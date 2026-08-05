"""CadQuery generator for the lean surface-of-revolution blade (V1 redesign, chunk 2).

Builds the both-face blade solid from a :class:`~fanopt.geometry.blade.BladeParams`:
a dished sector (surface of revolution about the pin/z-axis) carrying the free panel
displacement grid, with thick rib edges, unioned to the pivot boss. Also the
authoritative **swept-volume fold gate** — stack two adjacent blades one layer apart on
the pin and rotate through the fold; a non-empty intersection means collision.

Coordinate frame: the pin is the +z axis through the origin; a blade occupies the
angular wedge ``θ ∈ [-α, +α]`` (``α = INTER_BLADE_ANGLE_RAD/2``) about +x, radius
``r`` outward in the x-y plane. Blade height (z) is the dished mean surface
``rib_z_at(r) + displacement`` ± the local half-thickness (rib at the edges, panel
between). Per CLAUDE.md §4.1 this module imports cadquery unconditionally; environments
without it fail to import (tests skip at module load via ``find_spec``).
"""

from __future__ import annotations

import math

import cadquery as cq
import numpy as np
from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCP.ShapeFix import ShapeFix_Solid
from OCP.TopoDS import TopoDS

from fanopt.geometry.blade import (
    BLADE_ROOT_RADIUS_M,
    MERIDIAN_ROOT_FLAT_RADIUS_M,
    RIB_TIP_RADIUS_M,
    BladeParams,
    displacement_at,
    half_width_at,
    layer_spacing_m,
    panel_thickness_at,
    rib_thickness_at,
    rib_width_at,
    rib_z_at,
)
from fanopt.geometry.schema import (
    INTER_BLADE_ANGLE_RAD,
    PIVOT_BOSS_RADIUS_M,
    PIVOT_PIN_DIAMETER_M,
    RHO_PETG_KG_PER_M3,
)

__all__ = [
    "N_RADIAL_SECTIONS",
    "N_TANGENTIAL_SAMPLES",
    "make_blade_solid",
    "export_blade_step",
    "blade_trimesh",
    "blade_volume_m3",
    "blade_mass_kg",
    "fold_collision_volume_m3",
    "fold_penetration_m",
    "fold_collision_clear",
]

N_RADIAL_SECTIONS: int = 40
"""Radial cross-sections lofted into the CAD solid (:func:`make_blade_solid`, used by the CFD/FEA
mesh export and the offline CAD-boolean fold cross-check :func:`fold_collision_volume_m3`).

**40 is the campaign-locked resolution (ADR-0003).** It was previously set at runtime in every
notebook while the module default drifted to 60; it is now the DEFAULT so a run that forgets to set
it — or a spawn-based pool worker that re-imports this module fresh — cannot silently mesh at a
different resolution than the campaign objective did, which would make its results non-comparable.
Must stay well above the shipped-12: at a coarse count the meridian is faceted so a ``smooth``
(Catmull-Rom) rib gets chopped into a straight-segment zigzag — indistinguishable from ``linear`` and,
worse, feeding the CFD spurious sharp edges that can drive divergence. This density does NOT drive the
in-loop fold gate: that is the analytic :func:`fold_penetration_m`, which evaluates the smooth height
field directly (no facets), so the fold verdict is mesh-independent."""

N_TANGENTIAL_SAMPLES: int = 18
"""Tangential samples per cross-section across the wedge (paired with :data:`N_RADIAL_SECTIONS` for
the lofted CAD solid). Not used by the analytic fold gate."""

# The trapezoid planform starts at the pivot centre (r=0) so the blade root overlaps the boss.
_R_INNER_M: float = BLADE_ROOT_RADIUS_M
_FOLD_INTERSECT_EPS_M3: float = 5e-9
"""Swept-intersection volume (m³) below which the OFFLINE CAD-boolean cross-check
(:func:`fold_collision_volume_m3`) reads as clear. 5 mm³ absorbs the polyhedral faceting sliver a
sharp ``linear`` zigzag leaves between two rotated neighbours (the CAD boolean intersects faceted
approximations; the underlying smooth solids nest with the full clearance). NOT the in-loop gate —
that is the faceting-free analytic :func:`fold_penetration_m` / :data:`_FOLD_PENETRATION_EPS_M`; this
constant is only for the CAD deep-verify used to validate the analytic gate offline."""
_SEW_TOLERANCE_M: float = 1e-7

_FOLD_SAMPLE_RADIAL: int = 60
_FOLD_SAMPLE_TANGENTIAL: int = 30
"""Planform sampling density for the analytic fold gate (:func:`fold_penetration_m`). Dense enough
that the deepest interpenetration between two rotated neighbours is captured to well within
:data:`_FOLD_PENETRATION_EPS_M`; validated against the CAD swept-volume boolean."""

_FOLD_PENETRATION_EPS_M: float = 5e-5
"""0.05 mm — max fold interpenetration DEPTH below which the analytic gate reports ``clear``.

The analytic gate evaluates the smooth height field exactly (no polyhedral faceting), so this is a
true tolerance, not a faceting floor: it absorbs only the sub-grid sampling gap. A foldable design
nests with a gap (penetration ``≤ −FOLD_CLEARANCE_M ≈ −0.4 mm``); a real collision (e.g. a
checkerboard Way-2 wave exceeding the layer clearance) penetrates by mm — so 0.05 mm sits deep in
the empty band between the two, ~8× below the clearance margin and ~20×+ below a real collision."""


_ROOT_TAPER_END_M: float = 2.0 * MERIDIAN_ROOT_FLAT_RADIUS_M
"""True radius by which the panel-displacement root taper reaches full amplitude (18 mm)."""


def _root_taper(rho: float) -> float:
    """Radial weight (0→1) suppressing the panel mean-surface offset inside the boss footprint.

    The meridian is pinned flat for ``rho ≤`` :data:`MERIDIAN_ROOT_FLAT_RADIUS_M`, but the panel
    displacement grid could still lift the mean surface there — which climbs the blade root into
    the next stacked layer's boss and breaks the fold exactly as a steep meridian would. This
    ramps the displacement from 0 at the boss-flat radius to full amplitude at
    :data:`_ROOT_TAPER_END_M`, so no blade material rises within the buried near-boss region. The
    ramped span (inner ~8 % of the 220 mm blade, near the hub) carries negligible wind, so Way-2
    face-shaping freedom is untouched where it matters."""
    if rho <= MERIDIAN_ROOT_FLAT_RADIUS_M:
        return 0.0
    if rho >= _ROOT_TAPER_END_M:
        return 1.0
    t = (rho - MERIDIAN_ROOT_FLAT_RADIUS_M) / (_ROOT_TAPER_END_M - MERIDIAN_ROOT_FLAT_RADIUS_M)
    return t * t * (3.0 - 2.0 * t)  # smoothstep (C1 at both ends — no lip to catch a neighbour)


def _is_rib_rail(params: BladeParams, r_station: float, v: float) -> bool:
    """True iff the planform point is inside a rib rail (thick frame) rather than the panel.

    A rib rail runs within ``rib_width`` (tangential y-distance) of each trapezoid edge. Uniform
    (no-rib) blades have no rails, so this is always ``False``. Classification is a **planform**
    decision (fixed tangential distance from the straight edge), so it uses ``r_station``.
    """
    if params.uniform:
        return False
    edge_dist = half_width_at(r_station) * (1.0 - abs(v))  # y-distance to the nearest edge
    return edge_dist <= rib_width_at(r_station)


def _panel_window(params: BladeParams, r_station: float, v: float) -> float:
    """Tangential weight (0→1) that suppresses the panel mean displacement across the rib rails.

    **The fold fix (2026-07-29, part 2):** the panel mean-surface displacement grid pins only the
    trapezoid *edge* (``v = ±1``) to 0, but a rib rail is a whole *band* (``edge_dist ≤ rib_width``)
    — so without this window the thick rail rides UP on the still-nonzero displacement inside the
    band and pokes ABOVE the ``±t_rib/2`` containment envelope (measured ~0.8 mm of poke → ~7 mm³
    fold collision). Containment (the codec) only proves the thin *panel* fits inside the rib
    envelope; it never accounted for the rails riding the mean. Zeroing the displacement across the
    rail band puts the rails on the pure meridian (a true surface of revolution → folds any shape)
    and leaves the interior panel free to wave *within* the rib envelope — exactly the intended
    ribbed Way-2 model (rails = frame on the meridian; panel undulates between them). Uniform blades
    have no rails, so the window is 1 everywhere (the sheet waves to its edges; it nests by the
    codec's ``|offset| + t/2 ≤ e_cap/2`` bound instead).

    The window ramps smoothly (smoothstep) from 0 at the rail inner boundary to 1 one rail-width
    further in, so the panel blends into the frame with no sharp lip to catch a folded neighbour.
    """
    if params.uniform:
        return 1.0
    edge_dist = half_width_at(r_station) * (1.0 - abs(v))
    rail = rib_width_at(r_station)
    if edge_dist <= rail:
        return 0.0
    if edge_dist >= 2.0 * rail:
        return 1.0
    t = (edge_dist - rail) / rail
    return t * t * (3.0 - 2.0 * t)


def _half_thickness_m(params: BladeParams, r_station: float, rho: float, v: float) -> float:
    """Local half material thickness at planform station ``r_station``, true radius ``rho`` and
    tangential fraction ``v`` ∈ [-1, 1].

    The rib-rail classification (which nodes are thick rail vs thin panel) is a **planform**
    decision — a fixed tangential distance from the trapezoid edge — so it uses ``r_station``.
    The thickness *magnitude* is a **radial height field** and uses the true radius ``rho`` so the
    top/bottom surfaces are surfaces of revolution (see :func:`_surface_grids`).

    **Ribbed**: a rib rail (thick) runs within ``rib_width`` (tangential distance) of each
    trapezoid edge; the interior carries the thinner panel membrane. **Uniform**: no rails —
    the whole width is the panel sheet.
    """
    if _is_rib_rail(params, r_station, v):
        return rib_thickness_at(params, rho) / 2.0
    return panel_thickness_at(params, rho, v) / 2.0


def _surface_grids(
    params: BladeParams,
) -> tuple[list[list[cq.Vector]], list[list[cq.Vector]]]:
    """Top and bottom surface point grids ``[radial][tangential]`` (both faces free).

    **Planform** (unchanged): a cross-section at station ``x = r`` spans ``y ∈ [-w(r), +w(r)]``
    where ``w = half_width_at(r)`` grows linearly root→tip (a Cartesian trapezoid, NOT the retired
    ``y = r·sin θ`` sector).

    **Height field = surface of revolution** (the fold fix, 2026-07-29): the mean surface and the
    material thickness are functions of the **true radius** ``rho = √(x²+y²)`` (clamped to the
    valid radial domain at the tip corners), NOT the x-station. ``z = f(x)`` made a flat Cartesian
    strip whose profile went out of phase when a neighbour was rotated onto the pin, so multi-hump
    / zigzag meridians collided; ``z = f(rho)`` is a true surface of revolution, so rotated
    neighbours are congruent and **any** meridian shape nests when folded. Verified: same zigzag
    meridian collides 81 mm³ as ``f(x)`` vs ~3.5 mm³ (folds) as ``f(rho)``.
    """
    top: list[list[cq.Vector]] = []
    bot: list[list[cq.Vector]] = []
    for i in range(N_RADIAL_SECTIONS):
        r = _R_INNER_M + (RIB_TIP_RADIUS_M - _R_INNER_M) * i / (N_RADIAL_SECTIONS - 1)
        w = half_width_at(r)
        top_row: list[cq.Vector] = []
        bot_row: list[cq.Vector] = []
        for j in range(N_TANGENTIAL_SAMPLES + 1):
            v = -1.0 + 2.0 * j / N_TANGENTIAL_SAMPLES
            x, y = r, v * w
            rho = min(max(math.hypot(x, y), _R_INNER_M), RIB_TIP_RADIUS_M)
            panel_disp = _panel_window(params, r, v) * displacement_at(params, rho, v)
            mean = rib_z_at(params, rho) + _root_taper(rho) * panel_disp
            h = _half_thickness_m(params, r, rho, v)
            top_row.append(cq.Vector(x, y, mean + h))
            bot_row.append(cq.Vector(x, y, mean - h))
        top.append(top_row)
        bot.append(bot_row)
    return top, bot


def _tri(a: cq.Vector, b: cq.Vector, c: cq.Vector) -> cq.Face:
    return cq.Face.makeFromWires(cq.Wire.makePolygon([a, b, c], close=True))


def _quad(a: cq.Vector, b: cq.Vector, c: cq.Vector, d: cq.Vector) -> list[cq.Face]:
    """Two triangles for the (non-planar) quad a→b→c→d, wound consistently."""
    return [_tri(a, b, c), _tri(a, c, d)]


def _blade_faces(top: list[list[cq.Vector]], bot: list[list[cq.Vector]]) -> list[cq.Face]:
    """Closed triangulated boundary: top + bottom surfaces, tangential walls, radial caps."""
    ni, nj = N_RADIAL_SECTIONS, N_TANGENTIAL_SAMPLES
    faces: list[cq.Face] = []
    for i in range(ni - 1):
        for j in range(nj):
            faces += _quad(top[i][j], top[i][j + 1], top[i + 1][j + 1], top[i + 1][j])
            faces += _quad(bot[i][j], bot[i + 1][j], bot[i + 1][j + 1], bot[i][j + 1])
    for i in range(ni - 1):  # tangential edge walls (the two rib flanks)
        faces += _quad(top[i][0], bot[i][0], bot[i + 1][0], top[i + 1][0])
        faces += _quad(top[i][nj], top[i + 1][nj], bot[i + 1][nj], bot[i][nj])
    for j in range(nj):  # radial end caps (hub, tip)
        faces += _quad(top[0][j], bot[0][j], bot[0][j + 1], top[0][j + 1])
        faces += _quad(top[ni - 1][j], top[ni - 1][j + 1], bot[ni - 1][j + 1], bot[ni - 1][j])
    return faces


def _sew_solid(faces: list[cq.Face]) -> cq.Solid:
    """Sew a triangulated boundary into a watertight, outward-oriented solid."""
    sew = BRepBuilderAPI_Sewing(_SEW_TOLERANCE_M)
    for f in faces:
        sew.Add(f.wrapped)
    sew.Perform()
    shell = TopoDS.Shell_s(sew.SewedShape())
    return cq.Solid(ShapeFix_Solid().SolidFromShell(shell))


def _boss_solid(params: BladeParams, *, clearance_m: float | None = None) -> cq.Workplane:
    """Pivot boss: a ``PIVOT_BOSS_OD_M`` cylinder one layer tall, pin hole subtracted.

    ``clearance_m`` overrides the fold clearance → a shorter boss packs the deployed deck tighter.
    """
    s = layer_spacing_m(params, clearance_m=clearance_m)
    boss = cq.Workplane("XY").circle(PIVOT_BOSS_RADIUS_M).extrude(s).translate((0.0, 0.0, -s / 2.0))
    hole = (
        cq.Workplane("XY")
        .circle(PIVOT_PIN_DIAMETER_M / 2.0)
        .extrude(2.0 * s)
        .translate((0.0, 0.0, -s))
    )
    return boss.cut(hole)


def make_blade_solid(params: BladeParams, *, clearance_m: float | None = None) -> cq.Workplane:
    """Build one both-face blade solid (dished sector + rib edges + boss).

    Triangulates both surfaces + walls over ``N_RADIAL_SECTIONS × N_TANGENTIAL_SAMPLES``,
    sews them into a watertight solid, then unions the pivot boss. The result is a single
    valid solid in the blade's own frame (pin = +z axis); ``deploy``/fold place copies by
    rotation about +z. ``clearance_m`` overrides the fold clearance (sizes the boss height).
    """
    top, bot = _surface_grids(params)
    solid = _sew_solid(_blade_faces(top, bot))
    blade = cq.Workplane("XY").newObject([solid])
    return blade.union(_boss_solid(params, clearance_m=clearance_m))


def export_blade_step(params: BladeParams, path: str) -> str:
    """Write one blade design to a STEP file (open in any CAD viewer to 3D-render). Returns path."""
    cq.exporters.export(make_blade_solid(params), str(path))
    return str(path)


def blade_trimesh(params: BladeParams, tol: float = 0.0005) -> tuple[np.ndarray, np.ndarray]:
    """Triangulated surface of the blade solid as ``(vertices (N,3), faces (M,3))`` arrays.

    Tessellates the CAD solid to a triangle soup for 3D surface plotting (e.g. Plotly
    ``Mesh3d``). ``tol`` is the chordal deviation in metres — smaller renders finer.
    """
    verts, tris = make_blade_solid(params).val().tessellate(tol)
    vertices = np.array([[v.x, v.y, v.z] for v in verts], dtype=float)
    faces = np.array(tris, dtype=int)
    return vertices, faces


def blade_volume_m3(params: BladeParams) -> float:
    """Solid volume of one blade (m³) — authoritative vs the analytic mass proxy."""
    return make_blade_solid(params).val().Volume()


def blade_mass_kg(params: BladeParams, density_kg_per_m3: float = RHO_PETG_KG_PER_M3) -> float:
    """Mass of the whole fan (m³ × ρ × blade_count)."""
    return blade_volume_m3(params) * density_kg_per_m3 * params.blade_count


def fold_collision_volume_m3(
    params: BladeParams, *, n_swing_steps: int = 6, clearance_m: float | None = None
) -> float:
    """Max intersection volume between two adjacent blades across the fold swing (m³).

    Stacks blade *i+1* one layer above blade *i* on the pin and rotates it from the
    folded pose (0°) out to the deployed pitch (one inter-blade angle), sampling
    ``n_swing_steps`` intermediate angles. Any non-trivial intersection = a real
    collision the analytic proxies missed. ~0 confirms the design folds. ``clearance_m``
    checks a tighter fold clearance (a tighter deck must still show ~0 intersection).
    """
    blade = make_blade_solid(params, clearance_m=clearance_m).val()
    s = layer_spacing_m(params, clearance_m=clearance_m)
    worst = 0.0
    for k in range(n_swing_steps + 1):
        delta_deg = math.degrees(INTER_BLADE_ANGLE_RAD) * k / n_swing_steps
        other = blade.translate(cq.Vector(0.0, 0.0, s)).rotate(
            cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), delta_deg
        )
        common = blade.intersect(other)
        vol = common.Volume() if common is not None else 0.0
        worst = max(worst, vol)
    return worst


def _surface_top_bot(params: BladeParams, x: float, y: float) -> tuple[float, float] | None:
    """Top and bottom blade-surface z at planform point ``(x, y)`` — or ``None`` if outside the
    trapezoid. Evaluates the SAME mean/thickness height field the CAD solid is lofted from
    (:func:`_surface_grids`), so the analytic fold gate is consistent with the CAD boolean.
    """
    r = x  # planform radial station
    if r < _R_INNER_M or r > RIB_TIP_RADIUS_M:
        return None
    w = half_width_at(r)
    if w <= 0.0 or abs(y) > w:
        return None
    v = y / w
    rho = min(max(math.hypot(x, y), _R_INNER_M), RIB_TIP_RADIUS_M)
    panel_disp = _panel_window(params, r, v) * displacement_at(params, rho, v)
    mean = rib_z_at(params, rho) + _root_taper(rho) * panel_disp
    h = _half_thickness_m(params, r, rho, v)
    return mean + h, mean - h


def fold_penetration_m(
    params: BladeParams,
    *,
    n_swing_steps: int = 6,
    n_radial: int = _FOLD_SAMPLE_RADIAL,
    n_tangential: int = _FOLD_SAMPLE_TANGENTIAL,
    clearance_m: float | None = None,
) -> float:
    """Max z-penetration (m) between two adjacent blades across the fold swing — analytic gate.

    Stacks blade *i+1* one layer above blade *i* (``+layer_spacing``) and rotates it through the
    fold from the folded pose (0°) to the deployed pitch (one inter-blade angle). At each swing angle
    and each sampled planform point ``p`` of blade *i*, the neighbour occupies that column iff the
    un-rotated pre-image ``q = R₋Δ(p)`` lies in the planform; the two material z-spans then overlap by
    ``min(top_i(p), top_i(q)+s) − max(bot_i(p), bot_i(q)+s)``. The max over the grid and the swing is
    the deepest interpenetration: ``≤ 0`` means the blades nest with a gap (folds), ``> 0`` is a real
    collision depth.

    This replaces the CAD swept-volume boolean as the in-loop gate: it evaluates the smooth
    surface-of-revolution height field directly, so it has **no polyhedral faceting artifact**
    (exact, tight threshold), **cannot hang** (OCP's boolean hangs indefinitely on a valid-but-
    pathological solid — steep bipolar zigzag + checkerboard panel offsets — which is decode-
    reachable), and runs in ~ms instead of seconds. The boss annulus is not sampled: the boss is one
    ``layer_spacing`` tall and stacks exactly one layer apart, so it nests by construction. The CAD
    boolean :func:`fold_collision_volume_m3` is retained as an offline deep-verify cross-check.

    The default grid resolves the fold VERDICT reliably (validated: 0 false-negatives over 450+
    designs, including 25×-finer re-sampling), but it is not a tight bound on the penetration
    MAGNITUDE for extreme high-frequency panel textures — the sole positive-penetration mechanism (a
    rail riding the panel mean) is a low-frequency rail band the grid samples well, while the deep
    negative gaps of a high-frequency design can be under-resolved. Pass a finer ``n_radial`` /
    ``n_tangential`` if an exact depth is needed.
    """
    s = layer_spacing_m(params, clearance_m=clearance_m)
    alpha = INTER_BLADE_ANGLE_RAD
    swings = max(n_swing_steps, 1)  # guard n_swing_steps=0 (folded pose only) against /0
    worst = -math.inf
    for ir in range(n_radial):
        r = _R_INNER_M + (RIB_TIP_RADIUS_M - _R_INNER_M) * ir / (n_radial - 1)
        w = half_width_at(r)
        for it in range(n_tangential + 1):
            v = -1.0 + 2.0 * it / n_tangential
            x, y = r, v * w
            tb_i = _surface_top_bot(params, x, y)
            if tb_i is None:  # pragma: no cover - grid points are in-planform by construction
                continue
            top_i, bot_i = tb_i
            for k in range(swings + 1):
                delta = alpha * k / swings
                cos_d, sin_d = math.cos(-delta), math.sin(-delta)  # un-rotate the neighbour
                xq = x * cos_d - y * sin_d
                yq = x * sin_d + y * cos_d
                tb_q = _surface_top_bot(params, xq, yq)
                if tb_q is None:
                    continue  # neighbour absent in this column → no overlap here
                pen = min(top_i, tb_q[0] + s) - max(bot_i, tb_q[1] + s)
                if pen > worst:
                    worst = pen
    return worst


def fold_collision_clear(params: BladeParams, *, n_swing_steps: int = 6) -> bool:
    """True iff adjacent blades nest (never interpenetrate) through the fold swing.

    Uses the analytic :func:`fold_penetration_m` (fast, hang-proof, faceting-free). A design folds
    iff the deepest interpenetration stays within :data:`_FOLD_PENETRATION_EPS_M` — a small tolerance
    absorbing only the sub-grid sampling gap, far below any real collision.
    """
    return fold_penetration_m(params, n_swing_steps=n_swing_steps) <= _FOLD_PENETRATION_EPS_M
