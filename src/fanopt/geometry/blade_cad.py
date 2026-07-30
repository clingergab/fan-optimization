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
    "fold_collision_clear",
]

N_RADIAL_SECTIONS: int = 60
"""Radial cross-sections lofted along the blade (polyhedral approximation density).

60 (not 12): the meridian is faceted between these stations, so a coarse count chops a
``smooth`` (Catmull-Rom) rib into a straight-segment zigzag — indistinguishable from
``linear`` and, worse, feeding the CFD spurious sharp edges that can drive divergence. Raised
40→60 with the surface-of-revolution height (2026-07-29): the mean surface now varies with the
TRUE radius ``√(x²+y²)``, which varies tangentially within a section, so a steep meridian needs a
finer mesh for the two rotated neighbours' faceted surfaces to nest to within the fold clearance
(coarse facets leave a mm³-scale chordal sliver — numerical, not a real interference)."""

N_TANGENTIAL_SAMPLES: int = 18
"""Tangential samples per cross-section across the wedge. Raised 12→18 alongside
:data:`N_RADIAL_SECTIONS` (2026-07-29): the surface-of-revolution height varies tangentially
across a section (rho grows toward the trapezoid edges), so a steep meridian is chorded
tangentially too — more samples shrink the polyhedral fold-gate artifact."""

# The trapezoid planform starts at the pivot centre (r=0) so the blade root overlaps the boss.
_R_INNER_M: float = BLADE_ROOT_RADIUS_M
_FOLD_INTERSECT_EPS_M3: float = 1e-8
"""Swept-intersection volume (m³) below which the fold gate reports ``clear``.

10 mm³. NOT a numeric zero: the gate intersects two *polyhedral* approximations of a smooth
surface-of-revolution blade, and where the meridian is steep the two rotated neighbours' facets
chord-mismatch by a mm³-scale sliver even though the underlying smooth solids nest with the full
:data:`~fanopt.geometry.blade.FOLD_CLEARANCE_M` gap. Measured over 371 diverse decoded designs
(Sobol + range-corner steep zigzags + extremes) AND a 48-way extreme sweep, the artifact peaks at
~3.4 mm³ and does not shrink to zero at the boss-rim kink with mesh refinement — it is numerical,
not a real interference. A real interference is an order of magnitude larger and grows (or holds)
under refinement: WITHOUT the boss-flat meridian + root taper a steep meridian climbs the next
layer's boss at ~140 mm³. Those construction fixes make every range-valid design fold, so this
threshold's job is to pass the faceting noise (it sits ~3× above the floor, ~14× below a gross
collision) while remaining a live backstop against any future gross interference. 100 % of the
371-design probe folds clear at this threshold; the exact-zero rate is ~94 %."""
_SEW_TOLERANCE_M: float = 1e-7


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
    if not params.uniform:
        edge_dist = half_width_at(r_station) * (1.0 - abs(v))  # y-distance to the nearest edge
        if edge_dist <= rib_width_at(r_station):
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
            mean = rib_z_at(params, rho) + _root_taper(rho) * displacement_at(params, rho, v)
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


def _boss_solid(params: BladeParams) -> cq.Workplane:
    """Pivot boss: a ``PIVOT_BOSS_OD_M`` cylinder one layer tall, pin hole subtracted."""
    s = layer_spacing_m(params)
    boss = cq.Workplane("XY").circle(PIVOT_BOSS_RADIUS_M).extrude(s).translate((0.0, 0.0, -s / 2.0))
    hole = (
        cq.Workplane("XY")
        .circle(PIVOT_PIN_DIAMETER_M / 2.0)
        .extrude(2.0 * s)
        .translate((0.0, 0.0, -s))
    )
    return boss.cut(hole)


def make_blade_solid(params: BladeParams) -> cq.Workplane:
    """Build one both-face blade solid (dished sector + rib edges + boss).

    Triangulates both surfaces + walls over ``N_RADIAL_SECTIONS × N_TANGENTIAL_SAMPLES``,
    sews them into a watertight solid, then unions the pivot boss. The result is a single
    valid solid in the blade's own frame (pin = +z axis); ``deploy``/fold place copies by
    rotation about +z.
    """
    top, bot = _surface_grids(params)
    solid = _sew_solid(_blade_faces(top, bot))
    blade = cq.Workplane("XY").newObject([solid])
    return blade.union(_boss_solid(params))


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


def fold_collision_volume_m3(params: BladeParams, *, n_swing_steps: int = 6) -> float:
    """Max intersection volume between two adjacent blades across the fold swing (m³).

    Stacks blade *i+1* one layer above blade *i* on the pin and rotates it from the
    folded pose (0°) out to the deployed pitch (one inter-blade angle), sampling
    ``n_swing_steps`` intermediate angles. Any non-trivial intersection = a real
    collision the analytic proxies missed. ~0 confirms the design folds.
    """
    blade = make_blade_solid(params).val()
    s = layer_spacing_m(params)
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


def fold_collision_clear(params: BladeParams, *, n_swing_steps: int = 6) -> bool:
    """True iff adjacent blades never intersect through the fold swing (authoritative)."""
    return fold_collision_volume_m3(params, n_swing_steps=n_swing_steps) <= _FOLD_INTERSECT_EPS_M3
