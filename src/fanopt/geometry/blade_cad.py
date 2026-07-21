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
    RIB_TIP_RADIUS_M,
    BladeParams,
    displacement_at,
    layer_spacing_m,
    rib_thickness_at,
    rib_width_at,
    rib_z_at,
)
from fanopt.geometry.schema import (
    HUB_RADIUS_M,
    INTER_BLADE_ANGLE_RAD,
    PIVOT_BOSS_RADIUS_M,
    PIVOT_PIN_DIAMETER_M,
    RHO_PETG_KG_PER_M3,
)

__all__ = [
    "N_RADIAL_SECTIONS",
    "N_TANGENTIAL_SAMPLES",
    "make_blade_solid",
    "blade_trimesh",
    "blade_volume_m3",
    "blade_mass_kg",
    "fold_collision_volume_m3",
    "fold_collision_clear",
]

N_RADIAL_SECTIONS: int = 12
"""Radial cross-sections lofted along the blade (polyhedral approximation density)."""

N_TANGENTIAL_SAMPLES: int = 12
"""Tangential samples per cross-section across the wedge."""

_ALPHA_RAD: float = INTER_BLADE_ANGLE_RAD / 2.0
# Panel spans inward to the boss edge so the blade merges with the boss cylinder.
_R_INNER_M: float = PIVOT_BOSS_RADIUS_M
_FOLD_INTERSECT_EPS_M3: float = 1e-12  # numeric floor: below this = "clear"
_SEW_TOLERANCE_M: float = 1e-7


def _half_thickness_m(params: BladeParams, r: float, theta: float) -> float:
    """Local half material thickness: the rib (thick) near the wedge edges, else panel.

    A rib exists only in ``r ≥ HUB_RADIUS_M`` and within ``rib_width`` (arc) of an edge;
    the hub region and the wedge interior carry the thinner panel membrane.
    """
    if r >= HUB_RADIUS_M:
        edge_arc = r * (_ALPHA_RAD - abs(theta))
        if edge_arc <= rib_width_at(r):
            return rib_thickness_at(params, r) / 2.0
    return params.panel_thickness_nom_m / 2.0


def _surface_grids(
    params: BladeParams,
) -> tuple[list[list[cq.Vector]], list[list[cq.Vector]]]:
    """Top and bottom surface point grids ``[radial][tangential]`` (both faces free)."""
    top: list[list[cq.Vector]] = []
    bot: list[list[cq.Vector]] = []
    for i in range(N_RADIAL_SECTIONS):
        r = _R_INNER_M + (RIB_TIP_RADIUS_M - _R_INNER_M) * i / (N_RADIAL_SECTIONS - 1)
        top_row: list[cq.Vector] = []
        bot_row: list[cq.Vector] = []
        for j in range(N_TANGENTIAL_SAMPLES + 1):
            th = -_ALPHA_RAD + 2.0 * _ALPHA_RAD * j / N_TANGENTIAL_SAMPLES
            mean = rib_z_at(params, r) + displacement_at(params, r, th / _ALPHA_RAD)
            h = _half_thickness_m(params, r, th)
            x, y = r * math.cos(th), r * math.sin(th)
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


def _blade_faces(
    top: list[list[cq.Vector]], bot: list[list[cq.Vector]]
) -> list[cq.Face]:
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
    boss = (
        cq.Workplane("XY")
        .circle(PIVOT_BOSS_RADIUS_M)
        .extrude(s)
        .translate((0.0, 0.0, -s / 2.0))
    )
    hole = cq.Workplane("XY").circle(PIVOT_PIN_DIAMETER_M / 2.0).extrude(2.0 * s).translate(
        (0.0, 0.0, -s)
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


def blade_trimesh(
    params: BladeParams, tol: float = 0.0005
) -> tuple[np.ndarray, np.ndarray]:
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
