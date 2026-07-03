"""2D mid-radius cascade-slice mesher (Tier -1 / Phase 3, report-final.md §9.6).

Builds the CFD mesh for the unwrapped linear cascade: a rectangular 2D flow
domain (axis 0 = streamwise ≡ deployed +z; axis 1 = tangential/cross) with the
blade cross-sections cut out as internal walls. Writes the locked physical-group
marker names verbatim so the SU2 ``MARKER_*`` directives bind (§9.6):

- ``fan_surface`` — blade body boundaries (MARKER_HEATFLUX / MARKER_MONITORING)
- ``farfield``    — streamwise inlet + outlet arcs (MARKER_FAR)
- ``cascade_wall`` — cross-extent top + bottom (MARKER_SYM); forces freestream
  through the inter-blade gaps
- ``downstream_plane`` — internal line 0.300 m forward along +z (MARKER_ANALYZE),
  the §9.4 J_fan analysis plane

The mesher is geometry-agnostic: it meshes whatever cross-section polygons it is
handed, so the Layer-1/Path-A+ panel parameterization can evolve without
touching this module. ``baseline_cascade_polygons`` produces a simple flat-panel
corrugated cross-section for smoke tests and the Phase-2a baseline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import gmsh
import numpy as np

from fanopt.cfd.j_fan import ANALYSIS_PLANE_DISTANCE_M

__all__ = [
    "FAN_SURFACE_MARKER",
    "FARFIELD_MARKER",
    "CASCADE_WALL_MARKER",
    "DOWNSTREAM_PLANE_MARKER",
    "SliceMeshParams",
    "SliceMeshResult",
    "baseline_cascade_polygons",
    "build_cascade_slice_mesh",
]

# Locked marker names — must match the SU2 configs verbatim (§9.6 table).
FAN_SURFACE_MARKER = "fan_surface"
FARFIELD_MARKER = "farfield"
CASCADE_WALL_MARKER = "cascade_wall"
DOWNSTREAM_PLANE_MARKER = "downstream_plane"


@dataclass(frozen=True)
class SliceMeshParams:
    """Cascade-slice domain + sizing parameters (SI metres)."""

    streamwise_inlet_m: float = 0.30  # upstream extent (−z) from origin
    streamwise_outlet_m: float = 0.45  # downstream extent (+z); must exceed the plane
    cross_half_extent_m: float = 0.15  # tangential half-height of the domain
    mesh_size_m: float = 0.01  # far-field target element size
    wall_mesh_size_m: float = 0.001  # element size on the fan surface
    downstream_plane_m: float = ANALYSIS_PLANE_DISTANCE_M  # internal plane at +z
    include_downstream_plane: bool = False  # embed the MARKER_ANALYZE line

    def __post_init__(self) -> None:
        if self.streamwise_outlet_m <= self.downstream_plane_m:
            raise ValueError(
                f"streamwise_outlet_m ({self.streamwise_outlet_m}) must exceed the "
                f"downstream plane ({self.downstream_plane_m})"
            )
        if self.downstream_plane_m <= 0:
            raise ValueError("downstream_plane_m must be > 0 (plane is forward of the pivot)")
        if min(self.streamwise_inlet_m, self.cross_half_extent_m) <= 0:
            raise ValueError("domain extents must be positive")
        if min(self.mesh_size_m, self.wall_mesh_size_m) <= 0:
            raise ValueError("mesh sizes must be positive")


@dataclass(frozen=True)
class SliceMeshResult:
    """Summary of a written cascade-slice mesh."""

    path: Path
    n_nodes: int
    n_elements: int
    markers: tuple[str, ...]
    n_blades: int
    meta: dict[str, float] = field(default_factory=dict)


def baseline_cascade_polygons(
    *,
    n_blades: int = 5,
    panel_width_m: float = 0.045,
    panel_gap_m: float = 0.005,
    panel_thickness_m: float = 0.0038,
    rib_thickness_m: float = 0.002,
) -> list[np.ndarray]:
    """Flat-panel corrugated cross-section: one rectangle per blade panel.

    Panels are laid out along the cross (tangential) axis, each protruding on
    +z (streamwise) above the rib valley by ``panel_thickness − rib_thickness``
    (the one-sided rib-flat corrugation of §3.2.4). Returns closed ``(4, 2)``
    ``(streamwise, cross)`` polygons centred on cross = 0.
    """
    if n_blades < 1:
        raise ValueError(f"n_blades must be ≥ 1; got {n_blades}")
    pitch = panel_width_m + panel_gap_m
    total = n_blades * pitch - panel_gap_m
    start = -total / 2.0
    protrusion = panel_thickness_m - rib_thickness_m
    z_lo, z_hi = -rib_thickness_m / 2.0, rib_thickness_m / 2.0 + protrusion
    polys: list[np.ndarray] = []
    for i in range(n_blades):
        c0 = start + i * pitch
        c1 = c0 + panel_width_m
        polys.append(np.array([[z_lo, c0], [z_hi, c0], [z_hi, c1], [z_lo, c1]], dtype=float))
    return polys


def _add_polygon_loop(polygon: np.ndarray, mesh_size: float) -> int:
    """Add a closed polygon to the OCC kernel; return its curve-loop tag."""
    pts = [gmsh.model.occ.addPoint(float(x), float(y), 0.0, mesh_size) for x, y in polygon]
    lines = [gmsh.model.occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    return gmsh.model.occ.addCurveLoop(lines)


def build_cascade_slice_mesh(
    cross_section: Sequence[np.ndarray],
    params: SliceMeshParams,
    out_path: str | Path,
) -> SliceMeshResult:
    """Mesh a 2D cascade slice and write it to ``out_path`` (.su2 / .msh).

    ``cross_section`` is a sequence of closed ``(streamwise, cross)`` polygons
    (the solid blade bodies) — they are subtracted from the flow rectangle so
    their boundaries become the ``fan_surface`` wall. The rectangle's streamwise
    ends become ``farfield``, its cross ends ``cascade_wall``, and an internal
    line at ``params.downstream_plane_m`` becomes ``downstream_plane``.
    """
    polygons = [np.asarray(p, dtype=float) for p in cross_section]
    if not polygons:
        raise ValueError("cross_section must contain at least one polygon")
    for p in polygons:
        if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 3:
            raise ValueError(f"each polygon must be (>=3, 2); got shape {p.shape}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("cascade_slice")

        x0, x1 = -params.streamwise_inlet_m, params.streamwise_outlet_m
        y0, y1 = -params.cross_half_extent_m, params.cross_half_extent_m
        domain = gmsh.model.occ.addRectangle(x0, y0, 0.0, x1 - x0, y1 - y0)

        blade_loops = [_add_polygon_loop(p, params.wall_mesh_size_m) for p in polygons]
        blade_surfaces = [gmsh.model.occ.addPlaneSurface([loop]) for loop in blade_loops]
        gmsh.model.occ.synchronize()

        fluid, _ = gmsh.model.occ.cut(
            [(2, domain)], [(2, s) for s in blade_surfaces], removeTool=True
        )
        if params.include_downstream_plane:
            # Embed the MARKER_ANALYZE line. Note: gmsh's .su2 exporter writes
            # only boundary markers, so this internal line survives in .msh but
            # not .su2 — the 2D-slice objective is the surface-force proxy, which
            # needs no internal plane.
            line = gmsh.model.occ.addLine(
                gmsh.model.occ.addPoint(params.downstream_plane_m, y0, 0.0, params.mesh_size_m),
                gmsh.model.occ.addPoint(params.downstream_plane_m, y1, 0.0, params.mesh_size_m),
            )
            frag, _ = gmsh.model.occ.fragment(fluid, [(1, line)])
            gmsh.model.occ.synchronize()
            surfaces = [tag for dim, tag in frag if dim == 2]
        else:
            gmsh.model.occ.synchronize()
            surfaces = [tag for dim, tag in fluid]
        markers = _tag_markers(surfaces, params, x0, x1, y0, y1)

        gmsh.option.setNumber("Mesh.MeshSizeMax", params.mesh_size_m)
        gmsh.option.setNumber("Mesh.MeshSizeMin", params.wall_mesh_size_m)
        gmsh.model.mesh.generate(2)

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=2)
        n_elements = sum(len(t) for t in elem_tags)

        gmsh.write(str(out_path))
        return SliceMeshResult(
            path=out_path,
            n_nodes=len(node_tags),
            n_elements=n_elements,
            markers=markers,
            n_blades=len(polygons),
            meta={
                "streamwise_span_m": x1 - x0,
                "cross_span_m": y1 - y0,
                "downstream_plane_m": params.downstream_plane_m,
            },
        )
    finally:
        gmsh.finalize()


def _tag_markers(
    surfaces: Sequence[int],
    params: SliceMeshParams,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> tuple[str, ...]:
    """Assign physical groups by classifying boundary curves geometrically.

    Returns the marker names actually written to the mesh.
    """
    tol = min(params.wall_mesh_size_m, 1e-4)
    fan: set[int] = set()
    farfield: set[int] = set()
    cascade: set[int] = set()
    plane: set[int] = set()
    # combined=False keeps internal shared edges (the downstream line); dedup via sets.
    boundary = gmsh.model.getBoundary([(2, s) for s in surfaces], oriented=False, combined=False)
    for _, curve in boundary:
        com = gmsh.model.occ.getCenterOfMass(1, curve)
        cx, cy = com[0], com[1]
        on_inlet_outlet = abs(cx - x0) < tol or abs(cx - x1) < tol
        on_top_bottom = abs(cy - y0) < tol or abs(cy - y1) < tol
        on_plane = params.include_downstream_plane and abs(cx - params.downstream_plane_m) < tol
        if on_plane:
            plane.add(curve)
        elif on_inlet_outlet:
            farfield.add(curve)
        elif on_top_bottom:
            cascade.add(curve)
        else:
            fan.add(curve)

    gmsh.model.addPhysicalGroup(2, list(surfaces), name="fluid")
    written: list[str] = []
    for tags, name in (
        (fan, FAN_SURFACE_MARKER),
        (farfield, FARFIELD_MARKER),
        (cascade, CASCADE_WALL_MARKER),
        (plane, DOWNSTREAM_PLANE_MARKER),
    ):
        if tags:
            gmsh.model.addPhysicalGroup(1, sorted(tags), name=name)
            written.append(name)
    return tuple(written)
