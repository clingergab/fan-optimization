"""2D NACA-airfoil-in-farfield mesher for the Spike 0.6c solver-validation case.

Builds the CFD mesh for a single airfoil in unbounded flow: a large circular
far-field domain with the airfoil polyline cut out as an internal viscous wall.
Used by the oscillating-NACA-0012 benchmark that validates SU2's unsteady
pitching physics against published dynamic-stall data (the SU2 side of the
Spike 0.6c.2 cross-solver gate; the PyFR side needs the G4 GPU).

Two physical-group markers, written verbatim so the benchmark cfg's ``MARKER_*``
directives bind:

- ``airfoil``  — the airfoil surface (MARKER_HEATFLUX / MARKER_MONITORING)
- ``farfield`` — the outer circle (MARKER_FAR)

Geometry-agnostic: it meshes whatever closed polyline it is handed, so it works
for any NACA 4-digit section :func:`fanopt.cfd.airfoil_shapes.airfoil_polyline`
produces. Unstructured triangles with a Distance+Threshold boundary-layer
refinement toward the wall — enough for the low-Re (~40k) laminar benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gmsh
import numpy as np

__all__ = [
    "AIRFOIL_MARKER",
    "FARFIELD_MARKER",
    "AirfoilMeshParams",
    "AirfoilMeshResult",
    "build_airfoil_mesh",
]

# Locked marker names — must match the benchmark cfg's MARKER_* directives.
AIRFOIL_MARKER = "airfoil"
FARFIELD_MARKER = "farfield"


@dataclass(frozen=True)
class AirfoilMeshParams:
    """Far-field domain + sizing for the airfoil-in-freestream mesh (SI metres)."""

    farfield_radius_chords: float = 20.0  # domain radius in chord lengths
    wall_mesh_size_m: float = 0.004  # element size on the airfoil surface
    farfield_mesh_size_m: float = 0.5  # element size at the outer circle
    wall_refine_distance_m: float = 0.15  # distance over which wall sizing relaxes

    def __post_init__(self) -> None:
        if self.farfield_radius_chords <= 1.0:
            raise ValueError("farfield_radius_chords must be > 1 (domain must enclose the airfoil)")
        if min(self.wall_mesh_size_m, self.farfield_mesh_size_m) <= 0:
            raise ValueError("mesh sizes must be positive")
        if self.wall_mesh_size_m > self.farfield_mesh_size_m:
            raise ValueError("wall_mesh_size_m must be <= farfield_mesh_size_m")
        if self.wall_refine_distance_m <= 0:
            raise ValueError("wall_refine_distance_m must be > 0")


@dataclass(frozen=True)
class AirfoilMeshResult:
    """Summary of a written airfoil mesh."""

    path: Path
    n_nodes: int
    n_elements: int
    markers: tuple[str, ...]
    meta: dict[str, float] = field(default_factory=dict)


def _add_polyline_loop(polyline: np.ndarray, mesh_size: float) -> tuple[int, list[int]]:
    """Add a closed polyline to the OCC kernel; return (curve-loop tag, point tags)."""
    pts = [gmsh.model.occ.addPoint(float(x), float(y), 0.0, mesh_size) for x, y in polyline]
    lines = [gmsh.model.occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    return gmsh.model.occ.addCurveLoop(lines), pts


def build_airfoil_mesh(
    polyline: np.ndarray,
    params: AirfoilMeshParams,
    out_path: str | Path,
    *,
    chord_m: float = 1.0,
    motion_origin_x_m: float = 0.25,
) -> AirfoilMeshResult:
    """Mesh a single airfoil in a circular far-field and write it to ``out_path``.

    ``polyline`` is a closed ``(x, y)`` airfoil contour (see
    :func:`fanopt.cfd.airfoil_shapes.airfoil_polyline`). It is cut from a circle
    of radius ``farfield_radius_chords * chord_m`` centred on the pitching axis
    (``motion_origin_x_m``) so the airfoil surface becomes the ``airfoil`` wall
    and the circle becomes ``farfield``. Wall sizing relaxes to the far-field
    size over ``wall_refine_distance_m`` via a Distance+Threshold field.
    """
    poly = np.asarray(polyline, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        raise ValueError(f"polyline must be (>=3, 2); got shape {poly.shape}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    radius = params.farfield_radius_chords * chord_m

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("airfoil")

        domain = gmsh.model.occ.addDisk(motion_origin_x_m, 0.0, 0.0, radius, radius)
        airfoil_loop, _ = _add_polyline_loop(poly, params.wall_mesh_size_m)
        airfoil_surface = gmsh.model.occ.addPlaneSurface([airfoil_loop])
        gmsh.model.occ.synchronize()

        fluid, _ = gmsh.model.occ.cut(
            [(2, domain)], [(2, airfoil_surface)], removeTool=True
        )
        gmsh.model.occ.synchronize()
        surfaces = [tag for dim, tag in fluid if dim == 2]

        markers = _tag_markers(surfaces, radius)
        _apply_wall_refinement(markers, params)

        gmsh.option.setNumber("Mesh.MeshSizeMax", params.farfield_mesh_size_m)
        gmsh.option.setNumber("Mesh.MeshSizeMin", params.wall_mesh_size_m)
        gmsh.model.mesh.generate(2)

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        _, elem_tags, _ = gmsh.model.mesh.getElements(dim=2)
        n_elements = sum(len(t) for t in elem_tags)

        gmsh.write(str(out_path))
        return AirfoilMeshResult(
            path=out_path,
            n_nodes=len(node_tags),
            n_elements=n_elements,
            markers=markers,
            meta={
                "farfield_radius_m": radius,
                "chord_m": chord_m,
                "motion_origin_x_m": motion_origin_x_m,
            },
        )
    finally:
        gmsh.finalize()


def _tag_markers(surfaces: list[int], radius: float) -> tuple[str, ...]:
    """Classify boundary curves: the outer circle is farfield, the rest airfoil."""
    airfoil: set[int] = set()
    farfield: set[int] = set()
    boundary = gmsh.model.getBoundary([(2, s) for s in surfaces], oriented=False, combined=False)
    for _, curve in boundary:
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, curve)
        # The far-field circle spans ~2·radius; airfoil curves span ~1 chord (<<
        # radius, since radius = farfield_radius_chords · chord > chord). A closed
        # circle's centroid sits at the disk centre, so a radial test on the
        # centre-of-mass fails — the bounding-box extent separates them cleanly.
        extent = max(xmax - xmin, ymax - ymin)
        if extent > radius:
            farfield.add(curve)
        else:
            airfoil.add(curve)

    gmsh.model.addPhysicalGroup(2, list(surfaces), name="fluid")
    written: list[str] = []
    for tags, name in ((airfoil, AIRFOIL_MARKER), (farfield, FARFIELD_MARKER)):
        if tags:
            gmsh.model.addPhysicalGroup(1, sorted(tags), name=name)
            written.append(name)
    return tuple(written)


def _apply_wall_refinement(markers: tuple[str, ...], params: AirfoilMeshParams) -> None:
    """Distance-from-airfoil → Threshold size field (wall size near, farfield away)."""
    if AIRFOIL_MARKER not in markers:  # pragma: no cover - cut always yields the wall
        return
    wall_curves: list[int] = []
    for dim, tag in gmsh.model.getPhysicalGroups(1):
        if gmsh.model.getPhysicalName(dim, tag) == AIRFOIL_MARKER:
            wall_curves = list(gmsh.model.getEntitiesForPhysicalGroup(dim, tag))
            break
    dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist, "CurvesList", wall_curves)
    gmsh.model.mesh.field.setNumber(dist, "Sampling", 200)
    thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thr, "InField", dist)
    gmsh.model.mesh.field.setNumber(thr, "SizeMin", params.wall_mesh_size_m)
    gmsh.model.mesh.field.setNumber(thr, "SizeMax", params.farfield_mesh_size_m)
    gmsh.model.mesh.field.setNumber(thr, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(thr, "DistMax", params.wall_refine_distance_m)
    gmsh.model.mesh.field.setAsBackgroundMesh(thr)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
