"""3D volume mesher for high-fidelity CFD verification (Phase 5, Tier 0/1).

Builds the external-flow mesh for a solid body (a single V-unit blade for V1
verification — the full deployed fan is a thin multi-body geometry whose click-
mating faces are non-manifold and won't tet-mesh). The body is imported from a
STEP file (exported by the CadQuery generator), subtracted from a padded domain
box, and 3D tet-meshed with wall refinement. Marker names match the ``fan3d_*``
SU2 configs:

- ``fan_surface`` — the body boundary (MARKER_HEATFLUX / MARKER_MONITORING)
- ``farfield``    — the domain-box faces (MARKER_FAR)

Geometry-agnostic (like :mod:`fanopt.cfd.mesh_2d_slice`): meshes whatever solids
the STEP holds, so the caller chooses single-blade vs. (if it meshes) full fan.
Gmsh-only — no CadQuery dependency; geometry export happens at the call site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gmsh

__all__ = [
    "FAN_SURFACE_MARKER",
    "FARFIELD_MARKER",
    "VolumeMeshParams",
    "VolumeMeshResult",
    "build_volume_mesh",
]

FAN_SURFACE_MARKER = "fan_surface"
FARFIELD_MARKER = "farfield"


@dataclass(frozen=True)
class VolumeMeshParams:
    """External-flow domain + sizing (SI metres)."""

    pad_upstream_m: float = 0.15  # −z (into the incoming flow)
    pad_downstream_m: float = 0.30  # +z (wake — user-ward thrust direction)
    pad_lateral_m: float = 0.15  # x and y padding around the body
    wall_mesh_size_m: float = 0.002  # element size on the body surface
    farfield_mesh_size_m: float = 0.03  # element size at the domain boundary
    wall_refine_distance_m: float = 0.02  # distance over which size grows wall→farfield

    def __post_init__(self) -> None:
        if min(self.pad_upstream_m, self.pad_downstream_m, self.pad_lateral_m) <= 0:
            raise ValueError("domain padding must be positive")
        if min(self.wall_mesh_size_m, self.farfield_mesh_size_m) <= 0:
            raise ValueError("mesh sizes must be positive")
        if self.farfield_mesh_size_m < self.wall_mesh_size_m:
            raise ValueError("farfield_mesh_size_m must be >= wall_mesh_size_m")


@dataclass(frozen=True)
class VolumeMeshResult:
    """Summary of a written 3D volume mesh."""

    path: Path
    n_nodes: int
    n_elements: int
    markers: tuple[str, ...]
    meta: dict[str, float] = field(default_factory=dict)


def _domain_box(bb: tuple[float, ...], p: VolumeMeshParams) -> int:
    """Add the padded flow domain box around body bbox ``bb`` (x0,y0,z0,x1,y1,z1)."""
    x0 = bb[0] - p.pad_lateral_m
    y0 = bb[1] - p.pad_lateral_m
    z0 = bb[2] - p.pad_upstream_m
    x1 = bb[3] + p.pad_lateral_m
    y1 = bb[4] + p.pad_lateral_m
    z1 = bb[5] + p.pad_downstream_m
    return gmsh.model.occ.addBox(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0)


def _wall_refinement_field(fan_surfaces: list[int], p: VolumeMeshParams) -> None:
    """Distance-from-wall → Threshold size field: fine at the body, coarse far."""
    dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist, "SurfacesList", fan_surfaces)
    thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thr, "InField", dist)
    gmsh.model.mesh.field.setNumber(thr, "SizeMin", p.wall_mesh_size_m)
    gmsh.model.mesh.field.setNumber(thr, "SizeMax", p.farfield_mesh_size_m)
    gmsh.model.mesh.field.setNumber(thr, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(thr, "DistMax", p.wall_refine_distance_m)
    gmsh.model.mesh.field.setAsBackgroundMesh(thr)


def _tag_markers(fluid_vols: list[int], box_bb: tuple[float, ...], tol: float) -> tuple[str, ...]:
    """Classify boundary surfaces: on a domain-box face → farfield, else fan_surface."""
    x0, y0, z0, x1, y1, z1 = box_bb
    fan: set[int] = set()
    farfield: set[int] = set()
    boundary = gmsh.model.getBoundary([(3, v) for v in fluid_vols], oriented=False, combined=True)
    for _, surf in boundary:
        cx, cy, cz = gmsh.model.occ.getCenterOfMass(2, surf)[:3]
        on_box = (
            abs(cx - x0) < tol
            or abs(cx - x1) < tol
            or abs(cy - y0) < tol
            or abs(cy - y1) < tol
            or abs(cz - z0) < tol
            or abs(cz - z1) < tol
        )
        (farfield if on_box else fan).add(surf)

    gmsh.model.addPhysicalGroup(3, fluid_vols, name="fluid")
    written: list[str] = []
    for tags, name in ((fan, FAN_SURFACE_MARKER), (farfield, FARFIELD_MARKER)):
        if tags:
            gmsh.model.addPhysicalGroup(2, sorted(tags), name=name)
            written.append(name)
    return tuple(written)


def build_volume_mesh(
    step_path: str | Path, params: VolumeMeshParams, out_path: str | Path
) -> VolumeMeshResult:
    """Mesh the external flow around the STEP body; write ``.su2`` to ``out_path``.

    Imports the body, subtracts it from a padded domain box, wall-refines, and
    3D tet-meshes. Raises ``RuntimeError`` if the STEP has no solids or the cut
    leaves no fluid volume.
    """
    step_path = Path(step_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("volume")

        entities = gmsh.model.occ.importShapes(str(step_path))
        gmsh.model.occ.synchronize()
        body_vols = [t for d, t in entities if d == 3]
        if not body_vols:
            raise RuntimeError(f"{step_path}: no 3D solids to mesh")

        body_bb = gmsh.model.getBoundingBox(-1, -1)
        box = _domain_box(body_bb, params)
        gmsh.model.occ.synchronize()
        fluid, _ = gmsh.model.occ.cut([(3, box)], [(3, v) for v in body_vols], removeTool=True)
        gmsh.model.occ.synchronize()
        fluid_vols = [t for d, t in fluid if d == 3]
        if not fluid_vols:  # pragma: no cover - unreachable: padding always leaves fluid
            raise RuntimeError(f"{step_path}: cut left no fluid volume")

        box_bb = gmsh.model.getBoundingBox(3, fluid_vols[0])
        tol = min(params.wall_mesh_size_m, 1e-4)
        markers = _tag_markers(fluid_vols, box_bb, tol)
        fan_surfaces = [
            s for d, s in gmsh.model.getBoundary([(3, v) for v in fluid_vols], oriented=False)
        ]
        # refine only around the body (interior surfaces), not the box faces
        _wall_refinement_field([s for s in fan_surfaces if _is_interior(s, box_bb, tol)], params)

        gmsh.model.mesh.generate(3)
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        _, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
        n_elements = sum(len(t) for t in elem_tags)

        gmsh.write(str(out_path))
        return VolumeMeshResult(
            path=out_path,
            n_nodes=len(node_tags),
            n_elements=n_elements,
            markers=markers,
            meta={"n_bodies": float(len(body_vols))},
        )
    finally:
        gmsh.finalize()


def _is_interior(surf: int, box_bb: tuple[float, ...], tol: float) -> bool:
    x0, y0, z0, x1, y1, z1 = box_bb
    cx, cy, cz = gmsh.model.occ.getCenterOfMass(2, surf)[:3]
    on_box = (
        abs(cx - x0) < tol
        or abs(cx - x1) < tol
        or abs(cy - y0) < tol
        or abs(cy - y1) < tol
        or abs(cz - z0) < tol
        or abs(cz - z1) < tol
    )
    return not on_box
