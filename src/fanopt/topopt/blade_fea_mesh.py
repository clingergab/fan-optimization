"""Tet mesh of the SOLID blade for structural FEA, with region tags.

Unlike :mod:`fanopt.cfd.mesh` (which meshes the external *fluid* box around the body),
this meshes the blade **solid itself** and tags the regions the load cases and design
domain need:

- ``aero_skin`` (top / bottom panel faces) — the frozen, always-solid air-pushing surface;
- ``hub_boss`` nodes — clamped support (the pivot region);
- ``click_region`` facets — where the engagement load is applied.

Tagging is geometric on the extracted mesh (surface-facet normals + cylindrical radius),
not gmsh physical groups, so it is pure numpy and unit-testable without a mesher. Only
:func:`build_blade_fea_mesh` needs gmsh + CadQuery; the test module gates on those.

Blade CAD frame: pin on +z, radial extent about +x, thickness/bow along z — so the aero
faces have a z-dominant normal and the hub boss sits at small cylindrical radius.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cadquery as cq
import gmsh
import numpy as np

from fanopt.geometry.blade import RIB_TIP_RADIUS_M
from fanopt.geometry.blade_cad import make_blade_solid
from fanopt.geometry.schema import HUB_RADIUS_M

__all__ = [
    "FeaMeshParams",
    "BladeFeaMeshResult",
    "boundary_facets",
    "facet_geometry",
    "cylindrical_radius",
    "classify_aero_skin",
    "hub_support_nodes",
    "click_region_facets",
    "build_blade_fea_mesh",
]

_GMSH_TET_TYPE = 4  # gmsh element type id for the 4-node tetrahedron


@dataclass(frozen=True)
class FeaMeshParams:
    """Sizing + region-tag thresholds for the solid mesh (SI metres)."""

    mesh_size_m: float = 0.0015
    hub_radius_m: float = HUB_RADIUS_M  # nodes within this cylindrical radius are clamped
    tip_radius_m: float = RIB_TIP_RADIUS_M
    click_band_m: float = 0.010  # tip band treated as the click footprint (approximation)

    def __post_init__(self) -> None:
        if self.mesh_size_m <= 0:
            raise ValueError("mesh_size_m must be positive")


@dataclass(frozen=True)
class BladeFeaMeshResult:
    """Extracted tet mesh + geometric region tags for one blade."""

    nodes: np.ndarray  # (n_nodes, 3)
    tets: np.ndarray  # (n_elem, 4) int, 0-based
    aero_top_facets: np.ndarray  # (k, 3) int
    aero_bottom_facets: np.ndarray  # (k, 3) int
    click_facets: np.ndarray  # (k, 3) int
    support_node_ids: np.ndarray  # (s,) int
    meta: dict = field(default_factory=dict)


def boundary_facets(tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Surface triangles of a tet mesh + each one's opposite (interior) vertex.

    A triangular face on the boundary belongs to exactly one tet; interior faces are
    shared by two. Returns ``(facets (k,3), opposite_vertex (k,))`` with facet node order
    inherited from the owning tet (for outward-normal orientation).
    """
    tets = np.asarray(tets, dtype=int)
    faces = np.vstack([tets[:, [1, 2, 3]], tets[:, [0, 2, 3]], tets[:, [0, 1, 3]], tets[:, [0, 1, 2]]])
    opps = np.concatenate([tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]])
    keys = np.sort(faces, axis=1)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    on_boundary = counts[inv] == 1
    return faces[on_boundary], opps[on_boundary]


def facet_geometry(
    nodes: np.ndarray, facets: np.ndarray, opposite_vertex: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Facet ``(centers, outward_unit_normals, areas)``.

    If ``opposite_vertex`` is given, each normal is flipped to point away from that
    interior vertex (outward); otherwise the raw right-hand-rule normal is returned.
    """
    p0, p1, p2 = nodes[facets[:, 0]], nodes[facets[:, 1]], nodes[facets[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(cross, axis=1, keepdims=True)
    normals = cross / norm
    areas = 0.5 * norm[:, 0]
    centers = (p0 + p1 + p2) / 3.0
    if opposite_vertex is not None:
        outward = centers - nodes[opposite_vertex]
        flip = np.einsum("ij,ij->i", normals, outward) < 0.0
        normals[flip] *= -1.0
    return centers, normals, areas


def cylindrical_radius(points: np.ndarray) -> np.ndarray:
    """Radius from the pin axis (+z): ``√(x² + y²)``."""
    p = np.asarray(points, dtype=float)
    return np.hypot(p[:, 0], p[:, 1])


def classify_aero_skin(
    normals: np.ndarray, centers: np.ndarray, hub_radius_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Masks of ``(top, bottom)`` aero-skin facets — z-dominant normal, beyond the hub.

    The panel faces are the surfaces whose outward normal is dominated by ±z (the thin
    rib flanks and radial caps have in-plane normals). Facets inside the hub radius (the
    boss) are excluded so only the true air-pushing skin is frozen.
    """
    nz = np.abs(normals[:, 2])
    z_dominant = (nz >= np.abs(normals[:, 0])) & (nz >= np.abs(normals[:, 1]))
    beyond_hub = cylindrical_radius(centers) >= hub_radius_m
    aero = z_dominant & beyond_hub
    return aero & (normals[:, 2] > 0.0), aero & (normals[:, 2] < 0.0)


def hub_support_nodes(nodes: np.ndarray, hub_radius_m: float) -> np.ndarray:
    """Node ids within the hub/boss cylindrical radius — clamped in every load case."""
    return np.nonzero(cylindrical_radius(nodes) <= hub_radius_m)[0]


def click_region_facets(
    facets: np.ndarray, centers: np.ndarray, tip_radius_m: float, click_band_m: float
) -> np.ndarray:
    """Facets in the outer tip band (``r ≥ tip_radius − band``) — the click footprint.

    A radius-band approximation of the click engagement region; the exact footprint needs
    the click-mechanism geometry (a documented open gap).
    """
    return facets[cylindrical_radius(centers) >= tip_radius_m - click_band_m]


def _extract_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Pull ``(nodes, tets)`` arrays (0-based) from the current gmsh model."""
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.asarray(coords, dtype=float).reshape(-1, 3)
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
    etypes, _, enodes = gmsh.model.mesh.getElements(dim=3)
    blocks = [
        np.array([tag_to_idx[int(t)] for t in conn], dtype=int).reshape(-1, 4)
        for et, conn in zip(etypes, enodes)
        if et == _GMSH_TET_TYPE
    ]
    if not blocks:  # pragma: no cover - a solid always tet-meshes to >0 tets
        raise RuntimeError("mesh has no tetrahedra")
    return nodes, np.vstack(blocks)


def build_blade_fea_mesh(
    params, mesh_params: FeaMeshParams | None = None, out_path: str | Path | None = None
) -> BladeFeaMeshResult:
    """Mesh the blade solid and tag its aero skin, hub support, and click region.

    ``params`` is a :class:`fanopt.geometry.blade.BladeParams`. Exports the solid to STEP,
    tet-meshes it with gmsh at ``mesh_params.mesh_size_m``, extracts node/tet arrays, and
    runs the geometric taggers. ``out_path`` optionally writes the STEP for inspection.
    """
    mp = mesh_params or FeaMeshParams()
    solid = make_blade_solid(params)
    with tempfile.TemporaryDirectory() as tmp:
        step = Path(out_path) if out_path is not None else Path(tmp) / "blade.step"
        step.parent.mkdir(parents=True, exist_ok=True)
        cq.exporters.export(solid, str(step))
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("blade_fea")
            entities = gmsh.model.occ.importShapes(str(step))
            gmsh.model.occ.synchronize()
            if not [t for d, t in entities if d == 3]:  # pragma: no cover - make_blade_solid always yields a solid
                raise RuntimeError(f"{step}: no 3D solid to mesh")
            gmsh.option.setNumber("Mesh.MeshSizeMax", mp.mesh_size_m)
            gmsh.option.setNumber("Mesh.MeshSizeMin", mp.mesh_size_m * 0.5)
            gmsh.model.mesh.generate(3)
            nodes, tets = _extract_mesh()
        finally:
            gmsh.finalize()

    facets, opps = boundary_facets(tets)
    centers, normals, _ = facet_geometry(nodes, facets, opps)
    top, bottom = classify_aero_skin(normals, centers, mp.hub_radius_m)
    return BladeFeaMeshResult(
        nodes=nodes,
        tets=tets,
        aero_top_facets=facets[top],
        aero_bottom_facets=facets[bottom],
        click_facets=click_region_facets(facets, centers, mp.tip_radius_m, mp.click_band_m),
        support_node_ids=hub_support_nodes(nodes, mp.hub_radius_m),
        meta={"n_nodes": float(len(nodes)), "n_tets": float(len(tets))},
    )
