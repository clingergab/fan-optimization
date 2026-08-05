"""Turn a topology-optimized density field into a printable watertight surface mesh + STL.

The TO output is a per-tet density ``ρ`` on a scattered centroid cloud (``<name>_density.npy`` +
``<name>_centroids.npy``) — not a printable solid. This module does the standard TO→print conversion:
voxelize the scattered densities onto a regular grid, then marching-cubes the ``ρ = 0.5`` iso-surface
into a watertight triangle mesh, and write it as a binary STL you can slice.

The ``rmin`` density filter used during TO already guarantees a printable minimum feature size, so the
retained material is manufacturable; this module only changes representation (field → surface), it does
not alter the geometry. Pure numpy + scipy + scikit-image (marching cubes); no CadQuery.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

__all__ = ["carved_blade_mesh", "write_binary_stl", "laplacian_smooth"]


def laplacian_smooth(verts: np.ndarray, faces: np.ndarray, iterations: int = 8, lam: float = 0.5):
    """Laplacian-smooth a triangle mesh — relaxes the marching-cubes staircase for a cleaner render.

    Each vertex is nudged ``lam`` of the way toward the mean of its edge neighbours, ``iterations`` times.
    Topology (``faces``) is unchanged. Pure numpy/scipy; a mild smooth (default) is cosmetic, not a
    geometry change — do NOT smooth the mesh you dimension-check or slice for tight tolerances.
    """
    verts = np.asarray(verts, dtype=float)
    n = len(verts)
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])  # undirected
    adj = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    adj.data[:] = 1.0
    deg = np.asarray(adj.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    v = verts.copy()
    for _ in range(iterations):
        v = v + lam * ((adj @ v) / deg[:, None] - v)
    return v


def carved_blade_mesh(
    density: np.ndarray,
    centroids: np.ndarray,
    *,
    voxel_pitch_m: float,
    level: float = 0.5,
    outside_tol_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Watertight surface of the retained material (``ρ ≥ level``) via marching cubes.

    Voxelize the scattered tet densities onto a regular grid at ``voxel_pitch_m`` — each voxel takes
    the nearest centroid's density, or **0 (void)** beyond ``outside_tol_m`` so the blade doesn't
    balloon past its own mesh — then extract the ``level`` iso-surface. The grid is padded with void on
    all sides so the surface closes. Returns ``(verts (V,3) float, faces (F,3) int)`` in world metres.

    ``voxel_pitch_m`` sets the surface resolution; ~half the TO mesh size resolves the carved detail.
    """
    centroids = np.asarray(centroids, dtype=float)
    density = np.asarray(density, dtype=float)
    tree = cKDTree(centroids)

    # "Void outside the mesh" must key off the SOURCE cloud spacing, not the output voxel pitch: an
    # interior grid node can sit up to (√3/2)·spacing ≈ 0.87·spacing from its nearest centroid (a cubic
    # cell's body-centre) yet be deep inside solid material. A pitch-only tolerance therefore punches
    # speckle cavities once the voxel pitch drops below the cloud spacing. Keying tol to ~0.9·spacing
    # covers that worst case; the cost is the ρ=0.5 surface may sit up to ~tol proud of the true boundary
    # (mesh-resolution fidelity) — the smooth-CAD-skin route is the refinement for a production surface.
    if outside_tol_m is None:
        sample = centroids if len(centroids) <= 4000 else centroids[
            np.linspace(0, len(centroids) - 1, 4000).astype(int)
        ]
        spacing = float(np.median(tree.query(sample, k=2)[0][:, 1]))  # median nearest-neighbour distance
        tol = max(1.5 * voxel_pitch_m, 0.9 * spacing)
    else:
        tol = outside_tol_m

    lo = centroids.min(axis=0) - 2.0 * (voxel_pitch_m + tol)  # pad past tol so the iso-surface closes
    hi = centroids.max(axis=0) + 2.0 * (voxel_pitch_m + tol)
    dims = (np.ceil((hi - lo) / voxel_pitch_m).astype(int) + 1)
    axes = [lo[d] + np.arange(dims[d]) * voxel_pitch_m for d in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    dist, idx = tree.query(grid)
    field = np.where(dist <= tol, density[idx], 0.0).reshape(dims)  # void outside the mesh

    verts, faces, _normals, _values = marching_cubes(field, level=level, spacing=(voxel_pitch_m,) * 3)
    return verts + lo, faces  # grid-local coords -> world coords


def write_binary_stl(
    verts: np.ndarray, faces: np.ndarray, path: str | Path, *, scale: float = 1000.0
) -> Path:
    """Write ``(verts, faces)`` as a binary STL (facet normals from the triangle winding).

    ``scale`` converts the input coordinates to STL units. This codebase is SI (**metres**), but STL is
    conventionally **unitless and read as millimetres** by every mainstream slicer — so the default
    ``scale=1000`` writes millimetres, and an unscaled export would import 1000× too small. Pass
    ``scale=1.0`` only if ``verts`` are already in millimetres.
    """
    verts = (np.asarray(verts, dtype=np.float64) * scale).astype(np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    tris = verts[faces]  # (F, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, norms, out=np.zeros_like(n), where=norms > 0).astype(np.float32)

    rec = np.zeros(len(faces), dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("attr", "<u2")]))
    rec["n"] = n
    rec["v"] = tris.reshape(len(faces), 9)
    path = Path(path)
    path.write_bytes(b"\x00" * 80 + struct.pack("<I", len(faces)) + rec.tobytes())
    return path
