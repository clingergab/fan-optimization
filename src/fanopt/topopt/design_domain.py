"""Design-domain bookkeeping for 3D solid topology optimization of a blade mesh.

Turns a raw tet mesh of the blade solid into the arrays SIMP needs:

- **element centroids + volumes** (unstructured — tets have unequal volume, which the
  volume-weighted OC and density filter must respect);
- a **frozen mask** — the aero skin (a shell of fixed thickness on the top/bottom panel
  faces) is ``ρ ≡ 1`` and OUT of the design set, so TO can never punch a hole through the
  air-pushing surface. The carvable design region is everything else (rib + interior);
- a **density filter** (unstructured cone filter — the anti-checkerboard regularizer);
- a **per-element material orientation** field mapping the transversely-isotropic PETG
  frame (weak build axis = blade **width/tangential**) onto each element of the curved
  surface-of-revolution. A cheaper single global frame is offered for a v1 approximation.

Pure numpy + scipy.spatial (KDTree) — no mesh library, no FEA dependency; every function
takes plain arrays so it is testable on a hand-built element set. The blade's CAD frame
has the pin on **+z**, radial extent about **+x**, thickness/bow along **z**.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

__all__ = [
    "DesignDomain",
    "tet_centroids_and_volumes",
    "build_density_filter_unstructured",
    "classify_frozen_skin",
    "revolution_material_frames",
    "global_material_frame",
]

_PIN_AXIS = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class DesignDomain:
    """Everything SIMP needs about one blade mesh's element field.

    ``frozen_mask`` (aero skin, always solid) and ``design_mask`` (carvable) partition the
    elements; ``support_node_ids`` are clamped (the hub/pivot boss); ``material_frames`` is
    the per-element rotation (material→global) for the orthotropic stiffness.
    """

    element_centroids: np.ndarray  # (n, 3)
    element_volumes: np.ndarray  # (n,)
    frozen_mask: np.ndarray  # (n,) bool — aero skin, ρ≡1, non-design
    design_mask: np.ndarray  # (n,) bool — carvable
    support_node_ids: np.ndarray  # (k,) int
    material_frames: np.ndarray | None = None  # (n, 3, 3) or None
    meta: dict = field(default_factory=dict)


def tet_centroids_and_volumes(nodes: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroids (mean of 4 vertices) and signed-free volumes of tetrahedra.

    ``nodes`` is ``(n_nodes, 3)``, ``tets`` is ``(n_elem, 4)`` int. Volume is
    ``|det(v1-v0, v2-v0, v3-v0)| / 6`` (always positive — orientation-agnostic).
    """
    p = nodes[tets]  # (n_elem, 4, 3)
    centroids = p.mean(axis=1)
    e = p[:, 1:, :] - p[:, :1, :]  # (n_elem, 3, 3): edges from vertex 0
    volumes = np.abs(np.linalg.det(e)) / 6.0
    return centroids, volumes


def build_density_filter_unstructured(
    centroids: np.ndarray,
    r_min: float,
    element_volumes: np.ndarray | None = None,
    *,
    block_rows: int = 20_000,
) -> csr_matrix:
    """Row-normalized cone density filter on element centroids (``filtered = W @ ρ``).

    ``H_ij = max(0, r_min − ‖c_i − c_j‖)`` optionally volume-weighted by ``V_j`` (correct
    for unequal tet volumes), then each row is normalized so the filter is a weighted
    average (mesh-independent, preserves a uniform field). This is the standard SIMP
    regularizer that removes checkerboarding and imposes a minimum feature size ``r_min``.

    **Memory-bounded build.** At a fine mesh with a fixed physical ``r_min`` each element has
    hundreds of neighbours, so the whole filter is ~10⁸–10⁹ nonzeros. Building it in one shot
    (a full COO, then a concatenate, then ``tocsr``, then a ``diags @`` product) materialises
    several transient copies of that whole structure and spikes RAM into the tens of GB. Instead
    this fills a preallocated CSR **block by block** (``block_rows`` query points at a time via
    ``sparse_distance_matrix`` against the full tree), so the transient is one block's pair set,
    not the whole matrix — peak ≈ the output CSR itself. A first cheap pass (neighbour counts only,
    ``return_length``) sizes ``indptr`` so nothing is over-allocated or concatenated.
    """
    n = len(centroids)
    c = np.ascontiguousarray(centroids, dtype=float)
    tree = cKDTree(c)
    vols = None if element_volumes is None else np.asarray(element_volumes, dtype=float)

    # Pass 1: neighbour count per row (incl. self), memory-cheap — just an int array of length n.
    counts = np.asarray(tree.query_ball_point(c, r_min, return_length=True), dtype=np.int64)
    indices = np.empty(int(counts.sum()), dtype=np.int32)
    data = np.empty(int(counts.sum()), dtype=np.float64)
    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0

    # Pass 2: fill the CSR one row-block at a time; the only large transient is this block's COO.
    pos = 0
    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)
        rows_here = stop - start
        dmat = cKDTree(c[start:stop]).sparse_distance_matrix(tree, r_min, output_type="coo_matrix")
        loc, col = dmat.row, dmat.col  # loc: local row in [0, rows_here); col: global neighbour
        off = (start + loc) != col  # drop the zero-distance self pair; re-added explicitly below
        loc, col, w = loc[off], col[off], r_min - dmat.data[off]
        if vols is not None:
            w = w * vols[col]  # volume-weight by the neighbour (column) j
        self_w = np.full(rows_here, r_min) if vols is None else r_min * vols[start:stop]
        loc = np.concatenate([loc, np.arange(rows_here)])
        col = np.concatenate([col, np.arange(start, stop)])
        w = np.concatenate([w, self_w])
        order = np.argsort(loc, kind="stable")  # group by row so each row's entries are contiguous
        loc, col, w = loc[order], col[order], w[order]
        seg = np.bincount(loc, minlength=rows_here)
        w = w / np.repeat(np.bincount(loc, weights=w, minlength=rows_here), seg)  # per-row normalize
        indices[pos : pos + col.size] = col
        data[pos : pos + col.size] = w
        indptr[start + 1 : stop + 1] = pos + np.cumsum(seg)
        pos += col.size

    return csr_matrix((data[:pos], indices[:pos], indptr), shape=(n, n))


def classify_frozen_skin(
    element_centroids: np.ndarray, aero_surface_points: np.ndarray, skin_thickness_m: float
) -> np.ndarray:
    """Boolean mask of elements within ``skin_thickness_m`` of the aero surface.

    These form the frozen (always-solid) skin that keeps the air-pushing faces intact;
    the complement is the carvable design region. Distance is to the nearest supplied
    aero-surface point (the mesher passes the top/bottom panel facet nodes — NOT the rib
    flanks or hub, so only the true aero skin is frozen).
    """
    tree = cKDTree(aero_surface_points)
    dist, _ = tree.query(element_centroids)
    return dist <= skin_thickness_m


def _frame_from_radial(radial: np.ndarray, pin: np.ndarray) -> np.ndarray:
    """Right-handed material frame rows ``[radial, thickness, width]`` (width = weak axis).

    Material axis 1 = radial (strong, base-to-tip), axis 2 = through-thickness (strong,
    ≈ −pin), axis 3 = tangential/width (weak build direction). Returns a proper rotation
    (rows are the material axes in global coords) suitable for ``rotate_stiffness``.
    """
    width = np.cross(pin, radial)  # circumferential (tangential) = weak build axis
    width /= np.linalg.norm(width)
    thickness = np.cross(width, radial)  # = −pin, the strong through-thickness axis
    return np.stack([radial, thickness, width])


def revolution_material_frames(centroids: np.ndarray, pin_axis: np.ndarray = _PIN_AXIS) -> np.ndarray:
    """Per-element material frames for the surface of revolution about ``pin_axis``.

    At each element the radial direction is the centroid projected off the pin axis; the
    weak build axis follows the local tangential (width) direction. This is the physically
    faithful field for the curved blade (radial/width rotate with angle θ), versus the
    single :func:`global_material_frame` approximation.
    """
    pin = np.asarray(pin_axis, dtype=float)
    pin = pin / np.linalg.norm(pin)
    c = np.asarray(centroids, dtype=float)
    radial = c - np.outer(c @ pin, pin)  # remove the pin-axis component
    norm = np.linalg.norm(radial, axis=1, keepdims=True)
    # Elements exactly on the axis have no radial direction; fall back to a fixed one.
    degenerate = (norm[:, 0] == 0.0)
    radial = np.where(norm == 0.0, np.array([1.0, 0.0, 0.0]), radial / np.where(norm == 0.0, 1.0, norm))
    frames = np.stack([_frame_from_radial(r, pin) for r in radial])
    if degenerate.any():  # pragma: no cover - blade elements are all at r ≥ HUB_RADIUS
        frames[degenerate] = global_material_frame(pin_axis=pin)
    return frames


def global_material_frame(
    radial_dir: np.ndarray = np.array([1.0, 0.0, 0.0]), pin_axis: np.ndarray = _PIN_AXIS
) -> np.ndarray:
    """Single material frame (v1 approximation): weak axis = width at the blade centreline.

    Exact only where the local radial equals ``radial_dir`` (θ = 0); drifts by the blade
    half-angle at the tangential edges. Cheaper than the per-element field and adequate for
    a first structural pass.
    """
    pin = np.asarray(pin_axis, dtype=float)
    pin = pin / np.linalg.norm(pin)
    radial = np.asarray(radial_dir, dtype=float)
    radial = radial - (radial @ pin) * pin
    radial /= np.linalg.norm(radial)
    return _frame_from_radial(radial, pin)
