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
from scipy.sparse import coo_matrix, csr_matrix, diags
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
    centroids: np.ndarray, r_min: float, element_volumes: np.ndarray | None = None
) -> csr_matrix:
    """Row-normalized cone density filter on element centroids (``filtered = W @ ρ``).

    ``H_ij = max(0, r_min − ‖c_i − c_j‖)`` optionally volume-weighted by ``V_j`` (correct
    for unequal tet volumes), then each row is normalized so the filter is a weighted
    average (mesh-independent, preserves a uniform field). This is the standard SIMP
    regularizer that removes checkerboarding and imposes a minimum feature size ``r_min``.

    Built at the C level via ``sparse_distance_matrix`` — a per-element Python loop over
    neighbours blows up transient RAM at a fine mesh (~10s of GB from the intermediate lists
    at sub-mm resolution) and is the dominant setup-time cost; this stays in numpy/scipy.
    """
    n = len(centroids)
    tree = cKDTree(centroids)
    # Off-diagonal pairs within r_min as a COO (C-level; no per-point Python lists). The
    # zero-distance self pairs are the sparse blank, so they are added back explicitly below.
    dmat = tree.sparse_distance_matrix(tree, r_min, output_type="coo_matrix")
    off = dmat.row != dmat.col
    rows, cols, dist = dmat.row[off], dmat.col[off], dmat.data[off]
    w = r_min - dist  # cone weight, ≥ 0 since dist ≤ r_min
    self_w = np.full(n, r_min)  # self pair: dist 0 → weight r_min
    if element_volumes is not None:
        vols = np.asarray(element_volumes, dtype=float)
        w = w * vols[cols]  # volume-weight by the neighbour (column) j
        self_w = self_w * vols
    rows = np.concatenate([rows, np.arange(n)])
    cols = np.concatenate([cols, np.arange(n)])
    data = np.concatenate([w, self_w])
    unnormalized = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    row_sums = np.asarray(unnormalized.sum(axis=1)).ravel()
    return (diags(1.0 / row_sums) @ unnormalized).tocsr()


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
