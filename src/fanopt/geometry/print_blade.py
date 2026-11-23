"""Build the printable blade solid from a Stage-4 TO result, without inflating or hollowing it.

The printed solid is the union of the TO tets the optimizer counts as solid (``ρ ≥ 0.5``, the same
threshold the TO's own solid-only stress readout uses), rasterized exactly on the TO's own tet mesh, so
every outer surface sits on the CAD design boundary. On top of that it adds only: the TO's own frozen
aero skin re-imposed at voxel resolution (seals element-edge pinholes), and the operator-approved print
features — the pivot hub (solid within the clamped hub radius, the TO's support region) with the pin
bore, the LE/TE rib walls restored to a fixed width, and a thin tip-end cap — then drops floating specks.
All of it lives inside the TO domain, so the part never grows past the design envelope. The surface mesh
then gets exact CAD solids (:func:`apply_cad_solids`): the flat boss rebuilt as an exact slab between two
planes (voxel ripple on the boss faces would otherwise add to every layer of the folded stack) and,
optionally, the hold-detent bumps (added) and recesses (cut). :func:`boss_slab_at_junction_m` sizes and
places that slab to match the blade right where it leaves the boss, so neighbouring blades touch there.

Frames: the TO/CAD frame has the pin on +z, radial along +x, chord along y (LE/TE side walls face ±y).
The print frame stands the blade on its −y edge (on-edge): the −y side wall is rotated about the pin
axis until it is parallel to x, then ``(x, y, z) → (x, −z, y)`` puts the chord vertical and the pin
axis along print −y. numpy + scipy + scikit-image (marching cubes) + manifold3d (manifold-safe simplify).
"""

from __future__ import annotations

from dataclasses import dataclass

import manifold3d
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import marching_cubes

from fanopt.geometry.blade_detent import surface_z_range

__all__ = [
    "VoxelGrid",
    "BladePrintFeatures",
    "PrintBlade",
    "build_print_blade",
    "apply_cad_solids",
    "boss_slab_at_junction_m",
    "drop_sliver_shells",
    "simplify_manifold",
    "SOLID_DENSITY_LEVEL",
    "side_wall_planes",
    "edge_leveling_transform",
    "grid_for_points",
    "rasterize_tets",
    "boundary_shell",
    "keep_largest_component",
    "solid_with_print_features",
    "solid_surface_mesh",
    "merge_vertices",
    "mesh_volume",
    "layer_thin_fraction",
    "bed_contact_fraction",
]

# Same cut the TO uses to call an element "solid" (blade_topopt.SOLID_STRESS_THRESHOLD).
SOLID_DENSITY_LEVEL: float = 0.5
_CODE_SCALE = 254.0  # uint8 code: 0 = outside the domain, 1 + round(ρ·254) inside


@dataclass(frozen=True)
class VoxelGrid:
    """Regular grid of sample points ``lo + i·pitch`` (print frame, metres)."""

    lo: np.ndarray
    pitch: float
    shape: tuple[int, int, int]

    def points(self, idx: np.ndarray) -> np.ndarray:
        return self.lo + np.asarray(idx, dtype=float) * self.pitch


@dataclass(frozen=True)
class BladePrintFeatures:
    """The print-prep additions, all in the TO/CAD frame (metres)."""

    hub_radius_m: float  # solid within this cylindrical radius of the pin axis (TO clamp region)
    bore_radius_m: float  # pin bore, void within this radius of the pin axis
    rib_width_m: float  # LE/TE rib wall restored solid to this depth from each side wall
    tip_cap_m: float  # solid layer this thick behind the radial tip face
    tip_x_m: float  # radial position of the tip end face
    skin_m: float  # the TO's frozen aero-skin depth, held solid under every outer surface


def side_wall_planes(
    face_centers: np.ndarray, face_normals: np.ndarray, x_range_m: tuple[float, float]
) -> tuple[tuple[np.ndarray, float], tuple[np.ndarray, float]]:
    """Least-squares planes of the −y and +y side walls, as ``(unit_outward_normal, offset)``.

    A side-wall facet has a y-dominant outward normal; only facets with ``x`` in ``x_range_m`` are
    used (excludes the rounded hub and the tip face). A point ``p`` is ``n·p − offset`` outside the wall.
    """
    c = np.asarray(face_centers, dtype=float)
    n = np.asarray(face_normals, dtype=float)
    span = (c[:, 0] >= x_range_m[0]) & (c[:, 0] <= x_range_m[1])
    y_dom = np.abs(n[:, 1]) >= np.maximum(np.abs(n[:, 0]), np.abs(n[:, 2]))
    planes = []
    for sign in (-1.0, 1.0):
        pts = c[span & y_dom & (np.sign(n[:, 1]) == sign)]
        if len(pts) < 3:
            raise ValueError("not enough side-wall facets to fit a plane")
        # y = a + b·x + e·z  →  normal ∝ (−b, 1, −e) oriented outward (sign of y)
        a_mat = np.column_stack([np.ones(len(pts)), pts[:, 0], pts[:, 2]])
        a, b, e = np.linalg.lstsq(a_mat, pts[:, 1], rcond=None)[0]
        normal = sign * np.array([-b, 1.0, -e])
        scale = np.linalg.norm(normal)
        planes.append((normal / scale, sign * a / scale))
    return planes[0], planes[1]


def edge_leveling_transform(minus_y_plane: tuple[np.ndarray, float]) -> np.ndarray:
    """3×3 rotation TO frame → print frame that lays the −y side wall flat (on-edge, chord vertical).

    Rotates about the pin axis (+z) so the wall's in-plane direction becomes parallel to +x, then maps
    ``(x, y, z) → (x, −z, y)``: print +z = chord (up from the bed), print −y = pin axis.
    """
    normal, _ = minus_y_plane
    theta = np.arctan2(normal[0], -normal[1])  # angle of the wall's in-plane direction (−n_y, n_x)
    c, s = np.cos(-theta), np.sin(-theta)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    swap = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    return swap @ rz


def grid_for_points(points: np.ndarray, pitch_m: float, pad_voxels: int = 2) -> VoxelGrid:
    """Grid covering ``points`` with ``pad_voxels`` of empty margin on every side (closes the surface)."""
    p = np.asarray(points, dtype=float)
    lo = p.min(axis=0) - pad_voxels * pitch_m
    hi = p.max(axis=0) + pad_voxels * pitch_m
    nx, ny, nz = (int(v) for v in np.ceil((hi - lo) / pitch_m).astype(int) + 1)
    return VoxelGrid(lo=lo, pitch=float(pitch_m), shape=(nx, ny, nz))


def rasterize_tets(
    nodes: np.ndarray,
    tets: np.ndarray,
    density: np.ndarray,
    grid: VoxelGrid,
    max_batch_points: int = 8_000_000,
) -> np.ndarray:
    """Sample a per-tet density on ``grid``: uint8 code 0 outside every tet, ``1 + round(ρ·254)`` inside.

    Each grid point takes the density of the tet that contains it (barycentric test), so the domain
    boundary is the tet mesh's own boundary — no nearest-neighbour tolerance, no inflation.
    """
    nodes = np.asarray(nodes, dtype=float)
    tets = np.asarray(tets, dtype=np.int64)
    code = np.zeros(grid.shape, dtype=np.uint8)
    values = (
        1.0 + np.round(np.clip(np.asarray(density, dtype=float), 0.0, 1.0) * _CODE_SCALE)
    ).astype(np.uint8)
    p = nodes[tets]  # (n, 4, 3)
    lo_idx = np.ceil((p.min(axis=1) - grid.lo) / grid.pitch - 1e-9).astype(np.int64)
    hi_idx = np.floor((p.max(axis=1) - grid.lo) / grid.pitch + 1e-9).astype(np.int64)
    span = hi_idx - lo_idx + 1
    keep = np.all(span > 0, axis=1)
    order = np.nonzero(keep)[0]
    order = order[np.argsort(np.prod(span[order], axis=1), kind="stable")]
    edges = p[:, 1:, :] - p[:, :1, :]  # (n, 3, 3) rows = edge vectors
    shape = np.array(grid.shape)
    start = 0
    while start < len(order):
        end = min(start + 65536, len(order))
        while (
            end - start > 1
            and (end - start) * int(np.prod(span[order[start:end]].max(axis=0))) > max_batch_points
        ):
            end = start + (end - start) // 2
        sel = order[start:end]
        k = span[sel].max(axis=0)
        start = end
        off = np.stack(np.meshgrid(*[np.arange(v) for v in k], indexing="ij"), axis=-1).reshape(
            -1, 3
        )
        idx = lo_idx[sel][:, None, :] + off[None, :, :]  # (m, K, 3)
        valid = np.all(off[None, :, :] < span[sel][:, None, :], axis=2)
        valid &= np.all((idx >= 0) & (idx < shape), axis=2)
        pts = grid.lo + idx * grid.pitch
        inv = np.linalg.inv(np.transpose(edges[sel], (0, 2, 1)))  # columns = edges
        lam = np.einsum("mij,mkj->mki", inv, pts - p[sel][:, None, 0, :])
        inside = valid & np.all(lam >= -1e-9, axis=2) & (lam.sum(axis=2) <= 1.0 + 1e-9)
        hit = idx[inside]
        code[hit[:, 0], hit[:, 1], hit[:, 2]] = np.broadcast_to(values[sel][:, None], inside.shape)[
            inside
        ]
    return code


def boundary_shell(
    domain: np.ndarray, pitch_m: float, depth_m: float, slab: int = 256
) -> np.ndarray:
    """Domain voxels within ``depth_m`` of the domain's outer surface (x-slabs with a halo, exact EDT).

    A voxel centre sits ``edt − pitch/2`` from the surface it faces, so the cut is ``edt ≤ depth + pitch/2``.
    The halo exceeds the depth, so the false surface at each slab cut never reaches a kept voxel.
    """
    halo = int(np.ceil(depth_m / pitch_m)) + 2
    out = np.zeros_like(domain, dtype=bool)
    nx = domain.shape[0]
    for s in range(0, nx, slab):
        a, b = max(s - halo, 0), min(s + slab + halo, nx)
        block = np.pad(domain[a:b], 1, constant_values=False)
        dist = ndimage.distance_transform_edt(block)[1:-1, 1:-1, 1:-1] * pitch_m
        shell = domain[a:b] & (dist <= depth_m + 0.5 * pitch_m)
        e = min(s + slab, nx)
        out[s:e] = shell[s - a : e - a]
    return out


def keep_largest_component(solid: np.ndarray) -> tuple[np.ndarray, int]:
    """Drop every face-connected (6-neighbour) piece except the largest — floating specks can't print.

    Returns ``(cleaned, voxels_removed)``.
    """
    labels, n = ndimage.label(solid)
    if n <= 1:
        return solid, 0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep = labels == int(sizes.argmax())
    return keep, int(solid.sum() - keep.sum())


def solid_with_print_features(
    code: np.ndarray,
    grid: VoxelGrid,
    print_from_cad: np.ndarray,
    print_offset_m: np.ndarray,
    planes: tuple[tuple[np.ndarray, float], tuple[np.ndarray, float]],
    features: BladePrintFeatures,
    level: float = SOLID_DENSITY_LEVEL,
) -> tuple[np.ndarray, dict[str, int]]:
    """Print solid: TO-solid ∪ frozen skin ∪ hub ∪ rib walls ∪ tip cap, minus the bore — all inside the domain.

    The skin term re-imposes the TO's own frozen aero shell at voxel resolution: element-wise ``ρ ≥ 0.5``
    leaves pinholes where a carved tet meets the surface along an edge between two frozen face tets.
    Floating specks are dropped last. ``print = print_from_cad @ cad + print_offset_m``. Returns
    ``(solid, voxel_counts)``: voxels each feature switched void→solid (bore: solid→void; islands: dropped).
    """
    threshold = 1 + int(np.round(level * _CODE_SCALE))
    solid = code >= threshold
    counts = {"to_solid": int(solid.sum())}
    idx = np.argwhere(code > 0)
    cad = (
        grid.points(idx) - print_offset_m
    ) @ print_from_cad  # inverse of a rotation is its transpose
    r_pin = np.hypot(cad[:, 0], cad[:, 1])
    in_span = (r_pin >= features.hub_radius_m) & (cad[:, 0] <= features.tip_x_m)
    rib = np.zeros(len(idx), dtype=bool)
    for normal, offset in planes:
        rib |= (offset - cad @ normal) <= features.rib_width_m
    shell = boundary_shell(code > 0, grid.pitch, features.skin_m)
    masks = {
        "skin": shell[idx[:, 0], idx[:, 1], idx[:, 2]],
        "hub": r_pin <= features.hub_radius_m,
        "rib": rib & in_span,
        "tip_cap": cad[:, 0] >= features.tip_x_m - features.tip_cap_m,
    }
    for name, m in masks.items():
        target = idx[m]
        before = solid[target[:, 0], target[:, 1], target[:, 2]]
        counts[name] = int((~before).sum())
        solid[target[:, 0], target[:, 1], target[:, 2]] = True
    bore = idx[r_pin < features.bore_radius_m]
    counts["bore"] = int(solid[bore[:, 0], bore[:, 1], bore[:, 2]].sum())
    solid[bore[:, 0], bore[:, 1], bore[:, 2]] = False
    solid, counts["islands"] = keep_largest_component(solid)
    return solid, counts


def merge_vertices(
    verts: np.ndarray, faces: np.ndarray, decimals: int = 9
) -> tuple[np.ndarray, np.ndarray]:
    """Weld coincident vertices (slab seams) and drop faces that collapse."""
    key = np.round(np.asarray(verts, dtype=float), decimals)
    uniq, inverse = np.unique(key, axis=0, return_inverse=True)
    f = inverse.reshape(-1)[np.asarray(faces)]
    good = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    return uniq, f[good]


def solid_surface_mesh(
    solid: np.ndarray, grid: VoxelGrid, slab: int = 256, smooth_sigma_vox: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Iso-surface of a boolean solid (print frame, metres), extracted in x-slabs and welded.

    Slabs share one sample plane, so seam vertices are computed identically and weld exactly. An optional
    Gaussian blur (``smooth_sigma_vox`` voxels, applied with a halo so seams still match) rounds the voxel
    staircase and resolves diagonal-only voxel contacts that would otherwise give non-manifold edges.
    """
    nx = solid.shape[0]
    halo = int(np.ceil(4 * smooth_sigma_vox)) if smooth_sigma_vox > 0 else 0
    verts_all, faces_all, n_verts = [], [], 0
    for s in range(0, nx - 1, slab):
        e = min(s + slab, nx - 1)
        a, b = max(s - halo, 0), min(e + 1 + halo, nx)
        block = solid[a:b].astype(np.float32)
        if not block.any():
            continue
        if smooth_sigma_vox > 0:
            block = ndimage.gaussian_filter(block, smooth_sigma_vox, mode="constant")
        block = block[s - a : e + 1 - a]
        if block.max() < 0.5 or block.min() > 0.5:
            continue
        v, f, _n, _val = marching_cubes(block, level=0.5, spacing=(grid.pitch,) * 3)
        v[:, 0] += s * grid.pitch
        verts_all.append(v + grid.lo)
        faces_all.append(f + n_verts)
        n_verts += len(v)
    if not verts_all:
        raise ValueError("solid is empty")
    return merge_vertices(np.vstack(verts_all), np.vstack(faces_all))


def mesh_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Enclosed volume (absolute value of the signed divergence-theorem sum)."""
    t = np.asarray(verts, dtype=float)[np.asarray(faces)]
    return abs(float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum()) / 6.0)


def layer_thin_fraction(
    solid: np.ndarray, pitch_m: float, min_width_m: float, layer_axis: int = 2
) -> np.ndarray:
    """Per layer, the fraction of solid cross-section area thinner than ``min_width_m``.

    A pixel is "thin" when the local wall width — ``2·dt − pitch`` at the wall's ridge, spread over the
    wall by a max filter (``dt`` = in-layer distance to the nearest void pixel centre) — is under
    ``min_width_m``: the slicer would have to lay a wall narrower than the nozzle there. ``2·dt − pitch``
    is exact for odd pixel widths and one pixel low for even ones, so the check errs toward flagging.
    """
    out = np.zeros(solid.shape[layer_axis])
    radius_px = int(np.ceil(min_width_m / pitch_m))
    for k in range(solid.shape[layer_axis]):
        sl = np.take(solid, k, axis=layer_axis)
        n = int(sl.sum())
        if n == 0:
            continue
        dt = ndimage.distance_transform_edt(sl) * pitch_m
        width = 2.0 * ndimage.maximum_filter(dt, size=2 * radius_px + 1) - pitch_m
        out[k] = float((sl & (width < min_width_m)).sum()) / n
    return out


def bed_contact_fraction(solid: np.ndarray, layers: int = 2) -> float:
    """Fraction of the part's x-stations whose lowest solid voxel is within ``layers`` of the bed layer."""
    col = solid.any(axis=1)  # (x, z)
    present = col.any(axis=1)
    if not present.any():
        return 0.0
    first = col.argmax(axis=1)[present]
    return float((first <= first.min() + layers).mean())


def simplify_manifold(
    verts: np.ndarray, faces: np.ndarray, tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Manifold-preserving simplification (surface moves ≤ ``tol``), returned with outward winding.

    Raises ``ValueError`` if the input is not a valid closed 2-manifold.
    """
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if _signed_volume(v, f) < 0:
        f = f[:, ::-1]
    mesh = manifold3d.Mesh(vert_properties=v.astype(np.float32), tri_verts=f.astype(np.uint32))
    man = manifold3d.Manifold(mesh)
    if man.status() != manifold3d.Error.NoError:
        raise ValueError(f"mesh is not a closed 2-manifold: {man.status().name}")
    out = man.simplify(tol).to_mesh()
    return np.asarray(out.vert_properties, dtype=np.float64)[:, :3], np.asarray(
        out.tri_verts, dtype=np.int64
    )


def boss_slab_at_junction_m(
    nodes: np.ndarray,
    facets: np.ndarray,
    boss_radius_m: float,
    design_half_height_m: float,
    *,
    band_m: float = 1.0e-3,
    step_deg: float = 0.5,
) -> tuple[float, float]:
    """``(z_low, z_high)`` of a boss slab matching the blade where it leaves the boss.

    Samples the blade on a ring band just outside ``boss_radius_m`` (the blade is dished, so it must be
    sampled right at the boss edge); the slab is the thickest sample, centred on the median mid-surface
    there and shifted (if needed) to stay inside the design's own boss faces ``|z| ≤
    design_half_height_m``. Raises if the band never crosses the blade.
    """
    edges = nodes[facets] - nodes[np.roll(facets, 1, axis=1)]
    search = max(3.0e-3, float(np.linalg.norm(edges, axis=2).max()))
    thick, mid = [], []
    for r in boss_radius_m + band_m * (np.arange(4) + 0.5) / 4:
        for t in np.radians(np.arange(-180.0, 180.0, step_deg)):
            try:
                zb, zt = surface_z_range(nodes, facets, r * np.cos(t), r * np.sin(t), search)
            except ValueError:
                continue
            thick.append(zt - zb)
            mid.append((zt + zb) / 2.0)
    if not thick:
        raise ValueError("the junction band never crosses the blade")
    s = float(max(thick))
    if s > 2.0 * design_half_height_m:
        raise ValueError("blade at the junction is thicker than the design boss")
    lo = float(np.clip(np.median(mid) - s / 2.0, -design_half_height_m, design_half_height_m - s))
    return lo, lo + s


def _spheres(rows: np.ndarray, segments: int) -> list[manifold3d.Manifold]:
    return [
        manifold3d.Manifold.sphere(float(r), segments).translate((x, y, z)) for x, y, z, r in rows
    ]


def apply_cad_solids(
    verts: np.ndarray,
    faces: np.ndarray,
    print_from_cad: np.ndarray,
    print_offset_m: np.ndarray,
    *,
    boss: tuple[float, float, float, float] | None = None,
    add_spheres: np.ndarray | None = None,
    cut_spheres: np.ndarray | None = None,
    segments: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact CAD-frame solids applied to the print-frame mesh (manifold booleans).

    ``boss = (z_low, z_high, bore_radius, boss_radius)``: within ``boss_radius`` of the pin everything
    outside ``z_low ≤ z ≤ z_high`` is cut off and the ring ``bore < r ≤ boss_radius`` is filled — so the
    boss faces are exact planes one layer spacing apart. The blade outside the boss is untouched. Then ``add_spheres`` are
    unioned and ``cut_spheres`` subtracted (rows ``[x, y, z, radius]``, CAD frame).
    """
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    man = manifold3d.Manifold(
        manifold3d.Mesh(vert_properties=v.astype(np.float32), tri_verts=f.astype(np.uint32))
    )
    if man.status() != manifold3d.Error.NoError:
        raise ValueError(f"mesh is not a closed 2-manifold: {man.status().name}")
    to_print = np.hstack([print_from_cad, np.asarray(print_offset_m, dtype=float).reshape(3, 1)])
    cad: list[manifold3d.Manifold] = []
    if boss is not None:
        z_lo, z_hi, bore_r, boss_r = boss
        tall = 0.05
        cyl = manifold3d.Manifold.cylinder
        beyond = cyl(tall, boss_r, circular_segments=4 * segments).translate(
            (0.0, 0.0, z_hi)
        ) + cyl(tall, boss_r, circular_segments=4 * segments).translate((0.0, 0.0, z_lo - tall))
        ring = cyl(z_hi - z_lo, boss_r, circular_segments=4 * segments) - cyl(
            z_hi - z_lo + 1e-3, bore_r, circular_segments=segments
        ).translate((0.0, 0.0, -5e-4))
        man = man - beyond.transform(to_print)
        man = man + ring.translate((0.0, 0.0, z_lo)).transform(to_print)
    if add_spheres is not None and len(add_spheres):
        cad = _spheres(np.asarray(add_spheres), segments)
        man = man + manifold3d.Manifold.batch_boolean(cad, manifold3d.OpType.Add).transform(
            to_print
        )
    if cut_spheres is not None and len(cut_spheres):
        cad = _spheres(np.asarray(cut_spheres), segments)
        man = man - manifold3d.Manifold.batch_boolean(cad, manifold3d.OpType.Add).transform(
            to_print
        )
    out = man.to_mesh()
    return np.asarray(out.vert_properties, dtype=np.float64)[:, :3], np.asarray(
        out.tri_verts, dtype=np.int64
    )


def drop_sliver_shells(
    verts: np.ndarray, faces: np.ndarray, min_volume_m3: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """Drop closed surface pieces enclosing less than ``min_volume_m3`` (simplification debris).

    Pieces are vertex-connected face sets; the blade and its sealed internal voids enclose real volume
    (a void counts negative) and are kept.
    """
    f = np.asarray(faces, dtype=np.int64)
    edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]]])
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(verts),) * 2)
    _n, label = connected_components(graph, directed=False)
    t = np.asarray(verts, dtype=np.float64)[f]
    per_face = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])) / 6.0
    vol = np.bincount(label[f[:, 0]], weights=per_face, minlength=_n)
    keep = np.abs(vol[label[f[:, 0]]]) >= min_volume_m3
    used = np.unique(f[keep])
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return np.asarray(verts)[used], remap[f[keep]]


def _signed_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    t = verts[faces]
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum()) / 6.0


@dataclass(frozen=True)
class PrintBlade:
    """A printable blade: surface mesh + the voxel solid it came from, in the print frame (metres)."""

    verts: np.ndarray
    faces: np.ndarray
    solid: np.ndarray
    grid: VoxelGrid
    print_from_cad: np.ndarray
    print_offset_m: np.ndarray
    voxel_counts: dict[str, int]


def build_print_blade(
    nodes: np.ndarray,
    tets: np.ndarray,
    density: np.ndarray,
    face_centers: np.ndarray,
    face_normals: np.ndarray,
    *,
    hub_radius_m: float,
    bore_radius_m: float,
    rib_width_m: float,
    tip_cap_m: float,
    skin_m: float,
    pitch_m: float,
    smooth_sigma_vox: float = 0.6,
    simplify_tol_m: float = 3.0e-5,
    boss_z_m: tuple[float, float] | None = None,
    boss_radius_m: float | None = None,
    bump_spheres: np.ndarray | None = None,
    recess_spheres: np.ndarray | None = None,
) -> PrintBlade:
    """TO tet mesh + per-tet density (CAD frame) → on-edge printable blade.

    ``face_centers``/``face_normals`` are the mesh's outward boundary facets (CAD frame), used to fit the
    LE/TE side-wall planes. The print frame is centred in x and y with the bed at z = 0.
    ``boss_z_m`` (the slab's two face heights) + ``boss_radius_m`` make the flat
    boss exact (see :func:`apply_cad_solids`); ``bump_spheres`` /
    ``recess_spheres`` (CAD rows ``[x, y, z, r]``) add / cut the hold detent.
    """
    nodes = np.asarray(nodes, dtype=float)
    tip_x = float(nodes[:, 0].max())
    planes = side_wall_planes(face_centers, face_normals, (hub_radius_m + 0.005, tip_x - 0.005))
    rot = edge_leveling_transform(planes[0])
    p = nodes @ rot.T
    offset = np.array(
        [
            -(p[:, 0].min() + p[:, 0].max()) / 2.0,
            -(p[:, 1].min() + p[:, 1].max()) / 2.0,
            -p[:, 2].min(),
        ]
    )
    p = p + offset
    grid = grid_for_points(p, pitch_m)
    code = rasterize_tets(p, tets, density, grid)
    features = BladePrintFeatures(
        hub_radius_m=hub_radius_m,
        bore_radius_m=bore_radius_m,
        rib_width_m=rib_width_m,
        tip_cap_m=tip_cap_m,
        tip_x_m=tip_x,
        skin_m=skin_m,
    )
    solid, counts = solid_with_print_features(code, grid, rot, offset, planes, features)
    counts["domain"] = int((code > 0).sum())
    del code
    verts, faces = solid_surface_mesh(solid, grid, smooth_sigma_vox=smooth_sigma_vox)
    verts, faces = simplify_manifold(verts, faces, simplify_tol_m)
    boss = None
    if boss_z_m is not None and boss_radius_m is not None:
        boss = (*boss_z_m, bore_radius_m, boss_radius_m)
    if boss is not None or bump_spheres is not None or recess_spheres is not None:
        verts, faces = apply_cad_solids(
            verts,
            faces,
            rot,
            offset,
            boss=boss,
            add_spheres=bump_spheres,
            cut_spheres=recess_spheres,
        )
    verts, faces = drop_sliver_shells(verts, faces)
    return PrintBlade(verts, faces, solid, grid, rot, offset, counts)
