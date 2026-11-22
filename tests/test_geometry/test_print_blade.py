"""Tests for the TO-result → printable blade builder (fanopt.geometry.print_blade)."""

from __future__ import annotations

import importlib.util
import itertools

import numpy as np
import pytest

for _dep in ("skimage", "manifold3d"):
    if importlib.util.find_spec(_dep) is None:  # pragma: no cover - env-dependent
        pytest.skip(f"{_dep} not installed", allow_module_level=True)

import manifold3d  # noqa: E402

from fanopt.geometry.print_blade import (  # noqa: E402
    BladePrintFeatures,
    VoxelGrid,
    apply_cad_solids,
    bed_contact_fraction,
    boundary_shell,
    build_print_blade,
    drop_sliver_shells,
    edge_leveling_transform,
    flat_boss_radius_m,
    grid_for_points,
    keep_largest_component,
    layer_thin_fraction,
    merge_vertices,
    mesh_volume,
    rasterize_tets,
    side_wall_planes,
    simplify_manifold,
    solid_surface_mesh,
    solid_with_print_features,
)

MM = 1e-3
# A box "blade" in the CAD frame: radial x, chord y (side walls at ±10 mm), pin axis z through the origin.
BOX_LO = np.array([-5.0, -10.0, -2.0]) * MM
BOX_HI = np.array([50.0, 10.0, 2.0]) * MM


def _box_tets(lo, hi, cells):
    """Structured box split into Kuhn tetrahedra (6 per cell)."""
    axes = [np.linspace(lo[d], hi[d], cells[d] + 1) for d in range(3)]
    nodes = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    shape = [c + 1 for c in cells]

    def nid(i, j, k):
        return (i * shape[1] + j) * shape[2] + k

    tets = []
    for i, j, k in itertools.product(*[range(c) for c in cells]):
        for perm in itertools.permutations(range(3)):
            corner = [i, j, k]
            path = [nid(*corner)]
            for axis in perm:
                corner[axis] += 1
                path.append(nid(*corner))
            tets.append(path)
    return nodes, np.array(tets)


def _boundary_faces(nodes, tets):
    """Outward boundary facet centres + unit normals (faces used by exactly one tet)."""
    faces = np.vstack(
        [tets[:, [1, 2, 3]], tets[:, [0, 2, 3]], tets[:, [0, 1, 3]], tets[:, [0, 1, 2]]]
    )
    opposite = np.concatenate([tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]])
    key = np.sort(faces, axis=1)
    _u, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    once = cnt[inv.reshape(-1)] == 1
    f, o = faces[once], opposite[once]
    p = nodes[f]
    centers = p.mean(axis=1)
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    flip = np.einsum("ij,ij->i", n, nodes[o] - centers) > 0
    n[flip] *= -1.0
    return centers, n


def _features(**kw):
    base = dict(
        hub_radius_m=10 * MM, bore_radius_m=1.5 * MM, rib_width_m=1.0 * MM, tip_cap_m=0.6 * MM,
        tip_x_m=50 * MM, skin_m=0.5 * MM,
    )  # fmt: skip
    base.update(kw)
    return BladePrintFeatures(**base)


def _identity_frame():
    return np.eye(3), np.zeros(3)


def test_rasterize_marks_exactly_the_samples_inside_the_box():
    nodes, tets = _box_tets(BOX_LO, BOX_HI, (11, 4, 2))
    grid = grid_for_points(nodes, 0.5 * MM)
    code = rasterize_tets(nodes, tets, np.ones(len(tets)), grid)
    pts = grid.points(np.argwhere(np.ones(grid.shape, dtype=bool)))
    eps = 1e-9
    inside = np.all((pts >= BOX_LO - eps) & (pts <= BOX_HI + eps), axis=1).reshape(grid.shape)
    assert np.array_equal(code > 0, inside)


def test_rasterize_leaves_outside_empty():
    nodes, tets = _box_tets(BOX_LO, BOX_HI, (11, 4, 2))
    grid = grid_for_points(nodes, 0.5 * MM, pad_voxels=3)
    code = rasterize_tets(nodes, tets, np.ones(len(tets)), grid)
    assert code[:2].max() == 0 and code[:, :, :2].max() == 0  # padding stays outside every tet


def test_rasterize_encodes_per_tet_density():
    nodes, tets = _box_tets(BOX_LO, BOX_HI, (11, 4, 2))
    centroids = nodes[tets].mean(axis=1)
    rho = np.where(centroids[:, 0] < 20 * MM, 1.0, 0.2)
    grid = grid_for_points(nodes, 0.5 * MM)
    code = rasterize_tets(nodes, tets, rho, grid)
    solid_x = grid.points(np.argwhere(code >= 128))[:, 0]
    assert solid_x.max() <= 20 * MM + grid.pitch  # only the ρ=1 half reaches the solid code


def test_rasterize_small_batches_match_one_batch():
    nodes, tets = _box_tets(BOX_LO, BOX_HI, (11, 4, 2))
    grid = grid_for_points(nodes, 0.5 * MM)
    a = rasterize_tets(nodes, tets, np.ones(len(tets)), grid)
    b = rasterize_tets(nodes, tets, np.ones(len(tets)), grid, max_batch_points=500)
    assert np.array_equal(a > 0, b > 0)


def test_side_wall_planes_recover_box_walls():
    nodes, tets = _box_tets(BOX_LO, BOX_HI, (11, 4, 2))
    c, n = _boundary_faces(nodes, tets)
    (n_lo, d_lo), (n_hi, d_hi) = side_wall_planes(c, n, (15 * MM, 45 * MM))
    assert np.allclose(n_lo, [0, -1, 0]) and np.allclose(n_hi, [0, 1, 0])
    assert (d_lo, d_hi) == pytest.approx((10 * MM, 10 * MM))


def test_side_wall_planes_need_facets():
    nodes, tets = _box_tets(BOX_LO, BOX_HI, (11, 4, 2))
    c, n = _boundary_faces(nodes, tets)
    with pytest.raises(ValueError):
        side_wall_planes(c, n, (100 * MM, 200 * MM))


def test_edge_leveling_lays_a_slanted_wall_flat():
    slope = np.radians(5.0)  # −y wall: y = −(a + tan·x)  → outward normal (−sin, −cos, 0)
    rot = edge_leveling_transform((np.array([-np.sin(slope), -np.cos(slope), 0.0]), 0.006))
    a, b = np.array([0.0, -0.006, 0.0]), np.array([0.2, -0.006 - 0.2 * np.tan(slope), 0.0])
    assert (rot @ a)[2] == pytest.approx((rot @ b)[2], abs=1e-12)


def test_edge_leveling_is_a_proper_rotation():
    rot = edge_leveling_transform((np.array([-0.0883, -0.9961, 0.0]), 0.006))
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-3) and np.linalg.det(rot) == pytest.approx(
        1.0, abs=1e-3
    )


def test_edge_leveling_puts_the_pin_axis_along_print_minus_y():
    rot = edge_leveling_transform((np.array([0.0, -1.0, 0.0]), 0.006))
    assert np.allclose(rot @ np.array([0.0, 0.0, 1.0]), [0.0, -1.0, 0.0])


def test_boundary_shell_has_the_requested_depth():
    dom = np.zeros((30, 30, 30), dtype=bool)
    dom[5:25, 5:25, 5:25] = True
    shell = boundary_shell(dom, 1.0, 3.0, slab=7)  # small slab exercises the halo stitching
    assert shell.sum() == 20**3 - 14**3


def test_keep_largest_component_drops_specks():
    s = np.zeros((10, 10, 10), dtype=bool)
    s[1:6, 1:6, 1:6] = True
    s[8, 8, 8] = True
    kept, removed = keep_largest_component(s)
    assert removed == 1 and kept.sum() == 125


def test_keep_largest_component_single_body_unchanged():
    s = np.zeros((5, 5, 5), dtype=bool)
    s[1:4, 1:4, 1:4] = True
    kept, removed = keep_largest_component(s)
    assert removed == 0 and np.array_equal(kept, s)


def _void_box_solid():
    nodes, tets = _box_tets(BOX_LO, BOX_HI, (22, 8, 4))
    grid = grid_for_points(nodes, 0.25 * MM)
    code = rasterize_tets(
        nodes, tets, np.full(len(tets), 0.1), grid
    )  # all carved: only features remain
    c, n = _boundary_faces(nodes, tets)
    planes = side_wall_planes(c, n, (15 * MM, 45 * MM))
    rot, off = _identity_frame()
    solid, counts = solid_with_print_features(code, grid, rot, off, planes, _features())
    return solid, counts, grid


def _at(solid, grid, xyz_mm):
    idx = np.round((np.array(xyz_mm) * MM - grid.lo) / grid.pitch).astype(int)
    return bool(solid[tuple(idx)])


@pytest.fixture(scope="module")
def void_box():
    return _void_box_solid()


def test_features_keep_carved_core_hollow(void_box):
    solid, _c, grid = void_box
    assert not _at(solid, grid, (30, 0, 0))


def test_features_hold_the_skin(void_box):
    solid, _c, grid = void_box
    assert _at(solid, grid, (30, 0, 1.75))  # 0.25 mm under the top face


def test_features_restore_the_rib(void_box):
    solid, _c, grid = void_box
    assert _at(solid, grid, (30, 9.25, 0))  # inside the 1 mm rib strip of the +y wall


def test_features_cap_the_tip(void_box):
    solid, _c, grid = void_box
    assert _at(solid, grid, (49.75, 0, 0))


def test_features_fill_the_hub(void_box):
    solid, _c, grid = void_box
    assert _at(solid, grid, (5, 0, 0))


def test_features_open_the_bore(void_box):
    solid, _c, grid = void_box
    assert not _at(solid, grid, (0, 0, 0))


def test_features_report_every_count(void_box):
    _s, counts, _g = void_box
    assert set(counts) == {"to_solid", "skin", "hub", "rib", "tip_cap", "bore", "islands"}


def _cube_solid(n=30, lo=5, hi=25):
    s = np.zeros((n, n, n), dtype=bool)
    s[lo:hi, lo:hi, lo:hi] = True
    return s, VoxelGrid(lo=np.zeros(3), pitch=1.0, shape=(n, n, n))


def _edge_use(faces):
    e = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _u, c = np.unique(e, axis=0, return_counts=True)
    return set(c.tolist())


def test_surface_mesh_is_closed_across_slab_seams():
    s, g = _cube_solid()
    _v, f = solid_surface_mesh(s, g, slab=7)
    assert _edge_use(f) == {2}


def test_surface_mesh_volume_matches_the_solid():
    s, g = _cube_solid()
    v, f = solid_surface_mesh(s, g, slab=7)
    assert mesh_volume(v, f) == pytest.approx(20.0**3, rel=0.03)


def test_surface_mesh_smoothing_stays_closed():
    s, g = _cube_solid()
    _v, f = solid_surface_mesh(s, g, slab=7, smooth_sigma_vox=0.6)
    assert _edge_use(f) == {2}


def test_surface_mesh_rejects_empty_solid():
    s, g = _cube_solid(lo=0, hi=0)
    with pytest.raises(ValueError):
        solid_surface_mesh(s, g)


def test_merge_vertices_welds_duplicates():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=float)
    uv, f = merge_vertices(v, np.array([[0, 1, 2], [0, 3, 2]]))
    assert len(uv) == 3 and np.array_equal(uv[f[0]], uv[f[1]])


def test_merge_vertices_drops_collapsed_faces():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float)
    _uv, f = merge_vertices(v, np.array([[0, 1, 2]]))
    assert len(f) == 0


def test_mesh_volume_of_unit_tet():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    assert mesh_volume(v, f) == pytest.approx(1.0 / 6.0)


def test_thin_fraction_flags_a_sub_nozzle_wall():
    s = np.zeros((40, 40, 3), dtype=bool)
    s[5:35, 10:13, :] = True  # 3 px = 0.3 mm wall at 0.1 mm pitch
    assert layer_thin_fraction(s, 1e-4, 4e-4).min() == pytest.approx(1.0)


def test_thin_fraction_passes_a_thick_wall():
    s = np.zeros((40, 40, 3), dtype=bool)
    s[5:35, 10:20, :] = True  # 1.0 mm wall
    assert layer_thin_fraction(s, 1e-4, 4e-4).max() == pytest.approx(0.0)


def test_bed_contact_full_for_a_flat_bottom():
    s = np.zeros((10, 5, 10), dtype=bool)
    s[:, 2, 0:6] = True
    assert bed_contact_fraction(s) == pytest.approx(1.0)


def test_bed_contact_counts_raised_stations():
    s = np.zeros((10, 5, 10), dtype=bool)
    s[:5, 2, 0:6] = True
    s[5:, 2, 5:9] = True  # second half floats 5 layers up
    assert bed_contact_fraction(s) == pytest.approx(0.5)


def test_bed_contact_empty_is_zero():
    assert bed_contact_fraction(np.zeros((3, 3, 3), dtype=bool)) == 0.0


def test_simplify_manifold_orients_outward_and_keeps_volume():
    s, g = _cube_solid()
    v, f = solid_surface_mesh(s, g)
    sv, sf = simplify_manifold(v, f[:, ::-1], 0.01)  # feed it inside-out
    t = sv[sf]
    signed = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0
    assert signed == pytest.approx(mesh_volume(v, f), rel=1e-3)


def test_simplify_manifold_rejects_an_open_mesh():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    with pytest.raises(ValueError):
        simplify_manifold(v, np.array([[0, 2, 1], [0, 1, 3]]), 0.01)


@pytest.fixture(scope="module")
def built_box():
    nodes, tets = _box_tets(BOX_LO, BOX_HI, (22, 8, 4))
    centroids = nodes[tets].mean(axis=1)
    rho = np.where(centroids[:, 0] < 25 * MM, 1.0, 0.1)
    c, n = _boundary_faces(nodes, tets)
    blade = build_print_blade(
        nodes, tets, rho, c, n, hub_radius_m=10 * MM, bore_radius_m=1.5 * MM, rib_width_m=1.0 * MM,
        tip_cap_m=0.6 * MM, skin_m=0.5 * MM, pitch_m=0.25 * MM,
    )  # fmt: skip
    return blade


def test_build_print_blade_is_closed(built_box):
    assert _edge_use(built_box.faces) == {2}


def test_build_print_blade_sits_on_the_bed(built_box):
    assert built_box.verts[:, 2].min() == pytest.approx(0.0, abs=0.25 * MM)


def test_build_print_blade_stands_on_edge(built_box):
    extent = np.ptp(built_box.verts, axis=0)
    assert extent[2] == pytest.approx(20 * MM, abs=0.5 * MM)  # chord vertical


def test_build_print_blade_stays_inside_the_domain(built_box):
    assert mesh_volume(built_box.verts, built_box.faces) < np.prod(BOX_HI - BOX_LO)


def test_build_print_blade_counts_the_domain(built_box):
    assert built_box.voxel_counts["domain"] > built_box.voxel_counts["to_solid"]


def test_flat_boss_radius_is_the_nearest_node_beyond_the_faces():
    nodes = np.array([[1, 0, 1], [0, 3, -1], [5, 0, 2], [0, 7, -3]], dtype=float) * MM
    assert flat_boss_radius_m(nodes, 1 * MM) == pytest.approx(5 * MM)


def test_flat_boss_radius_when_everything_is_flat():
    nodes = np.array([[1, 0, 1], [0, 3, -1]], dtype=float) * MM
    assert flat_boss_radius_m(nodes, 1 * MM) == pytest.approx(3 * MM)


def _slab(half_z_m, size_m=40 * MM):
    m = manifold3d.Manifold.cube((size_m, size_m, 2 * half_z_m), center=True).to_mesh()
    return np.asarray(m.vert_properties, dtype=float)[:, :3], np.asarray(
        m.tri_verts, dtype=np.int64
    )


def _z_extent_within(verts, radius_m):
    near = np.hypot(verts[:, 0], verts[:, 1]) < radius_m
    return float(verts[near, 2].min()), float(verts[near, 2].max())


BOSS = (1.0 * MM, 0.5 * MM, 5.0 * MM, 8.0 * MM)  # half height, bore, boss radius, flat radius


def test_boss_trims_material_beyond_the_faces():
    v, f = apply_cad_solids(*_slab(1.06 * MM), *_identity_frame(), boss=BOSS)
    trimmed = 2 * np.pi * (8 * MM) ** 2 * 0.06 * MM
    assert mesh_volume(v, f) == pytest.approx((40 * MM) ** 2 * 2.12 * MM - trimmed, rel=1e-4)


def test_boss_trim_leaves_nothing_above_the_faces_inside_the_flat_radius():
    v, f = apply_cad_solids(*_slab(1.06 * MM), *_identity_frame(), boss=BOSS)
    assert np.abs(v[np.hypot(v[:, 0], v[:, 1]) < 7.9 * MM, 2]).max() <= 1.0 * MM + 1e-6


def test_boss_leaves_the_root_beyond_the_flat_radius():
    v, f = apply_cad_solids(*_slab(1.06 * MM), *_identity_frame(), boss=BOSS)
    far = np.hypot(v[:, 0], v[:, 1]) > 9 * MM
    assert v[far, 2].max() == pytest.approx(1.06 * MM, abs=1e-6)


def test_boss_fills_dips_only_inside_the_boss_radius():
    v, f = apply_cad_solids(*_slab(0.9 * MM), *_identity_frame(), boss=BOSS)
    assert _z_extent_within(v, 5.0 * MM + 1e-6)[1] == pytest.approx(
        1.0 * MM, abs=1e-6
    ) and mesh_volume(v, f) == pytest.approx(
        (40 * MM) ** 2 * 1.8 * MM + np.pi * (5 * MM) ** 2 * 0.2 * MM, rel=2e-3
    )


def test_boss_ring_keeps_the_bore_open():
    v, f = apply_cad_solids(*_slab(0.9 * MM), *_identity_frame(), boss=BOSS)
    bore_wall = np.abs(np.hypot(v[:, 0], v[:, 1]) - 0.5 * MM) < 0.02 * MM
    assert v[bore_wall, 2].max() == pytest.approx(1.0 * MM, abs=1e-6)


def test_added_sphere_adds_its_volume():
    sphere = np.array([[30 * MM, 0.0, 0.0, 2 * MM]])  # clear of the 40 mm slab
    v0, f0 = _slab(1 * MM)
    v, f = apply_cad_solids(v0, f0, *_identity_frame(), add_spheres=sphere)
    assert mesh_volume(v, f) - mesh_volume(v0, f0) == pytest.approx(
        4 / 3 * np.pi * (2 * MM) ** 3, rel=0.02
    )


def test_cut_sphere_removes_its_volume():
    sphere = np.array([[0.0, 0.0, 0.0, 0.5 * MM]])
    v0, f0 = _slab(1 * MM)
    v, f = apply_cad_solids(v0, f0, *_identity_frame(), cut_spheres=sphere)
    assert mesh_volume(v0, f0) - mesh_volume(v, f) == pytest.approx(
        4 / 3 * np.pi * (0.5 * MM) ** 3, rel=0.02
    )


def test_cad_solids_are_placed_through_the_print_transform():
    rot = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    off = np.array([0.0, 0.0, 50 * MM])
    sphere = np.array([[30 * MM, 0.0, 5 * MM, 1 * MM]])
    v, f = apply_cad_solids(*_slab(1 * MM), rot, off, add_spheres=sphere)
    blob = v[v[:, 0] > 25 * MM]
    assert blob.mean(axis=0) == pytest.approx(rot @ sphere[0, :3] + off, abs=0.05 * MM)


def test_cad_solids_keep_the_mesh_closed():
    sphere = np.array([[20 * MM, 0.0, 1 * MM, 1.5 * MM]])
    v, f = apply_cad_solids(*_slab(1.06 * MM), *_identity_frame(), boss=BOSS, add_spheres=sphere)
    assert _edge_use(f) == {2}


def test_cad_solids_reject_an_open_mesh():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    with pytest.raises(ValueError):
        apply_cad_solids(v, np.array([[0, 2, 1], [0, 1, 3]]), *_identity_frame())


@pytest.fixture(scope="module")
def built_box_with_boss():
    nodes, tets = _box_tets(BOX_LO, BOX_HI, (22, 8, 4))
    c, n = _boundary_faces(nodes, tets)
    bump = np.array([[30 * MM, 0.0, 2.5 * MM, 1.0 * MM]])
    blade = build_print_blade(
        nodes, tets, np.ones(len(tets)), c, n, hub_radius_m=10 * MM, bore_radius_m=1.5 * MM,
        rib_width_m=1.0 * MM, tip_cap_m=0.6 * MM, skin_m=0.5 * MM, pitch_m=0.25 * MM,
        boss_half_height_m=2.0 * MM, boss_radius_m=4.0 * MM, bump_spheres=bump,
    )  # fmt: skip
    cad = (blade.verts - blade.print_offset_m) @ blade.print_from_cad
    return blade, cad


def test_build_with_boss_makes_the_boss_exactly_the_layer_spacing(built_box_with_boss):
    _blade, cad = built_box_with_boss
    near = np.hypot(cad[:, 0], cad[:, 1]) < 3.5 * MM
    assert np.ptp(cad[near, 2]) == pytest.approx(4.0 * MM, abs=1e-6)


def test_build_with_a_bump_raises_it_off_the_face(built_box_with_boss):
    _blade, cad = built_box_with_boss
    assert cad[:, 2].max() == pytest.approx(3.5 * MM, abs=0.02 * MM)


def test_build_with_cad_solids_stays_closed(built_box_with_boss):
    assert _edge_use(built_box_with_boss[0].faces) == {2}


def _shells():
    """A 10 mm cube with a sealed 4 mm void inside, plus a separate 0.01 mm sliver tetrahedron."""
    outer = manifold3d.Manifold.cube((10 * MM,) * 3, center=True)
    void = manifold3d.Manifold.cube((4 * MM,) * 3, center=True)
    m = (outer - void).to_mesh()
    v = np.asarray(m.vert_properties, dtype=float)[:, :3]
    f = np.asarray(m.tri_verts, dtype=np.int64)
    tv = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float) * 0.01 * MM + 20 * MM
    tf = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]) + len(v)
    return np.vstack([v, tv]), np.vstack([f, tf])


def test_drop_sliver_shells_removes_the_speck():
    v, f = drop_sliver_shells(*_shells())
    assert v.max() < 6 * MM


def test_drop_sliver_shells_keeps_the_sealed_void():
    v, f = drop_sliver_shells(*_shells())
    assert mesh_volume(v, f) == pytest.approx((10 * MM) ** 3 - (4 * MM) ** 3, rel=1e-9)


def test_drop_sliver_shells_reindexes_faces():
    v, f = drop_sliver_shells(*_shells())
    assert f.max() == len(v) - 1 and _edge_use(f) == {2}
