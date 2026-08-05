"""Tests for the TO density-field -> printable STL conversion (fanopt.geometry.to_stl)."""

from __future__ import annotations

import importlib.util
import struct

import numpy as np
import pytest
from scipy.sparse import coo_matrix

if importlib.util.find_spec("skimage") is None:  # pragma: no cover - env-dependent
    pytest.skip("scikit-image not installed", allow_module_level=True)

from fanopt.geometry.to_stl import carved_blade_mesh, laplacian_smooth, write_binary_stl


def _cloud(spacing=0.001, box=0.020):
    ax = np.arange(0.0, box, spacing)
    return np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1).reshape(-1, 3)


def _block_field(centroids, lo=0.005, hi=0.015):
    inside = np.all((centroids >= lo) & (centroids <= hi), axis=1)
    return np.where(inside, 1.0, 0.1)  # solid inside the sub-box, void outside


def test_carved_mesh_returns_valid_triangles():
    cen = _cloud()
    v, f = carved_blade_mesh(_block_field(cen), cen, voxel_pitch_m=0.0005)
    assert v.shape[1] == 3 and f.shape[1] == 3
    assert len(f) > 0 and f.max() < len(v)  # every face indexes a real vertex


def test_carved_mesh_wraps_the_material_not_the_whole_cloud():
    # The 20 mm cloud holds a 10 mm solid block; the surface must bound ~the block, not the cloud.
    cen = _cloud()
    v, _f = carved_blade_mesh(_block_field(cen), cen, voxel_pitch_m=0.0005)
    extent_mm = (v.max(axis=0) - v.min(axis=0)) * 1e3
    assert np.all(extent_mm < 14.0) and np.all(extent_mm > 9.0)  # ~10 mm block, not the 20 mm cloud


def test_carved_mesh_void_is_excluded():
    # A fully-void field (all below level) has no iso-surface -> marching_cubes raises.
    cen = _cloud()
    with pytest.raises(ValueError):
        carved_blade_mesh(np.full(len(cen), 0.1), cen, voxel_pitch_m=0.0005)


def test_write_binary_stl_size_and_count(tmp_path):
    cen = _cloud()
    v, f = carved_blade_mesh(_block_field(cen), cen, voxel_pitch_m=0.0005)
    p = write_binary_stl(v, f, tmp_path / "blade.stl")
    raw = p.read_bytes()
    assert len(raw) == 84 + 50 * len(f)  # 80-byte header + uint32 count + 50 bytes/triangle
    (count,) = struct.unpack("<I", raw[80:84])
    assert count == len(f)


def test_carved_mesh_solid_is_pitch_invariant():
    # The void tolerance tracks the CLOUD spacing, so a solid region is captured the same at a fine or
    # coarse voxel pitch — no interior speckle when the pitch drops below the spacing (the old pitch-only
    # tolerance carved cavities at fine pitch, shrinking/fragmenting the volume).
    cen = _cloud()
    dens = _block_field(cen)

    def vol(v, f):
        t = v[f]
        return abs(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)

    fine = carved_blade_mesh(dens, cen, voxel_pitch_m=0.0004)  # pitch < 1 mm cloud spacing
    coarse = carved_blade_mesh(dens, cen, voxel_pitch_m=0.0008)
    assert vol(*fine) == pytest.approx(vol(*coarse), rel=0.15)  # same solid captured at both pitches


def test_carved_mesh_accepts_explicit_tolerance():
    cen = _cloud()
    v, f = carved_blade_mesh(_block_field(cen), cen, voxel_pitch_m=0.0005, outside_tol_m=0.001)
    assert len(f) > 0 and f.max() < len(v)


def test_laplacian_smooth_preserves_topology_and_reduces_roughness():
    cen = _cloud()
    v, f = carved_blade_mesh(_block_field(cen), cen, voxel_pitch_m=0.0004)
    vs = laplacian_smooth(v, f, iterations=8)
    assert vs.shape == v.shape and f.max() < len(vs) and np.isfinite(vs).all()

    def rough(V):
        e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        e = np.vstack([e, e[:, ::-1]])
        adj = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V))).tocsr()
        adj.data[:] = 1.0
        d = np.asarray(adj.sum(1)).ravel()
        d[d == 0] = 1.0
        return np.linalg.norm((adj @ V) / d[:, None] - V, axis=1).mean()

    assert rough(vs) < rough(v)  # smoother than the raw marching-cubes staircase


def test_write_binary_stl_scales_metres_to_mm(tmp_path):
    # SI metres in -> millimetres in the STL (default scale 1000), so slicers read the true size.
    v = np.array([[0.0, 0, 0], [0.220, 0, 0], [0.0, 0.05, 0]])  # 0.22 m blade-scale triangle
    f = np.array([[0, 1, 2]])
    p = write_binary_stl(v, f, tmp_path / "mm.stl")
    coords = np.frombuffer(p.read_bytes()[96:132], dtype="<f4").reshape(3, 3)  # 3 verts after the normal
    assert coords[:, 0].max() == pytest.approx(220.0, rel=1e-4)  # 0.22 m -> 220 mm, not 0.22


def test_write_binary_stl_scale_one_keeps_units(tmp_path):
    v = np.array([[0.0, 0, 0], [2.0, 0, 0], [0.0, 3.0, 0]])
    f = np.array([[0, 1, 2]])
    p = write_binary_stl(v, f, tmp_path / "raw.stl", scale=1.0)
    coords = np.frombuffer(p.read_bytes()[96:132], dtype="<f4").reshape(3, 3)
    assert coords[:, 0].max() == pytest.approx(2.0)  # scale=1 leaves coordinates untouched


def test_write_binary_stl_normals_are_unit(tmp_path):
    # A single triangle in the xy-plane -> facet normal must be +/- z, unit length.
    v = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]])
    f = np.array([[0, 1, 2]])
    p = write_binary_stl(v, f, tmp_path / "tri.stl")
    n = np.frombuffer(p.read_bytes()[84:96], dtype="<f4")
    assert np.allclose(np.abs(n), [0.0, 0.0, 1.0])
