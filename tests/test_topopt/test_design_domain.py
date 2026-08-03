"""Tests for design-domain classification, density filter, and material frames."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from fanopt.topopt.design_domain import (
    build_density_filter_unstructured,
    classify_frozen_skin,
    global_material_frame,
    revolution_material_frames,
    tet_centroids_and_volumes,
)

_UNIT_TET = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def test_unit_tet_volume():
    _, vol = tet_centroids_and_volumes(_UNIT_TET, np.array([[0, 1, 2, 3]]))
    assert vol[0] == pytest.approx(1.0 / 6.0)


def test_unit_tet_centroid():
    cen, _ = tet_centroids_and_volumes(_UNIT_TET, np.array([[0, 1, 2, 3]]))
    assert cen[0] == pytest.approx([0.25, 0.25, 0.25])


def test_tet_volume_orientation_agnostic():
    # Swapping two vertices flips the det sign; volume must stay positive.
    _, vol = tet_centroids_and_volumes(_UNIT_TET, np.array([[0, 2, 1, 3]]))
    assert vol[0] == pytest.approx(1.0 / 6.0)


def test_filter_rows_sum_to_one():
    centroids = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    w = build_density_filter_unstructured(centroids, r_min=1.5).toarray()
    assert np.allclose(w.sum(axis=1), 1.0)


def test_filter_preserves_uniform_field():
    centroids = np.random.default_rng(0).uniform(0, 1, size=(20, 3))
    w = build_density_filter_unstructured(centroids, r_min=0.4)
    assert np.allclose(w @ np.ones(20), 1.0)


def test_filter_isolated_element_is_self():
    # Two elements farther apart than r_min → each filters to itself only.
    centroids = np.array([[0.0, 0, 0], [10.0, 0, 0]])
    w = build_density_filter_unstructured(centroids, r_min=1.0).toarray()
    assert np.allclose(w, np.eye(2))


def test_filter_volume_weighting_favors_larger_neighbor():
    # Element 0 sits between two equidistant neighbors; the larger-volume one weighs more.
    centroids = np.array([[0.0, 0, 0], [1.0, 0, 0], [-1.0, 0, 0]])
    vols = np.array([1.0, 5.0, 1.0])
    w = build_density_filter_unstructured(centroids, r_min=1.5, element_volumes=vols).toarray()
    assert w[0, 1] > w[0, 2]


def test_filter_matches_reference_neighbor_loop():
    # Lock the C-level (sparse_distance_matrix) build to the plain per-neighbor reference, incl.
    # the self term and column-volume weighting, on a random cloud with unequal volumes.
    rng = np.random.default_rng(0)
    c = rng.random((300, 3)) * 0.05
    vols = rng.random(300) * 1e-7
    r_min = 0.012
    tree = cKDTree(c)
    n = len(c)
    ref = np.zeros((n, n))
    for i, nbrs in enumerate(tree.query_ball_point(c, r_min)):
        nbrs = np.asarray(nbrs, dtype=int)
        w = (r_min - np.linalg.norm(c[nbrs] - c[i], axis=1)) * vols[nbrs]
        ref[i, nbrs] = w / w.sum()
    got = build_density_filter_unstructured(c, r_min, vols).toarray()
    assert np.allclose(got, ref, atol=1e-12)


def test_filter_block_size_does_not_change_result():
    # The block-wise build must be invariant to block_rows: a tiny block (many blocks) must equal
    # a single block covering all rows. Locks the cross-block indptr/scatter accumulation.
    rng = np.random.default_rng(1)
    c = rng.random((250, 3)) * 0.05
    vols = rng.random(250) * 1e-7
    one_block = build_density_filter_unstructured(c, 0.012, vols, block_rows=1000).toarray()
    many_blocks = build_density_filter_unstructured(c, 0.012, vols, block_rows=7).toarray()
    assert np.allclose(one_block, many_blocks, atol=1e-15)


def test_filter_multi_block_rows_still_sum_to_one():
    # Row-normalization must hold when the build spans several blocks.
    c = np.random.default_rng(2).uniform(0, 1, size=(40, 3))
    w = build_density_filter_unstructured(c, r_min=0.5, block_rows=9)
    assert np.allclose(np.asarray(w.sum(axis=1)).ravel(), 1.0)


def test_frozen_skin_flags_near_surface_elements():
    surface = np.array([[x, y, 0.0] for x in np.linspace(0, 1, 5) for y in np.linspace(0, 1, 5)])
    centroids = np.array([[0.5, 0.5, 0.0008], [0.5, 0.5, 0.005]])
    frozen = classify_frozen_skin(centroids, surface, skin_thickness_m=0.0012)
    assert frozen.tolist() == [True, False]


def test_revolution_frame_orthonormal_and_proper():
    centroids = np.array([[0.1, 0.0, 0.0], [0.08, 0.05, 0.02], [0.05, -0.09, 0.03]])
    frames = revolution_material_frames(centroids)
    for r in frames:
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(r) == pytest.approx(1.0)


def test_revolution_weak_axis_is_tangential_on_x():
    # A centroid on +x has radial=x, so the weak (width) axis must be tangential = +y.
    frames = revolution_material_frames(np.array([[0.1, 0.0, 0.0]]))
    assert np.allclose(frames[0][2], [0.0, 1.0, 0.0])


def test_revolution_radial_axis_points_outward():
    # First material axis (radial) aligns with the in-plane centroid direction.
    c = np.array([[0.06, 0.08, 0.0]])
    frames = revolution_material_frames(c)
    assert np.allclose(frames[0][0], [0.6, 0.8, 0.0])


def test_global_frame_matches_centreline_revolution_frame():
    g = global_material_frame()
    r = revolution_material_frames(np.array([[0.1, 0.0, 0.0]]))[0]
    assert np.allclose(g, r)
