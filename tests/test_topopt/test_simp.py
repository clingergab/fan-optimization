"""Tests for SIMP interpolation, density filter, and OC update (§3.1)."""

from __future__ import annotations

import numpy as np
import pytest

from fanopt.topopt.simp import apply_filter, build_density_filter, oc_update, simp_modulus


def test_simp_modulus_solid_is_e0():
    assert simp_modulus(np.array([1.0]), 3.0, 1300e6, 1.3)[0] == pytest.approx(1300e6)


def test_simp_modulus_void_is_emin():
    assert simp_modulus(np.array([0.0]), 3.0, 1300e6, 1.3)[0] == pytest.approx(1.3)


def test_simp_modulus_penalizes_intermediate():
    # rho=0.5, p=3 -> E ~ emin + 0.125*(e0-emin); penalization pushes below linear.
    e = simp_modulus(np.array([0.5]), 3.0, 1.0, 0.0)[0]
    assert e == pytest.approx(0.125)


def test_filter_row_sums_unity():
    hs = build_density_filter(5, 8, 1.5)
    assert np.allclose(np.asarray(hs.sum(axis=1)).ravel(), 1.0)


def test_filter_preserves_uniform_field():
    hs = build_density_filter(5, 8, 1.5)
    x = np.full((5, 8), 0.4)
    assert np.allclose(apply_filter(hs, x), 0.4)


def test_filter_rejects_nonpositive_rmin():
    with pytest.raises(ValueError, match="rmin must be"):
        build_density_filter(4, 4, 0.0)


def test_filter_smooths_a_spike():
    hs = build_density_filter(5, 5, 2.0)
    x = np.zeros((5, 5))
    x[2, 2] = 1.0
    xf = apply_filter(hs, x)
    assert xf[2, 2] < 1.0  # spike spread
    assert xf[2, 3] > 0.0  # neighbor gained material


def _uniform_problem(nely=4, nelx=6, volfrac=0.4):
    x = np.full((nely, nelx), volfrac)
    free = np.ones((nely, nelx), dtype=bool)
    active = np.ones((nely, nelx), dtype=bool)
    hs = build_density_filter(nely, nelx, 1.5)
    dc = -np.ones((nely, nelx))  # uniform sensitivity
    dv = np.ones((nely, nelx))
    return x, dc, dv, hs, free, active


def test_oc_update_hits_volume_target():
    x, dc, dv, hs, free, active = _uniform_problem()
    x_new = oc_update(x, dc, dv, hs, volfrac=0.4, free=free, active=active)
    assert apply_filter(hs, x_new)[active].sum() == pytest.approx(0.4 * active.sum(), rel=1e-3)


def test_oc_update_respects_move_and_bounds():
    x, dc, dv, hs, free, active = _uniform_problem()
    x_new = oc_update(x, dc, dv, hs, volfrac=0.4, free=free, active=active, move=0.1)
    assert x_new.min() >= 0.0
    assert x_new.max() <= 1.0
    assert np.abs(x_new - x).max() <= 0.1 + 1e-9


def test_oc_update_is_scale_invariant_in_sensitivity():
    # Compliance-minimization is scale-invariant: multiplying the compliance
    # sensitivity by any positive constant (e.g. a larger load) must give the SAME
    # density update. A regression guard for the fixed-bisection-bounds bug that broke
    # low-compliance problems (a stiff panel under light load).
    x, _, dv, hs, free, active = _uniform_problem()
    dc = -(np.arange(x.size, dtype=float).reshape(x.shape) + 1.0)  # non-uniform
    x_small = oc_update(x, dc * 1e-9, dv, hs, volfrac=0.4, free=free, active=active)
    x_large = oc_update(x, dc * 1e6, dv, hs, volfrac=0.4, free=free, active=active)
    assert np.abs(x_small - x_large).max() < 1e-9


def test_oc_update_leaves_preserved_fixed():
    nely, nelx = 4, 6
    x = np.full((nely, nelx), 0.4)
    preserved = np.zeros((nely, nelx), dtype=bool)
    preserved[0, :] = True
    x[preserved] = 1.0
    active = np.ones((nely, nelx), dtype=bool)
    free = active & ~preserved
    hs = build_density_filter(nely, nelx, 1.5)
    x_new = oc_update(
        x, -np.ones_like(x), np.ones_like(x), hs, volfrac=0.5, free=free, active=active
    )
    assert np.allclose(x_new[preserved], 1.0)
