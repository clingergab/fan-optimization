"""Tests for SIMP interpolation, density filter, and OC update (§3.1)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import identity

from fanopt.topopt.simp import (
    apply_filter,
    build_density_filter,
    oc_update,
    oc_update_unstructured,
    simp_modulus,
)


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


# --- unstructured (3D tet) OC update -------------------------------------------------

def _unstructured_case(n=6, n_design=4, volfrac=0.4):
    """Flat n-element case: first n_design are carvable, the rest are frozen skin (ρ=1)."""
    x = np.full(n, volfrac)
    design = np.zeros(n, dtype=bool)
    design[:n_design] = True
    x[~design] = 1.0
    volumes = np.ones(n)
    filt = identity(n, format="csr")  # identity filter isolates the OC math
    dc = -np.ones(n)  # uniform compliance sensitivity
    return x, dc, volumes, filt, design


def test_oc_unstructured_hits_volume_weighted_target():
    x, dc, volumes, filt, design = _unstructured_case()
    x_new = oc_update_unstructured(x, dc, volumes, filt, volfrac=0.4, design_mask=design)
    vol = (volumes * x_new)[design].sum()
    assert vol == pytest.approx(0.4 * volumes[design].sum(), rel=1e-3)


def test_oc_unstructured_weights_by_volume():
    # Two design elements, equal (uniform) sensitivity but unequal volume. The volume
    # target must be met by MASS (V·ρ), so the big element cannot simply mirror the small.
    x = np.array([0.5, 0.5])
    volumes = np.array([1.0, 4.0])
    filt = identity(2, format="csr")
    design = np.array([True, True])
    x_new = oc_update_unstructured(x, -np.ones(2), volumes, filt, volfrac=0.4, design_mask=design)
    assert (volumes * x_new).sum() == pytest.approx(0.4 * volumes.sum(), rel=1e-3)


def test_oc_unstructured_leaves_frozen_fixed():
    x, dc, volumes, filt, design = _unstructured_case()
    x_new = oc_update_unstructured(x, dc, volumes, filt, volfrac=0.4, design_mask=design)
    assert np.allclose(x_new[~design], 1.0)  # frozen skin untouched


def test_oc_unstructured_respects_move_limit():
    x, dc, volumes, filt, design = _unstructured_case()
    x_new = oc_update_unstructured(x, dc, volumes, filt, volfrac=0.4, design_mask=design, move=0.1)
    assert np.abs(x_new - x)[design].max() <= 0.1 + 1e-9
    assert x_new.min() >= 0.0 and x_new.max() <= 1.0


def test_oc_unstructured_scale_invariant():
    x, _, volumes, filt, design = _unstructured_case()
    dc = -(np.arange(6, dtype=float) + 1.0)
    a = oc_update_unstructured(x, dc * 1e-9, volumes, filt, volfrac=0.4, design_mask=design)
    b = oc_update_unstructured(x, dc * 1e6, volumes, filt, volfrac=0.4, design_mask=design)
    assert np.abs(a - b).max() < 1e-9


def test_oc_unstructured_empty_design_returns_unchanged():
    # A fully-frozen mesh (no carvable elements) must not crash on b.max() of an empty array.
    x = np.array([1.0, 1.0, 1.0])
    out = oc_update_unstructured(
        x, -np.ones(3), np.ones(3), identity(3, format="csr"),
        volfrac=0.4, design_mask=np.zeros(3, dtype=bool),
    )
    assert np.array_equal(out, x)


def test_oc_unstructured_volume_sensitivity_changes_direction():
    # A larger volume-sensitivity on element 0 makes it "cost more" per unit density, so it
    # gets less material than element 1 despite equal compliance sensitivity.
    x = np.full(2, 0.5)
    design = np.array([True, True])
    out = oc_update_unstructured(
        x, -np.ones(2), np.ones(2), identity(2, format="csr"),
        volfrac=0.5, design_mask=design, volume_sensitivity=np.array([4.0, 1.0]),
    )
    assert out[0] < out[1]


def test_oc_unstructured_moves_material_to_sensitive_elements():
    # Element 0 far more compliance-sensitive than element 1 -> gets more material.
    x = np.full(4, 0.5)
    design = np.array([True, True, False, False])
    x[~design] = 1.0
    x_new = oc_update_unstructured(
        x, np.array([-10.0, -1.0, 0.0, 0.0]), np.ones(4), identity(4, format="csr"),
        volfrac=0.5, design_mask=design,
    )
    assert x_new[0] > x_new[1]
