"""Tests for the Phase 3 steady↔unsteady correlation gate."""

from __future__ import annotations

import numpy as np
import pytest

from fanopt.cfd.correlation import (
    R2_RETENTION_THRESHOLD,
    correlate,
    kendall_tau,
    pearson_r2,
    spearman_rho,
)


def test_perfect_linear_correlation_r2_is_one():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 3.0 * x + 5.0
    assert pearson_r2(x, y) == pytest.approx(1.0)


def test_r2_is_scale_invariant():
    # steady ~ O(10), unsteady ~ O(1e14): R² must ignore the scale gap.
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = x * 1e14
    assert pearson_r2(x, y) == pytest.approx(1.0)


def test_uncorrelated_low_r2():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 1.0, 4.0, 3.0])
    assert pearson_r2(x, y) < 0.5


def test_constant_series_is_zero_r2():
    assert pearson_r2(np.array([1.0, 2.0, 3.0]), np.array([5.0, 5.0, 5.0])) == 0.0


def test_pearson_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="equal-length"):
        pearson_r2(np.array([1.0, 2.0]), np.array([1.0]))


def test_pearson_rejects_single_point():
    with pytest.raises(ValueError, match="at least 2"):
        pearson_r2(np.array([1.0]), np.array([1.0]))


def test_kendall_tau_monotone_is_one():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert kendall_tau(x, x**2) == pytest.approx(1.0)


def test_kendall_tau_reversed_is_minus_one():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert kendall_tau(x, -x) == pytest.approx(-1.0)


def test_kendall_tau_constant_is_zero():
    assert kendall_tau(np.array([1.0, 2.0, 3.0]), np.array([5.0, 5.0, 5.0])) == 0.0


def test_spearman_rho_monotone_is_one():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman_rho(x, x**3) == pytest.approx(1.0)  # monotone but nonlinear


def test_spearman_rho_reversed_is_minus_one():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman_rho(x, -x) == pytest.approx(-1.0)


def test_spearman_rho_constant_is_zero():
    assert spearman_rho(np.array([1.0, 2.0, 3.0]), np.array([5.0, 5.0, 5.0])) == 0.0


def test_spearman_rejects_single_point():
    with pytest.raises(ValueError, match="at least 2"):
        spearman_rho(np.array([1.0]), np.array([1.0]))


def test_correlate_passes_above_threshold():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    res = correlate(x, 2.0 * x)
    assert res.passed is True
    assert res.r2 == pytest.approx(1.0)
    assert res.n == 5


def test_correlate_fails_below_threshold():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([4.0, 1.0, 3.0, 2.0])
    res = correlate(x, y)
    assert res.passed == (res.r2 >= R2_RETENTION_THRESHOLD)
    assert res.r2 < R2_RETENTION_THRESHOLD


def test_correlate_custom_threshold():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    res = correlate(x, 2.0 * x, threshold=0.99)
    assert res.meta["threshold"] == 0.99
