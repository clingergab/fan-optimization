"""Tests for the canonical J_fan post-processor (report-final.md §9.4)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from fanopt.cfd import j_fan
from fanopt.cfd.j_fan import (
    RHO_AIR_KG_PER_M3,
    THRUST_DIR,
    SteadyProxyResult,
    SteadyRun,
    UnsteadyResult,
    compute_j_fan,
    compute_j_fan_steady,
    plane_flux_from_velocity,
    plane_momentum_flux,
    reduce_cycles,
)

# --- locked constants ---------------------------------------------------------


def test_thrust_dir_is_plus_z():
    assert THRUST_DIR == (0.0, 0.0, 1.0)


def test_density_lock():
    assert pytest.approx(1.225) == RHO_AIR_KG_PER_M3


def test_plane_geometry_locks():
    assert pytest.approx(0.300) == j_fan.ANALYSIS_PLANE_DISTANCE_M
    assert pytest.approx(0.600) == j_fan.ANALYSIS_PLANE_SIZE_M


def test_cycle_locks():
    assert j_fan.N_CYCLES_CANONICAL == 5
    assert j_fan.N_CYCLES_DISCARD == 1
    assert j_fan.STEPS_PER_CYCLE == 200


# --- plane_momentum_flux ------------------------------------------------------


def test_plane_momentum_flux_uniform_known_value():
    u_n = np.array([2.0, 2.0])
    u_t = np.array([3.0, 3.0])
    area = np.array([0.5, 0.5])
    # ρ · Σ(u_n·u_t·dA) = 1.225 · (2·3·0.5 + 2·3·0.5) = 1.225 · 6
    assert plane_momentum_flux(u_n, u_t, area) == pytest.approx(1.225 * 6.0)


def test_plane_momentum_flux_custom_rho():
    got = plane_momentum_flux(np.array([1.0]), np.array([1.0]), np.array([1.0]), rho=2.0)
    assert got == pytest.approx(2.0)


def test_plane_momentum_flux_shape_mismatch_raises():
    with pytest.raises(ValueError, match="share shape"):
        plane_momentum_flux(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0]))


# --- plane_flux_from_velocity -------------------------------------------------


def test_plane_flux_from_velocity_normal_equals_thrust():
    # n̂ = t̂ = +z ⇒ integrand = ρ · w² · dA
    velocity = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    area = np.array([1.0, 1.0])
    assert plane_flux_from_velocity(velocity, area) == pytest.approx(1.225 * (4.0 + 4.0))


def test_plane_flux_reversed_flow_still_positive_when_n_equals_t():
    # w² is positive regardless of sign when n̂ == t̂ == +z
    velocity = np.array([[0.0, 0.0, -2.0]])
    area = np.array([1.0])
    assert plane_flux_from_velocity(velocity, area) == pytest.approx(1.225 * 4.0)


def test_plane_flux_signed_when_n_differs_from_t():
    # n̂ = +z, t̂ = +x ⇒ u_n·u_t = w·u, can be negative
    velocity = np.array([[3.0, 0.0, -2.0]])
    area = np.array([1.0])
    got = plane_flux_from_velocity(velocity, area, n_hat=(0, 0, 1), t_hat=(1, 0, 0))
    assert got == pytest.approx(1.225 * (-2.0 * 3.0))


def test_plane_flux_from_velocity_bad_velocity_shape():
    with pytest.raises(ValueError, match="velocity must be"):
        plane_flux_from_velocity(np.array([1.0, 2.0, 3.0]), np.array([1.0]))


def test_plane_flux_from_velocity_area_mismatch():
    with pytest.raises(ValueError, match="area must be"):
        plane_flux_from_velocity(np.zeros((3, 3)), np.array([1.0, 2.0]))


# --- reduce_cycles ------------------------------------------------------------


def test_reduce_cycles_constant_series():
    flux = np.full(8, 2.0)  # 2 cycles of 4 steps
    res = reduce_cycles(flux, steps_per_cycle=4, n_discard=1)
    assert res.j_fan == pytest.approx(2.0)
    assert res.n_avg == 1
    assert res.j_fan_se == 0.0
    assert res.extend_recommended is False


def test_reduce_cycles_discards_first_and_averages_rest():
    # cycle means [10, 1, 2, 3, 4]; discard cycle 0 ⇒ mean([1,2,3,4]) = 2.5
    flux = np.array([10, 10, 1, 1, 2, 2, 3, 3, 4, 4], dtype=float)
    res = reduce_cycles(flux, steps_per_cycle=2, n_discard=1)
    assert res.per_cycle == pytest.approx((1.0, 2.0, 3.0, 4.0))
    assert res.j_fan == pytest.approx(2.5)
    assert res.n_avg == 4


def test_reduce_cycles_standard_error():
    flux = np.array([10, 10, 1, 1, 2, 2, 3, 3, 4, 4], dtype=float)
    res = reduce_cycles(flux, steps_per_cycle=2, n_discard=1)
    expected_se = np.std([1, 2, 3, 4], ddof=1) / np.sqrt(4)
    assert res.j_fan_se == pytest.approx(expected_se)


def test_reduce_cycles_extend_recommended_when_cycles_disagree():
    # retained cycle means 1 vs 2 differ 67% > 5% threshold
    flux = np.array([0, 0, 1, 1, 2, 2, 2, 2, 2, 2], dtype=float)
    res = reduce_cycles(flux, steps_per_cycle=2, n_discard=1)
    assert res.extend_recommended is True
    assert res.cycle2_vs_cycle3_rel_diff == pytest.approx(1.0 / 1.5)


def test_reduce_cycles_no_extend_when_cycles_agree():
    flux = np.array([0, 0, 2.00, 2.00, 2.02, 2.02, 2.0, 2.0, 2.0, 2.0], dtype=float)
    res = reduce_cycles(flux, steps_per_cycle=2, n_discard=1)
    assert res.extend_recommended is False


def test_reduce_cycles_peak_clips_negative():
    # retained cycle samples [-2, 4] ⇒ mean = 1.0, peak (clip<0) = mean([0,4]) = 2.0
    flux = np.array([0, 0, -2, 4], dtype=float)
    res = reduce_cycles(flux, steps_per_cycle=2, n_discard=1)
    assert res.j_fan == pytest.approx(1.0)
    assert res.j_fan_peak == pytest.approx(2.0)


def test_reduce_cycles_meta_records_counts():
    res = reduce_cycles(np.zeros(6), steps_per_cycle=2, n_discard=1)
    assert res.meta["n_cycles_total"] == 3.0
    assert res.meta["n_discard"] == 1.0


def test_reduce_cycles_rejects_non_1d():
    with pytest.raises(ValueError, match="must be 1D"):
        reduce_cycles(np.zeros((2, 2)), steps_per_cycle=2)


def test_reduce_cycles_rejects_non_multiple_length():
    with pytest.raises(ValueError, match="not a positive multiple"):
        reduce_cycles(np.zeros(7), steps_per_cycle=2)


def test_reduce_cycles_rejects_empty():
    with pytest.raises(ValueError, match="not a positive multiple"):
        reduce_cycles(np.zeros(0), steps_per_cycle=2)


def test_reduce_cycles_rejects_bad_steps_per_cycle():
    with pytest.raises(ValueError, match="steps_per_cycle"):
        reduce_cycles(np.zeros(4), steps_per_cycle=0)


def test_reduce_cycles_rejects_bad_period():
    with pytest.raises(ValueError, match="period_s"):
        reduce_cycles(np.zeros(4), steps_per_cycle=2, period_s=0.0)


def test_reduce_cycles_rejects_discard_all():
    with pytest.raises(ValueError, match="n_discard"):
        reduce_cycles(np.zeros(4), steps_per_cycle=2, n_discard=2)


def test_reduce_cycles_single_retained_cycle_zero_rel():
    res = reduce_cycles(np.array([1, 1, 5, 5], dtype=float), steps_per_cycle=2, n_discard=1)
    assert res.n_avg == 1
    assert res.cycle2_vs_cycle3_rel_diff == 0.0
    assert res.extend_recommended is False


# --- compute_j_fan_steady -----------------------------------------------------


def test_steady_proxy_delta():
    runs = [SteadyRun(2.0, "productive"), SteadyRun(0.5, "return")]
    res = compute_j_fan_steady(runs)
    assert res.proxy_kind == "delta"
    assert res.j_fan_steady_proxy == pytest.approx(1.5)
    assert res.drag_productive == pytest.approx(2.0)
    assert res.drag_return == pytest.approx(0.5)


def test_steady_proxy_one_direction_warns(caplog):
    with caplog.at_level(logging.WARNING):
        res = compute_j_fan_steady([SteadyRun(2.0, "productive")])
    assert res.proxy_kind == "one_direction"
    assert res.j_fan_steady_proxy == pytest.approx(2.0)
    assert res.drag_return is None
    assert "one-direction" in caplog.text


def test_steady_proxy_empty_raises():
    with pytest.raises(ValueError, match="at least one"):
        compute_j_fan_steady([])


def test_steady_proxy_duplicate_stroke_raises():
    with pytest.raises(ValueError, match="at most one run per stroke"):
        compute_j_fan_steady([SteadyRun(1.0, "productive"), SteadyRun(2.0, "productive")])


def test_steady_run_rejects_bad_stroke():
    with pytest.raises(ValueError, match="stroke must be"):
        SteadyRun(1.0, "sideways")


# --- compute_j_fan dispatcher -------------------------------------------------


def test_compute_j_fan_single_steady_run():
    res = compute_j_fan(SteadyRun(1.0, "return"))
    assert isinstance(res, SteadyProxyResult)
    assert res.proxy_kind == "one_direction"


def test_compute_j_fan_steady_sequence():
    res = compute_j_fan([SteadyRun(2.0, "productive"), SteadyRun(0.5, "return")])
    assert isinstance(res, SteadyProxyResult)
    assert res.j_fan_steady_proxy == pytest.approx(1.5)


def test_compute_j_fan_unsteady_series():
    res = compute_j_fan(np.full(8, 3.0), steps_per_cycle=4, n_discard=1)
    assert isinstance(res, UnsteadyResult)
    assert res.j_fan == pytest.approx(3.0)


def test_compute_j_fan_rejects_2d():
    with pytest.raises(TypeError, match="1D instantaneous-flux"):
        compute_j_fan(np.zeros((2, 2)))


def test_compute_j_fan_rejects_string():
    with pytest.raises(TypeError):
        compute_j_fan("not-a-series")


def test_compute_j_fan_rejects_non_numeric_object():
    with pytest.raises(TypeError, match="could not read"):
        compute_j_fan(object())
