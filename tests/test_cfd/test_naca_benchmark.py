"""Tests for fanopt.cfd.naca_benchmark (config derivation + history reduction)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fanopt.cfd.naca_benchmark import (
    GAMMA_AIR,
    R_SPECIFIC_AIR_J_KGK,
    BenchmarkConfig,
    _loop_area,
    benchmark_metrics,
    pitch_angle_series,
)

# --- config validation ---


def test_config_rejects_supersonic_mach():
    with pytest.raises(ValueError, match="mach_number"):
        BenchmarkConfig(mach_number=1.5)


def test_config_rejects_negative_reduced_frequency():
    with pytest.raises(ValueError, match="reduced_frequency_k"):
        BenchmarkConfig(reduced_frequency_k=-0.1)


def test_config_rejects_out_of_range_pitch_axis():
    with pytest.raises(ValueError, match="motion_origin_frac"):
        BenchmarkConfig(motion_origin_frac=1.5)


# --- derived quantities ---


def test_freestream_velocity_matches_mach_times_sound_speed():
    cfg = BenchmarkConfig(mach_number=0.05, freestream_temperature_k=300.0)
    a = math.sqrt(GAMMA_AIR * R_SPECIFIC_AIR_J_KGK * 300.0)
    assert cfg.freestream_velocity_ms == pytest.approx(0.05 * a)


def test_pitching_omega_from_reduced_frequency():
    cfg = BenchmarkConfig(reduced_frequency_k=0.55, chord_m=1.0)
    expected = 2.0 * cfg.freestream_velocity_ms * 0.55 / 1.0
    assert cfg.pitching_omega_rad_s == pytest.approx(expected)


def test_time_step_and_iter_span_the_requested_cycles():
    cfg = BenchmarkConfig(n_cycles=5, steps_per_cycle=200)
    assert cfg.time_iter == 1000
    assert cfg.time_step_s == pytest.approx(cfg.period_s / 200)
    assert cfg.max_time_s == pytest.approx(5 * cfg.period_s)


def test_quarter_chord_pitch_axis():
    cfg = BenchmarkConfig(motion_origin_frac=0.25, chord_m=0.8)
    assert cfg.motion_origin_x_m == pytest.approx(0.2)


# --- pitch angle series ---


def test_pitch_series_length_is_time_iter():
    cfg = BenchmarkConfig(n_cycles=3, steps_per_cycle=50)
    assert pitch_angle_series(cfg).size == 150


def test_pitch_series_amplitude_matches_config():
    cfg = BenchmarkConfig(pitch_amplitude_deg=10.0, steps_per_cycle=360, n_cycles=1)
    series = pitch_angle_series(cfg)
    assert np.max(np.abs(series)) == pytest.approx(math.radians(10.0), rel=1e-3)


# --- loop area (hysteresis) ---


def test_loop_area_of_unit_circle_is_pi():
    theta = np.linspace(0, 2 * math.pi, 2000, endpoint=False)
    assert _loop_area(np.cos(theta), np.sin(theta)) == pytest.approx(math.pi, rel=1e-3)


def test_loop_area_of_degenerate_line_is_zero():
    x = np.array([0.0, 1.0, 2.0, 1.0])
    assert _loop_area(x, np.zeros_like(x)) == pytest.approx(0.0)


# --- metric reduction ---


def _synthetic(steps_per_cycle: int, n_cycles: int):
    cl = np.tile([0.0, 1.0, 0.0, -1.0], n_cycles)
    cd = np.tile([0.10, 0.20, 0.10, 0.20], n_cycles)
    alpha = np.tile([0.0, math.radians(10.0), 0.0, math.radians(-10.0)], n_cycles)
    assert steps_per_cycle == 4
    return cl, cd, alpha


def test_metrics_cl_max_over_post_transient_cycles():
    cl, cd, alpha = _synthetic(4, 3)
    m = benchmark_metrics(cl, cd, alpha, steps_per_cycle=4, n_discard=1)
    assert m.c_l_max == pytest.approx(1.0)


def test_metrics_cd_mean_over_post_transient_cycles():
    cl, cd, alpha = _synthetic(4, 3)
    m = benchmark_metrics(cl, cd, alpha, steps_per_cycle=4, n_discard=1)
    assert m.c_d_mean == pytest.approx(0.15)


def test_metrics_reports_alpha_at_cl_max_and_cycle_count():
    cl, cd, alpha = _synthetic(4, 3)
    m = benchmark_metrics(cl, cd, alpha, steps_per_cycle=4, n_discard=1)
    assert m.alpha_at_cl_max_deg == pytest.approx(10.0)
    assert m.n_cycles_used == 2
    assert math.isfinite(m.hysteresis_area)


def test_metrics_rejects_too_few_cycles():
    cl, cd, alpha = _synthetic(4, 1)
    with pytest.raises(ValueError, match="whole cycles"):
        benchmark_metrics(cl, cd, alpha, steps_per_cycle=4, n_discard=1)


def test_metrics_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        benchmark_metrics(np.zeros(8), np.zeros(8), np.zeros(4), steps_per_cycle=4)
