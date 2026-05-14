"""Unit tests for fanopt.physical.imu.

Validates W_cycle and kinematic-sanity extraction against analytic SHM
references. Synthesizes IMU traces at the locked spec amplitudes/frequencies
(2 Hz, 8.8 rad/s, 0.7 rad) so the sanity flags hit `True` on a clean trace.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.3, §3.2.3 kinematics.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from fanopt.physical.imu import (
    F_WAVE_TARGET_HZ,
    OMEGA_MAX_TARGET_RAD_PER_S,
    THETA_MAX_TARGET_RAD,
    IMUTrace,
    _cumulative_trapezoid,
    _dominant_frequency,
    analyze_imu_trace,
    compute_w_cycle,
    load_imu_csv,
)


# Locked kinematics (§3.2.3).
F_HZ = F_WAVE_TARGET_HZ          # 2.0 Hz
OMEGA = 2.0 * math.pi * F_HZ     # 12.566 rad/s SHM angular frequency
THETA_MAX = THETA_MAX_TARGET_RAD # 0.6981 rad = 40°
OMEGA_MAX = OMEGA_MAX_TARGET_RAD_PER_S  # 8.8 rad/s = θ_max · ω_SHM
SAMPLE_HZ = 500.0


def _shm_trace(duration_s: float = 5.0, theta_max: float = THETA_MAX,
               omega_shm: float = OMEGA, sample_hz: float = SAMPLE_HZ) -> IMUTrace:
    """θ(t) = θ_max · sin(ω·t), ω(t) = θ_max · ω · cos(ω·t)."""
    t = np.arange(0.0, duration_s, 1.0 / sample_hz)
    theta = theta_max * np.sin(omega_shm * t)
    omega = theta_max * omega_shm * np.cos(omega_shm * t)
    return IMUTrace(t_s=t, theta_rad=theta, omega_rad_per_s=omega)


# ----- W_cycle analytics ----------------------------------------------------


def test_w_cycle_matches_analytic_2_I_omega_max_squared() -> None:
    """For ideal SHM, ∫|I·ω·(dω/dt)| dt over one cycle = 2·I·ω_max².

    Proof sketch: dω/dt = -θ_max·ω²·sin(ω·t). So ω·(dω/dt) =
    -θ_max²·ω³·sin·cos = -½·θ_max²·ω³·sin(2ω·t). |·| integrated over one
    period T = 2π/ω: ∫₀ᵀ |sin(2ω·t)| dt = 2/ω. So
    ∫|ω·(dω/dt)| dt over one cycle = ½·θ_max²·ω³ · (2/ω) = θ_max²·ω² = ω_max².
    Multiply by I and we get I·ω_max². But the rectified integral covers TWO
    half-cycles of |sin|, so over one full period we get 2·I·ω_max².
    """
    I = 1.0e-3  # kg·m²
    # 5 full cycles to make n_cycles-averaging robust against edge effects.
    trace = _shm_trace(duration_s=5.0 / F_HZ)
    W = compute_w_cycle(trace, I)
    expected = 2.0 * I * OMEGA_MAX ** 2
    assert W == pytest.approx(expected, rel=0.02)


def test_w_cycle_scales_with_I_wrist() -> None:
    """W ∝ I exactly."""
    trace = _shm_trace(duration_s=2.5)
    W_a = compute_w_cycle(trace, 1.0e-3)
    W_b = compute_w_cycle(trace, 2.5e-3)
    assert W_b / W_a == pytest.approx(2.5, rel=1e-6)


def test_w_cycle_rejects_nonpositive_inertia() -> None:
    trace = _shm_trace()
    with pytest.raises(ValueError):
        compute_w_cycle(trace, 0.0)


# ----- kinematic-sanity flags -----------------------------------------------


def test_sanity_flags_pass_on_clean_locked_trace() -> None:
    trace = _shm_trace(duration_s=5.0)
    res = analyze_imu_trace(trace, I_wrist_kgm2=1.0e-3)
    assert res.f_wave_ok is True
    assert res.omega_max_ok is True
    assert res.theta_max_ok is True
    assert res.sanity_ok is True
    assert res.f_wave_Hz == pytest.approx(F_HZ, rel=0.05)
    assert res.omega_max_rad_per_s == pytest.approx(OMEGA_MAX, rel=0.01)
    assert res.theta_max_rad == pytest.approx(THETA_MAX, rel=0.01)


def test_sanity_flag_fires_when_amplitude_too_low() -> None:
    """Operator with weak wrist: θ_max = 0.3 rad → ω_max = 0.3·ω ≈ 3.8 → fails."""
    trace = _shm_trace(duration_s=5.0, theta_max=0.30)
    res = analyze_imu_trace(trace, I_wrist_kgm2=1.0e-3)
    assert res.omega_max_ok is False
    assert res.theta_max_ok is False
    assert res.sanity_ok is False


def test_sanity_flag_fires_when_frequency_drifts() -> None:
    """Operator at 1.5 Hz cadence (25% slow) — outside the 10% f_wave band."""
    omega_slow = 2.0 * math.pi * 1.5
    trace = _shm_trace(duration_s=5.0, omega_shm=omega_slow)
    res = analyze_imu_trace(trace, I_wrist_kgm2=1.0e-3)
    assert res.f_wave_ok is False
    assert res.sanity_ok is False


# ----- CSV loader -----------------------------------------------------------


def test_load_imu_csv_round_trip(tmp_path: Path) -> None:
    """Write a CSV, load it, get an equivalent IMUTrace back."""
    trace = _shm_trace(duration_s=2.5)
    path = tmp_path / "trial1.csv"
    lines = ["t_s,theta_rad,omega_rad_per_s"]
    for ti, th, om in zip(trace.t_s, trace.theta_rad, trace.omega_rad_per_s):
        lines.append(f"{ti:.6f},{th:.8f},{om:.8f}")
    path.write_text("\n".join(lines) + "\n")

    loaded = load_imu_csv(path)
    # %.6f / %.8f format strings lose ≲1e-8 of precision; compare with atol.
    np.testing.assert_allclose(loaded.t_s, trace.t_s, atol=1e-6)
    np.testing.assert_allclose(loaded.theta_rad, trace.theta_rad, atol=1e-7)
    np.testing.assert_allclose(loaded.omega_rad_per_s, trace.omega_rad_per_s, atol=1e-7)


def test_load_imu_csv_synthesizes_theta_when_omitted(tmp_path: Path) -> None:
    """If theta_rad column is missing, reconstruct it from omega via ∫ω dt."""
    trace = _shm_trace(duration_s=2.5)
    path = tmp_path / "trial_no_theta.csv"
    lines = ["t_s,omega_rad_per_s"]
    for ti, om in zip(trace.t_s, trace.omega_rad_per_s):
        lines.append(f"{ti:.6f},{om:.8f}")
    path.write_text("\n".join(lines) + "\n")

    loaded = load_imu_csv(path)
    # Reconstructed θ should match the analytic to within trapezoid accuracy.
    np.testing.assert_allclose(loaded.theta_rad, trace.theta_rad, atol=0.005)


def test_load_imu_csv_rejects_nonmonotonic_time(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(
        "t_s,theta_rad,omega_rad_per_s\n"
        "0.000,0.0,0.0\n"
        "0.010,0.0,0.0\n"
        "0.005,0.0,0.0\n"
        "0.020,0.0,0.0\n"
        "0.025,0.0,0.0\n"
        "0.030,0.0,0.0\n"
        "0.035,0.0,0.0\n"
        "0.040,0.0,0.0\n"
    )
    with pytest.raises(ValueError, match="monotonically"):
        load_imu_csv(path)


def test_load_imu_csv_rejects_missing_omega(tmp_path: Path) -> None:
    path = tmp_path / "no_omega.csv"
    path.write_text("t_s,theta_rad\n0.0,0.0\n0.01,0.01\n")
    with pytest.raises(ValueError, match="omega"):
        load_imu_csv(path)


def test_load_imu_csv_skips_comments_and_blanks(tmp_path: Path) -> None:
    trace = _shm_trace(duration_s=1.5)
    path = tmp_path / "with_comments.csv"
    lines = ["# header comment", "", "t_s,theta_rad,omega_rad_per_s"]
    for ti, th, om in zip(trace.t_s, trace.theta_rad, trace.omega_rad_per_s):
        lines.append(f"{ti:.6f},{th:.8f},{om:.8f}")
    path.write_text("\n".join(lines) + "\n")
    loaded = load_imu_csv(path)
    assert loaded.n_samples == trace.n_samples


def test_load_imu_csv_rejects_empty_file(tmp_path: Path) -> None:
    """No header, no data → ValueError."""
    path = tmp_path / "empty.csv"
    path.write_text("# only comments\n\n")
    with pytest.raises(ValueError, match="no rows found"):
        load_imu_csv(path)


def test_load_imu_csv_rejects_missing_t_s_column(tmp_path: Path) -> None:
    path = tmp_path / "no_t.csv"
    path.write_text("theta_rad,omega_rad_per_s\n0.0,0.0\n0.0,0.0\n")
    with pytest.raises(ValueError, match="t_s"):
        load_imu_csv(path)


def test_load_imu_csv_rejects_too_few_samples(tmp_path: Path) -> None:
    """t.size < 8 → ValueError before we waste time analyzing it."""
    path = tmp_path / "short.csv"
    lines = ["t_s,theta_rad,omega_rad_per_s"]
    for i in range(5):  # < 8 samples
        lines.append(f"{i*0.01},0.0,0.0")
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="must cover ≥1 cycle"):
        load_imu_csv(path)


# ----- IMUTrace edge cases --------------------------------------------------


def test_imu_trace_sample_rate_zero_when_single_sample() -> None:
    """Edge case: 1-sample trace returns 0 Hz from sample_rate_hz."""
    trace = IMUTrace(
        t_s=np.array([0.0]),
        theta_rad=np.array([0.0]),
        omega_rad_per_s=np.array([0.0]),
    )
    assert trace.sample_rate_hz == 0.0
    assert trace.duration_s == 0.0


def test_imu_trace_duration_and_sample_rate_on_normal_trace() -> None:
    trace = _shm_trace(duration_s=2.0, sample_hz=500.0)
    # np.arange is right-exclusive, so the actual last sample is t = 2.0 - 1/500.
    assert trace.duration_s == pytest.approx(2.0 - 1.0 / 500.0, abs=1e-6)
    assert trace.sample_rate_hz == pytest.approx(500.0, rel=0.01)


# ----- _dominant_frequency / _cumulative_trapezoid internals ----------------


def test_dominant_frequency_short_signal_returns_zero() -> None:
    """Direct test of the n<8 fallback in _dominant_frequency."""
    t = np.linspace(0, 1, 4)
    s = np.array([0.0, 1.0, 0.0, -1.0])
    assert _dominant_frequency(s, t) == 0.0


def test_dominant_frequency_zero_dt_returns_zero() -> None:
    """Degenerate t (all zeros) gives dt_mean ≤ 0 → 0 fallback."""
    t = np.zeros(16)
    s = np.linspace(-1, 1, 16)
    assert _dominant_frequency(s, t) == 0.0


def test_cumulative_trapezoid_size_mismatch() -> None:
    """Internal helper validates input sizes."""
    with pytest.raises(ValueError, match="size mismatch"):
        _cumulative_trapezoid(np.array([1.0, 2.0]), np.array([0.0, 1.0, 2.0]))


def test_cumulative_trapezoid_first_value_is_zero() -> None:
    """Integration starts at 0 by construction."""
    y = np.array([1.0, 1.0, 1.0, 1.0])
    x = np.array([0.0, 1.0, 2.0, 3.0])
    out = _cumulative_trapezoid(y, x)
    assert out[0] == 0.0
    # ∫1 dx from 0 to 3 = 3
    assert out[-1] == pytest.approx(3.0, rel=1e-12)
