"""Tests for scripts/diagnose_su2_pitching_regime.py.

Exercises the classification logic + cross-correlation phase detector
against synthetic SU2 histories with known force-regime properties.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import diagnose_su2_pitching_regime as diag

# ---------------------------------------------------------------------------
# Synthetic history generators
# ---------------------------------------------------------------------------


def _write_history(
    path: Path,
    *,
    n_cycles: int,
    n_per_cycle: int,
    cl_signal: np.ndarray,
    cd_signal: np.ndarray | None = None,
) -> Path:
    """Write a SU2-style history.csv with prescribed CL / CD per outer iter."""
    n_total = n_cycles * n_per_cycle
    assert len(cl_signal) == n_total
    if cd_signal is None:
        cd_signal = np.full(n_total, 0.05)
    period_s = 1.0
    dt = period_s / n_per_cycle
    lines = ["Time_Iter,Cur_Time,CL,CD"]
    for i in range(n_total):
        lines.append(f"{i},{i * dt:.6f},{cl_signal[i]:.6e},{cd_signal[i]:.6e}")
    path.write_text("\n".join(lines) + "\n")
    return path


def _wind_tunnel_cl(n_per_cycle: int, n_cycles: int, theta_max: float) -> np.ndarray:
    """CL ∝ α (in phase): wind-tunnel-like behavior."""
    t = np.arange(n_per_cycle * n_cycles) * (1.0 / n_per_cycle)
    omega = 2.0 * math.pi  # period = 1 s
    alpha = theta_max * np.sin(omega * t)
    return 5.0 * alpha  # CL = lift slope × α


def _added_mass_cl(n_per_cycle: int, n_cycles: int, theta_max: float) -> np.ndarray:
    """CL ∝ dα/dt (90° lead): added-mass-like behavior."""
    t = np.arange(n_per_cycle * n_cycles) * (1.0 / n_per_cycle)
    omega = 2.0 * math.pi
    # dα/dt = θ_max · ω · cos(ωt) — leads α=sin(ωt) by 90°.
    dalpha_dt = theta_max * omega * np.cos(omega * t)
    return 0.1 * dalpha_dt


def _biased_cl(n_per_cycle: int, n_cycles: int, theta_max: float) -> np.ndarray:
    """Wind-tunnel-like CL with a large additive bias (non-physical mean)."""
    base = _wind_tunnel_cl(n_per_cycle, n_cycles, theta_max)
    bias = 10.0 * theta_max  # bias ≈ 2× the amplitude → bias_ratio >> 0.2
    return base + bias


# ---------------------------------------------------------------------------
# _normalise_lag_degrees
# ---------------------------------------------------------------------------


def test_normalise_lag_in_range_unchanged() -> None:
    assert diag._normalise_lag_degrees(45.0) == 45.0
    assert diag._normalise_lag_degrees(-90.0) == -90.0
    assert diag._normalise_lag_degrees(180.0) == 180.0


def test_normalise_lag_wraps_above_180() -> None:
    assert diag._normalise_lag_degrees(270.0) == pytest.approx(-90.0)


def test_normalise_lag_wraps_below_minus_180() -> None:
    assert diag._normalise_lag_degrees(-270.0) == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# _phase_lag_via_xcorr
# ---------------------------------------------------------------------------


def _multi_period_t(n_periods: int = 5, n_per_period: int = 200) -> tuple[np.ndarray, float]:
    """Return (t, dt) for n_periods full periods. Multi-period signals avoid
    the cross-correlation edge effects that bias single-period estimates."""
    n = n_periods * n_per_period
    dt = 1.0 / n_per_period  # period = 1 s
    t = np.arange(n) * dt
    return t, dt


def test_phase_lag_zero_for_in_phase_signal() -> None:
    """sin(ωt) vs sin(ωt) → lag ≈ 0°."""
    t, dt = _multi_period_t()
    omega = 2.0 * math.pi
    alpha = np.sin(omega * t)
    cl = 2.5 * alpha  # in-phase, just scaled
    lag = diag._phase_lag_via_xcorr(cl, alpha, dt, omega)
    assert abs(lag) < 5.0  # within 5° of perfectly in-phase


def test_phase_lag_90deg_for_cosine_signal() -> None:
    """cos(ωt) leads sin(ωt) by 90°. Multi-period signal so the
    cross-correlation peak is well-resolved (single-period inputs hit
    edge effects that bias the estimate by ~10°)."""
    t, dt = _multi_period_t()
    omega = 2.0 * math.pi
    alpha = np.sin(omega * t)
    cl = np.cos(omega * t)  # leads α by 90°
    lag = diag._phase_lag_via_xcorr(cl, alpha, dt, omega)
    # CL = cos = sin(ωt + π/2) → CL is α shifted FORWARD in time → NEGATIVE lag.
    assert -95.0 < lag < -85.0  # within 5° of -90°


def test_phase_lag_180deg_for_negated_signal() -> None:
    """-sin(ωt) is anti-phase with sin(ωt) → lag ≈ ±180°."""
    t, dt = _multi_period_t()
    omega = 2.0 * math.pi
    alpha = np.sin(omega * t)
    cl = -alpha
    lag = diag._phase_lag_via_xcorr(cl, alpha, dt, omega)
    assert abs(abs(lag) - 180.0) < 5.0


def test_phase_lag_zero_for_zero_variance_signal() -> None:
    """Constant CL has no variance; phase lag is undefined → return 0."""
    n = 100
    omega = 2.0 * math.pi
    dt = 0.01
    cl = np.full(n, 0.5)
    alpha = np.sin(omega * np.arange(n) * dt)
    assert diag._phase_lag_via_xcorr(cl, alpha, dt, omega) == 0.0


# ---------------------------------------------------------------------------
# _classify
# ---------------------------------------------------------------------------


def test_classify_wind_tunnel_low_lag_no_bias() -> None:
    label, findings = diag._classify(bias_ratio=0.05, phase_lag_deg=15.0)
    assert label == "WIND_TUNNEL_LIKE"
    assert any("WIND-TUNNEL" in f for f in findings)


def test_classify_added_mass_90deg_lag_no_bias() -> None:
    label, findings = diag._classify(bias_ratio=0.05, phase_lag_deg=85.0)
    assert label == "ADDED_MASS_DOMINANCE"
    assert any("ADDED-MASS" in f for f in findings)


def test_classify_added_mass_negative_lag() -> None:
    """A 90° lead is the same regime as a 90° lag — abs value matters."""
    label, _findings = diag._classify(bias_ratio=0.05, phase_lag_deg=-90.0)
    assert label == "ADDED_MASS_DOMINANCE"


def test_classify_anti_phase_at_180() -> None:
    label, findings = diag._classify(bias_ratio=0.05, phase_lag_deg=178.0)
    assert label == "ANTI_PHASE"
    assert any("ANTI-PHASE" in f for f in findings)


def test_classify_intermediate_at_45() -> None:
    label, _findings = diag._classify(bias_ratio=0.05, phase_lag_deg=45.0)
    assert label == "INTERMEDIATE"


def test_classify_bias_warning_added_independent_of_phase() -> None:
    """High bias flag fires alongside the phase classification."""
    _label, findings = diag._classify(bias_ratio=0.5, phase_lag_deg=15.0)
    assert any("NON-PHYSICAL BIAS" in f for f in findings)
    # Phase classification still present.
    assert any("WIND-TUNNEL" in f for f in findings)


def test_classify_bias_just_under_threshold_no_warning() -> None:
    _label, findings = diag._classify(bias_ratio=0.15, phase_lag_deg=15.0)
    assert not any("NON-PHYSICAL BIAS" in f for f in findings)


# ---------------------------------------------------------------------------
# diagnose_history — end-to-end against synthetic histories
# ---------------------------------------------------------------------------


def test_diagnose_wind_tunnel_history_classifies_correctly(tmp_path: Path) -> None:
    n_cycles, n_per = 5, 200
    cl = _wind_tunnel_cl(n_per, n_cycles, theta_max=math.radians(10.0))
    history = _write_history(
        tmp_path / "history.csv", n_cycles=n_cycles, n_per_cycle=n_per, cl_signal=cl
    )
    result = diag.diagnose_history(
        history,
        theta_max_rad=math.radians(10.0),
        omega_shm_rad_per_s=2.0 * math.pi,
        n_cycles=n_cycles,
    )
    assert result["classification"] == "WIND_TUNNEL_LIKE"


def test_diagnose_added_mass_history_classifies_correctly(tmp_path: Path) -> None:
    n_cycles, n_per = 5, 200
    cl = _added_mass_cl(n_per, n_cycles, theta_max=math.radians(10.0))
    history = _write_history(
        tmp_path / "history.csv", n_cycles=n_cycles, n_per_cycle=n_per, cl_signal=cl
    )
    result = diag.diagnose_history(
        history,
        theta_max_rad=math.radians(10.0),
        omega_shm_rad_per_s=2.0 * math.pi,
        n_cycles=n_cycles,
    )
    assert result["classification"] == "ADDED_MASS_DOMINANCE"


def test_diagnose_biased_history_flags_non_physical(tmp_path: Path) -> None:
    n_cycles, n_per = 5, 200
    cl = _biased_cl(n_per, n_cycles, theta_max=math.radians(10.0))
    history = _write_history(
        tmp_path / "history.csv", n_cycles=n_cycles, n_per_cycle=n_per, cl_signal=cl
    )
    result = diag.diagnose_history(
        history,
        theta_max_rad=math.radians(10.0),
        omega_shm_rad_per_s=2.0 * math.pi,
        n_cycles=n_cycles,
    )
    # Wind-tunnel-like phase, but the bias warning is independent.
    assert result["classification"] == "WIND_TUNNEL_LIKE"
    assert any("NON-PHYSICAL BIAS" in f for f in result["findings"])
    assert result["metrics"]["bias_ratio_mean_over_amplitude"] > 0.2


def test_diagnose_cycle_zero_discarded(tmp_path: Path) -> None:
    """Cycle 0's data should NOT contribute to the kept-cycle metrics."""
    n_cycles, n_per = 5, 200
    # Wind-tunnel signal on cycles 1-4; cycle 0 is huge garbage.
    cl_clean = _wind_tunnel_cl(n_per, n_cycles, theta_max=math.radians(10.0))
    cl_with_transient = cl_clean.copy()
    cl_with_transient[:n_per] = 1.0e9  # garbage in cycle 0
    history = _write_history(
        tmp_path / "history.csv",
        n_cycles=n_cycles,
        n_per_cycle=n_per,
        cl_signal=cl_with_transient,
    )
    result = diag.diagnose_history(
        history,
        theta_max_rad=math.radians(10.0),
        omega_shm_rad_per_s=2.0 * math.pi,
        n_cycles=n_cycles,
    )
    # Should classify as WIND_TUNNEL_LIKE if the 1e9 transient was correctly
    # discarded (otherwise the cycle-mean would be huge and the bias warning
    # would dominate).
    assert result["classification"] == "WIND_TUNNEL_LIKE"
    # Amplitude is bounded by the clean signal, NOT the 1e9 transient.
    assert result["metrics"]["cl_amplitude_kept_cycles"] < 100.0


def test_diagnose_rejects_too_few_iters(tmp_path: Path) -> None:
    n_cycles, n_per = 5, 1  # only 5 iters total
    cl = np.zeros(5)
    history = _write_history(
        tmp_path / "history.csv", n_cycles=n_cycles, n_per_cycle=n_per, cl_signal=cl
    )
    with pytest.raises(ValueError, match="≥ 2 per cycle"):
        diag.diagnose_history(
            history,
            theta_max_rad=math.radians(10.0),
            omega_shm_rad_per_s=2.0 * math.pi,
            n_cycles=n_cycles,
        )


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_writes_json_and_png(tmp_path: Path) -> None:
    n_cycles, n_per = 5, 200
    cl = _wind_tunnel_cl(n_per, n_cycles, theta_max=math.radians(10.0))
    history = _write_history(
        tmp_path / "history.csv", n_cycles=n_cycles, n_per_cycle=n_per, cl_signal=cl
    )
    out_json = tmp_path / "diag.json"
    out_png = tmp_path / "diag.png"
    rc = diag.main(
        [
            "--history",
            str(history),
            "--omega-shm-rad-per-s",
            str(2.0 * math.pi),
            "--theta-max-rad",
            str(math.radians(10.0)),
            "--n-cycles",
            str(n_cycles),
            "--out-json",
            str(out_json),
            "--out-png",
            str(out_png),
        ]
    )
    assert rc == 0
    assert out_json.exists()
    assert out_png.exists()
    payload = json.loads(out_json.read_text())
    assert payload["classification"] == "WIND_TUNNEL_LIKE"


def test_cli_no_plot_skips_png(tmp_path: Path) -> None:
    n_cycles, n_per = 5, 200
    cl = _wind_tunnel_cl(n_per, n_cycles, theta_max=math.radians(10.0))
    history = _write_history(
        tmp_path / "history.csv", n_cycles=n_cycles, n_per_cycle=n_per, cl_signal=cl
    )
    out_json = tmp_path / "diag.json"
    out_png = tmp_path / "diag.png"
    rc = diag.main(
        [
            "--history",
            str(history),
            "--omega-shm-rad-per-s",
            str(2.0 * math.pi),
            "--out-json",
            str(out_json),
            "--out-png",
            str(out_png),
            "--no-plot",
        ]
    )
    assert rc == 0
    assert out_json.exists()
    assert not out_png.exists()


def test_cli_missing_history_exits_2(tmp_path: Path) -> None:
    rc = diag.main(
        [
            "--history",
            str(tmp_path / "nope.csv"),
            "--omega-shm-rad-per-s",
            str(2.0 * math.pi),
        ]
    )
    assert rc == 2
