"""Tests for fanopt.physical.calibration (measured vs predicted reduction)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fanopt.physical.anemometer import GRID_POINTS_M
from fanopt.physical.calibration import DesignMeasurement, calibrate, reduce_design


def _imu_csv(tmp_path, name="imu.csv", omega_amp=8.8, f_wave=2.0):
    fs = 200.0
    t = np.arange(0.0, 2.0, 1.0 / fs)
    omega = omega_amp * np.sin(2 * math.pi * f_wave * t)
    lines = ["t_s,omega_rad_per_s"] + [f"{ti},{wi}" for ti, wi in zip(t, omega, strict=False)]
    p = tmp_path / name
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _anemometer_csv(tmp_path, name="anemometer.csv", v=0.5):
    header = "point,x_m,y_m,z_m,v_mean_m_per_s,v_peak_m_per_s,notes"
    rows = [f"p{i + 1},{x},{y},0.3,{v},{v * 2},''" for i, (x, y) in enumerate(GRID_POINTS_M)]
    p = tmp_path / name
    p.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return p


def _acoustic_csv(tmp_path, name="acoustic.csv", freq=400.0, amp=0.4):
    fs = 8000.0
    t = np.arange(0.0, 0.5, 1.0 / fs)
    x = amp * np.sin(2 * math.pi * freq * t)
    lines = ["t_s,pressure_pa"] + [f"{ti},{xi}" for ti, xi in zip(t, x, strict=False)]
    p = tmp_path / name
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# --- reduce_design: presence / absence ---


def test_reduce_no_measurements_is_all_none(tmp_path):
    d = reduce_design("b8_i0", predicted_i_wrist_kgm2=0.008, predicted_j_fan_3d=1.1e10)
    assert d.n_measurements == 0
    assert d.w_cycle_j is None and d.j_fan_proxy_n is None and d.spl_db is None


def test_reduce_counts_present_channels(tmp_path):
    d = reduce_design(
        "b8_i0",
        predicted_i_wrist_kgm2=0.008,
        predicted_j_fan_3d=1.1e10,
        blade_count=8,
        imu_csv=_imu_csv(tmp_path),
        anemometer_csv=_anemometer_csv(tmp_path),
        acoustic_csv=_acoustic_csv(tmp_path),
    )
    assert d.n_measurements == 3
    assert d.w_cycle_j is not None and d.w_cycle_j > 0
    assert d.j_fan_proxy_n == pytest.approx(1.225 * 0.5 * 0.0182, rel=0.2) or d.j_fan_proxy_n > 0
    assert d.spl_db is not None


def test_reduce_skips_missing_paths(tmp_path):
    d = reduce_design(
        "b8_i0",
        predicted_i_wrist_kgm2=0.008,
        imu_csv=tmp_path / "nope.csv",
        anemometer_csv=_anemometer_csv(tmp_path),
    )
    assert d.n_measurements == 1
    assert d.w_cycle_j is None
    assert d.j_fan_proxy_n is not None


def test_reduce_imu_without_i_wrist_raises(tmp_path):
    with pytest.raises(ValueError, match="predicted_i_wrist_kgm2 is None"):
        reduce_design("b8_i0", imu_csv=_imu_csv(tmp_path))


def test_reduce_acoustic_uses_blade_pass_tone(tmp_path):
    # blade_count 200 × f_wave 2 Hz = 400 Hz — matches the injected tone.
    d = reduce_design(
        "x", blade_count=200, acoustic_csv=_acoustic_csv(tmp_path, freq=400.0, amp=0.4)
    )
    assert d.blade_pass_level_db is not None
    # 0.4 Pa tone → SPL = 20·log10((0.4/√2)/20µPa) ≈ 83 dB at the blade-pass bin.
    assert d.blade_pass_level_db == pytest.approx(83.0, abs=1.0)


# --- calibrate: cross-design rank ---


def test_calibrate_rank_preserved_when_measured_tracks_predicted():
    # predicted j_fan_3d ascending; measured proxy ascending → τ = 1.
    designs = [
        DesignMeasurement("a", predicted_j_fan_3d=1.0, j_fan_proxy_n=0.10),
        DesignMeasurement("b", predicted_j_fan_3d=2.0, j_fan_proxy_n=0.20),
        DesignMeasurement("c", predicted_j_fan_3d=3.0, j_fan_proxy_n=0.30),
    ]
    report = calibrate(designs)
    assert report["j_fan_rank"]["rank_preserved"] is True
    assert report["j_fan_rank"]["kendall_tau"] == pytest.approx(1.0)


def test_calibrate_rank_broken_when_measured_inverts_predicted():
    designs = [
        DesignMeasurement("a", predicted_j_fan_3d=1.0, j_fan_proxy_n=0.30),
        DesignMeasurement("b", predicted_j_fan_3d=2.0, j_fan_proxy_n=0.20),
        DesignMeasurement("c", predicted_j_fan_3d=3.0, j_fan_proxy_n=0.10),
    ]
    assert calibrate(designs)["j_fan_rank"]["rank_preserved"] is False


def test_calibrate_rank_none_with_fewer_than_two_paired():
    designs = [
        DesignMeasurement("a", predicted_j_fan_3d=1.0, j_fan_proxy_n=0.10),
        DesignMeasurement("b", predicted_j_fan_3d=2.0, j_fan_proxy_n=None),
    ]
    assert calibrate(designs)["j_fan_rank"]["rank_preserved"] is None


def test_calibrate_ignores_non_finite_pairs():
    designs = [
        DesignMeasurement("a", predicted_j_fan_3d=float("nan"), j_fan_proxy_n=0.10),
        DesignMeasurement("b", predicted_j_fan_3d=2.0, j_fan_proxy_n=0.20),
    ]
    assert calibrate(designs)["j_fan_rank"]["n"] == 1


def test_calibrate_counts_channels():
    designs = [
        DesignMeasurement("a", w_cycle_j=0.5, j_fan_proxy_n=0.1, spl_db=60.0, predicted_j_fan_3d=1.0),
        DesignMeasurement("b", w_cycle_j=0.6, predicted_j_fan_3d=2.0, j_fan_proxy_n=0.2),
    ]
    report = calibrate(designs)
    assert report["n_with_imu"] == 2
    assert report["n_with_anemometer"] == 2
    assert report["n_with_acoustic"] == 1
