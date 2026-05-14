"""CLI tests for scripts/run_spike_0_3_baseline.py.

Exercises:
- happy-path end-to-end (synthetic IMU + 9-point grid + Spike 0.2 inertia.json).
- inertia.json parsing (missing file, malformed JSON, missing key).
- IMU error path (bad CSV).
- anemometer error path (wrong row count, locked-grid mismatch).
- baseline.json schema sanity (Phase 6 consumes it).
- exit code is always 0 even on sanity warnings (warnings don't fail the run).

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.3; protocol in
docs/spike_0_3_protocol.md.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import run_spike_0_3_baseline as cli
from fanopt.physical.anemometer import GRID_POINTS_M, PLANE_AREA_M2, RHO_AIR_KG_PER_M3

# ---- fixtures --------------------------------------------------------------


F_HZ = 2.0
OMEGA_SHM = 2.0 * math.pi * F_HZ
THETA_MAX = 0.6981  # 40°
SAMPLE_HZ = 200.0


def _write_imu(path: Path, *, duration_s: float = 5.0, seed: int = 0) -> Path:
    """Synthesize an IMU trace at the locked-spec kinematics; write CSV."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, 1.0 / SAMPLE_HZ)
    theta = THETA_MAX * np.sin(OMEGA_SHM * t) + 0.001 * rng.standard_normal(t.size)
    omega = THETA_MAX * OMEGA_SHM * np.cos(OMEGA_SHM * t) + 0.005 * rng.standard_normal(t.size)
    lines = ["t_s,theta_rad,omega_rad_per_s"]
    for ti, th, om in zip(t, theta, omega, strict=True):
        lines.append(f"{ti:.5f},{th:.6f},{om:.6f}")
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_grid(path: Path, v: float = 0.3, v_peak: float = 0.7) -> Path:
    """Write a uniform-flow 9-point grid CSV."""
    lines = ["point,x_m,y_m,z_m,v_mean_m_per_s,v_peak_m_per_s,notes"]
    for i, (x, y) in enumerate(GRID_POINTS_M, start=1):
        lines.append(f"p{i},{x},{y},0.3,{v},{v_peak},")
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_inertia(path: Path, *, I_wrist: float = 5.7e-5, passed: bool = True) -> Path:
    path.write_text(json.dumps({"result": {"I_wrist_kgm2": I_wrist, "passed": passed}}))
    return path


@pytest.fixture
def tmp_spike03(tmp_path: Path) -> dict[str, Path]:
    """A complete, valid input set for the runner."""
    imu_files = [_write_imu(tmp_path / f"imu_trial{i}.csv", seed=i) for i in range(1, 6)]
    grid = _write_grid(tmp_path / "grid.csv")
    inertia = _write_inertia(tmp_path / "inertia.json")
    out = tmp_path / "baseline.json"
    return {
        "imu": imu_files,
        "grid": grid,
        "inertia": inertia,
        "out": out,
    }


def _invoke(args: dict[str, Path], **overrides) -> int:
    argv = [
        "--imu",
        *[str(p) for p in args["imu"]],
        "--anemometer",
        str(args["grid"]),
        "--inertia",
        str(args["inertia"]),
        "--out",
        str(args["out"]),
    ]
    for k, v in overrides.items():
        argv.extend([f"--{k.replace('_', '-')}", str(v)])
    return cli.main(argv)


# ---- happy path ------------------------------------------------------------


def test_runner_end_to_end(tmp_spike03: dict[str, Path]) -> None:
    rc = _invoke(tmp_spike03)
    assert rc == 0
    payload = json.loads(tmp_spike03["out"].read_text())
    b = payload["baseline"]
    # Mean velocity = 0.3 m/s uniform → J_fan_proxy = ρ · 0.3 · 0.36.
    assert b["J_fan_proxy_N"] == pytest.approx(RHO_AIR_KG_PER_M3 * 0.3 * PLANE_AREA_M2, rel=1e-9)
    assert b["J_fan_proxy_peak_N"] == pytest.approx(
        RHO_AIR_KG_PER_M3 * 0.7 * PLANE_AREA_M2, rel=1e-9
    )
    # W_cycle ≈ 2·I·ω_max² for ideal SHM.
    expected_W = 2.0 * 5.7e-5 * (THETA_MAX * OMEGA_SHM) ** 2
    assert b["W_cycle_J"] == pytest.approx(expected_W, rel=0.02)
    # J/W ratio derives from the above.
    assert b["J_per_W"] == pytest.approx(b["J_fan_proxy_N"] / b["W_cycle_J"], rel=1e-9)
    # IMU sanity flags should all pass on the synthetic spec-matched trace.
    assert payload["imu"]["kinematic_sanity_all_ok"] is True
    assert payload["imu"]["trial_consistency_ok"] is True


def test_baseline_json_schema(tmp_spike03: dict[str, Path]) -> None:
    """Phase 6 consumes specific keys from baseline.json; lock them down."""
    _invoke(tmp_spike03)
    payload = json.loads(tmp_spike03["out"].read_text())
    # Required top-level keys.
    assert {"spec_reference", "inputs", "anemometer", "imu", "baseline"} <= set(payload)
    # Required per-section keys.
    assert {"I_wrist_kgm2", "spike_0_2_passed", "rho_air_kg_per_m3", "A_plane_m2"} <= set(
        payload["inputs"]
    )
    assert {"J_fan_proxy_N", "v_mean_grid_m_per_s", "n_points"} <= set(payload["anemometer"])
    assert {"n_trials", "W_cycle_J_mean", "W_cycle_J_std", "kinematic_sanity_all_ok"} <= set(
        payload["imu"]
    )
    assert {"J_fan_proxy_N", "W_cycle_J", "J_per_W"} <= set(payload["baseline"])
    # Per-trial entries carry the IMU sanity flags.
    per = payload["imu"]["per_trial"]
    assert len(per) == 5
    for entry in per:
        assert {
            "W_cycle_J",
            "f_wave_Hz",
            "omega_max_rad_per_s",
            "theta_max_rad",
            "f_wave_ok",
            "omega_max_ok",
            "theta_max_ok",
            "sanity_ok",
        } <= set(entry)


def test_runner_warns_but_does_not_fail_on_kinematic_drift(
    tmp_path: Path, tmp_spike03: dict[str, Path]
) -> None:
    """Operator running at half-amplitude → sanity flags False but exit 0."""
    # Overwrite trial 1 with a weak-amplitude trace.
    weak = tmp_path / "imu_trial1.csv"
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 5.0, 1.0 / SAMPLE_HZ)
    theta = 0.30 * np.sin(OMEGA_SHM * t) + 0.001 * rng.standard_normal(t.size)
    omega = 0.30 * OMEGA_SHM * np.cos(OMEGA_SHM * t) + 0.005 * rng.standard_normal(t.size)
    lines = ["t_s,theta_rad,omega_rad_per_s"]
    for ti, th, om in zip(t, theta, omega, strict=True):
        lines.append(f"{ti:.5f},{th:.6f},{om:.6f}")
    weak.write_text("\n".join(lines) + "\n")
    tmp_spike03["imu"][0] = weak

    rc = _invoke(tmp_spike03)
    # Warning, not failure — Phase 0 wants the run to land artifacts even on
    # cadence drift, so the operator can re-shoot the bad trial without
    # re-doing everything.
    assert rc == 0
    payload = json.loads(tmp_spike03["out"].read_text())
    assert payload["imu"]["kinematic_sanity_all_ok"] is False


def test_runner_rho_air_override(tmp_spike03: dict[str, Path]) -> None:
    """--rho-air alters J_fan_proxy linearly."""
    _invoke(tmp_spike03, rho_air=1.0)
    payload = json.loads(tmp_spike03["out"].read_text())
    assert payload["inputs"]["rho_air_kg_per_m3"] == 1.0
    assert payload["baseline"]["J_fan_proxy_N"] == pytest.approx(
        1.0 * 0.3 * PLANE_AREA_M2, rel=1e-9
    )


# ---- inertia.json parsing --------------------------------------------------


def test_runner_errors_when_inertia_json_missing(
    tmp_path: Path, tmp_spike03: dict[str, Path]
) -> None:
    tmp_spike03["inertia"] = tmp_path / "nope.json"
    assert _invoke(tmp_spike03) == 2


def test_runner_errors_when_inertia_json_malformed(
    tmp_path: Path, tmp_spike03: dict[str, Path]
) -> None:
    """Malformed inertia JSON → graceful exit 2 (JSONDecodeError is a ValueError)."""
    bad = tmp_path / "bad_inertia.json"
    bad.write_text("not valid json")
    tmp_spike03["inertia"] = bad
    assert _invoke(tmp_spike03) == 2


def test_runner_errors_when_inertia_json_missing_key(
    tmp_path: Path, tmp_spike03: dict[str, Path]
) -> None:
    """Inertia JSON without result.I_wrist_kgm2 → exit 2."""
    bad = tmp_path / "no_key.json"
    bad.write_text(json.dumps({"result": {"passed": True}}))
    tmp_spike03["inertia"] = bad
    assert _invoke(tmp_spike03) == 2


def test_runner_propagates_spike_0_2_passed_flag(
    tmp_path: Path, tmp_spike03: dict[str, Path]
) -> None:
    """Surface Spike 0.2's pass flag in baseline.json for downstream auditors."""
    _write_inertia(tmp_spike03["inertia"], I_wrist=5.7e-5, passed=False)
    _invoke(tmp_spike03)
    payload = json.loads(tmp_spike03["out"].read_text())
    assert payload["inputs"]["spike_0_2_passed"] is False


# ---- IMU / anemometer error paths ------------------------------------------


def test_runner_errors_on_bad_imu_csv(tmp_path: Path, tmp_spike03: dict[str, Path]) -> None:
    bad = tmp_path / "bad_imu.csv"
    bad.write_text("oops,no,header\n1,2,3\n")
    tmp_spike03["imu"][0] = bad
    assert _invoke(tmp_spike03) == 2


def test_runner_errors_on_missing_imu_file(tmp_path: Path, tmp_spike03: dict[str, Path]) -> None:
    tmp_spike03["imu"][0] = tmp_path / "nope.csv"
    assert _invoke(tmp_spike03) == 2


def test_runner_errors_on_grid_with_wrong_positions(
    tmp_path: Path, tmp_spike03: dict[str, Path]
) -> None:
    """Operator's reticle slipped → load_anemometer_csv refuses the file."""
    bad = tmp_path / "shifted_grid.csv"
    lines = ["point,x_m,y_m,z_m,v_mean_m_per_s,v_peak_m_per_s,notes"]
    for i, (x, y) in enumerate(GRID_POINTS_M, start=1):
        # Shift p5 by 50 mm — well past the 5 mm tolerance.
        if i == 5:
            x = 0.05
        lines.append(f"p{i},{x},{y},0.3,0.3,0.7,")
    bad.write_text("\n".join(lines) + "\n")
    tmp_spike03["grid"] = bad
    assert _invoke(tmp_spike03) == 2


def test_runner_errors_on_short_grid(tmp_path: Path, tmp_spike03: dict[str, Path]) -> None:
    """Only 8 grid points → exit 2."""
    bad = tmp_path / "short_grid.csv"
    lines = ["point,x_m,y_m,z_m,v_mean_m_per_s,v_peak_m_per_s,notes"]
    for i, (x, y) in enumerate(GRID_POINTS_M[:-1], start=1):
        lines.append(f"p{i},{x},{y},0.3,0.3,0.7,")
    bad.write_text("\n".join(lines) + "\n")
    tmp_spike03["grid"] = bad
    assert _invoke(tmp_spike03) == 2


# ---- single-trial trial_consistency edge ----------------------------------


def test_runner_with_single_imu_trial(tmp_path: Path, tmp_spike03: dict[str, Path]) -> None:
    """With 1 IMU trial, trial-to-trial std is 0 → trial_consistency_ok still True
    (the gate is `std/mean < 20%` and 0/anything < 20%)."""
    tmp_spike03["imu"] = [tmp_spike03["imu"][0]]
    rc = _invoke(tmp_spike03)
    assert rc == 0
    payload = json.loads(tmp_spike03["out"].read_text())
    assert payload["imu"]["n_trials"] == 1
    assert payload["imu"]["W_cycle_J_std"] == 0.0
    assert payload["imu"]["trial_consistency_ok"] is True
