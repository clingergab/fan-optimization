"""CLI tests for scripts/spike_0_2_analyze.py.

Exercises:
- happy-path PASS (exit 0, results.json schema, JSON values).
- repeatability FAIL (exit 1, passed=False).
- cross-check FAIL (exit 1, passed=False).
- bad input → exit 2 (missing file, missing column, non-numeric).
- CSV decommenter handles '#' / blank lines.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.2; protocol in
docs/spike_0_2_protocol.md.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import spike_0_2_analyze as cli

# ---- fixtures --------------------------------------------------------------


@pytest.fixture
def kappa_value() -> float:
    """A realistic torsion constant for the SHM scenarios below."""
    return 1.234e-3


@pytest.fixture
def calibration_csv(tmp_path: Path, kappa_value: float) -> Path:
    path = tmp_path / "calibration.csv"
    path.write_text(
        "# calibration log line\n"
        "kappa_Nm_per_rad,T_ref_s,m_ref_kg,L_ref_m,I_ref_kgm2,method,notes\n"
        f"{kappa_value},1.540,0.050,0.120,6.0e-5,torsion_wire,smoke\n"
    )
    return path


def _measurements_csv(path: Path, periods: list[float]) -> Path:
    lines = ["trial,T_osc_s,amplitude_deg,notes"]
    for i, T in enumerate(periods, start=1):
        lines.append(f"{i},{T},8,")
    path.write_text("\n".join(lines) + "\n")
    return path


def _i_wrist(kappa: float, T: float) -> float:
    return kappa * (T / (2.0 * math.pi)) ** 2


# ---- happy path ------------------------------------------------------------


def test_cli_pass(tmp_path: Path, calibration_csv: Path, kappa_value: float) -> None:
    T = 1.50
    meas = _measurements_csv(tmp_path / "measurements.csv", [T] * 5)
    out = tmp_path / "results.json"
    I_gen = _i_wrist(kappa_value, T) * 1.05  # within ±10%

    rc = cli.main(
        [
            "--calibration",
            str(calibration_csv),
            "--measurements",
            str(meas),
            "--generator-i-wrist",
            str(I_gen),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    payload = json.loads(out.read_text())
    r = payload["result"]
    assert r["passed"] is True
    assert r["repeatability_passed"] is True
    assert r["cross_check_passed"] is True
    assert r["n_trials"] == 5
    assert math.isclose(r["I_wrist_kgm2"], _i_wrist(kappa_value, T), rel_tol=1e-9)
    # Schema sanity — the consumer (Phase 6 runner) relies on these keys.
    assert payload["spec_reference"].startswith("docs/plan_R11.md")
    assert payload["gates"]["repeatability_gate_pct"] == 3.0
    assert payload["gates"]["cross_check_gate_pct"] == 10.0
    assert payload["inputs"]["n_trials"] == 5


def test_cli_no_generator_value_still_passes_on_repeatability(
    tmp_path: Path, calibration_csv: Path
) -> None:
    """No --generator-i-wrist → cross-check skipped, exit 0 if repeatability OK."""
    meas = _measurements_csv(tmp_path / "measurements.csv", [1.50] * 5)
    out = tmp_path / "results.json"
    rc = cli.main(
        [
            "--calibration",
            str(calibration_csv),
            "--measurements",
            str(meas),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    r = json.loads(out.read_text())["result"]
    assert r["cross_check_pct"] is None
    assert r["cross_check_passed"] is None
    assert r["passed"] is True


# ---- failing-gate exit codes -----------------------------------------------


def test_cli_fails_repeatability(tmp_path: Path, calibration_csv: Path) -> None:
    """Periods spread enough that std/mean > 3% → exit 1."""
    meas = _measurements_csv(tmp_path / "measurements.csv", [1.20, 1.30, 1.40, 1.25, 1.35])
    out = tmp_path / "results.json"
    rc = cli.main(
        ["--calibration", str(calibration_csv), "--measurements", str(meas), "--out", str(out)]
    )
    assert rc == 1
    r = json.loads(out.read_text())["result"]
    assert r["repeatability_passed"] is False
    assert r["passed"] is False


def test_cli_fails_cross_check(tmp_path: Path, calibration_csv: Path, kappa_value: float) -> None:
    """Tight trials but generator off by 20% → cross-check fails → exit 1."""
    T = 1.50
    meas = _measurements_csv(tmp_path / "measurements.csv", [T] * 5)
    out = tmp_path / "results.json"
    I_gen = _i_wrist(kappa_value, T) * 1.20  # 20% off
    rc = cli.main(
        [
            "--calibration",
            str(calibration_csv),
            "--measurements",
            str(meas),
            "--generator-i-wrist",
            str(I_gen),
            "--out",
            str(out),
        ]
    )
    assert rc == 1
    r = json.loads(out.read_text())["result"]
    assert r["repeatability_passed"] is True
    assert r["cross_check_passed"] is False
    assert r["passed"] is False


# ---- bad-input exit codes (exit 2) -----------------------------------------


def test_cli_missing_calibration_file(tmp_path: Path) -> None:
    meas = _measurements_csv(tmp_path / "measurements.csv", [1.50] * 5)
    rc = cli.main(
        [
            "--calibration",
            str(tmp_path / "does_not_exist.csv"),
            "--measurements",
            str(meas),
            "--out",
            str(tmp_path / "results.json"),
        ]
    )
    assert rc == 2


def test_cli_missing_measurements_file(tmp_path: Path, calibration_csv: Path) -> None:
    rc = cli.main(
        [
            "--calibration",
            str(calibration_csv),
            "--measurements",
            str(tmp_path / "does_not_exist.csv"),
            "--out",
            str(tmp_path / "results.json"),
        ]
    )
    assert rc == 2


def test_cli_calibration_missing_kappa_column(tmp_path: Path) -> None:
    bad_calib = tmp_path / "bad_calib.csv"
    bad_calib.write_text("T_ref_s,m_ref_kg\n1.540,0.050\n")
    meas = _measurements_csv(tmp_path / "measurements.csv", [1.50] * 5)
    rc = cli.main(
        [
            "--calibration",
            str(bad_calib),
            "--measurements",
            str(meas),
            "--out",
            str(tmp_path / "results.json"),
        ]
    )
    assert rc == 2


def test_cli_calibration_nonnumeric_kappa(tmp_path: Path) -> None:
    bad_calib = tmp_path / "bad_calib.csv"
    bad_calib.write_text("kappa_Nm_per_rad,T_ref_s\n" "not_a_number,1.540\n")
    meas = _measurements_csv(tmp_path / "measurements.csv", [1.50] * 5)
    rc = cli.main(
        [
            "--calibration",
            str(bad_calib),
            "--measurements",
            str(meas),
            "--out",
            str(tmp_path / "results.json"),
        ]
    )
    assert rc == 2


def test_cli_measurements_missing_t_osc_column(tmp_path: Path, calibration_csv: Path) -> None:
    bad_meas = tmp_path / "bad_meas.csv"
    bad_meas.write_text("trial,amplitude_deg,notes\n1,8,\n2,8,\n")
    rc = cli.main(
        [
            "--calibration",
            str(calibration_csv),
            "--measurements",
            str(bad_meas),
            "--out",
            str(tmp_path / "results.json"),
        ]
    )
    assert rc == 2


def test_cli_measurements_nonnumeric_t_osc(tmp_path: Path, calibration_csv: Path) -> None:
    bad_meas = tmp_path / "bad_meas.csv"
    bad_meas.write_text("trial,T_osc_s,amplitude_deg,notes\n1,oops,8,\n2,1.50,8,\n")
    rc = cli.main(
        [
            "--calibration",
            str(calibration_csv),
            "--measurements",
            str(bad_meas),
            "--out",
            str(tmp_path / "results.json"),
        ]
    )
    assert rc == 2


def test_cli_only_one_trial_errors(tmp_path: Path, calibration_csv: Path) -> None:
    """analyze_trials requires ≥ 2 trials — single-trial CSV → exit 2."""
    meas = _measurements_csv(tmp_path / "measurements.csv", [1.50])
    rc = cli.main(
        [
            "--calibration",
            str(calibration_csv),
            "--measurements",
            str(meas),
            "--out",
            str(tmp_path / "results.json"),
        ]
    )
    assert rc == 2


def test_cli_empty_measurements_errors(tmp_path: Path, calibration_csv: Path) -> None:
    """Measurements file with header but no rows → exit 2."""
    bad_meas = tmp_path / "empty_meas.csv"
    bad_meas.write_text(
        "# operator notes; no rows recorded\n" "trial,T_osc_s,amplitude_deg,notes\n"
    )
    rc = cli.main(
        [
            "--calibration",
            str(calibration_csv),
            "--measurements",
            str(bad_meas),
            "--out",
            str(tmp_path / "results.json"),
        ]
    )
    assert rc == 2


def test_cli_empty_calibration_errors(tmp_path: Path) -> None:
    """No data rows (only header / comments) → exit 2."""
    bad_calib = tmp_path / "empty_calib.csv"
    bad_calib.write_text("# comments only\n" "kappa_Nm_per_rad,T_ref_s\n")
    meas = _measurements_csv(tmp_path / "measurements.csv", [1.50] * 5)
    rc = cli.main(
        [
            "--calibration",
            str(bad_calib),
            "--measurements",
            str(meas),
            "--out",
            str(tmp_path / "results.json"),
        ]
    )
    assert rc == 2


# ---- mount subtraction propagates -----------------------------------------


def test_cli_subtracts_mount_inertia(
    tmp_path: Path, calibration_csv: Path, kappa_value: float
) -> None:
    T = 1.50
    mount = 1.0e-6
    meas = _measurements_csv(tmp_path / "measurements.csv", [T] * 5)
    out = tmp_path / "results.json"
    rc = cli.main(
        [
            "--calibration",
            str(calibration_csv),
            "--measurements",
            str(meas),
            "--mount-i-wrist",
            str(mount),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    r = json.loads(out.read_text())["result"]
    assert r["I_wrist_kgm2"] == pytest.approx(_i_wrist(kappa_value, T) - mount, rel=1e-9)
