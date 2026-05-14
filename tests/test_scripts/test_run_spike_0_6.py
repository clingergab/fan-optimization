"""CLI smoke tests for scripts/run_spike_0_6.py.

Exercises:
- happy-path PASS with both sub-spikes (exit 0).
- sub-spike 0.6a FAIL (exit 1).
- sub-spike 0.6b FAIL (exit 1).
- missing column in 06b CSV (exit 2).
- aggregator records calibration overall_passed=True regardless of contents.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.6; protocol in
docs/spike_0_6_protocol.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_spike_0_6 as cli

# ---- fixtures --------------------------------------------------------------


@pytest.fixture
def budget_csv(tmp_path: Path) -> Path:
    path = tmp_path / "budget.csv"
    path.write_text(
        "# spike 0.6 compute-budget log\n"
        "platform,workload,wall_time_s,cu_consumed,cells,notes\n"
        "colab_pro_cpu,3d_unsteady_500k_5cycles_dtT200,3600,4.5,500000,smoke\n"
        "colab_pro_g4_gpu,3d_unsteady_500k_5cycles_dtT200,900,3.2,500000,smoke\n"
    )
    return path


def _write_06a(path: Path, wall_time_s: float, J_fan: str = "0.123") -> Path:
    path.write_text(
        "wall_time_s,J_fan_steady_proxy,stage_name,stage_wall_time_s\n"
        f"{wall_time_s},{J_fan},,\n"
        ",,cadquery,5.0\n"
        ",,gmsh_2d,10.0\n"
        ",,su2_2d_steady,580.0\n"
        ",,j_fan,5.0\n"
    )
    return path


def _write_06b(
    path: Path,
    wall_time_s: float,
    measured: float,
    P_N: float = 3.0,
    L_m: float = 1.0,
    E_Pa: float = 1.0,
    I_m4: float = 1000.0,
) -> Path:
    path.write_text(
        "wall_time_s,measured_tip_deflection_m,P_N,L_m,E_Pa,I_m4\n"
        f"{wall_time_s},{measured},{P_N},{L_m},{E_Pa},{I_m4}\n"
    )
    return path


# ---- happy path ------------------------------------------------------------


def test_cli_06a_pass_06b_pass(tmp_path: Path, budget_csv: Path) -> None:
    """Both sub-spikes pass -> exit 0."""
    a = _write_06a(tmp_path / "06a.csv", wall_time_s=600.0)
    b = _write_06b(tmp_path / "06b.csv", wall_time_s=30.0, measured=1.03e-3)
    out = tmp_path / "results.json"

    rc = cli.main(
        [
            "--budget-csv",
            str(budget_csv),
            "--06a-csv",
            str(a),
            "--06b-csv",
            str(b),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["spec_reference"].startswith("docs/plan_R11.md")
    r = payload["result"]
    assert r["overall_passed"] is True
    assert r["sub_06a"]["passed"] is True
    assert r["sub_06b"]["passed"] is True
    assert len(r["budget_entries"]) == 2
    assert r["budget_entries"][0]["platform"] == "colab_pro_cpu"
    assert r["budget_entries"][1]["cu_consumed"] == pytest.approx(3.2)
    # Stages survive the CSV round-trip.
    stages = r["sub_06a"]["pipeline_stages"]
    assert stages["su2_2d_steady"] == pytest.approx(580.0)
    # Gate constants surfaced in payload.
    assert payload["gates"]["m3_su2_wall_time_gate_s"] == 15 * 60
    assert payload["gates"]["m3_fea_wall_time_gate_s"] == 2 * 60
    assert payload["gates"]["m3_fea_tip_deflection_tolerance_pct"] == 5.0


def test_cli_no_sub_spikes_still_zero(tmp_path: Path, budget_csv: Path) -> None:
    """Budget rows only — no sub-spikes — calibration exit 0."""
    out = tmp_path / "results.json"
    rc = cli.main(["--budget-csv", str(budget_csv), "--out", str(out)])
    assert rc == 0
    r = json.loads(out.read_text())["result"]
    assert r["sub_06a"] is None
    assert r["sub_06b"] is None
    assert r["overall_passed"] is True


# ---- failing-gate exit codes -----------------------------------------------


def test_cli_06a_fail_returns_1(tmp_path: Path, budget_csv: Path) -> None:
    """Wall-time over 15 min on 0.6a -> exit 1, but overall_passed remains True."""
    a = _write_06a(tmp_path / "06a.csv", wall_time_s=16 * 60)
    b = _write_06b(tmp_path / "06b.csv", wall_time_s=30.0, measured=1.03e-3)
    out = tmp_path / "results.json"
    rc = cli.main(
        [
            "--budget-csv",
            str(budget_csv),
            "--06a-csv",
            str(a),
            "--06b-csv",
            str(b),
            "--out",
            str(out),
        ]
    )
    assert rc == 1
    r = json.loads(out.read_text())["result"]
    assert r["sub_06a"]["passed"] is False
    assert r["sub_06a"]["wall_time_passed"] is False
    assert r["sub_06b"]["passed"] is True
    # Calibration framing: aggregate is still PASS even when 06a fails.
    assert r["overall_passed"] is True


def test_cli_06b_fail_returns_1(tmp_path: Path, budget_csv: Path) -> None:
    """Tip-deflection 10% off on 0.6b -> exit 1."""
    a = _write_06a(tmp_path / "06a.csv", wall_time_s=600.0)
    b = _write_06b(tmp_path / "06b.csv", wall_time_s=30.0, measured=1.10e-3)
    out = tmp_path / "results.json"
    rc = cli.main(
        [
            "--budget-csv",
            str(budget_csv),
            "--06a-csv",
            str(a),
            "--06b-csv",
            str(b),
            "--out",
            str(out),
        ]
    )
    assert rc == 1
    r = json.loads(out.read_text())["result"]
    assert r["sub_06b"]["tip_deflection_passed"] is False
    assert r["sub_06b"]["passed"] is False
    assert r["overall_passed"] is True


# ---- bad-input exit codes (exit 2) -----------------------------------------


def test_cli_missing_06b_column_returns_2(tmp_path: Path, budget_csv: Path) -> None:
    """0.6b CSV missing required `P_N` column -> exit 2."""
    bad_b = tmp_path / "bad_06b.csv"
    bad_b.write_text(
        "wall_time_s,measured_tip_deflection_m,L_m,E_Pa,I_m4\n" "30.0,1.03e-3,1.0,1.0,1000.0\n"
    )
    rc = cli.main(
        [
            "--budget-csv",
            str(budget_csv),
            "--06b-csv",
            str(bad_b),
            "--out",
            str(tmp_path / "results.json"),
        ]
    )
    assert rc == 2


def test_cli_missing_budget_file_returns_2(tmp_path: Path) -> None:
    """Pointed at a non-existent budget CSV -> exit 2."""
    rc = cli.main(
        [
            "--budget-csv",
            str(tmp_path / "does_not_exist.csv"),
            "--out",
            str(tmp_path / "results.json"),
        ]
    )
    assert rc == 2


def test_cli_budget_missing_platform_returns_2(tmp_path: Path) -> None:
    bad = tmp_path / "bad_budget.csv"
    bad.write_text(
        "workload,wall_time_s,cu_consumed,cells,notes\n" "3d_unsteady,3600,4.5,500000,smoke\n"
    )
    rc = cli.main(
        [
            "--budget-csv",
            str(bad),
            "--out",
            str(tmp_path / "results.json"),
        ]
    )
    assert rc == 2
