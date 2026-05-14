"""CLI tests for scripts/run_spike_0_4.py.

Exercises:
- happy-path PASS (all four CSVs at canonical values; overall_passed=true).
- FAIL on force balance below threshold (v1_lock_fallback_armed=true).
- input error → exit 2 (missing column).

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.4; protocol in
docs/spike_0_4_protocol.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_spike_0_4 as cli


# ─────────────────────────────────────────────────────────────────────
# Canonical-input fixtures
# ─────────────────────────────────────────────────────────────────────


def _write_force_balance(path: Path, I_wrist: float = 1.0e-3, F_fric: float = 1.50) -> Path:
    """Default: I=1e-3 → F_inertial=0.44 N; F_fric=1.5 N >> 0.88 N → passes."""
    path.write_text(
        "# operator notes\n"
        "I_wrist_kgm2,F_friction_cumulative_N,notes\n"
        f"{I_wrist},{F_fric},smoke\n"
    )
    return path


def _write_clearance(path: Path, values: list[float] | None = None) -> Path:
    if values is None:
        values = [0.16, 0.17, 0.18, 0.19]
    lines = ["mating_surface,clearance_mm,notes"]
    for i, v in enumerate(values, start=1):
        lines.append(f"blade{i}_blade{i + 1},{v},")
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_engagement_force(
    path: Path,
    low: list[float] | None = None,
    high: list[float] | None = None,
) -> Path:
    if low is None:
        low = [0.8, 1.0, 1.2, 1.5, 1.8]
    if high is None:
        high = [1.5, 2.0, 2.5, 3.0]
    lines = ["trial,force_N,regime,notes"]
    trial = 1
    for f in low:
        lines.append(f"{trial},{f},low,")
        trial += 1
    for f in high:
        lines.append(f"{trial},{f},high,")
        trial += 1
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_cycle_inspections(
    path: Path,
    n_cycles: int = 1000,
    fracture_at: int | None = None,
) -> Path:
    lines = ["cycle,wear_observed,fracture,notes"]
    for c in range(100, n_cycles + 1, 100):
        frac = "true" if fracture_at is not None and c == fracture_at else "false"
        lines.append(f"{c},false,{frac},")
    path.write_text("\n".join(lines) + "\n")
    return path


# ─────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────


def test_cli_pass(tmp_path: Path) -> None:
    fb = _write_force_balance(tmp_path / "force_balance.csv")
    cl = _write_clearance(tmp_path / "clearance.csv")
    ef = _write_engagement_force(tmp_path / "engagement_force.csv")
    ci = _write_cycle_inspections(tmp_path / "cycle_inspections.csv")
    out = tmp_path / "results.json"

    rc = cli.main(
        [
            "--force-balance", str(fb),
            "--clearance", str(cl),
            "--engagement-force", str(ef),
            "--cycle-inspections", str(ci),
            "--alignment-gap-variation-mm", "0.30",
            "--high-amp-completed",
            "--out", str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    payload = json.loads(out.read_text())
    r = payload["result"]
    assert r["overall_passed"] is True
    assert r["v1_lock_fallback_armed"] is False
    assert r["force_balance"]["passed"] is True
    assert r["clearance"]["passed"] is True
    assert r["engagement_force"]["passed"] is True
    assert r["cycle_life"]["passed"] is True
    assert r["high_amp_engagement_force"]["passed"] is True

    # Schema sanity — Phase 6 consumer relies on these keys.
    assert payload["spec_reference"].startswith("docs/plan_R11.md")
    assert payload["gates"]["alpha_max_rad_per_s2"] == 110.0
    assert payload["gates"]["L_wrist_to_tip_m"] == 0.25
    assert payload["gates"]["force_balance_safety_factor"] == 2.0
    assert payload["gates"]["clearance_band_mm"] == [0.15, 0.20]
    assert payload["gates"]["cycle_target"] == 1000


def test_cli_pass_without_high_amp_rows(tmp_path: Path) -> None:
    """No 'high' regime rows: only low-regime engagement is scored."""
    fb = _write_force_balance(tmp_path / "force_balance.csv")
    cl = _write_clearance(tmp_path / "clearance.csv")
    ef = _write_engagement_force(tmp_path / "engagement_force.csv", high=[])
    ci = _write_cycle_inspections(tmp_path / "cycle_inspections.csv")
    out = tmp_path / "results.json"

    rc = cli.main(
        [
            "--force-balance", str(fb),
            "--clearance", str(cl),
            "--engagement-force", str(ef),
            "--cycle-inspections", str(ci),
            "--alignment-gap-variation-mm", "0.30",
            "--high-amp-completed",
            "--out", str(out),
        ]
    )
    assert rc == 0
    r = json.loads(out.read_text())["result"]
    assert r["high_amp_engagement_force"] is None
    assert r["overall_passed"] is True


# ─────────────────────────────────────────────────────────────────────
# Failure paths
# ─────────────────────────────────────────────────────────────────────


def test_cli_fails_force_balance_arms_v1_fallback(tmp_path: Path) -> None:
    """F_fric = 0.50 N < required 0.88 N → exit 1, fallback armed."""
    fb = _write_force_balance(tmp_path / "force_balance.csv", F_fric=0.50)
    cl = _write_clearance(tmp_path / "clearance.csv")
    ef = _write_engagement_force(tmp_path / "engagement_force.csv")
    ci = _write_cycle_inspections(tmp_path / "cycle_inspections.csv")
    out = tmp_path / "results.json"

    rc = cli.main(
        [
            "--force-balance", str(fb),
            "--clearance", str(cl),
            "--engagement-force", str(ef),
            "--cycle-inspections", str(ci),
            "--alignment-gap-variation-mm", "0.30",
            "--high-amp-completed",
            "--out", str(out),
        ]
    )
    assert rc == 1
    r = json.loads(out.read_text())["result"]
    assert r["overall_passed"] is False
    assert r["force_balance"]["passed"] is False
    assert r["v1_lock_fallback_armed"] is True
    assert r["force_balance"]["v1_lock_fallback_armed"] is True


def test_cli_fails_clearance_out_of_band(tmp_path: Path) -> None:
    fb = _write_force_balance(tmp_path / "force_balance.csv")
    cl = _write_clearance(
        tmp_path / "clearance.csv", values=[0.14, 0.17, 0.18, 0.21]
    )
    ef = _write_engagement_force(tmp_path / "engagement_force.csv")
    ci = _write_cycle_inspections(tmp_path / "cycle_inspections.csv")
    out = tmp_path / "results.json"

    rc = cli.main(
        [
            "--force-balance", str(fb),
            "--clearance", str(cl),
            "--engagement-force", str(ef),
            "--cycle-inspections", str(ci),
            "--alignment-gap-variation-mm", "0.30",
            "--high-amp-completed",
            "--out", str(out),
        ]
    )
    assert rc == 1
    r = json.loads(out.read_text())["result"]
    assert r["clearance"]["passed"] is False
    assert r["clearance"]["out_of_band_count"] == 2


def test_cli_fails_cycle_life_when_fractured(tmp_path: Path) -> None:
    fb = _write_force_balance(tmp_path / "force_balance.csv")
    cl = _write_clearance(tmp_path / "clearance.csv")
    ef = _write_engagement_force(tmp_path / "engagement_force.csv")
    ci = _write_cycle_inspections(
        tmp_path / "cycle_inspections.csv", fracture_at=600
    )
    out = tmp_path / "results.json"

    rc = cli.main(
        [
            "--force-balance", str(fb),
            "--clearance", str(cl),
            "--engagement-force", str(ef),
            "--cycle-inspections", str(ci),
            "--alignment-gap-variation-mm", "0.30",
            "--high-amp-completed",
            "--out", str(out),
        ]
    )
    assert rc == 1
    r = json.loads(out.read_text())["result"]
    assert r["cycle_life"]["passed"] is False
    assert r["cycle_life"]["first_fracture_cycle"] == 600


# ─────────────────────────────────────────────────────────────────────
# Input error → exit 2
# ─────────────────────────────────────────────────────────────────────


def test_cli_missing_column_exits_2(tmp_path: Path) -> None:
    """Missing 'F_friction_cumulative_N' column in force-balance CSV → exit 2."""
    fb = tmp_path / "force_balance.csv"
    fb.write_text("I_wrist_kgm2,notes\n1.0e-3,bad header\n")
    cl = _write_clearance(tmp_path / "clearance.csv")
    ef = _write_engagement_force(tmp_path / "engagement_force.csv")
    ci = _write_cycle_inspections(tmp_path / "cycle_inspections.csv")
    out = tmp_path / "results.json"

    rc = cli.main(
        [
            "--force-balance", str(fb),
            "--clearance", str(cl),
            "--engagement-force", str(ef),
            "--cycle-inspections", str(ci),
            "--alignment-gap-variation-mm", "0.30",
            "--high-amp-completed",
            "--out", str(out),
        ]
    )
    assert rc == 2


def test_cli_missing_file_exits_2(tmp_path: Path) -> None:
    cl = _write_clearance(tmp_path / "clearance.csv")
    ef = _write_engagement_force(tmp_path / "engagement_force.csv")
    ci = _write_cycle_inspections(tmp_path / "cycle_inspections.csv")
    rc = cli.main(
        [
            "--force-balance", str(tmp_path / "does_not_exist.csv"),
            "--clearance", str(cl),
            "--engagement-force", str(ef),
            "--cycle-inspections", str(ci),
            "--alignment-gap-variation-mm", "0.30",
            "--high-amp-completed",
            "--out", str(tmp_path / "results.json"),
        ]
    )
    assert rc == 2


def test_cli_empty_clearance_exits_2(tmp_path: Path) -> None:
    """Header-only clearance CSV → exit 2."""
    fb = _write_force_balance(tmp_path / "force_balance.csv")
    cl = tmp_path / "clearance.csv"
    cl.write_text("mating_surface,clearance_mm,notes\n")
    ef = _write_engagement_force(tmp_path / "engagement_force.csv")
    ci = _write_cycle_inspections(tmp_path / "cycle_inspections.csv")

    rc = cli.main(
        [
            "--force-balance", str(fb),
            "--clearance", str(cl),
            "--engagement-force", str(ef),
            "--cycle-inspections", str(ci),
            "--alignment-gap-variation-mm", "0.30",
            "--high-amp-completed",
            "--out", str(tmp_path / "results.json"),
        ]
    )
    assert rc == 2


def test_cli_bad_regime_exits_2(tmp_path: Path) -> None:
    """A regime value other than low/high → exit 2."""
    fb = _write_force_balance(tmp_path / "force_balance.csv")
    cl = _write_clearance(tmp_path / "clearance.csv")
    ef = tmp_path / "engagement_force.csv"
    ef.write_text("trial,force_N,regime,notes\n1,1.0,medium,\n")
    ci = _write_cycle_inspections(tmp_path / "cycle_inspections.csv")
    rc = cli.main(
        [
            "--force-balance", str(fb),
            "--clearance", str(cl),
            "--engagement-force", str(ef),
            "--cycle-inspections", str(ci),
            "--alignment-gap-variation-mm", "0.30",
            "--high-amp-completed",
            "--out", str(tmp_path / "results.json"),
        ]
    )
    assert rc == 2


def test_cli_nonfloat_clearance_exits_2(tmp_path: Path) -> None:
    fb = _write_force_balance(tmp_path / "force_balance.csv")
    cl = tmp_path / "clearance.csv"
    cl.write_text("mating_surface,clearance_mm,notes\nblade1_blade2,oops,\n")
    ef = _write_engagement_force(tmp_path / "engagement_force.csv")
    ci = _write_cycle_inspections(tmp_path / "cycle_inspections.csv")
    rc = cli.main(
        [
            "--force-balance", str(fb),
            "--clearance", str(cl),
            "--engagement-force", str(ef),
            "--cycle-inspections", str(ci),
            "--alignment-gap-variation-mm", "0.30",
            "--high-amp-completed",
            "--out", str(tmp_path / "results.json"),
        ]
    )
    assert rc == 2


# ---- --i-wrist-analytic path (V1 with Spike-0.2 deferred) -----------------


def _common_args_without_force_balance(tmp_path: Path) -> list[str]:
    """Build a non-force-balance argv (clearance/engagement/inspections only)."""
    cl = _write_clearance(tmp_path / "clearance.csv")
    ef = _write_engagement_force(tmp_path / "engagement_force.csv")
    ci = _write_cycle_inspections(tmp_path / "cycle_inspections.csv")
    return [
        "--clearance", str(cl),
        "--engagement-force", str(ef),
        "--cycle-inspections", str(ci),
        "--alignment-gap-variation-mm", "0.30",
        "--high-amp-completed",
        "--out", str(tmp_path / "results.json"),
    ]


def test_cli_analytic_iwrist_pass_uses_3x_safety_factor(tmp_path: Path) -> None:
    """With --i-wrist-analytic, the 3x factor is applied. F_fric=1.50 vs
    F_inertial = 1e-3 · 110 / 0.25 = 0.44 → 1.50 / 0.44 ≈ 3.4 > 3.0 → pass."""
    out = tmp_path / "results.json"
    rc = cli.main(
        _common_args_without_force_balance(tmp_path) + [
            "--i-wrist-analytic", "1.0e-3",
            "--f-friction-cumulative-n", "1.50",
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["gates"]["force_balance_safety_factor"] == 3.0
    assert payload["gates"]["i_wrist_source"] == "analytic"
    assert payload["result"]["force_balance"]["passed"] is True


def test_cli_analytic_iwrist_fails_under_3x_when_2x_would_have_passed(tmp_path: Path) -> None:
    """F_fric = 1.00 vs F_inertial = 0.44: 1.00 / 0.44 ≈ 2.27. Passes 2x
    (canonical) but fails 3x (analytic). Demonstrates the bumped factor."""
    out = tmp_path / "results.json"
    rc = cli.main(
        _common_args_without_force_balance(tmp_path) + [
            "--i-wrist-analytic", "1.0e-3",
            "--f-friction-cumulative-n", "1.00",
        ]
    )
    # Fails the analytic 3x gate even though it would have passed canonical 2x.
    payload = json.loads(out.read_text())
    assert payload["result"]["force_balance"]["passed"] is False
    assert payload["result"]["v1_lock_fallback_armed"] is True
    # Exit 1 because the force-balance gate failed (overall_passed=False).
    assert rc == 1


def test_cli_analytic_iwrist_requires_paired_friction(tmp_path: Path) -> None:
    """--i-wrist-analytic without --f-friction-cumulative-n must exit 2."""
    rc = cli.main(
        _common_args_without_force_balance(tmp_path) + [
            "--i-wrist-analytic", "1.0e-3",
        ]
    )
    assert rc == 2


def test_cli_friction_arg_requires_paired_analytic_iwrist(tmp_path: Path) -> None:
    """--f-friction-cumulative-n without --i-wrist-analytic must exit 2."""
    rc = cli.main(
        _common_args_without_force_balance(tmp_path) + [
            "--f-friction-cumulative-n", "1.50",
        ]
    )
    assert rc == 2


def test_cli_analytic_iwrist_rejects_nonpositive(tmp_path: Path) -> None:
    """Negative or zero I_wrist must exit 2."""
    rc = cli.main(
        _common_args_without_force_balance(tmp_path) + [
            "--i-wrist-analytic", "-0.001",
            "--f-friction-cumulative-n", "1.50",
        ]
    )
    assert rc == 2


def test_cli_analytic_iwrist_skips_force_balance_csv(tmp_path: Path) -> None:
    """When --i-wrist-analytic is used, the --force-balance CSV is not read.

    Confirms by pointing it at a non-existent path: CSV is never opened.
    """
    out = tmp_path / "results.json"
    rc = cli.main(
        _common_args_without_force_balance(tmp_path) + [
            "--i-wrist-analytic", "1.0e-3",
            "--f-friction-cumulative-n", "1.50",
            "--force-balance", str(tmp_path / "does_not_exist.csv"),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    # The fb_row is now a synthetic dict, not a CSV-parsed row.
    assert payload["inputs"]["force_balance_row"]["source"] == "analytic"


def test_cli_measured_path_unchanged_by_new_flag_default(tmp_path: Path) -> None:
    """Default (no --i-wrist-analytic) still reads the CSV and uses 2x factor."""
    fb = _write_force_balance(tmp_path / "force_balance.csv")
    out = tmp_path / "results.json"
    rc = cli.main(
        _common_args_without_force_balance(tmp_path) + [
            "--force-balance", str(fb),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["gates"]["force_balance_safety_factor"] == 2.0
    assert payload["gates"]["i_wrist_source"] == "measured"
