"""CLI smoke tests for scripts/run_spike_0_7c.py and run_spike_0_7c_smoke.py.

Spec reference: ``docs/plan_R11.md §Phase 0 Spike 0.7c``; protocol in
``docs/spike_0_7c_protocol.md``.

Covers:

* The smoke runner (``run_spike_0_7c_smoke``) end-to-end on synthetic
  data -- proves the harness produces a PASS without any CFD spend.
* An explicit-FAIL test that hand-rolls a ledger where BO is worse than
  Sobol on all three budgets and asserts the CLI exits 1.
* Input-error paths -- missing files / empty ledgers / malformed JSON
  must exit 2.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_spike_0_7c as cli
import run_spike_0_7c_smoke as smoke_cli

# ---- smoke runner end-to-end PASS ---------------------------------------


def test_smoke_runner_passes(tmp_path: Path) -> None:
    """Smoke runner with default args (improvement 25% > 5% gate) -> PASS."""
    rc = smoke_cli.main(
        [
            "--n-sobol",
            "20",
            "--n-bo",
            "30",
            "--d",
            "8",
            "--seed",
            "42",
            "--budgets",
            "30,100,300",
            "--target-bo-improvement-pct",
            "25.0",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0

    results = tmp_path / "results.json"
    assert results.exists()
    payload = json.loads(results.read_text())
    assert payload["passed"] is True
    assert payload["n_budgets_bo_beats"] >= 2
    assert payload["fallback_recommendation"] is None
    assert len(payload["per_budget"]) == 3


# ---- explicit FAIL ledger --------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def test_cli_returns_1_when_bo_loses_on_all_budgets(tmp_path: Path) -> None:
    """Hand-roll a ledger where BO is uniformly worse than Sobol -> rc=1."""
    sobol = [{"j_fan": 1.0, "wall_time_hours": 0.5} for _ in range(20)]
    # BO J_fan above Sobol on every record -> never beats by 5%.
    bo = [{"j_fan": 2.0, "wall_time_hours": 0.5} for _ in range(30)]
    sobol_p = tmp_path / "sobol.jsonl"
    bo_p = tmp_path / "bo.jsonl"
    _write_jsonl(sobol_p, sobol)
    _write_jsonl(bo_p, bo)
    out = tmp_path / "results.json"

    rc = cli.main(
        [
            "--sobol-results",
            str(sobol_p),
            "--bo-results",
            str(bo_p),
            "--budgets",
            "5,10,20",
            "--out",
            str(out),
        ]
    )
    assert rc == 1
    payload = json.loads(out.read_text())
    assert payload["passed"] is False
    assert payload["n_budgets_bo_beats"] == 0
    # No GP-fit-time labels supplied -> retune fallback.
    assert payload["fallback_recommendation"] == "retune_acquisition"


def test_cli_passes_with_explicit_pass_ledger(tmp_path: Path) -> None:
    """Hand-roll a ledger where BO is 10% better than Sobol everywhere -> rc=0."""
    sobol = [{"j_fan": 1.0, "wall_time_hours": 0.5} for _ in range(20)]
    bo = [{"j_fan": 0.9, "wall_time_hours": 0.5} for _ in range(30)]
    sobol_p = tmp_path / "sobol.jsonl"
    bo_p = tmp_path / "bo.jsonl"
    _write_jsonl(sobol_p, sobol)
    _write_jsonl(bo_p, bo)
    out = tmp_path / "results.json"

    rc = cli.main(
        [
            "--sobol-results",
            str(sobol_p),
            "--bo-results",
            str(bo_p),
            "--budgets",
            "5,10,20",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["passed"] is True
    assert payload["n_budgets_bo_beats"] == 3


# ---- fallback-recommendation propagation -----------------------------------


def test_cli_propagates_high_d_label_to_saasbo_fallback(tmp_path: Path) -> None:
    sobol = [{"j_fan": 1.0, "wall_time_hours": 0.5} for _ in range(20)]
    bo = [{"j_fan": 2.0, "wall_time_hours": 0.5} for _ in range(30)]
    sobol_p = tmp_path / "sobol.jsonl"
    bo_p = tmp_path / "bo.jsonl"
    _write_jsonl(sobol_p, sobol)
    _write_jsonl(bo_p, bo)
    out = tmp_path / "results.json"

    rc = cli.main(
        [
            "--sobol-results",
            str(sobol_p),
            "--bo-results",
            str(bo_p),
            "--budgets",
            "5,10,20",
            "--gp-fit-time-above-60s-on",
            "high_d",
            "--out",
            str(out),
        ]
    )
    assert rc == 1
    payload = json.loads(out.read_text())
    assert payload["fallback_recommendation"] == "saasbo"


def test_cli_propagates_architecture_label_to_arch_fallback(tmp_path: Path) -> None:
    sobol = [{"j_fan": 1.0, "wall_time_hours": 0.5} for _ in range(20)]
    bo = [{"j_fan": 2.0, "wall_time_hours": 0.5} for _ in range(30)]
    sobol_p = tmp_path / "sobol.jsonl"
    bo_p = tmp_path / "bo.jsonl"
    _write_jsonl(sobol_p, sobol)
    _write_jsonl(bo_p, bo)
    out = tmp_path / "results.json"

    rc = cli.main(
        [
            "--sobol-results",
            str(sobol_p),
            "--bo-results",
            str(bo_p),
            "--budgets",
            "5,10,20",
            "--gp-fit-time-above-60s-on",
            "wide_architecture_set",
            "--out",
            str(out),
        ]
    )
    assert rc == 1
    payload = json.loads(out.read_text())
    assert payload["fallback_recommendation"] == "fix_architecture_set"


# ---- input-error paths ----------------------------------------------------


def test_cli_missing_sobol_returns_2(tmp_path: Path) -> None:
    bo = [{"j_fan": 1.0, "wall_time_hours": 0.5}]
    bo_p = tmp_path / "bo.jsonl"
    _write_jsonl(bo_p, bo)
    out = tmp_path / "results.json"
    rc = cli.main(
        [
            "--sobol-results",
            str(tmp_path / "missing.jsonl"),
            "--bo-results",
            str(bo_p),
            "--budgets",
            "5",
            "--out",
            str(out),
        ]
    )
    assert rc == 2


def test_cli_missing_bo_returns_2(tmp_path: Path) -> None:
    sobol = [{"j_fan": 1.0, "wall_time_hours": 0.5}]
    sobol_p = tmp_path / "sobol.jsonl"
    _write_jsonl(sobol_p, sobol)
    out = tmp_path / "results.json"
    rc = cli.main(
        [
            "--sobol-results",
            str(sobol_p),
            "--bo-results",
            str(tmp_path / "missing.jsonl"),
            "--budgets",
            "5",
            "--out",
            str(out),
        ]
    )
    assert rc == 2


def test_cli_empty_sobol_returns_2(tmp_path: Path) -> None:
    sobol_p = tmp_path / "sobol.jsonl"
    sobol_p.write_text("")
    bo_p = tmp_path / "bo.jsonl"
    _write_jsonl(bo_p, [{"j_fan": 1.0, "wall_time_hours": 0.5}])
    out = tmp_path / "results.json"
    rc = cli.main(
        [
            "--sobol-results",
            str(sobol_p),
            "--bo-results",
            str(bo_p),
            "--budgets",
            "5",
            "--out",
            str(out),
        ]
    )
    assert rc == 2


def test_cli_invalid_budget_returns_2(tmp_path: Path) -> None:
    sobol_p = tmp_path / "sobol.jsonl"
    _write_jsonl(sobol_p, [{"j_fan": 1.0, "wall_time_hours": 0.5}])
    bo_p = tmp_path / "bo.jsonl"
    _write_jsonl(bo_p, [{"j_fan": 0.5, "wall_time_hours": 0.5}])
    out = tmp_path / "results.json"
    rc = cli.main(
        [
            "--sobol-results",
            str(sobol_p),
            "--bo-results",
            str(bo_p),
            "--budgets",
            "0,not_a_number",
            "--out",
            str(out),
        ]
    )
    assert rc == 2


def test_cli_malformed_jsonl_raises(tmp_path: Path) -> None:
    """A line that isn't valid JSON triggers SystemExit (input error)."""
    sobol_p = tmp_path / "sobol.jsonl"
    sobol_p.write_text('{"j_fan": 1.0, "wall_time_hours": 0.5}\nthis is not json\n')
    bo_p = tmp_path / "bo.jsonl"
    _write_jsonl(bo_p, [{"j_fan": 0.5, "wall_time_hours": 0.5}])
    out = tmp_path / "results.json"
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--sobol-results",
                str(sobol_p),
                "--bo-results",
                str(bo_p),
                "--budgets",
                "5",
                "--out",
                str(out),
            ]
        )
