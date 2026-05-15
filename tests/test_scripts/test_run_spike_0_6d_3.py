"""CLI smoke tests for ``scripts/run_spike_0_6d_3.py`` (advisory; no marker)."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import run_spike_0_6d_3 as sub3


def _history_with_mean(*, mean: float, n_cycles: int = 5, samples_per_cycle: int = 200) -> str:
    out = io.StringIO()
    out.write("Time_Iter,Inner_Iter,CFx,CD\n")
    total = n_cycles * samples_per_cycle
    for i in range(total):
        phase = 2.0 * math.pi * (i % samples_per_cycle) / samples_per_cycle
        cfx = mean + 0.1 * math.sin(phase)
        out.write(f"{i},0,{cfx:.6f},0.05\n")
    return out.getvalue()


def test_sub_3_writes_advisory_result_json_no_marker(tmp_path: Path) -> None:
    """sub_3 writes its result JSON; does NOT write a PASS/FAIL marker."""
    comp_p = tmp_path / "comp.csv"
    comp_p.write_text(_history_with_mean(mean=0.50))
    incomp_p = tmp_path / "incomp.csv"
    incomp_p.write_text(_history_with_mean(mean=0.55))  # +10%, within 20%

    rc = sub3.main(
        [
            "--comp-history-csv",
            str(comp_p),
            "--incomp-history-csv",
            str(incomp_p),
            "--result-json",
            str(tmp_path / "sub_3_result.json"),
        ]
    )
    assert rc == 0
    # No marker files written — sub_3 is advisory.
    assert not (tmp_path / "sub_3.PASS").exists()
    assert not (tmp_path / "sub_3.FAIL").exists()
    payload = json.loads((tmp_path / "sub_3_result.json").read_text())
    assert payload["result"]["passed"] is True
    # The note must communicate advisory semantics so downstream consumers know
    # the result doesn't gate Phase 4. We check for either "advisory" itself
    # or "does NOT block" / "not block Phase 4" phrasing.
    note = payload["advisory_note"].lower()
    assert "advisory" in note or "does not block" in note or "not block phase 4" in note


def test_sub_3_logs_pass_or_fail_in_payload(tmp_path: Path) -> None:
    """Disagreement >20% -> rc=1 but still advisory; payload records the result."""
    comp_p = tmp_path / "comp.csv"
    comp_p.write_text(_history_with_mean(mean=0.50))
    incomp_p = tmp_path / "incomp.csv"
    incomp_p.write_text(_history_with_mean(mean=0.80))  # +60%

    rc = sub3.main(
        [
            "--comp-history-csv",
            str(comp_p),
            "--incomp-history-csv",
            str(incomp_p),
            "--result-json",
            str(tmp_path / "sub_3_result.json"),
        ]
    )
    assert rc == 1  # advisory failure
    payload = json.loads((tmp_path / "sub_3_result.json").read_text())
    assert payload["result"]["passed"] is False


def test_sub_3_returns_2_when_history_missing(tmp_path: Path) -> None:
    rc = sub3.main(
        [
            "--comp-history-csv",
            str(tmp_path / "no_such.csv"),
            "--incomp-history-csv",
            str(tmp_path / "also_not.csv"),
            "--result-json",
            str(tmp_path / "sub_3_result.json"),
        ]
    )
    assert rc == 2
