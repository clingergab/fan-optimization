"""CLI smoke tests for the Spike 0.6c runners + aggregator (V1 scope).

Sub-spike 0.6c.2 (NACA 0012 numerical-consistency benchmark) was
deferred to Phase 5 on 2026-05-14 (see
``docs/phase_logs/spike_0_6c.md``). The 0.6c.2 runner
(``scripts/run_spike_0_6c_2.py``) was removed; its tests live in the
git history.

Exercises:

* ``scripts/run_spike_0_6c_1.py`` — cfg-only fallback path (no SU2
  installed in the sandbox, so the runner falls back to a parser-only
  check and exits 1 with ``sub_1.FAIL``).
* ``scripts/run_spike_0_6c.py`` — aggregator (now reads only sub_1).

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.6c; protocol in
docs/spike_0_6c_protocol.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import run_spike_0_6c as agg
import run_spike_0_6c_1 as sub1

# ---- sub-spike 0.6c.1 (cfg-only fallback path) ----------------------------


def test_sub_1_cfg_only_fallback_writes_fail_marker(tmp_path: Path) -> None:
    """Without SU2 on PATH the runner takes the cfg-only fallback path.

    The cfg parser sees MACH = 1e-9 + REF_DIMENSIONALIZATION =
    FREESTREAM_PRESS_EQ_ONE (the Round-9 HIGH-12 fallback path the
    template ships, since SU2 v8.0.1 rejects the primary
    FREESTREAM_OPTION = FREESTREAM_VELOCITY directive). With no SU2 log
    the outer-step gate cannot clear — so ``passed`` is False and the
    runner exits 1 with a ``sub_1.FAIL`` marker. The result JSON still
    records all the cfg-level fields (lock parts that DID pass).
    """
    result_json = tmp_path / "sub_1_result.json"
    marker_dir = tmp_path
    rc = sub1.main(
        [
            "--result-json",
            str(result_json),
            "--marker-dir",
            str(marker_dir),
        ]
    )
    # SU2 not installed in the test sandbox -> outer-step gate fails -> exit 1.
    # If SU2 happens to be installed in the running environment (unlikely
    # for CI), the gate could pass; we accept either 0 or 1 here since the
    # script's job is to faithfully report whichever path triggered.
    assert rc in (0, 1)
    payload = json.loads(result_json.read_text())
    assert payload["spec_reference"].startswith("docs/plan_R11.md")
    assert payload["mach_unsteady_lock"] == 1e-9
    r = payload["result"]
    # The cfg's lock invariants should parse OK regardless of SU2 install.
    assert r["parsed_ok"] is True
    assert r["mach_value"] == 1e-9
    # Round-9 HIGH-12 fallback path: REF_DIMENSIONALIZATION pins the
    # freestream state; FREESTREAM_OPTION is intentionally absent.
    assert r["freestream_option"] == ""
    assert r["ref_dimensionalization"] == "FREESTREAM_PRESS_EQ_ONE"
    # And the marker file matches the exit code.
    expected_marker = "sub_1.PASS" if rc == 0 else "sub_1.FAIL"
    assert (marker_dir / expected_marker).exists()


# ---- evidence-from-prior-run paths (2026-05-14 bugfix) --------------------


def _write_history_csv(path: Path, *, n_outer_steps: int) -> Path:
    """Write a minimal SU2-style history.csv with n_outer_steps rows."""
    lines = ['"Time_Iter","Inner_Iter","CL","CD"']
    for i in range(n_outer_steps):
        lines.append(f"{i},0,0.1,0.05")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_sub_1_passes_with_history_csv_evidence(tmp_path: Path) -> None:
    """Operator-recovery path: pass a Drive history.csv from a prior
    successful SU2 run; runner counts rows as outer-step evidence,
    writes sub_1.PASS without re-invoking SU2."""
    history = _write_history_csv(tmp_path / "history.csv", n_outer_steps=5)
    rc = sub1.main(
        [
            "--su2-history-csv",
            str(history),
            "--result-json",
            str(tmp_path / "sub_1_result.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert (tmp_path / "sub_1.PASS").exists()
    assert not (tmp_path / "sub_1.FAIL").exists()
    r = json.loads((tmp_path / "sub_1_result.json").read_text())["result"]
    assert r["passed"] is True
    assert r["outer_time_steps_completed"] >= 1


def test_sub_1_returns_2_when_history_csv_missing(tmp_path: Path) -> None:
    """If --su2-history-csv points at a non-existent file, exit 2 (input error)."""
    rc = sub1.main(
        [
            "--su2-history-csv",
            str(tmp_path / "no_such_history.csv"),
            "--result-json",
            str(tmp_path / "sub_1_result.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2


def test_sub_1_passes_with_su2_log_file_evidence(tmp_path: Path) -> None:
    """Operator-recovery path: pass a prior SU2 stdout log file directly."""
    log_path = tmp_path / "su2.log"
    log_path.write_text("Time_Iter: 0\nTime_Iter: 1\nTime_Iter: 2\n")
    rc = sub1.main(
        [
            "--su2-log-file",
            str(log_path),
            "--result-json",
            str(tmp_path / "sub_1_result.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert (tmp_path / "sub_1.PASS").exists()
    r = json.loads((tmp_path / "sub_1_result.json").read_text())["result"]
    assert r["outer_time_steps_completed"] == 3


def test_sub_1_returns_2_when_su2_log_file_missing(tmp_path: Path) -> None:
    rc = sub1.main(
        [
            "--su2-log-file",
            str(tmp_path / "no_such.log"),
            "--result-json",
            str(tmp_path / "sub_1_result.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2


def test_sub_1_returns_2_when_probe_mesh_missing(tmp_path: Path) -> None:
    """--probe-mesh that doesn't exist is an input error, not a silent failure.

    Catches the original 2026-05 Colab bug where the runner invoked SU2 with
    a nonexistent ``probe.su2`` and silently returned outer_steps=0.
    """
    rc = sub1.main(
        [
            "--probe-mesh",
            str(tmp_path / "no_such_probe.su2"),
            "--result-json",
            str(tmp_path / "sub_1_result.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2


def test_sub_1_history_csv_takes_priority_over_log_file(tmp_path: Path) -> None:
    """If both --su2-history-csv and --su2-log-file are supplied, history wins."""
    history = _write_history_csv(tmp_path / "history.csv", n_outer_steps=7)
    log_path = tmp_path / "su2.log"
    log_path.write_text("Time_Iter: 0\nTime_Iter: 1\n")  # 2 steps, conflicting
    rc = sub1.main(
        [
            "--su2-history-csv",
            str(history),
            "--su2-log-file",
            str(log_path),
            "--result-json",
            str(tmp_path / "sub_1_result.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    r = json.loads((tmp_path / "sub_1_result.json").read_text())["result"]
    # Should use history.csv (7 steps), NOT the log file (2 steps).
    assert r["outer_time_steps_completed"] == 7


# ---- aggregator (V1: gates on sub_1 only) ---------------------------------


def _write_sub_1_json(path: Path, *, passed: bool) -> Path:
    payload = {
        "spec_reference": "docs/plan_R11.md §Phase 0 Spike 0.6c (sub-spike 0.6c.1)",
        "lock_reference": "Round-9 HIGH-12 (= C12)",
        "mach_unsteady_lock": 1e-9,
        "result": {
            "cfg_path": "<test>",
            "parsed_ok": True,
            "mach_value": 1e-9,
            "freestream_option": "FREESTREAM_VELOCITY",
            "ref_dimensionalization": None,
            "outer_time_steps_completed": 1 if passed else 0,
            "error": None,
            "passed": passed,
        },
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def test_aggregator_pass_writes_pass_marker(tmp_path: Path) -> None:
    sub_1 = _write_sub_1_json(tmp_path / "sub_1_result.json", passed=True)
    out = tmp_path / "results.json"
    rc = agg.main(
        [
            "--sub-1-json",
            str(sub_1),
            "--out",
            str(out),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["result"]["overall_passed"] is True
    assert (tmp_path / "PASS").exists()
    assert not (tmp_path / "FAIL").exists()


def test_aggregator_fails_if_sub_1_fails(tmp_path: Path) -> None:
    sub_1 = _write_sub_1_json(tmp_path / "sub_1_result.json", passed=False)
    out = tmp_path / "results.json"
    rc = agg.main(
        [
            "--sub-1-json",
            str(sub_1),
            "--out",
            str(out),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 1
    payload = json.loads(out.read_text())
    assert payload["result"]["overall_passed"] is False
    assert (tmp_path / "FAIL").exists()


def test_aggregator_missing_sub_1_returns_2(tmp_path: Path) -> None:
    rc = agg.main(
        [
            "--sub-1-json",
            str(tmp_path / "nope.json"),
            "--out",
            str(tmp_path / "results.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2


def test_aggregator_clears_stale_opposite_marker(tmp_path: Path) -> None:
    """Writing PASS should clear any pre-existing FAIL marker (and vice versa)."""
    # Pre-place a stale FAIL marker.
    (tmp_path / "FAIL").write_text("")
    sub_1 = _write_sub_1_json(tmp_path / "sub_1_result.json", passed=True)
    rc = agg.main(
        [
            "--sub-1-json",
            str(sub_1),
            "--out",
            str(tmp_path / "results.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert (tmp_path / "PASS").exists()
    assert not (tmp_path / "FAIL").exists()


def test_aggregator_payload_includes_phase5_deferral_note(tmp_path: Path) -> None:
    """The result JSON must include the V1-scope note so downstream
    consumers (Phase 5 prep, audit) see that 0.6c.2 was deferred — not
    silently dropped."""
    sub_1 = _write_sub_1_json(tmp_path / "sub_1_result.json", passed=True)
    out = tmp_path / "results.json"
    agg.main(
        [
            "--sub-1-json",
            str(sub_1),
            "--out",
            str(out),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    payload = json.loads(out.read_text())
    assert "v1_scope_note" in payload
    assert "Phase 5" in payload["v1_scope_note"]
