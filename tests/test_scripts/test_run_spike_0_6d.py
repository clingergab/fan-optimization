"""CLI smoke tests for ``scripts/run_spike_0_6d.py`` aggregator.

Post-2026-05-15 redesign: the gate is sub-spike 0.6d.2's
frequency-consistency result ONLY. sub_1 (symmetry/dimensional) and
sub_3 (incompressible) are advisory + optional — recorded for Phase 5,
never gating.
"""

from __future__ import annotations

import json
from pathlib import Path

import run_spike_0_6d as agg


def _write_sub_2_json(path: Path, *, passed: bool) -> Path:
    """Write a redesigned (freq-consistency) 0.6d.2 result JSON."""
    payload = {
        "spec_reference": "docs/report-final.md §Phase 0 Spike 0.6d (sub-spike 0.6d.2)",
        "lock_reference": "H10 supplement; redesigned 2026-05-15",
        "mach_unsteady_lock": 1e-9,
        "result": {
            "omega_f1_rad_per_s": 6.2832,
            "omega_f2_rad_per_s": 12.5664,
            "recovered_ia_nondim_f1": 0.037,
            "recovered_ia_nondim_f2": 0.037 if passed else 0.10,
            "freq_consistency_rel_diff": 0.0 if passed else 0.92,
            "freq_consistency_tol": 0.25,
            "freq_consistency_passed": passed,
            "closed_form_ia_nondim": 0.0902,
            "closed_form_factor_f1": 0.41,
            "closed_form_factor_tol": 2.0,
            "closed_form_advisory_ok": False,
            "drag_to_added_mass_ratio_f1": 0.01,
            "passed": passed,
        },
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def _write_sub_1_json(path: Path, *, passed: bool) -> Path:
    payload = {
        "spec_reference": "docs/report-final.md §Phase 0 Spike 0.6d (sub-spike 0.6d.1)",
        "lock_reference": "H10 supplement (ADVISORY post-2026-05-15)",
        "mach_unsteady_lock": 1e-9,
        "result": {
            "history_path": "<test>",
            "n_cycles": 5,
            "force_cycle_avg": 0.001 if passed else 0.5,
            "force_cycle_peak": 0.5,
            "force_envelope": 0.5,
            "envelope_geometry": "test fixture",
            "symmetry_ratio": 0.002 if passed else 1.0,
            "symmetry_passed": passed,
            "magnitude_ratio_log10": 0.0,
            "magnitude_passed": True,
            "passed": passed,
        },
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def _write_sub_3_json(path: Path, *, passed: bool) -> Path:
    payload = {
        "spec_reference": "docs/report-final.md §Phase 0 Spike 0.6d (sub-spike 0.6d.3)",
        "lock_reference": "H10 supplement (ADVISORY)",
        "mach_unsteady_lock": 1e-9,
        "advisory_note": "Failure does NOT block Phase 4.",
        "result": {
            "compressible_force_cycle_avg": 0.5,
            "incompressible_force_cycle_avg": 0.55 if passed else 0.9,
            "relative_error": 0.1 if passed else 0.444,
            "tolerance": 0.2,
            "passed": passed,
        },
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def test_aggregator_pass_when_sub_2_consistent(tmp_path: Path) -> None:
    sub_2 = _write_sub_2_json(tmp_path / "sub_2.json", passed=True)
    out = tmp_path / "results.json"
    rc = agg.main(["--sub-2-json", str(sub_2), "--out", str(out), "--marker-dir", str(tmp_path)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["result"]["overall_passed"] is True
    assert (tmp_path / "PASS").exists()
    assert not (tmp_path / "FAIL").exists()


def test_aggregator_fail_when_sub_2_inconsistent(tmp_path: Path) -> None:
    sub_2 = _write_sub_2_json(tmp_path / "sub_2.json", passed=False)
    rc = agg.main(
        [
            "--sub-2-json",
            str(sub_2),
            "--out",
            str(tmp_path / "results.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 1
    assert (tmp_path / "FAIL").exists()


def test_aggregator_gate_ignores_advisory_sub_1_and_sub_3(tmp_path: Path) -> None:
    """sub_1 FAIL + sub_3 FAIL must NOT block when sub_2 passes."""
    sub_2 = _write_sub_2_json(tmp_path / "sub_2.json", passed=True)
    sub_1 = _write_sub_1_json(tmp_path / "sub_1.json", passed=False)
    sub_3 = _write_sub_3_json(tmp_path / "sub_3.json", passed=False)
    out = tmp_path / "results.json"
    rc = agg.main(
        [
            "--sub-2-json",
            str(sub_2),
            "--sub-1-json",
            str(sub_1),
            "--sub-3-json",
            str(sub_3),
            "--out",
            str(out),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["result"]["overall_passed"] is True
    # Advisory results still recorded for Phase 5.
    assert payload["result"]["sub_06d_1"]["passed"] is False
    assert payload["result"]["sub_06d_3"]["passed"] is False


def test_aggregator_works_with_sub_2_only(tmp_path: Path) -> None:
    """sub_1 / sub_3 optional — sub_2 alone determines the gate."""
    sub_2 = _write_sub_2_json(tmp_path / "sub_2.json", passed=True)
    out = tmp_path / "results.json"
    rc = agg.main(["--sub-2-json", str(sub_2), "--out", str(out), "--marker-dir", str(tmp_path)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["result"]["sub_06d_1"] is None
    assert payload["result"]["sub_06d_3"] is None


def test_aggregator_payload_documents_freq_consistency_gate(tmp_path: Path) -> None:
    sub_2 = _write_sub_2_json(tmp_path / "sub_2.json", passed=True)
    out = tmp_path / "results.json"
    agg.main(["--sub-2-json", str(sub_2), "--out", str(out), "--marker-dir", str(tmp_path)])
    payload = json.loads(out.read_text())
    assert "freq_consistency_passed" in payload["gate_note"]
    assert "62.5" in payload["phase5_step_62_5_pointer"]


def test_aggregator_missing_sub_2_returns_2(tmp_path: Path) -> None:
    rc = agg.main(
        [
            "--sub-2-json",
            str(tmp_path / "nope.json"),
            "--out",
            str(tmp_path / "results.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2


def test_aggregator_clears_stale_opposite_marker(tmp_path: Path) -> None:
    (tmp_path / "FAIL").write_text("")
    sub_2 = _write_sub_2_json(tmp_path / "sub_2.json", passed=True)
    rc = agg.main(
        [
            "--sub-2-json",
            str(sub_2),
            "--out",
            str(tmp_path / "results.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert (tmp_path / "PASS").exists()
    assert not (tmp_path / "FAIL").exists()


def test_aggregator_ignores_unreadable_advisory_sub_1(tmp_path: Path) -> None:
    """A malformed advisory sub_1 is ignored, not fatal — sub_2 still gates."""
    sub_2 = _write_sub_2_json(tmp_path / "sub_2.json", passed=True)
    bad_sub_1 = tmp_path / "sub_1.json"
    bad_sub_1.write_text("{ not json")
    rc = agg.main(
        [
            "--sub-2-json",
            str(sub_2),
            "--sub-1-json",
            str(bad_sub_1),
            "--out",
            str(tmp_path / "results.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    payload = json.loads((tmp_path / "results.json").read_text())
    assert payload["result"]["overall_passed"] is True
    assert payload["result"]["sub_06d_1"] is None  # unreadable → skipped
