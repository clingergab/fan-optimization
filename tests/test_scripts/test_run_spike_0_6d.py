"""CLI smoke tests for ``scripts/run_spike_0_6d.py`` aggregator."""

from __future__ import annotations

import json
from pathlib import Path

import run_spike_0_6d as agg


def _write_sub_1_json(path: Path, *, passed: bool) -> Path:
    payload = {
        "spec_reference": "docs/report-final.md §Phase 0 Spike 0.6d",
        "lock_reference": "H10 supplement",
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


def _write_sub_2_json(path: Path, *, passed: bool) -> Path:
    payload = {
        "spec_reference": "docs/report-final.md §Phase 0 Spike 0.6d",
        "lock_reference": "H10 supplement",
        "mach_unsteady_lock": 1e-9,
        "closed_form_reference": "Sedov/Newman",
        "result": {
            "history_path": "<test>",
            "chord_m": 1.0,
            "pivot_offset_normalized": -0.5,
            "pitching_omega_rad_per_s": 10.0,
            "pitching_amplitude_rad": 0.1,
            "su2_moment_peak": 1.0,
            "closed_form_moment_peak": 1.0 if passed else 1.5,
            "relative_error": 0.0 if passed else -0.333,
            "tolerance": 0.15,
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


def test_aggregator_pass_writes_pass_marker(tmp_path: Path) -> None:
    sub_1 = _write_sub_1_json(tmp_path / "sub_1.json", passed=True)
    sub_2 = _write_sub_2_json(tmp_path / "sub_2.json", passed=True)
    out = tmp_path / "results.json"
    rc = agg.main(
        [
            "--sub-1-json",
            str(sub_1),
            "--sub-2-json",
            str(sub_2),
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
    sub_1 = _write_sub_1_json(tmp_path / "sub_1.json", passed=False)
    sub_2 = _write_sub_2_json(tmp_path / "sub_2.json", passed=True)
    rc = agg.main(
        [
            "--sub-1-json",
            str(sub_1),
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


def test_aggregator_fails_if_sub_2_fails(tmp_path: Path) -> None:
    sub_1 = _write_sub_1_json(tmp_path / "sub_1.json", passed=True)
    sub_2 = _write_sub_2_json(tmp_path / "sub_2.json", passed=False)
    rc = agg.main(
        [
            "--sub-1-json",
            str(sub_1),
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


def test_aggregator_ignores_sub_3_for_marker_decision(tmp_path: Path) -> None:
    """sub_3 FAIL + sub_1 PASS + sub_2 PASS → PASS marker still writes."""
    sub_1 = _write_sub_1_json(tmp_path / "sub_1.json", passed=True)
    sub_2 = _write_sub_2_json(tmp_path / "sub_2.json", passed=True)
    sub_3 = _write_sub_3_json(tmp_path / "sub_3.json", passed=False)
    rc = agg.main(
        [
            "--sub-1-json",
            str(sub_1),
            "--sub-2-json",
            str(sub_2),
            "--sub-3-json",
            str(sub_3),
            "--out",
            str(tmp_path / "results.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert (tmp_path / "PASS").exists()
    payload = json.loads((tmp_path / "results.json").read_text())
    assert payload["result"]["sub_06d_3"]["passed"] is False
    assert payload["result"]["overall_passed"] is True


def test_aggregator_payload_includes_sub_3_advisory_result(tmp_path: Path) -> None:
    sub_1 = _write_sub_1_json(tmp_path / "sub_1.json", passed=True)
    sub_2 = _write_sub_2_json(tmp_path / "sub_2.json", passed=True)
    sub_3 = _write_sub_3_json(tmp_path / "sub_3.json", passed=True)
    agg.main(
        [
            "--sub-1-json",
            str(sub_1),
            "--sub-2-json",
            str(sub_2),
            "--sub-3-json",
            str(sub_3),
            "--out",
            str(tmp_path / "results.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    payload = json.loads((tmp_path / "results.json").read_text())
    assert payload["result"]["sub_06d_3"] is not None
    assert payload["result"]["sub_06d_3"]["passed"] is True


def test_aggregator_payload_documents_phase5_step_62_5_target(tmp_path: Path) -> None:
    """The aggregator payload must point downstream consumers at step 62.5."""
    sub_1 = _write_sub_1_json(tmp_path / "sub_1.json", passed=True)
    sub_2 = _write_sub_2_json(tmp_path / "sub_2.json", passed=True)
    agg.main(
        [
            "--sub-1-json",
            str(sub_1),
            "--sub-2-json",
            str(sub_2),
            "--out",
            str(tmp_path / "results.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    payload = json.loads((tmp_path / "results.json").read_text())
    assert "62.5" in payload["phase5_step_62_5_pointer"]


def test_aggregator_missing_sub_1_returns_2(tmp_path: Path) -> None:
    sub_2 = _write_sub_2_json(tmp_path / "sub_2.json", passed=True)
    rc = agg.main(
        [
            "--sub-1-json",
            str(tmp_path / "nope.json"),
            "--sub-2-json",
            str(sub_2),
            "--out",
            str(tmp_path / "results.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2


def test_aggregator_missing_sub_2_returns_2(tmp_path: Path) -> None:
    sub_1 = _write_sub_1_json(tmp_path / "sub_1.json", passed=True)
    rc = agg.main(
        [
            "--sub-1-json",
            str(sub_1),
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
    sub_1 = _write_sub_1_json(tmp_path / "sub_1.json", passed=True)
    sub_2 = _write_sub_2_json(tmp_path / "sub_2.json", passed=True)
    rc = agg.main(
        [
            "--sub-1-json",
            str(sub_1),
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


def test_aggregator_handles_missing_sub_3_file_gracefully(tmp_path: Path) -> None:
    """If --sub-3-json points to a non-existent file, sub_3 is skipped (not an error)."""
    sub_1 = _write_sub_1_json(tmp_path / "sub_1.json", passed=True)
    sub_2 = _write_sub_2_json(tmp_path / "sub_2.json", passed=True)
    rc = agg.main(
        [
            "--sub-1-json",
            str(sub_1),
            "--sub-2-json",
            str(sub_2),
            "--sub-3-json",
            str(tmp_path / "nonexistent_sub_3.json"),
            "--out",
            str(tmp_path / "results.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert (tmp_path / "PASS").exists()
    payload = json.loads((tmp_path / "results.json").read_text())
    assert payload["result"]["sub_06d_3"] is None
