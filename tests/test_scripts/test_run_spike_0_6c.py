"""CLI smoke tests for the Spike 0.6c runners + aggregator.

Exercises:

* ``scripts/run_spike_0_6c_1.py`` — cfg-only fallback path (no SU2
  installed in the sandbox, so the runner falls back to a parser-only
  check and exits 1 with `sub_1.FAIL`).
* ``scripts/run_spike_0_6c_2.py`` — canonical PASS + over-tolerance FAIL.
* ``scripts/run_spike_0_6c.py`` — aggregator combining both sub-spike
  result JSONs.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.6c; protocol in
docs/spike_0_6c_protocol.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_spike_0_6c as agg
import run_spike_0_6c_1 as sub1
import run_spike_0_6c_2 as sub2


# ---- fixtures -------------------------------------------------------------


def _measured_csv(path: Path, *, scale: float = 1.0) -> Path:
    """Write a 5-cycle measured CSV.

    Cycle 0 is the initial transient (discarded). Cycles 1-4 are each
    scaled by ``scale`` from the shipped NACA0012_REFERENCE — `scale=1.0`
    PASSes; `scale=1.20` FAILs the c_l_max metric by +20%.
    """
    from fanopt.cfd.spike_0_6c import NACA0012_REFERENCE as R

    rows = [
        "cycle_index,c_l_max,c_l_min,c_d_mean,c_l_hysteresis_area",
        f"0,99.0,-99.0,9.9,99.0",
    ]
    for i in range(1, 5):
        rows.append(
            f"{i},{R['c_l_max'] * scale},{R['c_l_min']},{R['c_d_mean']},{R['c_l_hysteresis_area']}"
        )
    path.write_text("\n".join(rows) + "\n")
    return path


def _reference_json(path: Path) -> Path:
    from fanopt.cfd.spike_0_6c import NACA0012_REFERENCE as R

    path.write_text(json.dumps(dict(R), indent=2))
    return path


# ---- sub-spike 0.6c.1 (cfg-only fallback path) ----------------------------


def test_sub_1_cfg_only_fallback_writes_fail_marker(tmp_path: Path) -> None:
    """Without SU2 on PATH the runner takes the cfg-only fallback path.

    The cfg parser sees MACH = 1e-9 + FREESTREAM_OPTION = FREESTREAM_VELOCITY
    in the spec's canonical block, but with no SU2 log the outer-step gate
    cannot clear — so ``passed`` is False and the runner exits 1 with a
    `sub_1.FAIL` marker. The result JSON still records all the cfg-level
    fields (lock parts that DID pass).
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
    assert r["freestream_option"] == "FREESTREAM_VELOCITY"
    # And the marker file matches the exit code.
    expected_marker = "sub_1.PASS" if rc == 0 else "sub_1.FAIL"
    assert (marker_dir / expected_marker).exists()


# ---- sub-spike 0.6c.2 (PASS + FAIL paths) ---------------------------------


def test_sub_2_canonical_pass(tmp_path: Path) -> None:
    measured = _measured_csv(tmp_path / "measured.csv", scale=1.0)
    result_json = tmp_path / "sub_2_result.json"
    marker_dir = tmp_path
    rc = sub2.main(
        [
            "--measured",
            str(measured),
            "--k-reduced",
            "0.55",
            "--reynolds",
            "40000",
            "--reference-source",
            "test:NACA0012",
            "--result-json",
            str(result_json),
            "--marker-dir",
            str(marker_dir),
        ]
    )
    assert rc == 0
    payload = json.loads(result_json.read_text())
    r = payload["result"]
    assert r["passed"] is True
    assert r["all_metrics_within_15pct"] is True
    assert r["k_reduced"] == pytest.approx(0.55)
    assert r["reynolds"] == pytest.approx(40000)
    # All four metrics compared, all PASS.
    by_name = {c["metric_name"]: c for c in r["comparisons"]}
    assert set(by_name.keys()) == {
        "c_l_max",
        "c_l_min",
        "c_d_mean",
        "c_l_hysteresis_area",
    }
    assert all(c["passed"] for c in r["comparisons"])
    assert (marker_dir / "sub_2.PASS").exists()
    assert not (marker_dir / "sub_2.FAIL").exists()


def test_sub_2_over_tolerance_fails(tmp_path: Path) -> None:
    """A 20% c_l_max deviation fails the ±15% gate; exit 1, FAIL marker."""
    measured = _measured_csv(tmp_path / "measured.csv", scale=1.20)
    result_json = tmp_path / "sub_2_result.json"
    marker_dir = tmp_path
    rc = sub2.main(
        [
            "--measured",
            str(measured),
            "--k-reduced",
            "0.55",
            "--reynolds",
            "40000",
            "--result-json",
            str(result_json),
            "--marker-dir",
            str(marker_dir),
        ]
    )
    assert rc == 1
    payload = json.loads(result_json.read_text())
    r = payload["result"]
    assert r["passed"] is False
    by_name = {c["metric_name"]: c for c in r["comparisons"]}
    assert by_name["c_l_max"]["passed"] is False
    assert by_name["c_l_min"]["passed"] is True
    assert (marker_dir / "sub_2.FAIL").exists()
    assert not (marker_dir / "sub_2.PASS").exists()


def test_sub_2_accepts_reference_json_override(tmp_path: Path) -> None:
    measured = _measured_csv(tmp_path / "measured.csv", scale=1.0)
    ref = _reference_json(tmp_path / "ref.json")
    rc = sub2.main(
        [
            "--measured",
            str(measured),
            "--reference",
            str(ref),
            "--k-reduced",
            "0.55",
            "--reynolds",
            "40000",
            "--result-json",
            str(tmp_path / "sub_2_result.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0


def test_sub_2_missing_column_returns_2(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "cycle_index,c_l_max,c_l_min,c_d_mean\n"
        "1,1.2,-1.2,0.085\n"
    )
    rc = sub2.main(
        [
            "--measured",
            str(bad),
            "--k-reduced",
            "0.55",
            "--reynolds",
            "40000",
            "--result-json",
            str(tmp_path / "sub_2_result.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2


def test_sub_2_missing_measured_file_returns_2(tmp_path: Path) -> None:
    rc = sub2.main(
        [
            "--measured",
            str(tmp_path / "nope.csv"),
            "--k-reduced",
            "0.55",
            "--reynolds",
            "40000",
            "--result-json",
            str(tmp_path / "sub_2_result.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2


# ---- aggregator -----------------------------------------------------------


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


def _write_sub_2_json(path: Path, *, passed: bool) -> Path:
    from fanopt.cfd.spike_0_6c import NACA0012_REFERENCE as R

    measured_scale = 1.0 if passed else 1.20
    cycles = [
        {
            "cycle_index": 0,
            "c_l_max": 99.0,
            "c_l_min": -99.0,
            "c_d_mean": 9.9,
            "c_l_hysteresis_area": 99.0,
        }
    ] + [
        {
            "cycle_index": i,
            "c_l_max": R["c_l_max"] * measured_scale,
            "c_l_min": R["c_l_min"],
            "c_d_mean": R["c_d_mean"],
            "c_l_hysteresis_area": R["c_l_hysteresis_area"],
        }
        for i in range(1, 5)
    ]
    pct_c_l_max = 100.0 * (measured_scale - 1.0)
    comparisons = [
        {
            "metric_name": "c_l_max",
            "measured": R["c_l_max"] * measured_scale,
            "reference": R["c_l_max"],
            "pct_diff": pct_c_l_max,
            "passed": passed,
        },
        {
            "metric_name": "c_l_min",
            "measured": R["c_l_min"],
            "reference": R["c_l_min"],
            "pct_diff": 0.0,
            "passed": True,
        },
        {
            "metric_name": "c_d_mean",
            "measured": R["c_d_mean"],
            "reference": R["c_d_mean"],
            "pct_diff": 0.0,
            "passed": True,
        },
        {
            "metric_name": "c_l_hysteresis_area",
            "measured": R["c_l_hysteresis_area"],
            "reference": R["c_l_hysteresis_area"],
            "pct_diff": 0.0,
            "passed": True,
        },
    ]
    payload = {
        "spec_reference": "docs/plan_R11.md §Phase 0 Spike 0.6c (sub-spike 0.6c.2)",
        "result": {
            "k_reduced": 0.55,
            "reynolds": 40000.0,
            "cycles": cycles,
            "reference_source": "test",
            "comparisons": comparisons,
            "all_metrics_within_15pct": passed,
            "passed": passed,
        },
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def test_aggregator_pass_writes_pass_marker(tmp_path: Path) -> None:
    sub_1 = _write_sub_1_json(tmp_path / "sub_1_result.json", passed=True)
    sub_2 = _write_sub_2_json(tmp_path / "sub_2_result.json", passed=True)
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
    sub_1 = _write_sub_1_json(tmp_path / "sub_1_result.json", passed=False)
    sub_2 = _write_sub_2_json(tmp_path / "sub_2_result.json", passed=True)
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
    assert rc == 1
    payload = json.loads(out.read_text())
    assert payload["result"]["overall_passed"] is False
    assert (tmp_path / "FAIL").exists()


def test_aggregator_fails_if_sub_2_fails(tmp_path: Path) -> None:
    sub_1 = _write_sub_1_json(tmp_path / "sub_1_result.json", passed=True)
    sub_2 = _write_sub_2_json(tmp_path / "sub_2_result.json", passed=False)
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
    assert rc == 1


def test_aggregator_missing_sub_1_returns_2(tmp_path: Path) -> None:
    sub_2 = _write_sub_2_json(tmp_path / "sub_2_result.json", passed=True)
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


def test_sub_2_fails_when_k_reduced_out_of_band(tmp_path: Path) -> None:
    """k_reduced=2.0 must hard-fail (exit 2) — not just warn — even if the
    rendered metrics would otherwise pass the ±15% gate."""
    measured = _measured_csv(tmp_path / "measured.csv", scale=1.0)
    rc = sub2.main(
        [
            "--measured", str(measured),
            "--k-reduced", "2.0",  # WAY outside [0.5, 0.6]
            "--reynolds", "40000",
            "--reference-source", "test:NACA0012",
            "--result-json", str(tmp_path / "sub_2_result.json"),
            "--marker-dir", str(tmp_path),
        ]
    )
    assert rc == 2


def test_sub_2_fails_when_reynolds_out_of_band(tmp_path: Path) -> None:
    """Re=1e6 must hard-fail."""
    measured = _measured_csv(tmp_path / "measured.csv", scale=1.0)
    rc = sub2.main(
        [
            "--measured", str(measured),
            "--k-reduced", "0.55",
            "--reynolds", "1000000",  # outside [30000, 50000]
            "--reference-source", "test:NACA0012",
            "--result-json", str(tmp_path / "sub_2_result.json"),
            "--marker-dir", str(tmp_path),
        ]
    )
    assert rc == 2


def test_sub_2_allow_out_of_band_permits_override(tmp_path: Path) -> None:
    """--allow-out-of-band makes the band check a warning instead of a gate."""
    measured = _measured_csv(tmp_path / "measured.csv", scale=1.0)
    rc = sub2.main(
        [
            "--measured", str(measured),
            "--k-reduced", "2.0",
            "--reynolds", "40000",
            "--reference-source", "test:CustomReference",
            "--result-json", str(tmp_path / "sub_2_result.json"),
            "--marker-dir", str(tmp_path),
            "--allow-out-of-band",
        ]
    )
    # Run completes; pass/fail depends on the actual metric comparison.
    assert rc in (0, 1)


def test_aggregator_clears_stale_opposite_marker(tmp_path: Path) -> None:
    """Writing PASS should clear any pre-existing FAIL marker (and vice versa)."""
    # Pre-place a stale FAIL marker.
    (tmp_path / "FAIL").write_text("")
    sub_1 = _write_sub_1_json(tmp_path / "sub_1_result.json", passed=True)
    sub_2 = _write_sub_2_json(tmp_path / "sub_2_result.json", passed=True)
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
