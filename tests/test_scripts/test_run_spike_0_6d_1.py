"""CLI smoke tests for ``scripts/run_spike_0_6d_1.py``."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import run_spike_0_6d_1 as sub1


def _sinusoidal_history(
    *, n_cycles: int, samples_per_cycle: int, amplitude: float, mean_bias: float = 0.0
) -> str:
    out = io.StringIO()
    out.write("Time_Iter,Inner_Iter,CFx,CD\n")
    total = n_cycles * samples_per_cycle
    for i in range(total):
        phase = 2.0 * math.pi * (i % samples_per_cycle) / samples_per_cycle
        cfx = mean_bias + amplitude * math.sin(phase)
        out.write(f"{i},0,{cfx:.6f},0.05\n")
    return out.getvalue()


def _write_history(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "history.csv"
    p.write_text(text)
    return p


def test_sub_1_passes_on_synthetic_clean_history(tmp_path: Path) -> None:
    """Symmetric trace within envelope → exit 0 + sub_1.PASS marker."""
    history = _write_history(
        tmp_path,
        _sinusoidal_history(n_cycles=5, samples_per_cycle=200, amplitude=0.5),
    )
    result_json = tmp_path / "sub_1_result.json"
    marker_dir = tmp_path
    rc = sub1.main(
        [
            "--history-csv",
            str(history),
            "--mass-kg",
            "0.05",
            "--omega-rad-per-s",
            "12.5664",
            "--r-cm-m",
            "0.1",
            "--result-json",
            str(result_json),
            "--marker-dir",
            str(marker_dir),
        ]
    )
    assert rc == 0
    assert (marker_dir / "sub_1.PASS").exists()
    assert not (marker_dir / "sub_1.FAIL").exists()
    payload = json.loads(result_json.read_text())
    assert payload["result"]["passed"] is True
    assert payload["mach_unsteady_lock"] == 1e-9


def test_sub_1_fails_on_biased_history(tmp_path: Path) -> None:
    """Non-zero-mean CFx → symmetry fails → exit 1 + sub_1.FAIL marker."""
    history = _write_history(
        tmp_path,
        _sinusoidal_history(n_cycles=5, samples_per_cycle=200, amplitude=1.0, mean_bias=2.0),
    )
    rc = sub1.main(
        [
            "--history-csv",
            str(history),
            "--mass-kg",
            "0.05",
            "--omega-rad-per-s",
            "12.5664",
            "--r-cm-m",
            "0.1",
            "--result-json",
            str(tmp_path / "sub_1_result.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 1
    assert (tmp_path / "sub_1.FAIL").exists()
    assert not (tmp_path / "sub_1.PASS").exists()


def test_sub_1_writes_result_json_with_all_diagnostics(tmp_path: Path) -> None:
    history = _write_history(
        tmp_path,
        _sinusoidal_history(n_cycles=5, samples_per_cycle=200, amplitude=0.5),
    )
    result_json = tmp_path / "sub_1_result.json"
    sub1.main(
        [
            "--history-csv",
            str(history),
            "--mass-kg",
            "0.05",
            "--omega-rad-per-s",
            "12.5664",
            "--r-cm-m",
            "0.1",
            "--envelope-geometry",
            "test fixture",
            "--result-json",
            str(result_json),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    r = json.loads(result_json.read_text())["result"]
    # All diagnostic fields present.
    for key in (
        "force_cycle_avg",
        "force_cycle_peak",
        "force_envelope",
        "envelope_geometry",
        "symmetry_ratio",
        "symmetry_passed",
        "magnitude_ratio_log10",
        "magnitude_passed",
        "passed",
    ):
        assert key in r, f"missing field: {key}"
    assert r["envelope_geometry"] == "test fixture"


def test_sub_1_clears_stale_opposite_marker(tmp_path: Path) -> None:
    """Writing PASS removes a pre-existing FAIL (and vice versa)."""
    (tmp_path / "sub_1.FAIL").write_text("")
    history = _write_history(
        tmp_path,
        _sinusoidal_history(n_cycles=5, samples_per_cycle=200, amplitude=0.5),
    )
    rc = sub1.main(
        [
            "--history-csv",
            str(history),
            "--mass-kg",
            "0.05",
            "--omega-rad-per-s",
            "12.5664",
            "--r-cm-m",
            "0.1",
            "--result-json",
            str(tmp_path / "sub_1_result.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert (tmp_path / "sub_1.PASS").exists()
    assert not (tmp_path / "sub_1.FAIL").exists()


def test_sub_1_returns_2_when_history_missing(tmp_path: Path) -> None:
    rc = sub1.main(
        [
            "--history-csv",
            str(tmp_path / "no_such.csv"),
            "--mass-kg",
            "0.05",
            "--r-cm-m",
            "0.1",
            "--result-json",
            str(tmp_path / "sub_1_result.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2
    assert not (tmp_path / "sub_1.PASS").exists()
    assert not (tmp_path / "sub_1.FAIL").exists()
