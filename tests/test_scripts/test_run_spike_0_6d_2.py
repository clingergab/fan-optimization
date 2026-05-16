"""CLI smoke tests for ``scripts/run_spike_0_6d_2.py`` (two-frequency
added-mass frequency-consistency gate, 2026-05-15 redesign)."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import pytest

import run_spike_0_6d_2 as sub2


def _moment_history(
    *, omega: float, theta_max: float, ia_nondim: float, n_cycles: int = 5, spc: int = 200
) -> str:
    out = io.StringIO()
    out.write("Time_Iter,Inner_Iter,CMz,CD\n")
    k_am = ia_nondim * omega**2 * theta_max
    for i in range(n_cycles * spc):
        phi = 2.0 * math.pi * (i % spc) / spc
        out.write(f"{i},0,{k_am * math.sin(phi):.8e},0.05\n")
    return out.getvalue()


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def test_sub_2_passes_when_two_frequencies_consistent(tmp_path: Path) -> None:
    th = 0.1745
    f1 = _write(tmp_path, "f1.csv", _moment_history(omega=6.2832, theta_max=th, ia_nondim=0.037))
    f2 = _write(tmp_path, "f2.csv", _moment_history(omega=12.5664, theta_max=th, ia_nondim=0.037))
    rc = sub2.main(
        [
            "--history-csv-f1",
            str(f1),
            "--omega-f1",
            "6.2832",
            "--history-csv-f2",
            str(f2),
            "--omega-f2",
            "12.5664",
            "--pitching-amplitude-rad",
            str(th),
            "--chord-m",
            "1.0",
            "--result-json",
            str(tmp_path / "sub_2_result.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert (tmp_path / "sub_2.PASS").exists()
    payload = json.loads((tmp_path / "sub_2_result.json").read_text())
    assert payload["result"]["passed"] is True
    assert payload["result"]["freq_consistency_passed"] is True


def test_sub_2_fails_when_frequency_inconsistent(tmp_path: Path) -> None:
    """Frequency-dependent recovered I_a (numerical-distortion proxy) → FAIL."""
    th = 0.1745
    f1 = _write(tmp_path, "f1.csv", _moment_history(omega=6.2832, theta_max=th, ia_nondim=0.037))
    f2 = _write(tmp_path, "f2.csv", _moment_history(omega=12.5664, theta_max=th, ia_nondim=0.10))
    rc = sub2.main(
        [
            "--history-csv-f1",
            str(f1),
            "--omega-f1",
            "6.2832",
            "--history-csv-f2",
            str(f2),
            "--omega-f2",
            "12.5664",
            "--pitching-amplitude-rad",
            str(th),
            "--chord-m",
            "1.0",
            "--result-json",
            str(tmp_path / "sub_2_result.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 1
    assert (tmp_path / "sub_2.FAIL").exists()


def test_sub_2_returns_2_when_history_missing(tmp_path: Path) -> None:
    f1 = _write(tmp_path, "f1.csv", _moment_history(omega=6.2832, theta_max=0.1, ia_nondim=0.037))
    rc = sub2.main(
        [
            "--history-csv-f1",
            str(f1),
            "--omega-f1",
            "6.2832",
            "--history-csv-f2",
            str(tmp_path / "nope.csv"),
            "--omega-f2",
            "12.5664",
            "--pitching-amplitude-rad",
            "0.1",
            "--chord-m",
            "1.0",
            "--result-json",
            str(tmp_path / "sub_2_result.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2


def test_sub_2_returns_2_when_no_moment_column(tmp_path: Path) -> None:
    """history.csv with no CMy/CMz/CM column → input error (zero projection)."""
    no_moment = "Time_Iter,Inner_Iter,CD\n" + "\n".join(f"{i},0,0.05" for i in range(1000)) + "\n"
    f1 = _write(tmp_path, "f1.csv", no_moment)
    f2 = _write(tmp_path, "f2.csv", no_moment)
    rc = sub2.main(
        [
            "--history-csv-f1",
            str(f1),
            "--omega-f1",
            "6.2832",
            "--history-csv-f2",
            str(f2),
            "--omega-f2",
            "12.5664",
            "--pitching-amplitude-rad",
            "0.1",
            "--chord-m",
            "1.0",
            "--result-json",
            str(tmp_path / "sub_2_result.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2


def test_sub_2_clears_stale_opposite_marker(tmp_path: Path) -> None:
    (tmp_path / "sub_2.FAIL").write_text("")
    th = 0.1745
    f1 = _write(tmp_path, "f1.csv", _moment_history(omega=6.2832, theta_max=th, ia_nondim=0.037))
    f2 = _write(tmp_path, "f2.csv", _moment_history(omega=12.5664, theta_max=th, ia_nondim=0.037))
    rc = sub2.main(
        [
            "--history-csv-f1",
            str(f1),
            "--omega-f1",
            "6.2832",
            "--history-csv-f2",
            str(f2),
            "--omega-f2",
            "12.5664",
            "--pitching-amplitude-rad",
            str(th),
            "--chord-m",
            "1.0",
            "--result-json",
            str(tmp_path / "sub_2_result.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert (tmp_path / "sub_2.PASS").exists()
    assert not (tmp_path / "sub_2.FAIL").exists()


def test_sub_2_payload_records_both_projections(tmp_path: Path) -> None:
    """The result JSON carries both per-frequency projections for Phase 5."""
    th = 0.1745
    f1 = _write(tmp_path, "f1.csv", _moment_history(omega=6.2832, theta_max=th, ia_nondim=0.037))
    f2 = _write(tmp_path, "f2.csv", _moment_history(omega=12.5664, theta_max=th, ia_nondim=0.037))
    sub2.main(
        [
            "--history-csv-f1",
            str(f1),
            "--omega-f1",
            "6.2832",
            "--history-csv-f2",
            str(f2),
            "--omega-f2",
            "12.5664",
            "--pitching-amplitude-rad",
            str(th),
            "--chord-m",
            "1.0",
            "--result-json",
            str(tmp_path / "sub_2_result.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    payload = json.loads((tmp_path / "sub_2_result.json").read_text())
    assert "projection_f1" in payload
    assert "projection_f2" in payload
    ia1 = payload["projection_f1"]["recovered_ia_nondim"]
    ia2 = payload["projection_f2"]["recovered_ia_nondim"]
    # Both runs planted the same I_a (0.037) → recovered values agree.
    assert ia1 == pytest.approx(0.037, rel=1e-2)
    assert ia2 == pytest.approx(0.037, rel=1e-2)
