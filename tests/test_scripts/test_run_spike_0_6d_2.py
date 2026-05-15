"""CLI smoke tests for ``scripts/run_spike_0_6d_2.py``."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import run_spike_0_6d_2 as sub2
from fanopt.cfd.spike_0_6d import compute_added_mass_moment_closed_form_2d_plate


def _moment_history(*, peak_moment: float, n_cycles: int = 5, samples_per_cycle: int = 200) -> str:
    out = io.StringIO()
    out.write("Time_Iter,Inner_Iter,CMz,CD\n")
    total = n_cycles * samples_per_cycle
    for i in range(total):
        phase = 2.0 * math.pi * (i % samples_per_cycle) / samples_per_cycle
        cmz = peak_moment * math.sin(phase)
        out.write(f"{i},0,{cmz:.6e},0.05\n")
    return out.getvalue()


def _write_history(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "history.csv"
    p.write_text(text)
    return p


def test_sub_2_passes_when_su2_within_15pct_of_closed_form(tmp_path: Path) -> None:
    closed = compute_added_mass_moment_closed_form_2d_plate(
        chord_m=1.0,
        pivot_offset_normalized=-0.5,
        pitching_omega_rad_per_s=10.0,
        pitching_amplitude_rad=0.1,
    )
    # Synthetic SU2 output: peak moment = +10% of closed form (within ±15%).
    history = _write_history(tmp_path, _moment_history(peak_moment=closed * 1.10))
    rc = sub2.main(
        [
            "--history-csv",
            str(history),
            "--chord-m",
            "1.0",
            "--pitching-omega-rad-per-s",
            "10.0",
            "--pitching-amplitude-rad",
            "0.1",
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


def test_sub_2_fails_outside_15pct(tmp_path: Path) -> None:
    closed = compute_added_mass_moment_closed_form_2d_plate(
        chord_m=1.0,
        pivot_offset_normalized=-0.5,
        pitching_omega_rad_per_s=10.0,
        pitching_amplitude_rad=0.1,
    )
    # +40% off — well outside ±15%.
    history = _write_history(tmp_path, _moment_history(peak_moment=closed * 1.40))
    rc = sub2.main(
        [
            "--history-csv",
            str(history),
            "--chord-m",
            "1.0",
            "--pitching-omega-rad-per-s",
            "10.0",
            "--pitching-amplitude-rad",
            "0.1",
            "--result-json",
            str(tmp_path / "sub_2_result.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 1
    assert (tmp_path / "sub_2.FAIL").exists()


def test_sub_2_returns_2_when_history_missing(tmp_path: Path) -> None:
    rc = sub2.main(
        [
            "--history-csv",
            str(tmp_path / "no_such.csv"),
            "--chord-m",
            "1.0",
            "--pitching-omega-rad-per-s",
            "10.0",
            "--pitching-amplitude-rad",
            "0.1",
            "--result-json",
            str(tmp_path / "sub_2_result.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2


def test_sub_2_returns_2_when_no_moment_column(tmp_path: Path) -> None:
    """A history.csv with no CMy/CMz/CM column → input error."""
    p = tmp_path / "history.csv"
    p.write_text("Time_Iter,Inner_Iter,CD\n0,0,0.05\n1,0,0.05\n")
    rc = sub2.main(
        [
            "--history-csv",
            str(p),
            "--chord-m",
            "1.0",
            "--pitching-omega-rad-per-s",
            "10.0",
            "--pitching-amplitude-rad",
            "0.1",
            "--result-json",
            str(tmp_path / "sub_2_result.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2


def test_sub_2_clears_stale_opposite_marker(tmp_path: Path) -> None:
    """Pre-existing FAIL → after PASS run, FAIL removed."""
    (tmp_path / "sub_2.FAIL").write_text("")
    closed = compute_added_mass_moment_closed_form_2d_plate(
        chord_m=1.0,
        pivot_offset_normalized=-0.5,
        pitching_omega_rad_per_s=10.0,
        pitching_amplitude_rad=0.1,
    )
    history = _write_history(tmp_path, _moment_history(peak_moment=closed * 1.05))
    rc = sub2.main(
        [
            "--history-csv",
            str(history),
            "--chord-m",
            "1.0",
            "--pitching-omega-rad-per-s",
            "10.0",
            "--pitching-amplitude-rad",
            "0.1",
            "--result-json",
            str(tmp_path / "sub_2_result.json"),
            "--marker-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert (tmp_path / "sub_2.PASS").exists()
    assert not (tmp_path / "sub_2.FAIL").exists()
