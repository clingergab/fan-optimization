"""Tests for the Phase 3 orchestration (report-final.md §Phase 3).

Skips at module load where ``gmsh`` is absent (phase3 imports mesh_2d_slice).
The SU2-running paths (``run_su2``, ``run_correlation_sweep``) are exercised by
the local integration run, not unit tests.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

if importlib.util.find_spec("gmsh") is None:  # pragma: no cover - env-dependent
    pytest.skip("gmsh not installed", allow_module_level=True)

from fanopt.cfd import phase3


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_sweep_designs_are_distinct():
    names = [d.name for d in phase3.sweep_designs()]
    assert len(names) == len(set(names))
    assert len(names) >= 4  # enough points to correlate


def test_extract_steady_drag_reads_cd(tmp_path):
    h = _write(tmp_path / "steady.csv", "Inner_Iter,CL,CD\n0,0.1,0.0\n1,0.1,7.5\n")
    assert phase3.extract_steady_drag(h) == pytest.approx(7.5)


def test_extract_unsteady_mean_cycle_averages(tmp_path):
    # 3 cycles × 4 steps; per-cycle CFx = [10, 2, 4]; discard cycle 0 → mean(2,4)=3.
    lines = ["Time_Iter,CFx"]
    for t, cyc in enumerate([10.0] * 4 + [2.0] * 4 + [4.0] * 4):
        lines.append(f"{t},{cyc}")
    h = _write(tmp_path / "unsteady.csv", "\n".join(lines) + "\n")
    assert phase3.extract_unsteady_mean(h, n_cycles=3) == pytest.approx(3.0)


def test_extract_unsteady_rms_amplitude(tmp_path):
    # 3 cycles × 4 steps; values ±10 → cycle-mean ~0 but RMS = 10. Discard cyc 0.
    vals = [10.0, -10.0, 10.0, -10.0] * 3
    lines = ["Time_Iter,CFx"] + [f"{t},{v}" for t, v in enumerate(vals)]
    h = _write(tmp_path / "u.csv", "\n".join(lines) + "\n")
    assert phase3.extract_unsteady_mean(h, n_cycles=3) == pytest.approx(0.0)
    assert phase3.extract_unsteady_rms(h, n_cycles=3) == pytest.approx(10.0)


def test_extract_unsteady_rms_rejects_too_short(tmp_path):
    h = _write(tmp_path / "u.csv", "Time_Iter,CFx\n0,1.0\n")
    with pytest.raises(ValueError, match="too short"):
        phase3.extract_unsteady_rms(h, n_cycles=3)


def test_prepare_design_case_writes_mesh_and_both_cfgs(tmp_path):
    d = phase3.sweep_designs()[0]
    info = phase3.prepare_design_case(d, tmp_path)
    assert Path(info["mesh"]).exists()
    assert (tmp_path / "steady.cfg").exists()
    assert (tmp_path / "unsteady.cfg").exists()
    assert info["n_nodes"] > 0


def test_unsteady_cfg_is_plunging_unsteady(tmp_path):
    phase3.prepare_design_case(phase3.sweep_designs()[0], tmp_path)
    cfg = (tmp_path / "unsteady.cfg").read_text()
    assert "PLUNGING_OMEGA=" in cfg
    assert "DUAL_TIME_STEPPING-2ND_ORDER" in cfg
    assert "MACH_NUMBER= 1e-9" in cfg


def test_find_su2_returns_path_or_none():
    # Just must not raise; returns a str path or None depending on env.
    result = phase3.find_su2()
    assert result is None or isinstance(result, str)


def test_run_correlation_sweep_errors_without_su2(tmp_path, monkeypatch):
    monkeypatch.delenv("SU2_RUN", raising=False)
    monkeypatch.setattr(phase3.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="SU2_CFD not found"):
        phase3.run_correlation_sweep(tmp_path, designs=phase3.sweep_designs()[:1])


def test_find_su2_from_su2_run_env(tmp_path, monkeypatch):
    (tmp_path / "SU2_CFD").write_text("#!/bin/sh\n")
    monkeypatch.setenv("SU2_RUN", str(tmp_path))
    assert phase3.find_su2() == str(tmp_path / "SU2_CFD")


def test_extract_unsteady_mean_rejects_too_short(tmp_path):
    h = tmp_path / "u.csv"
    h.write_text("Time_Iter,CFx\n0,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="too short"):
        phase3.extract_unsteady_mean(h, n_cycles=3)


def test_run_su2_success_returns_history(tmp_path, monkeypatch):
    # Mock the SU2 subprocess (external boundary) + drop a history.csv.
    def fake_run(cmd, cwd, stdout, stderr, env):
        (tmp_path / "history.csv").write_text("Time_Iter,CD\n0,1.0\n", encoding="utf-8")

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(phase3.subprocess, "run", fake_run)
    hist = phase3.run_su2("x.cfg", tmp_path, "/fake/SU2_CFD")
    assert hist.name == "history.csv"


def test_run_su2_failure_raises_with_log(tmp_path, monkeypatch):
    def fake_run(cmd, cwd, stdout, stderr, env):
        stdout.write("SU2 boom: invalid option\n")

        class R:
            returncode = 1

        return R()

    monkeypatch.setattr(phase3.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="SU2 failed"):
        phase3.run_su2("x.cfg", tmp_path, "/fake/SU2_CFD")
