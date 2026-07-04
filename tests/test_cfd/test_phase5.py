"""Tests for fanopt.cfd.phase5 (3D verification of top Pareto designs).

Geometry + gmsh meshing are real; the SU2 subprocess is mocked. Requires gmsh +
cadquery (blade geometry + 3D mesh).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

if importlib.util.find_spec("gmsh") is None:  # pragma: no cover - env-dependent
    pytest.skip("gmsh not installed", allow_module_level=True)
if importlib.util.find_spec("cadquery") is None:  # pragma: no cover - env-dependent
    pytest.skip("cadquery not installed", allow_module_level=True)

from fanopt.bo.codec import bounds, clip_to_bounds
from fanopt.cfd import phase3, phase5


def _mid_vector() -> np.ndarray:
    low, high = bounds()
    return clip_to_bounds((low + high) / 2.0)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _fake_su2_writing(series):
    def fake_run(cmd, cwd, stdout, stderr, env):
        lines = ["Time_Iter,CFx"] + [f"{t},{v}" for t, v in enumerate(series)]
        (Path(cwd) / "history.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

        class R:
            returncode = 0

        return R()

    return fake_run


# --- pure pieces ---


def test_extract_j_fan_3d_cycle_mean(tmp_path):
    # 3 cycles x 4 steps; per-cycle CFx = [10, 2, 4]; discard cycle 0 -> mean(2,4)=3.
    lines = ["Time_Iter,CFx"]
    for t, v in enumerate([10.0] * 4 + [2.0] * 4 + [4.0] * 4):
        lines.append(f"{t},{v}")
    h = _write(tmp_path / "u.csv", "\n".join(lines) + "\n")
    assert phase5.extract_j_fan_3d(h, n_cycles=3) == pytest.approx(3.0)


def test_extract_j_fan_3d_rejects_too_short(tmp_path):
    h = _write(tmp_path / "u.csv", "Time_Iter,CFx\n0,1.0\n")
    with pytest.raises(ValueError, match="too short"):
        phase5.extract_j_fan_3d(h, n_cycles=3)


def test_verify_ranking_preserved_when_correlated():
    res = [
        phase5.VerifyResult("a", j_fan_3d=1.0, j_fan_slice=1.0),
        phase5.VerifyResult("b", j_fan_3d=2.0, j_fan_slice=2.0),
        phase5.VerifyResult("c", j_fan_3d=3.0, j_fan_slice=3.0),
    ]
    out = phase5.verify_ranking(res)
    assert out["rank_preserved"] is True
    assert out["kendall_tau"] == pytest.approx(1.0)


def test_verify_ranking_broken_when_anticorrelated():
    res = [
        phase5.VerifyResult("a", j_fan_3d=3.0, j_fan_slice=1.0),
        phase5.VerifyResult("b", j_fan_3d=2.0, j_fan_slice=2.0),
        phase5.VerifyResult("c", j_fan_3d=1.0, j_fan_slice=3.0),
    ]
    assert phase5.verify_ranking(res)["rank_preserved"] is False


def test_verify_ranking_needs_two_points():
    res = [phase5.VerifyResult("a", j_fan_3d=1.0, j_fan_slice=1.0)]
    out = phase5.verify_ranking(res)
    assert out["rank_preserved"] is None
    assert out["n"] == 1


def test_verify_ranking_ignores_missing_slice():
    res = [
        phase5.VerifyResult("a", j_fan_3d=1.0, j_fan_slice=None),
        phase5.VerifyResult("b", j_fan_3d=2.0, j_fan_slice=2.0),
    ]
    assert phase5.verify_ranking(res)["n"] == 1


def test_verify_ranking_skips_failed_3d_runs():
    res = [
        phase5.VerifyResult("a", j_fan_3d=float("nan"), j_fan_slice=1.0),  # failed 3D
        phase5.VerifyResult("b", j_fan_3d=2.0, j_fan_slice=2.0),
        phase5.VerifyResult("c", j_fan_3d=3.0, j_fan_slice=3.0),
    ]
    assert phase5.verify_ranking(res)["n"] == 2  # the nan (failed) design skipped


def test_verify_ranking_reports_all_three_metrics():
    res = [
        phase5.VerifyResult("a", j_fan_3d=1.0, j_fan_slice=1.0),
        phase5.VerifyResult("b", j_fan_3d=2.0, j_fan_slice=2.0),
        phase5.VerifyResult("c", j_fan_3d=3.0, j_fan_slice=3.0),
    ]
    v = phase5.verify_ranking(res)["valid_only"]
    assert v["kendall_tau"] == pytest.approx(1.0)
    assert v["spearman_rho"] == pytest.approx(1.0)
    assert v["pearson_r2"] == pytest.approx(1.0)


def test_verify_ranking_flags_negative_jfan_as_suspect():
    res = [
        phase5.VerifyResult("a", j_fan_3d=1.0, j_fan_slice=1.0),
        phase5.VerifyResult("b", j_fan_3d=-5.0, j_fan_slice=2.0),  # net reverse thrust
    ]
    out = phase5.verify_ranking(res)
    assert out["n_suspect"] == 1
    assert out["suspect_designs"] == ["b"]
    assert out["valid_only"]["n"] == 1  # 'b' excluded from the valid set


def test_verify_ranking_exposes_ranking_that_only_holds_via_suspect():
    # slice ranks a>b (both positive); 3D inverts them; a degenerate negative
    # design 'c' is worst by both. all_finite τ is dragged positive by 'c', but
    # valid_only (excluding 'c') reveals the top-two ranking is actually broken.
    res = [
        phase5.VerifyResult("a", j_fan_3d=1.0, j_fan_slice=3.0),
        phase5.VerifyResult("b", j_fan_3d=2.0, j_fan_slice=2.0),
        phase5.VerifyResult("c", j_fan_3d=-9.0, j_fan_slice=1.0),  # degenerate worst
    ]
    out = phase5.verify_ranking(res)
    assert out["all_finite"]["kendall_tau"] > 0  # 'c' props it up
    assert out["valid_only"]["kendall_tau"] < 0  # honest: top-two inverted
    assert out["rank_preserved"] is False  # keys off valid_only


def test_verify_ranking_pairs_carry_suspect_flag():
    res = [
        phase5.VerifyResult("a", j_fan_3d=1.0, j_fan_slice=1.0),
        phase5.VerifyResult("b", j_fan_3d=float("nan"), j_fan_slice=2.0),
    ]
    pairs = {p["name"]: p for p in phase5.verify_ranking(res)["pairs"]}
    assert pairs["a"]["suspect"] is False
    assert pairs["b"]["suspect"] is True and pairs["b"]["j_fan_3d"] is None


def test_run_verification_penalizes_failed_design(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bad mesh")

    monkeypatch.setattr("fanopt.cfd.phase5.prepare_verification_case", boom)
    results = phase5.run_verification(
        [("d0", _mid_vector(), 1.0)], tmp_path, su2_bin="/fake/SU2_CFD"
    )
    assert np.isnan(results[0].j_fan_3d)
    assert results[0].meta.get("failed") == 1.0
    assert (tmp_path / "d0" / "FAILED.txt").exists()


# --- geometry → mesh → cfg (real) ---


def test_prepare_verification_case_builds_mesh_and_cfg(tmp_path):
    mesh = phase5.prepare_verification_case(_mid_vector(), tmp_path)
    assert mesh.n_nodes > 0
    assert (tmp_path / phase5.MESH_NAME).exists()
    cfg = (tmp_path / phase5.CFG_NAME).read_text()
    assert "PITCHING_OMEGA=" in cfg
    assert phase5.FAN_SURFACE_MARKER in cfg


# --- SU2-driving path (mocked subprocess) ---


def test_run_verification_errors_without_su2(tmp_path, monkeypatch):
    monkeypatch.delenv("SU2_RUN", raising=False)
    monkeypatch.setattr(phase3.shutil, "which", lambda _n: None)
    with pytest.raises(RuntimeError, match="SU2_CFD not found"):
        phase5.run_verification([("d0", _mid_vector(), 1.0)], tmp_path, su2_bin=None)


def test_run_verification_happy_path(tmp_path, monkeypatch):
    monkeypatch.setattr(phase3.subprocess, "run", _fake_su2_writing([6.0, -2.0] * 30))
    results = phase5.run_verification(
        [("d0", _mid_vector(), 1.5)], tmp_path, su2_bin="/fake/SU2_CFD"
    )
    assert len(results) == 1
    assert np.isfinite(results[0].j_fan_3d)
    assert results[0].j_fan_slice == 1.5
    assert results[0].meta["n_nodes"] > 0


def _make_fake_su2(tmp_path: Path) -> str:
    # A real executable (mocks don't cross process boundaries): writes history.csv.
    s = tmp_path / "fake_su2"
    s.write_text(
        "#!/bin/sh\n"
        "printf 'Time_Iter,CFx\\n' > history.csv\n"
        "i=0\n"
        "while [ $i -lt 120 ]; do printf '%d,5\\n' $i >> history.csv; i=$((i+1)); done\n"
    )
    s.chmod(0o755)
    return str(s)


def test_run_verification_progress_bar(tmp_path, monkeypatch):
    # progress=True must not change results (bar is display-only).
    monkeypatch.setattr(phase3.subprocess, "run", _fake_su2_writing([1.0, -1.0] * 30))
    results = phase5.run_verification(
        [("d0", _mid_vector(), 1.0)], tmp_path, su2_bin="/fake/SU2_CFD", progress=True
    )
    assert len(results) == 1


def test_run_verification_parallel_preserves_order(tmp_path):
    su2 = _make_fake_su2(tmp_path)
    designs = [("d0", _mid_vector(), 1.0), ("d1", _mid_vector(), 2.0)]
    results = phase5.run_verification(
        designs, tmp_path / "out", su2_bin=su2, n_workers=2, progress=True
    )
    assert [r.name for r in results] == ["d0", "d1"]  # order preserved across processes
    assert all(np.isfinite(r.j_fan_3d) for r in results)
    assert [r.j_fan_slice for r in results] == [1.0, 2.0]
