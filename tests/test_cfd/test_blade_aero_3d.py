"""Tests for the 3D objective aero evaluation (CFz thrust, both reductions, divergence guard)."""

from __future__ import annotations

from pathlib import Path

import pytest

from fanopt.cfd import blade_aero_3d as a3d
from fanopt.cfd.blade_aero_3d import evaluate_blade_aero_3d
from fanopt.cfd.phase5 import VerifyConfig
from fanopt.geometry.blade import BladeParams


class _FakeMesh:
    n_nodes = 1234


def _gentle() -> BladeParams:
    return BladeParams.from_dict(
        {
            "blade_count": 8,
            "rib_bow_knots_m": [0.006, 0.010, 0.012, 0.010, 0.006],
            "rib_bow_interp": "linear",
            "t_rib_hub_m": 0.005,
            "t_rib_tip_m": 0.005,
            "panel_offsets_m": [[0.0, 0.0, 0.0]] * 4,
            "panel_thickness_m": [[0.003, 0.003, 0.003]] * 4,
        }
    )


def _hist_writer(series, *, cfx_decoy=-999.0):
    # BOTH columns present (as a real SU2 history) with a DECOY CFx — so the test only passes
    # if the code reads CFz (thrust). A single-CFz-column history would resolve to CFz under the
    # buggy CFx-first default too, silently failing to guard the B1 fix.
    def fake_run(cfg_name, workdir, su2):
        lines = ["Time_Iter,CFx,CFz"] + [f"{t},{cfx_decoy},{v}" for t, v in enumerate(series)]
        p = Path(workdir) / "history.csv"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p
    return fake_run


def test_extracts_cycle_mean_and_peak(tmp_path, monkeypatch):
    monkeypatch.setattr(a3d, "prepare_blade_verification_case", lambda p, wd, cfg: _FakeMesh())
    # 3 cycles × 200 steps; per-cycle CFz = [5, 7, 9]; discard cycle 0 → mean(7,9)=8.
    monkeypatch.setattr(a3d, "run_su2", _hist_writer([5.0] * 200 + [7.0] * 200 + [9.0] * 200))
    res = evaluate_blade_aero_3d(_gentle(), tmp_path, cfg=VerifyConfig(n_cycles=3), su2_bin="/fake")
    assert res.j_fan_mean == pytest.approx(8.0)
    assert res.n_nodes == 1234.0
    assert res.j_fan_peak == res.j_fan_peak  # finite (not NaN)


def test_reads_cfz_not_cfx(tmp_path, monkeypatch):
    # Audit B1: the in-loop scorer must read CFz (thrust), not the parser's CFx-first default.
    # CFx carries a large decoy; CFz carries per-cycle [5,7,9] → discard cycle 0 → mean 8.0.
    monkeypatch.setattr(a3d, "prepare_blade_verification_case", lambda p, wd, cfg: _FakeMesh())
    monkeypatch.setattr(
        a3d, "run_su2", _hist_writer([5.0] * 200 + [7.0] * 200 + [9.0] * 200, cfx_decoy=-999.0)
    )
    res = evaluate_blade_aero_3d(_gentle(), tmp_path, cfg=VerifyConfig(n_cycles=3), su2_bin="/fake")
    assert res.j_fan_mean == pytest.approx(8.0)  # CFz, not the -999 CFx decoy


def test_rejects_short_diverged_run(tmp_path, monkeypatch):
    monkeypatch.setattr(a3d, "prepare_blade_verification_case", lambda p, wd, cfg: _FakeMesh())
    monkeypatch.setattr(a3d, "run_su2", _hist_writer([1.0] * 100))  # < 3×200
    with pytest.raises(ValueError, match="incomplete or diverged"):
        evaluate_blade_aero_3d(_gentle(), tmp_path, cfg=VerifyConfig(n_cycles=3), su2_bin="/fake")


def test_requires_su2(tmp_path, monkeypatch):
    monkeypatch.setattr(a3d, "find_su2", lambda: None)
    with pytest.raises(RuntimeError, match="SU2_CFD not found"):
        evaluate_blade_aero_3d(_gentle(), tmp_path)  # no su2_bin and find_su2 → None
