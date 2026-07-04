"""Tests for scripts/run_phase6_physical.py (recommended.json + measurements → report)."""

from __future__ import annotations

import json

import run_phase6_physical as script
from fanopt.physical.anemometer import GRID_POINTS_M


def _recommended(tmp_path):
    rec = {
        "recommended": [
            {"index": 0, "blade_count": 8, "i_wrist_kgm2": 0.008, "j_fan_3d": 1.1e10},
            {"index": 1, "blade_count": 10, "i_wrist_kgm2": 0.009, "j_fan_3d": 0.7e10},
        ]
    }
    p = tmp_path / "recommended.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    return p


def _anemometer(d_dir, v):
    header = "point,x_m,y_m,z_m,v_mean_m_per_s,v_peak_m_per_s,notes"
    rows = [f"p{i + 1},{x},{y},0.3,{v},{v * 2},''" for i, (x, y) in enumerate(GRID_POINTS_M)]
    (d_dir / "anemometer.csv").write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


def test_run_writes_report_and_ranks(tmp_path):
    rec = _recommended(tmp_path)
    meas = tmp_path / "meas"
    # Only design b8_i0 gets an anemometer reading; one measured design → rank None.
    (meas / "b8_i0").mkdir(parents=True)
    _anemometer(meas / "b8_i0", v=0.6)

    report = script.run(recommended_path=rec, measurements_dir=meas, out_dir=tmp_path / "out")
    assert (tmp_path / "out" / "physical_results.json").exists()
    assert report["n_designs"] == 2
    assert report["n_with_anemometer"] == 1
    assert report["j_fan_rank"]["rank_preserved"] is None  # <2 paired


def test_run_ranks_when_both_measured(tmp_path):
    rec = _recommended(tmp_path)
    meas = tmp_path / "meas"
    for name, v in (("b8_i0", 0.6), ("b10_i1", 0.4)):  # b8 predicted higher + measured higher
        (meas / name).mkdir(parents=True)
        _anemometer(meas / name, v=v)
    report = script.run(recommended_path=rec, measurements_dir=meas, out_dir=tmp_path / "out")
    assert report["n_with_anemometer"] == 2
    assert report["j_fan_rank"]["rank_preserved"] is True


def test_main_returns_zero(tmp_path):
    rec = _recommended(tmp_path)
    rc = script.main(
        ["--recommended", str(rec), "--measurements", str(tmp_path / "none"),
         "--out-dir", str(tmp_path / "out")]
    )
    assert rc == 0
    report = json.loads((tmp_path / "out" / "physical_results.json").read_text())
    assert report["n_with_imu"] == 0  # no measurements dir → all pending
