"""Tests for scripts/recommend_designs.py (Pareto + 3D verification → print list)."""

from __future__ import annotations

import json

import numpy as np

import recommend_designs as script
from fanopt.bo.codec import encode
from fanopt.bo.results import CHECKPOINT_NAME
from fanopt.geometry.envelope import Layer1Params, ThicknessGridField


def _vec(blade_count: int, thickness: float) -> np.ndarray:
    return encode(
        Layer1Params(
            blade_count=blade_count,
            camber_knots_m=(0.001, 0.001, 0.001),
            twist_knots_rad=(0.0, 0.0),
            thickness_field=ThicknessGridField.uniform(thickness),
            edge_profile="rounded",
            fourier_le_amplitudes=(0.0, 0.0, 0.0),
            fourier_te_amplitudes=(0.0, 0.0, 0.0),
        )
    )


def _fake_campaign(tmp_path):
    camp = tmp_path / "camp"
    camp.mkdir()
    x = np.array([_vec(8, 0.0022), _vec(10, 0.003), _vec(12, 0.0038)])
    y_raw = np.array([[2.0, 0.004, 0.001], [1.5, 0.005, 0.0009], [1.0, 0.006, 0.0008]])
    np.savez(camp / CHECKPOINT_NAME, x=x, y_raw=y_raw, iteration=1)
    return camp


def test_main_writes_recommended_json_without_verification(tmp_path):
    camp = _fake_campaign(tmp_path)
    rc = script.main(
        ["--campaign-dir", str(camp), "--out-dir", str(tmp_path / "out"), "--top-k", "3"]
    )
    assert rc == 0
    rec = json.loads((tmp_path / "out" / "recommended.json").read_text())
    assert rec["verification"] == "absent"
    assert len(rec["recommended"]) == 3
    assert all(d["j_fan_3d"] is None for d in rec["recommended"])


def test_main_merges_verification(tmp_path):
    camp = _fake_campaign(tmp_path)
    ver = tmp_path / "verification.json"
    ver.write_text(
        json.dumps(
            {
                "ranking": {"rank_preserved": True},
                "designs": [{"name": "b8_i0", "j_fan_3d": 1.9, "j_fan_slice": 2.0}],
            }
        ),
        encoding="utf-8",
    )
    summary = script.run(
        campaign_dir=camp, out_dir=tmp_path / "out", top_k=3, verification_path=ver
    )
    assert summary["verification"] == "present"
    assert summary["n_verified"] == 1
