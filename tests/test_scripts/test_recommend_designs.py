"""Tests for scripts/recommend_designs.py (Pareto + 3D verification → print list)."""

from __future__ import annotations

import json

import numpy as np
import pytest

import recommend_designs as script
from fanopt.bo.blade_codec import bounds as blade_bounds
from fanopt.bo.blade_codec import clip_to_bounds
from fanopt.bo.blade_codec import decode as blade_decode
from fanopt.bo.codec import encode
from fanopt.bo.results import CHECKPOINT_NAME
from fanopt.geometry.envelope import Layer1Params, ThicknessGridField
from fanopt.utils.ledger import design_hash


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


def _bvec(frac: float) -> np.ndarray:
    low, high = blade_bounds()
    return clip_to_bounds(low + (high - low) * frac)


def _bhash(vec: np.ndarray) -> str:
    return design_hash(blade_decode(vec).to_dict())


def _fake_shards(tmp_path):
    shared = tmp_path / "campaign_trapezoid"
    shared.mkdir()
    rows = [  # distinct mass so both trade off → both Pareto
        {"design_hash": _bhash(_bvec(f)), "vector": _bvec(f).tolist(), "j_fan": j,
         "mass_kg": m, "deflection_m": 1e-4, "blade_count": blade_decode(_bvec(f)).blade_count}
        for f, j, m in [(0.3, 2.0, 0.004), (0.6, 1.5, 0.003)]
    ]
    (shared / "evaluations_colab-0.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return shared


def test_main_shared_dir_writes_recommended_json(tmp_path):
    shared = _fake_shards(tmp_path)
    rc = script.main(
        ["--shared-dir", str(shared), "--out-dir", str(tmp_path / "out"), "--top-k", "3"]
    )
    assert rc == 0
    rec = json.loads((tmp_path / "out" / "recommended.json").read_text())
    assert rec["verification"] == "absent"
    assert rec["n_pareto"] == 2 and len(rec["recommended"]) == 2


def test_main_shared_dir_merges_verification_by_hash(tmp_path):
    shared = _fake_shards(tmp_path)
    ver = tmp_path / "verification.json"
    ver.write_text(
        json.dumps({"ranking": {"rank_preserved": True}, "designs": [
            {"name": f"00_{_bhash(_bvec(0.3))}", "j_fan_3d": 1.9e12, "j_fan_coarse": 2.0}]}),
        encoding="utf-8",
    )
    summary = script.run(
        shared_dir=shared, out_dir=tmp_path / "out", top_k=3, verification_path=ver
    )
    assert summary["verification"] == "present"
    assert summary["n_verified"] == 1


def test_main_requires_a_campaign_source(tmp_path):
    with pytest.raises(SystemExit):  # neither --shared-dir nor --campaign-dir
        script.main(["--out-dir", str(tmp_path / "out")])


def test_main_merges_verification(tmp_path):
    camp = _fake_campaign(tmp_path)
    ver = tmp_path / "verification.json"
    ver.write_text(
        json.dumps(
            {
                "ranking": {"rank_preserved": True},
                "designs": [{"name": "b8_i0", "j_fan_3d": 1.9, "j_fan_coarse": 2.0}],
            }
        ),
        encoding="utf-8",
    )
    summary = script.run(
        campaign_dir=camp, out_dir=tmp_path / "out", top_k=3, verification_path=ver
    )
    assert summary["verification"] == "present"
    assert summary["n_verified"] == 1
