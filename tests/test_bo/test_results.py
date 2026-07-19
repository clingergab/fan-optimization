"""Tests for fanopt.bo.results (campaign → Pareto + top-k diverse). Botorch-free."""

from __future__ import annotations

import json

import numpy as np

from fanopt.bo import results
from fanopt.bo.codec import encode
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


def _fake_campaign(tmp_path, x, y_raw, *, ledger=True):
    np.savez(tmp_path / results.CHECKPOINT_NAME, x=x, y_raw=y_raw, iteration=1)
    if ledger:
        lines = [json.dumps({"J_fan": float(y[0]), "I_wrist_kgm2": float(y[1])}) for y in y_raw]
        (tmp_path / results.LEDGER_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


# --- non-dominated sort ---


def test_non_dominated_mask_basic():
    y_max = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-1.0, -1.0]])
    mask = results.non_dominated_mask(y_max)
    assert mask[0] and mask[1]
    assert not mask[3]  # dominated by all


def test_non_dominated_single_point():
    assert results.non_dominated_mask(np.array([[1.0, 2.0, 3.0]])).tolist() == [True]


# --- pareto_designs ---


def test_pareto_designs_decodes_and_sorts_by_jfan():
    x = np.array([_vec(8, 0.0022), _vec(10, 0.003), _vec(12, 0.0038)])
    # obj = (J_fan↑, I_wrist↓, structural↓). Rows 0 and 1 non-dominated; row 2 dominated.
    y_raw = np.array([[2.0, 0.004, 0.001], [1.0, 0.003, 0.0009], [0.5, 0.009, 0.002]])
    pf = results.pareto_designs(x, y_raw)
    assert [d["index"] for d in pf] == [0, 1]  # J_fan descending
    assert pf[0]["blade_count"] == 8
    assert "edge_profile" in pf[0]


# --- diversity selection ---


def test_select_diverse_returns_all_when_few():
    x = np.array([_vec(8, 0.0022), _vec(10, 0.003)])
    assert set(results.select_diverse(x, [0, 1], 3)) == {0, 1}


def test_select_diverse_spreads():
    # three thin + one thick; picking 2 must include the outlier thick design.
    x = np.array([_vec(10, 0.00221), _vec(10, 0.00222), _vec(10, 0.00223), _vec(10, 0.0038)])
    picked = results.select_diverse(x, [0, 1, 2, 3], 2)
    assert 3 in picked
    assert len(picked) == 2


def test_select_diverse_empty():
    assert results.select_diverse(np.zeros((0, 35)), [], 3) == []


# --- load + analyze ---


def test_load_campaign_reads_checkpoint_and_ledger(tmp_path):
    x = np.array([_vec(8, 0.0022), _vec(10, 0.003)])
    y_raw = np.array([[1.0, 0.005, 0.001], [2.0, 0.007, 0.0008]])
    _fake_campaign(tmp_path, x, y_raw)
    data = results.load_campaign(tmp_path)
    assert data.x.shape == (2, 35)
    assert len(data.ledger_rows) == 2


def test_load_campaign_without_ledger(tmp_path):
    x = np.array([_vec(8, 0.0022)])
    y_raw = np.array([[1.0, 0.005, 0.001]])
    _fake_campaign(tmp_path, x, y_raw, ledger=False)
    data = results.load_campaign(tmp_path)
    assert data.ledger_rows == []


def test_analyze_flags_diverse_picks(tmp_path):
    x = np.array([_vec(8, 0.0022), _vec(10, 0.003), _vec(12, 0.0038)])
    y_raw = np.array([[2.0, 0.004, 0.001], [1.5, 0.005, 0.0009], [1.0, 0.006, 0.0008]])
    _fake_campaign(tmp_path, x, y_raw)
    summary = results.analyze(tmp_path, top_k=2)
    assert summary["n_evaluations"] == 3
    assert summary["n_pareto"] == 3  # all non-dominated (a proper tradeoff)
    assert len(summary["top_k_diverse"]) == 2
    assert all(d["diverse_pick"] for d in summary["top_k_diverse"])


def _campaign_three(tmp_path):
    x = np.array([_vec(8, 0.0022), _vec(10, 0.003), _vec(12, 0.0038)])
    y_raw = np.array([[2.0, 0.004, 0.001], [1.5, 0.005, 0.0009], [1.0, 0.006, 0.0008]])
    return _fake_campaign(tmp_path, x, y_raw)


def test_recommend_without_verification(tmp_path):
    _campaign_three(tmp_path)
    out = results.recommend(tmp_path, top_k=3)
    assert out["verification"] == "absent"
    assert out["n_verified"] == 0
    assert len(out["recommended"]) == 3
    assert all(d["j_fan_3d"] is None and d["verified"] is False for d in out["recommended"])


def test_recommend_with_verification(tmp_path):
    _campaign_three(tmp_path)
    ver = {
        "ranking": {"rank_preserved": True},
        "designs": [
            {"name": "b8_i0", "j_fan_3d": 1.8, "j_fan_slice": 2.0},
            {"name": "b10_i1", "j_fan_3d": 1.3, "j_fan_slice": 1.5},
        ],
    }
    vp = tmp_path / "verification.json"
    vp.write_text(json.dumps(ver), encoding="utf-8")
    out = results.recommend(tmp_path, top_k=3, verification_path=vp)
    assert out["verification"] == "present"
    assert out["n_verified"] == 2  # indices 0, 1 verified; 2 not
    verified = {d["index"] for d in out["recommended"] if d["verified"]}
    assert verified == {0, 1}


def test_recommend_ignores_missing_verification_file(tmp_path):
    _campaign_three(tmp_path)
    out = results.recommend(tmp_path, top_k=3, verification_path=tmp_path / "nope.json")
    assert out["verification"] == "absent"


def test_recommend_ranked_orders_all_verified_by_3d_jfan(tmp_path):
    _campaign_three(tmp_path)
    ver = {
        "ranking": {},
        "designs": [
            {"name": "b8_i0", "j_fan_3d": 1.8, "j_fan_slice": 2.0},
            {"name": "b10_i1", "j_fan_3d": 3.3, "j_fan_slice": 1.5},  # best in 3D, 2nd in 2D
            {"name": "b12_i2", "j_fan_3d": -5.0, "j_fan_slice": 1.0},  # negative → suspect
        ],
    }
    vp = tmp_path / "verification.json"
    vp.write_text(json.dumps(ver), encoding="utf-8")
    ranked = results.recommend(tmp_path, top_k=3, verification_path=vp)["ranked"]

    assert [d["index"] for d in ranked] == [1, 0, 2]  # 3.3 > 1.8 > -5.0 (suspect last)
    assert ranked[0]["verified"] is True and ranked[0]["blade_count"] == 10
    assert ranked[-1]["index"] == 2 and ranked[-1]["suspect"] is True
    assert ranked[-1]["verified"] is False


def test_recommend_ranked_sinks_failed_3d_runs_to_bottom(tmp_path):
    _campaign_three(tmp_path)
    ver = {
        "ranking": {},
        "designs": [
            {"name": "b8_i0", "j_fan_3d": None, "j_fan_slice": 2.0},  # failed 3D run
            {"name": "b10_i1", "j_fan_3d": 1.3, "j_fan_slice": 1.5},
        ],
    }
    vp = tmp_path / "verification.json"
    vp.write_text(json.dumps(ver), encoding="utf-8")
    ranked = results.recommend(tmp_path, top_k=3, verification_path=vp)["ranked"]
    assert ranked[0]["index"] == 1  # finite 3D J_fan ranks above the failed run
    assert ranked[-1]["index"] == 0 and ranked[-1]["j_fan_3d"] is None
    assert ranked[-1]["suspect"] is True
