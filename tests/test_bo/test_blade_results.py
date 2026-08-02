"""Tests for fanopt.bo.blade_results (Stage-3 shard campaign → Pareto + V1-pick consolidator).

Pure numpy + codec — cadquery/gmsh-free (the consolidator runs anywhere the shards land)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fanopt.bo import blade_results
from fanopt.bo.blade_codec import bounds, clip_to_bounds, decode
from fanopt.utils.ledger import design_hash


def _vec(frac: float) -> np.ndarray:
    low, high = bounds()
    return clip_to_bounds(low + (high - low) * frac)


def _hash(vec: np.ndarray) -> str:
    return design_hash(decode(vec).to_dict())


def _row(vec: np.ndarray, j_fan: float, mass: float = 0.05, defl: float = 1e-4) -> dict:
    return {
        "design_hash": _hash(vec),
        "vector": vec.tolist(),
        "j_fan": j_fan,
        "mass_kg": mass,
        "deflection_m": defl,
        "blade_count": decode(vec).blade_count,
    }


def _write(shared: Path, rows: list[dict], name: str = "evaluations_colab-0.jsonl") -> None:
    shared.mkdir(parents=True, exist_ok=True)
    (shared / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _verification(designs: list[dict]) -> dict:
    return {"ranking": {"rank_preserved": True}, "designs": designs}


# --- load_shard_rows ---


def test_load_shard_rows_dedups_across_shards(tmp_path):
    _write(tmp_path, [_row(_vec(0.3), 1.0), _row(_vec(0.6), 3.0)], "evaluations_colab-0.jsonl")
    _write(tmp_path, [_row(_vec(0.6), 3.0)], "evaluations_colab-1.jsonl")  # dup design_hash
    rows = blade_results.load_shard_rows(tmp_path)
    assert len(rows) == 2  # the duplicate is deduped


def test_load_shard_rows_drops_nonfinite_objectives(tmp_path):
    _write(tmp_path, [
        _row(_vec(0.3), 1.0),
        _row(_vec(0.6), float("nan")),  # failed J_fan
        _row(_vec(0.9), 2.0, mass=float("inf")),  # infeasible mass
    ])
    rows = blade_results.load_shard_rows(tmp_path)
    assert [r["j_fan"] for r in rows] == [1.0]  # only the fully-finite row


def test_load_shard_rows_drops_torn_and_missing_objective_rows(tmp_path):
    good = _row(_vec(0.4), 2.0)
    torn = {"design_hash": "t", "vector": [0.1, 0.2], "j_fan": 5.0, "mass_kg": 0.05,
            "deflection_m": 1e-4}  # short vector
    missing = {"design_hash": "m", "vector": _vec(0.7).tolist(), "j_fan": 4.0}  # no mass/deflection
    _write(tmp_path, [good, torn, missing])
    rows = blade_results.load_shard_rows(tmp_path)
    assert [r["j_fan"] for r in rows] == [2.0]  # only the well-formed row survives


def test_load_shard_rows_missing_dir(tmp_path):
    assert blade_results.load_shard_rows(tmp_path / "nope") == []


def test_load_shard_rows_skips_blank_and_garbage_lines(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "evaluations_colab-0.jsonl").write_text(
        "\n   \nnot valid json {{{\n" + json.dumps(_row(_vec(0.3), 1.0)) + "\n", encoding="utf-8"
    )
    rows = blade_results.load_shard_rows(tmp_path)
    assert [r["j_fan"] for r in rows] == [1.0]  # blank + malformed-JSON lines skipped, good row kept


# --- blade_pareto_designs ---


def test_blade_pareto_designs_sorts_by_jfan_desc_and_drops_dominated():
    # A(2.0, light-ish) & B(1.5, lightest) trade off → both Pareto; C(1.0, heavy) dominated by A.
    rows = [
        _row(_vec(0.3), 2.0, mass=0.004, defl=0.001),
        _row(_vec(0.6), 1.5, mass=0.003, defl=0.0009),
        _row(_vec(0.9), 1.0, mass=0.006, defl=0.002),
    ]
    pf = blade_results.blade_pareto_designs(rows)
    assert [d["j_fan"] for d in pf] == [2.0, 1.5]  # J_fan descending, dominated C dropped
    assert [d["index"] for d in pf] == [0, 1]  # Pareto rank position


def test_blade_pareto_designs_carries_decoded_blade_summary():
    pf = blade_results.blade_pareto_designs([_row(_vec(0.4), 2.0)])
    d = pf[0]
    assert d["rib_mode"] in {"ribbed", "uniform"}
    assert d["blade_count"] == decode(_vec(0.4)).blade_count
    assert d["design_hash"] == _hash(_vec(0.4))  # the verification join key


def test_blade_pareto_designs_empty():
    assert blade_results.blade_pareto_designs([]) == []


# --- recommend_blades ---


def test_recommend_blades_without_verification(tmp_path):
    _write(tmp_path, [_row(_vec(0.3), 2.0), _row(_vec(0.6), 1.5), _row(_vec(0.9), 1.0)])
    out = blade_results.recommend_blades(tmp_path, top_k=3)
    assert out["verification"] == "absent"
    assert out["n_verified"] == 0
    assert out["ranked"] == []  # nothing verified yet
    assert all(d["j_fan_3d"] is None and d["verified"] is False for d in out["recommended"])


def test_recommend_blades_joins_verification_by_hash(tmp_path):
    va, vb = _vec(0.3), _vec(0.6)
    _write(tmp_path, [_row(va, 2.0, mass=0.004, defl=0.001), _row(vb, 1.5, mass=0.003, defl=0.0009)])
    ver = tmp_path / "verification.json"
    ver.write_text(json.dumps(_verification([
        {"name": f"00_{_hash(va)}", "j_fan_3d": 1.9e12, "j_fan_coarse": 2.0},
        {"name": f"01_{_hash(vb)}", "j_fan_3d": 3.3e12, "j_fan_coarse": 1.5},  # best in 3D
    ])), encoding="utf-8")
    out = blade_results.recommend_blades(tmp_path, top_k=3, verification_path=ver)
    assert out["verification"] == "present"
    assert out["n_verified"] == 2
    # ranked by fine J_fan: vb (3.3e12) above va (1.9e12) even though va had higher coarse
    assert [r["design_hash"] for r in out["ranked"]] == [_hash(vb), _hash(va)]
    assert out["ranked"][0]["mass_kg"] == 0.003  # shard metric joined through by hash


def test_recommend_blades_negative_fine_is_verified_not_suspect(tmp_path):
    va, vb = _vec(0.3), _vec(0.6)
    _write(tmp_path, [_row(va, 2.0), _row(vb, 1.5)])
    ver = tmp_path / "verification.json"
    ver.write_text(json.dumps(_verification([
        {"name": f"00_{_hash(va)}", "j_fan_3d": 1.0e12, "j_fan_coarse": 2.0},
        {"name": f"01_{_hash(vb)}", "j_fan_3d": -5.0e11, "j_fan_coarse": 1.5},  # real, bad result
    ])), encoding="utf-8")
    ranked = {r["design_hash"]: r for r in blade_results.recommend_blades(
        tmp_path, top_k=3, verification_path=ver)["ranked"]}
    assert ranked[_hash(vb)]["verified"] is True  # negative is a finite, real result
    assert ranked[_hash(vb)]["suspect"] is False


def test_recommend_blades_failed_fine_run_is_suspect_and_sinks(tmp_path):
    va, vb = _vec(0.3), _vec(0.6)
    _write(tmp_path, [_row(va, 2.0), _row(vb, 1.5)])
    ver = tmp_path / "verification.json"
    ver.write_text(json.dumps(_verification([
        {"name": f"00_{_hash(va)}", "j_fan_3d": None, "j_fan_coarse": 2.0},  # failed 3D run
        {"name": f"01_{_hash(vb)}", "j_fan_3d": 1.2e12, "j_fan_coarse": 1.5},
    ])), encoding="utf-8")
    ranked = blade_results.recommend_blades(tmp_path, top_k=3, verification_path=ver)["ranked"]
    assert ranked[0]["design_hash"] == _hash(vb)  # finite fine value ranks above the failed run
    assert ranked[-1]["design_hash"] == _hash(va) and ranked[-1]["suspect"] is True


def test_recommend_blades_skips_verification_design_without_hash_name(tmp_path):
    # A verify design whose name isn't "{rank}_{hash}" can't be joined by hash → skipped, not crashed.
    _write(tmp_path, [_row(_vec(0.3), 2.0)])
    ver = tmp_path / "verification.json"
    ver.write_text(json.dumps(_verification([{"name": "nounderscore", "j_fan_3d": 1.0}])),
                   encoding="utf-8")
    out = blade_results.recommend_blades(tmp_path, top_k=3, verification_path=ver)
    assert out["verification"] == "absent" and out["ranked"] == []  # unparseable name yields no join


def test_recommend_blades_stars_top_k_by_fine_jfan_when_verified(tmp_path):
    # With verification in, the ★ (recommended_for_print = the promote-to-TO set) must follow the
    # FINE J_fan, NOT the coarse-Pareto diversity — the whole point of the fine tier is that coarse
    # can mislead. Here the fine order is INVERTED vs coarse.
    va, vb, vc = _vec(0.3), _vec(0.6), _vec(0.9)
    _write(tmp_path, [_row(va, 3.0, mass=0.006), _row(vb, 2.0, mass=0.005), _row(vc, 1.0, mass=0.004)])
    ver = tmp_path / "verification.json"
    ver.write_text(json.dumps(_verification([
        {"name": f"00_{_hash(va)}", "j_fan_3d": 1.0e12, "j_fan_coarse": 3.0},  # coarse-best, fine-worst
        {"name": f"01_{_hash(vb)}", "j_fan_3d": 2.0e12, "j_fan_coarse": 2.0},
        {"name": f"02_{_hash(vc)}", "j_fan_3d": 3.0e12, "j_fan_coarse": 1.0},  # coarse-worst, fine-BEST
    ])), encoding="utf-8")
    out = blade_results.recommend_blades(tmp_path, top_k=2, verification_path=ver)
    starred = {r["design_hash"] for r in out["ranked"] if r["recommended_for_print"]}
    assert starred == {_hash(vc), _hash(vb)}  # top-2 by FINE J_fan (c, b) — not coarse (a, b)


def test_recommend_blades_picks_top_k_diverse_from_pareto(tmp_path):
    # 4 non-dominated designs (J_fan↓ with mass↓ — a real tradeoff so all are Pareto); 3 cluster in
    # vector space + 1 outlier. top_k=2 diverse must include the outlier _vec(0.95).
    rows = [_row(_vec(0.50), 2.0, mass=0.006), _row(_vec(0.51), 1.9, mass=0.005),
            _row(_vec(0.52), 1.8, mass=0.004), _row(_vec(0.95), 1.7, mass=0.003)]
    _write(tmp_path, rows)
    out = blade_results.recommend_blades(tmp_path, top_k=2)
    assert out["n_pareto"] == 4  # all four trade off → all non-dominated
    assert len(out["recommended"]) == 2
    assert _hash(_vec(0.95)) in {d["design_hash"] for d in out["recommended"]}  # the outlier
