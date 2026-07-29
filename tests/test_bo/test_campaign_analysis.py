"""Tests for the pooled campaign-ledger analysis (numpy-only; no botorch needed)."""

from __future__ import annotations

import json

import numpy as np

from fanopt.bo.blade_codec import N_DIMS, bounds
from fanopt.bo.campaign_analysis import (
    campaign_report,
    failed_designs,
    find_shards,
    load_rows,
    pareto_indices,
    session_trajectories,
    shape_evolution,
    shape_summary,
    top_designs_shapes,
)


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _row(h, j, mass, defl, source="bo", ts="2026-07-28T00:00:00", blades=8):
    return {
        "design_hash": h,
        "j_fan": j,
        "mass_kg": mass,
        "deflection_m": defl,
        "source": source,
        "timestamp_iso": ts,
        "blade_count": blades,
    }


def test_dedup_across_shards_and_nan_split(tmp_path):
    a = tmp_path / "evaluations_A.jsonl"
    b = tmp_path / "evaluations_B.jsonl"
    _write(a, [_row("h1", 1.0e12, 0.09, 1e-3), _row("h2", 2.0e12, 0.10, 1e-3)])
    _write(
        b, [_row("h2", 2.0e12, 0.10, 1e-3), _row("h3", float("nan"), float("nan"), float("nan"))]
    )
    rep = campaign_report([a, b])
    assert rep["total_rows"] == 4
    assert rep["unique_designs"] == 3  # h2 deduped
    assert rep["finite"] == 2 and rep["failed_nan"] == 1


def test_progression_running_best_and_by_time_in_time_order(tmp_path):
    shard = tmp_path / "evaluations_A.jsonl"
    _write(
        shard,
        [
            _row("h1", 1.0e12, 0.09, 1e-3, ts="2026-07-28T00:00:01"),
            _row("h2", 3.0e12, 0.09, 1e-3, ts="2026-07-28T00:00:02"),
            _row("h3", 2.0e12, 0.09, 1e-3, ts="2026-07-28T00:00:03"),
        ],
    )
    p = campaign_report([shard])["progression"]
    assert p["running_best"] == [1.0e12, 3.0e12, 3.0e12]  # monotone (max-so-far)
    assert p["j_by_time"] == [1.0e12, 3.0e12, 2.0e12]  # actual values, ordered by timestamp
    assert p["n_new_bests"] == 2


def _srow(h, j, sess, ts):
    return _row(h, j, 0.09, 1e-3, ts=ts) | {"session_id": sess}


def test_per_session_trajectories_and_learning_means(tmp_path):
    shard = tmp_path / "evaluations_A.jsonl"
    # session A improves (1->2->4->6); session B flat (1,1). campaign_report groups by session_id.
    _write(
        shard,
        [
            _srow("a1", 1.0e12, "A", "t1"),
            _srow("a2", 2.0e12, "A", "t2"),
            _srow("a3", 4.0e12, "A", "t3"),
            _srow("a4", 6.0e12, "A", "t4"),
            _srow("b1", 1.0e12, "B", "t5"),
            _srow("b2", 1.0e12, "B", "t6"),
        ],
    )
    r = campaign_report([shard])
    bs = r["progression"]["by_session"]
    assert set(bs) == {"A", "B"} and bs["A"]["n"] == 4
    assert bs["A"]["second_half_mean"] > bs["A"]["first_half_mean"]  # A learned
    assert r["bo_mean"] is not None  # all rows are BO here; sobol_mean is None (no DoE rows)


def test_top_under_mass_cap_excludes_heavy(tmp_path):
    shard = tmp_path / "evaluations_A.jsonl"
    _write(
        shard,
        [
            _row("heavy", 5.0e12, 0.18, 1e-3),  # best J_fan but 180 g — over the cap
            _row("light", 3.0e12, 0.09, 1e-3),  # 90 g — eligible
        ],
    )
    elig = campaign_report([shard])["top_under_mass_cap"]["100.0g"]
    assert all(d["mass_g"] <= 100 for d in elig)
    assert elig and elig[0]["design_hash"] == "light"[:8]


def test_pareto_indices_excludes_dominated():
    # rows: (J_fan↑, mass↓, deflection↓). row1 dominates row0 (more J_fan, less mass).
    obj = np.array([[1.0, 0.10, 1e-3], [2.0, 0.09, 1e-3], [1.5, 0.20, 2e-3]])
    keep = pareto_indices(obj)
    assert 1 in keep and 0 not in keep  # row0 dominated by row1


def test_find_shards_surfaces_duplicate_folders(tmp_path):
    for folder in ("blade_campaign", "blade_campaign (1)"):
        d = tmp_path / folder
        d.mkdir()
        _write(d / "evaluations_colab-0.jsonl", [_row("h1", 1.0e12, 0.09, 1e-3)])
    found = find_shards(tmp_path)
    assert len(found) == 2  # both the real folder and the duplicate are surfaced


def test_shape_summary_buckets_surface_types(tmp_path):
    low, high = bounds()
    rng = np.random.default_rng(0)
    rows = []
    for i in range(9):
        v = (low + rng.random(N_DIMS) * (high - low)).tolist()
        rows.append(
            {
                "design_hash": f"h{i}",
                "vector": v,
                "j_fan": 1e12 + i * 1e11,
                "mass_kg": 0.09,
                "deflection_m": 1e-3,
                "source": "bo",
                "timestamp_iso": f"t{i:02d}",
            }
        )
    _write(tmp_path / "evaluations_A.jsonl", rows)
    s = shape_summary([tmp_path / "evaluations_A.jsonl"])
    assert s["n"] == 9
    assert sum(s["peak_counts"].values()) == 9
    assert set(s["peak_counts"]) <= {"hub", "mid", "tip"}
    assert "early_peak_dist" in s and "late_peak_dist" in s  # convergence check present


def test_top_designs_shapes_decodes_winners(tmp_path):
    low, high = bounds()
    rng = np.random.default_rng(1)
    rows = []
    for i in range(6):
        v = (low + rng.random(N_DIMS) * (high - low)).tolist()
        rows.append(
            {
                "design_hash": f"h{i}",
                "vector": v,
                "j_fan": 1e12 + i * 1e11,
                "mass_kg": 0.09,
                "deflection_m": 1e-3,
                "timestamp_iso": f"t{i}",
            }
        )
    _write(tmp_path / "evaluations_A.jsonl", rows)
    top = top_designs_shapes([tmp_path / "evaluations_A.jsonl"], k=3)
    assert len(top) == 3
    assert top[0]["j_fan"] >= top[1]["j_fan"] >= top[2]["j_fan"]  # sorted desc
    assert len(top[0]["knots_mm"]) == 5 and top[0]["peak"] in ("hub", "mid", "tip")


def test_session_trajectories_per_session_time_ordered_with_details(tmp_path):
    low, high = bounds()
    rng = np.random.default_rng(4)
    rows = []
    for s in ("colab-0", "colab-1"):
        for i in range(4):
            v = (low + rng.random(N_DIMS) * (high - low)).tolist()
            rows.append(
                {
                    "design_hash": f"{s}-{i}",
                    "vector": v,
                    "j_fan": 1e12 + i * 1e11,
                    "mass_kg": 0.09,
                    "deflection_m": 1e-3,
                    "session_id": s,
                    "timestamp_iso": f"2026-07-29T00:0{i}:00",
                }
            )
    _write(tmp_path / "evaluations_A.jsonl", rows)
    traj = session_trajectories([tmp_path / "evaluations_A.jsonl"])
    assert set(traj) == {"colab-0", "colab-1"} and len(traj["colab-0"]) == 4
    pt = traj["colab-0"][0]
    assert {"eval", "j_fan", "mass_g", "peak", "hash"} <= set(pt)  # hover fields present
    assert [d["eval"] for d in traj["colab-0"]] == [1, 2, 3, 4]  # time-ordered


def test_shape_evolution_samples_across_time_order(tmp_path):
    low, high = bounds()
    rng = np.random.default_rng(3)
    rows = []
    for i in range(20):
        v = (low + rng.random(N_DIMS) * (high - low)).tolist()
        rows.append(
            {
                "design_hash": f"h{i}",
                "vector": v,
                "j_fan": 1e12 + i * 1e10,
                "mass_kg": 0.09,
                "deflection_m": 1e-3,
                "timestamp_iso": f"2026-07-29T00:{i:02d}:00",
            }
        )
    _write(tmp_path / "evaluations_A.jsonl", rows)
    ev = shape_evolution([tmp_path / "evaluations_A.jsonl"], n_samples=5)
    assert len(ev) == 5
    assert ev[0]["eval_frac"] == 0.0 and ev[-1]["eval_frac"] == 1.0  # spans first->last
    assert [d["eval_index"] for d in ev] == sorted(d["eval_index"] for d in ev)  # time-ordered
    assert len(ev[0]["knots_mm"]) == 5


def test_failed_designs_lists_only_nan(tmp_path):
    low, high = bounds()
    rng = np.random.default_rng(2)
    v_ok = (low + rng.random(N_DIMS) * (high - low)).tolist()
    v_bad = (low + rng.random(N_DIMS) * (high - low)).tolist()
    _write(
        tmp_path / "evaluations_A.jsonl",
        [
            {
                "design_hash": "ok",
                "vector": v_ok,
                "j_fan": 2e12,
                "mass_kg": 0.09,
                "deflection_m": 1e-3,
            },
            {
                "design_hash": "bad",
                "vector": v_bad,
                "j_fan": float("nan"),
                "mass_kg": float("nan"),
                "deflection_m": float("nan"),
                "source": "bo",
            },
        ],
    )
    fails = failed_designs([tmp_path / "evaluations_A.jsonl"])
    assert [f["hash"] for f in fails] == ["bad"]
    assert len(fails[0]["vector"]) == N_DIMS  # vector kept so the failure can be re-rendered


def test_load_rows_skips_torn_lines(tmp_path):
    shard = tmp_path / "evaluations_A.jsonl"
    shard.write_text(json.dumps(_row("h1", 1.0e12, 0.09, 1e-3)) + "\n{bad json\n", encoding="utf-8")
    assert len(load_rows([shard])) == 1
