"""Tests for scripts/rerank_v1_effort.py (V1 campaign re-ranked by wind per inertial effort)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import rerank_v1_effort  # noqa: E402
from fanopt.bo.blade_codec import bounds, clip_to_bounds, decode  # noqa: E402
from fanopt.utils.ledger import design_hash  # noqa: E402


@pytest.fixture
def outputs(tmp_path):
    low, high = bounds()
    rng = np.random.default_rng(5)
    rows = []
    for k in range(6):
        vec = clip_to_bounds(low + (high - low) * rng.uniform(0.1, 0.9, low.size))
        h = design_hash(decode(vec).to_dict())
        rows.append(
            {
                "design_hash": h,
                "vector": vec.tolist(),
                "j_fan": float(k - 2),
                "mass_kg": 0.4 + k / 100,
            }
        )
    rows.append({**rows[0], "j_fan": float("nan")})
    (tmp_path / "evaluations_c0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cal = tmp_path / "cal.json"
    cal.write_text(
        json.dumps(
            {"model": {"wall_mm": 1.6, "core_fraction": 0.15, "base_g": 2.0, "rho_kg_m3": 1240}}
        )
    )
    summary = rerank_v1_effort.run(
        shards=tmp_path, calibration=cal, out_csv=tmp_path / "o.csv", out_json=tmp_path / "o.json",
        recipe_design=rows[0]["design_hash"],
    )  # fmt: skip
    return tmp_path, rows, summary


def test_one_csv_row_per_valid_design(outputs):
    tmp, _, _ = outputs
    assert len(list(csv.DictReader((tmp / "o.csv").open()))) == 6


def test_recipe_is_the_named_design_thinned_by_the_scale(outputs):
    _, _, s = outputs
    assert s["recipe"]["from"].endswith("× 0.8")


def test_fan_is_twelve_blades_under_the_mass_cap(outputs):
    tmp, _, s = outputs
    row = next(csv.DictReader((tmp / "o.csv").open()))
    assert float(row["fan_mass_kg"]) == pytest.approx(12 * float(row["blade_mass_kg"])) and (
        s["over_mass_cap"] == 0
    )


def test_ratio_column_is_wind_over_push_work(outputs):
    tmp, _, _ = outputs
    row = next(csv.DictReader((tmp / "o.csv").open()))
    assert float(row["j_fan_per_push_work"]) == pytest.approx(
        float(row["j_fan"]) / float(row["work_push_j"])
    )


def test_summary_names_the_best_designs_by_hash(outputs):
    _, rows, s = outputs
    best = s["rerank_vs_j_fan"]["by_inertial_work"]["best_by_score"]
    assert best == max(rows[:6], key=lambda r: r["j_fan"])["design_hash"]


def test_summary_is_written(outputs):
    tmp, _, s = outputs
    assert json.loads((tmp / "o.json").read_text()) == s


def test_main_runs(outputs, capsys):
    tmp, rows, _ = outputs
    rc = rerank_v1_effort.main([
        "--shards", str(tmp), "--calibration", str(tmp / "cal.json"), "--out-csv",
        str(tmp / "o2.csv"), "--out-json", str(tmp / "o2.json"), "--recipe-design",
        rows[0]["design_hash"],
    ])  # fmt: skip
    assert rc == 0 and "rerank_vs_j_fan" in capsys.readouterr().out
