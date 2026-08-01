"""Tests for fanopt.cfd.blade_verify (3D verification of the redesigned aero-first blade).

Geometry + gmsh meshing are real; the SU2 subprocess is mocked. Requires gmsh + cadquery.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

if importlib.util.find_spec("gmsh") is None:  # pragma: no cover - env-dependent
    pytest.skip("gmsh not installed", allow_module_level=True)
if importlib.util.find_spec("cadquery") is None:  # pragma: no cover - env-dependent
    pytest.skip("cadquery not installed", allow_module_level=True)

from fanopt.bo.blade_codec import bounds, clip_to_bounds, decode
from fanopt.cfd import blade_verify, phase3
from fanopt.geometry.blade import BladeParams
from fanopt.utils.ledger import design_hash


def _vec(frac: float) -> np.ndarray:
    low, high = bounds()
    return clip_to_bounds(low + (high - low) * frac)


def _pareto_entry(vec: np.ndarray, j_fan: float) -> dict[str, object]:
    return {
        "vector": vec.tolist(),
        "j_fan": j_fan,
        "mass_kg": 0.05,
        "deflection_m": 1e-4,
        "params": decode(vec).to_dict(),
    }


def _fake_su2_writing(series):
    def fake_run(cmd, cwd, stdout, stderr, env):
        # 3D thrust axis is CFz; tile to >= 5 cycles x 200 steps (the FINE tier, the verify default)
        # so the period guard passes at coarse OR fine.
        full = (list(series) * (1000 // len(series) + 1))[:1000]
        lines = ["Time_Iter,CFz"] + [f"{t},{v}" for t, v in enumerate(full)]
        (Path(cwd) / "history.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

        class R:
            returncode = 0

        return R()

    return fake_run


# --- designs_from_pareto (pure) ---


def test_designs_from_pareto_sorts_by_j_fan_desc():
    pareto = [_pareto_entry(_vec(f), j) for f, j in [(0.3, 1.0), (0.6, 3.0), (0.5, 2.0)]]
    designs = blade_verify.designs_from_pareto(pareto)
    assert [d[2] for d in designs] == [3.0, 2.0, 1.0]


def test_designs_from_pareto_top_k_truncates():
    pareto = [_pareto_entry(_vec(f), j) for f, j in [(0.3, 1.0), (0.6, 3.0), (0.5, 2.0)]]
    designs = blade_verify.designs_from_pareto(pareto, top_k=2)
    assert [d[2] for d in designs] == [3.0, 2.0]


def test_designs_from_pareto_carries_params_and_name():
    v = _vec(0.4)
    designs = blade_verify.designs_from_pareto([_pareto_entry(v, 1.0)])
    name, params, j_slice = designs[0]
    assert name.startswith("00_")
    assert isinstance(params, BladeParams)  # absolute params, not a range-dependent vector
    assert params == decode(v)
    assert j_slice == 1.0


def test_designs_from_pareto_names_stable_across_reruns():
    pareto = [_pareto_entry(_vec(0.6), 3.0), _pareto_entry(_vec(0.3), 1.0)]
    a = [d[0] for d in blade_verify.designs_from_pareto(pareto)]
    b = [d[0] for d in blade_verify.designs_from_pareto(pareto)]
    assert a == b


def test_load_pareto_round_trip(tmp_path):
    pareto = [_pareto_entry(_vec(0.5), 2.0)]
    p = tmp_path / "pareto.json"
    p.write_text(json.dumps(pareto), encoding="utf-8")
    assert blade_verify.load_pareto(p)[0]["j_fan"] == 2.0


# --- campaign-shard input (the real Stage-3 output: vector rows, no params, no pareto.json) ---


def _shard_row(vec: np.ndarray, j_fan: float) -> dict[str, object]:
    p = decode(vec)
    return {
        "design_hash": design_hash(p.to_dict()),
        "vector": vec.tolist(),  # campaign rows carry the vector, NOT a params dict
        "j_fan": j_fan,
        "mass_kg": 0.05,
        "deflection_m": 1e-4,
        "blade_count": p.blade_count,
    }


def _write_shard(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_designs_from_pareto_accepts_vector_only_rows():
    # campaign shard rows have 'vector' but no 'params'; the builder must decode them.
    v = _vec(0.4)
    designs = blade_verify.designs_from_pareto([{"vector": v.tolist(), "j_fan": 2.0}])
    assert designs[0][1] == decode(v)  # decoded to the right absolute BladeParams
    assert designs[0][2] == 2.0  # coarse J_fan carried through


def test_load_campaign_rows_dedups_across_shards_and_drops_nan(tmp_path):
    _write_shard(tmp_path / "evaluations_colab-0.jsonl",
                 [_shard_row(_vec(0.3), 1.0), _shard_row(_vec(0.6), 3.0)])
    _write_shard(tmp_path / "evaluations_colab-1.jsonl",
                 [_shard_row(_vec(0.6), 3.0),  # SAME design_hash as shard-0 → deduped
                  _shard_row(_vec(0.9), float("nan"))])  # failed/infeasible → dropped
    rows = blade_verify.load_campaign_rows(tmp_path)
    assert len(rows) == 2  # dup deduped, NaN dropped
    assert sorted(float(r["j_fan"]) for r in rows) == [1.0, 3.0]


def test_load_campaign_rows_drops_torn_vector_rows(tmp_path):
    # A concurrent Drive merge can leave a row with a finite j_fan but a truncated/overlong vector.
    # It must be dropped at read time (like the campaign's own _iter_rows guard), NOT survive to
    # crash decode() outside the fault-isolated pool and abort the whole multi-hour launch.
    good = _shard_row(_vec(0.4), 2.0)
    torn = {"design_hash": "torn", "vector": [0.1, 0.2, 0.3], "j_fan": 5.0}  # len 3 != N_DIMS
    non_numeric = {"design_hash": "bad", "vector": ["x"] * 33, "j_fan": 4.0}  # right len, not floats
    _write_shard(tmp_path / "evaluations_colab-0.jsonl", [good, torn, non_numeric])
    rows = blade_verify.load_campaign_rows(tmp_path)
    assert [float(r["j_fan"]) for r in rows] == [2.0]  # only the well-formed row survives


def test_top_designs_from_shards_ranks_by_coarse_and_decodes(tmp_path):
    _write_shard(
        tmp_path / "evaluations_colab-0.jsonl",
        [_shard_row(_vec(0.3), 1.0), _shard_row(_vec(0.6), 3.0), _shard_row(_vec(0.5), 2.0)],
    )
    designs = blade_verify.top_designs_from_shards(tmp_path, top_k=2)
    assert [d[2] for d in designs] == [3.0, 2.0]  # top-2 by coarse J_fan, best first
    assert all(isinstance(d[1], BladeParams) for d in designs)  # decoded from the stored vector


# --- geometry → mesh → cfg (real) ---


def test_prepare_blade_verification_case_builds_mesh_and_cfg(tmp_path):
    cfg = blade_verify.VerifyConfig()
    mesh = blade_verify.prepare_blade_verification_case(decode(_vec(0.5)), tmp_path, cfg)
    assert mesh.n_nodes > 0
    assert (tmp_path / blade_verify.MESH_NAME).exists()
    assert (tmp_path / blade_verify.STEP_NAME).exists()
    cfg = (tmp_path / blade_verify.CFG_NAME).read_text()
    assert "PITCHING_OMEGA=" in cfg
    assert blade_verify.FAN_SURFACE_MARKER in cfg


# --- SU2-driving path (mocked subprocess) ---


def test_verify_blades_happy_path(tmp_path, monkeypatch):
    monkeypatch.setattr(phase3.subprocess, "run", _fake_su2_writing([6.0, -2.0] * 30))
    pareto = [_pareto_entry(_vec(0.5), 2.0)]
    results, ranking = blade_verify.verify_blades(pareto, tmp_path, su2_bin="/fake/SU2_CFD")
    assert len(results) == 1
    assert np.isfinite(results[0].j_fan_3d)
    assert results[0].j_fan_coarse == 2.0
    assert results[0].meta["n_nodes"] > 0
    assert "kendall_tau" in ranking


def test_verify_blades_scales_j_fan_3d_to_whole_fan(tmp_path, monkeypatch):
    # extract_j_fan_3d is per-blade; verify_blades must scale it to WHOLE-FAN (× blade_count) so the
    # fine J_fan is the same quantity as the campaign's coarse (whole-fan) j_fan. Constant CFz=5 →
    # per-blade cycle-mean = 5.0 → whole-fan = 5.0 × blade_count.
    monkeypatch.setattr(phase3.subprocess, "run", _fake_su2_writing([5.0]))
    v = _vec(0.5)
    params = decode(v)
    results, _ = blade_verify.verify_blades(
        [_pareto_entry(v, 2.0)], tmp_path, su2_bin="/fake/SU2_CFD"
    )
    assert results[0].j_fan_3d == pytest.approx(5.0 * params.blade_count)
    assert params.blade_count == 12  # the locked count; whole-fan is 12× the per-blade thrust


def test_verify_blades_penalizes_failed_prep(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bad mesh")

    monkeypatch.setattr(blade_verify, "make_blade_solid", boom)
    pareto = [_pareto_entry(_vec(0.5), 2.0)]
    results, _ = blade_verify.verify_blades(pareto, tmp_path, su2_bin="/fake/SU2_CFD")
    assert np.isnan(results[0].j_fan_3d)
    assert list(tmp_path.glob("*/FAILED.txt"))
