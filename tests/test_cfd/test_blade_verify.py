"""Tests for fanopt.cfd.blade_verify (3D verification of the redesigned aero-first blade).

Geometry + gmsh meshing are real; the SU2 subprocess is mocked. Requires gmsh + cadquery.
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

from fanopt.bo.blade_codec import bounds, clip_to_bounds, decode
from fanopt.cfd import blade_verify, phase3
from fanopt.geometry.blade import BladeParams


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
        lines = ["Time_Iter,CFx"] + [f"{t},{v}" for t, v in enumerate(series)]
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
    import json

    pareto = [_pareto_entry(_vec(0.5), 2.0)]
    p = tmp_path / "pareto.json"
    p.write_text(json.dumps(pareto), encoding="utf-8")
    assert blade_verify.load_pareto(p)[0]["j_fan"] == 2.0


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
    assert results[0].j_fan_slice == 2.0
    assert results[0].meta["n_nodes"] > 0
    assert "kendall_tau" in ranking


def test_verify_blades_penalizes_failed_prep(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bad mesh")

    monkeypatch.setattr(blade_verify, "make_blade_solid", boom)
    pareto = [_pareto_entry(_vec(0.5), 2.0)]
    results, _ = blade_verify.verify_blades(pareto, tmp_path, su2_bin="/fake/SU2_CFD")
    assert np.isnan(results[0].j_fan_3d)
    assert list(tmp_path.glob("*/FAILED.txt"))
