"""Tests for fanopt.bo.objective (Phase 4 objective spine).

The gmsh slice-meshing is real; the SU2 subprocess (the external boundary) is
mocked. Skips at module load where gmsh is absent (prepare_slice_case meshes).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

if importlib.util.find_spec("gmsh") is None:  # pragma: no cover - env-dependent
    pytest.skip("gmsh not installed", allow_module_level=True)

from fanopt.bo import codec, objective
from fanopt.cfd import phase3


def _mid_vector() -> np.ndarray:
    low, high = codec.bounds()
    return codec.clip_to_bounds((low + high) / 2.0)


def _fake_su2_writing(series: list[float]):
    def fake_run(cmd, cwd, stdout, stderr, env):
        lines = ["Time_Iter,CFx"] + [f"{t},{v}" for t, v in enumerate(series)]
        (Path(cwd) / "history.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

        class R:
            returncode = 0

        return R()

    return fake_run


def test_prepare_slice_case_writes_mesh_and_unsteady_cfg(tmp_path):
    layer1 = codec.decode(_mid_vector())
    mesh = objective.prepare_slice_case(layer1, tmp_path)
    assert Path(mesh.path).exists()
    assert (tmp_path / objective.UNSTEADY_CFG).exists()
    assert mesh.n_nodes > 0


def test_prepare_slice_case_cfg_is_plunging_unsteady(tmp_path):
    objective.prepare_slice_case(codec.decode(_mid_vector()), tmp_path)
    cfg = (tmp_path / objective.UNSTEADY_CFG).read_text()
    assert "PLUNGING_OMEGA=" in cfg
    assert "DUAL_TIME_STEPPING-2ND_ORDER" in cfg


def test_evaluate_j_fan_returns_finite(tmp_path, monkeypatch):
    monkeypatch.setattr(phase3.subprocess, "run", _fake_su2_writing([10.0, -10.0] * 60))
    j_fan, meta = objective.evaluate_j_fan(_mid_vector(), tmp_path, su2_bin="/fake/SU2_CFD")
    assert np.isfinite(j_fan)
    assert meta["n_nodes"] > 0
    assert meta["blade_count"] in (8.0, 10.0, 12.0)


def test_evaluate_j_fan_errors_without_su2(tmp_path, monkeypatch):
    monkeypatch.delenv("SU2_RUN", raising=False)
    monkeypatch.setattr(phase3.shutil, "which", lambda _n: None)
    with pytest.raises(RuntimeError, match="SU2_CFD not found"):
        objective.evaluate_j_fan(_mid_vector(), tmp_path, su2_bin=None)


def test_evaluate_design_injects_both_objectives(tmp_path, monkeypatch):
    monkeypatch.setattr(phase3.subprocess, "run", _fake_su2_writing([5.0, -3.0] * 60))
    res = objective.evaluate_design(
        _mid_vector(),
        tmp_path,
        su2_bin="/fake/SU2_CFD",
        inertia_fn=lambda p: 1.5e-4,
        structural_fn=lambda p: 42.0,
    )
    assert np.isfinite(res.j_fan)
    assert res.i_wrist_kgm2 == pytest.approx(1.5e-4)
    assert res.structural == pytest.approx(42.0)


def test_evaluate_design_defaults_objectives_to_none(tmp_path, monkeypatch):
    monkeypatch.setattr(phase3.subprocess, "run", _fake_su2_writing([1.0, -1.0] * 60))
    res = objective.evaluate_design(_mid_vector(), tmp_path, su2_bin="/fake/SU2_CFD")
    assert res.i_wrist_kgm2 is None
    assert res.structural is None


def test_inertia_fn_receives_decoded_params(tmp_path, monkeypatch):
    monkeypatch.setattr(phase3.subprocess, "run", _fake_su2_writing([2.0, -2.0] * 60))
    seen = {}

    def spy(p):
        seen["blade_count"] = p.blade_count
        return 0.0

    objective.evaluate_design(_mid_vector(), tmp_path, su2_bin="/fake/SU2_CFD", inertia_fn=spy)
    assert seen["blade_count"] in (8, 10, 12)
