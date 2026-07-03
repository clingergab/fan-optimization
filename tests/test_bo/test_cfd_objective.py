"""Tests for fanopt.bo.cfd_objective (picklable CFD objective for process pools).

gmsh meshing is real; the SU2 subprocess (external boundary) is mocked in-process.
Requires gmsh + cadquery (I_wrist geometry).
"""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pytest

if importlib.util.find_spec("gmsh") is None:  # pragma: no cover - env-dependent
    pytest.skip("gmsh not installed", allow_module_level=True)
if importlib.util.find_spec("cadquery") is None:  # pragma: no cover - env-dependent
    pytest.skip("cadquery not installed", allow_module_level=True)

from fanopt.bo import codec
from fanopt.bo.cfd_objective import CfdObjective
from fanopt.cfd import phase3


def _mid_vector() -> np.ndarray:
    low, high = codec.bounds()
    return codec.clip_to_bounds((low + high) / 2.0)


def _fake_su2_writing(series):
    def fake_run(cmd, cwd, stdout, stderr, env):
        lines = ["Time_Iter,CFx"] + [f"{t},{v}" for t, v in enumerate(series)]
        (Path(cwd) / "history.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

        class R:
            returncode = 0

        return R()

    return fake_run


def test_cfd_objective_is_picklable():
    # The whole reason this class exists: it must round-trip through pickle so a
    # ProcessPoolExecutor can ship it to workers (a closure could not).
    obj = CfdObjective(out_dir=Path("/tmp/x"), su2_bin="/fake/SU2_CFD")
    restored = pickle.loads(pickle.dumps(obj))
    assert restored.out_dir == obj.out_dir
    assert restored.su2_bin == obj.su2_bin
    assert callable(restored)


def test_cfd_objective_returns_three_finite_objectives(tmp_path, monkeypatch):
    monkeypatch.setattr(phase3.subprocess, "run", _fake_su2_writing([5.0, -3.0] * 60))
    obj = CfdObjective(out_dir=tmp_path, su2_bin="/fake/SU2_CFD")
    j_fan, i_wrist, structural = obj(_mid_vector())
    assert np.isfinite(j_fan)
    assert i_wrist > 0.0  # real CadQuery inertia
    assert structural > 0.0  # real plate-bending deflection
