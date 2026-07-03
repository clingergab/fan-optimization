"""Tests for scripts/run_phase2a_baseline_cfd.py.

Skips at module load where ``gmsh`` is absent (the script imports mesh_2d_slice,
which requires gmsh).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

if importlib.util.find_spec("gmsh") is None:  # pragma: no cover - env-dependent
    pytest.skip("gmsh not installed", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import run_phase2a_baseline_cfd as p2a  # noqa: E402


def _history(path: Path, cd: float) -> Path:
    # 2D slice: the streamwise force is CD (drag along the freestream).
    path.write_text(f"Inner_Iter,CL,CD\n0,0.2,0.0\n1,0.2,{cd}\n", encoding="utf-8")
    return path


def test_extract_baseline_load_delta(tmp_path):
    prod = _history(tmp_path / "prod.csv", 2.0)
    ret = _history(tmp_path / "ret.csv", 0.5)
    load = p2a.extract_baseline_load(prod, ret)
    assert load["j_fan_steady_proxy"] == pytest.approx(1.5)
    assert load["drag_productive"] == pytest.approx(2.0)
    assert load["suggested_pressure_pa"] > 0.0


def test_prepare_baseline_case_writes_mesh_and_cfgs(tmp_path):
    manifest = p2a.prepare_baseline_case(tmp_path, n_blades=3)
    assert Path(manifest["mesh"]).exists()
    assert Path(manifest["productive_cfg"]).exists()
    assert Path(manifest["return_cfg"]).exists()
    assert manifest["n_nodes"] > 0


def test_prepared_cfg_wires_cascade_marker(tmp_path):
    manifest = p2a.prepare_baseline_case(tmp_path, n_blades=3)
    cfg = Path(manifest["productive_cfg"]).read_text()
    assert "MARKER_SYM= ( cascade_wall )" in cfg
    assert "fan_surface" in cfg


def test_main_prepare_returns_zero(tmp_path):
    rc = p2a.main(["prepare", "--out-dir", str(tmp_path), "--n-blades", "3"])
    assert rc == 0


def test_main_extract_writes_load_json(tmp_path):
    prod = _history(tmp_path / "prod.csv", 2.0)
    ret = _history(tmp_path / "ret.csv", 0.5)
    out = tmp_path / "load.json"
    rc = p2a.main(
        [
            "extract",
            "--productive-history",
            str(prod),
            "--return-history",
            str(ret),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert json.loads(out.read_text())["j_fan_steady_proxy"] == pytest.approx(1.5)
