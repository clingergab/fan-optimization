"""Tests for scripts/run_phase5_verify.py.

The 3D-verifying run (run_verification) is the expensive boundary and is mocked;
the CLI wiring — campaign load → top-k select → verification.json → exit 0 — is
exercised. Requires gmsh + cadquery (the phase5 import pulls them).
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

if importlib.util.find_spec("gmsh") is None:  # pragma: no cover - env-dependent
    pytest.skip("gmsh not installed", allow_module_level=True)
if importlib.util.find_spec("cadquery") is None:  # pragma: no cover - env-dependent
    pytest.skip("cadquery not installed", allow_module_level=True)

import run_phase5_verify as script
from fanopt.bo.codec import encode
from fanopt.bo.results import CHECKPOINT_NAME
from fanopt.cfd.phase5 import VerifyResult
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


def _fake_campaign(tmp_path):
    camp = tmp_path / "camp"
    camp.mkdir()
    x = np.array([_vec(8, 0.0022), _vec(10, 0.003), _vec(12, 0.0038)])
    y_raw = np.array([[2.0, 0.004, 0.001], [1.5, 0.005, 0.0009], [1.0, 0.006, 0.0008]])
    np.savez(camp / CHECKPOINT_NAME, x=x, y_raw=y_raw, iteration=1)
    return camp


def test_main_writes_verification_json(tmp_path, monkeypatch):
    camp = _fake_campaign(tmp_path)

    def fake_run_verification(designs, out_dir, **kwargs):
        return [
            VerifyResult(name, j_fan_3d=float(i + 1), j_fan_slice=j, meta={"n_nodes": 100.0})
            for i, (name, _vec_, j) in enumerate(designs)
        ]

    monkeypatch.setattr(script, "run_verification", fake_run_verification)
    rc = script.main(
        ["--campaign-dir", str(camp), "--out-dir", str(tmp_path / "out"), "--top-k", "2"]
    )
    assert rc == 0
    v = json.loads((tmp_path / "out" / "verification.json").read_text())
    assert "ranking" in v
    assert len(v["designs"]) == 2
    assert all("j_fan_3d" in d and "j_fan_slice" in d for d in v["designs"])


def test_designs_from_campaign_shapes(tmp_path):
    camp = _fake_campaign(tmp_path)
    designs = script._designs_from_campaign(camp, top_k=3)
    assert len(designs) >= 1
    name, vec, j_slice = designs[0]
    assert vec.shape[0] == 35
    assert isinstance(j_slice, float)


def test_run_checkpoints_verification_json_after_each_design(tmp_path, monkeypatch):
    """A mid-run disconnect must keep completed designs — verify the incremental write."""
    camp = _fake_campaign(tmp_path)
    out = tmp_path / "out"
    seen_counts = []

    def fake_run_verification(designs, out_dir, *, on_result=None, **kwargs):
        results = []
        for i, (name, _v, j) in enumerate(designs):
            r = VerifyResult(name, j_fan_3d=float(i + 1), j_fan_slice=j, meta={"n_nodes": 100.0})
            results.append(r)
            if on_result is not None:
                on_result(r)
                # verification.json on disk must already reflect designs done so far
                seen_counts.append(len(json.loads((out_dir / "verification.json").read_text())["designs"]))
        return results

    monkeypatch.setattr(script, "run_verification", fake_run_verification)
    script.run(campaign_dir=camp, out_dir=out, top_k=2)
    assert seen_counts == [1, 2]  # written after each design, not just at the end
