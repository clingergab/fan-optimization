"""Tests for scripts/run_phase5_verify_blade.py.

The 3D-verifying run (verify_blades) is the expensive boundary and is mocked; the CLI
wiring — pareto load → verify → verification.json → exit 0 — is exercised. Requires gmsh +
cadquery (the blade_verify import pulls them).
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

import run_phase5_verify_blade as script
from fanopt.bo.blade_codec import bounds, clip_to_bounds, decode
from fanopt.cfd.phase5 import VerifyResult


def _vec(frac: float) -> np.ndarray:
    low, high = bounds()
    return clip_to_bounds(low + (high - low) * frac)


def _fake_pareto(tmp_path):
    pareto = [
        {"vector": _vec(f).tolist(), "j_fan": j, "mass_kg": 0.05,
         "deflection_m": 1e-4, "params": decode(_vec(f)).to_dict()}
        for f, j in [(0.3, 1.0), (0.6, 3.0), (0.5, 2.0)]
    ]
    p = tmp_path / "pareto.json"
    p.write_text(json.dumps(pareto), encoding="utf-8")
    return p


def test_main_writes_verification_json(tmp_path, monkeypatch):
    p = _fake_pareto(tmp_path)

    def fake_verify_blades(pareto, out_dir, *, top_k=None, on_result=None, **kw):
        results = [
            VerifyResult(f"{i:02d}", j_fan_3d=float(i + 1), j_fan_slice=float(e["j_fan"]),
                         meta={"n_nodes": 100.0})
            for i, e in enumerate(pareto[:top_k] if top_k else pareto)
        ]
        for r in results:
            if on_result is not None:
                on_result(r)
        from fanopt.cfd.phase5 import verify_ranking

        return results, verify_ranking(results)

    monkeypatch.setattr(script, "verify_blades", fake_verify_blades)
    rc = script.main(["--pareto", str(p), "--out-dir", str(tmp_path / "out"), "--top-k", "2"])
    assert rc == 0
    v = json.loads((tmp_path / "out" / "verification.json").read_text())
    assert "ranking" in v
    assert len(v["designs"]) == 2
    assert all("j_fan_3d" in d and "j_fan_slice" in d for d in v["designs"])


def test_run_checkpoints_after_each_design(tmp_path, monkeypatch):
    p = _fake_pareto(tmp_path)
    out = tmp_path / "out"
    seen_counts = []

    def fake_verify_blades(pareto, out_dir, *, top_k=None, on_result=None, **kw):
        results = []
        for i, e in enumerate(pareto[:top_k] if top_k else pareto):
            r = VerifyResult(f"{i:02d}", j_fan_3d=float(i + 1), j_fan_slice=float(e["j_fan"]),
                             meta={"n_nodes": 100.0})
            results.append(r)
            if on_result is not None:
                on_result(r)
                seen_counts.append(
                    len(json.loads((out_dir / "verification.json").read_text())["designs"])
                )
        return results, {}

    monkeypatch.setattr(script, "verify_blades", fake_verify_blades)
    script.run(pareto_path=p, out_dir=out, top_k=2)
    assert seen_counts == [1, 2]  # incremental write after each design
