"""Tests for scripts/run_phase4_bo.py.

The CFD-driving campaign (run_campaign) is the expensive boundary and is mocked;
the CLI wiring — arg parsing → run() → pareto.json → exit 0 — is exercised.
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
if importlib.util.find_spec("botorch") is None:  # pragma: no cover - env-dependent
    pytest.skip("botorch not installed", allow_module_level=True)

import run_phase4_bo as script
from fanopt.bo.backbone import TrustRegionState
from fanopt.bo.codec import N_DIMS
from fanopt.bo.orchestration import CampaignConfig, CampaignState, sobol_doe


def _fake_state(n_iterations: int) -> CampaignState:
    x = sobol_doe(3, seed=0)
    y_raw = np.array([[1.0, 2.0, 3.0], [1.5, 1.0, 2.0], [0.5, 3.0, 1.0]])
    return CampaignState(
        x=x, y_raw=y_raw, tr=TrustRegionState(dim=N_DIMS), iteration=n_iterations, used_fallback=1
    )


def test_make_cfd_objective_returns_callable(tmp_path):
    obj = script.make_cfd_objective(tmp_path, su2_bin="/fake/SU2_CFD")
    assert callable(obj)  # not invoked → no CFD


def test_main_writes_pareto_json(tmp_path, monkeypatch):
    def fake_run_campaign(objective_fn, out_dir, cfg, **kwargs):
        return _fake_state(cfg.n_iterations)

    monkeypatch.setattr(script, "run_campaign", fake_run_campaign)
    rc = script.main(["--out-dir", str(tmp_path), "--n-init", "3", "--n-iterations", "2"])
    assert rc == 0
    summary = json.loads((tmp_path / "pareto.json").read_text())
    assert summary["n_iterations"] == 2
    assert summary["n_pareto"] >= 1
    assert summary["used_fallback"] == 1
    assert all("blade_count" in d for d in summary["pareto"])


def test_run_returns_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(script, "run_campaign", lambda *a, **k: _fake_state(4))
    summary = script.run(out_dir=tmp_path, cfg=CampaignConfig(n_iterations=4), su2_bin=None)
    assert summary["n_evaluations"] == 3
    assert (tmp_path / "pareto.json").exists()
