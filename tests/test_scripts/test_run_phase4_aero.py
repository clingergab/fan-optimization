"""Tests for scripts/run_phase4_aero.py.

The campaign wiring is tested with a cheap synthetic objective (no SU2); ``main`` arg →
config wiring is tested with the campaign runner mocked out. Skipped without botorch
(the backbone dependency the script imports).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("botorch") is None:  # pragma: no cover - env-dependent
    pytest.skip("botorch not installed", allow_module_level=True)

import run_phase4_aero as script
from fanopt.bo.blade_campaign import CampaignConfig

_SMALL = CampaignConfig(n_init=6, n_iterations=2, batch_size=1, num_restarts=2,
                        raw_samples=16, mc_samples=16)


def _synthetic(vector):
    v = np.asarray(vector, dtype=float)
    return (float(v[:8].sum()), float(v[2]), float(v[4]))


def test_run_with_synthetic_objective_writes_pareto(tmp_path):
    state = script.run(tmp_path, cfg=_SMALL, objective_fn=_synthetic)
    assert state.x.shape[0] == _SMALL.n_init + _SMALL.n_iterations
    assert (tmp_path / "pareto.json").exists()


def test_main_wires_args_into_config(tmp_path, monkeypatch):
    captured: dict = {}

    def fake_run(out_dir, **kw):
        captured["out_dir"] = out_dir
        captured.update(kw)

        class _State:
            x = np.zeros((3, 18))
            y_raw = np.zeros((3, 3))

        return _State()

    monkeypatch.setattr(script, "run", fake_run)
    monkeypatch.setattr(script, "pareto_designs", lambda s: [])
    rc = script.main(
        ["--out-dir", str(tmp_path), "--n-init", "4", "--n-iterations", "2", "--n-workers", "3"]
    )
    assert rc == 0
    cfg = captured["cfg"]
    assert (cfg.n_init, cfg.n_iterations, cfg.n_workers) == (4, 2, 3)
