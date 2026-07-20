"""Tests for fanopt.bo.blade_campaign (aero-first BO loop, synthetic objective).

Skipped without botorch (the backbone dependency). SU2 is never touched — the objective
is a cheap synthetic function, so this verifies the campaign *wiring* (DoE, loop, ledger,
checkpoint, resume, Pareto), not the CFD.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("botorch") is None:
    pytest.skip("botorch not installed", allow_module_level=True)

from fanopt.bo import blade_campaign as bc
from fanopt.bo.blade_codec import N_DIMS, bounds, decode

_SMALL = bc.CampaignConfig(
    n_init=6, n_iterations=2, batch_size=1, seed=0,
    num_restarts=2, raw_samples=16, mc_samples=16,
)


def _synthetic(vector):
    """Cheap 3-objective stand-in for the CFD objective (all finite)."""
    v = np.asarray(vector, dtype=float)
    return (float(v[:8].sum()), float(v[2]), float(v[4]))


# --- DoE + fallback ----------------------------------------------------------


def test_sobol_doe_shape_and_bounds():
    x = bc.sobol_doe(10, seed=1)
    low, high = bounds()
    assert x.shape == (10, N_DIMS)
    assert np.all(x >= low - 1e-9) and np.all(x <= high + 1e-9)


def test_fallback_designs_decode():
    vecs = bc.diverse_fallback_designs()
    assert len(vecs) >= 3
    for v in vecs:
        decode(v)  # each must decode to a valid BladeParams


# --- campaign end to end -----------------------------------------------------


def test_campaign_runs_and_persists(tmp_path):
    state = bc.run_campaign(_synthetic, tmp_path, _SMALL, resume=False)
    assert state.x.shape[0] == _SMALL.n_init + _SMALL.n_iterations  # batch_size 1
    assert state.y_raw.shape[1] == 3
    assert (tmp_path / bc.CHECKPOINT_NAME).exists()
    assert (tmp_path / bc.LEDGER_NAME).exists()


def test_ledger_has_a_row_per_eval(tmp_path):
    bc.run_campaign(_synthetic, tmp_path, _SMALL, resume=False)
    lines = (tmp_path / bc.LEDGER_NAME).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == _SMALL.n_init + _SMALL.n_iterations


def test_pareto_designs_nonempty(tmp_path):
    state = bc.run_campaign(_synthetic, tmp_path, _SMALL, resume=False)
    pareto = bc.pareto_designs(state)
    assert len(pareto) >= 1
    assert set(pareto[0]) >= {"vector", "j_fan", "mass_kg", "deflection_m", "params"}


def test_resume_continues_from_checkpoint(tmp_path):
    bc.run_campaign(_synthetic, tmp_path, _SMALL, resume=False)
    cfg2 = bc.CampaignConfig(**{**_SMALL.__dict__, "n_iterations": 4})
    state = bc.run_campaign(_synthetic, tmp_path, cfg2, resume=True)
    assert state.x.shape[0] == _SMALL.n_init + cfg2.n_iterations  # resumed, not restarted


def test_campaign_parallel_eval(tmp_path):
    # n_workers > 1 ships the (picklable) objective to a process pool — the Colab path.
    cfg = bc.CampaignConfig(**{**_SMALL.__dict__, "n_workers": 2})
    state = bc.run_campaign(_synthetic, tmp_path, cfg, resume=False)
    assert state.x.shape[0] == cfg.n_init + cfg.n_iterations
