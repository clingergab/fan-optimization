"""Tests for fanopt.bo.orchestration (campaign loop). Requires botorch."""

from __future__ import annotations

import importlib.util
import json
import threading
import time

import numpy as np
import pytest

if importlib.util.find_spec("botorch") is None:  # pragma: no cover - env-dependent
    pytest.skip("botorch not installed", allow_module_level=True)

from fanopt.bo import orchestration as orch
from fanopt.bo.codec import N_DIMS, bounds, decode

_LOW, _HIGH = bounds()


def _smooth_objective(v: np.ndarray) -> tuple[float, float, float]:
    xn = (v - _LOW) / (_HIGH - _LOW)
    j_fan = float(-np.sum((xn - 0.5) ** 2))  # maximize (peak at centre)
    i_wrist = float(np.sum(xn[:18]))  # minimize
    structural = float(np.sum((1.0 - xn) ** 2))  # minimize
    return (j_fan, i_wrist, structural)


def _fast_cfg(**kw) -> orch.CampaignConfig:
    base = dict(n_init=6, n_iterations=3, num_restarts=2, raw_samples=16, mc_samples=16, seed=0)
    base.update(kw)
    return orch.CampaignConfig(**base)


# --- DoE + diverse designs (no botorch fit needed) ---


def test_sobol_doe_shape_and_decodable():
    x = orch.sobol_doe(8, seed=1)
    assert x.shape == (8, N_DIMS)
    assert np.all(x >= _LOW - 1e-9) and np.all(x <= _HIGH + 1e-9)
    for v in x:
        decode(v)  # every DoE vector must decode


def test_diverse_fallback_designs_are_decodable():
    designs = orch.diverse_fallback_designs()
    assert len(designs) >= 5
    blade_counts = {decode(v).blade_count for v in designs}
    assert blade_counts == {8, 10, 12}  # spans the categorical


# --- full campaign ---


def test_campaign_runs_and_persists(tmp_path):
    state = orch.run_campaign(_smooth_objective, tmp_path, _fast_cfg())
    assert state.x.shape == (6 + 3, N_DIMS)
    assert state.y_raw.shape == (9, 3)
    # ledger has one line per evaluation
    lines = (tmp_path / orch.LEDGER_NAME).read_text().strip().splitlines()
    assert len(lines) == 9
    row = json.loads(lines[0])
    assert row["J_fan"] is not None
    assert row["config_hash"]  # cross-tier hash stamped
    assert (tmp_path / orch.CHECKPOINT_NAME).exists()


def test_pareto_designs_are_non_dominated(tmp_path):
    state = orch.run_campaign(_smooth_objective, tmp_path, _fast_cfg())
    pf = orch.pareto_designs(state)
    assert len(pf) >= 1
    assert all("j_fan" in d and "blade_count" in d for d in pf)


def test_campaign_resumes_from_checkpoint(tmp_path):
    orch.run_campaign(_smooth_objective, tmp_path, _fast_cfg(n_iterations=2))
    lines_after_first = len((tmp_path / orch.LEDGER_NAME).read_text().strip().splitlines())
    assert lines_after_first == 6 + 2
    # resume: continue to 4 iterations total (adds 2 more, DoE not repeated)
    state = orch.run_campaign(_smooth_objective, tmp_path, _fast_cfg(n_iterations=4), resume=True)
    assert state.iteration == 4
    assert state.x.shape[0] == 6 + 4


def test_stall_fallback_triggers(tmp_path):
    # DoE returns varied objectives (GP fits); every later point is dominated →
    # HV never improves → after stall_patience the diverse fallback is injected.
    counter = {"n": 0}

    def plateau_objective(v: np.ndarray) -> tuple[float, float, float]:
        counter["n"] += 1
        if counter["n"] <= 6:  # n_init varied rows
            k = float(counter["n"])
            return (-k, k, k)
        return (-100.0, 100.0, 100.0)  # always dominated

    state = orch.run_campaign(
        plateau_objective, tmp_path, _fast_cfg(n_iterations=6, stall_patience=2)
    )
    assert state.used_fallback >= 1
    # a fallback evaluation is tagged in the ledger
    sources = {
        json.loads(line)["params"]["source"]
        for line in (tmp_path / orch.LEDGER_NAME).read_text().strip().splitlines()
    }
    assert "fallback" in sources


def test_no_trust_region_path(tmp_path):
    state = orch.run_campaign(_smooth_objective, tmp_path, _fast_cfg(use_trust_region=False))
    assert state.x.shape[0] == 9


def test_evaluate_batch_calls_on_eval_per_design(tmp_path):
    calls = {"n": 0}

    def on_eval() -> None:
        calls["n"] += 1

    batch = orch.sobol_doe(3, seed=0)
    orch._evaluate_batch(
        _smooth_objective,
        tmp_path / orch.LEDGER_NAME,
        batch,
        iteration=0,
        source="test",
        on_eval=on_eval,
    )
    assert calls["n"] == 3


def test_progress_bar_runs_to_completion(tmp_path):
    # progress=True must not change results (bar is display-only).
    state = orch.run_campaign(_smooth_objective, tmp_path, _fast_cfg(), progress=True)
    assert state.x.shape[0] == 9


def test_parallel_doe_uses_multiple_threads(tmp_path):
    seen: set[str] = set()
    lock = threading.Lock()

    def obj(v: np.ndarray) -> tuple[float, float, float]:
        with lock:
            seen.add(threading.current_thread().name)
        time.sleep(0.02)  # encourage overlap so the pool dispatches across threads
        return _smooth_objective(v)

    orch.run_campaign(obj, tmp_path, _fast_cfg(n_init=4, n_iterations=0, n_workers=4))
    assert len(seen) > 1  # the DoE evaluated across worker threads


def test_parallel_results_match_serial(tmp_path):
    # DoE is deterministic (seeded Sobol); parallel must match serial exactly.
    serial = orch.run_campaign(
        _smooth_objective, tmp_path / "serial", _fast_cfg(n_init=4, n_iterations=0, n_workers=1)
    )
    parallel = orch.run_campaign(
        _smooth_objective, tmp_path / "par", _fast_cfg(n_init=4, n_iterations=0, n_workers=4)
    )
    assert np.allclose(serial.y_raw, parallel.y_raw)
    # ledger has one row per design in both
    for d in ("serial", "par"):
        lines = (tmp_path / d / orch.LEDGER_NAME).read_text().strip().splitlines()
        assert len(lines) == 4
