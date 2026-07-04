"""Tests for fanopt.bo.backbone (qLogNEHVI + TuRBO). Requires botorch."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("botorch") is None:  # pragma: no cover - env-dependent
    pytest.skip("botorch not installed", allow_module_level=True)

import torch

from fanopt.bo import backbone as bb
from fanopt.bo.codec import N_DIMS, bounds


def _synthetic(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    low, high = bounds()
    rng = np.random.default_rng(seed)
    x = low + rng.random((n, N_DIMS)) * (high - low)
    span = high - low
    xn = (x - low) / span
    j_fan = -np.sum((xn - 0.5) ** 2, axis=1)  # maximize (peak at centre)
    i_wrist = np.sum(xn, axis=1)  # minimize
    struct = np.sum(xn**2, axis=1)  # minimize
    y_raw = np.column_stack([j_fan, i_wrist, struct])
    return x, y_raw


# --- objective transforms ---


def test_to_maximization_flips_minimize_objectives():
    y = np.array([[10.0, 2.0, 3.0]])
    out = bb.to_maximization(y)
    assert out[0, 0] == 10.0
    assert out[0, 1] == -2.0
    assert out[0, 2] == -3.0


def test_to_maximization_rejects_wrong_width():
    with pytest.raises(ValueError, match="objectives"):
        bb.to_maximization(np.zeros((2, 2)))


def test_normalize_objectives_centers_and_scales():
    y = np.array([[1e12, 1.0], [2e12, 2.0], [3e12, 3.0]])
    y_norm, loc, scale = bb.normalize_objectives(y)
    assert np.allclose(y_norm.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(bb.apply_objective_norm(y, loc, scale), y_norm)


def test_normalize_handles_zero_variance_column():
    y = np.array([[5.0, 1.0], [5.0, 2.0]])  # column 0 constant → scale would be 0
    y_norm, _, _ = bb.normalize_objectives(y)
    assert np.all(np.isfinite(y_norm))


def test_sanitize_replaces_non_finite_with_penalty():
    y = np.array([[1.0, 2.0], [np.nan, 3.0], [np.inf, 1.0]])
    out = bb.sanitize_objectives(y)
    assert np.all(np.isfinite(out))
    assert out[1, 0] < 1.0 and out[2, 0] < 1.0  # penalized below the min finite


def test_sanitize_is_noop_when_all_finite():
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(bb.sanitize_objectives(y), y)


def test_normalize_enables_extreme_magnitude_proposal():
    # J_fan ~1e12 broke qNEHVI (multinomial inf/nan) before normalize_objectives.
    x, _ = _synthetic(12)
    low, high = bounds()
    rng = np.random.default_rng(1)
    y_raw = np.column_stack(
        [rng.uniform(-5e12, 5e12, 12), rng.uniform(5e-3, 9e-3, 12), rng.uniform(5e-4, 9e-4, 12)]
    )
    y_norm, _, _ = bb.normalize_objectives(bb.to_maximization(y_raw))
    model = bb.fit_gp(x, y_norm, low, high)
    ref = bb.infer_reference_point(y_norm)
    cand = bb.propose_candidates(
        model,
        x,
        y_norm,
        low,
        high,
        ref,
        batch_size=4,
        num_restarts=2,
        raw_samples=16,
        mc_samples=16,
    )
    assert cand.shape == (4, N_DIMS)
    assert np.all(np.isfinite(cand))


def test_reference_point_is_dominated_by_all():
    _, y_raw = _synthetic(10)
    y_max = bb.to_maximization(y_raw)
    ref = bb.infer_reference_point(y_max)
    assert np.all(y_max >= ref)


# --- Pareto / hypervolume ---


def test_pareto_mask_picks_non_dominated():
    y_max = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-1.0, -1.0]])
    mask = bb.pareto_mask(y_max)
    assert mask[0] and mask[1]
    assert not mask[3]  # dominated by everything


def test_hypervolume_positive_and_monotone():
    ref = np.array([-1.0, -1.0])
    small = bb.hypervolume(np.array([[0.0, 0.0]]), ref)
    bigger = bb.hypervolume(np.array([[0.0, 0.0], [0.5, 0.5]]), ref)
    assert small > 0.0
    assert bigger >= small


def test_hypervolume_empty_is_zero():
    assert bb.hypervolume(np.empty((0, 2)), np.array([-1.0, -1.0])) == 0.0


# --- trust region ---


def test_tr_grows_on_success_streak():
    tr = bb.TrustRegionState(dim=10, batch_size=1, length=0.4)
    for _ in range(tr.success_tol):
        tr.update(improved=True)
    assert tr.length == pytest.approx(0.8)


def test_tr_shrinks_on_failure_streak():
    tr = bb.TrustRegionState(dim=4, batch_size=4, length=0.8)  # failure_tol = 1
    tr.update(improved=False)
    assert tr.length == pytest.approx(0.4)


def test_tr_sets_restart_when_too_small():
    tr = bb.TrustRegionState(dim=1, batch_size=1)
    tr.length = 1.5 * tr.length_min  # one halving drops below length_min
    tr.failure_tol = 1
    tr.update(improved=False)
    assert tr.restart_triggered


# --- GP fit + acquisition (the real optimizer path) ---


def test_fit_gp_and_propose_full_domain():
    x, y_raw = _synthetic(10)
    low, high = bounds()
    y_max = bb.to_maximization(y_raw)
    model = bb.fit_gp(x, y_max, low, high)
    ref = bb.infer_reference_point(y_max)
    cand = bb.propose_candidates(
        model, x, y_max, low, high, ref, batch_size=2, num_restarts=2, raw_samples=16, mc_samples=16
    )
    assert cand.shape == (2, N_DIMS)
    assert np.all(cand >= low - 1e-9) and np.all(cand <= high + 1e-9)
    assert np.all(np.isfinite(cand))


@pytest.mark.skipif(
    importlib.util.find_spec("numpyro") is None,
    reason="SAASBO needs the fully_bayesian extra (numpyro)",
)
def test_fit_saas_gp_fallback_smoke():
    # SAASBO fully-Bayesian multi-output fit (ModelList of 1-output SAAS GPs).
    x, y_raw = _synthetic(6)
    low, high = bounds()
    y_max = bb.to_maximization(y_raw)
    model = bb.fit_saas_gp(x, y_max, low, high, warmup_steps=4, num_samples=4, thinning=1)
    assert len(model.models) == 3
    post = model.posterior(torch.zeros(1, N_DIMS, dtype=torch.double))
    assert torch.isfinite(post.mean).all()
    assert post.mean.shape[-1] == 3


def test_propose_inside_trust_region():
    x, y_raw = _synthetic(10)
    low, high = bounds()
    y_max = bb.to_maximization(y_raw)
    model = bb.fit_gp(x, y_max, low, high)
    ref = bb.infer_reference_point(y_max)
    tr = bb.TrustRegionState(dim=N_DIMS, batch_size=1, length=0.4)
    cand = bb.propose_candidates(
        model,
        x,
        y_max,
        low,
        high,
        ref,
        batch_size=1,
        tr_state=tr,
        num_restarts=2,
        raw_samples=16,
        mc_samples=16,
    )
    # Candidate must lie in the TR box around the incumbent (± length/2 in unit cube).
    span = high - low
    cand_n = (cand[0] - low) / span
    incumbent_n = ((x[int(np.argmax(y_max[:, 0]))]) - low) / span
    assert np.all(np.abs(cand_n - incumbent_n) <= tr.length / 2.0 + 1e-6)
