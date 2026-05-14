"""Unit tests for fanopt.bo.spike_0_7b.

Spec reference: ``docs/plan_R11.md §Phase 0 Spike 0.7b``; protocol in
``docs/spike_0_7b_protocol.md``.

Covers the three pass-gates the spike enforces:

* GP fit-time gate (≤ 60 s).
* Architecture-bandit K_promoted gate (= 4 for the synthetic objective).
* TuRBO trust-region update gate (shrink-on-fail + grow-on-success).
"""

from __future__ import annotations

import numpy as np
import pytest

from fanopt.bo.spike_0_7b import (
    EPISTEMIC_NOISE_FLOOR_DEFAULT,
    GP_FIT_TIME_GATE_S,
    K_PROMOTED_SANITY,
    ArchitectureBanditRecord,
    GpFitTiming,
    Spike07bResult,
    TurboTRRecord,
    analyze_07b,
    calibrate_epistemic_noise_floor,
    lhs_sample,
    synthetic_objective,
)

# ---- synthetic_objective ----------------------------------------------------


def test_synthetic_objective_smooth_and_deterministic() -> None:
    """Repeated calls at the same x return the same value; nearby x give nearby f."""
    x = np.full(40, 0.5)
    v1 = synthetic_objective(x)
    v2 = synthetic_objective(x)
    assert v1 == v2

    # At the centre (all 0.5), the quadratic + interaction terms vanish.
    assert v1 == pytest.approx(0.0, abs=1e-12)

    # Smoothness: small perturbation → small change.
    x_perturbed = x.copy()
    x_perturbed[0] = 0.51
    v_pert = synthetic_objective(x_perturbed)
    assert abs(v_pert - v1) < 0.05


def test_synthetic_objective_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        synthetic_objective(np.array([]))


def test_synthetic_objective_noise_requires_rng() -> None:
    with pytest.raises(ValueError, match="rng"):
        synthetic_objective(np.full(40, 0.5), noise_std=0.1)


def test_synthetic_objective_noise_uses_rng() -> None:
    rng = np.random.default_rng(0)
    x = np.full(40, 0.5)
    a = synthetic_objective(x, noise_std=0.1, rng=rng)
    b = synthetic_objective(x, noise_std=0.1, rng=rng)
    # Two draws from the same rng differ.
    assert a != b


# ---- lhs_sample -------------------------------------------------------------


def test_lhs_sample_in_unit_box() -> None:
    rng = np.random.default_rng(0)
    s = lhs_sample(8, 40, rng=rng)
    assert s.shape == (8, 40)
    assert s.min() >= 0.0
    assert s.max() <= 1.0


def test_lhs_sample_one_per_stratum() -> None:
    """Each axis has one sample per equal-width stratum (the LHS property)."""
    rng = np.random.default_rng(0)
    n, d = 8, 12
    s = lhs_sample(n, d, rng=rng)
    for j in range(d):
        # Map each sample to its stratum index in [0, n-1].
        strata = np.floor(s[:, j] * n).astype(int)
        strata = np.clip(strata, 0, n - 1)
        # Exactly one sample per stratum.
        counts = np.bincount(strata, minlength=n)
        assert np.all(counts == 1), f"axis {j}: stratum counts = {counts.tolist()}"


def test_lhs_sample_rejects_bad_sizes() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        lhs_sample(0, 5, rng=rng)
    with pytest.raises(ValueError):
        lhs_sample(5, 0, rng=rng)


# ---- GP fit-time gate -------------------------------------------------------


def test_gp_fit_timing_under_60s_passes() -> None:
    """A timing record at wall_time ≤ 60 s passes the gate."""
    timings = [
        GpFitTiming(iteration=0, wall_time_s=0.5, n_train=8, d=40, passed=True),
        GpFitTiming(iteration=1, wall_time_s=1.2, n_train=9, d=40, passed=True),
    ]
    tr_log = _make_passing_tr_log()
    bandit = _make_passing_bandit()
    res = analyze_07b(timings, tr_log, bandit)
    assert res.all_gp_fits_under_60s is True
    assert res.passed is True


def test_gp_fit_timing_over_60s_fails() -> None:
    """A single iteration over 60 s flips the gate to fail."""
    timings = [
        GpFitTiming(iteration=0, wall_time_s=0.5, n_train=8, d=40, passed=True),
        GpFitTiming(
            iteration=1,
            wall_time_s=GP_FIT_TIME_GATE_S + 1.0,
            n_train=9,
            d=40,
            passed=False,
        ),
    ]
    tr_log = _make_passing_tr_log()
    bandit = _make_passing_bandit()
    res = analyze_07b(timings, tr_log, bandit)
    assert res.all_gp_fits_under_60s is False
    assert res.passed is False


def test_gp_fit_timing_empty_fails() -> None:
    """Empty timings → no evidence of pass → overall fail."""
    res = analyze_07b([], _make_passing_tr_log(), _make_passing_bandit())
    assert res.all_gp_fits_under_60s is False
    assert res.passed is False


# ---- bandit K_promoted gate -------------------------------------------------


def test_k_promoted_4_passes_for_synthetic() -> None:
    """Exactly 4 promoted architectures passes the synthetic-objective K=4 gate."""
    bandit = [
        ArchitectureBanditRecord(architecture_id=f"a{i}", screened_count=2, promoted=True)
        for i in range(K_PROMOTED_SANITY)
    ] + [
        ArchitectureBanditRecord(
            architecture_id=f"a{i+K_PROMOTED_SANITY}", screened_count=2, promoted=False
        )
        for i in range(4)
    ]
    res = analyze_07b(
        [GpFitTiming(iteration=0, wall_time_s=0.1, n_train=8, d=40, passed=True)],
        _make_passing_tr_log(),
        bandit,
    )
    assert res.k_promoted == K_PROMOTED_SANITY
    assert res.k_promoted_passes is True


def test_k_promoted_3_fails() -> None:
    bandit = [
        ArchitectureBanditRecord(architecture_id=f"a{i}", screened_count=2, promoted=True)
        for i in range(3)
    ]
    res = analyze_07b(
        [GpFitTiming(iteration=0, wall_time_s=0.1, n_train=8, d=40, passed=True)],
        _make_passing_tr_log(),
        bandit,
    )
    assert res.k_promoted_passes is False
    assert res.passed is False


# ---- TuRBO TR update gate --------------------------------------------------


def test_turbo_tr_shrinks_on_failure_grows_on_success() -> None:
    """A TR log with one valid shrink and one valid grow passes the gate."""
    tr_log = (
        TurboTRRecord(iteration=0, center=(0.5,), length=0.4, success_count=0, failure_count=0),
        # Grow: success_count up, length up.
        TurboTRRecord(iteration=1, center=(0.5,), length=0.6, success_count=1, failure_count=0),
        # Shrink: failure_count up, length down.
        TurboTRRecord(iteration=2, center=(0.5,), length=0.3, success_count=1, failure_count=1),
    )
    res = analyze_07b(
        [GpFitTiming(iteration=0, wall_time_s=0.1, n_train=8, d=40, passed=True)],
        tr_log,
        _make_passing_bandit(),
    )
    assert res.turbo_trs_update_correctly is True
    assert res.passed is True


def test_turbo_tr_no_shrink_fails() -> None:
    """A TR log with only growth fails the gate."""
    tr_log = (
        TurboTRRecord(iteration=0, center=(0.5,), length=0.4, success_count=0, failure_count=0),
        TurboTRRecord(iteration=1, center=(0.5,), length=0.6, success_count=1, failure_count=0),
        TurboTRRecord(iteration=2, center=(0.5,), length=0.8, success_count=2, failure_count=0),
    )
    res = analyze_07b(
        [GpFitTiming(iteration=0, wall_time_s=0.1, n_train=8, d=40, passed=True)],
        tr_log,
        _make_passing_bandit(),
    )
    assert res.turbo_trs_update_correctly is False
    assert res.passed is False


def test_turbo_tr_too_short_fails() -> None:
    """A 1-record TR log cannot show a shrink + grow → fails."""
    tr_log = (
        TurboTRRecord(iteration=0, center=(0.5,), length=0.4, success_count=0, failure_count=0),
    )
    res = analyze_07b(
        [GpFitTiming(iteration=0, wall_time_s=0.1, n_train=8, d=40, passed=True)],
        tr_log,
        _make_passing_bandit(),
    )
    assert res.turbo_trs_update_correctly is False
    assert res.passed is False


# ---- aggregate --------------------------------------------------------------


def test_spike_07b_aggregates_all_three_gates() -> None:
    """All three gates pass → overall pass; any failing → overall fail."""
    timings = [GpFitTiming(iteration=0, wall_time_s=0.5, n_train=8, d=40, passed=True)]
    tr_log = _make_passing_tr_log()
    bandit = _make_passing_bandit()
    good = analyze_07b(timings, tr_log, bandit)
    assert isinstance(good, Spike07bResult)
    assert good.all_gp_fits_under_60s is True
    assert good.k_promoted_passes is True
    assert good.turbo_trs_update_correctly is True
    assert good.passed is True

    # Flip exactly one gate at a time → overall must fail each time.
    bad_timings = [
        GpFitTiming(
            iteration=0,
            wall_time_s=GP_FIT_TIME_GATE_S + 1.0,
            n_train=8,
            d=40,
            passed=False,
        )
    ]
    assert analyze_07b(bad_timings, tr_log, bandit).passed is False

    bad_bandit = [
        ArchitectureBanditRecord(architecture_id=f"a{i}", screened_count=2, promoted=True)
        for i in range(K_PROMOTED_SANITY - 1)
    ]
    assert analyze_07b(timings, tr_log, bad_bandit).passed is False

    bad_tr = (
        TurboTRRecord(iteration=0, center=(0.5,), length=0.4, success_count=0, failure_count=0),
        TurboTRRecord(iteration=1, center=(0.5,), length=0.4, success_count=0, failure_count=0),
    )
    assert analyze_07b(timings, bad_tr, bandit).passed is False


# ---- shared helpers --------------------------------------------------------


def _make_passing_tr_log() -> tuple[TurboTRRecord, ...]:
    return (
        TurboTRRecord(iteration=0, center=(0.5,), length=0.4, success_count=0, failure_count=0),
        TurboTRRecord(iteration=1, center=(0.5,), length=0.6, success_count=1, failure_count=0),
        TurboTRRecord(iteration=2, center=(0.5,), length=0.3, success_count=1, failure_count=1),
    )


def _make_passing_bandit() -> tuple[ArchitectureBanditRecord, ...]:
    return tuple(
        ArchitectureBanditRecord(
            architecture_id=f"a{i}", screened_count=2, promoted=(i < K_PROMOTED_SANITY)
        )
        for i in range(K_PROMOTED_SANITY + 4)
    )


# ---- EPISTEMIC_NOISE_FLOOR calibration -------------------------------------


def test_epistemic_noise_floor_default_is_1e_minus_6() -> None:
    """Per §0 row 43 — `max(measured, 1e-6)` floor."""
    assert EPISTEMIC_NOISE_FLOOR_DEFAULT == 1.0e-6


def test_calibrate_uses_sample_variance_when_above_floor() -> None:
    """Measured variance ≫ floor → returns the measured variance."""
    # Five replicates with std ≈ 0.01 → var ≈ 1.25e-4, well above 1e-6.
    replicates = [0.10, 0.11, 0.09, 0.10, 0.11]
    out = calibrate_epistemic_noise_floor(replicates)
    assert out > 1.0e-6
    assert out == pytest.approx(float(np.var(replicates, ddof=1)), rel=1e-9)


def test_calibrate_falls_back_to_floor_when_replicates_identical() -> None:
    """Zero variance → floor of 1e-6."""
    out = calibrate_epistemic_noise_floor([0.05, 0.05, 0.05])
    assert out == EPISTEMIC_NOISE_FLOOR_DEFAULT


def test_calibrate_requires_at_least_two_replicates() -> None:
    """Single-replicate variance is undefined."""
    with pytest.raises(ValueError, match="≥ 2 replicates"):
        calibrate_epistemic_noise_floor([0.05])
    with pytest.raises(ValueError, match="≥ 2 replicates"):
        calibrate_epistemic_noise_floor([])


def test_calibrate_floor_min_override() -> None:
    """Operator can raise the floor (e.g., for a noisier tier)."""
    out = calibrate_epistemic_noise_floor([0.05, 0.05, 0.05], floor_min=1.0e-4)
    assert out == 1.0e-4
