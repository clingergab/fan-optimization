"""Unit tests for fanopt.physical.inertia.

Validates the formula `I = κ · (T / 2π)²` against analytic reference geometries
(uniform rod transverse, plate) and exercises the Spike 0.2 pass-criteria gates.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.2; protocol in
docs/spike_0_2_protocol.md.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from fanopt.physical.inertia import (
    CROSS_CHECK_GATE_PCT,
    REPEATABILITY_GATE_PCT,
    analyze_trials,
    i_wrist_from_period,
    kappa_from_reference,
    rod_transverse_inertia,
)


# ----- analytic round-trip ---------------------------------------------------


def test_kappa_from_reference_round_trip() -> None:
    """κ → T → κ is exact (up to float)."""
    I_ref = 1.5e-4  # kg·m²
    T = 2.0  # s
    kappa = kappa_from_reference(I_ref_kgm2=I_ref, T_ref_s=T)
    I_recovered = i_wrist_from_period(kappa_Nm_per_rad=kappa, T_osc_s=T)
    assert math.isclose(I_recovered, I_ref, rel_tol=1e-12)


def test_rod_transverse_inertia_analytic() -> None:
    """(1/12)·m·L² for a uniform rod about its midpoint transverse axis."""
    m, L = 0.0521, 0.120
    expected = m * L * L / 12.0
    assert math.isclose(rod_transverse_inertia(m, L), expected, rel_tol=1e-12)


def test_rod_transverse_inertia_rejects_nonpositive() -> None:
    """rod_transverse_inertia validates m, L > 0 (defensive — used in calibration)."""
    with pytest.raises(ValueError, match="m, L must be > 0"):
        rod_transverse_inertia(0.0, 0.1)
    with pytest.raises(ValueError, match="m, L must be > 0"):
        rod_transverse_inertia(0.05, 0.0)
    with pytest.raises(ValueError, match="m, L must be > 0"):
        rod_transverse_inertia(-0.01, 0.1)


def test_period_for_known_rod() -> None:
    """A known rod should give a known period at a chosen κ.

    Hardware sanity: a 50 g, 120 mm rod (I ≈ 6.0e-5 kg·m²) on a wire with
    κ = 1.0e-3 N·m/rad gives T = 2π·√(I/κ) ≈ 1.539 s — well within the
    bench-stopwatch repeatability budget at 10 periods averaged.
    """
    m, L = 0.050, 0.120
    I_ref = rod_transverse_inertia(m, L)
    kappa = 1.0e-3
    T_expected = 2.0 * math.pi * math.sqrt(I_ref / kappa)
    assert math.isclose(T_expected, 1.539, abs_tol=0.005)


# ----- input validation ------------------------------------------------------


def test_i_wrist_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        i_wrist_from_period(kappa_Nm_per_rad=0.0, T_osc_s=1.0)
    with pytest.raises(ValueError):
        i_wrist_from_period(kappa_Nm_per_rad=1.0, T_osc_s=0.0)


def test_kappa_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        kappa_from_reference(I_ref_kgm2=0.0, T_ref_s=1.0)
    with pytest.raises(ValueError):
        kappa_from_reference(I_ref_kgm2=1.0, T_ref_s=0.0)


def test_analyze_requires_two_trials() -> None:
    with pytest.raises(ValueError):
        analyze_trials(kappa_Nm_per_rad=1.0, T_osc_trials_s=[1.0])


def test_analyze_rejects_negative_mount() -> None:
    with pytest.raises(ValueError):
        analyze_trials(
            kappa_Nm_per_rad=1.0,
            T_osc_trials_s=[1.0, 1.01, 0.99],
            mount_I_wrist_kgm2=-1e-6,
        )


# ----- repeatability gate ----------------------------------------------------


def test_repeatability_passes_when_trials_tight() -> None:
    """5 identical periods → repeatability ≈ 0% → passes < 3% gate."""
    res = analyze_trials(
        kappa_Nm_per_rad=1.234e-3,
        T_osc_trials_s=[1.500, 1.500, 1.500, 1.500, 1.500],
    )
    assert res.repeatability_pct == pytest.approx(0.0, abs=1e-9)
    assert res.repeatability_passed is True


def test_repeatability_fails_when_spread_above_3pct() -> None:
    """Spread the periods enough that the resulting I_wrist std/mean > 3%.

    I ∝ T², so a ±2.5% spread in T gives ~±5% spread in I. We want enough
    spread that the sample std/mean exceeds 3%.
    """
    res = analyze_trials(
        kappa_Nm_per_rad=1.0e-3,
        T_osc_trials_s=[1.20, 1.30, 1.40, 1.25, 1.35],
    )
    assert res.repeatability_pct > REPEATABILITY_GATE_PCT
    assert res.repeatability_passed is False
    assert res.passed is False


def test_per_trial_values_returned_in_order() -> None:
    trials = [1.50, 1.51, 1.49, 1.50, 1.50]
    res = analyze_trials(kappa_Nm_per_rad=1.234e-3, T_osc_trials_s=trials)
    assert len(res.per_trial_I_kgm2) == len(trials)
    for T_i, I_i in zip(trials, res.per_trial_I_kgm2, strict=True):
        assert math.isclose(I_i, i_wrist_from_period(1.234e-3, T_i), rel_tol=1e-12)


# ----- cross-check gate ------------------------------------------------------


def test_cross_check_passes_within_10pct() -> None:
    """Measured I within ±10% of the generator value → passes."""
    kappa = 1.234e-3
    T_measured = 1.50
    I_measured = i_wrist_from_period(kappa, T_measured)
    # Generator emits a value 5% above the measured — still inside the gate.
    res = analyze_trials(
        kappa_Nm_per_rad=kappa,
        T_osc_trials_s=[T_measured] * 5,
        generator_I_wrist_kgm2=I_measured * 1.05,
    )
    assert res.cross_check_pct == pytest.approx(100.0 * 0.05 / 1.05, abs=1e-9)
    assert res.cross_check_passed is True
    assert res.passed is True


def test_cross_check_fails_outside_10pct() -> None:
    kappa = 1.234e-3
    T_measured = 1.50
    I_measured = i_wrist_from_period(kappa, T_measured)
    res = analyze_trials(
        kappa_Nm_per_rad=kappa,
        T_osc_trials_s=[T_measured] * 5,
        generator_I_wrist_kgm2=I_measured * 1.20,  # 20% off
    )
    assert res.cross_check_pct > CROSS_CHECK_GATE_PCT
    assert res.cross_check_passed is False
    assert res.passed is False


def test_cross_check_optional_when_no_generator_value() -> None:
    """No generator value → cross_check fields are None; pass = repeatability only."""
    res = analyze_trials(
        kappa_Nm_per_rad=1.234e-3,
        T_osc_trials_s=[1.50, 1.501, 1.499, 1.500, 1.500],
    )
    assert res.cross_check_pct is None
    assert res.cross_check_passed is None
    assert res.passed is True
    assert res.repeatability_passed is True


def test_cross_check_rejects_nonpositive_generator() -> None:
    with pytest.raises(ValueError):
        analyze_trials(
            kappa_Nm_per_rad=1.0,
            T_osc_trials_s=[1.0, 1.01, 0.99],
            generator_I_wrist_kgm2=0.0,
        )


# ----- mount subtraction -----------------------------------------------------


def test_mount_inertia_subtracted_from_per_trial() -> None:
    """Mount-block contribution drops out of each per-trial I."""
    kappa = 1.0e-3
    T = 1.50
    mount = 1.0e-6
    res = analyze_trials(
        kappa_Nm_per_rad=kappa,
        T_osc_trials_s=[T] * 5,
        mount_I_wrist_kgm2=mount,
    )
    expected = i_wrist_from_period(kappa, T) - mount
    assert res.I_wrist_kgm2 == pytest.approx(expected, rel=1e-9)


def test_mount_too_large_errors() -> None:
    """Mount inertia exceeding the measured value → ValueError before stats."""
    kappa = 1.0e-3
    T = 1.50
    I_no_mount = i_wrist_from_period(kappa, T)
    with pytest.raises(ValueError):
        analyze_trials(
            kappa_Nm_per_rad=kappa,
            T_osc_trials_s=[T] * 5,
            mount_I_wrist_kgm2=I_no_mount * 2,
        )


# ----- realistic scenario ----------------------------------------------------


def test_realistic_spike_0_2_scenario() -> None:
    """End-to-end: calibrate κ from a known rod, then measure a 'fan'.

    Uses a known target I_fan = 5.0e-4 kg·m² and 5 trials with realistic
    1% jitter on T_osc; verifies the analyzer reports the right I and passes
    both gates.
    """
    rng = np.random.default_rng(0xFA0B)
    # Calibration: 50 g × 120 mm rod, T_ref measured cleanly.
    I_ref = rod_transverse_inertia(0.050, 0.120)
    T_ref = 1.0
    kappa = kappa_from_reference(I_ref_kgm2=I_ref, T_ref_s=T_ref)

    # Target fan: I = 5.0e-4 kg·m² → T_target = 2π·√(I/κ).
    I_fan = 5.0e-4
    T_target = 2.0 * math.pi * math.sqrt(I_fan / kappa)
    # 5 trials with 0.5% T jitter (well inside repeatability budget).
    T_trials = T_target * (1.0 + 0.005 * rng.standard_normal(5))

    # Generator predicts within 3% of truth.
    I_gen = I_fan * 1.03

    res = analyze_trials(
        kappa_Nm_per_rad=kappa,
        T_osc_trials_s=T_trials.tolist(),
        generator_I_wrist_kgm2=I_gen,
    )

    assert res.I_wrist_kgm2 == pytest.approx(I_fan, rel=0.02)
    assert res.repeatability_passed is True
    assert res.cross_check_passed is True
    assert res.passed is True
