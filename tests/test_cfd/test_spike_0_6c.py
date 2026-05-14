"""Tests for ``fanopt.cfd.spike_0_6c`` — the Tier-1 cfg sanity + NACA 0012
benchmark validation library that gates Phase 4 launch.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.6c (lines 1839-1844).
Lock reference: Round-9 HIGH-12 (= C12) for the unsteady MACH lock.
"""
from __future__ import annotations

import math

import pytest

from fanopt.cfd.spike_0_6c import (
    BENCHMARK_TOLERANCE_PCT,
    MACH_UNSTEADY_LOCK,
    NACA0012_REFERENCE,
    BenchmarkCycleData,
    Tier1CfgSanityResult,
    analyze_benchmark,
    analyze_spike_06c,
    check_tier1_cfg_sanity,
    compare_cycle_to_reference,
)


# ---- helpers --------------------------------------------------------------


def _cfg_tier1_primary(mach: float = MACH_UNSTEADY_LOCK) -> str:
    """Return a synthetic Tier-1 cfg using the primary (FREESTREAM_VELOCITY) syntax."""
    return (
        "SOLVER= NAVIER_STOKES\n"
        "KIND_TURB_MODEL= NONE\n"
        f"MACH_NUMBER= {mach}\n"
        "FREESTREAM_OPTION= FREESTREAM_VELOCITY\n"
        "FREESTREAM_VELOCITY= 0.0 0.0 0.001\n"
        "FREESTREAM_TEMPERATURE= 300.0\n"
        "FREESTREAM_PRESSURE= 101325.0\n"
        "TIME_DOMAIN= YES\n"
        "TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER\n"
        "TIME_STEP= 0.0025\n"
        "GRID_MOVEMENT= RIGID_MOTION\n"
    )


def _cfg_tier1_fallback() -> str:
    """Return a synthetic Tier-1 cfg using the fallback (REF_DIMENSIONALIZATION) syntax."""
    return (
        "SOLVER= NAVIER_STOKES\n"
        f"MACH_NUMBER= {MACH_UNSTEADY_LOCK}\n"
        "REF_DIMENSIONALIZATION= FREESTREAM_PRESS_EQ_ONE\n"
        "FREESTREAM_TEMPERATURE= 300.0\n"
        "FREESTREAM_PRESSURE= 101325.0\n"
        "TIME_DOMAIN= YES\n"
        "TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER\n"
        "GRID_MOVEMENT= RIGID_MOTION\n"
    )


def _su2_log_with_outer_step(n: int = 1) -> str:
    """Return a synthetic SU2 stdout showing ``n`` completed outer time steps."""
    lines = [f"Time_Iter: {i}\n  INNER_ITER 0  RMS[Rho] -3.0\n" for i in range(n)]
    return "".join(lines)


def _cycle(idx: int, c_l_max: float, c_l_min: float, c_d_mean: float, area: float) -> BenchmarkCycleData:
    return BenchmarkCycleData(
        cycle_index=idx,
        c_l_max=c_l_max,
        c_l_min=c_l_min,
        c_d_mean=c_d_mean,
        c_l_hysteresis_area=area,
    )


# ---- sub-spike 0.6c.1 (Tier-1 cfg sanity) ---------------------------------


def test_tier1_cfg_sanity_pass_with_mach_1e_minus_9() -> None:
    """A cfg with MACH = 1e-9 + FREESTREAM_VELOCITY + 1 completed outer step PASSes."""
    cfg = _cfg_tier1_primary()
    log = _su2_log_with_outer_step(1)
    res = check_tier1_cfg_sanity(cfg, su2_log=log)
    assert res.passed is True
    assert res.parsed_ok is True
    assert res.mach_value == MACH_UNSTEADY_LOCK
    assert res.freestream_option == "FREESTREAM_VELOCITY"
    assert res.outer_time_steps_completed >= 1
    assert res.error is None


def test_tier1_cfg_sanity_fail_with_steady_mach() -> None:
    """A cfg with MACH = 0.0064 (the STEADY-tier value) FAILs the unsteady sanity check.

    Round-9 HIGH-12 retired the 0.0064 value for unsteady cfgs — it remains
    correct for Tier -1 / Tier 0 steady cfgs only.
    """
    cfg = _cfg_tier1_primary(mach=0.0064)
    log = _su2_log_with_outer_step(1)
    res = check_tier1_cfg_sanity(cfg, su2_log=log)
    assert res.passed is False
    assert res.parsed_ok is True
    assert res.mach_value == pytest.approx(0.0064)


def test_tier1_cfg_sanity_accepts_either_freestream_option_or_ref_dimensionalization() -> None:
    """Both H10 syntaxes (primary + fallback) pass when the rest of the gate clears."""
    log = _su2_log_with_outer_step(1)

    primary = check_tier1_cfg_sanity(_cfg_tier1_primary(), su2_log=log)
    fallback = check_tier1_cfg_sanity(_cfg_tier1_fallback(), su2_log=log)

    assert primary.passed is True
    assert primary.freestream_option == "FREESTREAM_VELOCITY"
    assert primary.ref_dimensionalization is None

    assert fallback.passed is True
    assert fallback.freestream_option == ""
    assert fallback.ref_dimensionalization == "FREESTREAM_PRESS_EQ_ONE"


def test_tier1_cfg_sanity_fail_when_no_su2_log_passed() -> None:
    """Without an SU2 log the outer-step gate cannot be cleared — the
    cfg-only fallback path reports `passed=False` by design."""
    cfg = _cfg_tier1_primary()
    res = check_tier1_cfg_sanity(cfg, su2_log=None)
    assert res.passed is False
    assert res.outer_time_steps_completed == 0
    # Other parts of the check still pass.
    assert res.parsed_ok is True
    assert res.mach_value == MACH_UNSTEADY_LOCK


def test_tier1_cfg_sanity_fail_when_mach_directive_missing() -> None:
    """A cfg missing MACH_NUMBER produces a parse error and FAILs."""
    cfg = "SOLVER= NAVIER_STOKES\nKIND_TURB_MODEL= NONE\n"
    res = check_tier1_cfg_sanity(cfg, su2_log=_su2_log_with_outer_step(1))
    assert res.passed is False
    assert res.parsed_ok is False
    assert res.error is not None
    assert math.isnan(res.mach_value)


# ---- sub-spike 0.6c.2 (NACA 0012 benchmark) -------------------------------


def test_compare_cycle_under_15pct_passes() -> None:
    """A 5% deviation on every metric passes."""
    ref = {"c_l_max": 1.20, "c_l_min": -1.20, "c_d_mean": 0.085, "c_l_hysteresis_area": 0.45}
    measured = _cycle(
        idx=-1,
        c_l_max=1.26,            # +5%
        c_l_min=-1.14,           # -5%
        c_d_mean=0.08925,        # +5%
        area=0.4725,             # +5%
    )
    comps = compare_cycle_to_reference(measured, ref)
    assert len(comps) == 4
    assert all(c.passed for c in comps)


def test_compare_cycle_over_15pct_fails() -> None:
    """A 20% deviation on one metric flips that metric's comparison to FAIL."""
    ref = {"c_l_max": 1.20, "c_l_min": -1.20, "c_d_mean": 0.085, "c_l_hysteresis_area": 0.45}
    measured = _cycle(
        idx=-1,
        c_l_max=1.44,            # +20% -> FAIL
        c_l_min=-1.14,           # -5%
        c_d_mean=0.08925,        # +5%
        area=0.4725,             # +5%
    )
    comps = compare_cycle_to_reference(measured, ref)
    by_name = {c.metric_name: c for c in comps}
    assert by_name["c_l_max"].passed is False
    assert by_name["c_l_min"].passed is True
    assert by_name["c_d_mean"].passed is True
    assert by_name["c_l_hysteresis_area"].passed is True


def test_analyze_benchmark_discards_first_cycle_integrates_last_4() -> None:
    """The first cycle is dropped; the integrated value reflects the last 4."""
    # Cycle 0 has wildly wrong numbers (initial transient); cycles 1-4 all
    # match the reference exactly, so after dropping cycle 0 the integration
    # gives the reference value back.
    ref = dict(NACA0012_REFERENCE)
    bad_cycle = _cycle(0, 99.0, -99.0, 9.9, 99.0)
    good_cycles = [
        _cycle(
            i,
            ref["c_l_max"],
            ref["c_l_min"],
            ref["c_d_mean"],
            ref["c_l_hysteresis_area"],
        )
        for i in range(1, 5)
    ]
    result = analyze_benchmark(
        cycles=[bad_cycle, *good_cycles],
        reference=ref,
        k_reduced=0.55,
        reynolds=40_000,
        reference_source="test",
    )
    assert result.passed is True
    assert result.all_metrics_within_15pct is True
    assert len(result.cycles) == 5  # raw cycles preserved
    # And the comparisons reflect the integrated value matching the ref.
    for c in result.comparisons:
        assert abs(c.pct_diff) < 1e-6


def test_analyze_benchmark_fails_if_any_metric_over_15pct() -> None:
    """Even one over-tolerance metric fails the whole benchmark."""
    ref = dict(NACA0012_REFERENCE)
    cycles = [
        _cycle(0, 99.0, -99.0, 9.9, 99.0),  # dropped
        # The remaining four cycles each have c_l_max 20% too high; everything
        # else matches.
        _cycle(1, ref["c_l_max"] * 1.20, ref["c_l_min"], ref["c_d_mean"], ref["c_l_hysteresis_area"]),
        _cycle(2, ref["c_l_max"] * 1.20, ref["c_l_min"], ref["c_d_mean"], ref["c_l_hysteresis_area"]),
        _cycle(3, ref["c_l_max"] * 1.20, ref["c_l_min"], ref["c_d_mean"], ref["c_l_hysteresis_area"]),
        _cycle(4, ref["c_l_max"] * 1.20, ref["c_l_min"], ref["c_d_mean"], ref["c_l_hysteresis_area"]),
    ]
    result = analyze_benchmark(
        cycles=cycles,
        reference=ref,
        k_reduced=0.55,
        reynolds=40_000,
        reference_source="test",
    )
    assert result.passed is False
    assert result.all_metrics_within_15pct is False
    by_name = {c.metric_name: c for c in result.comparisons}
    assert by_name["c_l_max"].passed is False
    assert by_name["c_l_min"].passed is True


def test_analyze_benchmark_requires_more_than_one_cycle() -> None:
    """With < (discard + 1) cycles, the analyzer raises."""
    ref = dict(NACA0012_REFERENCE)
    with pytest.raises(ValueError):
        analyze_benchmark(
            cycles=[_cycle(0, 1.0, -1.0, 0.05, 0.4)],
            reference=ref,
            k_reduced=0.55,
            reynolds=40_000,
            reference_source="test",
        )


def test_tolerance_lock_is_15_pct() -> None:
    """Spike 0.6c locks the tolerance at ±15% per spec line 1843."""
    assert BENCHMARK_TOLERANCE_PCT == 15.0


# ---- aggregate Spike 0.6c -------------------------------------------------


def _passing_sub_1() -> Tier1CfgSanityResult:
    return check_tier1_cfg_sanity(_cfg_tier1_primary(), su2_log=_su2_log_with_outer_step(1))


def _passing_sub_2():
    ref = dict(NACA0012_REFERENCE)
    cycles = [_cycle(0, 99.0, -99.0, 9.9, 99.0)] + [
        _cycle(i, ref["c_l_max"], ref["c_l_min"], ref["c_d_mean"], ref["c_l_hysteresis_area"])
        for i in range(1, 5)
    ]
    return analyze_benchmark(
        cycles=cycles, reference=ref, k_reduced=0.55, reynolds=40_000, reference_source="test"
    )


def _failing_sub_2():
    ref = dict(NACA0012_REFERENCE)
    cycles = [_cycle(0, 99.0, -99.0, 9.9, 99.0)] + [
        _cycle(
            i,
            ref["c_l_max"] * 1.20,
            ref["c_l_min"],
            ref["c_d_mean"],
            ref["c_l_hysteresis_area"],
        )
        for i in range(1, 5)
    ]
    return analyze_benchmark(
        cycles=cycles, reference=ref, k_reduced=0.55, reynolds=40_000, reference_source="test"
    )


def test_spike_06c_passes_when_both_sub_spikes_pass() -> None:
    res = analyze_spike_06c(_passing_sub_1(), _passing_sub_2())
    assert res.sub_06c_1.passed is True
    assert res.sub_06c_2.passed is True
    assert res.overall_passed is True


def test_spike_06c_fails_if_either_sub_spike_fails() -> None:
    # sub_1 PASS, sub_2 FAIL -> overall FAIL
    res = analyze_spike_06c(_passing_sub_1(), _failing_sub_2())
    assert res.sub_06c_1.passed is True
    assert res.sub_06c_2.passed is False
    assert res.overall_passed is False

    # sub_1 FAIL (no SU2 log) + sub_2 PASS -> overall FAIL
    sub_1_fail = check_tier1_cfg_sanity(_cfg_tier1_primary(), su2_log=None)
    res2 = analyze_spike_06c(sub_1_fail, _passing_sub_2())
    assert res2.sub_06c_1.passed is False
    assert res2.sub_06c_2.passed is True
    assert res2.overall_passed is False
