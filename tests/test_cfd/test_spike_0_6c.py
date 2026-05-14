"""Tests for ``fanopt.cfd.spike_0_6c`` — the Tier-1 cfg sanity + NACA 0012
benchmark validation library that gates Phase 4 launch.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.6c (lines 1839-1844).
Lock reference: Round-9 HIGH-12 (= C12) for the unsteady MACH lock.
"""
from __future__ import annotations

import math

import pytest

from fanopt.cfd.spike_0_6c import (
    CONVERGENCE_METRICS,
    CONVERGENCE_TOLERANCE_PCT,
    MACH_UNSTEADY_LOCK,
    SYMMETRY_TOLERANCE_PCT,
    BenchmarkCycleData,
    Tier1CfgSanityResult,
    analyze_benchmark,
    analyze_spike_06c,
    check_convergence,
    check_symmetry,
    check_tier1_cfg_sanity,
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


# ---- sub-spike 0.6c.2 (NACA 0012 numerical consistency) -------------------


def _well_converged_cycles() -> list[BenchmarkCycleData]:
    """Four kept cycles with sub-1% variation and tight symmetry — gates pass."""
    # c_l_max ~ 0.70, c_l_min ~ -0.70, c_d_mean ~ 0.06, hysteresis ~ 0.1.
    # Each per-cycle value within ±0.5% of the cycle mean.
    return [
        _cycle(1, 0.700, -0.700, 0.060, 0.10),
        _cycle(2, 0.701, -0.699, 0.0604, 0.10),
        _cycle(3, 0.699, -0.701, 0.0598, 0.10),
        _cycle(4, 0.700, -0.700, 0.0602, 0.10),
    ]


def test_check_convergence_passes_for_tight_cycles() -> None:
    """Cycles within < 2% relative range pass each metric's convergence check."""
    checks = check_convergence(_well_converged_cycles())
    assert len(checks) == len(CONVERGENCE_METRICS)
    for c in checks:
        assert c.passed is True
        assert c.relative_range_pct < CONVERGENCE_TOLERANCE_PCT


def test_check_convergence_fails_when_range_exceeds_tolerance() -> None:
    """A metric whose relative range > 2% fails the convergence check."""
    cycles = [
        _cycle(1, 0.700, -0.700, 0.060, 0.10),
        _cycle(2, 0.800, -0.700, 0.060, 0.10),  # c_l_max bumped ~14% above
        _cycle(3, 0.700, -0.700, 0.060, 0.10),
        _cycle(4, 0.700, -0.700, 0.060, 0.10),
    ]
    checks = check_convergence(cycles)
    by_name = {c.metric_name: c for c in checks}
    assert by_name["c_l_max"].passed is False
    assert by_name["c_l_max"].relative_range_pct > CONVERGENCE_TOLERANCE_PCT
    assert by_name["c_l_min"].passed is True
    assert by_name["c_d_mean"].passed is True


def test_check_convergence_does_not_gate_hysteresis_area() -> None:
    """The hysteresis area is intentionally excluded from CONVERGENCE_METRICS.

    At k=0.55 the loop is near sign-inversion; a 2% relative-range gate on a
    near-zero quantity is numerically unstable. Hysteresis area lives in
    BenchmarkResult.diagnostic_hysteresis_area_mean (not gated).
    """
    assert "c_l_hysteresis_area" not in CONVERGENCE_METRICS


def test_check_symmetry_passes_when_c_l_max_equals_minus_c_l_min() -> None:
    """Symmetric airfoil + mean α = 0° → ⟨c_l_max⟩ ≈ -⟨c_l_min⟩."""
    sym = check_symmetry(_well_converged_cycles())
    assert sym.passed is True
    assert sym.asymmetry_pct < SYMMETRY_TOLERANCE_PCT
    assert sym.c_l_max_mean == pytest.approx(0.7, abs=1e-6)
    assert sym.c_l_min_mean == pytest.approx(-0.7, abs=1e-6)


def test_check_symmetry_fails_when_offsets_break_symmetry() -> None:
    """An 8% offset between |c_l_max| and |c_l_min| breaks the 5% symmetry gate."""
    cycles = [
        _cycle(1, 0.700, -0.640, 0.060, 0.10),
        _cycle(2, 0.701, -0.640, 0.060, 0.10),
        _cycle(3, 0.699, -0.641, 0.060, 0.10),
        _cycle(4, 0.700, -0.639, 0.060, 0.10),
    ]
    sym = check_symmetry(cycles)
    assert sym.passed is False
    assert sym.asymmetry_pct > SYMMETRY_TOLERANCE_PCT


def test_analyze_benchmark_discards_cycle_0_and_runs_gates_on_kept() -> None:
    """Cycle 0 is dropped as initial transient; gates apply to kept cycles."""
    bad_cycle = _cycle(0, 99.0, -99.0, 9.9, 99.0)  # initial transient
    good = _well_converged_cycles()
    result = analyze_benchmark(
        cycles=[bad_cycle, *good],
        k_reduced=0.55,
        reynolds=40_000,
    )
    assert result.passed is True
    assert result.convergence_passed is True
    assert result.symmetry_passed is True
    assert len(result.cycles) == 5
    # Convergence operated on kept cycles only — bad_cycle excluded.
    by_name = {c.metric_name: c for c in result.convergence}
    assert by_name["c_l_max"].values == (0.700, 0.701, 0.699, 0.700)


def test_analyze_benchmark_fails_if_convergence_fails() -> None:
    """A non-converged c_l_max across kept cycles fails the overall spike."""
    bad_cycle = _cycle(0, 99.0, -99.0, 9.9, 99.0)
    diverging = [
        _cycle(1, 0.700, -0.700, 0.060, 0.10),
        _cycle(2, 0.800, -0.700, 0.060, 0.10),
        _cycle(3, 0.900, -0.700, 0.060, 0.10),
        _cycle(4, 1.000, -0.700, 0.060, 0.10),
    ]
    result = analyze_benchmark(
        cycles=[bad_cycle, *diverging],
        k_reduced=0.55,
        reynolds=40_000,
    )
    assert result.passed is False
    assert result.convergence_passed is False


def test_analyze_benchmark_fails_if_symmetry_fails() -> None:
    """A converged-but-asymmetric run fails the overall spike via the symmetry gate."""
    bad_cycle = _cycle(0, 99.0, -99.0, 9.9, 99.0)
    asymmetric = [
        _cycle(1, 0.700, -0.500, 0.060, 0.10),
        _cycle(2, 0.701, -0.500, 0.060, 0.10),
        _cycle(3, 0.699, -0.500, 0.060, 0.10),
        _cycle(4, 0.700, -0.500, 0.060, 0.10),
    ]
    result = analyze_benchmark(
        cycles=[bad_cycle, *asymmetric],
        k_reduced=0.55,
        reynolds=40_000,
    )
    assert result.passed is False
    assert result.convergence_passed is True
    assert result.symmetry_passed is False


def test_analyze_benchmark_requires_more_than_one_cycle() -> None:
    """With < (discard + 1) cycles, the analyzer raises."""
    with pytest.raises(ValueError):
        analyze_benchmark(
            cycles=[_cycle(0, 1.0, -1.0, 0.05, 0.4)],
            k_reduced=0.55,
            reynolds=40_000,
        )


def test_convergence_tolerance_lock_is_2pct() -> None:
    """V1 lock: per-metric relative-range gate at 2%."""
    assert CONVERGENCE_TOLERANCE_PCT == 2.0


def test_symmetry_tolerance_lock_is_5pct() -> None:
    """V1 lock: C_L symmetry gate at 5%."""
    assert SYMMETRY_TOLERANCE_PCT == 5.0


def test_diagnostic_hysteresis_area_logged_but_not_gated() -> None:
    """The result exposes the cycle-mean hysteresis area for cross-solver use."""
    bad_cycle = _cycle(0, 99.0, -99.0, 9.9, 99.0)
    good = _well_converged_cycles()
    result = analyze_benchmark(
        cycles=[bad_cycle, *good],
        k_reduced=0.55,
        reynolds=40_000,
    )
    assert result.diagnostic_hysteresis_area_mean == pytest.approx(0.10, abs=1e-9)
    # And it didn't appear in convergence checks.
    assert all(c.metric_name != "c_l_hysteresis_area" for c in result.convergence)


# ---- aggregate Spike 0.6c -------------------------------------------------


def _passing_sub_1() -> Tier1CfgSanityResult:
    return check_tier1_cfg_sanity(_cfg_tier1_primary(), su2_log=_su2_log_with_outer_step(1))


def _passing_sub_2():
    bad_cycle = _cycle(0, 99.0, -99.0, 9.9, 99.0)
    return analyze_benchmark(
        cycles=[bad_cycle, *_well_converged_cycles()],
        k_reduced=0.55,
        reynolds=40_000,
    )


def _failing_sub_2():
    bad_cycle = _cycle(0, 99.0, -99.0, 9.9, 99.0)
    diverging = [
        _cycle(1, 0.700, -0.700, 0.060, 0.10),
        _cycle(2, 0.800, -0.700, 0.060, 0.10),
        _cycle(3, 0.900, -0.700, 0.060, 0.10),
        _cycle(4, 1.000, -0.700, 0.060, 0.10),
    ]
    return analyze_benchmark(
        cycles=[bad_cycle, *diverging],
        k_reduced=0.55,
        reynolds=40_000,
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
