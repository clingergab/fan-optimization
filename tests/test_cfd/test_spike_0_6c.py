"""Tests for ``fanopt.cfd.spike_0_6c`` — the Tier-1 cfg sanity check that
gates Phase 4 launch (V1 scope).

Sub-spike 0.6c.2 (NACA 0012 numerical-consistency benchmark) was
deferred to Phase 5 on 2026-05-14 after the diagnostic confirmed the
moving-body-in-still-air cfg can't be validated against any published
wind-tunnel reference (see ``docs/phase_logs/spike_0_6c.md``). These
tests cover only sub-spike 0.6c.1 + the simplified aggregator.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.6c (lines 1839-1844).
Lock reference: Round-9 HIGH-12 (= C12) for the unsteady MACH lock.
"""

from __future__ import annotations

import math

import pytest

from fanopt.cfd.spike_0_6c import (
    MACH_UNSTEADY_LOCK,
    Tier1CfgSanityResult,
    analyze_spike_06c,
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


# ---- aggregate Spike 0.6c (V1: 0.6c.1 only) -------------------------------


def _passing_sub_1() -> Tier1CfgSanityResult:
    return check_tier1_cfg_sanity(_cfg_tier1_primary(), su2_log=_su2_log_with_outer_step(1))


def _failing_sub_1() -> Tier1CfgSanityResult:
    return check_tier1_cfg_sanity(_cfg_tier1_primary(), su2_log=None)


def test_spike_06c_passes_when_sub_1_passes() -> None:
    res = analyze_spike_06c(_passing_sub_1())
    assert res.sub_06c_1.passed is True
    assert res.overall_passed is True


def test_spike_06c_fails_when_sub_1_fails() -> None:
    res = analyze_spike_06c(_failing_sub_1())
    assert res.sub_06c_1.passed is False
    assert res.overall_passed is False
