"""Unit tests for fanopt.physical.click_rig.

Validates the Spike 0.4 sub-gates against analytic-known inputs and
boundary cases. The H8 lever-arm lock (L_wrist_to_tip = 0.25 m) is pinned
by the first test — if anyone re-introduces 0.20 m from the pivot, this
test fails immediately.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.4; protocol in
docs/spike_0_4_protocol.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from fanopt.physical.click_rig import (
    ALIGNMENT_GAP_MAX_MM,
    ALPHA_MAX_RAD_PER_S2,
    CLEARANCE_MAX_MM,
    CLEARANCE_MIN_MM,
    CYCLE_TARGET,
    ENGAGEMENT_FORCE_MAX_N,
    ENGAGEMENT_FORCE_MIN_N,
    FORCE_BALANCE_SAFETY_FACTOR,
    HIGH_AMP_CYCLE_TARGET,
    HIGH_AMP_FORCE_MAX_N,
    HIGH_AMP_FORCE_MIN_N,
    L_WRIST_TO_TIP_M,
    Spike04Result,
    analyze_clearance,
    analyze_cycle_life,
    analyze_engagement_force,
    analyze_force_balance,
    force_balance_passes,
    inertial_force_at_click,
)

# ─────────────────────────────────────────────────────────────────────
# Constants pinning the H8 / H6 locks
# ─────────────────────────────────────────────────────────────────────


def test_module_constants_match_h8_h6_locks() -> None:
    """Spec locks: α_max = 110 rad/s² (H8/H6), L_wrist_to_tip = 0.25 m (H8)."""
    assert ALPHA_MAX_RAD_PER_S2 == 110.0
    assert L_WRIST_TO_TIP_M == 0.25
    assert FORCE_BALANCE_SAFETY_FACTOR == 2.0


def test_clearance_band_constants() -> None:
    """Per-mating-surface clearance band per Spike 0.4 spec."""
    assert CLEARANCE_MIN_MM == 0.15
    assert CLEARANCE_MAX_MM == 0.20


def test_engagement_force_band_constants() -> None:
    """Click engagement force bands per Spike 0.4 spec."""
    assert ENGAGEMENT_FORCE_MIN_N == 0.5
    assert ENGAGEMENT_FORCE_MAX_N == 2.0
    assert HIGH_AMP_FORCE_MIN_N == 1.0
    assert HIGH_AMP_FORCE_MAX_N == 4.0


def test_cycle_targets() -> None:
    """1000-cycle low-amp run + 100-cycle high-amp segment, < 1 mm alignment."""
    assert CYCLE_TARGET == 1000
    assert HIGH_AMP_CYCLE_TARGET == 100
    assert ALIGNMENT_GAP_MAX_MM == 1.0


# ─────────────────────────────────────────────────────────────────────
# inertial_force_at_click — H8 lever-arm lock pinned exactly
# ─────────────────────────────────────────────────────────────────────


def test_inertial_force_at_click_uses_h8_lever_arm() -> None:
    """`I=1e-3 kg·m²` × `α=110 rad/s²` / `0.25 m` = 0.44 N exactly.

    If anyone divides by 0.20 m (the pivot-to-tip distance) instead of
    0.25 m (the wrist-to-tip distance, H8 lock), this test fires.
    """
    F = inertial_force_at_click(I_wrist_kgm2=1.0e-3, alpha_max=110.0)
    assert pytest.approx(0.001 * 110.0 / 0.25, rel=1e-12) == F
    assert pytest.approx(0.44, rel=1e-12) == F


def test_inertial_force_at_click_custom_args() -> None:
    """Explicit alpha_max / lever_arm_m round-trip through F = τ / L."""
    I = 5.0e-4  # noqa: E741 -- moment of inertia (scientific convention)
    a = 200.0
    L = 0.30
    F = inertial_force_at_click(I_wrist_kgm2=I, alpha_max=a, lever_arm_m=L)
    assert pytest.approx(I * a / L, rel=1e-12) == F


def test_inertial_force_at_click_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        inertial_force_at_click(I_wrist_kgm2=0.0)
    with pytest.raises(ValueError):
        inertial_force_at_click(I_wrist_kgm2=1e-3, alpha_max=0.0)
    with pytest.raises(ValueError):
        inertial_force_at_click(I_wrist_kgm2=1e-3, lever_arm_m=0.0)


# ─────────────────────────────────────────────────────────────────────
# force_balance_passes / analyze_force_balance — H6 lock
# ─────────────────────────────────────────────────────────────────────


def test_force_balance_passes_at_exactly_2x() -> None:
    """Boundary case: cumulative friction == 2 × inertial → passes (≥, not >)."""
    F_in = 0.44
    assert force_balance_passes(F_friction_cumulative_N=0.88, F_inertial_at_click_N=F_in)


def test_force_balance_fails_just_below_2x() -> None:
    F_in = 0.44
    assert not force_balance_passes(F_friction_cumulative_N=0.88 - 1e-9, F_inertial_at_click_N=F_in)


def test_force_balance_arms_v1_fallback_when_failing() -> None:
    """Failing the H6 force balance auto-arms the V1 rib-tab fallback."""
    # I_wrist = 1e-3 kg·m² → F_inertial = 0.44 N → required friction 0.88 N.
    res = analyze_force_balance(
        I_wrist_kgm2=1.0e-3,
        F_friction_cumulative_N=0.50,  # well below 0.88 N
    )
    assert res.passed is False
    assert res.v1_lock_fallback_armed is True
    assert res.required_friction_N == pytest.approx(0.88, rel=1e-12)
    assert res.margin_ratio == pytest.approx(0.50 / 0.44, rel=1e-12)


def test_force_balance_does_not_arm_fallback_when_passing() -> None:
    res = analyze_force_balance(
        I_wrist_kgm2=1.0e-3,
        F_friction_cumulative_N=1.50,  # > 2 × 0.44 = 0.88
    )
    assert res.passed is True
    assert res.v1_lock_fallback_armed is False
    assert res.tau_inertial_peak_Nm == pytest.approx(1e-3 * 110.0, rel=1e-12)


# ─────────────────────────────────────────────────────────────────────
# Clearance band
# ─────────────────────────────────────────────────────────────────────


def test_clearance_band_boundary() -> None:
    """[0.149, 0.150, 0.200, 0.201] → only 0.149 and 0.201 out-of-band."""
    res = analyze_clearance([0.149, 0.150, 0.200, 0.201])
    assert res.n_measurements == 4
    assert res.out_of_band_count == 2
    oob_vals = [r.clearance_mm for r in res.out_of_band_rows]
    assert oob_vals == [0.149, 0.201]
    assert res.passed is False


def test_clearance_all_in_band_passes() -> None:
    res = analyze_clearance([0.16, 0.17, 0.18, 0.19])
    assert res.out_of_band_count == 0
    assert res.passed is True
    assert res.min_mm == 0.16
    assert res.max_mm == 0.19
    assert res.mean_mm == pytest.approx(0.175, rel=1e-12)


def test_clearance_requires_at_least_one_row() -> None:
    with pytest.raises(ValueError, match="≥ 1"):
        analyze_clearance([])


def test_clearance_labels_propagate() -> None:
    res = analyze_clearance([0.16, 0.18], labels=["blade1_blade2", "blade2_blade3"])
    assert [r.mating_surface for r in res.rows] == ["blade1_blade2", "blade2_blade3"]


def test_clearance_label_length_mismatch_errors() -> None:
    with pytest.raises(ValueError, match="labels length"):
        analyze_clearance([0.16, 0.18], labels=["only_one"])


# ─────────────────────────────────────────────────────────────────────
# Engagement force
# ─────────────────────────────────────────────────────────────────────


def test_engagement_force_low_regime_band() -> None:
    """All trials inside [0.5, 2.0] N → passes; one below or above → fails."""
    res = analyze_engagement_force([0.6, 1.0, 1.5, 1.9], high_amplitude=False)
    assert res.regime == "low"
    assert res.passed is True
    assert res.band_min_N == ENGAGEMENT_FORCE_MIN_N
    assert res.band_max_N == ENGAGEMENT_FORCE_MAX_N

    res_low_outlier = analyze_engagement_force([0.4, 1.0, 1.5], high_amplitude=False)
    assert res_low_outlier.passed is False
    assert res_low_outlier.out_of_band_count == 1

    res_high_outlier = analyze_engagement_force([1.0, 1.5, 2.1], high_amplitude=False)
    assert res_high_outlier.passed is False
    assert res_high_outlier.out_of_band_count == 1


def test_engagement_force_high_regime_band() -> None:
    """All trials inside [1.0, 4.0] N → passes; outlier → fails."""
    res = analyze_engagement_force([1.2, 2.0, 3.0, 3.8], high_amplitude=True)
    assert res.regime == "high"
    assert res.passed is True
    assert res.band_min_N == HIGH_AMP_FORCE_MIN_N
    assert res.band_max_N == HIGH_AMP_FORCE_MAX_N

    res_outlier = analyze_engagement_force([0.9, 2.0, 3.0], high_amplitude=True)
    assert res_outlier.passed is False
    assert res_outlier.out_of_band_count == 1


def test_engagement_force_mean_std() -> None:
    res = analyze_engagement_force([1.0, 1.0, 1.0], high_amplitude=False)
    assert res.mean_N == pytest.approx(1.0)
    assert res.std_N == pytest.approx(0.0)

    res2 = analyze_engagement_force([1.0, 2.0], high_amplitude=False)
    assert res2.mean_N == pytest.approx(1.5)
    # Sample stdev of [1, 2] is sqrt(0.5) ≈ 0.7071
    assert res2.std_N == pytest.approx(0.5**0.5, rel=1e-9)


def test_engagement_force_requires_at_least_one_trial() -> None:
    with pytest.raises(ValueError, match="≥ 1 trial"):
        analyze_engagement_force([], high_amplitude=False)


# ─────────────────────────────────────────────────────────────────────
# Cycle life
# ─────────────────────────────────────────────────────────────────────


def _clean_inspections(n_cycles: int) -> list[dict]:
    """Return inspections every 100 cycles up to n_cycles, all clean."""
    return [
        {"cycle": c, "wear_observed": False, "fracture": False, "notes": "clean"}
        for c in range(100, n_cycles + 1, 100)
    ]


def test_cycle_life_pass_when_no_wear_through_1000_cycles() -> None:
    res = analyze_cycle_life(
        inspections=_clean_inspections(1000),
        alignment_gap_variation_mm=0.3,
        high_amp_completed=True,
        high_amp_failure_cycle=None,
    )
    assert res.total_cycles_completed == 1000
    assert res.first_wear_cycle is None
    assert res.first_fracture_cycle is None
    assert res.low_amp_passed is True
    assert res.high_amp_passed is True
    assert res.alignment_passed is True
    assert res.passed is True


def test_cycle_life_wear_alone_does_not_fail_low_amp() -> None:
    """Wear is logged but only fracture fails the low-amp gate."""
    insp = _clean_inspections(1000)
    insp[5]["wear_observed"] = True  # cycle 600
    res = analyze_cycle_life(
        inspections=insp,
        alignment_gap_variation_mm=0.3,
        high_amp_completed=True,
        high_amp_failure_cycle=None,
    )
    assert res.first_wear_cycle == 600
    assert res.first_fracture_cycle is None
    assert res.low_amp_passed is True
    assert res.passed is True


def test_cycle_life_fail_when_fracture_before_1000() -> None:
    """Detent fracture at cycle 600 → low-amp gate fails → overall fail."""
    insp = _clean_inspections(1000)
    insp[5]["fracture"] = True  # cycle 600 has fracture=True
    res = analyze_cycle_life(
        inspections=insp,
        alignment_gap_variation_mm=0.3,
        high_amp_completed=True,
        high_amp_failure_cycle=None,
    )
    assert res.first_fracture_cycle == 600
    assert res.low_amp_passed is False
    assert res.passed is False


def test_cycle_life_fail_when_total_under_1000() -> None:
    """Cycling stopped early (no fracture but total < 1000) → low-amp fails."""
    res = analyze_cycle_life(
        inspections=_clean_inspections(700),
        alignment_gap_variation_mm=0.3,
        high_amp_completed=True,
        high_amp_failure_cycle=None,
    )
    assert res.total_cycles_completed == 700
    assert res.low_amp_passed is False
    assert res.passed is False


def test_cycle_life_high_amp_fracture_fails_segment() -> None:
    """Fracture inside the 100-cycle high-amplitude segment → segment fails."""
    res = analyze_cycle_life(
        inspections=_clean_inspections(1000),
        alignment_gap_variation_mm=0.3,
        high_amp_completed=False,
        high_amp_failure_cycle=42,
    )
    assert res.high_amp_passed is False
    assert res.passed is False


def test_cycle_life_high_amp_not_completed_fails_segment() -> None:
    res = analyze_cycle_life(
        inspections=_clean_inspections(1000),
        alignment_gap_variation_mm=0.3,
        high_amp_completed=False,
        high_amp_failure_cycle=None,
    )
    assert res.high_amp_passed is False
    assert res.passed is False


def test_alignment_gap_under_1mm_passes() -> None:
    """gap = 0.999 mm → passes; gap = 1.0 mm → fails (strictly <)."""
    res_pass = analyze_cycle_life(
        inspections=_clean_inspections(1000),
        alignment_gap_variation_mm=0.999,
        high_amp_completed=True,
        high_amp_failure_cycle=None,
    )
    assert res_pass.alignment_passed is True
    assert res_pass.passed is True

    res_fail = analyze_cycle_life(
        inspections=_clean_inspections(1000),
        alignment_gap_variation_mm=1.0,
        high_amp_completed=True,
        high_amp_failure_cycle=None,
    )
    assert res_fail.alignment_passed is False
    assert res_fail.passed is False


def test_cycle_life_rejects_missing_cycle_column() -> None:
    bad = [{"wear_observed": False, "fracture": False, "notes": ""}]
    with pytest.raises(ValueError, match="'cycle'"):
        analyze_cycle_life(
            inspections=bad,
            alignment_gap_variation_mm=0.3,
            high_amp_completed=True,
            high_amp_failure_cycle=None,
        )


# ─────────────────────────────────────────────────────────────────────
# Spike04Result serialization
# ─────────────────────────────────────────────────────────────────────


def test_spike_04_result_serializes_via_asdict() -> None:
    """asdict must produce a JSON-serializable nested dict."""
    fb = analyze_force_balance(
        I_wrist_kgm2=1.0e-3,
        F_friction_cumulative_N=1.50,
    )
    cl = analyze_clearance([0.16, 0.18])
    ef = analyze_engagement_force([0.8, 1.0, 1.2], high_amplitude=False)
    cy = analyze_cycle_life(
        inspections=_clean_inspections(1000),
        alignment_gap_variation_mm=0.3,
        high_amp_completed=True,
        high_amp_failure_cycle=None,
    )
    result = Spike04Result(
        force_balance=fb,
        clearance=cl,
        engagement_force=ef,
        cycle_life=cy,
        overall_passed=True,
        v1_lock_fallback_armed=fb.v1_lock_fallback_armed,
    )
    d = asdict(result)
    assert d["overall_passed"] is True
    assert d["v1_lock_fallback_armed"] is False
    assert d["force_balance"]["passed"] is True
    assert d["clearance"]["passed"] is True
    assert d["engagement_force"]["passed"] is True
    assert d["cycle_life"]["passed"] is True
    # Round-trip the dict through JSON to confirm nothing exotic leaks through.
    json.dumps(d)
