"""Unit tests for fanopt.geometry.manufacturability.

Covers both:
- :class:`Layer4Params` BO parameter schema (bounds validation)
- :func:`run_manufacturability_filter` §N7 filter protocol (scaffold tier;
  geometry-level checks return PENDING_CADQUERY until Phase 1)
"""

from __future__ import annotations

import json

import pytest

from fanopt.geometry.manufacturability import (
    CLICK_CHAMFER_ANGLE_RANGE_DEG,
    CLICK_DESIGN_CLEARANCE_RANGE_M,
    LAYER_HEIGHTS_M,
    MANUFACTURABILITY_PASS_THRESHOLD,
    MODERATE_FAILURE_PENALTY,
    PRINT_ORIENTATIONS,
    SOFT_FAILURE_PENALTY,
    CheckResult,
    CheckSeverity,
    CheckStatus,
    Layer4Params,
    _aggregate_score,
    run_manufacturability_filter,
)
from fanopt.geometry.schema import DETENT_RADIUS_RANGE_M


def _canonical_kwargs() -> dict:
    return dict(
        print_orientation="flat",
        layer_height_m=0.0002,
        click_chamfer_angle_deg=45.0,
        click_detent_size_m=0.0004,
        click_design_clearance_m=0.00018,
    )


def test_canonical_validates() -> None:
    p = Layer4Params(**_canonical_kwargs())
    assert p.print_orientation == "flat"


def test_unknown_print_orientation_fails() -> None:
    kw = _canonical_kwargs()
    kw["print_orientation"] = "diagonal"
    with pytest.raises(ValueError, match="print_orientation"):
        Layer4Params(**kw)


def test_all_locked_print_orientations_pass() -> None:
    for po in PRINT_ORIENTATIONS:
        kw = _canonical_kwargs()
        kw["print_orientation"] = po
        Layer4Params(**kw)


def test_layer_height_not_in_locked_set_fails() -> None:
    """Layer height is a discrete categorical."""
    kw = _canonical_kwargs()
    kw["layer_height_m"] = 0.00025  # 0.25 mm — not in the locked set
    with pytest.raises(ValueError, match="layer_height_m"):
        Layer4Params(**kw)


def test_all_locked_layer_heights_pass() -> None:
    for lh in LAYER_HEIGHTS_M:
        kw = _canonical_kwargs()
        kw["layer_height_m"] = lh
        Layer4Params(**kw)


def test_chamfer_angle_below_30deg_fails() -> None:
    kw = _canonical_kwargs()
    kw["click_chamfer_angle_deg"] = 25.0
    with pytest.raises(ValueError, match="click_chamfer_angle_deg"):
        Layer4Params(**kw)


def test_chamfer_angle_above_60deg_fails() -> None:
    kw = _canonical_kwargs()
    kw["click_chamfer_angle_deg"] = 65.0
    with pytest.raises(ValueError, match="click_chamfer_angle_deg"):
        Layer4Params(**kw)


def test_chamfer_angle_at_bounds_inclusive() -> None:
    lo, hi = CLICK_CHAMFER_ANGLE_RANGE_DEG
    for ang in (lo, hi):
        kw = _canonical_kwargs()
        kw["click_chamfer_angle_deg"] = ang
        Layer4Params(**kw)


def test_detent_size_below_locked_range_fails() -> None:
    kw = _canonical_kwargs()
    kw["click_detent_size_m"] = 0.0001  # 0.1 mm < 0.3 mm floor
    with pytest.raises(ValueError, match="click_detent_size_m"):
        Layer4Params(**kw)


def test_detent_size_at_locked_bounds_pass() -> None:
    lo, hi = DETENT_RADIUS_RANGE_M
    for d in (lo, hi):
        kw = _canonical_kwargs()
        kw["click_detent_size_m"] = d
        Layer4Params(**kw)


def test_design_clearance_below_range_fails() -> None:
    kw = _canonical_kwargs()
    kw["click_design_clearance_m"] = 0.00010
    with pytest.raises(ValueError, match="click_design_clearance_m"):
        Layer4Params(**kw)


def test_design_clearance_at_bounds_pass() -> None:
    lo, hi = CLICK_DESIGN_CLEARANCE_RANGE_M
    for c in (lo, hi):
        kw = _canonical_kwargs()
        kw["click_design_clearance_m"] = c
        Layer4Params(**kw)


def test_round_trip_to_from_dict() -> None:
    p = Layer4Params(**_canonical_kwargs())
    d = p.to_dict()
    recovered = Layer4Params.from_dict(d)
    assert recovered == p


# ---- §9.7.3 manufacturability filter --------------------------------------


def test_filter_returns_all_protocol_checks() -> None:
    """The filter emits one CheckResult per §N7 row. Checks #9 (noise) and #10
    (TPMS) are removed with the porosity fields per V1-Slim S1."""
    result = run_manufacturability_filter({})
    check_ids = [c.check_id for c in result.checks]
    expected = {"1", "2", "3", "4", "5", "6", "7", "8", "11", "12", "13", "14"}
    assert set(check_ids) == expected


def test_filter_canonical_design_passes() -> None:
    """In the scaffold, geometry-level checks are PENDING_CADQUERY (no
    pass/fail signal) and hard bounds register as PASSED. Score = 1.0."""
    result = run_manufacturability_filter({})
    assert result.score == 1.0
    assert result.passed is True


def test_filter_severity_assignments_match_plan() -> None:
    """Plan §9.7.3 penalty mapping: critical={3,5,6,14}, moderate={1,2,12},
    soft={4,8,13}, hard_bound={7,11} (#9/#10 porosity checks cut per S1)."""
    result = run_manufacturability_filter({})
    by_id = {c.check_id: c for c in result.checks}
    critical_ids = {"3", "5", "6", "14"}
    moderate_ids = {"1", "2", "12"}
    soft_ids = {"4", "8", "13"}
    hard_bound_ids = {"7", "11"}
    for cid in critical_ids:
        assert by_id[cid].severity == CheckSeverity.CRITICAL
    for cid in moderate_ids:
        assert by_id[cid].severity == CheckSeverity.MODERATE
    for cid in soft_ids:
        assert by_id[cid].severity == CheckSeverity.SOFT
    for cid in hard_bound_ids:
        assert by_id[cid].severity == CheckSeverity.HARD_BOUND


def test_filter_hard_bounds_register_as_passed() -> None:
    """Plan §9.7.3: hard parameter bounds (#7, #11) are upstream-enforced and
    never reach the filter as failures — they record PASSED. (#9/#10 porosity
    bounds cut per S1.)"""
    result = run_manufacturability_filter({})
    by_id = {c.check_id: c for c in result.checks}
    for cid in ("7", "11"):
        assert by_id[cid].status == CheckStatus.PASSED


def test_filter_geometry_level_checks_pending() -> None:
    """All geometry-level checks return PENDING_CADQUERY in the scaffold."""
    result = run_manufacturability_filter({})
    pending_set = set(result.pending_cadquery)
    # Every non-hard-bound check should be pending until Phase 1 CadQuery
    # lands.
    expected_pending = {"1", "2", "3", "4", "5", "6", "8", "12", "13", "14"}
    assert pending_set == expected_pending


def test_filter_pass_threshold_is_0_5() -> None:
    """Plan §9.7.3 threshold lock — used by the BO loop to reject infeasible
    designs."""
    assert MANUFACTURABILITY_PASS_THRESHOLD == 0.5


def test_filter_result_is_json_serializable() -> None:
    """The filter result dict round-trips through json."""
    result = run_manufacturability_filter({})
    json.dumps(result.to_dict())


def test_filter_critical_failures_list_empty_in_scaffold() -> None:
    """No checks have FAILED status in the scaffold; critical_failures list
    is empty."""
    result = run_manufacturability_filter({})
    assert result.critical_failures == ()


# ---- _aggregate_score (scoring arithmetic) --------------------------------


def _failed_check(check_id: str, severity: CheckSeverity) -> CheckResult:
    """Synthetic FAILED CheckResult for scoring-arithmetic tests."""
    return CheckResult(
        check_id=check_id,
        name=f"synthetic check {check_id}",
        severity=severity,
        status=CheckStatus.FAILED,
        message="synthetic failure for scoring test",
    )


def test_aggregate_score_all_passed_is_1() -> None:
    """No FAILED checks → score stays at 1.0."""
    checks = (
        CheckResult("1", "n", CheckSeverity.HARD_BOUND, CheckStatus.PASSED, "m"),
        CheckResult("2", "n", CheckSeverity.HARD_BOUND, CheckStatus.PASSED, "m"),
    )
    score, crit, pending = _aggregate_score(checks)
    assert score == 1.0
    assert crit == []
    assert pending == []


def test_aggregate_score_one_moderate_subtracts_0_3() -> None:
    """Plan §9.7.3: a moderate failure subtracts 0.3."""
    checks = (_failed_check("1", CheckSeverity.MODERATE),)
    score, _crit, _pending = _aggregate_score(checks)
    assert score == pytest.approx(1.0 - MODERATE_FAILURE_PENALTY)


def test_aggregate_score_one_soft_subtracts_0_1() -> None:
    """Plan §9.7.3: a soft failure subtracts 0.1."""
    checks = (_failed_check("4", CheckSeverity.SOFT),)
    score, _crit, _pending = _aggregate_score(checks)
    assert score == pytest.approx(1.0 - SOFT_FAILURE_PENALTY)


def test_aggregate_score_two_moderates_plus_soft() -> None:
    """Penalties are additive: 2×0.3 + 0.1 → 0.3."""
    checks = (
        _failed_check("1", CheckSeverity.MODERATE),
        _failed_check("2", CheckSeverity.MODERATE),
        _failed_check("4", CheckSeverity.SOFT),
    )
    score, _crit, _pending = _aggregate_score(checks)
    assert score == pytest.approx(0.3, abs=1e-9)


def test_aggregate_score_critical_drives_to_zero() -> None:
    """Plan §9.7.3: any critical failure forces score = 0 regardless of other
    penalties."""
    checks = (
        _failed_check("3", CheckSeverity.CRITICAL),
        # Add a soft failure that would otherwise count.
        _failed_check("4", CheckSeverity.SOFT),
    )
    score, crit, _pending = _aggregate_score(checks)
    assert score == 0.0
    assert crit == ["3"]


def test_aggregate_score_multiple_criticals_all_listed() -> None:
    checks = (
        _failed_check("3", CheckSeverity.CRITICAL),
        _failed_check("5", CheckSeverity.CRITICAL),
        _failed_check("14", CheckSeverity.CRITICAL),
    )
    score, crit, _pending = _aggregate_score(checks)
    assert score == 0.0
    assert crit == ["3", "5", "14"]


def test_aggregate_score_clamps_negative_to_zero() -> None:
    """Many moderate failures would drive score below 0 without the clamp."""
    checks = tuple(_failed_check(str(i), CheckSeverity.MODERATE) for i in range(10))
    score, _crit, _pending = _aggregate_score(checks)
    assert score == 0.0


def test_aggregate_score_lists_pending_cadquery_check_ids() -> None:
    """PENDING_CADQUERY checks neither pass nor fail — listed for callers."""
    checks = (
        CheckResult(
            "2",
            "overhang",
            CheckSeverity.MODERATE,
            CheckStatus.PENDING_CADQUERY,
            "needs CadQuery",
        ),
        CheckResult(
            "5",
            "voids",
            CheckSeverity.CRITICAL,
            CheckStatus.PENDING_CADQUERY,
            "needs CadQuery",
        ),
    )
    score, crit, pending = _aggregate_score(checks)
    assert score == 1.0  # pending checks don't penalise
    assert crit == []
    assert pending == ["2", "5"]


def test_aggregate_score_hard_bound_failure_does_not_penalise() -> None:
    """Hard parameter bounds are upstream-enforced; a (hypothetical) FAILED
    hard-bound check should NOT subtract from the score (the comment in
    _aggregate_score documents this invariant)."""
    checks = (_failed_check("7", CheckSeverity.HARD_BOUND),)
    score, crit, _pending = _aggregate_score(checks)
    assert score == 1.0
    assert crit == []
