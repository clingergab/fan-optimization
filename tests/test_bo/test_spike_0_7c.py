"""Unit tests for fanopt.bo.spike_0_7c.

Spec reference: ``docs/plan_R11.md §Phase 0 Spike 0.7c``; protocol in
``docs/spike_0_7c_protocol.md``.

Covers:

* ``compute_iso_compute_point`` -- budget truncation, best-J_fan
  selection, 5% gate boundary.
* ``analyze_spike_07c`` -- pass rule "BO beats Sobol on >= 2 of 3
  budgets", and fallback recommendation routing.
* ``IsoComputePoint`` -- serialisable via ``dataclasses.asdict``.
"""
from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from fanopt.bo.spike_0_7c import (
    BO_MINUS_SOBOL_PCT_GATE,
    BUDGETS_HOURS,
    BUDGETS_PASS_THRESHOLD,
    BUDGETS_TOTAL_HOURS,
    IsoComputePoint,
    Spike07cResult,
    analyze_spike_07c,
    compute_iso_compute_point,
)


# ---- locks --------------------------------------------------------------


def test_locks_match_spec() -> None:
    """Sanity-check the constant values against the spec."""
    assert BUDGETS_HOURS == (30, 100, 300)
    assert BUDGETS_TOTAL_HOURS == 430
    assert BO_MINUS_SOBOL_PCT_GATE == 5.0
    assert BUDGETS_PASS_THRESHOLD == 2


# ---- compute_iso_compute_point -----------------------------------------


def test_compute_iso_compute_point_truncates_at_budget() -> None:
    """Cumulative wall_time exceeding the budget mid-stream truncates the prefix.

    Three Sobol records at 10 h each: budget=25 h fits the first two
    (cumulative 20 h) but not the third (cumulative 30 h > 25 h).
    """
    sobol = [
        {"j_fan": 5.0, "wall_time_hours": 10.0},
        {"j_fan": 3.0, "wall_time_hours": 10.0},
        # This third record would push cumulative to 30 h > 25 h, so it
        # MUST NOT be counted.
        {"j_fan": 1.0, "wall_time_hours": 10.0},
    ]
    bo = [
        {"j_fan": 4.0, "wall_time_hours": 10.0},
        {"j_fan": 2.0, "wall_time_hours": 10.0},
    ]
    pt = compute_iso_compute_point(25, sobol, bo)
    # Sobol best within budget = min(5.0, 3.0) = 3.0 (NOT 1.0).
    assert pt.sobol_best_j_fan == 3.0
    assert pt.sample_count["sobol"] == 2
    # BO best within budget = min(4.0, 2.0) = 2.0.
    assert pt.bo_best_j_fan == 2.0
    assert pt.sample_count["bo"] == 2


def test_compute_iso_compute_point_takes_best_j_fan() -> None:
    """Best-J_fan is the minimum (lower-is-better convention)."""
    sobol = [
        {"j_fan": 10.0, "wall_time_hours": 1.0},
        {"j_fan": 4.0, "wall_time_hours": 1.0},
        {"j_fan": 7.0, "wall_time_hours": 1.0},
    ]
    bo = [
        {"j_fan": 8.0, "wall_time_hours": 1.0},
        {"j_fan": 2.0, "wall_time_hours": 1.0},
        {"j_fan": 5.0, "wall_time_hours": 1.0},
    ]
    pt = compute_iso_compute_point(10, sobol, bo)
    assert pt.sobol_best_j_fan == 4.0
    assert pt.bo_best_j_fan == 2.0
    assert pt.sample_count == {"sobol": 3, "bo": 3}


def test_compute_iso_compute_point_empty_prefix_yields_inf() -> None:
    """If even the first record overruns, best is +inf and bo_beats is False."""
    sobol = [{"j_fan": 5.0, "wall_time_hours": 50.0}]
    bo = [{"j_fan": 1.0, "wall_time_hours": 50.0}]
    pt = compute_iso_compute_point(10, sobol, bo)
    assert pt.sobol_best_j_fan == float("inf")
    assert pt.bo_best_j_fan == float("inf")
    assert pt.bo_beats is False
    assert pt.sample_count == {"sobol": 0, "bo": 0}


def test_compute_iso_compute_point_rejects_negative_wall_time() -> None:
    """Negative wall_time per record is meaningless under the budget lock."""
    sobol = [{"j_fan": 5.0, "wall_time_hours": -1.0}]
    bo = [{"j_fan": 1.0, "wall_time_hours": 1.0}]
    with pytest.raises(ValueError, match="negative"):
        compute_iso_compute_point(10, sobol, bo)


def test_compute_iso_compute_point_custom_keys() -> None:
    """The j_fan / wall_time keys are configurable."""
    sobol = [{"loss": 5.0, "cost_hr": 1.0}]
    bo = [{"loss": 2.0, "cost_hr": 1.0}]
    pt = compute_iso_compute_point(
        10, sobol, bo, j_fan_key="loss", wall_time_key="cost_hr"
    )
    assert pt.sobol_best_j_fan == 5.0
    assert pt.bo_best_j_fan == 2.0


def test_bo_beats_sobol_when_5pct_better() -> None:
    """BO at 95% of Sobol's best is exactly the 5% gate boundary -- counts as a beat."""
    sobol = [{"j_fan": 100.0, "wall_time_hours": 1.0}]
    bo = [{"j_fan": 95.0, "wall_time_hours": 1.0}]
    pt = compute_iso_compute_point(10, sobol, bo)
    assert pt.bo_minus_sobol_pct == pytest.approx(5.0)
    assert pt.bo_beats is True


def test_bo_beats_sobol_boundary_at_exactly_5pct() -> None:
    """Strict >= 5% gate -- 4.999% does NOT beat; 5.001% does."""
    # Just below the gate.
    sobol_below = [{"j_fan": 100.0, "wall_time_hours": 1.0}]
    bo_below = [{"j_fan": 95.001, "wall_time_hours": 1.0}]
    pt_below = compute_iso_compute_point(10, sobol_below, bo_below)
    assert pt_below.bo_minus_sobol_pct < 5.0
    assert pt_below.bo_beats is False

    # Just above the gate.
    sobol_above = [{"j_fan": 100.0, "wall_time_hours": 1.0}]
    bo_above = [{"j_fan": 94.999, "wall_time_hours": 1.0}]
    pt_above = compute_iso_compute_point(10, sobol_above, bo_above)
    assert pt_above.bo_minus_sobol_pct > 5.0
    assert pt_above.bo_beats is True


def test_bo_loses_when_worse_than_sobol() -> None:
    """BO with higher J_fan than Sobol -- negative delta, bo_beats=False."""
    sobol = [{"j_fan": 100.0, "wall_time_hours": 1.0}]
    bo = [{"j_fan": 110.0, "wall_time_hours": 1.0}]
    pt = compute_iso_compute_point(10, sobol, bo)
    assert pt.bo_minus_sobol_pct == pytest.approx(-10.0)
    assert pt.bo_beats is False


# ---- analyze_spike_07c -- pass criterion ---------------------------------


def _pt(budget: int, *, bo_beats: bool, pct: float = 10.0) -> IsoComputePoint:
    """Build a synthetic IsoComputePoint for analyzer tests."""
    return IsoComputePoint(
        budget_hours=budget,
        sobol_best_j_fan=100.0,
        bo_best_j_fan=100.0 - pct,
        bo_minus_sobol_pct=pct if bo_beats else -pct,
        bo_beats=bo_beats,
        sample_count={"sobol": 10, "bo": 10},
    )


def test_analyze_passes_when_bo_beats_on_2_of_3_budgets() -> None:
    pts = [
        _pt(30, bo_beats=True),
        _pt(100, bo_beats=False),
        _pt(300, bo_beats=True),
    ]
    res = analyze_spike_07c(pts)
    assert isinstance(res, Spike07cResult)
    assert res.n_budgets_bo_beats == 2
    assert res.passed is True
    assert res.fallback_recommendation is None


def test_analyze_passes_when_bo_beats_on_all_3_budgets() -> None:
    pts = [
        _pt(30, bo_beats=True),
        _pt(100, bo_beats=True),
        _pt(300, bo_beats=True),
    ]
    res = analyze_spike_07c(pts)
    assert res.n_budgets_bo_beats == 3
    assert res.passed is True
    assert res.fallback_recommendation is None


def test_analyze_fails_when_bo_beats_on_only_1_of_3_budgets() -> None:
    pts = [
        _pt(30, bo_beats=True),
        _pt(100, bo_beats=False),
        _pt(300, bo_beats=False),
    ]
    res = analyze_spike_07c(pts)
    assert res.n_budgets_bo_beats == 1
    assert res.passed is False
    # Default fallback when no GP-fit-time labels supplied: retune.
    assert res.fallback_recommendation == "retune_acquisition"


def test_analyze_fails_when_bo_beats_on_0_of_3_budgets() -> None:
    pts = [
        _pt(30, bo_beats=False),
        _pt(100, bo_beats=False),
        _pt(300, bo_beats=False),
    ]
    res = analyze_spike_07c(pts)
    assert res.n_budgets_bo_beats == 0
    assert res.passed is False
    assert res.fallback_recommendation == "retune_acquisition"


# ---- analyze_spike_07c -- fallback routing ------------------------------


def test_fallback_recommends_saasbo_when_high_d_gp_fit_blocking() -> None:
    """High-D-routed labels -> SAASBO (with <=500 inducing points per §6.2.2)."""
    pts = [
        _pt(30, bo_beats=False),
        _pt(100, bo_beats=False),
        _pt(300, bo_beats=False),
    ]
    res = analyze_spike_07c(pts, gp_fit_time_above_60s_on=("high_d",))
    assert res.passed is False
    assert res.fallback_recommendation == "saasbo"


def test_fallback_recommends_saasbo_when_dimensionality_label_used() -> None:
    pts = [
        _pt(30, bo_beats=True),
        _pt(100, bo_beats=False),
        _pt(300, bo_beats=False),
    ]
    res = analyze_spike_07c(
        pts, gp_fit_time_above_60s_on=("dimensionality_blowup",)
    )
    assert res.passed is False
    assert res.fallback_recommendation == "saasbo"


def test_fallback_recommends_architecture_set_reduction_when_wide_set_blocking() -> None:
    """Wide-set-routed labels -> collapse Layer 2 categoricals."""
    pts = [
        _pt(30, bo_beats=False),
        _pt(100, bo_beats=True),
        _pt(300, bo_beats=False),
    ]
    res = analyze_spike_07c(
        pts, gp_fit_time_above_60s_on=("wide_architecture_set",)
    )
    assert res.passed is False
    assert res.fallback_recommendation == "fix_architecture_set"


def test_fallback_recommends_architecture_set_when_layer_2_label_used() -> None:
    pts = [
        _pt(30, bo_beats=False),
        _pt(100, bo_beats=False),
        _pt(300, bo_beats=False),
    ]
    res = analyze_spike_07c(
        pts, gp_fit_time_above_60s_on=("layer_2_categorical_blowup",)
    )
    assert res.passed is False
    assert res.fallback_recommendation == "fix_architecture_set"


def test_fallback_recommends_retune_when_no_gp_breach() -> None:
    """No 60-s-breach labels -> the BO simply under-exploited; retune is the first move."""
    pts = [
        _pt(30, bo_beats=True),
        _pt(100, bo_beats=False),
        _pt(300, bo_beats=False),
    ]
    res = analyze_spike_07c(pts, gp_fit_time_above_60s_on=())
    assert res.passed is False
    assert res.fallback_recommendation == "retune_acquisition"


# ---- serialisation ------------------------------------------------------


def test_iso_compute_point_serializes_via_asdict() -> None:
    """asdict produces a JSON-serialisable dict of the dataclass fields."""
    pt = IsoComputePoint(
        budget_hours=30,
        sobol_best_j_fan=100.0,
        bo_best_j_fan=90.0,
        bo_minus_sobol_pct=10.0,
        bo_beats=True,
        sample_count={"sobol": 10, "bo": 25},
    )
    d = asdict(pt)
    assert d["budget_hours"] == 30
    assert d["sobol_best_j_fan"] == 100.0
    assert d["bo_best_j_fan"] == 90.0
    assert d["bo_minus_sobol_pct"] == 10.0
    assert d["bo_beats"] is True
    assert d["sample_count"] == {"sobol": 10, "bo": 25}
    # Round-trips through json -- confirms no exotic types leaked into the dataclass.
    s = json.dumps(d, sort_keys=True)
    assert json.loads(s) == d
