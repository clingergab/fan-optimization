"""Unit tests for fanopt.physical.fab_noise.

Exercises the CV computation and the Spike 0.5 roll-up gates (J_fan, mass,
10-point dimension, three-point bend) against analytic-known inputs.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.5; protocol in
docs/spike_0_5_protocol.md.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict

import pytest

from fanopt.physical.fab_noise import (
    CV_GATE_PCT,
    N_BLADES_REQUIRED,
    BladeMeasurements,
    FabNoiseResult,
    PerMeasurementCV,
    analyze_fab_noise,
    coefficient_of_variation_pct,
)

# ─────────────────────────────────────────────────────────────────────
# Fixture helpers
# ─────────────────────────────────────────────────────────────────────


def _tight_dims(base_mm: float = 25.000) -> tuple[float, ...]:
    """10 caliper readings all equal to `base_mm` — zero CV at every point."""
    return tuple([base_mm] * 10)


def _make_blade(
    blade_id: int,
    mass_g: float = 5.00,
    dims: tuple[float, ...] | None = None,
    bend_mm: float = 1.20,
    j_fan: float = 0.350,
) -> BladeMeasurements:
    return BladeMeasurements(
        blade_id=blade_id,
        mass_g=mass_g,
        dimension_mm_10pt=dims if dims is not None else _tight_dims(),
        three_point_bend_deflection_mm=bend_mm,
        j_fan_proxy=j_fan,
    )


# ─────────────────────────────────────────────────────────────────────
# coefficient_of_variation_pct — analytic-known values
# ─────────────────────────────────────────────────────────────────────


def test_coefficient_of_variation_known_values() -> None:
    """Two anchors: zero spread → 0%; 8/10/12 → 20%.

    For [8, 10, 12]: mean = 10, sample std (ddof=1) = sqrt(((-2)² + 0² + 2²) / 2)
                              = sqrt(8 / 2) = 2.0 → CV = 100 * 2.0 / 10 = 20.0%.
    """
    assert coefficient_of_variation_pct([10.0, 10.0, 10.0]) == pytest.approx(0.0, abs=1e-12)
    assert coefficient_of_variation_pct([8.0, 10.0, 12.0]) == pytest.approx(20.0, abs=1e-9)


def test_cv_raises_on_zero_mean() -> None:
    """mean = 0 → ValueError (division would explode)."""
    with pytest.raises(ValueError, match="mean must be > 0"):
        coefficient_of_variation_pct([-1.0, 0.0, 1.0])


def test_cv_raises_on_negative_mean() -> None:
    """mean < 0 → ValueError (CV is undefined for negative-mean quantities)."""
    with pytest.raises(ValueError, match="mean must be > 0"):
        coefficient_of_variation_pct([-2.0, -1.0, -3.0])


def test_cv_raises_on_single_value() -> None:
    """N < 2 → ValueError (sample std needs ≥ 2)."""
    with pytest.raises(ValueError, match="need ≥ 2 values"):
        coefficient_of_variation_pct([5.0])


def test_cv_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="need ≥ 2 values"):
        coefficient_of_variation_pct([])


# ─────────────────────────────────────────────────────────────────────
# Spike 0.5 roll-up — happy path
# ─────────────────────────────────────────────────────────────────────


def test_fab_noise_passes_when_all_metrics_under_5pct() -> None:
    """Tight spread (< 1%) on every metric → overall_passed = True."""
    # ±0.2% mass, ±0.4% bend, ±0.3% J_fan — all comfortably under 5%.
    blades = [
        _make_blade(1, mass_g=5.000, bend_mm=1.200, j_fan=0.3500),
        _make_blade(2, mass_g=5.010, bend_mm=1.205, j_fan=0.3510),
        _make_blade(3, mass_g=4.990, bend_mm=1.195, j_fan=0.3490),
    ]
    result = analyze_fab_noise(blades)
    assert isinstance(result, FabNoiseResult)
    assert result.overall_passed is True
    assert result.mass_cv.passed is True
    assert result.j_fan_cv.passed is True
    assert result.dimension_cv.passed is True
    assert result.bend_cv.passed is True
    # Every CV strictly below the gate.
    for cv in (result.mass_cv, result.j_fan_cv, result.dimension_cv, result.bend_cv):
        assert cv.cv_pct < CV_GATE_PCT


def test_fab_noise_fails_when_j_fan_cv_over_5pct() -> None:
    """J_fan spread 0.30 / 0.40 / 0.50 → CV ≈ 25% → fails 5% gate."""
    blades = [
        _make_blade(1, j_fan=0.30),
        _make_blade(2, j_fan=0.40),
        _make_blade(3, j_fan=0.50),
    ]
    result = analyze_fab_noise(blades)
    assert result.j_fan_cv.passed is False
    assert result.j_fan_cv.cv_pct > CV_GATE_PCT
    assert result.overall_passed is False
    # The other metrics still pass on their own — the fail is metric-specific.
    assert result.mass_cv.passed is True
    assert result.dimension_cv.passed is True
    assert result.bend_cv.passed is True


def test_fab_noise_fails_when_mass_cv_over_5pct_even_if_j_fan_under() -> None:
    """A tight J_fan does NOT rescue a sloppy mass — overall gate is conjunctive."""
    blades = [
        _make_blade(1, mass_g=4.50, j_fan=0.3500),
        _make_blade(2, mass_g=5.00, j_fan=0.3505),
        _make_blade(3, mass_g=5.50, j_fan=0.3495),
    ]
    result = analyze_fab_noise(blades)
    assert result.j_fan_cv.passed is True
    assert result.mass_cv.passed is False
    assert result.mass_cv.cv_pct > CV_GATE_PCT
    assert result.overall_passed is False


def test_fab_noise_fails_when_bend_cv_over_5pct() -> None:
    """A tight J_fan does NOT rescue a sloppy bend deflection either."""
    blades = [
        _make_blade(1, bend_mm=1.00, j_fan=0.3500),
        _make_blade(2, bend_mm=1.20, j_fan=0.3505),
        _make_blade(3, bend_mm=1.40, j_fan=0.3495),
    ]
    result = analyze_fab_noise(blades)
    assert result.bend_cv.passed is False
    assert result.bend_cv.cv_pct > CV_GATE_PCT
    assert result.overall_passed is False


# ─────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────


def test_fab_noise_requires_at_least_3_blades() -> None:
    """N < N_BLADES_REQUIRED (3) → ValueError per spec."""
    blades = [_make_blade(1), _make_blade(2)]
    with pytest.raises(ValueError, match="need ≥ 3 blades"):
        analyze_fab_noise(blades)


def test_fab_noise_accepts_more_than_3_blades_for_outlier_diagnosis() -> None:
    """Spec floor is 3; 4 is permitted (operator may print an extra copy)."""
    blades = [
        _make_blade(1, mass_g=5.00),
        _make_blade(2, mass_g=5.01),
        _make_blade(3, mass_g=4.99),
        _make_blade(4, mass_g=5.00),
    ]
    result = analyze_fab_noise(blades)
    assert len(result.per_blade) == 4
    assert result.overall_passed is True


def test_fab_noise_rejects_ragged_dimension_arrays() -> None:
    """Per-point CV aggregation needs all blades to share the same 10 points."""
    blades = [
        BladeMeasurements(1, 5.0, _tight_dims(), 1.2, 0.35),
        BladeMeasurements(2, 5.0, tuple([25.0] * 9), 1.2, 0.35),  # 9 points, not 10
        BladeMeasurements(3, 5.0, _tight_dims(), 1.2, 0.35),
    ]
    with pytest.raises(ValueError, match="dimension readings"):
        analyze_fab_noise(blades)


def test_fab_noise_rejects_empty_dimension_array() -> None:
    blades = [
        BladeMeasurements(1, 5.0, (), 1.2, 0.35),
        BladeMeasurements(2, 5.0, (), 1.2, 0.35),
        BladeMeasurements(3, 5.0, (), 1.2, 0.35),
    ]
    with pytest.raises(ValueError, match="≥ 1 dimension reading"):
        analyze_fab_noise(blades)


# ─────────────────────────────────────────────────────────────────────
# Dimensional-CV aggregation
# ─────────────────────────────────────────────────────────────────────


def test_dimension_cv_aggregates_across_10_points() -> None:
    """The reported dimension CV is the mean of per-point CVs across blades.

    Construct three blades whose per-point CVs are known by hand:
      point 0: values [9.8, 10.0, 10.2]  → mean 10.0, std 0.2  → CV = 2.0%
      point 1: values [9.9, 10.0, 10.1]  → mean 10.0, std 0.1  → CV = 1.0%
      points 2-9: all 10.0                → CV = 0.0% each
    Mean per-point CV = (2.0 + 1.0 + 8 × 0.0) / 10 = 0.30%. Well under 5%.
    """
    blade1_dims = (9.8, 9.9, *tuple([10.0] * 8))
    blade2_dims = (10.0, 10.0, *tuple([10.0] * 8))
    blade3_dims = (10.2, 10.1, *tuple([10.0] * 8))
    blades = [
        BladeMeasurements(1, 5.0, blade1_dims, 1.2, 0.35),
        BladeMeasurements(2, 5.0, blade2_dims, 1.2, 0.35),
        BladeMeasurements(3, 5.0, blade3_dims, 1.2, 0.35),
    ]
    result = analyze_fab_noise(blades)
    assert result.dimension_cv.cv_pct == pytest.approx(0.30, abs=1e-6)
    assert result.dimension_cv.passed is True


def test_dimension_cv_fails_when_aggregate_over_5pct() -> None:
    """One badly varying point (10% CV) is enough to drive the aggregate > 5%
    when nine other points are clean only at the 0% level — but the spec's
    aggregation is the *mean*, so the failure must be quite large at one point.

    Build a per-point pattern where every point has CV = 8% → aggregate = 8%.
    """
    dims1 = tuple([9.2] * 10)
    dims2 = tuple([10.0] * 10)
    dims3 = tuple([10.8] * 10)
    # mean = 10, std = sqrt((0.8² + 0² + 0.8²) / 2) = 0.8 → CV = 8.0% per point.
    blades = [
        BladeMeasurements(1, 5.0, dims1, 1.2, 0.35),
        BladeMeasurements(2, 5.0, dims2, 1.2, 0.35),
        BladeMeasurements(3, 5.0, dims3, 1.2, 0.35),
    ]
    result = analyze_fab_noise(blades)
    assert result.dimension_cv.cv_pct == pytest.approx(8.0, abs=1e-6)
    assert result.dimension_cv.passed is False
    assert result.overall_passed is False


# ─────────────────────────────────────────────────────────────────────
# Serialization (downstream Phase-6 consumers expect asdict-safe records)
# ─────────────────────────────────────────────────────────────────────


def test_fab_noise_result_serializes_via_asdict() -> None:
    """`dataclasses.asdict` round-trips the full result without raising."""
    blades = [
        _make_blade(1),
        _make_blade(2, mass_g=5.005),
        _make_blade(3, mass_g=4.995),
    ]
    result = analyze_fab_noise(blades)
    d = asdict(result)
    # Top-level keys round-trip.
    assert set(d.keys()) == {
        "per_blade",
        "mass_cv",
        "j_fan_cv",
        "dimension_cv",
        "bend_cv",
        "overall_passed",
    }
    # Sub-records are dicts with the canonical CV fields.
    for key in ("mass_cv", "j_fan_cv", "dimension_cv", "bend_cv"):
        sub = d[key]
        assert set(sub.keys()) == {"metric_name", "mean", "std", "cv_pct", "passed"}
    # Per-blade list preserves input order and field names.
    assert len(d["per_blade"]) == 3
    for blade_dict, blade_obj in zip(d["per_blade"], blades, strict=True):
        assert blade_dict["blade_id"] == blade_obj.blade_id
        assert blade_dict["mass_g"] == blade_obj.mass_g
        # `dimension_mm_10pt` is a tuple on the dataclass; asdict preserves it
        # as a tuple per Python's dataclasses behaviour.
        assert tuple(blade_dict["dimension_mm_10pt"]) == blade_obj.dimension_mm_10pt


# ─────────────────────────────────────────────────────────────────────
# Constants sanity
# ─────────────────────────────────────────────────────────────────────


def test_constants_match_spec() -> None:
    """Hard-coded spec locks. If these change, the spec changed too."""
    assert CV_GATE_PCT == 5.0
    assert N_BLADES_REQUIRED == 3


def test_per_measurement_cv_is_frozen() -> None:
    cv = PerMeasurementCV(metric_name="x", mean=1.0, std=0.0, cv_pct=0.0, passed=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cv.cv_pct = 99.0  # type: ignore[misc]
