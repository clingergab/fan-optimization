"""Single-blade fabrication-noise floor for J_fan deltas.

Implements Spike 0.5 (`docs/spike_0_5_protocol.md`,
`docs/plan_R11.md §Phase 0 Spike 0.5`).

The projected 15-30% J_fan gain from panel topology + airfoil optimization
must clear the printer's part-to-part noise floor. This spike measures that
floor: print three identical copies of a single representative blade on the
same printer with the same settings, assemble each into a 10-blade fan with
9 unchanged Spike 0.3 baseline blades, and measure the resulting
J_fan-proxy. The coefficient of variation (CV = std / mean × 100) across
the three single-blade fans is the noise floor against which every
subsequent J_fan delta must be compared.

Per-blade we also record three independent fabrication-quality signals so
the CV breakdown tells the operator *what* is varying (not just *that*
something is): mass on a jewelry scale, dimensional accuracy at 10 caliper
points, and three-point bend deflection under a known load. The roll-up
verdict (`overall_passed`) requires every metric to clear the 5% CV gate so
a metric-specific failure (e.g., bend stiffness varies 8% but mass varies
2%) is not papered over by a coincidentally tight J_fan CV.

Pass criterion (Spike 0.5):
- `CV_J_fan_pct < CV_GATE_PCT` (5%) across the three single-blade fans, AND
- mass, dimension, and bend-deflection CVs likewise each < 5%.

Mitigation if the J_fan CV exceeds 5%: tighten the print process (linear /
pressure advance per the Spike 0.4 fallback tree) or commit only to gains
> 15% (memo issue #16); the achieved CV is recorded in the Drive / JSONL
ledger and all downstream J_fan deltas are compared against it.

All functions are pure: no file I/O, no globals, no side effects. The CLI
wrapper (`scripts/run_spike_0_5.py`) handles CSV ingestion and results.json
emission.

References:
- Spec: `docs/plan_R11.md §Phase 0 Spike 0.5`
- Depends on: Spike 0.4 click-feature geometry (`src/fanopt/physical/click_rig.py`)
- Baseline J_fan proxy: Spike 0.3 (`scripts/run_spike_0_3_baseline.py`)
- Protocol: `docs/spike_0_5_protocol.md`
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable

__all__ = [
    "CV_GATE_PCT",
    "N_BLADES_REQUIRED",
    "coefficient_of_variation_pct",
    "BladeMeasurements",
    "PerMeasurementCV",
    "FabNoiseResult",
    "analyze_fab_noise",
]


# ─────────────────────────────────────────────────────────────────────
# Module-level constants (Spike 0.5 locks)
# ─────────────────────────────────────────────────────────────────────

# Per-metric CV pass criterion. Three single-blade fans must agree within
# this fraction of the mean (sample std / mean × 100). The spec sets it at
# 5%; tighter than the 15% lower-bound of the projected J_fan gain so the
# fabrication noise can't masquerade as a real improvement.
CV_GATE_PCT: float = 5.0

# Spec lock: three identical printed copies of a single representative
# blade. The blade-to-blade variation is the actual quantity of interest —
# printing three full fans would conflate blade noise with assembly noise.
N_BLADES_REQUIRED: int = 3


# ─────────────────────────────────────────────────────────────────────
# Coefficient of variation
# ─────────────────────────────────────────────────────────────────────


def coefficient_of_variation_pct(values: Iterable[float]) -> float:
    """Return `100 · std(ddof=1) / mean` for the given values.

    Sample standard deviation (ddof=1) is used because the three blades
    are a sample of the printer's fabrication-noise population, not the
    full population. With N = 3 this is what "CV across three copies"
    means physically — anything else under-estimates the spread.

    Parameters
    ----------
    values : iterable of floats. Must have at least 2 entries and the
        mean must be > 0; otherwise raises ValueError.

    Returns
    -------
    float : CV in percent (already multiplied by 100).
    """
    data = [float(v) for v in values]
    if len(data) < 2:
        raise ValueError(
            f"coefficient_of_variation_pct: need ≥ 2 values, got {len(data)}"
        )
    mean = sum(data) / len(data)
    if mean <= 0:
        raise ValueError(
            f"coefficient_of_variation_pct: mean must be > 0, got {mean}"
        )
    std = statistics.stdev(data)
    return 100.0 * std / mean


# ─────────────────────────────────────────────────────────────────────
# Per-blade measurement record
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BladeMeasurements:
    """One printed blade's fabrication-quality measurements.

    The blade is assembled into a 10-blade fan with 9 unchanged Spike 0.3
    baseline blades; `j_fan_proxy` is that assembled fan's anemometer
    plane-integral J_fan-proxy under the canonical Spike 0.3 protocol.
    """

    blade_id: int
    """Operator-assigned identifier (typically 1, 2, 3 for the three copies)."""

    mass_g: float
    """Mass of the printed single blade on the jewelry scale, grams."""

    dimension_mm_10pt: tuple[float, ...]
    """Caliper readings at 10 nominally identical points on the blade, mm.
    The 10-point convention is the spec — every dimension is measured at
    the same anatomical location across the three blades."""

    three_point_bend_deflection_mm: float
    """Deflection under the calibrated three-point bend load, mm."""

    j_fan_proxy: float
    """J_fan-proxy for the 10-blade fan assembled with this single blade +
    9 unchanged Spike 0.3 baseline blades, same units as Spike 0.3."""


# ─────────────────────────────────────────────────────────────────────
# Per-metric CV verdict
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PerMeasurementCV:
    """CV verdict for one metric across the three printed blades.

    Serializable via `dataclasses.asdict`. The CLI wrapper writes a table
    of these to stdout with ✓/✗ marks per metric.
    """

    metric_name: str
    """Operator-facing label, e.g., 'mass_g', 'j_fan_proxy', 'dimension_mm',
    'bend_deflection_mm'."""

    mean: float
    """Mean of the per-blade values."""

    std: float
    """Sample standard deviation (ddof=1) of the per-blade values."""

    cv_pct: float
    """`100 · std / mean`. Pass: `< CV_GATE_PCT` (5%)."""

    passed: bool
    """True iff `cv_pct < CV_GATE_PCT`."""


def _per_metric_cv(metric_name: str, values: Iterable[float]) -> PerMeasurementCV:
    """Build a PerMeasurementCV from raw values; helper for analyze_fab_noise."""
    data = [float(v) for v in values]
    mean = sum(data) / len(data)
    # `coefficient_of_variation_pct` validates len ≥ 2 and mean > 0.
    cv_pct = coefficient_of_variation_pct(data)
    std = statistics.stdev(data)
    return PerMeasurementCV(
        metric_name=metric_name,
        mean=mean,
        std=std,
        cv_pct=cv_pct,
        passed=cv_pct < CV_GATE_PCT,
    )


# ─────────────────────────────────────────────────────────────────────
# Spike 0.5 roll-up
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FabNoiseResult:
    """Top-level Spike 0.5 verdict.

    `overall_passed` requires every per-metric CV gate to clear: J_fan,
    mass, dimensions (averaged across the 10 caliper points), and bend
    deflection. A metric-specific failure flags *what* is varying so the
    mitigation step (tighten print process vs. raise the gain threshold)
    targets the right knob.

    Serializable via `dataclasses.asdict`. The CLI wrapper writes this to
    `data/spike_0_5/results.json` and exits 0 iff `overall_passed`.
    """

    per_blade: tuple[BladeMeasurements, ...]
    """Per-blade input rows, in input order, for the run log."""

    mass_cv: PerMeasurementCV
    """CV across the three printed-blade masses."""

    j_fan_cv: PerMeasurementCV
    """CV across the three J_fan-proxy values (the spec criterion)."""

    dimension_cv: PerMeasurementCV
    """Dimensional-CV roll-up. Per-point CV is computed for each of the
    10 caliper positions; the reported CV is the *mean* of those 10
    per-point CVs (mean / std / passed fields reflect that aggregation).
    See `analyze_fab_noise` for the exact aggregation rule."""

    bend_cv: PerMeasurementCV
    """CV across the three three-point bend deflections."""

    overall_passed: bool
    """True iff *every* metric's CV is below `CV_GATE_PCT` (5%)."""


def analyze_fab_noise(blades: Iterable[BladeMeasurements]) -> FabNoiseResult:
    """Compute the Spike 0.5 fabrication-noise verdict for three printed blades.

    Parameters
    ----------
    blades : iterable of `BladeMeasurements`. Must have exactly
        `N_BLADES_REQUIRED` (3) entries per spec — fewer raises
        ValueError. More are accepted (the spike permits a 4th or 5th
        copy for outlier diagnosis) but the canonical pass criterion is
        defined over three.

    Returns
    -------
    FabNoiseResult with per-metric CVs and overall pass/fail.

    Notes
    -----
    Dimensional-CV aggregation: for each of the 10 caliper points, compute
    the CV across the three blades. The reported `dimension_cv.cv_pct` is
    the *mean* of those 10 per-point CVs, and `passed` is True iff that
    mean is below `CV_GATE_PCT`. This catches systematic per-point
    variation (e.g., one corner of every blade is consistently 0.3 mm
    over-extruded) without being dominated by one rogue point. All 10
    points must share the same anatomical convention across blades —
    the protocol locks this.

    The `dimension_cv.mean` / `.std` fields report the cross-blade mean
    of the per-blade *mean* dimension and the std of those per-blade
    means, so they remain physically meaningful even though `cv_pct` is
    aggregated differently.
    """
    blade_list = tuple(blades)
    if len(blade_list) < N_BLADES_REQUIRED:
        raise ValueError(
            f"analyze_fab_noise: need ≥ {N_BLADES_REQUIRED} blades per spec, "
            f"got {len(blade_list)}"
        )

    # All blades must have the same number of dimension readings — otherwise
    # per-point CV aggregation is ill-defined.
    n_dims = len(blade_list[0].dimension_mm_10pt)
    if n_dims < 1:
        raise ValueError(
            "analyze_fab_noise: each blade must have ≥ 1 dimension reading, "
            "got 0 in blade_id=" + str(blade_list[0].blade_id)
        )
    for b in blade_list[1:]:
        if len(b.dimension_mm_10pt) != n_dims:
            raise ValueError(
                f"analyze_fab_noise: blade_id={b.blade_id} has "
                f"{len(b.dimension_mm_10pt)} dimension readings, "
                f"expected {n_dims} to match blade_id={blade_list[0].blade_id}"
            )

    masses = [b.mass_g for b in blade_list]
    j_fans = [b.j_fan_proxy for b in blade_list]
    bends = [b.three_point_bend_deflection_mm for b in blade_list]

    mass_cv = _per_metric_cv("mass_g", masses)
    j_fan_cv = _per_metric_cv("j_fan_proxy", j_fans)
    bend_cv = _per_metric_cv("bend_deflection_mm", bends)

    # Dimension CV: per-point CV across blades, then mean across the 10
    # points. Reported `mean` / `std` are derived from per-blade *mean*
    # dimensions so they remain interpretable on the run-log table.
    per_point_cvs: list[float] = []
    for j in range(n_dims):
        col = [b.dimension_mm_10pt[j] for b in blade_list]
        per_point_cvs.append(coefficient_of_variation_pct(col))
    aggregated_dim_cv = sum(per_point_cvs) / len(per_point_cvs)

    per_blade_mean_dim = [
        sum(b.dimension_mm_10pt) / len(b.dimension_mm_10pt) for b in blade_list
    ]
    dim_mean = sum(per_blade_mean_dim) / len(per_blade_mean_dim)
    dim_std = statistics.stdev(per_blade_mean_dim)
    dimension_cv = PerMeasurementCV(
        metric_name="dimension_mm",
        mean=dim_mean,
        std=dim_std,
        cv_pct=aggregated_dim_cv,
        passed=aggregated_dim_cv < CV_GATE_PCT,
    )

    overall_passed = (
        j_fan_cv.passed
        and mass_cv.passed
        and dimension_cv.passed
        and bend_cv.passed
    )

    return FabNoiseResult(
        per_blade=blade_list,
        mass_cv=mass_cv,
        j_fan_cv=j_fan_cv,
        dimension_cv=dimension_cv,
        bend_cv=bend_cv,
        overall_passed=overall_passed,
    )
