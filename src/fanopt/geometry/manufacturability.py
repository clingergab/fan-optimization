"""Geometry Layer 4 — manufacturing + click features + §N7 filter.

Two responsibilities, both under plan §9.7 / §N7:

1. :class:`Layer4Params` — BO design-parameter schema for the
   manufacturing categoricals (print orientation, layer height) and
   panel-edge click parameters. Schema validation at construction.

2. :func:`run_manufacturability_filter` — the §9.7.3 11-check (now 14+
   row, post-rounds-12/13/14) manufacturability filter that downstream
   geometry consumers run on the assembled blade description. Schema-
   level checks (#9, #10, #11) are upstream-enforced by the layer
   dataclasses so they never reach the filter. Geometry-level checks
   (#1, #2, #3, #4, #5, #6, #7, #8, #12, #13, #14) require CadQuery
   shapes; today's scaffold marks them as ``CheckStatus.PENDING_CADQUERY``
   so the filter protocol is complete while the actual shape inspection
   waits on Phase 1 generator code.

Scoring (plan §9.7.3):

- Critical failures (#3, #5, #6, #14): immediate score = 0.
- Moderate failures (#1, #2, #12): each −0.3.
- Soft failures (#4, #8, #13): each −0.1.
- Hard parameter bounds (#7, #9, #10, #11): no penalty step (upstream).

Threshold: score ≥ 0.5 ⇒ printable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from fanopt.geometry.schema import DETENT_RADIUS_RANGE_M

__all__ = [
    "PRINT_ORIENTATIONS",
    "LAYER_HEIGHTS_M",
    "CLICK_CHAMFER_ANGLE_RANGE_DEG",
    "CLICK_DESIGN_CLEARANCE_RANGE_M",
    "MANUFACTURABILITY_PASS_THRESHOLD",
    "CRITICAL_FAILURE_PENALTY",
    "MODERATE_FAILURE_PENALTY",
    "SOFT_FAILURE_PENALTY",
    "Layer4Params",
    "CheckSeverity",
    "CheckStatus",
    "CheckResult",
    "ManufacturabilityResult",
    "run_manufacturability_filter",
    "_aggregate_score",
]


PRINT_ORIENTATIONS: tuple[str, ...] = ("flat", "edge", "custom-angle")
"""Plan §6.2.1 + §7.4.3. ``flat`` (= rib-flat) is the default and triggers
plano-convex camber on Layer 1 (enforced by :class:`BladeDesignParams`)."""

LAYER_HEIGHTS_M: tuple[float, ...] = (0.0001, 0.00015, 0.0002)
"""{0.1, 0.15, 0.2} mm slicer setting — discrete categorical."""

CLICK_CHAMFER_ANGLE_RANGE_DEG: tuple[float, float] = (30.0, 60.0)
"""Plan §6.2.1: '30-60°'."""

CLICK_DESIGN_CLEARANCE_RANGE_M: tuple[float, float] = (0.00015, 0.00020)
"""0.15-0.20 mm per mating surface."""


# ---------------------------------------------------------------------------
# Manufacturability filter — scoring constants (plan §9.7.3)
# ---------------------------------------------------------------------------

MANUFACTURABILITY_PASS_THRESHOLD: float = 0.5
"""Plan §9.7.3 scoring: ``< 0.5`` is infeasible (design rejected)."""

CRITICAL_FAILURE_PENALTY: float = 1.0
"""Critical failures (#3, #5, #6, #14) drive score to 0 directly. Encoded
as a full-score subtraction floor here for uniformity with the soft/moderate
penalty pattern."""

MODERATE_FAILURE_PENALTY: float = 0.3
"""Each moderate failure (#1, #2, #12) subtracts 0.3."""

SOFT_FAILURE_PENALTY: float = 0.1
"""Each soft failure (#4, #8, #13) subtracts 0.1."""


@dataclass(frozen=True)
class Layer4Params:
    """Layer 4 design parameters — manufacturing + click features."""

    print_orientation: str
    layer_height_m: float
    click_chamfer_angle_deg: float
    click_detent_size_m: float
    click_design_clearance_m: float

    def __post_init__(self) -> None:
        if self.print_orientation not in PRINT_ORIENTATIONS:
            raise ValueError(
                f"print_orientation must be one of {PRINT_ORIENTATIONS}, "
                f"got {self.print_orientation!r}"
            )
        if not any(abs(self.layer_height_m - lh) < 1e-9 for lh in LAYER_HEIGHTS_M):
            raise ValueError(
                f"layer_height_m = {self.layer_height_m} must match one of "
                f"{LAYER_HEIGHTS_M} (mm: 0.1 / 0.15 / 0.2)"
            )
        ang_lo, ang_hi = CLICK_CHAMFER_ANGLE_RANGE_DEG
        if not (ang_lo <= self.click_chamfer_angle_deg <= ang_hi):
            raise ValueError(
                f"click_chamfer_angle_deg = {self.click_chamfer_angle_deg} "
                f"outside range [{ang_lo}, {ang_hi}]"
            )
        det_lo, det_hi = DETENT_RADIUS_RANGE_M
        if not (det_lo <= self.click_detent_size_m <= det_hi):
            raise ValueError(
                f"click_detent_size_m = {self.click_detent_size_m} outside "
                f"locked range {DETENT_RADIUS_RANGE_M}"
            )
        cl_lo, cl_hi = CLICK_DESIGN_CLEARANCE_RANGE_M
        if not (cl_lo <= self.click_design_clearance_m <= cl_hi):
            raise ValueError(
                f"click_design_clearance_m = {self.click_design_clearance_m} "
                f"outside range [{cl_lo}, {cl_hi}]"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "print_orientation": self.print_orientation,
            "layer_height_m": self.layer_height_m,
            "click_chamfer_angle_deg": self.click_chamfer_angle_deg,
            "click_detent_size_m": self.click_detent_size_m,
            "click_design_clearance_m": self.click_design_clearance_m,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Layer4Params:
        return cls(
            print_orientation=str(d["print_orientation"]),
            layer_height_m=float(d["layer_height_m"]),
            click_chamfer_angle_deg=float(d["click_chamfer_angle_deg"]),
            click_detent_size_m=float(d["click_detent_size_m"]),
            click_design_clearance_m=float(d["click_design_clearance_m"]),
        )


# ---------------------------------------------------------------------------
# Manufacturability filter — protocol (Phase 1 fills the geometry checks)
# ---------------------------------------------------------------------------


class CheckSeverity(str, Enum):
    """Penalty class per plan §9.7.3."""

    CRITICAL = "critical"
    MODERATE = "moderate"
    SOFT = "soft"
    HARD_BOUND = "hard_bound"
    """Upstream-enforced parameter bound; the filter records its existence
    but never penalises (violations cannot reach the filter)."""


class CheckStatus(str, Enum):
    """Per-check evaluation outcome."""

    PASSED = "passed"
    FAILED = "failed"
    PENDING_CADQUERY = "pending_cadquery"
    """Geometry-level check that requires actual shape inspection; the
    scaffold cannot evaluate it without CadQuery."""


@dataclass(frozen=True)
class CheckResult:
    """One §N7 manufacturability check's outcome."""

    check_id: str
    name: str
    severity: CheckSeverity
    status: CheckStatus
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "name": self.name,
            "severity": self.severity.value,
            "status": self.status.value,
            "message": self.message,
        }


@dataclass(frozen=True)
class ManufacturabilityResult:
    """Aggregate filter outcome.

    ``score`` is the plan §9.7.3 manufacturability_score in [0, 1].
    ``passed = score >= MANUFACTURABILITY_PASS_THRESHOLD``.

    ``critical_failures`` lists check_ids that drove score to 0. If any
    fired, ``score = 0`` regardless of moderate/soft penalties.
    """

    score: float
    passed: bool
    checks: tuple[CheckResult, ...]
    critical_failures: tuple[str, ...]
    pending_cadquery: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "critical_failures": list(self.critical_failures),
            "pending_cadquery": list(self.pending_cadquery),
        }


# ---- per-check stubs ------------------------------------------------------


def _pending_check(check_id: str, name: str, severity: CheckSeverity, message: str) -> CheckResult:
    """A geometry-level check that needs Phase-1 CadQuery code to evaluate.

    The scaffold returns ``PENDING_CADQUERY`` so the filter protocol is
    complete; the eventual implementation replaces these stubs with real
    shape inspection.
    """
    return CheckResult(
        check_id=check_id,
        name=name,
        severity=severity,
        status=CheckStatus.PENDING_CADQUERY,
        message=message,
    )


def _bound_check(check_id: str, name: str, message: str) -> CheckResult:
    """A hard parameter bound that's enforced upstream by the layer
    dataclasses' ``__post_init__``. The filter records its existence for
    audit traceability."""
    return CheckResult(
        check_id=check_id,
        name=name,
        severity=CheckSeverity.HARD_BOUND,
        status=CheckStatus.PASSED,
        message=message,
    )


def run_manufacturability_filter(
    geometry_description: dict[str, Any],
) -> ManufacturabilityResult:
    """Apply the §9.7.3 11+row manufacturability filter to a blade description.

    ``geometry_description`` is the dict produced by
    :func:`fanopt.geometry.generator.generate_blade` — it carries the
    per-layer descriptions, the panel-domain mask, and the parameter
    trace. Today's scaffold runs the protocol; geometry-level checks
    return :attr:`CheckStatus.PENDING_CADQUERY`. Hard parameter bounds
    (#7, #9, #10, #11) record as passed (upstream-enforced).

    Score arithmetic per plan §9.7.3:

    - Start at 1.0.
    - Each moderate failure subtracts 0.3.
    - Each soft failure subtracts 0.1.
    - Any critical failure drives the score to 0.
    - ``PENDING_CADQUERY`` checks contribute neither pass nor fail — they
      are listed in ``pending_cadquery`` so callers can see what's still
      unverifiable in the scaffold tier.
    """
    del geometry_description  # consumed by real Phase-1 checks; scaffold ignores
    checks = (
        _pending_check(
            "1",
            "Minimum feature size ≥ 0.8 mm",
            CheckSeverity.MODERATE,
            "All features ≥ 0.8 mm (2× nozzle for 0.4 mm). TPMS exemption "
            "via check #1b. Schema enforces MIN_FEATURE_SIZE_M floor; "
            "geometry-level skin-edge scan deferred to Phase 1.",
        ),
        _pending_check(
            "2",
            "Overhang angle ≤ 45°",
            CheckSeverity.MODERATE,
            "Overhangs ≤ 45° from vertical without support, ≤ 60° with "
            "bridging. TPMS surfaces are self-supporting (exempt). "
            "Requires CadQuery face-normal scan.",
        ),
        _pending_check(
            "3",
            "Connectivity (single component)",
            CheckSeverity.CRITICAL,
            "Blade must be a single connected component. Requires " "CadQuery solid-count check.",
        ),
        _pending_check(
            "4",
            "Bridging ≤ 8 mm",
            CheckSeverity.SOFT,
            "Any horizontal span > 8 mm flagged. Requires CadQuery "
            "free-span geometric measurement.",
        ),
        _pending_check(
            "5",
            "Internal voids — exit path ≥ 1 mm (TPMS exempt)",
            CheckSeverity.CRITICAL,
            "Any fully-enclosed void without ≥ 1 mm exit path → unprintable. "
            "TPMS lattices exempt. Requires CadQuery void-detection scan.",
        ),
        _pending_check(
            "6",
            "Edge clearance ≥ 1 mm from envelope",
            CheckSeverity.CRITICAL,
            "All features ≥ 1 mm from blade outer envelope. Layer 2 fields "
            "guarantee by construction; Layer 3 enforces via position "
            "margins (schema). Geometry-level skin-feature distance check "
            "deferred to Phase 1.",
        ),
        _bound_check(
            "7",
            "Click feature 5 mm exclusion (panel-edge)",
            "Layer 2 fields + Layer 3 primitive are constrained to keep "
            "≥ 5 mm clearance from CLICK_FOOTPRINT_X_RANGE on every "
            "panel's outer tangential edge. Hard parameter bound — "
            "upstream-enforced.",
        ),
        _pending_check(
            "8",
            "Aspect ratio ≤ 20:1",
            CheckSeverity.SOFT,
            "No feature with aspect ratio > 20:1. Requires CadQuery "
            "bounding-box aspect measurement on each feature solid.",
        ),
        _bound_check(
            "9",
            "Noise field threshold retains ≥ 40% material",
            "Hard parameter bound — schema enforces " "NOISE_THRESHOLD_RETENTION_MIN floor.",
        ),
        _bound_check(
            "10",
            "TPMS cell size ≥ 3× min feature size",
            "Hard parameter bound — schema enforces TPMS_CELL_SIZE_MIN_M " "(2.4 mm) floor.",
        ),
        _bound_check(
            "11",
            "Fourier envelope amplitude ≤ ±15%",
            "Hard parameter bound — schema enforces "
            "FOURIER_AMPLITUDE_RELATIVE_MAX cap on every harmonic.",
        ),
        _pending_check(
            "12",
            "Layer-adhesion Z-thin-section flag",
            CheckSeverity.MODERATE,
            "Hard fail if > 5% of panel volume is Z-thinner than 1.5× "
            "layer height. Requires CadQuery volume-by-z-thickness "
            "analysis.",
        ),
        _pending_check(
            "13",
            "Warpage proxy (large planar face)",
            CheckSeverity.SOFT,
            "Face with bounding-box aspect > 8:1 AND area > 1000 mm² "
            "flagged. Requires CadQuery face-aspect measurement.",
        ),
        _pending_check(
            "14",
            "Support-scar location on functional surfaces",
            CheckSeverity.CRITICAL,
            "Under print_orientation='flat', the panel's bottom face's "
            "outward normal must point in −z. Requires CadQuery face-"
            "orientation check on the calibrated wall-roughness face.",
        ),
    )

    score, critical_failures, pending_cadquery = _aggregate_score(checks)
    return ManufacturabilityResult(
        score=score,
        passed=score >= MANUFACTURABILITY_PASS_THRESHOLD,
        checks=checks,
        critical_failures=tuple(critical_failures),
        pending_cadquery=tuple(pending_cadquery),
    )


def _aggregate_score(
    checks: tuple[CheckResult, ...],
) -> tuple[float, list[str], list[str]]:
    """Apply plan §9.7.3 scoring arithmetic to a tuple of CheckResults.

    Public test surface — kept module-private (underscore prefix) but
    importable by ``tests/test_geometry/test_manufacturability.py`` so the
    FAILED-path penalty arithmetic can be exercised without waiting on
    Phase 1 CadQuery code to surface real failures.

    Returns ``(score, critical_failures, pending_cadquery)``. ``score`` is
    clamped to ``[0.0, 1.0]``; any CRITICAL failure forces score to 0.
    """
    score = 1.0
    critical_failures: list[str] = []
    pending_cadquery: list[str] = []
    for c in checks:
        if c.status == CheckStatus.PENDING_CADQUERY:
            pending_cadquery.append(c.check_id)
            continue
        if c.status == CheckStatus.FAILED:
            if c.severity == CheckSeverity.CRITICAL:
                critical_failures.append(c.check_id)
            elif c.severity == CheckSeverity.MODERATE:
                score -= MODERATE_FAILURE_PENALTY
            elif c.severity == CheckSeverity.SOFT:
                score -= SOFT_FAILURE_PENALTY
            # HARD_BOUND never FAILS (upstream-enforced).

    if critical_failures:
        score = 0.0
    score = max(0.0, score)
    return score, critical_failures, pending_cadquery
