"""Click-feature tolerance, cycle life, and V1-lock force balance.

Implements Spike 0.4 (`docs/spike_0_4_protocol.md`,
`docs/plan_R11.md §Phase 0 Spike 0.4`).

Three independent measurements feed one Spike 0.4 verdict:

1. **V1-lock force balance (H6 lock).** Cumulative click-engagement friction
   across the 9 inter-blade pairs at the deployed position must exceed the
   inertial force that would back-drive the lock at the wrist's peak angular
   acceleration. Inertial torque from Spike 0.2 inertia and α_max = 110 rad/s²
   is converted to a tangential force AT the click location using the
   **wrist-to-tip lever arm L_wrist_to_tip = 0.25 m** (H8 lock —
   d_handle + L_blade = 0.05 + 0.20). The click chamfer + detent live at the
   panel's outer tangential edge at the tip, NOT on a rib face.

   Pass: `F_friction_cumulative ≥ 2 × F_inertial_at_click` (factor-of-2 safety
   margin). On failure, the V1 fallback geometry (printed rib-tab on each
   guard blade) is auto-armed via the `v1_lock_fallback_armed` flag, which
   downstream sets `params.layer4.v1_lock_fallback_enabled = True`.

2. **As-printed clearance at the click feature.** Target band 0.15-0.20 mm per
   mating surface. Out-of-band → re-tune slicer or printer calibration.

3. **Click engagement force + 1000-cycle fatigue + high-amplitude segment.**
   - Low regime: 0.5-2 N (canonical deploy-from-folded force).
   - 1000 deploy/fold cycles, inspect every 100. No detent fracture / wear.
   - High-amplitude stress segment: +100 cycles at 1-4 N (~2× design point).
   - Deployed-state alignment gap variation < 1 mm across adjacent blade tips.

   Fallback on detent fracture → embedded neodymium magnetic catch
   (~20-40 g, within the 100 g C9 mass constraint).

All functions are pure: no file I/O, no globals, no side effects. The CLI
wrapper (`scripts/run_spike_0_4.py`) is responsible for CSV ingestion and
results.json emission.

References:
- Spec: `docs/plan_R11.md §Phase 0 Spike 0.4`
- Locks: H6 (force balance), H8 (lever arm 0.25 m), C9 (mass cap 100 g)
- Inertia input: `src/fanopt/physical/inertia.py` (Spike 0.2)
- Protocol: `docs/spike_0_4_protocol.md`
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable

__all__ = [
    "ALPHA_MAX_RAD_PER_S2",
    "L_WRIST_TO_TIP_M",
    "FORCE_BALANCE_SAFETY_FACTOR",
    "FORCE_BALANCE_SAFETY_FACTOR_ANALYTIC",
    "CLEARANCE_MIN_MM",
    "CLEARANCE_MAX_MM",
    "ENGAGEMENT_FORCE_MIN_N",
    "ENGAGEMENT_FORCE_MAX_N",
    "HIGH_AMP_FORCE_MIN_N",
    "HIGH_AMP_FORCE_MAX_N",
    "CYCLE_TARGET",
    "HIGH_AMP_CYCLE_TARGET",
    "ALIGNMENT_GAP_MAX_MM",
    "inertial_force_at_click",
    "force_balance_passes",
    "ForceBalanceResult",
    "analyze_force_balance",
    "ClearanceRow",
    "ClearanceResult",
    "analyze_clearance",
    "EngagementForceResult",
    "analyze_engagement_force",
    "CycleLifeResult",
    "analyze_cycle_life",
    "Spike04Result",
]


# ─────────────────────────────────────────────────────────────────────
# Module-level constants (Spike 0.4 locks)
# ─────────────────────────────────────────────────────────────────────

# H8/H6 lock: peak wrist angular acceleration used in inertial-force balance.
ALPHA_MAX_RAD_PER_S2: float = 110.0

# H8 lever-arm lock: click feature sits at the panel's outer tangential edge at
# the tip, i.e., d_handle (0.05 m) + L_blade (0.20 m) = 0.25 m from the wrist
# axis. **Not** 0.20 m from the pivot pin.
L_WRIST_TO_TIP_M: float = 0.25

# Pass-criterion safety factor for the V1-lock force balance.
FORCE_BALANCE_SAFETY_FACTOR: float = 2.0
"""Canonical 2x factor when I_wrist is measured (Spike 0.2 PASS)."""

FORCE_BALANCE_SAFETY_FACTOR_ANALYTIC: float = 3.0
"""Bumped 3x factor when I_wrist comes from the analytic generator emission
instead of a Spike 0.2 bench measurement. Absorbs the unverified-inertia
uncertainty for V1 (Spike 0.2 deferred to V2 per docs/phase_logs/phase_0_signoff.md)."""

# Per-mating-surface clearance band at the click feature.
CLEARANCE_MIN_MM: float = 0.15
CLEARANCE_MAX_MM: float = 0.20

# Click engagement force band, low regime (canonical design point).
ENGAGEMENT_FORCE_MIN_N: float = 0.5
ENGAGEMENT_FORCE_MAX_N: float = 2.0

# Click engagement force band, high-amplitude stress segment (~2× design).
HIGH_AMP_FORCE_MIN_N: float = 1.0
HIGH_AMP_FORCE_MAX_N: float = 4.0

# Cycle-life targets.
CYCLE_TARGET: int = 1000
HIGH_AMP_CYCLE_TARGET: int = 100

# Deployed-state alignment gap variation across adjacent blade tips.
ALIGNMENT_GAP_MAX_MM: float = 1.0


# ─────────────────────────────────────────────────────────────────────
# V1-lock force balance (H6 lock)
# ─────────────────────────────────────────────────────────────────────


def inertial_force_at_click(
    I_wrist_kgm2: float,
    alpha_max: float = ALPHA_MAX_RAD_PER_S2,
    lever_arm_m: float = L_WRIST_TO_TIP_M,
) -> float:
    """Convert wrist inertial torque to tangential force at the click feature.

    `τ_inertial_peak = I_wrist · α_max`, then `F = τ / L_wrist_to_tip`. The
    lever arm is the **wrist-to-tip** distance (H8 lock: 0.25 m), since the
    click chamfer + detent live at the panel's outer tangential edge at the
    tip, not on a rib face.

    Parameters
    ----------
    I_wrist_kgm2 : measured I_wrist from Spike 0.2, kg·m².
    alpha_max : peak wrist angular acceleration, rad/s² (default 110).
    lever_arm_m : wrist-to-tip lever arm, m (default 0.25 — H8 lock).

    Returns
    -------
    F_inertial_at_click : float
        Tangential force AT the click location, newtons.
    """
    if I_wrist_kgm2 <= 0:
        raise ValueError(f"I_wrist_kgm2 must be > 0, got {I_wrist_kgm2}")
    if alpha_max <= 0:
        raise ValueError(f"alpha_max must be > 0, got {alpha_max}")
    if lever_arm_m <= 0:
        raise ValueError(f"lever_arm_m must be > 0, got {lever_arm_m}")
    tau = I_wrist_kgm2 * alpha_max
    return tau / lever_arm_m


def force_balance_passes(
    F_friction_cumulative_N: float,
    F_inertial_at_click_N: float,
    safety_factor: float = FORCE_BALANCE_SAFETY_FACTOR,
) -> bool:
    """Return True iff cumulative friction ≥ safety_factor × inertial force.

    Pass criterion (Spike 0.4 V1-lock force balance):
    `F_friction_cumulative ≥ safety_factor · F_inertial_at_click` with the
    canonical safety factor of 2.0.
    """
    if F_friction_cumulative_N < 0:
        raise ValueError(
            f"F_friction_cumulative_N must be ≥ 0, got {F_friction_cumulative_N}"
        )
    if F_inertial_at_click_N <= 0:
        raise ValueError(
            f"F_inertial_at_click_N must be > 0, got {F_inertial_at_click_N}"
        )
    if safety_factor <= 0:
        raise ValueError(f"safety_factor must be > 0, got {safety_factor}")
    return F_friction_cumulative_N >= safety_factor * F_inertial_at_click_N


@dataclass(frozen=True)
class ForceBalanceResult:
    """V1-lock force balance verdict (H6 lock).

    Serializable via `dataclasses.asdict`. `v1_lock_fallback_armed` mirrors
    `not passed`: the printed rib-tab fallback geometry auto-arms when the
    force balance fails.
    """

    I_wrist_kgm2: float
    """Measured wrist inertia from Spike 0.2, kg·m²."""

    F_friction_cumulative_N: float
    """Measured cumulative friction across the 9 inter-blade pairs, N."""

    tau_inertial_peak_Nm: float
    """`I_wrist · α_max`, N·m."""

    F_inertial_at_click_N: float
    """`τ_inertial_peak / L_wrist_to_tip`, N. The thing friction must beat."""

    required_friction_N: float
    """`safety_factor × F_inertial_at_click_N`, N."""

    margin_ratio: float
    """`F_friction_cumulative_N / F_inertial_at_click_N`. Pass if ≥ 2.0."""

    passed: bool
    """Pass iff cumulative friction ≥ safety_factor × inertial force."""

    v1_lock_fallback_armed: bool
    """True iff the V1 fallback (rib-tab snap-fit) auto-arms (= not passed)."""


def analyze_force_balance(
    I_wrist_kgm2: float,
    F_friction_cumulative_N: float,
    *,
    alpha_max: float = ALPHA_MAX_RAD_PER_S2,
    lever_arm_m: float = L_WRIST_TO_TIP_M,
    safety_factor: float = FORCE_BALANCE_SAFETY_FACTOR,
) -> ForceBalanceResult:
    """Compute the Spike 0.4 V1-lock force-balance verdict.

    Parameters
    ----------
    I_wrist_kgm2 : measured wrist inertia from Spike 0.2, kg·m².
    F_friction_cumulative_N : sum of click-engagement friction across the
        9 inter-blade pairs at the deployed position, N. Force gauge applied
        tangentially at `(x = L_blade, y = ±panel_tangential_outer)`.
    alpha_max : peak wrist angular acceleration, rad/s² (default 110).
    lever_arm_m : wrist-to-tip lever arm, m (default 0.25 — H8 lock).
    safety_factor : safety factor in the pass criterion (default 2.0).

    Returns
    -------
    ForceBalanceResult with pass/fail and the fallback-arming flag.
    """
    # `inertial_force_at_click` validates I_wrist / alpha_max / lever_arm.
    F_inertial = inertial_force_at_click(
        I_wrist_kgm2, alpha_max=alpha_max, lever_arm_m=lever_arm_m
    )
    tau = I_wrist_kgm2 * alpha_max
    passed = force_balance_passes(
        F_friction_cumulative_N=F_friction_cumulative_N,
        F_inertial_at_click_N=F_inertial,
        safety_factor=safety_factor,
    )
    margin_ratio = F_friction_cumulative_N / F_inertial
    return ForceBalanceResult(
        I_wrist_kgm2=I_wrist_kgm2,
        F_friction_cumulative_N=F_friction_cumulative_N,
        tau_inertial_peak_Nm=tau,
        F_inertial_at_click_N=F_inertial,
        required_friction_N=safety_factor * F_inertial,
        margin_ratio=margin_ratio,
        passed=passed,
        v1_lock_fallback_armed=not passed,
    )


# ─────────────────────────────────────────────────────────────────────
# As-printed clearance at the click feature
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClearanceRow:
    """One mating-surface clearance measurement (Spike 0.4 Step 3)."""

    mating_surface: str
    """Operator label (e.g., 'blade1_blade2_outer')."""

    clearance_mm: float
    """Measured clearance at the click feature, mm."""

    in_band: bool
    """True iff `CLEARANCE_MIN_MM ≤ clearance_mm ≤ CLEARANCE_MAX_MM`."""


@dataclass(frozen=True)
class ClearanceResult:
    """Aggregated clearance verdict across all measured mating surfaces."""

    rows: tuple[ClearanceRow, ...]
    """Per-mating-surface measurements with per-row pass/fail."""

    n_measurements: int
    out_of_band_count: int
    out_of_band_rows: tuple[ClearanceRow, ...]
    """Rows that fell outside the [0.15, 0.20] mm band."""

    min_mm: float
    max_mm: float
    mean_mm: float

    band_min_mm: float = CLEARANCE_MIN_MM
    band_max_mm: float = CLEARANCE_MAX_MM

    passed: bool = False
    """True iff every row is in-band."""


def analyze_clearance(
    measurements_mm: Iterable[float],
    *,
    labels: Iterable[str] | None = None,
) -> ClearanceResult:
    """Score per-mating-surface clearance against the [0.15, 0.20] mm band.

    Parameters
    ----------
    measurements_mm : per-mating-surface clearance measurements, mm. One row
        per surface; must have ≥ 1 row.
    labels : optional operator labels for each row (e.g.,
        'blade1_blade2_outer'). Defaults to 'surface_{i}' if not provided.

    Returns
    -------
    ClearanceResult with per-row pass/fail and aggregate pass/fail.
    """
    values = tuple(float(v) for v in measurements_mm)
    if not values:
        raise ValueError("clearance: need ≥ 1 measurement row, got 0")

    label_tuple: tuple[str, ...]
    if labels is None:
        label_tuple = tuple(f"surface_{i + 1}" for i in range(len(values)))
    else:
        label_tuple = tuple(str(s) for s in labels)
        if len(label_tuple) != len(values):
            raise ValueError(
                f"labels length {len(label_tuple)} != measurements length {len(values)}"
            )

    rows = tuple(
        ClearanceRow(
            mating_surface=lab,
            clearance_mm=v,
            in_band=(CLEARANCE_MIN_MM <= v <= CLEARANCE_MAX_MM),
        )
        for lab, v in zip(label_tuple, values, strict=True)
    )
    out_of_band = tuple(r for r in rows if not r.in_band)
    return ClearanceResult(
        rows=rows,
        n_measurements=len(values),
        out_of_band_count=len(out_of_band),
        out_of_band_rows=out_of_band,
        min_mm=min(values),
        max_mm=max(values),
        mean_mm=sum(values) / len(values),
        passed=(len(out_of_band) == 0),
    )


# ─────────────────────────────────────────────────────────────────────
# Click engagement force
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EngagementForceResult:
    """Click engagement-force verdict for one regime (low or high amplitude)."""

    regime: str
    """'low' (0.5-2 N) or 'high' (1-4 N)."""

    forces_N: tuple[float, ...]
    """Per-trial engagement forces (deploy from folded), N."""

    n_trials: int
    mean_N: float
    std_N: float
    """Sample standard deviation (ddof=1) if n_trials ≥ 2, else 0.0."""

    band_min_N: float
    band_max_N: float
    out_of_band_count: int
    passed: bool
    """True iff every trial is in-band."""


def analyze_engagement_force(
    forces_N: Iterable[float],
    *,
    high_amplitude: bool = False,
) -> EngagementForceResult:
    """Score per-trial click engagement force against the appropriate band.

    Parameters
    ----------
    forces_N : per-trial engagement forces (deploy from folded), N.
    high_amplitude : if True, score against the 1-4 N band (high-amplitude
        stress segment). If False, score against the canonical 0.5-2 N band.

    Returns
    -------
    EngagementForceResult with per-regime pass/fail.
    """
    values = tuple(float(v) for v in forces_N)
    if not values:
        raise ValueError("engagement force: need ≥ 1 trial, got 0")

    if high_amplitude:
        lo, hi, regime = HIGH_AMP_FORCE_MIN_N, HIGH_AMP_FORCE_MAX_N, "high"
    else:
        lo, hi, regime = ENGAGEMENT_FORCE_MIN_N, ENGAGEMENT_FORCE_MAX_N, "low"

    in_band = [lo <= v <= hi for v in values]
    out_of_band_count = sum(1 for ok in in_band if not ok)
    mean_N = sum(values) / len(values)
    std_N = statistics.stdev(values) if len(values) >= 2 else 0.0
    return EngagementForceResult(
        regime=regime,
        forces_N=values,
        n_trials=len(values),
        mean_N=mean_N,
        std_N=std_N,
        band_min_N=lo,
        band_max_N=hi,
        out_of_band_count=out_of_band_count,
        passed=(out_of_band_count == 0),
    )


# ─────────────────────────────────────────────────────────────────────
# Cycle life
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CycleLifeResult:
    """1000-cycle fatigue + high-amplitude segment + alignment-gap verdict."""

    n_inspections: int
    """Number of inspection rows ingested."""

    total_cycles_completed: int
    """Largest cycle index recorded across inspections (0 if no rows)."""

    first_wear_cycle: int | None
    """Cycle index at first observed `wear_observed=True`; None if never."""

    first_fracture_cycle: int | None
    """Cycle index at first observed `fracture=True`; None if never."""

    cycle_target: int = CYCLE_TARGET
    high_amp_cycle_target: int = HIGH_AMP_CYCLE_TARGET

    high_amp_completed: bool = False
    """Operator-reported: did the 100-cycle high-amplitude segment finish?"""

    high_amp_failure_cycle: int | None = None
    """Cycle index within the high-amplitude segment where detent fractured."""

    high_amp_passed: bool = False
    """True iff the high-amplitude segment completed without fracture."""

    alignment_gap_variation_mm: float = 0.0
    """Worst-case gap variation across adjacent blade tips, mm."""

    alignment_passed: bool = False
    """True iff `alignment_gap_variation_mm < ALIGNMENT_GAP_MAX_MM`."""

    low_amp_passed: bool = False
    """True iff no fracture observed within `CYCLE_TARGET` cycles."""

    passed: bool = False
    """Roll-up: low-amp + high-amp + alignment all pass."""


def analyze_cycle_life(
    inspections: Iterable[dict],
    alignment_gap_variation_mm: float,
    high_amp_completed: bool,
    high_amp_failure_cycle: int | None,
) -> CycleLifeResult:
    """Score the 1000-cycle fatigue run + high-amplitude segment + alignment.

    Parameters
    ----------
    inspections : iterable of dicts with keys
        `{"cycle": int, "wear_observed": bool, "fracture": bool, "notes": str}`.
        Inspection rows from the every-100-cycle audit during the 1000-cycle
        low-amplitude run.
    alignment_gap_variation_mm : worst-case gap variation across adjacent
        blade tips at full deployment, mm.
    high_amp_completed : True iff the operator completed the 100-cycle
        high-amplitude stress segment (1-4 N engagement force).
    high_amp_failure_cycle : cycle index within the high-amplitude segment at
        which the detent fractured; None if no fracture observed.

    Returns
    -------
    CycleLifeResult with sub-gates and overall pass/fail.

    Notes
    -----
    `low_amp_passed` requires (a) the 1000-cycle target was reached, and
    (b) no fracture before that cycle. Wear observations alone do not fail
    the gate — only fracture does — but they are surfaced via
    `first_wear_cycle` for the run log.
    """
    rows = list(inspections)
    cycles: list[int] = []
    first_wear: int | None = None
    first_fracture: int | None = None
    for i, row in enumerate(rows, start=1):
        try:
            c = int(row["cycle"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"inspection row {i}: missing/invalid 'cycle': {row}") from e
        cycles.append(c)
        if bool(row.get("wear_observed", False)) and first_wear is None:
            first_wear = c
        if bool(row.get("fracture", False)) and first_fracture is None:
            first_fracture = c

    total = max(cycles) if cycles else 0

    if first_fracture is not None and first_fracture <= CYCLE_TARGET:
        low_amp_passed = False
    else:
        low_amp_passed = total >= CYCLE_TARGET

    if high_amp_failure_cycle is not None:
        high_amp_passed = False
    else:
        high_amp_passed = bool(high_amp_completed)

    alignment_passed = alignment_gap_variation_mm < ALIGNMENT_GAP_MAX_MM

    passed = low_amp_passed and high_amp_passed and alignment_passed

    return CycleLifeResult(
        n_inspections=len(rows),
        total_cycles_completed=total,
        first_wear_cycle=first_wear,
        first_fracture_cycle=first_fracture,
        high_amp_completed=bool(high_amp_completed),
        high_amp_failure_cycle=high_amp_failure_cycle,
        high_amp_passed=high_amp_passed,
        alignment_gap_variation_mm=float(alignment_gap_variation_mm),
        alignment_passed=alignment_passed,
        low_amp_passed=low_amp_passed,
        passed=passed,
    )


# ─────────────────────────────────────────────────────────────────────
# Spike 0.4 roll-up
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Spike04Result:
    """Top-level Spike 0.4 verdict.

    Serializable via `dataclasses.asdict`. The CLI wrapper writes this to
    `data/spike_0_4/results.json` and the runner exits 0 iff `overall_passed`.
    """

    force_balance: ForceBalanceResult
    clearance: ClearanceResult
    engagement_force: EngagementForceResult
    cycle_life: CycleLifeResult

    overall_passed: bool
    """True iff every sub-gate passes."""

    v1_lock_fallback_armed: bool
    """Mirrors `force_balance.v1_lock_fallback_armed`. Drives
    `params.layer4.v1_lock_fallback_enabled` downstream."""

    high_amp_engagement_force: EngagementForceResult | None = None
    """Optional: high-amplitude-regime engagement-force scoring, when the
    operator recorded forces from the 100-cycle stress segment."""
