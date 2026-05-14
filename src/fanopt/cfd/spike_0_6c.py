"""Spike 0.6c — Tier-1 unsteady-config benchmark validation (H10 lock).

Implements ``docs/plan_R11.md §Phase 0 Spike 0.6c`` (gates Phase 4 launch).

**Two sub-spikes, both required to PASS:**

1. **0.6c.1 — Tier-1 cfg sanity check.** Render the canonical Tier-1 cfg
   (``configs/su2/fan3d_unsteady.cfg.j2``) and verify it parses without
   error, that it carries the Round-9 HIGH-12 lock (``MACH = 1e-9``), and
   that SU2 can complete one outer time-step on a probe mesh. The
   Round-9 HIGH-12 lock pins the unsteady cfg to ``MACH = 1e-9`` with
   ``FREESTREAM_OPTION = FREESTREAM_VELOCITY`` (primary path) or
   ``REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE`` (fallback path).
   This sub-spike validates the rendered cfg, not the CROSS_TIER dict
   (``CROSS_TIER`` does NOT carry ``MACH`` per the C12 lock — MACH is
   tier-specific and lives in the TIER_SPECIFIC[1] block).

2. **0.6c.2 — Published-benchmark validation.** Run a NACA 0012
   oscillating-airfoil case (pitching about quarter-chord at
   ``k_reduced ≈ 0.55``, ``Re ≈ 40k``) through the working Tier-1 cfg.
   Discard cycle 1, integrate cycles 2-5 for the lift/drag metrics,
   compare to published references (McAlister/Carr UH110A or Anderson
   oscillating-airfoil DB). PASS iff every metric is within ±15% of the
   published value.

**Why this gates Phase 4.** The compressible-with-low-Mach-prec +
RIGID_MOTION + near-zero-ambient + 5-cycle-dual-time-stepping numerics
combination is locked on engineering judgment. Without benchmark
validation, the entire Tier 1 dataset (the only "true J_fan" tier) rests
on unvalidated numerics — a silent error in any of the locked numerics
would propagate through every Phase 4/5 Tier-1 result.

References:

* Spec: ``docs/plan_R11.md §Phase 0 Spike 0.6c`` (lines 1839-1844).
* Protocol: ``docs/spike_0_6c_protocol.md``.
* Lock callouts: Round-9 HIGH-12 (= C12, unsteady MACH lock).
* Companion CI gate: ``tests/test_cfd/test_unsteady_freestream_consistency.py``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

__all__ = [
    "BENCHMARK_TOLERANCE_PCT",
    "BENCHMARK_K_REDUCED_MIN",
    "BENCHMARK_K_REDUCED_MAX",
    "BENCHMARK_RE_MIN",
    "BENCHMARK_RE_MAX",
    "BENCHMARK_CYCLES_TOTAL",
    "BENCHMARK_CYCLES_DISCARD",
    "MACH_UNSTEADY_LOCK",
    "NACA0012_REFERENCE",
    "Tier1CfgSanityResult",
    "BenchmarkCycleData",
    "BenchmarkComparison",
    "BenchmarkResult",
    "Spike06cResult",
    "check_tier1_cfg_sanity",
    "compare_cycle_to_reference",
    "analyze_benchmark",
    "analyze_spike_06c",
]


# ----- locks (from §Phase 0 Spike 0.6c + Round-9 HIGH-12) -------------------

BENCHMARK_TOLERANCE_PCT: float = 15.0
"""Sub-spike 0.6c.2 tolerance: each measured metric must match its
reference within ±15%. Source: spec line 1843."""

BENCHMARK_K_REDUCED_MIN: float = 0.5
BENCHMARK_K_REDUCED_MAX: float = 0.6
"""Reduced-frequency band for the NACA 0012 benchmark
(``k_reduced ≈ 0.5-0.6``). Source: spec line 1843."""

BENCHMARK_RE_MIN: float = 30_000.0
BENCHMARK_RE_MAX: float = 50_000.0
"""Reynolds-number band for the NACA 0012 benchmark
(``Re ≈ 30k-50k``). Source: spec line 1843."""

BENCHMARK_CYCLES_TOTAL: int = 5
BENCHMARK_CYCLES_DISCARD: int = 1
"""5 cycles total; discard the first (initial-transient) cycle, integrate
the remaining 4. Source: spec line 1843."""

MACH_UNSTEADY_LOCK: float = 1e-9
"""Tier-1 unsteady MACH value (Round-9 HIGH-12 = C12). The steady tiers
(Tier -1 / Tier 0) use ``MACH = 0.0064``; ``CROSS_TIER`` does NOT carry
``MACH`` — the value lives in ``TIER_SPECIFIC[1]`` per §9.4.1."""


# ----- shipped reference data ----------------------------------------------

NACA0012_REFERENCE: dict[str, float] = {
    "c_l_max": 1.20,
    "c_l_min": -1.20,
    "c_d_mean": 0.085,
    "c_l_hysteresis_area": 0.45,
}
"""Default reference dataset for the NACA 0012 oscillating-airfoil
benchmark (k_reduced ≈ 0.55, Re ≈ 40k, pitching about quarter-chord,
±10° amplitude).

Values are representative ranges from the McAlister/Carr UH110A airfoil
studies and the Anderson oscillating-airfoil database (specifically the
low-Re symmetric-foil subset). Operators running the benchmark should
override this dict with the exact published values for the case they
chose to reproduce — the shipped numbers exist so the runner has a
template default, not to substitute for citing a specific paper.

Override mechanism: ``scripts/run_spike_0_6c_2.py --reference <path.json>``.
"""


# ----- sub-spike 0.6c.1 (Tier-1 cfg sanity check) ---------------------------


@dataclass(frozen=True)
class Tier1CfgSanityResult:
    """Outcome of ``check_tier1_cfg_sanity`` for a single rendered cfg.

    Pass criterion (gate for Phase 4 launch via the aggregator):
      * ``outer_time_steps_completed >= 1``
      * ``mach_value == MACH_UNSTEADY_LOCK``
      * EITHER ``freestream_option == "FREESTREAM_VELOCITY"``
        OR ``ref_dimensionalization == "FREESTREAM_PRESS_EQ_ONE"``

    The H10 lock allows EITHER syntax (the spec authors anticipated a
    fallback path); both pass under Round-9 HIGH-12.
    """

    cfg_path: str
    parsed_ok: bool
    mach_value: float
    freestream_option: str
    ref_dimensionalization: str | None
    outer_time_steps_completed: int
    error: str | None
    passed: bool


def _parse_cfg_directive(cfg_text: str, key: str) -> str | None:
    """Extract one ``KEY = VALUE`` line's value, stripping inline comments.

    SU2 cfg lines look like ``KEY= VALUE % optional comment`` (with optional
    whitespace and optional ``=`` spacing). Returns the trimmed value string,
    or ``None`` if the key does not appear in ``cfg_text``.
    """
    pattern = rf"^\s*{re.escape(key)}\s*=\s*([^%\n]+?)(?:\s*%.*)?$"
    match = re.search(pattern, cfg_text, re.MULTILINE)
    return match.group(1).strip() if match else None


def _count_completed_outer_steps(su2_log: str) -> int:
    """Count the number of outer (time-step) iterations SU2 completed.

    SU2's stdout under unsteady mode prints ``Time Iter`` lines. We count
    distinct integer ``Time_Iter`` markers as a robust lower bound on the
    completed outer-step count. If no such markers appear, we fall back to
    a permissive ``"Time step:"`` or ``"OUTER_ITER"`` match.
    """
    if not su2_log:
        return 0
    # Primary: explicit "Time_Iter: N" or "Time Iter: N" patterns.
    primary = re.findall(r"Time[_ ]Iter\s*[:=]\s*(\d+)", su2_log)
    if primary:
        return len({int(v) for v in primary})
    # Fallback: count "Time step:" or "Iter:" lines in summary tables.
    fallback = re.findall(r"^\s*\d+\s+Time\s+step\s+", su2_log, re.MULTILINE)
    if fallback:
        return len(fallback)
    # Last resort: any "OUTER_ITER" markers.
    return len(re.findall(r"OUTER_ITER\b", su2_log))


def check_tier1_cfg_sanity(
    cfg_text: str,
    *,
    su2_log: str | None = None,
    cfg_path: str = "<rendered>",
) -> Tier1CfgSanityResult:
    """Validate that the rendered Tier-1 cfg satisfies the Round-9 HIGH-12 lock.

    Parameters
    ----------
    cfg_text : the rendered SU2 ``.cfg`` text (key = value lines).
    su2_log : optional SU2 stdout from running the cfg on a probe mesh.
        When provided, the outer-time-step count is parsed from the log;
        when absent, the sanity check passes the parser-only path with
        ``outer_time_steps_completed = 0`` (the cfg-only fallback that
        the runner uses when ``SU2_CFD`` is not on PATH).

        Note: the sub-spike's pass criterion requires
        ``outer_time_steps_completed >= 1``, so passing ``su2_log=None``
        produces ``passed=False`` — by design, since the cfg-only fallback
        is a partial check until SU2 is available.
    cfg_path : optional path label for the result record (informational).

    Returns
    -------
    ``Tier1CfgSanityResult`` with the four lock fields populated and the
    overall ``passed`` flag.
    """
    error: str | None = None
    mach_value = math.nan
    freestream_option = ""
    ref_dimensionalization: str | None = None
    parsed_ok = False

    try:
        mach_str = _parse_cfg_directive(cfg_text, "MACH_NUMBER")
        if mach_str is None:
            raise ValueError("MACH_NUMBER directive not found in cfg")
        mach_value = float(mach_str)

        freestream_option = _parse_cfg_directive(cfg_text, "FREESTREAM_OPTION") or ""
        ref_dimensionalization = _parse_cfg_directive(
            cfg_text, "REF_DIMENSIONALIZATION"
        )
        parsed_ok = True
    except (ValueError, TypeError) as e:
        error = str(e)

    if su2_log is not None:
        outer_steps = _count_completed_outer_steps(su2_log)
    else:
        outer_steps = 0

    # Pass criterion:
    #  - cfg parsed
    #  - MACH matches the Round-9 HIGH-12 lock (1e-9)
    #  - either primary OR fallback freestream-syntax path is present
    #  - SU2 completed at least one outer time step on the probe mesh
    mach_ok = parsed_ok and mach_value == MACH_UNSTEADY_LOCK
    syntax_ok = (
        freestream_option == "FREESTREAM_VELOCITY"
        or ref_dimensionalization == "FREESTREAM_PRESS_EQ_ONE"
    )
    outer_ok = outer_steps >= 1
    passed = parsed_ok and mach_ok and syntax_ok and outer_ok

    return Tier1CfgSanityResult(
        cfg_path=cfg_path,
        parsed_ok=parsed_ok,
        mach_value=mach_value,
        freestream_option=freestream_option,
        ref_dimensionalization=ref_dimensionalization,
        outer_time_steps_completed=outer_steps,
        error=error,
        passed=passed,
    )


# ----- sub-spike 0.6c.2 (NACA 0012 benchmark validation) --------------------


@dataclass(frozen=True)
class BenchmarkCycleData:
    """Per-cycle lift/drag aggregates extracted from one pitching cycle.

    Fields are independent metrics — each is compared to its published
    reference value separately and each must individually pass the
    ±15% tolerance gate.

    * ``c_l_max`` — peak lift coefficient over the cycle.
    * ``c_l_min`` — trough lift coefficient over the cycle.
    * ``c_d_mean`` — cycle-mean drag coefficient.
    * ``c_l_hysteresis_area`` — signed area inside the ``C_l(α)`` loop.
    """

    cycle_index: int
    c_l_max: float
    c_l_min: float
    c_d_mean: float
    c_l_hysteresis_area: float


@dataclass(frozen=True)
class BenchmarkComparison:
    """One metric's measured / reference / pct-diff comparison.

    ``pct_diff`` is signed (measured > reference => positive), but the
    ±15% gate is two-sided so ``passed = |pct_diff| < BENCHMARK_TOLERANCE_PCT``.
    """

    metric_name: str
    measured: float
    reference: float
    pct_diff: float
    passed: bool


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregated outcome of sub-spike 0.6c.2."""

    k_reduced: float
    reynolds: float
    cycles: tuple[BenchmarkCycleData, ...]
    reference_source: str
    comparisons: tuple[BenchmarkComparison, ...]
    all_metrics_within_15pct: bool
    passed: bool


def _signed_pct_diff(measured: float, reference: float) -> float:
    """Return ``100 * (measured - reference) / reference``.

    For a zero reference value the function returns ``inf`` if the
    measured value is non-zero (the comparison will fail the ±15% gate
    by construction), and ``0.0`` if both values are zero.
    """
    if reference == 0.0:
        return 0.0 if measured == 0.0 else math.inf
    return 100.0 * (measured - reference) / reference


def compare_cycle_to_reference(
    measured: BenchmarkCycleData,
    reference: dict[str, float],
) -> tuple[BenchmarkComparison, ...]:
    """Compare every shared metric between ``measured`` and ``reference``.

    Iterates the four canonical metric names; for each, computes the
    signed % difference and tags it ``passed`` iff
    ``|pct_diff| < BENCHMARK_TOLERANCE_PCT``. Metric names absent from
    ``reference`` are skipped silently — the caller decides which subset
    of metrics the campaign tracks.

    Parameters
    ----------
    measured : aggregated cycle data (usually the integrated last-4-cycles
        record produced by ``analyze_benchmark``).
    reference : published-benchmark values keyed by metric name.

    Returns
    -------
    Tuple of ``BenchmarkComparison`` records, one per metric present in
    both ``measured`` and ``reference``.
    """
    metric_names = ("c_l_max", "c_l_min", "c_d_mean", "c_l_hysteresis_area")
    out: list[BenchmarkComparison] = []
    for name in metric_names:
        if name not in reference:
            continue
        meas = float(getattr(measured, name))
        ref = float(reference[name])
        pct = _signed_pct_diff(meas, ref)
        out.append(
            BenchmarkComparison(
                metric_name=name,
                measured=meas,
                reference=ref,
                pct_diff=pct,
                # Plan says "within ±15%" — inclusive boundary.
                passed=abs(pct) <= BENCHMARK_TOLERANCE_PCT,
            )
        )
    return tuple(out)


def _integrate_cycles(
    kept: Iterable[BenchmarkCycleData],
) -> BenchmarkCycleData:
    """Integrate (here: average) per-cycle metrics over the kept cycles.

    The "integrate the last 4" spec wording reduces to a per-metric mean
    for the four scalar aggregates we track: max / min / mean / area.
    Each per-cycle value is already a cycle-level scalar; averaging four
    of them produces the multi-cycle steady-state estimate.

    The returned ``BenchmarkCycleData.cycle_index`` is set to ``-1`` to
    flag the record as an aggregate (not one of the raw cycles).
    """
    kept_t = tuple(kept)
    if not kept_t:
        raise ValueError("No cycles to integrate")
    n = len(kept_t)
    return BenchmarkCycleData(
        cycle_index=-1,
        c_l_max=sum(c.c_l_max for c in kept_t) / n,
        c_l_min=sum(c.c_l_min for c in kept_t) / n,
        c_d_mean=sum(c.c_d_mean for c in kept_t) / n,
        c_l_hysteresis_area=sum(c.c_l_hysteresis_area for c in kept_t) / n,
    )


def analyze_benchmark(
    cycles: Iterable[BenchmarkCycleData],
    reference: dict[str, float],
    *,
    k_reduced: float,
    reynolds: float,
    reference_source: str,
) -> BenchmarkResult:
    """Run the sub-spike 0.6c.2 analysis: discard 1 cycle, integrate 4, compare.

    Parameters
    ----------
    cycles : iterable of per-cycle data records, ordered by cycle index.
        The first ``BENCHMARK_CYCLES_DISCARD`` cycles (= 1) are discarded
        as initial-transient; the remaining cycles are integrated.
    reference : published-benchmark values keyed by metric name.
    k_reduced : reduced-frequency parameter of the simulation. Recorded
        for the result so the operator can verify it sits in the
        ``[BENCHMARK_K_REDUCED_MIN, BENCHMARK_K_REDUCED_MAX]`` band.
    reynolds : Reynolds number of the simulation. Recorded similarly for
        the ``[BENCHMARK_RE_MIN, BENCHMARK_RE_MAX]`` band check.
    reference_source : free-form provenance string (e.g., "McAlister/Carr
        UH110A 1978, fig 7c").

    Returns
    -------
    ``BenchmarkResult`` with the per-metric comparisons and overall pass flag.

    Notes
    -----
    The pass criterion has two parts: (a) every metric within ±15% of
    its reference (``all_metrics_within_15pct``), AND (b) at least one
    metric was actually compared (``passed`` is False if the reference
    dict shares no metrics with the cycle data, since the gate would be
    vacuously true otherwise).
    """
    cycles_t = tuple(cycles)
    if len(cycles_t) < BENCHMARK_CYCLES_DISCARD + 1:
        raise ValueError(
            f"Need at least {BENCHMARK_CYCLES_DISCARD + 1} cycles "
            f"(discard {BENCHMARK_CYCLES_DISCARD}, integrate >=1); "
            f"got {len(cycles_t)}"
        )

    kept = cycles_t[BENCHMARK_CYCLES_DISCARD:]
    integrated = _integrate_cycles(kept)
    comparisons = compare_cycle_to_reference(integrated, reference)

    all_within = len(comparisons) > 0 and all(c.passed for c in comparisons)

    return BenchmarkResult(
        k_reduced=float(k_reduced),
        reynolds=float(reynolds),
        cycles=cycles_t,
        reference_source=reference_source,
        comparisons=comparisons,
        all_metrics_within_15pct=all_within,
        passed=all_within,
    )


# ----- aggregate Spike 0.6c result ------------------------------------------


@dataclass(frozen=True)
class Spike06cResult:
    """Aggregate roll-up of sub-spikes 0.6c.1 and 0.6c.2.

    The Phase 4 launch gate (``scripts/launch_phase4.py``) checks for the
    ``phase0/spike_0_6c/PASS`` marker, which is written by the aggregator
    iff ``overall_passed`` is True. ``overall_passed`` is True iff BOTH
    sub-spikes individually passed.
    """

    sub_06c_1: Tier1CfgSanityResult
    sub_06c_2: BenchmarkResult
    overall_passed: bool


def analyze_spike_06c(
    sub_1: Tier1CfgSanityResult,
    sub_2: BenchmarkResult,
) -> Spike06cResult:
    """Combine the two sub-spike results into the gating ``Spike06cResult``.

    ``overall_passed`` is the strict AND of the two sub-spike pass flags
    (no partial-credit / informational override — Phase 4 launch is hard
    gated on both passing per spec line 1844).
    """
    return Spike06cResult(
        sub_06c_1=sub_1,
        sub_06c_2=sub_2,
        overall_passed=bool(sub_1.passed and sub_2.passed),
    )
