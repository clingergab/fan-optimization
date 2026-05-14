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

2. **0.6c.2 — NACA 0012 numerical-consistency benchmark.** Run a NACA 0012
   oscillating-airfoil case (pitching about quarter-chord at
   ``k_reduced ≈ 0.55``, ``Re ≈ 40k``, ±10°) through the working Tier-1
   cfg. PASS iff TWO internal-consistency gates clear:

   (a) **Cycle-to-cycle convergence.** After discarding cycle 0
       (initial transient), the relative range of ``c_l_max``,
       ``c_l_min``, and ``c_d_mean`` across kept cycles is < 2% each.
   (b) **C_L symmetry.** With mean α = 0° and a symmetric airfoil,
       ``|⟨c_l_max⟩ + ⟨c_l_min⟩| / max(|⟨c_l_max⟩|, |⟨c_l_min⟩|) < 5%``
       on the kept-cycle averages.

   **Why no literature reference comparison.** A targeted literature
   survey (2026-05) confirmed the (Re=40k, k=0.55, ±10°, mean α=0°,
   c/4 pivot) operating point is in a gap between the well-studied
   low-Re/low-k attached-pitching regime and the moderate-Re/high-k
   dynamic-stall regime. The nearest published neighbors (Kim & Chang
   2013 at Re=48k k=0.1 ±6°; MDPI 2025 at Re=66k) disagree on k,
   amplitude, and Re. At this Re, inter-study scatter on C_L_max is
   ≥25% even between published studies. The ±15% literature-comparison
   gate that early draft protocols enumerated is **not defensible**.
   Internal-consistency gates substitute: they validate that SU2 is
   solving its own equations consistently. Quantitative cross-solver
   validation (SU2 vs PyFR) is deferred to Phase 5 where PyFR is
   already provisioned.

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
    "CONVERGENCE_TOLERANCE_PCT",
    "SYMMETRY_TOLERANCE_PCT",
    "BENCHMARK_K_REDUCED_MIN",
    "BENCHMARK_K_REDUCED_MAX",
    "BENCHMARK_RE_MIN",
    "BENCHMARK_RE_MAX",
    "BENCHMARK_CYCLES_TOTAL",
    "BENCHMARK_CYCLES_DISCARD",
    "CONVERGENCE_METRICS",
    "MACH_UNSTEADY_LOCK",
    "Tier1CfgSanityResult",
    "BenchmarkCycleData",
    "ConvergenceCheck",
    "SymmetryCheck",
    "BenchmarkResult",
    "Spike06cResult",
    "check_tier1_cfg_sanity",
    "check_convergence",
    "check_symmetry",
    "analyze_benchmark",
    "analyze_spike_06c",
]


# ----- locks (from §Phase 0 Spike 0.6c + Round-9 HIGH-12) -------------------

CONVERGENCE_TOLERANCE_PCT: float = 2.0
"""Per-metric cycle-to-cycle convergence gate: relative range across the
kept cycles must be < 2% for ``c_l_max``, ``c_l_min``, ``c_d_mean``.

Sourced from the researcher's 2026-05 internal-consistency proposal —
'loop closes within 2% over settling cycles' — translated to a per-metric
relative-range check on the kept cycles."""

SYMMETRY_TOLERANCE_PCT: float = 5.0
"""C_L symmetry gate: ``|⟨c_l_max⟩ + ⟨c_l_min⟩| / max(|⟨c_l_max⟩|,
|⟨c_l_min⟩|) < 5%`` on kept-cycle averages. Required by NACA-0012
geometric symmetry combined with mean α = 0°.

Sourced from the researcher's 2026-05 internal-consistency proposal —
'C_L_min ≈ −C_L_max within 5%'."""

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
"""5 cycles total; discard the first (initial-transient) cycle, keep the
remaining 4 for the convergence + symmetry gates. Source: spec line 1843."""

CONVERGENCE_METRICS: tuple[str, ...] = ("c_l_max", "c_l_min", "c_d_mean")
"""Metrics gated by the convergence check. ``c_l_hysteresis_area`` is
intentionally excluded — at k=0.55 the loop is near sign-inversion and a
2% relative-range gate on a near-zero quantity is numerically unstable.
The hysteresis area is logged as a diagnostic in ``BenchmarkResult``."""

MACH_UNSTEADY_LOCK: float = 1e-9
"""Tier-1 unsteady MACH value (Round-9 HIGH-12 = C12). The steady tiers
(Tier -1 / Tier 0) use ``MACH = 0.0064``; ``CROSS_TIER`` does NOT carry
``MACH`` — the value lives in ``TIER_SPECIFIC[1]`` per §9.4.1."""


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


# ----- sub-spike 0.6c.2 (NACA 0012 numerical-consistency benchmark) ---------


@dataclass(frozen=True)
class BenchmarkCycleData:
    """Per-cycle lift/drag aggregates extracted from one pitching cycle.

    * ``c_l_max`` — peak lift coefficient over the cycle.
    * ``c_l_min`` — trough lift coefficient over the cycle.
    * ``c_d_mean`` — cycle-mean drag coefficient.
    * ``c_l_hysteresis_area`` — signed area inside the ``C_l(α)`` loop.
      Logged as diagnostic; not gated (see ``CONVERGENCE_METRICS``).
    """

    cycle_index: int
    c_l_max: float
    c_l_min: float
    c_d_mean: float
    c_l_hysteresis_area: float


@dataclass(frozen=True)
class ConvergenceCheck:
    """Per-metric cycle-to-cycle convergence record.

    ``relative_range_pct`` is ``100 * (max - min) / |mean|`` over the kept
    cycles. ``passed = relative_range_pct <= CONVERGENCE_TOLERANCE_PCT``.

    For a near-zero mean the relative form is unstable; the check uses
    ``inf`` for ``relative_range_pct`` when ``|mean| < 1e-9`` and a
    non-zero range, which fails the gate by construction.
    """

    metric_name: str
    values: tuple[float, ...]
    mean: float
    relative_range_pct: float
    passed: bool


@dataclass(frozen=True)
class SymmetryCheck:
    """C_L symmetry record on kept-cycle averages.

    ``asymmetry_pct = 100 * |c_l_max_mean + c_l_min_mean| /
    max(|c_l_max_mean|, |c_l_min_mean|)``. With ``mean α = 0°`` and the
    NACA 0012's geometric symmetry, this value should be small.
    """

    c_l_max_mean: float
    c_l_min_mean: float
    asymmetry_pct: float
    passed: bool


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregated outcome of sub-spike 0.6c.2.

    Pass criterion: ``convergence_passed AND symmetry_passed``. Both gates
    are internal-consistency checks on the SU2 output alone — no
    literature-reference comparison (see module docstring).
    """

    k_reduced: float
    reynolds: float
    cycles: tuple[BenchmarkCycleData, ...]
    convergence: tuple[ConvergenceCheck, ...]
    symmetry: SymmetryCheck
    diagnostic_hysteresis_area_mean: float
    convergence_passed: bool
    symmetry_passed: bool
    passed: bool


def _relative_range_pct(values: tuple[float, ...]) -> tuple[float, float]:
    """Return ``(mean, relative_range_pct)`` for the value sequence.

    ``relative_range_pct = 100 * (max - min) / |mean|`` if ``|mean| >= 1e-9``;
    otherwise returns ``(mean, inf)`` if the range is non-zero and
    ``(mean, 0.0)`` if all values are identical to zero.
    """
    if not values:
        raise ValueError("relative_range_pct requires at least one value")
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    rng = max(values) - min(values)
    if abs(mean) < 1e-9:
        return mean, 0.0 if rng == 0.0 else math.inf
    return mean, 100.0 * rng / abs(mean)


def check_convergence(
    kept: Iterable[BenchmarkCycleData],
    *,
    metrics: tuple[str, ...] = CONVERGENCE_METRICS,
    tolerance_pct: float = CONVERGENCE_TOLERANCE_PCT,
) -> tuple[ConvergenceCheck, ...]:
    """Per-metric cycle-to-cycle convergence check over the kept cycles.

    For each metric in ``metrics``, extract the per-cycle values, compute
    the relative range, and tag ``passed = relative_range_pct <=
    tolerance_pct``.
    """
    kept_t = tuple(kept)
    if not kept_t:
        raise ValueError("check_convergence requires at least one kept cycle")
    out: list[ConvergenceCheck] = []
    for name in metrics:
        values = tuple(float(getattr(c, name)) for c in kept_t)
        mean, rrange = _relative_range_pct(values)
        out.append(
            ConvergenceCheck(
                metric_name=name,
                values=values,
                mean=mean,
                relative_range_pct=rrange,
                passed=rrange <= tolerance_pct,
            )
        )
    return tuple(out)


def check_symmetry(
    kept: Iterable[BenchmarkCycleData],
    *,
    tolerance_pct: float = SYMMETRY_TOLERANCE_PCT,
) -> SymmetryCheck:
    """C_L symmetry check on kept-cycle averages.

    With ``mean α = 0°`` and a symmetric airfoil, ``⟨c_l_max⟩ ≈ -⟨c_l_min⟩``
    is required by physics. A non-zero asymmetry indicates either a
    geometry / motion-axis error or a numerical bias.
    """
    kept_t = tuple(kept)
    if not kept_t:
        raise ValueError("check_symmetry requires at least one kept cycle")
    c_l_max_mean = sum(c.c_l_max for c in kept_t) / len(kept_t)
    c_l_min_mean = sum(c.c_l_min for c in kept_t) / len(kept_t)
    denom = max(abs(c_l_max_mean), abs(c_l_min_mean))
    if denom < 1e-9:
        # Both averages near zero — vacuously symmetric, but the spike has
        # almost certainly failed for other reasons. Pass the check; the
        # convergence gate will catch the degenerate case.
        asym_pct = 0.0
    else:
        asym_pct = 100.0 * abs(c_l_max_mean + c_l_min_mean) / denom
    return SymmetryCheck(
        c_l_max_mean=c_l_max_mean,
        c_l_min_mean=c_l_min_mean,
        asymmetry_pct=asym_pct,
        passed=asym_pct <= tolerance_pct,
    )


def analyze_benchmark(
    cycles: Iterable[BenchmarkCycleData],
    *,
    k_reduced: float,
    reynolds: float,
) -> BenchmarkResult:
    """Run the sub-spike 0.6c.2 analysis: discard 1 cycle, run gates on the rest.

    Parameters
    ----------
    cycles : iterable of per-cycle data records, ordered by cycle index.
        The first ``BENCHMARK_CYCLES_DISCARD`` cycles (= 1) are discarded
        as initial-transient; the remaining cycles feed the gates.
    k_reduced : reduced-frequency parameter of the simulation. Recorded
        so the operator can verify it sits in
        ``[BENCHMARK_K_REDUCED_MIN, BENCHMARK_K_REDUCED_MAX]``.
    reynolds : Reynolds number of the simulation. Recorded similarly.

    Returns
    -------
    ``BenchmarkResult`` with the convergence + symmetry records and
    overall pass flag. The hysteresis-area cycle-mean is logged
    diagnostically (not gated).
    """
    cycles_t = tuple(cycles)
    if len(cycles_t) < BENCHMARK_CYCLES_DISCARD + 1:
        raise ValueError(
            f"Need at least {BENCHMARK_CYCLES_DISCARD + 1} cycles "
            f"(discard {BENCHMARK_CYCLES_DISCARD}, keep >=1); "
            f"got {len(cycles_t)}"
        )

    kept = cycles_t[BENCHMARK_CYCLES_DISCARD:]
    convergence = check_convergence(kept)
    symmetry = check_symmetry(kept)

    convergence_passed = all(c.passed for c in convergence)
    symmetry_passed = symmetry.passed
    hysteresis_mean = sum(c.c_l_hysteresis_area for c in kept) / len(kept)

    return BenchmarkResult(
        k_reduced=float(k_reduced),
        reynolds=float(reynolds),
        cycles=cycles_t,
        convergence=convergence,
        symmetry=symmetry,
        diagnostic_hysteresis_area_mean=hysteresis_mean,
        convergence_passed=convergence_passed,
        symmetry_passed=symmetry_passed,
        passed=convergence_passed and symmetry_passed,
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
