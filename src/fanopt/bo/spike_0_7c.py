"""Spike 0.7c -- Sobol-vs-BO iso-compute comparison harness.

Implements ``docs/plan_R11.md §Phase 0 Spike 0.7c`` (lines ~1859-1867).

**Purpose.** Verify that the production GP + qMFKG BO inner-loop beats a
uniform-random Sobol baseline by at least 5% on at least 2 of 3 fixed
CFD-hour budgets ``B in {30, 100, 300}``. The total experiment is
30 + 100 + 300 = 430 cumulative-compute hours, booked under Phase 0
(NOT against the 1000-h Phase 4 stop rule -- per the spec's H7 budget-
allocation lock; the Phase 4 budget counter starts at the
``phase4-launch`` git tag created by ``scripts/launch_phase4.py``).

**Budget accounting -- locked rules.**

1. "Hours" means cumulative tier-(-1) + tier-0 + tier-1 compute including
   Sobol seed runs. There is no "free" Sobol allowance.
2. The three budgets run serially with results gating the next: B=30
   must complete before B=100 starts (so the 100-h GP uses the 30-h
   seed data); B=100 must complete before B=300 starts.
3. The comparison axis is cumulative compute, not wall-clock. Colab Pro
   CPU runs 2-4 parallel sessions, so 100 h of cumulative compute is
   roughly 25-50 h of wall-clock.

**Pass criterion.** BO's best ``J_fan`` exceeds (i.e., is lower than, by
the lower-is-better convention) Sobol's best ``J_fan`` by at least
``BO_MINUS_SOBOL_PCT_GATE = 5.0%`` on at least
``BUDGETS_PASS_THRESHOLD = 2`` of the 3 budgets.

**Fallback if the spike fails.** Per spec, the choice depends on which
sub-axis the GP fit time exceeded 60 s on:

* High-D GP fit blocking -> switch to SAASBO with the <=500-inducing-
  point cap from §6.2.2.
* Wide architecture set blocking -> collapse Layer 2 categoricals
  (e.g., pin Layer 2 activation profile to one combination).

If BO simply under-exploits without hitting the 60-s gate, the
recommendation is to re-tune TuRBO / qMFKG hyperparameters (returned as
``None`` from the analyzer; surfaced in the operator protocol).

**Sobol-seed reuse.** The 50 Sobol-seed runs at tier -1 also double as
the GP seed set for Phase 4 -- so the day's compute is not wasted even
if BO does win cleanly. Records are written to
``gdrive/fan-optimization/phase0/sobol_seed/results.jsonl``.

**Convention: lower-is-better.** ``J_fan`` is a minimisation objective
(per the panel-aero loss formulation in §6). "Best so far" therefore
means the running minimum, and "BO beats Sobol by X%" means BO's best
``J_fan`` is X% lower. The 5% gate is computed as
``(sobol_best - bo_best) / |sobol_best| * 100``.

References:

* Spec: ``docs/plan_R11.md §Phase 0 Spike 0.7c`` (lines ~1859-1867).
* H7 budget-allocation lock: 430 h booked under Phase 0 only.
* §6.2.3 stop-rule accounting: same cumulative-compute axis.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

__all__ = [
    "BUDGETS_HOURS",
    "BUDGETS_TOTAL_HOURS",
    "BO_MINUS_SOBOL_PCT_GATE",
    "BUDGETS_PASS_THRESHOLD",
    "SOBOL_SEED_COUNT",
    "BO_ITERATIONS_DEFAULT",
    "IsoComputePoint",
    "Spike07cResult",
    "compute_iso_compute_point",
    "analyze_spike_07c",
    "record_to_jsonl",
]


# ----- locks -----------------------------------------------------------------

BUDGETS_HOURS: tuple[int, int, int] = (30, 100, 300)
"""The three iso-compute checkpoints (cumulative CFD hours). Source:
spec line 1862 -- "given equal CFD budgets B in {30, 100, 300} hours"."""

BUDGETS_TOTAL_HOURS: int = sum(BUDGETS_HOURS)
"""430 h total. Booked under Phase 0 per the H7 budget-allocation lock;
does NOT count against the Phase 4 1000-h stop rule."""

BO_MINUS_SOBOL_PCT_GATE: float = 5.0
"""BO must beat Sobol by >=5% on best-J_fan at each checkpoint to count
as a win for that budget. Source: spec line 1863."""

BUDGETS_PASS_THRESHOLD: int = 2
"""BO must beat Sobol on at least this many of the three budgets to pass
the spike. Source: spec line 1863 -- ">= 5% on at least 2 of the 3
budgets"."""

SOBOL_SEED_COUNT: int = 50
"""Number of Sobol seed evaluations at tier -1. Source: spec line 1861."""

BO_ITERATIONS_DEFAULT: int = 100
"""Number of BO iterations on the same architecture set with production
GP + qMFKG. Source: spec line 1861."""


# ----- data classes ---------------------------------------------------------


@dataclass(frozen=True)
class IsoComputePoint:
    """One iso-compute budget checkpoint.

    Attributes
    ----------
    budget_hours :
        The CFD-hour cap for this checkpoint (one of ``BUDGETS_HOURS``).
    sobol_best_j_fan :
        Best (= minimum) ``J_fan`` from Sobol records whose cumulative
        wall_time was <= ``budget_hours``.
    bo_best_j_fan :
        Best (= minimum) ``J_fan`` from BO records whose cumulative
        wall_time was <= ``budget_hours``.
    bo_minus_sobol_pct :
        ``(sobol_best - bo_best) / abs(sobol_best) * 100``. Positive
        means BO is better (lower J_fan).
    bo_beats :
        True iff ``bo_minus_sobol_pct >= BO_MINUS_SOBOL_PCT_GATE``.
    sample_count :
        Mapping ``{"sobol": n_sobol_used, "bo": n_bo_used}`` -- how many
        records from each stream fit inside the cumulative-compute cap.
    """

    budget_hours: int
    sobol_best_j_fan: float
    bo_best_j_fan: float
    bo_minus_sobol_pct: float
    bo_beats: bool
    sample_count: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Spike07cResult:
    """Aggregated outcome across all three budgets.

    Attributes
    ----------
    per_budget :
        One ``IsoComputePoint`` per budget in ``BUDGETS_HOURS`` order.
    n_budgets_bo_beats :
        Count of budgets where ``bo_beats`` is True.
    passed :
        True iff ``n_budgets_bo_beats >= BUDGETS_PASS_THRESHOLD``.
    fallback_recommendation :
        ``None`` if passed, else one of:

        * ``"saasbo"`` -- spec fallback when high-D GP fit blocks
          (switch to SAASBO with <=500 inducing points).
        * ``"fix_architecture_set"`` -- spec fallback when a wide
          architecture set blocks (collapse Layer 2 categoricals).
        * ``"retune_acquisition"`` -- BO under-exploits without
          breaching the 60-s GP gate; re-tune TuRBO / qMFKG
          hyperparameters (not a spec-listed fallback but the natural
          next step before reaching for the bigger hammers).
    """

    per_budget: tuple[IsoComputePoint, ...]
    n_budgets_bo_beats: int
    passed: bool
    fallback_recommendation: str | None


# ----- per-budget aggregation -----------------------------------------------


def _truncate_at_budget(
    records: Iterable[Mapping[str, float]],
    budget_hours: float,
    *,
    wall_time_key: str,
) -> list[Mapping[str, float]]:
    """Return the prefix of records whose cumulative wall_time <= budget.

    Stops including records as soon as the next record would push the
    cumulative wall_time strictly above ``budget_hours``. This matches
    the spec's locked accounting: only fully-completed evaluations
    count.
    """
    out: list[Mapping[str, float]] = []
    cum = 0.0
    for r in records:
        w = float(r[wall_time_key])
        if w < 0.0:
            raise ValueError(
                f"record has negative {wall_time_key}={w!r}; budget accounting "
                "requires non-negative per-record wall_time"
            )
        if cum + w > budget_hours:
            break
        cum += w
        out.append(r)
    return out


def compute_iso_compute_point(
    budget_hours: int,
    sobol_results: Iterable[Mapping[str, float]],
    bo_results: Iterable[Mapping[str, float]],
    *,
    j_fan_key: str = "j_fan",
    wall_time_key: str = "wall_time_hours",
) -> IsoComputePoint:
    """Compute one budget checkpoint from raw evaluation records.

    Both ``sobol_results`` and ``bo_results`` are streams of dict-like
    records with at least:

    * ``j_fan_key`` -- the panel-aero scalar objective (lower is better).
    * ``wall_time_key`` -- per-record cumulative-compute hours (tier
      -1 / 0 / 1 inclusive per the budget-accounting lock).

    Records are consumed *in order*; the prefix whose cumulative
    wall_time fits inside ``budget_hours`` is used. The "best" within
    that prefix is the minimum ``j_fan``.

    Edge cases
    ----------
    * Empty truncated prefix (budget too small for even one record):
      represented with ``+inf`` for that stream's best, and a 0%
      delta + ``bo_beats=False``. The caller's pass criterion is
      already conservative for tiny budgets.
    * Both streams empty: ``bo_minus_sobol_pct = 0.0``,
      ``bo_beats = False``.
    """
    sobol_used = _truncate_at_budget(
        sobol_results, float(budget_hours), wall_time_key=wall_time_key
    )
    bo_used = _truncate_at_budget(
        bo_results, float(budget_hours), wall_time_key=wall_time_key
    )

    sobol_best = (
        min(float(r[j_fan_key]) for r in sobol_used) if sobol_used else float("inf")
    )
    bo_best = (
        min(float(r[j_fan_key]) for r in bo_used) if bo_used else float("inf")
    )

    if sobol_best == float("inf") and bo_best == float("inf"):
        pct = 0.0
        bo_beats = False
    elif sobol_best == 0.0:
        # Sobol hit exactly zero -- treat any improvement as +inf%,
        # any tie as 0%, any regression as -inf%. The gate is then a
        # simple sign check against +inf, which is True iff bo_best < 0.
        if bo_best < sobol_best:
            pct = float("inf")
            bo_beats = True
        elif bo_best == sobol_best:
            pct = 0.0
            bo_beats = False
        else:
            pct = float("-inf")
            bo_beats = False
    else:
        pct = (sobol_best - bo_best) / abs(sobol_best) * 100.0
        bo_beats = pct >= BO_MINUS_SOBOL_PCT_GATE

    return IsoComputePoint(
        budget_hours=int(budget_hours),
        sobol_best_j_fan=float(sobol_best),
        bo_best_j_fan=float(bo_best),
        bo_minus_sobol_pct=float(pct),
        bo_beats=bool(bo_beats),
        sample_count={"sobol": len(sobol_used), "bo": len(bo_used)},
    )


# ----- overall analyzer -----------------------------------------------------


def _fallback_recommendation(
    gp_fit_time_above_60s_on: tuple[str, ...],
) -> str:
    """Map the GP-fit-time-exceeded sub-axis labels to a spec fallback.

    Per spec line 1865-1866: "The choice between SAASBO and architecture-
    set-reduction depends on which sub-axis the GP-fit-time exceeds 60 s
    for; both fallbacks are pre-specified in Spike 0.7b."

    Heuristics applied in order:

    1. Any label hinting at training-set dimensionality -- ``"high_d"``,
       ``"dimensionality"``, ``"n_train"``, ``"inducing"`` -- routes to
       SAASBO.
    2. Any label hinting at architecture-set width -- ``"architecture"``,
       ``"wide_set"``, ``"categorical"``, ``"layer_2"`` -- routes to
       architecture-set reduction.
    3. Anything else (including the empty tuple, which signals "BO
       under-exploited without breaching the 60-s GP gate") routes to
       ``"retune_acquisition"``.
    """
    labels = tuple(s.lower() for s in gp_fit_time_above_60s_on)
    saasbo_hints = ("high_d", "dimensionality", "n_train", "inducing")
    arch_hints = ("architecture", "wide_set", "categorical", "layer_2")
    if any(any(h in lab for h in saasbo_hints) for lab in labels):
        return "saasbo"
    if any(any(h in lab for h in arch_hints) for lab in labels):
        return "fix_architecture_set"
    return "retune_acquisition"


def analyze_spike_07c(
    iso_compute_points: Iterable[IsoComputePoint],
    *,
    gp_fit_time_above_60s_on: tuple[str, ...] = (),
) -> Spike07cResult:
    """Aggregate the per-budget points into the overall pass/fail result.

    Parameters
    ----------
    iso_compute_points :
        Iterable of ``IsoComputePoint`` (one per budget). Order is
        preserved in the returned ``per_budget`` tuple.
    gp_fit_time_above_60s_on :
        Optional tuple of sub-axis labels indicating where the GP fit
        time exceeded 60 s during the underlying BO run. Empty tuple
        (default) means no 60-s breaches were observed -- under that
        case any fallback recommendation is the "retune_acquisition"
        catch-all rather than a SAASBO or architecture-set switch.

    Returns
    -------
    ``Spike07cResult`` with ``passed = True`` iff BO beat Sobol on at
    least ``BUDGETS_PASS_THRESHOLD`` budgets.
    """
    pts = tuple(iso_compute_points)
    n_beats = sum(1 for p in pts if p.bo_beats)
    passed = n_beats >= BUDGETS_PASS_THRESHOLD
    fallback = None if passed else _fallback_recommendation(gp_fit_time_above_60s_on)
    return Spike07cResult(
        per_budget=pts,
        n_budgets_bo_beats=int(n_beats),
        passed=bool(passed),
        fallback_recommendation=fallback,
    )


# ----- JSONL ledger helpers --------------------------------------------------


def record_to_jsonl(records: Iterable[Mapping[str, object]], path: Path) -> None:
    """Append-write a stream of evaluation records to a JSONL ledger.

    The Sobol-seed ledger lives at
    ``gdrive/fan-optimization/phase0/sobol_seed/results.jsonl`` per the
    spec. This helper is the canonical writer for both that file and
    the BO inner-loop ledger that ``run_spike_0_7c.py`` consumes.

    Each record is serialised on its own line via ``json.dumps`` with
    ``sort_keys=True`` for deterministic byte-exact output. Parent
    directories are created if missing. Use append mode so the ledger
    can be appended-to across multiple budget tranches without
    re-reading or re-writing prior records (matches the serial-budget
    locked rule).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(dict(r), sort_keys=True))
            fh.write("\n")
