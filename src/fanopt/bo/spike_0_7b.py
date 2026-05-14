"""Spike 0.7b — BO infrastructure scaling sanity check.

Implements `docs/plan_R11.md §Phase 0 Spike 0.7b` (lines ~1855-1858).

**Purpose.** Gate Phase 4 launch on three throughput / behaviour checks
against the high-D BO infrastructure (architecture bandit + TuRBO +
multi-fidelity GP) at the spec's 37-46-dimensional search space using a
*synthetic* objective (no CFD). The three checks are:

1.  **GP fit-time gate.** Wall-clock per-iteration GP fit must stay
    ≤ ``GP_FIT_TIME_GATE_S = 60 s``. If we cannot fit a smooth synthetic
    quadratic in 60 s at 37-46 D, no CFD-coupled run will ever close the
    loop inside Phase 4's budget.
2.  **Architecture-bandit promotion gate.** With the synthetic objective
    the bandit must promote ``K_PROMOTED_SANITY = 4`` architectures. K = 4
    is hard-coded **for this synthetic sanity check only** — the
    production K is set later from Phase 3's measured R² on the
    multi-fidelity bridge (per the upstream spec).
3.  **TuRBO trust-region update gate.** Trust regions must shrink after a
    failure-count increment and grow after a success-threshold crossing.

The three gates correspond directly to the spike's pass criteria and to
the fallback decisions documented in
``docs/spike_0_7b_protocol.md``.

**GP backend.** Production runs use BoTorch's ``SingleTaskGP``. This
library is GP-backend-agnostic: it only consumes per-iteration timings.
The driver script ``scripts/run_spike_0_7b.py`` tries BoTorch first and
falls back to a numpy-only RBF GP (Cholesky solve) on sandboxes without
the ``[bo]`` extras installed. Either way, the gate is a wall-clock
budget — the fallback yields a fair lower bound (numpy/Cholesky on CPU
is in the same complexity class as BoTorch's exact GP on CPU).

References:

* Spec: ``docs/plan_R11.md §Phase 0 Spike 0.7b``.
* Locks: ``GP_FIT_TIME_GATE_S = 60`` (spec); K = 4 for the synthetic
  sanity check (spec).
* Fallbacks if the spike fails: switch TuRBO → SAASBO with ≤ 500
  inducing points, or freeze categoricals to shrink D. See
  ``docs/spike_0_7b_protocol.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats.qmc import LatinHypercube

__all__ = [
    "GP_FIT_TIME_GATE_S",
    "N_LHS_DEFAULT",
    "K_PROMOTED_SANITY",
    "D_DEFAULT",
    "D_MIN",
    "D_MAX",
    "EPISTEMIC_NOISE_FLOOR_DEFAULT",
    "calibrate_epistemic_noise_floor",
    "synthetic_objective",
    "lhs_sample",
    "GpFitTiming",
    "TurboTRRecord",
    "ArchitectureBanditRecord",
    "Spike07bResult",
    "analyze_07b",
]


# ----- locks -----------------------------------------------------------------

GP_FIT_TIME_GATE_S: float = 60.0
"""Per-iteration GP fit wall-clock budget. Source: spec line 1856."""

N_LHS_DEFAULT: int = 8
"""LHS samples per spike run. Spec range: 5-10."""

K_PROMOTED_SANITY: int = 4
"""Architectures the bandit must promote with the synthetic objective.

Hard-coded **only** for the synthetic-objective sanity check. Production
K is derived later from Phase 3's measured R² on the multi-fidelity
bridge.
"""

D_DEFAULT: int = 40
"""Default design-space dimensionality (mid-range of spec's 37-46)."""

D_MIN: int = 37
D_MAX: int = 46


# ----- epistemic noise floor (Round-1 §0 row 43 lock) ------------------------

EPISTEMIC_NOISE_FLOOR_DEFAULT: float = 1.0e-6
"""Floor used in the GP's `train_Yvar` constant.

**This is a PLACEHOLDER** awaiting the canonical calibration step that
Spike 0.7b is supposed to perform once per campaign:

  Run M ≥ 3 replicate Tier-1 evaluations on the same design (perturbed
  initial conditions). Compute the per-design variance of J_fan across
  replicates. The floor is then `max(measured_variance, 1.0e-6)`.

The Round-1 §0 row 43 lock requires this calibrated value to be passed
to `SingleTaskMultiFidelityGP(train_Yvar=...)` as a SCALAR per tier
(NOT per-observation — physics-driven limit-cycle variance from vortex
shedding would otherwise tag the very designs we want to find as
low-confidence and weaken the cross-fidelity correlation kernel).

Until the calibration replicates run, downstream consumers should treat
`EPISTEMIC_NOISE_FLOOR_DEFAULT` as a conservative placeholder that
errs on the side of "trust the GP slightly less". Replace via
`calibrate_epistemic_noise_floor(replicate_j_fans)` once the data is in
hand and pin the result in `configs/bo_epistemic_noise.json`.

Per-tier floors are permitted (the lock says "tier-specific noise floors
are allowed but each is a fixed scalar"); a single scalar is the default.
"""


def calibrate_epistemic_noise_floor(
    replicate_J_fans: Iterable[float],
    *,
    floor_min: float = EPISTEMIC_NOISE_FLOOR_DEFAULT,
) -> float:
    """Compute `max(measured_variance, floor_min)` from M replicate J_fan values.

    Spec reference: docs/plan_R11.md §0 row 43.

    The replicates must be drawn from the **same physical design** at the
    **same fidelity tier**, perturbing only initial conditions / random
    seeds. M = 3 is the minimum the spike validates; more is better.

    Returns the per-tier noise variance to feed to
    `SingleTaskMultiFidelityGP(train_Yvar=...)` as a constant scalar.
    """
    values = np.asarray(list(replicate_J_fans), dtype=float)
    if values.size < 2:
        raise ValueError(f"need ≥ 2 replicates to estimate variance; got {values.size}")
    # Sample variance (ddof=1) — same convention as inertia.analyze_trials.
    measured = float(values.var(ddof=1))
    return max(measured, floor_min)


# ----- synthetic objective ---------------------------------------------------


def synthetic_objective(
    x: np.ndarray,
    *,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> float:
    """High-D smooth synthetic objective.

    Sum of quadratics centred at 0.5 with a small number of pairwise
    interaction terms. Smooth and differentiable, so a vanilla GP with
    an RBF kernel should fit it well in ≤ 60 s at 37-46 D given the
    sample sizes the spike uses (≤ a few hundred).

    Parameters
    ----------
    x : array of shape (d,) — point in [0, 1]^d.
    noise_std : optional observation-noise std (default 0, deterministic).
    rng : numpy ``Generator`` used to draw the noise sample; only consulted
        when ``noise_std > 0``.

    Returns
    -------
    float — scalar objective (lower is better; the spike only cares about
    fit time + bandit behaviour, not minimisation).

    Notes
    -----
    * Sum of axis-aligned quadratics gives smooth curvature in every
      coordinate so the GP length-scales are well-defined.
    * 5 fixed cross-coordinate interaction terms (one per pair from a
      short index list) inject mild correlation without breaking
      smoothness — enough for the GP fit to be non-trivial but still
      ≤ 60 s.
    """
    x = np.asarray(x, dtype=float).ravel()
    d = x.size
    if d == 0:
        raise ValueError("x must be non-empty")
    centred = x - 0.5
    base = float(np.sum(centred * centred))

    # A handful of interaction terms — indexed modulo d so the formula
    # works at any dimensionality in [D_MIN, D_MAX] and beyond.
    interaction_pairs = (
        (0, 1),
        (2, 5),
        (3, 7),
        (6, 9),
        (4, 8),
    )
    interaction = 0.0
    for i, j in interaction_pairs:
        ii = i % d
        jj = j % d
        interaction += 0.25 * centred[ii] * centred[jj]

    value = base + interaction
    if noise_std > 0.0:
        if rng is None:
            raise ValueError("noise_std > 0 requires an rng (numpy.random.Generator)")
        value += float(noise_std * rng.standard_normal())
    return value


# ----- Latin-hypercube sampler ----------------------------------------------


def lhs_sample(n: int, d: int, *, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` Latin-hypercube samples in [0, 1]^d via scipy.

    Parameters
    ----------
    n : number of samples (must be ≥ 1).
    d : dimensionality (must be ≥ 1).
    rng : numpy ``Generator`` used to seed scipy's LatinHypercube sampler.

    Returns
    -------
    np.ndarray of shape (n, d), each column being a permutation over the
    n equal-width strata of [0, 1].
    """
    if n < 1:
        raise ValueError(f"n must be ≥ 1, got {n}")
    if d < 1:
        raise ValueError(f"d must be ≥ 1, got {d}")
    seed = int(rng.integers(0, 2**31 - 1))
    sampler = LatinHypercube(d=d, seed=seed)
    return np.asarray(sampler.random(n=n), dtype=float)


# ----- per-iteration records -------------------------------------------------


@dataclass(frozen=True)
class GpFitTiming:
    """One iteration of GP fit-time measurement."""

    iteration: int
    wall_time_s: float
    n_train: int
    d: int
    passed: bool
    """True iff ``wall_time_s ≤ GP_FIT_TIME_GATE_S``."""


@dataclass(frozen=True)
class TurboTRRecord:
    """One iteration of TuRBO trust-region state."""

    iteration: int
    center: tuple[float, ...]
    length: float
    success_count: int
    failure_count: int


@dataclass(frozen=True)
class ArchitectureBanditRecord:
    """One architecture screened by the outer bandit."""

    architecture_id: str
    screened_count: int
    promoted: bool


@dataclass(frozen=True)
class Spike07bResult:
    """Aggregated outcome of all three pass-gates."""

    gp_fit_timings: tuple[GpFitTiming, ...]
    turbo_trs: tuple[TurboTRRecord, ...]
    bandit_records: tuple[ArchitectureBanditRecord, ...]
    k_promoted: int
    all_gp_fits_under_60s: bool
    k_promoted_passes: bool
    turbo_trs_update_correctly: bool
    passed: bool


# ----- aggregator ------------------------------------------------------------


def _verify_turbo_tr_updates(turbo_trs: Sequence[TurboTRRecord]) -> bool:
    """Return True iff the TR log shows at least one valid shrink and one valid grow.

    *Shrink:* between two consecutive records (same TR), the failure
    count increases and the length decreases.
    *Grow:* between two consecutive records (same TR), the success count
    increases and the length increases.

    This is a behaviour gate, not a sufficiency check: the spike's job is
    to confirm the TR-update wiring works, not to evaluate TuRBO's full
    state machine.
    """
    if len(turbo_trs) < 2:
        return False
    saw_shrink = False
    saw_grow = False
    prev = turbo_trs[0]
    for cur in turbo_trs[1:]:
        if cur.failure_count > prev.failure_count and cur.length < prev.length:
            saw_shrink = True
        if cur.success_count > prev.success_count and cur.length > prev.length:
            saw_grow = True
        prev = cur
    return saw_shrink and saw_grow


def analyze_07b(
    gp_fit_timings: Iterable[GpFitTiming],
    turbo_trs: Iterable[TurboTRRecord],
    bandit_records: Iterable[ArchitectureBanditRecord],
    k_promoted_expected: int = K_PROMOTED_SANITY,
) -> Spike07bResult:
    """Aggregate per-iteration logs into the three pass-gates.

    Parameters
    ----------
    gp_fit_timings : iterable of ``GpFitTiming`` — one per BO iteration.
    turbo_trs : iterable of ``TurboTRRecord`` — TR state over the run.
    bandit_records : iterable of ``ArchitectureBanditRecord`` — one per
        screened architecture, with ``promoted`` set by the bandit.
    k_promoted_expected : expected number of promoted architectures
        (default ``K_PROMOTED_SANITY = 4``).

    Returns
    -------
    ``Spike07bResult`` with all three gates and the overall pass flag.
    """
    gp_t = tuple(gp_fit_timings)
    turbo_t = tuple(turbo_trs)
    bandit_t = tuple(bandit_records)

    all_under_60s = len(gp_t) > 0 and all(t.passed for t in gp_t)
    k_promoted = sum(1 for r in bandit_t if r.promoted)
    k_passes = k_promoted == k_promoted_expected
    tr_ok = _verify_turbo_tr_updates(turbo_t)
    overall = all_under_60s and k_passes and tr_ok

    return Spike07bResult(
        gp_fit_timings=gp_t,
        turbo_trs=turbo_t,
        bandit_records=bandit_t,
        k_promoted=k_promoted,
        all_gp_fits_under_60s=all_under_60s,
        k_promoted_passes=k_passes,
        turbo_trs_update_correctly=tr_ok,
        passed=overall,
    )
