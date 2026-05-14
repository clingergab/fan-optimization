#!/usr/bin/env python
"""Spike 0.7c Sobol-seed generator -- writes the 50-sample JSONL ledger.

Spec reference: ``docs/plan_R11.md §Phase 0 Spike 0.7c``; protocol in
``docs/spike_0_7c_protocol.md`` step 1.

What this script does:

1. Draws ``--n`` Sobol samples in [0, 1]^d via
   ``scipy.stats.qmc.Sobol``.
2. Evaluates each sample via ``_evaluate`` -- by default a synthetic
   stub on ``fanopt.bo.spike_0_7b.synthetic_objective``. Swap the hook
   for the production CFD bridge before running this for real.
3. Writes one JSONL record per sample to ``--out`` with at least:
   ``iteration``, ``tier``, ``params_hash``, ``j_fan``,
   ``wall_time_hours``, ``source``.

These records *also serve as the Phase 4 GP initialisation set*, per the
H7 budget-allocation lock. So the canonical ledger path is
``gdrive/fan-optimization/phase0/sobol_seed/results.jsonl``.

The synthetic-stub ``wall_time_hours`` is a placeholder. When the
production CFD bridge replaces ``_evaluate``, the real per-record
wall_time comes from the CFD runner's timing log.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats.qmc import Sobol

REPO_ROOT = Path(__file__).resolve().parents[1]
# Plan §Phase 0 Spike 0.7c (H7 lock) names the canonical path as
# `gdrive/fan-optimization/phase0/sobol_seed/results.jsonl`. We mirror that
# path under the repo's `data/` tree so the script works hermetically; on
# production runs the operator overrides `--out` to the actual Drive path.
DEFAULT_OUT = (
    REPO_ROOT / "data" / "phase0" / "sobol_seed" / "results.jsonl"
)
"""Default sandbox-local path matching the structure of the production Drive
path. Override `--out` to write directly to the Drive location during the
canonical Phase 0 seed run."""

from fanopt.bo.spike_0_7b import synthetic_objective  # noqa: E402
from fanopt.bo.spike_0_7c import SOBOL_SEED_COUNT, record_to_jsonl  # noqa: E402


def _sobol_samples(n: int, d: int, *, seed: int) -> np.ndarray:
    """Draw ``n`` Sobol samples in [0, 1]^d.

    Uses ``scipy.stats.qmc.Sobol`` with scrambling for low-discrepancy
    coverage. The seed is forwarded so two invocations with identical
    flags produce identical sample sets.
    """
    sampler = Sobol(d=d, scramble=True, seed=int(seed))
    return np.asarray(sampler.random(n=n), dtype=float)


def _hash_row(row: np.ndarray) -> str:
    """Cheap deterministic hash of a parameter vector."""
    h = hashlib.sha1(np.asarray(row, dtype=float).tobytes()).hexdigest()
    return h[:16]


def _evaluate(
    x: np.ndarray,
    *,
    tier: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Synthetic evaluation stub.

    Swap this hook for the production CFD bridge when you wire in the
    real runner. Contract:

    * Input: 1-D parameter vector in [0, 1]^d, plus the target tier.
    * Output: dict with at least ``j_fan`` and ``wall_time_hours``.

    The synthetic stub returns ``synthetic_objective(x)`` and a
    placeholder ``wall_time_hours`` drawn from a small log-uniform
    band so the ledger is non-degenerate for harness-level testing.
    """
    j = float(synthetic_objective(x))
    # Synthetic wall_time: tier -1 should be cheap (~0.05 - 0.2 h on
    # average per the spec's cost tuple), modulated by a small lognormal
    # for realism. NOT the production timing -- replace with the CFD
    # runner's measured wall_time in production.
    base = {-1: 0.1, 0: 1.0, 1: 5.0}.get(tier, 0.1)
    wall = float(base * np.exp(rng.normal(0.0, 0.2)))
    return {"j_fan": j, "wall_time_hours": wall}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n",
        type=int,
        default=SOBOL_SEED_COUNT,
        help="Number of Sobol samples. Default: %(default)s (the spec lock).",
    )
    p.add_argument(
        "--tier",
        type=int,
        default=-1,
        help="Tier label on each record. Default: %(default)s (tier -1 per spec).",
    )
    p.add_argument(
        "--d",
        type=int,
        default=40,
        help="Design-space dimensionality. Default: %(default)s (mid-range of 37-46).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sobol scrambling seed. Default: %(default)s.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Where to write the JSONL ledger. Default: %(default)s.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.n <= 0:
        print(f"[spike_0_7c_seed] ERROR: --n must be > 0, got {args.n}", file=sys.stderr)
        return 2
    if args.d <= 0:
        print(f"[spike_0_7c_seed] ERROR: --d must be > 0, got {args.d}", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    X = _sobol_samples(args.n, args.d, seed=args.seed)

    records: list[dict[str, Any]] = []
    for i, row in enumerate(X):
        ev = _evaluate(row, tier=args.tier, rng=rng)
        records.append(
            {
                "iteration": int(i),
                "tier": int(args.tier),
                "params_hash": _hash_row(row),
                "j_fan": float(ev["j_fan"]),
                "wall_time_hours": float(ev["wall_time_hours"]),
                "source": "sobol_seed",
            }
        )

    # Append-write so the ledger can grow across multiple invocations if
    # the operator chooses (e.g., expanding the seed set after a partial
    # run). For the canonical 50-sample run, the output file should be
    # fresh -- the protocol emphasises a single invocation with --n 50.
    if args.out.exists():
        print(
            f"[spike_0_7c_seed] WARNING: {args.out} already exists; "
            "new records will be APPENDED. Remove the file first for a "
            "fresh ledger.",
            file=sys.stderr,
        )
    record_to_jsonl(records, args.out)
    print(
        f"[spike_0_7c_seed] Wrote {len(records)} Sobol-seed records "
        f"(d={args.d}, tier={args.tier}, seed={args.seed}) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
