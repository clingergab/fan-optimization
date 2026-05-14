#!/usr/bin/env python
"""Spike 0.7c smoke runner -- synthetic Sobol + BO end-to-end check.

Spec reference: ``docs/plan_R11.md §Phase 0 Spike 0.7c``; protocol in
``docs/spike_0_7c_protocol.md``.

The full Spike-0.7c experiment is 430 cumulative-compute hours of CFD
(50 Sobol seed runs at tier -1 + 100 BO iterations at the production
GP+qMFKG config, summed across three serial 30 / 100 / 300 h budgets).
That is not runnable in this sandbox.

This script verifies the *harness* end-to-end on a synthetic objective
in seconds:

1. Generates 50 synthetic Sobol records on the same synthetic objective
   the BO infrastructure uses (``fanopt.bo.spike_0_7b.synthetic_objective``).
   Each record gets a small simulated ``wall_time_hours``.
2. Generates 100 synthetic BO records that improve monotonically -- the
   "BO" stream is just a noisy-monotone descent toward the synthetic-
   objective minimum, scaled so it cleanly beats the Sobol baseline by
   more than the 5% gate on at least 2 of 3 budgets.
3. Writes both ledgers to temp JSONL files.
4. Invokes ``run_spike_0_7c.main`` to run the comparison.
5. Asserts the overall result is PASS; returns 0 on PASS, 1 otherwise.

This lets us smoke-test the comparison logic without spending a single
CFD hour.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats.qmc import Sobol

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fanopt.bo.spike_0_7b import lhs_sample, synthetic_objective  # noqa: E402
from fanopt.bo.spike_0_7c import (  # noqa: E402
    BO_ITERATIONS_DEFAULT,
    BUDGETS_HOURS,
    SOBOL_SEED_COUNT,
    record_to_jsonl,
)

import run_spike_0_7c as cli  # noqa: E402


def _synth_sobol_records(
    n: int,
    d: int,
    *,
    rng: np.random.Generator,
    wall_time_per_record: float,
) -> list[dict[str, Any]]:
    """Generate n synthetic Sobol-seed records.

    Uses ``scipy.stats.qmc.Sobol`` (a hard runtime dep) for low-discrepancy
    coverage. Each record is timestamped with a fixed per-record wall_time
    so cumulative-compute arithmetic is exact.
    """
    seed = int(rng.integers(0, 2**31 - 1))
    sampler = Sobol(d=d, scramble=True, seed=seed)
    X = np.asarray(sampler.random(n=n), dtype=float)

    records: list[dict[str, Any]] = []
    for i, row in enumerate(X):
        # Tier-(-1) Sobol seed runs; deterministic synthetic objective.
        j = float(synthetic_objective(row))
        records.append(
            {
                "iteration": i,
                "tier": -1,
                "params_hash": _hash_row(row),
                "j_fan": j,
                "wall_time_hours": float(wall_time_per_record),
                "source": "sobol_seed",
            }
        )
    return records


def _synth_bo_records(
    n: int,
    d: int,
    *,
    sobol_best: float,
    rng: np.random.Generator,
    wall_time_per_record: float,
    target_improvement_pct: float = 25.0,
) -> list[dict[str, Any]]:
    """Generate n synthetic BO-iteration records.

    The synthetic BO stream simulates a competent BO loop on the same
    synthetic objective: starting near (but slightly better than)
    Sobol's best, descending monotonically (with small noise) toward a
    final J_fan that is ``target_improvement_pct`` below the Sobol best.

    This is a *behavioural stand-in*, not a real BO run. Its purpose is
    to exercise the comparison harness end-to-end so we can confirm the
    truncation / best-so-far / gate logic is wired up correctly.
    """
    # Start at sobol_best (the BO loop initialises from the Sobol-seed
    # set) and descend exponentially toward target. Add bounded noise so
    # the sequence is not bit-perfectly monotone.
    start = sobol_best
    target = sobol_best * (1.0 - target_improvement_pct / 100.0)
    # Floor at a small positive number so the synthetic objective stays
    # in a numerically sensible range (synthetic_objective is >= 0 by
    # construction; the BO simulator should not produce negative values).
    target = max(target, 1e-6)
    records: list[dict[str, Any]] = []
    cur = start
    for i in range(n):
        # Exponential descent toward target with a small fraction per step.
        cur = cur + (target - cur) * 0.05
        noise = float(rng.normal(0.0, 0.001 * abs(start)))
        observed = max(cur + noise, 1e-9)
        # Synthesise a fake parameters vector so the record schema is
        # representative (and the params_hash is non-trivial).
        x_dummy = rng.random(d)
        records.append(
            {
                "iteration": i,
                "tier": 0 if i % 5 != 0 else 1,
                "params_hash": _hash_row(x_dummy),
                "j_fan": float(observed),
                "wall_time_hours": float(wall_time_per_record),
                "source": "bo_inner_loop",
            }
        )
    return records


def _hash_row(row: np.ndarray) -> str:
    """Cheap deterministic hash of a parameter vector."""
    h = hashlib.sha1(np.asarray(row, dtype=float).tobytes()).hexdigest()
    return h[:16]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n-sobol",
        type=int,
        default=SOBOL_SEED_COUNT,
        help="Number of synthetic Sobol records. Default: %(default)s.",
    )
    p.add_argument(
        "--n-bo",
        type=int,
        default=BO_ITERATIONS_DEFAULT,
        help="Number of synthetic BO records. Default: %(default)s.",
    )
    p.add_argument(
        "--d",
        type=int,
        default=12,
        help=(
            "Dimensionality of the synthetic objective. Small here so "
            "the smoke test is fast (production is 37-46 D). Default: %(default)s."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the synthetic data. Default: %(default)s.",
    )
    p.add_argument(
        "--budgets",
        type=str,
        default=",".join(str(b) for b in BUDGETS_HOURS),
        help=(
            "Budgets to compare on. Default: %(default)s (the three spec "
            "checkpoints; the synthetic per-record wall_time is scaled so "
            "all checkpoints are reachable inside the synthetic ledger)."
        ),
    )
    p.add_argument(
        "--target-bo-improvement-pct",
        type=float,
        default=25.0,
        help=(
            "Target final BO improvement over Sobol best, in percent. "
            "Default: %(default)s -- well above the 5%% gate so the "
            "smoke test passes deterministically."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write the synthetic ledgers + results.json. Default: tempdir.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    rng = np.random.default_rng(args.seed)

    # Budget arithmetic: we need the sum of per-record wall_times to span
    # the largest budget (300 by default). Place per-record wall_time so
    # the smallest budget already contains a useful number of records
    # from each stream.
    budgets = tuple(int(b.strip()) for b in args.budgets.split(",") if b.strip())
    largest = max(budgets)
    # Place per-record wall_time so the smallest budget still contains
    # >= 5 records from each stream. The smallest budget / (5 * max(n))
    # ensures that. We then cap so all records fit inside the largest
    # budget too.
    n_max = max(args.n_sobol, args.n_bo)
    smallest = min(budgets)
    wall_time_low = smallest / max(5 * n_max, 1)
    wall_time_high = largest / max(n_max, 1)
    per_record_wall_time = min(wall_time_high, max(wall_time_low, 1e-6))

    sobol = _synth_sobol_records(
        args.n_sobol,
        args.d,
        rng=rng,
        wall_time_per_record=per_record_wall_time,
    )
    sobol_best = min(r["j_fan"] for r in sobol)
    bo = _synth_bo_records(
        args.n_bo,
        args.d,
        sobol_best=sobol_best,
        rng=rng,
        wall_time_per_record=per_record_wall_time,
        target_improvement_pct=args.target_bo_improvement_pct,
    )

    out_dir = args.out_dir
    cleanup_tmp = False
    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="spike_0_7c_smoke_"))
        cleanup_tmp = False  # leave for inspection if the user wants it
    out_dir.mkdir(parents=True, exist_ok=True)

    sobol_path = out_dir / "sobol_results.jsonl"
    bo_path = out_dir / "bo_results.jsonl"
    # `record_to_jsonl` appends, so make sure we start from a clean file.
    if sobol_path.exists():
        sobol_path.unlink()
    if bo_path.exists():
        bo_path.unlink()
    record_to_jsonl(sobol, sobol_path)
    record_to_jsonl(bo, bo_path)

    results_path = out_dir / "results.json"
    rc = cli.main(
        [
            "--sobol-results", str(sobol_path),
            "--bo-results", str(bo_path),
            "--budgets", args.budgets,
            "--out", str(results_path),
        ]
    )

    if rc != 0:
        print(
            f"[spike_0_7c_smoke] FAIL: comparison harness returned rc={rc}",
            file=sys.stderr,
        )
        return 1

    payload = json.loads(results_path.read_text())
    if not payload.get("passed"):
        print(
            "[spike_0_7c_smoke] FAIL: harness reported overall failure on "
            "synthetic data -- the BO synthetic stream should beat Sobol on all "
            "three budgets by construction.",
            file=sys.stderr,
        )
        return 1

    print(
        f"[spike_0_7c_smoke] PASS -- synthetic harness end-to-end OK "
        f"(ledgers + results in {out_dir})"
    )
    # Sanity-check the per-budget table is non-empty.
    if len(payload["per_budget"]) != len(budgets):
        print(
            f"[spike_0_7c_smoke] WARNING: expected {len(budgets)} budgets, "
            f"got {len(payload['per_budget'])}",
            file=sys.stderr,
        )
    if cleanup_tmp:  # pragma: no cover -- left as False; explicit toggle.
        shutil.rmtree(out_dir, ignore_errors=True)
    # Defensive: ensure all math values are finite.
    for p in payload["per_budget"]:
        if not math.isfinite(p["bo_minus_sobol_pct"]):
            print(
                f"[spike_0_7c_smoke] WARNING: non-finite delta at "
                f"budget={p['budget_hours']}: {p['bo_minus_sobol_pct']!r}",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
