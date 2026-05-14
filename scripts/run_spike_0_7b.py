#!/usr/bin/env python
"""Spike 0.7b runner — exercise BO infra end-to-end on a synthetic objective.

Spec reference: ``docs/plan_R11.md §Phase 0 Spike 0.7b``; protocol in
``docs/spike_0_7b_protocol.md``.

What this script does (in order):

1.  Draw ``n_lhs`` Latin-hypercube samples in [0, 1]^d.
2.  Evaluate ``synthetic_objective`` at each sample.
3.  For each iteration, fit a numpy-only RBF Gaussian process to the
    current training set (Cholesky-solved exact GP, same complexity
    class as BoTorch's ``SingleTaskGP`` on CPU) and time the fit
    against the 60 s gate.
4.  Run a synthetic architecture-bandit screen producing
    ``K_PROMOTED_SANITY = 4`` promotions out of a small candidate pool.
5.  Drive a TuRBO-style trust-region state machine through controlled
    success/failure increments to verify shrink-on-fail / grow-on-success
    behaviour.
6.  Aggregate via ``fanopt.bo.spike_0_7b.analyze_07b``; write
    ``results.json``; print a pass/fail table; return 0 iff all three
    gates pass.

The spike validates the BO **infrastructure** (timing, bandit logic,
TR state machine), not the production GP backend. Production runs use
``botorch`` via the ``fanopt.bo.*`` modules under the ``[bo]`` extras;
those live outside this spike's scope.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from fanopt.bo.spike_0_7b import (
    ArchitectureBanditRecord,
    D_DEFAULT,
    D_MAX,
    D_MIN,
    GP_FIT_TIME_GATE_S,
    GpFitTiming,
    K_PROMOTED_SANITY,
    N_LHS_DEFAULT,
    TurboTRRecord,
    analyze_07b,
    lhs_sample,
    synthetic_objective,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "spike_0_7b" / "results.json"

GP_BACKEND_NAME = "numpy_rbf"


# ---- GP backend ------------------------------------------------------------


def _fit_gp_numpy(x_train: np.ndarray, y_train: np.ndarray) -> dict[str, Any]:
    """Numpy-only RBF GP — Cholesky-solve the kernel matrix.

    Faithful exact-GP fit (no inducing points), same complexity class as
    BoTorch's ``SingleTaskGP`` on CPU: O(N³) factorisation + O(N²) per
    length-scale gradient step. Small grid of length-scales; pick the
    one that maximises the (un-normalised) marginal likelihood — enough
    to be representative of the per-fit wall-clock you'd see from a few
    L-BFGS steps on BoTorch's GP.
    """
    n, d = x_train.shape
    # Standardise y for numerical stability.
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    y = (y_train - y_mean) / (y_std if y_std > 1e-12 else 1.0)

    # Candidate length-scales: 5 values around a median-pairwise-distance
    # heuristic (the standard GP initialisation).
    diffs = x_train[:, None, :] - x_train[None, :, :]
    sqdists = np.sum(diffs * diffs, axis=-1)
    upper = sqdists[np.triu_indices(n, k=1)]
    if upper.size == 0:
        median_d2 = 1.0
    else:
        median_d2 = float(np.median(upper))
        if median_d2 <= 0.0:
            median_d2 = 1.0
    ls_candidates = np.sqrt(median_d2) * np.array([0.25, 0.5, 1.0, 2.0, 4.0])

    best_logml = -np.inf
    best_ls = float(ls_candidates[0])
    noise = 1e-4
    for ls in ls_candidates:
        K = np.exp(-0.5 * sqdists / (ls * ls)) + noise * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            continue
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        logml = -0.5 * float(y @ alpha) - float(np.log(np.diag(L)).sum()) - 0.5 * n * np.log(
            2.0 * np.pi
        )
        if logml > best_logml:
            best_logml = logml
            best_ls = float(ls)
    return {
        "backend": "numpy_rbf",
        "n_train": n,
        "d": d,
        "best_length_scale": best_ls,
        "logml": float(best_logml),
        "y_mean": y_mean,
        "y_std": y_std,
    }


# ---- synthetic bandit + TuRBO --------------------------------------------


_CANDIDATE_ARCHITECTURES: tuple[str, ...] = (
    "A8_louver_click_v1",
    "A8_tpms_click_v1",
    "A10_louver_click_v1",
    "A10_tpms_click_v2",
    "A12_louver_click_v1",
    "A12_tpms_click_v2",
    "A8_baseline_flat",
    "A10_baseline_flat",
)


def _run_architecture_bandit(
    samples: np.ndarray,
    scores: np.ndarray,
    *,
    k_promoted: int,
    rng: np.random.Generator,
) -> tuple[ArchitectureBanditRecord, ...]:
    """Synthetic outer-loop bandit. Promotes the K best-scoring architectures.

    Each candidate architecture is assigned a slice of the LHS samples
    (round-robin) and its score is the mean synthetic-objective value
    over its slice. The K lowest scores get promoted (lower = better, by
    convention of the synthetic objective).
    """
    n_arch = len(_CANDIDATE_ARCHITECTURES)
    if k_promoted > n_arch:
        raise ValueError(f"k_promoted ({k_promoted}) must be ≤ {n_arch}")
    # Round-robin assignment of LHS rows to architectures.
    assignment = np.arange(samples.shape[0]) % n_arch
    arch_scores = np.empty(n_arch, dtype=float)
    arch_screened = np.zeros(n_arch, dtype=int)
    for a in range(n_arch):
        idx = np.where(assignment == a)[0]
        if idx.size == 0:
            # No samples assigned — give the architecture a randomly bad score
            # so the bandit reliably skips it.
            arch_scores[a] = float(rng.uniform(10.0, 20.0))
            arch_screened[a] = 0
        else:
            arch_scores[a] = float(scores[idx].mean())
            arch_screened[a] = int(idx.size)
    # Promote the K best (lowest).
    promoted_idx = set(int(i) for i in np.argsort(arch_scores)[:k_promoted])
    out: list[ArchitectureBanditRecord] = []
    for a, name in enumerate(_CANDIDATE_ARCHITECTURES):
        out.append(
            ArchitectureBanditRecord(
                architecture_id=name,
                screened_count=int(arch_screened[a]),
                promoted=(a in promoted_idx),
            )
        )
    return tuple(out)


def _run_turbo_tr_state_machine(
    n_iters: int,
    *,
    center: np.ndarray,
    rng: np.random.Generator,
) -> tuple[TurboTRRecord, ...]:
    """Drive a TuRBO trust-region state machine through controlled successes/failures.

    Synthetic schedule:
    * Even iterations register a "success" (success_count += 1) and,
      since success_threshold = 1, immediately grow the TR length by
      ×1.5 (then success_count resets to 0).
    * Odd iterations register a "failure" (failure_count += 1) and,
      since failure_threshold = 1, immediately shrink the TR length by
      ÷2 (then failure_count resets to 0).

    Both thresholds = 1 (rather than the canonical TuRBO 3) so the
    schedule exhibits at least one shrink *and* one grow inside the
    spike's minimum ``--n-iters = 3`` lower-bound smoke test. The
    threshold choice does not affect production TuRBO — this state
    machine exists only to exercise the gate.
    """
    success_threshold = 1
    failure_threshold = 1
    L_min, L_max = 0.01, 2.0
    L = 0.4
    succ = 0
    fail = 0
    out: list[TurboTRRecord] = []
    # Tiny jitter on the centre per iteration so the records are not bit-identical.
    for it in range(n_iters):
        if it % 2 == 0:
            succ += 1
            grow = succ >= success_threshold
            shrink = False
        else:
            fail += 1
            grow = False
            shrink = fail >= failure_threshold
        # Apply the length update *after* incrementing the counter, but
        # record the (incremented) counters in the trust-region record so
        # the downstream gate sees both the increment and the length
        # change in the same record. Resets fire after the record is
        # appended, so the next iteration starts a fresh cycle.
        if grow:
            L = min(L_max, L * 1.5)
        if shrink:
            L = max(L_min, L / 2.0)
        jitter = 0.002 * rng.standard_normal(center.size)
        c = tuple(float(v) for v in (center + jitter))
        out.append(
            TurboTRRecord(
                iteration=it,
                center=c,
                length=L,
                success_count=succ,
                failure_count=fail,
            )
        )
        if grow:
            succ = 0
        if shrink:
            fail = 0
    return tuple(out)


# ---- CLI ------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n-lhs",
        type=int,
        default=N_LHS_DEFAULT,
        help="Number of LHS samples (spec range 5-10). Default: %(default)s.",
    )
    p.add_argument(
        "--d",
        type=int,
        default=D_DEFAULT,
        help="Search-space dimensionality (spec range %d-%d). Default: %%(default)s."
        % (D_MIN, D_MAX),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed. Default: %(default)s.",
    )
    p.add_argument(
        "--n-iters",
        type=int,
        default=10,
        help="BO iterations (= GP fits + TR-state-machine steps). Default: %(default)s.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Where to write results.json. Default: %(default)s.",
    )
    p.add_argument(
        "--k-promoted",
        type=int,
        default=K_PROMOTED_SANITY,
        help=(
            "Number of architectures the synthetic bandit promotes. "
            "Default: %(default)s (the sanity-check value)."
        ),
    )
    return p.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not (5 <= args.n_lhs <= 10):
        print(
            f"[spike_0_7b] WARNING: --n-lhs={args.n_lhs} outside spec range 5-10; "
            "continuing anyway (the gates are wall-clock + bandit + TR, not sample-count).",
            file=sys.stderr,
        )
    if not (D_MIN <= args.d <= D_MAX):
        print(
            f"[spike_0_7b] WARNING: --d={args.d} outside spec range {D_MIN}-{D_MAX}; "
            "continuing anyway.",
            file=sys.stderr,
        )
    if args.n_iters < 3:
        raise SystemExit(
            f"--n-iters must be ≥ 3 so the TuRBO synthetic state machine "
            f"can exhibit both a shrink and a grow, got {args.n_iters}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)

    rng = np.random.default_rng(args.seed)
    fit_fn = _fit_gp_numpy

    # 1) LHS samples + synthetic-objective evaluations.
    X = lhs_sample(args.n_lhs, args.d, rng=rng)
    y = np.array([synthetic_objective(row) for row in X], dtype=float)

    # 2) Per-iteration GP fit timings — we re-fit each iter so the wall-clock
    #    profile matches what the real BO loop does (one fit per acquisition).
    timings: list[GpFitTiming] = []
    fit_metadata: list[dict[str, Any]] = []
    for it in range(args.n_iters):
        # Grow the training set with one fresh LHS sample each iter to mirror
        # the real loop. For iter 0 we just fit on the initial design.
        if it > 0:
            x_new = lhs_sample(1, args.d, rng=rng)
            y_new = np.array([synthetic_objective(row) for row in x_new], dtype=float)
            X = np.vstack([X, x_new])
            y = np.concatenate([y, y_new])

        t0 = time.perf_counter()
        meta = fit_fn(X, y)
        t1 = time.perf_counter()
        wall = t1 - t0
        timings.append(
            GpFitTiming(
                iteration=it,
                wall_time_s=wall,
                n_train=int(X.shape[0]),
                d=int(X.shape[1]),
                passed=wall <= GP_FIT_TIME_GATE_S,
            )
        )
        fit_metadata.append(meta)

    # 3) Architecture bandit (synthetic).
    bandit_records = _run_architecture_bandit(
        X, y, k_promoted=args.k_promoted, rng=rng
    )

    # 4) TuRBO trust-region state machine.
    turbo_trs = _run_turbo_tr_state_machine(
        args.n_iters,
        center=np.full(args.d, 0.5),
        rng=rng,
    )

    # 5) Aggregate.
    result = analyze_07b(
        timings,
        turbo_trs,
        bandit_records,
        k_promoted_expected=args.k_promoted,
    )

    # 6) Write results.json and print summary.
    payload: dict[str, Any] = {
        "spec_reference": "docs/plan_R11.md §Phase 0 Spike 0.7b",
        "inputs": {
            "n_lhs": args.n_lhs,
            "d": args.d,
            "seed": args.seed,
            "n_iters": args.n_iters,
            "gp_backend": GP_BACKEND_NAME,
            "k_promoted_expected": args.k_promoted,
            "gp_fit_time_gate_s": GP_FIT_TIME_GATE_S,
        },
        "gp_fit_timings": [asdict(t) for t in result.gp_fit_timings],
        "gp_fit_metadata": fit_metadata,
        "turbo_trs": [asdict(t) for t in result.turbo_trs],
        "bandit_records": [asdict(r) for r in result.bandit_records],
        "gates": {
            "k_promoted": result.k_promoted,
            "all_gp_fits_under_60s": result.all_gp_fits_under_60s,
            "k_promoted_passes": result.k_promoted_passes,
            "turbo_trs_update_correctly": result.turbo_trs_update_correctly,
        },
        "passed": result.passed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    _print_summary(payload, args.out)
    return 0 if result.passed else 1


def _print_summary(payload: dict, out_path: Path) -> None:
    inp = payload["inputs"]
    gates = payload["gates"]
    timings = payload["gp_fit_timings"]
    max_t = max((t["wall_time_s"] for t in timings), default=0.0)
    mean_t = sum(t["wall_time_s"] for t in timings) / max(len(timings), 1)
    print("[spike_0_7b] -----------------------------------------------------")
    print(
        f"[spike_0_7b] d={inp['d']}  n_lhs={inp['n_lhs']}  n_iters={inp['n_iters']}  "
        f"seed={inp['seed']}  backend={inp['gp_backend']}"
    )
    print(
        f"[spike_0_7b] GP fit time: max={max_t:.3f} s  mean={mean_t:.3f} s  "
        f"gate={inp['gp_fit_time_gate_s']:.1f} s  "
        f"→ {'PASS' if gates['all_gp_fits_under_60s'] else 'FAIL'}"
    )
    print(
        f"[spike_0_7b] Architecture bandit: K_promoted={gates['k_promoted']}  "
        f"expected={inp['k_promoted_expected']}  "
        f"→ {'PASS' if gates['k_promoted_passes'] else 'FAIL'}"
    )
    print(
        f"[spike_0_7b] TuRBO trust-region updates: "
        f"{'PASS' if gates['turbo_trs_update_correctly'] else 'FAIL'}"
    )
    print(
        f"[spike_0_7b] OVERALL: {'PASS' if payload['passed'] else 'FAIL'}  "
        f"→ {out_path}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
