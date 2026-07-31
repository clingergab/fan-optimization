"""Run one async distributed BO session for the aero-first blade campaign (Stage 3.B).

Thin wrapper: build the 3D objective (:class:`~fanopt.bo.blade_objective_3d.Blade3DObjective`,
cycle-mean CFz, ADR-0004) and hand it to
:func:`~fanopt.bo.distributed_campaign.run_distributed_session`. Launch one of these per Colab
session, all pointed at the SAME ``--shared-dir`` on Drive with distinct ``--session-index``;
they coordinate through the shared ledger without communicating. The objective is injectable so
the wiring is tested without CFD.

    python3 scripts/run_blade_campaign_distributed.py \
        --shared-dir /content/drive/MyDrive/fanopt/blade_campaign \
        --session-id colab-A --session-index 0 --n-sessions 3 \
        --budget 300 --n-init 24 --batch-size 8 --n-workers 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fanopt.bo.blade_campaign import stage3_seed_designs
from fanopt.bo.blade_objective_3d import Blade3DObjective
from fanopt.bo.distributed_campaign import (
    DistributedConfig,
    ObjectiveFn,
    pareto_from_ledger,
    run_async_session,
    run_distributed_session,
    validate_async,
)
from fanopt.cfd.phase3 import find_su2
from fanopt.cfd.phase5 import VerifyConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shared-dir", type=Path, required=True, help="shared Drive ledger dir")
    p.add_argument("--session-id", required=True, help="unique label for this session")
    p.add_argument("--session-index", type=int, default=0, help="0-based; slices the DoE")
    p.add_argument("--n-sessions", type=int, default=1, help="total concurrent sessions")
    p.add_argument("--budget", type=int, default=300, help="stop when shared ledger hits this")
    p.add_argument("--n-init", type=int, default=16, help="shared Sobol DoE size (cold-start)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-workers", type=int, default=1, help="CFD evals in parallel per session")
    p.add_argument("--seed", type=int, default=0, help="shared Sobol seed")
    p.add_argument(
        "--inject-seeds",
        action="store_true",
        help="dispatch the two operator-locked Stage-3 seeds (A = panel+ribs, B = no-rib uniform) "
        "AHEAD of the Sobol DoE (blade_campaign.stage3_seed_designs). Identical across sessions — "
        "the claim/ledger path dedups them, so each is evaluated exactly once.",
    )
    p.add_argument(
        "--explore-fraction",
        type=float,
        default=0.0,
        help="fraction of BO-phase dispatches that are space-filling exploration (e.g. 0.25)",
    )
    p.add_argument("--su2-bin", default=None, help="SU2_CFD path; default = auto-discover")
    p.add_argument("--metric", default="mean", choices=("mean", "peak"), help="ADR-0004: mean")
    p.add_argument("--n-cycles", type=int, default=None, help="VerifyConfig override (fidelity)")
    p.add_argument("--inner-iter", type=int, default=None, help="VerifyConfig override (fidelity)")
    p.add_argument(
        "--cfd-out",
        type=Path,
        default=None,
        help="CFD scratch dir; default shared/cfd. Use LOCAL disk (e.g. /content/cfd) — "
        "SU2 does heavy small-file I/O that is very slow on a Drive FUSE mount.",
    )
    p.add_argument(
        "--drive-diag",
        action="store_true",
        help="persist each design's full CFD output to Drive (out_dir/designs copied per "
        "success). OFF by default: that per-design Drive copytree is O(N) and lags "
        "the sync of the small ledger/claim files coordination depends on.",
    )
    p.add_argument("--poll-seconds", type=float, default=5.0)
    p.add_argument(
        "--claim-ttl",
        type=float,
        default=DistributedConfig().claim_ttl_seconds,
        help="steal a claim older than this (s); MUST exceed worst-case eval wall time",
    )
    p.add_argument(
        "--sync",
        action="store_true",
        help="use the synchronous batch loop instead of the default async-per-worker loop",
    )
    p.add_argument("--max-iters", type=int, default=100_000, help="sync loop only")
    return p


def _verify_config(args: argparse.Namespace) -> VerifyConfig:
    base = VerifyConfig()
    return VerifyConfig(
        n_cycles=args.n_cycles if args.n_cycles is not None else base.n_cycles,
        inner_iter=args.inner_iter if args.inner_iter is not None else base.inner_iter,
    )


def main(argv: list[str] | None = None, objective_fn: ObjectiveFn | None = None) -> int:
    args = build_parser().parse_args(argv)
    shared = args.shared_dir
    shared.mkdir(parents=True, exist_ok=True)

    if objective_fn is None:  # real campaign; tests inject a synthetic objective
        su2_bin = args.su2_bin or find_su2()
        if su2_bin is None:
            raise SystemExit("SU2_CFD not found — pass --su2-bin")
        objective_fn = Blade3DObjective(
            out_dir=args.cfd_out or (shared / "cfd"),
            su2_bin=su2_bin,
            # diag on Drive only if asked; otherwise diag==workdir (local) so no per-design copytree
            diag_dir=shared if args.drive_diag else None,
            metric=args.metric,
            cfg=_verify_config(args),
        )

    cfg = DistributedConfig(
        total_budget=args.budget,
        n_init=args.n_init,
        batch_size=args.batch_size,
        seed=args.seed,
        n_workers=args.n_workers,
        explore_fraction=args.explore_fraction,
        poll_seconds=args.poll_seconds,
        claim_ttl_seconds=args.claim_ttl,
        seed_designs=stage3_seed_designs() if args.inject_seeds else None,
    )
    if args.sync:
        x, _ = run_distributed_session(
            objective_fn,
            shared,
            cfg,
            session_id=args.session_id,
            session_index=args.session_index,
            n_sessions=args.n_sessions,
            max_iters=args.max_iters,
            on_batch=lambda n: print(f"[{args.session_id}] +{n} evaluated", flush=True),
        )
    else:
        # LIVE async check: print a verdict on every completion — so batching is caught within the
        # first wave (~one eval in), NOT after the whole multi-day run. `busy` is how many workers
        # were running when this design was dispatched; == n_workers means the freed slot was
        # refilled immediately (async). Utilization firms up once >n_workers evals have landed.
        def _on_event(e: dict) -> None:
            busy = e["inflight_at_dispatch"] + 1
            v = validate_async(shared, args.n_workers)
            tag = (
                "async"
                if v["is_async"]
                else ("warming up" if v["n_evals"] <= args.n_workers else "BATCHING?")
            )
            print(
                f"[{args.session_id}] done {e['design_hash'][:8]} ({e['source']}) — "
                f"{busy}/{args.n_workers} workers busy at dispatch | "
                f"utilization={v['utilization']:.0%} n={v['n_evals']} -> {tag}",
                flush=True,
            )

        x, _ = run_async_session(
            objective_fn,
            shared,
            cfg,
            session_id=args.session_id,
            session_index=args.session_index,
            n_sessions=args.n_sessions,
            on_event=_on_event,
        )
        v = validate_async(shared, args.n_workers)
        print(
            f"[{args.session_id}] FINAL ASYNC CHECK: worker utilization={v['utilization']:.0%} "
            f"over {v['n_evals']} evals; is_async={v['is_async']}; per-session={v['per_session']}"
        )
    front = pareto_from_ledger(shared)
    print(f"[{args.session_id}] ledger has {len(x)} evals; {len(front)} on the Pareto front")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
