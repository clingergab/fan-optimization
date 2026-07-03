#!/usr/bin/env python
"""Phase 4 — multi-objective BO panel shape optimization (V1-slim Stage 2).

Thin CLI over ``fanopt.bo.orchestration.run_campaign``: wires the real CFD
objective (decode → Path A+ slice → unsteady SU2 → J_fan, plus CadQuery I_wrist
and plate-bending panel stiffness) and drives the qLogNEHVI + TuRBO campaign,
persisting a JSONL ledger + checkpoints and writing the final Pareto set.

    export SU2_RUN="$HOME/su2-local/extracted/bin"
    python scripts/run_phase4_bo.py --out-dir data/phase4_bo --n-init 8 --n-iterations 20

The CFD objective is expensive (~minutes/eval); a full campaign runs on Colab. Use
small --n-init/--n-iterations for a local smoke.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fanopt.bo.cfd_objective import CfdObjective
from fanopt.bo.objective import PRODUCTION_EVAL_CFG, SliceEvalConfig
from fanopt.bo.orchestration import CampaignConfig, ObjectiveFn, pareto_designs, run_campaign


def make_cfd_objective(
    out_dir: Path, *, su2_bin: str | None = None, eval_cfg: SliceEvalConfig | None = None
) -> ObjectiveFn:
    """Build the real 3-objective CFD evaluation callable for the campaign.

    Returns a picklable :class:`CfdObjective` (not a closure) so the campaign's
    process pool can ship it to worker processes. Each design gets a stable
    per-hash workdir under ``out_dir/designs`` so a resumed campaign reuses prior
    CFD output rather than recomputing it.
    """
    return CfdObjective(out_dir=out_dir, su2_bin=su2_bin, eval_cfg=eval_cfg or SliceEvalConfig())


def run(
    *,
    out_dir: Path,
    cfg: CampaignConfig,
    su2_bin: str | None = None,
    eval_cfg: SliceEvalConfig | None = None,
    progress: bool = True,
) -> dict[str, object]:
    """Run the campaign and write ``pareto.json``; return the summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    objective_fn = make_cfd_objective(out_dir, su2_bin=su2_bin, eval_cfg=eval_cfg)
    state = run_campaign(objective_fn, out_dir, cfg, progress=progress)
    pareto = pareto_designs(state)
    summary: dict[str, object] = {
        "n_evaluations": int(state.x.shape[0]),
        "n_iterations": int(state.iteration),
        "used_fallback": int(state.used_fallback),
        "n_pareto": len(pareto),
        "pareto": pareto,
    }
    (out_dir / "pareto.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/phase4_bo"))
    parser.add_argument("--n-init", type=int, default=8)
    parser.add_argument("--n-iterations", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stall-patience", type=int, default=5)
    parser.add_argument(
        "--workers", type=int, default=1, help="Parallel CFD processes (DoE + batch); ≈ n_cores."
    )
    parser.add_argument("--no-trust-region", action="store_true")
    parser.add_argument(
        "--production",
        action="store_true",
        help="Use the locked 5-cycle / dt=T/200 unsteady resolution (else the fast demo).",
    )
    parser.add_argument("--su2-bin", default=None, help="Path to SU2_CFD (default: $SU2_RUN/PATH)")
    parser.add_argument("--no-progress", action="store_true", help="Disable the tqdm progress bar.")
    args = parser.parse_args(argv)

    cfg = CampaignConfig(
        n_init=args.n_init,
        n_iterations=args.n_iterations,
        batch_size=args.batch_size,
        seed=args.seed,
        stall_patience=args.stall_patience,
        use_trust_region=not args.no_trust_region,
        n_workers=args.workers,
    )
    eval_cfg = PRODUCTION_EVAL_CFG if args.production else None
    summary = run(
        out_dir=args.out_dir,
        cfg=cfg,
        su2_bin=args.su2_bin,
        eval_cfg=eval_cfg,
        progress=not args.no_progress,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "pareto"}, indent=2))
    print(
        f"[phase4] {summary['n_pareto']} Pareto designs from "
        f"{summary['n_evaluations']} evals → {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
