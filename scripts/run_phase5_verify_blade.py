#!/usr/bin/env python
"""Phase 5 — 3D verification of the aero-first campaign's top Pareto blades.

Loads a finished aero campaign's ``pareto.json`` (from ``scripts/run_phase4_aero.py``),
picks the top-k designs by slice ``J_fan``, and re-evaluates each with a 3D unsteady SU2
run (single redesigned blade, external flow) — then checks the cheap 2D-slice ranking
survives the 3D physics (Kendall τ between slice and 3D ``J_fan``). Writes
``verification.json``, rewritten after each design so a mid-run disconnect keeps progress.

    export SU2_RUN="$HOME/su2-local/extracted/bin"
    python scripts/run_phase5_verify_blade.py --pareto data/phase4_aero/pareto.json --top-k 3

The 3D unsteady runs are expensive; geometry + meshing + cfg are local.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fanopt.cfd.blade_verify import load_pareto, verify_blades
from fanopt.cfd.phase5 import VerifyConfig, VerifyResult, verify_ranking


def _summary(results: list[VerifyResult]) -> dict[str, Any]:
    return {
        "ranking": verify_ranking(results),
        "designs": [
            {
                "name": r.name,
                "j_fan_3d": r.j_fan_3d,
                "j_fan_slice": r.j_fan_slice,
                "n_nodes": r.meta.get("n_nodes"),
            }
            for r in results
        ],
    }


def run(
    *,
    pareto_path: Path,
    out_dir: Path,
    top_k: int | None,
    su2_bin: str | None = None,
    cfg: VerifyConfig | None = None,
    n_workers: int = 1,
    progress: bool = True,
) -> dict[str, object]:
    """Verify the top-k blades and write ``verification.json``; return the summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pareto = load_pareto(pareto_path)

    done: list[VerifyResult] = []

    def _checkpoint(r: VerifyResult) -> None:
        done.append(r)
        (out_dir / "verification.json").write_text(
            json.dumps(_summary(done), indent=2), encoding="utf-8"
        )

    results, _ = verify_blades(
        pareto,
        out_dir,
        top_k=top_k,
        cfg=cfg or VerifyConfig(),
        su2_bin=su2_bin,
        n_workers=n_workers,
        progress=progress,
        on_result=_checkpoint,
    )
    summary = _summary(results)  # final write with order-preserved results
    (out_dir / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pareto", type=Path, default=Path("data/phase4_aero/pareto.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/phase5_verify_blade"))
    parser.add_argument(
        "--top-k", type=int, default=3, help="Top designs by slice J_fan (0 = all)."
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Parallel designs (processes); ~ min(top_k, cores)."
    )
    parser.add_argument("--su2-bin", default=None, help="Path to SU2_CFD (default: $SU2_RUN/PATH)")
    parser.add_argument("--no-progress", action="store_true", help="Disable the tqdm progress bar.")
    args = parser.parse_args(argv)

    summary = run(
        pareto_path=args.pareto,
        out_dir=args.out_dir,
        top_k=args.top_k or None,
        su2_bin=args.su2_bin,
        n_workers=args.workers,
        progress=not args.no_progress,
    )
    ranking: dict[str, Any] = summary["ranking"]  # type: ignore[assignment]
    print(json.dumps(ranking, indent=2))
    valid = ranking["valid_only"]
    suspects = ranking["suspect_designs"]
    print(
        f"[phase5-blade] verified {len(summary['designs'])} designs → "  # type: ignore[arg-type]
        f"rank_preserved={ranking['rank_preserved']} "
        f"(valid n={valid['n']}: τ={valid['kendall_tau']}, ρ={valid['spearman_rho']}, "
        f"R²={valid['pearson_r2']})"
    )
    if suspects:
        print(
            f"[phase5-blade] {ranking['n_suspect']} suspect "
            f"(negative/failed 3D J_fan): {suspects}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
