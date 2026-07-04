#!/usr/bin/env python
"""Phase 5 — 3D high-fidelity verification of the top Pareto designs.

Loads a finished Phase-4 campaign, picks the top-k structurally-diverse Pareto
designs, and re-evaluates each with a 3D unsteady SU2 run (single V-unit blade,
external flow) — then checks the cheap 2D-slice ranking survives the 3D physics
(Kendall τ between slice and 3D J_fan). Writes ``verification.json``.

    export SU2_RUN="$HOME/su2-local/extracted/bin"
    python scripts/run_phase5_verify.py --campaign-dir data/phase4_bo --top-k 3

The 3D unsteady runs are expensive (Colab). Geometry + meshing + cfg are local.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from fanopt.bo.results import analyze
from fanopt.cfd.phase5 import VerifyConfig, run_verification, verify_ranking


def _designs_from_campaign(
    campaign_dir: Path, top_k: int
) -> list[tuple[str, np.ndarray, float | None]]:
    summary = analyze(campaign_dir, top_k=top_k)
    designs: list[tuple[str, np.ndarray, float | None]] = []
    for d in summary["top_k_diverse"]:
        name = f"b{d['blade_count']}_i{d['index']}"
        designs.append((name, np.asarray(d["vector"], dtype=float), float(d["j_fan"])))
    return designs


def run(
    *,
    campaign_dir: Path,
    out_dir: Path,
    top_k: int,
    su2_bin: str | None = None,
    cfg: VerifyConfig | None = None,
    n_workers: int = 1,
    progress: bool = True,
) -> dict[str, object]:
    """Verify the top-k designs and write ``verification.json``; return the summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    designs = _designs_from_campaign(campaign_dir, top_k)
    results = run_verification(
        designs,
        out_dir,
        cfg=cfg or VerifyConfig(),
        su2_bin=su2_bin,
        n_workers=n_workers,
        progress=progress,
    )
    ranking = verify_ranking(results)
    summary: dict[str, object] = {
        "ranking": ranking,
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
    (out_dir / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=Path("data/phase4_bo"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/phase5_verify"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--workers", type=int, default=1, help="Parallel designs (processes); ≈ min(top_k, cores)."
    )
    parser.add_argument("--su2-bin", default=None, help="Path to SU2_CFD (default: $SU2_RUN/PATH)")
    parser.add_argument("--no-progress", action="store_true", help="Disable the tqdm progress bar.")
    args = parser.parse_args(argv)

    summary = run(
        campaign_dir=args.campaign_dir,
        out_dir=args.out_dir,
        top_k=args.top_k,
        su2_bin=args.su2_bin,
        n_workers=args.workers,
        progress=not args.no_progress,
    )
    ranking: dict[str, Any] = summary["ranking"]  # type: ignore[assignment]
    print(json.dumps(ranking, indent=2))
    valid = ranking["valid_only"]
    suspects = ranking["suspect_designs"]
    print(
        f"[phase5] verified {len(summary['designs'])} designs → "  # type: ignore[arg-type]
        f"rank_preserved={ranking['rank_preserved']} "
        f"(valid n={valid['n']}: τ={valid['kendall_tau']}, ρ={valid['spearman_rho']}, "
        f"R²={valid['pearson_r2']})"
    )
    if suspects:
        print(f"[phase5] ⚠ {ranking['n_suspect']} suspect (negative/failed 3D J_fan): {suspects}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
