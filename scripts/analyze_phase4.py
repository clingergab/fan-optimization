#!/usr/bin/env python
"""Analyze a Phase 4 BO campaign — Pareto front + top-k diverse print picks.

Reads a campaign directory (``checkpoint.npz`` + ``evaluations.jsonl`` written by
``run_phase4_bo.py``), reconstructs the 3-objective Pareto front, selects the
structurally-diverse designs to print in Phase 5, and writes ``analysis.json``
(+ an optional Pareto scatter PNG).

    python scripts/analyze_phase4.py --out-dir data/phase4_bo --top-k 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from fanopt.bo.results import analyze


def run(*, out_dir: Path, top_k: int, plot: bool) -> dict[str, Any]:
    summary = analyze(out_dir, top_k=top_k)
    (out_dir / "analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if plot:
        picks = {d["index"] for d in summary["top_k_diverse"]}
        _write_plot(out_dir, summary["pareto"], picks)
    return summary


def _write_plot(out_dir: Path, pareto: list[dict[str, Any]], picks: set[Any]) -> None:
    """Headless (Agg) Pareto scatter — J_fan vs I_wrist, coloured by deflection."""
    j = [d["j_fan"] for d in pareto]
    iw = [d["i_wrist_kgm2"] for d in pareto]
    st = [d["structural_m"] * 1000.0 for d in pareto]
    fig = Figure(figsize=(5.6, 4.2))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    sc = ax.scatter(j, iw, c=st, s=70, cmap="viridis")
    for d in pareto:
        if d["index"] in picks:
            ax.scatter(
                [d["j_fan"]], [d["i_wrist_kgm2"]], s=180, facecolors="none", edgecolors="red"
            )
    ax.set_xlabel("J_fan  (maximize →)")
    ax.set_ylabel("I_wrist  kg·m²  (← minimize)")
    fig.colorbar(sc, ax=ax, label="panel deflection (mm, minimize)")
    ax.set_title(f"Phase 4 Pareto — {len(pareto)} designs (red = print picks)")
    fig.tight_layout()
    fig.savefig(out_dir / "pareto.png", dpi=120)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/phase4_bo"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    summary = run(out_dir=args.out_dir, top_k=args.top_k, plot=not args.no_plot)
    picks = summary["top_k_diverse"]
    print(
        f"[phase4-analyze] {summary['n_pareto']} Pareto designs from "
        f"{summary['n_evaluations']} evals → {len(picks)} diverse print picks"  # type: ignore[arg-type]
    )
    for d in picks:  # type: ignore[union-attr]
        print(
            f"  b{d['blade_count']} {d['edge_profile']}: J_fan={d['j_fan']:.3e} "
            f"I_wrist={d['i_wrist_kgm2']:.3e} defl={d['structural_m'] * 1000:.3f}mm"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
