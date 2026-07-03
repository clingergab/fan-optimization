#!/usr/bin/env python
"""Phase 3 — steady↔unsteady 2D-slice correlation gate (report-final.md §Phase 3).

Thin CLI wrapper around ``fanopt.cfd.phase3``: sweeps a spread of designs, runs
the cheap steady slice + the true 2D-unsteady plunging slice for each (SU2 —
local macOS binary via Rosetta, or Colab), correlates the two force series, and
writes a correlation JSON. R² ≥ 0.4 retains the steady tier as a screening
fidelity.

    export SU2_RUN="$HOME/su2-local/extracted/bin"   # or put SU2_CFD on PATH
    python scripts/run_phase3_correlation.py --out-dir data/phase3_sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fanopt.cfd.correlation import CorrelationResult
from fanopt.cfd.phase3 import DesignPoint, DesignResult, run_correlation_sweep


def _summary(corr: CorrelationResult, results: list[DesignResult]) -> dict[str, object]:
    return {
        "metric": "steady_cd_vs_unsteady_rms",
        "r2": round(corr.r2, 4),
        "pearson_r": round(corr.pearson_r, 4),
        "kendall_tau": round(corr.kendall_tau, 4),
        "n_designs": corr.n,
        "threshold": corr.meta["threshold"],
        "passed": corr.passed,
        "designs": [
            {
                "name": r.name,
                "steady_cd": r.steady_cd,
                "unsteady_mean": r.unsteady_mean,
                "unsteady_rms": r.unsteady_rms,
                "meta": r.meta,
            }
            for r in results
        ],
    }


def run(
    *, out_dir: Path, su2_bin: str | None, designs: list[DesignPoint] | None = None
) -> dict[str, object]:
    """Run the sweep and write ``correlation.json`` to ``out_dir``.

    ``designs`` defaults to the full :func:`sweep_designs` set; pass a subset for
    a faster demo (the Phase 3 notebook runs the first few designs).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    corr, results = run_correlation_sweep(out_dir, designs=designs, su2_bin=su2_bin)
    summary = _summary(corr, results)
    (out_dir / "correlation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/phase3_sweep"))
    parser.add_argument(
        "--su2-bin", default=None, help="Path to SU2_CFD (default: $SU2_RUN/SU2_CFD or PATH)"
    )
    args = parser.parse_args(argv)

    summary = run(out_dir=args.out_dir, su2_bin=args.su2_bin)
    print(json.dumps(summary, indent=2))
    verdict = "PASS" if summary["passed"] else "FAIL"
    print(f"[phase3] {verdict} r2={summary['r2']} tau={summary['kendall_tau']} → {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
