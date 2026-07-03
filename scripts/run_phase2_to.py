#!/usr/bin/env python
"""Phase 2 — rib SIMP topology optimization (report-final.md §Phase 2 / §3.1).

Thin CLI wrapper around ``fanopt.topopt``: builds the locked rib problem, runs
the plate-bending SIMP loop locally (numpy/scipy — no FEniCSx/Colab), and writes
the optimized density field + a result JSON.

    python scripts/run_phase2_to.py --out-dir data/phase2_rib_to
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fanopt.topopt.loads import DEFAULT_AERO_PRESSURE_PA, DEFAULT_ELEM_SIZE_M, build_rib_problem
from fanopt.topopt.solver import RibTOResult, run_rib_topology_optimization


def _result_summary(result: RibTOResult) -> dict[str, object]:
    return {
        "converged": result.converged,
        "iterations": result.iterations,
        "volume_fraction": round(result.volume_fraction, 4),
        "u_tip_max_m": result.u_tip_max_m,
        "u_tip_max_mm": round(result.u_tip_max_m * 1000.0, 4),
        "u_tip_under_1mm": bool(result.u_tip_max_m < 0.001),
        "compliance_initial": result.compliance_history[0],
        "compliance_final": result.compliance_history[-1],
        "compliance_reduction_pct": round(
            100.0 * (1.0 - result.compliance_history[-1] / result.compliance_history[0]), 2
        ),
        "meta": result.meta,
    }


def run(
    *,
    elem_size_m: float,
    pressure_pa: float,
    volfrac: float,
    max_iters: int,
    out_dir: Path,
) -> dict[str, object]:
    """Build + run the rib TO and write artifacts to ``out_dir``. Returns the summary."""
    problem = build_rib_problem(elem_size_m=elem_size_m, pressure_pa=pressure_pa, volfrac=volfrac)
    result = run_rib_topology_optimization(problem, max_iters=max_iters)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "rib_density.npy", result.density)
    np.save(out_dir / "rib_active_mask.npy", problem.active)
    summary = _result_summary(result)
    (out_dir / "phase2_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elem-size-m", type=float, default=DEFAULT_ELEM_SIZE_M)
    parser.add_argument("--pressure-pa", type=float, default=DEFAULT_AERO_PRESSURE_PA)
    parser.add_argument("--volfrac", type=float, default=0.40)
    parser.add_argument("--max-iters", type=int, default=80)
    parser.add_argument("--out-dir", type=Path, default=Path("data/phase2_rib_to"))
    args = parser.parse_args(argv)

    summary = run(
        elem_size_m=args.elem_size_m,
        pressure_pa=args.pressure_pa,
        volfrac=args.volfrac,
        max_iters=args.max_iters,
        out_dir=args.out_dir,
    )
    print(json.dumps(summary, indent=2))
    print(f"[phase2] wrote rib_density.npy + phase2_result.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
