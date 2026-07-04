#!/usr/bin/env python
"""Run the oscillating-NACA-0012 SU2 solver-validation benchmark (Spike 0.6c.2).

Meshes a NACA 0012 in a circular far-field, runs the pitching unsteady SU2 case
(Re, reduced frequency k, ±amplitude about the quarter chord), and reduces the
lift/drag history to C_L,max, C_d,mean and the C_L-alpha hysteresis-loop area.
Writes ``metrics.json`` and prints a summary.

    python scripts/run_naca_benchmark.py --workdir data/spike_0_6c/naca --re 40000 --k 0.55

NOTE: the PASS/FAIL cross-solver gate (SU2 vs PyFR p=3 vs published dynamic-stall
data) is a separate step — it needs reference numbers this script does not invent.
This produces the SU2-side observables for that comparison.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from fanopt.cfd.naca_benchmark import BenchmarkConfig, BenchmarkMetrics, run_benchmark


def run(
    *,
    workdir: Path,
    reynolds_number: float,
    reduced_frequency_k: float,
    pitch_amplitude_deg: float,
    n_cycles: int,
    steps_per_cycle: int,
    su2_bin: str | None = None,
) -> BenchmarkMetrics:
    """Run one benchmark, write ``metrics.json`` to ``workdir``, return the metrics."""
    cfg = BenchmarkConfig(
        reynolds_number=reynolds_number,
        reduced_frequency_k=reduced_frequency_k,
        pitch_amplitude_deg=pitch_amplitude_deg,
        n_cycles=n_cycles,
        steps_per_cycle=steps_per_cycle,
    )
    metrics = run_benchmark(cfg, workdir, su2_bin=su2_bin)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "metrics.json").write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path("data/spike_0_6c/naca"))
    parser.add_argument("--re", dest="reynolds_number", type=float, default=40000.0)
    parser.add_argument("--k", dest="reduced_frequency_k", type=float, default=0.55)
    parser.add_argument("--amplitude-deg", type=float, default=10.0)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--steps-per-cycle", type=int, default=200)
    parser.add_argument("--su2-bin", default=None, help="path to SU2_CFD (else autodetect)")
    args = parser.parse_args(argv)

    metrics = run(
        workdir=args.workdir,
        reynolds_number=args.reynolds_number,
        reduced_frequency_k=args.reduced_frequency_k,
        pitch_amplitude_deg=args.amplitude_deg,
        n_cycles=args.cycles,
        steps_per_cycle=args.steps_per_cycle,
        su2_bin=args.su2_bin,
    )
    print(
        f"[naca-benchmark] Re={args.reynolds_number:.0f} k={args.reduced_frequency_k} "
        f"±{args.amplitude_deg}deg | {metrics.n_cycles_used} cycles used"
    )
    print(
        f"  C_L,max={metrics.c_l_max:.4f} (at alpha={metrics.alpha_at_cl_max_deg:.2f}deg)  "
        f"C_d,mean={metrics.c_d_mean:.5f}  hysteresis_area={metrics.hysteresis_area:.5f}"
    )
    print("  NOTE: PASS/FAIL vs PyFR + published data is a separate step (reference numbers needed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
