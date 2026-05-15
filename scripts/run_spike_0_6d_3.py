#!/usr/bin/env python
"""Spike 0.6d.3 — SU2 incompressible-mode advisory cross-check.

Spec reference: docs/report-final.md §Phase 0 Spike 0.6d (sub-spike 0.6d.3);
protocol in docs/spike_0_6d_protocol.md.

**Advisory — NOT gating.** Compare dimensional cycle-averaged forces from
SU2 compressible-with-MACH=1e-9 vs SU2 native incompressible-mode
(``INC_NAVIER_STOKES``). The aggregator records the result but does NOT
use it to write the Phase-4-gate marker; failure is documented as a
Phase-5 step-62.5 investigation item.

Inputs:

* ``--comp-history-csv`` — SU2 history.csv from the compressible-mode run.
* ``--incomp-history-csv`` — SU2 history.csv from the INC_NAVIER_STOKES run.
* ``--n-cycles`` — total cycles each ran.
* ``--tolerance`` — relative-error tolerance (default ±20%).

Outputs:

* ``data/spike_0_6d/sub_3_result.json`` — Tier1IncompResult.
* No marker file (advisory).

Exit codes:

* ``0`` — agreement within tolerance.
* ``1`` — disagreement outside tolerance (still advisory; aggregator ignores).
* ``2`` — input error.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from fanopt.cfd.spike_0_6d import (
    MACH_UNSTEADY_LOCK,
    check_incompressible_cross,
    cycle_averages_and_peaks,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "data" / "spike_0_6d"
DEFAULT_RESULT_JSON = DEFAULT_DIR / "sub_3_result.json"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comp-history-csv", type=Path, required=True)
    p.add_argument("--incomp-history-csv", type=Path, required=True)
    p.add_argument("--n-cycles", type=int, default=5)
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.20,
        help="Relative-error tolerance (default: ±20%%; advisory only).",
    )
    p.add_argument("--result-json", type=Path, default=DEFAULT_RESULT_JSON)
    return p.parse_args(argv)


def _force_cycle_avg(path: Path, n_cycles: int) -> float | None:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not text.strip():
        return None
    avg, _, _ = cycle_averages_and_peaks(text, n_cycles=n_cycles)
    return avg


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    comp_avg = _force_cycle_avg(args.comp_history_csv, args.n_cycles)
    incomp_avg = _force_cycle_avg(args.incomp_history_csv, args.n_cycles)
    if comp_avg is None:
        print(
            f"[spike_0_6d.3] comp history.csv unreadable: {args.comp_history_csv}",
            file=sys.stderr,
        )
        return 2
    if incomp_avg is None:
        print(
            f"[spike_0_6d.3] incomp history.csv unreadable: {args.incomp_history_csv}",
            file=sys.stderr,
        )
        return 2

    result = check_incompressible_cross(
        compressible_force_cycle_avg=comp_avg,
        incompressible_force_cycle_avg=incomp_avg,
        tolerance=args.tolerance,
    )

    payload = {
        "spec_reference": "docs/report-final.md §Phase 0 Spike 0.6d (sub-spike 0.6d.3)",
        "lock_reference": "H10 supplement (ADVISORY; not gating)",
        "mach_unsteady_lock": MACH_UNSTEADY_LOCK,
        "advisory_note": (
            "Failure does NOT block Phase 4. The independent-codebase "
            "cross-check (OpenFOAM `pimpleFoam`) is the gating answer "
            "in Phase 5 step 62.5."
        ),
        "result": asdict(result),
    }

    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(payload, indent=2) + "\n")

    print(
        f"[spike_0_6d.3] comp_avg     = {result.compressible_force_cycle_avg:+.6e}   "
        f"incomp_avg = {result.incompressible_force_cycle_avg:+.6e}"
    )
    print(
        f"[spike_0_6d.3] rel_error    = {result.relative_error:.4f}   "
        f"tolerance = ±{result.tolerance:.2f}"
    )
    print(
        f"[spike_0_6d.3] ADVISORY     = {'PASS' if result.passed else 'FAIL'} "
        f"(does NOT gate Phase 4)"
    )
    print(f"[spike_0_6d.3] result JSON  = {args.result_json}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
