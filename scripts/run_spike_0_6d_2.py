#!/usr/bin/env python
"""Spike 0.6d.2 — 2D thin-plate added-mass analytic check (GATING).

Spec reference: docs/report-final.md §Phase 0 Spike 0.6d (sub-spike 0.6d.2);
protocol in docs/spike_0_6d_protocol.md.

**Pass criterion (V1 — gates Phase 4 launch alongside 0.6c.1):**
SU2 cycle-peak inviscid-phase pitching moment about the pivot is within
±15% of the closed-form Sedov/Newman added-mass moment for a 2D plate.

The SU2 inviscid-phase moment is the cycle-peak |moment| over cycles 2-N
(discarding the start-up transient cycle 1). For pure added-mass dominance
(which is what the production Tier-1 numerics SHOULD produce on a 2D
thin-plate body in still air), the peak moment occurs at peak θ̈ — i.e.,
at zero θ where the body's angular acceleration is maximal.

Inputs:

* ``--history-csv`` — SU2 history.csv from the 2D thin-plate pitching run.
* ``--chord-m`` — plate chord length (m).
* ``--pivot-offset-normalized`` — ``a = (x_pivot - c/2) / (c/2)``; -0.5 at
  quarter-chord per protocol default.
* ``--pitching-omega-rad-per-s`` — pitching angular frequency.
* ``--pitching-amplitude-rad`` — pitch amplitude θ_max.
* ``--n-cycles`` — total cycles SU2 executed (5 per Tier-1 lock).

Outputs:

* ``data/spike_0_6d/sub_2_result.json`` — full Tier1AddedMassResult.
* ``data/spike_0_6d/sub_2.PASS`` or ``data/spike_0_6d/sub_2.FAIL`` marker.

Exit codes:

* ``0`` — passed.
* ``1`` — failed.
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
    check_added_mass,
    cycle_averages_and_peaks,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "data" / "spike_0_6d"
DEFAULT_RESULT_JSON = DEFAULT_DIR / "sub_2_result.json"
DEFAULT_MARKER_DIR = DEFAULT_DIR

# Moment column candidates in SU2's CSV. CMz is the standard for 2D
# planar pitching about y-axis (sign convention may differ; we take |peak|).
MOMENT_COLUMN_CANDIDATES = ("cmy", "cmz", "cm", "cmoment")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history-csv", type=Path, required=True)
    p.add_argument("--chord-m", type=float, required=True)
    p.add_argument(
        "--pivot-offset-normalized",
        type=float,
        default=-0.5,
        help="`a = (x_pivot - c/2) / (c/2)`. -0.5 at quarter-chord (default).",
    )
    p.add_argument("--pitching-omega-rad-per-s", type=float, required=True)
    p.add_argument("--pitching-amplitude-rad", type=float, required=True)
    p.add_argument(
        "--n-cycles",
        type=int,
        default=5,
        help="Total cycles SU2 ran. Default: %(default)d.",
    )
    p.add_argument(
        "--fluid-density-kg-per-m3",
        type=float,
        default=1.225,
        help="Freestream density (kg/m³) for the closed-form. Default: sea-level air.",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.15,
        help="Relative-error tolerance vs the closed form. Default: ±15%%.",
    )
    p.add_argument("--result-json", type=Path, default=DEFAULT_RESULT_JSON)
    p.add_argument("--marker-dir", type=Path, default=DEFAULT_MARKER_DIR)
    return p.parse_args(argv)


def _write_marker(marker_dir: Path, passed: bool) -> Path:
    marker_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("sub_2.PASS", "sub_2.FAIL"):
        stale_path = marker_dir / stale
        if stale_path.exists():
            stale_path.unlink()
    marker = marker_dir / ("sub_2.PASS" if passed else "sub_2.FAIL")
    marker.write_text("")
    return marker


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.history_csv.exists():
        print(f"[spike_0_6d.2] history.csv not found: {args.history_csv}", file=sys.stderr)
        return 2
    try:
        history_text = args.history_csv.read_text(encoding="utf-8")
    except OSError as e:
        print(f"[spike_0_6d.2] cannot read history.csv: {e}", file=sys.stderr)
        return 2
    if not history_text.strip():
        print(f"[spike_0_6d.2] history.csv is empty: {args.history_csv}", file=sys.stderr)
        return 2

    _, moment_peak, _ = cycle_averages_and_peaks(
        history_text,
        n_cycles=args.n_cycles,
        force_column_candidates=MOMENT_COLUMN_CANDIDATES,
    )
    if moment_peak == 0:
        print(
            f"[spike_0_6d.2] no moment column ({MOMENT_COLUMN_CANDIDATES}) in history.csv",
            file=sys.stderr,
        )
        return 2

    result = check_added_mass(
        su2_moment_peak=moment_peak,
        chord_m=args.chord_m,
        pivot_offset_normalized=args.pivot_offset_normalized,
        pitching_omega_rad_per_s=args.pitching_omega_rad_per_s,
        pitching_amplitude_rad=args.pitching_amplitude_rad,
        fluid_density_kg_per_m3=args.fluid_density_kg_per_m3,
        tolerance=args.tolerance,
        history_path=str(args.history_csv),
    )

    payload = {
        "spec_reference": "docs/report-final.md §Phase 0 Spike 0.6d (sub-spike 0.6d.2)",
        "lock_reference": "H10 supplement; mirrors Round-9 HIGH-12 / C12 production numerics",
        "mach_unsteady_lock": MACH_UNSTEADY_LOCK,
        "closed_form_reference": "Sedov/Newman: I_a = π ρ b⁴ (1/8 + a²) per unit span",
        "result": asdict(result),
    }

    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(payload, indent=2) + "\n")
    marker = _write_marker(args.marker_dir, result.passed)

    print(f"[spike_0_6d.2] history       = {args.history_csv}")
    print(
        f"[spike_0_6d.2] M_su2_peak    = {result.su2_moment_peak:.6e}   "
        f"M_closed_form = {result.closed_form_moment_peak:.6e}"
    )
    print(
        f"[spike_0_6d.2] rel_error     = {result.relative_error:+.4f}   "
        f"tolerance = ±{result.tolerance:.2f}"
    )
    print(f"[spike_0_6d.2] OVERALL       = {'PASS' if result.passed else 'FAIL'} -> {marker}")
    print(f"[spike_0_6d.2] result JSON   = {args.result_json}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
