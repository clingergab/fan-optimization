#!/usr/bin/env python
"""Spike 0.6d.1 — Tier-1 symmetry + dimensional-force sanity (GATING).

Spec reference: docs/report-final.md §Phase 0 Spike 0.6d (sub-spike 0.6d.1);
protocol in docs/spike_0_6d_protocol.md.

**Pass criterion (V1 — gates Phase 4 launch alongside 0.6c.1):**

* Symmetry: ``|F_cycle_avg| < 0.05 × F_cycle_peak`` for the body-frame-symmetric
  force component (CFx for pitching about y-axis).
* Magnitude: dimensional ``F_cycle_peak`` within ±1 order of magnitude of the
  analytic envelope ``F_envelope = m × ω² × r_cm`` for the geometry the cfg
  ran on. F4 decision (2026-05-14): default geometry is the NACA 0012 mesh
  from Spike 0.6c Cell 6 — the V1 flat-panel baseline mesh isn't generated
  until Phase 4 step 46.

Inputs:

* ``--history-csv`` — SU2 history.csv from a production-Tier-1 run.
* ``--n-cycles`` — total cycles the run executed (5 per Tier-1 lock).
* ``--mass-kg`` — mass of the pitching body (for the envelope).
* ``--omega-rad-per-s`` — pitching angular frequency (12.5664 per Tier-1).
* ``--r-cm-m`` — radius from pivot to center-of-mass (for the envelope).
* ``--envelope-geometry`` — human-readable label written into the result JSON.

Outputs:

* ``data/spike_0_6d/sub_1_result.json`` — full Tier1SymmetryDimensionalResult.
* ``data/spike_0_6d/sub_1.PASS`` or ``data/spike_0_6d/sub_1.FAIL`` — local
  sub-spike marker (the aggregator reads these to write the overall
  ``PASS``/``FAIL`` at ``data/spike_0_6d/PASS``).

Exit codes:

* ``0`` — passed; ``sub_1.PASS`` written.
* ``1`` — failed; ``sub_1.FAIL`` written. (Diagnostic-result JSON is still written.)
* ``2`` — input error (history.csv missing / malformed).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from fanopt.cfd.spike_0_6d import (
    MACH_UNSTEADY_LOCK,
    check_symmetry_dimensional,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "data" / "spike_0_6d"
DEFAULT_RESULT_JSON = DEFAULT_DIR / "sub_1_result.json"
DEFAULT_MARKER_DIR = DEFAULT_DIR


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--history-csv",
        type=Path,
        required=True,
        help="Path to the SU2 history.csv produced by a production-Tier-1 run.",
    )
    p.add_argument(
        "--n-cycles",
        type=int,
        default=5,
        help="Total cycles the SU2 run executed. Default: %(default)d (Tier-1 lock).",
    )
    p.add_argument(
        "--mass-kg",
        type=float,
        required=True,
        help="Mass of the pitching body (kg) for the analytic envelope.",
    )
    p.add_argument(
        "--omega-rad-per-s",
        type=float,
        default=12.5664,
        help="Pitching angular frequency (rad/s). Default: %(default)s (Tier-1 lock).",
    )
    p.add_argument(
        "--r-cm-m",
        type=float,
        required=True,
        help="Radius from pivot to center-of-mass (m) for the analytic envelope.",
    )
    p.add_argument(
        "--envelope-geometry",
        type=str,
        default="NACA 0012 from Spike 0.6c Cell 6",
        help="Human-readable geometry label (written into the result JSON).",
    )
    p.add_argument(
        "--result-json",
        type=Path,
        default=DEFAULT_RESULT_JSON,
        help="Where to write the result JSON. Default: %(default)s.",
    )
    p.add_argument(
        "--marker-dir",
        type=Path,
        default=DEFAULT_MARKER_DIR,
        help="Where to write the sub_1.PASS / sub_1.FAIL marker. Default: %(default)s.",
    )
    return p.parse_args(argv)


def _write_marker(marker_dir: Path, passed: bool) -> Path:
    marker_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("sub_1.PASS", "sub_1.FAIL"):
        stale_path = marker_dir / stale
        if stale_path.exists():
            stale_path.unlink()
    marker = marker_dir / ("sub_1.PASS" if passed else "sub_1.FAIL")
    marker.write_text("")
    return marker


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.history_csv.exists():
        print(f"[spike_0_6d.1] history.csv not found: {args.history_csv}", file=sys.stderr)
        return 2
    try:
        history_text = args.history_csv.read_text(encoding="utf-8")
    except OSError as e:
        print(f"[spike_0_6d.1] cannot read history.csv: {e}", file=sys.stderr)
        return 2
    if not history_text.strip():
        print(f"[spike_0_6d.1] history.csv is empty: {args.history_csv}", file=sys.stderr)
        return 2

    result = check_symmetry_dimensional(
        history_text,
        n_cycles=args.n_cycles,
        mass_kg=args.mass_kg,
        omega_rad_per_s=args.omega_rad_per_s,
        r_cm_m=args.r_cm_m,
        envelope_geometry=args.envelope_geometry,
        history_path=str(args.history_csv),
    )

    payload = {
        "spec_reference": "docs/report-final.md §Phase 0 Spike 0.6d (sub-spike 0.6d.1)",
        "lock_reference": "H10 supplement; mirrors Round-9 HIGH-12 / C12 production numerics",
        "mach_unsteady_lock": MACH_UNSTEADY_LOCK,
        "result": asdict(result),
    }

    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(payload, indent=2) + "\n")
    marker = _write_marker(args.marker_dir, result.passed)

    print(f"[spike_0_6d.1] history       = {args.history_csv}")
    print(f"[spike_0_6d.1] envelope      = {result.envelope_geometry}")
    print(
        f"[spike_0_6d.1] F_cycle_avg   = {result.force_cycle_avg:+.4e}  "
        f"F_cycle_peak = {result.force_cycle_peak:.4e}  "
        f"F_envelope = {result.force_envelope:.4e}"
    )
    print(
        f"[spike_0_6d.1] symmetry      = {'PASS' if result.symmetry_passed else 'FAIL'} "
        f"(|F_avg|/F_peak = {result.symmetry_ratio:.4f}, threshold 0.05)"
    )
    print(
        f"[spike_0_6d.1] magnitude     = {'PASS' if result.magnitude_passed else 'FAIL'} "
        f"(log10(F_peak/F_envelope) = {result.magnitude_ratio_log10:+.3f}, tolerance ±1.0)"
    )
    print(f"[spike_0_6d.1] OVERALL       = {'PASS' if result.passed else 'FAIL'} -> {marker}")
    print(f"[spike_0_6d.1] result JSON   = {args.result_json}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
