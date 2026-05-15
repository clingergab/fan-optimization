#!/usr/bin/env python
"""Spike 0.6d — aggregator + Phase 4 launch gate marker (H10 supplement).

Spec reference: docs/report-final.md §Phase 0 Spike 0.6d (2026-05-14 addition);
protocol in docs/spike_0_6d_protocol.md.

**Gate semantics:** ``overall_passed = sub_06d_1.passed AND sub_06d_2.passed``.
Sub-spike 0.6d.3 is ADVISORY — its result is recorded in the result JSON but
does NOT affect the marker decision.

Phase 4 launch (``scripts/launch_phase4.py``) refuses to create the
``phase4-launch`` git tag unless BOTH ``data/spike_0_6c/PASS`` AND
``data/spike_0_6d/PASS`` are present.

Inputs:

* ``--sub-1-json`` — sub-spike 0.6d.1 result JSON.
* ``--sub-2-json`` — sub-spike 0.6d.2 result JSON.
* ``--sub-3-json`` — (optional) sub-spike 0.6d.3 advisory result JSON.

Outputs:

* ``data/spike_0_6d/results.json`` — aggregate result.
* ``data/spike_0_6d/PASS`` or ``data/spike_0_6d/FAIL`` — Phase 4 gate marker.

Exit codes:

* ``0`` — sub_1 + sub_2 both passed; ``PASS`` marker written.
* ``1`` — sub_1 or sub_2 failed; ``FAIL`` marker written.
* ``2`` — input error (missing / malformed gating result JSON).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

from fanopt.cfd.spike_0_6d import (
    Tier1AddedMassResult,
    Tier1IncompResult,
    Tier1SymmetryDimensionalResult,
    analyze_spike_06d,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "data" / "spike_0_6d"
DEFAULT_SUB_1_JSON = DEFAULT_DIR / "sub_1_result.json"
DEFAULT_SUB_2_JSON = DEFAULT_DIR / "sub_2_result.json"
DEFAULT_SUB_3_JSON = DEFAULT_DIR / "sub_3_result.json"
DEFAULT_OUT_JSON = DEFAULT_DIR / "results.json"


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"result JSON not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: invalid JSON: {e}") from e


def _get_result_block(payload: dict, path: Path) -> dict:
    r = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(r, dict):
        raise ValueError(f"{path}: missing 'result' block")
    return r


def _float_or_nan(v) -> float:
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return math.nan


def _load_sub_1(path: Path) -> Tier1SymmetryDimensionalResult:
    r = _get_result_block(_load_json(path), path)
    return Tier1SymmetryDimensionalResult(
        history_path=str(r.get("history_path", "")),
        n_cycles=int(r.get("n_cycles", 0)),
        force_cycle_avg=_float_or_nan(r.get("force_cycle_avg")),
        force_cycle_peak=_float_or_nan(r.get("force_cycle_peak")),
        force_envelope=_float_or_nan(r.get("force_envelope")),
        envelope_geometry=str(r.get("envelope_geometry", "")),
        symmetry_ratio=_float_or_nan(r.get("symmetry_ratio")),
        symmetry_passed=bool(r.get("symmetry_passed", False)),
        magnitude_ratio_log10=_float_or_nan(r.get("magnitude_ratio_log10")),
        magnitude_passed=bool(r.get("magnitude_passed", False)),
        passed=bool(r.get("passed", False)),
    )


def _load_sub_2(path: Path) -> Tier1AddedMassResult:
    r = _get_result_block(_load_json(path), path)
    return Tier1AddedMassResult(
        history_path=str(r.get("history_path", "")),
        chord_m=_float_or_nan(r.get("chord_m")),
        pivot_offset_normalized=_float_or_nan(r.get("pivot_offset_normalized")),
        pitching_omega_rad_per_s=_float_or_nan(r.get("pitching_omega_rad_per_s")),
        pitching_amplitude_rad=_float_or_nan(r.get("pitching_amplitude_rad")),
        su2_moment_peak=_float_or_nan(r.get("su2_moment_peak")),
        closed_form_moment_peak=_float_or_nan(r.get("closed_form_moment_peak")),
        relative_error=_float_or_nan(r.get("relative_error")),
        tolerance=_float_or_nan(r.get("tolerance")),
        passed=bool(r.get("passed", False)),
    )


def _load_sub_3(path: Path) -> Tier1IncompResult:
    r = _get_result_block(_load_json(path), path)
    return Tier1IncompResult(
        compressible_force_cycle_avg=_float_or_nan(r.get("compressible_force_cycle_avg")),
        incompressible_force_cycle_avg=_float_or_nan(r.get("incompressible_force_cycle_avg")),
        relative_error=_float_or_nan(r.get("relative_error")),
        tolerance=_float_or_nan(r.get("tolerance")),
        passed=bool(r.get("passed", False)),
    )


def _write_marker(out_dir: Path, passed: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("PASS", "FAIL"):
        stale_path = out_dir / stale
        if stale_path.exists():
            stale_path.unlink()
    marker = out_dir / ("PASS" if passed else "FAIL")
    marker.write_text("")
    return marker


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sub-1-json", type=Path, default=DEFAULT_SUB_1_JSON)
    p.add_argument("--sub-2-json", type=Path, default=DEFAULT_SUB_2_JSON)
    p.add_argument(
        "--sub-3-json",
        type=Path,
        default=None,
        help=(
            "Optional path to sub-spike 0.6d.3 advisory result JSON. If absent or "
            "missing, sub_3 is logged as 'not run' and does not affect the gate."
        ),
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--marker-dir", type=Path, default=DEFAULT_DIR)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        sub_1 = _load_sub_1(args.sub_1_json)
        sub_2 = _load_sub_2(args.sub_2_json)
    except (FileNotFoundError, ValueError) as e:
        print(f"[spike_0_6d] input error: {e}", file=sys.stderr)
        return 2

    sub_3 = None
    sub_3_note = "not run"
    if args.sub_3_json is not None and args.sub_3_json.exists():
        try:
            sub_3 = _load_sub_3(args.sub_3_json)
            sub_3_note = "PASS" if sub_3.passed else "FAIL (advisory; does not gate)"
        except ValueError as e:
            print(f"[spike_0_6d] sub_3 advisory result unreadable; ignoring: {e}", file=sys.stderr)

    result = analyze_spike_06d(sub_1, sub_2, sub_3)

    payload = {
        "spec_reference": "docs/report-final.md §Phase 0 Spike 0.6d",
        "lock_reference": "H10 supplement (2026-05-14 addition); mirrors Round-9 HIGH-12 / C12",
        "gate_note": (
            "overall_passed = sub_1.passed AND sub_2.passed. sub_3 is ADVISORY "
            "and does NOT affect the gate. Phase 4 launch requires BOTH "
            "data/spike_0_6c/PASS AND data/spike_0_6d/PASS."
        ),
        "phase5_step_62_5_pointer": (
            "Absolute-accuracy validation (regime-appropriate published "
            "reference + OpenFOAM cross-codebase) is the Phase 5 step 62.5 "
            "deliverable; see docs/report-final.md."
        ),
        "result": {
            "sub_06d_1": asdict(result.sub_06d_1),
            "sub_06d_2": asdict(result.sub_06d_2),
            "sub_06d_3": asdict(result.sub_06d_3) if result.sub_06d_3 is not None else None,
            "overall_passed": result.overall_passed,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    marker = _write_marker(args.marker_dir, result.overall_passed)

    print(f"[spike_0_6d] spec       = {payload['spec_reference']}")
    print(
        f"[spike_0_6d] sub_06d_1  = "
        f"{'PASS' if result.sub_06d_1.passed else 'FAIL'} "
        f"(symmetry={'PASS' if result.sub_06d_1.symmetry_passed else 'FAIL'}, "
        f"magnitude={'PASS' if result.sub_06d_1.magnitude_passed else 'FAIL'})"
    )
    print(
        f"[spike_0_6d] sub_06d_2  = "
        f"{'PASS' if result.sub_06d_2.passed else 'FAIL'} "
        f"(rel_err={result.sub_06d_2.relative_error:+.4f}, "
        f"tolerance ±{result.sub_06d_2.tolerance:.2f})"
    )
    print(f"[spike_0_6d] sub_06d_3  = {sub_3_note} (advisory; does not gate)")
    print(f"[spike_0_6d] OVERALL    = {'PASS' if result.overall_passed else 'FAIL'}  -> {marker}")
    print(f"[spike_0_6d] results    = {args.out}")

    return 0 if result.overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
