#!/usr/bin/env python
"""Spike 0.6d — aggregator + Phase 4 launch gate marker (H10 supplement).

Spec reference: docs/report-final.md §Phase 0 Spike 0.6d; protocol in
docs/spike_0_6d_protocol.md. Gate redesigned 2026-05-15 — see
docs/phase_logs/phase_0_signoff.md Note 3.

**Gate semantics (post-2026-05-15 redesign):** ``overall_passed =
sub_06d_2.freq_consistency_passed`` ONLY. Sub-spike 0.6d.2's two-frequency
added-mass consistency check is the sole, normalization-invariant Phase-4
gate. Sub-spike 0.6d.1 (symmetry/dimensional) and 0.6d.3 (incompressible)
are ADVISORY — their results are recorded in the result JSON (they feed
Phase 5 step 62.5) but do NOT affect the marker decision.

Phase 4 launch (``scripts/launch_phase4.py``) refuses to create the
``phase4-launch`` git tag unless BOTH ``data/spike_0_6c/PASS`` AND
``data/spike_0_6d/PASS`` are present.

Inputs:

* ``--sub-2-json`` — sub-spike 0.6d.2 result JSON (REQUIRED; the gate).
* ``--sub-1-json`` — (optional) sub-spike 0.6d.1 advisory result JSON.
* ``--sub-3-json`` — (optional) sub-spike 0.6d.3 advisory result JSON.

Outputs:

* ``data/spike_0_6d/results.json`` — aggregate result.
* ``data/spike_0_6d/PASS`` or ``data/spike_0_6d/FAIL`` — Phase 4 gate marker.

Exit codes:

* ``0`` — sub_2 frequency-consistency passed; ``PASS`` marker written.
* ``1`` — sub_2 frequency-consistency failed; ``FAIL`` marker written.
* ``2`` — input error (sub_2 result JSON missing / malformed).
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
DEFAULT_SUB_2_JSON = DEFAULT_DIR / "sub_2_result.json"
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
    """Advisory 0.6d.1 loader (does NOT gate, post-2026-05-15 redesign)."""
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
    """GATING 0.6d.2 loader — the redesigned freq-consistency result."""
    r = _get_result_block(_load_json(path), path)
    return Tier1AddedMassResult(
        omega_f1_rad_per_s=_float_or_nan(r.get("omega_f1_rad_per_s")),
        omega_f2_rad_per_s=_float_or_nan(r.get("omega_f2_rad_per_s")),
        recovered_ia_nondim_f1=_float_or_nan(r.get("recovered_ia_nondim_f1")),
        recovered_ia_nondim_f2=_float_or_nan(r.get("recovered_ia_nondim_f2")),
        freq_consistency_rel_diff=_float_or_nan(r.get("freq_consistency_rel_diff")),
        freq_consistency_tol=_float_or_nan(r.get("freq_consistency_tol")),
        freq_consistency_passed=bool(r.get("freq_consistency_passed", False)),
        closed_form_ia_nondim=_float_or_nan(r.get("closed_form_ia_nondim")),
        closed_form_factor_f1=_float_or_nan(r.get("closed_form_factor_f1")),
        closed_form_factor_tol=_float_or_nan(r.get("closed_form_factor_tol")),
        closed_form_advisory_ok=bool(r.get("closed_form_advisory_ok", False)),
        drag_to_added_mass_ratio_f1=_float_or_nan(r.get("drag_to_added_mass_ratio_f1")),
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
    p.add_argument(
        "--sub-2-json",
        type=Path,
        default=DEFAULT_SUB_2_JSON,
        help="GATING: sub-spike 0.6d.2 freq-consistency result JSON (required).",
    )
    p.add_argument(
        "--sub-1-json",
        type=Path,
        default=None,
        help=(
            "Optional ADVISORY sub-spike 0.6d.1 result JSON (post-2026-05-15 "
            "redesign: recorded for Phase 5, does NOT gate)."
        ),
    )
    p.add_argument(
        "--sub-3-json",
        type=Path,
        default=None,
        help=(
            "Optional ADVISORY sub-spike 0.6d.3 result JSON. Recorded for "
            "Phase 5; does NOT gate."
        ),
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--marker-dir", type=Path, default=DEFAULT_DIR)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        sub_2 = _load_sub_2(args.sub_2_json)  # the ONLY gating input
    except (FileNotFoundError, ValueError) as e:
        print(f"[spike_0_6d] input error (sub_2 is required): {e}", file=sys.stderr)
        return 2

    sub_1 = None
    sub_1_note = "not run"
    if args.sub_1_json is not None and args.sub_1_json.exists():
        try:
            sub_1 = _load_sub_1(args.sub_1_json)
            sub_1_note = "PASS" if sub_1.passed else "FAIL (advisory; does not gate)"
        except ValueError as e:
            print(f"[spike_0_6d] sub_1 advisory result unreadable; ignoring: {e}", file=sys.stderr)

    sub_3 = None
    sub_3_note = "not run"
    if args.sub_3_json is not None and args.sub_3_json.exists():
        try:
            sub_3 = _load_sub_3(args.sub_3_json)
            sub_3_note = "PASS" if sub_3.passed else "FAIL (advisory; does not gate)"
        except ValueError as e:
            print(f"[spike_0_6d] sub_3 advisory result unreadable; ignoring: {e}", file=sys.stderr)

    result = analyze_spike_06d(sub_2, sub_1, sub_3)

    payload = {
        "spec_reference": "docs/report-final.md §Phase 0 Spike 0.6d",
        "lock_reference": "H10 supplement; redesigned 2026-05-15 (freq-consistency gate)",
        "gate_note": (
            "overall_passed = sub_2.freq_consistency_passed ONLY (the "
            "normalization-invariant added-mass frequency-consistency check). "
            "sub_1 (symmetry/dimensional) and sub_3 (incompressible) are "
            "ADVISORY — recorded for Phase 5, do NOT gate. Phase 4 launch "
            "requires BOTH data/spike_0_6c/PASS AND data/spike_0_6d/PASS."
        ),
        "phase5_step_62_5_pointer": (
            "Absolute-accuracy validation (regime-appropriate published "
            "reference + OpenFOAM cross-codebase) is the Phase 5 step 62.5 "
            "deliverable; see docs/report-final.md."
        ),
        "result": {
            "sub_06d_2": asdict(result.sub_06d_2),
            "sub_06d_1": asdict(result.sub_06d_1) if result.sub_06d_1 is not None else None,
            "sub_06d_3": asdict(result.sub_06d_3) if result.sub_06d_3 is not None else None,
            "overall_passed": result.overall_passed,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    marker = _write_marker(args.marker_dir, result.overall_passed)

    print(f"[spike_0_6d] spec       = {payload['spec_reference']}")
    print(
        f"[spike_0_6d] sub_06d_2  = "
        f"{'PASS' if result.sub_06d_2.passed else 'FAIL'} (GATING; "
        f"freq rel_diff={result.sub_06d_2.freq_consistency_rel_diff:.4f}, "
        f"tol {result.sub_06d_2.freq_consistency_tol}; "
        f"closed-form advisory_ok={result.sub_06d_2.closed_form_advisory_ok})"
    )
    print(f"[spike_0_6d] sub_06d_1  = {sub_1_note} (ADVISORY; does not gate)")
    print(f"[spike_0_6d] sub_06d_3  = {sub_3_note} (ADVISORY; does not gate)")
    print(f"[spike_0_6d] OVERALL    = {'PASS' if result.overall_passed else 'FAIL'}  -> {marker}")
    print(f"[spike_0_6d] results    = {args.out}")

    return 0 if result.overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
