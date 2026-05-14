#!/usr/bin/env python
"""Spike 0.6c — aggregator + Phase 4 launch gate marker.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.6c (lines 1839-1844);
protocol in docs/spike_0_6c_protocol.md.

Reads the per-sub-spike result JSONs written by
``scripts/run_spike_0_6c_1.py`` and ``scripts/run_spike_0_6c_2.py``,
rolls them up into a ``Spike06cResult``, and writes the
``phase0/spike_0_6c/PASS`` marker iff BOTH sub-spikes individually
passed.

Phase 4 launch (``scripts/launch_phase4.py``) refuses to create the
``phase4-launch`` git tag if the ``PASS`` marker is absent.

Inputs (defaults point at the per-runner output paths):

* ``--sub-1-json`` — sub-spike 0.6c.1 result JSON.
* ``--sub-2-json`` — sub-spike 0.6c.2 result JSON.

Outputs:

* ``data/spike_0_6c/results.json`` — aggregate result.
* ``data/spike_0_6c/PASS`` or ``data/spike_0_6c/FAIL`` — Phase 4 launch
  gate marker.

Exit codes:

* ``0`` — both sub-spikes passed; ``PASS`` marker written.
* ``1`` — at least one sub-spike failed; ``FAIL`` marker written.
* ``2`` — input error (missing / malformed result JSON).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

from fanopt.cfd.spike_0_6c import (
    BenchmarkCycleData,
    BenchmarkResult,
    ConvergenceCheck,
    SymmetryCheck,
    Tier1CfgSanityResult,
    analyze_spike_06c,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "data" / "spike_0_6c"
DEFAULT_SUB_1_JSON = DEFAULT_DIR / "sub_1_result.json"
DEFAULT_SUB_2_JSON = DEFAULT_DIR / "sub_2_result.json"
DEFAULT_OUT_JSON = DEFAULT_DIR / "results.json"


# ---- result loaders -------------------------------------------------------


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"result JSON not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: invalid JSON: {e}") from e


def _load_sub_1(path: Path) -> Tier1CfgSanityResult:
    payload = _load_json(path)
    r = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(r, dict):
        raise ValueError(f"{path}: missing 'result' block")
    try:
        mach = r["mach_value"]
        return Tier1CfgSanityResult(
            cfg_path=str(r.get("cfg_path", "")),
            parsed_ok=bool(r.get("parsed_ok", False)),
            mach_value=(
                float(mach)
                if mach is not None and not (isinstance(mach, str) and mach.lower() == "nan")
                else math.nan
            ),
            freestream_option=str(r.get("freestream_option", "") or ""),
            ref_dimensionalization=r.get("ref_dimensionalization"),
            outer_time_steps_completed=int(r.get("outer_time_steps_completed", 0)),
            error=r.get("error"),
            passed=bool(r.get("passed", False)),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{path}: malformed sub_1 result: {e}") from e


def _load_sub_2(path: Path) -> BenchmarkResult:
    payload = _load_json(path)
    r = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(r, dict):
        raise ValueError(f"{path}: missing 'result' block")
    try:
        cycles = tuple(
            BenchmarkCycleData(
                cycle_index=int(c["cycle_index"]),
                c_l_max=float(c["c_l_max"]),
                c_l_min=float(c["c_l_min"]),
                c_d_mean=float(c["c_d_mean"]),
                c_l_hysteresis_area=float(c["c_l_hysteresis_area"]),
            )
            for c in r.get("cycles", [])
        )
        convergence = tuple(
            ConvergenceCheck(
                metric_name=str(c["metric_name"]),
                values=tuple(float(v) for v in c["values"]),
                mean=float(c["mean"]),
                relative_range_pct=float(c["relative_range_pct"]),
                passed=bool(c["passed"]),
            )
            for c in r.get("convergence", [])
        )
        sym = r["symmetry"]
        symmetry = SymmetryCheck(
            c_l_max_mean=float(sym["c_l_max_mean"]),
            c_l_min_mean=float(sym["c_l_min_mean"]),
            asymmetry_pct=float(sym["asymmetry_pct"]),
            passed=bool(sym["passed"]),
        )
        return BenchmarkResult(
            k_reduced=float(r["k_reduced"]),
            reynolds=float(r["reynolds"]),
            cycles=cycles,
            convergence=convergence,
            symmetry=symmetry,
            diagnostic_hysteresis_area_mean=float(r.get("diagnostic_hysteresis_area_mean", 0.0)),
            convergence_passed=bool(r.get("convergence_passed", False)),
            symmetry_passed=bool(r.get("symmetry_passed", False)),
            passed=bool(r.get("passed", False)),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{path}: malformed sub_2 result: {e}") from e


# ---- I/O ------------------------------------------------------------------


def _write_marker(out_dir: Path, passed: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("PASS", "FAIL"):
        stale_path = out_dir / stale
        if stale_path.exists():
            stale_path.unlink()
    marker = out_dir / ("PASS" if passed else "FAIL")
    marker.write_text("")
    return marker


# ---- CLI ------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sub-1-json",
        type=Path,
        default=DEFAULT_SUB_1_JSON,
        help="Sub-spike 0.6c.1 result JSON. Default: %(default)s.",
    )
    p.add_argument(
        "--sub-2-json",
        type=Path,
        default=DEFAULT_SUB_2_JSON,
        help="Sub-spike 0.6c.2 result JSON. Default: %(default)s.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_JSON,
        help="Where to write the aggregate results.json. Default: %(default)s.",
    )
    p.add_argument(
        "--marker-dir",
        type=Path,
        default=DEFAULT_DIR,
        help="Where to write the PASS / FAIL marker. Default: %(default)s.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        sub_1 = _load_sub_1(args.sub_1_json)
        sub_2 = _load_sub_2(args.sub_2_json)
    except (FileNotFoundError, ValueError) as e:
        print(f"[spike_0_6c] input error: {e}", file=sys.stderr)
        return 2

    result = analyze_spike_06c(sub_1, sub_2)

    payload = {
        "spec_reference": "docs/plan_R11.md §Phase 0 Spike 0.6c",
        "lock_reference": "H10 (Tier-1 cfg benchmark) + Round-9 HIGH-12 / C12 (unsteady MACH)",
        "phase4_gate_note": (
            "Phase 4 launch (scripts/launch_phase4.py) refuses to create the "
            "`phase4-launch` tag if data/spike_0_6c/PASS is absent."
        ),
        "result": {
            "sub_06c_1": asdict(result.sub_06c_1),
            "sub_06c_2": {
                "k_reduced": result.sub_06c_2.k_reduced,
                "reynolds": result.sub_06c_2.reynolds,
                "cycles": [asdict(c) for c in result.sub_06c_2.cycles],
                "convergence": [asdict(c) for c in result.sub_06c_2.convergence],
                "symmetry": asdict(result.sub_06c_2.symmetry),
                "diagnostic_hysteresis_area_mean": (
                    result.sub_06c_2.diagnostic_hysteresis_area_mean
                ),
                "convergence_passed": result.sub_06c_2.convergence_passed,
                "symmetry_passed": result.sub_06c_2.symmetry_passed,
                "passed": result.sub_06c_2.passed,
            },
            "overall_passed": result.overall_passed,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    marker = _write_marker(args.marker_dir, result.overall_passed)

    print(f"[spike_0_6c] spec       = {payload['spec_reference']}")
    print(
        f"[spike_0_6c] sub_06c_1  = "
        f"{'PASS' if result.sub_06c_1.passed else 'FAIL'} "
        f"(MACH={result.sub_06c_1.mach_value!r}, "
        f"FREESTREAM_OPTION={result.sub_06c_1.freestream_option!r}, "
        f"outer_steps={result.sub_06c_1.outer_time_steps_completed})"
    )
    print(
        f"[spike_0_6c] sub_06c_2  = "
        f"{'PASS' if result.sub_06c_2.passed else 'FAIL'} "
        f"(convergence_passed={result.sub_06c_2.convergence_passed}, "
        f"symmetry_passed={result.sub_06c_2.symmetry_passed})"
    )
    print(
        f"[spike_0_6c] OVERALL    = " f"{'PASS' if result.overall_passed else 'FAIL'}  -> {marker}"
    )
    print(f"[spike_0_6c] results    = {args.out}")

    return 0 if result.overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
