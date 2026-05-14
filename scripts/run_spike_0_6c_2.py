#!/usr/bin/env python
"""Spike 0.6c.2 — NACA 0012 oscillating-airfoil benchmark validation runner.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.6c (lines 1839-1844);
protocol in docs/spike_0_6c_protocol.md.

Consumes the operator-supplied per-cycle measurements from a 5-cycle
NACA 0012 pitching simulation (run through the locked Tier-1 cfg with
``k_reduced ≈ 0.55``, ``Re ≈ 40k``, pitching about quarter-chord) and
compares the integrated last-4-cycle metrics to published references.

Inputs
------

``--measured`` (required)
    CSV with one row per cycle. Columns:
    ``cycle_index, c_l_max, c_l_min, c_d_mean, c_l_hysteresis_area``.
    The aggregator discards the first cycle (initial transient) and
    integrates the remaining 4 per the spec.

``--reference``
    Optional JSON file with reference values keyed by metric name. If
    omitted, the runner ships a default ``NACA0012_REFERENCE`` dict
    representative of the McAlister/Carr UH110A studies.

``--k-reduced`` / ``--reynolds``
    Reduced-frequency and Reynolds number actually run. Recorded in the
    result so the operator can verify both fall in the spec bands
    ([0.5, 0.6] and [30000, 50000]).

``--reference-source``
    Free-form provenance string for the reference values (e.g.,
    "McAlister/Carr UH110A 1978, fig 7c").

Outputs
-------

* ``data/spike_0_6c/sub_2_result.json`` — serialized
  ``BenchmarkResult``.
* ``data/spike_0_6c/sub_2.PASS`` or ``data/spike_0_6c/sub_2.FAIL`` —
  marker file consumed by the Spike 0.6c aggregator.

Exit codes
----------

* ``0`` — all metrics within ±15% of their references.
* ``1`` — at least one metric outside the ±15% tolerance.
* ``2`` — input error (missing file, missing column, non-numeric).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

from fanopt.cfd.spike_0_6c import (
    BENCHMARK_K_REDUCED_MAX,
    BENCHMARK_K_REDUCED_MIN,
    BENCHMARK_RE_MAX,
    BENCHMARK_RE_MIN,
    BENCHMARK_TOLERANCE_PCT,
    NACA0012_REFERENCE,
    BenchmarkCycleData,
    analyze_benchmark,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "spike_0_6c"
DEFAULT_RESULT_JSON = DEFAULT_OUTPUT_DIR / "sub_2_result.json"


# ---- CSV plumbing ---------------------------------------------------------


def _decomment(lines):
    for line in lines:
        s = line.lstrip()
        if not s or s.startswith("#"):
            continue
        yield line


def _read_measured(path: Path) -> tuple[BenchmarkCycleData, ...]:
    if not path.exists():
        raise FileNotFoundError(f"measured CSV not found: {path}")
    required = {
        "cycle_index",
        "c_l_max",
        "c_l_min",
        "c_d_mean",
        "c_l_hysteresis_area",
    }
    with path.open(newline="") as f:
        reader = csv.DictReader(_decomment(f))
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no measured rows found")
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(
            f"{path}: missing required columns {sorted(missing)} "
            f"(found: {sorted(rows[0].keys())})"
        )
    out: list[BenchmarkCycleData] = []
    for i, row in enumerate(rows, start=1):
        try:
            out.append(
                BenchmarkCycleData(
                    cycle_index=int(float(row["cycle_index"])),
                    c_l_max=float(row["c_l_max"]),
                    c_l_min=float(row["c_l_min"]),
                    c_d_mean=float(row["c_d_mean"]),
                    c_l_hysteresis_area=float(row["c_l_hysteresis_area"]),
                )
            )
        except (TypeError, ValueError) as e:
            raise ValueError(f"{path} row {i}: parse error: {e}") from e
    return tuple(out)


def _load_reference(path: Path | None) -> tuple[dict[str, float], str]:
    if path is None:
        return dict(NACA0012_REFERENCE), "shipped:NACA0012_REFERENCE"
    if not path.exists():
        raise FileNotFoundError(f"reference JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: reference JSON must be an object at top level")
    out: dict[str, float] = {}
    for k, v in data.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{path}: value for {k!r} is not numeric ({v!r})") from e
    return out, f"file:{path}"


# ---- I/O helpers ----------------------------------------------------------


def _write_result(
    out_path: Path,
    result_payload: dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result_payload, indent=2) + "\n")


def _write_marker(out_dir: Path, passed: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("sub_2.PASS", "sub_2.FAIL"):
        stale_path = out_dir / stale
        if stale_path.exists():
            stale_path.unlink()
    marker = out_dir / ("sub_2.PASS" if passed else "sub_2.FAIL")
    marker.write_text("")
    return marker


# ---- CLI ------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--measured",
        type=Path,
        required=True,
        help="CSV with per-cycle measured aerodynamic coefficients.",
    )
    p.add_argument(
        "--reference",
        type=Path,
        default=None,
        help=(
            "Optional reference JSON. If omitted, uses the shipped "
            "NACA0012_REFERENCE default."
        ),
    )
    p.add_argument(
        "--k-reduced",
        type=float,
        required=True,
        help=(
            f"Reduced frequency of the run (spec band: "
            f"{BENCHMARK_K_REDUCED_MIN}-{BENCHMARK_K_REDUCED_MAX})."
        ),
    )
    p.add_argument(
        "--reynolds",
        type=float,
        required=True,
        help=(
            f"Reynolds number of the run (spec band: "
            f"{BENCHMARK_RE_MIN:.0f}-{BENCHMARK_RE_MAX:.0f})."
        ),
    )
    p.add_argument(
        "--reference-source",
        type=str,
        default="operator-supplied",
        help="Free-form provenance string for the reference values.",
    )
    p.add_argument(
        "--result-json",
        type=Path,
        default=DEFAULT_RESULT_JSON,
        help="Where to write the sub_2 result JSON. Default: %(default)s.",
    )
    p.add_argument(
        "--marker-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write the PASS / FAIL marker. Default: %(default)s.",
    )
    p.add_argument(
        "--allow-out-of-band",
        action="store_true",
        help=(
            "Permit --k-reduced / --reynolds outside the locked NACA 0012 "
            "spec band [0.5, 0.6] / [30000, 50000]. Use ONLY when "
            "intentionally validating against a different published "
            "reference case (document the substitution in the phase log)."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        cycles = _read_measured(args.measured)
        reference, source_label = _load_reference(args.reference)
    except (FileNotFoundError, ValueError) as e:
        print(f"[spike_0_6c_2] input error: {e}", file=sys.stderr)
        return 2

    # Hard gate on band mismatch — operator overrides via --allow-out-of-band
    # only when intentionally validating against a different reference case.
    # Per the audit finding: a run at k=2.0 with metrics happening to align
    # with the (unrelated) NACA-0012 reference should NOT mint a PASS marker.
    band_violations: list[str] = []
    if not (BENCHMARK_K_REDUCED_MIN <= args.k_reduced <= BENCHMARK_K_REDUCED_MAX):
        band_violations.append(
            f"k_reduced={args.k_reduced} outside spec band "
            f"[{BENCHMARK_K_REDUCED_MIN}, {BENCHMARK_K_REDUCED_MAX}]"
        )
    if not (BENCHMARK_RE_MIN <= args.reynolds <= BENCHMARK_RE_MAX):
        band_violations.append(
            f"reynolds={args.reynolds} outside spec band "
            f"[{BENCHMARK_RE_MIN:.0f}, {BENCHMARK_RE_MAX:.0f}]"
        )
    if band_violations:
        if args.allow_out_of_band:
            for msg in band_violations:
                print(f"[spike_0_6c_2] WARNING (override accepted): {msg}", file=sys.stderr)
        else:
            for msg in band_violations:
                print(f"[spike_0_6c_2] FAIL: {msg}", file=sys.stderr)
            print(
                "[spike_0_6c_2] refuse to run benchmark outside the locked band. "
                "Use --allow-out-of-band only when intentionally validating "
                "against a different published reference case (and document the "
                "substitution in the phase log).",
                file=sys.stderr,
            )
            return 2

    try:
        result = analyze_benchmark(
            cycles=cycles,
            reference=reference,
            k_reduced=args.k_reduced,
            reynolds=args.reynolds,
            reference_source=(
                args.reference_source
                if args.reference is not None
                else f"{args.reference_source} ({source_label})"
            ),
        )
    except ValueError as e:
        print(f"[spike_0_6c_2] analysis error: {e}", file=sys.stderr)
        return 2

    payload = {
        "spec_reference": "docs/plan_R11.md §Phase 0 Spike 0.6c (sub-spike 0.6c.2)",
        "lock_reference": "H10 (= Round-9 HIGH-10): Tier-1 cfg benchmark validation",
        "tolerance_pct": BENCHMARK_TOLERANCE_PCT,
        "inputs": {
            "measured_csv": str(args.measured),
            "reference_source": result.reference_source,
            "k_reduced": result.k_reduced,
            "reynolds": result.reynolds,
        },
        "result": {
            "k_reduced": result.k_reduced,
            "reynolds": result.reynolds,
            "reference_source": result.reference_source,
            "cycles": [asdict(c) for c in result.cycles],
            "comparisons": [asdict(c) for c in result.comparisons],
            "all_metrics_within_15pct": result.all_metrics_within_15pct,
            "passed": result.passed,
        },
    }

    _write_result(args.result_json, payload)
    marker = _write_marker(args.marker_dir, result.passed)

    print(
        f"[spike_0_6c_2] cycles               {len(result.cycles)} "
        f"(discard 1, integrate {len(result.cycles) - 1})"
    )
    print(
        f"[spike_0_6c_2] k_reduced            {result.k_reduced} "
        f"(band [{BENCHMARK_K_REDUCED_MIN}, {BENCHMARK_K_REDUCED_MAX}])"
    )
    print(
        f"[spike_0_6c_2] reynolds             {result.reynolds:g} "
        f"(band [{BENCHMARK_RE_MIN:.0f}, {BENCHMARK_RE_MAX:.0f}])"
    )
    print(f"[spike_0_6c_2] reference_source     {result.reference_source}")
    for c in result.comparisons:
        mark = "ok" if c.passed else "XX"
        print(
            f"  - {c.metric_name:<22} meas={c.measured:>10.4f}  "
            f"ref={c.reference:>10.4f}  pct={c.pct_diff:+7.2f}%  [{mark}]"
        )
    print(f"[spike_0_6c_2] passed               {result.passed}")
    print(f"[spike_0_6c_2] result_json          {args.result_json}")
    print(f"[spike_0_6c_2] marker               {marker}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
