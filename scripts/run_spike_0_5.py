#!/usr/bin/env python
"""Spike 0.5 analyzer — single-blade fabrication-noise floor.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.5; protocol in
docs/spike_0_5_protocol.md.

Reads one CSV the operator filled out at the bench:

  --measurements   One row per printed blade (3 required per spec).
                   Columns expected:
                     blade_id, mass_g, d1_mm, d2_mm, ..., d10_mm,
                     bend_deflection_mm, j_fan_proxy, notes
                   Any column matching the regex ^d\\d+_mm$ is treated as
                   a dimension measurement; the spec requires ≥ 10 such
                   columns (10-point caliper grid).

Writes results.json with the FabNoiseResult dataclass. A per-metric CV
table is printed to stdout with ✓/✗ marks for each gate.

Exit codes:
  0 — PASS (every metric's CV below 5%)
  1 — FAIL (at least one metric's CV at or above 5%)
  2 — input error (missing file, missing column, bad parse, < 3 blades, etc.)

Pass criterion (§Phase 0 Spike 0.5):
  - J_fan-proxy CV < 5% across the three single-blade fans, AND
  - mass, 10-point dimension, and bend-deflection CVs each < 5%.

Mitigation if J_fan CV ≥ 5%:
  Tighten the print process (linear / pressure advance per Spike 0.4
  fallback) or commit only to gains > 15% (memo issue #16). Record the
  achieved CV in the Drive / JSONL ledger; all subsequent J_fan deltas
  compare against this floor.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

from fanopt.physical.fab_noise import (
    CV_GATE_PCT,
    N_BLADES_REQUIRED,
    BladeMeasurements,
    analyze_fab_noise,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "spike_0_5"
DEFAULT_MEASUREMENTS = DEFAULT_DATA_DIR / "measurements.csv"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "results.json"

# Regex for "dN_mm" dimension columns (N is a positive integer).
DIMENSION_COL_RE = re.compile(r"^d(\d+)_mm$")

# Per spec the 10-point caliper grid is the floor: anything less is an
# operator data-entry error, not a permissible variation.
MIN_DIMENSION_COLS = 10


# ─────────────────────────────────────────────────────────────────────
# CSV ingestion helpers
# ─────────────────────────────────────────────────────────────────────


def _decomment(lines):
    """Skip blank lines and lines starting with '#' (CSV-friendly comments)."""
    for line in lines:
        s = line.lstrip()
        if not s or s.startswith("#"):
            continue
        yield line


def _parse_float(path: Path, row_idx: int, col: str, raw: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"{path} row {row_idx}: {col} is not a float: {raw!r}"
        ) from e


def _parse_int(path: Path, row_idx: int, col: str, raw: str) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"{path} row {row_idx}: {col} is not an int: {raw!r}"
        ) from e


def _dimension_columns(fieldnames: list[str]) -> list[str]:
    """Return dN_mm columns in ascending N order."""
    pairs: list[tuple[int, str]] = []
    for name in fieldnames:
        m = DIMENSION_COL_RE.match(name)
        if m is not None:
            pairs.append((int(m.group(1)), name))
    pairs.sort(key=lambda p: p[0])
    return [name for _, name in pairs]


def _read_measurements(path: Path) -> tuple[list[BladeMeasurements], list[dict], list[str]]:
    """Return (parsed blades, full rows, dimension-column names in order)."""
    with path.open(newline="") as f:
        reader = csv.DictReader(_decomment(f))
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not rows:
        raise ValueError(f"{path}: no measurement rows found")

    dim_cols = _dimension_columns(fieldnames)
    if len(dim_cols) < MIN_DIMENSION_COLS:
        raise ValueError(
            f"{path}: need ≥ {MIN_DIMENSION_COLS} dimension columns matching "
            f"'dN_mm' (found {len(dim_cols)}: {dim_cols})"
        )

    required = ("blade_id", "mass_g", "bend_deflection_mm", "j_fan_proxy")
    for col in required:
        if col not in fieldnames:
            raise ValueError(
                f"{path}: missing required column {col!r} "
                f"(found: {fieldnames})"
            )

    blades: list[BladeMeasurements] = []
    for i, row in enumerate(rows, start=1):
        # Catch a header that drifts row-to-row (DictReader should not let
        # this happen but defend anyway since CSVs are operator-edited).
        missing = [c for c in required if c not in row]
        if missing:
            raise ValueError(
                f"{path} row {i}: missing column(s) {missing} "
                f"(found: {list(row.keys())})"
            )
        for c in dim_cols:
            if c not in row:
                raise ValueError(
                    f"{path} row {i}: missing dimension column {c!r} "
                    f"(found: {list(row.keys())})"
                )

        blade_id = _parse_int(path, i, "blade_id", row["blade_id"])
        mass_g = _parse_float(path, i, "mass_g", row["mass_g"])
        bend_mm = _parse_float(
            path, i, "bend_deflection_mm", row["bend_deflection_mm"]
        )
        j_fan = _parse_float(path, i, "j_fan_proxy", row["j_fan_proxy"])
        dims = tuple(
            _parse_float(path, i, c, row[c]) for c in dim_cols
        )
        blades.append(
            BladeMeasurements(
                blade_id=blade_id,
                mass_g=mass_g,
                dimension_mm_10pt=dims,
                three_point_bend_deflection_mm=bend_mm,
                j_fan_proxy=j_fan,
            )
        )

    if len(blades) < N_BLADES_REQUIRED:
        raise ValueError(
            f"{path}: need ≥ {N_BLADES_REQUIRED} blade rows per spec, "
            f"got {len(blades)}"
        )

    return blades, rows, dim_cols


# ─────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurements",
        type=Path,
        default=DEFAULT_MEASUREMENTS,
        help="Per-blade measurements CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write results.json (default: %(default)s)",
    )
    return parser.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        blades, raw_rows, dim_cols = _read_measurements(args.measurements)
    except (FileNotFoundError, ValueError) as e:
        print(f"[spike_0_5] input error: {e}", file=sys.stderr)
        return 2

    try:
        result = analyze_fab_noise(blades)
    except ValueError as e:
        print(f"[spike_0_5] analysis error: {e}", file=sys.stderr)
        return 2

    payload = {
        "spec_reference": "docs/plan_R11.md §Phase 0 Spike 0.5",
        "inputs": {
            "measurements_path": str(args.measurements),
            "n_blades": len(blades),
            "dimension_columns": dim_cols,
            "measurement_rows": raw_rows,
        },
        "gates": {
            "cv_gate_pct": CV_GATE_PCT,
            "n_blades_required": N_BLADES_REQUIRED,
        },
        "result": asdict(result),
        # Canonical top-level alias for downstream Phase 6 consumers — the
        # plan says "the achieved CV is the noise floor" and "all subsequent
        # J_fan deltas must be compared against this floor". Surface it
        # explicitly rather than burying it under result.j_fan_cv.cv_pct.
        "published_noise_floor_pct": result.j_fan_cv.cv_pct,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    _print_table(payload, args.out)
    return 0 if result.overall_passed else 1


def _mark(ok: bool) -> str:
    return "✓" if ok else "✗"


def _print_table(payload: dict, out_path: Path) -> None:
    r = payload["result"]
    n = payload["inputs"]["n_blades"]
    print(f"\n[spike_0_5] n_blades       = {n} (required ≥ {N_BLADES_REQUIRED})")
    print(
        f"[spike_0_5] CV gate        = < {CV_GATE_PCT}% (per metric)"
    )
    print(
        f"[spike_0_5] {'metric':<22} {'mean':>12} {'std':>12} "
        f"{'cv_pct':>8}   verdict"
    )
    for cv in (r["mass_cv"], r["dimension_cv"], r["bend_cv"], r["j_fan_cv"]):
        print(
            f"[spike_0_5] {cv['metric_name']:<22} "
            f"{cv['mean']:>12.4f} {cv['std']:>12.4f} "
            f"{cv['cv_pct']:>7.3f}%  {_mark(cv['passed'])}"
        )
    overall = "PASS" if r["overall_passed"] else "FAIL"
    print(f"[spike_0_5] {overall}")
    if not r["overall_passed"]:
        # The spec's mitigation rule (memo issue #16): when CV exceeds 5%,
        # either tighten the print process or commit only to gains > 15%.
        # Print the achieved J_fan CV so the operator can quote it into
        # the Drive / JSONL ledger directly.
        print(
            f"[spike_0_5] mitigation: record achieved J_fan CV "
            f"= {r['j_fan_cv']['cv_pct']:.3f}% in Drive/JSONL ledger; "
            f"compare all subsequent J_fan deltas against this floor."
        )
    print(f"[spike_0_5] wrote          {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
