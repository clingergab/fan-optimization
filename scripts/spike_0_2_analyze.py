#!/usr/bin/env python
"""Spike 0.2 analyzer — torsional-pendulum I_wrist from measured trials.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.2; protocol in
docs/spike_0_2_protocol.md.

Reads two CSVs the operator filled out at the bench:

  --calibration   Step-1 output: κ from the reference rod (one row used).
                  Columns expected: kappa_Nm_per_rad, T_ref_s, m_ref_kg,
                  L_ref_m, I_ref_kgm2, method, notes
                  Only `kappa_Nm_per_rad` is required; the rest are read
                  for the result log.

  --measurements  Step-2 output: one row per trial of the baseline fan.
                  Columns expected: trial, T_osc_s, amplitude_deg, notes
                  Only `T_osc_s` is required.

Computes I_wrist = κ · (T̄_osc / 2π)² per trial, then mean / repeatability /
cross-check vs the generator's emitted I_wrist_kgm2 (passed via
--generator-i-wrist). Writes results.json and prints a pass/fail table.

Pass criteria (§Phase 0 Spike 0.2):
  1. repeatability % < 3
  2. cross-check % < 10 (if --generator-i-wrist is provided)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

from fanopt.physical.inertia import (
    CROSS_CHECK_GATE_PCT,
    REPEATABILITY_GATE_PCT,
    analyze_trials,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION = REPO_ROOT / "data" / "spike_0_2" / "calibration.csv"
DEFAULT_MEASUREMENTS = REPO_ROOT / "data" / "spike_0_2" / "measurements.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "spike_0_2" / "results.json"


def _read_single_kappa(path: Path) -> tuple[float, dict[str, str]]:
    """Return (kappa, full_row) from the first non-comment row of calibration.csv."""
    with path.open(newline="") as f:
        reader = csv.DictReader(_decomment(f))
        row = next(iter(reader), None)
    if row is None:
        raise ValueError(f"{path}: no calibration rows found")
    if "kappa_Nm_per_rad" not in row:
        raise ValueError(
            f"{path}: missing 'kappa_Nm_per_rad' column " f"(found: {list(row.keys())})"
        )
    try:
        kappa = float(row["kappa_Nm_per_rad"])
    except ValueError as e:
        raise ValueError(
            f"{path}: kappa_Nm_per_rad is not a float: {row['kappa_Nm_per_rad']!r}"
        ) from e
    return kappa, row


def _read_trials(path: Path) -> list[tuple[int, float, dict[str, str]]]:
    """Return list of (trial_index, T_osc_s, full_row) from measurements.csv."""
    with path.open(newline="") as f:
        reader = csv.DictReader(_decomment(f))
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no measurement rows found")
    out: list[tuple[int, float, dict[str, str]]] = []
    for i, row in enumerate(rows, start=1):
        if "T_osc_s" not in row:
            raise ValueError(
                f"{path} row {i}: missing 'T_osc_s' column " f"(found: {list(row.keys())})"
            )
        try:
            T = float(row["T_osc_s"])
        except ValueError as e:
            raise ValueError(f"{path} row {i}: T_osc_s is not a float: {row['T_osc_s']!r}") from e
        trial_idx = int(row.get("trial", i) or i)
        out.append((trial_idx, T, row))
    return out


def _decomment(lines):
    """Skip blank lines and lines starting with '#' (CSV-friendly comments)."""
    for line in lines:
        s = line.lstrip()
        if not s or s.startswith("#"):
            continue
        yield line


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help="κ-calibration CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--measurements",
        type=Path,
        default=DEFAULT_MEASUREMENTS,
        help="T_osc trials CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--generator-i-wrist",
        type=float,
        default=None,
        help=(
            "I_wrist_kgm2 emitted by the §9.7 generator for the same fan "
            "(skip cross-check if omitted; the spike then only checks "
            "repeatability and is incomplete per §Phase 0 Spike 0.2)."
        ),
    )
    parser.add_argument(
        "--mount-i-wrist",
        type=float,
        default=0.0,
        help=(
            "Empty-rig inertia to subtract per trial (kg·m²). Measure once on "
            "the mount with no fan attached; defaults to 0."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write results.json (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        kappa, calib_row = _read_single_kappa(args.calibration)
        trials = _read_trials(args.measurements)
    except (FileNotFoundError, ValueError) as e:
        print(f"[spike_0_2] input error: {e}", file=sys.stderr)
        return 2

    try:
        result = analyze_trials(
            kappa_Nm_per_rad=kappa,
            T_osc_trials_s=[T for (_, T, _) in trials],
            generator_I_wrist_kgm2=args.generator_i_wrist,
            mount_I_wrist_kgm2=args.mount_i_wrist,
        )
    except ValueError as e:
        print(f"[spike_0_2] analysis error: {e}", file=sys.stderr)
        return 2

    payload = {
        "spec_reference": "docs/plan_R11.md §Phase 0 Spike 0.2",
        "inputs": {
            "kappa_Nm_per_rad": kappa,
            "calibration_row": calib_row,
            "n_trials": len(trials),
            "T_osc_trials_s": [T for (_, T, _) in trials],
            "trial_rows": [r for (_, _, r) in trials],
            "generator_I_wrist_kgm2": args.generator_i_wrist,
            "mount_I_wrist_kgm2": args.mount_i_wrist,
        },
        "gates": {
            "repeatability_gate_pct": REPEATABILITY_GATE_PCT,
            "cross_check_gate_pct": CROSS_CHECK_GATE_PCT,
        },
        "result": asdict(result),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    _print_table(payload, args.out)
    return 0 if result.passed else 1


def _print_table(payload: dict, out_path: Path) -> None:
    r = payload["result"]
    print(f"\n[spike_0_2] κ          = {payload['inputs']['kappa_Nm_per_rad']:.6e} N·m/rad")
    print(f"[spike_0_2] n_trials   = {r['n_trials']}")
    print(
        f"[spike_0_2] I_wrist    = {r['I_wrist_kgm2']:.6e} kg·m²  "
        f"(std {r['I_wrist_std_kgm2']:.3e})"
    )
    rep_mark = "✓" if r["repeatability_passed"] else "✗"
    print(
        f"[spike_0_2] repeat     = {r['repeatability_pct']:.2f}%  "
        f"(gate < {REPEATABILITY_GATE_PCT}%)  {rep_mark}"
    )
    if r["cross_check_pct"] is None:
        print("[spike_0_2] cross-check= (skipped — no --generator-i-wrist)")
    else:
        x_mark = "✓" if r["cross_check_passed"] else "✗"
        print(
            f"[spike_0_2] cross-chk  = {r['cross_check_pct']:.2f}%  "
            f"(gate < {CROSS_CHECK_GATE_PCT}%)  {x_mark}"
        )
    overall = "PASS" if r["passed"] else "FAIL"
    print(f"[spike_0_2] {overall}")
    print(f"[spike_0_2] wrote      {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
