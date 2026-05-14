#!/usr/bin/env python
"""Spike 0.6 aggregator — Colab compute-budget probe + M3 sub-spike rollup.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.6; protocol in
docs/spike_0_6_protocol.md.

Reads three CSVs the operator filled out from the Colab probe and the two
M3 sub-spike runners:

  --budget-csv   Compute-budget rows, one row per (platform, workload). At
                 minimum the spec asks for two rows: Colab CPU 3D unsteady,
                 Colab GPU 3D unsteady (500K cells, 5 pitching cycles,
                 dt = T/200 — the Tier-1 locked config).
                 Columns: platform, workload, wall_time_s, cu_consumed,
                          cells, notes

  --06a-csv      One row of the M3 SU2 Tier-1 end-to-end timing, optionally
                 with per-stage breakdown rows.
                 Columns: wall_time_s, J_fan_steady_proxy, stage_name,
                          stage_wall_time_s
                 The first row carries `wall_time_s` and
                 `J_fan_steady_proxy`; subsequent rows carry per-stage
                 entries (stage_name + stage_wall_time_s) with the first
                 two columns blank.

  --06b-csv      One row of the M3 FEA cantilever measurement.
                 Columns: wall_time_s, measured_tip_deflection_m, P_N,
                          L_m, E_Pa, I_m4

Writes results.json and prints a pass/fail table.

Exit codes:
  0  — both gates (sub-spike 0.6a + sub-spike 0.6b) passed (or absent &
       informational). Calibration always reports PASS per spec § "Status".
  1  — at least one sub-spike was provided AND failed its gate.
  2  — input error (missing file, missing column, non-numeric).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

from fanopt.utils.compute_probe import (
    M3_FEA_TIP_DEFLECTION_TOLERANCE_PCT,
    M3_FEA_WALL_TIME_GATE_S,
    M3_SU2_WALL_TIME_GATE_S,
    ComputeBudgetEntry,
    analyze_06a,
    analyze_06b,
    analyze_spike_06,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "spike_0_6" / "results.json"


# ---- CSV plumbing ---------------------------------------------------------


def _decomment(lines):
    """Skip blank lines and lines starting with '#' (CSV-friendly comments)."""
    for line in lines:
        s = line.lstrip()
        if not s or s.startswith("#"):
            continue
        yield line


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(_decomment(f))
        return list(reader)


def _opt_float(row: dict[str, str], key: str) -> float | None:
    val = row.get(key, "")
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() in {"none", "nan", "null"}:
        return None
    try:
        return float(s)
    except ValueError as e:
        raise ValueError(f"column {key!r}: not a float: {val!r}") from e


def _opt_int(row: dict[str, str], key: str) -> int | None:
    val = row.get(key, "")
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() in {"none", "null"}:
        return None
    try:
        return int(float(s))
    except ValueError as e:
        raise ValueError(f"column {key!r}: not an int: {val!r}") from e


def _required_float(row: dict[str, str], key: str, context: str) -> float:
    if key not in row:
        raise ValueError(f"{context}: missing required column {key!r}")
    s = str(row.get(key, "")).strip()
    if s == "":
        raise ValueError(f"{context}: required column {key!r} is empty")
    try:
        return float(s)
    except ValueError as e:
        raise ValueError(f"{context}: column {key!r} is not a float: {s!r}") from e


def _read_budget(path: Path) -> list[ComputeBudgetEntry]:
    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"{path}: no budget rows found")
    required = {"platform", "workload", "wall_time_s"}
    out: list[ComputeBudgetEntry] = []
    for i, row in enumerate(rows, start=1):
        missing = required - set(row.keys())
        if missing:
            raise ValueError(
                f"{path} row {i}: missing required columns {sorted(missing)} "
                f"(found: {sorted(row.keys())})"
            )
        platform = (row.get("platform") or "").strip()
        workload = (row.get("workload") or "").strip()
        if not platform or not workload:
            raise ValueError(f"{path} row {i}: platform / workload must be non-empty")
        wall = _required_float(row, "wall_time_s", f"{path} row {i}")
        cu = _opt_float(row, "cu_consumed")
        cells = _opt_int(row, "cells")
        notes = (row.get("notes") or "").strip()
        out.append(
            ComputeBudgetEntry(
                platform=platform,
                workload=workload,
                wall_time_s=wall,
                cu_consumed=cu,
                cells=cells,
                notes=notes,
            )
        )
    return out


def _read_06a(
    path: Path,
) -> tuple[float, float | None, dict[str, float]]:
    """Return (wall_time_s, J_fan_steady_proxy, stages) from the 0.6a CSV."""
    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"{path}: no rows found")

    head = rows[0]
    if "wall_time_s" not in head:
        raise ValueError(
            f"{path} row 1: missing required column 'wall_time_s' "
            f"(found: {sorted(head.keys())})"
        )
    wall_time_s = _required_float(head, "wall_time_s", f"{path} row 1")
    j_fan = _opt_float(head, "J_fan_steady_proxy")

    stages: dict[str, float] = {}
    for i, row in enumerate(rows, start=1):
        stage_name = (row.get("stage_name") or "").strip()
        stage_wall = (row.get("stage_wall_time_s") or "").strip()
        if not stage_name and not stage_wall:
            continue
        if not stage_name or not stage_wall:
            raise ValueError(
                f"{path} row {i}: stage_name and stage_wall_time_s must be "
                "provided together (or both blank)."
            )
        try:
            stages[stage_name] = float(stage_wall)
        except ValueError as e:
            raise ValueError(
                f"{path} row {i}: stage_wall_time_s not a float: {stage_wall!r}"
            ) from e
    return wall_time_s, j_fan, stages


def _read_06b(path: Path) -> dict[str, float]:
    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"{path}: no rows found")
    row = rows[0]
    required = ("wall_time_s", "measured_tip_deflection_m", "P_N", "L_m", "E_Pa", "I_m4")
    out: dict[str, float] = {}
    for key in required:
        out[key] = _required_float(row, key, f"{path} row 1")
    return out


# ---- argument parsing -----------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budget-csv",
        type=Path,
        default=None,
        help="Compute-budget CSV (Colab CPU / GPU rows). Optional — informational.",
    )
    parser.add_argument(
        "--06a-csv",
        dest="sub_06a_csv",
        type=Path,
        default=None,
        help="Sub-spike 0.6a CSV (M3 SU2 Tier-1 timing).",
    )
    parser.add_argument(
        "--06b-csv",
        dest="sub_06b_csv",
        type=Path,
        default=None,
        help="Sub-spike 0.6b CSV (M3 FEA cantilever).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write results.json (default: %(default)s).",
    )
    return parser.parse_args(argv)


# ---- main -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        budget = _read_budget(args.budget_csv) if args.budget_csv is not None else []
        if args.sub_06a_csv is not None:
            wall_a, j_fan, stages = _read_06a(args.sub_06a_csv)
            sub_06a = analyze_06a(wall_time_s=wall_a, J_fan_steady_proxy=j_fan, stages=stages)
        else:
            sub_06a = None
        if args.sub_06b_csv is not None:
            params = _read_06b(args.sub_06b_csv)
            sub_06b = analyze_06b(
                wall_time_s=params["wall_time_s"],
                measured_deflection_m=params["measured_tip_deflection_m"],
                P_N=params["P_N"],
                L_m=params["L_m"],
                E_Pa=params["E_Pa"],
                I_m4=params["I_m4"],
            )
        else:
            sub_06b = None
    except (FileNotFoundError, ValueError) as e:
        print(f"[spike_0_6] input error: {e}", file=sys.stderr)
        return 2

    result = analyze_spike_06(budget_entries=budget, sub_06a=sub_06a, sub_06b=sub_06b)

    payload = {
        "spec_reference": "docs/plan_R11.md §Phase 0 Spike 0.6",
        "status_note": (
            "Spike 0.6 is calibration, not a gate (per spec § 'Status'). "
            "Sub-spikes 0.6a and 0.6b ARE gates for their downstream phases."
        ),
        "gates": {
            "m3_su2_wall_time_gate_s": M3_SU2_WALL_TIME_GATE_S,
            "m3_fea_wall_time_gate_s": M3_FEA_WALL_TIME_GATE_S,
            "m3_fea_tip_deflection_tolerance_pct": M3_FEA_TIP_DEFLECTION_TOLERANCE_PCT,
        },
        "result": {
            "budget_entries": [asdict(b) for b in result.budget_entries],
            "sub_06a": asdict(result.sub_06a) if result.sub_06a is not None else None,
            "sub_06b": asdict(result.sub_06b) if result.sub_06b is not None else None,
            "overall_passed": result.overall_passed,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    _print_table(payload, args.out)

    # Aggregator exit code: 0 iff every supplied sub-spike passed. If a sub-
    # spike CSV was not supplied, treat it as "not yet run" — that's
    # informational and exits 0, matching the calibration framing.
    failures = [s for s in (result.sub_06a, result.sub_06b) if s is not None and not s.passed]
    return 1 if failures else 0


def _print_table(payload: dict, out_path: Path) -> None:
    print(f"\n[spike_0_6] spec ref   = {payload['spec_reference']}")
    print("[spike_0_6] status     = calibration (overall_passed always True)")

    budget = payload["result"]["budget_entries"]
    print(f"[spike_0_6] budget rows = {len(budget)}")
    for b in budget:
        cu = "-" if b["cu_consumed"] is None else f"{b['cu_consumed']:.2f}"
        cells = "-" if b["cells"] is None else f"{b['cells']}"
        print(
            f"  - {b['platform']:<24} {b['workload']:<36}"
            f" t={b['wall_time_s']:.0f}s  cu={cu}  cells={cells}"
        )

    a = payload["result"]["sub_06a"]
    if a is None:
        print("[spike_0_6] sub-0.6a   = (skipped — no --06a-csv)")
    else:
        mark = "PASS" if a["passed"] else "FAIL"
        wt_mark = "ok" if a["wall_time_passed"] else "XX"
        jf_mark = "ok" if a["j_fan_finite"] else "XX"
        print(
            f"[spike_0_6] sub-0.6a   = wall={a['m3_wall_time_s']:.1f}s "
            f"(gate <= {M3_SU2_WALL_TIME_GATE_S:.0f}s, {wt_mark})"
            f"  J_fan_finite={jf_mark}  -> {mark}"
        )

    b = payload["result"]["sub_06b"]
    if b is None:
        print("[spike_0_6] sub-0.6b   = (skipped — no --06b-csv)")
    else:
        mark = "PASS" if b["passed"] else "FAIL"
        wt_mark = "ok" if b["wall_time_passed"] else "XX"
        td_mark = "ok" if b["tip_deflection_passed"] else "XX"
        print(
            f"[spike_0_6] sub-0.6b   = wall={b['wall_time_s']:.1f}s "
            f"(gate <= {M3_FEA_WALL_TIME_GATE_S:.0f}s, {wt_mark})"
            f"  tip_pct={b['tip_deflection_pct']:.2f}%"
            f" (gate <= {M3_FEA_TIP_DEFLECTION_TOLERANCE_PCT}%, {td_mark})"
            f"  -> {mark}"
        )

    print(f"[spike_0_6] wrote      {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
