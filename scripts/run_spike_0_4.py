#!/usr/bin/env python
"""Spike 0.4 analyzer — click feature tolerance, cycle life, and V1-lock force balance.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.4; protocol in
docs/spike_0_4_protocol.md.

Reads four CSVs the operator filled out at the bench:

  --force-balance      Step-2 output: one row used.
                       Columns expected:
                         I_wrist_kgm2, F_friction_cumulative_N, notes

  --clearance          Step-3 output: ≥ 1 row per mating surface.
                       Columns expected:
                         mating_surface, clearance_mm, notes

  --engagement-force   Step-4 + Step-6 output: rows for the low regime
                       (deploy-from-folded) and optionally rows for the
                       high-amplitude stress segment.
                       Columns expected:
                         trial, force_N, regime, notes
                       `regime` must be 'low' or 'high'.

  --cycle-inspections  Step-5 output: one row per every-100-cycle inspection.
                       Columns expected:
                         cycle, wear_observed, fracture, notes

Also takes:
  --alignment-gap-variation-mm   worst-case gap variation across deployed
                                  blade tips, mm.
  --high-amp-completed           bool (default false).
  --high-amp-failure-cycle       int|None — cycle index at which the detent
                                  fractured within the 100-cycle high-amp
                                  segment; omit if no fracture.

Writes results.json with the Spike04Result dataclass.  Pass/fail table is
printed to stdout with ✓/✗ marks for each sub-gate.

Exit codes:
  0 — overall PASS (every sub-gate passes)
  1 — overall FAIL (at least one sub-gate failed)
  2 — input error (missing file, missing column, bad parse, etc.)

Pass criteria (§Phase 0 Spike 0.4):
  - V1-lock force balance: F_friction_cumulative ≥ 2 × F_inertial_at_click
    (H6 lock; H8 lever-arm lock: L_wrist_to_tip = 0.25 m).
  - Clearance: every measurement within [0.15, 0.20] mm per mating surface.
  - Engagement force (low regime): every trial within [0.5, 2.0] N.
  - Cycle life: no fracture across 1000 deploy/fold cycles.
  - High-amplitude segment: 100 cycles at 1-4 N completed without fracture.
  - Alignment: gap variation < 1 mm across adjacent blade tips.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

from fanopt.physical.click_rig import (
    ALIGNMENT_GAP_MAX_MM,
    ALPHA_MAX_RAD_PER_S2,
    CLEARANCE_MAX_MM,
    CLEARANCE_MIN_MM,
    CYCLE_TARGET,
    ENGAGEMENT_FORCE_MAX_N,
    ENGAGEMENT_FORCE_MIN_N,
    FORCE_BALANCE_SAFETY_FACTOR,
    FORCE_BALANCE_SAFETY_FACTOR_ANALYTIC,
    HIGH_AMP_CYCLE_TARGET,
    HIGH_AMP_FORCE_MAX_N,
    HIGH_AMP_FORCE_MIN_N,
    L_WRIST_TO_TIP_M,
    Spike04Result,
    analyze_clearance,
    analyze_cycle_life,
    analyze_engagement_force,
    analyze_force_balance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "spike_0_4"
DEFAULT_FORCE_BALANCE = DEFAULT_DATA_DIR / "force_balance.csv"
DEFAULT_CLEARANCE = DEFAULT_DATA_DIR / "clearance.csv"
DEFAULT_ENGAGEMENT_FORCE = DEFAULT_DATA_DIR / "engagement_force.csv"
DEFAULT_CYCLE_INSPECTIONS = DEFAULT_DATA_DIR / "cycle_inspections.csv"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "results.json"


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


def _require_columns(path: Path, row: dict, required: tuple[str, ...], row_idx: int) -> None:
    missing = [c for c in required if c not in row]
    if missing:
        raise ValueError(
            f"{path} row {row_idx}: missing column(s) {missing} " f"(found: {list(row.keys())})"
        )


def _parse_float(path: Path, row_idx: int, col: str, raw: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{path} row {row_idx}: {col} is not a float: {raw!r}") from e


def _parse_int(path: Path, row_idx: int, col: str, raw: str) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{path} row {row_idx}: {col} is not an int: {raw!r}") from e


def _parse_bool(path: Path, row_idx: int, col: str, raw: str) -> bool:
    s = (raw or "").strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f", ""}:
        return False
    raise ValueError(f"{path} row {row_idx}: {col} is not a bool: {raw!r}")


def _read_force_balance(path: Path) -> tuple[float, float, dict]:
    """Return (I_wrist_kgm2, F_friction_cumulative_N, full_row)."""
    with path.open(newline="") as f:
        reader = csv.DictReader(_decomment(f))
        row = next(iter(reader), None)
    if row is None:
        raise ValueError(f"{path}: no force-balance rows found")
    _require_columns(path, row, ("I_wrist_kgm2", "F_friction_cumulative_N"), 1)
    I_wrist = _parse_float(path, 1, "I_wrist_kgm2", row["I_wrist_kgm2"])
    F_fric = _parse_float(path, 1, "F_friction_cumulative_N", row["F_friction_cumulative_N"])
    return I_wrist, F_fric, row


def _read_clearance(path: Path) -> tuple[tuple[float, ...], tuple[str, ...], list[dict]]:
    """Return (clearances_mm, labels, full_rows)."""
    with path.open(newline="") as f:
        reader = csv.DictReader(_decomment(f))
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no clearance rows found")
    clearances: list[float] = []
    labels: list[str] = []
    for i, row in enumerate(rows, start=1):
        _require_columns(path, row, ("mating_surface", "clearance_mm"), i)
        labels.append(str(row["mating_surface"]).strip() or f"surface_{i}")
        clearances.append(_parse_float(path, i, "clearance_mm", row["clearance_mm"]))
    return tuple(clearances), tuple(labels), rows


def _read_engagement_force(
    path: Path,
) -> tuple[list[float], list[float], list[dict]]:
    """Return (low_regime_forces_N, high_regime_forces_N, full_rows)."""
    with path.open(newline="") as f:
        reader = csv.DictReader(_decomment(f))
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no engagement-force rows found")
    low: list[float] = []
    high: list[float] = []
    for i, row in enumerate(rows, start=1):
        _require_columns(path, row, ("force_N", "regime"), i)
        f_n = _parse_float(path, i, "force_N", row["force_N"])
        regime = (row["regime"] or "").strip().lower()
        if regime == "low":
            low.append(f_n)
        elif regime == "high":
            high.append(f_n)
        else:
            raise ValueError(
                f"{path} row {i}: regime must be 'low' or 'high', got {row['regime']!r}"
            )
    if not low:
        raise ValueError(f"{path}: no rows with regime='low' (canonical 0.5-2 N band required)")
    return low, high, rows


def _read_cycle_inspections(path: Path) -> list[dict]:
    """Return list of dicts suitable for analyze_cycle_life."""
    with path.open(newline="") as f:
        reader = csv.DictReader(_decomment(f))
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no cycle-inspection rows found")
    out: list[dict] = []
    for i, row in enumerate(rows, start=1):
        _require_columns(path, row, ("cycle", "wear_observed", "fracture"), i)
        out.append(
            {
                "cycle": _parse_int(path, i, "cycle", row["cycle"]),
                "wear_observed": _parse_bool(path, i, "wear_observed", row["wear_observed"]),
                "fracture": _parse_bool(path, i, "fracture", row["fracture"]),
                "notes": row.get("notes", ""),
            }
        )
    return out


# ─────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-balance",
        type=Path,
        default=DEFAULT_FORCE_BALANCE,
        help=(
            "V1-lock force-balance CSV (default: %(default)s). "
            "Required UNLESS --i-wrist-analytic is supplied with "
            "--f-friction-cumulative-n, in which case the CSV is skipped "
            "entirely and the analytic value is used instead."
        ),
    )
    parser.add_argument(
        "--i-wrist-analytic",
        type=float,
        default=None,
        help=(
            "Bypass the Spike-0.2 measured I_wrist and use the analytic "
            "value emitted by the §9.7 generator (kg·m²). When supplied, "
            "the canonical 2x safety factor is replaced by 3x to absorb "
            "the unverified-inertia uncertainty. Use this path when "
            "Spike 0.2 is deferred to V2 (see data/spike_0_2/deferral.json)."
        ),
    )
    parser.add_argument(
        "--f-friction-cumulative-n",
        type=float,
        default=None,
        help=(
            "Bench-measured cumulative click friction (N), used together "
            "with --i-wrist-analytic to skip reading --force-balance CSV. "
            "The friction measurement is still required from the bench — "
            "only the I_wrist component is substituted."
        ),
    )
    parser.add_argument(
        "--clearance",
        type=Path,
        default=DEFAULT_CLEARANCE,
        help="As-printed clearance CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--engagement-force",
        type=Path,
        default=DEFAULT_ENGAGEMENT_FORCE,
        help="Click engagement-force CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--cycle-inspections",
        type=Path,
        default=DEFAULT_CYCLE_INSPECTIONS,
        help="1000-cycle inspection CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--alignment-gap-variation-mm",
        type=float,
        required=True,
        help="Worst-case gap variation across deployed blade tips, mm.",
    )
    parser.add_argument(
        "--high-amp-completed",
        action="store_true",
        help="Set if the 100-cycle high-amplitude stress segment completed.",
    )
    parser.add_argument(
        "--high-amp-failure-cycle",
        type=int,
        default=None,
        help=(
            "Cycle index within the high-amplitude segment at which the "
            "detent fractured. Omit if no fracture."
        ),
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

    # Validate the analytic-I_wrist / measured-friction pairing.
    using_analytic = args.i_wrist_analytic is not None
    if using_analytic and args.f_friction_cumulative_n is None:
        print(
            "[spike_0_4] --i-wrist-analytic requires --f-friction-cumulative-n "
            "(the friction measurement still comes from the bench; only the "
            "I_wrist component is substituted).",
            file=sys.stderr,
        )
        return 2
    if args.f_friction_cumulative_n is not None and not using_analytic:
        print(
            "[spike_0_4] --f-friction-cumulative-n requires --i-wrist-analytic; "
            "if you have a measured I_wrist, supply it via the --force-balance CSV.",
            file=sys.stderr,
        )
        return 2

    try:
        if using_analytic:
            if args.i_wrist_analytic <= 0:
                print(
                    f"[spike_0_4] --i-wrist-analytic must be > 0, got " f"{args.i_wrist_analytic}",
                    file=sys.stderr,
                )
                return 2
            I_wrist = float(args.i_wrist_analytic)
            F_fric = float(args.f_friction_cumulative_n)
            fb_row: dict = {
                "source": "analytic",
                "I_wrist_kgm2": str(I_wrist),
                "F_friction_cumulative_N": str(F_fric),
                "notes": (
                    "Spike-0.2 deferred; I_wrist from §9.7 generator analytic "
                    "emission. Safety factor bumped 2x -> 3x."
                ),
            }
        else:
            I_wrist, F_fric, fb_row = _read_force_balance(args.force_balance)
        clearances, labels, clearance_rows = _read_clearance(args.clearance)
        low_forces, high_forces, ef_rows = _read_engagement_force(args.engagement_force)
        inspections = _read_cycle_inspections(args.cycle_inspections)
    except (FileNotFoundError, ValueError) as e:
        print(f"[spike_0_4] input error: {e}", file=sys.stderr)
        return 2

    # Pick the safety factor based on whether I_wrist is measured or analytic.
    safety_factor = (
        FORCE_BALANCE_SAFETY_FACTOR_ANALYTIC if using_analytic else FORCE_BALANCE_SAFETY_FACTOR
    )

    try:
        force_balance = analyze_force_balance(
            I_wrist_kgm2=I_wrist,
            F_friction_cumulative_N=F_fric,
            safety_factor=safety_factor,
        )
        clearance = analyze_clearance(clearances, labels=labels)
        engagement_force = analyze_engagement_force(low_forces, high_amplitude=False)
        high_amp_ef = (
            analyze_engagement_force(high_forces, high_amplitude=True) if high_forces else None
        )
        cycle_life = analyze_cycle_life(
            inspections=inspections,
            alignment_gap_variation_mm=args.alignment_gap_variation_mm,
            high_amp_completed=args.high_amp_completed,
            high_amp_failure_cycle=args.high_amp_failure_cycle,
        )
    except ValueError as e:
        print(f"[spike_0_4] analysis error: {e}", file=sys.stderr)
        return 2

    overall = (
        force_balance.passed
        and clearance.passed
        and engagement_force.passed
        and cycle_life.passed
        and (high_amp_ef is None or high_amp_ef.passed)
    )

    result = Spike04Result(
        force_balance=force_balance,
        clearance=clearance,
        engagement_force=engagement_force,
        cycle_life=cycle_life,
        overall_passed=overall,
        v1_lock_fallback_armed=force_balance.v1_lock_fallback_armed,
        high_amp_engagement_force=high_amp_ef,
    )

    payload = {
        "spec_reference": "docs/plan_R11.md §Phase 0 Spike 0.4",
        "inputs": {
            "force_balance_row": fb_row,
            "clearance_rows": clearance_rows,
            "engagement_force_rows": ef_rows,
            "cycle_inspection_rows": inspections,
            "alignment_gap_variation_mm": args.alignment_gap_variation_mm,
            "high_amp_completed": bool(args.high_amp_completed),
            "high_amp_failure_cycle": args.high_amp_failure_cycle,
        },
        "gates": {
            "alpha_max_rad_per_s2": ALPHA_MAX_RAD_PER_S2,
            "L_wrist_to_tip_m": L_WRIST_TO_TIP_M,
            "force_balance_safety_factor": safety_factor,
            "force_balance_safety_factor_canonical": FORCE_BALANCE_SAFETY_FACTOR,
            "force_balance_safety_factor_analytic": FORCE_BALANCE_SAFETY_FACTOR_ANALYTIC,
            "i_wrist_source": "analytic" if using_analytic else "measured",
            "clearance_band_mm": [CLEARANCE_MIN_MM, CLEARANCE_MAX_MM],
            "engagement_force_band_N_low": [
                ENGAGEMENT_FORCE_MIN_N,
                ENGAGEMENT_FORCE_MAX_N,
            ],
            "engagement_force_band_N_high": [
                HIGH_AMP_FORCE_MIN_N,
                HIGH_AMP_FORCE_MAX_N,
            ],
            "cycle_target": CYCLE_TARGET,
            "high_amp_cycle_target": HIGH_AMP_CYCLE_TARGET,
            "alignment_gap_max_mm": ALIGNMENT_GAP_MAX_MM,
        },
        "result": asdict(result),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    _print_table(payload, args.out)
    return 0 if overall else 1


def _mark(ok: bool) -> str:
    return "✓" if ok else "✗"


def _print_table(payload: dict, out_path: Path) -> None:
    r = payload["result"]
    fb = r["force_balance"]
    cl = r["clearance"]
    ef = r["engagement_force"]
    cy = r["cycle_life"]
    haf = r.get("high_amp_engagement_force")

    print(f"\n[spike_0_4] I_wrist                = " f"{fb['I_wrist_kgm2']:.6e} kg·m²")
    print(
        f"[spike_0_4] τ_inertial_peak        = "
        f"{fb['tau_inertial_peak_Nm']:.4f} N·m  "
        f"(α_max = {ALPHA_MAX_RAD_PER_S2} rad/s²)"
    )
    print(
        f"[spike_0_4] F_inertial_at_click    = "
        f"{fb['F_inertial_at_click_N']:.4f} N  "
        f"(L_wrist_to_tip = {L_WRIST_TO_TIP_M} m, H8 lock)"
    )
    print(
        f"[spike_0_4] F_friction_cumulative  = "
        f"{fb['F_friction_cumulative_N']:.4f} N  "
        f"(required ≥ {fb['required_friction_N']:.4f} N)  "
        f"{_mark(fb['passed'])}"
    )
    print(f"[spike_0_4] V1 fallback armed      = " f"{fb['v1_lock_fallback_armed']}")
    print(
        f"[spike_0_4] clearance              = "
        f"{cl['min_mm']:.3f}-{cl['max_mm']:.3f} mm "
        f"(band [{CLEARANCE_MIN_MM}, {CLEARANCE_MAX_MM}])  "
        f"out-of-band: {cl['out_of_band_count']}/{cl['n_measurements']}  "
        f"{_mark(cl['passed'])}"
    )
    print(
        f"[spike_0_4] engagement (low)       = "
        f"mean {ef['mean_N']:.3f} N (std {ef['std_N']:.3f}), "
        f"band [{ENGAGEMENT_FORCE_MIN_N}, {ENGAGEMENT_FORCE_MAX_N}]  "
        f"out-of-band: {ef['out_of_band_count']}/{ef['n_trials']}  "
        f"{_mark(ef['passed'])}"
    )
    if haf is not None:
        print(
            f"[spike_0_4] engagement (high)      = "
            f"mean {haf['mean_N']:.3f} N (std {haf['std_N']:.3f}), "
            f"band [{HIGH_AMP_FORCE_MIN_N}, {HIGH_AMP_FORCE_MAX_N}]  "
            f"out-of-band: {haf['out_of_band_count']}/{haf['n_trials']}  "
            f"{_mark(haf['passed'])}"
        )
    print(
        f"[spike_0_4] cycle life (low-amp)   = "
        f"{cy['total_cycles_completed']}/{CYCLE_TARGET} cycles, "
        f"first_fracture={cy['first_fracture_cycle']}  "
        f"{_mark(cy['low_amp_passed'])}"
    )
    print(
        f"[spike_0_4] high-amp segment       = "
        f"completed={cy['high_amp_completed']}, "
        f"fracture_cycle={cy['high_amp_failure_cycle']}  "
        f"{_mark(cy['high_amp_passed'])}"
    )
    print(
        f"[spike_0_4] alignment-gap variation= "
        f"{cy['alignment_gap_variation_mm']:.3f} mm  "
        f"(gate < {ALIGNMENT_GAP_MAX_MM} mm)  "
        f"{_mark(cy['alignment_passed'])}"
    )
    overall = "PASS" if r["overall_passed"] else "FAIL"
    print(f"[spike_0_4] {overall}")
    print(f"[spike_0_4] wrote                  {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
