#!/usr/bin/env python
"""Spike 0.7c CLI -- Sobol-vs-BO iso-compute comparison.

Spec reference: ``docs/plan_R11.md §Phase 0 Spike 0.7c``; protocol in
``docs/spike_0_7c_protocol.md``.

What this script does:

1. Reads two JSONL ledgers -- ``--sobol-results`` and ``--bo-results``.
   Each line is one evaluation record with at least ``j_fan`` and
   ``wall_time_hours``.
2. For each budget in ``--budgets`` (default 30,100,300), computes one
   ``IsoComputePoint`` via ``fanopt.bo.spike_0_7c.compute_iso_compute_point``.
3. Aggregates via ``analyze_spike_07c`` and writes ``results.json``.
4. Prints a per-budget comparison table marking each BO-beats outcome
   with a check (PASS) or cross (FAIL).
5. Exits 0 on PASS, 1 on FAIL, 2 on input error.

This script does NOT run any CFD or any BO: it is the *analysis* end of
the harness. The CFD / BO outer loop produces the JSONL ledgers
externally (Phase 4 production runs).

Use ``scripts/run_spike_0_7c_smoke.py`` for an end-to-end smoke test
on a synthetic objective without any external dependencies.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fanopt.bo.spike_0_7c import (
    BO_MINUS_SOBOL_PCT_GATE,
    BUDGETS_HOURS,
    BUDGETS_PASS_THRESHOLD,
    BUDGETS_TOTAL_HOURS,
    analyze_spike_07c,
    compute_iso_compute_point,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "spike_0_7c" / "results.json"


# ---- IO helpers ------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts. Blank lines are skipped."""
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"[spike_0_7c] {path}:{lineno}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(rec, dict):
                raise SystemExit(
                    f"[spike_0_7c] {path}:{lineno}: top-level value must be an object"
                )
            out.append(rec)
    return out


def _parse_budgets(raw: str) -> tuple[int, ...]:
    """Parse '30,100,300' into a tuple of ints. Rejects empty / non-positive."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("--budgets must list at least one positive integer")
    try:
        vals = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"--budgets must be comma-separated integers, got {raw!r}") from exc
    if any(v <= 0 for v in vals):
        raise ValueError(f"--budgets must all be positive, got {vals}")
    return vals


def _parse_labels(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(p.strip() for p in raw.split(",") if p.strip())


# ---- CLI ------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sobol-results",
        type=Path,
        required=True,
        help="JSONL with the 50 Sobol-seed evaluation records.",
    )
    p.add_argument(
        "--bo-results",
        type=Path,
        required=True,
        help="JSONL with the 100 BO-iteration evaluation records.",
    )
    p.add_argument(
        "--budgets",
        type=str,
        default=",".join(str(b) for b in BUDGETS_HOURS),
        help=(
            "Comma-separated CFD-hour budgets. Default: %(default)s "
            "(the three spec checkpoints summing to "
            f"{BUDGETS_TOTAL_HOURS} h booked under Phase 0)."
        ),
    )
    p.add_argument(
        "--gp-fit-time-above-60s-on",
        type=str,
        default="",
        help=(
            "Comma-separated sub-axis labels where the GP fit time "
            "exceeded 60 s during the underlying BO run "
            "(e.g., 'high_d' or 'wide_architecture_set'). Used only "
            "for the fallback recommendation when the spike fails."
        ),
    )
    p.add_argument(
        "--j-fan-key",
        type=str,
        default="j_fan",
        help="Key in each record holding the scalar J_fan. Default: %(default)s.",
    )
    p.add_argument(
        "--wall-time-key",
        type=str,
        default="wall_time_hours",
        help="Key in each record holding the per-record wall_time. Default: %(default)s.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Where to write results.json. Default: %(default)s.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        budgets = _parse_budgets(args.budgets)
    except ValueError as exc:
        print(f"[spike_0_7c] ERROR: {exc}", file=sys.stderr)
        return 2

    if not args.sobol_results.exists():
        print(
            f"[spike_0_7c] ERROR: --sobol-results not found: {args.sobol_results}",
            file=sys.stderr,
        )
        return 2
    if not args.bo_results.exists():
        print(
            f"[spike_0_7c] ERROR: --bo-results not found: {args.bo_results}",
            file=sys.stderr,
        )
        return 2

    sobol_records = _load_jsonl(args.sobol_results)
    bo_records = _load_jsonl(args.bo_results)

    if not sobol_records:
        print(
            f"[spike_0_7c] ERROR: no Sobol records in {args.sobol_results}",
            file=sys.stderr,
        )
        return 2
    if not bo_records:
        print(
            f"[spike_0_7c] ERROR: no BO records in {args.bo_results}",
            file=sys.stderr,
        )
        return 2

    iso_points = tuple(
        compute_iso_compute_point(
            b,
            sobol_records,
            bo_records,
            j_fan_key=args.j_fan_key,
            wall_time_key=args.wall_time_key,
        )
        for b in budgets
    )

    labels = _parse_labels(args.gp_fit_time_above_60s_on)
    result = analyze_spike_07c(iso_points, gp_fit_time_above_60s_on=labels)

    payload: dict[str, Any] = {
        "spec_reference": "docs/plan_R11.md §Phase 0 Spike 0.7c",
        "inputs": {
            "sobol_results": str(args.sobol_results),
            "bo_results": str(args.bo_results),
            "budgets_hours": list(budgets),
            "gp_fit_time_above_60s_on": list(labels),
            "j_fan_key": args.j_fan_key,
            "wall_time_key": args.wall_time_key,
            "bo_minus_sobol_pct_gate": BO_MINUS_SOBOL_PCT_GATE,
            "budgets_pass_threshold": BUDGETS_PASS_THRESHOLD,
        },
        "per_budget": [asdict(p) for p in result.per_budget],
        "n_budgets_bo_beats": result.n_budgets_bo_beats,
        "passed": result.passed,
        "fallback_recommendation": result.fallback_recommendation,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    _print_summary(payload, args.out)
    return 0 if result.passed else 1


# ---- table printing -------------------------------------------------------


def _print_summary(payload: dict[str, Any], out_path: Path) -> None:
    gate = payload["inputs"]["bo_minus_sobol_pct_gate"]
    threshold = payload["inputs"]["budgets_pass_threshold"]
    print("[spike_0_7c] -----------------------------------------------------")
    print(
        f"[spike_0_7c] Iso-compute comparison: BO must beat Sobol by "
        f">= {gate:.1f}% on >= {threshold} of {len(payload['per_budget'])} budgets."
    )
    header = (
        f"  {'Budget(h)':>10}  {'n_sobol':>8}  {'n_bo':>5}  "
        f"{'sobol_best':>12}  {'bo_best':>12}  {'BO-Sobol%':>10}  Beats?"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for p in payload["per_budget"]:
        mark = "PASS" if p["bo_beats"] else "FAIL"
        sc = p["sample_count"]
        print(
            f"  {p['budget_hours']:>10d}  "
            f"{sc.get('sobol', 0):>8d}  "
            f"{sc.get('bo', 0):>5d}  "
            f"{p['sobol_best_j_fan']:>12.6g}  "
            f"{p['bo_best_j_fan']:>12.6g}  "
            f"{p['bo_minus_sobol_pct']:>+10.2f}  "
            f"{mark}"
        )
    print(
        f"[spike_0_7c] BO beats Sobol on "
        f"{payload['n_budgets_bo_beats']} / {len(payload['per_budget'])} budgets."
    )
    overall = "PASS" if payload["passed"] else "FAIL"
    print(f"[spike_0_7c] OVERALL: {overall}  -> {out_path}")
    if payload["fallback_recommendation"]:
        print(
            f"[spike_0_7c] Fallback recommendation: {payload['fallback_recommendation']}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
