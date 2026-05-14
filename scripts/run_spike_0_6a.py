#!/usr/bin/env python
"""Spike 0.6a — M3 SU2 Tier-1 end-to-end timing runner.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.6 sub-spike 0.6a; protocol
in docs/spike_0_6_protocol.md §06a.

Drives one Tier-1 case (CadQuery -> Gmsh 2D corrugated slice -> SU2 2D
steady -> `j_fan.py`) end-to-end on the MacBook M3 and writes a single-row
CSV at `data/spike_0_6/06a.csv` (plus per-stage breakdown rows) for the
aggregator to consume.

Pass criterion (gate for any local-M3 SU2 use):
  * wall-time <= 15 min
  * `J_fan_steady_proxy` finite

Fallback when SU2 is not installed locally: print a clear notice and exit 0
with a marker — the gate is reported as "gated-on-availability" so the
aggregator sees a missing 06a CSV rather than spurious data.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "spike_0_6" / "06a.csv"


def _su2_available() -> bool:
    return shutil.which("SU2_CFD") is not None


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the 06a CSV row (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the actual SU2 invocation; write a placeholder row.",
    )
    return parser.parse_args(argv)


def _write_row(
    out_path: Path,
    wall_time_s: float,
    J_fan_steady_proxy: float | None,
    stages: dict[str, float],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wall_time_s", "J_fan_steady_proxy", "stage_name", "stage_wall_time_s"])
        # First row carries totals; stage_name / stage_wall_time_s blank.
        writer.writerow(
            [
                f"{wall_time_s:.3f}",
                "" if J_fan_steady_proxy is None else f"{J_fan_steady_proxy:.6e}",
                "",
                "",
            ]
        )
        for name, dt in stages.items():
            writer.writerow(["", "", name, f"{dt:.3f}"])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not _su2_available() and not args.dry_run:
        # Fail loud — silent-pass under SU2-absence hides the gate from CI
        # and from local M3 viability checks. The protocol's "fallback"
        # path (shift smoke_test.py to Colab) is a downstream consequence,
        # not a license for this gate to silently green.
        print(
            "[spike_0_6a] SU2_CFD not found on PATH. The 0.6a gate requires "
            "a real SU2 invocation on the M3 to validate local viability. "
            "Install SU2 (see environment.yml notes) or run with `--dry-run` "
            "to exercise the timing harness only. Aggregator will see no "
            "06a CSV and the sub-spike will be reported as NOT RUN.",
            file=sys.stderr,
        )
        return 2

    # Stub: time a placeholder pipeline. Real implementation lands when
    # `src/fanopt/cfd/` exposes a Tier-1 entry point. For now this just
    # demonstrates the timing harness and emits a dry-run row so the
    # aggregator's input plumbing is testable.
    stages: dict[str, float] = {}

    t_start = time.perf_counter()

    t0 = time.perf_counter()
    # CadQuery panel generation — placeholder.
    stages["cadquery"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    # Gmsh 2D corrugated slice — placeholder.
    stages["gmsh_2d"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    # SU2 2D steady — placeholder. Real invocation: subprocess.run(['SU2_CFD', ...]).
    stages["su2_2d_steady"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    # j_fan.py reduction — placeholder.
    stages["j_fan"] = time.perf_counter() - t0

    wall_time_s = time.perf_counter() - t_start

    if args.dry_run:
        # Plumbing exercise only — write a placeholder finite value so the
        # aggregator's input parser is testable.
        J_fan_steady_proxy: float = 0.0
        _write_row(args.out, wall_time_s, J_fan_steady_proxy, stages)
        print(f"[spike_0_6a] wrote      {args.out}")
        print(f"[spike_0_6a] wall_time  {wall_time_s:.3f}s")
        print("[spike_0_6a] DRY-RUN — placeholder J_fan_steady_proxy = 0.0")
        return 0

    # Non-dry-run with SU2 available but no real pipeline yet: refuse to
    # write a misleading row. The pipeline (CadQuery → Gmsh → SU2 → j_fan)
    # is wired up by Phase 1; until then this gate can only be exercised
    # via --dry-run.
    print(
        "[spike_0_6a] SU2 is available but the CadQuery → Gmsh → SU2 → j_fan "
        "pipeline is not yet wired (Phase 1 dependency). Re-run with "
        "`--dry-run` to exercise the timing harness, or land the Phase 1 "
        "geometry generator first.",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
