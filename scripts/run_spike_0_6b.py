#!/usr/bin/env python
"""Spike 0.6b — M3 FEA cantilever-rib timing runner.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.6 sub-spike 0.6b; protocol
in docs/spike_0_6_protocol.md §06b.

Runs one FEniCSx (or CalculiX) static FEA case on the M3 -- a simple
cantilever rib under a 5 N tip load -- and writes a single-row CSV at
`data/spike_0_6/06b.csv` for the aggregator to consume.

Cantilever spec (PETG, locked-ish defaults):
  * E_PETG = 1300 MPa
  * cross-section: b = 12 mm, h = 2 mm (rectangular, bending about b-axis)
  * length: L = 200 mm
  * tip load: P = 5 N

Analytic reference: `delta = P L^3 / (3 E I)` with `I = b h^3 / 12`.

Pass criterion (gate for Phase 5 step 59.5 / step 64.5 local FEA):
  * wall-time <= 2 min
  * |measured - analytic| / analytic * 100 <= 5%

Fallback when FEniCSx is not installed locally: print a clear notice and
exit 0 — the gate is reported as "gated-on-availability".
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "spike_0_6" / "06b.csv"

# Locked cantilever spec — keep in sync with docs/spike_0_6_protocol.md §06b.
E_PETG_PA: float = 1.300e9
B_M: float = 0.012
H_M: float = 0.002
L_M: float = 0.200
P_N: float = 5.0


def _fenicsx_available() -> bool:
    return importlib.util.find_spec("dolfinx") is not None


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the 06b CSV row (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip the actual FEA solve; write a row using the analytic "
            "deflection as 'measured' (useful for plumbing tests)."
        ),
    )
    return parser.parse_args(argv)


def _i_rect_m4(b: float, h: float) -> float:
    return b * (h**3) / 12.0


def _analytic_tip_deflection_m(P: float, L: float, E: float, I_m4: float) -> float:
    return P * (L**3) / (3.0 * E * I_m4)


def _solve_cantilever_1d_eb() -> float:
    """Solve a 1D Euler-Bernoulli cantilever via FEniCSx.

    Beam: clamped at x=0, point load P at x=L. The 5% gate exists to verify
    that FEniCSx on the M3 produces the analytic answer within tolerance —
    that's the point of the spike (verify the local FEA toolchain works).

    **Stub:** returns the analytic value verbatim. This trivially passes
    the 5% gate and does NOT exercise FEniCSx. Real implementation must
    `import dolfinx` and build a 1D mesh + Hermite elements + solve.
    Until then the wall-time gate is also meaningless (microseconds, not
    minutes). Sub-spike 0.6b cannot validate M3 FEA viability via this
    stub — gate must remain DRY-RUN-only or land the real solver.
    """
    raise NotImplementedError(
        "0.6b real solver not implemented. The stub used to return the "
        "analytic value verbatim, which silently passed the 5% gate "
        "without exercising FEniCSx — defeating the spike. Land the real "
        "dolfinx-based cantilever solver, or run with --dry-run only."
    )


def _write_row(
    out_path: Path,
    wall_time_s: float,
    measured_tip_deflection_m: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "wall_time_s",
                "measured_tip_deflection_m",
                "P_N",
                "L_m",
                "E_Pa",
                "I_m4",
            ]
        )
        writer.writerow(
            [
                f"{wall_time_s:.3f}",
                f"{measured_tip_deflection_m:.6e}",
                f"{P_N:.3f}",
                f"{L_M:.6f}",
                f"{E_PETG_PA:.3e}",
                f"{_i_rect_m4(B_M, H_M):.6e}",
            ]
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not _fenicsx_available() and not args.dry_run:
        # Fail loud — silent-pass under FEniCSx-absence hides the gate.
        print(
            "[spike_0_6b] FEniCSx (dolfinx) not importable. The 0.6b gate "
            "requires a real FEniCSx solve on the M3 to validate local FEA "
            "viability. Install FEniCSx (conda-forge fenics-dolfinx) or "
            "run with `--dry-run` to exercise the timing harness only. "
            "Aggregator will see no 06b CSV and the sub-spike will be "
            "reported as NOT RUN.",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        # Plumbing exercise only — write the analytic value as the
        # placeholder "measured" so the aggregator's parser is testable.
        I_m4 = _i_rect_m4(B_M, H_M)
        measured = _analytic_tip_deflection_m(P_N, L_M, E_PETG_PA, I_m4)
        wall_time_s = 0.0
        _write_row(args.out, wall_time_s, measured)
        analytic = measured
        pct = 0.0
        print(f"[spike_0_6b] wrote      {args.out}")
        print("[spike_0_6b] DRY-RUN — analytic placeholder, no FEniCSx invoked.")
        print(f"[spike_0_6b] measured   {measured:.6e} m  (analytic {analytic:.6e} m, {pct:.2f}%)")
        return 0

    t_start = time.perf_counter()
    measured = _solve_cantilever_1d_eb()  # raises NotImplementedError until real solver lands
    wall_time_s = time.perf_counter() - t_start

    _write_row(args.out, wall_time_s, measured)
    analytic = _analytic_tip_deflection_m(P_N, L_M, E_PETG_PA, _i_rect_m4(B_M, H_M))
    pct = 100.0 * abs(measured - analytic) / analytic
    print(f"[spike_0_6b] wrote      {args.out}")
    print(f"[spike_0_6b] wall_time  {wall_time_s:.3f}s")
    print(f"[spike_0_6b] measured   {measured:.6e} m  (analytic {analytic:.6e} m)")
    print(f"[spike_0_6b] tip_pct    {pct:.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
