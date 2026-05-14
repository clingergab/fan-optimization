#!/usr/bin/env python
"""Parse a SU2 history.csv from the pitching-airfoil benchmark into the
per-cycle measured.csv that ``scripts/run_spike_0_6c_2.py`` consumes.

**Input** (SU2 history.csv from the Tier-1 unsteady benchmark run):

  Columns include at minimum:
    Time_Iter, Inner_Iter, Cur_Time, CL, CD, CMz  (or equivalents)

  Each row is one outer time step (or one inner iter per outer step,
  depending on the SU2 build). For unsteady runs we want the converged
  inner-iter value per outer step, which is the last row at each
  Time_Iter index.

**Output** (one row per pitching cycle):

  cycle_index,c_l_max,c_l_min,c_d_mean,c_l_hysteresis_area
  0,...
  1,...
  ...

The downstream ``run_spike_0_6c_2.py`` analyzer discards cycle 0
(initial transient) and averages cycles 1-4 for the ±15% gate.

**Cycle slicing.** The benchmark renders TIME_STEP = T/200 and TIME_ITER
= n_cycles * 200, so each cycle is exactly 200 outer time steps. We
split the history into `n_cycles` equal slices of 200 rows each.

**Hysteresis area.** C_l(α) traces a closed loop over one cycle. The
signed area is computed via the shoelace formula on (alpha, C_l). The
sign indicates leading vs. lagging stall.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.6c.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]


def _detect_column(header: list[str], candidates: Iterable[str]) -> str | None:
    """Return the first column from `candidates` that appears in `header`
    (case-insensitive substring match). SU2 history column names vary
    across versions ("CL", "CL ", "C_L", "AERO_COEFF_CL", etc.).
    """
    lower = [h.strip().strip('"').lower() for h in header]
    for cand in candidates:
        c = cand.lower()
        for i, h in enumerate(lower):
            if c == h or c == h.strip():
                return header[i]
    return None


def _read_history(path: Path) -> tuple[list[dict[str, float]], dict[str, str]]:
    """Read SU2 history.csv. Returns (rows, column_map).

    rows is a list of dicts keyed by the *canonical* names
    {time_iter, time, cl, cd, alpha}. column_map records which SU2
    column was matched to each canonical name (for the report).
    """
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path}: empty history file")
        header = [h.strip().strip('"') for h in header]

        time_iter_col = _detect_column(header, ("Time_Iter", "Inner_Iter", "Iter"))
        time_col = _detect_column(header, ("Cur_Time", "Time", "Time(s)"))
        cl_col = _detect_column(header, ("CL", "C_L", "CLift", "Aero_CL"))
        cd_col = _detect_column(header, ("CD", "C_D", "CDrag", "Aero_CD"))
        # alpha-of-attack column is optional; if missing we'll reconstruct
        # from the pitching kinematics.
        alpha_col = _detect_column(header, ("AoA", "Alpha", "PITCH_ANGLE"))

        if time_iter_col is None:
            raise ValueError(
                f"{path}: no recognized time-iter column "
                f"(looked for Time_Iter / Inner_Iter / Iter). "
                f"Found columns: {header}"
            )
        if cl_col is None or cd_col is None:
            raise ValueError(
                f"{path}: no recognized lift/drag columns "
                f"(looked for CL/CD). Found columns: {header}"
            )

        rows: list[dict[str, float]] = []
        idx = {col: header.index(col) for col in header}
        for line_idx, raw in enumerate(reader, start=2):
            if not raw:
                continue
            try:
                row = {
                    "time_iter": float(raw[idx[time_iter_col]]),
                    "cl": float(raw[idx[cl_col]]),
                    "cd": float(raw[idx[cd_col]]),
                }
                if time_col is not None:
                    row["time"] = float(raw[idx[time_col]])
                if alpha_col is not None:
                    row["alpha"] = float(raw[idx[alpha_col]])
                rows.append(row)
            except (ValueError, IndexError) as e:
                raise ValueError(
                    f"{path}:{line_idx}: malformed row: {e}; raw={raw}"
                ) from e

    column_map = {
        "time_iter": time_iter_col,
        "time": time_col or "",
        "cl": cl_col,
        "cd": cd_col,
        "alpha": alpha_col or "",
    }
    return rows, column_map


def _per_outer_iter(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    """Collapse multiple inner-iter rows into one row per outer time-step.

    SU2 emits one history row per (Time_Iter, Inner_Iter) pair when
    OUTPUT_WRT_FREQ_INNER > 0 — we want the LAST inner-iter row at each
    time-iter index (the converged value).
    """
    by_iter: dict[int, dict[str, float]] = {}
    for r in rows:
        ti = int(r["time_iter"])
        # Last write wins — assumes file is in monotonic outer-iter order.
        by_iter[ti] = r
    return [by_iter[k] for k in sorted(by_iter)]


def _split_cycles(rows: list[dict[str, float]], n_cycles: int) -> list[list[dict[str, float]]]:
    """Split the time history into n_cycles equal slices."""
    if n_cycles <= 0:
        raise ValueError(f"n_cycles must be > 0, got {n_cycles}")
    n = len(rows)
    if n < n_cycles:
        raise ValueError(
            f"only {n} outer iters in history; need ≥ {n_cycles} (one per cycle)"
        )
    per_cycle = n // n_cycles
    if per_cycle * n_cycles != n:
        # Not strictly an error — but warn since the spec locks TIME_ITER
        # = 200 · n_cycles so non-divisibility means the SU2 run was
        # truncated mid-cycle.
        print(
            f"[parse_su2_history] WARNING: {n} outer iters / {n_cycles} cycles "
            f"= {per_cycle} per cycle with {n - per_cycle * n_cycles} extras; "
            f"truncating last cycle.",
            file=sys.stderr,
        )
    return [
        rows[i * per_cycle : (i + 1) * per_cycle]
        for i in range(n_cycles)
    ]


def _hysteresis_area_shoelace(alphas: list[float], cls: list[float]) -> float:
    """Signed area enclosed by the (alpha, C_l) loop.

    The shoelace formula on a closed polygon gives twice the signed area
    of the enclosed region. We treat the per-cycle history as a closed
    loop (last point connects back to first) and report |area|.
    """
    n = len(alphas)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += alphas[i] * cls[j] - alphas[j] * cls[i]
    return abs(s) / 2.0


def cycles_from_rows(
    rows: list[dict[str, float]],
    *,
    n_cycles: int,
    theta_max_rad: float,
    omega_shm_rad_per_s: float,
) -> list[dict[str, float]]:
    """Compute per-cycle (c_l_max, c_l_min, c_d_mean, c_l_hysteresis_area).

    `theta_max_rad`, `omega_shm_rad_per_s` are needed only when the
    history file lacks an explicit alpha column — we reconstruct
    `alpha(t) = theta_max · sin(omega_shm · t)` from the row's `time`
    column. If neither is present, hysteresis area uses the row index
    as a proxy x-axis (still gives a consistent number across cycles).
    """
    per_iter = _per_outer_iter(rows)
    cycles = _split_cycles(per_iter, n_cycles=n_cycles)

    out: list[dict[str, float]] = []
    for i, cyc in enumerate(cycles):
        cls = [r["cl"] for r in cyc]
        cds = [r["cd"] for r in cyc]
        # Build alphas: prefer recorded alpha, fall back to reconstructed,
        # fall back to row index.
        if all("alpha" in r for r in cyc):
            alphas = [r["alpha"] for r in cyc]
        elif all("time" in r for r in cyc):
            alphas = [theta_max_rad * math.sin(omega_shm_rad_per_s * r["time"]) for r in cyc]
        else:
            alphas = [float(j) for j in range(len(cyc))]
        out.append(
            {
                "cycle_index": float(i),
                "c_l_max": max(cls),
                "c_l_min": min(cls),
                "c_d_mean": sum(cds) / len(cds),
                "c_l_hysteresis_area": _hysteresis_area_shoelace(alphas, cls),
            }
        )
    return out


def write_measured_csv(cycles: list[dict[str, float]], out_path: Path) -> Path:
    """Write the cycle aggregates to `measured.csv` in the format
    ``run_spike_0_6c_2.py --measured`` expects."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["cycle_index", "c_l_max", "c_l_min", "c_d_mean", "c_l_hysteresis_area"]
        )
        for c in cycles:
            writer.writerow(
                [
                    int(c["cycle_index"]),
                    f"{c['c_l_max']:.6f}",
                    f"{c['c_l_min']:.6f}",
                    f"{c['c_d_mean']:.6f}",
                    f"{c['c_l_hysteresis_area']:.6f}",
                ]
            )
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        type=Path,
        required=True,
        help="SU2 history.csv from the pitching-airfoil benchmark run.",
    )
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=5,
        help="Number of cycles in the run. Default 5 (Spike 0.6c.2 canonical).",
    )
    parser.add_argument(
        "--theta-max-rad",
        type=float,
        default=0.1745,  # 10° — typical NACA 0012 benchmark amplitude
        help="Pitching amplitude in radians. Default 0.1745 (= 10°).",
    )
    parser.add_argument(
        "--omega-shm-rad-per-s",
        type=float,
        required=True,
        help=(
            "SHM angular frequency in rad/s (the benchmark's "
            "PITCHING_OMEGA y-component magnitude). Needed to reconstruct "
            "alpha(t) when the history file lacks an AoA column."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output measured.csv path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows, _column_map = _read_history(args.history)
    except (FileNotFoundError, ValueError) as e:
        print(f"[parse_su2_history] input error: {e}", file=sys.stderr)
        return 2

    try:
        cycles = cycles_from_rows(
            rows,
            n_cycles=args.n_cycles,
            theta_max_rad=args.theta_max_rad,
            omega_shm_rad_per_s=abs(args.omega_shm_rad_per_s),
        )
    except ValueError as e:
        print(f"[parse_su2_history] analysis error: {e}", file=sys.stderr)
        return 2

    path = write_measured_csv(cycles, args.out)
    print(f"[parse_su2_history] wrote {len(cycles)} cycle rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
