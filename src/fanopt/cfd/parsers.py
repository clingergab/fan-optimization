"""SU2 output parsers feeding the canonical J_fan post-processor.

Thin file-I/O boundary (per CLAUDE.md §4.5): these functions read SU2 output
files and hand validated numpy arrays / :class:`~fanopt.cfd.j_fan.SteadyRun`
objects to ``fanopt.cfd.j_fan``. All numerics live in ``j_fan``.

- ``parse_su2_history_thrust`` — converged surface force projected on t̂ (+ẑ)
  from a steady SU2 ``history.csv`` (feeds the steady two-eval proxy).
- ``steady_run_from_history`` — convenience wrapper returning a ``SteadyRun``.
- ``parse_su2_plane_flow_csv`` — plane velocity field + area weights from an
  SU2 ``SURFACE_FLOW`` export on the analysis-plane marker (feeds the unsteady
  metric).
- ``plane_flux_series_from_csvs`` — a time-ordered list of plane exports →
  the per-time-step instantaneous flux series ``reduce_cycles`` consumes.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from fanopt.cfd.j_fan import (
    THRUST_DIR,
    SteadyRun,
    plane_flux_from_velocity,
)

__all__ = [
    "parse_su2_history_thrust",
    "steady_run_from_history",
    "parse_su2_plane_flow_csv",
    "plane_flux_series_from_csvs",
]

# SU2 column-name candidates (names vary across builds/versions).
_THRUST_Z_CANDIDATES = ("CFz", "CZ", "Fz", "Aero_CFz", "CForce_z")
_VEL_CANDIDATES = {
    "x": ("Velocity_x", "Velocity_0", "Vel_x", "u"),
    "y": ("Velocity_y", "Velocity_1", "Vel_y", "v"),
    "z": ("Velocity_z", "Velocity_2", "Vel_z", "w"),
}
_AREA_CANDIDATES = ("Area", "Cell_Area", "dA", "Weight")


def _detect_column(header: Sequence[str], candidates: Iterable[str]) -> str | None:
    """First header entry equal (case-insensitively) to a candidate."""
    lower = {h.strip().strip('"').lower(): h for h in header}
    for cand in candidates:
        hit = lower.get(cand.lower())
        if hit is not None:
            return hit
    return None


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path}: empty CSV")
        header = [h.strip().strip('"') for h in header]
        rows = [r for r in reader if r]
    return header, rows


def parse_su2_history_thrust(
    path: str | Path,
    *,
    thrust_candidates: Iterable[str] = _THRUST_Z_CANDIDATES,
) -> float:
    """Converged surface force projected on t̂ (+ẑ) from a steady history.csv.

    Returns the value in the last (converged) row of the z-force column. Raises
    ``ValueError`` if no recognized z-force column is present or the file is
    empty of data rows.
    """
    header, rows = _read_csv(Path(path))
    col = _detect_column(header, thrust_candidates)
    if col is None:
        raise ValueError(
            f"{path}: no recognized t̂ (z) force column; looked for "
            f"{tuple(thrust_candidates)}; found {header}"
        )
    if not rows:
        raise ValueError(f"{path}: history has a header but no data rows")
    idx = header.index(col)
    last = rows[-1]
    try:
        return float(last[idx])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"{path}: malformed final row for column {col!r}: {last}") from exc


def steady_run_from_history(
    path: str | Path,
    *,
    stroke: str,
    design_hash: str = "",
    thrust_candidates: Iterable[str] = _THRUST_Z_CANDIDATES,
) -> SteadyRun:
    """Build a :class:`SteadyRun` from a steady SU2 history.csv."""
    thrust = parse_su2_history_thrust(path, thrust_candidates=thrust_candidates)
    return SteadyRun(thrust=thrust, stroke=stroke, design_hash=design_hash)


def parse_su2_plane_flow_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Plane velocity field ``(N, 3)`` + per-sample area weights ``(N,)``.

    Reads an SU2 ``SURFACE_FLOW`` CSV export on the analysis-plane marker. Needs
    three velocity columns and an area/weight column (names auto-detected).
    """
    header, rows = _read_csv(Path(path))
    vel_cols = {axis: _detect_column(header, cands) for axis, cands in _VEL_CANDIDATES.items()}
    missing = [axis for axis, c in vel_cols.items() if c is None]
    if missing:
        raise ValueError(f"{path}: missing velocity column(s) for axes {missing}; found {header}")
    area_col = _detect_column(header, _AREA_CANDIDATES)
    if area_col is None:
        raise ValueError(
            f"{path}: no recognized area/weight column; looked for "
            f"{_AREA_CANDIDATES}; found {header}"
        )
    if not rows:
        raise ValueError(f"{path}: plane export has a header but no data rows")

    vx = header.index(vel_cols["x"])  # type: ignore[arg-type]
    vy = header.index(vel_cols["y"])  # type: ignore[arg-type]
    vz = header.index(vel_cols["z"])  # type: ignore[arg-type]
    ia = header.index(area_col)
    try:
        velocity = np.array([[float(r[vx]), float(r[vy]), float(r[vz])] for r in rows], dtype=float)
        area = np.array([float(r[ia]) for r in rows], dtype=float)
    except (ValueError, IndexError) as exc:
        raise ValueError(f"{path}: malformed numeric data: {exc}") from exc
    return velocity, area


def plane_flux_series_from_csvs(
    paths: Sequence[str | Path],
    *,
    n_hat: Sequence[float] = THRUST_DIR,
    t_hat: Sequence[float] = THRUST_DIR,
) -> np.ndarray:
    """Per-time-step instantaneous flux series from time-ordered plane exports.

    ``paths`` must be in ascending time-step order (one plane export per SU2
    outer time step). Returns a 1D array suitable for
    :func:`fanopt.cfd.j_fan.reduce_cycles`.
    """
    if not paths:
        raise ValueError("plane_flux_series_from_csvs requires at least one path")
    series = []
    for p in paths:
        velocity, area = parse_su2_plane_flow_csv(p)
        series.append(plane_flux_from_velocity(velocity, area, n_hat=n_hat, t_hat=t_hat))
    return np.asarray(series, dtype=float)


_UNSTEADY_FORCE_CANDIDATES = ("CFx", "CD", "CFz")
_TIME_ITER_CANDIDATES = ("Time_Iter", "Time_Iter ", "TimeIter", "Cur_Time")


def parse_su2_unsteady_force_series(
    path: str | Path, *, force_candidates: Iterable[str] = _UNSTEADY_FORCE_CANDIDATES
) -> np.ndarray:
    """Per-outer-time-step force from an unsteady SU2 history.csv.

    SU2 writes one row per (Time_Iter, Inner_Iter); we keep the LAST inner-iter
    value at each outer step (the converged sub-iteration). Returns a 1D array
    of length = number of outer time steps — feed it to
    :func:`fanopt.cfd.j_fan.reduce_cycles` to get the cycle-averaged J_fan.
    """
    header, rows = _read_csv(Path(path))
    fcol = _detect_column(header, force_candidates)
    if fcol is None:
        raise ValueError(
            f"{path}: no recognized unsteady force column; looked for "
            f"{tuple(force_candidates)}; found {header}"
        )
    tcol = _detect_column(header, _TIME_ITER_CANDIDATES)
    if tcol is None:
        raise ValueError(f"{path}: no Time_Iter column; found {header}")
    if not rows:
        raise ValueError(f"{path}: unsteady history has no data rows")
    fi, ti = header.index(fcol), header.index(tcol)
    by_step: dict[int, float] = {}
    try:
        for r in rows:
            by_step[int(float(r[ti]))] = float(r[fi])  # last write per outer step wins
    except (ValueError, IndexError) as exc:
        raise ValueError(f"{path}: malformed unsteady row: {exc}") from exc
    return np.array([by_step[k] for k in sorted(by_step)], dtype=float)
