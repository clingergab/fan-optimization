"""Map a CFD surface-pressure field onto the FEA skin — the aero → structural bridge.

The productive- and return-stroke structural load cases are driven by the aerodynamic
pressure on the panel, which comes from re-running a chosen blade with SU2 surface output
on. This module reads that ``surface_flow`` CSV as **plain data** (topopt must not depend
on :mod:`fanopt.cfd`) and interpolates the pressure onto the FEA skin facet centres, so
:mod:`fanopt.topopt.loadcases` can turn it into nodal forces.

Pure numpy + scipy.spatial — no FEA dependency; the pressures are read from disk, never
invented.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["load_surface_pressure", "map_pressure_to_facets"]

_X_KEYS = ("x", "x_coord")
_Y_KEYS = ("y", "y_coord")
_Z_KEYS = ("z", "z_coord")
_PRESSURE_KEYS = ("pressure", "p")


def _clean(name: str) -> str:
    return name.strip().strip('"').strip().lower()


def _find_column(fieldnames: list[str], keys: tuple[str, ...], label: str) -> str:
    cleaned = {_clean(f): f for f in fieldnames}
    for k in keys:
        if k in cleaned:
            return cleaned[k]
    raise ValueError(f"no {label} column in {fieldnames} (looked for {keys})")


def load_surface_pressure(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse an SU2 ``surface_flow`` CSV into ``(points (n,3), pressure (n,))``.

    Column names are matched case-insensitively (SU2 quotes its headers), so ``"x"`` /
    ``"Pressure"`` resolve regardless of quoting. Raises if the coordinate or pressure
    columns are absent — a malformed harvest should fail loudly, not silently zero-load.
    """
    with Path(csv_path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"empty CSV: {csv_path}")
        xc = _find_column(reader.fieldnames, _X_KEYS, "x")
        yc = _find_column(reader.fieldnames, _Y_KEYS, "y")
        zc = _find_column(reader.fieldnames, _Z_KEYS, "z")
        pc = _find_column(reader.fieldnames, _PRESSURE_KEYS, "pressure")
        pts, press = [], []
        for row in reader:
            pts.append((float(row[xc]), float(row[yc]), float(row[zc])))
            press.append(float(row[pc]))
    return np.asarray(pts, dtype=float), np.asarray(press, dtype=float)


def map_pressure_to_facets(
    surface_points: np.ndarray,
    surface_pressure: np.ndarray,
    facet_centers: np.ndarray,
    *,
    k: int = 1,
    eps: float = 1e-12,
) -> np.ndarray:
    """Interpolate CFD pressure onto FEA facet centres (nearest, or ``k``-NN inverse-distance).

    ``k = 1`` takes the nearest CFD sample (piecewise-constant); ``k > 1`` inverse-distance-
    weights the ``k`` nearest, smoothing sampling noise. ``eps`` guards a facet coincident
    with a sample point (zero distance → that sample's value verbatim).
    """
    tree = cKDTree(surface_points)
    dist, idx = tree.query(facet_centers, k=k)
    if k == 1:
        return surface_pressure[idx]
    weights = 1.0 / (dist + eps)
    return (weights * surface_pressure[idx]).sum(axis=1) / weights.sum(axis=1)
