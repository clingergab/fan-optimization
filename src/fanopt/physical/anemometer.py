"""Physical J_fan_proxy from a 9-point anemometer grid.

Implements the L8 lock at `docs/plan_R11.md §Phase 6 step 78` (also reused
in Spike 0.3): the handheld anemometer measures point velocity, so the
plane integral over the §9.4 600×600 mm plane at 300 mm is approximated by
a 3×3 grid (200 mm pitch) and multiplied by the plane area.

  J_fan_proxy ≈ ρ_air · ⟨v⟩_grid · A_plane

This is intentionally coarse — the 3×3 grid undersamples the wake — but it
bounds the variance of a single-point reading and matches the §9.4 plane
convention better than reading one point.

Reference:
- Spec: `docs/plan_R11.md §Phase 6 step 78` (L8 lock)
- Protocol: `docs/spike_0_3_protocol.md`
- Plane definition: §9.4 (600×600 mm at z = +300 mm above pivot)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "RHO_AIR_KG_PER_M3",
    "PLANE_HALF_EXTENT_M",
    "PLANE_AREA_M2",
    "PLANE_Z_M",
    "GRID_POINTS_M",
    "AnemometerGrid",
    "AnemometerResult",
    "load_anemometer_csv",
    "j_fan_proxy_from_grid",
    "analyze_anemometer_grid",
]


# §9.4 plane geometry (L8 lock).
RHO_AIR_KG_PER_M3: float = 1.225
PLANE_HALF_EXTENT_M: float = 0.300  # 600 mm × 600 mm plane → ±300 mm
PLANE_AREA_M2: float = (2.0 * PLANE_HALF_EXTENT_M) ** 2  # 0.36 m²
PLANE_Z_M: float = 0.300  # 300 mm in +z from pivot

# 3×3 grid points at (x, y) = ±200, 0 mm relative to pivot. Same row-major
# raster order used in the protocol/log templates (p1 .. p9).
_PITCH_M: float = 0.200
GRID_POINTS_M: tuple[tuple[float, float], ...] = (
    (-_PITCH_M, -_PITCH_M),  # p1
    (0.0, -_PITCH_M),  # p2
    (+_PITCH_M, -_PITCH_M),  # p3
    (-_PITCH_M, 0.0),  # p4
    (0.0, 0.0),  # p5
    (+_PITCH_M, 0.0),  # p6
    (-_PITCH_M, +_PITCH_M),  # p7
    (0.0, +_PITCH_M),  # p8
    (+_PITCH_M, +_PITCH_M),  # p9
)
_EXPECTED_POINTS = len(GRID_POINTS_M)


@dataclass(frozen=True)
class AnemometerGrid:
    """One run of 9 anemometer measurements on the §9.4 plane."""

    labels: tuple[str, ...]
    """Per-point labels (e.g., 'p1' .. 'p9') in CSV order."""

    xy_m: tuple[tuple[float, float], ...]
    """Per-point (x, y) on the plane, in m."""

    v_mean_m_per_s: tuple[float, ...]
    """Per-point time-averaged velocity over the 10-cycle window."""

    v_peak_m_per_s: tuple[float, ...] | None
    """Per-point peak velocity (optional; None if not recorded)."""


@dataclass(frozen=True)
class AnemometerResult:
    """Plane-integrated J_fan_proxy from the 9-point grid."""

    J_fan_proxy_N: float
    """ρ · ⟨v_mean⟩ · A_plane — the canonical proxy."""

    J_fan_proxy_peak_N: float | None
    """ρ · ⟨v_peak⟩ · A_plane — diagnostic; None if peaks were not recorded."""

    v_mean_grid_m_per_s: float
    """Spatial mean across the 9 points."""

    v_mean_grid_std_m_per_s: float
    """Spatial std (ddof=1) across the 9 points — diagnostic of spatial uniformity."""

    n_points: int


def load_anemometer_csv(path: Path | str) -> AnemometerGrid:
    """Load the 9-row CSV the operator filled out at the bench.

    Expected columns (header row required):
      point, x_m, y_m, z_m, v_mean_m_per_s, v_peak_m_per_s, notes

    `v_peak_m_per_s` is optional. The grid coordinates (`x_m`, `y_m`) are
    validated against `GRID_POINTS_M` within 5 mm tolerance — if the
    operator's labels disagree with the locked grid positions, the file
    fails loudly rather than silently producing a wrong plane integral.
    """
    path = Path(path)
    raw: list[dict[str, str]] = []
    header: list[str] | None = None
    with path.open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            cols = [c.strip() for c in s.split(",")]
            if header is None:
                header = cols
                continue
            raw.append(dict(zip(header, cols, strict=False)))
    if header is None or not raw:
        raise ValueError(f"{path}: no rows found")

    if len(raw) != _EXPECTED_POINTS:
        raise ValueError(f"{path}: expected {_EXPECTED_POINTS} grid points, got {len(raw)}")

    labels: list[str] = []
    xy: list[tuple[float, float]] = []
    v_mean: list[float] = []
    v_peak: list[float] = []
    has_peak = True

    for i, row in enumerate(raw):
        for k in ("point", "x_m", "y_m", "v_mean_m_per_s"):
            if k not in row:
                raise ValueError(f"{path} row {i + 1}: missing column {k!r}")
        labels.append(row["point"])
        x = float(row["x_m"])
        y = float(row["y_m"])
        xy.append((x, y))
        v_mean.append(float(row["v_mean_m_per_s"]))
        peak_raw = row.get("v_peak_m_per_s", "")
        if peak_raw in ("", None):
            has_peak = False
        else:
            v_peak.append(float(peak_raw))

    # Cross-check the operator's recorded grid against the locked positions.
    if not _grid_matches(tuple(xy)):
        raise ValueError(
            f"{path}: recorded (x, y) positions do not match the locked 3×3 grid "
            f"({GRID_POINTS_M}). Did the reticle slip or did you record the "
            f"wrong order?"
        )

    return AnemometerGrid(
        labels=tuple(labels),
        xy_m=tuple(xy),
        v_mean_m_per_s=tuple(v_mean),
        v_peak_m_per_s=tuple(v_peak) if has_peak else None,
    )


def j_fan_proxy_from_grid(
    v_mean_grid_m_per_s: float,
    *,
    rho_air_kg_per_m3: float = RHO_AIR_KG_PER_M3,
    A_plane_m2: float = PLANE_AREA_M2,
) -> float:
    """`ρ · v · A` — single-number proxy.

    Intentionally simple: this is the *bench-anemometer* approximation of the
    §9.4 plane integral. CFD evaluations of J_fan use a true surface
    integration with directional decomposition; do not mix the two.
    """
    return rho_air_kg_per_m3 * v_mean_grid_m_per_s * A_plane_m2


def analyze_anemometer_grid(
    grid: AnemometerGrid,
    *,
    rho_air_kg_per_m3: float = RHO_AIR_KG_PER_M3,
    A_plane_m2: float = PLANE_AREA_M2,
) -> AnemometerResult:
    """Plane-mean + plane-mean-peak J_fan_proxy from the 9-point grid."""
    v_mean = np.asarray(grid.v_mean_m_per_s, dtype=float)
    plane_v_mean = float(v_mean.mean())
    plane_v_std = float(v_mean.std(ddof=1)) if v_mean.size > 1 else 0.0
    J_mean = j_fan_proxy_from_grid(
        plane_v_mean, rho_air_kg_per_m3=rho_air_kg_per_m3, A_plane_m2=A_plane_m2
    )
    if grid.v_peak_m_per_s is not None:
        v_peak = np.asarray(grid.v_peak_m_per_s, dtype=float)
        J_peak: float | None = j_fan_proxy_from_grid(
            float(v_peak.mean()),
            rho_air_kg_per_m3=rho_air_kg_per_m3,
            A_plane_m2=A_plane_m2,
        )
    else:
        J_peak = None
    return AnemometerResult(
        J_fan_proxy_N=J_mean,
        J_fan_proxy_peak_N=J_peak,
        v_mean_grid_m_per_s=plane_v_mean,
        v_mean_grid_std_m_per_s=plane_v_std,
        n_points=int(v_mean.size),
    )


# ---------- internal --------------------------------------------------------


def _grid_matches(xy: tuple[tuple[float, float], ...], tol_m: float = 0.005) -> bool:
    """True if `xy` corresponds (in any order) to the locked grid within tol."""
    if len(xy) != len(
        GRID_POINTS_M
    ):  # pragma: no cover  (load_anemometer_csv enforces count first)
        return False
    remaining = list(GRID_POINTS_M)
    for x, y in xy:
        # Find a nearest locked point within tol; consume it.
        match_idx: int | None = None
        for i, (xl, yl) in enumerate(remaining):
            if abs(x - xl) <= tol_m and abs(y - yl) <= tol_m:
                match_idx = i
                break
        if match_idx is None:
            return False
        remaining.pop(match_idx)
    return not remaining
