"""Geometry Layer 1 — outer envelope + Fourier LE/TE.

Implements the Layer 1 BO design-parameter schema per plan §6.2.1
(`docs/report-final.md` §6.2.1 + §3.2 + §9.7). Layer 1 covers the panel's
smooth outer shape: blade count, camber/twist/thickness profiles, edge
profile family, and Fourier-modulated leading/trailing edges. ~14 vars.

The actual CadQuery generator (``make_outer_envelope``) will land in
Phase 1; this module ships the schema + validators that the BO loop and
the eventual generator both consume.

Locked constants imported from :mod:`fanopt.geometry.schema`:
- ``BLADE_COUNTS`` — {8, 10, 12}; 14 removed for ergonomic infeasibility
  (>180° past straight-line, Round 7).
- ``PANEL_THICKNESS_MIN_M`` / ``PANEL_THICKNESS_MAX_M`` — 2.2-3.8 mm
  (chamfer-clearance floor + folded-collision ceiling).

The panel thickness is a Path A+ :class:`ThicknessGridField` (control-point grid
+ corrugation), replacing the legacy 3-knot spline (see plan_v1_slim_latest.md
§10). ``ThicknessGridField.from_radial_knots`` bridges the old 3-knot form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from fanopt.geometry.schema import (
    BLADE_COUNTS,
    CORRUGATION_AMPLITUDE_MAX_M,
    CORRUGATION_WAVELENGTH_RANGE,
    PANEL_THICKNESS_MAX_M,
    PANEL_THICKNESS_MIN_M,
    THICKNESS_GRID_RADIAL_COUNT,
    THICKNESS_GRID_TANGENTIAL_COUNT,
)

__all__ = [
    "CAMBER_KNOT_COUNT_RANGE",
    "CAMBER_RANGE_M",
    "TWIST_KNOT_COUNT_RANGE",
    "TWIST_RANGE_RAD",
    "EDGE_PROFILES",
    "FOURIER_HARMONIC_COUNT",
    "FOURIER_AMPLITUDE_RELATIVE_MAX",
    "ThicknessGridField",
    "Layer1Params",
]


CAMBER_KNOT_COUNT_RANGE: tuple[int, int] = (3, 4)
"""Camber spline knot count: 3 or 4. Plan §6.2.1."""

CAMBER_RANGE_M: tuple[float, float] = (0.0, 0.005)
"""0-5 mm. Plan §6.2.1. Plano-convex constraint when ``print_orientation = flat``
(Layer 4) applies to the top face only; the bottom face is planar."""

TWIST_KNOT_COUNT_RANGE: tuple[int, int] = (2, 3)
"""Twist distribution knot count: 2 or 3."""

TWIST_RANGE_RAD: tuple[float, float] = (-math.radians(10.0), math.radians(10.0))
"""±10°. Plan §6.2.1."""

EDGE_PROFILES: tuple[str, ...] = ("sharp", "rounded", "mildly-serrated")
"""Edge profile category — architecture-bandit variable."""

FOURIER_HARMONIC_COUNT: int = 3
"""k = 1, 2, 3 harmonics on each of LE and TE (phases fixed at 0, π/3, 2π/3)."""

FOURIER_AMPLITUDE_RELATIVE_MAX: float = 0.15
"""±15% of mean envelope radius. Plan §6.2.1: 'bounded so envelope stays
within ±15% of mean'."""

_THICKNESS_NOMINAL_M: float = (PANEL_THICKNESS_MIN_M + PANEL_THICKNESS_MAX_M) / 2.0


@dataclass(frozen=True)
class ThicknessGridField:
    """Path A+ panel thickness field (plan_v1_slim_latest.md §10).

    A radial × tangential control-point grid of panel thicknesses plus a
    corrugation family. :meth:`thickness_at` bilinearly interpolates the grid
    and adds the corrugation, clamped to the [MIN, MAX] thickness lock. A uniform
    grid with zero corrugation is a flat cambered panel — the neutral baseline
    the optimizer reaches when smooth beats faceted; nothing forces corrugation.
    """

    grid_m: tuple[tuple[float, ...], ...]
    corrugation_amplitude_m: float = 0.0
    corrugation_wavelength: float = 0.5
    corrugation_phase_rad: float = 0.0
    corrugation_orientation_rad: float = 0.0

    def __post_init__(self) -> None:
        self._validate_grid()
        self._validate_corrugation()

    def _validate_grid(self) -> None:
        rows, cols = THICKNESS_GRID_RADIAL_COUNT, THICKNESS_GRID_TANGENTIAL_COUNT
        if len(self.grid_m) != rows:
            raise ValueError(f"grid_m must have {rows} radial rows, got {len(self.grid_m)}")
        for i, row in enumerate(self.grid_m):
            if len(row) != cols:
                raise ValueError(f"grid_m[{i}] must have {cols} tangential points, got {len(row)}")
            for j, val in enumerate(row):
                if not (PANEL_THICKNESS_MIN_M <= val <= PANEL_THICKNESS_MAX_M):
                    raise ValueError(
                        f"grid_m[{i}][{j}] = {val} outside "
                        f"[{PANEL_THICKNESS_MIN_M}, {PANEL_THICKNESS_MAX_M}]"
                    )

    def _validate_corrugation(self) -> None:
        if not (0.0 <= self.corrugation_amplitude_m <= CORRUGATION_AMPLITUDE_MAX_M):
            raise ValueError(
                f"corrugation_amplitude_m = {self.corrugation_amplitude_m} outside "
                f"[0, {CORRUGATION_AMPLITUDE_MAX_M}]"
            )
        lo, hi = CORRUGATION_WAVELENGTH_RANGE
        if not (lo <= self.corrugation_wavelength <= hi):
            raise ValueError(
                f"corrugation_wavelength = {self.corrugation_wavelength} outside [{lo}, {hi}]"
            )
        if not (0.0 <= self.corrugation_phase_rad < 2.0 * math.pi):
            raise ValueError(
                f"corrugation_phase_rad = {self.corrugation_phase_rad} outside [0, 2π)"
            )
        if not (0.0 <= self.corrugation_orientation_rad < math.pi):
            raise ValueError(
                f"corrugation_orientation_rad = {self.corrugation_orientation_rad} outside [0, π)"
            )

    @classmethod
    def uniform(cls, thickness_m: float = _THICKNESS_NOMINAL_M) -> ThicknessGridField:
        """Flat baseline: every control point at ``thickness_m``, no corrugation."""
        row = tuple(thickness_m for _ in range(THICKNESS_GRID_TANGENTIAL_COUNT))
        return cls(grid_m=tuple(row for _ in range(THICKNESS_GRID_RADIAL_COUNT)))

    @classmethod
    def from_radial_knots(cls, knots_m: tuple[float, ...]) -> ThicknessGridField:
        """Bridge from the legacy 3-knot radial spline: each knot becomes a
        tangentially-uniform radial row (no corrugation). Preserves the old
        radial thickness profile exactly."""
        if len(knots_m) != THICKNESS_GRID_RADIAL_COUNT:
            raise ValueError(
                f"from_radial_knots needs {THICKNESS_GRID_RADIAL_COUNT} knots, got {len(knots_m)}"
            )
        rows = tuple(tuple(k for _ in range(THICKNESS_GRID_TANGENTIAL_COUNT)) for k in knots_m)
        return cls(grid_m=rows)

    @property
    def is_flat(self) -> bool:
        """True when the field is a uniform-thickness flat panel (no facets/corrugation)."""
        first = self.grid_m[0][0]
        uniform = all(v == first for row in self.grid_m for v in row)
        return uniform and self.corrugation_amplitude_m == 0.0

    def thickness_at(self, u: float, v: float) -> float:
        """Panel thickness at parametric ``(u, v)`` — u∈[0,1] radial, v∈[-1,1] tangential.

        Bilinear over the grid + corrugation, clamped to the thickness lock.
        """
        rows, cols = THICKNESS_GRID_RADIAL_COUNT, THICKNESS_GRID_TANGENTIAL_COUNT
        u = min(max(u, 0.0), 1.0)
        v = min(max(v, -1.0), 1.0)
        v_n = (v + 1.0) / 2.0  # tangential in [0, 1]
        fr = u * (rows - 1)
        i0 = min(int(fr), rows - 2)
        tr = fr - i0
        ft = v_n * (cols - 1)
        j0 = min(int(ft), cols - 2)
        tt = ft - j0
        g = self.grid_m
        top = g[i0][j0] * (1.0 - tt) + g[i0][j0 + 1] * tt
        bot = g[i0 + 1][j0] * (1.0 - tt) + g[i0 + 1][j0 + 1] * tt
        base = top * (1.0 - tr) + bot * tr
        theta = self.corrugation_orientation_rad
        arg = (
            2.0
            * math.pi
            * (u * math.sin(theta) + v_n * math.cos(theta))
            / self.corrugation_wavelength
            + self.corrugation_phase_rad
        )
        thickness = base + self.corrugation_amplitude_m * math.sin(arg)
        return min(max(thickness, PANEL_THICKNESS_MIN_M), PANEL_THICKNESS_MAX_M)

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_m": [list(row) for row in self.grid_m],
            "corrugation_amplitude_m": self.corrugation_amplitude_m,
            "corrugation_wavelength": self.corrugation_wavelength,
            "corrugation_phase_rad": self.corrugation_phase_rad,
            "corrugation_orientation_rad": self.corrugation_orientation_rad,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThicknessGridField:
        return cls(
            grid_m=tuple(tuple(float(x) for x in row) for row in data["grid_m"]),
            corrugation_amplitude_m=float(data.get("corrugation_amplitude_m", 0.0)),
            corrugation_wavelength=float(data.get("corrugation_wavelength", 0.5)),
            corrugation_phase_rad=float(data.get("corrugation_phase_rad", 0.0)),
            corrugation_orientation_rad=float(data.get("corrugation_orientation_rad", 0.0)),
        )


@dataclass(frozen=True)
class Layer1Params:
    """Layer 1 design parameters — outer envelope + Fourier modulation.

    Construction validates bounds; ``ValueError`` is raised at instantiation
    on any out-of-range value. Use :meth:`to_dict` / :meth:`from_dict` for
    BO-loop serialisation.
    """

    blade_count: int
    camber_knots_m: tuple[float, ...]
    twist_knots_rad: tuple[float, ...]
    thickness_field: ThicknessGridField
    edge_profile: str
    fourier_le_amplitudes: tuple[float, float, float]
    fourier_te_amplitudes: tuple[float, float, float]

    def __post_init__(self) -> None:
        self._validate_blade_count()
        self._validate_camber()
        self._validate_twist()
        # thickness_field self-validates in its own __post_init__ (Path A+).
        self._validate_edge_profile()
        self._validate_fourier(self.fourier_le_amplitudes, "fourier_le_amplitudes")
        self._validate_fourier(self.fourier_te_amplitudes, "fourier_te_amplitudes")

    def _validate_blade_count(self) -> None:
        if self.blade_count not in BLADE_COUNTS:
            raise ValueError(f"blade_count must be one of {BLADE_COUNTS}, got {self.blade_count}")

    def _validate_camber(self) -> None:
        lo, hi = CAMBER_KNOT_COUNT_RANGE
        n = len(self.camber_knots_m)
        if not (lo <= n <= hi):
            raise ValueError(f"camber_knots_m must have {lo}-{hi} elements, got {n}")
        c_lo, c_hi = CAMBER_RANGE_M
        for i, v in enumerate(self.camber_knots_m):
            if not (c_lo <= v <= c_hi):
                raise ValueError(f"camber_knots_m[{i}] = {v} outside range [{c_lo}, {c_hi}]")

    def _validate_twist(self) -> None:
        lo, hi = TWIST_KNOT_COUNT_RANGE
        n = len(self.twist_knots_rad)
        if not (lo <= n <= hi):
            raise ValueError(f"twist_knots_rad must have {lo}-{hi} elements, got {n}")
        t_lo, t_hi = TWIST_RANGE_RAD
        for i, v in enumerate(self.twist_knots_rad):
            if not (t_lo <= v <= t_hi):
                raise ValueError(f"twist_knots_rad[{i}] = {v} outside range [{t_lo}, {t_hi}]")

    def _validate_edge_profile(self) -> None:
        if self.edge_profile not in EDGE_PROFILES:
            raise ValueError(
                f"edge_profile must be one of {EDGE_PROFILES}, got " f"{self.edge_profile!r}"
            )

    def _validate_fourier(self, amps: tuple[float, float, float], name: str) -> None:
        if len(amps) != FOURIER_HARMONIC_COUNT:
            raise ValueError(
                f"{name} must have exactly {FOURIER_HARMONIC_COUNT} "
                f"elements (k=1,2,3), got {len(amps)}"
            )
        for k, a in enumerate(amps, start=1):
            if abs(a) > FOURIER_AMPLITUDE_RELATIVE_MAX:
                raise ValueError(
                    f"{name}[k={k}] = {a} exceeds "
                    f"±{FOURIER_AMPLITUDE_RELATIVE_MAX} envelope-relative bound"
                )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict for BO ledger / serialisation."""
        return {
            "blade_count": self.blade_count,
            "camber_knots_m": list(self.camber_knots_m),
            "twist_knots_rad": list(self.twist_knots_rad),
            "thickness_field": self.thickness_field.to_dict(),
            "edge_profile": self.edge_profile,
            "fourier_le_amplitudes": list(self.fourier_le_amplitudes),
            "fourier_te_amplitudes": list(self.fourier_te_amplitudes),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Layer1Params:
        """Inverse of :meth:`to_dict`. Validates on construction."""
        return cls(
            blade_count=int(d["blade_count"]),
            camber_knots_m=tuple(float(v) for v in d["camber_knots_m"]),
            twist_knots_rad=tuple(float(v) for v in d["twist_knots_rad"]),
            thickness_field=ThicknessGridField.from_dict(d["thickness_field"]),
            edge_profile=str(d["edge_profile"]),
            fourier_le_amplitudes=tuple(float(v) for v in d["fourier_le_amplitudes"]),  # type: ignore[arg-type]
            fourier_te_amplitudes=tuple(float(v) for v in d["fourier_te_amplitudes"]),  # type: ignore[arg-type]
        )
