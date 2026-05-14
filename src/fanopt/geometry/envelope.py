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
- ``PANEL_THICKNESS_KNOT_COUNT`` — 3 spline knots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from fanopt.geometry.schema import (
    BLADE_COUNTS,
    PANEL_THICKNESS_KNOT_COUNT,
    PANEL_THICKNESS_MAX_M,
    PANEL_THICKNESS_MIN_M,
)

__all__ = [
    "CAMBER_KNOT_COUNT_RANGE",
    "CAMBER_RANGE_M",
    "TWIST_KNOT_COUNT_RANGE",
    "TWIST_RANGE_RAD",
    "EDGE_PROFILES",
    "FOURIER_HARMONIC_COUNT",
    "FOURIER_AMPLITUDE_RELATIVE_MAX",
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
    thickness_knots_m: tuple[float, float, float]
    edge_profile: str
    fourier_le_amplitudes: tuple[float, float, float]
    fourier_te_amplitudes: tuple[float, float, float]

    def __post_init__(self) -> None:
        self._validate_blade_count()
        self._validate_camber()
        self._validate_twist()
        self._validate_thickness()
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

    def _validate_thickness(self) -> None:
        n = len(self.thickness_knots_m)
        if n != PANEL_THICKNESS_KNOT_COUNT:
            raise ValueError(
                f"thickness_knots_m must have exactly "
                f"{PANEL_THICKNESS_KNOT_COUNT} elements, got {n}"
            )
        for i, v in enumerate(self.thickness_knots_m):
            if not (PANEL_THICKNESS_MIN_M <= v <= PANEL_THICKNESS_MAX_M):
                raise ValueError(
                    f"thickness_knots_m[{i}] = {v} outside range "
                    f"[{PANEL_THICKNESS_MIN_M}, {PANEL_THICKNESS_MAX_M}]"
                )

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
            "thickness_knots_m": list(self.thickness_knots_m),
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
            thickness_knots_m=tuple(float(v) for v in d["thickness_knots_m"]),  # type: ignore[arg-type]
            edge_profile=str(d["edge_profile"]),
            fourier_le_amplitudes=tuple(float(v) for v in d["fourier_le_amplitudes"]),  # type: ignore[arg-type]
            fourier_te_amplitudes=tuple(float(v) for v in d["fourier_te_amplitudes"]),  # type: ignore[arg-type]
        )
