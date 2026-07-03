"""BO design-vector ↔ params codec (Phase 4, V1-slim single-fidelity).

The optimizer works in a flat, bounded, all-float vector; the geometry and CFD
pipeline works in structured, validated dataclasses. This module is the bijection
between them. :data:`SEARCH_SPACE` is the ordered vector layout; :func:`encode`
and :func:`decode` move between a :class:`~fanopt.geometry.envelope.Layer1Params`
and its vector; :func:`bounds` gives the box the acquisition optimizes inside.

Scope (35 vars, V1-final): the 18-point thickness grid, the 4-var corrugation
family, camber + twist (airfoil mean surface), Fourier LE/TE edge modulation (6),
and the ``edge_profile`` + ``blade_count`` categoricals. Deferred by decision
(2026-07-03, plan §10 note): **Layer-2 louvers → V2** (their value is a 2D-slice
drag-asymmetry model best done with reference validation; corrugation + camber
already give the ASO its asymmetry levers), and **print_orientation is fixed at
'flat'** (V1 plano-convex baseline). Consequence: ``twist`` is geometrically inert
under the flat lock (the generator zeros it; the slice doesn't apply it) — it is
retained as a reserved axis for a future 'edge' orientation, not an active lever.

Categoricals are carried as a continuous coordinate in ``[0, len(choices))`` that
:func:`decode` floors to a choice index — the standard continuous-relaxation
handling so a single GP sees one homogeneous float box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fanopt.geometry.envelope import (
    CAMBER_RANGE_M,
    EDGE_PROFILES,
    FOURIER_AMPLITUDE_RELATIVE_MAX,
    FOURIER_HARMONIC_COUNT,
    TWIST_RANGE_RAD,
    Layer1Params,
    ThicknessGridField,
)
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
    "Var",
    "SEARCH_SPACE",
    "N_DIMS",
    "encode",
    "decode",
    "bounds",
    "clip_to_bounds",
]

_N_CAMBER_KNOTS = 3
_N_TWIST_KNOTS = 2
_N_FOURIER = FOURIER_HARMONIC_COUNT  # 3 harmonics on each of LE and TE
# 2π and π ceilings for the periodic corrugation vars (open interval → tiny epsilon
# below the top so a decoded value never trips the ``< 2π`` / ``< π`` validators).
_TWO_PI = 2.0 * np.pi
_PERIODIC_EPS = 1e-9


@dataclass(frozen=True)
class Var:
    """One coordinate of the BO vector.

    ``kind`` is ``"continuous"`` (value used directly, bounded ``[low, high]``) or
    ``"categorical"`` (value in ``[0, n_choices)``, floored to a ``choices`` index).
    """

    name: str
    low: float
    high: float
    kind: str = "continuous"
    choices: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("continuous", "categorical"):
            raise ValueError(f"kind must be continuous|categorical; got {self.kind!r}")
        if self.kind == "categorical" and not self.choices:
            raise ValueError(f"categorical var {self.name!r} needs choices")
        if self.high <= self.low:
            raise ValueError(f"var {self.name!r} needs high > low; got [{self.low}, {self.high}]")


def _build_search_space() -> tuple[Var, ...]:
    vars_: list[Var] = []
    for i in range(THICKNESS_GRID_RADIAL_COUNT):
        for j in range(THICKNESS_GRID_TANGENTIAL_COUNT):
            vars_.append(Var(f"t_{i}_{j}", PANEL_THICKNESS_MIN_M, PANEL_THICKNESS_MAX_M))
    vars_.append(Var("corrugation_amplitude_m", 0.0, CORRUGATION_AMPLITUDE_MAX_M))
    vars_.append(Var("corrugation_wavelength", *CORRUGATION_WAVELENGTH_RANGE))
    vars_.append(Var("corrugation_phase_rad", 0.0, _TWO_PI))
    vars_.append(Var("corrugation_orientation_rad", 0.0, float(np.pi)))
    for k in range(_N_CAMBER_KNOTS):
        vars_.append(Var(f"camber_{k}", *CAMBER_RANGE_M))
    for k in range(_N_TWIST_KNOTS):
        vars_.append(Var(f"twist_{k}", *TWIST_RANGE_RAD))
    _fa = FOURIER_AMPLITUDE_RELATIVE_MAX
    for k in range(_N_FOURIER):
        vars_.append(Var(f"fourier_le_{k}", -_fa, _fa))
    for k in range(_N_FOURIER):
        vars_.append(Var(f"fourier_te_{k}", -_fa, _fa))
    vars_.append(
        Var(
            "edge_profile",
            0.0,
            float(len(EDGE_PROFILES)),
            kind="categorical",
            choices=EDGE_PROFILES,
        )
    )
    vars_.append(
        Var("blade_count", 0.0, float(len(BLADE_COUNTS)), kind="categorical", choices=BLADE_COUNTS)
    )
    return tuple(vars_)


SEARCH_SPACE: tuple[Var, ...] = _build_search_space()
N_DIMS: int = len(SEARCH_SPACE)

_IDX = {v.name: i for i, v in enumerate(SEARCH_SPACE)}


def bounds() -> tuple[np.ndarray, np.ndarray]:
    """``(low, high)`` arrays of shape ``(N_DIMS,)`` — the acquisition box."""
    low = np.array([v.low for v in SEARCH_SPACE], dtype=float)
    high = np.array([v.high for v in SEARCH_SPACE], dtype=float)
    return low, high


def clip_to_bounds(vec: np.ndarray) -> np.ndarray:
    """Clip a vector into the search box (continuous dims clamped, categorical
    coordinates kept strictly below their upper index so :func:`decode` floors
    to a valid choice)."""
    low, high = bounds()
    out = np.clip(np.asarray(vec, dtype=float), low, high)
    for i, v in enumerate(SEARCH_SPACE):
        if v.kind == "categorical":
            out[i] = min(out[i], v.high - 1e-9)
    return out


def _decode_categorical(value: float, var: Var) -> Any:
    assert var.choices is not None
    idx = int(np.floor(value))
    idx = min(max(idx, 0), len(var.choices) - 1)
    return var.choices[idx]


def decode(vec: np.ndarray) -> Layer1Params:
    """Vector → validated :class:`Layer1Params` (raises ``ValueError`` on bounds)."""
    arr = np.asarray(vec, dtype=float)
    if arr.shape != (N_DIMS,):
        raise ValueError(f"vector must have shape ({N_DIMS},); got {arr.shape}")

    grid = tuple(
        tuple(float(arr[_IDX[f"t_{i}_{j}"]]) for j in range(THICKNESS_GRID_TANGENTIAL_COUNT))
        for i in range(THICKNESS_GRID_RADIAL_COUNT)
    )
    field = ThicknessGridField(
        grid_m=grid,
        corrugation_amplitude_m=float(arr[_IDX["corrugation_amplitude_m"]]),
        corrugation_wavelength=float(arr[_IDX["corrugation_wavelength"]]),
        corrugation_phase_rad=min(
            float(arr[_IDX["corrugation_phase_rad"]]), _TWO_PI - _PERIODIC_EPS
        ),
        corrugation_orientation_rad=min(
            float(arr[_IDX["corrugation_orientation_rad"]]), float(np.pi) - _PERIODIC_EPS
        ),
    )
    camber = tuple(float(arr[_IDX[f"camber_{k}"]]) for k in range(_N_CAMBER_KNOTS))
    twist = tuple(float(arr[_IDX[f"twist_{k}"]]) for k in range(_N_TWIST_KNOTS))
    fourier_le = tuple(float(arr[_IDX[f"fourier_le_{k}"]]) for k in range(_N_FOURIER))
    fourier_te = tuple(float(arr[_IDX[f"fourier_te_{k}"]]) for k in range(_N_FOURIER))
    edge_profile: str = _decode_categorical(
        float(arr[_IDX["edge_profile"]]), SEARCH_SPACE[_IDX["edge_profile"]]
    )
    blade_count: int = _decode_categorical(
        float(arr[_IDX["blade_count"]]), SEARCH_SPACE[_IDX["blade_count"]]
    )
    return Layer1Params(
        blade_count=blade_count,
        camber_knots_m=camber,
        twist_knots_rad=twist,
        thickness_field=field,
        edge_profile=edge_profile,
        fourier_le_amplitudes=fourier_le,  # type: ignore[arg-type]
        fourier_te_amplitudes=fourier_te,  # type: ignore[arg-type]
    )


def encode(params: Layer1Params) -> np.ndarray:
    """:class:`Layer1Params` → vector (inverse of :func:`decode` on searched dims).

    ``decode(encode(p))`` round-trips every searched dimension (thickness grid,
    corrugation, camber, twist, Fourier LE/TE, edge profile, blade count).
    """
    if len(params.camber_knots_m) != _N_CAMBER_KNOTS:
        raise ValueError(f"encode expects {_N_CAMBER_KNOTS} camber knots")
    if len(params.twist_knots_rad) != _N_TWIST_KNOTS:
        raise ValueError(f"encode expects {_N_TWIST_KNOTS} twist knots")
    out = np.zeros(N_DIMS, dtype=float)
    field = params.thickness_field
    for i in range(THICKNESS_GRID_RADIAL_COUNT):
        for j in range(THICKNESS_GRID_TANGENTIAL_COUNT):
            out[_IDX[f"t_{i}_{j}"]] = field.grid_m[i][j]
    out[_IDX["corrugation_amplitude_m"]] = field.corrugation_amplitude_m
    out[_IDX["corrugation_wavelength"]] = field.corrugation_wavelength
    out[_IDX["corrugation_phase_rad"]] = field.corrugation_phase_rad
    out[_IDX["corrugation_orientation_rad"]] = field.corrugation_orientation_rad
    for k in range(_N_CAMBER_KNOTS):
        out[_IDX[f"camber_{k}"]] = params.camber_knots_m[k]
    for k in range(_N_TWIST_KNOTS):
        out[_IDX[f"twist_{k}"]] = params.twist_knots_rad[k]
    for k in range(_N_FOURIER):
        out[_IDX[f"fourier_le_{k}"]] = params.fourier_le_amplitudes[k]
        out[_IDX[f"fourier_te_{k}"]] = params.fourier_te_amplitudes[k]
    out[_IDX["edge_profile"]] = float(EDGE_PROFILES.index(params.edge_profile)) + 0.5
    out[_IDX["blade_count"]] = float(BLADE_COUNTS.index(params.blade_count)) + 0.5
    return out
