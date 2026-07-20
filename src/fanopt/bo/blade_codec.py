"""BO design-vector ↔ :class:`BladeParams` codec (V1 redesign, lean free-panel space).

The bijection between the optimizer's flat bounded float vector and the structured
:class:`~fanopt.geometry.blade.BladeParams`. Successor to the 35-var ``bo.codec``:
rib meridian (2) + rib thickness (2) + a free panel displacement grid
(``PANEL_GRID_RADIAL_COUNT × PANEL_GRID_TANGENTIAL_COUNT``) + panel thickness (1) +
the ``blade_count`` categorical, per ``docs/blade_architecture_redesign.md`` §7.2.

Categoricals are carried as a continuous coordinate in ``[0, len(choices))`` that
:func:`decode` floors to a choice index — the standard continuous-relaxation handling
so a single GP sees one homogeneous float box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fanopt.geometry.blade import (
    PANEL_GRID_RADIAL_COUNT,
    PANEL_GRID_TANGENTIAL_COUNT,
    PANEL_OFFSET_RANGE_M,
    PANEL_THICKNESS_NOM_RANGE_M,
    RIB_BOW_RANGE_M,
    RIB_THICKNESS_RANGE_M,
    BladeParams,
)
from fanopt.geometry.schema import BLADE_COUNTS

__all__ = [
    "Var",
    "SEARCH_SPACE",
    "N_DIMS",
    "encode",
    "decode",
    "bounds",
    "clip_to_bounds",
]


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
    vars_: list[Var] = [
        Var("rib_bow_mid_m", *RIB_BOW_RANGE_M),
        Var("rib_bow_tip_m", *RIB_BOW_RANGE_M),
        Var("t_rib_hub_m", *RIB_THICKNESS_RANGE_M),
        Var("t_rib_tip_m", *RIB_THICKNESS_RANGE_M),
    ]
    for i in range(PANEL_GRID_RADIAL_COUNT):
        for j in range(PANEL_GRID_TANGENTIAL_COUNT):
            vars_.append(Var(f"panel_z_{i}_{j}", *PANEL_OFFSET_RANGE_M))
    vars_.append(Var("panel_thickness_nom_m", *PANEL_THICKNESS_NOM_RANGE_M))
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
    """Clip a vector into the search box (continuous dims clamped; the categorical
    coordinate kept strictly below its upper index so :func:`decode` floors valid)."""
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


def decode(vec: np.ndarray) -> BladeParams:
    """Vector → validated :class:`BladeParams` (raises ``ValueError`` on shape/bounds)."""
    arr = np.asarray(vec, dtype=float)
    if arr.shape != (N_DIMS,):
        raise ValueError(f"vector must have shape ({N_DIMS},); got {arr.shape}")
    grid = tuple(
        tuple(float(arr[_IDX[f"panel_z_{i}_{j}"]]) for j in range(PANEL_GRID_TANGENTIAL_COUNT))
        for i in range(PANEL_GRID_RADIAL_COUNT)
    )
    blade_count: int = _decode_categorical(
        float(arr[_IDX["blade_count"]]), SEARCH_SPACE[_IDX["blade_count"]]
    )
    return BladeParams(
        blade_count=blade_count,
        rib_bow_mid_m=float(arr[_IDX["rib_bow_mid_m"]]),
        rib_bow_tip_m=float(arr[_IDX["rib_bow_tip_m"]]),
        t_rib_hub_m=float(arr[_IDX["t_rib_hub_m"]]),
        t_rib_tip_m=float(arr[_IDX["t_rib_tip_m"]]),
        panel_offsets_m=grid,
        panel_thickness_nom_m=float(arr[_IDX["panel_thickness_nom_m"]]),
    )


def encode(params: BladeParams) -> np.ndarray:
    """:class:`BladeParams` → vector. ``decode(encode(p))`` round-trips every dimension."""
    out = np.zeros(N_DIMS, dtype=float)
    out[_IDX["rib_bow_mid_m"]] = params.rib_bow_mid_m
    out[_IDX["rib_bow_tip_m"]] = params.rib_bow_tip_m
    out[_IDX["t_rib_hub_m"]] = params.t_rib_hub_m
    out[_IDX["t_rib_tip_m"]] = params.t_rib_tip_m
    for i in range(PANEL_GRID_RADIAL_COUNT):
        for j in range(PANEL_GRID_TANGENTIAL_COUNT):
            out[_IDX[f"panel_z_{i}_{j}"]] = params.panel_offsets_m[i][j]
    out[_IDX["panel_thickness_nom_m"]] = params.panel_thickness_nom_m
    out[_IDX["blade_count"]] = float(BLADE_COUNTS.index(params.blade_count)) + 0.5
    return out
