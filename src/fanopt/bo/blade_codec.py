"""BO design-vector ↔ :class:`BladeParams` codec (V1 redesign, **feasible by construction**).

The bijection between the optimizer's flat bounded float vector and the structured
:class:`~fanopt.geometry.blade.BladeParams`. Crucially, :func:`decode` maps every vector
to a **fold- and containment-feasible** blade, so the BO searches a space where ~100 % of
designs are valid instead of penalizing a mostly-infeasible space (uniform sampling of
raw physical thickness/offsets is almost never feasible). Concretely:

- **rib thickness** is a knob in ``[0, 1]`` mapped to ``[T_MIN, cap(blade_count)]`` where
  ``cap`` is the thickest rib that still folds under the stack-height limit for that blade
  count — so the fold constraint holds by construction;
- **panel thickness** is a knob mapped to ``[P_MIN, min(rib)]`` — so ``panel ≤ rib`` always;
- **panel offsets** are knobs in ``[-1, 1]`` scaled by the *local* containable envelope
  ``(t_rib(r) − panel)/2`` — so the panel never pokes past the rib.

``rib_bow`` (unconstrained) and ``blade_count`` (categorical) stay direct. Mass is not
forced here (it depends on the whole design); it stays a soft check in the objective, and
is rarely violated once ribs are fold-thinned. ``decode(encode(p))`` round-trips any
feasible ``p``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from fanopt.geometry.blade import (
    FOLD_CLEARANCE_M,
    MAX_FOLDED_STACK_HEIGHT_M,
    PANEL_GRID_RADIAL_COUNT,
    PANEL_GRID_TANGENTIAL_COUNT,
    PANEL_THICKNESS_NOM_RANGE_M,
    RIB_BOW_RANGE_M,
    RIB_THICKNESS_RANGE_M,
    BladeParams,
    estimate_mass_kg,
    panel_radial_stations,
)
from fanopt.geometry.schema import BLADE_COUNTS, HUB_RADIUS_M, L_RIB_M, MAX_TOTAL_MASS_KG

__all__ = [
    "Var",
    "SEARCH_SPACE",
    "N_DIMS",
    "rib_thickness_cap_m",
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


@lru_cache(maxsize=None)
def _mass_thickness_cap_m(blade_count: int) -> float:
    """Max uniform rib thickness (m) keeping the mass proxy ≤ the cap for this blade count.

    Evaluated at the heaviest panel a given rib allows (``min(rib, P_HI)``, flat), so any
    lighter panel is safe too. Binary search on the monotone mass–thickness relation.
    """
    lo, hi = RIB_THICKNESS_RANGE_M
    flat = tuple((0.0,) * PANEL_GRID_TANGENTIAL_COUNT for _ in range(PANEL_GRID_RADIAL_COUNT))

    def mass(t: float) -> float:
        return estimate_mass_kg(
            BladeParams(
                blade_count=blade_count, rib_bow_mid_m=0.0, rib_bow_tip_m=0.0,
                t_rib_hub_m=t, t_rib_tip_m=t, panel_offsets_m=flat,
                panel_thickness_nom_m=min(t, _P_HI),
            )
        )

    if mass(lo) > MAX_TOTAL_MASS_KG:
        return lo  # even the thinnest rib is over (this blade count barely fits)
    if mass(hi) <= MAX_TOTAL_MASS_KG:
        return hi
    a, b = lo, hi
    for _ in range(28):
        m = 0.5 * (a + b)
        if mass(m) <= MAX_TOTAL_MASS_KG:
            a = m
        else:
            b = m
    return a


def rib_thickness_cap_m(blade_count: int) -> float:
    """Thickest rib (m) that both **folds** and keeps **mass ≤ cap** for ``blade_count``.

    ``min`` of the fold cap (``N·(t+c) ≤ stack``), the mass cap, and the rib range max —
    so more blades ⇒ thinner ribs, and the design is fold + mass feasible by construction.
    """
    lo, hi = RIB_THICKNESS_RANGE_M
    fold_cap = MAX_FOLDED_STACK_HEIGHT_M / blade_count - FOLD_CLEARANCE_M
    cap = min(fold_cap, _mass_thickness_cap_m(blade_count), hi)
    return min(max(cap, lo), hi)


def _build_search_space() -> tuple[Var, ...]:
    vars_: list[Var] = [
        Var("rib_bow_mid_m", *RIB_BOW_RANGE_M),
        Var("rib_bow_tip_m", *RIB_BOW_RANGE_M),
        Var("t_rib_hub_k", 0.0, 1.0),  # knob → [T_MIN, cap(blade_count)]
        Var("t_rib_tip_k", 0.0, 1.0),
    ]
    for i in range(PANEL_GRID_RADIAL_COUNT):
        for j in range(PANEL_GRID_TANGENTIAL_COUNT):
            vars_.append(Var(f"panel_z_{i}_{j}", -1.0, 1.0))  # fraction of containable envelope
    vars_.append(Var("panel_thickness_k", 0.0, 1.0))  # knob → [P_MIN, min(rib)]
    vars_.append(
        Var("blade_count", 0.0, float(len(BLADE_COUNTS)), kind="categorical", choices=BLADE_COUNTS)
    )
    return tuple(vars_)


SEARCH_SPACE: tuple[Var, ...] = _build_search_space()
N_DIMS: int = len(SEARCH_SPACE)

_IDX = {v.name: i for i, v in enumerate(SEARCH_SPACE)}
_T_LO = RIB_THICKNESS_RANGE_M[0]
_P_LO, _P_HI = PANEL_THICKNESS_NOM_RANGE_M


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


def _c01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def _radial_fracs() -> list[float]:
    return [(r - HUB_RADIUS_M) / L_RIB_M for r in panel_radial_stations()]


def decode(vec: np.ndarray) -> BladeParams:
    """Vector → **feasible** :class:`BladeParams` (fold + containment by construction)."""
    arr = np.asarray(vec, dtype=float)
    if arr.shape != (N_DIMS,):
        raise ValueError(f"vector must have shape ({N_DIMS},); got {arr.shape}")

    blade_count: int = _decode_categorical(
        float(arr[_IDX["blade_count"]]), SEARCH_SPACE[_IDX["blade_count"]]
    )
    # Rib thickness: knob → [T_MIN, fold cap]. Panel: knob → [P_MIN, min(rib)].
    t_cap = rib_thickness_cap_m(blade_count)
    t_hub = _T_LO + _c01(float(arr[_IDX["t_rib_hub_k"]])) * (t_cap - _T_LO)
    t_tip = _T_LO + _c01(float(arr[_IDX["t_rib_tip_k"]])) * (t_cap - _T_LO)
    p_hi = max(min(t_hub, t_tip, _P_HI), _P_LO)  # panel ≤ min(rib) and within its own range
    panel = _P_LO + _c01(float(arr[_IDX["panel_thickness_k"]])) * (p_hi - _P_LO)

    # Offsets: fraction of the local containable envelope (t_rib(r) − panel)/2 ≥ 0.
    grid: list[tuple[float, ...]] = []
    for i, u in enumerate(_radial_fracs()):
        t_rib_i = t_hub * (1.0 - u) + t_tip * u
        allow = max((t_rib_i - panel) / 2.0, 0.0)
        grid.append(
            tuple(
                min(max(float(arr[_IDX[f"panel_z_{i}_{j}"]]), -1.0), 1.0) * allow
                for j in range(PANEL_GRID_TANGENTIAL_COUNT)
            )
        )

    return BladeParams(
        blade_count=blade_count,
        rib_bow_mid_m=float(arr[_IDX["rib_bow_mid_m"]]),
        rib_bow_tip_m=float(arr[_IDX["rib_bow_tip_m"]]),
        t_rib_hub_m=t_hub,
        t_rib_tip_m=t_tip,
        panel_offsets_m=tuple(grid),
        panel_thickness_nom_m=panel,
    )


def encode(params: BladeParams) -> np.ndarray:
    """:class:`BladeParams` → vector. ``decode(encode(p))`` round-trips any feasible ``p``."""
    out = np.zeros(N_DIMS, dtype=float)
    out[_IDX["rib_bow_mid_m"]] = params.rib_bow_mid_m
    out[_IDX["rib_bow_tip_m"]] = params.rib_bow_tip_m

    t_cap = rib_thickness_cap_m(params.blade_count)
    t_span = t_cap - _T_LO
    out[_IDX["t_rib_hub_k"]] = _c01((params.t_rib_hub_m - _T_LO) / t_span) if t_span > 0 else 0.0
    out[_IDX["t_rib_tip_k"]] = _c01((params.t_rib_tip_m - _T_LO) / t_span) if t_span > 0 else 0.0

    p_hi = max(min(params.t_rib_hub_m, params.t_rib_tip_m, _P_HI), _P_LO)
    p_span = p_hi - _P_LO
    out[_IDX["panel_thickness_k"]] = (
        _c01((params.panel_thickness_nom_m - _P_LO) / p_span) if p_span > 0 else 0.0
    )

    for i, u in enumerate(_radial_fracs()):
        t_rib_i = params.t_rib_hub_m * (1.0 - u) + params.t_rib_tip_m * u
        allow = max((t_rib_i - params.panel_thickness_nom_m) / 2.0, 0.0)
        for j in range(PANEL_GRID_TANGENTIAL_COUNT):
            frac = params.panel_offsets_m[i][j] / allow if allow > 0 else 0.0
            out[_IDX[f"panel_z_{i}_{j}"]] = min(max(frac, -1.0), 1.0)

    out[_IDX["blade_count"]] = float(BLADE_COUNTS.index(params.blade_count)) + 0.5
    return out
