"""BO design-vector ↔ :class:`BladeParams` codec (V1 redesign, **feasible by construction**).

The bijection between the optimizer's flat bounded float vector and the structured
:class:`~fanopt.geometry.blade.BladeParams`. Crucially, :func:`decode` maps every vector
to a **fold- and containment-feasible** blade, so the BO searches a space where ~100 % of
designs are valid instead of penalizing a mostly-infeasible space (uniform sampling of
raw physical thickness/offsets is almost never feasible). Concretely:

- **rib thickness** is a knob in ``[0, 1]`` mapped to ``[T_MIN, fold_cap(blade_count, bow)]``
  where ``fold_cap`` is the thickest per-layer z-envelope that still folds under the stack-
  height limit for that blade count — so the fold constraint holds by construction;
- **panel thickness** is a knob mapped to ``[P_MIN, min(rib)]`` — so ``panel ≤ rib`` always
  (ribbed); the **uniform** (no-rib) family instead bounds each node's ``|offset| + t/2`` by
  half the fold envelope so the wavy sheet nests by construction;
- **panel offsets** are knobs in ``[-1, 1]`` scaled by the *local* containable envelope
  ``(t_rib(r) − panel)/2`` (ribbed) or the nesting envelope (uniform).

``rib_bow`` (unconstrained) and ``blade_count`` / ``rib_mode`` (categorical) stay direct. Mass
is **not** forced here — it depends on the whole design and stays a purely soft check in the
objective. (The fold cap is intentionally mass-free: at the 22 cm trapezoid a mass ceiling
would collapse the rib range to a point for 10–12 blades. Mass-cap reconciliation is a
separate operator decision; see ``MAX_TOTAL_MASS_KG`` / audit N8.) ``decode(encode(p))``
round-trips any feasible ``p``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fanopt.geometry.blade import (
    BLADE_ROOT_RADIUS_M,
    FOLD_CLEARANCE_M,
    MAX_FOLDED_STACK_HEIGHT_M,
    PANEL_GRID_RADIAL_COUNT,
    PANEL_GRID_TANGENTIAL_COUNT,
    PANEL_THICKNESS_NOM_RANGE_M,
    RIB_BOW_INTERP_MODES,
    RIB_BOW_KNOT_COUNT,
    RIB_BOW_RANGE_M,
    RIB_MODES,
    RIB_THICKNESS_RANGE_M,
    RIB_TIP_RADIUS_M,
    BladeParams,
    panel_radial_stations,
    rib_meridian_extent_m,
)
from fanopt.geometry.schema import BLADE_COUNTS

__all__ = [
    "Var",
    "SEARCH_SPACE",
    "N_DIMS",
    "fold_envelope_cap_m",
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


# Upper bound on the rib meridian's out-of-plane extent for any codec-emittable design. Uniform
# Catmull-Rom overshoots the knot hull by up to ~25% of the knot span (empirically 1.25× at the
# range corners); 1.3× is a safe upper bound for the bow-blind (bow_extent_m=None) fold cap.
_MAX_MERIDIAN_EXTENT_M: float = 1.3 * (RIB_BOW_RANGE_M[1] - RIB_BOW_RANGE_M[0])


def fold_envelope_cap_m(blade_count: int, bow_extent_m: float | None = None) -> float:
    """Thickest **z-envelope** (m) per blade layer that still folds under the stack-height cap.

    The panel-aware per-layer footprint (:func:`fanopt.geometry.blade.blade_z_envelope_m`) that
    both the thickest rib (ribbed) and the panel's own thickness+wave spread (uniform) must stay
    under. Fold model (matches :func:`fanopt.geometry.blade.folded_stack_height_m`):
    ``(N−1)·(e+c) + bow + e ≤ MAX`` ⇒ ``e ≤ (MAX − (N−1)·c − bow) / N``. ``decode`` passes the
    design's EXACT ``bow_extent_m`` (incl. Catmull-Rom overshoot) so the cap is tight and
    feasible-by-construction; ``None`` falls back to ``_MAX_MERIDIAN_EXTENT_M`` — a true UPPER
    bound on the extent (so the cap is a genuine fold-safe lower bound, since it decreases in bow).
    """
    bow = _MAX_MERIDIAN_EXTENT_M if bow_extent_m is None else bow_extent_m
    return (MAX_FOLDED_STACK_HEIGHT_M - (blade_count - 1) * FOLD_CLEARANCE_M - bow) / blade_count


def rib_thickness_cap_m(blade_count: int, bow_extent_m: float | None = None) -> float:
    """Thickest rib (m) that **folds** under the stack-height cap for ``blade_count``.

    The bow-aware :func:`fold_envelope_cap_m` clamped to the rib range — so more blades ⇒ thinner
    ribs, and a ribbed design is fold-feasible by construction. Mass is deliberately NOT a factor
    (a soft objective check instead; see the module docstring).
    """
    lo, hi = RIB_THICKNESS_RANGE_M
    cap = min(fold_envelope_cap_m(blade_count, bow_extent_m), hi)
    return min(max(cap, lo), hi)


def _build_search_space() -> tuple[Var, ...]:
    vars_: list[Var] = [Var(f"rib_bow_k{i}", *RIB_BOW_RANGE_M) for i in range(RIB_BOW_KNOT_COUNT)]
    vars_.append(
        Var("rib_bow_interp", 0.0, float(len(RIB_BOW_INTERP_MODES)),
            kind="categorical", choices=RIB_BOW_INTERP_MODES)
    )
    vars_.append(
        Var("rib_mode", 0.0, float(len(RIB_MODES)), kind="categorical", choices=RIB_MODES)
    )
    vars_ += [
        Var("t_rib_hub_k", 0.0, 1.0),  # knob → [T_MIN, cap(blade_count)]
        Var("t_rib_tip_k", 0.0, 1.0),
    ]
    for i in range(PANEL_GRID_RADIAL_COUNT):
        for j in range(PANEL_GRID_TANGENTIAL_COUNT):
            vars_.append(Var(f"panel_z_{i}_{j}", -1.0, 1.0))  # fraction of containable envelope
    for i in range(PANEL_GRID_RADIAL_COUNT):
        for j in range(PANEL_GRID_TANGENTIAL_COUNT):
            vars_.append(Var(f"panel_thick_{i}_{j}", 0.0, 1.0))  # knob → [P_MIN, min(rib) at node]
    vars_.append(
        Var("blade_count", 0.0, float(len(BLADE_COUNTS)), kind="categorical", choices=BLADE_COUNTS)
    )
    return tuple(vars_)


SEARCH_SPACE: tuple[Var, ...] = _build_search_space()
N_DIMS: int = len(SEARCH_SPACE)

_IDX = {v.name: i for i, v in enumerate(SEARCH_SPACE)}
_T_LO = RIB_THICKNESS_RANGE_M[0]
_P_LO, _P_HI = PANEL_THICKNESS_NOM_RANGE_M
# Ribbed rib floor: a rib must be at least as thick as the panel floor to contain a panel
# (panel ≥ P_LO), so the ribbed rib range starts at max(rib_floor, panel_floor).
_T_LO_RIBBED = max(_T_LO, _P_LO)


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
    span = RIB_TIP_RADIUS_M - BLADE_ROOT_RADIUS_M
    return [(r - BLADE_ROOT_RADIUS_M) / span for r in panel_radial_stations()]


def _uniform_t_rib_m(thicks: tuple[tuple[float, ...], ...]) -> float:
    """Rib-thickness field value stored on a uniform (no-rib) design.

    Uniform blades have no rib rails, so ``t_rib`` is unused by the geometry — but it is a
    ``BladeParams`` field, so it must be set deterministically from the panel (the sheet's own
    thickness) for ``decode``/``encode`` to round-trip. Clamped into the valid rib range.
    """
    t = max(max(row) for row in thicks)
    return min(max(t, RIB_THICKNESS_RANGE_M[0]), RIB_THICKNESS_RANGE_M[1])


def decode(vec: np.ndarray) -> BladeParams:
    """Vector → **feasible** :class:`BladeParams` (fold + containment/nesting by construction).

    Ribbed and uniform (no-rib) designs are both feasible-by-construction: ribbed pins the panel
    inside the rib envelope; uniform bounds each node's ``|offset| + t/2`` by half the fold
    envelope cap so the wavy sheet still nests under the stack-height limit — the same hard fold
    constraint, letting design B wave as freely as design A.
    """
    arr = np.asarray(vec, dtype=float)
    if arr.shape != (N_DIMS,):
        raise ValueError(f"vector must have shape ({N_DIMS},); got {arr.shape}")

    blade_count: int = _decode_categorical(
        float(arr[_IDX["blade_count"]]), SEARCH_SPACE[_IDX["blade_count"]]
    )
    uniform = (
        _decode_categorical(float(arr[_IDX["rib_mode"]]), SEARCH_SPACE[_IDX["rib_mode"]])
        == "uniform"
    )
    # Bow decoded first so the fold cap below uses the design's EXACT bow extent.
    interp = _decode_categorical(
        float(arr[_IDX["rib_bow_interp"]]), SEARCH_SPACE[_IDX["rib_bow_interp"]]
    )
    knots = tuple(
        min(max(float(arr[_IDX[f"rib_bow_k{i}"]]), RIB_BOW_RANGE_M[0]), RIB_BOW_RANGE_M[1])
        for i in range(RIB_BOW_KNOT_COUNT)
    )
    e_cap = fold_envelope_cap_m(blade_count, rib_meridian_extent_m(knots, interp))

    offsets: list[tuple[float, ...]] = []
    thicks: list[tuple[float, ...]] = []

    if uniform:
        # No-rib sheet: thickness knob → [P_MIN, min(P_HI, e_cap)] (sheet fits its own layer);
        # offset knob → fraction of the nesting-containable half-envelope (e_cap − t)/2, so each
        # node's |offset| + t/2 ≤ e_cap/2 ⇒ the whole sheet's z-spread ≤ e_cap ⇒ nests by
        # construction. The tangential edges are UNPINNED in the geometry, so the sheet waves free.
        p_hi = max(min(_P_HI, e_cap), _P_LO)
        for _i in range(PANEL_GRID_RADIAL_COUNT):
            off_row: list[float] = []
            th_row: list[float] = []
            for j in range(PANEL_GRID_TANGENTIAL_COUNT):
                thick = _P_LO + _c01(float(arr[_IDX[f"panel_thick_{_i}_{j}"]])) * (p_hi - _P_LO)
                allow = max((e_cap - thick) / 2.0, 0.0)
                off = min(max(float(arr[_IDX[f"panel_z_{_i}_{j}"]]), -1.0), 1.0) * allow
                off_row.append(off)
                th_row.append(thick)
            offsets.append(tuple(off_row))
            thicks.append(tuple(th_row))
        thicks_t = tuple(thicks)
        t_rib = _uniform_t_rib_m(thicks_t)
        return BladeParams(
            blade_count=blade_count, rib_bow_knots_m=knots, rib_bow_interp=interp,
            t_rib_hub_m=t_rib, t_rib_tip_m=t_rib,
            panel_offsets_m=tuple(offsets), panel_thickness_m=thicks_t, uniform=True,
        )

    # Ribbed: rib thickness knob → [T_LO_RIBBED, bow-aware fold cap] (rib ≥ panel floor so it can
    # contain a panel); per node: thickness → [P_MIN, min(t_rib(r), P_HI)]; offset → fraction of
    # the local containable envelope (t_rib(r) − thickness)/2 ≥ 0. Two free grids ⇒ independent faces.
    t_cap = max(rib_thickness_cap_m(blade_count, rib_meridian_extent_m(knots, interp)), _T_LO_RIBBED)
    t_hub = _T_LO_RIBBED + _c01(float(arr[_IDX["t_rib_hub_k"]])) * (t_cap - _T_LO_RIBBED)
    t_tip = _T_LO_RIBBED + _c01(float(arr[_IDX["t_rib_tip_k"]])) * (t_cap - _T_LO_RIBBED)
    for i, u in enumerate(_radial_fracs()):
        t_rib_i = t_hub * (1.0 - u) + t_tip * u
        p_hi_i = max(min(t_rib_i, _P_HI), _P_LO)
        off_row = []
        th_row = []
        for j in range(PANEL_GRID_TANGENTIAL_COUNT):
            thick = _P_LO + _c01(float(arr[_IDX[f"panel_thick_{i}_{j}"]])) * (p_hi_i - _P_LO)
            allow = max((t_rib_i - thick) / 2.0, 0.0)
            off = min(max(float(arr[_IDX[f"panel_z_{i}_{j}"]]), -1.0), 1.0) * allow
            off_row.append(off)
            th_row.append(thick)
        offsets.append(tuple(off_row))
        thicks.append(tuple(th_row))

    return BladeParams(
        blade_count=blade_count, rib_bow_knots_m=knots, rib_bow_interp=interp,
        t_rib_hub_m=t_hub, t_rib_tip_m=t_tip,
        panel_offsets_m=tuple(offsets), panel_thickness_m=tuple(thicks), uniform=False,
    )


def encode(params: BladeParams) -> np.ndarray:
    """:class:`BladeParams` → vector. ``decode(encode(p))`` round-trips any feasible ``p``."""
    out = np.zeros(N_DIMS, dtype=float)
    for i in range(RIB_BOW_KNOT_COUNT):
        out[_IDX[f"rib_bow_k{i}"]] = params.rib_bow_knots_m[i]
    out[_IDX["rib_bow_interp"]] = float(RIB_BOW_INTERP_MODES.index(params.rib_bow_interp)) + 0.5
    out[_IDX["rib_mode"]] = float(RIB_MODES.index("uniform" if params.uniform else "ribbed")) + 0.5
    out[_IDX["blade_count"]] = float(BLADE_COUNTS.index(params.blade_count)) + 0.5

    e_cap = fold_envelope_cap_m(
        params.blade_count, rib_meridian_extent_m(params.rib_bow_knots_m, params.rib_bow_interp)
    )

    if params.uniform:
        # t_rib knobs are dead in uniform mode (decode ignores them and recomputes t_rib from
        # the panel), so leave them at 0. Invert the same per-node maps decode used.
        p_hi = max(min(_P_HI, e_cap), _P_LO)
        p_span = p_hi - _P_LO
        for i in range(PANEL_GRID_RADIAL_COUNT):
            for j in range(PANEL_GRID_TANGENTIAL_COUNT):
                thick = params.panel_thickness_m[i][j]
                out[_IDX[f"panel_thick_{i}_{j}"]] = _c01((thick - _P_LO) / p_span) if p_span > 0 else 0.0
                allow = max((e_cap - thick) / 2.0, 0.0)
                frac = params.panel_offsets_m[i][j] / allow if allow > 0 else 0.0
                out[_IDX[f"panel_z_{i}_{j}"]] = min(max(frac, -1.0), 1.0)
        return out

    # Same bow-aware cap decode used, so encode(decode(v)) round-trips the thickness knob.
    t_cap = max(rib_thickness_cap_m(
        params.blade_count, rib_meridian_extent_m(params.rib_bow_knots_m, params.rib_bow_interp)
    ), _T_LO_RIBBED)
    t_span = t_cap - _T_LO_RIBBED
    out[_IDX["t_rib_hub_k"]] = _c01((params.t_rib_hub_m - _T_LO_RIBBED) / t_span) if t_span > 0 else 0.0
    out[_IDX["t_rib_tip_k"]] = _c01((params.t_rib_tip_m - _T_LO_RIBBED) / t_span) if t_span > 0 else 0.0

    for i, u in enumerate(_radial_fracs()):
        t_rib_i = params.t_rib_hub_m * (1.0 - u) + params.t_rib_tip_m * u
        p_hi_i = max(min(t_rib_i, _P_HI), _P_LO)
        p_span_i = p_hi_i - _P_LO
        for j in range(PANEL_GRID_TANGENTIAL_COUNT):
            thick = params.panel_thickness_m[i][j]
            out[_IDX[f"panel_thick_{i}_{j}"]] = (
                _c01((thick - _P_LO) / p_span_i) if p_span_i > 0 else 0.0
            )
            allow = max((t_rib_i - thick) / 2.0, 0.0)
            frac = params.panel_offsets_m[i][j] / allow if allow > 0 else 0.0
            out[_IDX[f"panel_z_{i}_{j}"]] = min(max(frac, -1.0), 1.0)

    return out
