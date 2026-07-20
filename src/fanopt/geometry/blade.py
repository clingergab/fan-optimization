"""Lean surface-of-revolution blade parameterization (V1 redesign).

Replaces the 35-var plano-convex codec with the blade of
``docs/blade_architecture_redesign.md`` §7. The blade is a curved rib frame that is
a **surface of revolution about the pivot axis** — so all blades are congruent under
rotation and **nest by construction** when folded — carrying a free both-face aero
panel *inside* the rib thickness envelope.

Design variables (per §7.2, panel widened to a free displacement grid so the
optimizer discovers the panel shape *type* — camber, base→tip zigzag, louvers, … —
not just a camber magnitude):
- rib meridian ``z_rib(r)``: ``rib_bow_mid_m``, ``rib_bow_tip_m`` (the ``)`` generatrix,
  anchored ``z = 0`` at the boss),
- rib thickness ``t_rib(r)``: ``t_rib_hub_m``, ``t_rib_tip_m`` (thin at the hub — the
  fold constraint binds there),
- panel aero surface: a ``PANEL_GRID_RADIAL_COUNT × PANEL_GRID_TANGENTIAL_COUNT`` grid
  of surface-normal offsets ``panel_offsets_m`` (tangential edges pinned to the ribs),
- ``panel_thickness_nom_m`` (SIMP TO fills within this, bounded ``≤ t_rib``),
- ``blade_count`` ∈ {8, 10, 12} (outer bandit).

The geometry helpers and constraint margins here are **fast analytic proxies** for the
in-loop feasibility check. The authoritative fold/mass checks are the CAD swept-volume
boolean and the meshed solid (chunk 2+); this module is the cheap gate that keeps the
optimizer inside a buildable region. Locked constants come from
:mod:`fanopt.geometry.schema` — this module never re-declares them.

V2 direction (``V2_backlog.md`` ML track): the displacement grid is still a *bounded
basis*; the free-form/neural-implicit route removes the grid so the panel can take any
shape at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from fanopt.geometry.schema import (
    BLADE_COUNTS,
    HUB_RADIUS_M,
    INTER_BLADE_ANGLE_RAD,
    L_RIB_M,
    MAX_TOTAL_MASS_KG,
    PIVOT_BOSS_RADIUS_M,
    RHO_PETG_KG_PER_M3,
    RIB_BASE_WIDTH_M,
    RIB_TIP_WIDTH_M,
)

__all__ = [
    "RIB_BOW_RANGE_M",
    "RIB_THICKNESS_RANGE_M",
    "PANEL_THICKNESS_NOM_RANGE_M",
    "PANEL_GRID_RADIAL_COUNT",
    "PANEL_GRID_TANGENTIAL_COUNT",
    "PANEL_OFFSET_MAX_M",
    "PANEL_OFFSET_RANGE_M",
    "FOLD_CLEARANCE_M",
    "MAX_FOLDED_STACK_HEIGHT_M",
    "RIB_TIP_RADIUS_M",
    "BladeParams",
    "panel_radial_stations",
    "rib_z_at",
    "rib_thickness_at",
    "rib_width_at",
    "displacement_at",
    "layer_spacing_m",
    "folded_stack_height_m",
    "fold_margin_m",
    "containment_margin_m",
    "estimate_mass_kg",
    "mass_margin_kg",
    "feasible",
]

# --- New parameterization bounds (this module's own ranges, like the old
# envelope.py's CAMBER_RANGE_M — NOT locked schema constants). -----------------

RIB_BOW_RANGE_M: tuple[float, float] = (0.0, 0.030)
"""Out-of-plane rise of the ``)`` rib meridian at mid-span / tip (0–30 mm)."""

RIB_THICKNESS_RANGE_M: tuple[float, float] = (0.002, 0.006)
"""Rib z-thickness envelope at hub / tip. 2 mm floor = FDM minimum feature."""

PANEL_THICKNESS_NOM_RANGE_M: tuple[float, float] = (0.0012, 0.003)
"""Nominal panel membrane thickness (1.2–3 mm). Held ``≤ t_rib`` by containment."""

PANEL_GRID_RADIAL_COUNT: int = 4
"""Radial control rows of the panel displacement grid (base→tip; enough for steps)."""

PANEL_GRID_TANGENTIAL_COUNT: int = 3
"""Interior tangential control points per row. The two rib edges are pinned to 0."""

PANEL_OFFSET_MAX_M: float = (RIB_THICKNESS_RANGE_M[1] - PANEL_THICKNESS_NOM_RANGE_M[0]) / 2.0
"""±2.4 mm — the largest surface offset that can ever fit inside a rib slab. The
local containment constraint (thinner rib / thicker panel) is tighter and penalized."""

PANEL_OFFSET_RANGE_M: tuple[float, float] = (-PANEL_OFFSET_MAX_M, PANEL_OFFSET_MAX_M)

FOLD_CLEARANCE_M: float = 0.0004
"""0.4 mm per-interface fold clearance for PETG FDM (§4.5)."""

MAX_FOLDED_STACK_HEIGHT_M: float = 0.035
"""Ergonomic bound on the folded-bundle thickness (this module's design bound, not a
locked schema constant). The fan z-stacks like a deck: layer spacing = thickest rib +
clearance, so the folded stack (and the deployed z-stagger) is ``N × layer_spacing``.
Thick ribs → fat bundle; this is the pressure that keeps ribs thin."""

RIB_TIP_RADIUS_M: float = HUB_RADIUS_M + L_RIB_M
"""0.185 m — outer rib radius (hub + rib radial extent)."""

_MID_RADIUS_M: float = HUB_RADIUS_M + 0.5 * L_RIB_M
_MARGIN_SAMPLES: int = 21  # radial sampling for the nesting constraint


@dataclass(frozen=True)
class BladeParams:
    """Lean surface-of-revolution blade design (§7.2). Validates ranges on construction.

    Range validation guards the search box; *feasibility* (nesting / containment /
    mass) is a separate soft check via :func:`feasible` — the optimizer may propose a
    range-valid but infeasible design and get penalized, exactly as the old loop did.
    """

    blade_count: int
    rib_bow_mid_m: float
    rib_bow_tip_m: float
    t_rib_hub_m: float
    t_rib_tip_m: float
    panel_offsets_m: tuple[tuple[float, ...], ...]
    panel_thickness_nom_m: float

    def __post_init__(self) -> None:
        if self.blade_count not in BLADE_COUNTS:
            raise ValueError(f"blade_count must be one of {BLADE_COUNTS}, got {self.blade_count}")
        self._check("rib_bow_mid_m", self.rib_bow_mid_m, RIB_BOW_RANGE_M)
        self._check("rib_bow_tip_m", self.rib_bow_tip_m, RIB_BOW_RANGE_M)
        self._check("t_rib_hub_m", self.t_rib_hub_m, RIB_THICKNESS_RANGE_M)
        self._check("t_rib_tip_m", self.t_rib_tip_m, RIB_THICKNESS_RANGE_M)
        self._check(
            "panel_thickness_nom_m", self.panel_thickness_nom_m, PANEL_THICKNESS_NOM_RANGE_M
        )
        self._check_grid()

    @staticmethod
    def _check(name: str, value: float, rng: tuple[float, float]) -> None:
        lo, hi = rng
        if not (lo <= value <= hi):
            raise ValueError(f"{name} = {value} outside range [{lo}, {hi}]")

    def _check_grid(self) -> None:
        rows, cols = PANEL_GRID_RADIAL_COUNT, PANEL_GRID_TANGENTIAL_COUNT
        if len(self.panel_offsets_m) != rows:
            raise ValueError(
                f"panel_offsets_m must have {rows} radial rows, got {len(self.panel_offsets_m)}"
            )
        lo, hi = PANEL_OFFSET_RANGE_M
        for i, row in enumerate(self.panel_offsets_m):
            if len(row) != cols:
                raise ValueError(
                    f"panel_offsets_m[{i}] must have {cols} tangential points, got {len(row)}"
                )
            for j, val in enumerate(row):
                if not (lo <= val <= hi):
                    raise ValueError(f"panel_offsets_m[{i}][{j}] = {val} outside [{lo}, {hi}]")

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict for the BO ledger / serialisation."""
        return {
            "blade_count": self.blade_count,
            "rib_bow_mid_m": self.rib_bow_mid_m,
            "rib_bow_tip_m": self.rib_bow_tip_m,
            "t_rib_hub_m": self.t_rib_hub_m,
            "t_rib_tip_m": self.t_rib_tip_m,
            "panel_offsets_m": [list(row) for row in self.panel_offsets_m],
            "panel_thickness_nom_m": self.panel_thickness_nom_m,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BladeParams:
        """Inverse of :meth:`to_dict`. Validates on construction."""
        return cls(
            blade_count=int(d["blade_count"]),
            rib_bow_mid_m=float(d["rib_bow_mid_m"]),
            rib_bow_tip_m=float(d["rib_bow_tip_m"]),
            t_rib_hub_m=float(d["t_rib_hub_m"]),
            t_rib_tip_m=float(d["t_rib_tip_m"]),
            panel_offsets_m=tuple(tuple(float(x) for x in row) for row in d["panel_offsets_m"]),
            panel_thickness_nom_m=float(d["panel_thickness_nom_m"]),
        )


def _radial_frac(r: float) -> float:
    """Radial position normalized to [0, 1] over the rib span (hub → tip), clamped."""
    return min(max((r - HUB_RADIUS_M) / L_RIB_M, 0.0), 1.0)


def panel_radial_stations() -> list[float]:
    """Radii of the panel grid's radial control rows (hub → tip, evenly spaced)."""
    n = PANEL_GRID_RADIAL_COUNT
    return [HUB_RADIUS_M + (RIB_TIP_RADIUS_M - HUB_RADIUS_M) * i / (n - 1) for i in range(n)]


def rib_z_at(params: BladeParams, r: float) -> float:
    """Out-of-plane height of the ``)`` rib meridian at radius ``r`` (0 at the boss).

    Piecewise-linear through (hub, 0), (mid, ``rib_bow_mid_m``), (tip, ``rib_bow_tip_m``).
    Any generatrix revolved about the pivot axis nests; the shape only sets aero/fold
    envelope, so a two-segment curve is enough for the lean blade.
    """
    r = min(max(r, HUB_RADIUS_M), RIB_TIP_RADIUS_M)
    if r <= _MID_RADIUS_M:
        t = (r - HUB_RADIUS_M) / (_MID_RADIUS_M - HUB_RADIUS_M)
        return params.rib_bow_mid_m * t
    t = (r - _MID_RADIUS_M) / (RIB_TIP_RADIUS_M - _MID_RADIUS_M)
    return params.rib_bow_mid_m * (1.0 - t) + params.rib_bow_tip_m * t


def rib_thickness_at(params: BladeParams, r: float) -> float:
    """Rib z-thickness at ``r`` — linear hub→tip envelope (the fold-relevant ceiling)."""
    u = _radial_frac(r)
    return params.t_rib_hub_m * (1.0 - u) + params.t_rib_tip_m * u


def rib_width_at(r: float) -> float:
    """Rib tangential width at ``r`` — locked linear taper (4 mm base → 6 mm tip)."""
    u = _radial_frac(r)
    return RIB_BASE_WIDTH_M * (1.0 - u) + RIB_TIP_WIDTH_M * u


def displacement_at(params: BladeParams, r: float, v: float) -> float:
    """Panel mean-surface offset (from the rib mean surface) at radius ``r`` and
    tangential ``v`` ∈ [-1, 1]. Bilinear over the offset grid with the two rib edges
    (v = ±1) pinned to 0, so the panel blends into the frame. This free grid is what
    lets the optimizer discover camber, a base→tip zigzag, louvers, etc.
    """
    rows, cols = PANEL_GRID_RADIAL_COUNT, PANEL_GRID_TANGENTIAL_COUNT
    v = min(max(v, -1.0), 1.0)
    u = _radial_frac(r)
    fr = u * (rows - 1)
    i0 = min(int(fr), rows - 2)
    tr = fr - i0
    # Tangential stations include the pinned edges: cols + 2 evenly spaced over [-1, 1].
    s = (v + 1.0) / 2.0 * (cols + 1)
    j0 = min(int(s), cols)
    tt = s - j0
    top_row = (0.0, *params.panel_offsets_m[i0], 0.0)
    bot_row = (0.0, *params.panel_offsets_m[i0 + 1], 0.0)
    top = top_row[j0] * (1.0 - tt) + top_row[j0 + 1] * tt
    bot = bot_row[j0] * (1.0 - tt) + bot_row[j0 + 1] * tt
    return top * (1.0 - tr) + bot * tr


def _radial_samples() -> list[float]:
    step = L_RIB_M / (_MARGIN_SAMPLES - 1)
    return [HUB_RADIUS_M + step * k for k in range(_MARGIN_SAMPLES)]


def layer_spacing_m(params: BladeParams) -> float:
    """Z-stack layer spacing set by the boss = thickest rib + clearance.

    The fan folds by z-stacking (a deck), so adjacent blades sit one layer apart on
    the pin. The spacing must clear the thickest rib section (``t_rib`` is linear, so
    the max is at an endpoint) at every radius, since the blades are rigid.
    """
    return max(params.t_rib_hub_m, params.t_rib_tip_m) + FOLD_CLEARANCE_M


def folded_stack_height_m(params: BladeParams) -> float:
    """Folded-bundle thickness ``blade_count × layer_spacing`` (= the deployed z-stagger)."""
    return params.blade_count * layer_spacing_m(params)


def fold_margin_m(params: BladeParams) -> float:
    """``MAX_FOLDED_STACK_HEIGHT_M − folded_stack_height``. ≥ 0 ⇒ folds acceptably thin.

    The real fold cost is stack height, not hub packing — thick ribs make a fat bundle.
    The CAD swept-volume boolean is the authoritative no-collision check through the swing.
    """
    return MAX_FOLDED_STACK_HEIGHT_M - folded_stack_height_m(params)


def containment_margin_m(params: BladeParams) -> float:
    """Min over grid nodes of ``(t_rib(r) − panel_nom)/2 − |offset|``. ≥ 0 ⇒ contained.

    The cambered/undulating panel membrane must fit inside the rib thickness envelope
    so it can never poke out and strike a neighbour (§4.1 rule 2). Bilinear extrema sit
    at the control nodes, so checking the nodes is exact. Couples panel relief to rib
    thickness: bigger undulations need a thicker rib — traded against nesting + mass.
    """
    margins: list[float] = []
    for r, row in zip(panel_radial_stations(), params.panel_offsets_m):
        allow = (rib_thickness_at(params, r) - params.panel_thickness_nom_m) / 2.0
        for offset in row:
            margins.append(allow - abs(offset))
    return min(margins)


def estimate_mass_kg(params: BladeParams) -> float:
    """Coarse assembly-mass estimate (kg): (2 ribs + panel + boss) × blade_count × ρ_PETG.

    A fast analytic proxy for the mass cap in-loop; the meshed CAD solid is
    authoritative. Panel tangential width is one blade slot minus its two edge ribs;
    the small extra area from panel undulation is neglected in this proxy.
    """
    samples = _radial_samples()
    dr = L_RIB_M / (_MARGIN_SAMPLES - 1)
    rib_vol = 0.0
    panel_vol = 0.0
    for r in samples:
        w_rib = rib_width_at(r)
        rib_vol += 2.0 * w_rib * rib_thickness_at(params, r) * dr
        w_panel = max(0.0, r * INTER_BLADE_ANGLE_RAD - 2.0 * w_rib)
        panel_vol += w_panel * params.panel_thickness_nom_m * dr
    boss_vol = math.pi * PIVOT_BOSS_RADIUS_M**2 * params.t_rib_hub_m
    vol_per_blade = rib_vol + panel_vol + boss_vol
    return vol_per_blade * params.blade_count * RHO_PETG_KG_PER_M3


def mass_margin_kg(params: BladeParams) -> float:
    """``MAX_TOTAL_MASS_KG − estimate_mass_kg``. ≥ 0 ⇒ under the C9 100 g cap."""
    return MAX_TOTAL_MASS_KG - estimate_mass_kg(params)


def feasible(params: BladeParams) -> bool:
    """True iff the fold (stack-height), containment, and mass proxies are all satisfied."""
    return (
        fold_margin_m(params) >= 0.0
        and containment_margin_m(params) >= 0.0
        and mass_margin_kg(params) >= 0.0
    )
