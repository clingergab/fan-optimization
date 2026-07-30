"""Lean surface-of-revolution blade parameterization (V1 redesign).

Replaces the 35-var plano-convex codec with the blade of
``docs/blade_architecture_redesign.md`` §7. The blade is a curved rib frame that is
a **surface of revolution about the pivot axis** — so all blades are congruent under
rotation and **nest by construction** when folded — carrying a free both-face aero
panel *inside* the rib thickness envelope.

Design variables (per §7.2, panel widened to a free displacement grid so the
optimizer discovers the panel shape *type* — camber, base→tip zigzag, louvers, … —
not just a camber magnitude):
- rib meridian ``z_rib(r)``: ``rib_bow_knots_m`` (K radial knots, anchored ``z = 0`` at the
  boss) with ``rib_bow_interp`` (linear pleats vs smooth camber) — the ``)`` generatrix,
- rib thickness ``t_rib(r)``: ``t_rib_hub_m``, ``t_rib_tip_m`` (thin at the hub — the
  fold constraint binds there),
- panel aero surface: two ``PANEL_GRID_RADIAL_COUNT × PANEL_GRID_TANGENTIAL_COUNT`` grids —
  mean-surface offsets ``panel_offsets_m`` (tangential edges pinned to the ribs) and
  membrane thickness ``panel_thickness_m`` — which together give the **two faces independent
  shape** (``top = mean + t/2``, ``bot = mean − t/2``; each ``≤ t_rib`` for folding),
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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from fanopt.geometry.schema import (
    BLADE_COUNTS,
    MAX_TOTAL_MASS_KG,
    PIVOT_BOSS_RADIUS_M,
    RHO_PETG_KG_PER_M3,
    RIB_BASE_WIDTH_M,
    RIB_TIP_WIDTH_M,
)

__all__ = [
    "RIB_BOW_RANGE_M",
    "RIB_BOW_KNOT_COUNT",
    "RIB_BOW_INTERP_MODES",
    "RIB_MODES",
    "RIB_THICKNESS_RANGE_M",
    "PANEL_THICKNESS_NOM_RANGE_M",
    "PANEL_GRID_RADIAL_COUNT",
    "PANEL_GRID_TANGENTIAL_COUNT",
    "PANEL_OFFSET_MAX_M",
    "PANEL_OFFSET_RANGE_M",
    "FOLD_CLEARANCE_M",
    "MAX_FOLDED_STACK_HEIGHT_M",
    "RIB_TIP_RADIUS_M",
    "BLADE_ROOT_RADIUS_M",
    "BLADE_COUNT",
    "ROOT_HALF_WIDTH_M",
    "TIP_HALF_WIDTH_M",
    "BladeParams",
    "half_width_at",
    "panel_radial_stations",
    "rib_bow_stations",
    "rib_z_at",
    "rib_meridian_extent_m",
    "rib_thickness_at",
    "rib_width_at",
    "displacement_at",
    "panel_thickness_at",
    "blade_z_envelope_m",
    "layer_spacing_m",
    "folded_rib_bow_extent_m",
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
"""Out-of-plane rise of the ``)`` rib meridian at each knot (0–30 mm)."""

RIB_BOW_KNOT_COUNT: int = 5
"""Free radial control knots of the rib meridian (hub pinned to 0, knots hub→tip).

The meridian is a surface of revolution, so **any** radial profile nests when folded —
unlike the panel grid (fold-limited to sub-mm). More knots let the optimizer sculpt
radial camber, S-curves, or sharp pleats/zigzags at full amplitude, not just one cup.
``2`` reproduces the original (mid, tip) two-segment meridian."""

RIB_BOW_INTERP_MODES: tuple[str, ...] = ("linear", "smooth")
"""Radial interpolation between meridian knots: ``linear`` (crisp pleats / zigzag) or
``smooth`` (Catmull-Rom dished camber). A categorical BO knob — the optimizer tries both."""

RIB_MODES: tuple[str, ...] = ("ribbed", "uniform")
"""Blade topology family (categorical BO knob). ``ribbed`` = a thick rib rail down each
tangential edge carrying a thinner contained panel between them (design A). ``uniform`` =
a single-thickness sheet with NO rib rails, free to wave across its whole width (design B);
nesting is governed by the fold stack-height constraint instead of rib containment."""

RIB_THICKNESS_RANGE_M: tuple[float, float] = (0.002, 0.012)
"""Rib z-thickness envelope at hub / tip. 2 mm floor = FDM minimum feature. Widened to
12 mm (2026-07-22) so a thick rib frame can give the panel real room to sculpt (thick
ribs ⇄ chunkier fold + more mass — see MAX_FOLDED_STACK_HEIGHT_M / MAX_TOTAL_MASS_KG)."""

PANEL_THICKNESS_NOM_RANGE_M: tuple[float, float] = (0.003, 0.010)
"""Nominal panel membrane thickness. Floor raised to 3 mm (2026-07-29 trapezoid redesign):
3 mm is the printable base-panel minimum for both design families (ribbed panel-between-rails
and the uniform no-rib sheet). Held ``≤ t_rib`` by containment in ribbed mode; in uniform mode
the sheet IS the blade, so the fold stack-height constraint bounds it instead."""

PANEL_GRID_RADIAL_COUNT: int = 4
"""Radial control rows of the panel displacement grid (base→tip; enough for steps)."""

PANEL_GRID_TANGENTIAL_COUNT: int = 3
"""Interior tangential control points per row. The two rib edges are pinned to 0."""

PANEL_OFFSET_MAX_M: float = (RIB_THICKNESS_RANGE_M[1] - PANEL_THICKNESS_NOM_RANGE_M[0]) / 2.0
"""Largest surface offset that can ever fit inside a rib slab (auto-derived from the rib +
panel ranges). The local containment constraint (thinner rib / thicker panel) is tighter
and penalized."""

PANEL_OFFSET_RANGE_M: tuple[float, float] = (-PANEL_OFFSET_MAX_M, PANEL_OFFSET_MAX_M)

FOLD_CLEARANCE_M: float = 0.0004
"""0.4 mm per-interface fold clearance for PETG FDM (§4.5)."""

MAX_FOLDED_STACK_HEIGHT_M: float = 0.090
"""Ergonomic bound on the folded-bundle thickness (this module's design bound, not a
locked schema constant). Widened 35 → 90 mm (2026-07-22) so thick ribs are fold-feasible
— a deliberately chunkier folded fan, the price of giving the panel room to sculpt. The
fan z-stacks like a deck: layer spacing = thickest rib +
clearance, so the folded stack (and the deployed z-stagger) is ``N × layer_spacing``.
Thick ribs → fat bundle; this is the pressure that keeps ribs thin."""

RIB_TIP_RADIUS_M: float = 0.220
"""0.220 m — blade tip radius = 22 cm blade length (2026-07-29 trapezoid redesign).

The live blade length is governed here, NOT by ``schema.L_BLADE_M`` (a dead no-op for live
geometry) nor ``schema.L_RIB_M`` (which still drives the legacy plano-convex / rib-TO stack
at the retired 185 mm; decoupled here to keep this length change inside the live blade)."""

BLADE_ROOT_RADIUS_M: float = 0.0
"""Root radius. The trapezoid planform starts at the pivot centre (r = 0) so the blade root
overlaps the pivot boss and unions into ONE solid (not a tangent-but-separate part)."""

BLADE_COUNT: int = 12
"""Blade count — FIXED at 12 (operator, 2026-07-29 trapezoid redesign; ADR-0005). No longer a
BO variable: the deployed span (12 × 13.3° × 22 cm ≈ 43 cm) requires the count be fixed, so the
codec dropped the ``blade_count`` categorical and every design is a 12-blade fan. 12 is a member
of ``schema.BLADE_COUNTS`` (still the validation set for :class:`BladeParams`)."""

ROOT_HALF_WIDTH_M: float = PIVOT_BOSS_RADIUS_M
"""6 mm — root tangential half-width. Root width = 12 mm = boss diameter, so the root meets
the boss exactly and the union is a single solid."""

TIP_HALF_WIDTH_M: float = 0.0255
"""25.5 mm — tip tangential half-width. Tip width = 51 mm ≈ flush-tile width at the 13.3°
deploy pitch over a 220 mm span, so adjacent deployed tips sit gapless."""

_BLADE_SPAN_M: float = RIB_TIP_RADIUS_M - BLADE_ROOT_RADIUS_M

_MARGIN_SAMPLES: int = 21  # radial sampling for the nesting constraint


@dataclass(frozen=True)
class BladeParams:
    """Lean surface-of-revolution blade design (§7.2). Validates ranges on construction.

    Range validation guards the search box; *feasibility* (nesting / containment /
    mass) is a separate soft check via :func:`feasible` — the optimizer may propose a
    range-valid but infeasible design and get penalized, exactly as the old loop did.
    """

    blade_count: int
    rib_bow_knots_m: tuple[float, ...]
    rib_bow_interp: str
    t_rib_hub_m: float
    t_rib_tip_m: float
    panel_offsets_m: tuple[tuple[float, ...], ...]
    panel_thickness_m: tuple[tuple[float, ...], ...]
    uniform: bool = False
    """``False`` = ribbed (A): rib rails + contained panel. ``True`` = uniform (B): a single
    no-rib sheet that waves freely and nests via the fold constraint (see :data:`RIB_MODES`)."""

    def __post_init__(self) -> None:
        if self.blade_count not in BLADE_COUNTS:
            raise ValueError(f"blade_count must be one of {BLADE_COUNTS}, got {self.blade_count}")
        if len(self.rib_bow_knots_m) != RIB_BOW_KNOT_COUNT:
            raise ValueError(
                f"rib_bow_knots_m must have {RIB_BOW_KNOT_COUNT} knots, "
                f"got {len(self.rib_bow_knots_m)}"
            )
        for i, k in enumerate(self.rib_bow_knots_m):
            self._check(f"rib_bow_knots_m[{i}]", k, RIB_BOW_RANGE_M)
        if self.rib_bow_interp not in RIB_BOW_INTERP_MODES:
            raise ValueError(
                f"rib_bow_interp must be one of {RIB_BOW_INTERP_MODES}, got {self.rib_bow_interp!r}"
            )
        self._check("t_rib_hub_m", self.t_rib_hub_m, RIB_THICKNESS_RANGE_M)
        self._check("t_rib_tip_m", self.t_rib_tip_m, RIB_THICKNESS_RANGE_M)
        self._check_grid("panel_offsets_m", self.panel_offsets_m, PANEL_OFFSET_RANGE_M)
        self._check_grid("panel_thickness_m", self.panel_thickness_m, PANEL_THICKNESS_NOM_RANGE_M)

    @staticmethod
    def _check(name: str, value: float, rng: tuple[float, float]) -> None:
        lo, hi = rng
        if not (lo <= value <= hi):
            raise ValueError(f"{name} = {value} outside range [{lo}, {hi}]")

    @staticmethod
    def _check_grid(
        name: str, grid: tuple[tuple[float, ...], ...], rng: tuple[float, float]
    ) -> None:
        rows, cols = PANEL_GRID_RADIAL_COUNT, PANEL_GRID_TANGENTIAL_COUNT
        if len(grid) != rows:
            raise ValueError(f"{name} must have {rows} radial rows, got {len(grid)}")
        lo, hi = rng
        for i, row in enumerate(grid):
            if len(row) != cols:
                raise ValueError(f"{name}[{i}] must have {cols} tangential points, got {len(row)}")
            for j, val in enumerate(row):
                if not (lo <= val <= hi):
                    raise ValueError(f"{name}[{i}][{j}] = {val} outside [{lo}, {hi}]")

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict for the BO ledger / serialisation."""
        return {
            "blade_count": self.blade_count,
            "rib_bow_knots_m": list(self.rib_bow_knots_m),
            "rib_bow_interp": self.rib_bow_interp,
            "t_rib_hub_m": self.t_rib_hub_m,
            "t_rib_tip_m": self.t_rib_tip_m,
            "panel_offsets_m": [list(row) for row in self.panel_offsets_m],
            "panel_thickness_m": [list(row) for row in self.panel_thickness_m],
            "uniform": self.uniform,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BladeParams:
        """Inverse of :meth:`to_dict`. Validates on construction.

        Back-compatible with pre-enrichment schemas: the two-knot meridian
        (``rib_bow_mid_m`` / ``rib_bow_tip_m``) is resampled onto the knot stations, and a
        scalar ``panel_thickness_nom_m`` is broadcast to a uniform thickness grid — so old
        ``pareto.json`` / ledger rows still load.
        """
        if "rib_bow_knots_m" in d:
            knots = tuple(float(k) for k in d["rib_bow_knots_m"])
            interp = str(d.get("rib_bow_interp", "linear"))
        else:
            mid, tip = float(d["rib_bow_mid_m"]), float(d["rib_bow_tip_m"])
            knots = tuple(_legacy_rib_z(mid, tip, r) for r in rib_bow_stations())
            interp = "linear"
        if "panel_thickness_m" in d:
            thickness = tuple(tuple(float(x) for x in row) for row in d["panel_thickness_m"])
        else:
            t = float(d["panel_thickness_nom_m"])
            thickness = tuple(
                (t,) * PANEL_GRID_TANGENTIAL_COUNT for _ in range(PANEL_GRID_RADIAL_COUNT)
            )
        return cls(
            blade_count=int(d["blade_count"]),
            rib_bow_knots_m=knots,
            rib_bow_interp=interp,
            t_rib_hub_m=float(d["t_rib_hub_m"]),
            t_rib_tip_m=float(d["t_rib_tip_m"]),
            panel_offsets_m=tuple(tuple(float(x) for x in row) for row in d["panel_offsets_m"]),
            panel_thickness_m=thickness,
            uniform=bool(d.get("uniform", False)),
        )


def _radial_frac(r: float) -> float:
    """Radial position normalized to [0, 1] over the blade span (root r=0 → tip), clamped."""
    return min(max((r - BLADE_ROOT_RADIUS_M) / _BLADE_SPAN_M, 0.0), 1.0)


def half_width_at(r: float) -> float:
    """Trapezoid tangential half-width at radius ``r`` (linear root→tip).

    The planform is a Cartesian trapezoid, NOT a pie-slice sector: half-width grows linearly
    from :data:`ROOT_HALF_WIDTH_M` at the root (r=0, = boss radius) to :data:`TIP_HALF_WIDTH_M`
    at the tip. A point at tangential fraction ``v ∈ [-1, 1]`` sits at ``y = v · half_width(r)``,
    ``x = r`` — decoupling width from ``r · angle`` (the retired sector assumption).
    """
    u = _radial_frac(r)
    return ROOT_HALF_WIDTH_M + (TIP_HALF_WIDTH_M - ROOT_HALF_WIDTH_M) * u


def panel_radial_stations() -> list[float]:
    """Radii of the panel grid's radial control rows (root → tip, evenly spaced)."""
    n = PANEL_GRID_RADIAL_COUNT
    return [BLADE_ROOT_RADIUS_M + _BLADE_SPAN_M * i / (n - 1) for i in range(n)]


def rib_bow_stations() -> list[float]:
    """Radii of the meridian's free knots — evenly spaced root(exclusive)→tip.

    The root itself is pinned to ``z = 0`` (the blade meets the boss), so the ``K`` knots
    sit at ``root + (i+1)/K · span`` for ``i in 0..K-1`` (the last is the tip).
    """
    k = RIB_BOW_KNOT_COUNT
    return [BLADE_ROOT_RADIUS_M + _BLADE_SPAN_M * (i + 1) / k for i in range(k)]


def _legacy_rib_z(mid: float, tip: float, r: float) -> float:
    """Old two-segment meridian z at ``r``: (root, 0) → (mid_radius, ``mid``) → (tip, ``tip``).

    Used only by :meth:`BladeParams.from_dict` to resample pre-enrichment designs onto the
    current knot stations.
    """
    mid_radius = BLADE_ROOT_RADIUS_M + 0.5 * _BLADE_SPAN_M
    r = min(max(r, BLADE_ROOT_RADIUS_M), RIB_TIP_RADIUS_M)
    if r <= mid_radius:
        return mid * (r - BLADE_ROOT_RADIUS_M) / (mid_radius - BLADE_ROOT_RADIUS_M)
    t = (r - mid_radius) / (RIB_TIP_RADIUS_M - mid_radius)
    return mid * (1.0 - t) + tip * t


def _catmull_rom(ys: list[float], seg: int, t: float) -> float:
    """Uniform Catmull-Rom value in segment ``[seg, seg+1]`` at local ``t`` ∈ [0, 1].

    Endpoints duplicate the boundary knot (clamped tangents). Knots are evenly spaced, so
    the uniform form applies.
    """
    n = len(ys)
    p0 = ys[seg - 1] if seg - 1 >= 0 else ys[seg]
    p1 = ys[seg]
    p2 = ys[seg + 1]
    p3 = ys[seg + 2] if seg + 2 < n else ys[seg + 1]
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t
    )


def _meridian_z(knots: Sequence[float], interp: str, r: float) -> float:
    """Meridian height at ``r`` from knots + interp alone (no :class:`BladeParams`)."""
    r = min(max(r, BLADE_ROOT_RADIUS_M), RIB_TIP_RADIUS_M)
    xs = [BLADE_ROOT_RADIUS_M, *rib_bow_stations()]
    ys = [0.0, *knots]
    # Locate the segment [i, i+1] containing r (xs is strictly increasing).
    seg = 0
    while seg < len(xs) - 2 and r > xs[seg + 1]:
        seg += 1
    t = (r - xs[seg]) / (xs[seg + 1] - xs[seg])
    if interp == "smooth":
        return _catmull_rom(ys, seg, t)
    return ys[seg] * (1.0 - t) + ys[seg + 1] * t


def rib_z_at(params: BladeParams, r: float) -> float:
    """Out-of-plane height of the ``)`` rib meridian at radius ``r`` (0 at the boss).

    Interpolates ``rib_bow_interp`` (``linear`` or ``smooth`` Catmull-Rom) through the hub
    (pinned to 0) and the ``rib_bow_knots_m`` at :func:`rib_bow_stations`. Any generatrix
    revolved about the pivot axis nests when folded, so the meridian is the fold-free lever
    for radial camber / pleats — amplitude bounded only by the bow range and mass.
    """
    return _meridian_z(params.rib_bow_knots_m, params.rib_bow_interp, r)


def rib_meridian_extent_m(knots: Sequence[float], interp: str) -> float:
    """Peak-to-trough extent ``max z(r) − min z(r)`` of the meridian from knots + interp.

    Independent of :class:`BladeParams`, so the codec can compute a design's exact bow extent
    (including Catmull-Rom overshoot) to cap rib thickness bow-aware *before* a full
    ``BladeParams`` exists. This is the true fold-relevant out-of-plane extent.
    """
    z = [_meridian_z(knots, interp, r) for r in _radial_samples()]
    return max(z) - min(z)


def rib_thickness_at(params: BladeParams, r: float) -> float:
    """Rib z-thickness at ``r`` — linear hub→tip envelope (the fold-relevant ceiling)."""
    u = _radial_frac(r)
    return params.t_rib_hub_m * (1.0 - u) + params.t_rib_tip_m * u


def rib_width_at(r: float) -> float:
    """Rib tangential width at ``r`` — locked linear taper (4 mm base → 6 mm tip)."""
    u = _radial_frac(r)
    return RIB_BASE_WIDTH_M * (1.0 - u) + RIB_TIP_WIDTH_M * u


def displacement_at(params: BladeParams, r: float, v: float) -> float:
    """Panel mean-surface offset (from the meridian mean surface) at radius ``r`` and
    tangential ``v`` ∈ [-1, 1]. Bilinear over the offset grid. This free grid is what lets the
    optimizer discover camber, a base→tip zigzag, louvers, etc.

    **Ribbed** (``uniform=False``): the two rib edges (v = ±1) are pinned to 0 so the panel
    blends into the frame. **Uniform** (``uniform=True``): there is no rib frame, so the edges
    are UNPINNED — padded by the nearest interior column instead of 0 — so the no-rib sheet can
    wave/angle right out to its tangential edges, the same freedom the ribbed panel has inside.
    """
    rows, cols = PANEL_GRID_RADIAL_COUNT, PANEL_GRID_TANGENTIAL_COUNT
    v = min(max(v, -1.0), 1.0)
    u = _radial_frac(r)
    fr = u * (rows - 1)
    i0 = min(int(fr), rows - 2)
    tr = fr - i0
    # Tangential stations: cols + 2 points evenly spaced over [-1, 1] (interior + two edges).
    s = (v + 1.0) / 2.0 * (cols + 1)
    j0 = min(int(s), cols)
    tt = s - j0
    ra, rb = params.panel_offsets_m[i0], params.panel_offsets_m[i0 + 1]
    if params.uniform:
        top_row = (ra[0], *ra, ra[-1])  # edges follow the nearest interior column (unpinned)
        bot_row = (rb[0], *rb, rb[-1])
    else:
        top_row = (0.0, *ra, 0.0)  # edges pinned to the rib frame
        bot_row = (0.0, *rb, 0.0)
    top = top_row[j0] * (1.0 - tt) + top_row[j0 + 1] * tt
    bot = bot_row[j0] * (1.0 - tt) + bot_row[j0 + 1] * tt
    return top * (1.0 - tr) + bot * tr


def panel_thickness_at(params: BladeParams, r: float, v: float) -> float:
    """Panel membrane thickness at radius ``r`` and tangential ``v`` ∈ [-1, 1].

    Bilinear over ``panel_thickness_m``, tangential edges padded by the nearest column (NOT
    pinned to 0 like the offset grid — thickness stays positive). Paired with
    :func:`displacement_at` (the mean-surface offset), a free thickness grid gives the two
    faces **independent** shape: ``top = mean + thickness/2``, ``bot = mean − thickness/2``
    (a per-node (offset, thickness) is a bijection with a per-node (top, bottom)).
    """
    rows, cols = PANEL_GRID_RADIAL_COUNT, PANEL_GRID_TANGENTIAL_COUNT
    v = min(max(v, -1.0), 1.0)
    u = _radial_frac(r)
    fr = u * (rows - 1)
    i0 = min(int(fr), rows - 2)
    tr = fr - i0
    s = (v + 1.0) / 2.0 * (cols + 1)
    j0 = min(int(s), cols)
    tt = s - j0
    ra, rb = params.panel_thickness_m[i0], params.panel_thickness_m[i0 + 1]
    pa = (ra[0], *ra, ra[-1])
    pb = (rb[0], *rb, rb[-1])
    va = pa[j0] * (1.0 - tt) + pa[j0 + 1] * tt
    vb = pb[j0] * (1.0 - tt) + pb[j0 + 1] * tt
    return va * (1.0 - tr) + vb * tr


def _radial_samples() -> list[float]:
    step = _BLADE_SPAN_M / (_MARGIN_SAMPLES - 1)
    return [BLADE_ROOT_RADIUS_M + step * k for k in range(_MARGIN_SAMPLES)]


def blade_z_envelope_m(params: BladeParams) -> float:
    """Full local material thickness in z, relative to the meridian mean surface (m).

    This is the per-layer footprint that sets the fold pitch (``layer_spacing``). It is the
    span from the highest top surface to the lowest bottom surface, measured off the meridian:

    - **Ribbed**: the rib rails sit at ``±t_rib/2`` and (by containment) enclose the panel, so
      the envelope is the thickest rib. Panel nodes are included defensively.
    - **Uniform**: there are no rails, so the envelope is the panel's own top-to-bottom spread
      ``max(offset + t/2) − min(offset − t/2)`` — a wavier / thicker no-rib sheet folds fatter,
      which is exactly the pressure that keeps design B nesting under the stack-height cap.

    Panel-aware (was rib-only): design B (no ribs) now folds correctly.
    """
    tops: list[float] = []
    bots: list[float] = []
    for off_row, th_row in zip(params.panel_offsets_m, params.panel_thickness_m, strict=True):
        for off, thick in zip(off_row, th_row, strict=True):
            tops.append(off + thick / 2.0)
            bots.append(off - thick / 2.0)
    if not params.uniform:
        rib_max = max(params.t_rib_hub_m, params.t_rib_tip_m)
        tops.append(rib_max / 2.0)
        bots.append(-rib_max / 2.0)
    return max(tops) - min(bots)


def layer_spacing_m(params: BladeParams) -> float:
    """Z-stack layer spacing = the blade's local material envelope + clearance.

    The fan folds by z-stacking (a deck), so adjacent blades sit one layer apart on the pin.
    The spacing must clear the thickest section at every radius. This is
    :func:`blade_z_envelope_m` — panel-aware, so it equals the thickest rib for a ribbed blade
    and the panel's own thickness spread for a uniform no-rib blade. The boss is one layer tall,
    so boss height = ``max(rib, panel) + clearance`` too.
    """
    return blade_z_envelope_m(params) + FOLD_CLEARANCE_M


def folded_rib_bow_extent_m(params: BladeParams) -> float:
    """Peak-to-trough out-of-plane extent of the rib meridian ``max z(r) − min z(r)``.

    The meridian rises up to ~30 mm on the **same z axis the fan folds on**, so it adds to
    the folded stack — a Catmull-Rom overshoot below 0 widens the extent further.
    """
    return rib_meridian_extent_m(params.rib_bow_knots_m, params.rib_bow_interp)


def folded_stack_height_m(params: BladeParams) -> float:
    """Folded-bundle z-extent, **bow-aware**.

    Nested dishes: ``(N−1)·layer_spacing`` between blade bases + the top blade's full
    footprint ``bow_extent + envelope``. The top-blade footprint is the panel-aware
    :func:`blade_z_envelope_m` (was rib-only ``t_rib_max``), so a uniform no-rib blade's own
    thickness spread counts. The CAD swept-volume boolean remains the authoritative check.
    """
    return (
        (params.blade_count - 1) * layer_spacing_m(params)
        + folded_rib_bow_extent_m(params)
        + blade_z_envelope_m(params)
    )


def fold_margin_m(params: BladeParams) -> float:
    """``MAX_FOLDED_STACK_HEIGHT_M − folded_stack_height``. ≥ 0 ⇒ folds acceptably thin.

    The real fold cost is stack height, not hub packing — thick ribs make a fat bundle.
    The CAD swept-volume boolean is the authoritative no-collision check through the swing.
    """
    return MAX_FOLDED_STACK_HEIGHT_M - folded_stack_height_m(params)


def containment_margin_m(params: BladeParams) -> float:
    """Min over grid nodes of ``(t_rib(r) − panel_nom)/2 − |offset|``. ≥ 0 ⇒ contained.

    **Ribbed**: the cambered/undulating panel membrane must fit inside the rib thickness
    envelope so it can never poke out and strike a neighbour (§4.1 rule 2). Bilinear extrema
    sit at the control nodes, so checking the nodes is exact. Couples panel relief to rib
    thickness: bigger undulations need a thicker rib — traded against nesting + mass.

    **Uniform**: there is no rib frame to contain the panel within — the sheet *is* the blade.
    Its nesting is governed by the fold stack-height constraint instead, so this returns
    :func:`fold_margin_m` (the panel-nesting fold margin) rather than a rib-containment margin.
    """
    if params.uniform:
        return fold_margin_m(params)
    margins: list[float] = []
    for r, off_row, th_row in zip(
        panel_radial_stations(), params.panel_offsets_m, params.panel_thickness_m
    ):
        for offset, thick in zip(off_row, th_row):
            allow = (rib_thickness_at(params, r) - thick) / 2.0
            margins.append(allow - abs(offset))
    return min(margins)


def estimate_mass_kg(params: BladeParams) -> float:
    """Coarse assembly-mass estimate (kg): (material + boss) × blade_count × ρ_PETG.

    A fast analytic proxy for the mass cap in-loop; the meshed CAD solid is authoritative.
    Tangential width is the trapezoid ``2·half_width(r)`` (NOT the retired ``r·angle`` sector):
    **ribbed** carries two edge rails + the interior panel; **uniform** carries the full-width
    sheet with no rails. The small extra area from panel undulation is neglected in this proxy.

    Integration is **trapezoidal** (endpoints half-weighted) over the ``_MARGIN_SAMPLES``
    stations. The material rides the **bowed** meridian, so each station's radial length element
    is scaled by the local arc factor ``ds/dr = √(1 + (dz/dr)²)`` — a big bow costs mass.
    """
    samples = _radial_samples()
    n = len(samples)
    dr = _BLADE_SPAN_M / (n - 1)
    z = [rib_z_at(params, r) for r in samples]
    material_vol = 0.0
    for k, r in enumerate(samples):
        trap = 0.5 if k in (0, n - 1) else 1.0  # trapezoidal endpoint weight
        if k == 0:
            arc = math.hypot(dr, z[1] - z[0]) / dr
        elif k == n - 1:
            arc = math.hypot(dr, z[-1] - z[-2]) / dr
        else:
            arc = math.hypot(2.0 * dr, z[k + 1] - z[k - 1]) / (2.0 * dr)
        full_width = 2.0 * half_width_at(r)
        if params.uniform:
            material_vol += trap * full_width * panel_thickness_at(params, r, 0.0) * arc * dr
        else:
            w_rib = rib_width_at(r)
            material_vol += trap * 2.0 * w_rib * rib_thickness_at(params, r) * arc * dr
            w_panel = max(0.0, full_width - 2.0 * w_rib)
            material_vol += trap * w_panel * panel_thickness_at(params, r, 0.0) * arc * dr
    boss_vol = math.pi * PIVOT_BOSS_RADIUS_M**2 * layer_spacing_m(params)
    vol_per_blade = material_vol + boss_vol
    return vol_per_blade * params.blade_count * RHO_PETG_KG_PER_M3


def mass_margin_kg(params: BladeParams) -> float:
    """``MAX_TOTAL_MASS_KG − estimate_mass_kg`` (kg). Reported margin only — NOT a feasibility gate.

    Mass is a **soft** Pareto objective, not a hard constraint (operator, 2026-07-29): the 22 cm /
    12-blade seeds run ~360 g, over the ``MAX_TOTAL_MASS_KG`` = 300 g reference, and gating on it
    would erase the whole search. ``MAX_TOTAL_MASS_KG`` stays a documented reference (audit N8), so
    this margin can go negative without making a design infeasible; :func:`feasible` ignores it.
    """
    return MAX_TOTAL_MASS_KG - estimate_mass_kg(params)


def feasible(params: BladeParams) -> bool:
    """True iff the fold (stack-height) and containment proxies are satisfied.

    Mass is deliberately NOT a gate (operator, 2026-07-29): it is a soft Pareto objective the BO
    trades, reported via :func:`estimate_mass_kg` / :func:`mass_margin_kg`, never a feasibility
    veto. Only the two geometric buildability proxies gate here.
    """
    return fold_margin_m(params) >= 0.0 and containment_margin_m(params) >= 0.0
