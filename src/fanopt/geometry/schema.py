"""Locked geometry + kinematics constants for the folding-fan project.

Single source of truth for the locked numerical constants the plan
(`docs/plan_R11.md`) pins across §0, §2.1, §2.3, §3.1.2, §3.2.0, §3.2.3,
§6.4. Any module that needs HUB_RADIUS, the panel-pivot region, the click
footprint, the rib taper, or the C11 PITCHING_OMEGA sign should import
from here rather than re-defining literals.

The full JSON schema + Pydantic validation for design parameters lands in
Phase 1 alongside the §9.7 generator. This module ships only the
non-negotiable locked constants so downstream spikes (0.2 — 0.7) stop
hard-coding the same numbers in scattered places.

Locks referenced:
- §0 row 25 panel-pivot architecture
- §0 row 26 / §3.2.0 coordinate convention + C11 PITCHING_OMEGA sign
- §0 row 27 wrist axis = +y
- §0 row 28 mass cap (C9)
- §0 row 29 CoM cap
- §0 row 33 panel thickness range
- §0 row 45 click-footprint dual rib-band lock (HUB_RADIUS + RIB_TIP_TAPER)
- §0 row 47 plano-convex envelope (rib-flat default)
- §2.1 dimensions table
- §3.1.2 PANEL_PIVOT_REGION + CLICK_FOOTPRINT_*
- §3.2.3 H8 kinematics symbol table
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    # geometry — radial
    "L_BLADE_M",
    "D_HANDLE_M",
    "L_WRIST_TO_TIP_M",
    "HUB_RADIUS_M",
    "RIB_TIP_TAPER_M",
    "L_RIB_M",
    # geometry — rib
    "RIB_BASE_WIDTH_M",
    "RIB_TIP_WIDTH_M",
    "RIB_THICKNESS_M",
    # geometry — pivot / boss
    "PIVOT_CENTER_X_M",
    "PIVOT_BOSS_OD_M",
    "PIVOT_BOSS_RADIUS_M",
    "PIVOT_PIN_DIAMETER_M",
    "PANEL_PIVOT_REGION_RADIUS_M",
    # geometry — click
    "CLICK_FOOTPRINT_X_RANGE_M",
    "CHAMFER_CLEARANCE_M",
    "CLICK_CHAMFER_BEVEL_RANGE_M",
    "DETENT_RADIUS_RANGE_M",
    "CLICK_CHAMFER_ANGLE_DEG",
    # geometry — panel
    "PANEL_THICKNESS_MIN_M",
    "PANEL_THICKNESS_MAX_M",
    "PANEL_THICKNESS_KNOT_COUNT",
    "THICKNESS_GRID_RADIAL_COUNT",
    "THICKNESS_GRID_TANGENTIAL_COUNT",
    "CORRUGATION_AMPLITUDE_MAX_M",
    "CORRUGATION_WAVELENGTH_RANGE",
    # geometry — blade-count + pitch
    "BLADE_COUNT_DEFAULT",
    "BLADE_COUNTS",
    "INTER_BLADE_ANGLE_DEG",
    "INTER_BLADE_ANGLE_RAD",
    # mass + CoM caps
    "MAX_TOTAL_MASS_KG",
    "MAX_R_COM_WRIST_M",
    # kinematics (H8 symbol table)
    "F_WAVE_HZ",
    "T_CYCLE_S",
    "THETA_MAX_RAD",
    "OMEGA_SHM_RAD_PER_S",
    "OMEGA_BLADE_MAX_RAD_PER_S",
    "ALPHA_MAX_RAD_PER_S2",
    "V_TIP_M_PER_S",
    # C11 freestream / pitching signs
    "PITCHING_OMEGA_VEC",
    "PITCHING_AMPL_VEC",
    "FREESTREAM_PRODUCTIVE_VEC",
    "FREESTREAM_RETURN_VEC",
    # material — PETG
    "RHO_PETG_KG_PER_M3",
    "E_PETG_XY_PA",
    "E_PETG_Z_PA",
    "NU_PETG",
    "SIGMA_Y_PETG_XY_PA",
    "SIGMA_Y_PETG_Z_PA",
    # material — pivot pin (steel/brass)
    "RHO_PIN_STEEL_KG_PER_M3",
    "RHO_PIN_BRASS_KG_PER_M3",
    # masks
    "CircularMask",
    "PANEL_PIVOT_REGION",
    "panel_tangential_outer_at_tip_m",
    "click_footprint_y_range_panel_edge_m",
]


# ---------------------------------------------------------------------------
# Radial geometry — §0 row 45 dual rib-band lock + §2.1
# ---------------------------------------------------------------------------

L_BLADE_M: float = 0.200
"""Blade length from pivot to tip (§2.1). Panel spans full L_BLADE_M radially."""

D_HANDLE_M: float = 0.050
"""Wrist-axis-to-pivot offset along +x (§0 row 27)."""

L_WRIST_TO_TIP_M: float = D_HANDLE_M + L_BLADE_M
"""0.25 m. The canonical lever arm for τ → F at the click region (H8 lock)."""

HUB_RADIUS_M: float = 0.020
"""Inner rib boundary; rib is absent for x < HUB_RADIUS (C7 / Architectural D)."""

RIB_TIP_TAPER_M: float = 0.015
"""Outer rib taper-out length; rib is absent for x > L_BLADE − RIB_TIP_TAPER
(Architectural A). Click footprint lives in this rib-absent tip region."""

L_RIB_M: float = L_BLADE_M - HUB_RADIUS_M - RIB_TIP_TAPER_M
"""Rib radial extent: 0.165 m."""


# ---------------------------------------------------------------------------
# Rib cross-section — H12 lock
# ---------------------------------------------------------------------------

RIB_BASE_WIDTH_M: float = 0.004
"""Rib width at root (4 mm, H12 lock — NOT a BO variable)."""

RIB_TIP_WIDTH_M: float = 0.006
"""Rib width at tip (6 mm, H12 UP-taper)."""

RIB_THICKNESS_M: float = 0.002
"""Rib z-thickness (FDM minimum feature size)."""


# ---------------------------------------------------------------------------
# Pivot region — §2.1, §3.1.2
# ---------------------------------------------------------------------------

PIVOT_CENTER_X_M: float = 0.008
"""Pivot pin x position from rib base end. Same across all blades."""

PIVOT_BOSS_OD_M: float = 0.012
"""12 mm OD circular boss centered on the pivot pin (§2.1)."""

PIVOT_BOSS_RADIUS_M: float = PIVOT_BOSS_OD_M / 2.0
"""6 mm. Pivot K_tt = 2.42 derived from d/w = 3/12 = 0.25."""

PIVOT_PIN_DIAMETER_M: float = 0.003

PANEL_PIVOT_REGION_RADIUS_M: float = PIVOT_BOSS_RADIUS_M + 0.001
"""7 mm — 12 mm boss + 1 mm clearance. PANEL_PIVOT_REGION keep-out radius."""


# ---------------------------------------------------------------------------
# Click feature — Round-9 HIGH-8 Option A (corner bevel, NOT full-z face)
# ---------------------------------------------------------------------------

CLICK_FOOTPRINT_X_RANGE_M: tuple[float, float] = (L_BLADE_M - 0.010, L_BLADE_M)
"""Last 10 mm of the blade radially — fully inside the rib-absent tip band."""

CHAMFER_CLEARANCE_M: float = 0.0001
"""0.1 mm per-face design clearance (§0 row 33)."""

CLICK_CHAMFER_BEVEL_RANGE_M: tuple[float, float] = (0.0005, 0.0010)
"""0.5 – 1.0 mm corner bevel — Option A (NOT a full-panel-thickness face)."""

DETENT_RADIUS_RANGE_M: tuple[float, float] = (0.0003, 0.0005)
"""Hemispherical detent bump radius range."""

CLICK_CHAMFER_ANGLE_DEG: float = 45.0
"""±1° tolerance asserted by test_click_z_lap.py."""


# ---------------------------------------------------------------------------
# Panel thickness — §0 row 33
# ---------------------------------------------------------------------------

PANEL_THICKNESS_MIN_M: float = 0.0022
"""2.2 mm — chamfer clearance floor (rib_thickness + 2·chamfer_clearance)."""

PANEL_THICKNESS_MAX_M: float = 0.0038
"""3.8 mm — folded-stack collision ceiling (2·rib_thickness − folded_clearance)."""

PANEL_THICKNESS_KNOT_COUNT: int = 3
"""3 spline knots at (t0, t1, t2). Superseded by the Path A+ thickness grid
(see THICKNESS_GRID_* below); retained for the legacy Layer-1 spline until the
grid integration lands."""

# Path A+ panel thickness field (plan_v1_slim_latest.md §10): a control-point
# thickness grid + a corrugation family replaces the 3-knot spline. Each grid
# point is independently bounded to [PANEL_THICKNESS_MIN_M, PANEL_THICKNESS_MAX_M]
# so the thickness lock + folded-collision floor hold by construction.
THICKNESS_GRID_RADIAL_COUNT: int = 3
"""Radial control-point count (hub→tip) of the Path A+ thickness grid."""

THICKNESS_GRID_TANGENTIAL_COUNT: int = 6
"""Tangential control-point count (across local panel width) of the grid."""

CORRUGATION_AMPLITUDE_MAX_M: float = 0.0008
"""0.8 mm — max corrugation half-amplitude added onto the grid before clamping."""

CORRUGATION_WAVELENGTH_RANGE: tuple[float, float] = (0.2, 1.0)
"""Normalized corrugation wavelength range (fraction of the parametric domain)."""


# ---------------------------------------------------------------------------
# Blade count + pitch — C8 lock; 14-blade ergonomic trim (Round 7)
# ---------------------------------------------------------------------------

BLADE_COUNT_DEFAULT: int = 10

BLADE_COUNTS: tuple[int, ...] = (8, 10, 12)
"""14 removed for ergonomic infeasibility (>180° past straight-line, Round 7)."""

INTER_BLADE_ANGLE_DEG: float = 13.3
"""C8 lock — centerline-to-centerline AND tangential blade width."""

INTER_BLADE_ANGLE_RAD: float = math.radians(INTER_BLADE_ANGLE_DEG)
"""≈ 0.232 rad. Used in panel-width formula: panel_width(r) = r · 0.232 − 2·rib_width(r) − 0.5 mm."""


# ---------------------------------------------------------------------------
# Mass + CoM caps — C9, §0 row 29
# ---------------------------------------------------------------------------

MAX_TOTAL_MASS_KG: float = 0.300
"""Total assembly mass cap. Relaxed C9 100 g → 120 g (2026-07-21) → 300 g (2026-07-22),
user-authorized, to let the optimizer genuinely explore thick ribs / thick panels /
independent faces (which are heavy — even a 6 mm-rib/3 mm-panel blade is ~187 g). This is
a deliberately heavy *exploration* budget (V2 data), NOT a practical hand-fan mass. NOTE:
the CLAUDE.md §3 / plan C9 lock text should be updated to match."""

MAX_R_COM_WRIST_M: float = 0.160
"""d_handle 0.05 m + 0.55·L_blade 0.20 m. Distribution-based, not mass-based."""


# ---------------------------------------------------------------------------
# Kinematics — §3.2.3 H8 symbol table
# ---------------------------------------------------------------------------

F_WAVE_HZ: float = 2.0

T_CYCLE_S: float = 1.0 / F_WAVE_HZ

THETA_MAX_RAD: float = 0.6981  # 40°

OMEGA_SHM_RAD_PER_S: float = 2.0 * math.pi * F_WAVE_HZ
"""≈ 12.566 rad/s. The SHM angular frequency of the pitching motion."""

OMEGA_BLADE_MAX_RAD_PER_S: float = THETA_MAX_RAD * OMEGA_SHM_RAD_PER_S
"""≈ 8.8 rad/s. Peak instantaneous blade angular velocity."""

ALPHA_MAX_RAD_PER_S2: float = THETA_MAX_RAD * OMEGA_SHM_RAD_PER_S**2
"""≈ 110 rad/s². Peak angular acceleration (Phase 2 inertial load)."""

V_TIP_M_PER_S: float = OMEGA_BLADE_MAX_RAD_PER_S * L_WRIST_TO_TIP_M
"""≈ 2.20 m/s. Peak tip velocity."""


# ---------------------------------------------------------------------------
# C11 sign convention — §3.2.0
# ---------------------------------------------------------------------------

PITCHING_OMEGA_VEC: tuple[float, float, float] = (0.0, -OMEGA_SHM_RAD_PER_S, 0.0)
"""C11 lock — NEGATIVE y component. Right-hand-rule on productive stroke."""

PITCHING_AMPL_VEC: tuple[float, float, float] = (0.0, +THETA_MAX_RAD, 0.0)

FREESTREAM_PRODUCTIVE_VEC: tuple[float, float, float] = (0.0, 0.0, -1.0)
"""Air flows -z relative to a stationary fan swept +z toward user."""

FREESTREAM_RETURN_VEC: tuple[float, float, float] = (0.0, 0.0, +1.0)


# ---------------------------------------------------------------------------
# Materials — §10.1, §3.1.7
# ---------------------------------------------------------------------------

RHO_PETG_KG_PER_M3: float = 1270.0

E_PETG_XY_PA: float = 1.30e9
"""FDM PETG, NOT injection-molded datasheet value (2100 MPa)."""

E_PETG_Z_PA: float = 1.00e9

NU_PETG: float = 0.38

SIGMA_Y_PETG_XY_PA: float = 45.0e6
"""Yield in plane (rib-flat orientation)."""

SIGMA_Y_PETG_Z_PA: float = 30.0e6
"""Yield through interlayer (Z-direction). Per FDM anisotropy."""

RHO_PIN_STEEL_KG_PER_M3: float = 7850.0
RHO_PIN_BRASS_KG_PER_M3: float = 8500.0


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CircularMask:
    """A circular keep-out region in (x, y) coordinates on the panel plane.

    Used for `PANEL_PIVOT_REGION` (the 12 mm boss + clearance). Layer 2 / 3
    generative carving must clip against this mask BEFORE Boolean
    subtraction (§9.7.1 Step 0).
    """

    center_x_m: float
    center_y_m: float
    radius_m: float

    def contains(self, x_m: float, y_m: float) -> bool:
        dx = x_m - self.center_x_m
        dy = y_m - self.center_y_m
        return (dx * dx + dy * dy) <= (self.radius_m * self.radius_m)


PANEL_PIVOT_REGION: CircularMask = CircularMask(
    center_x_m=PIVOT_CENTER_X_M,
    center_y_m=0.0,
    radius_m=PANEL_PIVOT_REGION_RADIUS_M,
)
"""7 mm-radius circular keep-out around the pivot pin centerline. Covers the
full 12 mm OD boss + 1 mm clearance regardless of orientation (item #41
lock — replaces the prior 8 mm rectangular bound that was narrower than
the boss in y)."""


# ---------------------------------------------------------------------------
# Derived helpers for tangential geometry
# ---------------------------------------------------------------------------


def panel_tangential_outer_at_tip_m(
    blade_count: int = BLADE_COUNT_DEFAULT,
) -> float:
    """Half-pitch at the tip — `r * inter_blade_angle_rad / 2` at r = L_wrist_to_tip.

    Used for the click footprint Y range. The default 10-blade fan gives
    ≈ 0.029 m at the wrist-relative tip, but the *plan-locked* tangential
    extent uses r = L_BLADE_M (NOT L_WRIST_TO_TIP_M) for the panel half-
    pitch at the radial position where the click sits — that yields the
    canonical ~0.0225 m figure used throughout the plan.

    Returns half the panel tangential pitch at r = L_BLADE_M.
    """
    if blade_count <= 0:
        raise ValueError(f"blade_count must be > 0, got {blade_count}")
    # The plan's panel-tangential-outer formula uses r = L_BLADE_M (the
    # radial position of the click footprint at the blade tip), with the
    # locked 13.3° inter-blade angle.
    return 0.5 * L_BLADE_M * INTER_BLADE_ANGLE_RAD


def click_footprint_y_range_panel_edge_m(
    blade_count: int = BLADE_COUNT_DEFAULT,
) -> tuple[float, float]:
    """`(panel_tangential_outer − 0.005, panel_tangential_outer)`. 5 mm band."""
    outer = panel_tangential_outer_at_tip_m(blade_count=blade_count)
    return (outer - 0.005, outer)
