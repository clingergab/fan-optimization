"""Locked-constant regression tests for fanopt.geometry.schema.

If anyone changes one of these constants the test fires with a pointer to
the plan lock that authored it. This is intentional — the constants are
load-bearing across the project and drift in any of them propagates to
SU2 cfg numerics, structural FEA loads, BO Pareto axes, etc.

Spec references: see schema.py docstring.
"""

from __future__ import annotations

import math

import pytest

from fanopt.geometry import schema as s

# ---- radial geometry (§0 row 45 dual rib-band lock) ------------------------


def test_blade_length_locked_to_200_mm() -> None:
    assert s.L_BLADE_M == 0.200


def test_handle_offset_locked() -> None:
    assert s.D_HANDLE_M == 0.050


def test_wrist_to_tip_is_handle_plus_blade() -> None:
    """L_WRIST_TO_TIP_M = d_handle + L_blade = 0.25 m (H8 lever-arm lock)."""
    assert pytest.approx(0.250, abs=1e-12) == s.L_WRIST_TO_TIP_M


def test_hub_radius_locked() -> None:
    assert s.HUB_RADIUS_M == 0.020


def test_rib_tip_taper_locked() -> None:
    assert s.RIB_TIP_TAPER_M == 0.015


def test_rib_radial_extent_is_165_mm() -> None:
    """L_RIB = L_blade − HUB_RADIUS − RIB_TIP_TAPER = 0.165 m."""
    assert pytest.approx(0.165, abs=1e-12) == s.L_RIB_M


# ---- rib cross-section (H12) ----------------------------------------------


def test_rib_base_width_4mm() -> None:
    assert s.RIB_BASE_WIDTH_M == 0.004


def test_rib_tip_width_6mm() -> None:
    assert s.RIB_TIP_WIDTH_M == 0.006


def test_rib_thickness_2mm() -> None:
    assert s.RIB_THICKNESS_M == 0.002


# ---- pivot / boss ----------------------------------------------------------


def test_pivot_boss_geometry() -> None:
    assert s.PIVOT_CENTER_X_M == 0.008
    assert s.PIVOT_BOSS_OD_M == 0.012
    assert s.PIVOT_BOSS_RADIUS_M == 0.006
    assert s.PIVOT_PIN_DIAMETER_M == 0.003


def test_panel_pivot_region_radius_includes_1mm_clearance() -> None:
    """7 mm = 6 mm boss radius + 1 mm clearance."""
    assert pytest.approx(0.007, abs=1e-12) == s.PANEL_PIVOT_REGION_RADIUS_M


def test_pivot_mask_contains_origin_and_boss_edge() -> None:
    """Mask must cover (8,0) and (14,0); reject (16,0) and (8,8)."""
    m = s.PANEL_PIVOT_REGION
    assert m.contains(0.008, 0.000) is True
    assert m.contains(0.014, 0.000) is True  # 6 mm out — boss edge
    assert m.contains(0.015, 0.000) is True  # 7 mm out — exactly at clearance
    assert m.contains(0.016, 0.000) is False  # 8 mm out
    assert m.contains(0.008, 0.008) is False  # 8 mm above center


# ---- click footprint -------------------------------------------------------


def test_click_x_range_last_10mm() -> None:
    """(L_blade − 0.010, L_blade) = (0.190, 0.200)."""
    lo, hi = s.CLICK_FOOTPRINT_X_RANGE_M
    assert lo == pytest.approx(0.190, abs=1e-12)
    assert hi == pytest.approx(0.200, abs=1e-12)


def test_click_x_range_fully_in_rib_absent_tip_band() -> None:
    """Lower bound (0.190) is ≥ rib-tip-taper boundary (0.185)."""
    lo, _ = s.CLICK_FOOTPRINT_X_RANGE_M
    rib_taper_x = s.L_BLADE_M - s.RIB_TIP_TAPER_M
    assert lo >= rib_taper_x


def test_chamfer_geometry_option_a() -> None:
    """HIGH-8 Round-9 Option A: 0.5–1.0 mm corner bevel, NOT full-z face."""
    bevel_lo, bevel_hi = s.CLICK_CHAMFER_BEVEL_RANGE_M
    assert bevel_lo == 0.0005
    assert bevel_hi == 0.0010
    assert s.CLICK_CHAMFER_ANGLE_DEG == 45.0


def test_detent_radius_range() -> None:
    lo, hi = s.DETENT_RADIUS_RANGE_M
    assert lo == 0.0003
    assert hi == 0.0005


def test_chamfer_clearance_0_1mm() -> None:
    """0.1 mm per-face design clearance (§0 row 33)."""
    assert pytest.approx(0.0001, abs=1e-12) == s.CHAMFER_CLEARANCE_M


# ---- panel thickness range ------------------------------------------------


def test_panel_thickness_range() -> None:
    assert s.PANEL_THICKNESS_MIN_M == 0.0022
    assert s.PANEL_THICKNESS_MAX_M == 0.0038
    assert s.PANEL_THICKNESS_KNOT_COUNT == 3


def test_panel_thickness_min_respects_chamfer_floor() -> None:
    """min ≥ rib_thickness + 2·chamfer_clearance."""
    floor = s.RIB_THICKNESS_M + 2 * s.CHAMFER_CLEARANCE_M
    assert floor <= s.PANEL_THICKNESS_MIN_M


# ---- blade count + pitch (C8 + MED-10) -------------------------------------


def test_blade_count_default() -> None:
    assert s.BLADE_COUNT_DEFAULT == 10


def test_blade_counts_excludes_14() -> None:
    """MED-10 trim — 14 removed."""
    assert s.BLADE_COUNTS == (8, 10, 12)
    assert 14 not in s.BLADE_COUNTS


def test_inter_blade_angle_radians() -> None:
    """13.3° ≈ 0.232 rad."""
    assert s.INTER_BLADE_ANGLE_DEG == 13.3
    assert pytest.approx(0.232, abs=1e-3) == s.INTER_BLADE_ANGLE_RAD


def test_deployed_extent_for_10_blades_is_133deg() -> None:
    """10 × 13.3° = 133.0° per C8 lock."""
    extent = s.BLADE_COUNT_DEFAULT * s.INTER_BLADE_ANGLE_DEG
    assert extent == pytest.approx(133.0, abs=1e-9)


# ---- mass + CoM caps -------------------------------------------------------


def test_mass_cap_c9() -> None:
    # C9's 100 g relaxed to 120 g (2026-07-21, user-authorized) for more design headroom.
    assert s.MAX_TOTAL_MASS_KG == 0.120


def test_com_cap() -> None:
    """d_handle + 0.55·L_blade = 0.05 + 0.55·0.20 = 0.16 m."""
    assert pytest.approx(0.160, abs=1e-12) == s.MAX_R_COM_WRIST_M


# ---- kinematics (H8 symbol table) ------------------------------------------


def test_kinematics_h8_symbol_table() -> None:
    assert s.F_WAVE_HZ == 2.0
    assert s.T_CYCLE_S == 0.5
    assert s.THETA_MAX_RAD == 0.6981
    assert pytest.approx(2.0 * math.pi * 2.0, abs=1e-9) == s.OMEGA_SHM_RAD_PER_S
    assert pytest.approx(8.77, abs=0.05) == s.OMEGA_BLADE_MAX_RAD_PER_S
    assert pytest.approx(110.0, abs=1.0) == s.ALPHA_MAX_RAD_PER_S2
    assert pytest.approx(2.20, abs=0.02) == s.V_TIP_M_PER_S


# ---- C11 sign lock --------------------------------------------------------


def test_c11_pitching_omega_negative_y() -> None:
    """Right-hand-rule on productive stroke: ω points in −y."""
    x, y, z = s.PITCHING_OMEGA_VEC
    assert x == 0.0
    assert y < 0  # critical — positive y would invert productive stroke
    assert z == 0.0


def test_c11_pitching_ampl_positive_y() -> None:
    """θ_max amplitude is +y about the wrist axis."""
    _, y, _ = s.PITCHING_AMPL_VEC
    assert y > 0


def test_freestream_productive_minus_z() -> None:
    """Stationary-fan CFD: air flows −z when productive stroke sweeps +z."""
    assert s.FREESTREAM_PRODUCTIVE_VEC == (0.0, 0.0, -1.0)


def test_freestream_return_plus_z() -> None:
    assert s.FREESTREAM_RETURN_VEC == (0.0, 0.0, +1.0)


# ---- materials -----------------------------------------------------------


def test_petg_properties_fdm() -> None:
    """FDM-printed PETG, NOT injection-molded datasheet."""
    assert s.RHO_PETG_KG_PER_M3 == 1270.0
    assert s.E_PETG_XY_PA == 1.30e9
    assert s.E_PETG_Z_PA == 1.00e9
    assert s.NU_PETG == 0.38
    assert s.SIGMA_Y_PETG_XY_PA == 45e6
    assert s.SIGMA_Y_PETG_Z_PA == 30e6  # σ_y_Z = 30 MPa per Round-1 lock


def test_pin_densities() -> None:
    assert s.RHO_PIN_STEEL_KG_PER_M3 == 7850.0
    assert s.RHO_PIN_BRASS_KG_PER_M3 == 8500.0


# ---- derived helpers -----------------------------------------------------


def test_panel_tangential_outer_at_tip() -> None:
    """Half of (r·inter_blade_angle) at r = L_BLADE = 0.0232 m."""
    outer = s.panel_tangential_outer_at_tip_m()
    expected = 0.5 * s.L_BLADE_M * s.INTER_BLADE_ANGLE_RAD
    assert outer == pytest.approx(expected, rel=1e-12)
    # Sanity vs plan-cited ~0.0225 m: this is r=0.20 (NOT r=0.25), so a bit
    # different. The plan locks 0.0225 m as the click-edge y; the helper
    # returns the formula's geometric value.
    assert outer == pytest.approx(0.0232, abs=0.001)


def test_click_footprint_y_range_5mm_band() -> None:
    lo, hi = s.click_footprint_y_range_panel_edge_m()
    assert hi - lo == pytest.approx(0.005, abs=1e-12)


def test_panel_tangential_outer_rejects_nonpositive_blade_count() -> None:
    with pytest.raises(ValueError):
        s.panel_tangential_outer_at_tip_m(blade_count=0)


# ---- mask sanity ---------------------------------------------------------


def test_circular_mask_self_consistency() -> None:
    m = s.CircularMask(0.0, 0.0, 1.0)
    assert m.contains(0.0, 0.0) is True
    assert m.contains(1.0, 0.0) is True
    assert m.contains(0.0, 1.0) is True
    assert m.contains(1.1, 0.0) is False
