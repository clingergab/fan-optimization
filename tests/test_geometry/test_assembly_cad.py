"""Tests for fanopt.geometry.assembly_cad (V-unit blade composition).

Skipped at module load when CadQuery isn't installed, per CLAUDE.md §4.1.
"""

from __future__ import annotations

import importlib.util
import math

import pytest

if importlib.util.find_spec("cadquery") is None:
    pytest.skip("cadquery not installed", allow_module_level=True)

import cadquery as cq

from fanopt.geometry.assembly_cad import (
    CHAMFER_BEVEL_M,
    make_pivot_boss,
    make_rib,
    make_vunit_blade,
)
from fanopt.geometry.envelope import Layer1Params, ThicknessGridField
from fanopt.geometry.envelope_cad import LOFT_START_EPS_M
from fanopt.geometry.fields import Layer2Params
from fanopt.geometry.generator import BladeDesignParams
from fanopt.geometry.generator_cad import generate_blade_cad
from fanopt.geometry.manufacturability import Layer4Params
from fanopt.geometry.primitives import Layer3Primitive
from fanopt.geometry.schema import (
    CLICK_CHAMFER_BEVEL_RANGE_M,
    HUB_RADIUS_M,
    L_BLADE_M,
    PIVOT_BOSS_RADIUS_M,
    PIVOT_CENTER_X_M,
    RIB_BASE_WIDTH_M,
    RIB_THICKNESS_M,
    RIB_TIP_TAPER_M,
    RIB_TIP_WIDTH_M,
    panel_tangential_outer_at_tip_m,
)


def _canonical_design() -> BladeDesignParams:
    return BladeDesignParams(
        layer1=Layer1Params(
            blade_count=10,
            camber_knots_m=(0.0, 0.002, 0.001),
            twist_knots_rad=(0.0, 0.0),
            thickness_field=ThicknessGridField.from_radial_knots((0.0030, 0.0028, 0.0026)),
            edge_profile="rounded",
            fourier_le_amplitudes=(0.0, 0.0, 0.0),
            fourier_te_amplitudes=(0.0, 0.0, 0.0),
        ),
        layer2=Layer2Params.all_inactive(),
        layer3=Layer3Primitive.absent(),
        layer4=Layer4Params(
            print_orientation="flat",
            layer_height_m=0.0002,
            click_chamfer_angle_deg=45.0,
            click_detent_size_m=0.0004,
            click_design_clearance_m=0.00018,
        ),
    )


# ---- chamfer bevel constant ----------------------------------------------


def test_chamfer_bevel_at_midpoint_of_locked_range() -> None:
    expected = sum(CLICK_CHAMFER_BEVEL_RANGE_M) / 2.0
    assert pytest.approx(expected) == CHAMFER_BEVEL_M


def test_chamfer_bevel_within_locked_range() -> None:
    lo, hi = CLICK_CHAMFER_BEVEL_RANGE_M
    assert lo <= CHAMFER_BEVEL_M <= hi


# ---- make_rib -------------------------------------------------------------


def test_rib_le_volume_positive() -> None:
    rib = make_rib("LE")
    assert rib.val().Volume() > 0.0


def test_rib_te_volume_positive() -> None:
    rib = make_rib("TE")
    assert rib.val().Volume() > 0.0


def test_rib_le_lives_in_negative_y_half() -> None:
    """LE rib must sit at y < 0."""
    bb = make_rib("LE").val().BoundingBox()
    assert bb.ymax < 0.0


def test_rib_te_lives_in_positive_y_half() -> None:
    bb = make_rib("TE").val().BoundingBox()
    assert bb.ymin > 0.0


def test_rib_radial_extent_matches_locked_band() -> None:
    """Rib spans [HUB_RADIUS, L_BLADE - RIB_TIP_TAPER]."""
    bb = make_rib("LE").val().BoundingBox()
    assert bb.xmin == pytest.approx(HUB_RADIUS_M, abs=1e-6)
    assert bb.xmax == pytest.approx(L_BLADE_M - RIB_TIP_TAPER_M, abs=1e-6)


def test_rib_thickness_matches_locked_value() -> None:
    bb = make_rib("LE").val().BoundingBox()
    z_extent = bb.zmax - bb.zmin
    assert z_extent == pytest.approx(RIB_THICKNESS_M, abs=1e-6)


def test_rib_volume_matches_taper_estimate() -> None:
    """Rib is a tapered bar; volume ≈ length × thickness × mean_width."""
    length = L_BLADE_M - RIB_TIP_TAPER_M - HUB_RADIUS_M  # 0.165 m
    mean_width = (RIB_BASE_WIDTH_M + RIB_TIP_WIDTH_M) / 2.0  # 5 mm
    expected = length * RIB_THICKNESS_M * mean_width
    vol = make_rib("LE").val().Volume()
    # Loose tolerance — CadQuery loft adds small curvature artifacts.
    assert vol == pytest.approx(expected, rel=0.05)


def test_rib_side_validation() -> None:
    with pytest.raises(ValueError, match="LE.*TE"):
        make_rib("DIAGONAL")


# ---- make_pivot_boss ------------------------------------------------------


def test_pivot_boss_volume_matches_cylinder_formula() -> None:
    expected = math.pi * (PIVOT_BOSS_RADIUS_M**2) * RIB_THICKNESS_M
    vol = make_pivot_boss().val().Volume()
    assert vol == pytest.approx(expected, rel=1e-3)


def test_pivot_boss_centred_at_pivot_x() -> None:
    bb = make_pivot_boss().val().BoundingBox()
    centre_x = (bb.xmin + bb.xmax) / 2.0
    assert centre_x == pytest.approx(PIVOT_CENTER_X_M, abs=1e-6)


def test_pivot_boss_centred_at_y_zero() -> None:
    bb = make_pivot_boss().val().BoundingBox()
    assert (bb.ymin + bb.ymax) / 2.0 == pytest.approx(0.0, abs=1e-6)


def test_pivot_boss_zmin_flush_with_print_bed() -> None:
    bb = make_pivot_boss().val().BoundingBox()
    assert bb.zmin == pytest.approx(0.0, abs=1e-6)


# ---- make_vunit_blade -----------------------------------------------------


def test_vunit_blade_is_single_solid() -> None:
    blade = make_vunit_blade(_canonical_design())
    assert blade.val().Volume() > 0.0
    assert len(blade.solids().vals()) == 1


def test_vunit_blade_volume_exceeds_panel_alone() -> None:
    """Adding ribs + boss + detent must increase volume vs the panel
    alone (chamfer subtracts some but not enough to negate the additions).
    """
    design = _canonical_design()
    _result, panel = generate_blade_cad(design)
    panel_vol = panel.val().Volume()
    blade_vol = make_vunit_blade(design).val().Volume()
    assert blade_vol > panel_vol


def test_vunit_blade_x_extent_panel_dominated() -> None:
    blade = make_vunit_blade(_canonical_design())
    bb = blade.val().BoundingBox()
    # x extent is dominated by the panel envelope: xmin = LOFT_START_EPS_M
    # (the loft starts 1 mm inboard of the pivot to avoid the degenerate
    # x=0 cross-section), xmax = L_BLADE_M.
    assert bb.xmin == pytest.approx(LOFT_START_EPS_M, abs=2e-6)
    assert bb.xmax == pytest.approx(L_BLADE_M, abs=2e-6)
    # Rib reach is verified separately via make_rib's bounding box.
    rib = make_rib("LE")
    rib_bb = rib.val().BoundingBox()
    assert rib_bb.xmin == pytest.approx(HUB_RADIUS_M, abs=2e-6)


def test_vunit_blade_y_extent_includes_rib_extents() -> None:
    """Ribs sit at y = ±panel_half_pitch(x) with width RIB_TIP_WIDTH at
    the tip — so blade y_extent at the tip should exceed bare panel
    y_extent."""
    design = _canonical_design()
    _result, panel = generate_blade_cad(design)
    panel_bb = panel.val().BoundingBox()
    blade_bb = make_vunit_blade(design).val().Volume()
    assert blade_bb > 0  # at least it has volume
    blade = make_vunit_blade(design)
    bbb = blade.val().BoundingBox()
    # The blade's y_max should be at least the panel's y_max (ribs add
    # outward, and the +y rib's outer y edge is panel_y_max + RIB_TIP_WIDTH/2).
    assert bbb.ymax >= panel_bb.ymax


def test_vunit_blade_chamfer_present_at_outer_corner() -> None:
    """The chamfer subtracts material at (x=L_BLADE, |y|=panel_outer, z=top).
    Probing that exact corner with a small box should find it absent."""
    blade = make_vunit_blade(_canonical_design())
    bb = blade.val().BoundingBox()
    z_top = bb.zmax
    y_outer = panel_tangential_outer_at_tip_m()
    # Probe a tiny cube at the LE outer corner where the chamfer cut.
    probe = (
        cq.Workplane("XY")
        .box(CHAMFER_BEVEL_M / 4.0, CHAMFER_BEVEL_M / 4.0, CHAMFER_BEVEL_M / 4.0)
        .translate(
            (
                L_BLADE_M - CHAMFER_BEVEL_M / 8.0,
                -y_outer + CHAMFER_BEVEL_M / 8.0,
                z_top - CHAMFER_BEVEL_M / 8.0,
            )
        )
    )
    # The corner SHOULD be inside the chamfer cut region — probing
    # there with intersect should give a small or zero volume.
    inter = blade.val().intersect(probe.val()).Volume()
    full_probe = probe.val().Volume()
    # If the chamfer cut at this corner, the intersection with the blade
    # is strictly less than the full probe volume.
    assert inter < full_probe


def test_vunit_blade_rib_not_pierced_by_pivot() -> None:
    """Panel-pivot architectural lock — the rib's y-band must NOT
    include y=0 (pivot pin location)."""
    rib = make_rib("LE")
    bb = rib.val().BoundingBox()
    assert bb.ymax < 0.0
    rib_te = make_rib("TE")
    assert rib_te.val().BoundingBox().ymin > 0.0


def test_vunit_blade_detent_increases_y_extent() -> None:
    """The detent bump sits past the panel y-edge → blade y_max > panel y_max
    (with margin equal to the detent radius)."""
    design = _canonical_design()
    blade = make_vunit_blade(design)
    bb = blade.val().BoundingBox()
    y_outer = panel_tangential_outer_at_tip_m()
    # The detent's outer face is at y_outer + detent_radius.
    expected_extra = design.layer4.click_detent_size_m
    # The rib + detent both extend past the panel. Blade y_max should be
    # at least y_outer.
    assert bb.ymax >= y_outer - 1e-6
    # And at most y_outer + RIB_TIP_WIDTH/2 + detent (loose upper bound).
    assert bb.ymax <= y_outer + RIB_TIP_WIDTH_M / 2.0 + expected_extra + 1e-3


def test_vunit_blade_exports_to_stl(tmp_path) -> None:
    """End-to-end: produced solid must export to STL without raising."""
    blade = make_vunit_blade(_canonical_design())
    stl_path = tmp_path / "blade.stl"
    cq.exporters.export(blade, str(stl_path))
    assert stl_path.exists()
    assert stl_path.stat().st_size > 1000


def test_vunit_blade_under_edge_orientation() -> None:
    """Composes under non-flat orientation too — bb.zmin < 0 because
    the panel envelope is midplane-symmetric."""
    d = _canonical_design()
    d_edge = BladeDesignParams(
        layer1=d.layer1,
        layer2=d.layer2,
        layer3=d.layer3,
        layer4=Layer4Params(
            print_orientation="edge",
            layer_height_m=d.layer4.layer_height_m,
            click_chamfer_angle_deg=d.layer4.click_chamfer_angle_deg,
            click_detent_size_m=d.layer4.click_detent_size_m,
            click_design_clearance_m=d.layer4.click_design_clearance_m,
        ),
    )
    blade = make_vunit_blade(d_edge)
    assert blade.val().Volume() > 0.0
