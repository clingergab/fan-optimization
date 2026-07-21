"""Tests for fanopt.geometry.blade (lean surface-of-revolution blade params)."""

from __future__ import annotations

import pytest

from fanopt.geometry.blade import (
    FOLD_CLEARANCE_M,
    MAX_FOLDED_STACK_HEIGHT_M,
    PANEL_GRID_RADIAL_COUNT,
    PANEL_GRID_TANGENTIAL_COUNT,
    PANEL_OFFSET_RANGE_M,
    PANEL_THICKNESS_NOM_RANGE_M,
    RIB_BOW_RANGE_M,
    RIB_THICKNESS_RANGE_M,
    RIB_TIP_RADIUS_M,
    BladeParams,
    containment_margin_m,
    displacement_at,
    estimate_mass_kg,
    feasible,
    fold_margin_m,
    folded_stack_height_m,
    layer_spacing_m,
    mass_margin_kg,
    panel_radial_stations,
    rib_bow_stations,
    rib_thickness_at,
    rib_width_at,
    rib_z_at,
)
from fanopt.geometry.schema import (
    HUB_RADIUS_M,
    RIB_BASE_WIDTH_M,
    RIB_TIP_WIDTH_M,
)

_SAMPLE_GRID = (
    (0.0003, 0.0005, 0.0003),
    (0.0004, 0.0006, 0.0004),
    (0.0005, 0.0007, 0.0005),
    (0.0006, 0.0008, 0.0006),
)


def _sample(blade_count: int = 8) -> BladeParams:
    """A feasible lean blade (all three constraint proxies satisfied)."""
    return BladeParams(
        blade_count=blade_count,
        rib_bow_knots_m=(0.005, 0.010, 0.013, 0.017, 0.020),
        rib_bow_interp="linear",
        t_rib_hub_m=0.0025,
        t_rib_tip_m=0.0035,
        panel_offsets_m=_SAMPLE_GRID,
        panel_thickness_m=((0.0013, 0.0013, 0.0013), (0.0013, 0.0013, 0.0013), (0.0013, 0.0013, 0.0013), (0.0013, 0.0013, 0.0013)),
    )


# --- construction + validation ----------------------------------------------


def test_valid_construction():
    assert _sample().blade_count == 8


def test_invalid_blade_count_raises():
    with pytest.raises(ValueError, match="blade_count"):
        BladeParams.from_dict({**_sample().to_dict(), "blade_count": 9})


@pytest.mark.parametrize(
    "field,value",
    [
        ("t_rib_hub_m", RIB_THICKNESS_RANGE_M[0] - 0.001),
        ("t_rib_tip_m", RIB_THICKNESS_RANGE_M[1] + 0.001),
    ],
)
def test_out_of_range_scalar_field_raises(field, value):
    kwargs = _sample().to_dict()
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        BladeParams.from_dict(kwargs)


def test_out_of_range_bow_knot_raises():
    kwargs = _sample().to_dict()
    kwargs["rib_bow_knots_m"][2] = RIB_BOW_RANGE_M[1] + 0.001
    with pytest.raises(ValueError, match="rib_bow_knots_m"):
        BladeParams.from_dict(kwargs)


def test_wrong_bow_knot_count_raises():
    kwargs = _sample().to_dict()
    kwargs["rib_bow_knots_m"] = [0.01, 0.02]  # too few
    with pytest.raises(ValueError, match="rib_bow_knots_m"):
        BladeParams.from_dict(kwargs)


def test_invalid_bow_interp_raises():
    kwargs = _sample().to_dict()
    kwargs["rib_bow_interp"] = "cubic"
    with pytest.raises(ValueError, match="rib_bow_interp"):
        BladeParams.from_dict(kwargs)


def test_from_dict_backcompat_legacy_meridian():
    # Pre-enrichment schema (rib_bow_mid_m / rib_bow_tip_m) still loads, resampled to knots.
    legacy = {k: v for k, v in _sample().to_dict().items()
              if k not in ("rib_bow_knots_m", "rib_bow_interp")}
    legacy["rib_bow_mid_m"], legacy["rib_bow_tip_m"] = 0.010, 0.020
    p = BladeParams.from_dict(legacy)
    assert len(p.rib_bow_knots_m) == len(rib_bow_stations())
    assert rib_z_at(p, RIB_TIP_RADIUS_M) == pytest.approx(0.020)  # tip knot = old tip bow


def test_out_of_range_grid_value_raises():
    kwargs = _sample().to_dict()
    kwargs["panel_offsets_m"] = [list(r) for r in _SAMPLE_GRID]
    kwargs["panel_offsets_m"][0][0] = PANEL_OFFSET_RANGE_M[1] + 0.001
    with pytest.raises(ValueError, match="panel_offsets_m"):
        BladeParams.from_dict(kwargs)


def test_out_of_range_thickness_grid_value_raises():
    kwargs = _sample().to_dict()
    kwargs["panel_thickness_m"][0][0] = PANEL_THICKNESS_NOM_RANGE_M[1] + 0.001
    with pytest.raises(ValueError, match="panel_thickness_m"):
        BladeParams.from_dict(kwargs)


def test_wrong_grid_shape_raises():
    kwargs = _sample().to_dict()
    kwargs["panel_offsets_m"] = [[0.0, 0.0, 0.0]]  # only 1 radial row
    with pytest.raises(ValueError, match="radial rows"):
        BladeParams.from_dict(kwargs)


def test_wrong_grid_row_width_raises():
    kwargs = _sample().to_dict()
    kwargs["panel_offsets_m"] = [[0.0, 0.0] for _ in range(PANEL_GRID_RADIAL_COUNT)]
    with pytest.raises(ValueError, match="tangential points"):
        BladeParams.from_dict(kwargs)


def test_to_from_dict_roundtrip():
    p = _sample(blade_count=12)
    assert BladeParams.from_dict(p.to_dict()) == p


# --- rib meridian + thickness + width ---------------------------------------


def test_rib_z_zero_at_hub():
    assert rib_z_at(_sample(), HUB_RADIUS_M) == pytest.approx(0.0)


def test_rib_z_hits_each_knot_at_its_station_linear():
    p = _sample()
    for station, knot in zip(rib_bow_stations(), p.rib_bow_knots_m):
        assert rib_z_at(p, station) == pytest.approx(knot)


def test_rib_z_hits_each_knot_at_its_station_smooth():
    # Catmull-Rom interpolates through its control points, so knots are hit exactly.
    p = BladeParams(**{**_sample().to_dict(), "rib_bow_interp": "smooth"})
    for station, knot in zip(rib_bow_stations(), p.rib_bow_knots_m):
        assert rib_z_at(p, station) == pytest.approx(knot)


def test_rib_z_equals_last_knot_at_tip():
    assert rib_z_at(_sample(), RIB_TIP_RADIUS_M) == pytest.approx(_sample().rib_bow_knots_m[-1])


def test_rib_z_linear_midpoint_between_two_knots():
    # Two equal adjacent knots ⇒ the linear meridian is flat between them.
    knots = (0.010, 0.010, 0.010, 0.010, 0.010)
    p = BladeParams(**{**_sample().to_dict(), "rib_bow_knots_m": knots})
    s0, s1 = rib_bow_stations()[1], rib_bow_stations()[2]
    assert rib_z_at(p, 0.5 * (s0 + s1)) == pytest.approx(0.010)


def test_rib_thickness_endpoints():
    p = _sample()
    assert rib_thickness_at(p, HUB_RADIUS_M) == pytest.approx(p.t_rib_hub_m)
    assert rib_thickness_at(p, RIB_TIP_RADIUS_M) == pytest.approx(p.t_rib_tip_m)


def test_rib_width_endpoints():
    assert rib_width_at(HUB_RADIUS_M) == pytest.approx(RIB_BASE_WIDTH_M)
    assert rib_width_at(RIB_TIP_RADIUS_M) == pytest.approx(RIB_TIP_WIDTH_M)


# --- panel displacement grid ------------------------------------------------


def test_displacement_zero_at_both_ribs():
    p = _sample()
    assert displacement_at(p, 0.10, -1.0) == pytest.approx(0.0)
    assert displacement_at(p, 0.10, +1.0) == pytest.approx(0.0)


def test_displacement_hits_control_node_value():
    # At a radial control row and the mid tangential node (v=0, the middle interior
    # point for 3 interior points), the surface equals that grid value.
    p = _sample()
    r0 = panel_radial_stations()[0]
    assert displacement_at(p, r0, 0.0) == pytest.approx(_SAMPLE_GRID[0][1])


def test_displacement_grid_expresses_base_to_tip_zigzag():
    # Alternating-sign rows produce a genuine base→tip zigzag (the shape the lean
    # camber basis could not represent).
    zz = tuple(
        tuple((0.0006 if i % 2 == 0 else -0.0006) for _ in range(PANEL_GRID_TANGENTIAL_COUNT))
        for i in range(PANEL_GRID_RADIAL_COUNT)
    )
    p = BladeParams(**{**_sample().to_dict(), "panel_offsets_m": zz})
    disp = [displacement_at(p, r, 0.0) for r in panel_radial_stations()]
    signs = [d > 0 for d in disp]
    assert signs == [True, False, True, False]


# --- fold (z-stack) + constraint margins ------------------------------------


def test_layer_spacing_is_thickest_rib_plus_clearance():
    p = _sample()  # t_rib_tip (0.0035) is the thickest section
    assert layer_spacing_m(p) == pytest.approx(p.t_rib_tip_m + FOLD_CLEARANCE_M)


def test_folded_stack_height_is_count_times_spacing():
    p = _sample()
    assert folded_stack_height_m(p) == pytest.approx(p.blade_count * layer_spacing_m(p))


def test_fold_margin_is_cap_minus_stack_height():
    p = _sample()
    assert fold_margin_m(p) == pytest.approx(MAX_FOLDED_STACK_HEIGHT_M - folded_stack_height_m(p))


def test_fold_margin_negative_for_fat_stack():
    # Many thick-ribbed blades → the folded bundle exceeds the ergonomic cap.
    p = BladeParams(**{**_sample().to_dict(), "blade_count": 12, "t_rib_tip_m": 0.006})
    assert fold_margin_m(p) < 0.0


def test_containment_margin_positive_for_feasible():
    assert containment_margin_m(_sample()) > 0.0


def test_containment_margin_negative_when_offset_exceeds_rib():
    big = tuple(
        tuple(0.0024 for _ in range(PANEL_GRID_TANGENTIAL_COUNT))
        for _ in range(PANEL_GRID_RADIAL_COUNT)
    )
    p = BladeParams(**{**_sample().to_dict(), "panel_offsets_m": big})
    assert containment_margin_m(p) < 0.0


def test_estimate_mass_positive():
    assert estimate_mass_kg(_sample()) > 0.0


def test_estimate_mass_scales_with_blade_count():
    assert estimate_mass_kg(_sample(12)) > estimate_mass_kg(_sample(8))


def test_mass_margin_is_cap_minus_estimate():
    from fanopt.geometry.schema import MAX_TOTAL_MASS_KG

    p = _sample()
    assert mass_margin_kg(p) == pytest.approx(MAX_TOTAL_MASS_KG - estimate_mass_kg(p))


def test_feasible_true_for_good_design():
    assert feasible(_sample()) is True


def test_feasible_false_when_any_constraint_violated():
    p = BladeParams(**{**_sample().to_dict(), "t_rib_hub_m": 0.006})
    assert feasible(p) is False
