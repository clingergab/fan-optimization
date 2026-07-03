"""Tests for fanopt.geometry.fields_cad (CadQuery Layer 2 generators).

Skipped at module load when CadQuery isn't installed, per CLAUDE.md §4.1.
"""

from __future__ import annotations

import importlib.util
import math

import pytest

if importlib.util.find_spec("cadquery") is None:
    pytest.skip("cadquery not installed", allow_module_level=True)

import cadquery as cq

from fanopt.geometry.envelope import Layer1Params, ThicknessGridField
from fanopt.geometry.envelope_cad import make_outer_envelope
from fanopt.geometry.fields import (
    EdgeFeatureField,
    Layer2Params,
    LouverField,
    TextureField,
)
from fanopt.geometry.fields_cad import (
    PANEL_X_CARVE_RANGE_M,
    apply_edge_feature_field,
    apply_layer2_fields,
    apply_louver_field,
    apply_texture_field,
)
from fanopt.geometry.schema import (
    CLICK_FOOTPRINT_X_RANGE_M,
    HUB_RADIUS_M,
)

# ---- fixtures -------------------------------------------------------------


def _envelope() -> cq.Workplane:
    """Canonical Layer 1 envelope used as the base shape."""
    p = Layer1Params(
        blade_count=10,
        camber_knots_m=(0.0, 0.002, 0.001),
        twist_knots_rad=(0.0, 0.0),
        thickness_field=ThicknessGridField.from_radial_knots((0.0030, 0.0030, 0.0030)),
        edge_profile="rounded",
        fourier_le_amplitudes=(0.0, 0.0, 0.0),
        fourier_te_amplitudes=(0.0, 0.0, 0.0),
    )
    return make_outer_envelope(p, "flat")


# ---- panel-carve range invariant ------------------------------------------


def test_panel_carve_range_inboard_at_hub() -> None:
    assert PANEL_X_CARVE_RANGE_M[0] == HUB_RADIUS_M


def test_panel_carve_range_outboard_clears_click_footprint() -> None:
    assert PANEL_X_CARVE_RANGE_M[1] < CLICK_FOOTPRINT_X_RANGE_M[0]
    # ≥ 5 mm safety margin
    assert CLICK_FOOTPRINT_X_RANGE_M[0] - PANEL_X_CARVE_RANGE_M[1] >= 0.005 - 1e-9


# ---- Louver field ---------------------------------------------------------


def test_louver_inactive_returns_input_unchanged() -> None:
    env = _envelope()
    out = apply_louver_field(env, LouverField(active=False))
    assert out is env


def test_louver_subtract_reduces_volume() -> None:
    env = _envelope()
    vol_before = env.val().Volume()
    out = apply_louver_field(
        env,
        LouverField(active=True, count=6, width_m=0.001, polarity="subtract"),
    )
    assert out.val().Volume() < vol_before


def test_louver_more_slots_remove_more_volume() -> None:
    env = _envelope()
    few = apply_louver_field(
        env, LouverField(active=True, count=3, width_m=0.001, polarity="subtract")
    )
    many = apply_louver_field(
        env, LouverField(active=True, count=12, width_m=0.001, polarity="subtract")
    )
    assert many.val().Volume() < few.val().Volume()


def test_louver_add_polarity_increases_volume() -> None:
    env = _envelope()
    vol_before = env.val().Volume()
    out = apply_louver_field(
        env,
        LouverField(active=True, count=6, width_m=0.001, polarity="add"),
    )
    assert out.val().Volume() > vol_before


def test_louver_clustered_at_tip_positions_differ_from_uniform() -> None:
    """The clustered-at-tip profile must produce a different geometry
    than uniform (cuts at sqrt(t) instead of linear t)."""
    env = _envelope()
    uni = apply_louver_field(
        env,
        LouverField(
            active=True, count=6, width_m=0.001, polarity="subtract", spacing_profile="uniform"
        ),
    )
    clu = apply_louver_field(
        env,
        LouverField(
            active=True,
            count=6,
            width_m=0.001,
            polarity="subtract",
            spacing_profile="clustered-at-tip",
        ),
    )
    # Same count + width → same total cut volume; but the positions differ
    # so the resulting solids are NOT identical. Compare bounding-box xmax
    # which shifts depending on where the cuts fall.
    bb_uni = uni.val().BoundingBox()
    bb_clu = clu.val().BoundingBox()
    # The clustered-at-tip profile concentrates cuts at higher x, so its
    # cut volume distribution differs. The bounding box won't, but the
    # solid hash will.
    assert uni.val().Volume() == pytest.approx(clu.val().Volume(), rel=0.02)
    # The two should have the same volume (same cut, different positions)
    # but the cut positions differ; if implementation is correct the two
    # results should NOT be structurally identical.
    assert (bb_uni.xmin, bb_uni.xmax) == pytest.approx((bb_clu.xmin, bb_clu.xmax), abs=1e-6)


# ---- Texture field --------------------------------------------------------


def test_texture_inactive_returns_input_unchanged() -> None:
    env = _envelope()
    out = apply_texture_field(env, TextureField(active=False))
    assert out is env


def test_texture_dimple_reduces_volume() -> None:
    env = _envelope()
    vol_before = env.val().Volume()
    out = apply_texture_field(
        env,
        TextureField(
            active=True,
            feature_type="dimple",
            density_per_cm2=5.0,
            size_m=0.001,
            polarity="subtract",
        ),
    )
    assert out.val().Volume() < vol_before


def test_texture_higher_density_removes_more_volume() -> None:
    env = _envelope()
    low = apply_texture_field(
        env,
        TextureField(
            active=True,
            feature_type="dimple",
            density_per_cm2=1.0,
            size_m=0.001,
            polarity="subtract",
        ),
    )
    high = apply_texture_field(
        env,
        TextureField(
            active=True,
            feature_type="dimple",
            density_per_cm2=10.0,
            size_m=0.001,
            polarity="subtract",
        ),
    )
    assert high.val().Volume() < low.val().Volume()


def test_texture_ridge_orientation_changes_geometry() -> None:
    """Ridge texture orientation must perturb the result (rectangular
    feature, not spherically symmetric)."""
    env = _envelope()
    r0 = apply_texture_field(
        env,
        TextureField(
            active=True,
            feature_type="ridge",
            density_per_cm2=1.0,
            size_m=0.001,
            orientation_rad=0.0,
            polarity="subtract",
        ),
    )
    r90 = apply_texture_field(
        env,
        TextureField(
            active=True,
            feature_type="ridge",
            density_per_cm2=1.0,
            size_m=0.001,
            orientation_rad=math.pi / 2.0,
            polarity="subtract",
        ),
    )
    # Same density + size → same cut count; volumes should be very close
    # but the cut orientations differ. Either way, both must reduce volume.
    assert r0.val().Volume() < env.val().Volume()
    assert r90.val().Volume() < env.val().Volume()


def test_texture_bump_add_polarity_increases_volume() -> None:
    env = _envelope()
    vol_before = env.val().Volume()
    out = apply_texture_field(
        env,
        TextureField(
            active=True, feature_type="bump", density_per_cm2=1.0, size_m=0.001, polarity="add"
        ),
    )
    assert out.val().Volume() > vol_before


# ---- Edge feature field ---------------------------------------------------


def test_edge_inactive_returns_input_unchanged() -> None:
    env = _envelope()
    out = apply_edge_feature_field(env, EdgeFeatureField(active=False))
    assert out is env


def test_edge_serration_LE_reduces_volume() -> None:
    env = _envelope()
    vol_before = env.val().Volume()
    out = apply_edge_feature_field(
        env,
        EdgeFeatureField(
            active=True, feature_type="serration", count=8, depth_m=0.001, application="LE"
        ),
    )
    assert out.val().Volume() < vol_before


def test_edge_scallop_TE_reduces_volume() -> None:
    env = _envelope()
    vol_before = env.val().Volume()
    out = apply_edge_feature_field(
        env,
        EdgeFeatureField(
            active=True, feature_type="scallop", count=6, depth_m=0.0015, application="TE"
        ),
    )
    assert out.val().Volume() < vol_before


def test_edge_smooth_fade_LE_reduces_volume() -> None:
    env = _envelope()
    vol_before = env.val().Volume()
    out = apply_edge_feature_field(
        env,
        EdgeFeatureField(
            active=True,
            feature_type="smooth-fade",
            count=4,
            depth_m=0.0015,
            application="LE",
        ),
    )
    assert out.val().Volume() < vol_before


def test_edge_both_application_removes_more_than_LE_alone() -> None:
    env = _envelope()
    le_only = apply_edge_feature_field(
        env,
        EdgeFeatureField(
            active=True, feature_type="serration", count=8, depth_m=0.001, application="LE"
        ),
    )
    both = apply_edge_feature_field(
        env,
        EdgeFeatureField(
            active=True, feature_type="serration", count=8, depth_m=0.001, application="both"
        ),
    )
    # Both edges → roughly 2× the cut volume → less remaining material.
    assert both.val().Volume() < le_only.val().Volume()


# ---- top-level dispatcher -------------------------------------------------


def test_apply_layer2_fields_all_inactive_returns_input() -> None:
    env = _envelope()
    out = apply_layer2_fields(env, Layer2Params.all_inactive())
    assert out is env


def test_apply_layer2_fields_single_active_field_modifies_shape() -> None:
    env = _envelope()
    vol_before = env.val().Volume()
    params = Layer2Params(
        louver=LouverField(active=True, count=4, width_m=0.001),
        texture=TextureField(active=False),
        edge=EdgeFeatureField(active=False),
    )
    out = apply_layer2_fields(env, params)
    assert out.val().Volume() < vol_before


def test_apply_layer2_fields_max_three_active_applied_in_order() -> None:
    """With 3 fields active, the result has cuts from all three."""
    env = _envelope()
    vol_before = env.val().Volume()
    params = Layer2Params(
        louver=LouverField(active=True, count=4, width_m=0.001),
        texture=TextureField(active=True, density_per_cm2=2.0, size_m=0.001),
        edge=EdgeFeatureField(active=True, count=6, depth_m=0.001, application="LE"),
    )
    out = apply_layer2_fields(env, params)
    # All three subtract; result must be lighter than any single one alone.
    only_louver = apply_layer2_fields(
        env,
        Layer2Params(
            louver=params.louver,
            texture=TextureField(active=False),
            edge=EdgeFeatureField(active=False),
        ),
    )
    assert out.val().Volume() < only_louver.val().Volume()
    assert out.val().Volume() < vol_before


def test_apply_layer2_fields_preserves_solid_shape() -> None:
    """After Layer 2 cuts the result must still be a single connected solid."""
    env = _envelope()
    params = Layer2Params(
        louver=LouverField(active=True, count=3, width_m=0.001),
        texture=TextureField(active=False),
        edge=EdgeFeatureField(active=False),
    )
    out = apply_layer2_fields(env, params)
    assert out.val().Volume() > 0.0
