"""Unit tests for fanopt.geometry.fields (Layer 2 BO design parameters)."""

from __future__ import annotations

import math

import pytest

from fanopt.geometry.fields import (
    EDGE_FEATURE_TYPES,
    LOUVER_COUNT_RANGE,
    LOUVER_SPACING_PROFILES,
    MAX_ACTIVE_FIELDS,
    EdgeFeatureField,
    Layer2Params,
    LouverField,
    TextureField,
)

# ---- per-field validation -------------------------------------------------


def test_inactive_field_skips_param_validation() -> None:
    """An inactive field with bad params still constructs (validation skipped)."""
    f = LouverField(active=False, count=999, angle_rad=99.9)  # would be invalid if active
    assert f.active is False


def test_louver_count_below_3_fails() -> None:
    with pytest.raises(ValueError, match="LouverField.count"):
        LouverField(active=True, count=2)


def test_louver_count_above_12_fails() -> None:
    with pytest.raises(ValueError, match="LouverField.count"):
        LouverField(active=True, count=13)


def test_louver_count_in_range_passes() -> None:
    for c in (LOUVER_COUNT_RANGE[0], 6, LOUVER_COUNT_RANGE[1]):
        LouverField(active=True, count=c, angle_rad=0.0, width_m=0.001)


def test_louver_angle_over_60deg_fails() -> None:
    with pytest.raises(ValueError, match="LouverField.angle_rad"):
        LouverField(active=True, count=6, angle_rad=math.radians(70.0))


def test_louver_width_below_min_fails() -> None:
    with pytest.raises(ValueError, match="LouverField.width_m"):
        LouverField(active=True, count=6, width_m=0.0001)  # 0.1 mm < 0.5 mm floor


def test_louver_spacing_profile_unknown_fails() -> None:
    with pytest.raises(ValueError, match="LouverField.spacing_profile"):
        LouverField(active=True, count=6, width_m=0.001, spacing_profile="diagonally-rotated")


def test_louver_all_locked_spacing_profiles_pass() -> None:
    for sp in LOUVER_SPACING_PROFILES:
        LouverField(active=True, count=6, width_m=0.001, spacing_profile=sp)


def test_texture_feature_type_unknown_fails() -> None:
    with pytest.raises(ValueError, match="TextureField.feature_type"):
        TextureField(active=True, feature_type="vortex-generator")


def test_texture_size_above_3mm_fails() -> None:
    with pytest.raises(ValueError, match="TextureField.size_m"):
        TextureField(active=True, size_m=0.004)


def test_edge_feature_type_unknown_fails() -> None:
    with pytest.raises(ValueError, match="EdgeFeatureField.feature_type"):
        EdgeFeatureField(active=True, feature_type="wavy-blob")


def test_edge_feature_all_locked_types_pass() -> None:
    for ft in EDGE_FEATURE_TYPES:
        EdgeFeatureField(active=True, feature_type=ft, count=5, depth_m=0.001)


def test_edge_feature_count_above_24_fails() -> None:
    """EDGE_FEATURE_COUNT_RANGE upper bound is 24."""
    with pytest.raises(ValueError, match="EdgeFeatureField.count"):
        EdgeFeatureField(active=True, feature_type="serration", count=25, depth_m=0.001)


def test_edge_feature_count_below_3_fails() -> None:
    """EDGE_FEATURE_COUNT_RANGE lower bound is 3."""
    with pytest.raises(ValueError, match="EdgeFeatureField.count"):
        EdgeFeatureField(active=True, feature_type="serration", count=2, depth_m=0.001)


# ---- Layer2Params aggregator + cardinality bound -------------------------


def test_all_inactive_canonical() -> None:
    p = Layer2Params.all_inactive()
    assert p.active_count() == 0


def test_one_active_passes() -> None:
    p = Layer2Params(
        louver=LouverField(active=True, count=6, width_m=0.001),
        texture=TextureField(active=False),
        edge=EdgeFeatureField(active=False),
    )
    assert p.active_count() == 1


def test_three_active_passes_at_cap() -> None:
    p = Layer2Params(
        louver=LouverField(active=True, count=6, width_m=0.001),
        texture=TextureField(active=True, size_m=0.001),
        edge=EdgeFeatureField(active=True, count=5, depth_m=0.001),
    )
    assert p.active_count() == MAX_ACTIVE_FIELDS


def test_layer2_round_trip() -> None:
    p = Layer2Params(
        louver=LouverField(active=True, count=6, width_m=0.001, angle_rad=0.5),
        texture=TextureField(active=False),
        edge=EdgeFeatureField(active=False),
    )
    d = p.to_dict()
    recovered = Layer2Params.from_dict(d)
    assert recovered == p


def test_inactive_field_with_default_args_validates_via_constructor() -> None:
    """LouverField(active=False) without other args uses the class defaults."""
    f = LouverField(active=False)
    assert f.count == 6  # default
