"""Unit tests for fanopt.geometry.fields (Layer 2 BO design parameters)."""
from __future__ import annotations

import math

import pytest

from fanopt.geometry.fields import (
    EDGE_FEATURE_TYPES,
    LOUVER_COUNT_RANGE,
    LOUVER_SPACING_PROFILES,
    MAX_ACTIVE_FIELDS,
    NOISE_THRESHOLD_RETENTION_MIN,
    TPMS_CELL_SIZE_MIN_M,
    TPMS_LATTICE_TYPES,
    EdgeFeatureField,
    Layer2Params,
    LouverField,
    NoiseField,
    TextureField,
    TpmsField,
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
        LouverField(
            active=True, count=6, width_m=0.001, spacing_profile="diagonally-rotated"
        )


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


def test_noise_threshold_below_40pct_fails() -> None:
    with pytest.raises(ValueError, match="threshold_retention"):
        NoiseField(active=True, threshold_retention=0.30)


def test_noise_threshold_above_1_fails() -> None:
    with pytest.raises(ValueError, match="threshold_retention"):
        NoiseField(active=True, threshold_retention=1.5)


def test_noise_threshold_at_floor_passes() -> None:
    NoiseField(active=True, threshold_retention=NOISE_THRESHOLD_RETENTION_MIN)


def test_tpms_cell_size_below_3x_min_feature_fails() -> None:
    """Plan: cell_size ≥ 3 × MIN_FEATURE_SIZE_M = 2.4 mm."""
    with pytest.raises(ValueError, match="TpmsField.cell_size_m"):
        TpmsField(active=True, cell_size_m=0.001)  # 1 mm < 2.4 mm floor


def test_tpms_cell_size_at_floor_passes() -> None:
    TpmsField(active=True, cell_size_m=TPMS_CELL_SIZE_MIN_M)


def test_tpms_lattice_type_unknown_fails() -> None:
    with pytest.raises(ValueError, match="TpmsField.lattice_type"):
        TpmsField(active=True, lattice_type="fractal-pearl")


def test_tpms_all_locked_lattice_types_pass() -> None:
    for lt in TPMS_LATTICE_TYPES:
        TpmsField(active=True, lattice_type=lt, cell_size_m=TPMS_CELL_SIZE_MIN_M)


# ---- Layer2Params aggregator + cardinality bound -------------------------


def test_all_inactive_canonical() -> None:
    p = Layer2Params.all_inactive()
    assert p.active_count() == 0


def test_one_active_passes() -> None:
    p = Layer2Params(
        louver=LouverField(active=True, count=6, width_m=0.001),
        texture=TextureField(active=False),
        edge=EdgeFeatureField(active=False),
        noise=NoiseField(active=False),
        tpms=TpmsField(active=False),
    )
    assert p.active_count() == 1


def test_three_active_passes_at_cap() -> None:
    p = Layer2Params(
        louver=LouverField(active=True, count=6, width_m=0.001),
        texture=TextureField(active=True, size_m=0.001),
        edge=EdgeFeatureField(active=True, count=5, depth_m=0.001),
        noise=NoiseField(active=False),
        tpms=TpmsField(active=False),
    )
    assert p.active_count() == MAX_ACTIVE_FIELDS


def test_four_active_fails_over_cap() -> None:
    with pytest.raises(ValueError, match="MAX_ACTIVE_FIELDS"):
        Layer2Params(
            louver=LouverField(active=True, count=6, width_m=0.001),
            texture=TextureField(active=True, size_m=0.001),
            edge=EdgeFeatureField(active=True, count=5, depth_m=0.001),
            noise=NoiseField(active=True, threshold_retention=0.5),
            tpms=TpmsField(active=False),
        )


def test_layer2_round_trip() -> None:
    p = Layer2Params(
        louver=LouverField(active=True, count=6, width_m=0.001, angle_rad=0.5),
        texture=TextureField(active=False),
        edge=EdgeFeatureField(active=False),
        noise=NoiseField(active=True, threshold_retention=0.6, x_scale=12.0),
        tpms=TpmsField(active=False),
    )
    d = p.to_dict()
    recovered = Layer2Params.from_dict(d)
    assert recovered == p


def test_inactive_field_with_default_args_validates_via_constructor() -> None:
    """LouverField(active=False) without other args uses the class defaults."""
    f = LouverField(active=False)
    assert f.count == 6  # default
