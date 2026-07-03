"""Tests for fanopt.bo.codec (BO design-vector ↔ Layer1Params codec)."""

from __future__ import annotations

import numpy as np
import pytest

from fanopt.bo import codec
from fanopt.geometry.envelope import Layer1Params, ThicknessGridField
from fanopt.geometry.schema import (
    BLADE_COUNTS,
    PANEL_THICKNESS_MAX_M,
    PANEL_THICKNESS_MIN_M,
    THICKNESS_GRID_RADIAL_COUNT,
    THICKNESS_GRID_TANGENTIAL_COUNT,
)


def _sample_params(blade_count: int = 10) -> Layer1Params:
    grid = tuple(
        tuple(0.0025 + 0.0001 * j for j in range(THICKNESS_GRID_TANGENTIAL_COUNT))
        for _ in range(THICKNESS_GRID_RADIAL_COUNT)
    )
    field = ThicknessGridField(
        grid_m=grid,
        corrugation_amplitude_m=0.0005,
        corrugation_wavelength=0.4,
        corrugation_phase_rad=1.0,
        corrugation_orientation_rad=0.5,
    )
    return Layer1Params(
        blade_count=blade_count,
        camber_knots_m=(0.001, 0.002, 0.001),
        twist_knots_rad=(0.05, -0.05),
        thickness_field=field,
        edge_profile="rounded",
        fourier_le_amplitudes=(0.0, 0.0, 0.0),
        fourier_te_amplitudes=(0.0, 0.0, 0.0),
    )


def test_n_dims_matches_search_space():
    assert len(codec.SEARCH_SPACE) == codec.N_DIMS
    # 18 thickness + 4 corrugation + 3 camber + 2 twist + 1 blade_count = 28.
    assert codec.N_DIMS == 28


def test_bounds_shapes_and_ordering():
    low, high = codec.bounds()
    assert low.shape == high.shape == (codec.N_DIMS,)
    assert np.all(high > low)


def test_encode_decode_round_trips_searched_dims():
    p = _sample_params()
    p2 = codec.decode(codec.encode(p))
    assert p2.blade_count == p.blade_count
    assert p2.thickness_field.grid_m == p.thickness_field.grid_m
    assert p2.thickness_field.corrugation_amplitude_m == pytest.approx(
        p.thickness_field.corrugation_amplitude_m
    )
    assert p2.camber_knots_m == pytest.approx(p.camber_knots_m)
    assert p2.twist_knots_rad == pytest.approx(p.twist_knots_rad)


def test_round_trip_preserves_each_blade_count():
    for bc in BLADE_COUNTS:
        p2 = codec.decode(codec.encode(_sample_params(bc)))
        assert p2.blade_count == bc


def test_decode_wrong_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        codec.decode(np.zeros(codec.N_DIMS - 1))


def test_decode_lower_bounds_is_valid():
    low, _ = codec.bounds()
    p = codec.decode(low)
    assert p.thickness_field.grid_m[0][0] == pytest.approx(PANEL_THICKNESS_MIN_M)


def test_decode_upper_bounds_is_valid():
    _, high = codec.bounds()
    # clip keeps categorical strictly below its top index so decode is valid.
    p = codec.decode(codec.clip_to_bounds(high))
    assert p.thickness_field.grid_m[0][0] == pytest.approx(PANEL_THICKNESS_MAX_M)
    assert p.blade_count == BLADE_COUNTS[-1]


def test_categorical_flooring_maps_ranges_to_choices():
    _, high = codec.bounds()
    bc_idx = next(i for i, v in enumerate(codec.SEARCH_SPACE) if v.name == "blade_count")
    low, _ = codec.bounds()
    for k, choice in enumerate(BLADE_COUNTS):
        vec = low.copy()
        vec[bc_idx] = k + 0.5
        assert codec.decode(vec).blade_count == choice


def test_clip_to_bounds_clamps_out_of_box():
    low, high = codec.bounds()
    over = high + 1.0
    clipped = codec.clip_to_bounds(over)
    assert np.all(clipped <= high)
    assert np.all(clipped >= low)


def test_clip_keeps_categorical_decodable_at_top():
    _, high = codec.bounds()
    clipped = codec.clip_to_bounds(high + 5.0)
    codec.decode(clipped)  # must not raise


def test_var_rejects_bad_kind():
    with pytest.raises(ValueError, match="kind"):
        codec.Var("x", 0.0, 1.0, kind="ordinal")


def test_var_rejects_categorical_without_choices():
    with pytest.raises(ValueError, match="choices"):
        codec.Var("x", 0.0, 1.0, kind="categorical")


def test_var_rejects_degenerate_range():
    with pytest.raises(ValueError, match="high > low"):
        codec.Var("x", 1.0, 1.0)


def test_encode_rejects_wrong_camber_count():
    p = _sample_params()
    bad = Layer1Params(
        blade_count=10,
        camber_knots_m=(0.001, 0.002, 0.001, 0.0),
        twist_knots_rad=(0.05, -0.05),
        thickness_field=p.thickness_field,
        edge_profile="rounded",
        fourier_le_amplitudes=(0.0, 0.0, 0.0),
        fourier_te_amplitudes=(0.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="camber knots"):
        codec.encode(bad)


def test_encode_rejects_wrong_twist_count():
    p = _sample_params()
    bad = Layer1Params(
        blade_count=10,
        camber_knots_m=(0.001, 0.002, 0.001),
        twist_knots_rad=(0.05, -0.05, 0.02),
        thickness_field=p.thickness_field,
        edge_profile="rounded",
        fourier_le_amplitudes=(0.0, 0.0, 0.0),
        fourier_te_amplitudes=(0.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="twist knots"):
        codec.encode(bad)
