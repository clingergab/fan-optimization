"""Tests for fanopt.bo.blade_codec (BO vector ↔ BladeParams codec, free-panel space)."""

from __future__ import annotations

import numpy as np
import pytest

from fanopt.bo import blade_codec as codec
from fanopt.geometry.blade import (
    PANEL_GRID_RADIAL_COUNT,
    PANEL_GRID_TANGENTIAL_COUNT,
    BladeParams,
    containment_margin_m,
    fold_margin_m,
)
from fanopt.geometry.schema import BLADE_COUNTS


def _random_vec(seed: int) -> np.ndarray:
    low, high = codec.bounds()
    rng = np.random.default_rng(seed)
    return codec.clip_to_bounds(low + rng.random(codec.N_DIMS) * (high - low))

_SAMPLE_GRID = (
    (0.0003, 0.0005, 0.0003),
    (0.0004, 0.0006, 0.0004),
    (0.0005, 0.0007, 0.0005),
    (0.0006, 0.0008, 0.0006),
)


def _sample(blade_count: int = 10) -> BladeParams:
    return BladeParams(
        blade_count=blade_count,
        rib_bow_mid_m=0.010,
        rib_bow_tip_m=0.020,
        t_rib_hub_m=0.0025,
        t_rib_tip_m=0.0035,
        panel_offsets_m=_SAMPLE_GRID,
        panel_thickness_nom_m=0.0013,
    )


def test_n_dims_matches_layout():
    # 2 rib meridian + 2 rib thickness + grid + 1 panel thickness + 1 blade_count.
    expected = 4 + PANEL_GRID_RADIAL_COUNT * PANEL_GRID_TANGENTIAL_COUNT + 2
    assert codec.N_DIMS == expected


def test_grid_var_count():
    grid_vars = [v for v in codec.SEARCH_SPACE if v.name.startswith("panel_z_")]
    assert len(grid_vars) == PANEL_GRID_RADIAL_COUNT * PANEL_GRID_TANGENTIAL_COUNT


def test_leading_and_trailing_names():
    names = [v.name for v in codec.SEARCH_SPACE]
    assert names[:4] == ["rib_bow_mid_m", "rib_bow_tip_m", "t_rib_hub_k", "t_rib_tip_k"]
    assert names[-2:] == ["panel_thickness_k", "blade_count"]


def test_bounds_shapes():
    low, high = codec.bounds()
    assert low.shape == (codec.N_DIMS,) and high.shape == (codec.N_DIMS,)


def test_bounds_high_above_low():
    low, high = codec.bounds()
    assert np.all(high > low)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_encode_decode_roundtrips_decoded_design(seed):
    # decode maps a vector into the feasible space; encode/decode is then idempotent.
    p = codec.decode(_random_vec(seed))
    assert codec.decode(codec.encode(p)) == p


def test_decode_is_always_fold_and_containment_feasible():
    # The whole point of the reparam: EVERY vector decodes to a foldable, contained
    # blade (mass stays a soft check, so it's excluded here).
    for seed in range(150):
        p = codec.decode(_random_vec(seed))
        assert fold_margin_m(p) >= -1e-12
        assert containment_margin_m(p) >= -1e-12


def test_decode_wrong_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        codec.decode(np.zeros(codec.N_DIMS - 1))


def test_clip_clamps_continuous_dims():
    low, high = codec.bounds()
    clipped = codec.clip_to_bounds(high + 1.0)
    assert np.all(clipped[:-1] <= high[:-1] + 1e-12)


def test_clip_keeps_categorical_below_upper():
    _, high = codec.bounds()
    clipped = codec.clip_to_bounds(high + 1.0)
    assert clipped[-1] < high[-1]


@pytest.mark.parametrize("blade_count", BLADE_COUNTS)
def test_decode_maps_blade_count(blade_count):
    vec = codec.encode(_sample(blade_count))
    assert codec.decode(vec).blade_count == blade_count


def test_var_rejects_bad_bounds():
    with pytest.raises(ValueError, match="high > low"):
        codec.Var("x", 1.0, 1.0)


def test_var_categorical_needs_choices():
    with pytest.raises(ValueError, match="choices"):
        codec.Var("c", 0.0, 3.0, kind="categorical")


def test_var_rejects_bad_kind():
    with pytest.raises(ValueError, match="continuous|categorical"):
        codec.Var("x", 0.0, 1.0, kind="ordinal")
