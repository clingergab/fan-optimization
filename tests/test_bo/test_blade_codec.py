"""Tests for fanopt.bo.blade_codec (BO vector ↔ BladeParams codec, free-panel space)."""

from __future__ import annotations

import numpy as np
import pytest

from fanopt.bo import blade_codec as codec
from fanopt.geometry.blade import (
    PANEL_GRID_RADIAL_COUNT,
    PANEL_GRID_TANGENTIAL_COUNT,
    RIB_BOW_KNOT_COUNT,
    BladeParams,
    containment_margin_m,
    displacement_at,
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
_P3 = tuple((0.003, 0.003, 0.003) for _ in range(4))


def _sample(blade_count: int = 10) -> BladeParams:
    return BladeParams(
        blade_count=blade_count,
        rib_bow_knots_m=(0.005, 0.010, 0.013, 0.017, 0.020),
        rib_bow_interp="linear",
        t_rib_hub_m=0.005,
        t_rib_tip_m=0.006,
        panel_offsets_m=_SAMPLE_GRID,
        panel_thickness_m=_P3,
        uniform=False,
    )


def _vec_with(rib_mode: str = "ribbed", blade_count: int = 10, panel_z: float = 0.0) -> np.ndarray:
    """A search vector with a chosen rib_mode / blade_count and all panel_z knobs at ``panel_z``."""
    low, high = codec.bounds()
    v = (low + high) / 2.0
    v[codec._IDX["rib_mode"]] = codec.RIB_MODES.index(rib_mode) + 0.5
    v[codec._IDX["blade_count"]] = BLADE_COUNTS.index(blade_count) + 0.5
    for i in range(PANEL_GRID_RADIAL_COUNT):
        for j in range(PANEL_GRID_TANGENTIAL_COUNT):
            v[codec._IDX[f"panel_z_{i}_{j}"]] = panel_z
    return codec.clip_to_bounds(v)


def test_n_dims_matches_layout():
    # K meridian knots + 1 interp + 1 rib_mode + 2 rib thickness + 2 panel grids
    # (offset + thickness) + 1 blade_count.
    expected = (
        RIB_BOW_KNOT_COUNT + 1 + 1 + 2 + 2 * PANEL_GRID_RADIAL_COUNT * PANEL_GRID_TANGENTIAL_COUNT + 1
    )
    assert codec.N_DIMS == expected


def test_grid_var_count():
    n = PANEL_GRID_RADIAL_COUNT * PANEL_GRID_TANGENTIAL_COUNT
    offset_vars = [v for v in codec.SEARCH_SPACE if v.name.startswith("panel_z_")]
    thick_vars = [v for v in codec.SEARCH_SPACE if v.name.startswith("panel_thick_")]
    assert len(offset_vars) == n  # mean-surface offset grid
    assert len(thick_vars) == n  # per-node thickness grid (independent faces)


def test_leading_and_trailing_names():
    names = [v.name for v in codec.SEARCH_SPACE]
    assert names[:RIB_BOW_KNOT_COUNT] == [f"rib_bow_k{i}" for i in range(RIB_BOW_KNOT_COUNT)]
    assert names[RIB_BOW_KNOT_COUNT] == "rib_bow_interp"
    assert names[-1] == "blade_count"
    assert names[-2].startswith("panel_thick_")


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


def test_search_space_has_rib_mode_categorical():
    v = next(v for v in codec.SEARCH_SPACE if v.name == "rib_mode")
    assert v.kind == "categorical" and v.choices == codec.RIB_MODES


def test_decode_uniform_mode_sets_flag():
    assert codec.decode(_vec_with(rib_mode="uniform")).uniform is True


def test_decode_ribbed_mode_clears_flag():
    assert codec.decode(_vec_with(rib_mode="ribbed")).uniform is False


def test_uniform_offset_knobs_are_live():
    # B-proper: cranking the panel_z knobs in uniform mode must move the (unpinned) sheet edge —
    # the no-rib panel has the same wave freedom as the ribbed one, not a dead pinned-flat panel.
    flat = codec.decode(_vec_with(rib_mode="uniform", blade_count=8, panel_z=0.0))
    waved = codec.decode(_vec_with(rib_mode="uniform", blade_count=8, panel_z=0.9))
    assert displacement_at(waved, 0.10, 1.0) > displacement_at(flat, 0.10, 1.0)
    assert displacement_at(waved, 0.10, 1.0) > 0.0  # edge is unpinned (nonzero)


def test_uniform_decode_is_fold_and_nesting_feasible():
    # Every uniform vector decodes to a sheet that nests under the fold cap by construction.
    for seed in range(60):
        low, high = codec.bounds()
        rng = np.random.default_rng(1000 + seed)
        v = codec.clip_to_bounds(low + rng.random(codec.N_DIMS) * (high - low))
        v[codec._IDX["rib_mode"]] = codec.RIB_MODES.index("uniform") + 0.5
        p = codec.decode(v)
        assert p.uniform is True
        assert fold_margin_m(p) >= -1e-12
        assert containment_margin_m(p) >= -1e-12


@pytest.mark.parametrize("rib_mode", ["ribbed", "uniform"])
def test_roundtrip_both_modes(rib_mode):
    p = codec.decode(_vec_with(rib_mode=rib_mode, blade_count=8, panel_z=0.7))
    assert codec.decode(codec.encode(p)) == p


def test_none_fallback_cap_is_fold_safe_for_smooth_overshoot():
    # L1: the bow_extent=None fallback must be a fold-SAFE lower bound — even for the worst
    # Catmull-Rom overshoot, cap(None) must not exceed cap(exact extent) (never over-optimistic,
    # which would certify an unfoldable rib). The nominal knot span alone is NOT a safe bound.
    from fanopt.geometry.blade import rib_meridian_extent_m

    worst = [0.03, 0.03, 0.0, 0.0, 0.03]  # empirical worst-overshoot knot corner (smooth)
    exact = rib_meridian_extent_m(worst, "smooth")
    for n in (8, 10, 12):
        assert codec.rib_thickness_cap_m(n, None) <= codec.rib_thickness_cap_m(n, exact) + 1e-12
