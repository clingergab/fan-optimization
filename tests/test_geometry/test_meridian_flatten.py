"""Tests for fanopt.geometry.meridian_flatten (unbend a blade at the same reach and true wall)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from fanopt.bo.blade_codec import bounds, clip_to_bounds, decode
from fanopt.geometry.blade import (
    MERIDIAN_ROOT_FLAT_RADIUS_M,
    rib_bow_stations,
    rib_z_at,
)
from fanopt.geometry.meridian_flatten import flat_meridian_params, flatten_meridian

# Linear meridian rising 4 mm per station: a constant slope inside each station interval.
_KNOTS = (0.004, 0.008, 0.012, 0.016, 0.020)


@pytest.fixture(scope="module")
def ramp():
    low, high = bounds()
    params = decode(clip_to_bounds(low + (high - low) * 0.5))
    return replace(params, rib_bow_knots_m=_KNOTS, rib_bow_interp="linear")


def _station_mid_and_slope() -> tuple[float, float]:
    s = rib_bow_stations()
    return (s[1] + s[2]) / 2.0, (_KNOTS[2] - _KNOTS[1]) / (s[2] - s[1])


def _pair_at(params, rho: float, half: float, *, angle: float = 0.0) -> np.ndarray:
    x, y = rho * math.cos(angle), rho * math.sin(angle)
    mid = rib_z_at(params, rho)
    return np.array([[x, y, mid + half], [x, y, mid - half]])


def test_flat_params_zero_every_knot(ramp):
    assert flat_meridian_params(ramp).rib_bow_knots_m == (0.0,) * 5


def test_flat_params_keep_the_other_knobs(ramp):
    flat = flat_meridian_params(ramp)
    assert replace(flat, rib_bow_knots_m=ramp.rib_bow_knots_m) == ramp


def test_meridian_point_lands_on_the_plane(ramp):
    rho, _ = _station_mid_and_slope()
    out = flatten_meridian(np.array([[rho, 0.0, rib_z_at(ramp, rho)]]), ramp)
    assert out[0, 2] == pytest.approx(0.0, abs=1e-9)


def test_wall_shrinks_by_the_cosine_of_the_slope(ramp):
    rho, slope = _station_mid_and_slope()
    out = flatten_meridian(_pair_at(ramp, rho, 1.0e-3), ramp)
    assert out[0, 2] - out[1, 2] == pytest.approx(2.0e-3 / math.hypot(1.0, slope), rel=1e-6)


def test_wall_uses_the_true_radius_off_the_axis(ramp):
    rho, slope = _station_mid_and_slope()
    out = flatten_meridian(_pair_at(ramp, rho, 1.0e-3, angle=0.3), ramp)
    assert out[0, 2] - out[1, 2] == pytest.approx(2.0e-3 / math.hypot(1.0, slope), rel=1e-6)


def test_planform_is_unchanged(ramp):
    rho, _ = _station_mid_and_slope()
    pts = _pair_at(ramp, rho, 1.0e-3, angle=0.3)
    np.testing.assert_allclose(flatten_meridian(pts, ramp)[:, :2], pts[:, :2])


def test_boss_region_is_unchanged(ramp):
    pts = np.array([[MERIDIAN_ROOT_FLAT_RADIUS_M / 2.0, 0.0, 1.0e-3]])
    np.testing.assert_allclose(flatten_meridian(pts, ramp), pts)


def test_flat_blade_is_unchanged(ramp):
    flat = flat_meridian_params(ramp)
    rho, _ = _station_mid_and_slope()
    pts = np.array([[rho, 0.01, 1.0e-3], [rho, 0.01, -1.0e-3]])
    np.testing.assert_allclose(flatten_meridian(pts, flat), pts)


def test_input_nodes_are_not_modified(ramp):
    rho, _ = _station_mid_and_slope()
    pts = _pair_at(ramp, rho, 1.0e-3)
    before = pts.copy()
    flatten_meridian(pts, ramp)
    np.testing.assert_array_equal(pts, before)
