"""Tests for the open/closed hold-detent placement (fanopt.geometry.rib_detent)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fanopt.bo.blade_codec import N_DIMS, decode
from fanopt.geometry.rib_detent import (
    DEFAULT_DEPLOY_DIRECTION,
    DetentSeat,
    rib_detent_seats,
    rib_half_angle_deg,
)
from fanopt.geometry.schema import INTER_BLADE_ANGLE_RAD

PITCH_DEG = math.degrees(INTER_BLADE_ANGLE_RAD)


def _params() -> object:
    return decode(np.full(N_DIMS, 0.5))


def test_half_angle_is_about_half_pitch_at_the_tip():
    # The blade is ~one pitch wide: its angular half-width at the tip ~= half the deploy pitch. That
    # coincidence is what lets one rotation carry a bump from rib to rib (closed->open).
    assert rib_half_angle_deg(0.219) == pytest.approx(PITCH_DEG / 2.0, abs=0.3)


def test_half_angle_grows_toward_the_hub():
    assert rib_half_angle_deg(0.09) > rib_half_angle_deg(0.19)


def test_returns_requested_number_of_seats():
    seats = rib_detent_seats(_params(), n_seats=5)
    assert len(seats) == 5
    assert all(isinstance(s, DetentSeat) for s in seats)


def test_seats_span_the_requested_band():
    seats = rib_detent_seats(_params(), n_seats=4, r_min_m=0.09, r_max_m=0.12)
    assert seats[0].radius_m == pytest.approx(0.09)
    assert seats[-1].radius_m == pytest.approx(0.12)


def test_closed_recess_sits_on_the_bump_rib():
    # CLOSED = blades aligned, so the bump lands under its OWN rib angle on the neighbour.
    for s in rib_detent_seats(_params()):
        assert s.closed_recess_angle_deg == pytest.approx(s.bump_angle_deg)


def test_open_recess_is_one_pitch_across_from_closed():
    # OPEN = neighbour rotated one deploy pitch, so the bump lands one pitch away in the neighbour frame.
    for s in rib_detent_seats(_params(), deploy_direction=-1):
        assert s.open_recess_angle_deg - s.closed_recess_angle_deg == pytest.approx(PITCH_DEG)


def test_default_direction_puts_bumps_on_the_left_rib():
    # deploy_direction=-1 (the notebook default): the overlapping rib is the LEFT (negative-angle) one,
    # and the open landing swings across toward the RIGHT rib (matches the hand-derived layout).
    seats = rib_detent_seats(_params(), deploy_direction=-1)
    assert all(s.bump_angle_deg < 0 for s in seats)  # left rib
    assert all(s.open_recess_angle_deg > 0 for s in seats)  # swings across to the far (right) side


def test_direction_flips_the_bump_rib():
    left = rib_detent_seats(_params(), deploy_direction=-1)
    right = rib_detent_seats(_params(), deploy_direction=1)
    assert left[0].bump_angle_deg == pytest.approx(-right[0].bump_angle_deg)


def test_open_and_closed_recesses_both_land_on_the_blade():
    # Both seats must be within the neighbour planform (|angle| <= half-width) or they can't catch.
    for s in rib_detent_seats(_params()):
        alpha = rib_half_angle_deg(s.radius_m)
        assert abs(s.closed_recess_angle_deg) <= alpha + 1e-9
        assert abs(s.open_recess_angle_deg) <= alpha + 1e-9


def test_travel_grows_with_radius():
    seats = rib_detent_seats(_params(), n_seats=3, r_min_m=0.09, r_max_m=0.12)
    assert seats[0].travel_mm < seats[-1].travel_mm
    assert seats[-1].travel_mm == pytest.approx(0.12 * INTER_BLADE_ANGLE_RAD * 1e3, abs=0.1)


def test_bump_is_inside_the_rib_edge():
    # The bump sits a margin inside the physical rib so a real feature has material around it.
    for s in rib_detent_seats(_params()):
        assert abs(s.bump_angle_deg) < rib_half_angle_deg(s.radius_m)


def test_tip_band_without_overlap_raises():
    # Right at the tip the overlap collapses: a seat there can't hold OPEN, so it must error.
    with pytest.raises(ValueError, match="hold OPEN"):
        rib_detent_seats(_params(), n_seats=2, r_min_m=0.215, r_max_m=0.219)


def test_outer_band_too_thin_to_hold_open_raises():
    # The dual-hold band is inboard; pushing r_max out to 0.18 leaves no room for the open seat.
    with pytest.raises(ValueError, match="hold OPEN"):
        rib_detent_seats(_params(), n_seats=4, r_min_m=0.1, r_max_m=0.18)


def test_bad_direction_raises():
    with pytest.raises(ValueError, match="deploy_direction"):
        rib_detent_seats(_params(), deploy_direction=0)


def test_bad_band_raises():
    with pytest.raises(ValueError, match="r_min_m"):
        rib_detent_seats(_params(), r_min_m=0.2, r_max_m=0.1)


def test_bad_count_raises():
    with pytest.raises(ValueError, match="n_seats"):
        rib_detent_seats(_params(), n_seats=0)


def test_default_direction_matches_the_notebook():
    assert DEFAULT_DEPLOY_DIRECTION == -1
