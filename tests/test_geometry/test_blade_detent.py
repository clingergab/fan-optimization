"""Tests for sizing the hold detent on the blade surface (fanopt.geometry.blade_detent).

A flat plate (bottom z = 0, top z = 3 mm) stacked at 3.8 mm spacing has a 0.8 mm gap everywhere, so every
size is checkable by hand.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fanopt.geometry.blade_detent import (
    DetentGeometry,
    DetentSpec,
    detent_geometry,
    surface_z_range,
)
from fanopt.geometry.rib_detent import DetentSeat

MM = 1e-3
SPACING = 3.8 * MM
SEAT = DetentSeat(
    radius_m=100 * MM,
    bump_angle_deg=-5.0,
    closed_recess_angle_deg=-5.0,
    open_recess_angle_deg=5.0,
    travel_mm=17.5,
)


def _plate(thickness_m=3 * MM, step_m=2 * MM):
    """Top + bottom triangulated sheets over x 50..150 mm, y -30..30 mm."""
    xs = np.arange(50 * MM, 150 * MM + 1e-9, step_m)
    ys = np.arange(-30 * MM, 30 * MM + 1e-9, step_m)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    nx, ny = gx.shape
    flat = np.column_stack([gx.ravel(), gy.ravel()])
    nodes = np.vstack(
        [
            np.column_stack([flat, np.zeros(len(flat))]),
            np.column_stack([flat, np.full(len(flat), thickness_m)]),
        ]
    )
    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1
            tris += [[a, b, c], [a, c, d]]
    tris = np.array(tris)
    return nodes, np.vstack([tris, tris + len(flat)])


@pytest.fixture(scope="module")
def plate():
    return _plate()


@pytest.fixture(scope="module")
def geo(plate):
    return detent_geometry([SEAT], *plate, SPACING)


def _xy(angle_deg):
    t = math.radians(angle_deg)
    return 100 * MM * math.cos(t), 100 * MM * math.sin(t)


def test_surface_range_of_the_plate(plate):
    assert surface_z_range(*plate, *_xy(0.0)) == pytest.approx((0.0, 3 * MM))


def test_surface_range_ignores_vertical_facets(plate):
    nodes, facets = plate
    x, y = _xy(0.0)
    wall = np.array([[x - 1 * MM, y, -9 * MM], [x + 1 * MM, y, -9 * MM], [x, y, 9 * MM]])
    nodes2 = np.vstack([nodes, wall])
    facets2 = np.vstack([facets, [[len(nodes), len(nodes) + 1, len(nodes) + 2]]])
    assert surface_z_range(nodes2, facets2, x, y) == pytest.approx((0.0, 3 * MM))


def test_surface_range_raises_off_the_part(plate):
    with pytest.raises(ValueError, match="no surface"):
        surface_z_range(*plate, 0.0, 0.0)


def test_one_bump_and_two_recesses_per_seat(geo):
    assert isinstance(geo, DetentGeometry) and (len(geo.bumps), len(geo.recesses)) == (1, 2)


def test_bump_apex_crosses_the_gap_plus_interference(geo):
    x, y, z, r = geo.bumps[0]
    assert z + r == pytest.approx(3 * MM + 0.8 * MM + 0.3 * MM)


def test_bump_is_cut_from_the_fixed_sphere(geo):
    assert geo.bumps[0][3] == pytest.approx(1.6 * MM)


def test_report_records_the_bump_base_width(geo):
    _x, _y, z, r = geo.bumps[0]
    assert geo.seats[0]["bump_base_diameter_mm"] == pytest.approx(
        2 * math.sqrt(r**2 - (3 * MM - z) ** 2) / MM
    )


def test_seated_bump_clears_the_recess_floor(geo):
    bump_apex = geo.bumps[0][2] + geo.bumps[0][3]
    recess_floor_above = SPACING + geo.recesses[0][2] + geo.recesses[0][3]  # neighbour layer
    assert recess_floor_above - bump_apex == pytest.approx(0.05 * MM)


def test_recess_is_wider_than_the_bump(geo):
    assert geo.recesses[0][3] == pytest.approx(geo.bumps[0][3] + 0.15 * MM)


def test_recesses_sit_at_the_closed_and_open_angles(geo):
    got = [tuple(r[:2]) for r in geo.recesses]
    assert got == [pytest.approx(_xy(-5.0)), pytest.approx(_xy(5.0))]


def test_report_records_sizes(geo):
    row = geo.seats[0]
    assert (row["gap_closed_mm"], row["bump_height_mm"], row["recess_depth_open_mm"]) == (
        pytest.approx(0.8),
        pytest.approx(1.1),
        pytest.approx(0.35),
    )


def test_interference_is_configurable(plate):
    g = detent_geometry([SEAT], *plate, SPACING, DetentSpec(interference_m=0.5 * MM))
    assert g.seats[0]["bump_height_mm"] == pytest.approx(1.3)


def test_bump_that_would_poke_through_a_thin_blade_is_rejected():
    with pytest.raises(ValueError, match="pokes out"):
        detent_geometry([SEAT], *_plate(thickness_m=1.5 * MM), SPACING - 1.5 * MM)


def test_recess_too_deep_for_the_blade_is_rejected(plate):
    with pytest.raises(ValueError, match="recess would leave"):
        detent_geometry([SEAT], *plate, SPACING, DetentSpec(seat_clearance_m=2.0 * MM))


def test_bump_taller_than_its_sphere_is_rejected(plate):
    with pytest.raises(ValueError, match="taller than its sphere"):
        detent_geometry([SEAT], *plate, SPACING, DetentSpec(interference_m=1.0 * MM))


@pytest.fixture(scope="module")
def pressed(plate):
    """Spacing 2.8 mm < the 3 mm plate: the blades press together (gap -0.2 mm)."""
    return detent_geometry([SEAT], *plate, 2.8 * MM)


def test_pressed_blades_report_the_negative_gap(pressed):
    assert pressed.seats[0]["gap_closed_mm"] == pytest.approx(-0.2)


def test_pressed_blades_get_a_bump_of_just_the_interference(pressed):
    assert pressed.seats[0]["bump_height_mm"] == pytest.approx(0.3)


def test_pressed_blades_recess_takes_the_bump_plus_clearance(pressed):
    assert pressed.seats[0]["recess_depth_closed_mm"] == pytest.approx(0.35)


def test_report_records_the_wall_under_the_recess(geo):
    assert geo.seats[0]["wall_under_recess_mm"] == pytest.approx(3.0 - 0.35)


def test_no_seats_gives_empty_arrays(plate):
    g = detent_geometry([], *plate, SPACING)
    assert g.bumps.shape == (0, 4) and g.recesses.shape == (0, 4)
