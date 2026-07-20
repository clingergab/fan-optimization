"""Tests for fanopt.cfd.blade_slice (solid blade → 2D cascade-slice cross-section)."""

from __future__ import annotations

import numpy as np
import pytest

from fanopt.cfd.blade_slice import (
    INTER_BLADE_GAP_M,
    blade_chord_span_m,
    blade_slice_polygons,
    radius_at_u,
)
from fanopt.geometry.blade import RIB_TIP_RADIUS_M, BladeParams
from fanopt.geometry.schema import HUB_RADIUS_M, INTER_BLADE_ANGLE_RAD

_CAMBER_GRID = (
    (0.0003, 0.0005, 0.0003),
    (0.0004, 0.0006, 0.0004),
    (0.0005, 0.0007, 0.0005),
    (0.0006, 0.0008, 0.0006),
)
_FLAT_GRID = tuple((0.0, 0.0, 0.0) for _ in range(4))


def _params(grid) -> BladeParams:
    return BladeParams(
        blade_count=10,
        rib_bow_mid_m=0.010,
        rib_bow_tip_m=0.020,
        t_rib_hub_m=0.0025,
        t_rib_tip_m=0.0035,
        panel_offsets_m=grid,
        panel_thickness_nom_m=0.0013,
    )


# --- radius / chord ----------------------------------------------------------


def test_radius_at_u_endpoints():
    assert radius_at_u(0.0) == pytest.approx(HUB_RADIUS_M)
    assert radius_at_u(1.0) == pytest.approx(RIB_TIP_RADIUS_M)


def test_radius_at_u_out_of_range_raises():
    with pytest.raises(ValueError, match="radial_u"):
        radius_at_u(1.5)


def test_chord_span_is_arc_minus_gap():
    r = 0.10
    assert blade_chord_span_m(r) == pytest.approx(r * INTER_BLADE_ANGLE_RAD - INTER_BLADE_GAP_M)


def test_chord_span_grows_with_radius():
    assert blade_chord_span_m(0.18) > blade_chord_span_m(0.05)


# --- slice polygons ----------------------------------------------------------


def test_returns_n_panels():
    polys = blade_slice_polygons(_params(_CAMBER_GRID), n_panels=5)
    assert len(polys) == 5


def test_polygon_point_count_is_two_faces():
    polys = blade_slice_polygons(_params(_CAMBER_GRID), n_panels=1, n_samples=24)
    assert polys[0].shape == (48, 2)  # bottom face + top face


def test_flat_blade_has_uniform_thickness_and_flat_mean():
    # Zero displacement → both faces flat, separated by exactly panel_thickness_nom.
    p = _params(_FLAT_GRID)
    poly = blade_slice_polygons(p, n_panels=1, n_samples=20)[0]
    z = poly[:, 0]
    assert (z.max() - z.min()) == pytest.approx(p.panel_thickness_nom_m)


def test_cambered_blade_bows_the_mean_surface():
    # Non-zero displacement → the mean surface varies across the chord (both faces bow).
    poly = blade_slice_polygons(_params(_CAMBER_GRID), n_panels=1, n_samples=24)[0]
    n = poly.shape[0] // 2
    bottom_z = poly[:n, 0]
    assert bottom_z.max() - bottom_z.min() > 1e-5  # the face is not flat


def test_edges_pinned_to_zero_camber():
    # v = ±1 (first/last tangential sample) → displacement 0 → face at ±half thickness.
    p = _params(_CAMBER_GRID)
    poly = blade_slice_polygons(p, n_panels=1, n_samples=24)[0]
    n = poly.shape[0] // 2
    bottom_z = poly[:n, 0]  # bottom face left→right
    assert bottom_z[0] == pytest.approx(-p.panel_thickness_nom_m / 2.0)
    assert bottom_z[-1] == pytest.approx(-p.panel_thickness_nom_m / 2.0)


def test_cascade_centered_on_cross_zero():
    polys = blade_slice_polygons(_params(_CAMBER_GRID), n_panels=5, n_samples=16)
    cross_all = np.concatenate([poly[:, 1] for poly in polys])
    assert cross_all.mean() == pytest.approx(0.0, abs=1e-9)


def test_invalid_n_panels_raises():
    with pytest.raises(ValueError, match="n_panels"):
        blade_slice_polygons(_params(_CAMBER_GRID), n_panels=0)


def test_invalid_n_samples_raises():
    with pytest.raises(ValueError, match="n_samples"):
        blade_slice_polygons(_params(_CAMBER_GRID), n_samples=1)
