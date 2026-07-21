"""Tests for fanopt.geometry.blade_viz (blade surface grids for 3D plots)."""

from __future__ import annotations

import numpy as np
import pytest

from fanopt.geometry.blade import BladeParams
from fanopt.geometry.blade_viz import blade_surface_xyz

_GRID = (
    (0.0004, 0.0009, 0.0004),
    (0.0005, 0.0011, 0.0005),
    (0.0006, 0.0013, 0.0006),
    (0.0007, 0.0015, 0.0007),
)
_FLAT = tuple((0.0, 0.0, 0.0) for _ in range(4))


def _params(grid) -> BladeParams:
    return BladeParams(
        blade_count=10, rib_bow_knots_m=(0.005, 0.010, 0.013, 0.017, 0.020), rib_bow_interp="linear",
        t_rib_hub_m=0.0025, t_rib_tip_m=0.0035, panel_offsets_m=grid, panel_thickness_nom_m=0.0016,
    )


def test_shapes_match_requested_resolution():
    x, y, z = blade_surface_xyz(_params(_GRID), n_radial=30, n_tangential=20)
    assert x.shape == y.shape == z.shape == (30, 20)


def test_top_is_above_bottom():
    _, _, z_top = blade_surface_xyz(_params(_GRID), face="top")
    _, _, z_bot = blade_surface_xyz(_params(_GRID), face="bottom")
    assert np.all(z_top >= z_bot)


def test_camber_changes_the_surface():
    _, _, z_flat = blade_surface_xyz(_params(_FLAT), face="top")
    _, _, z_camber = blade_surface_xyz(_params(_GRID), face="top")
    # Same dish + thickness, but the displacement grid must alter the surface.
    assert not np.allclose(z_flat, z_camber)


def test_invalid_face_raises():
    with pytest.raises(ValueError, match="face"):
        blade_surface_xyz(_params(_GRID), face="side")
