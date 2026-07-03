"""Tests for fanopt.cfd.panel_slice (Path A+ panel → 2D slice cross-section)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fanopt.cfd.panel_slice import (
    PanelSliceLayout,
    panel_layout_at_radius,
    panel_slice_polygons,
    rib_width_at_radius_m,
)
from fanopt.geometry.envelope import ThicknessGridField
from fanopt.geometry.schema import (
    L_BLADE_M,
    PANEL_THICKNESS_MAX_M,
    PANEL_THICKNESS_MIN_M,
    RIB_BASE_WIDTH_M,
    RIB_THICKNESS_M,
    RIB_TIP_WIDTH_M,
)


def test_rib_width_tapers_base_to_tip():
    assert rib_width_at_radius_m(0.0) == pytest.approx(RIB_BASE_WIDTH_M)
    assert rib_width_at_radius_m(L_BLADE_M) == pytest.approx(RIB_TIP_WIDTH_M)
    mid = rib_width_at_radius_m(0.5 * L_BLADE_M)
    assert RIB_BASE_WIDTH_M < mid < RIB_TIP_WIDTH_M


def test_layout_pitch_equals_arc_at_radius():
    lay = panel_layout_at_radius(0.5)
    # Full pitch is r · inter_blade_angle (13.3°); width + gap must reconstruct it.
    r = 0.5 * L_BLADE_M
    assert lay.pitch_m == pytest.approx(r * math.radians(13.3))
    assert lay.panel_width_m > 0
    assert lay.panel_gap_m > 0


def test_layout_rejects_too_inboard_radius():
    with pytest.raises(ValueError, match="non-positive panel width"):
        panel_layout_at_radius(0.05)  # r = 10 mm: rib widths swamp the pitch


def test_layout_rejects_out_of_range_u():
    with pytest.raises(ValueError, match="radial_u"):
        panel_layout_at_radius(1.5)


def test_polygons_count_matches_n_panels():
    field = ThicknessGridField.uniform()
    polys = panel_slice_polygons(field, n_panels=4)
    assert len(polys) == 4


def test_polygons_are_closed_2d_with_expected_vertex_count():
    field = ThicknessGridField.uniform()
    polys = panel_slice_polygons(field, n_samples=10)
    # 2 bottom vertices + n_samples top vertices.
    assert polys[0].shape == (12, 2)
    assert polys[0].shape[1] == 2


def test_flat_field_top_face_is_uniform_thickness():
    t = 0.003
    field = ThicknessGridField.uniform(t)
    poly = panel_slice_polygons(field, n_samples=8)[0]
    z_lo = -RIB_THICKNESS_M / 2.0
    top_z = poly[2:, 0]  # top-face streamwise coords
    assert np.allclose(top_z, z_lo + t)


def test_corrugation_makes_top_face_non_uniform():
    grid = ThicknessGridField.uniform(0.003).grid_m
    field = ThicknessGridField(
        grid_m=grid, corrugation_amplitude_m=0.0008, corrugation_wavelength=0.3
    )
    poly = panel_slice_polygons(field, n_samples=32)[0]
    top_z = poly[2:, 0]
    assert top_z.std() > 0.0  # corrugation varies the top face


def test_top_face_respects_thickness_lock():
    grid = ThicknessGridField.uniform(PANEL_THICKNESS_MAX_M).grid_m
    field = ThicknessGridField(grid_m=grid, corrugation_amplitude_m=0.0008)
    poly = panel_slice_polygons(field, n_samples=40)[0]
    z_lo = -RIB_THICKNESS_M / 2.0
    thickness = poly[2:, 0] - z_lo
    assert thickness.max() <= PANEL_THICKNESS_MAX_M + 1e-12
    assert thickness.min() >= PANEL_THICKNESS_MIN_M - 1e-12


def test_polygons_centred_on_cross_zero():
    field = ThicknessGridField.uniform()
    polys = panel_slice_polygons(field, n_panels=5)
    all_cross = np.concatenate([p[:, 1] for p in polys])
    assert all_cross.mean() == pytest.approx(0.0, abs=1e-9)


def test_injected_layout_overrides_radius_formula():
    field = ThicknessGridField.uniform()
    lay = PanelSliceLayout(panel_width_m=0.02, panel_gap_m=0.004, radius_m=0.1)
    poly = panel_slice_polygons(field, n_panels=1, layout=lay)[0]
    cross = poly[:, 1]
    assert cross.max() - cross.min() == pytest.approx(0.02)


def test_rejects_bad_n_panels():
    with pytest.raises(ValueError, match="n_panels"):
        panel_slice_polygons(ThicknessGridField.uniform(), n_panels=0)


def test_rejects_bad_n_samples():
    with pytest.raises(ValueError, match="n_samples"):
        panel_slice_polygons(ThicknessGridField.uniform(), n_samples=1)


def test_layout_rejects_non_positive_width():
    with pytest.raises(ValueError, match="panel_width_m must be > 0"):
        PanelSliceLayout(panel_width_m=0.0, panel_gap_m=0.004, radius_m=0.1)


def test_layout_rejects_negative_gap():
    with pytest.raises(ValueError, match="panel_gap_m must be >= 0"):
        PanelSliceLayout(panel_width_m=0.02, panel_gap_m=-0.001, radius_m=0.1)
