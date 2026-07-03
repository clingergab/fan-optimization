"""Tests for fanopt.bo.structural (panel-stiffness tip-deflection objective)."""

from __future__ import annotations

import pytest

from fanopt.bo.structural import NOMINAL_PRESSURE_PA, panel_tip_deflection_m
from fanopt.geometry.envelope import Layer1Params, ThicknessGridField


def _layer1(field: ThicknessGridField) -> Layer1Params:
    return Layer1Params(
        blade_count=10,
        camber_knots_m=(0.001, 0.002, 0.001),
        twist_knots_rad=(0.05, -0.05),
        thickness_field=field,
        edge_profile="rounded",
        fourier_le_amplitudes=(0.0, 0.0, 0.0),
        fourier_te_amplitudes=(0.0, 0.0, 0.0),
    )


def test_deflection_positive_finite():
    d = panel_tip_deflection_m(_layer1(ThicknessGridField.uniform(0.003)))
    assert d > 0.0
    assert d < 1.0


def test_thicker_panel_deflects_less():
    thin = panel_tip_deflection_m(_layer1(ThicknessGridField.uniform(0.0022)))
    thick = panel_tip_deflection_m(_layer1(ThicknessGridField.uniform(0.0038)))
    assert thick < thin


def test_deflection_scales_linearly_with_pressure():
    field = ThicknessGridField.uniform(0.003)
    d1 = panel_tip_deflection_m(_layer1(field), pressure_pa=10.0)
    d2 = panel_tip_deflection_m(_layer1(field), pressure_pa=20.0)
    assert d2 == pytest.approx(2.0 * d1, rel=1e-6)


def test_corrugation_stiffens_relative_to_flat_same_mean():
    # A corrugated field of the same mean thickness is stiffer (t^3 convexity),
    # so it deflects less than the flat panel at that mean.
    flat = ThicknessGridField.uniform(0.003)
    corrugated = ThicknessGridField(
        grid_m=flat.grid_m, corrugation_amplitude_m=0.0008, corrugation_wavelength=0.25
    )
    d_flat = panel_tip_deflection_m(_layer1(flat))
    d_corr = panel_tip_deflection_m(_layer1(corrugated))
    assert d_corr < d_flat


def test_default_pressure_is_nominal():
    field = ThicknessGridField.uniform(0.003)
    assert panel_tip_deflection_m(_layer1(field)) == pytest.approx(
        panel_tip_deflection_m(_layer1(field), pressure_pa=NOMINAL_PRESSURE_PA)
    )


def test_rejects_bad_mesh_dims():
    with pytest.raises(ValueError, match="nelx, nely"):
        panel_tip_deflection_m(_layer1(ThicknessGridField.uniform(0.003)), nelx=0)
