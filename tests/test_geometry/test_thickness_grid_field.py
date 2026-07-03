"""Tests for the Path A+ thickness grid field (plan_v1_slim_latest.md §10)."""

from __future__ import annotations

import math

import pytest

from fanopt.geometry.envelope import ThicknessGridField
from fanopt.geometry.schema import (
    PANEL_THICKNESS_MAX_M,
    PANEL_THICKNESS_MIN_M,
    THICKNESS_GRID_RADIAL_COUNT,
    THICKNESS_GRID_TANGENTIAL_COUNT,
)

R, T = THICKNESS_GRID_RADIAL_COUNT, THICKNESS_GRID_TANGENTIAL_COUNT
MID = (PANEL_THICKNESS_MIN_M + PANEL_THICKNESS_MAX_M) / 2.0


def _rows(*values: float) -> tuple[tuple[float, ...], ...]:
    """Grid where radial row i is uniformly ``values[i]``."""
    return tuple(tuple(v for _ in range(T)) for v in values)


# --- uniform / not-forced baseline --------------------------------------------


def test_uniform_is_flat():
    assert ThicknessGridField.uniform().is_flat is True


def test_uniform_thickness_constant_everywhere():
    field = ThicknessGridField.uniform(0.003)
    assert field.thickness_at(0.0, -1.0) == pytest.approx(0.003)
    assert field.thickness_at(1.0, 1.0) == pytest.approx(0.003)
    assert field.thickness_at(0.5, 0.0) == pytest.approx(0.003)


def test_uniform_defaults_to_nominal():
    assert ThicknessGridField.uniform().grid_m[0][0] == pytest.approx(MID)


def test_varying_grid_is_not_flat():
    grid = _rows(0.0022, 0.003, 0.0038)
    assert ThicknessGridField(grid_m=grid).is_flat is False


def test_corrugation_makes_not_flat():
    field = ThicknessGridField(grid_m=_rows(0.003, 0.003, 0.003), corrugation_amplitude_m=0.0005)
    assert field.is_flat is False


# --- bilinear interpolation ---------------------------------------------------


def test_radial_endpoints():
    field = ThicknessGridField(grid_m=_rows(0.0022, 0.003, 0.0038))
    assert field.thickness_at(0.0, 0.0) == pytest.approx(0.0022)
    assert field.thickness_at(1.0, 0.0) == pytest.approx(0.0038)


def test_radial_midpoint_interpolates():
    field = ThicknessGridField(grid_m=_rows(0.0022, 0.003, 0.0038))
    # u=0.25 → between row0 (0.0022) and row1 (0.003), halfway → 0.0026
    assert field.thickness_at(0.25, 0.0) == pytest.approx(0.0026)


def test_tangential_interpolation():
    # each row: first 3 points MIN, last 3 MAX
    row = (PANEL_THICKNESS_MIN_M,) * 3 + (PANEL_THICKNESS_MAX_M,) * 3
    field = ThicknessGridField(grid_m=tuple(row for _ in range(R)))
    assert field.thickness_at(0.0, -1.0) == pytest.approx(PANEL_THICKNESS_MIN_M)
    assert field.thickness_at(0.0, 1.0) == pytest.approx(PANEL_THICKNESS_MAX_M)


# --- corrugation --------------------------------------------------------------


def test_corrugation_adds_expected_bump():
    field = ThicknessGridField(
        grid_m=_rows(0.003, 0.003, 0.003),
        corrugation_amplitude_m=0.0005,
        corrugation_wavelength=0.5,
        corrugation_orientation_rad=0.0,  # tangential ridges
    )
    # θ=0 ⇒ arg = 2π·v_n/λ; at v_n=0.125 (v=-0.75) ⇒ sin(π/2)=1 ⇒ +amplitude
    assert field.thickness_at(0.0, -0.75) == pytest.approx(0.0035)


def test_corrugation_clamps_at_max():
    field = ThicknessGridField(
        grid_m=_rows(*([PANEL_THICKNESS_MAX_M] * R)),
        corrugation_amplitude_m=0.0005,
        corrugation_wavelength=0.5,
    )
    # base already at MAX; a positive bump clamps back to MAX
    assert field.thickness_at(0.0, -0.75) == pytest.approx(PANEL_THICKNESS_MAX_M)


def test_corrugation_clamps_at_min():
    field = ThicknessGridField(
        grid_m=_rows(*([PANEL_THICKNESS_MIN_M] * R)),
        corrugation_amplitude_m=0.0005,
        corrugation_wavelength=0.5,
    )
    # v_n=0.375 (v=-0.25) ⇒ sin(1.5π) = -1 ⇒ negative bump clamps to MIN
    assert field.thickness_at(0.0, -0.25) == pytest.approx(PANEL_THICKNESS_MIN_M)


# --- validation ---------------------------------------------------------------


def test_rejects_wrong_radial_count():
    with pytest.raises(ValueError, match="radial rows"):
        ThicknessGridField(grid_m=_rows(0.003, 0.003))


def test_rejects_wrong_tangential_count():
    # 3 radial rows (passes the radial check) but row 0 is one point short.
    grid = (tuple([0.003] * (T - 1)),) + _rows(0.003, 0.003, 0.003)[1:]
    with pytest.raises(ValueError, match="tangential points"):
        ThicknessGridField(grid_m=grid)


def test_rejects_grid_value_out_of_range():
    with pytest.raises(ValueError, match="outside"):
        ThicknessGridField(grid_m=_rows(0.003, 0.003, 0.005))


def test_rejects_amplitude_out_of_range():
    with pytest.raises(ValueError, match="corrugation_amplitude_m"):
        ThicknessGridField(grid_m=_rows(0.003, 0.003, 0.003), corrugation_amplitude_m=0.01)


def test_rejects_wavelength_out_of_range():
    with pytest.raises(ValueError, match="corrugation_wavelength"):
        ThicknessGridField(grid_m=_rows(0.003, 0.003, 0.003), corrugation_wavelength=0.1)


def test_rejects_phase_out_of_range():
    with pytest.raises(ValueError, match="corrugation_phase_rad"):
        ThicknessGridField(grid_m=_rows(0.003, 0.003, 0.003), corrugation_phase_rad=7.0)


def test_rejects_orientation_out_of_range():
    with pytest.raises(ValueError, match="corrugation_orientation_rad"):
        ThicknessGridField(grid_m=_rows(0.003, 0.003, 0.003), corrugation_orientation_rad=math.pi)


# --- serialization ------------------------------------------------------------


def test_to_from_dict_roundtrip():
    field = ThicknessGridField(
        grid_m=_rows(0.0022, 0.003, 0.0038),
        corrugation_amplitude_m=0.0004,
        corrugation_wavelength=0.6,
        corrugation_phase_rad=1.0,
        corrugation_orientation_rad=0.5,
    )
    assert ThicknessGridField.from_dict(field.to_dict()) == field


def test_from_dict_defaults_corrugation_off():
    field = ThicknessGridField.from_dict({"grid_m": [[0.003] * T for _ in range(R)]})
    assert field.is_flat is True
