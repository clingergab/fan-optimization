"""Tests for fanopt.physical.print_effort (as-printed mass, wrist inertia, inertial effort, re-ranking)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from fanopt.bo.blade_codec import bounds, clip_to_bounds, decode
from fanopt.geometry.blade import RIB_TIP_RADIUS_M, half_width_at
from fanopt.physical.print_effort import (
    BladeSheet,
    PrintMassModel,
    areal_density_kg_m2,
    blade_mass_kg,
    blade_sheet,
    design_effort,
    fan_com_wrist_m,
    fan_inertia_wrist_kgm2,
    fit_print_mass_model,
    inertial_work_per_cycle_j,
    mean_thickness_m,
    rerank_summary,
    scaled_to_mean_thickness,
)

_MODEL = PrintMassModel(wall_m=1.6e-3, core_fraction=0.15, base_kg=0.002, rho_kg_m3=1240.0)


@pytest.fixture(scope="module")
def flat():
    low, high = bounds()
    p = decode(clip_to_bounds(low + (high - low) * 0.5))
    zeros = tuple(tuple(0.0 for _ in row) for row in p.panel_offsets_m)
    return replace(p, rib_bow_knots_m=(0.0,) * 5, panel_offsets_m=zeros)


@pytest.fixture(scope="module")
def bowed(flat):
    return replace(flat, rib_bow_knots_m=(0.01, 0.02, 0.02, 0.01, 0.0), rib_bow_interp="smooth")


def _one_cell(x=0.1, y=0.0, z=0.01, t=1.0e-3, area=1.0e-4) -> BladeSheet:
    return BladeSheet(*(np.array([v]) for v in (x, y, z, t, area)))


_SOLID = PrintMassModel(wall_m=1.0, core_fraction=1.0, base_kg=0.0, rho_kg_m3=1000.0)


def test_sheet_covers_the_trapezoid(flat):
    r = np.linspace(0.0, RIB_TIP_RADIUS_M, 2001)
    area = np.trapezoid([2.0 * half_width_at(x) for x in r], r)
    assert blade_sheet(flat).area_m2.sum() == pytest.approx(area, rel=1e-3)


def test_flat_sheet_wall_is_the_vertical_thickness(flat):
    sheet = blade_sheet(flat, n_radial=10, n_tangential=6)
    assert np.all(np.abs(sheet.z_m) < 1e-12)


def test_bow_thins_the_true_wall(flat, bowed):
    assert mean_thickness_m(blade_sheet(bowed)) < mean_thickness_m(blade_sheet(flat))


def test_bow_adds_surface_area(flat, bowed):
    assert blade_sheet(bowed).area_m2.sum() > blade_sheet(flat).area_m2.sum()


def test_bow_keeps_the_volume(flat, bowed):
    # The thickness is laid out along z, so wall × surface area = vertical thickness × planform area.
    def volume(p):
        s = blade_sheet(p)
        return float((s.thickness_m * s.area_m2).sum())

    assert volume(bowed) == pytest.approx(volume(flat), rel=2e-3)


def test_scaling_hits_the_target_mean(bowed):
    assert mean_thickness_m(scaled_to_mean_thickness(blade_sheet(bowed), 2.5e-3)) == pytest.approx(
        2.5e-3
    )


def test_scaling_keeps_the_mid_surface(bowed):
    sheet = blade_sheet(bowed)
    np.testing.assert_array_equal(scaled_to_mean_thickness(sheet, 2.0e-3).z_m, sheet.z_m)


def test_scaling_rejects_a_non_positive_target(bowed):
    with pytest.raises(ValueError, match="positive"):
        scaled_to_mean_thickness(blade_sheet(bowed), 0.0)


def test_wall_thinner_than_the_shell_is_solid():
    assert areal_density_kg_m2(np.array([1.0e-3]), _MODEL)[0] == pytest.approx(1240.0 * 1.0e-3)


def test_wall_thicker_than_the_shell_adds_sparse_core():
    rho = areal_density_kg_m2(np.array([2.6e-3]), _MODEL)[0]
    assert rho == pytest.approx(1240.0 * (1.6e-3 + 0.15 * 1.0e-3))


def test_blade_mass_adds_the_base():
    assert blade_mass_kg(_one_cell(), _SOLID) == pytest.approx(1000.0 * 1.0e-3 * 1.0e-4)


def test_fit_recovers_a_known_model(flat, bowed):
    sheets = [
        scaled_to_mean_thickness(blade_sheet(p), t)
        for p in (flat, bowed)
        for t in (1.4e-3, 2.0e-3, 2.6e-3, 3.4e-3)
    ]
    fit = fit_print_mass_model(sheets, [blade_mass_kg(s, _MODEL) for s in sheets], rho_kg_m3=1240.0)
    assert (fit.wall_m, fit.core_fraction, fit.base_kg) == pytest.approx(
        (_MODEL.wall_m, _MODEL.core_fraction, _MODEL.base_kg), rel=1e-3
    )


def test_fit_needs_three_blades(flat):
    sheet = blade_sheet(flat)
    with pytest.raises(ValueError, match="at least 3"):
        fit_print_mass_model([sheet, sheet], [0.01, 0.01], rho_kg_m3=1240.0)


def test_fit_needs_one_mass_per_blade(flat):
    sheet = blade_sheet(flat)
    with pytest.raises(ValueError, match="masses"):
        fit_print_mass_model([sheet] * 3, [0.01] * 2, rho_kg_m3=1240.0)


def test_inertia_of_one_cell_is_mass_times_lever_squared():
    i = fan_inertia_wrist_kgm2(
        _one_cell(), _SOLID, blade_count=1, pitch_rad=0.0, wrist_offset_m=0.05
    )
    assert i == pytest.approx(1.0e-4 * (0.15**2 + 0.01**2))


def test_inertia_counts_the_base_at_the_pin():
    model = replace(_SOLID, base_kg=0.002)
    i = fan_inertia_wrist_kgm2(
        _one_cell(), model, blade_count=1, pitch_rad=0.0, wrist_offset_m=0.05
    )
    assert i == pytest.approx(1.0e-4 * (0.15**2 + 0.01**2) + 0.002 * 0.05**2)


def test_spread_blades_sit_closer_to_the_wrist():
    i = fan_inertia_wrist_kgm2(
        _one_cell(z=0.0), _SOLID, blade_count=2, pitch_rad=math.pi / 2, wrist_offset_m=0.05
    )
    assert i == pytest.approx(2 * 1.0e-4 * (0.1 * math.cos(math.pi / 4) + 0.05) ** 2)


def test_com_of_one_cell_is_its_lever():
    c = fan_com_wrist_m(_one_cell(), _SOLID, blade_count=1, pitch_rad=0.0, wrist_offset_m=0.05)
    assert c == pytest.approx(0.15)


def test_com_is_pulled_toward_the_pin_by_the_base():
    model = replace(_SOLID, base_kg=1.0e-4)
    c = fan_com_wrist_m(_one_cell(), model, blade_count=1, pitch_rad=0.0, wrist_offset_m=0.05)
    assert c == pytest.approx(0.10)


def test_push_only_work_is_i_theta_squared_omega_squared():
    assert inertial_work_per_cycle_j(0.005, 0.7, 12.0)[0] == pytest.approx(0.005 * 0.49 * 144.0)


def test_push_and_brake_is_twice_push_only():
    push, both = inertial_work_per_cycle_j(0.005, 0.7, 12.0)
    assert both == pytest.approx(2.0 * push)


def test_design_effort_fan_is_twelve_blades(bowed):
    e = design_effort(
        bowed, _MODEL, recipe_thickness_m=2.5e-3, blade_count=12, pitch_rad=0.232,
        wrist_offset_m=0.05, theta_max_rad=0.6981, omega_rad_s=4 * math.pi,
    )  # fmt: skip
    assert e["fan_mass_kg"] == pytest.approx(12 * e["blade_mass_kg"])


def test_design_effort_work_matches_its_inertia(bowed):
    e = design_effort(
        bowed, _MODEL, recipe_thickness_m=2.5e-3, blade_count=12, pitch_rad=0.232,
        wrist_offset_m=0.05, theta_max_rad=0.6981, omega_rad_s=4 * math.pi,
    )  # fmt: skip
    assert e["work_push_j"] == pytest.approx(
        e["inertia_wrist_kgm2"] * 0.6981**2 * (4 * math.pi) ** 2
    )


def test_equal_effort_keeps_the_ranking():
    s = rerank_summary(np.array([3.0, 1.0, 2.0, 5.0]), np.ones(4), top_k=(2,))
    assert (s["spearman"], s["top_k_kept"]["2"]) == (pytest.approx(1.0), 1.0)


def test_effort_can_change_the_best_design():
    s = rerank_summary(np.array([2.0, 1.9, 1.0]), np.array([2.0, 1.0, 1.0]), top_k=(1,))
    assert (s["best_by_score"], s["best_by_ratio"], s["top_k_kept"]["1"]) == (0, 1, 0.0)
