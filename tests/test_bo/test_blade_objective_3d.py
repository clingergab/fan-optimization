"""Tests for the 3D whole-fan objective (audit B3 + the metric knob)."""

from __future__ import annotations

import numpy as np
import pytest

from fanopt.bo import blade_objective_3d as obj3d
from fanopt.bo.blade_codec import bounds, clip_to_bounds
from fanopt.bo.blade_objective_3d import Blade3DObjective, blade_panel_deflection_m, whole_fan_j_fan
from fanopt.cfd.blade_aero_3d import Blade3DAeroResult
from fanopt.geometry.blade import BLADE_COUNT, BladeParams, half_width_at, panel_radial_stations
from fanopt.geometry.schema import E_PETG_XY_PA


def _feasible_vector() -> np.ndarray:
    low, high = bounds()
    return clip_to_bounds((low + high) / 2.0)  # decode-feasible by construction


def _defl_params(thick_grid: tuple) -> BladeParams:
    """A 12-blade trapezoid with a flat panel and the given thickness grid (≥ 3 mm floor)."""
    return BladeParams(
        blade_count=BLADE_COUNT,
        rib_bow_knots_m=(0.005, 0.010, 0.013, 0.017, 0.020),
        rib_bow_interp="linear",
        t_rib_hub_m=0.006,
        t_rib_tip_m=0.008,
        panel_offsets_m=tuple((0.0, 0.0, 0.0) for _ in range(4)),
        panel_thickness_m=thick_grid,
    )


def _fake_aero(mean: float, peak: float):
    def _f(params, workdir, **kw):
        return Blade3DAeroResult(j_fan_mean=mean, j_fan_peak=peak, n_nodes=1000.0)
    return _f


def test_whole_fan_scales_by_blade_count():
    assert whole_fan_j_fan(2.5, 8) == pytest.approx(20.0)


def test_bad_metric_raises(tmp_path):
    with pytest.raises(ValueError):
        Blade3DObjective(out_dir=tmp_path, metric="bogus")


def test_default_metric_is_cycle_mean_per_n1(tmp_path):
    # N1 (2026-07-26) selected cycle-mean CFz: it discriminates net wind (scoop +, flat ~0),
    # whereas peak is ~equal across designs. Lock the decision.
    assert Blade3DObjective(out_dir=tmp_path).metric == "mean"


def test_objective_peak_metric_scales_per_blade_by_count(tmp_path, monkeypatch):
    monkeypatch.setattr(obj3d, "evaluate_blade_aero_3d", _fake_aero(3.0, 10.0))
    vec = _feasible_vector()
    p = obj3d.decode(vec)
    j_fan, mass, defl = Blade3DObjective(out_dir=tmp_path, su2_bin="/fake", metric="peak")(vec)
    assert j_fan == pytest.approx(10.0 * p.blade_count)  # peak × blade_count
    assert mass > 0.0 and defl > 0.0


def test_objective_mean_metric_uses_cycle_mean(tmp_path, monkeypatch):
    monkeypatch.setattr(obj3d, "evaluate_blade_aero_3d", _fake_aero(3.0, 10.0))
    vec = _feasible_vector()
    p = obj3d.decode(vec)
    j_fan, _, _ = Blade3DObjective(out_dir=tmp_path, su2_bin="/fake", metric="mean")(vec)
    assert j_fan == pytest.approx(3.0 * p.blade_count)  # mean × blade_count


def test_whole_fan_scales_by_fixed_blade_count_12(tmp_path, monkeypatch):
    # ADR-0005: blade_count is FIXED at 12 (no longer a codec dimension), so whole-fan wind is
    # always per-blade × 12 — every decoded design is a 12-blade fan.
    monkeypatch.setattr(obj3d, "evaluate_blade_aero_3d", _fake_aero(3.0, 10.0))
    vec = _feasible_vector()
    assert obj3d.decode(vec).blade_count == 12
    j_fan, _, _ = Blade3DObjective(out_dir=tmp_path, su2_bin="/fake", metric="mean")(vec)
    assert j_fan == pytest.approx(3.0 * 12)


# --- panel deflection proxy (trapezoid; ADR-0004 live path) ------------------


def test_deflection_positive_on_trapezoid():
    p = _defl_params(tuple((0.003, 0.003, 0.003) for _ in range(4)))
    assert blade_panel_deflection_m(p) > 0.0


def test_deflection_grows_as_panel_thins():
    thick = _defl_params(tuple((0.006, 0.006, 0.006) for _ in range(4)))
    thin = _defl_params(tuple((0.003, 0.003, 0.003) for _ in range(4)))
    assert blade_panel_deflection_m(thin) > blade_panel_deflection_m(thick)


def test_deflection_uses_trapezoid_width_not_pie_slice_span():
    # Load-bearing: the deflection must use the trapezoid full width 2·half_width_at(r) (NOT the
    # retired r·Δθ sector span / L_RIB_M). For a uniform thickness t, the area-weighted mean
    # reduces to (α·p/(E·t³))·Σ w⁵ / Σ w with w(r) = 2·half_width_at(r). Match it exactly.
    t = 0.004
    p = _defl_params(tuple((t, t, t) for _ in range(4)))
    widths = [2.0 * half_width_at(r) for r in panel_radial_stations()]
    expected = (
        obj3d._CLAMPED_PLATE_ALPHA * obj3d.AERO_PRESSURE_PA / (E_PETG_XY_PA * t**3)
    ) * (sum(w**5 for w in widths) / sum(widths))
    assert blade_panel_deflection_m(p) == pytest.approx(expected)


def test_infeasible_returns_nan(tmp_path, monkeypatch):
    monkeypatch.setattr(obj3d, "feasible", lambda p: False)
    j_fan, mass, defl = Blade3DObjective(out_dir=tmp_path)(_feasible_vector())
    assert np.isnan(j_fan) and np.isnan(mass) and np.isnan(defl)


def test_unfoldable_returns_nan_before_cfd(tmp_path, monkeypatch):
    # The CAD fold gate is a cheap backstop BEFORE the minutes-long CFD: an un-foldable design is
    # excluded (NaN) and the aero eval is never reached, so nothing un-foldable is ever evaluated.
    monkeypatch.setattr(obj3d, "fold_collision_clear", lambda p: False)

    def boom(*a, **k):
        raise AssertionError("CFD must not run for an un-foldable design")

    monkeypatch.setattr(obj3d, "evaluate_blade_aero_3d", boom)
    j_fan, mass, defl = Blade3DObjective(out_dir=tmp_path, su2_bin="/fake")(_feasible_vector())
    assert np.isnan(j_fan) and np.isnan(mass) and np.isnan(defl)
    assert list(tmp_path.rglob("UNFOLDABLE.txt"))  # marker written for diagnosis


def test_cfd_failure_returns_nan(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("SU2 diverged")
    monkeypatch.setattr(obj3d, "evaluate_blade_aero_3d", boom)
    j_fan, _, _ = Blade3DObjective(out_dir=tmp_path, su2_bin="/fake")(_feasible_vector())
    assert np.isnan(j_fan)
