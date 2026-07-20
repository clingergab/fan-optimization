"""Tests for fanopt.bo.blade_objective (vector → (J_fan, mass, deflection))."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fanopt.bo import blade_objective as bobj
from fanopt.bo.blade_codec import encode
from fanopt.cfd.blade_aero import BladeAeroResult
from fanopt.geometry.blade import BladeParams

_FEASIBLE_GRID = (
    (0.0003, 0.0005, 0.0003),
    (0.0004, 0.0006, 0.0004),
    (0.0005, 0.0007, 0.0005),
    (0.0006, 0.0008, 0.0006),
)


def _feasible() -> BladeParams:
    return BladeParams(
        blade_count=8,
        rib_bow_mid_m=0.010,
        rib_bow_tip_m=0.020,
        t_rib_hub_m=0.0025,
        t_rib_tip_m=0.0035,
        panel_offsets_m=_FEASIBLE_GRID,
        panel_thickness_nom_m=0.0013,
    )


def _infeasible() -> BladeParams:
    # 12 blades with a thick rib → folded stack far exceeds the cap.
    return BladeParams(**{**_feasible().to_dict(), "blade_count": 12, "t_rib_tip_m": 0.006})


# --- analytic deflection -----------------------------------------------------


def test_deflection_positive():
    assert bobj.blade_panel_deflection_m(_feasible()) > 0.0


def test_deflection_grows_as_panel_thins():
    thick = BladeParams(**{**_feasible().to_dict(), "panel_thickness_nom_m": 0.003})
    thin = BladeParams(**{**_feasible().to_dict(), "panel_thickness_nom_m": 0.0012})
    assert bobj.blade_panel_deflection_m(thin) > bobj.blade_panel_deflection_m(thick)


# --- objective ---------------------------------------------------------------


def test_infeasible_penalized_without_su2(tmp_path):
    # No su2_bin, no mock: an infeasible design must NOT reach the solver.
    obj = bobj.BladeObjective(out_dir=tmp_path)
    out = obj(encode(_infeasible()))
    assert all(math.isnan(v) for v in out)


def test_infeasible_writes_marker(tmp_path):
    bobj.BladeObjective(out_dir=tmp_path)(encode(_infeasible()))
    assert list(tmp_path.glob("designs/*/INFEASIBLE.txt"))


def test_feasible_happy_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bobj, "evaluate_blade_aero",
        lambda *a, **k: BladeAeroResult(j_fan=4.2e10, steady_cd=0.25, n_nodes=6000),
    )
    obj = bobj.BladeObjective(out_dir=tmp_path, su2_bin="/fake/SU2_CFD")
    j_fan, mass, defl = obj(encode(_feasible()))
    assert j_fan == pytest.approx(4.2e10)
    assert mass > 0.0 and defl > 0.0


def test_solver_failure_penalized(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("SU2 diverged")

    monkeypatch.setattr(bobj, "evaluate_blade_aero", boom)
    out = bobj.BladeObjective(out_dir=tmp_path, su2_bin="/fake/SU2_CFD")(encode(_feasible()))
    assert all(math.isnan(v) for v in out)
    assert list(tmp_path.glob("designs/*/FAILED.txt"))


def test_returns_three_objectives(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bobj, "evaluate_blade_aero",
        lambda *a, **k: BladeAeroResult(j_fan=1.0, steady_cd=0.1, n_nodes=100),
    )
    out = bobj.BladeObjective(out_dir=tmp_path, su2_bin="x")(encode(_feasible()))
    assert len(out) == 3 and len(bobj.OBJECTIVE_NAMES) == 3
