"""Tests for fanopt.bo.blade_objective (vector → (J_fan, mass, deflection))."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fanopt.bo import blade_objective as bobj
from fanopt.bo.blade_codec import bounds, clip_to_bounds, decode, encode
from fanopt.cfd.blade_aero import BladeAeroResult
from fanopt.geometry.blade import BladeParams, feasible

_FEASIBLE_GRID = (
    (0.0003, 0.0005, 0.0003),
    (0.0004, 0.0006, 0.0004),
    (0.0005, 0.0007, 0.0005),
    (0.0006, 0.0008, 0.0006),
)


def _feasible() -> BladeParams:
    return BladeParams(
        blade_count=8,
        rib_bow_knots_m=(0.005, 0.010, 0.013, 0.017, 0.020),
        rib_bow_interp="linear",
        t_rib_hub_m=0.0025,
        t_rib_tip_m=0.0035,
        panel_offsets_m=_FEASIBLE_GRID,
        panel_thickness_m=((0.0013, 0.0013, 0.0013), (0.0013, 0.0013, 0.0013), (0.0013, 0.0013, 0.0013), (0.0013, 0.0013, 0.0013)),
    )


def _infeasible_vector() -> np.ndarray:
    """A BO vector that *decodes* to an infeasible (mass-over-cap) design.

    The codec is feasible-by-construction for fold + containment and caps rib thickness by
    mass, but the mass proxy is approximate, so a fraction of decodes still tip over the cap.
    Search deterministically for one so the objective's infeasible short-circuit is exercised.
    """
    low, high = bounds()
    rng = np.random.default_rng(0)
    for _ in range(2000):
        v = clip_to_bounds(low + rng.random(len(low)) * (high - low))
        if not feasible(decode(v)):
            return v
    raise AssertionError("expected some vector to decode to an infeasible design")


# --- analytic deflection -----------------------------------------------------


def test_deflection_positive():
    assert bobj.blade_panel_deflection_m(_feasible()) > 0.0


def test_deflection_grows_as_panel_thins():
    grid_thick = tuple((0.003, 0.003, 0.003) for _ in range(4))
    grid_thin = tuple((0.0012, 0.0012, 0.0012) for _ in range(4))
    thick = BladeParams(**{**_feasible().to_dict(), "panel_thickness_m": grid_thick})
    thin = BladeParams(**{**_feasible().to_dict(), "panel_thickness_m": grid_thin})
    assert bobj.blade_panel_deflection_m(thin) > bobj.blade_panel_deflection_m(thick)


# --- objective ---------------------------------------------------------------


def test_infeasible_penalized_without_su2(tmp_path):
    # No su2_bin, no mock: an infeasible design must NOT reach the solver.
    obj = bobj.BladeObjective(out_dir=tmp_path)
    out = obj(_infeasible_vector())
    assert all(math.isnan(v) for v in out)


def test_infeasible_writes_marker(tmp_path):
    bobj.BladeObjective(out_dir=tmp_path)(_infeasible_vector())
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


def test_success_persists_cfd_output_to_diag_dir(tmp_path, monkeypatch):
    scratch, drive = tmp_path / "scratch", tmp_path / "drive"

    def fake_eval(params, workdir, **kw):
        from pathlib import Path

        Path(workdir).mkdir(parents=True, exist_ok=True)
        (Path(workdir) / "history.csv").write_text("iter,CD\n1,0.2\n", encoding="utf-8")
        return BladeAeroResult(j_fan=1.0, steady_cd=0.2, n_nodes=10)

    monkeypatch.setattr(bobj, "evaluate_blade_aero", fake_eval)
    bobj.BladeObjective(out_dir=scratch, diag_dir=drive, su2_bin="x")(encode(_feasible()))
    # the CFD history.csv was copied from the (ephemeral) scratch to the persistent dir
    assert list(drive.glob("designs/*/history.csv"))
