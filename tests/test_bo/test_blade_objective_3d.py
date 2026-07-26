"""Tests for the 3D whole-fan objective (audit B3 + the metric knob)."""

from __future__ import annotations

import numpy as np
import pytest

from fanopt.bo import blade_objective_3d as obj3d
from fanopt.bo.blade_codec import bounds, clip_to_bounds
from fanopt.bo.blade_objective_3d import Blade3DObjective, whole_fan_j_fan
from fanopt.cfd.blade_aero_3d import Blade3DAeroResult


def _feasible_vector() -> np.ndarray:
    low, high = bounds()
    return clip_to_bounds((low + high) / 2.0)  # decode-feasible by construction


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


def test_blade_count_actually_moves_the_objective(tmp_path, monkeypatch):
    # B3's whole point: two designs identical but for blade_count give different whole-fan wind.
    monkeypatch.setattr(obj3d, "evaluate_blade_aero_3d", _fake_aero(3.0, 10.0))
    low, high = bounds()
    from fanopt.bo.blade_codec import SEARCH_SPACE, _IDX  # noqa: PLC0415 (test-local introspection)

    vec = clip_to_bounds((low + high) / 2.0)
    counts = SEARCH_SPACE[_IDX["blade_count"]].choices
    vals = []
    for c in (counts[0], counts[-1]):
        v = vec.copy()
        v[_IDX["blade_count"]] = counts.index(c) + 0.5  # select that categorical
        vals.append(Blade3DObjective(out_dir=tmp_path, su2_bin="/fake")(v)[0])
    assert vals[0] != vals[1]  # blade_count is no longer aero-inert


def test_infeasible_returns_nan(tmp_path, monkeypatch):
    monkeypatch.setattr(obj3d, "feasible", lambda p: False)
    j_fan, mass, defl = Blade3DObjective(out_dir=tmp_path)(_feasible_vector())
    assert np.isnan(j_fan) and np.isnan(mass) and np.isnan(defl)


def test_cfd_failure_returns_nan(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("SU2 diverged")
    monkeypatch.setattr(obj3d, "evaluate_blade_aero_3d", boom)
    j_fan, _, _ = Blade3DObjective(out_dir=tmp_path, su2_bin="/fake")(_feasible_vector())
    assert np.isnan(j_fan)
