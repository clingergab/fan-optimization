"""Sign-discriminator gate for the steady two-eval proxy (report-final.md §9.4.1).

Unit-level stand-in for the named CI gate: the full version generates a 45°
productive louver, runs both steady CFD evals, and asserts the delta is
positive. Here we exercise the ``j_fan`` delta logic directly with force values
representing a productive-scooping louver (catches air on the +z sweep, feathers
on the return) so the sign convention is locked without a CFD run.
"""

from __future__ import annotations

import pytest

from fanopt.cfd.j_fan import SteadyRun, compute_j_fan_steady


def test_productive_louver_scores_positive():
    # Louver scoops on the productive stroke (high drag) and feathers on return.
    runs = [SteadyRun(2.0, "productive"), SteadyRun(0.4, "return")]
    res = compute_j_fan_steady(runs)
    assert res.proxy_kind == "delta"
    assert res.j_fan_steady_proxy > 0.0


def test_symmetric_design_scores_near_zero():
    runs = [SteadyRun(1.5, "productive"), SteadyRun(1.5, "return")]
    res = compute_j_fan_steady(runs)
    assert res.j_fan_steady_proxy == pytest.approx(0.0)


def test_anti_productive_louver_scores_negative():
    # A louver oriented the wrong way feathers on productive, scoops on return.
    runs = [SteadyRun(0.4, "productive"), SteadyRun(2.0, "return")]
    res = compute_j_fan_steady(runs)
    assert res.j_fan_steady_proxy < 0.0
