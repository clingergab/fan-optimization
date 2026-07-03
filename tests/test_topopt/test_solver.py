"""Tests for the rib SIMP TO loop (report-final.md §3.1 / §9.2)."""

from __future__ import annotations

import numpy as np
import pytest

from fanopt.topopt.loads import build_rib_problem
from fanopt.topopt.solver import run_rib_topology_optimization


def _small_result(max_iters=20):
    # 1.5 mm elements: fast, and the preserved zone (~30%) stays below the 40%
    # volume target so the problem is feasible.
    p = build_rib_problem(elem_size_m=0.0015)
    return p, run_rib_topology_optimization(p, max_iters=max_iters)


def test_compliance_decreases():
    _p, r = _small_result()
    assert r.compliance_history[-1] < r.compliance_history[0]


def test_compliance_settles_downward_after_transient():
    # SIMP starts from uniform intermediate density (artificially stiff under the
    # penalty), so compliance bumps up ~1 iter before penalization drives it down.
    # After that transient it is monotone non-increasing (within OC wobble).
    _p, r = _small_result(max_iters=25)
    h = r.compliance_history
    peak = h.index(max(h))
    tail = h[peak:]
    assert all(tail[i + 1] <= tail[i] * 1.02 for i in range(len(tail) - 1))


def test_volume_target_met():
    p, r = _small_result(max_iters=40)
    assert r.volume_fraction == pytest.approx(p.volfrac, abs=0.03)


def test_density_within_bounds():
    _p, r = _small_result()
    assert r.density.min() >= 0.0
    assert r.density.max() <= 1.0 + 1e-9


def test_preserved_zone_stays_solid():
    p, r = _small_result(max_iters=40)
    assert r.density[p.preserved].min() > 0.9


def test_u_tip_is_finite_positive():
    _p, r = _small_result()
    assert np.isfinite(r.u_tip_max_m)
    assert r.u_tip_max_m > 0.0


def test_result_reports_iterations_and_history():
    _p, r = _small_result(max_iters=15)
    assert 1 <= r.iterations <= 15
    assert len(r.compliance_history) == r.iterations


def test_converges_on_coarse_problem():
    p = build_rib_problem(elem_size_m=0.0015)
    r = run_rib_topology_optimization(p, max_iters=120, tol=0.02)
    assert r.converged
