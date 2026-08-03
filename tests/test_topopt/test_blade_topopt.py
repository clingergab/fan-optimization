"""Tests for per-design 3D SIMP topology optimization (fanopt.topopt.blade_topopt).

The build_problem / SIMP-loop logic is exercised on a hand-built tet **slab** (a cantilever
plate positioned beyond the hub radius) so the FEA math runs without paying for gmsh meshing;
the gmsh→CadQuery mesh path is covered end-to-end in test_run_phase2_blade_to. Requires the
[topopt] extra (scikit-fem for the FEA, and gmsh/cadquery which blade_fea_mesh imports).
"""

from __future__ import annotations

import importlib.util
import json
import types

import numpy as np
import pytest

for _dep in ("skfem", "gmsh", "cadquery"):
    if importlib.util.find_spec(_dep) is None:  # pragma: no cover - env-dependent
        pytest.skip(f"{_dep} not installed", allow_module_level=True)

from fanopt.topopt.blade_fea_mesh import BladeFeaMeshResult
from fanopt.topopt.blade_topopt import (
    BladeTOResult,
    build_blade_to_problem,
    run_blade_to_batch,
    run_blade_topology_optimization,
    von_mises,
)


def _slab_mesh(nx=5, ny=4, nz=4, x0=0.05, x1=0.06, y=0.004, t=0.008) -> BladeFeaMeshResult:
    """Structured tet mesh of a slab ``[x0,x1]×[-y,y]×[-t/2,t/2]`` (6 tets / hex cell).

    Sits at radius > HUB_RADIUS so its ±z faces classify as aero skin; the ``x=x0`` face is
    the clamped support (a cantilever plate), so the productive pressure bends it in z.
    Kept compact (~2 mm node spacing) so the surface-node-distance skin classifier resolves a
    frozen outer layer + a carvable core — the real blade mesh is fine enough (1.5 mm) for this.
    """
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(-y, y, ny + 1)
    zs = np.linspace(-t / 2, t / 2, nz + 1)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)

    def nid(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    # 6-tet (Freudenthal) split of every hex cell.
    hexes = [(0, 1, 2, 3, 4, 5, 6, 7)]
    splits = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = [
                    nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                    nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1),
                ]
                for s in splits:
                    tets.append([c[hexes[0][v]] for v in s])
    tets = np.asarray(tets, dtype=int)
    support = np.nonzero(np.isclose(grid[:, 0], x0))[0]
    return BladeFeaMeshResult(
        nodes=grid, tets=tets,
        aero_top_facets=np.zeros((0, 3), int), aero_bottom_facets=np.zeros((0, 3), int),
        click_facets=np.zeros((0, 3), int), support_node_ids=support,
    )


# --- von Mises --------------------------------------------------------------------------

def test_von_mises_hydrostatic_is_zero():
    s = np.array([[5.0, 5.0, 5.0, 0.0, 0.0, 0.0]])  # pure pressure -> no deviatoric stress
    assert von_mises(s)[0] == pytest.approx(0.0)


def test_von_mises_uniaxial_equals_axial_stress():
    s = np.array([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert von_mises(s)[0] == pytest.approx(10.0)


def test_von_mises_pure_shear():
    s = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 4.0]])  # σ_xy=4 -> vM = sqrt(3)*4
    assert von_mises(s)[0] == pytest.approx(np.sqrt(3) * 4.0)


# --- problem assembly -------------------------------------------------------------------

def test_build_problem_partitions_frozen_and_design():
    prob = build_blade_to_problem(_slab_mesh(), skin_thickness_m=0.0025, volfrac=0.4)
    d = prob.domain
    n = len(d.element_volumes)
    assert d.frozen_mask.sum() > 0  # a skin froze
    assert d.design_mask.sum() > 0  # an interior remains carvable
    assert int(d.frozen_mask.sum() + d.design_mask.sum()) == n  # exact partition
    assert not np.any(d.frozen_mask & d.design_mask)


def test_build_problem_freezes_the_surface_not_the_core():
    prob = build_blade_to_problem(_slab_mesh(t=0.006), skin_thickness_m=0.0025)
    d = prob.domain
    z = np.abs(d.element_centroids[:, 2])
    # frozen elements sit nearer the ±z faces (larger |z|) than carvable ones on average.
    assert z[d.frozen_mask].mean() > z[d.design_mask].mean()


def test_build_problem_rho_min_matches_emin():
    prob = build_blade_to_problem(_slab_mesh(), penal=3.0)
    # ρ_min^penal == E_min factor keeps the void stiffness equal to the modified-SIMP floor.
    assert prob.rho_min**3.0 == pytest.approx(1e-9, rel=1e-6)


# --- the SIMP loop ----------------------------------------------------------------------

def _run_slab(**kw):
    prob = build_blade_to_problem(_slab_mesh(), skin_thickness_m=0.0025, volfrac=0.4)
    return prob, run_blade_topology_optimization(prob, **kw)


def test_to_keeps_frozen_skin_solid():
    prob, res = _run_slab(max_iters=6)
    assert res.density[prob.domain.frozen_mask].min() == pytest.approx(1.0)


def test_to_carves_the_design_region():
    prob, res = _run_slab(max_iters=6)
    assert res.density[prob.domain.design_mask].min() < 0.5  # material was removed
    assert 0.0 < res.volume_removed_frac < 1.0


def test_to_reduces_compliance():
    _prob, res = _run_slab(max_iters=8)
    assert res.compliance_history[-1] < res.compliance_history[0]


def test_to_reports_finite_structural_metrics():
    _prob, res = _run_slab(max_iters=5)
    assert np.isfinite(res.u_tip_max_m) and res.u_tip_max_m > 0.0
    assert np.isfinite(res.max_von_mises_pa) and res.max_von_mises_pa >= 0.0
    assert res.mass_kg > 0.0


def test_to_hits_design_volume_target_when_converged():
    _prob, res = _run_slab(max_iters=60, tol=1e-4)
    assert res.design_volume_fraction == pytest.approx(0.4, abs=0.05)


def test_to_flags_convergence_and_stops_early():
    # A loose tol (> the 0.2 move limit) converges on the first step -> the break fires.
    _prob, res = _run_slab(max_iters=40, tol=0.5)
    assert res.converged
    assert res.iterations < 40


# --- batch driver -----------------------------------------------------------------------

def _fake_result(removed=0.3):
    return BladeTOResult(
        density=np.array([1.0, 0.2]), compliance_history=(2.0, 1.0),
        design_volume_fraction=0.4, volume_removed_frac=removed, mass_kg=0.02,
        u_tip_max_m=5e-4, max_von_mises_pa=1e6, converged=True, iterations=3,
        meta={"n_design": 1.0, "n_frozen": 1.0},
    )


def test_batch_writes_density_and_summary(tmp_path):
    calls = []

    def fake_opt(params, **kw):
        calls.append(kw)
        return _fake_result()

    stub = types.SimpleNamespace(uniform=False)
    designs = [("00_aaa", stub), ("01_bbb", stub)]
    summary = run_blade_to_batch(designs, tmp_path, optimize=fake_opt, volfrac=0.4)
    assert summary["n_succeeded"] == 2
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "00_aaa_density.npy").exists()
    assert (tmp_path / "01_bbb_density.npy").exists()
    assert calls[0]["volfrac"] == 0.4  # kwargs threaded through


def test_batch_resumes_from_sidecars(tmp_path):
    stub = types.SimpleNamespace(uniform=False)
    calls = []

    def counting_opt(params, **kw):
        calls.append(1)
        return _fake_result()

    designs = [("00_aaa", stub), ("01_bbb", stub)]
    run_blade_to_batch(designs, tmp_path, optimize=counting_opt)
    assert len(calls) == 2  # first pass runs both
    summary = run_blade_to_batch(designs, tmp_path, optimize=counting_opt)
    assert len(calls) == 2  # second pass reuses sidecars — optimize NOT called again
    assert summary["n_succeeded"] == 2  # completed designs still reported


def test_batch_writes_summary_after_each_design(tmp_path):
    # A summary.json must exist even if the process is killed after the first design.
    seen = []

    def opt(params, **kw):
        # summary.json should already reflect the prior design(s) by the time this runs.
        p = tmp_path / "summary.json"
        seen.append(json.loads(p.read_text())["n_succeeded"] if p.exists() else 0)
        return _fake_result()

    stub = types.SimpleNamespace(uniform=False)
    run_blade_to_batch([("00_a", stub), ("01_b", stub)], tmp_path, optimize=opt)
    assert seen == [0, 1]  # summary grew incrementally, not written only at the end


def test_batch_empty_designs_writes_zero_summary(tmp_path):
    summary = run_blade_to_batch([], tmp_path, optimize=lambda *a, **k: _fake_result())
    assert summary["n_designs"] == 0 and summary["n_succeeded"] == 0
    assert (tmp_path / "summary.json").exists()


def test_batch_isolates_a_failing_design(tmp_path):
    def flaky(params, **kw):
        if params.tag == "boom":
            raise RuntimeError("mesh blew up")
        return _fake_result()

    ok = types.SimpleNamespace(uniform=False, tag="ok")
    bad = types.SimpleNamespace(uniform=False, tag="boom")
    designs = [("00_ok", ok), ("01_bad", bad), ("02_ok", ok)]
    summary = run_blade_to_batch(designs, tmp_path, optimize=flaky)
    assert summary["n_succeeded"] == 2
    bad = next(r for r in summary["designs"] if r["name"] == "01_bad")
    assert "error" in bad and "mesh blew up" in bad["error"]
    assert not (tmp_path / "01_bad_density.npy").exists()  # no artifact for the failure
