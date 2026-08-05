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

from fanopt.bo.blade_codec import N_DIMS, decode
from fanopt.topopt.blade_fea_mesh import BladeFeaMeshResult, FeaMeshParams, build_blade_fea_mesh
from fanopt.topopt.blade_topopt import (
    DEFAULT_RMIN_M,
    DEFAULT_SKIN_THICKNESS_M,
    BladeTOResult,
    _align_density,
    _fsync_path,
    _load_cases,
    _passes_screen,
    _screen_ladder,
    _screen_result,
    _write_json_durable,
    build_blade_to_problem,
    rescreen_blade_from_density,
    rescreen_blade_to_batch,
    run_blade_to_batch,
    run_blade_topology_optimization,
    topology_optimize_blade_screened,
    von_mises,
)

_LOAD_CASE_NAMES = ("productive_stroke", "return_stroke", "inertial", "click_engagement")


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


# --- screened search (min-mass-that-passes-the-screen) ----------------------------------

def _screen_res(u_tip_mm, vm_mpa, removed=0.5):
    return BladeTOResult(
        density=np.array([1.0, 0.2]), compliance_history=(2.0, 1.0), design_volume_fraction=0.4,
        volume_removed_frac=removed, mass_kg=0.02, u_tip_max_m=u_tip_mm * 1e-3,
        max_von_mises_pa=vm_mpa * 1e6, converged=True, iterations=3, meta={},
    )


def test_passes_screen_requires_both_limits():
    assert _passes_screen(_screen_res(0.5, 5.0), 1e-3, 15e6)
    assert not _passes_screen(_screen_res(2.0, 5.0), 1e-3, 15e6)  # deflection over limit
    assert not _passes_screen(_screen_res(0.5, 20.0), 1e-3, 15e6)  # stress over allowable


def test_screen_ladder_accepts_most_aggressive_that_passes():
    calls = []

    def run(vf):
        calls.append(vf)
        return _screen_res(0.5, 5.0)  # passes immediately at the lowest volfrac

    res, vf, trace, passed = _screen_ladder(run, (0.25, 0.35, 0.45), 1e-3, 15e6)
    assert passed and vf == 0.25 and calls == [0.25] and len(trace) == 1  # stopped at first pass


def test_screen_ladder_climbs_until_it_passes():
    def run(vf):
        return _screen_res(2.0 if vf < 0.30 else 0.5, 5.0)  # 0.25 too soft, 0.35 ok

    res, vf, trace, passed = _screen_ladder(run, (0.25, 0.35, 0.45), 1e-3, 15e6)
    assert passed and vf == 0.35 and len(trace) == 2


def test_screen_ladder_all_fail_returns_safest_flagged():
    def run(vf):
        return _screen_res(5.0, 5.0)  # never meets the deflection limit

    res, vf, trace, passed = _screen_ladder(run, (0.25, 0.35, 0.45), 1e-3, 15e6)
    assert not passed and vf == 0.45 and len(trace) == 3  # safest attempt, flagged


def test_screened_real_design_records_ladder_meta():
    # Relaxed limits so the first (lowest) rung passes -> one OC run, fast; checks the real path
    # meshes once and stamps the screen metadata onto the result.
    params = decode(np.full(N_DIMS, 0.5))
    res = topology_optimize_blade_screened(
        params, mesh_params=FeaMeshParams(mesh_size_m=0.006), max_iters=1,
        volfrac_ladder=(0.4, 0.6), u_tip_limit_m=1.0, sigma_allow_pa=1e12,
    )
    assert res.meta["accepted_volfrac"] == 0.4
    assert res.meta["screen_passed"] == 1.0
    assert len(res.meta["ladder_trace"]) == 1  # stopped at the first passing rung


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


def test_batch_parallel_runs_designs_concurrently(tmp_path):
    # Two real designs via a process pool (default optimize is picklable). Coarse mesh + 1 iter.
    # on_result fires in the MAIN process (not pickled to workers), so a closure is fine.
    seen = []
    designs = [("00_a", decode(np.full(N_DIMS, 0.5))), ("01_b", decode(np.full(N_DIMS, 0.45)))]
    summary = run_blade_to_batch(
        designs, tmp_path, n_workers=2, on_result=lambda name, res: seen.append(name),
        mesh_params=FeaMeshParams(mesh_size_m=0.006), max_iters=1, tol=1e-6,
    )
    assert summary["n_succeeded"] == 2
    assert summary["n_workers"] == 2
    assert sorted(seen) == ["00_a", "01_b"]  # callback fired for each completed design
    assert (tmp_path / "00_a_density.npy").exists()
    assert (tmp_path / "01_b_density.npy").exists()


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


# --- inertial mass: full-solid in the OC loop, carved (honest) in the final screen ------

def test_carved_density_lightens_only_the_inertial_load():
    prob = build_blade_to_problem(_slab_mesh(), skin_thickness_m=0.0025, volfrac=0.4)
    rho = np.where(prob.domain.frozen_mask, 1.0, 0.5)  # interior half-carved, skin solid
    full = _load_cases(prob, None)  # OC-loop loads (full-solid inertial)
    carved = _load_cases(prob, element_density=rho)  # final-screen loads (as-printed mass)
    # inertial is case index 2 (productive, return, inertial, click); a lighter blade -> less load.
    assert np.abs(carved[2].node_forces).sum() < np.abs(full[2].node_forces).sum()


def test_carved_density_leaves_aero_load_unchanged():
    prob = build_blade_to_problem(_slab_mesh(), skin_thickness_m=0.0025, volfrac=0.4)
    rho = np.where(prob.domain.frozen_mask, 1.0, 0.5)
    full = _load_cases(prob, None)
    carved = _load_cases(prob, element_density=rho)
    assert np.allclose(full[0].node_forces, carved[0].node_forces)  # productive aero is ρ-independent


# --- per-load-case breakdown (persisted, not just the max-across-loads) ------------------

def test_to_records_all_four_loads_in_breakdown():
    _prob, res = _run_slab(max_iters=3)
    assert set(res.u_tip_by_load_m) == set(_LOAD_CASE_NAMES)
    assert set(res.vm_by_load_pa) == set(_LOAD_CASE_NAMES)


def test_to_max_equals_worst_of_the_per_load_breakdown():
    _prob, res = _run_slab(max_iters=3)
    assert res.u_tip_max_m == pytest.approx(max(res.u_tip_by_load_m.values()))
    assert res.max_von_mises_pa == pytest.approx(max(res.vm_by_load_pa.values()))


# --- re-screen from saved density (no re-optimization) ----------------------------------

def test_align_density_identity_on_same_centroids():
    c = np.random.default_rng(0).random((10, 3))
    d = np.random.default_rng(1).random(10)
    rho, dist = _align_density(c, c, d)
    assert np.array_equal(rho, d) and dist == 0.0


def test_align_density_remaps_reordered_centroids():
    rng = np.random.default_rng(0)
    c = rng.random((10, 3))
    d = rng.random(10)
    perm = rng.permutation(10)
    rho, dist = _align_density(c, c[perm], d[perm])  # saved arrays permuted; rebuilt is original order
    assert np.allclose(rho, d) and dist == pytest.approx(0.0)


def test_align_density_none_requires_matching_count():
    with pytest.raises(ValueError, match="rebuilt mesh"):
        _align_density(np.zeros((5, 3)), None, np.zeros(4))


def test_align_density_none_passes_through_on_match():
    d = np.arange(5.0)
    rho, dist = _align_density(np.zeros((5, 3)), None, d)
    assert np.array_equal(rho, d) and dist == 0.0


def test_align_density_raises_on_divergent_mesh():
    # A rebuilt element far from every saved centroid = mesh divergence -> must RAISE, not silently smear.
    saved = np.zeros((5, 3))
    rebuilt = np.vstack([np.zeros((4, 3)), [[10.0, 0.0, 0.0]]])  # last point 10 m from any saved
    with pytest.raises(ValueError, match="diverged"):
        _align_density(rebuilt, saved, np.arange(5.0), max_dist_m=0.5)


def test_align_density_within_tolerance_passes():
    saved = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    rebuilt = np.array([[0.1, 0, 0], [1.1, 0, 0]])  # each 0.1 from a saved point; tol 0.5
    rho, dist = _align_density(rebuilt, saved, np.array([2.0, 3.0]), max_dist_m=0.5)
    assert np.allclose(rho, [2.0, 3.0]) and dist == pytest.approx(0.1)


def test_align_density_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        _align_density(np.zeros((3, 3)), np.zeros((5, 3)), np.zeros(4))  # density 4 != saved centroids 5


def test_screen_result_reports_all_screen_fields():
    prob = build_blade_to_problem(_slab_mesh(), skin_thickness_m=0.0025, volfrac=0.4)
    rho = np.where(prob.domain.frozen_mask, 1.0, 0.5)
    res = _screen_result(prob, rho, converged=True, iterations=7)
    assert set(res.u_tip_by_load_m) == set(_LOAD_CASE_NAMES)
    assert res.u_tip_max_m == pytest.approx(max(res.u_tip_by_load_m.values()))
    assert 0.0 < res.max_von_mises_solid_pa <= res.max_von_mises_pa
    assert res.converged and res.iterations == 7 and res.compliance_history == ()


def test_rescreen_batch_reproduces_original_screen(tmp_path):
    # Re-screening the saved density must reproduce the original run's screen EXACTLY (same density,
    # same deterministic mesh) — DEFLECTION *and both stress numbers*, since solid-only stress is the
    # binding verdict. max_iters=6 polarizes the field so solid (rho>=0.5) is more than just the skin.
    designs = [("00_a", decode(np.full(N_DIMS, 0.5)))]
    mp = FeaMeshParams(mesh_size_m=0.006)
    orig = run_blade_to_batch(designs, tmp_path, mesh_params=mp, max_iters=6, tol=1e-6)
    summary = rescreen_blade_to_batch(
        designs, tmp_path, mesh_params=mp, n_workers=1, j_fan_by_name={"00_a": 9.6e12}
    )
    assert summary["n_succeeded"] == 1 and summary["rescreened"] is True
    assert (tmp_path / "rescreen_summary.json").exists()
    o = orig["designs"][0]
    r = json.loads((tmp_path / "00_a_rescreen.json").read_text())
    assert r["u_tip_max_mm"] == pytest.approx(o["u_tip_max_mm"], rel=1e-6)
    assert r["max_von_mises_mpa"] == pytest.approx(o["max_von_mises_mpa"], rel=1e-6)
    assert r["max_von_mises_solid_mpa"] == pytest.approx(o["max_von_mises_solid_mpa"], rel=1e-6)
    assert r["remap_max_dist_m"] == pytest.approx(0.0, abs=1e-9)  # deterministic mesh -> identity
    assert r["j_fan_3d"] == pytest.approx(9.6e12)  # fine-J_fan carried into the record for ranking


def test_rescreen_repins_skin_solid():
    # The aero skin is always-solid (holes forbidden). If the loaded density has de-pinned skin (rho<1)
    # after any remap noise, the re-screen must re-pin it to 1.0 on the rebuilt mesh before screening.
    params = decode(np.full(N_DIMS, 0.5))
    mp = FeaMeshParams(mesh_size_m=0.006)
    problem = build_blade_to_problem(
        build_blade_fea_mesh(params, mesh_params=mp),
        hub_radius_m=mp.hub_radius_m, tip_radius_m=mp.tip_radius_m, click_band_m=mp.click_band_m,
        skin_thickness_m=DEFAULT_SKIN_THICKNESS_M, rmin_m=DEFAULT_RMIN_M, build_filter=False,
    )
    frozen = problem.domain.frozen_mask
    density = np.full(len(frozen), 0.5)
    density[frozen] = 0.3  # DE-PIN the skin (invalid); the re-screen must repair it
    res = rescreen_blade_from_density(params, density, problem.domain.element_centroids, mesh_params=mp)
    assert np.all(res.density[frozen] == 1.0)  # skin restored to solid


def test_rescreen_batch_leaves_original_sidecar_untouched(tmp_path):
    designs = [("00_a", decode(np.full(N_DIMS, 0.5)))]
    mp = FeaMeshParams(mesh_size_m=0.006)
    run_blade_to_batch(designs, tmp_path, mesh_params=mp, max_iters=1, tol=1e-6)
    before = (tmp_path / "00_a.json").read_text()
    rescreen_blade_to_batch(designs, tmp_path, mesh_params=mp, n_workers=1)
    assert (tmp_path / "00_a.json").read_text() == before  # rescreen writes a separate sidecar


def test_rescreen_serial_isolates_missing_density_and_fires_callback(tmp_path):
    good = decode(np.full(N_DIMS, 0.5))
    mp = FeaMeshParams(mesh_size_m=0.006)
    run_blade_to_batch([("00_a", good)], tmp_path, mesh_params=mp, max_iters=1, tol=1e-6)
    seen = []
    summary = rescreen_blade_to_batch(
        [("00_a", good), ("99_missing", good)], tmp_path, mesh_params=mp, n_workers=1,
        on_result=lambda name, res: seen.append(name),
    )
    assert summary["n_succeeded"] == 1  # 00_a screens; 99_missing has no saved density -> isolated error
    assert seen == ["00_a"]  # serial callback fired only for the successful design
    assert "error" in next(r for r in summary["designs"] if r["name"] == "99_missing")


def test_rescreen_batch_parallel(tmp_path):
    designs = [("00_a", decode(np.full(N_DIMS, 0.5))), ("01_b", decode(np.full(N_DIMS, 0.45)))]
    mp = FeaMeshParams(mesh_size_m=0.006)
    run_blade_to_batch(designs, tmp_path, mesh_params=mp, max_iters=1, tol=1e-6)
    seen = []
    summary = rescreen_blade_to_batch(
        designs, tmp_path, mesh_params=mp, n_workers=2, on_result=lambda name, res: seen.append(name)
    )
    assert summary["n_succeeded"] == 2 and summary["n_workers"] == 2
    assert sorted(seen) == ["00_a", "01_b"]  # parallel callback fired for each design
    assert (tmp_path / "00_a_rescreen.json").exists() and (tmp_path / "01_b_rescreen.json").exists()


# --- solid-only stress (gray-element artifact check) ------------------------------------

def test_solid_only_stress_is_a_subset_of_all_element_stress():
    # σ_VM over solid (ρ≥0.5) elements only must be positive and never exceed the all-element peak,
    # since the solid set is a subset. This is the honest as-printed stress used to check whether a
    # design's peak σ_VM is a gray-element artifact.
    _prob, res = _run_slab(max_iters=3)
    assert 0.0 < res.max_von_mises_solid_pa <= res.max_von_mises_pa


def test_solid_only_stress_recorded_per_load():
    _prob, res = _run_slab(max_iters=3)
    assert set(res.vm_solid_by_load_pa) == set(_LOAD_CASE_NAMES)


def _fake_result_with_loads():
    return BladeTOResult(
        density=np.array([1.0, 0.2]), compliance_history=(2.0, 1.0), design_volume_fraction=0.4,
        volume_removed_frac=0.3, mass_kg=0.02, u_tip_max_m=5e-4, max_von_mises_pa=1e6,
        max_von_mises_solid_pa=4e5,
        u_tip_by_load_m={"productive_stroke": 3e-4, "inertial": 5e-4},
        vm_by_load_pa={"productive_stroke": 1e6, "inertial": 4e5},
        vm_solid_by_load_pa={"productive_stroke": 4e5, "inertial": 2e5},
        converged=True, iterations=3, meta={"n_design": 1.0, "n_frozen": 1.0},
    )


def test_batch_serial_fires_on_result_per_design(tmp_path):
    seen = []
    stub = types.SimpleNamespace(uniform=False)
    run_blade_to_batch(
        [("00_a", stub), ("01_b", stub)], tmp_path, n_workers=1,
        optimize=lambda *a, **k: _fake_result(), on_result=lambda name, res: seen.append(name),
    )
    assert seen == ["00_a", "01_b"]  # serial path invokes the callback in order


def test_batch_sidecar_persists_per_load_breakdown(tmp_path):
    stub = types.SimpleNamespace(uniform=False)
    run_blade_to_batch([("00_a", stub)], tmp_path, optimize=lambda *a, **k: _fake_result_with_loads())
    rec = json.loads((tmp_path / "00_a.json").read_text())
    assert rec["u_tip_by_load_mm"]["inertial"] == pytest.approx(0.5)  # 5e-4 m -> 0.5 mm
    assert rec["vm_by_load_mpa"]["productive_stroke"] == pytest.approx(1.0)  # 1e6 Pa -> 1 MPa
    assert rec["max_von_mises_solid_mpa"] == pytest.approx(0.4)  # 4e5 Pa -> 0.4 MPa
    assert rec["vm_solid_by_load_mpa"]["inertial"] == pytest.approx(0.2)  # 2e5 Pa -> 0.2 MPa


# --- durable-write helpers (Colab Drive crash safety) -----------------------------------

def test_fsync_path_flushes_existing_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"payload")
    _fsync_path(p)  # must not raise on a plain file
    assert p.read_bytes() == b"payload"


def test_write_json_durable_is_atomic_and_leaves_no_tmp(tmp_path):
    p = tmp_path / "rec.json"
    _write_json_durable(p, {"a": 1, "b": [2, 3]})
    assert json.loads(p.read_text()) == {"a": 1, "b": [2, 3]}
    assert not (tmp_path / "rec.json.tmp").exists()  # tmp renamed away, not left behind


def test_write_json_durable_overwrites_previous(tmp_path):
    p = tmp_path / "rec.json"
    _write_json_durable(p, {"v": 1})
    _write_json_durable(p, {"v": 2})
    assert json.loads(p.read_text()) == {"v": 2}
