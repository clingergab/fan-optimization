"""Tests for the anisotropic linear-elastic FEA. Requires scikit-fem."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("skfem") is None:
    pytest.skip("scikit-fem required", allow_module_level=True)

from skfem import MeshTet

from fanopt.topopt.fea import (
    assemble_global_stiffness,
    build_fea_model,
    compliance,
    element_strains,
    element_strain_energies,
    element_stresses,
    solve_displacements,
    solve_displacements_multi,
)
from fanopt.topopt.material import isotropic_stiffness, transversely_isotropic_stiffness

_C_ISO = isotropic_stiffness(1.0e9, 0.3)


def _cube(refine: int = 2) -> tuple[np.ndarray, np.ndarray]:
    m = MeshTet().refined(refine)
    return m.p.T.copy(), m.t.T.copy()


def _nodal_field_to_dofs(model, field: np.ndarray) -> np.ndarray:
    u = np.zeros(model.n_dofs)
    nd = model.basis.nodal_dofs
    for c in range(3):
        u[nd[c]] = field[:, c]
    return u


def test_volumes_sum_to_unit_cube():
    nodes, tets = _cube(2)
    model = build_fea_model(nodes, tets, _C_ISO, np.array([], dtype=int))
    assert model.volumes.sum() == pytest.approx(1.0)


def test_patch_uniaxial_strain_reproduced_exactly():
    nodes, tets = _cube(1)
    model = build_fea_model(nodes, tets, _C_ISO, np.array([], dtype=int))
    field = np.zeros_like(nodes)
    field[:, 0] = 0.01 * nodes[:, 0]  # u_x = 0.01 x → ε_xx = 0.01
    eps = element_strains(model, _nodal_field_to_dofs(model, field))
    assert np.allclose(eps, [0.01, 0, 0, 0, 0, 0], atol=1e-12)


def test_patch_shear_strain_reproduced_exactly():
    nodes, tets = _cube(1)
    model = build_fea_model(nodes, tets, _C_ISO, np.array([], dtype=int))
    field = np.zeros_like(nodes)
    field[:, 0] = 0.02 * nodes[:, 1]  # u_x = 0.02 y → engineering γ_xy = 0.02 (slot 5)
    eps = element_strains(model, _nodal_field_to_dofs(model, field))
    assert np.allclose(eps[:, 5], 0.02, atol=1e-12) and np.allclose(eps[:, :5], 0.0, atol=1e-12)


def test_stress_equals_C_times_strain():
    nodes, tets = _cube(1)
    model = build_fea_model(nodes, tets, _C_ISO, np.array([], dtype=int))
    field = np.zeros_like(nodes)
    field[:, 0] = 0.01 * nodes[:, 0]
    u = _nodal_field_to_dofs(model, field)
    sig = element_stresses(model, u)
    assert np.allclose(sig[0], _C_ISO @ np.array([0.01, 0, 0, 0, 0, 0]))


def test_stiffness_is_symmetric():
    nodes, tets = _cube(2)
    model = build_fea_model(nodes, tets, _C_ISO, np.array([], dtype=int))
    k = assemble_global_stiffness(model)
    assert abs((k - k.T)).max() < 1.0  # Pa-scale noise on ~1e9 entries


def test_rigid_translation_has_no_strain_energy():
    nodes, tets = _cube(2)
    model = build_fea_model(nodes, tets, _C_ISO, np.array([], dtype=int))
    k = assemble_global_stiffness(model)
    trans = np.zeros_like(nodes)
    trans[:, 0] = 1.0  # uniform +x translation
    ku = k @ _nodal_field_to_dofs(model, trans)
    assert np.abs(ku).max() < 1.0  # negligible vs ~1e9 stiffness


def test_cantilever_deflects_in_load_direction():
    nodes, tets = _cube(3)
    support = np.nonzero(nodes[:, 0] < 1e-9)[0]
    model = build_fea_model(nodes, tets, _C_ISO, support)
    forces = np.zeros_like(nodes)
    tip = np.nonzero(nodes[:, 0] > 1 - 1e-9)[0]
    forces[tip, 2] = 1.0 / len(tip)  # +z shear at the far face
    u, f = solve_displacements(model, forces)
    tip_dz = u[model.basis.nodal_dofs[2, tip]].mean()
    assert tip_dz > 0.0
    assert compliance(u, f) > 0.0


def test_multi_solve_matches_single_solve_per_load():
    # Factorize-once/back-substitute must reproduce the per-load direct solve exactly.
    nodes, tets = _cube(3)
    support = np.nonzero(nodes[:, 0] < 1e-9)[0]
    model = build_fea_model(nodes, tets, _C_ISO, support)
    tip = np.nonzero(nodes[:, 0] > 1 - 1e-9)[0]
    loads = []
    for axis in (0, 1, 2):
        fce = np.zeros_like(nodes)
        fce[tip, axis] = 1.0 / len(tip)
        loads.append(fce)
    k = assemble_global_stiffness(model)
    multi = solve_displacements_multi(model, loads, k)
    for fce, (u_m, _) in zip(loads, multi):
        u_s, _ = solve_displacements(model, fce, k)
        assert np.allclose(u_m, u_s, atol=1e-10)


def test_element_energies_sum_to_compliance():
    nodes, tets = _cube(2)
    support = np.nonzero(nodes[:, 0] < 1e-9)[0]
    model = build_fea_model(nodes, tets, _C_ISO, support)
    forces = np.zeros_like(nodes)
    tip = np.nonzero(nodes[:, 0] > 1 - 1e-9)[0]
    forces[tip, 2] = 1.0 / len(tip)
    u, f = solve_displacements(model, forces)
    assert element_strain_energies(model, u).sum() == pytest.approx(compliance(u, f), rel=1e-6)


def test_material_frames_identity_matches_global():
    nodes, tets = _cube(1)
    frames = np.broadcast_to(np.eye(3), (len(tets), 3, 3)).copy()
    with_frames = build_fea_model(nodes, tets, _C_ISO, np.array([], dtype=int), material_frames=frames)
    global_frame = build_fea_model(nodes, tets, _C_ISO, np.array([], dtype=int))
    assert np.allclose(with_frames.base_stiffness, global_frame.base_stiffness)


def test_density_scales_stiffness_by_penal_power():
    nodes, tets = _cube(2)
    model = build_fea_model(nodes, tets, _C_ISO, np.array([], dtype=int))
    k_full = assemble_global_stiffness(model)
    k_half = assemble_global_stiffness(model, density=0.5 * np.ones(len(tets)), penal=3.0)
    assert abs(k_half - 0.125 * k_full).max() < 1.0  # 0.5^3 = 0.125


def test_weak_build_axis_is_more_compliant_than_strong():
    # Transversely isotropic: axis-3 (z) weak (E_z<E_xy). Pull the cube axially along z
    # (weak) vs x (strong); equal load, identical geometry → z is more compliant.
    nodes, tets = _cube(2)
    c = transversely_isotropic_stiffness(1.30e9, 1.00e9, 0.38, 0.38, 0.28e9)

    def axial_compliance(axis: int) -> float:
        support = np.nonzero(nodes[:, axis] < 1e-9)[0]
        model = build_fea_model(nodes, tets, c, support)
        forces = np.zeros_like(nodes)
        far = np.nonzero(nodes[:, axis] > 1 - 1e-9)[0]
        forces[far, axis] = 1.0 / len(far)
        u, f = solve_displacements(model, forces)
        return compliance(u, f)

    assert axial_compliance(2) > axial_compliance(0)
