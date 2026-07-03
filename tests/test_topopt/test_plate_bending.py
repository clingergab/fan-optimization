"""Tests for the Mindlin plate-bending FE core (report-final.md §3.1 / §9.2)."""

from __future__ import annotations

import numpy as np
import pytest

from fanopt.topopt.plate_bending import (
    DOF_PER_NODE,
    PlateMesh,
    assemble_global_stiffness,
    element_stiffness_unit,
    solve_displacements,
)


def test_element_stiffness_symmetric():
    ke = element_stiffness_unit(nu=0.3, t=0.002, dx=0.001, dy=0.001)
    assert np.allclose(ke, ke.T)


def test_element_stiffness_shape():
    ke = element_stiffness_unit(nu=0.3, t=0.002, dx=0.001, dy=0.001)
    assert ke.shape == (12, 12)


def test_element_is_positive_semidefinite():
    ke = element_stiffness_unit(nu=0.3, t=0.002, dx=0.001, dy=0.001)
    eig = np.linalg.eigvalsh(ke)
    assert eig.min() > -1e-9 * ke.max()


def test_rigid_body_translation_has_zero_energy():
    # Uniform transverse translation (w=1, rotations=0) is a rigid-body mode:
    # no curvature, no shear -> zero strain energy.
    ke = element_stiffness_unit(nu=0.3, t=0.002, dx=0.001, dy=0.001)
    d = np.zeros(12)
    d[0::3] = 1.0  # all four w-DOFs = 1
    assert float(d @ ke @ d) == pytest.approx(0.0, abs=1e-12 * ke.max())


def test_mesh_node_and_dof_counts():
    mesh = PlateMesh(nelx=4, nely=3, dx=1.0, dy=1.0)
    assert mesh.n_nodes == 5 * 4
    assert mesh.n_dofs == 5 * 4 * DOF_PER_NODE


def test_mesh_element_dofs_length():
    mesh = PlateMesh(nelx=4, nely=3, dx=1.0, dy=1.0)
    assert mesh.element_dofs(0, 0).shape == (12,)


def test_assemble_scales_with_modulus():
    mesh = PlateMesh(nelx=3, nely=3, dx=1.0, dy=1.0)
    k1 = assemble_global_stiffness(mesh, np.ones((3, 3)), nu=0.3, t=0.01)
    k2 = assemble_global_stiffness(mesh, np.full((3, 3), 2.0), nu=0.3, t=0.01)
    assert np.allclose((k2 - 2.0 * k1).toarray(), 0.0, atol=1e-12)


def test_assemble_rejects_shape_mismatch():
    mesh = PlateMesh(nelx=3, nely=3, dx=1.0, dy=1.0)
    with pytest.raises(ValueError, match="e_elem must be"):
        assemble_global_stiffness(mesh, np.ones((2, 2)), nu=0.3, t=0.01)


def test_solve_no_free_dofs_returns_zeros():
    mesh = PlateMesh(nelx=2, nely=2, dx=1.0, dy=1.0)
    k = assemble_global_stiffness(mesh, np.ones((2, 2)), nu=0.3, t=0.01)
    f = np.ones(mesh.n_dofs)
    u = solve_displacements(k, f, np.arange(mesh.n_dofs))
    assert np.allclose(u, 0.0)


def test_clamped_square_plate_matches_analytic():
    """Uniform-load clamped square plate: w_max = 0.00126·q·a⁴/D (Timoshenko)."""
    n, a, t, e, nu, q = 20, 1.0, 0.01, 1.0, 0.3, 1.0
    dxy = a / n
    mesh = PlateMesh(n, n, dxy, dxy)
    k = assemble_global_stiffness(mesh, np.full((n, n), e), nu, t)
    f = np.zeros(mesh.n_dofs)
    for ey in range(n):
        for ex in range(n):
            for nd in mesh.element_nodes(ex, ey):
                f[DOF_PER_NODE * nd] += q * dxy * dxy / 4.0
    fixed = []
    for ix in range(n + 1):
        for iy in range(n + 1):
            if ix in (0, n) or iy in (0, n):
                nid = mesh.node_id(ix, iy)
                fixed += [3 * nid, 3 * nid + 1, 3 * nid + 2]
    u = solve_displacements(k, f, np.array(fixed))
    w_center = u[3 * mesh.node_id(n // 2, n // 2)]
    d = e * t**3 / (12 * (1 - nu**2))
    w_analytic = 0.00126 * q * a**4 / d
    assert w_center == pytest.approx(w_analytic, rel=0.03)
