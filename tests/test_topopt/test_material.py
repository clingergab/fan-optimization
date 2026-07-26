"""Tests for the orthotropic/transversely-isotropic PETG constitutive model."""

from __future__ import annotations

import numpy as np
import pytest

from fanopt.topopt.material import (
    failure_index,
    in_plane_shear_modulus,
    interlaminar_shear,
    isotropic_stiffness,
    orthotropic_stiffness,
    rotate_stiffness,
    stress_bond_matrix,
    transversely_isotropic_stiffness,
    weak_axis_normal_stress,
)

# PETG-like transversely-isotropic constants (Pa); G_z from the cited conservative bracket.
_E_P = 1.30e9
_E_Z = 1.00e9
_NU_P = 0.38
_NU_ZP = 0.38
_G_Z = 0.28e9


def _rotation_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_in_plane_shear_modulus_matches_isotropic_relation():
    assert in_plane_shear_modulus(_E_P, _NU_P) == pytest.approx(_E_P / (2 * (1 + _NU_P)))


def test_isotropic_stiffness_c11_closed_form():
    e, nu = 2.0e9, 0.3
    c = isotropic_stiffness(e, nu)
    expected = e * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    assert c[0, 0] == pytest.approx(expected)


def test_isotropic_stiffness_shear_diagonal_is_G():
    e, nu = 2.0e9, 0.3
    c = isotropic_stiffness(e, nu)
    assert c[3, 3] == pytest.approx(e / (2 * (1 + nu)))


def test_orthotropic_stiffness_is_symmetric():
    c = transversely_isotropic_stiffness(_E_P, _E_Z, _NU_P, _NU_ZP, _G_Z)
    assert np.allclose(c, c.T)


def test_orthotropic_stiffness_positive_definite():
    c = transversely_isotropic_stiffness(_E_P, _E_Z, _NU_P, _NU_ZP, _G_Z)
    assert np.all(np.linalg.eigvalsh(c) > 0.0)


def test_inadmissible_constants_raise():
    # ν too large for the modulus ratio → non-positive-definite compliance inverse.
    with pytest.raises(ValueError):
        orthotropic_stiffness(1e9, 1e9, 1e9, 0.95, 0.95, 0.95, 4e8, 4e8, 4e8)


def test_compliance_round_trip_recovers_E1():
    c = transversely_isotropic_stiffness(_E_P, _E_Z, _NU_P, _NU_ZP, _G_Z)
    s = np.linalg.inv(c)
    assert 1.0 / s[0, 0] == pytest.approx(_E_P)


def test_compliance_round_trip_recovers_Ez():
    c = transversely_isotropic_stiffness(_E_P, _E_Z, _NU_P, _NU_ZP, _G_Z)
    s = np.linalg.inv(c)
    assert 1.0 / s[2, 2] == pytest.approx(_E_Z)


def test_transversely_isotropic_reduces_to_isotropic():
    e, nu = 2.0e9, 0.3
    g = in_plane_shear_modulus(e, nu)
    ti = transversely_isotropic_stiffness(e, e, nu, nu, g)
    assert np.allclose(ti, isotropic_stiffness(e, nu))


def test_rotate_identity_is_noop():
    c = transversely_isotropic_stiffness(_E_P, _E_Z, _NU_P, _NU_ZP, _G_Z)
    assert np.allclose(rotate_stiffness(c, np.eye(3)), c)


def test_rotating_isotropic_is_invariant():
    # Any rotation of an isotropic material is the same material — catches Bond-matrix bugs.
    c = isotropic_stiffness(2.0e9, 0.3)
    assert np.allclose(rotate_stiffness(c, _rotation_z(0.7)), c, atol=1e-3)


def test_rotate_round_trip_recovers_original():
    # Rotate by R then by Rᵀ (= R⁻¹): the Bond matrix is a homomorphism, so C returns.
    c = transversely_isotropic_stiffness(_E_P, _E_Z, _NU_P, _NU_ZP, _G_Z)
    rot = _rotation_z(0.9)
    # atol in Pa: stiffnesses are ~1e9, so 1 Pa is a 1e-9 relative tolerance.
    assert np.allclose(rotate_stiffness(rotate_stiffness(c, rot), rot.T), c, atol=1.0)


def test_rotation_about_symmetry_axis_is_invariant():
    # Rotation about the weak axis (axis 3) is a no-op — that IS transverse isotropy.
    c = transversely_isotropic_stiffness(_E_P, _E_Z, _NU_P, _NU_ZP, _G_Z)
    assert np.allclose(rotate_stiffness(c, _rotation_z(0.6)), c, atol=1.0)


def test_rotation_off_symmetry_axis_changes_stiffness():
    # Tilting the weak axis away from z (rotation about x) is NOT a no-op (sanity it acts).
    c = transversely_isotropic_stiffness(_E_P, _E_Z, _NU_P, _NU_ZP, _G_Z)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(0.6), -np.sin(0.6)], [0.0, np.sin(0.6), np.cos(0.6)]])
    assert not np.allclose(rotate_stiffness(c, rot_x), c)


def test_stress_bond_matrix_rejects_non_3x3():
    with pytest.raises(ValueError):
        stress_bond_matrix(np.eye(2))


def test_weak_axis_normal_stress_picks_sigma33():
    stress = np.array([10.0, 20.0, 99.0, 1.0, 2.0, 3.0])
    assert weak_axis_normal_stress(stress) == pytest.approx(99.0)


def test_interlaminar_shear_is_resultant_of_sigma23_sigma13():
    stress = np.array([0.0, 0.0, 0.0, 3.0, 4.0, 9.0])
    assert interlaminar_shear(stress) == pytest.approx(5.0)


def test_failure_index_weak_normal_dominates():
    # Pure σ33 at the weak yield → index exactly 1.0 (the weak-Z gate).
    stress = np.array([0.0, 0.0, 30e6, 0.0, 0.0, 0.0])
    assert failure_index(stress, 45e6, 30e6) == pytest.approx(1.0)


def test_failure_index_uses_lower_weak_yield():
    # Same stress magnitude read against weak vs in-plane yield → weak gives a bigger index.
    stress = np.array([30e6, 0.0, 30e6, 0.0, 0.0, 0.0])
    assert failure_index(stress, 45e6, 30e6) == pytest.approx(1.0)


def test_failure_index_includes_interlaminar_shear_when_strength_given():
    stress = np.array([0.0, 0.0, 0.0, 20e6, 0.0, 0.0])
    assert failure_index(stress, 45e6, 30e6, tau_interlaminar=20e6) == pytest.approx(1.0)
