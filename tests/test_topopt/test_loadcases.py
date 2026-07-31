"""Tests for the four locked structural load cases."""

from __future__ import annotations

import numpy as np
import pytest

from fanopt.geometry.schema import ALPHA_MAX_RAD_PER_S2, L_WRIST_TO_TIP_M, RHO_PETG_KG_PER_M3
from fanopt.topopt.loadcases import (
    assemble_load_cases,
    click_load,
    default_angular_acceleration,
    inertial_body_load,
    productive_stroke_load,
    return_stroke_load,
    tip_force_from_wrist_torque,
)

# One triangle facet in the z=0 plane, outward normal +z.
_FACET_IDS = np.array([[0, 1, 2]])
_FACET_AREA = np.array([2.0])
_FACET_NORMAL = np.array([[0.0, 0.0, 1.0]])
_FACET_PRESSURE = np.array([100.0])

# One tet: base triangle + apex, node ids 0..3.
_ELEM_IDS = np.array([[0, 1, 2, 3]])
_ELEM_VOL = np.array([1e-6])


def test_angular_acceleration_magnitude_is_alpha_max():
    assert np.linalg.norm(default_angular_acceleration()) == pytest.approx(ALPHA_MAX_RAD_PER_S2)


def test_angular_acceleration_points_negative_y():
    # C11 sign lock: the stroke/inertial axis is negative-y.
    a = default_angular_acceleration()
    assert a[1] < 0 and a[0] == pytest.approx(0.0) and a[2] == pytest.approx(0.0)


def test_tip_force_uses_wrist_lever_arm_not_blade():
    # H8 lock: τ→F divides by 0.27 m (ADR-0005 lever), so 1 N·m → ≈ 3.70 N.
    assert tip_force_from_wrist_torque(1.0) == pytest.approx(1.0 / L_WRIST_TO_TIP_M)
    assert tip_force_from_wrist_torque(1.0) == pytest.approx(3.7037, abs=1e-4)


def test_pressure_load_total_force_is_minus_pnA():
    f = productive_stroke_load(3, _FACET_IDS, _FACET_AREA, _FACET_NORMAL, _FACET_PRESSURE)
    # Σ nodal force = −p·A·n = −100·2·ẑ.
    assert np.allclose(f.sum(axis=0), [0.0, 0.0, -200.0])


def test_pressure_load_splits_equally_across_facet_nodes():
    f = productive_stroke_load(3, _FACET_IDS, _FACET_AREA, _FACET_NORMAL, _FACET_PRESSURE)
    assert np.allclose(f[0], f[1]) and np.allclose(f[1], f[2])


def test_return_stroke_reverses_productive():
    prod = productive_stroke_load(3, _FACET_IDS, _FACET_AREA, _FACET_NORMAL, _FACET_PRESSURE)
    ret = return_stroke_load(3, _FACET_IDS, _FACET_AREA, _FACET_NORMAL, _FACET_PRESSURE)
    assert np.allclose(ret, -prod)


def test_inertial_load_direction_from_alpha_cross_r():
    # α along −y, element at +x → a = α×r = (0,−a,0)×(x,0,0) = (0,0,+a·x): force in +z.
    centroids = np.array([[0.05, 0.0, 0.0]])
    f = inertial_body_load(4, _ELEM_IDS, _ELEM_VOL, centroids)
    total = f.sum(axis=0)
    assert total[2] > 0 and total[0] == pytest.approx(0.0) and total[1] == pytest.approx(0.0)


def test_inertial_load_total_equals_rho_vol_accel():
    centroids = np.array([[0.05, 0.0, 0.0]])
    f = inertial_body_load(4, _ELEM_IDS, _ELEM_VOL, centroids)
    a = default_angular_acceleration()
    expected = RHO_PETG_KG_PER_M3 * _ELEM_VOL[0] * np.cross(a, centroids[0])
    assert np.allclose(f.sum(axis=0), expected)


def test_inertial_load_scales_with_density():
    centroids = np.array([[0.05, 0.0, 0.0]])
    full = inertial_body_load(4, _ELEM_IDS, _ELEM_VOL, centroids)
    half = inertial_body_load(4, _ELEM_IDS, _ELEM_VOL, centroids, element_density=np.array([0.5]))
    assert np.allclose(half.sum(axis=0), 0.5 * full.sum(axis=0))


def test_click_load_total_equals_applied_force():
    f = click_load(5, np.array([1, 3]), np.array([0.0, 0.0, -2.0]))
    assert np.allclose(f.sum(axis=0), [0.0, 0.0, -2.0])


def test_click_load_splits_across_nodes():
    f = click_load(5, np.array([1, 3]), np.array([0.0, 0.0, -2.0]))
    assert np.allclose(f[1], [0.0, 0.0, -1.0]) and np.allclose(f[3], [0.0, 0.0, -1.0])


def test_assemble_returns_four_named_cases():
    cases = assemble_load_cases(
        4, _FACET_IDS, _FACET_AREA, _FACET_NORMAL, _FACET_PRESSURE,
        _ELEM_IDS, _ELEM_VOL, np.array([[0.05, 0.0, 0.0]]),
        np.array([3]), np.array([0.0, 0.0, -2.0]),
    )
    assert [c.name for c in cases] == [
        "productive_stroke", "return_stroke", "inertial", "click_engagement",
    ]
