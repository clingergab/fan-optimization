"""Tests for the 12-blade fan posing (fanopt.geometry.fan_pose)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fanopt.bo.blade_codec import N_DIMS, decode
from fanopt.geometry.blade import BLADE_COUNT, FOLD_CLEARANCE_M, layer_spacing_m
from fanopt.geometry.fan_pose import apply_pose, fan_blade_poses, pose_fan
from fanopt.geometry.schema import INTER_BLADE_ANGLE_DEG


def test_folded_poses_are_all_aligned():
    poses = fan_blade_poses(12, 0.004, deployed=False)
    assert all(rot == 0.0 for rot, _z in poses)  # folded = pure z-stack, no fan-out
    assert [z for _rot, z in poses] == [i * 0.004 for i in range(12)]


def test_deployed_poses_are_symmetric_about_zero():
    poses = fan_blade_poses(12, 0.004, deployed=True)
    rots = [rot for rot, _z in poses]
    assert rots[0] == pytest.approx(-INTER_BLADE_ANGLE_DEG * 5.5)  # (0 - (12-1)/2) * pitch
    assert rots[-1] == pytest.approx(INTER_BLADE_ANGLE_DEG * 5.5)
    assert rots[-1] - rots[0] == pytest.approx(11 * INTER_BLADE_ANGLE_DEG)  # full 11-gap spread


def test_direction_reverses_deploy_fan_out():
    fwd = fan_blade_poses(12, 0.004, deployed=True, direction=1)
    rev = fan_blade_poses(12, 0.004, deployed=True, direction=-1)
    assert [r for r, _z in rev] == [-r for r, _z in fwd]  # every rotation negated
    assert [z for _r, z in rev] == [z for _r, z in fwd]  # z-layers unchanged (fold-symmetric)


def test_pose_fan_direction_passes_through():
    params = decode(np.full(N_DIMS, 0.5))
    v = np.array([[0.1, 0.0, 0.0]])
    f = np.array([[0, 0, 0]])
    fwd = pose_fan(v, f, params, deployed=True, direction=1)
    rev = pose_fan(v, f, params, deployed=True, direction=-1)
    # blade 0's posed x,y is mirrored across the x-axis (opposite rotation sign) but same radius
    assert np.hypot(*fwd[0][0][0, :2]) == pytest.approx(np.hypot(*rev[0][0][0, :2]))
    assert fwd[0][0][0, 1] == pytest.approx(-rev[0][0][0, 1])


def test_deployed_z_layers_match_folded():
    folded = fan_blade_poses(12, 0.004, deployed=False)
    deployed = fan_blade_poses(12, 0.004, deployed=True)
    assert [z for _r, z in folded] == [z for _r, z in deployed]  # same pin layers, only rotation differs


def test_apply_pose_rotates_about_pin_axis():
    r = apply_pose(np.array([[1.0, 0.0, 0.0]]), 90.0, 0.0)
    assert np.allclose(r, [[0.0, 1.0, 0.0]], atol=1e-12)  # +x rotates to +y about +z


def test_apply_pose_applies_z_offset():
    r = apply_pose(np.array([[1.0, 2.0, 3.0]]), 0.0, 0.005)
    assert np.allclose(r, [[1.0, 2.0, 3.005]])


def test_apply_pose_preserves_pin_axis_distance():
    # Rotation about +z must not change a point's distance from the pin axis or its z.
    v = np.array([[0.1, 0.05, 0.02]])
    r = apply_pose(v, 37.0, 0.0)
    assert np.hypot(*r[0, :2]) == pytest.approx(np.hypot(*v[0, :2]))
    assert r[0, 2] == pytest.approx(v[0, 2])


def test_pose_fan_returns_one_mesh_per_blade():
    params = decode(np.full(N_DIMS, 0.5))
    v = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.15, 0.01, 0.0]])
    f = np.array([[0, 1, 2]])
    fan = pose_fan(v, f, params, deployed=True)
    assert len(fan) == BLADE_COUNT
    assert all(faces is f for _v, faces in fan)  # faces shared, only vertices move


def test_pose_fan_honors_params_blade_count():
    # blade_count comes from the DESIGN, not a hard-coded 12: a (hypothetical) 8-blade design poses
    # as 8 blades, not 12. Guards the single-source-of-truth for the count.
    params = dataclasses.replace(decode(np.full(N_DIMS, 0.5)), blade_count=8)
    v = np.array([[0.1, 0.0, 0.0]])
    f = np.array([[0, 0, 0]])
    assert len(pose_fan(v, f, params, deployed=True)) == 8


def test_pose_fan_explicit_blade_count_overrides_params():
    params = decode(np.full(N_DIMS, 0.5))  # blade_count == 12
    v = np.array([[0.1, 0.0, 0.0]])
    f = np.array([[0, 0, 0]])
    assert len(pose_fan(v, f, params, deployed=False, blade_count=3)) == 3


def test_pose_fan_clearance_tightens_the_z_stack():
    params = decode(np.full(N_DIMS, 0.5))
    v = np.array([[0.1, 0.0, 0.0]])
    f = np.array([[0, 0, 0]])
    base = pose_fan(v, f, params, deployed=False)
    tight = pose_fan(v, f, params, deployed=False, clearance_m=0.2e-3)
    # the z between adjacent layers drops by exactly the clearance reduction
    dz_base = base[1][0][0, 2] - base[0][0][0, 2]
    dz_tight = tight[1][0][0, 2] - tight[0][0][0, 2]
    assert dz_base - dz_tight == pytest.approx(FOLD_CLEARANCE_M - 0.2e-3)


def test_pose_fan_folded_stacks_in_z_only():
    params = decode(np.full(N_DIMS, 0.5))
    v = np.array([[0.1, 0.0, 0.0]])
    f = np.array([[0, 0, 0]])
    fan = pose_fan(v, f, params, deployed=False)
    spacing = layer_spacing_m(params)
    # folded: each blade is the base shifted purely in z by i*spacing (x,y unchanged)
    for i, (vv, _f) in enumerate(fan):
        assert np.allclose(vv[0], [0.1, 0.0, i * spacing])
