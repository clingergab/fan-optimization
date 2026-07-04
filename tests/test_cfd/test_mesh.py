"""Tests for fanopt.cfd.mesh (3D external-flow volume mesher). Requires gmsh."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

if importlib.util.find_spec("gmsh") is None:  # pragma: no cover - env-dependent
    pytest.skip("gmsh not installed", allow_module_level=True)

import gmsh

from fanopt.cfd.mesh import (
    FAN_SURFACE_MARKER,
    FARFIELD_MARKER,
    VolumeMeshParams,
    build_volume_mesh,
)


def _make_box_step(path: Path) -> Path:
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("body")
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, 0.02, 0.02, 0.003)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


def _make_surface_step(path: Path) -> Path:
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("surf")
        gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, 0.02, 0.02)  # 2D only, no solid
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


_COARSE = VolumeMeshParams(wall_mesh_size_m=0.006, farfield_mesh_size_m=0.05)


# --- param validation (pure) ---


def test_params_reject_negative_pad():
    with pytest.raises(ValueError, match="padding must be positive"):
        VolumeMeshParams(pad_lateral_m=-0.1)


def test_params_reject_farfield_smaller_than_wall():
    with pytest.raises(ValueError, match="farfield_mesh_size_m must be >="):
        VolumeMeshParams(wall_mesh_size_m=0.05, farfield_mesh_size_m=0.01)


def test_params_reject_nonpositive_size():
    with pytest.raises(ValueError, match="mesh sizes must be positive"):
        VolumeMeshParams(wall_mesh_size_m=0.0)


# --- meshing ---


def test_build_volume_mesh_produces_valid_mesh(tmp_path):
    step = _make_box_step(tmp_path / "body.step")
    res = build_volume_mesh(step, _COARSE, tmp_path / "vol.su2")
    assert res.n_nodes > 0
    assert res.n_elements > 0
    assert res.path.exists()
    assert int(res.meta["n_bodies"]) == 1


def test_both_markers_tagged(tmp_path):
    step = _make_box_step(tmp_path / "body.step")
    res = build_volume_mesh(step, _COARSE, tmp_path / "vol.su2")
    assert FAN_SURFACE_MARKER in res.markers
    assert FARFIELD_MARKER in res.markers


def test_step_without_solids_raises(tmp_path):
    step = _make_surface_step(tmp_path / "surf.step")
    with pytest.raises(RuntimeError, match="no 3D solids"):
        build_volume_mesh(step, _COARSE, tmp_path / "vol.su2")
