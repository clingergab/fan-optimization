"""Tests for fanopt.cfd.airfoil_mesh (2D NACA-airfoil-in-farfield mesher). Needs gmsh."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("gmsh") is None:  # pragma: no cover - env-dependent
    pytest.skip("gmsh not installed", allow_module_level=True)

from fanopt.cfd.airfoil_mesh import (
    AIRFOIL_MARKER,
    FARFIELD_MARKER,
    AirfoilMeshParams,
    build_airfoil_mesh,
)
from fanopt.cfd.airfoil_shapes import airfoil_polyline

# Coarse case keeps the mesh cheap; the numbers only need to be structurally sane.
_COARSE = AirfoilMeshParams(farfield_radius_chords=8.0, wall_mesh_size_m=0.02, farfield_mesh_size_m=1.0)


def _poly() -> np.ndarray:
    return np.array(airfoil_polyline(40, chord=1.0), dtype=float)


# --- param validation ---


def test_params_reject_tiny_farfield():
    with pytest.raises(ValueError, match="farfield_radius_chords"):
        AirfoilMeshParams(farfield_radius_chords=1.0)


def test_params_reject_wall_coarser_than_farfield():
    with pytest.raises(ValueError, match="wall_mesh_size_m must be"):
        AirfoilMeshParams(wall_mesh_size_m=1.0, farfield_mesh_size_m=0.5)


def test_params_reject_nonpositive_refine_distance():
    with pytest.raises(ValueError, match="wall_refine_distance_m"):
        AirfoilMeshParams(wall_refine_distance_m=0.0)


def test_build_rejects_open_polyline_shape():
    with pytest.raises(ValueError, match="polyline must be"):
        build_airfoil_mesh(np.zeros((2, 2)), _COARSE, "/tmp/none.su2")


# --- mesh generation ---


def test_build_writes_mesh_file(tmp_path):
    out = tmp_path / "airfoil.su2"
    res = build_airfoil_mesh(_poly(), _COARSE, out)
    assert out.exists()
    assert res.n_nodes > 0
    assert res.n_elements > 0


def test_build_tags_airfoil_and_farfield_markers(tmp_path):
    res = build_airfoil_mesh(_poly(), _COARSE, tmp_path / "a.su2")
    assert set(res.markers) == {AIRFOIL_MARKER, FARFIELD_MARKER}


def test_build_records_farfield_radius_in_meta(tmp_path):
    res = build_airfoil_mesh(_poly(), _COARSE, tmp_path / "a.su2", chord_m=2.0)
    assert res.meta["farfield_radius_m"] == pytest.approx(8.0 * 2.0)
    assert res.meta["chord_m"] == pytest.approx(2.0)
