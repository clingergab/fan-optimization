"""Tests for the 2D cascade-slice mesher (report-final.md §9.6).

Skips at module level where ``gmsh`` is not installed (the mesher imports it
unconditionally per CLAUDE.md §4.1).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("gmsh") is None:  # pragma: no cover - env-dependent
    pytest.skip("gmsh not installed", allow_module_level=True)

from fanopt.cfd.mesh_2d_slice import (
    CASCADE_WALL_MARKER,
    DOWNSTREAM_PLANE_MARKER,
    FAN_SURFACE_MARKER,
    FARFIELD_MARKER,
    SliceMeshParams,
    baseline_cascade_polygons,
    build_cascade_slice_mesh,
)

# --- baseline_cascade_polygons ------------------------------------------------


def test_baseline_polygon_count():
    polys = baseline_cascade_polygons(n_blades=5)
    assert len(polys) == 5


def test_baseline_polygon_shape():
    polys = baseline_cascade_polygons(n_blades=3)
    assert all(p.shape == (4, 2) for p in polys)


def test_baseline_corrugation_panel_above_rib():
    # §3.2.4 invariant: panel protrudes above the rib on +z (streamwise axis 0).
    polys = baseline_cascade_polygons(panel_thickness_m=0.0038, rib_thickness_m=0.002)
    panel_top = max(p[:, 0].max() for p in polys)
    rib_top = 0.002 / 2.0
    assert panel_top > rib_top


def test_baseline_cross_section_centered():
    polys = baseline_cascade_polygons(n_blades=4)
    lo = min(p[:, 1].min() for p in polys)
    hi = max(p[:, 1].max() for p in polys)
    assert lo == pytest.approx(-hi)


def test_baseline_rejects_zero_blades():
    with pytest.raises(ValueError, match="n_blades"):
        baseline_cascade_polygons(n_blades=0)


# --- SliceMeshParams validation ----------------------------------------------


def test_params_outlet_must_exceed_plane():
    with pytest.raises(ValueError, match="must exceed the"):
        SliceMeshParams(streamwise_outlet_m=0.1, downstream_plane_m=0.3)


def test_params_plane_must_be_positive():
    with pytest.raises(ValueError, match="downstream_plane_m must be"):
        SliceMeshParams(downstream_plane_m=-0.1, streamwise_outlet_m=0.45)


def test_params_extents_positive():
    with pytest.raises(ValueError, match="extents must be positive"):
        SliceMeshParams(cross_half_extent_m=-0.1)


def test_params_mesh_sizes_positive():
    with pytest.raises(ValueError, match="mesh sizes must be positive"):
        SliceMeshParams(mesh_size_m=0.0)


# --- build_cascade_slice_mesh -------------------------------------------------


def test_build_default_markers_and_counts(tmp_path):
    out = tmp_path / "slice.su2"
    res = build_cascade_slice_mesh(baseline_cascade_polygons(n_blades=4), SliceMeshParams(), out)
    assert res.markers == (FAN_SURFACE_MARKER, FARFIELD_MARKER, CASCADE_WALL_MARKER)
    assert res.n_nodes > 0
    assert res.n_elements > 0
    assert res.n_blades == 4


def test_build_writes_markers_to_su2(tmp_path):
    out = tmp_path / "slice.su2"
    res = build_cascade_slice_mesh(baseline_cascade_polygons(n_blades=3), SliceMeshParams(), out)
    text = out.read_text()
    assert all(m in text for m in res.markers)


def test_build_includes_downstream_plane_when_requested(tmp_path):
    out = tmp_path / "slice.msh"
    res = build_cascade_slice_mesh(
        baseline_cascade_polygons(n_blades=3),
        SliceMeshParams(include_downstream_plane=True),
        out,
    )
    assert DOWNSTREAM_PLANE_MARKER in res.markers
    assert DOWNSTREAM_PLANE_MARKER in out.read_text()


def test_build_meta_records_geometry(tmp_path):
    out = tmp_path / "slice.su2"
    res = build_cascade_slice_mesh(baseline_cascade_polygons(n_blades=2), SliceMeshParams(), out)
    assert res.meta["downstream_plane_m"] == pytest.approx(0.300)
    assert res.meta["streamwise_span_m"] > 0
    assert res.meta["cross_span_m"] > 0


def test_build_rejects_empty_cross_section(tmp_path):
    with pytest.raises(ValueError, match="at least one polygon"):
        build_cascade_slice_mesh([], SliceMeshParams(), tmp_path / "x.su2")


def test_build_rejects_degenerate_polygon(tmp_path):
    bad = [np.array([[0.0, 0.0], [1.0, 0.0]])]  # only 2 points
    with pytest.raises(ValueError, match=r"polygon must be"):
        build_cascade_slice_mesh(bad, SliceMeshParams(), tmp_path / "x.su2")
