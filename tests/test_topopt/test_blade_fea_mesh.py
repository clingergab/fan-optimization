"""Tests for the solid FEA mesher + geometric region taggers. Requires gmsh + CadQuery."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("gmsh") is None or importlib.util.find_spec("cadquery") is None:
    pytest.skip("gmsh + cadquery required", allow_module_level=True)

from fanopt.geometry.blade import BladeParams
from fanopt.topopt.blade_fea_mesh import (
    FeaMeshParams,
    boundary_facets,
    build_blade_fea_mesh,
    classify_aero_skin,
    click_region_facets,
    cylindrical_radius,
    facet_geometry,
    hub_support_nodes,
)

_TRI = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _gentle_design() -> BladeParams:
    return BladeParams.from_dict(
        {
            "blade_count": 8,
            "rib_bow_knots_m": [0.006, 0.010, 0.012, 0.010, 0.006],
            "rib_bow_interp": "linear",
            "t_rib_hub_m": 0.005,
            "t_rib_tip_m": 0.005,
            "panel_offsets_m": [[0.0, 0.0, 0.0]] * 4,
            "panel_thickness_m": [[0.003, 0.003, 0.003]] * 4,
        }
    )


def test_single_tet_has_four_boundary_facets():
    facets, _ = boundary_facets(np.array([[0, 1, 2, 3]]))
    assert len(facets) == 4


def test_shared_face_is_interior():
    # Two tets sharing face {0,1,2}: 4+4 faces − 2 shared = 6 boundary facets.
    facets, _ = boundary_facets(np.array([[0, 1, 2, 3], [0, 1, 2, 4]]))
    assert len(facets) == 6


def test_facet_geometry_area_normal_center():
    centers, normals, areas = facet_geometry(_TRI, np.array([[0, 1, 2]]))
    assert areas[0] == pytest.approx(0.5)
    assert np.allclose(normals[0], [0.0, 0.0, 1.0])
    assert np.allclose(centers[0], [1 / 3, 1 / 3, 0.0])


def test_facet_normal_oriented_away_from_opposite_vertex():
    nodes = np.vstack([_TRI, [0.0, 0.0, 1.0]])  # opposite vertex above the triangle
    _, normals, _ = facet_geometry(nodes, np.array([[0, 1, 2]]), np.array([3]))
    assert np.allclose(normals[0], [0.0, 0.0, -1.0])  # flipped to point away (downward)


def test_cylindrical_radius():
    assert cylindrical_radius(np.array([[3.0, 4.0, 9.0]]))[0] == pytest.approx(5.0)


def test_classify_aero_skin_splits_top_bottom_excludes_hub_and_edges():
    normals = np.array([[0, 0, 1.0], [0, 0, -1.0], [1.0, 0, 0], [0, 0, 1.0]])
    centers = np.array([[0.1, 0, 0], [0.1, 0, 0], [0.1, 0, 0], [0.01, 0, 0]])
    top, bottom = classify_aero_skin(normals, centers, hub_radius_m=0.02)
    assert top.tolist() == [True, False, False, False]
    assert bottom.tolist() == [False, True, False, False]


def test_hub_support_nodes_within_radius():
    nodes = np.array([[0.01, 0, 0], [0.05, 0, 0]])
    assert hub_support_nodes(nodes, hub_radius_m=0.02).tolist() == [0]


def test_click_region_facets_selects_tip_band():
    facets = np.array([[0, 1, 2], [3, 4, 5]])
    centers = np.array([[0.18, 0, 0], [0.05, 0, 0]])
    got = click_region_facets(facets, centers, tip_radius_m=0.185, click_band_m=0.01)
    assert got.tolist() == [[0, 1, 2]]


def test_mesh_params_rejects_nonpositive_size():
    with pytest.raises(ValueError):
        FeaMeshParams(mesh_size_m=0.0)


def test_build_blade_fea_mesh_produces_tagged_mesh():
    res = build_blade_fea_mesh(_gentle_design(), FeaMeshParams(mesh_size_m=0.004))
    assert res.tets.shape[1] == 4 and len(res.tets) > 0
    assert res.nodes.shape[1] == 3 and len(res.nodes) > 0
    assert len(res.aero_top_facets) > 0 and len(res.aero_bottom_facets) > 0
    assert len(res.support_node_ids) > 0


def test_build_blade_fea_mesh_tet_indices_in_range():
    res = build_blade_fea_mesh(_gentle_design(), FeaMeshParams(mesh_size_m=0.004))
    assert res.tets.min() >= 0 and res.tets.max() < len(res.nodes)
