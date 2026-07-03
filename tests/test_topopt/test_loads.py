"""Tests for the rib TO problem setup (report-final.md §3.1.2a / §9.2)."""

from __future__ import annotations

import numpy as np

from fanopt.topopt.loads import (
    DEFAULT_VOLFRAC,
    LoadCase,
    build_rib_problem,
)


def test_build_rib_problem_mesh_from_locked_geometry():
    p = build_rib_problem(elem_size_m=0.001)
    # L_rib = 165 mm, tip width 6 mm -> 165 x 6 elements at 1 mm.
    assert p.mesh.nelx == 165
    assert p.mesh.nely == 6


def test_active_mask_tapers_root_to_tip():
    p = build_rib_problem(elem_size_m=0.001)
    root_width = int(p.active[:, 0].sum())
    tip_width = int(p.active[:, -1].sum())
    assert root_width < tip_width  # H12 up-taper (4 mm root -> 6 mm tip)


def test_preserved_is_one_element_per_active_column():
    p = build_rib_problem(elem_size_m=0.001)
    cols_with_active = np.where(p.active.any(axis=0))[0]
    per_col = p.preserved.sum(axis=0)[cols_with_active]
    assert np.all(per_col == 1)


def test_free_is_active_minus_preserved():
    p = build_rib_problem(elem_size_m=0.001)
    assert np.array_equal(p.free, p.active & ~p.preserved)


def test_fixed_dofs_are_three_per_root_node():
    p = build_rib_problem(elem_size_m=0.001)
    assert p.fixed_dofs.size % 3 == 0
    assert p.fixed_dofs.size > 0


def test_two_load_cases_productive_and_return():
    p = build_rib_problem(elem_size_m=0.002)
    names = [lc.name for lc in p.load_cases]
    assert names == ["productive", "return"]


def test_return_is_negated_productive():
    p = build_rib_problem(elem_size_m=0.002)
    prod, ret = p.load_cases[0], p.load_cases[1]
    assert np.allclose(ret.forces, -prod.forces)


def test_productive_pressure_gives_positive_w_loads():
    p = build_rib_problem(elem_size_m=0.002, pressure_pa=10.0)
    # w-DOFs are indices 0, 3, 6, ...; a positive pressure loads them positively.
    w_loads = p.load_cases[0].forces[0::3]
    assert w_loads.max() > 0.0
    assert w_loads.min() >= 0.0


def test_default_volfrac():
    p = build_rib_problem(elem_size_m=0.002)
    assert p.volfrac == DEFAULT_VOLFRAC


def test_load_case_default_weight():
    lc = LoadCase("x", np.zeros(3))
    assert lc.weight == 0.5
