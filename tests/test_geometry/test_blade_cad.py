"""Tests for fanopt.geometry.blade_cad (CadQuery blade solid + swept-volume fold gate).

Skipped at module load when CadQuery isn't installed, per CLAUDE.md §4.1.
"""

from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("cadquery") is None:
    pytest.skip("cadquery not installed", allow_module_level=True)

from fanopt.geometry.blade import (
    BladeParams,
    containment_margin_m,
    estimate_mass_kg,
)
from fanopt.geometry.blade_cad import (
    blade_mass_kg,
    blade_trimesh,
    blade_volume_m3,
    fold_collision_clear,
    fold_collision_volume_m3,
    make_blade_solid,
)

_SAMPLE_GRID = (
    (0.0003, 0.0005, 0.0003),
    (0.0004, 0.0006, 0.0004),
    (0.0005, 0.0007, 0.0005),
    (0.0006, 0.0008, 0.0006),
)


def _sample(blade_count: int = 8) -> BladeParams:
    return BladeParams(
        blade_count=blade_count,
        rib_bow_knots_m=(0.005, 0.010, 0.013, 0.017, 0.020),
        rib_bow_interp="linear",
        t_rib_hub_m=0.0025,
        t_rib_tip_m=0.0035,
        panel_offsets_m=_SAMPLE_GRID,
        panel_thickness_nom_m=0.0013,
    )


def _contains_violating() -> BladeParams:
    """Panel offsets poke past the thin rib envelope (containment violated)."""
    big = tuple((0.0024, 0.0024, 0.0024) for _ in range(4))
    return BladeParams(**{**_sample().to_dict(), "panel_offsets_m": big})


def test_solid_is_valid():
    assert make_blade_solid(_sample()).val().isValid() is True


def test_trimesh_shapes_and_indices_valid():
    V, F = blade_trimesh(_sample())
    assert V.ndim == 2 and V.shape[1] == 3 and V.shape[0] > 0
    assert F.ndim == 2 and F.shape[1] == 3 and F.shape[0] > 0
    assert int(F.max()) < V.shape[0] and int(F.min()) >= 0  # every face indexes a real vertex


def test_trimesh_finer_tol_more_triangles():
    _, coarse = blade_trimesh(_sample(), tol=0.002)
    _, fine = blade_trimesh(_sample(), tol=0.0002)
    assert fine.shape[0] >= coarse.shape[0]


def test_solid_is_single_body():
    assert len(make_blade_solid(_sample()).val().Solids()) == 1


def test_volume_positive_and_sane():
    # A single blade of this fan is a few cm³ — bounded well away from 0 and 1 cm³.
    vol = blade_volume_m3(_sample())
    assert 1e-6 < vol < 5e-5


def test_cad_mass_matches_analytic_proxy():
    # The coarse analytic proxy should be in the right ballpark of the real solid mass.
    cad = blade_mass_kg(_sample())
    analytic = estimate_mass_kg(_sample())
    assert cad == pytest.approx(analytic, rel=0.4)


def test_cad_mass_scales_with_blade_count():
    assert blade_mass_kg(_sample(12)) > blade_mass_kg(_sample(8))


def test_feasible_design_folds_clear():
    assert fold_collision_clear(_sample()) is True


def test_feasible_design_zero_collision_volume():
    assert fold_collision_volume_m3(_sample()) == pytest.approx(0.0, abs=1e-12)


def test_containment_violation_collides_when_folded():
    # The swept-volume CAD gate independently confirms the analytic containment
    # constraint: a panel poking past the rib actually collides with its neighbour.
    bad = _contains_violating()
    assert containment_margin_m(bad) < 0.0  # analytic says infeasible
    assert fold_collision_clear(bad) is False  # real geometry agrees


def test_containment_violation_positive_collision_volume():
    assert fold_collision_volume_m3(_contains_violating()) > 0.0
