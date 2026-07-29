"""Tests for fanopt.geometry.blade_cad (CadQuery blade solid + swept-volume fold gate).

Skipped at module load when CadQuery isn't installed, per CLAUDE.md §4.1.
"""

from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("cadquery") is None:
    pytest.skip("cadquery not installed", allow_module_level=True)

from fanopt.geometry.blade import (
    RIB_TIP_RADIUS_M,
    BladeParams,
    containment_margin_m,
    estimate_mass_kg,
    half_width_at,
)
from fanopt.geometry.blade_cad import (
    blade_mass_kg,
    blade_trimesh,
    blade_volume_m3,
    export_blade_step,
    fold_collision_clear,
    fold_collision_volume_m3,
    make_blade_solid,
)
from fanopt.geometry.schema import PIVOT_BOSS_RADIUS_M

_SAMPLE_GRID = (
    (0.0003, 0.0005, 0.0003),
    (0.0004, 0.0006, 0.0004),
    (0.0005, 0.0007, 0.0005),
    (0.0006, 0.0008, 0.0006),
)


# Trapezoid (Cartesian) blades nest under a modest meridian bow; a large bow no longer nests
# the way the retired surface-of-revolution sector did (rotation-about-pin no longer maps the
# meridian onto itself), so the CAD samples use a gentle bow.
def _sample(blade_count: int = 8) -> BladeParams:
    return BladeParams(
        blade_count=blade_count,
        rib_bow_knots_m=(0.001, 0.002, 0.003, 0.0045, 0.006),
        rib_bow_interp="linear",
        t_rib_hub_m=0.005,
        t_rib_tip_m=0.006,
        panel_offsets_m=_SAMPLE_GRID,
        panel_thickness_m=tuple((0.003, 0.003, 0.003) for _ in range(4)),
    )


def _uniform_sample(blade_count: int = 8, tip_bow: float = 0.006) -> BladeParams:
    """A no-rib single-sheet blade (design B): a radially-cambered sheet (meridian bow), edges
    unpinned. Tangential offset waves on a bare sheet do NOT nest under the trapezoid fold
    rotation (unlike a ribbed panel, which folds because its waves hide inside the nesting rib
    slab), so a fold-safe no-rib blade carries its shape as radial camber, not tangential relief.
    """
    k = tuple(tip_bow * (i + 1) / 5 for i in range(5))
    return BladeParams(
        blade_count=blade_count,
        rib_bow_knots_m=k,
        rib_bow_interp="linear",
        t_rib_hub_m=0.0035,
        t_rib_tip_m=0.0035,
        panel_offsets_m=tuple((0.0, 0.0, 0.0) for _ in range(4)),
        panel_thickness_m=tuple((0.0035, 0.0035, 0.0035) for _ in range(4)),
        uniform=True,
    )


def test_export_blade_step_writes_a_file(tmp_path):
    fp = tmp_path / "blade.step"
    export_blade_step(_sample(), str(fp))
    assert fp.exists() and fp.stat().st_size > 0  # a real STEP file (for 3D rendering)


def _contains_violating() -> BladeParams:
    """Panel offsets poke past the thin rib envelope (containment violated)."""
    big = tuple((0.0024, 0.0024, 0.0024) for _ in range(4))
    return BladeParams(**{**_sample().to_dict(), "panel_offsets_m": big})


def test_solid_is_valid():
    assert make_blade_solid(_sample()).val().isValid() is True


@pytest.mark.slow
def test_trimesh_shapes_and_indices_valid():
    V, F = blade_trimesh(_sample())
    assert V.ndim == 2 and V.shape[1] == 3 and V.shape[0] > 0
    assert F.ndim == 2 and F.shape[1] == 3 and F.shape[0] > 0
    assert int(F.max()) < V.shape[0] and int(F.min()) >= 0  # every face indexes a real vertex


@pytest.mark.slow
def test_trimesh_finer_tol_more_triangles():
    _, coarse = blade_trimesh(_sample(), tol=0.002)
    _, fine = blade_trimesh(_sample(), tol=0.0002)
    assert fine.shape[0] >= coarse.shape[0]


def test_solid_is_single_body():
    assert len(make_blade_solid(_sample()).val().Solids()) == 1


def test_uniform_solid_is_valid_single_body():
    # Design B (no-rib sheet) also unions with the boss into one valid solid.
    solid = make_blade_solid(_uniform_sample()).val()
    assert solid.isValid() is True
    assert len(solid.Solids()) == 1


def test_trapezoid_dimensions():
    # The blade is a 220 mm-long trapezoid: 12 mm root (= boss dia) → 51 mm tip.
    bb = make_blade_solid(_sample()).val().BoundingBox()
    assert bb.xmax == pytest.approx(RIB_TIP_RADIUS_M, abs=1e-4)  # tip radius = 22 cm
    assert (bb.ymax - bb.ymin) == pytest.approx(2.0 * half_width_at(RIB_TIP_RADIUS_M), abs=1e-4)
    assert 2.0 * half_width_at(RIB_TIP_RADIUS_M) == pytest.approx(0.051, abs=1e-4)  # 51 mm tip
    assert 2.0 * PIVOT_BOSS_RADIUS_M == pytest.approx(0.012)  # 12 mm root = boss dia


def test_volume_positive_and_sane():
    # A single 22 cm trapezoid blade of this fan is tens of cm³ — bounded away from 0 and 1e-4 m³.
    vol = blade_volume_m3(_sample())
    assert 1e-6 < vol < 1e-4


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


def test_uniform_cambered_sheet_folds_clear():
    # B-proper: the swept-volume CAD gate confirms a radially-cambered no-rib sheet (edges
    # unpinned, no rib rails) nests through the fold swing.
    assert fold_collision_clear(_uniform_sample()) is True


def test_containment_violation_collides_when_folded():
    # The swept-volume CAD gate independently confirms the analytic containment
    # constraint: a panel poking past the rib actually collides with its neighbour.
    bad = _contains_violating()
    assert containment_margin_m(bad) < 0.0  # analytic says infeasible
    assert fold_collision_clear(bad) is False  # real geometry agrees


def test_containment_violation_positive_collision_volume():
    assert fold_collision_volume_m3(_contains_violating()) > 0.0
