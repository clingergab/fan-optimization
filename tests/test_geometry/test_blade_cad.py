"""Tests for fanopt.geometry.blade_cad (CadQuery blade solid + swept-volume fold gate).

Skipped at module load when CadQuery isn't installed, per CLAUDE.md §4.1.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("cadquery") is None:
    pytest.skip("cadquery not installed", allow_module_level=True)

from fanopt.bo.blade_codec import SEARCH_SPACE, decode
from fanopt.geometry.blade import (
    RIB_BOW_RANGE_M,
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


# The blade height is a surface of revolution (height = f(true radius √(x²+y²)), 2026-07-29), so
# rotated neighbours are congruent and a meridian bow of ANY shape/amplitude nests when folded —
# the CAD samples exercise that, not a gentle-bow workaround.
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
    unpinned. The meridian is a surface of revolution, so its radial camber nests through the
    fold swing for the bare sheet the same way it does for a ribbed blade.
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


def test_panel_overpoke_is_conservatively_infeasible_but_still_folds():
    # Surface-of-revolution + panel-aware layer spacing (2026-07-29): a panel that pokes past the
    # rib rail is now a *uniformly thicker* blade whose neighbour (one layer up, congruent under
    # rotation) nests with full clearance — it FOLDS. The analytic containment_margin stays a
    # conservative proxy (still flags it), but the authoritative CAD gate is the truth and says
    # clear. Pre-fix (Cartesian strip, rib-only spacing) this same design collided.
    bad = _contains_violating()
    assert containment_margin_m(bad) < 0.0  # analytic proxy: conservatively infeasible
    assert fold_collision_clear(bad) is True  # authoritative CAD gate: folds


def test_panel_overpoke_collision_volume_is_negligible():
    # The over-poke folds with only faceting-scale residual, far below the clear threshold.
    assert fold_collision_volume_m3(_contains_violating()) == pytest.approx(0.0, abs=1e-8)


# --- Surface-of-revolution fold fix (2026-07-29): height = f(true radius √(x²+y²)) makes rotated
# neighbours congruent, so a meridian of ANY shape — multi-hump, zigzag, base→tip wave, at full
# amplitude — nests when folded. The pre-fix Cartesian strip (z = f(x-station)) collided ~80 mm³
# on these; full design freedom (no monotonic / tip-loaded restriction) is the whole point. ---

_MAX_BOW: float = RIB_BOW_RANGE_M[1]  # 30 mm — the top of the meridian range


def _meridian_blade(knots, interp="linear", uniform=False):
    t = 0.0035 if uniform else 0.004
    return BladeParams(
        blade_count=12,
        rib_bow_knots_m=knots,
        rib_bow_interp=interp,
        t_rib_hub_m=t,
        t_rib_tip_m=t,
        panel_offsets_m=tuple((0.0, 0.0, 0.0) for _ in range(4)),
        panel_thickness_m=tuple((t, t, t) for _ in range(4)),
        uniform=uniform,
    )


def test_multihump_linear_meridian_folds_clear():
    # A base→tip multi-hump (0, 30, 0, 30, 0 mm) — the shape the pre-fix strip could not fold.
    p = _meridian_blade((0.0, _MAX_BOW, 0.0, _MAX_BOW, 0.0))
    assert fold_collision_clear(p) is True


def test_alternating_zigzag_meridian_folds_clear():
    # Steep-rooted alternating zigzag (30, 0, 30, 0, 30 mm) — the pre-fix worst case (~140 mm³).
    p = _meridian_blade((_MAX_BOW, 0.0, _MAX_BOW, 0.0, _MAX_BOW))
    assert fold_collision_clear(p) is True


def test_smooth_multihump_meridian_folds_clear():
    # Catmull-Rom (smooth) multi-hump, with its knot-hull overshoot, also nests.
    p = _meridian_blade((0.0, _MAX_BOW, 0.0, _MAX_BOW, 0.0), interp="smooth")
    assert fold_collision_clear(p) is True


def test_full_amplitude_monotonic_bow_folds_clear():
    # A full-amplitude base→tip ramp to the 30 mm ceiling folds (no amplitude cap on the meridian).
    p = _meridian_blade((0.006, 0.012, 0.018, 0.024, _MAX_BOW))
    assert fold_collision_clear(p) is True


def test_uniform_zigzag_meridian_folds_clear():
    # The no-rib sheet (design B) nests a full-amplitude zigzag meridian the same way design A does.
    p = _meridian_blade((_MAX_BOW, 0.0, _MAX_BOW, 0.0, _MAX_BOW), uniform=True)
    assert fold_collision_clear(p) is True


def _way2_checkerboard() -> BladeParams:
    """Way-2 design: independent top/bottom face waves (checkerboard offsets + varied thickness),
    within the rib thickness envelope by construction (built through the codec, which scales each
    offset to the local containable envelope)."""
    idx = {v.name: i for i, v in enumerate(SEARCH_SPACE)}
    vec = np.zeros(len(SEARCH_SPACE))
    vec[idx["rib_mode"]] = 0.5  # ribbed
    vec[idx["rib_bow_interp"]] = 0.5  # linear
    for i in range(5):
        vec[idx[f"rib_bow_k{i}"]] = 0.006
    vec[idx["t_rib_hub_k"]] = 1.0  # thick ribs → room for independent face relief
    vec[idx["t_rib_tip_k"]] = 1.0
    for i in range(4):
        for j in range(3):
            vec[idx[f"panel_z_{i}_{j}"]] = 0.9 if (i + j) % 2 == 0 else -0.9
            vec[idx[f"panel_thick_{i}_{j}"]] = 0.3 if (i + j) % 2 == 0 else 0.7
    return decode(vec)


def test_way2_independent_faces_folds_clear():
    # Way-2 (independent top/bottom face waves within the thickness envelope) nests by interlock +
    # containment — the root taper keeps the near-boss panel from climbing the next layer's boss.
    p = _way2_checkerboard()
    assert not p.uniform  # sanity: this is the ribbed independent-face family
    assert fold_collision_clear(p) is True
