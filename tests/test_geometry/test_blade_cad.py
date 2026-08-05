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
    FOLD_CLEARANCE_M,
    RIB_BOW_RANGE_M,
    RIB_TIP_RADIUS_M,
    BladeParams,
    containment_margin_m,
    estimate_mass_kg,
    half_width_at,
)
from fanopt.geometry.blade_cad import (
    _panel_window,
    blade_mass_kg,
    blade_trimesh,
    blade_volume_m3,
    boss_trimesh,
    carved_blade_with_boss,
    export_blade_step,
    fold_collision_clear,
    fold_collision_volume_m3,
    fold_penetration_m,
    make_blade_solid,
)
import fanopt.geometry.blade_cad as blade_cad_mod
from fanopt.geometry.schema import PIVOT_BOSS_RADIUS_M
from fanopt.geometry.to_stl import carved_blade_mesh


def test_n_radial_sections_default_is_the_campaign_lock():
    # ADR-0003 locks the geometry resolution to 40. The module DEFAULT must equal it so a run that
    # doesn't set it at runtime (or a spawn-based pool worker that re-imports fresh) can't silently
    # mesh at a resolution the campaign objective never used — which would make results non-comparable.
    assert blade_cad_mod.N_RADIAL_SECTIONS == 40

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
    # A full-amplitude base→tip ramp to the +20 mm ceiling folds (no amplitude cap on the meridian).
    p = _meridian_blade((0.004, 0.008, 0.012, 0.016, _MAX_BOW))
    assert fold_collision_clear(p) is True


def test_uniform_zigzag_meridian_folds_clear():
    # The no-rib sheet (design B) nests a full-amplitude zigzag meridian the same way design A does.
    p = _meridian_blade((_MAX_BOW, 0.0, _MAX_BOW, 0.0, _MAX_BOW), uniform=True)
    assert fold_collision_clear(p) is True


_MIN_BOW: float = RIB_BOW_RANGE_M[0]  # −20 mm — the bottom of the bipolar meridian range


def test_bipolar_multisign_meridian_folds_clear():
    # Bipolar range (2026-07-30): a multi-sign wave that dips below the base plane then rises
    # (−20, +10, −15, +12, +20 mm) folds — a surface of revolution nests any sign/shape.
    p = _meridian_blade((_MIN_BOW, 0.010, -0.015, 0.012, _MAX_BOW))
    assert fold_collision_clear(p) is True


def test_bipolar_deep_scoop_folds_clear():
    # A full-amplitude down-cup (all knots at −20 mm) — impossible in the retired up-only range.
    p = _meridian_blade((_MIN_BOW, _MIN_BOW, _MIN_BOW, _MIN_BOW, _MIN_BOW))
    assert fold_collision_clear(p) is True


def test_bipolar_zigzag_meridian_folds_clear():
    # The steep bipolar alternating zigzag (+20,−20,+20,−20,+20) — the worst faceting driver on the
    # relaxed range — still folds clear at the 90×27 default mesh (measured 1.1 mm³ ≪ 5 mm³ gate).
    p = _meridian_blade((_MAX_BOW, _MIN_BOW, _MAX_BOW, _MIN_BOW, _MAX_BOW))
    assert fold_collision_clear(p) is True


# --- Analytic fold gate (2026-07-30): fold_collision_clear evaluates the SMOOTH surface-of-
# revolution height field directly (fold_penetration_m) instead of the CAD swept-volume boolean.
# It is ~ms (not seconds), faceting-free (exact), and CANNOT hang — OCP's boolean hangs indefinitely
# on a valid-but-pathological solid (steep bipolar zigzag + checkerboard panel offsets) that is
# decode-reachable and actually FOLDS. The analytic gate gives it a correct verdict, restoring
# feasible-by-construction. The CAD boolean stays as an offline deep-verify cross-check. ---


def _checkerboard_design() -> BladeParams:
    """A steep bipolar-zigzag meridian + checkerboard (adjacent opposite-sign) max panel offsets +
    thin panel, ribbed — the design class that HANGS OCP's boolean. It is decode-reachable and (per
    the analytic gate) actually folds; the CAD boolean simply cannot compute it."""
    idx = {v.name: i for i, v in enumerate(SEARCH_SPACE)}
    lo, hi = RIB_BOW_RANGE_M
    vec = np.zeros(len(SEARCH_SPACE))
    vec[idx["rib_mode"]] = 0.5  # ribbed
    vec[idx["rib_bow_interp"]] = 0.5  # linear
    for i, s in enumerate((hi, lo, hi, lo, hi)):  # steep bipolar zigzag
        vec[idx[f"rib_bow_k{i}"]] = s
    for i in range(4):
        for j in range(3):
            vec[idx[f"panel_z_{i}_{j}"]] = 1.0 if (i + j) % 2 == 0 else -1.0  # checkerboard
            vec[idx[f"panel_thick_{i}_{j}"]] = 0.0  # thin panel → widest offset swing
    return decode(vec)


def test_analytic_gate_good_design_gap_is_the_fold_clearance():
    # A flat contained design nests with a gap of exactly one fold clearance: penetration = −0.4 mm.
    assert fold_penetration_m(_sample()) == pytest.approx(-FOLD_CLEARANCE_M, abs=5e-5)


def test_analytic_gate_is_fast_and_never_hangs_on_checkerboard():
    # The design that hangs OCP's boolean is resolved by the analytic gate in ~ms — and it FOLDS.
    p = _checkerboard_design()
    assert fold_collision_clear(p) is True  # foldable — the boolean's hang was not a real collision


def test_analytic_gate_catches_rail_lift_collision(monkeypatch):
    # Positive control: disabling the rib-rail window recreates the rail-lift bug (rails ride the
    # panel mean above the containment envelope). The analytic gate must report a real penetration.
    p = _max_offset_ribbed(1.0)
    monkeypatch.setattr(blade_cad_mod, "_panel_window", lambda params, r, v: 1.0)
    assert fold_penetration_m(p) > 0.0  # interpenetration detected
    assert fold_collision_clear(p) is False


def test_analytic_penetration_negative_when_clear():
    assert fold_penetration_m(_sample()) < 0.0  # a gap, not an overlap


def test_tighter_clearance_still_nests():
    # A reduced fold clearance (tighter deck to shrink the deployed gap) must still fold: the nesting
    # gap tracks the clearance, so penetration = -clearance, still <= 0.
    p = _sample()
    assert fold_penetration_m(p, clearance_m=0.2e-3) == pytest.approx(-0.2e-3, abs=5e-5)
    assert fold_penetration_m(p, clearance_m=0.2e-3) < 0.0


def test_analytic_gate_handles_zero_swing_steps():
    # Guard: n_swing_steps=0 (folded pose only) must not divide by zero.
    assert fold_penetration_m(_sample(), n_swing_steps=0) == pytest.approx(-FOLD_CLEARANCE_M, abs=5e-5)


def _synthetic_puck() -> tuple[np.ndarray, np.ndarray]:
    """A solid disk of unit-density centroids (radius 15 mm, 3 z-layers) — a stand-in TO cloud whose
    hub column the boss-fusion voids and rebuilds. Not a real blade; just exercises the fuse plumbing."""
    xs = np.arange(-0.015, 0.0151, 0.002)
    grid = np.stack(np.meshgrid(xs, xs, np.array([-0.002, 0.0, 0.002]), indexing="ij"), axis=-1)
    pts = grid.reshape(-1, 3)
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) <= 0.015]
    return np.ones(len(pts)), pts


def test_boss_trimesh_is_a_valid_indexed_mesh():
    v, f = boss_trimesh(_sample())
    assert v.ndim == 2 and v.shape[1] == 3 and len(v) > 0
    assert int(f.max()) < len(v) and int(f.min()) >= 0


def test_boss_trimesh_clearance_shortens_the_boss():
    # A tighter fold clearance builds a shorter boss (the fold pitch), so the tessellated z-extent drops
    # by exactly the clearance reduction — this is what pulls the deployed deck (and its gap) together.
    tall = boss_trimesh(_sample())[0][:, 2]
    short = boss_trimesh(_sample(), clearance_m=0.2e-3)[0][:, 2]
    dz = (tall.max() - tall.min()) - (short.max() - short.min())
    assert dz == pytest.approx(FOLD_CLEARANCE_M - 0.2e-3, abs=3e-5)


def test_carved_blade_with_boss_is_a_valid_fused_mesh():
    dens, pts = _synthetic_puck()
    v, f = carved_blade_with_boss(dens, pts, _sample(), voxel_pitch_m=0.002, clearance_m=0.3e-3)
    assert v.ndim == 2 and v.shape[1] == 3 and len(v) > 0
    assert int(f.max()) < len(v) and int(f.min()) >= 0  # face indices valid after the concat offset


def test_carved_blade_with_boss_adds_the_cad_boss_body():
    # Fusing must add vertices beyond the holed dish alone: the CAD boss is a second body in the mesh.
    dens, pts = _synthetic_puck()
    voided = dens.copy()
    voided[np.hypot(pts[:, 0], pts[:, 1]) < PIVOT_BOSS_RADIUS_M] = 0.0
    dish_v, _ = carved_blade_mesh(voided, pts, voxel_pitch_m=0.002)
    fused_v, _ = carved_blade_with_boss(dens, pts, _sample(), voxel_pitch_m=0.002)
    assert len(fused_v) > len(dish_v)


def test_carved_blade_with_boss_voids_inside_the_boss_od_for_overlap():
    # M3 fix: the hub is voided a hair INSIDE the boss OD (PIVOT_BOSS_RADIUS_M − overlap), so the CAD boss
    # (radius = full OD) overlaps the retained dish by a solid ring instead of abutting it on a coincident
    # cylinder. Verify the void radius is strictly inside the OD and the [OD−overlap, OD) ring is RETAINED.
    assert 0.0 < blade_cad_mod._BOSS_FUSE_OVERLAP_M < PIVOT_BOSS_RADIUS_M
    dens, pts = _synthetic_puck()
    r = np.hypot(pts[:, 0], pts[:, 1])
    void_mask = r < PIVOT_BOSS_RADIUS_M - blade_cad_mod._BOSS_FUSE_OVERLAP_M
    ring = (r >= PIVOT_BOSS_RADIUS_M - blade_cad_mod._BOSS_FUSE_OVERLAP_M) & (r < PIVOT_BOSS_RADIUS_M)
    assert ring.any()  # the fixture actually has centroids in the overlap ring this guards
    assert not void_mask[ring].any()  # ring centroids are NOT voided → dish keeps material to bond to


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


# --- Rib-rail window fold fix (2026-07-29, part 2): the panel mean displacement is windowed to
# ZERO across the rib-rail band, so the thick rail sits on the pure meridian (surface of revolution)
# instead of riding up on the panel wave. Pre-fix, a max-panel-offset ribbed design lifted the rail
# ~0.8 mm above the ±t_rib/2 containment envelope → a real ~7 mm³ fold collision that GREW under
# mesh refinement and passed the loose 10 mm³ gate (a false-negative). ---


def _max_offset_ribbed(sign: float = 1.0) -> BladeParams:
    """Ribbed design at the containment boundary: THICK ribs (so the offset envelope is real) with
    every panel offset knob at ±1 (max displacement), thin 3 mm panel, flat meridian. Decoded so it
    is exactly BO-reachable. This is the design class whose rib rails used to ride up on the panel
    wave and collide when folded — thick ribs are essential, since the offset allowance is
    ``(t_rib − panel)/2`` (thin ribs would leave zero offset room and no rail-lift to exercise)."""
    idx = {v.name: i for i, v in enumerate(SEARCH_SPACE)}
    vec = np.zeros(len(SEARCH_SPACE))
    vec[idx["rib_mode"]] = 0.5  # ribbed
    vec[idx["rib_bow_interp"]] = 0.5  # linear, flat meridian (all knots 0)
    vec[idx["t_rib_hub_k"]] = 1.0  # thick ribs → real containable offset envelope
    vec[idx["t_rib_tip_k"]] = 1.0
    for i in range(4):
        for j in range(3):
            vec[idx[f"panel_z_{i}_{j}"]] = sign  # max |offset| — the containment boundary
            vec[idx[f"panel_thick_{i}_{j}"]] = 0.0  # thin 3 mm panel → deepest rail/panel step
    return decode(vec)


def test_max_offset_ribbed_positive_folds_clear():
    p = _max_offset_ribbed(1.0)
    assert not p.uniform
    assert fold_collision_clear(p) is True


def test_max_offset_ribbed_negative_folds_clear():
    assert fold_collision_clear(_max_offset_ribbed(-1.0)) is True


def test_max_offset_ribbed_penetration_is_a_clearance_gap():
    # The former ~7 mm³ real collision is gone: the analytic gate shows the rails now nest with a
    # gap (negative penetration), because the window keeps them on the meridian.
    assert fold_penetration_m(_max_offset_ribbed(1.0)) < 0.0


@pytest.mark.slow
def test_max_offset_ribbed_cad_boolean_agrees_it_folds():
    # Offline CAD-boolean cross-check on this key design agrees with the analytic gate: it folds.
    assert fold_collision_volume_m3(_max_offset_ribbed(1.0)) == pytest.approx(0.0, abs=1e-8)


def test_rib_rail_carries_no_panel_displacement():
    # Unit invariant behind the fix: inside the rib-rail band the panel window is 0 (rail rides the
    # pure meridian), and it reaches 1 in the deep interior (panel free to wave).
    p = _max_offset_ribbed(1.0)
    assert _panel_window(p, 0.11, 1.0) == 0.0  # trapezoid edge → rail → no displacement
    assert _panel_window(p, 0.11, 0.0) == pytest.approx(1.0)  # centreline → full panel wave


def test_uniform_window_is_one_everywhere():
    # No-rib sheet has no rails, so the panel wave is never windowed (it waves to its edges).
    p = _uniform_sample()
    assert _panel_window(p, 0.11, 1.0) == 1.0
    assert _panel_window(p, 0.11, 0.0) == 1.0
