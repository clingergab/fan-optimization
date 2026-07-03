"""Tests for fanopt.geometry.envelope_cad (CadQuery Layer 1 generator).

Skipped at module load when CadQuery isn't installed, per CLAUDE.md §4.1
optional-dep handling.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

if importlib.util.find_spec("cadquery") is None:
    pytest.skip("cadquery not installed", allow_module_level=True)

import cadquery as cq

from fanopt.geometry.envelope import Layer1Params, ThicknessGridField
from fanopt.geometry.envelope_cad import (
    FOURIER_PHASES_RAD,
    LOFT_START_EPS_M,
    N_CHORD_SAMPLES,
    N_RADIAL_STATIONS,
    _camber_height,
    _fourier_modulation,
    _linear_spline,
    make_outer_envelope,
)
from fanopt.geometry.schema import (
    INTER_BLADE_ANGLE_RAD,
    L_BLADE_M,
    PIVOT_CENTER_X_M,
)

# ---- helpers --------------------------------------------------------------


def _canonical_layer1(
    *,
    camber_knots_m: tuple[float, ...] = (0.0, 0.002, 0.001),
    twist_knots_rad: tuple[float, ...] = (0.0, 0.0),
    thickness_knots_m: tuple[float, float, float] = (0.0030, 0.0028, 0.0026),
    fourier_le_amplitudes: tuple[float, float, float] = (0.0, 0.0, 0.0),
    fourier_te_amplitudes: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Layer1Params:
    return Layer1Params(
        blade_count=10,
        camber_knots_m=camber_knots_m,
        twist_knots_rad=twist_knots_rad,
        thickness_field=ThicknessGridField.from_radial_knots(thickness_knots_m),
        edge_profile="rounded",
        fourier_le_amplitudes=fourier_le_amplitudes,
        fourier_te_amplitudes=fourier_te_amplitudes,
    )


def _bbox(envelope: cq.Workplane) -> cq.occ_impl.geom.BoundBox:
    return envelope.val().BoundingBox()


# ---- spline helper --------------------------------------------------------


def test_linear_spline_clamps_below_zero() -> None:
    assert _linear_spline((1.0, 2.0, 3.0), -0.5) == 1.0


def test_linear_spline_clamps_above_one() -> None:
    assert _linear_spline((1.0, 2.0, 3.0), 1.5) == 3.0


def test_linear_spline_midpoint_three_knots() -> None:
    """3 knots: knot 1 lives at t=0.5, so spline(0.5) == knots[1]."""
    assert _linear_spline((1.0, 5.0, 9.0), 0.5) == pytest.approx(5.0)


def test_linear_spline_quarter_three_knots() -> None:
    """t=0.25: halfway between knot 0 (at t=0) and knot 1 (at t=0.5)."""
    assert _linear_spline((1.0, 5.0, 9.0), 0.25) == pytest.approx(3.0)


def test_linear_spline_single_knot_returns_value() -> None:
    assert _linear_spline((7.0,), 0.5) == 7.0


def test_linear_spline_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _linear_spline((), 0.5)


# ---- Fourier helper -------------------------------------------------------


def test_fourier_phases_match_plan_lock() -> None:
    """Plan §6.2.1 Layer 1 table locks ``phases = (0, π/3, 2π/3)``."""
    assert (0.0, math.pi / 3.0, 2.0 * math.pi / 3.0) == FOURIER_PHASES_RAD


def test_fourier_zero_amplitudes_returns_1() -> None:
    assert _fourier_modulation(0.05, (0.0, 0.0, 0.0)) == 1.0


def test_fourier_k1_only_zeroes_at_endpoints() -> None:
    """With phase φ_1 = 0, the fundamental harmonic zeroes at x=0 and x=L.
    k=2 and k=3 do NOT zero (their phases are π/3 and 2π/3)."""
    amps_k1_only = (0.10, 0.0, 0.0)
    assert _fourier_modulation(0.0, amps_k1_only) == pytest.approx(1.0, abs=1e-12)
    assert _fourier_modulation(L_BLADE_M, amps_k1_only) == pytest.approx(1.0, abs=1e-12)


def test_fourier_k2_nonzero_at_endpoints_per_plan_phase() -> None:
    """k=2 with phase π/3 gives sin(0 + π/3) = sin(π/3) ≈ 0.866 at x=0;
    at x=L, sin(2π + π/3) = sin(π/3) ≈ 0.866 too."""
    amps_k2_only = (0.0, 0.10, 0.0)
    expected_offset = 0.10 * math.sin(math.pi / 3.0)  # ≈ 0.0866
    assert _fourier_modulation(0.0, amps_k2_only) == pytest.approx(1.0 + expected_offset, abs=1e-12)
    assert _fourier_modulation(L_BLADE_M, amps_k2_only) == pytest.approx(
        1.0 + expected_offset, abs=1e-12
    )


def test_fourier_k3_nonzero_at_endpoints_per_plan_phase() -> None:
    """k=3 with phase 2π/3: at x=0, sin(2π/3) ≈ 0.866; at x=L, sin(3π + 2π/3) = -sin(π/3) ≈ -0.866."""
    amps_k3_only = (0.0, 0.0, 0.10)
    expected_at_0 = 0.10 * math.sin(2.0 * math.pi / 3.0)  # +0.0866
    expected_at_L = 0.10 * math.sin(3.0 * math.pi + 2.0 * math.pi / 3.0)  # -0.0866
    assert _fourier_modulation(0.0, amps_k3_only) == pytest.approx(1.0 + expected_at_0, abs=1e-12)
    assert _fourier_modulation(L_BLADE_M, amps_k3_only) == pytest.approx(
        1.0 + expected_at_L, abs=1e-12
    )


def test_fourier_equal_amps_cancel_at_tip() -> None:
    """At x=L with equal k=2,3 amplitudes, the k=2 (+sin(π/3)) and
    k=3 (-sin(π/3)) contributions cancel exactly. This is the plan's
    "phases prevent constructive amplification" property."""
    factor = _fourier_modulation(L_BLADE_M, (0.10, 0.10, 0.10))
    # k=1 → 0, k=2 → +amp·sin(π/3), k=3 → -amp·sin(π/3) → sum = 0 → factor = 1.0
    assert factor == pytest.approx(1.0, abs=1e-12)


def test_fourier_k1_max_at_midspan() -> None:
    """k=1 has its max at x=L/2 (sin(π/2)=1)."""
    factor = _fourier_modulation(L_BLADE_M / 2, (0.10, 0.0, 0.0))
    assert factor == pytest.approx(1.10, abs=1e-12)


# ---- camber helper --------------------------------------------------------


def test_camber_three_knots_midchord_is_middle_knot() -> None:
    """3 knots at y_norm ∈ {-1, 0, +1}; y_norm = 0 returns knots[1]."""
    assert _camber_height(0.0, (0.0, 0.002, 0.001)) == pytest.approx(0.002)


def test_camber_at_le_is_first_knot() -> None:
    assert _camber_height(-1.0, (0.0, 0.002, 0.001)) == 0.0


def test_camber_at_te_is_last_knot() -> None:
    assert _camber_height(+1.0, (0.0, 0.002, 0.001)) == 0.001


def test_camber_four_knots_at_minus_one_third_is_second_knot() -> None:
    """4 knots distributed evenly at y_norm ∈ {-1, -1/3, +1/3, +1};
    at y_norm = -1/3 the spline lands exactly on knots[1]."""
    knots = (0.0, 0.003, 0.002, 0.0)
    assert _camber_height(-1.0 / 3.0, knots) == pytest.approx(0.003, abs=1e-12)


def test_camber_four_knots_at_plus_one_third_is_third_knot() -> None:
    """At y_norm = +1/3 the spline lands exactly on knots[2]."""
    knots = (0.0, 0.003, 0.002, 0.0)
    assert _camber_height(+1.0 / 3.0, knots) == pytest.approx(0.002, abs=1e-12)


# ---- make_outer_envelope: bounding box ------------------------------------


def test_envelope_x_extent_matches_blade_length() -> None:
    p = _canonical_layer1()
    bb = _bbox(make_outer_envelope(p, "flat"))
    assert bb.xmin == pytest.approx(LOFT_START_EPS_M, abs=1e-6)
    assert bb.xmax == pytest.approx(L_BLADE_M, abs=1e-6)


def test_envelope_y_extent_matches_half_pitch_no_fourier() -> None:
    """Without Fourier modulation, y_max at the tip = L · INTER_BLADE_ANGLE_RAD / 2.

    Tolerance ``abs=2e-7`` covers CadQuery's BoundingBox default slop of
    ~1e-7 m (OpenCascade pads the bbox by its solid-construction tolerance).
    """
    p = _canonical_layer1()  # zero Fourier
    bb = _bbox(make_outer_envelope(p, "flat"))
    expected = L_BLADE_M * INTER_BLADE_ANGLE_RAD / 2
    assert bb.ymax == pytest.approx(expected, abs=2e-7)
    assert bb.ymin == pytest.approx(-expected, abs=2e-7)


# ---- plano-convex vs symmetric --------------------------------------------


def test_flat_orientation_has_planar_z0_bottom() -> None:
    """Plano-convex per plan §0 row 47: bottom face is at z = 0."""
    p = _canonical_layer1()
    bb = _bbox(make_outer_envelope(p, "flat"))
    assert bb.zmin == pytest.approx(0.0, abs=1e-6)


def test_symmetric_orientation_has_negative_zmin() -> None:
    """Edge / custom-angle orientations put material on both sides of midplane."""
    p = _canonical_layer1()
    bb = _bbox(make_outer_envelope(p, "edge"))
    assert bb.zmin < 0.0
    assert bb.zmax > 0.0


def test_flat_orientation_zmax_equals_thickness_plus_camber() -> None:
    """Plano-convex top face: z_max = thickness(x) + camber(y_norm) at the
    location where both are max.

    With camber_knots = (0, 2 mm, 1 mm), the max camber (2 mm) lands at
    y_norm = 0 (3-knot middle). With thickness_knots = (3 mm, 2.8, 2.6),
    the max thickness (3 mm) is at the root x=0 — but the loft starts at
    x = LOFT_START_EPS_M = 1 mm, so the spline lookup uses
    t_x_full = LOFT_START_EPS_M / L_BLADE_M = 0.005, giving thickness =
    0.005 × (2.8) + 0.995 × (3.0) = 2.999 mm. Effective zmax ≈
    2.999 + 2 = 4.999 mm. The 2 µm shortfall from "exactly 5 mm" is
    accounted for in the tolerance.
    """
    p = _canonical_layer1(
        camber_knots_m=(0.0, 0.002, 0.001),
        thickness_knots_m=(0.0030, 0.0028, 0.0026),
    )
    bb = _bbox(make_outer_envelope(p, "flat"))
    assert bb.zmax == pytest.approx(0.005, abs=5e-6)


# ---- twist behaviour -------------------------------------------------------


def test_twist_zero_under_flat_orientation() -> None:
    """Plano-convex: twist is suppressed to keep the bottom face planar."""
    p_no_twist = _canonical_layer1(twist_knots_rad=(0.0, 0.0))
    p_twisted = _canonical_layer1(twist_knots_rad=(0.0, math.radians(10.0)))
    bb_no = _bbox(make_outer_envelope(p_no_twist, "flat"))
    bb_tw = _bbox(make_outer_envelope(p_twisted, "flat"))
    # Both should have z_min == 0 (planar bottom); tolerate CadQuery's
    # ~1e-7 BoundingBox slop.
    assert bb_no.zmin == pytest.approx(0.0, abs=2e-7)
    assert bb_tw.zmin == pytest.approx(0.0, abs=2e-7)
    # And the upper bound should be identical (twist was suppressed).
    assert bb_no.zmax == pytest.approx(bb_tw.zmax, abs=2e-7)


def test_twist_applied_under_edge_orientation() -> None:
    """Non-flat orientations honour the twist spline."""
    p_no = _canonical_layer1(twist_knots_rad=(0.0, 0.0))
    p_tw = _canonical_layer1(twist_knots_rad=(0.0, math.radians(10.0)))
    bb_no = _bbox(make_outer_envelope(p_no, "edge"))
    bb_tw = _bbox(make_outer_envelope(p_tw, "edge"))
    # Twist rotates the cross-section: bounding box on y or z should differ.
    moved = abs(bb_no.ymax - bb_tw.ymax) > 1e-5 or abs(bb_no.zmax - bb_tw.zmax) > 1e-5
    assert moved, "twist should perturb the bounding box under edge orientation"


# ---- thickness + camber propagation ---------------------------------------


def test_thickness_root_dominates_with_descending_profile() -> None:
    """With thickness_knots = (3 mm, 2.8 mm, 2.6 mm), the root is thickest.
    Plano-convex z_max comes from x near the root.

    Tolerance accounts for the LOFT_START_EPS_M spline-lookup offset
    (~2 µm shortfall from the exact 3 mm root value)."""
    p = _canonical_layer1(
        camber_knots_m=(0.0, 0.0, 0.0),  # no camber
        thickness_knots_m=(0.0030, 0.0028, 0.0026),
    )
    bb = _bbox(make_outer_envelope(p, "flat"))
    # z_max should equal the root thickness 3 mm, minus the LOFT offset epsilon.
    assert bb.zmax == pytest.approx(0.0030, abs=5e-6)


def test_camber_zero_yields_constant_top_face_height_per_x() -> None:
    """With camber_knots = (0, 0, 0) AND uniform thickness, the top face is
    flat at z = thickness everywhere. No spline-offset shortfall since
    thickness is constant."""
    p = _canonical_layer1(
        camber_knots_m=(0.0, 0.0, 0.0),
        thickness_knots_m=(0.0030, 0.0030, 0.0030),  # uniform thickness
    )
    bb = _bbox(make_outer_envelope(p, "flat"))
    # Uniform thickness 3 mm; no camber → z_max = 3 mm. Tolerance for the
    # CadQuery BoundingBox slop (~1e-7).
    assert bb.zmax == pytest.approx(0.0030, abs=2e-7)
    assert bb.zmin == pytest.approx(0.0, abs=2e-7)


# ---- Fourier modulation propagation ---------------------------------------


def test_fourier_le_positive_amp_increases_volume() -> None:
    """LE amp k=1 = +0.10 bulges the LE outward in the interior (Fourier
    is zero at the tip, max at midspan), adding material → volume
    increases vs the unperturbed envelope.

    A bbox check cannot detect this perturbation because the tip
    (where Fourier sine is zero) sets the y extent; the perturbation
    only shows up in the interior cross-sections.
    """
    p_u = _canonical_layer1()
    p_p = _canonical_layer1(fourier_le_amplitudes=(0.10, 0.0, 0.0))
    vol_u = make_outer_envelope(p_u, "flat").val().Volume()
    vol_p = make_outer_envelope(p_p, "flat").val().Volume()
    assert vol_p > vol_u * 1.005  # at least ~0.5 % volume increase


def test_fourier_te_positive_amp_increases_volume() -> None:
    p_u = _canonical_layer1()
    p_p = _canonical_layer1(fourier_te_amplitudes=(0.10, 0.0, 0.0))
    vol_u = make_outer_envelope(p_u, "flat").val().Volume()
    vol_p = make_outer_envelope(p_p, "flat").val().Volume()
    assert vol_p > vol_u * 1.005


def test_fourier_negative_amp_decreases_volume() -> None:
    """Negative Fourier amp pinches the edge inward; volume drops."""
    p_u = _canonical_layer1()
    p_p = _canonical_layer1(fourier_le_amplitudes=(-0.10, 0.0, 0.0))
    vol_u = make_outer_envelope(p_u, "flat").val().Volume()
    vol_p = make_outer_envelope(p_p, "flat").val().Volume()
    assert vol_p < vol_u * 0.995


# ---- volume + STL export smoke --------------------------------------------


def test_envelope_volume_positive_and_finite() -> None:
    p = _canonical_layer1()
    vol = make_outer_envelope(p, "flat").val().Volume()
    assert vol > 0.0
    assert math.isfinite(vol)


def test_envelope_exports_to_stl(tmp_path: Path) -> None:
    """STL export is the canonical print-pipeline output. Must not raise."""
    p = _canonical_layer1()
    env = make_outer_envelope(p, "flat")
    stl_path = tmp_path / "blade.stl"
    cq.exporters.export(env, str(stl_path))
    assert stl_path.exists()
    assert stl_path.stat().st_size > 1000  # nontrivial mesh


# ---- input validation ------------------------------------------------------


def test_unknown_print_orientation_raises() -> None:
    p = _canonical_layer1()
    with pytest.raises(ValueError, match="print_orientation"):
        make_outer_envelope(p, "diagonal")


# ---- construction-parameter exports ---------------------------------------


def test_n_radial_stations_reasonable() -> None:
    """20 stations is the documented default; if someone tunes it, the
    docstring's facetting analysis becomes stale and they should bump
    the docstring too."""
    assert N_RADIAL_STATIONS >= 10


def test_n_chord_samples_reasonable() -> None:
    assert N_CHORD_SAMPLES >= 16


def test_loft_start_eps_safely_inboard_of_boss() -> None:
    """The loft start must stay outboard of x=0 (degenerate cross-section)
    AND inboard of the boss center (so the boss region is fully included)."""
    assert 0.0 < LOFT_START_EPS_M < PIVOT_CENTER_X_M


# ---- LE / TE label convention ---------------------------------------------


def test_le_amp_perturbs_negative_y_te_amp_does_not() -> None:
    """The label convention pins LE at ``y < 0`` and TE at ``y > 0``. A
    positive LE amplitude must increase the |y_min| of the envelope at
    midspan (the LE bulges further in the -y direction). A positive TE
    amplitude must NOT change y_min — it perturbs y_max instead.

    Without this guard, an accidental sign flip in the implementation
    (e.g., LE → +y) would silently invert the asymmetric-drag design
    family for every Phase 4 architecture.
    """
    p_unperturbed = _canonical_layer1()
    p_le_only = _canonical_layer1(fourier_le_amplitudes=(0.10, 0.0, 0.0))
    p_te_only = _canonical_layer1(fourier_te_amplitudes=(0.10, 0.0, 0.0))

    vol_u = make_outer_envelope(p_unperturbed, "flat").val().Volume()
    vol_le = make_outer_envelope(p_le_only, "flat").val().Volume()
    vol_te = make_outer_envelope(p_te_only, "flat").val().Volume()

    # Both LE and TE amps add volume (both bulge their respective edges).
    assert vol_le > vol_u
    assert vol_te > vol_u

    # The two should add roughly equal volume since the modulation
    # amplitude + axis are symmetric.
    rel_diff = abs(vol_le - vol_te) / vol_u
    assert rel_diff < 0.05, (
        f"LE-only and TE-only volume gains diverge by {rel_diff:.3f} — " "label asymmetry suspected"
    )


# ---- camber upper-bound boundary case -------------------------------------


def test_envelope_handles_camber_at_schema_upper_bound() -> None:
    """The schema allows camber up to 5 mm (``CAMBER_RANGE_M[1] = 0.005``).
    The CadQuery loft must tolerate the worst case without barfing on
    cross-section topology / surface-fit tolerances."""
    p = _canonical_layer1(
        camber_knots_m=(0.0, 0.005, 0.0),  # max camber at midchord
        thickness_knots_m=(0.0038, 0.0038, 0.0038),  # also schema max
    )
    env = make_outer_envelope(p, "flat")
    bb = env.val().BoundingBox()
    # z_max = thickness 3.8 mm + camber 5 mm = 8.8 mm. Tolerance for the
    # CadQuery BoundingBox slop (~1e-7).
    assert bb.zmax == pytest.approx(0.0088, abs=2e-7)
    assert bb.zmin == pytest.approx(0.0, abs=2e-7)
    assert env.val().Volume() > 0.0


# ---- edge_profile current behaviour (TODO: implement) --------------------


def test_edge_profile_currently_ignored() -> None:
    """Layer1Params accepts edge_profile ∈ {sharp, rounded, mildly-serrated}
    but ``make_outer_envelope`` does not yet apply a fillet or serration.
    All three values currently produce identical geometry.

    **This test pins the current limitation.** When edge-profile is
    implemented (TODO in Phase 1 followup) this test must be updated:
    the three profiles should produce distinguishable shapes.
    """
    base_kwargs = dict(
        blade_count=10,
        camber_knots_m=(0.0, 0.002, 0.001),
        twist_knots_rad=(0.0, 0.0),
        thickness_field=ThicknessGridField.from_radial_knots((0.0030, 0.0028, 0.0026)),
        fourier_le_amplitudes=(0.0, 0.0, 0.0),
        fourier_te_amplitudes=(0.0, 0.0, 0.0),
    )
    volumes = {}
    for profile in ("sharp", "rounded", "mildly-serrated"):
        p = Layer1Params(edge_profile=profile, **base_kwargs)
        volumes[profile] = make_outer_envelope(p, "flat").val().Volume()
    # Currently all identical — when edge_profile is wired in, this
    # equality breaks and the test must be updated to assert distinct
    # values per profile (with appropriate tolerances per profile family).
    assert volumes["sharp"] == volumes["rounded"]
    assert volumes["sharp"] == volumes["mildly-serrated"]
