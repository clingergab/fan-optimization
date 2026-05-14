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

from fanopt.geometry.envelope import Layer1Params
from fanopt.geometry.envelope_cad import (
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
        thickness_knots_m=thickness_knots_m,
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


def test_fourier_zero_amplitudes_returns_1() -> None:
    assert _fourier_modulation(0.05, (0.0, 0.0, 0.0)) == 1.0


def test_fourier_zero_at_endpoints() -> None:
    """sin(kπ·x/L) zeroes at x=0 and x=L for any integer k → factor = 1."""
    amps = (0.1, 0.05, 0.08)
    assert _fourier_modulation(0.0, amps) == pytest.approx(1.0)
    assert _fourier_modulation(L_BLADE_M, amps) == pytest.approx(1.0, abs=1e-12)


def test_fourier_k1_max_at_midspan() -> None:
    """k=1 has its max at x=L/2 (sin(π/2)=1)."""
    factor = _fourier_modulation(L_BLADE_M / 2, (0.10, 0.0, 0.0))
    assert factor == pytest.approx(1.10, abs=1e-9)


# ---- camber helper --------------------------------------------------------


def test_camber_three_knots_midchord_is_middle_knot() -> None:
    """3 knots at y_norm ∈ {-1, 0, +1}; y_norm = 0 returns knots[1]."""
    assert _camber_height(0.0, (0.0, 0.002, 0.001)) == pytest.approx(0.002)


def test_camber_at_le_is_first_knot() -> None:
    assert _camber_height(-1.0, (0.0, 0.002, 0.001)) == 0.0


def test_camber_at_te_is_last_knot() -> None:
    assert _camber_height(+1.0, (0.0, 0.002, 0.001)) == 0.001


# ---- make_outer_envelope: bounding box ------------------------------------


def test_envelope_x_extent_matches_blade_length() -> None:
    p = _canonical_layer1()
    bb = _bbox(make_outer_envelope(p, "flat"))
    assert bb.xmin == pytest.approx(LOFT_START_EPS_M, abs=1e-6)
    assert bb.xmax == pytest.approx(L_BLADE_M, abs=1e-6)


def test_envelope_y_extent_matches_half_pitch_no_fourier() -> None:
    """Without Fourier modulation, y_max at the tip = L · INTER_BLADE_ANGLE_RAD / 2."""
    p = _canonical_layer1()  # zero Fourier
    bb = _bbox(make_outer_envelope(p, "flat"))
    expected = L_BLADE_M * INTER_BLADE_ANGLE_RAD / 2
    assert bb.ymax == pytest.approx(expected, rel=1e-3)
    assert bb.ymin == pytest.approx(-expected, rel=1e-3)


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
    location where both are max. With camber_knots = (0, 0.002, 0.001), the
    max camber is 0.002 mm; thickness max is 0.0030 mm (knot at t=0);
    zmax should reach 0.0030 + 0.002 = 0.005."""
    p = _canonical_layer1(
        camber_knots_m=(0.0, 0.002, 0.001),
        thickness_knots_m=(0.0030, 0.0028, 0.0026),
    )
    bb = _bbox(make_outer_envelope(p, "flat"))
    assert bb.zmax == pytest.approx(0.005, abs=1e-4)


# ---- twist behaviour -------------------------------------------------------


def test_twist_zero_under_flat_orientation() -> None:
    """Plano-convex: twist is suppressed to keep the bottom face planar."""
    p_no_twist = _canonical_layer1(twist_knots_rad=(0.0, 0.0))
    p_twisted = _canonical_layer1(twist_knots_rad=(0.0, math.radians(10.0)))
    bb_no = _bbox(make_outer_envelope(p_no_twist, "flat"))
    bb_tw = _bbox(make_outer_envelope(p_twisted, "flat"))
    # Both should have z_min == 0 (planar bottom).
    assert bb_no.zmin == pytest.approx(0.0, abs=1e-6)
    assert bb_tw.zmin == pytest.approx(0.0, abs=1e-6)
    # And the upper bound should be the same (twist was suppressed).
    assert bb_no.zmax == pytest.approx(bb_tw.zmax, abs=1e-4)


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
    """With thickness_knots = (3.0 mm, 2.8, 2.6), the root is thickest. The
    plano-convex z_max therefore comes from x near the root (modulated by
    chord-position camber)."""
    p = _canonical_layer1(
        camber_knots_m=(0.0, 0.0, 0.0),  # no camber
        thickness_knots_m=(0.0030, 0.0028, 0.0026),
    )
    bb = _bbox(make_outer_envelope(p, "flat"))
    # z_max should equal the root thickness 3.0 mm.
    assert bb.zmax == pytest.approx(0.0030, abs=1e-4)


def test_camber_zero_yields_constant_top_face_height_per_x() -> None:
    """With camber_knots = (0, 0, 0), the top face at any x is at z = thickness(x)."""
    p = _canonical_layer1(
        camber_knots_m=(0.0, 0.0, 0.0),
        thickness_knots_m=(0.0030, 0.0030, 0.0030),  # uniform thickness
    )
    bb = _bbox(make_outer_envelope(p, "flat"))
    # Uniform thickness 3.0 mm; no camber → z_max = 3.0 mm.
    assert bb.zmax == pytest.approx(0.0030, abs=1e-4)
    assert bb.zmin == pytest.approx(0.0, abs=1e-6)


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
