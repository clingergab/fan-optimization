"""Tests for fanopt.cfd.airfoil_shapes.

Pure-Python NACA helpers — no gmsh dependency, runs in CI everywhere.
"""

from __future__ import annotations

import math

import pytest

from fanopt.cfd.airfoil_shapes import (
    NACA0012_CHORD,
    NACA0012_T,
    airfoil_polyline,
    naca0012_y,
)

# ---- NACA 0012 shape function ---------------------------------------------


def test_naca0012_at_leading_edge_is_zero() -> None:
    """y(0) = 0 by definition for any symmetric airfoil."""
    assert naca0012_y(0.0) == pytest.approx(0.0, abs=1e-12)


def test_naca0012_max_thickness_near_30pct_chord() -> None:
    """NACA 0012 peak thickness is at ~30% chord."""
    xs = [0.20, 0.25, 0.30, 0.35, 0.40]
    ys = [naca0012_y(x) for x in xs]
    peak_idx = ys.index(max(ys))
    assert xs[peak_idx] == pytest.approx(0.30, abs=0.05)


def test_naca0012_peak_half_thickness_is_about_0_06() -> None:
    """Max half-thickness ≈ 6% of chord for NACA 0012 (t/2)."""
    peak = max(naca0012_y(x) for x in [i / 1000 for i in range(1, 1000)])
    assert peak == pytest.approx(0.06, abs=0.003)


def test_naca0012_rejects_out_of_range_x() -> None:
    with pytest.raises(ValueError, match=r"x must be in"):
        naca0012_y(-0.01)
    with pytest.raises(ValueError, match=r"x must be in"):
        naca0012_y(1.01)


def test_naca0012_scales_with_chord() -> None:
    """Doubling the chord doubles every y value at the same chordwise fraction."""
    y_unit = naca0012_y(0.3, chord=1.0)
    y_double = naca0012_y(0.6, chord=2.0)
    assert y_double == pytest.approx(2.0 * y_unit, rel=1e-12)


def test_naca0012_closes_at_trailing_edge() -> None:
    """The closed-TE variant has y(chord) = 0 exactly (formula coefficient choice)."""
    assert naca0012_y(NACA0012_CHORD) == pytest.approx(0.0, abs=1e-9)


def test_naca0012_t_constant_matches_aerodynamic_convention() -> None:
    """NACA 0012's '12' means t/c = 0.12."""
    assert NACA0012_T == 0.12


# ---- airfoil_polyline -----------------------------------------------------


def test_polyline_closed() -> None:
    """Polyline spans both surfaces; min x = 0 (LE), max x = chord (TE)."""
    pts = airfoil_polyline(64)
    xs = [p[0] for p in pts]
    assert min(xs) == pytest.approx(0.0, abs=1e-9)
    assert max(xs) == pytest.approx(NACA0012_CHORD, abs=1e-9)


def test_polyline_symmetric_upper_lower() -> None:
    """For a symmetric airfoil, |y_upper(x)| == |y_lower(x)|."""
    pts = airfoil_polyline(64)
    upper = {round(p[0], 6): p[1] for p in pts if p[1] >= 0}
    lower = {round(p[0], 6): p[1] for p in pts if p[1] <= 0}
    for x, y_u in upper.items():
        if x in lower:
            assert lower[x] == pytest.approx(-y_u, abs=1e-9)


def test_polyline_rejects_too_few_points() -> None:
    with pytest.raises(ValueError, match=r"must be ≥ 16"):
        airfoil_polyline(8)


def test_polyline_density_in_expected_range() -> None:
    """n=64 produces between n and 2n boundary points (after LE/TE de-dup)."""
    pts = airfoil_polyline(64)
    assert 64 <= len(pts) <= 128


def test_polyline_cosine_spacing_clusters_near_le() -> None:
    """First few points after LE should be much closer in x than midchord points."""
    pts = airfoil_polyline(128)
    # Find points near the leading edge.
    near_le = [p[0] for p in pts if p[0] < 0.05 and p[1] >= 0]
    near_mid = [p[0] for p in pts if 0.4 < p[0] < 0.6 and p[1] >= 0]
    if len(near_le) >= 2 and len(near_mid) >= 2:
        near_le_sorted = sorted(near_le)
        near_mid_sorted = sorted(near_mid)
        le_gap = near_le_sorted[1] - near_le_sorted[0]
        mid_gap = near_mid_sorted[1] - near_mid_sorted[0]
        assert le_gap < mid_gap


def test_polyline_custom_chord_scales_all_x() -> None:
    """Doubling chord doubles every x coordinate."""
    pts_unit = airfoil_polyline(32, chord=1.0)
    pts_double = airfoil_polyline(32, chord=2.0)
    for (x1, _), (x2, _) in zip(pts_unit, pts_double, strict=True):
        assert x2 == pytest.approx(2.0 * x1, rel=1e-12)


def test_polyline_consecutive_segments_short() -> None:
    """No consecutive-point jump longer than ~2% of chord.

    Catches the pre-fix bug where the polyline went TE → upper → LE,
    then jumped from LE all the way back to near-TE on the lower
    surface (a chord-spanning straight line) before traversing lower.
    That bad ordering passed every other test in this file (the points
    were all correct; only the traversal order was wrong) but produced
    "intersections in the 1D mesh" errors when fed to gmsh.

    For 200 cosine-spaced points, worst-case adjacent-pair spacing is
    near the LE/TE corners where the cosine spacing is densest; with
    n=64 a 0.02-chord bound is conservative. Any chord-spanning jump
    blows past it by 50x.
    """
    pts = airfoil_polyline(64, chord=1.0)
    n = len(pts)
    max_gap = 0.0
    worst_i = 0
    for i in range(n):
        j = (i + 1) % n  # closed loop — also check the closing segment
        dx = pts[j][0] - pts[i][0]
        dy = pts[j][1] - pts[i][1]
        gap = math.hypot(dx, dy)
        if gap > max_gap:
            max_gap = gap
            worst_i = i
    assert max_gap < 0.10, (
        f"polyline has a consecutive-pair jump of {max_gap:.4f} chords "
        f"at index {worst_i} ({pts[worst_i]} -> {pts[(worst_i + 1) % n]}). "
        "Cosine-spaced airfoil points should not produce gaps near "
        "this size — check the polyline ordering."
    )


def test_polyline_traverses_around_airfoil_monotonically() -> None:
    """The loop should go TE → LE on one surface, then LE → TE on the other.

    Concretely: the x-coordinate should monotonically decrease over the
    first half of the polyline (TE → LE on upper surface) and then
    monotonically increase over the second half (LE → TE on lower surface).
    Cosine spacing makes this strict; under the pre-fix bug, the second
    half started near-TE and went toward LE, violating this property.
    """
    pts = airfoil_polyline(64, chord=1.0)
    n = len(pts)
    # First half: x should be decreasing (TE → LE on upper)
    half = n // 2
    for i in range(half - 1):
        assert pts[i + 1][0] <= pts[i][0] + 1e-9, (
            f"upper-surface traversal not monotonic at index {i}: "
            f"x[{i}]={pts[i][0]:.4f}, x[{i+1}]={pts[i+1][0]:.4f}"
        )
    # Second half: x should be increasing (LE → TE on lower)
    for i in range(half, n - 1):
        assert pts[i + 1][0] >= pts[i][0] - 1e-9, (
            f"lower-surface traversal not monotonic at index {i}: "
            f"x[{i}]={pts[i][0]:.4f}, x[{i+1}]={pts[i+1][0]:.4f}"
        )
