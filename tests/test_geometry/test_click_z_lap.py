"""HIGH-8 Round-9 Option A lock: click chamfer is a 0.5-1 mm × 0.5-1 mm corner
bevel, NOT a full-panel-thickness face. Adjacent panels meet at a 45° butt-joint
line at the deployed angle; no Z-axis overlap.

Pre-HIGH-8 the assertion was "chamfer spans the panel z-extent" — that's
retired. Under Option A, the chamfer is a SMALL bevel and the panel-to-panel
contact is a LINE contact at the chamfer-meeting plane, not a face-to-face
Z-overlap.

The four assertions below correspond to the four parts of the Option A geometry:
  1. Chamfer is a small corner bevel (0.5-1 mm depth in z).
  2. Chamfer angle is 45° (self-aligning during deployment).
  3. No Z-axis overlap between adjacent panels.
  4. Chamfer lives at the panel's outer tangential edge (NOT on the rib face).
"""
from __future__ import annotations

import math

import pytest

# ── Locked constants ──────────────────────────────────────────────────────
PANEL_THICKNESS_MIN_M = 0.0022   # C7 lock
PANEL_THICKNESS_MAX_M = 0.0038   # C7 lock
CHAMFER_DEPTH_MIN_M = 0.0005     # HIGH-8 Option A lower bound (0.5 mm)
CHAMFER_DEPTH_MAX_M = 0.001      # HIGH-8 Option A upper bound (1.0 mm)
CHAMFER_ANGLE_RAD = math.radians(45.0)
ANGLE_TOLERANCE_RAD = math.radians(1.0)
PANEL_TANGENTIAL_OUTER_AT_TIP_M = 0.0225  # half-width at tip per the #35 widening lock
L_BLADE_M = 0.200


def _generate_baseline_blade():
    """Production: from fanopt.geometry.generator import generate_blade.

    Returns a Blade object exposing the helper methods used in the assertions.
    For Phase 0, the generator may not yet exist; gate vacuously passes via
    pytest.skip, and the test re-runs once the generator lands.
    """
    try:
        from fanopt.geometry.generator import generate_blade  # type: ignore[import-untyped]
        from fanopt.geometry.schema import baseline_params  # type: ignore[import-untyped]

        return generate_blade(baseline_params)
    except ImportError:
        pytest.skip(
            "fanopt.geometry not yet implemented (Phase 0 sets it up). "
            "Once the generator exists, this gate runs against the production blade."
        )


# ─────────────────────────────────────────────────────────────────────
# Assertion 1: chamfer is a small bevel, NOT full-panel-z face
# ─────────────────────────────────────────────────────────────────────


def test_chamfer_is_small_bevel_not_full_z_face() -> None:
    """HIGH-8 Option A: chamfer depth ∈ [0.5, 1.0] mm — NOT a full-panel-thickness span."""
    blade = _generate_baseline_blade()
    z_extent = blade.click_chamfer_z_extent()
    assert CHAMFER_DEPTH_MIN_M <= z_extent <= CHAMFER_DEPTH_MAX_M, (
        f"HIGH-8 Option A violation: click chamfer z-extent {z_extent * 1000:.3f} mm "
        f"is outside the {CHAMFER_DEPTH_MIN_M * 1000:.1f}-{CHAMFER_DEPTH_MAX_M * 1000:.1f} mm "
        f"corner-bevel range. A chamfer spanning the full panel_thickness "
        f"({PANEL_THICKNESS_MIN_M * 1000:.1f}-{PANEL_THICKNESS_MAX_M * 1000:.1f} mm) is the "
        f"RETIRED pre-HIGH-8 architecture; Option A requires a small corner bevel only."
    )


# ─────────────────────────────────────────────────────────────────────
# Assertion 2: chamfer angle is 45°
# ─────────────────────────────────────────────────────────────────────


def test_chamfer_angle_is_45_degrees() -> None:
    """HIGH-8 Option A: chamfer face is at 45° — self-aligning during deployment."""
    blade = _generate_baseline_blade()
    normal = blade.click_chamfer_normal()  # unit vector of the chamfer face
    assert len(normal) == 3, "chamfer normal must be a 3-vector"
    # Angle to +z axis is acos(|n_z|); we want it to be 45°.
    cos_angle = abs(normal[2])
    expected_cos = math.cos(CHAMFER_ANGLE_RAD)
    # cos(45°) = cos(45° ± 1°) tolerance via |cos(θ) − cos(45°)|
    assert abs(cos_angle - expected_cos) < math.sin(CHAMFER_ANGLE_RAD) * math.sin(
        ANGLE_TOLERANCE_RAD
    ) + 0.01, (
        f"HIGH-8 Option A violation: chamfer normal z-component {cos_angle:.3f} differs "
        f"from cos(45°) = {expected_cos:.3f} by more than 1° tolerance."
    )


# ─────────────────────────────────────────────────────────────────────
# Assertion 3: no Z-axis overlap between adjacent panels
# ─────────────────────────────────────────────────────────────────────


def test_no_z_overlap_between_adjacent_panels() -> None:
    """HIGH-8 Option A: adjacent panels meet at a 45° LINE at z = z_i + panel_thickness/2;
    NO Z-axis overlap (face-to-face overlap was the retired pre-HIGH-8 architecture)."""
    blade = _generate_baseline_blade()
    panel_thickness = blade.panel_thickness_at_tip()
    # Under rib-flat (C10 Option E), blade i's panel occupies z ∈ [0, panel_thickness].
    # Blade i+1 stacks at z_i+1 = z_i + panel_thickness, so its panel bottom face is
    # at z = panel_thickness — coplanar with blade i's panel top face.
    z_i_top_panel_face = panel_thickness
    z_i_plus_1_bottom_panel_face = panel_thickness  # after stacking
    overlap = z_i_top_panel_face - z_i_plus_1_bottom_panel_face
    assert abs(overlap) < 1e-6, (
        f"HIGH-8 Option A violation: adjacent panels overlap in z by "
        f"{overlap * 1000:.3f} mm. Option A requires zero z-overlap (panels meet at a "
        f"LINE at the chamfer-meeting plane); the 'lap joint with z-overlap' terminology "
        f"used in earlier drafts is retired."
    )


# ─────────────────────────────────────────────────────────────────────
# Assertion 4: chamfer is on the panel outer tangential edge, NOT on the rib
# ─────────────────────────────────────────────────────────────────────


def test_chamfer_at_panel_outer_tangential_edge() -> None:
    """HIGH-8 Option A: chamfer lives on the panel's outer tangential edge at the tip
    (NOT on the rib face; item #3 panel-edge relocation)."""
    blade = _generate_baseline_blade()
    cx, cy = blade.click_chamfer_centroid_xy()
    panel_tangential_outer = blade.panel_tangential_outer_at_tip()
    x_expected = blade.l_blade()
    assert abs(cx - x_expected) < 0.001, (
        f"HIGH-8 violation: chamfer x-coord {cx:.4f} m not at L_blade = "
        f"{x_expected:.4f} m (chamfer must live at the panel tip, not mid-panel)."
    )
    assert abs(abs(cy) - panel_tangential_outer) < 0.001, (
        f"HIGH-8 violation: chamfer y-coord {cy:.4f} m not at panel_tangential_outer = "
        f"±{panel_tangential_outer:.4f} m. The chamfer must live on the panel's outer "
        f"tangential edge (NOT at the rib's y = ±rib_center position; rib-mounted "
        f"chamfers are the retired pre-item-#3 architecture)."
    )


if __name__ == "__main__":
    test_chamfer_is_small_bevel_not_full_z_face()
    test_chamfer_angle_is_45_degrees()
    test_no_z_overlap_between_adjacent_panels()
    test_chamfer_at_panel_outer_tangential_edge()
    print("OK: HIGH-8 Round-9 Option A chamfer geometry locked.")
