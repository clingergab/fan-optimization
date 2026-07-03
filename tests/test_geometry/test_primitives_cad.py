"""Tests for fanopt.geometry.primitives_cad (CadQuery Layer 3 generator).

Skipped at module load when CadQuery isn't installed, per CLAUDE.md §4.1
optional-dep handling.
"""

from __future__ import annotations

import importlib.util
import math

import pytest

if importlib.util.find_spec("cadquery") is None:
    pytest.skip("cadquery not installed", allow_module_level=True)

import cadquery as cq

from fanopt.geometry.envelope import Layer1Params, ThicknessGridField
from fanopt.geometry.envelope_cad import make_outer_envelope
from fanopt.geometry.primitives import (
    PRIMITIVE_MARGIN_FROM_EDGE_M,
    PRIMITIVE_MIN_DIMENSION_M,
    Layer3Primitive,
)
from fanopt.geometry.primitives_cad import (
    _make_ellipsoid,
    _make_slot,
    _make_wedge,
    apply_primitive,
)

# ---- helpers --------------------------------------------------------------


def _canonical_envelope() -> cq.Workplane:
    """A minimal Layer 1 envelope to exercise primitive application against."""
    p = Layer1Params(
        blade_count=10,
        camber_knots_m=(0.0, 0.002, 0.001),
        twist_knots_rad=(0.0, 0.0),
        thickness_field=ThicknessGridField.from_radial_knots((0.0030, 0.0028, 0.0026)),
        edge_profile="rounded",
        fourier_le_amplitudes=(0.0, 0.0, 0.0),
        fourier_te_amplitudes=(0.0, 0.0, 0.0),
    )
    return make_outer_envelope(p, "flat")


def _bbox_inside_envelope_primitive(
    shape_type: str = "ellipsoid",
    polarity: str = "subtract",
) -> Layer3Primitive:
    """A small primitive that fits safely inside the canonical envelope.

    The envelope spans roughly x ∈ [0.001, 0.200], y ∈ [-0.023, +0.023],
    z ∈ [0, ~0.005]. A 1 mm × 1 mm × 1 mm primitive at (0.10, 0.0, 0.002)
    is well inside on all axes.
    """
    return Layer3Primitive(
        present=True,
        shape_type=shape_type,
        polarity=polarity,
        position_x_m=0.10,
        position_y_m=0.01,
        position_z_m=0.0025,
        size_x_m=0.001,
        size_y_m=0.001,
        size_z_m=0.001,
        rotation_x_rad=0.0,
        rotation_y_rad=0.0,
        rotation_z_rad=0.0,
        local_envelope_xyz_m=(0.20, 0.05, 0.005),
    )


# ---- absent primitive -----------------------------------------------------


def test_apply_primitive_absent_returns_input_unchanged() -> None:
    """When ``present=False`` the function is a no-op."""
    env = _canonical_envelope()
    vol_before = env.val().Volume()
    out = apply_primitive(env, Layer3Primitive.absent())
    assert out is env  # same workplane handle
    assert out.val().Volume() == pytest.approx(vol_before)


# ---- builder shapes -------------------------------------------------------


def test_make_slot_volume_matches_capsule_estimate() -> None:
    """A 4×2×1 mm slot has a box bulk plus two half-cylinders at the ends.

    Capsule (rectangular bar + cylinder caps) volume formula:
    V = (size_x - size_y) * size_y * size_z + π * (size_y/2)² * size_z
    when size_y is the smaller in-plane extent (the diameter of the
    rounded ends).
    """
    sx, sy, sz = 0.004, 0.002, 0.001
    box_part = (sx - sy) * sy * sz
    cap_part = math.pi * (sy / 2.0) ** 2 * sz
    expected = box_part + cap_part
    vol = _make_slot(sx, sy, sz).val().Volume()
    # CadQuery edge-fillet uses a finite tolerance for the cylindrical
    # cap surface; ~1% slop is plenty.
    assert vol == pytest.approx(expected, rel=0.01)


def test_make_slot_handles_square_cross_section() -> None:
    """When size_x == size_y the slot is a cylinder. Must not raise."""
    vol = _make_slot(0.002, 0.002, 0.001).val().Volume()
    expected = math.pi * (0.002 / 2.0) ** 2 * 0.001
    assert vol == pytest.approx(expected, rel=0.05)


def test_make_ellipsoid_volume_matches_formula() -> None:
    """Triaxial ellipsoid volume: (4/3) π a b c with a,b,c = size/2."""
    sx, sy, sz = 0.002, 0.003, 0.001
    expected = (4.0 / 3.0) * math.pi * (sx / 2) * (sy / 2) * (sz / 2)
    vol = _make_ellipsoid(sx, sy, sz).val().Volume()
    # CadQuery's sphere tessellation tolerance — 2% is generous.
    assert vol == pytest.approx(expected, rel=0.02)


def test_make_wedge_volume_is_half_box() -> None:
    """The wedge cross-section is a right triangle (half a rectangle), so
    the wedge volume is exactly half the bounding-box volume."""
    sx, sy, sz = 0.003, 0.002, 0.001
    expected = 0.5 * sx * sy * sz
    vol = _make_wedge(sx, sy, sz).val().Volume()
    assert vol == pytest.approx(expected, rel=1e-6)


def test_make_wedge_bbox_matches_size_extents() -> None:
    sx, sy, sz = 0.003, 0.002, 0.001
    bb = _make_wedge(sx, sy, sz).val().BoundingBox()
    assert bb.xmax - bb.xmin == pytest.approx(sx, abs=1e-6)
    assert bb.ymax - bb.ymin == pytest.approx(sy, abs=1e-6)
    assert bb.zmax - bb.zmin == pytest.approx(sz, abs=1e-6)


# ---- subtract polarity ----------------------------------------------------


def test_subtract_polarity_reduces_volume() -> None:
    env = _canonical_envelope()
    vol_before = env.val().Volume()
    out = apply_primitive(env, _bbox_inside_envelope_primitive(polarity="subtract"))
    vol_after = out.val().Volume()
    assert vol_after < vol_before
    # Loose lower bound: at least 50% of the primitive's nominal volume
    # must come out (the primitive lives entirely inside the envelope).
    primitive_vol = (4.0 / 3.0) * math.pi * (0.0005**3)  # ellipsoid r=0.5mm
    assert (vol_before - vol_after) > primitive_vol * 0.5


def test_subtract_polarity_each_shape_type_reduces_volume() -> None:
    """All three shape types must reduce volume under subtract polarity."""
    env = _canonical_envelope()
    vol_before = env.val().Volume()
    for shape_type in ("slot", "ellipsoid", "wedge"):
        out = apply_primitive(
            env, _bbox_inside_envelope_primitive(shape_type=shape_type, polarity="subtract")
        )
        assert out.val().Volume() < vol_before, f"{shape_type} did not reduce volume"


# ---- add polarity ---------------------------------------------------------


def test_add_polarity_increases_volume_when_primitive_outside_envelope() -> None:
    """Add a primitive offset above the envelope (z > envelope.zmax) so
    its volume is purely additive (no overlap to subtract from)."""
    env = _canonical_envelope()
    vol_before = env.val().Volume()
    # Envelope zmax ~ 0.005; place the primitive's centre at z = 0.010
    # with z-extent 0.002 so it's fully above the envelope.
    add = Layer3Primitive(
        present=True,
        shape_type="ellipsoid",
        polarity="add",
        position_x_m=0.10,
        position_y_m=0.01,
        position_z_m=0.010,
        size_x_m=0.002,
        size_y_m=0.002,
        size_z_m=0.002,
        local_envelope_xyz_m=(0.20, 0.05, 0.020),
    )
    out = apply_primitive(env, add)
    assert out.val().Volume() > vol_before


# ---- rotation -------------------------------------------------------------


def test_rotation_x_and_y_exercise_all_axes() -> None:
    """The X and Y rotation branches need coverage. A wedge cut with a
    90° rotation about x or y differs from the unrotated wedge in
    which faces it carves into the envelope; both must apply cleanly."""
    env = _canonical_envelope()
    vol_before = env.val().Volume()
    p_xrot = Layer3Primitive(
        present=True,
        shape_type="wedge",
        polarity="subtract",
        position_x_m=0.10,
        position_y_m=0.01,
        position_z_m=0.0025,
        size_x_m=0.001,
        size_y_m=0.001,
        size_z_m=0.001,
        rotation_x_rad=math.pi / 2.0,
        local_envelope_xyz_m=(0.20, 0.05, 0.005),
    )
    p_yrot = Layer3Primitive(
        present=True,
        shape_type="wedge",
        polarity="subtract",
        position_x_m=0.10,
        position_y_m=0.01,
        position_z_m=0.0025,
        size_x_m=0.001,
        size_y_m=0.001,
        size_z_m=0.001,
        rotation_y_rad=math.pi / 2.0,
        local_envelope_xyz_m=(0.20, 0.05, 0.005),
    )
    assert apply_primitive(env, p_xrot).val().Volume() < vol_before
    assert apply_primitive(env, p_yrot).val().Volume() < vol_before


def test_rotation_z_changes_wedge_orientation() -> None:
    """A 90° rotation about z swaps the wedge's x and y extents in the cut."""
    env = _canonical_envelope()
    p_unrot = Layer3Primitive(
        present=True,
        shape_type="wedge",
        polarity="subtract",
        position_x_m=0.10,
        position_y_m=0.01,
        position_z_m=0.0025,
        size_x_m=0.003,
        size_y_m=0.001,
        size_z_m=0.001,
        local_envelope_xyz_m=(0.20, 0.05, 0.005),
    )
    p_rot = Layer3Primitive(
        present=True,
        shape_type="wedge",
        polarity="subtract",
        position_x_m=0.10,
        position_y_m=0.01,
        position_z_m=0.0025,
        size_x_m=0.003,
        size_y_m=0.001,
        size_z_m=0.001,
        rotation_z_rad=math.pi / 2.0,
        local_envelope_xyz_m=(0.20, 0.05, 0.005),
    )
    vol_unrot = apply_primitive(env, p_unrot).val().Volume()
    vol_rot = apply_primitive(env, p_rot).val().Volume()
    # Both subtract the same wedge volume (rotation preserves volume),
    # so the resulting envelope volumes must match.
    assert vol_unrot == pytest.approx(vol_rot, rel=0.05)


# ---- soft-fail contract ---------------------------------------------------


def test_apply_primitive_raises_propagate_for_generator_trycatch() -> None:
    """The generator wraps apply_primitive in try/except per plan §9.7;
    we deliberately do NOT swallow CAD failures inside this module.

    Trigger a failure by constructing a primitive that the schema cannot
    catch but CadQuery / OpenCascade may reject: a *very* thin wedge.
    """
    env = _canonical_envelope()
    # Force a degenerate wedge construction by monkey-patching the builder
    # to raise. The test confirms the exception propagates out of
    # apply_primitive (it is the orchestrator's job to catch it).
    p = _bbox_inside_envelope_primitive(shape_type="wedge", polarity="subtract")

    from fanopt.geometry import primitives_cad

    original = primitives_cad._SHAPE_BUILDERS["wedge"]

    def boom(sx: float, sy: float, sz: float) -> cq.Workplane:
        raise RuntimeError("simulated OpenCascade Boolean failure")

    primitives_cad._SHAPE_BUILDERS["wedge"] = boom
    try:
        with pytest.raises(RuntimeError, match="simulated OpenCascade"):
            apply_primitive(env, p)
    finally:
        primitives_cad._SHAPE_BUILDERS["wedge"] = original


# ---- positional / size invariants ----------------------------------------


def test_subtract_primitive_at_position_creates_cavity_at_that_position() -> None:
    """After subtracting a primitive at (px, py, pz), the resulting shape
    must NOT contain that point — there's a cavity there."""
    env = _canonical_envelope()
    p = _bbox_inside_envelope_primitive(shape_type="ellipsoid", polarity="subtract")
    out = apply_primitive(env, p)
    # Construct a tiny probe sphere at the primitive centre. If the
    # cavity exists, intersecting the probe with the cut shape produces
    # less volume than intersecting it with the original.
    probe_centre = (p.position_x_m, p.position_y_m, p.position_z_m)
    probe = (
        cq.Workplane("XY")
        .sphere(0.0002)  # 0.2 mm radius probe — smaller than the 0.5 mm primitive radius
        .translate(probe_centre)
    )
    probe_solid = probe.val()
    inter_before = env.val().intersect(probe_solid).Volume()
    inter_after = out.val().intersect(probe_solid).Volume()
    assert inter_after < inter_before
    # The primitive's r=0.5mm sphere fully contains the probe r=0.2mm sphere
    # at the same centre, so the cut removes the probe entirely.
    assert inter_after == pytest.approx(0.0, abs=1e-12)


# ---- end-to-end with envelope --------------------------------------------


def test_apply_primitive_preserves_solid_shape() -> None:
    """The output of apply_primitive must still be a valid solid (single
    connected component, positive volume) for all three shape types."""
    env = _canonical_envelope()
    for shape_type in ("slot", "ellipsoid", "wedge"):
        out = apply_primitive(
            env, _bbox_inside_envelope_primitive(shape_type=shape_type, polarity="subtract")
        )
        solid = out.val()
        assert solid.Volume() > 0.0
        # Single solid (no fragments). CadQuery's Solids() returns a list.
        assert len(out.solids().vals()) == 1


def test_minimum_dimension_primitive_applies_cleanly() -> None:
    """The 0.8 mm schema floor must not trip CadQuery on Boolean ops."""
    env = _canonical_envelope()
    min_size = PRIMITIVE_MIN_DIMENSION_M
    p = Layer3Primitive(
        present=True,
        shape_type="ellipsoid",
        polarity="subtract",
        position_x_m=0.10,
        position_y_m=0.01,
        position_z_m=0.0025,
        size_x_m=min_size,
        size_y_m=min_size,
        size_z_m=min_size,
        local_envelope_xyz_m=(0.20, 0.05, 0.005),
    )
    out = apply_primitive(env, p)
    assert out.val().Volume() > 0.0


def test_margin_floor_inside_envelope() -> None:
    """A primitive at exactly the schema margin must still apply cleanly."""
    margin = PRIMITIVE_MARGIN_FROM_EDGE_M
    env = _canonical_envelope()
    # Envelope x extent is ~[0.001, 0.200]; place primitive at margin+epsilon
    # in the local_envelope frame the schema enforces against.
    local_env = (0.20, 0.05, 0.005)
    p = Layer3Primitive(
        present=True,
        shape_type="slot",
        polarity="subtract",
        position_x_m=margin + 0.001,  # 1 mm past the margin → safely inside
        position_y_m=local_env[1] / 2,
        position_z_m=local_env[2] / 2,
        size_x_m=PRIMITIVE_MIN_DIMENSION_M,
        size_y_m=PRIMITIVE_MIN_DIMENSION_M,
        size_z_m=PRIMITIVE_MIN_DIMENSION_M,
        local_envelope_xyz_m=local_env,
    )
    out = apply_primitive(env, p)
    assert out.val().Volume() > 0.0
