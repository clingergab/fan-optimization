"""Tests for fanopt.geometry.generator_cad (CadQuery wrapper)."""

from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("cadquery") is None:
    pytest.skip("cadquery not installed", allow_module_level=True)

import cadquery as cq

from fanopt.geometry.envelope import Layer1Params, ThicknessGridField
from fanopt.geometry.fields import Layer2Params, LouverField
from fanopt.geometry.generator import (
    BladeDesignParams,
    GenerationStatus,
    generate_blade,
)
from fanopt.geometry.generator_cad import generate_blade_cad
from fanopt.geometry.manufacturability import Layer4Params
from fanopt.geometry.primitives import Layer3Primitive


def _canonical_design() -> BladeDesignParams:
    return BladeDesignParams(
        layer1=Layer1Params(
            blade_count=10,
            camber_knots_m=(0.0, 0.002, 0.001),
            twist_knots_rad=(0.0, 0.0),
            thickness_field=ThicknessGridField.from_radial_knots((0.0030, 0.0028, 0.0026)),
            edge_profile="rounded",
            fourier_le_amplitudes=(0.05, 0.0, 0.0),
            fourier_te_amplitudes=(0.0, 0.05, 0.0),
        ),
        layer2=Layer2Params.all_inactive(),
        layer3=Layer3Primitive.absent(),
        layer4=Layer4Params(
            print_orientation="flat",
            layer_height_m=0.0002,
            click_chamfer_angle_deg=45.0,
            click_detent_size_m=0.0004,
            click_design_clearance_m=0.00018,
        ),
    )


def test_returns_scaffold_result_and_shape() -> None:
    """Wrapper returns the (result, shape) tuple — both non-None on a
    canonical valid design."""
    design = _canonical_design()
    result, shape = generate_blade_cad(design)
    assert result is not None
    assert shape is not None


def test_wrapper_overrides_manufacturability_with_cadquery_inspection() -> None:
    """The wrapper's result swaps the scaffold's PENDING_CADQUERY-heavy
    filter for the real CadQuery inspection. Layer descriptions + panel
    domain mask + params + generator_version must still match the
    scaffold; only ``manufacturability`` (and possibly ``status``) differ.
    """
    design = _canonical_design()
    scaffold_only = generate_blade(design)
    wrapped, _shape = generate_blade_cad(design)
    assert wrapped.layer_descriptions == scaffold_only.layer_descriptions
    assert wrapped.panel_domain_mask == scaffold_only.panel_domain_mask
    assert wrapped.params == scaffold_only.params
    assert wrapped.generator_version == scaffold_only.generator_version
    # Manufacturability differs: scaffold returns many PENDING; wrapper
    # returns mostly PASSED with only #1 and #8 PENDING.
    assert set(wrapped.manufacturability.pending_cadquery) == {"1", "8"}
    assert len(scaffold_only.manufacturability.pending_cadquery) > 2


def test_status_ok_on_canonical_design() -> None:
    design = _canonical_design()
    result, _shape = generate_blade_cad(design)
    assert result.status == GenerationStatus.OK


def test_shape_is_cadquery_workplane() -> None:
    """The shape is a Workplane that we can inspect via .val().BoundingBox()."""
    design = _canonical_design()
    _result, shape = generate_blade_cad(design)
    assert isinstance(shape, cq.Workplane)
    bb = shape.val().BoundingBox()
    assert bb.xmax > bb.xmin
    assert bb.ymax > bb.ymin
    assert bb.zmax > bb.zmin


def test_shape_honours_print_orientation_flat() -> None:
    """Under flat orientation the envelope's z_min == 0 (plano-convex)."""
    design = _canonical_design()  # print_orientation="flat"
    _result, shape = generate_blade_cad(design)
    bb = shape.val().BoundingBox()
    assert bb.zmin == pytest.approx(0.0, abs=1e-6)


def test_shape_honours_print_orientation_edge() -> None:
    """Under edge orientation z_min < 0 (midplane-symmetric)."""
    d = _canonical_design()
    d_edge = BladeDesignParams(
        layer1=d.layer1,
        layer2=d.layer2,
        layer3=d.layer3,
        layer4=Layer4Params(
            print_orientation="edge",
            layer_height_m=d.layer4.layer_height_m,
            click_chamfer_angle_deg=d.layer4.click_chamfer_angle_deg,
            click_detent_size_m=d.layer4.click_detent_size_m,
            click_design_clearance_m=d.layer4.click_design_clearance_m,
        ),
    )
    _result, shape = generate_blade_cad(d_edge)
    bb = shape.val().BoundingBox()
    assert bb.zmin < 0.0


def test_layer2_louver_activation_reduces_volume() -> None:
    """Activating a louver subtract field must reduce the returned shape's
    volume. Replaced the prior 'Layer 2 unaffected' pin once
    apply_layer2_fields landed (2026-05-21).
    """
    d_inactive = _canonical_design()
    d_with_louver = BladeDesignParams(
        layer1=d_inactive.layer1,
        layer2=Layer2Params(
            louver=LouverField(active=True, count=6, width_m=0.001, polarity="subtract"),
            texture=d_inactive.layer2.texture,
            edge=d_inactive.layer2.edge,
        ),
        layer3=d_inactive.layer3,
        layer4=d_inactive.layer4,
    )
    _r_inactive, shape_inactive = generate_blade_cad(d_inactive)
    _r_active, shape_active = generate_blade_cad(d_with_louver)
    assert shape_active.val().Volume() < shape_inactive.val().Volume()


def test_layer3_subtract_reduces_volume() -> None:
    """Layer 3 subtract primitive must reduce the returned shape's volume.

    Replaced the prior "Layer 3 unaffected" pin once apply_primitive
    landed (2026-05-21). The test now exercises the integration:
    a subtract primitive inside the envelope must remove material.
    """
    # Trapezoidal envelope: at x=0.10, y_max ≈ 0.10 * INTER_BLADE_ANGLE_RAD / 2 ≈ 0.0116.
    # Position y=0.005 with size 0.001 lives safely inside that band.
    env = (0.200, 0.050, 0.005)
    d_inactive = _canonical_design()
    d_with_prim = BladeDesignParams(
        layer1=d_inactive.layer1,
        layer2=d_inactive.layer2,
        layer3=Layer3Primitive(
            present=True,
            shape_type="ellipsoid",
            polarity="subtract",
            position_x_m=0.10,
            position_y_m=0.005,
            position_z_m=0.0025,
            size_x_m=0.001,
            size_y_m=0.001,
            size_z_m=0.001,
            local_envelope_xyz_m=env,
        ),
        layer4=d_inactive.layer4,
    )
    _r_inactive, shape_inactive = generate_blade_cad(d_inactive)
    _r_present, shape_present = generate_blade_cad(d_with_prim)
    assert shape_present.val().Volume() < shape_inactive.val().Volume()


def test_layer3_cad_failure_degrades_status_and_returns_envelope() -> None:
    """If apply_primitive raises, the wrapper degrades status to
    LAYER3_FAILED and returns the pre-Layer-3 envelope."""
    from fanopt.geometry import generator_cad

    original = generator_cad.apply_primitive

    def boom(_shape, _primitive):
        raise RuntimeError("simulated OpenCascade failure")

    generator_cad.apply_primitive = boom
    try:
        d = _canonical_design()
        result, shape = generate_blade_cad(d)
        assert result.status == GenerationStatus.LAYER3_FAILED
        # Shape volume equals the bare envelope's volume (no primitive applied).
        from fanopt.geometry.envelope_cad import make_outer_envelope

        env_only = make_outer_envelope(d.layer1, d.layer4.print_orientation)
        assert shape.val().Volume() == pytest.approx(env_only.val().Volume(), rel=1e-9)
    finally:
        generator_cad.apply_primitive = original
