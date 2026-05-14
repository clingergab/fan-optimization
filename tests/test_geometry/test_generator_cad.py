"""Tests for fanopt.geometry.generator_cad (CadQuery wrapper)."""

from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("cadquery") is None:
    pytest.skip("cadquery not installed", allow_module_level=True)

import cadquery as cq

from fanopt.geometry.envelope import Layer1Params
from fanopt.geometry.fields import Layer2Params
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
            thickness_knots_m=(0.0030, 0.0028, 0.0026),
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


def test_scaffold_result_unchanged_from_generate_blade() -> None:
    """The first element of the returned tuple is exactly what the
    scaffold's ``generate_blade`` would have returned standalone."""
    design = _canonical_design()
    scaffold_only = generate_blade(design)
    wrapped, _shape = generate_blade_cad(design)
    assert wrapped == scaffold_only


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
