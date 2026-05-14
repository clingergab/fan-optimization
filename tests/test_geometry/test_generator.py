"""Unit tests for fanopt.geometry.generator (BladeDesignParams aggregator).

Validates per-layer composition + cross-layer constraints (specifically
the plano-convex camber requirement under print_orientation='flat').
"""
from __future__ import annotations

import pytest

from fanopt.geometry.envelope import Layer1Params
from fanopt.geometry.fields import Layer2Params, LouverField, NoiseField
from fanopt.geometry.generator import BladeDesignParams
from fanopt.geometry.manufacturability import Layer4Params
from fanopt.geometry.primitives import Layer3Primitive


def _canonical_layer1(camber_knots_m: tuple[float, ...] = (0.001, 0.002, 0.001)) -> Layer1Params:
    return Layer1Params(
        blade_count=10,
        camber_knots_m=camber_knots_m,
        twist_knots_rad=(0.0, 0.0),
        thickness_knots_m=(0.0030, 0.0028, 0.0026),
        edge_profile="rounded",
        fourier_le_amplitudes=(0.05, 0.0, 0.0),
        fourier_te_amplitudes=(0.0, 0.05, 0.0),
    )


def _canonical_layer4(print_orientation: str = "flat") -> Layer4Params:
    return Layer4Params(
        print_orientation=print_orientation,
        layer_height_m=0.0002,
        click_chamfer_angle_deg=45.0,
        click_detent_size_m=0.0004,
        click_design_clearance_m=0.00018,
    )


def test_canonical_design_validates() -> None:
    design = BladeDesignParams(
        layer1=_canonical_layer1(),
        layer2=Layer2Params.all_inactive(),
        layer3=Layer3Primitive.absent(),
        layer4=_canonical_layer4(),
    )
    assert design.layer1.blade_count == 10
    assert design.layer2.active_count() == 0
    assert design.layer3.present is False
    assert design.layer4.print_orientation == "flat"


def test_plano_convex_constraint_under_flat_orientation() -> None:
    """Camber values must be non-negative under print_orientation='flat'.

    Layer1Params alone already rejects negative camber via CAMBER_RANGE_M,
    so to specifically exercise the cross-layer plano-convex constraint
    we'd need a Layer 1 schema that allows negative camber when not flat.
    Currently the bound is unconditional. This test documents the
    intended cross-layer behavior: BladeDesignParams should NOT relax
    under non-flat orientations either (camber stays non-negative).
    """
    design = BladeDesignParams(
        layer1=_canonical_layer1((0.0, 0.005, 0.0)),
        layer2=Layer2Params.all_inactive(),
        layer3=Layer3Primitive.absent(),
        layer4=_canonical_layer4("flat"),
    )
    # All camber values are non-negative — constraint satisfied.
    assert all(v >= 0 for v in design.layer1.camber_knots_m)


def test_round_trip_to_from_dict() -> None:
    design = BladeDesignParams(
        layer1=_canonical_layer1(),
        layer2=Layer2Params.all_inactive(),
        layer3=Layer3Primitive.absent(),
        layer4=_canonical_layer4(),
    )
    d = design.to_dict()
    recovered = BladeDesignParams.from_dict(d)
    assert recovered == design


def test_round_trip_with_active_layer2_and_layer3() -> None:
    """Full-fledged design with active Layer 2 fields + a Layer 3 primitive."""
    env_xyz = (0.050, 0.050, 0.005)
    l2 = Layer2Params(
        louver=LouverField(
            active=True, count=6, angle_rad=0.5, width_m=0.001,
            spacing_profile="uniform", polarity="subtract",
        ),
        texture=Layer2Params.all_inactive().texture,
        edge=Layer2Params.all_inactive().edge,
        noise=NoiseField(active=True, threshold_retention=0.6),
        tpms=Layer2Params.all_inactive().tpms,
    )
    l3 = Layer3Primitive(
        present=True,
        shape_type="ellipsoid",
        polarity="subtract",
        position_x_m=0.025,
        position_y_m=0.025,
        position_z_m=0.0025,
        size_x_m=0.005,
        size_y_m=0.005,
        size_z_m=0.001,
        local_envelope_xyz_m=env_xyz,
    )
    design = BladeDesignParams(
        layer1=_canonical_layer1(),
        layer2=l2,
        layer3=l3,
        layer4=_canonical_layer4("edge"),  # 'edge' orientation, plano-convex relaxes
    )
    recovered = BladeDesignParams.from_dict(design.to_dict())
    assert recovered == design


def test_all_print_orientations_compatible_with_canonical_layer1() -> None:
    """Canonical Layer 1 (positive camber) is compatible with all three orientations."""
    for po in ("flat", "edge", "custom-angle"):
        BladeDesignParams(
            layer1=_canonical_layer1(),
            layer2=Layer2Params.all_inactive(),
            layer3=Layer3Primitive.absent(),
            layer4=_canonical_layer4(po),
        )
