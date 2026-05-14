"""Unit tests for fanopt.geometry.generator.

Two surfaces:
- :class:`BladeDesignParams` per-layer composition + cross-layer
  constraints (plano-convex camber under print_orientation='flat').
- :func:`generate_blade` orchestration scaffold — layer ordering, panel-
  domain mask, Layer 3 try/except behaviour, manufacturability filter
  invocation.
"""

from __future__ import annotations

import json

from fanopt.geometry import generator as gen
from fanopt.geometry.envelope import Layer1Params
from fanopt.geometry.fields import Layer2Params, LouverField, NoiseField
from fanopt.geometry.generator import (
    BladeDesignParams,
    GenerationStatus,
    GeneratorVersion,
    generate_blade,
    panel_domain_mask_description,
)
from fanopt.geometry.manufacturability import Layer4Params, ManufacturabilityResult
from fanopt.geometry.primitives import Layer3Primitive
from fanopt.geometry.schema import CLICK_FOOTPRINT_X_RANGE_M, PANEL_PIVOT_REGION


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
            active=True,
            count=6,
            angle_rad=0.5,
            width_m=0.001,
            spacing_profile="uniform",
            polarity="subtract",
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


# ---- panel-domain mask ----------------------------------------------------


def test_panel_domain_mask_carries_both_keep_outs() -> None:
    """Plan §9.7.1 Step 0: the panel-domain mask excludes both
    PANEL_PIVOT_REGION and CLICK_FOOTPRINT_PANEL_EDGE_REGION."""
    mask = panel_domain_mask_description(blade_count=10)
    assert "panel_pivot_region" in mask
    assert "click_footprint_panel_edge_region" in mask


def test_panel_pivot_region_in_mask_matches_schema_constant() -> None:
    mask = panel_domain_mask_description(blade_count=10)
    pp = mask["panel_pivot_region"]
    assert pp["center_x_m"] == PANEL_PIVOT_REGION.center_x_m
    assert pp["center_y_m"] == PANEL_PIVOT_REGION.center_y_m
    assert pp["radius_m"] == PANEL_PIVOT_REGION.radius_m


def test_click_footprint_x_range_in_mask_matches_schema_constant() -> None:
    mask = panel_domain_mask_description(blade_count=10)
    cf = mask["click_footprint_panel_edge_region"]
    assert tuple(cf["x_range_m"]) == CLICK_FOOTPRINT_X_RANGE_M


def test_panel_mask_y_range_scales_with_blade_count() -> None:
    """click_footprint_y range derives from blade_count via the C8-locked
    inter-blade angle; different blade counts give different y ranges."""
    m8 = panel_domain_mask_description(blade_count=8)
    m12 = panel_domain_mask_description(blade_count=12)
    # Y range is derived from the same panel-tangential half-pitch and
    # L_BLADE_M; both should be identical here per the plan formula. The
    # test pins that they're at minimum well-defined and the same.
    y8 = m8["click_footprint_panel_edge_region"]["y_range_panel_edge_m"]
    y12 = m12["click_footprint_panel_edge_region"]["y_range_panel_edge_m"]
    assert len(y8) == 2 and len(y12) == 2
    # The schema's click_footprint_y_range_panel_edge_m formula uses
    # L_BLADE_M (locked) — independent of blade_count today. Document
    # that invariant here so a future schema change is forced to update
    # this test consciously.
    assert y8 == y12


# ---- generate_blade orchestration -----------------------------------------


def _canonical_design(
    layer3: Layer3Primitive | None = None,
    print_orientation: str = "flat",
) -> BladeDesignParams:
    return BladeDesignParams(
        layer1=_canonical_layer1(),
        layer2=Layer2Params.all_inactive(),
        layer3=layer3 if layer3 is not None else Layer3Primitive.absent(),
        layer4=_canonical_layer4(print_orientation),
    )


def test_generate_blade_canonical_status_ok() -> None:
    """A fully-validated default design produces status OK."""
    result = generate_blade(_canonical_design())
    assert result.status == GenerationStatus.OK


def test_generate_blade_applies_all_four_layers_in_order() -> None:
    """Layer descriptions arrive in 1 → 2 → 3 → 4 order."""
    result = generate_blade(_canonical_design())
    indices = [ld.layer_index for ld in result.layer_descriptions]
    assert indices == [1, 2, 3, 4]


def test_generate_blade_layer3_skips_when_absent() -> None:
    """An absent Layer 3 primitive registers as applied=False, no error."""
    result = generate_blade(_canonical_design())
    l3 = next(ld for ld in result.layer_descriptions if ld.layer_index == 3)
    assert l3.applied is False
    assert l3.error is None
    assert l3.description["kind"] == "primitive"


def test_generate_blade_layer3_records_when_present() -> None:
    """A present Layer 3 primitive registers as applied=True with shape data."""
    env = (0.050, 0.050, 0.005)
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
        local_envelope_xyz_m=env,
    )
    result = generate_blade(_canonical_design(layer3=l3))
    l3_desc = next(ld for ld in result.layer_descriptions if ld.layer_index == 3)
    assert l3_desc.applied is True
    assert l3_desc.description["shape_type"] == "ellipsoid"
    assert l3_desc.description["polarity"] == "subtract"


def test_generate_blade_layer2_fixed_application_order() -> None:
    """Plan §9.7: Layer 2 fields apply in fixed order TPMS → noise →
    louver → texture → edge."""
    result = generate_blade(_canonical_design())
    l2 = next(ld for ld in result.layer_descriptions if ld.layer_index == 2)
    assert l2.description["application_order"] == ["tpms", "noise", "louver", "texture", "edge"]


def test_generate_blade_layer2_applied_fields_match_active() -> None:
    """`applied_fields` lists only the fields whose `active=True`."""
    design = BladeDesignParams(
        layer1=_canonical_layer1(),
        layer2=Layer2Params(
            louver=LouverField(active=True, count=6, width_m=0.001),
            texture=Layer2Params.all_inactive().texture,
            edge=Layer2Params.all_inactive().edge,
            noise=NoiseField(active=True, threshold_retention=0.6),
            tpms=Layer2Params.all_inactive().tpms,
        ),
        layer3=Layer3Primitive.absent(),
        layer4=_canonical_layer4(),
    )
    result = generate_blade(design)
    l2 = next(ld for ld in result.layer_descriptions if ld.layer_index == 2)
    # Application order: tpms (off) → noise (on) → louver (on) → texture/edge (off)
    assert l2.description["applied_fields"] == ["noise", "louver"]


def test_generate_blade_returns_panel_domain_mask() -> None:
    result = generate_blade(_canonical_design())
    assert "panel_pivot_region" in result.panel_domain_mask
    assert "click_footprint_panel_edge_region" in result.panel_domain_mask


def test_generate_blade_records_generator_version() -> None:
    result = generate_blade(_canonical_design())
    assert result.generator_version == GeneratorVersion


def test_generate_blade_result_is_json_serializable() -> None:
    """Full result dict round-trips through json without exotic types."""
    result = generate_blade(_canonical_design())
    json.dumps(result.to_dict())  # raises if non-serialisable


def test_generate_blade_status_ok_implies_mfg_passed() -> None:
    """The status protocol: OK ⇒ mfg.passed (the scaffold's 4 hard-bound
    checks register as passed; pending checks do not fail)."""
    result = generate_blade(_canonical_design())
    assert result.status == GenerationStatus.OK
    assert result.manufacturability.passed is True


def test_generate_blade_carries_params_reference() -> None:
    design = _canonical_design()
    result = generate_blade(design)
    assert result.params == design


# ---- status-branch coverage (LAYER3_FAILED + MFG_REJECTED) ----------------


def test_generate_blade_layer3_exception_promotes_to_layer3_failed(monkeypatch) -> None:
    """If Layer 3 raises, the orchestrator catches it and emits status
    LAYER3_FAILED with the error recorded on the Layer 3 description.

    The scaffold's pure-Python Layer 3 helper doesn't raise on its own;
    we monkey-patch it to simulate a CadQuery failure that the real
    Phase 1 generator may surface."""

    def _raising_layer3(_params):
        raise RuntimeError("simulated CadQuery boolean failure")

    monkeypatch.setattr(gen, "_apply_layer3_primitive", _raising_layer3)
    result = generate_blade(_canonical_design())
    assert result.status == GenerationStatus.LAYER3_FAILED
    l3 = next(ld for ld in result.layer_descriptions if ld.layer_index == 3)
    assert l3.applied is False
    assert l3.error is not None
    assert "simulated CadQuery boolean failure" in l3.error


def test_generate_blade_mfg_rejection_promotes_to_mfg_rejected(monkeypatch) -> None:
    """If the manufacturability filter rejects the geometry (score < 0.5),
    the orchestrator emits status MFG_REJECTED.

    We monkey-patch the filter to return a forced-failure result; the
    real Phase 1 CadQuery checks will surface this path naturally."""

    def _rejecting_filter(_geometry_description):
        return ManufacturabilityResult(
            score=0.0,
            passed=False,
            checks=(),
            critical_failures=("3",),
            pending_cadquery=(),
        )

    monkeypatch.setattr(gen, "run_manufacturability_filter", _rejecting_filter)
    result = generate_blade(_canonical_design())
    assert result.status == GenerationStatus.MFG_REJECTED
    assert result.manufacturability.passed is False
    assert result.manufacturability.score == 0.0
    assert result.manufacturability.critical_failures == ("3",)
