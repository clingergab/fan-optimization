"""Top-level blade-design parameter aggregator + orchestration scaffold.

Two public surfaces:

1. :class:`BladeDesignParams` — the 4-layer nested schema with cross-layer
   validation (plano-convex camber under rib-flat).

2. :func:`generate_blade` — pure-Python orchestration scaffold that lays
   out the deterministic Layer 1 → 2 → 3 → 4 application order, the
   panel-domain mask (PANEL_PIVOT_REGION + CLICK_FOOTPRINT exclusions),
   and the Layer 3 try/except wrapping per plan §9.7. Returns a
   :class:`GenerationResult` carrying a JSON-friendly geometry
   description. Real CadQuery shape construction lands in Phase 1;
   today's scaffold returns dict-descriptions so the orchestration
   logic is testable + freezable now.

Reference: plan §9.7.1 generator orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from fanopt.geometry.envelope import Layer1Params
from fanopt.geometry.fields import Layer2Params
from fanopt.geometry.manufacturability import (
    Layer4Params,
    ManufacturabilityResult,
    run_manufacturability_filter,
)
from fanopt.geometry.primitives import Layer3Primitive
from fanopt.geometry.schema import (
    CLICK_FOOTPRINT_X_RANGE_M,
    PANEL_PIVOT_REGION,
    click_footprint_y_range_panel_edge_m,
)

__all__ = [
    "BladeDesignParams",
    "GeneratorVersion",
    "GenerationStatus",
    "LayerDescription",
    "GenerationResult",
    "panel_domain_mask_description",
    "generate_blade",
]


GeneratorVersion: str = "0.1.0-scaffold"
"""Version stamp on every GenerationResult — bumps when the orchestration
contract changes (NOT when individual CadQuery helpers swap in)."""


@dataclass(frozen=True)
class BladeDesignParams:
    """One blade's full design parameter set across all 4 layers.

    Each layer validates its own bounds in ``__post_init__``. This
    aggregator additionally enforces cross-layer constraints.
    """

    layer1: Layer1Params
    layer2: Layer2Params
    layer3: Layer3Primitive
    layer4: Layer4Params

    def __post_init__(self) -> None:
        if self.layer4.print_orientation == "flat":
            for i, v in enumerate(self.layer1.camber_knots_m):
                if (
                    v < 0
                ):  # pragma: no cover -- defense in depth; Layer1Params CAMBER_RANGE_M already rejects negative camber upstream
                    raise ValueError(
                        f"BladeDesignParams: plano-convex constraint violated — "
                        f"camber_knots_m[{i}] = {v} < 0 under "
                        f"print_orientation='flat' (rib-flat). The bottom "
                        f"face is planar; camber values must be non-negative."
                    )

    def to_dict(self) -> dict[str, Any]:
        """Full JSON-friendly serialisation across all four layers."""
        return {
            "layer1": self.layer1.to_dict(),
            "layer2": self.layer2.to_dict(),
            "layer3": self.layer3.to_dict(),
            "layer4": self.layer4.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BladeDesignParams:
        return cls(
            layer1=Layer1Params.from_dict(d["layer1"]),
            layer2=Layer2Params.from_dict(d["layer2"]),
            layer3=Layer3Primitive.from_dict(d["layer3"]),
            layer4=Layer4Params.from_dict(d["layer4"]),
        )


# ---------------------------------------------------------------------------
# Orchestration scaffold
# ---------------------------------------------------------------------------


class GenerationStatus(str, Enum):
    """Final state of :func:`generate_blade`.

    ``OK``                — all four layers applied; design passes the
                            manufacturability filter (score >= 0.5).
    ``LAYER3_FAILED``     — Layer 3 raised; the orchestrator caught it,
                            kept the Layer-2-carved geometry, and emitted
                            a warning. Per plan §9.7: Layer 3 is the only
                            layer where CAD failures are tolerated. The
                            design is otherwise valid and may still pass.
    ``MFG_REJECTED``      — manufacturability filter score < 0.5
                            (infeasible — BO sees J_fan=0, mass=∞).
    """

    OK = "ok"
    LAYER3_FAILED = "layer3_failed"
    MFG_REJECTED = "mfg_rejected"


@dataclass(frozen=True)
class LayerDescription:
    """JSON-friendly per-layer description emitted by the scaffold.

    The real CadQuery generator will replace ``description`` with a
    shape handle; today the dict carries the parameter trace the
    eventual helper would consume.
    """

    layer_index: int
    applied: bool
    description: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "applied": self.applied,
            "description": self.description,
            "error": self.error,
        }


@dataclass(frozen=True)
class GenerationResult:
    """Aggregate output of :func:`generate_blade`."""

    status: GenerationStatus
    generator_version: str
    layer_descriptions: tuple[LayerDescription, ...]
    panel_domain_mask: dict[str, Any]
    manufacturability: ManufacturabilityResult
    params: BladeDesignParams = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "generator_version": self.generator_version,
            "layers": [ld.to_dict() for ld in self.layer_descriptions],
            "panel_domain_mask": self.panel_domain_mask,
            "manufacturability": self.manufacturability.to_dict(),
            "params": self.params.to_dict(),
        }


# ---------------------------------------------------------------------------
# Panel-domain mask — plan §9.7.1 Step 0 + §9.7.3 Check 6/7
# ---------------------------------------------------------------------------


def panel_domain_mask_description(blade_count: int) -> dict[str, Any]:
    """Serialise the panel-domain mask (preserved regions) for blade_count.

    The §9.7.1 Step 0 panel-domain mask excludes Layer 2/3 carving from
    two regions:

    - **PANEL_PIVOT_REGION** — 7 mm-radius circular keep-out around the
      pivot at (PIVOT_CENTER_X_M, 0). Covers the 12 mm boss + 1 mm
      clearance.
    - **CLICK_FOOTPRINT_PANEL_EDGE_REGION** — 5 × 5 mm rectangle at the
      panel's outer tangential edge at the tip, ``CLICK_FOOTPRINT_X_RANGE_M
      × CLICK_FOOTPRINT_Y_RANGE_PANEL_EDGE``.

    Returned dict is a stable contract the eventual CadQuery generator
    consumes when carving panels.
    """
    cf_y = click_footprint_y_range_panel_edge_m(blade_count=blade_count)
    return {
        "panel_pivot_region": {
            "kind": "circular_mask",
            "center_x_m": PANEL_PIVOT_REGION.center_x_m,
            "center_y_m": PANEL_PIVOT_REGION.center_y_m,
            "radius_m": PANEL_PIVOT_REGION.radius_m,
        },
        "click_footprint_panel_edge_region": {
            "kind": "rectangular_mask",
            "x_range_m": list(CLICK_FOOTPRINT_X_RANGE_M),
            "y_range_panel_edge_m": list(cf_y),
        },
    }


# ---------------------------------------------------------------------------
# Per-layer scaffold helpers — each returns a dict description, no CadQuery
# ---------------------------------------------------------------------------


def _apply_layer1_envelope(params: Layer1Params) -> dict[str, Any]:
    """Stub for Layer 1 envelope construction.

    Real implementation (Phase 1) will call CadQuery to build the
    Layer 1 plano-convex envelope. Today returns the parameter trace.
    """
    return {
        "kind": "outer_envelope",
        "blade_count": params.blade_count,
        "edge_profile": params.edge_profile,
        "camber_knot_count": len(params.camber_knots_m),
        "twist_knot_count": len(params.twist_knots_rad),
        "thickness_knot_count": len(params.thickness_knots_m),
        "fourier_le_amplitude_max_abs": max(abs(a) for a in params.fourier_le_amplitudes),
        "fourier_te_amplitude_max_abs": max(abs(a) for a in params.fourier_te_amplitudes),
    }


def _apply_layer2_fields(params: Layer2Params) -> dict[str, Any]:
    """Stub for Layer 2 field application.

    Plan §9.7: fields are applied in a fixed order to avoid order-
    dependent CAD edge cases: TPMS → noise → louver → texture → edge.
    Today the scaffold records the active fields in that order.
    """
    fixed_order: tuple[tuple[str, Any], ...] = (
        ("tpms", params.tpms),
        ("noise", params.noise),
        ("louver", params.louver),
        ("texture", params.texture),
        ("edge", params.edge),
    )
    applied = [name for name, f in fixed_order if f.active]
    return {
        "kind": "field_layer",
        "application_order": [name for name, _ in fixed_order],
        "applied_fields": applied,
        "active_count": params.active_count(),
    }


def _apply_layer3_primitive(params: Layer3Primitive) -> dict[str, Any]:
    """Stub for Layer 3 primitive application.

    Plan §9.7: this is the only step where CAD failures are tolerated.
    The orchestrator wraps the call in try/except; failures degrade the
    GenerationStatus to ``LAYER3_FAILED`` rather than aborting.
    """
    if not params.present:
        return {"kind": "primitive", "applied": False, "reason": "params.present=False"}
    return {
        "kind": "primitive",
        "applied": True,
        "shape_type": params.shape_type,
        "polarity": params.polarity,
        "position_m": [params.position_x_m, params.position_y_m, params.position_z_m],
        "size_m": [params.size_x_m, params.size_y_m, params.size_z_m],
    }


def _apply_layer4_features(params: Layer4Params) -> dict[str, Any]:
    """Stub for Layer 4 manufacturing-feature application.

    The Layer 4 panel-edge click chamfer + detent live in the
    CLICK_FOOTPRINT_X_RANGE × CLICK_FOOTPRINT_Y_RANGE_PANEL_EDGE region;
    they are written into the panel by ``make_panel_solid`` (Phase 1).
    Today the scaffold records the manufacturing parameters that drive
    that step.
    """
    return {
        "kind": "manufacturing_features",
        "print_orientation": params.print_orientation,
        "layer_height_m": params.layer_height_m,
        "click_chamfer_angle_deg": params.click_chamfer_angle_deg,
        "click_detent_size_m": params.click_detent_size_m,
        "click_design_clearance_m": params.click_design_clearance_m,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def generate_blade(params: BladeDesignParams) -> GenerationResult:
    """Run the 4-layer hybrid generator on a validated parameter set.

    Order: Layer 1 (envelope) → Layer 2 (fields, fixed sub-order) →
    Layer 3 (capped primitive, wrapped in try/except) → Layer 4
    (manufacturing features). The panel-domain mask is precomputed and
    carried through to the manufacturability filter.

    Phase-0 scaffolding: each layer helper returns a dict description
    rather than a CadQuery shape; the real generator swaps the helpers
    when Phase 1 lands.

    Returns
    -------
    A :class:`GenerationResult` with the per-layer descriptions, the
    panel-domain mask, the manufacturability filter outcome, and the
    final status.
    """
    layer_descs: list[LayerDescription] = []

    # Layer 1
    l1_desc = _apply_layer1_envelope(params.layer1)
    layer_descs.append(LayerDescription(1, True, l1_desc))

    # Layer 2
    l2_desc = _apply_layer2_fields(params.layer2)
    layer_descs.append(LayerDescription(2, True, l2_desc))

    # Layer 3 — wrapped in try/except per plan §9.7 (the only layer
    # where CAD failures are tolerated). Failures degrade status but
    # don't abort the pipeline.
    layer3_failed = False
    try:
        l3_desc = _apply_layer3_primitive(params.layer3)
        layer_descs.append(LayerDescription(3, l3_desc.get("applied", False), l3_desc))
    except Exception as e:  # pragma: no cover -- scaffold helpers don't raise yet
        layer3_failed = True
        layer_descs.append(
            LayerDescription(
                3,
                applied=False,
                description={"kind": "primitive", "applied": False},
                error=f"{type(e).__name__}: {e}",
            )
        )

    # Layer 4
    l4_desc = _apply_layer4_features(params.layer4)
    layer_descs.append(LayerDescription(4, True, l4_desc))

    panel_mask = panel_domain_mask_description(params.layer1.blade_count)

    # Run the manufacturability filter on the assembled description.
    geometry_description = {
        "layers": [ld.to_dict() for ld in layer_descs],
        "panel_domain_mask": panel_mask,
        "params": params.to_dict(),
    }
    mfg = run_manufacturability_filter(geometry_description)

    if layer3_failed:
        status = GenerationStatus.LAYER3_FAILED
    elif not mfg.passed:
        status = GenerationStatus.MFG_REJECTED
    else:
        status = GenerationStatus.OK

    return GenerationResult(
        status=status,
        generator_version=GeneratorVersion,
        layer_descriptions=tuple(layer_descs),
        panel_domain_mask=panel_mask,
        manufacturability=mfg,
        params=params,
    )
