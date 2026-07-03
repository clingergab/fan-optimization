"""Rotational-inertia objective — I_wrist of a deployed fan (Phase 4 Pareto axis).

Injected as ``inertia_fn`` into :func:`fanopt.bo.objective.evaluate_design`. Builds
the full deployed-fan CadQuery geometry for a design's Layer-1 parameters (with
neutral Layer-2/3/4 — the panel-shape search does not touch those here) and sums
each blade's moment of inertia about the wrist (+y) axis. Requires CadQuery; the
module imports it unconditionally at the top per CLAUDE.md §4.1, so importing
this module in a CadQuery-less environment fails cleanly and immediately.
"""

from __future__ import annotations

from fanopt.geometry.envelope import Layer1Params
from fanopt.geometry.fan_assembly import compute_i_wrist_kgm2, deploy_fan
from fanopt.geometry.fields import Layer2Params
from fanopt.geometry.generator import BladeDesignParams
from fanopt.geometry.manufacturability import Layer4Params
from fanopt.geometry.primitives import Layer3Primitive

__all__ = ["NEUTRAL_LAYER4", "fan_i_wrist_kgm2"]

# Neutral print/click manufacturing params (matches the Phase-1 smoke baseline);
# they do not enter the panel-shape aero search but are needed to build geometry.
NEUTRAL_LAYER4 = Layer4Params(
    print_orientation="flat",
    layer_height_m=0.0002,
    click_chamfer_angle_deg=45.0,
    click_detent_size_m=0.0004,
    click_design_clearance_m=0.00018,
)


def fan_i_wrist_kgm2(layer1: Layer1Params) -> float:
    """Total I_wrist (kg·m²) of the deployed fan for these Layer-1 parameters.

    Sum of each deployed blade's inertia about the wrist axis (parallel-axis
    handled per-blade inside :func:`compute_i_wrist_kgm2`).
    """
    params = BladeDesignParams(
        layer1=layer1,
        layer2=Layer2Params.all_inactive(),
        layer3=Layer3Primitive.absent(),
        layer4=NEUTRAL_LAYER4,
    )
    _, per_blade = deploy_fan(params)
    return float(sum(compute_i_wrist_kgm2(blade) for blade in per_blade))
