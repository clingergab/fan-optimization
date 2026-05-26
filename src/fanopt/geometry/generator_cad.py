"""CadQuery-aware orchestration wrapper around the scaffold ``generate_blade``.

Hard-requires CadQuery. The scaffold in :mod:`fanopt.geometry.generator`
stays pure-Python (dict-descriptions only) so its module-level imports
don't pull in CadQuery; this wrapper adds the real-shape construction
on top.

Consumer surface (Phase 1 onward):

    >>> from fanopt.geometry.generator_cad import generate_blade_cad
    >>> result, shape = generate_blade_cad(design)
    >>> shape  # cq.Workplane: envelope ∘ Layer 2 fields ∘ Layer 3 primitive

Pipeline order matches the scaffold (plan §9.7): Layer 1 envelope →
Layer 2 fields (locked sub-order TPMS → noise → louver → texture →
edge) → Layer 3 primitive. Layer 4 click chamfer + detent land in the
V-unit assembly composition (assembly_cad).

Per plan §9.7, Layer 3 is the only step where CadQuery / OpenCascade
failures are tolerated. This wrapper runs ``apply_primitive`` in a
try/except: on failure the returned shape is the pre-Layer-3 shape
(envelope ∘ Layer 2) and the scaffold result's status is overridden to
:attr:`GenerationStatus.LAYER3_FAILED`. Layer 2 application is
expected to succeed (safe-by-construction per the schema bounds) and
is NOT wrapped.
"""

from __future__ import annotations

import dataclasses

import cadquery as cq

from fanopt.geometry.envelope_cad import make_outer_envelope
from fanopt.geometry.fields_cad import apply_layer2_fields
from fanopt.geometry.generator import (
    BladeDesignParams,
    GenerationResult,
    GenerationStatus,
    generate_blade,
)
from fanopt.geometry.manufacturability_cad import run_manufacturability_filter_cad
from fanopt.geometry.primitives_cad import apply_primitive

__all__ = ["generate_blade_cad"]


def generate_blade_cad(
    params: BladeDesignParams,
) -> tuple[GenerationResult, cq.Workplane]:
    """Run the orchestration scaffold AND construct the CadQuery shape.

    Returns
    -------
    (GenerationResult, cq.Workplane)
        The scaffold result (status, layer descriptions, manufacturability
        protocol, panel-domain mask, params trace) AND the CadQuery
        Workplane holding ``envelope ∘ Layer 2 fields ∘ Layer 3 primitive``.
        If Layer 3 raises during CadQuery construction, the returned shape
        is the pre-Layer-3 shape and the result's ``status`` is overridden
        to ``LAYER3_FAILED``.

    Notes
    -----
    Layer 4 click chamfer + detent are applied as part of the V-unit
    blade composition (``assembly_cad.make_vunit_blade``), not in this
    panel-only generator.
    """
    result = generate_blade(params)
    shape = make_outer_envelope(
        params.layer1, print_orientation=params.layer4.print_orientation
    )

    shape = apply_layer2_fields(shape, params.layer2)
    pre_layer3 = shape

    layer3_failed = False
    try:
        shape = apply_primitive(shape, params.layer3)
    except Exception:
        # Plan §9.7: Layer 3 is the only step where CAD failures are
        # tolerated. Degrade status; return the pre-Layer-3 shape.
        layer3_failed = True
        shape = pre_layer3

    mfg_cad = run_manufacturability_filter_cad(shape, params.layer4)

    if layer3_failed:
        status = GenerationStatus.LAYER3_FAILED
    elif not mfg_cad.passed:
        status = GenerationStatus.MFG_REJECTED
    else:
        status = GenerationStatus.OK

    result = dataclasses.replace(result, status=status, manufacturability=mfg_cad)
    return result, shape
