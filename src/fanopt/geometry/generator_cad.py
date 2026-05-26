"""CadQuery-aware orchestration wrapper around the scaffold ``generate_blade``.

Hard-requires CadQuery. The scaffold in :mod:`fanopt.geometry.generator`
stays pure-Python (dict-descriptions only) so its module-level imports
don't pull in CadQuery; this wrapper adds the real-shape construction
on top.

Consumer surface (Phase 1 onward):

    >>> from fanopt.geometry.generator_cad import generate_blade_cad
    >>> result, shape = generate_blade_cad(design)
    >>> shape  # a cq.Workplane with the Layer 1 envelope + Layer 3 cut/fuse

Per plan §9.7, Layer 3 is the only step where CadQuery / OpenCascade
failures are tolerated. This wrapper runs ``apply_primitive`` in a
try/except: on failure the returned shape is the pre-Layer-3 envelope
and the scaffold result's status is overridden to
:attr:`GenerationStatus.LAYER3_FAILED` (preserving the rest of the
scaffold's metadata).

Phase 1 follow-up: Layer 2 field application + the click chamfer /
detent + the V-unit blade composition land in subsequent modules.
This wrapper today produces envelope ∘ Layer 3.
"""

from __future__ import annotations

import dataclasses

import cadquery as cq

from fanopt.geometry.envelope_cad import make_outer_envelope
from fanopt.geometry.generator import (
    BladeDesignParams,
    GenerationResult,
    GenerationStatus,
    generate_blade,
)
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
        Workplane holding the Layer 1 envelope with the Layer 3 primitive
        Boolean-applied. If Layer 3 raises during CadQuery construction,
        the returned shape is the pre-Layer-3 envelope and the result's
        ``status`` is overridden to ``LAYER3_FAILED``.

    Notes
    -----
    The shape returned today is envelope ∘ Layer 3. Layer 2 field
    application (fields_cad) and Layer 4 click features (assembly_cad)
    land in subsequent Phase-1 modules. The scaffold result still
    records the per-layer metadata for the layers not yet applied.
    """
    result = generate_blade(params)
    envelope = make_outer_envelope(
        params.layer1, print_orientation=params.layer4.print_orientation
    )

    try:
        shape = apply_primitive(envelope, params.layer3)
    except Exception:
        # Plan §9.7: Layer 3 is the only step where CAD failures are
        # tolerated. Degrade status; return the pre-Layer-3 envelope.
        result = dataclasses.replace(result, status=GenerationStatus.LAYER3_FAILED)
        shape = envelope

    return result, shape
