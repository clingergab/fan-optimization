"""CadQuery-aware orchestration wrapper around the scaffold ``generate_blade``.

Hard-requires CadQuery. The scaffold in :mod:`fanopt.geometry.generator`
stays pure-Python (dict-descriptions only) so its module-level imports
don't pull in CadQuery; this wrapper adds the real-shape construction
on top.

Consumer surface (Phase 1 onward):

    >>> from fanopt.geometry.generator_cad import generate_blade_cad
    >>> result, shape = generate_blade_cad(design)
    >>> shape  # a cq.Workplane with the Layer 1 envelope

The scaffold result is unchanged from :func:`fanopt.geometry.generator.generate_blade`
so all the existing orchestration tests + the manufacturability filter
continue to apply.

Phase 1 follow-up: ``shape`` will progress through the Layer 2 / 3 /
``make_panel_solid`` pipeline (per plan §9.7.1) as the corresponding
CadQuery helpers land. Today the wrapper produces only the Layer 1
envelope; Layer 2 / 3 cuts and the click-feature write remain to do.
"""

from __future__ import annotations

import cadquery as cq

from fanopt.geometry.envelope_cad import make_outer_envelope
from fanopt.geometry.generator import (
    BladeDesignParams,
    GenerationResult,
    generate_blade,
)

__all__ = ["generate_blade_cad"]


def generate_blade_cad(
    params: BladeDesignParams,
) -> tuple[GenerationResult, cq.Workplane]:
    """Run the orchestration scaffold AND construct the Layer 1 envelope.

    Returns
    -------
    (GenerationResult, cq.Workplane)
        The scaffold result (status, layer descriptions, manufacturability
        protocol, panel-domain mask, params trace) AND the CadQuery
        Workplane holding the Layer 1 outer envelope solid.

    Notes
    -----
    The shape returned here is *only* the Layer 1 envelope — Layer 2
    fields, Layer 3 primitive, and the Layer 4 click chamfer/detent
    are not yet written into it. The scaffold's :class:`GenerationResult`
    still records the Layer 2 / 3 / 4 application metadata in its
    ``layer_descriptions``; the geometry-side application of those layers
    waits on the corresponding CadQuery helpers (apply_louver, etc.).
    """
    result = generate_blade(params)
    envelope = make_outer_envelope(params.layer1, print_orientation=params.layer4.print_orientation)
    return result, envelope
