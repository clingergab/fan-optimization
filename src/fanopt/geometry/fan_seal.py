"""Deployed-fan sealing analysis — how big the inter-blade gaps are, and thinning the ribs to close them.

When the fan deploys, adjacent blades sit one ``layer_spacing`` apart on the pin. The hub seals (the boss
is a full layer tall, so stacked bosses touch), but out along the span the thinner panel/rib underfills
the layer, leaving a z-gap the wind leaks through:  ``gap = layer_spacing − local_thickness``.

``layer_spacing`` is set by the **thickest structural section (the rib)** — the boss just follows it — so
thinning the rib toward the panel thickness pulls every gap down to ~the fold clearance. That trades a
little stiffness (the rib is a stiffener), not wind (the panel *shape* makes the wind). Pure numpy.
"""

from __future__ import annotations

import dataclasses

from fanopt.geometry.blade import (
    FOLD_CLEARANCE_M,
    RIB_THICKNESS_RANGE_M,
    BladeParams,
    blade_z_envelope_m,
    layer_spacing_m,
)

__all__ = [
    "panel_thickness_range_m",
    "deployed_panel_gap_m",
    "deployed_gap_range_m",
    "rib_containment_floor_m",
    "thin_ribs",
]


def deployed_gap_range_m(params: BladeParams, clearance_m: float | None = None) -> tuple[float, float]:
    """``(min_gap, max_gap)`` vertical slot between adjacent deployed blades (m), with a clearance override.

    Surface-of-revolution ⇒ the bow cancels and the slot is ``envelope + clearance − h_top − h_bot``.
    The MIN is where both facing surfaces are the thickest section (the rib) → the slot collapses to the
    ``clearance``. The MAX is where both are the thinnest panel → ``envelope + clearance − panel_min``.
    ``clearance_m`` defaults to the locked ``FOLD_CLEARANCE_M``; pass a smaller value to model a tighter
    fold (thinner boss). The envelope tracks ``params`` (so pass thinned-rib params to model the rib fix).
    """
    cl = FOLD_CLEARANCE_M if clearance_m is None else clearance_m
    env = blade_z_envelope_m(params)
    return cl, env + cl - panel_thickness_range_m(params)[0]


def panel_thickness_range_m(params: BladeParams) -> tuple[float, float]:
    """``(min, max)`` panel membrane thickness over the design's thickness grid."""
    vals = [t for row in params.panel_thickness_m for t in row]
    return min(vals), max(vals)


def deployed_panel_gap_m(params: BladeParams) -> float:
    """Worst inter-panel air gap between adjacent deployed blades (m).

    ``layer_spacing − thinnest panel`` — the tallest open slot the wind sees where two panels overlap
    (at the rib the gap is only the fold clearance; at the thin panel it is this). 0 would mean the
    panel fills the whole layer (no gap); the boss already does that at the hub.
    """
    return layer_spacing_m(params) - panel_thickness_range_m(params)[0]


def rib_containment_floor_m(params: BladeParams) -> float:
    """Minimum rib thickness that still **contains** the offset panel: ``max(t_panel + 2·|offset|)``.

    The rib rails sit at ``±t_rib/2``; the panel membrane sits at ``offset ± t_panel/2``. Containment
    needs ``t_rib/2 ≥ |offset| + t_panel/2`` at every grid node, i.e. ``t_rib ≥ t_panel + 2·|offset|``.
    Clamping the rib to the panel *thickness* alone (ignoring the mean-surface offset) drives containment
    negative — the panel pokes past the rib and the folded stack collides.
    """
    return max(
        t + 2.0 * abs(off)
        for orow, trow in zip(params.panel_offsets_m, params.panel_thickness_m, strict=True)
        for off, t in zip(orow, trow, strict=True)
    )


def thin_ribs(params: BladeParams, target_rib_m: float) -> BladeParams:
    """Return ``params`` with both rib-thickness knots reduced to ``target_rib_m`` (never increased).

    Clamped to ``[rib_containment_floor_m, current rib]`` so the rib still **contains** the offset panel
    (``t_panel + 2·|offset|``, NOT just ``t_panel``) and never grows. Reducing the rib lowers
    ``layer_spacing`` → closes the deployed gaps and thins the fold; the boss follows automatically.
    Re-verify structure/fold after — this trades rib stiffness (there is margin) for sealing.
    """
    floor = max(rib_containment_floor_m(params), RIB_THICKNESS_RANGE_M[0])
    target = max(floor, min(target_rib_m, params.t_rib_hub_m, params.t_rib_tip_m))
    return dataclasses.replace(params, t_rib_hub_m=target, t_rib_tip_m=target)
