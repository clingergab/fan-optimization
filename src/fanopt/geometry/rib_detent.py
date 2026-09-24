"""Open/closed HOLD detent placement — a row of seats along the overlapping rib.

The fan holds its open and closed positions with a soft nesting detent (a hold, not an audible
click — the spring is the whole floppy blade, ~grams of force). One row of **bumps** on the
overlapping rib's top face, plus **two recesses** on the neighbour-above's underside per bump:

- **CLOSED** (blades aligned): the bump sits directly under the neighbour's *same* rib → the
  neighbour's underside recess there catches it.
- **OPEN** (neighbour rotated one deploy pitch): the blade planform is ≈ one pitch wide, so a
  13.3° rotation slides the bump across to under the neighbour's *far* rib → a second recess catches it.

This module is the geometry only — where each bump and its two seats sit (radius + angle). It's
pure and design-independent in the *angular* layout (the planform is locked); the surface z at each
site is sampled at fusion time. Applying the features to a printable mesh (bumps by union, recesses
by boolean) lives elsewhere. See ``docs`` / the fan-review notebook for the print flow.
"""

from __future__ import annotations

import dataclasses
import math

from fanopt.geometry.blade import BladeParams, half_width_at
from fanopt.geometry.schema import INTER_BLADE_ANGLE_RAD

__all__ = ["DetentSeat", "rib_half_angle_deg", "rib_detent_seats"]

# Deploy fan-out direction the physical fan is assembled to open in (matches the fan-review notebook's
# DEPLOY_DIRECTION). It selects WHICH rib overlaps the neighbour, so the bump/recess must be built for
# the same direction the fan actually opens — a wrong-direction build puts the bump over open air.
DEFAULT_DEPLOY_DIRECTION: int = -1

# Usable radial band for DUAL (open+closed) seats. The CLOSED seat sits on the blade at any radius, but
# the OPEN seat only lands where the DEPLOYED overlap is wide enough to hold both recesses inside their
# ribs: 2*alpha >= pitch + 2*edge_margin. That overlap widens toward the hub, so dual-hold seats live
# inboard (~80-120 mm) — NOT out at the tip, where the overlap collapses to a sliver. Travel there is
# still 19-28 mm, ample for a soft nesting seat.
DEFAULT_R_MIN_M: float = 0.08
DEFAULT_R_MAX_M: float = 0.12
# Keep the bump/recess this far (as an angle) inside the physical rib edge so a real feature has material
# around it and does not fall off the tapered edge under FDM placement tolerance.
DEFAULT_EDGE_MARGIN_RAD: float = math.radians(1.2)


@dataclasses.dataclass(frozen=True)
class DetentSeat:
    """One hold site: a bump on the overlapping rib + its closed/open recesses on the neighbour underside.

    Angles are in the blade's own frame (degrees, 0 = radial centreline, sign follows +z rotation). The
    bump is on the LOWER blade's rib TOP; the two recess angles are where that bump lands on the
    neighbour-above's UNDERSIDE when the fan is closed vs open.
    """

    radius_m: float
    bump_angle_deg: float
    closed_recess_angle_deg: float
    open_recess_angle_deg: float
    travel_mm: float


def rib_half_angle_deg(r_m: float) -> float:
    """Angular half-width of the blade planform at radius ``r`` (deg): ``atan(half_width(r) / r)``.

    This is the angle from the radial centreline to a rib edge — it GROWS toward the hub (the blade
    subtends more angle there) and is ≈ half the deploy pitch at the tip (why the blade is ~one pitch wide).
    """
    return math.degrees(math.atan(half_width_at(r_m) / r_m))


def rib_detent_seats(
    params: BladeParams,
    *,
    deploy_direction: int = DEFAULT_DEPLOY_DIRECTION,
    n_seats: int = 5,
    r_min_m: float = DEFAULT_R_MIN_M,
    r_max_m: float = DEFAULT_R_MAX_M,
    edge_margin_rad: float = DEFAULT_EDGE_MARGIN_RAD,
) -> list[DetentSeat]:
    """Placements for the open/closed hold detent: ``n_seats`` bump+recess sites along the overlapping rib.

    ``deploy_direction`` (±1) picks which rib overlaps the neighbour and MUST match the assembled fan's
    open direction. Seats are spaced over ``[r_min_m, r_max_m]``. Raises if the band/count is invalid or a
    seat's open landing would fall off the neighbour (no overlap → cannot hold open there).
    """
    if deploy_direction not in (1, -1):
        raise ValueError("deploy_direction must be +1 or -1")
    if n_seats < 1:
        raise ValueError("n_seats must be >= 1")
    if not 0.0 < r_min_m <= r_max_m:
        raise ValueError("require 0 < r_min_m <= r_max_m")

    pitch_deg = math.degrees(INTER_BLADE_ANGLE_RAD)
    edge_deg = math.degrees(edge_margin_rad)
    radii = (
        [r_min_m]
        if n_seats == 1
        else [r_min_m + (r_max_m - r_min_m) * k / (n_seats - 1) for k in range(n_seats)]
    )

    seats: list[DetentSeat] = []
    for r in radii:
        alpha = rib_half_angle_deg(r)
        # Bump on the overlapping rib, a hair inside the edge. deploy_direction sets which rib: the one
        # whose OPEN landing (bump_angle - direction*pitch) stays on the neighbour planform.
        bump = deploy_direction * (alpha - edge_deg)
        closed = bump  # folded: bump lands under the neighbour's SAME rib
        open_ = bump - deploy_direction * pitch_deg  # open: neighbour rotated one pitch
        # The OPEN seat must sit inside the FAR rib by the same margin, else there's no room to hold open
        # at this radius (overlap too thin). This bites at the outer radii and defines the usable band.
        if abs(open_) > alpha - edge_deg:
            raise ValueError(
                f"overlap too thin at r={r * 1e3:.0f} mm to hold OPEN (open seat {open_:.2f} deg vs rib "
                f"{alpha:.2f} deg, margin {edge_deg:.2f}); move r_max_m inboard (dual-hold band is ~80-120 mm)"
            )
        seats.append(
            DetentSeat(
                radius_m=r,
                bump_angle_deg=bump,
                closed_recess_angle_deg=closed,
                open_recess_angle_deg=open_,
                travel_mm=r * math.radians(pitch_deg) * 1e3,
            )
        )
    return seats
