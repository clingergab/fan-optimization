"""Size the open/closed hold detent on the real blade surfaces: one bump + two recesses per seat.

:func:`fanopt.geometry.rib_detent.rib_detent_seats` places each seat (radius + bump / closed / open
angles). This module samples the design surface there and turns each seat into solids (CAD frame, m):

- a **bump** — a spherical dome (fixed sphere radius) on the blade's TOP face at the bump angle, tall
  enough to cross the gap to the neighbour above plus ``interference_m``: the neighbour must flex up
  that much to slide past, which is the hold. Where the blades are thicker than the layer spacing the
  "gap" is negative: the blades press together there, so it counts as zero (the bump stands
  ``interference_m`` proud of the contact);
- two **recesses** — spherical dimples in the blade's UNDERSIDE at the closed and open angles, each deep
  enough that the neighbour-below's bump drops in with ``seat_clearance_m`` to spare.

Every blade is identical, so each one carries both: bumps for the blade above, recesses for the blade
below. The gap at a seat is ``layer_spacing + z_bottom(landing) − z_top(bump)`` (folded and open fans keep
the same z-layers; opening only rotates each blade one pitch about the pin).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fanopt.geometry.rib_detent import DetentSeat

__all__ = [
    "DetentSpec",
    "DetentGeometry",
    "DEFAULT_DETENT_SPEC",
    "surface_z_range",
    "detent_geometry",
]

_SPHERE_FLOOR_MARGIN_M = 0.05e-3  # keep the buried sphere off the underside (no sliver faces)


@dataclass(frozen=True)
class DetentSpec:
    """Feature sizing (metres)."""

    # The first printed bumps (Ø3 mm base, 0.9–1.1 mm tall) were cut from a ~1.6 mm sphere; keeping
    # the sphere fixed keeps the tested dome shape whatever the bump height.
    bump_radius_m: float = 1.6e-3
    # Bump height above the blade face where the blades touch: ~1 mm, the size of the first printed bumps
    # (the operator's choice); the recess is cut to take all of it.
    interference_m: float = 1.0e-3
    # Recess depth past the bump: the hold is the bump height it must climb; extra depth only adds play.
    seat_clearance_m: float = 0.05e-3
    recess_radial_clearance_m: float = 0.15e-3
    min_wall_m: float = 0.8e-3


DEFAULT_DETENT_SPEC = DetentSpec()


@dataclass(frozen=True)
class DetentGeometry:
    """Spheres as ``(N, 4)`` rows ``[x, y, z, radius]`` (CAD frame, m) + a per-seat report."""

    bumps: np.ndarray
    recesses: np.ndarray
    seats: list[dict[str, float]]


def surface_z_range(
    nodes: np.ndarray, facets: np.ndarray, x: float, y: float, search_m: float = 3.0e-3
) -> tuple[float, float]:
    """``(z_bottom, z_top)`` where the vertical line through ``(x, y)`` crosses the boundary surface.

    Facets nearly parallel to z (side walls, sliver projections) are skipped. Raises if the line misses.
    """
    c = nodes[facets].mean(axis=1)
    tri = nodes[facets[np.hypot(c[:, 0] - x, c[:, 1] - y) < search_m]]
    a, b, d = tri[:, 0], tri[:, 1], tri[:, 2]
    v0, v1 = b[:, :2] - a[:, :2], d[:, :2] - a[:, :2]
    v2 = np.array([x, y]) - a[:, :2]
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    ok = np.abs(den) > 1e-3 * d00 * d11
    safe = np.where(ok, den, 1.0)
    v = np.where(ok, (d11 * d20 - d01 * d21) / safe, -1.0)
    w = np.where(ok, (d00 * d21 - d01 * d20) / safe, -1.0)
    u = 1.0 - v - w
    hit = (u >= -1e-9) & (v >= -1e-9) & (w >= -1e-9)
    if not hit.any():
        raise ValueError(f"no surface above/below ({x * 1e3:.2f}, {y * 1e3:.2f}) mm")
    z = (u * a[:, 2] + v * b[:, 2] + w * d[:, 2])[hit]
    return float(z.min()), float(z.max())


def _xy(r: float, angle_deg: float) -> tuple[float, float]:
    t = math.radians(angle_deg)
    return r * math.cos(t), r * math.sin(t)


def detent_geometry(
    seats: list[DetentSeat],
    nodes: np.ndarray,
    facets: np.ndarray,
    layer_spacing_m: float,
    spec: DetentSpec = DEFAULT_DETENT_SPEC,
) -> DetentGeometry:
    """Bump + closed/open recess spheres for each seat, sized on the surface ``(nodes, facets)``.

    Bumps are unioned before recesses are cut, so the buried part of a bump sphere is harmless as long
    as it stays inside the blade. Raises if a bump sphere would poke out of the underside, or if a recess
    would leave less than ``spec.min_wall_m`` of blade under the top face.
    """
    bumps, recesses, report = [], [], []
    for s in seats:
        bx, by = _xy(s.radius_m, s.bump_angle_deg)
        ox, oy = _xy(s.radius_m, s.open_recess_angle_deg)
        z_bot_b, z_top_b = surface_z_range(nodes, facets, bx, by)
        z_bot_o, _ = surface_z_range(nodes, facets, ox, oy)
        gap = {
            "closed": layer_spacing_m + z_bot_b - z_top_b,
            "open": layer_spacing_m + z_bot_o - z_top_b,
        }
        seat = {k: max(g, 0.0) for k, g in gap.items()}
        h = max(seat.values()) + spec.interference_m
        rb = spec.bump_radius_m
        if h > rb:
            raise ValueError(
                f"r={s.radius_m * 1e3:.0f} mm: bump {h * 1e3:.2f} mm is taller than its sphere radius"
            )
        bumps.append([bx, by, z_top_b + h - rb, rb])
        rr = rb + spec.recess_radial_clearance_m
        depth = {k: h - g + spec.seat_clearance_m for k, g in seat.items()}
        for (x, y, zb), k in (((bx, by, z_bot_b), "closed"), ((ox, oy, z_bot_o), "open")):
            recesses.append([x, y, zb + depth[k] - rr, rr])
        sphere_floor = (z_top_b + h - 2.0 * rb) - z_bot_b
        if sphere_floor < _SPHERE_FLOOR_MARGIN_M:
            raise ValueError(
                f"r={s.radius_m * 1e3:.0f} mm: bump sphere pokes out of the underside "
                f"({sphere_floor * 1e3:.2f} mm above it); blade too thin for this bump"
            )
        wall = (z_top_b - z_bot_b) - max(depth.values())
        if wall < spec.min_wall_m:
            raise ValueError(
                f"r={s.radius_m * 1e3:.0f} mm: recess would leave {wall * 1e3:.2f} mm of blade "
                f"(< {spec.min_wall_m * 1e3:.2f} mm)"
            )
        report.append(
            {
                "radius_mm": s.radius_m * 1e3,
                "bump_angle_deg": s.bump_angle_deg,
                "open_recess_angle_deg": s.open_recess_angle_deg,
                "travel_mm": s.travel_mm,
                "blade_thickness_mm": (z_top_b - z_bot_b) * 1e3,
                "gap_closed_mm": gap["closed"] * 1e3,
                "gap_open_mm": gap["open"] * 1e3,
                "bump_height_mm": h * 1e3,
                "bump_base_diameter_mm": 2.0 * math.sqrt(2.0 * rb * h - h * h) * 1e3,
                "recess_depth_closed_mm": depth["closed"] * 1e3,
                "recess_depth_open_mm": depth["open"] * 1e3,
                "wall_under_recess_mm": wall * 1e3,
            }
        )
    return DetentGeometry(np.array(bumps).reshape(-1, 4), np.array(recesses).reshape(-1, 4), report)
