"""Make a blade thinner without changing its shape: squeeze nodes toward the blade's mid-surface.

The mid-surface is halfway between the top and bottom faces along z (the pin axis). Each node keeps its
``(x, y)`` and moves to ``mid + factor · (z − mid)``, so the outline, the side walls and the mid-surface
(camber, waves, dish) are unchanged and only the z-thickness scales by ``factor``. On a tilted part the
thickness normal to the surface scales by the same factor.

:func:`cap_thickness_in_overlap` thins only the strips where neighbouring blades overlap in the open fan,
down to a cap (the boss thickness), so neighbours lie flat there instead of propping each other apart.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

__all__ = [
    "surface_z_grid",
    "squeeze_toward_mid_surface",
    "overlap_region",
    "cap_thickness_in_overlap",
]


def surface_z_grid(
    nodes: np.ndarray, facets: np.ndarray, pitch_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(xs, ys, top, bottom)``: highest / lowest surface z over each grid node (NaN off the part).

    Facets nearly parallel to z (side walls) are skipped.
    """
    lo_xy = nodes[:, :2].min(axis=0) - pitch_m
    n = np.ceil((nodes[:, :2].max(axis=0) + pitch_m - lo_xy) / pitch_m).astype(int) + 1
    xs, ys = lo_xy[0] + np.arange(n[0]) * pitch_m, lo_xy[1] + np.arange(n[1]) * pitch_m
    top = np.full(n, -np.inf)
    bot = np.full(n, np.inf)
    a, b, c = (nodes[facets[:, k]] for k in range(3))
    lo = np.ceil((np.minimum(np.minimum(a, b), c)[:, :2] - lo_xy) / pitch_m).astype(int)
    hi = np.floor((np.maximum(np.maximum(a, b), c)[:, :2] - lo_xy) / pitch_m).astype(int)
    span = np.maximum(hi - lo + 1, 0)
    v0, v1 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    good = (np.abs(den) > 1e-6 * d00 * d11) & (span.min(axis=1) > 0)
    if good.any():
        for di in range(int(span[good, 0].max())):
            for dj in range(int(span[good, 1].max())):
                m = good & (span[:, 0] > di) & (span[:, 1] > dj)
                i, j = lo[m, 0] + di, lo[m, 1] + dj
                p = np.column_stack([xs[i], ys[j]]) - a[m, :2]
                d20, d21 = (p * v0[m]).sum(1), (p * v1[m]).sum(1)
                v = (d11[m] * d20 - d01[m] * d21) / den[m]
                w = (d00[m] * d21 - d01[m] * d20) / den[m]
                u = 1.0 - v - w
                hit = (u >= -1e-9) & (v >= -1e-9) & (w >= -1e-9)
                z = (u * a[m, 2] + v * b[m, 2] + w * c[m, 2])[hit]
                np.maximum.at(top, (i[hit], j[hit]), z)
                np.minimum.at(bot, (i[hit], j[hit]), z)
    top[np.isinf(top)] = np.nan
    bot[np.isinf(bot)] = np.nan
    return xs, ys, top, bot


def squeeze_toward_mid_surface(
    nodes: np.ndarray, facets: np.ndarray, factor: float, *, pitch_m: float = 0.25e-3
) -> np.ndarray:
    """Nodes with the z-thickness scaled by ``factor`` (0 < factor ≤ 1) about the mid-surface.

    ``facets`` is the boundary surface of ``nodes``. Nodes off the sampled grid (on the side walls) use
    the nearest sampled mid-surface.
    """
    if not 0.0 < factor <= 1.0:
        raise ValueError(f"thickness factor must be in (0, 1], got {factor}")
    out = np.array(nodes, dtype=float)
    if factor == 1.0:
        return out
    xs, ys, top, bot = surface_z_grid(out, facets, pitch_m)
    top, bot = _filled(top), _filled(bot)
    z_mid = RegularGridInterpolator((xs, ys), (top + bot) / 2.0)(out[:, :2])
    out[:, 2] = z_mid + factor * (out[:, 2] - z_mid)
    return out


def _filled(grid: np.ndarray) -> np.ndarray:
    """``grid`` with every NaN replaced by its nearest sampled value."""
    missing = np.isnan(grid)
    if missing.all():
        raise ValueError("the surface has no faces to measure the mid-surface on")
    nearest = ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True)
    return grid[tuple(nearest)]


def overlap_region(
    xs: np.ndarray, ys: np.ndarray, footprint: np.ndarray, angle_rad: float
) -> np.ndarray:
    """Grid nodes of ``footprint`` that a copy turned by ``+angle`` or ``−angle`` about z also covers."""
    pitch = xs[1] - xs[0]
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    covered = np.zeros_like(footprint)
    for a in (angle_rad, -angle_rad):
        c, s = math.cos(a), math.sin(a)
        i = np.rint((c * gx + s * gy - xs[0]) / pitch).astype(int)
        j = np.rint((-s * gx + c * gy - ys[0]) / pitch).astype(int)
        inside = (i >= 0) & (i < len(xs)) & (j >= 0) & (j < len(ys))
        covered[inside] |= footprint[i[inside], j[inside]]
    return footprint & covered


def cap_thickness_in_overlap(
    nodes: np.ndarray,
    facets: np.ndarray,
    cap_m: float,
    open_angle_rad: float,
    *,
    pitch_m: float = 0.25e-3,
    blend_m: float = 3.0e-3,
) -> np.ndarray:
    """Nodes thinned (about the mid-surface) to at most ``cap_m`` where the open neighbours overlap.

    The overlap is where a copy of the blade turned by ``±open_angle_rad`` covers it. Thinner places are
    untouched; the cap eases off over ``blend_m`` outside the overlap so no step is left.
    """
    if cap_m <= 0.0:
        raise ValueError(f"thickness cap must be positive, got {cap_m}")
    out = np.array(nodes, dtype=float)
    xs, ys, top, bot = surface_z_grid(out, facets, pitch_m)
    region = overlap_region(xs, ys, ~np.isnan(top), open_angle_rad)
    top, bot = _filled(top), _filled(bot)
    weight = np.clip(1.0 - ndimage.distance_transform_edt(~region) * pitch_m / blend_m, 0.0, 1.0)
    thick = top - bot
    capped = thick - weight * np.maximum(thick - cap_m, 0.0)
    factor = np.divide(capped, thick, out=np.ones_like(thick), where=thick > 0.0)
    z_mid = RegularGridInterpolator((xs, ys), (top + bot) / 2.0)(out[:, :2])
    f = RegularGridInterpolator((xs, ys), factor)(out[:, :2])
    out[:, 2] = z_mid + f * (out[:, 2] - z_mid)
    return out
