"""Unbend a blade's meridian: the same blade, flat, at the same reach and the same true wall thickness.

The blade's mean surface is the meridian ``z = rib_z_at(rho)`` (a surface of revolution about the pin)
plus a small panel offset. Each node keeps its ``(x, y)``, so the planform and the 220 mm reach are
unchanged, and its height above the meridian is scaled by ``cos`` of the local meridian slope: the
blade's thickness was laid out along z, so on a slope its wall measured normal to the surface is
``cos(slope)`` thinner, and the flat blade keeps that true wall. Tets keep their connectivity, so a
density mapped onto the curved mesh carries over unchanged.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from fanopt.geometry.blade import RIB_TIP_RADIUS_M, BladeParams, rib_z_at

__all__ = ["flat_meridian_params", "flatten_meridian"]


def flat_meridian_params(params: BladeParams) -> BladeParams:
    """``params`` with every meridian knot at 0 (a flat blade); all other knobs unchanged."""
    return replace(params, rib_bow_knots_m=tuple(0.0 for _ in params.rib_bow_knots_m))


def flatten_meridian(
    nodes: np.ndarray, params: BladeParams, *, step_m: float = 0.1e-3
) -> np.ndarray:
    """Nodes moved onto the flat blade: ``z → (z − meridian(rho)) · cos(meridian slope at rho)``."""
    out = np.array(nodes, dtype=float)
    rs = np.arange(0.0, RIB_TIP_RADIUS_M + step_m, step_m)
    zs = np.array([rib_z_at(params, r) for r in rs])
    cos = 1.0 / np.sqrt(1.0 + np.gradient(zs, rs) ** 2)
    rho = np.hypot(out[:, 0], out[:, 1])
    out[:, 2] = (out[:, 2] - np.interp(rho, rs, zs)) * np.interp(rho, rs, cos)
    return out
