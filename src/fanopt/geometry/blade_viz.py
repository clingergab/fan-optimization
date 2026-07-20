"""Surface point grids of a blade for 3D visualization (Plotly / matplotlib).

Pure-numpy evaluation of a :class:`~fanopt.geometry.blade.BladeParams` into ``(X, Y, Z)``
meshgrids for a surface plot — no CadQuery needed, so notebooks can render the optimized
blades without the CAD stack. The blade sits in its own frame (pin = +z axis); the top
face is the dished mean surface (``rib_z + displacement``) plus the local half thickness
(the rib ridge at the tangential edges, the thinner panel between).
"""

from __future__ import annotations

import math

import numpy as np

from fanopt.geometry.blade import (
    RIB_TIP_RADIUS_M,
    BladeParams,
    displacement_at,
    rib_thickness_at,
    rib_width_at,
    rib_z_at,
)
from fanopt.geometry.schema import HUB_RADIUS_M, INTER_BLADE_ANGLE_RAD

__all__ = ["blade_surface_xyz"]

_ALPHA_RAD: float = INTER_BLADE_ANGLE_RAD / 2.0


def _half_thickness_m(params: BladeParams, r: float, theta: float) -> float:
    if r >= HUB_RADIUS_M and r * (_ALPHA_RAD - abs(theta)) <= rib_width_at(r):
        return rib_thickness_at(params, r) / 2.0
    return params.panel_thickness_nom_m / 2.0


def blade_surface_xyz(
    params: BladeParams,
    *,
    n_radial: int = 40,
    n_tangential: int = 40,
    face: str = "top",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(X, Y, Z)`` grids (each ``(n_radial, n_tangential)``, metres) of a blade face.

    ``face`` is ``"top"`` (mean + half thickness) or ``"bottom"`` (mean − half thickness).
    Radius spans hub→tip; the tangential angle spans the blade's wedge.
    """
    if face not in ("top", "bottom"):
        raise ValueError(f"face must be 'top' or 'bottom'; got {face!r}")
    sign = 1.0 if face == "top" else -1.0
    radii = np.linspace(HUB_RADIUS_M, RIB_TIP_RADIUS_M, n_radial)
    thetas = np.linspace(-_ALPHA_RAD, _ALPHA_RAD, n_tangential)
    x = np.empty((n_radial, n_tangential))
    y = np.empty((n_radial, n_tangential))
    z = np.empty((n_radial, n_tangential))
    for i, r in enumerate(radii):
        for j, th in enumerate(thetas):
            mean = rib_z_at(params, float(r)) + displacement_at(params, float(r), th / _ALPHA_RAD)
            h = _half_thickness_m(params, float(r), float(th))
            x[i, j] = r * math.cos(th)
            y[i, j] = r * math.sin(th)
            z[i, j] = mean + sign * h
    return x, y, z
