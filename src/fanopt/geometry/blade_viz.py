"""Surface point grids of a blade for 3D visualization (Plotly / matplotlib).

Pure-numpy evaluation of a :class:`~fanopt.geometry.blade.BladeParams` into ``(X, Y, Z)``
meshgrids for a surface plot — no CadQuery needed, so notebooks can render the optimized
blades without the CAD stack. The blade sits in its own frame (pin = +z axis); the top
face is the dished mean surface (``rib_z + displacement``) plus the local half thickness
(the rib ridge at the tangential edges, the thinner panel between).
"""

from __future__ import annotations

import numpy as np

from fanopt.geometry.blade import (
    BLADE_ROOT_RADIUS_M,
    RIB_TIP_RADIUS_M,
    BladeParams,
    displacement_at,
    half_width_at,
    panel_thickness_at,
    rib_thickness_at,
    rib_width_at,
    rib_z_at,
)

__all__ = ["blade_surface_xyz"]


def _half_thickness_m(params: BladeParams, r: float, v: float) -> float:
    if not params.uniform and half_width_at(r) * (1.0 - abs(v)) <= rib_width_at(r):
        return rib_thickness_at(params, r) / 2.0
    return panel_thickness_at(params, r, v) / 2.0


def blade_surface_xyz(
    params: BladeParams,
    *,
    n_radial: int = 40,
    n_tangential: int = 40,
    face: str = "top",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(X, Y, Z)`` grids (each ``(n_radial, n_tangential)``, metres) of a blade face.

    ``face`` is ``"top"`` (mean + half thickness) or ``"bottom"`` (mean − half thickness).
    Radius (``x``) spans root→tip; ``y`` spans the trapezoid width ``[-w(r), +w(r)]``.
    """
    if face not in ("top", "bottom"):
        raise ValueError(f"face must be 'top' or 'bottom'; got {face!r}")
    sign = 1.0 if face == "top" else -1.0
    radii = np.linspace(BLADE_ROOT_RADIUS_M, RIB_TIP_RADIUS_M, n_radial)
    vs = np.linspace(-1.0, 1.0, n_tangential)
    x = np.empty((n_radial, n_tangential))
    y = np.empty((n_radial, n_tangential))
    z = np.empty((n_radial, n_tangential))
    for i, r in enumerate(radii):
        w = half_width_at(float(r))
        for j, v in enumerate(vs):
            mean = rib_z_at(params, float(r)) + displacement_at(params, float(r), float(v))
            h = _half_thickness_m(params, float(r), float(v))
            x[i, j] = r
            y[i, j] = v * w
            z[i, j] = mean + sign * h
    return x, y, z
