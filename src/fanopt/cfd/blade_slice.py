"""New solid blade → 2D cascade-slice cross-section (aero-first V1 spine).

Successor to the plano-convex ``panel_slice.py``: at a radial station it sweeps the
tangential span of the redesigned solid blade and evaluates **both** faces of the aero
surface — the displacement-grid mean surface ``±`` half the panel thickness — to build
cascade ``(streamwise = z, cross = tangential)`` polygons for
:func:`fanopt.cfd.mesh_2d_slice.build_cascade_slice_mesh`.

The airfoil chord **is** the tangential span (the blade sits broadside to the z-flow it
pushes), so ``v ∈ [-1, 1]`` sweeps the chord and the displacement grid's camber/zigzag
is the aero lever. Unlike the old plano-convex slice (flat bottom), the new solid blade
has **both faces free** — the mean surface bows and the section is symmetric about it.
The dish height ``rib_z(r)`` is a constant z-offset at a fixed radius (aerodynamically
inert per slice), so the camber line here is the displacement grid alone.
"""

from __future__ import annotations

import numpy as np

from fanopt.geometry.blade import (
    BladeParams,
    RIB_TIP_RADIUS_M,
    displacement_at,
)
from fanopt.geometry.schema import HUB_RADIUS_M, INTER_BLADE_ANGLE_RAD, L_RIB_M

__all__ = [
    "INTER_BLADE_GAP_M",
    "radius_at_u",
    "blade_chord_span_m",
    "blade_slice_polygons",
]

INTER_BLADE_GAP_M: float = 0.0005
"""Small tangential gap between adjacent deployed blades (cascade pitch − chord)."""


def radius_at_u(radial_u: float) -> float:
    """Radius (m) at parametric ``radial_u`` ∈ [0, 1] over the blade span (hub→tip)."""
    if not (0.0 <= radial_u <= 1.0):
        raise ValueError(f"radial_u must be in [0, 1]; got {radial_u}")
    return HUB_RADIUS_M + radial_u * L_RIB_M


def blade_chord_span_m(radius_m: float) -> float:
    """Tangential chord (span) of the blade at ``radius_m``: ``r·Δθ − gap`` (≥ 0)."""
    return max(radius_m * INTER_BLADE_ANGLE_RAD - INTER_BLADE_GAP_M, 0.0)


def blade_slice_polygons(
    params: BladeParams,
    *,
    radial_u: float = 0.5,
    n_panels: int = 5,
    n_samples: int = 24,
) -> list[np.ndarray]:
    """Cascade cross-section polygons for the solid blade at radial station ``radial_u``.

    Sweeps the tangential chord and evaluates ``displacement_at(r, v)`` (v ∈ [-1, 1]) for
    the mean surface, then offsets ``± panel_thickness_nom/2`` for the two free faces.
    Returns ``n_panels`` closed ``(streamwise = z, cross = tangential)`` polygons centred
    on ``cross = 0``, tiled at the cascade pitch — ready for ``build_cascade_slice_mesh``.
    """
    if n_panels < 1:
        raise ValueError(f"n_panels must be >= 1; got {n_panels}")
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2; got {n_samples}")

    r = radius_at_u(radial_u)
    width = blade_chord_span_m(r)
    if width <= 0.0:  # pragma: no cover - unreachable; even the hub radius gives r·Δθ > gap
        raise ValueError(f"radial_u={radial_u} (r={r:.4f} m) gives non-positive chord span")

    local = np.linspace(0.0, width, n_samples)
    v_param = 2.0 * local / width - 1.0
    mean = np.array([displacement_at(params, r, float(v)) for v in v_param], dtype=float)
    half_t = params.panel_thickness_nom_m / 2.0
    z_top = mean + half_t
    z_bot = mean - half_t

    pitch = width + INTER_BLADE_GAP_M
    total = n_panels * pitch - INTER_BLADE_GAP_M
    start = -total / 2.0

    polys: list[np.ndarray] = []
    for i in range(n_panels):
        c0 = start + i * pitch
        cross = c0 + local
        # Closed loop: bottom face left→right, then top face right→left (both curved).
        bottom = np.column_stack([z_bot, cross])
        top = np.column_stack([z_top[::-1], cross[::-1]])
        polys.append(np.vstack([bottom, top]))
    return polys
