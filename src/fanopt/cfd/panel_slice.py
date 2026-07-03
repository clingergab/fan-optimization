"""Path A+ panel → 2D mid-radius cascade-slice cross-section (Phase 4 spine).

Bridges the Path A+ panel thickness field (``ThicknessGridField``, plan
``plan_v1_slim_latest.md`` §10) to the geometry-agnostic slice mesher
(:func:`fanopt.cfd.mesh_2d_slice.build_cascade_slice_mesh`). Where
``baseline_cascade_polygons`` emits flat rectangles for smoke tests, this module
emits the *real* corrugated/thickness-varying cross-section the optimizer
searches over: at a fixed radial station ``u`` it sweeps the tangential span and
evaluates ``thickness_field.thickness_at(u, v)`` to build the top face over the
planar bottom, one closed ``(streamwise, cross)`` polygon per panel, tiled.

The 2D slice sees the *thickness/corrugation* shape, which is the dominant driver
of the cascade force. Camber and twist (Layer-1 mean-surface) shift/rotate the
whole section and fold in as an optional refinement without changing this
interface. Panel width + gap default to the plan's radius-dependent layout
formula (§ schema ``INTER_BLADE_ANGLE_RAD``) but are injectable for tests.
"""

from __future__ import annotations

import numpy as np

from fanopt.geometry.envelope import ThicknessGridField, camber_height_at
from fanopt.geometry.schema import (
    HUB_RADIUS_M,
    INTER_BLADE_ANGLE_RAD,
    L_BLADE_M,
    RIB_BASE_WIDTH_M,
    RIB_THICKNESS_M,
    RIB_TIP_TAPER_M,
    RIB_TIP_WIDTH_M,
)

__all__ = [
    "PanelSliceLayout",
    "rib_width_at_radius_m",
    "panel_layout_at_radius",
    "panel_slice_polygons",
]

# Tangential inter-panel clearance baked into the plan's panel-width formula
# (panel_width(r) = r·INTER_BLADE_ANGLE − 2·rib_width(r) − PANEL_TANGENTIAL_GAP_M).
PANEL_TANGENTIAL_GAP_M: float = 0.0005

_RIB_INNER_R_M = HUB_RADIUS_M
_RIB_OUTER_R_M = L_BLADE_M - RIB_TIP_TAPER_M


class PanelSliceLayout:
    """Tangential layout of one cascade slice: panel width + inter-panel gap (m)."""

    __slots__ = ("panel_width_m", "panel_gap_m", "radius_m")

    def __init__(self, panel_width_m: float, panel_gap_m: float, radius_m: float) -> None:
        if panel_width_m <= 0:
            raise ValueError(f"panel_width_m must be > 0; got {panel_width_m}")
        if panel_gap_m < 0:
            raise ValueError(f"panel_gap_m must be >= 0; got {panel_gap_m}")
        self.panel_width_m = panel_width_m
        self.panel_gap_m = panel_gap_m
        self.radius_m = radius_m

    @property
    def pitch_m(self) -> float:
        return self.panel_width_m + self.panel_gap_m


def rib_width_at_radius_m(radius_m: float) -> float:
    """Rib width at radius: linear taper base→tip (H12 lock).

    ``RIB_BASE_WIDTH_M`` (4 mm) at the hub end of the rib to ``RIB_TIP_WIDTH_M``
    (6 mm) at the rib tip, clamped outside the rib's radial extent.
    """
    span = _RIB_OUTER_R_M - _RIB_INNER_R_M
    frac = (radius_m - _RIB_INNER_R_M) / span
    frac = min(max(frac, 0.0), 1.0)
    return RIB_BASE_WIDTH_M + (RIB_TIP_WIDTH_M - RIB_BASE_WIDTH_M) * frac


def panel_layout_at_radius(radial_u: float) -> PanelSliceLayout:
    """Panel width + gap at parametric radius ``radial_u`` ∈ [0, 1] (u·L_BLADE_M).

    Uses the plan's locked layout formula: the full tangential pitch at radius
    ``r`` is ``r · INTER_BLADE_ANGLE_RAD`` (fixed 13.3°/blade), from which two rib
    widths and the tangential clearance are removed to leave the panel width. The
    gap is what remains of the pitch (2·rib_width + clearance).
    """
    if not (0.0 <= radial_u <= 1.0):
        raise ValueError(f"radial_u must be in [0, 1]; got {radial_u}")
    radius_m = radial_u * L_BLADE_M
    pitch_m = radius_m * INTER_BLADE_ANGLE_RAD
    gap_m = 2.0 * rib_width_at_radius_m(radius_m) + PANEL_TANGENTIAL_GAP_M
    panel_width_m = pitch_m - gap_m
    if panel_width_m <= 0:
        raise ValueError(
            f"radial_u={radial_u} (r={radius_m:.4f} m) gives non-positive panel width "
            f"({panel_width_m:.4f} m); slice radius is too far inboard"
        )
    return PanelSliceLayout(panel_width_m=panel_width_m, panel_gap_m=gap_m, radius_m=radius_m)


def panel_slice_polygons(
    thickness_field: ThicknessGridField,
    *,
    radial_u: float = 0.5,
    n_panels: int = 5,
    n_samples: int = 24,
    camber_knots_m: tuple[float, ...] | None = None,
    layout: PanelSliceLayout | None = None,
) -> list[np.ndarray]:
    """Cross-section polygons for a Path A+ panel at radial station ``radial_u``.

    Sweeps the tangential span of each panel and evaluates
    ``thickness_field.thickness_at(radial_u, v)`` (v ∈ [-1, 1]) to build the top
    face over a planar bottom at ``z = -RIB_THICKNESS_M/2``. Returns ``n_panels``
    closed ``(streamwise, cross)`` polygons centred on ``cross = 0``, ready for
    :func:`fanopt.cfd.mesh_2d_slice.build_cascade_slice_mesh`.

    ``camber_knots_m`` (Layer-1 chordwise camber) bows the top face across the
    tangential span, matching the CadQuery blade's mean surface — the smooth
    asymmetry lever the ASO uses for directed thrust. In the slice the airfoil
    chord *is* the tangential span, so ``v`` doubles as the camber's ``y_norm``.
    ``layout`` overrides the radius-derived panel width/gap. ``n_samples`` is the
    tangential resolution of the top face.
    """
    if n_panels < 1:
        raise ValueError(f"n_panels must be >= 1; got {n_panels}")
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2; got {n_samples}")
    lay = layout if layout is not None else panel_layout_at_radius(radial_u)

    z_lo = -RIB_THICKNESS_M / 2.0
    total = n_panels * lay.pitch_m - lay.panel_gap_m
    start = -total / 2.0
    # Tangential samples across one panel [0, width], mapped to v ∈ [-1, 1].
    local = np.linspace(0.0, lay.panel_width_m, n_samples)
    v_param = 2.0 * local / lay.panel_width_m - 1.0
    top_offsets = np.array(
        [thickness_field.thickness_at(radial_u, float(v)) for v in v_param], dtype=float
    )
    if camber_knots_m is not None:
        top_offsets = top_offsets + np.array(
            [camber_height_at(camber_knots_m, float(v)) for v in v_param], dtype=float
        )

    polys: list[np.ndarray] = []
    for i in range(n_panels):
        c0 = start + i * lay.pitch_m
        cross = c0 + local
        z_top = z_lo + top_offsets
        # Boundary: bottom-left → bottom-right → top face (right→left back to left).
        bottom = np.array([[z_lo, c0], [z_lo, c0 + lay.panel_width_m]], dtype=float)
        top = np.column_stack([z_top[::-1], cross[::-1]])
        polys.append(np.vstack([bottom, top]))
    return polys
