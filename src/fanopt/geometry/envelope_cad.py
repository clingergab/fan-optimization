"""Geometry Layer 1 — CadQuery envelope generator.

Implements plan §9.7.1 Step 0's ``make_outer_envelope(layer1, print_orientation)``
using a CadQuery cross-section loft. Hard-imports CadQuery — tests that
exercise this module gate themselves on ``importlib.util.find_spec("cadquery")``
+ ``pytest.skip(allow_module_level=True)`` per CLAUDE.md §4.1.

Construction (full Layer 1 — plan §6.2.1):

- **Trapezoidal half-pitch panel** in ``(x, y)``: ``y_max(x) = x · INTER_BLADE_ANGLE_RAD / 2``.
- **Thickness field (Path A+)**: ``params.thickness_field`` — a control-point
  grid + corrugation evaluated per ``(u=x/L_BLADE_M, v=y/y_max(x))`` on the top
  face; bilinear over the grid, clamped to the thickness lock (§10 slim plan).
- **Chordwise camber**: 3-4 spline knots at ``camber_knots_m``, linearly
  interpolated along the chordwise direction (``y / y_max(x)`` ∈ [-1, +1]).
- **Spanwise twist**: 2-3 spline knots at ``twist_knots_rad``, rotated about
  the x-axis per radial station.
- **Fourier LE/TE modulation**: k=1,2,3 amplitudes perturb the LE
  (``y < 0``) and TE (``y > 0``) edges by
  ``y_max(x) · (1 + Σ amp_k · sin(k π x / L + φ_k))``
  with the plan-locked phase offsets ``φ = (0, π/3, 2π/3)`` for
  ``k = (1, 2, 3)`` (plan §6.2.1 Layer 1 table: "phases fixed at 0, π/3,
  2π/3"). Only the k=1 harmonic zeroes at the endpoints; k=2 and k=3
  contribute non-zero modulation at the root and tip per the plan's
  phase choice (this avoids constructive amplification of all three
  harmonics at the same x). Amplitudes are bounded to ±15 % by
  ``Layer1Params`` so the envelope stays within ±15 % of mean.

- **LE / TE label convention**: ``LE`` (leading edge) is the ``y < 0``
  panel edge; ``TE`` (trailing edge) is ``y > 0``. For a sweeping fan
  blade the aerodynamic LE depends on stroke direction; this parameter
  label is a fixed geometric convention so the BO can search the two
  edges independently.
- **Print-orientation switch**: ``flat`` (rib-flat default) uses a planar
  bottom face at ``z = 0`` with camber + thickness on the top face only
  (plano-convex per §0 row 47); ``edge`` / ``custom-angle`` uses a
  midplane-symmetric profile where camber appears on both faces.
- **Twist under plano-convex**: spanwise twist rotates the whole cross-
  section about ``x`` and therefore tilts the bottom face. Under
  ``print_orientation == "flat"`` that would break the planar-bottom
  constraint; the generator forces twist to zero in that case
  (top-face-only twist is a Phase-1-followup refinement). Under
  ``edge`` / ``custom-angle`` twist is honoured as specified.

The cross-section at ``x = 0`` is degenerate (zero panel width) and is
skipped; the loft starts at ``x = LOFT_START_EPS_M`` (1 mm in from the
pivot) so the lofted solid has a well-defined volume. **Trade-off:** a
1 mm initial offset keeps the boss region intact (boss centered at
``PIVOT_CENTER_X_M = 8 mm``, radius ``PANEL_PIVOT_REGION_RADIUS_M =
7 mm``, so the boss spans ``x ∈ [1 mm, 15 mm]`` — the loft start at
1 mm is exactly at the inboard boss edge). A larger offset (e.g.,
5 mm) would clip into the boss; a smaller offset would create a
near-degenerate initial cross-section that strains CadQuery's loft
tolerance.

The returned shape is the **full envelope** (panel + rib region as one
solid). The orchestrator subtracts the Phase 2 rib mask later
(``panel_domain = full_envelope.cut(rib_mask_3d)``); this module
constructs only the envelope.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import cadquery as cq

from fanopt.geometry.envelope import Layer1Params
from fanopt.geometry.manufacturability import PRINT_ORIENTATIONS
from fanopt.geometry.schema import INTER_BLADE_ANGLE_RAD, L_BLADE_M

__all__ = [
    "N_RADIAL_STATIONS",
    "N_CHORD_SAMPLES",
    "LOFT_START_EPS_M",
    "FOURIER_PHASES_RAD",
    "make_outer_envelope",
]


FOURIER_PHASES_RAD: tuple[float, float, float] = (0.0, math.pi / 3.0, 2.0 * math.pi / 3.0)
"""Plan §6.2.1 Layer 1 table: "phases fixed at 0, π/3, 2π/3" for k=1,2,3
Fourier harmonics. Locked constant; exported so tests + downstream callers
can pin the phase convention without re-deriving it."""


N_RADIAL_STATIONS: int = 20
"""Number of radial loft slices spanning ``[LOFT_START_EPS_M, L_BLADE_M]``.

20 stations is enough to resolve the cubic camber + Fourier (k≤3)
modulation without an inflated mesh; bump to 40+ if Fourier amplitudes
near the ±15 % bound show visible facetting."""

N_CHORD_SAMPLES: int = 32
"""Points sampling each cross-section's top/bottom edge along the chord.

32 samples per edge × 2 edges = 64 vertices per cross-section, which
captures linear-spline camber + smooth top-face curvature without
over-tessellating."""

LOFT_START_EPS_M: float = 0.001
"""Inboard start of the loft (1 mm in from the pivot). Avoids the
degenerate ``x = 0`` cross-section (zero panel width) while keeping
the boss region intact: the loft starts well inside the
``PIVOT_CENTER_X_M = 0.008`` boss center."""


# ---------------------------------------------------------------------------
# Spline + Fourier helpers
# ---------------------------------------------------------------------------


def _linear_spline(knots: tuple[float, ...], t: float) -> float:
    """Piecewise-linear interpolation of `knots` at parameter ``t ∈ [0, 1]``.

    With N knots, segments span equal partitions of ``[0, 1]``:
    knot ``i`` is at ``t = i / (N - 1)``. Out-of-range ``t`` is clamped.
    """
    if len(knots) == 0:
        raise ValueError("knots must be non-empty")
    if len(knots) == 1:
        return float(knots[0])
    if t <= 0.0:
        return float(knots[0])
    if t >= 1.0:
        return float(knots[-1])
    n_segments = len(knots) - 1
    seg_t = t * n_segments
    i = int(seg_t)
    frac = seg_t - i
    return float(knots[i]) * (1.0 - frac) + float(knots[i + 1]) * frac


def _fourier_modulation(x: float, amplitudes: tuple[float, float, float]) -> float:
    """Edge-position modulation factor at radial position ``x``.

    Returns ``1 + Σ_{k=1}^{3} amp_k · sin(k π x / L_BLADE_M + φ_k)`` with
    the plan-locked phase offsets ``φ = (0, π/3, 2π/3)`` per plan §6.2.1
    Layer 1 table.

    The k=1 harmonic (φ_1 = 0) zeroes at ``x = 0`` and ``x = L_BLADE_M``,
    so the fundamental modulation is internal-only. The k=2 and k=3
    harmonics have non-zero values at the endpoints by design — the
    plan's phase choice ensures the three harmonics don't all peak at
    the same ``x``, avoiding constructive amplification.

    Per ``FOURIER_AMPLITUDE_RELATIVE_MAX = 0.15`` each ``amp_k`` is
    bounded to ±0.15. The schema's per-harmonic cap keeps any one
    harmonic's contribution to ±15 % of mean.
    """
    factor = 1.0
    for k, amp in enumerate(amplitudes, start=1):
        phase = FOURIER_PHASES_RAD[k - 1]
        factor += amp * math.sin(k * math.pi * x / L_BLADE_M + phase)
    return factor


def _camber_height(y_norm: float, camber_knots_m: tuple[float, ...]) -> float:
    """Camber height at chord-normalised position ``y_norm ∈ [-1, +1]``.

    Knots distributed evenly across ``[-1, +1]``: with 3 knots they land at
    ``y_norm ∈ {-1, 0, +1}``; with 4 knots at ``{-1, -1/3, +1/3, +1}``.
    Outside the range, the value is clamped (Layer 1 only ever evaluates
    inside ``[-1, +1]``; the clamp is defensive).

    All camber values are non-negative under flat orientation (enforced by
    ``Layer1Params`` ``CAMBER_RANGE_M = (0, 0.005)``).
    """
    t = (y_norm + 1.0) / 2.0
    return _linear_spline(camber_knots_m, t)


# ---------------------------------------------------------------------------
# Cross-section construction
# ---------------------------------------------------------------------------


def _section_points(
    y_le_neg: float,
    y_te_pos: float,
    z_bottom_fn: Callable[[float], float],
    z_top_fn: Callable[[float], float],
    n_chord: int,
) -> list[tuple[float, float]]:
    """Build the closed cross-section polyline in ``(y, z)`` at one ``x``.

    Traversal: LE-bottom → TE-bottom → TE-top → LE-top, then ``close()`` in
    the caller. Generates ``2 · (n_chord + 1)`` vertices total. Linear
    interpolation along y; nonlinearity in z is captured via the
    ``z_bottom_fn`` / ``z_top_fn`` callables (e.g., camber).
    """
    pts: list[tuple[float, float]] = []
    # Bottom edge: y from LE (negative) to TE (positive).
    for i in range(n_chord + 1):
        t = i / n_chord
        y = y_le_neg * (1.0 - t) + y_te_pos * t
        pts.append((y, z_bottom_fn(y)))
    # Top edge: y from TE back to LE.
    for i in range(n_chord + 1):
        t = i / n_chord
        y = y_te_pos * (1.0 - t) + y_le_neg * t
        pts.append((y, z_top_fn(y)))
    return pts


def _apply_twist(pts_yz: list[tuple[float, float]], twist_rad: float) -> list[tuple[float, float]]:
    """Rotate cross-section points by ``twist_rad`` about the x-axis (= ``(y, z)`` origin)."""
    if abs(twist_rad) < 1e-12:
        return pts_yz
    c = math.cos(twist_rad)
    s = math.sin(twist_rad)
    return [(y * c - z * s, y * s + z * c) for (y, z) in pts_yz]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_outer_envelope(
    params: Layer1Params,
    print_orientation: str,
) -> cq.Workplane:
    """Construct the Layer 1 outer envelope as a CadQuery solid.

    Parameters
    ----------
    params : Layer1Params
        Validated Layer 1 design parameters (camber/twist/thickness splines,
        edge profile, Fourier LE/TE amplitudes, blade_count).
    print_orientation : str
        One of :data:`fanopt.geometry.manufacturability.PRINT_ORIENTATIONS`.
        ``"flat"`` (rib-flat default) triggers the plano-convex construction
        (planar bottom at ``z = 0``); ``"edge"`` and ``"custom-angle"`` use
        the midplane-symmetric construction.

    Returns
    -------
    cq.Workplane
        Single lofted solid wrapped in a Workplane. The solid spans
        ``x ∈ [LOFT_START_EPS_M, L_BLADE_M]``; bounding box on ``y``
        reflects the panel half-pitch plus Fourier modulation; bounding
        box on ``z`` reflects the thickness profile + camber.
    """
    if print_orientation not in PRINT_ORIENTATIONS:
        raise ValueError(
            f"print_orientation must be one of {PRINT_ORIENTATIONS}, " f"got {print_orientation!r}"
        )
    flat = print_orientation == "flat"

    # Build the loft Workplane by chaining one workplane per radial station.
    # CadQuery's loft consumes the pending wires accumulated on the workplane,
    # so we add (workplane, polyline, close) for each station.
    dx = (L_BLADE_M - LOFT_START_EPS_M) / N_RADIAL_STATIONS

    def _station_data(
        x: float,
    ) -> tuple[float, float, float, float, Callable[[float], float], Callable[[float], float]]:
        """Per-station scalars + z-functions of y."""
        # Spline parameter = full radial fraction. BO parameters span
        # ``[0, 1]`` along the full blade; the 1 mm loft-start offset must
        # not pollute the spline lookup.
        t_x_full = x / L_BLADE_M
        y_max_base = (x * INTER_BLADE_ANGLE_RAD) / 2.0
        y_le_neg = -y_max_base * _fourier_modulation(x, params.fourier_le_amplitudes)
        y_te_pos = +y_max_base * _fourier_modulation(x, params.fourier_te_amplitudes)
        # Path A+ thickness is a 2D field t(u, v) over (radial u = t_x_full,
        # tangential v = y_norm); evaluate per-y inside the z-functions. The
        # representative scalar (tangential centre) is kept for the bbox tuple.
        thickness = params.thickness_field.thickness_at(t_x_full, 0.0)
        twist_rad = _linear_spline(params.twist_knots_rad, t_x_full)
        if flat:
            # Plano-convex constraint: planar bottom face. Whole-cross-section
            # twist would tilt the bottom and break the build-plate contact.
            # Suppress twist here; Phase 1 followup may add top-face-only twist.
            twist_rad = 0.0

        def _y_norm(y: float) -> float:
            yn = y / y_max_base if y_max_base > 1e-12 else 0.0
            return max(-1.0, min(1.0, yn))

        def _thickness_local(y_norm: float) -> float:
            return params.thickness_field.thickness_at(t_x_full, y_norm)

        if flat:

            def z_bottom(_y: float) -> float:
                return 0.0

            def z_top(y: float) -> float:
                y_norm = _y_norm(y)
                return _thickness_local(y_norm) + _camber_height(y_norm, params.camber_knots_m)

        else:

            def z_bottom(y: float) -> float:
                y_norm = _y_norm(y)
                return (
                    -_thickness_local(y_norm) / 2.0
                    - _camber_height(y_norm, params.camber_knots_m) / 2.0
                )

            def z_top(y: float) -> float:
                y_norm = _y_norm(y)
                return (
                    +_thickness_local(y_norm) / 2.0
                    + _camber_height(y_norm, params.camber_knots_m) / 2.0
                )

        return y_le_neg, y_te_pos, thickness, twist_rad, z_bottom, z_top

    # First cross-section.
    x0 = LOFT_START_EPS_M
    y_le, y_te, _thk, twist_rad, z_bot, z_top = _station_data(x0)
    pts0 = _section_points(y_le, y_te, z_bot, z_top, N_CHORD_SAMPLES)
    pts0 = _apply_twist(pts0, twist_rad)

    wp = cq.Workplane("YZ", origin=(x0, 0.0, 0.0)).polyline(pts0).close()

    # Chained cross-sections.
    for i in range(1, N_RADIAL_STATIONS + 1):
        x = LOFT_START_EPS_M + i * dx
        y_le, y_te, _thk, twist_rad, z_bot, z_top = _station_data(x)
        pts = _section_points(y_le, y_te, z_bot, z_top, N_CHORD_SAMPLES)
        pts = _apply_twist(pts, twist_rad)
        wp = wp.workplane(offset=dx, centerOption="ProjectedOrigin").polyline(pts).close()

    return wp.loft(combine=True)
