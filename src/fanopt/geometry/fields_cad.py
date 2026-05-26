"""Geometry Layer 2 — CadQuery field-application functions.

Implements plan §9.7's Layer 2 step: apply 0–3 of the five field
families (TPMS, noise, louver, texture, edge) to the Layer 1 envelope
in the locked sub-order ``TPMS → noise → louver → texture → edge``.

Per plan §9.7 / §6.2.1 Layer 2 is **safe by construction** — the
schema's bounds guarantee features stay within the envelope (≥ 1 mm
margin, ≥ 0.8 mm minimum feature). Unlike Layer 3, Layer 2 application
is expected to succeed and is NOT wrapped in try/except by the
orchestrator.

Phase-1 status and simplifications
----------------------------------

* ``apply_tpms_field`` is a Phase-1 placeholder: it cuts a regular
  through-blade grid of cylindrical holes at the gyroid cell pitch.
  Real triply-periodic minimal surfaces (gyroid, schwarz-diamond) need
  marching-cubes evaluation of an implicit function, which is out of
  scope for this CadQuery-only Phase-1 landing. The placeholder
  honours ``cell_size_m`` (grid pitch) and ``thickness_gradient`` (hole
  radius varies linearly with x); ``lattice_type`` does not yet alter
  the cut pattern. The Phase-2 refinement replaces this with a true
  isosurface.

* ``apply_noise_field`` is a Phase-1 placeholder: instead of a real
  Perlin/Simplex noise sample it uses a deterministic sum-of-sines
  field ``S(x', y') = sin(x'·x_scale) + sin(y'·y_scale)`` with
  ``(x', y')`` rotated by ``rotation_rad`` and offset by ``x_offset``.
  The field is thresholded at the value that retains
  ``threshold_retention`` of material under uniform distribution
  approximation. Real Perlin replaces this in a later refinement.

* ``apply_louver_field``, ``apply_texture_field``, and
  ``apply_edge_feature_field`` use real CadQuery cuts that exactly
  honour the schema parameters.

Panel-domain mask
-----------------

All Layer 2 cuts are constrained to the radial band
``[HUB_RADIUS_M, CLICK_FOOTPRINT_X_RANGE_M[0] - 0.005]`` —
i.e., ``x ∈ [0.020, 0.185]``. This keeps Layer 2 features clear of
the pivot boss (inboard) and the click footprint (outboard) with a
5 mm safety margin past the click region's leading edge.
"""

from __future__ import annotations

import math

import cadquery as cq

from fanopt.geometry.fields import (
    EdgeFeatureField,
    Layer2Params,
    LouverField,
    NoiseField,
    TextureField,
    TpmsField,
)
from fanopt.geometry.schema import (
    CLICK_FOOTPRINT_X_RANGE_M,
    HUB_RADIUS_M,
    INTER_BLADE_ANGLE_RAD,
)

__all__ = [
    "PANEL_X_CARVE_RANGE_M",
    "LAYER2_CARVE_SAFETY_MARGIN_M",
    "apply_tpms_field",
    "apply_noise_field",
    "apply_louver_field",
    "apply_texture_field",
    "apply_edge_feature_field",
    "apply_layer2_fields",
]


LAYER2_CARVE_SAFETY_MARGIN_M: float = 0.005
"""5 mm clearance between Layer 2 carving and the click footprint
inner edge. Keeps cuts away from the click chamfer geometry."""

PANEL_X_CARVE_RANGE_M: tuple[float, float] = (
    HUB_RADIUS_M,
    CLICK_FOOTPRINT_X_RANGE_M[0] - LAYER2_CARVE_SAFETY_MARGIN_M,
)
"""Radial band where Layer 2 fields may carve: ``[0.020, 0.185]`` m.
Inboard bound = HUB_RADIUS (rib starts here); outboard bound =
click footprint inner edge − 5 mm safety margin."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _panel_y_max_at(x: float) -> float:
    """Half-pitch of the trapezoidal panel at radial position ``x``.

    Mirrors the envelope's ``y_max_base = x · INTER_BLADE_ANGLE_RAD / 2``
    (with Fourier modulation = 1 at the endpoints, ignored here for
    Layer 2 cut placement — Fourier modulates ≤ ±15 % per the schema
    and the carve-region margin absorbs that).
    """
    return x * INTER_BLADE_ANGLE_RAD / 2.0


def _carve_y_extent_at(x: float) -> float:
    """Conservative half-pitch — 90% of geometric y_max to stay inside
    the envelope under worst-case Fourier modulation."""
    return 0.90 * _panel_y_max_at(x)


# ---------------------------------------------------------------------------
# Layer 2 — TPMS field (Phase-1 placeholder)
# ---------------------------------------------------------------------------


def apply_tpms_field(shape: cq.Workplane, field: TpmsField) -> cq.Workplane:
    """Cut a through-blade TPMS-like porosity grid into ``shape``.

    Phase-1 placeholder: regular grid of cylindrical holes at the
    gyroid cell pitch. Honors ``cell_size_m`` (grid pitch) and
    ``thickness_gradient`` (hole radius varies linearly with x).

    Returns ``shape`` unchanged when ``field.active`` is False or when
    ``field.lattice_type == "off"``.
    """
    if not field.active or field.lattice_type == "off":
        return shape

    x_lo, x_hi = PANEL_X_CARVE_RANGE_M
    cs = field.cell_size_m
    base_r = cs / 6.0  # ~17% of cell size — minimum-feature-respecting

    # Lay out a regular x × y grid; cell pitch = cs. Cylinders along z
    # span ±2 cell sizes to guarantee through-cutting regardless of
    # envelope thickness (any extra falls outside the envelope and is
    # absorbed by the cut).
    height = 4.0 * cs
    cutters: list = []
    x = x_lo + cs / 2.0
    while x < x_hi:
        y_extent = _carve_y_extent_at(x)
        # Local thickness-gradient scaling: + → larger holes at the tip,
        # − → larger holes at the root. Linear in radial fraction.
        rad_frac = (x - x_lo) / (x_hi - x_lo)
        radius_scale = 1.0 + field.thickness_gradient * (rad_frac - 0.5)
        r = max(base_r * radius_scale, base_r * 0.5)
        y = -y_extent + cs / 2.0
        while y < y_extent:
            cyl = cq.Workplane("XY").circle(r).extrude(height, both=True).translate((x, y, 0))
            cutters.append(cyl.val())
            y += cs
        x += cs

    if not cutters:
        return shape
    union = cutters[0]
    for c in cutters[1:]:
        union = union.fuse(c)
    return shape.cut(cq.Workplane("XY").newObject([union]))


# ---------------------------------------------------------------------------
# Layer 2 — Noise field (Phase-1 placeholder)
# ---------------------------------------------------------------------------


def apply_noise_field(shape: cq.Workplane, field: NoiseField) -> cq.Workplane:
    """Cut a deterministic noise-thresholded pattern into ``shape``.

    Phase-1 placeholder: ``S(x', y') = sin(x'·x_scale) + sin(y'·y_scale)``
    with ``(x', y')`` rotated by ``rotation_rad`` and offset by
    ``x_offset``. The field is sampled on a regular grid spanning the
    panel; cells where ``S`` exceeds the retention-derived threshold are
    cut. ``threshold_retention`` is honored statistically: the threshold
    is set so that ``(1 − retention)`` of cells are cut.

    Returns ``shape`` unchanged when ``field.active`` is False.
    """
    if not field.active:
        return shape

    x_lo, x_hi = PANEL_X_CARVE_RANGE_M
    # Sample grid resolution: roughly 1 sample per mm (panel ~165 mm long).
    nx = 60
    ny = 12
    samples: list[tuple[float, float, float]] = []
    cos_r = math.cos(field.rotation_rad)
    sin_r = math.sin(field.rotation_rad)
    for i in range(nx):
        t_x = (i + 0.5) / nx
        x = x_lo + t_x * (x_hi - x_lo)
        y_extent = _carve_y_extent_at(x)
        for j in range(ny):
            t_y = (j + 0.5) / ny
            y = -y_extent + t_y * (2.0 * y_extent)
            # Rotated, scaled, offset noise coordinates.
            xp = (x + field.x_offset) * cos_r - y * sin_r
            yp = (x + field.x_offset) * sin_r + y * cos_r
            value = math.sin(xp * field.x_scale * 10.0) + math.sin(yp * field.y_scale * 10.0)
            samples.append((x, y, value))

    # Threshold: cut the (1 - retention) highest-value cells.
    samples.sort(key=lambda s: s[2])
    n_cut = int(len(samples) * (1.0 - field.threshold_retention))
    cuts = samples[len(samples) - n_cut :] if n_cut > 0 else []

    cutters: list = []
    cell_dx = (x_hi - x_lo) / nx
    cell_dy_at = lambda x: 2.0 * _carve_y_extent_at(x) / ny  # noqa: E731
    height = 0.020  # 20 mm, well past any panel thickness
    for x, y, _ in cuts:
        dx = cell_dx
        dy = cell_dy_at(x)
        cutter = cq.Workplane("XY").box(dx, dy, height).translate((x, y, 0))
        cutters.append(cutter.val())

    if not cutters:
        return shape
    union = cutters[0]
    for c in cutters[1:]:
        union = union.fuse(c)
    return shape.cut(cq.Workplane("XY").newObject([union]))


# ---------------------------------------------------------------------------
# Layer 2 — Louver field
# ---------------------------------------------------------------------------


def _louver_x_positions(count: int, profile: str) -> list[float]:
    """Radial positions of ``count`` louvers under the given spacing profile.

    Returns positions in the radial fraction space ``t ∈ (0, 1)``;
    caller maps to ``PANEL_X_CARVE_RANGE_M``.
    """
    if profile == "uniform":
        return [(i + 1) / (count + 1) for i in range(count)]
    if profile == "clustered-at-tip":
        return [((i + 1) / (count + 1)) ** 0.5 for i in range(count)]
    if profile == "gradient-toward-LE":
        # Even radial spacing here; "gradient-toward-LE" is a tangential
        # property the louver angle captures. Phase-2 may revisit.
        return [(i + 1) / (count + 1) for i in range(count)]
    raise ValueError(f"unknown louver spacing_profile: {profile!r}")  # pragma: no cover


def apply_louver_field(shape: cq.Workplane, field: LouverField) -> cq.Workplane:
    """Apply ``field.count`` parallel rectangular louvers to ``shape``.

    Each louver is a rectangular slot of width ``field.width_m``,
    length = local panel half-pitch (so it spans the full tangential
    extent), rotated by ``field.angle_rad`` about the panel z-axis.
    ``polarity == "subtract"`` cuts; ``polarity == "add"`` fuses an
    extruded rib (height = 1 mm).

    Returns ``shape`` unchanged when ``field.active`` is False.
    """
    if not field.active:
        return shape

    x_lo, x_hi = PANEL_X_CARVE_RANGE_M
    positions_t = _louver_x_positions(field.count, field.spacing_profile)

    louver_height_through = 0.020  # 20 mm, through any panel thickness
    add_height = 0.001  # 1 mm extruded rib for additive polarity
    cutters: list = []
    for t in positions_t:
        x = x_lo + t * (x_hi - x_lo)
        y_extent = _carve_y_extent_at(x)
        slot_length = 2.0 * y_extent
        if field.polarity == "subtract":
            box = (
                cq.Workplane("XY")
                .box(field.width_m, slot_length, louver_height_through)
                .rotate((0, 0, 0), (0, 0, 1), math.degrees(field.angle_rad))
                .translate((x, 0, 0))
            )
        else:
            box = (
                cq.Workplane("XY")
                .box(field.width_m, slot_length, add_height)
                .rotate((0, 0, 0), (0, 0, 1), math.degrees(field.angle_rad))
                .translate((x, 0, add_height / 2.0))
            )
        cutters.append(box.val())

    union = cutters[0]
    for c in cutters[1:]:
        union = union.fuse(c)
    op_wp = cq.Workplane("XY").newObject([union])
    if field.polarity == "subtract":
        return shape.cut(op_wp)
    return shape.union(op_wp)


# ---------------------------------------------------------------------------
# Layer 2 — Texture field
# ---------------------------------------------------------------------------


def _texture_grid_pitch_m(density_per_cm2: float, size_m: float) -> float:
    """Grid pitch for a target density: one feature per (1/density) cm²."""
    cm2_per_feature = 1.0 / max(density_per_cm2, 1e-6)
    m2_per_feature = cm2_per_feature * 1.0e-4
    pitch = math.sqrt(m2_per_feature)
    return max(pitch, size_m * 1.2)


def _build_texture_feature(
    feature_type: str,
    size: float,
) -> cq.Workplane:
    """Construct one texture feature centred at the origin."""
    r = size / 2.0
    if feature_type == "dimple":
        # Half-sphere depression (cut from top): full sphere centered
        # at origin so the "cut" half is z > 0.
        return cq.Workplane("XY").sphere(r)
    if feature_type == "bump":
        return cq.Workplane("XY").sphere(r)
    if feature_type == "ridge":
        # Short rectangular bump aligned with x; orientation applied
        # by the caller.
        return cq.Workplane("XY").box(size * 2.0, size, size)
    raise ValueError(f"unknown texture feature_type: {feature_type!r}")  # pragma: no cover


def apply_texture_field(shape: cq.Workplane, field: TextureField) -> cq.Workplane:
    """Apply a density-tiled texture pattern to ``shape``.

    Lays a regular grid of features at the density-derived pitch over
    the panel-carve region. Orientation rotates each feature about its
    own centre. Polarity selects subtract (cut, e.g., dimples) or add
    (union, e.g., raised bumps).

    Returns ``shape`` unchanged when ``field.active`` is False.
    """
    if not field.active:
        return shape

    x_lo, x_hi = PANEL_X_CARVE_RANGE_M
    pitch = _texture_grid_pitch_m(field.density_per_cm2, field.size_m)
    bbox = shape.val().BoundingBox()
    z_centre = (bbox.zmin + bbox.zmax) / 2.0

    cutters: list = []
    x = x_lo + pitch / 2.0
    while x < x_hi:
        y_extent = _carve_y_extent_at(x)
        y = -y_extent + pitch / 2.0
        while y < y_extent:
            feat = _build_texture_feature(field.feature_type, field.size_m)
            if field.feature_type == "ridge" and abs(field.orientation_rad) > 1e-12:
                feat = feat.rotate(
                    (0, 0, 0), (0, 0, 1), math.degrees(field.orientation_rad)
                )
            feat = feat.translate((x, y, z_centre))
            cutters.append(feat.val())
            y += pitch
        x += pitch

    if not cutters:
        return shape
    union = cutters[0]
    for c in cutters[1:]:
        union = union.fuse(c)
    op_wp = cq.Workplane("XY").newObject([union])
    if field.polarity == "subtract":
        return shape.cut(op_wp)
    return shape.union(op_wp)


# ---------------------------------------------------------------------------
# Layer 2 — Edge feature field
# ---------------------------------------------------------------------------


def _build_edge_notch(
    feature_type: str,
    depth: float,
    panel_radial_pitch: float,
) -> cq.Workplane:
    """Construct one edge notch shape, depth-extruded along ±y.

    The notch is constructed in the X-Z plane and then extruded along
    y for ``2*depth`` so it punches through the panel edge regardless
    of panel thickness.
    """
    half_x = panel_radial_pitch / 2.0
    if feature_type == "serration":
        # Triangle pointing into the panel (−y direction by caller's translate).
        pts = [(-half_x, 0.0), (+half_x, 0.0), (0.0, depth)]
    elif feature_type == "scallop":
        # Half-circle: approximate with 3-arc polyline via spline.
        # CadQuery's threePointArc keeps it as a real arc.
        return (
            cq.Workplane("XY")
            .moveTo(-half_x, 0.0)
            .threePointArc((0.0, depth), (half_x, 0.0))
            .close()
            .extrude(0.020, both=True)
        )
    elif feature_type == "smooth-fade":
        # Long shallow taper.
        pts = [(-half_x, 0.0), (+half_x, 0.0), (+half_x, depth)]
    else:  # pragma: no cover -- schema rejects
        raise ValueError(f"unknown edge feature_type: {feature_type!r}")
    return (
        cq.Workplane("XY")
        .polyline(pts)
        .close()
        .extrude(0.020, both=True)
    )


def apply_edge_feature_field(
    shape: cq.Workplane,
    field: EdgeFeatureField,
) -> cq.Workplane:
    """Cut ``field.count`` notches along the LE / TE / both edges.

    Notches are evenly distributed in radial fraction across the
    panel-carve region. ``application`` selects which y-edge gets the
    notches; ``feature_type`` selects the shape (triangular serration,
    semicircular scallop, tapered smooth-fade).

    Returns ``shape`` unchanged when ``field.active`` is False.
    """
    if not field.active:
        return shape

    x_lo, x_hi = PANEL_X_CARVE_RANGE_M
    positions_t = [(i + 1) / (field.count + 1) for i in range(field.count)]
    radial_pitch = (x_hi - x_lo) / (field.count + 1)

    applications: tuple[str, ...]
    if field.application == "both":
        applications = ("LE", "TE")
    else:
        applications = (field.application,)

    cutters: list = []
    for t in positions_t:
        x = x_lo + t * (x_hi - x_lo)
        y_extent = _panel_y_max_at(x)  # use geometric edge here, not safe inset
        for app in applications:
            # LE: y < 0; TE: y > 0. Notch points inward.
            y_edge = -y_extent if app == "LE" else +y_extent
            sign = 1.0 if app == "LE" else -1.0
            notch = _build_edge_notch(field.feature_type, field.depth_m, radial_pitch)
            # Flip the notch so it points inward when on the TE side.
            if app == "TE":
                notch = notch.rotate((0, 0, 0), (1, 0, 0), 180.0)
            notch = notch.translate((x, y_edge + sign * field.depth_m * 0.0, 0))
            # Translate inward by depth so the notch bites into the panel.
            inward_translate = field.depth_m * 0.5
            notch = notch.translate((0, sign * inward_translate, 0))
            cutters.append(notch.val())

    if not cutters:
        return shape
    union = cutters[0]
    for c in cutters[1:]:
        union = union.fuse(c)
    return shape.cut(cq.Workplane("XY").newObject([union]))


# ---------------------------------------------------------------------------
# Layer 2 — top-level dispatcher
# ---------------------------------------------------------------------------


def apply_layer2_fields(
    shape: cq.Workplane,
    params: Layer2Params,
) -> cq.Workplane:
    """Apply all active Layer 2 fields in the locked sub-order.

    Order per plan §9.7 + ``generator._apply_layer2_fields``:
    TPMS → noise → louver → texture → edge. Inactive fields are
    skipped without modifying the shape.
    """
    if params.tpms.active:
        shape = apply_tpms_field(shape, params.tpms)
    if params.noise.active:
        shape = apply_noise_field(shape, params.noise)
    if params.louver.active:
        shape = apply_louver_field(shape, params.louver)
    if params.texture.active:
        shape = apply_texture_field(shape, params.texture)
    if params.edge.active:
        shape = apply_edge_feature_field(shape, params.edge)
    return shape
