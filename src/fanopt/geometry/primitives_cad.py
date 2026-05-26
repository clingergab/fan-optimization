"""Geometry Layer 3 — CadQuery primitive application.

Implements plan §9.7's Layer 3 step: place one capped 0-1 primitive
(slot / ellipsoid / wedge) on the Layer-2-carved envelope, either as
an additive boss or as a subtractive cut. Per plan §9.7 this is the
**only** generator step where CAD failures are tolerated; the
orchestrator (:func:`fanopt.geometry.generator.generate_blade`) wraps
the call in try/except. Failures degrade
``GenerationStatus`` to ``LAYER3_FAILED`` rather than aborting.

The shape returned mirrors the input ``cq.Workplane``; when
``primitive.present`` is ``False`` the input is returned unchanged. All
schema bounds (margin ≥ 1 mm from envelope edge, size ≥ 0.8 mm, size
≤ 30 % of local envelope, rotation ∈ [−π, π]) are enforced upstream by
``Layer3Primitive.__post_init__``; this module trusts them and focuses
on the CadQuery construction.

Position coordinates are panel-local in metres. The convention matches
``Layer3Primitive``: ``position_x_m`` / ``position_y_m`` /
``position_z_m`` is the primitive centre in the panel coordinate frame
(same frame the envelope is constructed in — ``x`` radial from pivot,
``y`` tangential half-pitch, ``z`` plano-convex out-of-plane). Sizes
are full extents along each principal axis. Rotations are extrinsic
Euler angles (rad) applied in X → Y → Z order before translation.
"""

from __future__ import annotations

import math

import cadquery as cq

from fanopt.geometry.primitives import Layer3Primitive

__all__ = [
    "apply_primitive",
]


def _make_slot(size_x: float, size_y: float, size_z: float) -> cq.Workplane:
    """Rounded slot — capsule extruded along z, centred on the origin.

    Uses CadQuery's ``slot2D`` primitive (a 2D slot is two semicircles
    joined by a rectangle) extruded ±z/2. For a square cross-section
    (size_x == size_y) the slot2D primitive degenerates; fall back to
    a plain cylinder along z to avoid the OpenCascade ``BRep_API``
    failure that surfaces when slot2D's fillet radius equals half its
    length.
    """
    if abs(size_x - size_y) < 1e-9:
        r = max(size_x, size_y) / 2.0
        return (
            cq.Workplane("XY")
            .circle(r)
            .extrude(size_z / 2.0, both=True)
        )
    longer = max(size_x, size_y)
    shorter = min(size_x, size_y)
    angle = 0.0 if size_x >= size_y else 90.0
    return (
        cq.Workplane("XY")
        .slot2D(longer, shorter, angle=angle)
        .extrude(size_z / 2.0, both=True)
    )


def _make_ellipsoid(size_x: float, size_y: float, size_z: float) -> cq.Workplane:
    """Triaxial ellipsoid — unit sphere scaled by ``(size_x/2, size_y/2, size_z/2)``."""
    sphere = cq.Workplane("XY").sphere(1.0)
    matrix = cq.Matrix(
        [
            [size_x / 2.0, 0.0, 0.0, 0.0],
            [0.0, size_y / 2.0, 0.0, 0.0],
            [0.0, 0.0, size_z / 2.0, 0.0],
        ]
    )
    transformed = sphere.val().transformGeometry(matrix)
    return cq.Workplane("XY").newObject([transformed])


def _make_wedge(size_x: float, size_y: float, size_z: float) -> cq.Workplane:
    """Right-triangular prism wedge.

    Cross-section in the X-Z plane: a right triangle with the hypotenuse
    rising from ``(-size_x/2, -size_z/2)`` to ``(+size_x/2, +size_z/2)``,
    extruded along y for ``size_y``. Centred on the origin.
    """
    hx = size_x / 2.0
    hy = size_y / 2.0
    hz = size_z / 2.0
    pts = [(-hx, -hz), (+hx, -hz), (+hx, +hz)]
    return (
        cq.Workplane("XZ")
        .polyline(pts)
        .close()
        .extrude(size_y, both=False)
        .translate((0.0, -hy, 0.0))
    )


_SHAPE_BUILDERS = {
    "slot": _make_slot,
    "ellipsoid": _make_ellipsoid,
    "wedge": _make_wedge,
}


def _rotate_extrinsic_xyz(
    shape: cq.Workplane,
    rx: float,
    ry: float,
    rz: float,
) -> cq.Workplane:
    """Apply extrinsic X → Y → Z Euler rotation about the world origin."""
    if abs(rx) > 1e-12:
        shape = shape.rotate((0, 0, 0), (1, 0, 0), math.degrees(rx))
    if abs(ry) > 1e-12:
        shape = shape.rotate((0, 0, 0), (0, 1, 0), math.degrees(ry))
    if abs(rz) > 1e-12:
        shape = shape.rotate((0, 0, 0), (0, 0, 1), math.degrees(rz))
    return shape


def apply_primitive(
    shape: cq.Workplane,
    primitive: Layer3Primitive,
) -> cq.Workplane:
    """Apply the Layer 3 capped primitive to ``shape``.

    Parameters
    ----------
    shape : cq.Workplane
        The shape after Layers 1 + 2 (envelope + fields). Returned
        unmodified when ``primitive.present`` is ``False``.
    primitive : Layer3Primitive
        Validated Layer 3 design parameters. Schema bounds (margin,
        size, rotation, envelope-fraction) are upstream-enforced.

    Returns
    -------
    cq.Workplane
        The shape with the primitive Boolean-fused (``polarity="add"``)
        or cut (``polarity="subtract"``).

    Raises
    ------
    ValueError
        If ``primitive.shape_type`` is not in the registered builders.
        ``Layer3Primitive`` rejects unknown shape types at construction
        time, so this branch is defensive.
    Exception
        Any CadQuery / OpenCascade Boolean failure. Per plan §9.7 the
        orchestrator catches these and degrades to LAYER3_FAILED.
    """
    if not primitive.present:
        return shape

    builder = _SHAPE_BUILDERS.get(primitive.shape_type)
    if builder is None:  # pragma: no cover -- schema rejects unknown types
        raise ValueError(
            f"apply_primitive: unknown shape_type {primitive.shape_type!r}; "
            f"Layer3Primitive should have rejected this upstream"
        )

    prim = builder(primitive.size_x_m, primitive.size_y_m, primitive.size_z_m)
    prim = _rotate_extrinsic_xyz(
        prim,
        primitive.rotation_x_rad,
        primitive.rotation_y_rad,
        primitive.rotation_z_rad,
    )
    prim = prim.translate(
        (
            primitive.position_x_m,
            primitive.position_y_m,
            primitive.position_z_m,
        )
    )

    if primitive.polarity == "subtract":
        return shape.cut(prim)
    return shape.union(prim)
