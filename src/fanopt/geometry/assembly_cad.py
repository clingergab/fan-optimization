"""V-unit blade composition — panel + 2 ribs + pivot boss + click features.

Implements plan §2.1 / §2.2 / §2.3 V-unit assembly on top of the
``generator_cad`` panel output. The V-unit is one printable blade:

* **Panel** — the aerodynamic surface (envelope ∘ Layer 2 fields ∘
  Layer 3 primitive), already produced by ``generate_blade_cad``.
* **2 ribs** — tapered structural bars along the panel's tangential
  y-edges, radial extent ``[HUB_RADIUS_M, L_BLADE_M − RIB_TIP_TAPER_M]``,
  width tapers ``RIB_BASE_WIDTH_M → RIB_TIP_WIDTH_M`` per H12 lock,
  thickness ``RIB_THICKNESS_M``. Panel-pivot architecture: the ribs are
  NEVER pierced by the pivot pin (rib at ``y = ±panel_half_pitch(x)``,
  pivot pin at ``y = 0``).
* **Pivot boss** — 12 mm OD circular boss centered on the pivot pin.
  Phase-1: integrated into the panel envelope at the hub region; this
  module adds an extra ``RIB_THICKNESS_M``-tall extrusion to give the
  panel-pivot region positive build-up over the bare envelope.
* **Click chamfer** — Round-9 HIGH-8 Option A: a 0.5–1 mm corner bevel
  at the panel's outer tangential edge inside ``CLICK_FOOTPRINT_X_RANGE_M``.
  NOT a full-panel-thickness face — a small triangular wedge subtracted
  from the (x = L_BLADE_M, |y| = panel_outer, z = top) corner at
  ``layer4.click_chamfer_angle_deg`` (45° default per the H8 lock).
* **Click detent** — hemispherical bump of radius
  ``layer4.click_detent_size_m`` on the +y outer edge at the click
  footprint centre, providing the "click" engagement with the adjacent
  blade's mating cup.

Phase-1 approximations and follow-ups
-------------------------------------

* Rib geometry is a single linear loft between two end cross-sections,
  not a sweep along the curved y-edge. The panel's y-edge angle is
  ``arctan(INTER_BLADE_ANGLE_RAD / 2) ≈ 6.65°`` so the linear loft is
  a tight approximation; a Phase-2 sweep refines it.
* The receiving cup mating the detent (on the adjacent blade) is NOT
  added — only the outgoing detent bump. Adjacent-blade mating is
  assembled at fan-composition time, not blade-level.
* The Layer 4 ``click_design_clearance_m`` is recorded in the assembly
  metadata but NOT applied as a per-face offset; Phase-2 work.
* The chamfer is applied via CadQuery ``.chamfer`` on the panel's
  outer-tangential corner edges inside the click footprint, with size
  fixed at the midpoint of ``CLICK_CHAMFER_BEVEL_RANGE_M`` (0.75 mm)
  per the lock. The ``click_chamfer_angle_deg`` only affects the
  panel-pair butt-joint angle (45° lock); the chamfer size itself is
  not a BO variable.
"""

from __future__ import annotations

import cadquery as cq

from fanopt.geometry.generator import BladeDesignParams
from fanopt.geometry.generator_cad import generate_blade_cad
from fanopt.geometry.schema import (
    CLICK_CHAMFER_BEVEL_RANGE_M,
    CLICK_FOOTPRINT_X_RANGE_M,
    HUB_RADIUS_M,
    INTER_BLADE_ANGLE_RAD,
    L_BLADE_M,
    PIVOT_BOSS_RADIUS_M,
    PIVOT_CENTER_X_M,
    RIB_BASE_WIDTH_M,
    RIB_THICKNESS_M,
    RIB_TIP_TAPER_M,
    RIB_TIP_WIDTH_M,
    click_footprint_y_range_panel_edge_m,
    panel_tangential_outer_at_tip_m,
)

__all__ = [
    "CHAMFER_BEVEL_M",
    "make_vunit_blade",
    "make_rib",
    "make_pivot_boss",
]


CHAMFER_BEVEL_M: float = sum(CLICK_CHAMFER_BEVEL_RANGE_M) / 2.0
"""0.75 mm — midpoint of the locked 0.5-1 mm bevel range."""


def _rib_y_at(x: float) -> float:
    """y-position of the panel-edge midline at radial x (positive side).

    The panel's tangential half-pitch at x is ``x · INTER_BLADE_ANGLE / 2``.
    The rib midline runs along this edge.
    """
    return x * INTER_BLADE_ANGLE_RAD / 2.0


def make_rib(side: str) -> cq.Workplane:
    """Construct one rib bar along the LE or TE panel edge.

    Lofts between the hub-end and tip-end rectangular cross-sections.
    The cross-section is centred on the panel y-edge midline; width
    tapers ``RIB_BASE_WIDTH_M → RIB_TIP_WIDTH_M``; thickness is
    ``RIB_THICKNESS_M`` along z, sitting flush with the panel bottom
    at z = 0 under flat orientation.

    Parameters
    ----------
    side : str
        ``"LE"`` (y < 0 panel edge) or ``"TE"`` (y > 0 panel edge).
    """
    if side not in ("LE", "TE"):
        raise ValueError(f"side must be 'LE' or 'TE', got {side!r}")

    sign = -1.0 if side == "LE" else +1.0
    x_root = HUB_RADIUS_M
    x_tip = L_BLADE_M - RIB_TIP_TAPER_M

    y_root_centre = sign * _rib_y_at(x_root)
    y_tip_centre = sign * _rib_y_at(x_tip)

    # Cross-section at each end: rectangle in (y, z) centred on the
    # rib midline. y-width tapers along x; z spans [0, RIB_THICKNESS_M].
    def section_pts(y_c: float, half_w: float) -> list[tuple[float, float]]:
        return [
            (y_c - half_w, 0.0),
            (y_c + half_w, 0.0),
            (y_c + half_w, RIB_THICKNESS_M),
            (y_c - half_w, RIB_THICKNESS_M),
        ]

    root_pts = section_pts(y_root_centre, RIB_BASE_WIDTH_M / 2.0)
    tip_pts = section_pts(y_tip_centre, RIB_TIP_WIDTH_M / 2.0)

    wp = (
        cq.Workplane("YZ", origin=(x_root, 0.0, 0.0))
        .polyline(root_pts)
        .close()
        .workplane(offset=x_tip - x_root, centerOption="ProjectedOrigin")
        .polyline(tip_pts)
        .close()
        .loft(combine=True)
    )
    return wp


def make_pivot_boss() -> cq.Workplane:
    """12 mm OD circular boss centred on the pivot pin.

    Sits at z ∈ [0, RIB_THICKNESS_M] so it shares the print-bed face
    with the panel + ribs under flat orientation. Radius =
    ``PIVOT_BOSS_RADIUS_M`` (6 mm). The 3 mm pivot pin hole is NOT
    drilled here — it's added at fan-composition time so the boss
    can mate with the pin's locked diameter.
    """
    return (
        cq.Workplane("XY")
        .circle(PIVOT_BOSS_RADIUS_M)
        .extrude(RIB_THICKNESS_M)
        .translate((PIVOT_CENTER_X_M, 0.0, 0.0))
    )


def _apply_click_chamfer(blade: cq.Workplane) -> cq.Workplane:
    """Round-9 HIGH-8 Option A chamfer.

    Subtracts a small triangular wedge from each outer-tangential
    corner of the panel inside ``CLICK_FOOTPRINT_X_RANGE_M``. The
    wedge is sized to ``CHAMFER_BEVEL_M`` and oriented at 45°
    (right-hand-rule per the H8 lock).

    The chamfer is applied as a Boolean subtract rather than CadQuery's
    ``.chamfer()`` method because ``.chamfer()`` requires identifying
    specific edges, which is brittle on Boolean-rich shapes; the
    explicit wedge subtract is geometry-stable.
    """
    x_lo, x_hi = CLICK_FOOTPRINT_X_RANGE_M
    bb = blade.val().BoundingBox()
    z_top = bb.zmax
    bevel = CHAMFER_BEVEL_M

    cutters: list[cq.Workplane] = []
    for sign in (-1.0, +1.0):
        y_outer = sign * panel_tangential_outer_at_tip_m()
        # Wedge: triangular prism with the diagonal cutting from the
        # outer y-edge to the top z-face. Polyline in (y, z) at the
        # corner; extrude along x for the click footprint length.
        pts = [
            (y_outer, z_top),
            (y_outer, z_top - bevel),
            (y_outer - sign * bevel, z_top),
        ]
        wedge = (
            cq.Workplane("YZ", origin=(x_lo, 0.0, 0.0))
            .polyline(pts)
            .close()
            .extrude(x_hi - x_lo)
        )
        cutters.append(wedge)

    union = cutters[0].val()
    for c in cutters[1:]:
        union = union.fuse(c.val())
    return blade.cut(cq.Workplane("XY").newObject([union]))


def _apply_click_detent(
    blade: cq.Workplane,
    detent_radius_m: float,
    blade_count: int,
) -> cq.Workplane:
    """Add a hemispherical detent bump on the +y outer edge.

    Placed at the centre of ``CLICK_FOOTPRINT_X_RANGE_M`` along x, on
    the panel's outer tangential edge at that x (the panel y-edge
    tapers with x). The sphere centre sits exactly on the y-edge so
    half the sphere is inside the panel material (fuses cleanly) and
    half protrudes outward as the engagement bump. Radius from
    ``layer4.click_detent_size_m``. The mating receiving cup on the
    adjacent blade is NOT added here — adjacent-blade mating is
    fan-composition-level work.
    """
    del blade_count  # Phase-1: panel y_max at the click x uses the locked geometry
    x_lo, x_hi = CLICK_FOOTPRINT_X_RANGE_M
    x_centre = (x_lo + x_hi) / 2.0
    # Panel half-pitch at the click-centre x — the geometric y-edge.
    y_edge = x_centre * INTER_BLADE_ANGLE_RAD / 2.0
    # Embed the sphere centre 1/3 of the radius INSIDE the panel material
    # so the Boolean fusion has a robust overlap volume; a tangent-contact
    # placement at y=y_edge leaves OpenCascade with a thin-shell artifact
    # and the fuse returns a Compound with two pieces.
    y_centre = y_edge - detent_radius_m / 3.0
    bb = blade.val().BoundingBox()
    z_mid = (bb.zmin + bb.zmax) / 2.0

    bump = (
        cq.Workplane("XY")
        .sphere(detent_radius_m)
        .translate((x_centre, y_centre, z_mid))
    )
    return blade.union(bump, clean=True)


def make_vunit_blade(params: BladeDesignParams) -> cq.Workplane:
    """Compose one V-unit blade: panel + 2 ribs + pivot boss + click features.

    Phase 1 deliverable. The output is a single printable solid (under
    flat orientation, plano-convex with z ≥ 0 throughout) representing
    one of N blades in the deployed fan.

    Parameters
    ----------
    params : BladeDesignParams
        Validated 4-layer design. Layer 1 envelope + Layer 2 fields +
        Layer 3 primitive go into the panel via ``generate_blade_cad``;
        Layer 4 click-feature params drive the chamfer + detent.

    Returns
    -------
    cq.Workplane
        Single solid wrapped in a Workplane. Bounding box spans
        ``x ∈ [HUB_RADIUS_M, L_BLADE_M]``, ``y`` includes the rib
        extents past the panel edges, ``z`` includes the rib
        thickness + the panel camber.
    """
    _result, panel = generate_blade_cad(params)
    rib_le = make_rib("LE")
    rib_te = make_rib("TE")
    boss = make_pivot_boss()

    blade = (
        panel.union(rib_le, clean=True)
        .union(rib_te, clean=True)
        .union(boss, clean=True)
    )
    blade = _apply_click_chamfer(blade)
    blade = _apply_click_detent(
        blade,
        detent_radius_m=params.layer4.click_detent_size_m,
        blade_count=params.layer1.blade_count,
    )
    return blade


def _click_footprint_y_centre(blade_count: int) -> float:
    """Helper for tests + downstream consumers — the y midline of the
    click footprint band at the tip."""
    lo, hi = click_footprint_y_range_panel_edge_m(blade_count=blade_count)
    return (lo + hi) / 2.0
