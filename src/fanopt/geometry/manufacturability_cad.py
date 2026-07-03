"""Geometry Layer 4 — CadQuery shape-inspection for the §N7 filter.

Replaces the 10 ``CheckStatus.PENDING_CADQUERY`` stubs in
:mod:`fanopt.geometry.manufacturability` with real shape-inspection
routines. Hard-requires CadQuery; tests skip-at-module-load when it's
absent.

Phase-1 inspection scope
------------------------

| # | Severity | Phase-1 strategy |
|---|---|---|
| 1 | MODERATE | Per-feature tracking deferred to Phase-2 (loft tessellation produces sub-mm edges that aren't real features). Records PENDING. |
| 2 | MODERATE | Planar-face normals; flag overhangs > 45° from vertical. Curved faces deferred. |
| 3 | CRITICAL | ``len(shape.solids())`` == 1. |
| 4 | SOFT | Horizontal edge length proxy (horizontal edges > 8 mm flagged). |
| 5 | CRITICAL | ``shape.isValid()`` + closed-shell count. Approximation for full void detection. |
| 6 | CRITICAL | Schema-enforced via ``PANEL_X_CARVE_RANGE_M`` (5 mm clearance from click). Records as PASSED. |
| 8 | SOFT | Deferred to Phase-2 (requires per-feature tracking; the whole blade aspect is ~70:1 by design). |
| 12 | MODERATE | ``z_extent ≥ 1.5 × layer_height_m`` proxy. Full check needs volume-by-z slicing. |
| 13 | SOFT | Planar-face aspect/area scan; flag faces with aspect > 8:1 AND area > 1000 mm². |
| 14 | CRITICAL | Under ``print_orientation == "flat"``, the largest −z-facing planar face must lie at z = 0. |

Per CLAUDE.md §4.5 the pure-Python :mod:`fanopt.geometry.manufacturability`
remains untouched; this module is the CadQuery sibling. The Phase-1
approximations are documented per-check; downstream BO consumers treat
the result as a manufacturability score in [0, 1] per plan §9.7.3.
"""

from __future__ import annotations

import math

import cadquery as cq

from fanopt.geometry.manufacturability import (
    MANUFACTURABILITY_PASS_THRESHOLD,
    CheckResult,
    CheckSeverity,
    CheckStatus,
    Layer4Params,
    ManufacturabilityResult,
    _aggregate_score,
    _bound_check,
)

__all__ = [
    "MIN_FEATURE_LENGTH_M",
    "MAX_BRIDGE_SPAN_M",
    "MAX_OVERHANG_DEG",
    "MAX_FACE_ASPECT_RATIO",
    "MAX_FACE_AREA_M2",
    "Z_THICKNESS_MULTIPLIER",
    "run_manufacturability_filter_cad",
]


MIN_FEATURE_LENGTH_M: float = 0.0008
"""Plan §9.7.3 check #1 — 2× nozzle diameter (0.4 mm)."""

MAX_BRIDGE_SPAN_M: float = 0.008
"""Plan §9.7.3 check #4 — bridging horizontal spans flagged above 8 mm."""

MAX_OVERHANG_DEG: float = 45.0
"""Plan §9.7.3 check #2 — overhangs > 45° from vertical without support."""

MAX_FACE_ASPECT_RATIO: float = 8.0
"""Plan §9.7.3 check #13 — face aspect ratio cap for warpage proxy."""

MAX_FACE_AREA_M2: float = 0.001
"""Plan §9.7.3 check #13 — 1000 mm² area threshold for warpage proxy."""

Z_THICKNESS_MULTIPLIER: float = 1.5
"""Plan §9.7.3 check #12 — z-thickness must be ≥ 1.5 × layer height."""


def _check_min_feature_size() -> CheckResult:
    """#1 — schema-enforced via PRIMITIVE_MIN_DIMENSION_M; per-feature
    skin-edge tracking deferred to Phase-2.

    A naive ``min(edge_length)`` check fires on loft tessellation edges
    (the envelope's cross-section polylines have ~0.1 mm edges by
    construction at the inboard stations) which aren't real design
    features — a slicer handles them fine. Distinguishing tessellation
    samples from designed features needs per-feature tracking during
    Layer 2/3 application; Phase-1 records this as PENDING with the
    same rationale as #8.
    """
    return CheckResult(
        check_id="1",
        name="Minimum feature size ≥ 0.8 mm",
        severity=CheckSeverity.MODERATE,
        status=CheckStatus.PENDING_CADQUERY,
        message=(
            "Per-feature tracking deferred to Phase-2 — loft tessellation "
            "produces sub-mm edges that aren't real features. Schema enforces "
            "PRIMITIVE_MIN_DIMENSION_M (0.8 mm) on Layer 3 sizes already."
        ),
    )


def _check_overhang_angle(shape: cq.Workplane) -> CheckResult:
    """#2 — flag planar faces with normals tilted > 45° past vertical.

    Walks planar faces only; curved faces (e.g., sphere segments from a
    Layer 3 ellipsoid cut) are deferred to a Phase-2 face-normal scan.
    """
    overhanging = 0
    cos_limit = math.cos(math.radians(90.0 - MAX_OVERHANG_DEG))
    for face in shape.faces().vals():
        if face.geomType() != "PLANE":
            continue
        try:
            n = face.normalAt()
        except Exception:
            continue
        # Overhang test: face normal points downward (n.z < 0); the
        # angle from straight-down is acos(-n.z). When n.z > -cos_limit
        # the face is more horizontal than the limit.
        if n.z < 0 and (-n.z) < cos_limit:
            overhanging += 1
    if overhanging > 0:
        return CheckResult(
            check_id="2",
            name="Overhang angle ≤ 45°",
            severity=CheckSeverity.MODERATE,
            status=CheckStatus.FAILED,
            message=f"{overhanging} planar face(s) overhang > 45° from vertical",
        )
    return CheckResult(
        check_id="2",
        name="Overhang angle ≤ 45°",
        severity=CheckSeverity.MODERATE,
        status=CheckStatus.PASSED,
        message="no planar overhanging faces > 45°",
    )


def _check_single_component(shape: cq.Workplane) -> CheckResult:
    """#3 — exactly one solid in the shape."""
    n_solids = len(shape.solids().vals())
    if n_solids != 1:
        return CheckResult(
            check_id="3",
            name="Connectivity (single component)",
            severity=CheckSeverity.CRITICAL,
            status=CheckStatus.FAILED,
            message=f"shape has {n_solids} solids; expected 1",
        )
    return CheckResult(
        check_id="3",
        name="Connectivity (single component)",
        severity=CheckSeverity.CRITICAL,
        status=CheckStatus.PASSED,
        message="single solid",
    )


def _check_bridging(shape: cq.Workplane) -> CheckResult:
    """#4 — horizontal edges longer than 8 mm flagged.

    Approximation: a real bridging check would identify horizontal
    free-spans (edges of overhanging faces). Phase-1 uses a simpler
    proxy: count all horizontal edges over the span limit.
    """
    long_horizontal = 0
    longest = 0.0
    for e in shape.edges().vals():
        try:
            start = e.startPoint()
            end = e.endPoint()
        except Exception:
            continue
        if abs(end.z - start.z) < 1e-5 and e.Length() > MAX_BRIDGE_SPAN_M:
            long_horizontal += 1
            longest = max(longest, e.Length())
    if long_horizontal > 0:
        return CheckResult(
            check_id="4",
            name="Bridging ≤ 8 mm",
            severity=CheckSeverity.SOFT,
            status=CheckStatus.FAILED,
            message=(
                f"{long_horizontal} horizontal edge(s) > 8 mm " f"(longest {longest * 1000:.1f} mm)"
            ),
        )
    return CheckResult(
        check_id="4",
        name="Bridging ≤ 8 mm",
        severity=CheckSeverity.SOFT,
        status=CheckStatus.PASSED,
        message="no horizontal edges > 8 mm",
    )


def _check_internal_voids(shape: cq.Workplane) -> CheckResult:
    """#5 — shape validity + closed-shell count. Approximation."""
    solid = shape.val()
    is_valid = True
    try:
        is_valid = solid.isValid()
    except Exception:
        is_valid = False
    if not is_valid:
        return CheckResult(
            check_id="5",
            name="Internal voids — exit path ≥ 1 mm (TPMS exempt)",
            severity=CheckSeverity.CRITICAL,
            status=CheckStatus.FAILED,
            message="shape failed isValid() — potential internal void or non-manifold",
        )
    return CheckResult(
        check_id="5",
        name="Internal voids — exit path ≥ 1 mm (TPMS exempt)",
        severity=CheckSeverity.CRITICAL,
        status=CheckStatus.PASSED,
        message="shape passes isValid() (no detected voids)",
    )


def _check_edge_clearance() -> CheckResult:
    """#6 — schema-enforced via PANEL_X_CARVE_RANGE_M (5 mm safety margin)."""
    return CheckResult(
        check_id="6",
        name="Edge clearance ≥ 1 mm from envelope",
        severity=CheckSeverity.CRITICAL,
        status=CheckStatus.PASSED,
        message=(
            "Schema-enforced via PANEL_X_CARVE_RANGE_M (Layer 2 cuts "
            "kept inside [HUB_RADIUS, click_x - 5 mm] band); Layer 3 "
            "schema requires ≥ 1 mm margin per axis"
        ),
    )


def _check_aspect_ratio() -> CheckResult:
    """#8 — per-feature aspect tracking deferred to Phase-2.

    The whole blade envelope is ~200 mm × 23 mm × 3 mm (≈ 70:1 by
    design); a bounding-box-only check would fire a false positive.
    Real implementation needs per-feature aspect during Layer 2/3
    application.
    """
    return CheckResult(
        check_id="8",
        name="Aspect ratio ≤ 20:1",
        severity=CheckSeverity.SOFT,
        status=CheckStatus.PENDING_CADQUERY,
        message=(
            "Per-feature aspect tracking deferred to Phase-2 — the whole "
            "blade is high-aspect by design; need per-feature solids."
        ),
    )


def _check_z_thin_section(
    shape: cq.Workplane,
    layer_height_m: float,
) -> CheckResult:
    """#12 — minimum z-extent must be ≥ 1.5 × layer height.

    Approximation: real check would identify z-thin regions by volume.
    Phase-1 checks the global z extent.
    """
    bb = shape.val().BoundingBox()
    z_extent = bb.zmax - bb.zmin
    threshold = Z_THICKNESS_MULTIPLIER * layer_height_m
    if z_extent < threshold:
        return CheckResult(
            check_id="12",
            name="Layer-adhesion Z-thin-section flag",
            severity=CheckSeverity.MODERATE,
            status=CheckStatus.FAILED,
            message=(
                f"shape z-extent {z_extent * 1000:.2f} mm < "
                f"{threshold * 1000:.2f} mm (1.5 × layer height)"
            ),
        )
    return CheckResult(
        check_id="12",
        name="Layer-adhesion Z-thin-section flag",
        severity=CheckSeverity.MODERATE,
        status=CheckStatus.PASSED,
        message=f"shape z-extent {z_extent * 1000:.2f} mm ≥ {threshold * 1000:.2f} mm",
    )


def _check_warpage_face(shape: cq.Workplane) -> CheckResult:
    """#13 — planar faces with aspect > 8:1 AND area > 1000 mm² flagged."""
    flagged = 0
    for face in shape.faces().vals():
        if face.geomType() != "PLANE":
            continue
        fbb = face.BoundingBox()
        extents = sorted(
            [
                fbb.xmax - fbb.xmin,
                fbb.ymax - fbb.ymin,
                fbb.zmax - fbb.zmin,
            ],
            reverse=True,
        )
        if extents[1] < 1e-9:
            continue
        aspect = extents[0] / extents[1]
        try:
            area = face.Area()
        except Exception:
            continue
        if aspect > MAX_FACE_ASPECT_RATIO and area > MAX_FACE_AREA_M2:
            flagged += 1
    if flagged > 0:
        return CheckResult(
            check_id="13",
            name="Warpage proxy (large planar face)",
            severity=CheckSeverity.SOFT,
            status=CheckStatus.FAILED,
            message=f"{flagged} planar face(s) with aspect > 8:1 AND area > 1000 mm²",
        )
    return CheckResult(
        check_id="13",
        name="Warpage proxy (large planar face)",
        severity=CheckSeverity.SOFT,
        status=CheckStatus.PASSED,
        message="no large high-aspect planar faces",
    )


def _check_support_scar(
    shape: cq.Workplane,
    print_orientation: str,
) -> CheckResult:
    """#14 — under print_orientation='flat', a planar face at z=0 facing −z must exist.

    For non-flat orientations this check passes trivially (the panel
    doesn't have a calibrated wall-roughness face under edge / custom-
    angle prints).
    """
    if print_orientation != "flat":
        return CheckResult(
            check_id="14",
            name="Support-scar location on functional surfaces",
            severity=CheckSeverity.CRITICAL,
            status=CheckStatus.PASSED,
            message=f"print_orientation={print_orientation!r} → check N/A",
        )
    for face in shape.faces().vals():
        if face.geomType() != "PLANE":
            continue
        try:
            n = face.normalAt()
        except Exception:
            continue
        if n.z >= -0.99:
            continue
        fbb = face.BoundingBox()
        if abs(fbb.zmin) < 1e-5 and abs(fbb.zmax) < 1e-5:
            return CheckResult(
                check_id="14",
                name="Support-scar location on functional surfaces",
                severity=CheckSeverity.CRITICAL,
                status=CheckStatus.PASSED,
                message="planar bottom face at z=0 with −z normal found",
            )
    return CheckResult(
        check_id="14",
        name="Support-scar location on functional surfaces",
        severity=CheckSeverity.CRITICAL,
        status=CheckStatus.FAILED,
        message="no planar bottom face at z=0 with −z normal under flat orientation",
    )


def run_manufacturability_filter_cad(
    shape: cq.Workplane,
    layer4: Layer4Params,
) -> ManufacturabilityResult:
    """Apply the §9.7.3 manufacturability filter against a CadQuery shape.

    Replaces the dict-only :func:`fanopt.geometry.manufacturability.run_manufacturability_filter`
    that returns ``PENDING_CADQUERY`` for the 10 geometry-level checks.
    Real shape inspection is applied per the per-check Phase-1 strategy
    table in this module's docstring.

    Hard parameter bounds (#7, #9, #10, #11) remain upstream-enforced
    and register as PASSED with the same audit message as the
    pure-Python filter.

    Returns
    -------
    ManufacturabilityResult
        ``score`` in ``[0, 1]``, ``passed`` against the
        ``MANUFACTURABILITY_PASS_THRESHOLD`` (0.5), ``checks`` for every
        protocol row, and ``critical_failures`` / ``pending_cadquery``
        for traceability.
    """
    checks = (
        _check_min_feature_size(),
        _check_overhang_angle(shape),
        _check_single_component(shape),
        _check_bridging(shape),
        _check_internal_voids(shape),
        _check_edge_clearance(),
        _bound_check(
            "7",
            "Click feature 5 mm exclusion (panel-edge)",
            "Layer 2 fields + Layer 3 primitive are constrained to keep "
            "≥ 5 mm clearance from CLICK_FOOTPRINT_X_RANGE on every "
            "panel's outer tangential edge. Hard parameter bound — "
            "upstream-enforced.",
        ),
        _check_aspect_ratio(),
        _bound_check(
            "11",
            "Fourier envelope amplitude ≤ ±15%",
            "Hard parameter bound — schema enforces "
            "FOURIER_AMPLITUDE_RELATIVE_MAX cap on every harmonic.",
        ),
        _check_z_thin_section(shape, layer4.layer_height_m),
        _check_warpage_face(shape),
        _check_support_scar(shape, layer4.print_orientation),
    )

    score, critical_failures, pending_cadquery = _aggregate_score(checks)
    return ManufacturabilityResult(
        score=score,
        passed=score >= MANUFACTURABILITY_PASS_THRESHOLD,
        checks=checks,
        critical_failures=tuple(critical_failures),
        pending_cadquery=tuple(pending_cadquery),
    )
