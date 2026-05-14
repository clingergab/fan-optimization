"""Spike 0.7a — Generative-geometry sanity check (library half).

Spec reference: ``docs/plan_R11.md §Phase 0 Spike 0.7`` (sub-spike 0.7a).
Protocol: ``docs/spike_0_7a_protocol.md``.

Purpose
-------
Phase 2b runs the CadQuery 4-layer generator across ~37-46 design variables.
Spike 0.7a is the *gate* that catches three failure modes before Phase 2b
launches:

1. **Random parameter coverage** — 10 random parameter dicts drawn from the
   JSON-schema bounds must round-trip through the generator + manufacturability
   filter with a sane pass rate (≥ 7/10 pass OR ≥ 3 fail with documented
   reasons).
2. **Click-feature footprint invariant** — the click chamfer footprint at
   ``(x = L_blade ± rib_tip_taper, y = ±panel_tangential_outer ≈ ±0.0225 m)``
   must be bit-for-bit intact on every blade, regardless of how aggressively
   Layer 2/3 carving is configured.
3. **Panel-domain invariant** (§9.7.1) — Layer 2/3 carving must never reach
   into the rib material from the Phase 2 SIMP TO output. The rib region is
   ``[HUB_RADIUS, L_blade − RIB_TIP_TAPER] × ([rib_center − rib_width/2,
   rib_center + rib_width/2] ∪ [−rib_center − rib_width/2, −rib_center +
   rib_width/2])``.

Hand-picked adversarial parameter sets exercise each invariant on the
boundary; ``analyze_07a`` enforces that **every** adversarial set is blocked
(by the manufacturability filter OR by the click / rib check).

Design notes
------------
The four checks (``generator_fn``, ``manuf_fn``, ``click_check_fn``,
``rib_check_fn``) are *injected* so the library is testable without the full
CadQuery pipeline. Production callers wire in the real implementations from
``fanopt.geometry.{generator,manufacturability}``; tests substitute fakes that
return canned pass/fail outcomes.

Until Phase 1 lands the real generator, the production wiring uses a small
**bounding-box shim** (see ``scripts/run_spike_0_7a.py``) that performs the
weaker "no carving in the rib bbox" check. The bbox check is strictly weaker
than the bit-for-bit panel-domain invariant the spike specifies, BUT it
catches the failure mode the adversarial sets are designed to expose. The
shim is clearly marked in its docstring and the protocol documents the
limitation.

Pass criterion (``Spike07aResult.passed``):
    * ALL adversarial sets are blocked (rejected by manufacturability OR
      fail click/rib check), AND
    * (≥ 7 of 10 random sets pass) OR (≥ 3 fail with non-empty
      ``rejection_reasons``).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

__all__ = [
    "HUB_RADIUS_M",
    "L_BLADE_M",
    "RIB_TIP_TAPER_M",
    "PANEL_TANGENTIAL_OUTER_M",
    "SCHEMA_BOUNDS",
    "ADVERSARIAL_PARAM_SETS",
    "GeomSanityRecord",
    "Spike07aResult",
    "RANDOM_PASS_FRACTION_GATE",
    "RANDOM_DOCUMENTED_FAIL_GATE",
    "random_param_set_within_bounds",
    "evaluate_param_set",
    "analyze_07a",
    "hash_params",
]

# ── Architectural locks (CLAUDE.md panel-pivot architecture) ─────────────
HUB_RADIUS_M: float = 0.020
L_BLADE_M: float = 0.200
RIB_TIP_TAPER_M: float = 0.015
# Panel half-width at the tip (Round-9 HIGH-8 Option A; widened per item #35 lock).
PANEL_TANGENTIAL_OUTER_M: float = 0.0225

# Pass-criterion gates from §Phase 0 Spike 0.7a.
RANDOM_PASS_FRACTION_GATE: float = 0.7  # ≥7/10 random sets must pass
RANDOM_DOCUMENTED_FAIL_GATE: int = 3  # OR ≥3 fail with documented reasons


# ── Schema bounds shim ───────────────────────────────────────────────────
#
# `src/fanopt/geometry/schema.py` is a Phase 0 scaffold (empty until Phase 1
# implements the full ~37-46-variable JSON schema). To unblock Spike 0.7a we
# inline the locked subset of bounds we know about (per `docs/plan_R11.md`
# §6.2.1 / §3.2). When the real schema lands, replace this dict with a
# `from fanopt.geometry.schema import SCHEMA_BOUNDS` import.
#
# Each entry is one of:
#   ("float", lo, hi)              — continuous, drawn uniformly in [lo, hi]
#   ("int", lo, hi)                — discrete uniform in [lo, hi] inclusive
#   ("choice", [v1, v2, ...])      — uniform pick from a categorical list
#   ("bool",)                      — Bernoulli(0.5)
#
# Convention: this is the same schema the BO infrastructure will sample from.
# The bounds below match the locked ranges in plan_R11.md; extending them
# requires bumping `SCHEMA_VERSION` and re-running this spike.
SCHEMA_VERSION: str = "spike_0_7a_shim_v1"

SCHEMA_BOUNDS: dict[str, tuple[Any, ...]] = {
    # Layer 1 — outer envelope
    "blade_count": ("choice", [8, 10, 12]),  # C8 lock; 14-blade is retired
    "panel_thickness_root_m": ("float", 0.0022, 0.0038),  # C7 lock
    "panel_thickness_mid_m": ("float", 0.0022, 0.0038),
    "panel_thickness_tip_m": ("float", 0.0022, 0.0038),
    "rib_base_width_m": ("float", 0.004, 0.004),  # locked per Round-9 H12
    "rib_tip_width_m": ("float", 0.006, 0.006),  # locked per Round-9 H12
    "camber_c0": ("float", -0.02, 0.02),
    "camber_c1": ("float", -0.02, 0.02),
    "camber_c2": ("float", -0.02, 0.02),
    "twist_root_rad": ("float", -0.10, 0.10),
    "twist_tip_rad": ("float", -0.10, 0.10),
    "edge_profile": ("choice", ["sharp", "rounded", "blunt"]),
    "fourier_le_a1": ("float", -0.0015, 0.0015),
    "fourier_le_a2": ("float", -0.0015, 0.0015),
    "fourier_te_a1": ("float", -0.0015, 0.0015),
    "fourier_te_a2": ("float", -0.0015, 0.0015),
    # Layer 2 — macro-pattern fields (0-3 of 5 active)
    "louver_active": ("bool",),
    "louver_spacing_m": ("float", 0.005, 0.020),
    "louver_depth_m": ("float", 0.0002, 0.0010),
    "louver_angle_rad": ("float", -0.5, 0.5),
    "louver_cluster_tip": ("float", 0.0, 1.0),  # 0 = uniform, 1 = clustered at tip
    "texture_active": ("bool",),
    "texture_amp_m": ("float", 0.0001, 0.0008),
    "texture_freq_per_m": ("float", 50.0, 500.0),
    "edge_feature_active": ("bool",),
    "edge_feature_depth_m": ("float", 0.0002, 0.0010),
    "noise_active": ("bool",),
    "noise_threshold": ("float", 0.3, 0.9),
    "tpms_active": ("bool",),
    "tpms_cell_size_m": ("float", 0.006, 0.030),  # min cell = 3× min feature
    "tpms_rotation_rad": ("float", 0.0, 1.5708),
    # Layer 3 — capped 0-1 primitive
    "prim_active": ("bool",),
    "prim_type": ("choice", ["slot", "ellipsoid", "wedge"]),
    "prim_polarity": ("choice", ["add", "subtract"]),
    "prim_x_m": ("float", HUB_RADIUS_M + 0.005, L_BLADE_M - RIB_TIP_TAPER_M - 0.005),
    "prim_y_m": ("float", -PANEL_TANGENTIAL_OUTER_M, PANEL_TANGENTIAL_OUTER_M),
    "prim_size_x_m": ("float", 0.003, 0.015),
    "prim_size_y_m": ("float", 0.003, 0.015),
    "prim_rotation_rad": ("float", 0.0, 1.5708),
    # Layer 4 — manufacturing
    "print_orientation": ("choice", ["flat", "edge"]),
    "layer_height_m": ("float", 0.00015, 0.00030),
    "chamfer_depth_m": ("float", 0.0005, 0.0010),  # Round-9 HIGH-8 Option A
    "detent_size_m": ("float", 0.0003, 0.0005),
    "design_clearance_m": ("float", 0.00015, 0.00020),
}


def hash_params(params: dict[str, Any]) -> str:
    """Stable short hash of a parameter dict, for record identity in logs.

    JSON-encoded with sorted keys so two equivalent dicts hash equal.
    """
    encoded = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _draw_one(rng: np.random.Generator, spec: tuple[Any, ...]) -> Any:
    kind = spec[0]
    if kind == "float":
        lo, hi = spec[1], spec[2]
        return float(rng.uniform(lo, hi))
    if kind == "int":
        lo, hi = spec[1], spec[2]
        return int(rng.integers(lo, hi + 1))
    if kind == "choice":
        choices = list(spec[1])
        return choices[int(rng.integers(0, len(choices)))]
    if kind == "bool":
        return bool(rng.integers(0, 2))
    raise ValueError(f"unknown schema bound kind: {kind!r}")


def random_param_set_within_bounds(
    rng: np.random.Generator, n: int
) -> list[dict[str, Any]]:
    """Draw ``n`` random parameter dicts from ``SCHEMA_BOUNDS``.

    Each dict is independent (no joint constraints applied — the
    manufacturability filter is the gate that rejects infeasible joint
    combinations).

    Parameters
    ----------
    rng : np.random.Generator
        Caller's RNG. Use ``np.random.default_rng(seed)`` for reproducibility.
    n : int
        How many parameter sets to draw. Must be ≥ 1.

    Returns
    -------
    list[dict[str, Any]]
        Length-``n`` list of parameter dicts. Each dict carries an
        ``is_adversarial: False`` flag so downstream code can mix random +
        adversarial sets in one pipeline.
    """
    if n < 1:
        raise ValueError(f"n must be ≥ 1, got {n}")
    out: list[dict[str, Any]] = []
    for _ in range(n):
        params: dict[str, Any] = {"is_adversarial": False}
        for name, spec in SCHEMA_BOUNDS.items():
            params[name] = _draw_one(rng, spec)
        out.append(params)
    return out


# ── Adversarial parameter sets (spec §0.7a sub-clause a/b/c) ─────────────
#
# Each adversarial set targets one invariant. The comments explain (1) what
# the set would have broken before the §9.7.1 panel-domain invariant + the
# new "≥5 mm from outer-rib click region" Check 7, and (2) which of the four
# downstream checks is expected to catch it.

ADVERSARIAL_PARAM_SETS: list[dict[str, Any]] = [
    # ── (a) Layer 2 louver clustered at tip ─────────────────────────────
    # Pushes louver cuts toward the outer-rib edge. Under the OLD guard-only
    # Check 7 (which only protected guard sticks, not inner blades), these
    # cuts would have invaded the click-feature footprint at the inner
    # blades' tips. Under a no-panel-mask generator they would also have
    # punched through the rib material at the tip taper region.
    {
        "is_adversarial": True,
        "_adversarial_id": "a_louver_clustered_tip",
        "_adversarial_target": "click_footprint",
        "_expected_block": "click_check or manuf",
        "blade_count": 10,
        "panel_thickness_root_m": 0.0030,
        "panel_thickness_mid_m": 0.0030,
        "panel_thickness_tip_m": 0.0030,
        "rib_base_width_m": 0.004,
        "rib_tip_width_m": 0.006,
        "camber_c0": 0.0,
        "camber_c1": 0.0,
        "camber_c2": 0.0,
        "twist_root_rad": 0.0,
        "twist_tip_rad": 0.0,
        "edge_profile": "rounded",
        "fourier_le_a1": 0.0,
        "fourier_le_a2": 0.0,
        "fourier_te_a1": 0.0,
        "fourier_te_a2": 0.0,
        "louver_active": True,
        "louver_spacing_m": 0.005,  # minimum — densest cuts
        "louver_depth_m": 0.0010,  # maximum depth
        "louver_angle_rad": 0.0,
        "louver_cluster_tip": 1.0,  # full cluster-at-tip
        "texture_active": False,
        "texture_amp_m": 0.0003,
        "texture_freq_per_m": 100.0,
        "edge_feature_active": False,
        "edge_feature_depth_m": 0.0005,
        "noise_active": False,
        "noise_threshold": 0.5,
        "tpms_active": False,
        "tpms_cell_size_m": 0.015,
        "tpms_rotation_rad": 0.0,
        "prim_active": False,
        "prim_type": "slot",
        "prim_polarity": "subtract",
        "prim_x_m": 0.100,
        "prim_y_m": 0.0,
        "prim_size_x_m": 0.005,
        "prim_size_y_m": 0.005,
        "prim_rotation_rad": 0.0,
        "print_orientation": "flat",
        "layer_height_m": 0.0002,
        "chamfer_depth_m": 0.0008,
        "detent_size_m": 0.0004,
        "design_clearance_m": 0.00018,
    },
    # ── (b) Layer 2 TPMS at minimum cell size, click-region rotation ────
    # TPMS at minimum cell with rotation aligned to put through-cuts across
    # the click region AND across the ribs. Pre-§9.7.1 this would have
    # carved through the rib material from the Phase 2 SIMP TO output;
    # pre-Check-7-widening it would also have hit the click footprint at the
    # inner blades.
    {
        "is_adversarial": True,
        "_adversarial_id": "b_tpms_min_cell_click_rotation",
        "_adversarial_target": "rib_material + click_footprint",
        "_expected_block": "rib_check or manuf",
        "blade_count": 10,
        "panel_thickness_root_m": 0.0030,
        "panel_thickness_mid_m": 0.0030,
        "panel_thickness_tip_m": 0.0030,
        "rib_base_width_m": 0.004,
        "rib_tip_width_m": 0.006,
        "camber_c0": 0.0,
        "camber_c1": 0.0,
        "camber_c2": 0.0,
        "twist_root_rad": 0.0,
        "twist_tip_rad": 0.0,
        "edge_profile": "rounded",
        "fourier_le_a1": 0.0,
        "fourier_le_a2": 0.0,
        "fourier_te_a1": 0.0,
        "fourier_te_a2": 0.0,
        "louver_active": False,
        "louver_spacing_m": 0.010,
        "louver_depth_m": 0.0005,
        "louver_angle_rad": 0.0,
        "louver_cluster_tip": 0.0,
        "texture_active": False,
        "texture_amp_m": 0.0003,
        "texture_freq_per_m": 100.0,
        "edge_feature_active": False,
        "edge_feature_depth_m": 0.0005,
        "noise_active": False,
        "noise_threshold": 0.5,
        "tpms_active": True,
        "tpms_cell_size_m": 0.006,  # minimum cell
        "tpms_rotation_rad": 0.7854,  # 45° — through-cuts cross both ribs + click
        "prim_active": False,
        "prim_type": "slot",
        "prim_polarity": "subtract",
        "prim_x_m": 0.100,
        "prim_y_m": 0.0,
        "prim_size_x_m": 0.005,
        "prim_size_y_m": 0.005,
        "prim_rotation_rad": 0.0,
        "print_orientation": "flat",
        "layer_height_m": 0.0002,
        "chamfer_depth_m": 0.0008,
        "detent_size_m": 0.0004,
        "design_clearance_m": 0.00018,
    },
    # ── (c) Layer 3 primitive at the bounds-edge ────────────────────────
    # Primitive positioned at the bounds-edge of the new "≥5 mm from
    # outer-rib click region" constraint. Pre-constraint this would have
    # invaded the click-feature footprint at (x ≈ L_blade − rib_tip_taper,
    # y ≈ ±panel_tangential_outer); now the manufacturability filter must
    # catch it OR the click-check must catch it if filter has a hole.
    {
        "is_adversarial": True,
        "_adversarial_id": "c_primitive_bounds_edge",
        "_adversarial_target": "click_footprint (≥5 mm constraint)",
        "_expected_block": "manuf or click_check",
        "blade_count": 10,
        "panel_thickness_root_m": 0.0030,
        "panel_thickness_mid_m": 0.0030,
        "panel_thickness_tip_m": 0.0030,
        "rib_base_width_m": 0.004,
        "rib_tip_width_m": 0.006,
        "camber_c0": 0.0,
        "camber_c1": 0.0,
        "camber_c2": 0.0,
        "twist_root_rad": 0.0,
        "twist_tip_rad": 0.0,
        "edge_profile": "rounded",
        "fourier_le_a1": 0.0,
        "fourier_le_a2": 0.0,
        "fourier_te_a1": 0.0,
        "fourier_te_a2": 0.0,
        "louver_active": False,
        "louver_spacing_m": 0.010,
        "louver_depth_m": 0.0005,
        "louver_angle_rad": 0.0,
        "louver_cluster_tip": 0.0,
        "texture_active": False,
        "texture_amp_m": 0.0003,
        "texture_freq_per_m": 100.0,
        "edge_feature_active": False,
        "edge_feature_depth_m": 0.0005,
        "noise_active": False,
        "noise_threshold": 0.5,
        "tpms_active": False,
        "tpms_cell_size_m": 0.015,
        "tpms_rotation_rad": 0.0,
        "prim_active": True,
        "prim_type": "slot",
        "prim_polarity": "subtract",
        # Park the primitive exactly at L_blade - rib_tip_taper - 0.005
        # (the "5 mm clearance from outer-rib click region" boundary). Its
        # size is the max allowed, so its bbox kisses the click footprint.
        "prim_x_m": L_BLADE_M - RIB_TIP_TAPER_M - 0.005,
        "prim_y_m": PANEL_TANGENTIAL_OUTER_M - 0.005,
        "prim_size_x_m": 0.015,
        "prim_size_y_m": 0.015,
        "prim_rotation_rad": 0.0,
        "print_orientation": "flat",
        "layer_height_m": 0.0002,
        "chamfer_depth_m": 0.0008,
        "detent_size_m": 0.0004,
        "design_clearance_m": 0.00018,
    },
]


@dataclass(frozen=True)
class GeomSanityRecord:
    """One row of the Spike 0.7a results table.

    Attributes
    ----------
    params_hash : str
        12-char SHA-256 hash of the parameter dict; identifies the row in
        downstream artifacts (STL filenames, log entries).
    is_adversarial : bool
        True iff the row came from ``ADVERSARIAL_PARAM_SETS``.
    generated : bool
        Did the generator return a non-None blade object?
    manufacturability_passed : bool
        Did the §N7 11-check filter accept the design?
    click_footprint_intact : bool
        Bit-for-bit (or bbox-shim) check: did the click chamfer footprint at
        ``(x = L_blade ± rib_tip_taper, y = ±panel_tangential_outer)`` survive
        Layer 2/3 carving?
    rib_material_preserved : bool
        §9.7.1 panel-domain invariant: did Layer 2/3 carving stay within the
        panel domain (i.e., no Boolean subtraction reached into the rib bbox)?
    rejection_reasons : tuple[str, ...]
        Free-form strings explaining each False above. Empty tuple iff every
        check passed.
    """

    params_hash: str
    is_adversarial: bool
    generated: bool
    manufacturability_passed: bool
    click_footprint_intact: bool
    rib_material_preserved: bool
    rejection_reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        """Aggregate pass = generated AND manuf AND click AND rib."""
        return (
            self.generated
            and self.manufacturability_passed
            and self.click_footprint_intact
            and self.rib_material_preserved
        )

    @property
    def blocked(self) -> bool:
        """Aggregate block = any of the four checks rejected the design.

        Used in the adversarial-coverage assertion: every adversarial design
        MUST be blocked.
        """
        return not self.passed


GeneratorFn = Callable[[dict[str, Any]], Any]
ManufFn = Callable[[dict[str, Any], Any], tuple[bool, tuple[str, ...]]]
ClickCheckFn = Callable[[dict[str, Any], Any], tuple[bool, tuple[str, ...]]]
RibCheckFn = Callable[[dict[str, Any], Any], tuple[bool, tuple[str, ...]]]


def evaluate_param_set(
    params: dict[str, Any],
    *,
    generator_fn: GeneratorFn,
    manuf_fn: ManufFn,
    click_check_fn: ClickCheckFn,
    rib_check_fn: RibCheckFn,
) -> GeomSanityRecord:
    """Run one parameter set through the four-stage pipeline.

    Stages:
        1. ``generator_fn(params) -> blade | None``
        2. ``manuf_fn(params, blade) -> (passed, reasons)``
        3. ``click_check_fn(params, blade) -> (intact, reasons)``
        4. ``rib_check_fn(params, blade) -> (preserved, reasons)``

    If the generator returns ``None`` (or raises a caught exception), the
    remaining stages short-circuit to ``False`` with an "ungenerated" reason.

    Parameters
    ----------
    params : dict[str, Any]
        One parameter dict. Must contain the ``is_adversarial`` flag (random
        sets default to False; ``ADVERSARIAL_PARAM_SETS`` carry True).
    generator_fn, manuf_fn, click_check_fn, rib_check_fn : callables
        Injectable for testability. Production wiring lives in
        ``scripts/run_spike_0_7a.py``.

    Returns
    -------
    GeomSanityRecord
        Frozen record summarizing the four checks + accumulated rejection
        reasons.
    """
    is_adv = bool(params.get("is_adversarial", False))
    h = hash_params(params)
    reasons: list[str] = []

    # Stage 1 — generation
    blade: Any = None
    generated = False
    try:
        blade = generator_fn(params)
        generated = blade is not None
        if not generated:
            reasons.append("generator returned None")
    except Exception as exc:  # noqa: BLE001 — boundary, capture verbatim
        reasons.append(f"generator raised {type(exc).__name__}: {exc}")
        generated = False

    if not generated:
        # Short-circuit: a blade that didn't generate trivially fails the
        # remaining three checks. Mark them False with the propagated reason.
        return GeomSanityRecord(
            params_hash=h,
            is_adversarial=is_adv,
            generated=False,
            manufacturability_passed=False,
            click_footprint_intact=False,
            rib_material_preserved=False,
            rejection_reasons=tuple(reasons) or ("ungenerated",),
        )

    # Stage 2 — manufacturability filter
    try:
        manuf_ok, manuf_reasons = manuf_fn(params, blade)
    except Exception as exc:  # noqa: BLE001
        manuf_ok, manuf_reasons = False, (f"manuf raised {type(exc).__name__}: {exc}",)
    if not manuf_ok:
        reasons.extend(f"manuf: {r}" for r in manuf_reasons)

    # Stage 3 — click footprint
    try:
        click_ok, click_reasons = click_check_fn(params, blade)
    except Exception as exc:  # noqa: BLE001
        click_ok, click_reasons = False, (f"click raised {type(exc).__name__}: {exc}",)
    if not click_ok:
        reasons.extend(f"click: {r}" for r in click_reasons)

    # Stage 4 — rib material preservation
    try:
        rib_ok, rib_reasons = rib_check_fn(params, blade)
    except Exception as exc:  # noqa: BLE001
        rib_ok, rib_reasons = False, (f"rib raised {type(exc).__name__}: {exc}",)
    if not rib_ok:
        reasons.extend(f"rib: {r}" for r in rib_reasons)

    return GeomSanityRecord(
        params_hash=h,
        is_adversarial=is_adv,
        generated=True,
        manufacturability_passed=manuf_ok,
        click_footprint_intact=click_ok,
        rib_material_preserved=rib_ok,
        rejection_reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class Spike07aResult:
    """Aggregated Spike 0.7a outcome over all records.

    Attributes
    ----------
    records : tuple[GeomSanityRecord, ...]
        One per parameter set (random + adversarial), in input order.
    n_random : int
        Count of records with ``is_adversarial = False``.
    n_adversarial : int
        Count of records with ``is_adversarial = True``.
    n_passing : int
        Count of records with ``record.passed = True`` (random subset only —
        passing adversarial would be a *failure* of the spike).
    adversarial_blocked_count : int
        Count of adversarial records that were blocked (by any of the four
        checks). Pass criterion: ``== n_adversarial``.
    passed : bool
        True iff:
          - every adversarial set is blocked, AND
          - (≥ 70 % of random sets pass) OR (≥ 3 random sets fail with
            non-empty ``rejection_reasons``).
    """

    records: tuple[GeomSanityRecord, ...]
    n_random: int
    n_adversarial: int
    n_passing: int
    adversarial_blocked_count: int
    passed: bool


def analyze_07a(records: Iterable[GeomSanityRecord]) -> Spike07aResult:
    """Aggregate per-design records into a single pass/fail verdict.

    See ``Spike07aResult.passed`` for the gate logic.
    """
    recs = tuple(records)
    if not recs:
        raise ValueError("analyze_07a: no records provided")

    random_recs = tuple(r for r in recs if not r.is_adversarial)
    adv_recs = tuple(r for r in recs if r.is_adversarial)

    n_random = len(random_recs)
    n_adversarial = len(adv_recs)
    n_passing = sum(1 for r in random_recs if r.passed)
    adv_blocked = sum(1 for r in adv_recs if r.blocked)

    # Adversarial coverage: every adversarial set must be blocked.
    adv_ok = (n_adversarial > 0) and (adv_blocked == n_adversarial)

    # Random-set acceptance: ≥70 % pass OR ≥3 fail with documented reasons.
    documented_fails = sum(
        1 for r in random_recs if (not r.passed) and len(r.rejection_reasons) > 0
    )
    pass_fraction = (n_passing / n_random) if n_random else 0.0
    random_ok = (
        pass_fraction >= RANDOM_PASS_FRACTION_GATE
        or documented_fails >= RANDOM_DOCUMENTED_FAIL_GATE
    )

    passed = adv_ok and random_ok

    return Spike07aResult(
        records=recs,
        n_random=n_random,
        n_adversarial=n_adversarial,
        n_passing=n_passing,
        adversarial_blocked_count=adv_blocked,
        passed=passed,
    )
