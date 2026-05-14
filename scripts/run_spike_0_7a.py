#!/usr/bin/env python
"""Spike 0.7a — generative-geometry sanity check (CLI runner).

Spec reference: ``docs/plan_R11.md §Phase 0 Spike 0.7`` (sub-spike 0.7a).
Protocol: ``docs/spike_0_7a_protocol.md``.

What this script does
---------------------
1. Draws N random parameter sets from ``SCHEMA_BOUNDS`` (default N=10).
2. Concatenates the 3 hand-picked adversarial sets from ``ADVERSARIAL_PARAM_SETS``.
3. Runs each design through the 4-stage pipeline (generator → manuf filter →
   click-footprint check → rib-material check).
4. Writes ``data/spike_0_7a/results.json`` + a per-design subdirectory carrying
   the params dict (and an STL path placeholder if the generator emits one).
5. Prints a pass/fail table.

Exit codes
----------
- 0 — pass: all adversarial sets blocked AND ≥7/10 random sets pass (or ≥3
  failed with documented reasons).
- 1 — fail: pass criterion not met (at least one adversarial slipped through
  OR the random-set pass rate is too low without enough documented failures).
- 2 — error: bad input arguments or pipeline misconfiguration.

Generator wiring (shim)
-----------------------
The production generator (``fanopt.geometry.generator.generate_blade``) is a
Phase 0 / Phase 1 scaffold. Until it lands, this script wires in a small
**bounding-box shim** that:

  * "Generates" a blade as a dict of bbox metadata.
  * Runs a manufacturability pre-check on rule-level invariants (no
    primitive in the click region, TPMS cell size ≥ 3× min feature, …).
  * Asserts the click-footprint bbox is unmodified.
  * Asserts no Layer 2/3 carving reaches into the rib bbox (§9.7.1 panel-
    domain invariant — weaker bbox version of the bit-for-bit check).

The shim is the SAME shim used by the regression tests in
``tests/test_geometry/test_click_feature_preservation.py`` and
``tests/test_geometry/test_panel_mask.py``. When Phase 1 lands the real
generator + manufacturability filter, replace the four functions below with
their production imports; the rest of this script is unchanged.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from fanopt.geometry.spike_0_7a import (
    ADVERSARIAL_PARAM_SETS,
    HUB_RADIUS_M,
    L_BLADE_M,
    PANEL_TANGENTIAL_OUTER_M,
    RANDOM_DOCUMENTED_FAIL_GATE,
    RANDOM_PASS_FRACTION_GATE,
    RIB_TIP_TAPER_M,
    GeomSanityRecord,
    Spike07aResult,
    analyze_07a,
    evaluate_param_set,
    random_param_set_within_bounds,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "spike_0_7a"

# Rib bbox under panel-pivot architecture (CLAUDE.md):
#   x ∈ [HUB_RADIUS, L_blade − RIB_TIP_TAPER] = [0.020, 0.185] m
#   y ∈ [±rib_center − rib_width/2, ±rib_center + rib_width/2]
#
# We pick rib_center = PANEL_TANGENTIAL_OUTER_M + rib_tip_width/2 ≈ 0.0255 m
# at the tip — the rib sits just outside the panel's tangential half-width.
# For the bbox shim we use the locked rib_tip_width = 0.006 m.
RIB_CENTER_M: float = PANEL_TANGENTIAL_OUTER_M + 0.003  # ≈ 0.0255
RIB_HALF_WIDTH_M: float = 0.003  # rib_tip_width / 2

# Click-feature footprint:
#   (x = L_blade ± rib_tip_taper, y = ±panel_tangential_outer ≈ ±0.0225 m)
# Treat the footprint as a small bbox of 1 mm radius around each of the four
# corner points (panel's outer tangential edge at both tips).
CLICK_FOOTPRINT_RADIUS_M: float = 0.001
CLICK_CLEARANCE_M: float = 0.005  # the "≥5 mm from outer-rib click region" lock


# ── Bounding-box generator + check shims ─────────────────────────────────


def shim_generator_fn(params: dict[str, Any]) -> dict[str, Any] | None:
    """Produce a stand-in 'blade' dict from the parameter set.

    The shim doesn't actually mesh anything. It computes bounding boxes for
    every Layer 2/3 feature the parameter set would have CadQuery carve, so
    the downstream check shims can run their bbox-overlap tests. Returns
    ``None`` only if the parameter dict is structurally invalid (missing
    required keys); the manufacturability filter is responsible for
    rejecting *physically* infeasible designs.

    Shim limitations (documented in the protocol):
      * No actual STL is emitted; ``stl_path`` is None.
      * No mesh-quality check.
      * Click-footprint check is bbox, not bit-for-bit (real check needs the
        Phase 2 SIMP TO output as a reference voxelization).
    """
    required = {
        "blade_count",
        "panel_thickness_tip_m",
        "louver_active",
        "tpms_active",
        "prim_active",
    }
    if not required.issubset(params):
        return None

    features: list[dict[str, Any]] = []

    # Layer 2 — louver bboxes (one band per spacing step).
    if params.get("louver_active", False):
        spacing = float(params["louver_spacing_m"])
        depth = float(params["louver_depth_m"])
        cluster = float(params.get("louver_cluster_tip", 0.0))
        # Cluster-at-tip: under full cluster the cuts crowd the outer 20 % of
        # the blade — model that as the cuts' x-range pinned to the tip side.
        x_lo = HUB_RADIUS_M + (L_BLADE_M - HUB_RADIUS_M) * cluster * 0.8
        x_hi = L_BLADE_M
        # Cuts span the panel tangentially.
        y_lo = -PANEL_TANGENTIAL_OUTER_M
        y_hi = +PANEL_TANGENTIAL_OUTER_M
        features.append(
            {
                "kind": "louver",
                "bbox": (x_lo, x_hi, y_lo, y_hi),
                "spacing_m": spacing,
                "depth_m": depth,
                "cluster_tip": cluster,
            }
        )

    # Layer 2 — TPMS through-cuts (the failure mode is when TPMS rotation
    # aligns through-cuts across BOTH the click footprint and the rib).
    if params.get("tpms_active", False):
        cell = float(params["tpms_cell_size_m"])
        rot = float(params["tpms_rotation_rad"])
        # TPMS covers the entire panel domain (and bleeds out by half a cell
        # if not panel-masked).
        x_lo = HUB_RADIUS_M - cell / 2.0
        x_hi = L_BLADE_M + cell / 2.0
        y_lo = -PANEL_TANGENTIAL_OUTER_M - cell / 2.0
        y_hi = +PANEL_TANGENTIAL_OUTER_M + cell / 2.0
        features.append(
            {
                "kind": "tpms",
                "bbox": (x_lo, x_hi, y_lo, y_hi),
                "cell_m": cell,
                "rotation_rad": rot,
            }
        )

    # Layer 3 — primitive.
    if params.get("prim_active", False):
        cx = float(params["prim_x_m"])
        cy = float(params["prim_y_m"])
        sx = float(params["prim_size_x_m"])
        sy = float(params["prim_size_y_m"])
        features.append(
            {
                "kind": "primitive",
                "polarity": params.get("prim_polarity", "subtract"),
                "bbox": (cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2),
                "size": (sx, sy),
            }
        )

    return {
        "kind": "bbox_shim_blade",
        "features": features,
        "stl_path": None,
        "blade_count": int(params["blade_count"]),
    }


def shim_manuf_fn(
    params: dict[str, Any], blade: dict[str, Any]
) -> tuple[bool, tuple[str, ...]]:
    """Bbox-level manufacturability filter shim.

    Implements the subset of §N7 checks that can be answered from the
    bbox shim:

      * TPMS cell size ≥ 3× min feature (=3 × 0.0005 m = 0.0015 m baseline,
        but the schema lower bound is already 0.006 m, so this is mostly a
        sanity check).
      * Layer 3 primitive must be ≥5 mm from the click region (new lock).
      * Louver spacing ≥ 5 mm (schema lower bound).
      * Panel thickness within [2.2, 3.8] mm (C7).
    """
    reasons: list[str] = []

    # Panel thickness bounds (C7).
    for k in ("panel_thickness_root_m", "panel_thickness_mid_m", "panel_thickness_tip_m"):
        v = float(params[k])
        if not (0.0022 <= v <= 0.0038):
            reasons.append(f"{k}={v:.4f} outside [0.0022, 0.0038] m (C7)")

    # TPMS cell size lower-bound.
    if params.get("tpms_active", False):
        cell = float(params["tpms_cell_size_m"])
        if cell < 0.006:
            reasons.append(f"tpms_cell_size_m={cell:.4f} below 0.006 m floor")

    # Louver spacing.
    if params.get("louver_active", False):
        spc = float(params["louver_spacing_m"])
        if spc < 0.005:
            reasons.append(f"louver_spacing_m={spc:.4f} below 0.005 m floor")

    # Layer 3 primitive — must stay ≥5 mm clear of every click footprint
    # corner. The four corners are (L_blade ± rib_tip_taper,
    # ±panel_tangential_outer). At the inner blades, ± rib_tip_taper
    # collapses to a single x at L_blade (the tip); guard sticks use
    # L_blade − rib_tip_taper. We check both.
    if params.get("prim_active", False):
        prim_cx = float(params["prim_x_m"])
        prim_cy = float(params["prim_y_m"])
        prim_sx = float(params["prim_size_x_m"])
        prim_sy = float(params["prim_size_y_m"])
        prim_x_hi = prim_cx + prim_sx / 2
        prim_y_outer = abs(prim_cy) + prim_sy / 2
        # Distance from primitive bbox to the click footprint (worst-case at
        # x = L_blade − rib_tip_taper, |y| = panel_tangential_outer).
        click_x = L_BLADE_M - RIB_TIP_TAPER_M
        dx = max(0.0, prim_x_hi - (click_x - CLICK_CLEARANCE_M))
        dy = max(0.0, prim_y_outer - (PANEL_TANGENTIAL_OUTER_M - CLICK_CLEARANCE_M))
        if dx > 0.0 and dy > 0.0:
            reasons.append(
                f"prim bbox overlaps click-clearance region "
                f"(dx={dx*1000:.2f} mm, dy={dy*1000:.2f} mm into the ≥5 mm zone)"
            )

    _ = blade  # bbox shim doesn't need geometric introspection beyond params
    return (len(reasons) == 0), tuple(reasons)


def shim_click_check_fn(
    params: dict[str, Any], blade: dict[str, Any]
) -> tuple[bool, tuple[str, ...]]:
    """Click-footprint bbox check (shim for the bit-for-bit invariant).

    The four click-footprint corner bboxes are tiny squares of radius
    ``CLICK_FOOTPRINT_RADIUS_M`` around::

        (x, y) ∈ {L_blade, L_blade − rib_tip_taper} × {±panel_tangential_outer}

    Any Layer 2/3 feature whose bbox overlaps one of these MUST be reported
    as a footprint violation.
    """
    reasons: list[str] = []
    click_xs = (L_BLADE_M, L_BLADE_M - RIB_TIP_TAPER_M)
    click_ys = (+PANEL_TANGENTIAL_OUTER_M, -PANEL_TANGENTIAL_OUTER_M)
    click_boxes = [
        (x - CLICK_FOOTPRINT_RADIUS_M, x + CLICK_FOOTPRINT_RADIUS_M,
         y - CLICK_FOOTPRINT_RADIUS_M, y + CLICK_FOOTPRINT_RADIUS_M)
        for x in click_xs
        for y in click_ys
    ]

    for feat in blade.get("features", []):
        # Only subtractive features can violate the footprint. Treat all
        # Layer 2 fields as subtractive (louvers and TPMS cut material).
        polarity = feat.get("polarity", "subtract")
        if polarity != "subtract":
            continue
        fx_lo, fx_hi, fy_lo, fy_hi = feat["bbox"]
        for cx_lo, cx_hi, cy_lo, cy_hi in click_boxes:
            if (
                fx_lo < cx_hi
                and fx_hi > cx_lo
                and fy_lo < cy_hi
                and fy_hi > cy_lo
            ):
                reasons.append(
                    f"{feat['kind']} bbox "
                    f"({fx_lo:.4f},{fy_lo:.4f})-({fx_hi:.4f},{fy_hi:.4f}) "
                    f"overlaps click footprint "
                    f"({cx_lo:.4f},{cy_lo:.4f})-({cx_hi:.4f},{cy_hi:.4f})"
                )
                break  # one reason per feature is enough
    _ = params
    return (len(reasons) == 0), tuple(reasons)


def shim_rib_check_fn(
    params: dict[str, Any], blade: dict[str, Any]
) -> tuple[bool, tuple[str, ...]]:
    """§9.7.1 panel-domain invariant check (bbox shim).

    The rib bboxes are the two strips at ``y ∈ [±rib_center ± rib_half_width]``
    spanning ``x ∈ [HUB_RADIUS, L_blade − RIB_TIP_TAPER]``. Any subtractive
    Layer 2/3 feature whose bbox overlaps a rib bbox violates the panel-
    domain invariant.

    Bit-for-bit comparison would require the Phase 2 SIMP TO output as a
    reference voxelization; the bbox check is strictly weaker but catches
    the failure mode the adversarial sets are designed to expose.
    """
    reasons: list[str] = []
    rib_boxes = [
        (
            HUB_RADIUS_M,
            L_BLADE_M - RIB_TIP_TAPER_M,
            +RIB_CENTER_M - RIB_HALF_WIDTH_M,
            +RIB_CENTER_M + RIB_HALF_WIDTH_M,
        ),
        (
            HUB_RADIUS_M,
            L_BLADE_M - RIB_TIP_TAPER_M,
            -RIB_CENTER_M - RIB_HALF_WIDTH_M,
            -RIB_CENTER_M + RIB_HALF_WIDTH_M,
        ),
    ]

    for feat in blade.get("features", []):
        polarity = feat.get("polarity", "subtract")
        if polarity != "subtract":
            continue
        fx_lo, fx_hi, fy_lo, fy_hi = feat["bbox"]
        for rx_lo, rx_hi, ry_lo, ry_hi in rib_boxes:
            if (
                fx_lo < rx_hi
                and fx_hi > rx_lo
                and fy_lo < ry_hi
                and fy_hi > ry_lo
            ):
                reasons.append(
                    f"{feat['kind']} bbox overlaps rib bbox "
                    f"y∈[{ry_lo:.4f},{ry_hi:.4f}] (panel-domain invariant)"
                )
                break
    _ = params
    return (len(reasons) == 0), tuple(reasons)


# ── CLI plumbing ─────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-random",
        type=int,
        default=10,
        help="How many random parameter sets to draw (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write results.json + per-design subdirs (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-adversarial",
        action="store_true",
        help="Skip the 3 adversarial sets (debug only — fails the spike).",
    )
    return parser.parse_args(argv)


def _record_to_json(rec: GeomSanityRecord) -> dict[str, Any]:
    d = dataclasses.asdict(rec)
    d["rejection_reasons"] = list(rec.rejection_reasons)
    d["passed"] = rec.passed
    return d


def _result_to_json(res: Spike07aResult) -> dict[str, Any]:
    return {
        "n_random": res.n_random,
        "n_adversarial": res.n_adversarial,
        "n_passing": res.n_passing,
        "adversarial_blocked_count": res.adversarial_blocked_count,
        "passed": res.passed,
        "records": [_record_to_json(r) for r in res.records],
    }


def _write_per_design_subdir(
    out_dir: Path, params: dict[str, Any], record: GeomSanityRecord
) -> Path:
    sub = out_dir / record.params_hash
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "params.json").write_text(json.dumps(params, indent=2, default=str) + "\n")
    (sub / "record.json").write_text(json.dumps(_record_to_json(record), indent=2) + "\n")
    # STL placeholder — production wiring will write the real STL here.
    (sub / "stl_path.txt").write_text("(shim: no STL emitted)\n")
    return sub


def _print_table(res: Spike07aResult, out_path: Path) -> None:
    print()
    print(
        f"[spike_0_7a] {'hash':<12} {'kind':<10} {'gen':<3} {'manuf':<5} "
        f"{'click':<5} {'rib':<3} {'overall':<7}  reasons"
    )
    for r in res.records:
        kind = "adv" if r.is_adversarial else "rand"
        reasons = "; ".join(r.rejection_reasons) if r.rejection_reasons else "-"
        if len(reasons) > 80:
            reasons = reasons[:77] + "..."
        overall = "PASS" if r.passed else ("BLOCKED" if r.blocked else "?")
        print(
            f"[spike_0_7a] {r.params_hash:<12} {kind:<10} "
            f"{'Y' if r.generated else 'N':<3} "
            f"{'Y' if r.manufacturability_passed else 'N':<5} "
            f"{'Y' if r.click_footprint_intact else 'N':<5} "
            f"{'Y' if r.rib_material_preserved else 'N':<3} "
            f"{overall:<7}  {reasons}"
        )
    print()
    print(f"[spike_0_7a] n_random        = {res.n_random}")
    print(f"[spike_0_7a] n_adversarial   = {res.n_adversarial}")
    print(f"[spike_0_7a] n_passing       = {res.n_passing} of {res.n_random}")
    print(
        f"[spike_0_7a] adv blocked     = {res.adversarial_blocked_count} of "
        f"{res.n_adversarial}  (must be all)"
    )
    print(
        f"[spike_0_7a] gates           = pass_frac≥{RANDOM_PASS_FRACTION_GATE} "
        f"OR documented_fails≥{RANDOM_DOCUMENTED_FAIL_GATE}"
    )
    print(f"[spike_0_7a] {'PASS' if res.passed else 'FAIL'}")
    print(f"[spike_0_7a] wrote           {out_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_random < 1:
        print(f"[spike_0_7a] --n-random must be ≥ 1, got {args.n_random}", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    try:
        random_sets = random_param_set_within_bounds(rng, args.n_random)
    except ValueError as exc:
        print(f"[spike_0_7a] error drawing random sets: {exc}", file=sys.stderr)
        return 2

    adversarial_sets = [] if args.skip_adversarial else list(ADVERSARIAL_PARAM_SETS)
    all_sets: list[dict[str, Any]] = list(random_sets) + adversarial_sets

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[GeomSanityRecord] = []
    for params in all_sets:
        rec = evaluate_param_set(
            params,
            generator_fn=shim_generator_fn,
            manuf_fn=shim_manuf_fn,
            click_check_fn=shim_click_check_fn,
            rib_check_fn=shim_rib_check_fn,
        )
        records.append(rec)
        _write_per_design_subdir(args.output_dir, params, rec)

    try:
        result = analyze_07a(records)
    except ValueError as exc:
        print(f"[spike_0_7a] error analyzing records: {exc}", file=sys.stderr)
        return 2

    payload = {
        "spec_reference": "docs/plan_R11.md §Phase 0 Spike 0.7 (sub-spike 0.7a)",
        "seed": args.seed,
        "n_random_requested": args.n_random,
        "skip_adversarial": args.skip_adversarial,
        "gates": {
            "random_pass_fraction_gate": RANDOM_PASS_FRACTION_GATE,
            "random_documented_fail_gate": RANDOM_DOCUMENTED_FAIL_GATE,
        },
        "result": _result_to_json(result),
    }
    out_path = args.output_dir / "results.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    _print_table(result, out_path)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
