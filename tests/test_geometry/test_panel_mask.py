"""§Phase 0 Spike 0.7a regression — §9.7.1 panel-domain invariant.

The spike-0.7a sub-clause requires: ``the rib material from the Phase 2 TO
output is bit-for-bit preserved (no Boolean subtraction reaches into the rib
region — enforced by the §9.7.1 panel-domain invariant)``.

The rib region (under panel-pivot architecture, CLAUDE.md) lives at::

    x ∈ [HUB_RADIUS, L_blade − RIB_TIP_TAPER] = [0.020, 0.185] m
    y ∈ [+rib_center − rib_width/2, +rib_center + rib_width/2]
        ∪ [−rib_center − rib_width/2, −rib_center + rib_width/2]

Bit-for-bit comparison requires the Phase 2 SIMP TO output as a reference
voxelization. Until Phase 2 lands, we run the **bounding-box shim** check
from ``scripts/run_spike_0_7a.py``: any subtractive Layer 2/3 feature whose
bbox overlaps a rib bbox MUST be flagged.

The bbox check is strictly weaker than the bit-for-bit check, BUT it catches
the failure mode the spike's adversarial set (b) is designed to expose
(a Layer 2 TPMS at minimum cell size rotated to put through-cuts across the
ribs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_spike_0_7a as runner  # noqa: E402
from fanopt.geometry.spike_0_7a import (  # noqa: E402
    ADVERSARIAL_PARAM_SETS,
    HUB_RADIUS_M,
    L_BLADE_M,
    RIB_TIP_TAPER_M,
    evaluate_param_set,
)

BBOX_SHIM: bool = True  # flip to False when bit-for-bit voxel comparator lands


def _tpms_adversarial() -> dict:
    for p in ADVERSARIAL_PARAM_SETS:
        if p.get("_adversarial_id", "").startswith("b_tpms"):
            return p
    raise AssertionError("adversarial set (b) not found")


def _safe_baseline_params() -> dict:
    p = dict(_tpms_adversarial())
    p["louver_active"] = False
    p["tpms_active"] = False
    p["prim_active"] = False
    return p


# ── Assertion 1: rib bbox spans the locked radial extent ────────────────


def test_rib_bbox_matches_panel_pivot_architecture_lock() -> None:
    """Under panel-pivot architecture the rib radial extent is locked to
    [HUB_RADIUS, L_blade − RIB_TIP_TAPER] = [0.020, 0.185] m. The bbox
    used in the panel-domain invariant check must match that lock."""
    assert pytest.approx(0.020) == HUB_RADIUS_M
    assert pytest.approx(0.200) == L_BLADE_M
    assert pytest.approx(0.015) == RIB_TIP_TAPER_M
    rib_x_hi = L_BLADE_M - RIB_TIP_TAPER_M
    assert rib_x_hi == pytest.approx(0.185)
    # Rib lives outside the panel's tangential half-width (panel-pivot
    # architecture: rib carries no pivot hole; the panel does).
    assert runner.RIB_CENTER_M > runner.PANEL_TANGENTIAL_OUTER_M


# ── Assertion 2: TPMS-min-cell adversarial set is blocked ───────────────


def test_tpms_min_cell_with_rib_crossing_rotation_is_blocked() -> None:
    """Spike 0.7a adversarial (b): TPMS at minimum cell size rotated 45° to
    put through-cuts across BOTH ribs and the click region must be flagged
    by the rib panel-domain invariant check (and/or the click check).

    BBOX_SHIM: this is a bounding-box overlap check. When the bit-for-bit
    voxel comparator lands, the assertion changes to ``rib voxels in the
    output STL == rib voxels in the Phase 2 SIMP TO output``."""
    if not BBOX_SHIM:
        pytest.skip(
            "depends on Phase 2 SIMP TO output; will unblock after Phase 2 "
            "with bit-for-bit voxel comparison"
        )

    params = _tpms_adversarial()
    rec = evaluate_param_set(
        params,
        generator_fn=runner.shim_generator_fn,
        manuf_fn=runner.shim_manuf_fn,
        click_check_fn=runner.shim_click_check_fn,
        rib_check_fn=runner.shim_rib_check_fn,
    )
    assert rec.blocked, (
        "Adversarial TPMS-min-cell set slipped through every check — "
        f"reasons={rec.rejection_reasons}"
    )
    # The rib check is the primary defender for this adversarial set; we
    # accept blocking by the click check too (TPMS rotated 45° at min cell
    # size hits both regions).
    caught_at_rib = not rec.rib_material_preserved
    caught_at_click = not rec.click_footprint_intact
    assert caught_at_rib or caught_at_click, (
        "Set was blocked but neither rib nor click flagged it — " f"reasons={rec.rejection_reasons}"
    )


# ── Assertion 3: a safe parameter set preserves rib material trivially ──


def test_safe_params_preserve_rib_material() -> None:
    """Baseline sanity: with all Layer 2/3 off, the rib check is trivially
    satisfied."""
    params = _safe_baseline_params()
    rec = evaluate_param_set(
        params,
        generator_fn=runner.shim_generator_fn,
        manuf_fn=runner.shim_manuf_fn,
        click_check_fn=runner.shim_click_check_fn,
        rib_check_fn=runner.shim_rib_check_fn,
    )
    assert (
        rec.rib_material_preserved
    ), f"Safe params somehow violated rib invariant — reasons={rec.rejection_reasons}"


# ── Assertion 4: additive Layer 3 primitive does NOT count as carving ────


def test_additive_primitive_does_not_violate_panel_domain() -> None:
    """The panel-domain invariant prohibits *subtractive* carving in the rib
    region. An *additive* primitive sitting on top of the rib is not a
    carving event and must not trigger the rib check."""
    params = _safe_baseline_params()
    params["prim_active"] = True
    params["prim_polarity"] = "add"  # additive
    params["prim_x_m"] = 0.100  # mid-blade
    params["prim_y_m"] = runner.RIB_CENTER_M  # on top of a rib
    params["prim_size_x_m"] = 0.010
    params["prim_size_y_m"] = 0.005

    rec = evaluate_param_set(
        params,
        generator_fn=runner.shim_generator_fn,
        manuf_fn=runner.shim_manuf_fn,
        click_check_fn=runner.shim_click_check_fn,
        rib_check_fn=runner.shim_rib_check_fn,
    )
    assert rec.rib_material_preserved, (
        "Additive primitive on top of a rib must NOT trip the rib check — "
        f"reasons={rec.rejection_reasons}"
    )


if __name__ == "__main__":
    test_rib_bbox_matches_panel_pivot_architecture_lock()
    test_tpms_min_cell_with_rib_crossing_rotation_is_blocked()
    test_safe_params_preserve_rib_material()
    test_additive_primitive_does_not_violate_panel_domain()
    print("OK: §9.7.1 panel-domain invariant holds under adversarial Layer 2/3.")
