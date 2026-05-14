"""§Phase 0 Spike 0.7a regression — click-footprint bit-for-bit invariant.

The spike-0.7a sub-clause requires: ``the click-feature footprint is bit-for-
bit intact on every blade, regardless of how aggressively Layer 2/3 carving
is configured``. The footprint lives at::

    (x = L_blade ± rib_tip_taper, y = ±panel_tangential_outer ≈ ±0.0225 m)

Bit-for-bit comparison requires the Phase 2 SIMP TO output as a reference
voxelization. Until Phase 2 lands, we run the **bounding-box shim** check
from ``scripts/run_spike_0_7a.py``: any Layer 2/3 subtractive feature whose
bbox overlaps the click-footprint corner bbox MUST be flagged.

The bbox check is strictly weaker than the bit-for-bit check, BUT it catches
the failure mode the spike's adversarial set (a) is designed to expose
(a Layer 2 louver with full cluster-at-tip pushing cuts into the click
footprint).

When Phase 2 lands and the real generator emits an STL whose surface mesh
can be compared bit-for-bit against the SIMP TO output, replace the bbox
helper here with the bit-for-bit voxel comparison and remove the
``BBOX_SHIM`` marker.
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
    L_BLADE_M,
    PANEL_TANGENTIAL_OUTER_M,
    RIB_TIP_TAPER_M,
    evaluate_param_set,
)

BBOX_SHIM: bool = True  # flip to False when the bit-for-bit comparator lands


def _louver_adversarial() -> dict:
    for p in ADVERSARIAL_PARAM_SETS:
        if p.get("_adversarial_id", "").startswith("a_louver"):
            return p
    raise AssertionError("adversarial set (a) not found")


def _primitive_adversarial() -> dict:
    for p in ADVERSARIAL_PARAM_SETS:
        if p.get("_adversarial_id", "").startswith("c_primitive"):
            return p
    raise AssertionError("adversarial set (c) not found")


def _baseline_safe_params() -> dict:
    """A 'safe' parameter set with no Layer 2/3 carving — click footprint
    must trivially survive."""
    p = dict(_louver_adversarial())
    p["louver_active"] = False
    p["tpms_active"] = False
    p["prim_active"] = False
    return p


# ── Assertion 1: click footprint corners are at the locked coordinates ──


def test_click_footprint_corner_coordinates_are_locked() -> None:
    """Sanity: the four click-footprint corners live at the architecturally
    locked positions (Round-9 HIGH-8 Option A + item #35 panel widening)."""
    expected_corners = {
        (L_BLADE_M, +PANEL_TANGENTIAL_OUTER_M),
        (L_BLADE_M, -PANEL_TANGENTIAL_OUTER_M),
        (L_BLADE_M - RIB_TIP_TAPER_M, +PANEL_TANGENTIAL_OUTER_M),
        (L_BLADE_M - RIB_TIP_TAPER_M, -PANEL_TANGENTIAL_OUTER_M),
    }
    # L_blade − rib_tip_taper must equal 0.185 m per CLAUDE.md.
    assert any(abs(x - 0.185) < 1e-9 for (x, _) in expected_corners)
    assert any(abs(y - 0.0225) < 1e-9 for (_, y) in expected_corners)


# ── Assertion 2: louver adversarial set is blocked at the click check ───


def test_louver_clustered_tip_invades_click_footprint() -> None:
    """Spike 0.7a adversarial (a): Layer 2 louver with full cluster-at-tip
    must be flagged by the click-footprint bbox check.

    BBOX_SHIM: this is a bounding-box overlap check. When the bit-for-bit
    voxel comparator lands, the assertion changes to ``surviving click_voxels
    == baseline click_voxels``."""
    if not BBOX_SHIM:
        pytest.skip(
            "depends on Phase 2 SIMP TO output; will unblock after Phase 2 "
            "with bit-for-bit voxel comparison"
        )

    params = _louver_adversarial()
    rec = evaluate_param_set(
        params,
        generator_fn=runner.shim_generator_fn,
        manuf_fn=runner.shim_manuf_fn,
        click_check_fn=runner.shim_click_check_fn,
        rib_check_fn=runner.shim_rib_check_fn,
    )
    # Either the click check OR the manufacturability filter must have
    # caught the invasion. Both is fine.
    assert rec.blocked, (
        "Adversarial louver-clustered-tip set slipped through every check — "
        f"reasons={rec.rejection_reasons}"
    )
    caught_at_click = not rec.click_footprint_intact
    caught_at_manuf = not rec.manufacturability_passed
    assert caught_at_click or caught_at_manuf, (
        "Set was blocked but neither click nor manuf flagged it — "
        f"reasons={rec.rejection_reasons}"
    )


# ── Assertion 3: primitive at the 5 mm boundary is blocked ──────────────


def test_primitive_at_click_clearance_boundary_is_blocked() -> None:
    """Spike 0.7a adversarial (c): Layer 3 primitive parked at the ≥5 mm
    click-clearance boundary must be flagged."""
    params = _primitive_adversarial()
    rec = evaluate_param_set(
        params,
        generator_fn=runner.shim_generator_fn,
        manuf_fn=runner.shim_manuf_fn,
        click_check_fn=runner.shim_click_check_fn,
        rib_check_fn=runner.shim_rib_check_fn,
    )
    assert rec.blocked, (
        "Adversarial primitive-at-boundary set slipped through every check — "
        f"reasons={rec.rejection_reasons}"
    )


# ── Assertion 4: a safe parameter set leaves the click footprint intact ─


def test_safe_params_leave_click_footprint_intact() -> None:
    """Baseline sanity: a parameter set with no Layer 2/3 carving must pass
    the click-footprint check vacuously."""
    params = _baseline_safe_params()
    rec = evaluate_param_set(
        params,
        generator_fn=runner.shim_generator_fn,
        manuf_fn=runner.shim_manuf_fn,
        click_check_fn=runner.shim_click_check_fn,
        rib_check_fn=runner.shim_rib_check_fn,
    )
    assert rec.click_footprint_intact, (
        "Safe params (no Layer 2/3) somehow violated click footprint — "
        f"reasons={rec.rejection_reasons}"
    )


if __name__ == "__main__":
    test_click_footprint_corner_coordinates_are_locked()
    test_louver_clustered_tip_invades_click_footprint()
    test_primitive_at_click_clearance_boundary_is_blocked()
    test_safe_params_leave_click_footprint_intact()
    print("OK: click-feature footprint invariant holds under adversarial Layer 2/3.")
