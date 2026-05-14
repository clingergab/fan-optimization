"""HIGH-12 Round-9 lock: §9.4.1 unsteady cfg freestream consistency.

Four assertions per locked wording:
  1. FREESTREAM_OPTION = FREESTREAM_VELOCITY  (or fallback
     REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE)
  2. MACH_NUMBER ≤ 1e-6
  3. FREESTREAM_VELOCITY magnitude < 0.01 · V_tip
  4. FREESTREAM_DIRECTION NOT set (or consistent with FREESTREAM_VELOCITY sign)

These four together guarantee that the Tier-1 unsteady simulation runs as
"moving body in still air" — the locked C2 + HIGH-12 physics — rather than as
"moving body in V_tip-magnitude tailwind", which is what SU2's default
FREESTREAM_OPTION = TEMPERATURE_FS would produce silently.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from fanopt.cfd.configs import render_unsteady_cfg

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_spec_path() -> Path:
    """Spec lives at either docs/report-final.md or repo-root report-final.md."""
    for candidate in (REPO_ROOT / "docs" / "report-final.md", REPO_ROOT / "report-final.md"):
        if candidate.exists():
            return candidate
    return REPO_ROOT / "docs" / "report-final.md"


SPEC_PATH = _resolve_spec_path()

V_TIP_MPS = 2.20
MACH_MAX = 1e-6
VEL_FRACTION_OF_V_TIP_MAX = 0.01  # FREESTREAM_VELOCITY magnitude must be < 1% of V_tip


def _render_unsteady_cfg() -> str:
    """Render the Tier-1 unsteady cfg via the production template.

    The renderer is now wired up (see fanopt.cfd.configs.render_unsteady_cfg)
    and required by this gate. The probe-mesh placeholder is fine for the
    parse-only invariants this test checks.
    """
    return render_unsteady_cfg(
        mesh_filename="probe.su2",
        marker_fan="FAN",
        marker_farfield="FARFIELD",
    )


def _parse_directive(cfg: str, key: str) -> str | None:
    """Extract the value of a single SU2 cfg directive, stripping trailing comments."""
    pattern = rf"^\s*{re.escape(key)}\s*=\s*([^%\n]+?)(?:\s*%.*)?$"
    m = re.search(pattern, cfg, re.MULTILINE)
    return m.group(1).strip() if m else None


def test_unsteady_freestream_option_set_or_fallback() -> None:
    """Assertion 1: FREESTREAM_OPTION = FREESTREAM_VELOCITY (primary) OR
    REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE (fallback).

    Either path forces SU2 to NOT compute freestream from MACH × c_ref. Without
    one of these, SU2's default TEMPERATURE_FS computes freestream as
    MACH × c_ref = 2.20 m/s = V_tip — producing zero body-vs-ambient relative
    velocity and physically nonsensical CFD predictions.
    """
    cfg = _render_unsteady_cfg()
    primary = _parse_directive(cfg, "FREESTREAM_OPTION") == "FREESTREAM_VELOCITY"
    fallback = (
        _parse_directive(cfg, "REF_DIMENSIONALIZATION") == "FREESTREAM_PRESS_EQ_ONE"
    )
    assert primary or fallback, (
        "HIGH-12 Round-9 violation: unsteady cfg must use FREESTREAM_OPTION = "
        "FREESTREAM_VELOCITY (primary) OR REF_DIMENSIONALIZATION = "
        "FREESTREAM_PRESS_EQ_ONE (fallback per Spike 0.6c.1). Without one, SU2's "
        "default TEMPERATURE_FS computes freestream from MACH × c_ref — Tier 1 "
        "would run with zero body-vs-ambient relative velocity."
    )


def test_unsteady_mach_below_1e_6() -> None:
    """Assertion 2: MACH_NUMBER ≤ 1e-6.

    Tier 1 is "moving body in still air" per C2 + HIGH-12. The body's relative
    velocity comes from GRID_MOVEMENT = RIGID_MOTION, not from MACH-based
    freestream. A non-zero MACH would inject a tailwind on top of the body's
    pitching motion.
    """
    cfg = _render_unsteady_cfg()
    mach_str = _parse_directive(cfg, "MACH_NUMBER")
    assert mach_str is not None, "HIGH-12: MACH_NUMBER directive not found in unsteady cfg"
    mach = float(mach_str)
    assert mach <= MACH_MAX, (
        f"HIGH-12 Round-9 violation: MACH_NUMBER = {mach} exceeds {MACH_MAX} "
        f"(unsteady cfg must use near-zero Mach; body motion provides relative "
        f"velocity via GRID_MOVEMENT = RIGID_MOTION, not via MACH-based tailwind)."
    )


def test_unsteady_freestream_velocity_below_one_percent_v_tip() -> None:
    """Assertion 3: |FREESTREAM_VELOCITY| < 0.01 · V_tip = 0.022 m/s.

    The 1 mm/s nominal value (numerically nonzero for Riemann far-field
    stability) sits at 2000× below V_tip. Any value approaching V_tip would
    break the body-vs-ambient frame separation.
    """
    cfg = _render_unsteady_cfg()
    vel_str = _parse_directive(cfg, "FREESTREAM_VELOCITY")
    if vel_str is None:
        # Fallback path uses REF_DIMENSIONALIZATION; velocity check vacuously passes.
        return
    components = [float(x) for x in vel_str.split()]
    magnitude = sum(c**2 for c in components) ** 0.5
    threshold = VEL_FRACTION_OF_V_TIP_MAX * V_TIP_MPS
    assert magnitude < threshold, (
        f"HIGH-12 Round-9 violation: FREESTREAM_VELOCITY magnitude {magnitude:.6f} m/s "
        f"exceeds 0.01 · V_tip = {threshold:.6f} m/s — body-vs-ambient frame separation broken."
    )


def test_unsteady_freestream_direction_not_set_or_consistent() -> None:
    """Assertion 4: FREESTREAM_DIRECTION is either NOT set (preferred) OR
    consistent with FREESTREAM_VELOCITY direction.

    Under FREESTREAM_OPTION = FREESTREAM_VELOCITY override, direction comes
    from the velocity vector. Setting both causes SU2-version-dependent silent
    axis coupling. If both are set, their signs MUST match component-wise.
    """
    cfg = _render_unsteady_cfg()
    direction_str = _parse_directive(cfg, "FREESTREAM_DIRECTION")
    if direction_str is None:
        # Preferred: not set under FREESTREAM_VELOCITY override.
        return
    vel_str = _parse_directive(cfg, "FREESTREAM_VELOCITY")
    if vel_str is None:
        # Fallback path; no velocity vector to compare against.
        return
    dir_components = [float(x) for x in direction_str.split()]
    vel_components = [float(x) for x in vel_str.split()]
    assert len(dir_components) == len(vel_components) == 3, (
        "HIGH-12: FREESTREAM_DIRECTION and FREESTREAM_VELOCITY must each be 3-vectors"
    )
    for axis, (d, v) in enumerate(zip(dir_components, vel_components)):
        if abs(v) > 1e-9:
            assert (d > 0) == (v > 0), (
                f"HIGH-12 Round-9 violation: FREESTREAM_DIRECTION sign on axis {axis} "
                f"({d}) contradicts FREESTREAM_VELOCITY sign ({v}) — ambiguous "
                f"freestream direction in unsteady cfg."
            )


if __name__ == "__main__":
    test_unsteady_freestream_option_set_or_fallback()
    test_unsteady_mach_below_1e_6()
    test_unsteady_freestream_velocity_below_one_percent_v_tip()
    test_unsteady_freestream_direction_not_set_or_consistent()
    print("OK: §9.4.1 unsteady cfg freestream consistency.")
