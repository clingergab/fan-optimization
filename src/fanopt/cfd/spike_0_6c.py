"""Spike 0.6c.1 — Tier-1 unsteady-config sanity check (H10 lock).

Implements ``docs/plan_R11.md §Phase 0 Spike 0.6c`` for the V1 scope.

**Phase 4 launch gate (V1):** sub-spike 0.6c.1 — render the canonical
Tier-1 cfg (``configs/su2/fan3d_unsteady.cfg.j2``) and verify it
parses, that it carries the Round-9 HIGH-12 lock (``MACH = 1e-9`` with
``FREESTREAM_OPTION = FREESTREAM_VELOCITY`` primary OR
``REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE`` fallback), and
that SU2 can complete one outer time-step on a probe mesh.

**Sub-spike 0.6c.2 deferred to Phase 5** (2026-05-14 decision; full
record in ``docs/phase_logs/spike_0_6c.md`` → "2026-05-14 diagnostic
addendum"). The benchmark cfg copied the production Tier-1 MACH=1e-9
numerics, which produce body-in-still-air added-mass/drag forces with
2× the prescribed pitching frequency and a large positive bias. These
can't be validated against any published wind-tunnel NACA 0012
oscillating-airfoil benchmark in the same frame. Quantitative
cross-solver validation (SU2 vs PyFR) takes over in Phase 5 where PyFR
is already provisioned.

**Why 0.6c.1 alone is sufficient for the Phase 4 launch gate.** Sub-
spike 0.6c.1 confirms the production Tier-1 cfg parses cleanly under
the deployed SU2 build AND that SU2 completes at least one outer
time-step on a probe mesh. That's enough to know the cfg is
SYNTACTICALLY VALID and the solver-cfg combination LAUNCHES. The
remaining numerical-validation work (does the solver produce
*correct* numbers?) is the Phase 5 cross-solver gate, not a Phase 4
launch prerequisite.

References:

* Spec: ``docs/plan_R11.md §Phase 0 Spike 0.6c`` (lines 1839-1844).
* Protocol: ``docs/spike_0_6c_protocol.md``.
* Lock callouts: Round-9 HIGH-12 (= C12, unsteady MACH lock).
* Companion CI gate: ``tests/test_cfd/test_unsteady_freestream_consistency.py``.
* Decision record: ``docs/phase_logs/spike_0_6c.md``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

__all__ = [
    "MACH_UNSTEADY_LOCK",
    "Tier1CfgSanityResult",
    "Spike06cResult",
    "check_tier1_cfg_sanity",
    "analyze_spike_06c",
]


MACH_UNSTEADY_LOCK: float = 1e-9
"""Tier-1 unsteady MACH value (Round-9 HIGH-12 = C12). The steady tiers
(Tier -1 / Tier 0) use ``MACH = 0.0064``; ``CROSS_TIER`` does NOT carry
``MACH`` — the value lives in ``TIER_SPECIFIC[1]`` per §9.4.1."""


# ----- sub-spike 0.6c.1 (Tier-1 cfg sanity check) ---------------------------


@dataclass(frozen=True)
class Tier1CfgSanityResult:
    """Outcome of ``check_tier1_cfg_sanity`` for a single rendered cfg.

    Pass criterion (V1 — gate for Phase 4 launch):
      * ``outer_time_steps_completed >= 1``
      * ``mach_value == MACH_UNSTEADY_LOCK``
      * EITHER ``freestream_option == "FREESTREAM_VELOCITY"``
        OR ``ref_dimensionalization == "FREESTREAM_PRESS_EQ_ONE"``

    The H10 lock allows EITHER syntax (the spec authors anticipated a
    fallback path); both pass under Round-9 HIGH-12.
    """

    cfg_path: str
    parsed_ok: bool
    mach_value: float
    freestream_option: str
    ref_dimensionalization: str | None
    outer_time_steps_completed: int
    error: str | None
    passed: bool


def _parse_cfg_directive(cfg_text: str, key: str) -> str | None:
    """Extract one ``KEY = VALUE`` line's value, stripping inline comments.

    SU2 cfg lines look like ``KEY= VALUE % optional comment`` (with optional
    whitespace and optional ``=`` spacing). Returns the trimmed value string,
    or ``None`` if the key does not appear in ``cfg_text``.
    """
    pattern = rf"^\s*{re.escape(key)}\s*=\s*([^%\n]+?)(?:\s*%.*)?$"
    match = re.search(pattern, cfg_text, re.MULTILINE)
    return match.group(1).strip() if match else None


def _count_completed_outer_steps(su2_log: str) -> int:
    """Count the number of outer (time-step) iterations SU2 completed.

    SU2's stdout under unsteady mode prints ``Time Iter`` lines. We count
    distinct integer ``Time_Iter`` markers as a robust lower bound on the
    completed outer-step count. If no such markers appear, we fall back to
    a permissive ``"Time step:"`` or ``"OUTER_ITER"`` match.
    """
    if not su2_log:
        return 0
    # Primary: explicit "Time_Iter: N" or "Time Iter: N" patterns.
    primary = re.findall(r"Time[_ ]Iter\s*[:=]\s*(\d+)", su2_log)
    if primary:
        return len({int(v) for v in primary})
    # Fallback: count "Time step:" or "Iter:" lines in summary tables.
    fallback = re.findall(r"^\s*\d+\s+Time\s+step\s+", su2_log, re.MULTILINE)
    if fallback:
        return len(fallback)
    # Last resort: any "OUTER_ITER" markers.
    return len(re.findall(r"OUTER_ITER\b", su2_log))


def check_tier1_cfg_sanity(
    cfg_text: str,
    *,
    su2_log: str | None = None,
    cfg_path: str = "<rendered>",
) -> Tier1CfgSanityResult:
    """Validate that the rendered Tier-1 cfg satisfies the Round-9 HIGH-12 lock.

    Parameters
    ----------
    cfg_text : the rendered SU2 ``.cfg`` text (key = value lines).
    su2_log : optional SU2 stdout from running the cfg on a probe mesh.
        When provided, the outer-time-step count is parsed from the log;
        when absent, the sanity check passes the parser-only path with
        ``outer_time_steps_completed = 0`` (the cfg-only fallback that
        the runner uses when ``SU2_CFD`` is not on PATH).

        Note: the sub-spike's pass criterion requires
        ``outer_time_steps_completed >= 1``, so passing ``su2_log=None``
        produces ``passed=False`` — by design, since the cfg-only fallback
        is a partial check until SU2 is available.
    cfg_path : optional path label for the result record (informational).

    Returns
    -------
    ``Tier1CfgSanityResult`` with the four lock fields populated and the
    overall ``passed`` flag.
    """
    error: str | None = None
    mach_value = math.nan
    freestream_option = ""
    ref_dimensionalization: str | None = None
    parsed_ok = False

    try:
        mach_str = _parse_cfg_directive(cfg_text, "MACH_NUMBER")
        if mach_str is None:
            raise ValueError("MACH_NUMBER directive not found in cfg")
        mach_value = float(mach_str)

        freestream_option = _parse_cfg_directive(cfg_text, "FREESTREAM_OPTION") or ""
        ref_dimensionalization = _parse_cfg_directive(cfg_text, "REF_DIMENSIONALIZATION")
        parsed_ok = True
    except (ValueError, TypeError) as e:
        error = str(e)

    outer_steps = _count_completed_outer_steps(su2_log) if su2_log is not None else 0

    # Pass criterion:
    #  - cfg parsed
    #  - MACH matches the Round-9 HIGH-12 lock (1e-9)
    #  - either primary OR fallback freestream-syntax path is present
    #  - SU2 completed at least one outer time step on the probe mesh
    mach_ok = parsed_ok and mach_value == MACH_UNSTEADY_LOCK
    syntax_ok = (
        freestream_option == "FREESTREAM_VELOCITY"
        or ref_dimensionalization == "FREESTREAM_PRESS_EQ_ONE"
    )
    outer_ok = outer_steps >= 1
    passed = parsed_ok and mach_ok and syntax_ok and outer_ok

    return Tier1CfgSanityResult(
        cfg_path=cfg_path,
        parsed_ok=parsed_ok,
        mach_value=mach_value,
        freestream_option=freestream_option,
        ref_dimensionalization=ref_dimensionalization,
        outer_time_steps_completed=outer_steps,
        error=error,
        passed=passed,
    )


# ----- aggregate Spike 0.6c result (V1: gates on 0.6c.1 only) ---------------


@dataclass(frozen=True)
class Spike06cResult:
    """Roll-up of Spike 0.6c (V1 scope: 0.6c.1 only).

    The Phase 4 launch gate (``scripts/launch_phase4.py``) checks for
    the ``data/spike_0_6c/PASS`` marker, which is written by the
    aggregator iff ``overall_passed`` is True. ``overall_passed`` is
    True iff ``sub_06c_1.passed`` is True.

    Sub-spike 0.6c.2 (NACA 0012 benchmark) was deferred to Phase 5 on
    2026-05-14 (see module docstring). The aggregator no longer carries
    a sub_06c_2 field.
    """

    sub_06c_1: Tier1CfgSanityResult
    overall_passed: bool


def analyze_spike_06c(sub_1: Tier1CfgSanityResult) -> Spike06cResult:
    """Roll up the sub-spike-1 result into the gating ``Spike06cResult``.

    V1 scope (post-2026-05-14): ``overall_passed`` is just
    ``sub_1.passed``. Sub-spike 0.6c.2 lives in Phase 5 now.
    """
    return Spike06cResult(
        sub_06c_1=sub_1,
        overall_passed=bool(sub_1.passed),
    )
