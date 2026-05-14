"""Rotational inertia from a torsional pendulum.

Implements Spike 0.2 (`docs/spike_0_2_protocol.md`,
`docs/plan_R11.md §Phase 0 Spike 0.2`).

All inertias are about the **+y wrist axis through the handle-grip point**
(§0 row 27, §6.4). This module does not know about the pivot pin axis at all;
it just plugs into the torsion-pendulum formula `I = κ · (T / 2π)²`.

Pass criteria (Spike 0.2):
- repeatability `std/mean × 100 < 3%` across N trials of the same fan
- cross-check `|I_meas − I_gen| / I_gen × 100 < 10%` vs the generator's
  `I_wrist_kgm2`

References:
- Spec: `docs/plan_R11.md §Phase 0 Spike 0.2`
- Generator: `src/fanopt/geometry/*` — emits `I_wrist_kgm2` via
  `i_wrist_assembly` (§6.4)
- Protocol: `docs/spike_0_2_protocol.md`
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

__all__ = [
    "REPEATABILITY_GATE_PCT",
    "CROSS_CHECK_GATE_PCT",
    "i_wrist_from_period",
    "kappa_from_reference",
    "rod_transverse_inertia",
    "InertiaResult",
    "analyze_trials",
]

# Pass-criterion gates from §Phase 0 Spike 0.2.
REPEATABILITY_GATE_PCT: float = 3.0
CROSS_CHECK_GATE_PCT: float = 10.0


def i_wrist_from_period(kappa_Nm_per_rad: float, T_osc_s: float) -> float:
    """I = κ · (T / 2π)².

    Single-shot inertia from one period measurement.

    Parameters
    ----------
    kappa_Nm_per_rad : torsion constant, N·m / rad (≡ kg·m²/s²)
    T_osc_s : measured period of the pendulum, seconds (ONE period — divide
        out the 10-period total at recording time, not here)
    """
    if kappa_Nm_per_rad <= 0:
        raise ValueError(f"kappa must be > 0, got {kappa_Nm_per_rad}")
    if T_osc_s <= 0:
        raise ValueError(f"T_osc must be > 0, got {T_osc_s}")
    return kappa_Nm_per_rad * (T_osc_s / (2.0 * math.pi)) ** 2


def kappa_from_reference(I_ref_kgm2: float, T_ref_s: float) -> float:
    """κ = I_ref · (2π / T_ref)².

    Inverse of `i_wrist_from_period` — use during calibration when I_ref is
    known analytically and T_ref is measured.
    """
    if I_ref_kgm2 <= 0:
        raise ValueError(f"I_ref must be > 0, got {I_ref_kgm2}")
    if T_ref_s <= 0:
        raise ValueError(f"T_ref must be > 0, got {T_ref_s}")
    return I_ref_kgm2 * (2.0 * math.pi / T_ref_s) ** 2


def rod_transverse_inertia(m_kg: float, L_m: float) -> float:
    """Analytic I for a uniform rod about its transverse-midpoint axis.

    I = (1/12) · m · L². Used in calibration: lay a known rod across the
    pendulum, measure its period, recover κ.
    """
    if m_kg <= 0 or L_m <= 0:
        raise ValueError(f"m, L must be > 0; got m={m_kg}, L={L_m}")
    return m_kg * L_m * L_m / 12.0


@dataclass(frozen=True)
class InertiaResult:
    """Output of `analyze_trials`. Serializable via dataclasses.asdict."""

    I_wrist_kgm2: float
    """Mean of per-trial inertia (the published number)."""

    I_wrist_std_kgm2: float
    """Sample standard deviation (ddof=1) across trials."""

    n_trials: int

    repeatability_pct: float
    """std / mean × 100. Pass: < REPEATABILITY_GATE_PCT (3%)."""

    repeatability_passed: bool

    cross_check_pct: float | None
    """|I_meas − I_gen| / I_gen × 100. None if no generator value provided.
    Pass: < CROSS_CHECK_GATE_PCT (10%)."""

    cross_check_passed: bool | None

    passed: bool
    """Repeatability passes AND (cross-check passes or was not requested)."""

    per_trial_I_kgm2: tuple[float, ...]
    """Individual trial values, in input order, for the run log."""


def analyze_trials(
    kappa_Nm_per_rad: float,
    T_osc_trials_s: Iterable[float],
    *,
    generator_I_wrist_kgm2: float | None = None,
    mount_I_wrist_kgm2: float = 0.0,
) -> InertiaResult:
    """Compute I_wrist + pass-criteria from N trial periods.

    Parameters
    ----------
    kappa_Nm_per_rad : torsion constant from calibration (Step 1).
    T_osc_trials_s : per-trial periods, seconds (Step 2 — 5 trials).
    generator_I_wrist_kgm2 : optional `I_wrist_kgm2` from the §9.7 generator
        for the same physical fan; cross-check is skipped if None.
    mount_I_wrist_kgm2 : optional mount-block contribution to subtract.
        Pendulum measures `I_total = I_fan + I_mount`; supply the empty-rig
        inertia to recover just the fan. Defaults to 0 (assume the mount is
        negligible or already incorporated into κ).

    Returns
    -------
    InertiaResult with measured I_wrist, repeatability %, cross-check %, and
    overall pass/fail. Both gates use the §Phase 0 Spike 0.2 thresholds:
    repeatability < 3%, cross-check < 10%.

    Notes
    -----
    Repeatability uses sample std (ddof=1) because trials are a sample of the
    physical measurement noise, not the full population. With N=5 trials,
    ddof=1 is what the spec's "< 3% across 5 measurements" intends — anything
    else under-estimates the spread.
    """
    trials = tuple(T_osc_trials_s)
    if len(trials) < 2:
        raise ValueError(
            f"need ≥ 2 trials to estimate repeatability; got {len(trials)}"
        )
    if mount_I_wrist_kgm2 < 0:
        raise ValueError(f"mount_I_wrist must be ≥ 0, got {mount_I_wrist_kgm2}")

    per_trial = np.asarray(
        [i_wrist_from_period(kappa_Nm_per_rad, t) - mount_I_wrist_kgm2 for t in trials],
        dtype=float,
    )
    mean_I = float(per_trial.mean())
    std_I = float(per_trial.std(ddof=1))

    if mean_I <= 0:
        raise ValueError(
            f"mean I_wrist ≤ 0 after mount subtraction ({mean_I:.3e}); "
            "mount_I_wrist may be too large for these trials."
        )

    repeatability_pct = 100.0 * std_I / mean_I
    repeatability_passed = repeatability_pct < REPEATABILITY_GATE_PCT

    cross_check_pct: float | None
    cross_check_passed: bool | None
    if generator_I_wrist_kgm2 is None:
        cross_check_pct = None
        cross_check_passed = None
    elif generator_I_wrist_kgm2 <= 0:
        raise ValueError(
            f"generator_I_wrist must be > 0 to cross-check, got {generator_I_wrist_kgm2}"
        )
    else:
        cross_check_pct = 100.0 * abs(mean_I - generator_I_wrist_kgm2) / generator_I_wrist_kgm2
        cross_check_passed = cross_check_pct < CROSS_CHECK_GATE_PCT

    overall = repeatability_passed and (cross_check_passed is not False)

    return InertiaResult(
        I_wrist_kgm2=mean_I,
        I_wrist_std_kgm2=std_I,
        n_trials=len(trials),
        repeatability_pct=repeatability_pct,
        repeatability_passed=repeatability_passed,
        cross_check_pct=cross_check_pct,
        cross_check_passed=cross_check_passed,
        passed=overall,
        per_trial_I_kgm2=tuple(float(x) for x in per_trial),
    )
