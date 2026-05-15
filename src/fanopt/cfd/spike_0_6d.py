"""Spike 0.6d — Tier-1 quantitative-sanity counter-checks (H10 supplement).

Three sub-spikes that compensate for the 2026-05-14 Sub-spike 0.6c.2
deferral by providing three independent quantitative checks on the
production Tier-1 cfg before Phase 4 burns ~1300 GPU-hours:

* **0.6d.1 (gating)** — symmetry + dimensional-force sanity. Given an
  SU2 history.csv on whatever geometry was run (default: NACA 0012 mesh
  from Spike 0.6c Cell 6), verify cycle-averaged force is near zero
  (symmetry of periodic pitching) AND dimensional cycle-peak force lies
  within ±1 order of magnitude of the analytic body-in-still-air
  envelope ``m × ω² × r_cm``.

* **0.6d.2 (gating)** — 2D thin-plate added-mass analytic check.
  Compare SU2's inviscid-phase pitching moment against the closed-form
  Sedov/Newman added-mass moment for a 2D plate pitching about its
  quarter-chord.

* **0.6d.3 (advisory)** — SU2 incompressible-mode cross-check. Compare
  compressible-with-MACH=1e-9 dimensional forces against native
  ``INC_NAVIER_STOKES``-mode forces. Failure is logged but does not
  block Phase 4.

References:

* Spec: ``docs/report-final.md §Phase 0 Spike 0.6d`` (2026-05-14 addition).
* Protocol: ``docs/spike_0_6d_protocol.md``.
* Motivation: ``docs/phase_logs/phase_0_signoff.md`` Note 2;
  ``docs/phase_logs/spike_0_6c.md`` Note 1.
* Lock callouts: Round-9 HIGH-12 (= C12; production MACH=1e-9 lock that
  these sub-spikes test against).
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from fanopt.cfd.spike_0_6c import MACH_UNSTEADY_LOCK

__all__ = [
    "MACH_UNSTEADY_LOCK",
    "Tier1SymmetryDimensionalResult",
    "Tier1AddedMassResult",
    "Tier1IncompResult",
    "Spike06dResult",
    "parse_history_csv",
    "cycle_averages_and_peaks",
    "compute_dimensional_envelope",
    "check_symmetry_dimensional",
    "compute_added_mass_moment_closed_form_2d_plate",
    "check_added_mass",
    "check_incompressible_cross",
    "analyze_spike_06d",
]


# ---- shared helpers --------------------------------------------------------


def parse_history_csv(history_csv_text: str) -> list[dict[str, float]]:
    """Parse SU2 history.csv text into a list of typed dict rows.

    SU2 history headers are typically ``"Time_Iter","Inner_Iter","CL","CD",...``
    with quoted strings; this helper accepts both quoted and unquoted variants.
    Non-numeric cells are dropped from the row.
    """
    reader = csv.DictReader(io.StringIO(history_csv_text), skipinitialspace=True)
    rows: list[dict[str, float]] = []
    for raw in reader:
        row: dict[str, float] = {}
        for k, v in raw.items():
            if k is None:
                continue
            key = k.strip().strip('"').lower()
            if v is None:
                continue
            try:
                row[key] = float(str(v).strip().strip('"'))
            except (TypeError, ValueError):
                continue
        if row:
            rows.append(row)
    return rows


def _per_outer_iter(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    """Collapse multi-inner-iter rows: keep the LAST inner row per outer step."""
    by_outer: dict[int, dict[str, float]] = {}
    for r in rows:
        if "time_iter" not in r:
            continue
        by_outer[int(r["time_iter"])] = r
    return [by_outer[k] for k in sorted(by_outer)]


def _column(rows: list[dict[str, float]], candidates: Sequence[str]) -> list[float]:
    """Return the first column found among ``candidates`` (case-insensitive)."""
    for cand in candidates:
        key = cand.lower()
        if rows and key in rows[0]:
            return [r[key] for r in rows if key in r]
    return []


def cycle_averages_and_peaks(
    history_csv_text: str,
    *,
    n_cycles: int,
    discard_first_cycle: bool = True,
    force_column_candidates: Sequence[str] = ("cfx", "cl", "cforce_x", "fx"),
) -> tuple[float, float, int]:
    """Compute cycle-averaged and cycle-peak |force| from an SU2 history.

    Returns ``(cycle_avg, cycle_peak, n_rows_used)`` where ``cycle_avg`` is
    the average over the kept cycles' rows (a value near zero is the
    physical-symmetry expectation) and ``cycle_peak`` is the maximum |force|
    over the same window.

    ``n_cycles`` is the total cycles SU2 ran; if ``discard_first_cycle=True``,
    cycle 0 is dropped (initial-transient suppression — standard practice for
    body-in-still-air dual-time-stepping).
    """
    rows = _per_outer_iter(parse_history_csv(history_csv_text))
    force = _column(rows, force_column_candidates)
    if not force or n_cycles < 1:
        return 0.0, 0.0, 0

    total = len(force)
    per_cycle = total // n_cycles
    if per_cycle == 0:
        # Not enough rows for the requested cycle count; use the whole window.
        kept = force
    else:
        start = per_cycle if discard_first_cycle and n_cycles > 1 else 0
        kept = force[start:]

    if not kept:
        return 0.0, 0.0, 0

    avg = sum(kept) / len(kept)
    peak = max(abs(x) for x in kept)
    return avg, peak, len(kept)


# ---- sub-spike 0.6d.1 ------------------------------------------------------


@dataclass(frozen=True)
class Tier1SymmetryDimensionalResult:
    """Outcome of ``check_symmetry_dimensional`` for one history.csv.

    Pass criterion:
      * symmetry: ``|F_cycle_avg| < 0.05 × F_cycle_peak`` (physical symmetry
        of periodic pitching about a fixed axis)
      * magnitude: ``F_cycle_peak`` within ±1 order of magnitude of the
        analytic envelope ``F_envelope = m × ω² × r_cm`` for the geometry
        the cfg actually ran on
    """

    history_path: str
    n_cycles: int
    force_cycle_avg: float
    force_cycle_peak: float
    force_envelope: float
    envelope_geometry: str  # human-readable: "NACA 0012, m=0.05 kg, r=0.25 m"
    symmetry_ratio: float  # |F_avg| / F_peak (0.0 if peak is 0)
    symmetry_passed: bool
    magnitude_ratio_log10: float  # log10(F_peak / F_envelope)
    magnitude_passed: bool
    passed: bool


def compute_dimensional_envelope(
    *,
    mass_kg: float,
    omega_rad_per_s: float,
    r_cm_m: float,
) -> float:
    """Analytic body-in-still-air force envelope ``F = m × ω² × r``.

    This is the order-of-magnitude scaling for a rigid body pitching in
    still fluid: peak inertial reaction × moment-arm scaling. The added-mass
    term enters at the same order for plate-like bodies in air (ρ_air × c² ≪
    body mass for our solids), so a single envelope captures both for
    sanity-check purposes.
    """
    if mass_kg < 0 or omega_rad_per_s < 0 or r_cm_m < 0:
        raise ValueError("envelope inputs must be non-negative")
    return mass_kg * omega_rad_per_s * omega_rad_per_s * r_cm_m


def check_symmetry_dimensional(
    history_csv_text: str,
    *,
    n_cycles: int,
    mass_kg: float,
    omega_rad_per_s: float,
    r_cm_m: float,
    envelope_geometry: str,
    history_path: str = "<inline>",
    symmetry_threshold: float = 0.05,
    magnitude_log10_tolerance: float = 1.0,
    force_column_candidates: Sequence[str] = ("cfx", "cl", "cforce_x", "fx"),
) -> Tier1SymmetryDimensionalResult:
    """Run the 0.6d.1 symmetry + dimensional-force checks on a history.csv.

    The two checks combine into ``passed = symmetry_passed AND magnitude_passed``.
    """
    avg, peak, _ = cycle_averages_and_peaks(
        history_csv_text,
        n_cycles=n_cycles,
        force_column_candidates=force_column_candidates,
    )
    envelope = compute_dimensional_envelope(
        mass_kg=mass_kg, omega_rad_per_s=omega_rad_per_s, r_cm_m=r_cm_m
    )

    symmetry_ratio = abs(avg) / peak if peak > 0 else math.inf
    symmetry_passed = symmetry_ratio < symmetry_threshold

    magnitude_ratio_log10 = math.log10(peak / envelope) if envelope > 0 and peak > 0 else math.nan
    magnitude_passed = (
        not math.isnan(magnitude_ratio_log10)
        and abs(magnitude_ratio_log10) <= magnitude_log10_tolerance
    )

    return Tier1SymmetryDimensionalResult(
        history_path=history_path,
        n_cycles=n_cycles,
        force_cycle_avg=avg,
        force_cycle_peak=peak,
        force_envelope=envelope,
        envelope_geometry=envelope_geometry,
        symmetry_ratio=symmetry_ratio,
        symmetry_passed=bool(symmetry_passed),
        magnitude_ratio_log10=magnitude_ratio_log10,
        magnitude_passed=bool(magnitude_passed),
        passed=bool(symmetry_passed and magnitude_passed),
    )


# ---- sub-spike 0.6d.2 ------------------------------------------------------


@dataclass(frozen=True)
class Tier1AddedMassResult:
    """Outcome of ``check_added_mass`` for a 2D thin-plate pitching run.

    Pass criterion: SU2 cycle-peak inviscid-phase pitching moment is within
    ``tolerance`` (default ±15%) of the closed-form Sedov/Newman added-mass
    moment.
    """

    history_path: str
    chord_m: float
    pivot_offset_normalized: float  # a = (x_pivot - c/2) / (c/2); -0.5 at quarter-chord
    pitching_omega_rad_per_s: float
    pitching_amplitude_rad: float
    su2_moment_peak: float
    closed_form_moment_peak: float
    relative_error: float  # (su2 - closed) / closed
    tolerance: float
    passed: bool


def compute_added_mass_moment_closed_form_2d_plate(
    *,
    chord_m: float,
    pivot_offset_normalized: float,
    pitching_omega_rad_per_s: float,
    pitching_amplitude_rad: float,
    fluid_density_kg_per_m3: float = 1.225,
) -> float:
    """Sedov/Newman closed-form peak added-mass moment for a 2D pitching plate.

    For a 2D flat plate of chord ``c`` (semi-chord ``b = c/2``) pitching with
    angular displacement ``θ(t) = θ_max × sin(ω t)`` about a pivot located
    at normalized offset ``a`` from mid-chord (``a = (x_pivot − c/2) / b``,
    so quarter-chord is ``a = −0.5``), the per-unit-span added-mass
    moment-of-inertia about the pivot is (Newman, *Marine Hydrodynamics*,
    §4.20):

        I_a = π ρ b^4 (1/8 + a²)

    The added-mass moment is purely inertial:

        M_added(t) = −I_a × θ̈(t)

    Its peak magnitude is ``|M_added,peak| = I_a × ω² × θ_max``. This is the
    value SU2's inviscid-phase moment (sampled at the instant of peak ``θ̈``)
    should match.

    Returns the per-unit-span peak moment in N·m / m (= N).
    """
    if chord_m <= 0 or pitching_omega_rad_per_s <= 0 or pitching_amplitude_rad <= 0:
        raise ValueError("chord, omega, and amplitude must be positive")
    b = chord_m / 2.0
    inertia_added = (
        math.pi * fluid_density_kg_per_m3 * b**4 * (1.0 / 8.0 + pivot_offset_normalized**2)
    )
    return inertia_added * pitching_omega_rad_per_s**2 * pitching_amplitude_rad


def check_added_mass(
    *,
    su2_moment_peak: float,
    chord_m: float,
    pivot_offset_normalized: float,
    pitching_omega_rad_per_s: float,
    pitching_amplitude_rad: float,
    fluid_density_kg_per_m3: float = 1.225,
    tolerance: float = 0.15,
    history_path: str = "<inline>",
) -> Tier1AddedMassResult:
    """Compare SU2 inviscid-phase moment with the Sedov/Newman closed form."""
    closed_form = compute_added_mass_moment_closed_form_2d_plate(
        chord_m=chord_m,
        pivot_offset_normalized=pivot_offset_normalized,
        pitching_omega_rad_per_s=pitching_omega_rad_per_s,
        pitching_amplitude_rad=pitching_amplitude_rad,
        fluid_density_kg_per_m3=fluid_density_kg_per_m3,
    )
    rel_err = math.nan if closed_form == 0 else (su2_moment_peak - closed_form) / closed_form
    passed = (not math.isnan(rel_err)) and abs(rel_err) <= tolerance
    return Tier1AddedMassResult(
        history_path=history_path,
        chord_m=chord_m,
        pivot_offset_normalized=pivot_offset_normalized,
        pitching_omega_rad_per_s=pitching_omega_rad_per_s,
        pitching_amplitude_rad=pitching_amplitude_rad,
        su2_moment_peak=su2_moment_peak,
        closed_form_moment_peak=closed_form,
        relative_error=rel_err,
        tolerance=tolerance,
        passed=bool(passed),
    )


# ---- sub-spike 0.6d.3 (advisory) -------------------------------------------


@dataclass(frozen=True)
class Tier1IncompResult:
    """Outcome of ``check_incompressible_cross`` (advisory, NOT gating).

    Pass criterion (advisory): ``|F_comp - F_incomp| / max(|F_comp|,|F_incomp|)
    <= tolerance`` (default ±20%).
    """

    compressible_force_cycle_avg: float
    incompressible_force_cycle_avg: float
    relative_error: float
    tolerance: float
    passed: bool  # advisory; aggregator ignores this for the marker decision


def check_incompressible_cross(
    *,
    compressible_force_cycle_avg: float,
    incompressible_force_cycle_avg: float,
    tolerance: float = 0.20,
) -> Tier1IncompResult:
    """Compare compressible-mode-with-MACH=1e-9 against native incompressible.

    Advisory only; the aggregator does not use this result to write the
    Phase-4-gate marker. Failure is documented as a Phase-5 investigation
    item (see Phase 5 step 62.5).
    """
    denom = max(abs(compressible_force_cycle_avg), abs(incompressible_force_cycle_avg))
    if denom == 0:
        rel_err = math.nan
    else:
        rel_err = abs(compressible_force_cycle_avg - incompressible_force_cycle_avg) / denom
    passed = (not math.isnan(rel_err)) and rel_err <= tolerance
    return Tier1IncompResult(
        compressible_force_cycle_avg=compressible_force_cycle_avg,
        incompressible_force_cycle_avg=incompressible_force_cycle_avg,
        relative_error=rel_err,
        tolerance=tolerance,
        passed=bool(passed),
    )


# ---- aggregate Spike 0.6d result -------------------------------------------


@dataclass(frozen=True)
class Spike06dResult:
    """Roll-up of Spike 0.6d (V1 scope, post-2026-05-14).

    Gate semantics:
      * ``overall_passed = sub_06d_1.passed AND sub_06d_2.passed`` —
        sub_06d_3 is ADVISORY and does NOT affect the gate decision.
      * ``scripts/launch_phase4.py`` reads ``data/spike_0_6d/PASS``, which
        is written by ``scripts/run_spike_0_6d.py`` iff ``overall_passed``.
    """

    sub_06d_1: Tier1SymmetryDimensionalResult
    sub_06d_2: Tier1AddedMassResult
    sub_06d_3: Tier1IncompResult | None = field(default=None)
    overall_passed: bool = False


def analyze_spike_06d(
    sub_1: Tier1SymmetryDimensionalResult,
    sub_2: Tier1AddedMassResult,
    sub_3: Tier1IncompResult | None = None,
) -> Spike06dResult:
    """Roll up the sub-spike results. ``sub_3`` advisory; does not gate."""
    overall_passed = bool(sub_1.passed and sub_2.passed)
    return Spike06dResult(
        sub_06d_1=sub_1,
        sub_06d_2=sub_2,
        sub_06d_3=sub_3,
        overall_passed=overall_passed,
    )
