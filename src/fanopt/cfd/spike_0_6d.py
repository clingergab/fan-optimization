"""Spike 0.6d — Tier-1 added-mass frequency-consistency gate (H10 supplement).

Compensates for the 2026-05-14 Sub-spike 0.6c.2 deferral with an
independent quantitative check on the production Tier-1 numerics before
Phase 4 burns ~1300 GPU-hours.

**Gate redesign (2026-05-15).** The live Colab run exposed that the
original 0.6d.1 magnitude check conflated SU2's q_ref=1 nondimensional
output with a dimensional Newton envelope, and its symmetry criterion
is ill-posed for the fan's net-work regime. The gate was redesigned to
rest solely on a normalization-invariant, parameter-free falsification
test (0.6d.2 below); 0.6d.1 and 0.6d.3 are now advisory.

* **0.6d.2 (GATING)** — 2D thin-plate added-mass frequency-consistency.
  Run the plate at two pitching frequencies (ω₁, ω₂) and Fourier-project
  each moment-coefficient trace onto the added-mass (sin φ) basis. The
  recovered ``I_a = a_sin/(ω²·θ_max)`` is a pure geometric/fluid constant
  — it MUST be frequency-independent. Disagreement falsifies the Tier-1
  numerics independent of the FREESTREAM_PRESS_EQ_ONE nondimensionalisation
  (q_ref is a fixed constant, identical both runs, cancels in the ratio).
  A Sedov/Newman closed-form magnitude comparison is computed too but is
  ADVISORY (its absolute interpretation needs SU2's exact reference-state
  handling, which is Phase 5 step 62.5's job).

* **0.6d.1 (advisory)** — symmetry + dimensional-force sanity. Recorded
  for Phase 5 but does NOT gate (demoted 2026-05-15; see above).

* **0.6d.3 (advisory)** — SU2 incompressible-mode cross-check. Recorded
  for Phase 5 but does NOT gate.

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
    "AddedMassProjection",
    "Tier1AddedMassResult",
    "Tier1IncompResult",
    "Spike06dResult",
    "parse_history_csv",
    "cycle_averages_and_peaks",
    "compute_dimensional_envelope",
    "check_symmetry_dimensional",
    "compute_added_mass_moment_closed_form_2d_plate",
    "recover_added_mass_projection",
    "check_added_mass_freq_consistency",
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
class AddedMassProjection:
    """Fourier projection of one SU2 2D-plate run's moment-coefficient trace.

    The prescribed motion is ``θ(t) = θ_max·sin(ωt)``, so ``θ̈ ∝ -sin(ωt)``
    and the pure added-mass moment ``M_am = -I_a·θ̈ ∝ +sin(ωt)`` — i.e. the
    added-mass component is the **sin(phase) projection** of the moment
    coefficient; the velocity-in-phase (damping/drag fundamental) component
    is the **cos(phase) projection**.

    ``recovered_ia_nondim`` = ``a_sin / (ω²·θ_max)`` is the added-mass
    "inertia" in whatever fixed nondimensionalisation SU2 emits (under
    ``REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE`` the reference
    dynamic pressure is a frequency-independent constant, so this quantity
    is comparable run-to-run without recovering dimensional N·m).
    """

    history_path: str
    omega_rad_per_s: float
    pitching_amplitude_rad: float
    n_cycles: int
    a_sin: float  # added-mass-phase Fourier coefficient (∝ sin(ωt))
    a_cos: float  # damping/drag-phase Fourier coefficient (∝ cos(ωt))
    recovered_ia_nondim: float  # a_sin / (ω²·θ_max)
    drag_to_added_mass_ratio: float  # |a_cos| / |a_sin| (regime diagnostic)


@dataclass(frozen=True)
class Tier1AddedMassResult:
    """Outcome of the 2D thin-plate added-mass verification (sub-spike 0.6d.2).

    **Gating criterion (G1 — frequency-consistency, normalization-invariant):**
    the recovered added-mass coefficient must agree between two pitching
    frequencies. ``I_a = πρb⁴(1/8+a²)`` is a pure geometric/fluid constant
    with NO frequency dependence; if SU2's unsteady solver (MACH=1e-9 +
    low-Mach preconditioning) is producing physically-faithful added-mass
    forces, ``a_sin/(ω²·θ_max)`` MUST be the same at ω₁ and ω₂. A mismatch
    falsifies the numerics — independent of the FREESTREAM_PRESS_EQ_ONE
    nondimensionalisation (q_ref is a fixed constant, identical both runs,
    so it cancels in the ratio). This is the Phase-4 de-risk.

    **Advisory diagnostic (closed-form magnitude):** the recovered
    coefficient is also compared to the Sedov/Newman analytic ``I_a``
    nondimensionalised under the *assumed* FREESTREAM_PRESS_EQ_ONE
    convention (q_ref=1, A_ref=L_ref=chord ⇒ divide by chord²). This is
    reported and flagged if off by more than ``closed_form_factor_tol``,
    but does NOT gate, because SU2's exact reference-state handling at
    MACH=1e-9 is not pinned in Phase 0 (that is Phase 5 step 62.5's job).
    """

    omega_f1_rad_per_s: float
    omega_f2_rad_per_s: float
    recovered_ia_nondim_f1: float
    recovered_ia_nondim_f2: float
    freq_consistency_rel_diff: float  # |Ia1-Ia2| / mean(|Ia1|,|Ia2|)
    freq_consistency_tol: float
    freq_consistency_passed: bool  # GATING
    closed_form_ia_nondim: float  # analytic I_a / chord² (assumed nondim)
    closed_form_factor_f1: float  # recovered_f1 / closed_form (advisory)
    closed_form_factor_tol: float
    closed_form_advisory_ok: bool  # advisory only — NOT gated
    drag_to_added_mass_ratio_f1: float  # regime diagnostic (advisory)
    passed: bool  # == freq_consistency_passed (the gate)


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

    The added-mass moment is purely inertial: ``M_added(t) = −I_a × θ̈(t)``,
    peak magnitude ``|M_added,peak| = I_a × ω² × θ_max``.

    Returns the per-unit-span peak moment in N·m / m (= N).
    """
    if chord_m <= 0 or pitching_omega_rad_per_s <= 0 or pitching_amplitude_rad <= 0:
        raise ValueError("chord, omega, and amplitude must be positive")
    b = chord_m / 2.0
    inertia_added = (
        math.pi * fluid_density_kg_per_m3 * b**4 * (1.0 / 8.0 + pivot_offset_normalized**2)
    )
    return inertia_added * pitching_omega_rad_per_s**2 * pitching_amplitude_rad


def _project_fundamental(
    series: list[float], *, n_cycles: int, discard_first_cycle: bool = True
) -> tuple[float, float]:
    """Project ``series`` onto sin(φ), cos(φ) at the fundamental.

    Phase is reconstructed from the cycle fraction (uniform sampling, integer
    cycles): row ``j`` of the kept window has ``φ_j = 2π·j / rows_per_cycle``.
    Returns ``(a_sin, a_cos)`` — the amplitudes of the sin/cos fundamental
    components (``2/N · Σ x·trig(φ)`` over an integer number of cycles).
    """
    n = len(series)
    if n == 0 or n_cycles < 1:
        return 0.0, 0.0
    rows_per_cycle = n // n_cycles
    if rows_per_cycle == 0:
        kept = series
        kept_offset = 0
    elif discard_first_cycle and n_cycles > 1:
        kept = series[rows_per_cycle:]
        kept_offset = rows_per_cycle
    else:
        kept = series
        kept_offset = 0
    if not kept or rows_per_cycle == 0:
        return 0.0, 0.0
    s_acc = 0.0
    c_acc = 0.0
    for k, x in enumerate(kept):
        phi = 2.0 * math.pi * (kept_offset + k) / rows_per_cycle
        s_acc += x * math.sin(phi)
        c_acc += x * math.cos(phi)
    return 2.0 * s_acc / len(kept), 2.0 * c_acc / len(kept)


def recover_added_mass_projection(
    history_csv_text: str,
    *,
    omega_rad_per_s: float,
    pitching_amplitude_rad: float,
    n_cycles: int,
    moment_column_candidates: Sequence[str] = ("cmy", "cmz", "cm", "cmoment"),
    history_path: str = "<inline>",
) -> AddedMassProjection:
    """Recover the added-mass coefficient from one SU2 2D-plate run.

    Fourier-projects the moment-coefficient trace onto the added-mass
    (sin φ) and damping (cos φ) bases over cycles 2..N, then forms the
    normalization-stable ``recovered_ia_nondim = a_sin / (ω²·θ_max)``.
    """
    if omega_rad_per_s <= 0 or pitching_amplitude_rad <= 0:
        raise ValueError("omega and amplitude must be positive")
    rows = _per_outer_iter(parse_history_csv(history_csv_text))
    moment = _column(rows, moment_column_candidates)
    a_sin, a_cos = _project_fundamental(moment, n_cycles=n_cycles)
    recovered_ia = a_sin / (omega_rad_per_s**2 * pitching_amplitude_rad)
    drag_ratio = abs(a_cos) / abs(a_sin) if a_sin != 0 else math.inf
    return AddedMassProjection(
        history_path=history_path,
        omega_rad_per_s=omega_rad_per_s,
        pitching_amplitude_rad=pitching_amplitude_rad,
        n_cycles=n_cycles,
        a_sin=a_sin,
        a_cos=a_cos,
        recovered_ia_nondim=recovered_ia,
        drag_to_added_mass_ratio=drag_ratio,
    )


def check_added_mass_freq_consistency(
    proj_f1: AddedMassProjection,
    proj_f2: AddedMassProjection,
    *,
    chord_m: float,
    pivot_offset_normalized: float,
    fluid_density_kg_per_m3: float = 1.225,
    freq_consistency_tol: float = 0.25,
    closed_form_factor_tol: float = 2.0,
) -> Tier1AddedMassResult:
    """Gate on frequency-consistency of the recovered added-mass coefficient.

    G1 (gating): ``|Ia(ω₁) − Ia(ω₂)| / mean`` < ``freq_consistency_tol``.
    Normalization-invariant — the only thing being compared is SU2 against
    SU2 at a different frequency, and ``I_a`` must be frequency-independent.

    Closed-form magnitude (advisory): recovered (avg) vs the Sedov/Newman
    ``I_a`` nondimensionalised under the assumed FREESTREAM_PRESS_EQ_ONE
    convention. Flagged if off by > ``closed_form_factor_tol``× ; never gates.
    """
    ia1 = proj_f1.recovered_ia_nondim
    ia2 = proj_f2.recovered_ia_nondim
    mean_mag = (abs(ia1) + abs(ia2)) / 2.0
    rel_diff = math.inf if mean_mag == 0 else abs(ia1 - ia2) / mean_mag
    freq_ok = rel_diff <= freq_consistency_tol

    # Advisory closed-form comparison under the assumed nondim convention:
    # moment-coefficient nondim divides dimensional N·m by q_ref·A_ref·L_ref;
    # under FREESTREAM_PRESS_EQ_ONE q_ref=1 and A_ref=L_ref=chord ⇒ /chord².
    b = chord_m / 2.0
    ia_analytic = (
        math.pi * fluid_density_kg_per_m3 * b**4 * (1.0 / 8.0 + pivot_offset_normalized**2)
    )
    closed_form_ia_nondim = ia_analytic / (chord_m**2) if chord_m > 0 else math.nan
    avg_recovered = (ia1 + ia2) / 2.0
    if closed_form_ia_nondim == 0 or math.isnan(closed_form_ia_nondim):
        factor = math.nan
    else:
        factor = avg_recovered / closed_form_ia_nondim
    advisory_ok = (
        not math.isnan(factor)
        and factor != 0
        and (1.0 / closed_form_factor_tol) <= abs(factor) <= closed_form_factor_tol
    )

    return Tier1AddedMassResult(
        omega_f1_rad_per_s=proj_f1.omega_rad_per_s,
        omega_f2_rad_per_s=proj_f2.omega_rad_per_s,
        recovered_ia_nondim_f1=ia1,
        recovered_ia_nondim_f2=ia2,
        freq_consistency_rel_diff=rel_diff,
        freq_consistency_tol=freq_consistency_tol,
        freq_consistency_passed=bool(freq_ok),
        closed_form_ia_nondim=closed_form_ia_nondim,
        closed_form_factor_f1=factor,
        closed_form_factor_tol=closed_form_factor_tol,
        closed_form_advisory_ok=bool(advisory_ok),
        drag_to_added_mass_ratio_f1=proj_f1.drag_to_added_mass_ratio,
        passed=bool(freq_ok),  # the gate IS the frequency-consistency check
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
    """Roll-up of Spike 0.6d (V1 scope, post-2026-05-15 redesign).

    Gate semantics (redesigned 2026-05-15):
      * ``overall_passed = sub_06d_2.freq_consistency_passed`` ONLY.
        Sub-spike 0.6d.2's two-frequency added-mass consistency check is
        the sole Phase-4 gate — it is the one normalization-invariant,
        parameter-free falsification test of the Tier-1 unsteady numerics.
      * Sub-spike 0.6d.1 (symmetry + dimensional envelope) is now
        **ADVISORY** — it was demoted on 2026-05-15 after the live Colab
        run showed its magnitude check conflated SU2's q_ref=1
        nondimensional output with a dimensional Newton envelope, and its
        symmetry criterion is ill-posed for the fan's net-work regime.
      * Sub-spike 0.6d.3 (incompressible cross-check) remains ADVISORY.
      * ``scripts/launch_phase4.py`` reads ``data/spike_0_6d/PASS``, which
        ``scripts/run_spike_0_6d.py`` writes iff ``overall_passed``.

    sub_1 / sub_3 results are still recorded (they feed Phase 5 step 62.5)
    but do NOT affect the gate decision.
    """

    sub_06d_2: Tier1AddedMassResult
    sub_06d_1: Tier1SymmetryDimensionalResult | None = field(default=None)
    sub_06d_3: Tier1IncompResult | None = field(default=None)
    overall_passed: bool = False


def analyze_spike_06d(
    sub_2: Tier1AddedMassResult,
    sub_1: Tier1SymmetryDimensionalResult | None = None,
    sub_3: Tier1IncompResult | None = None,
) -> Spike06dResult:
    """Roll up the sub-spike results.

    The gate is sub_2's frequency-consistency check ONLY. sub_1 and sub_3
    are advisory (recorded for Phase 5, do not gate) — see ``Spike06dResult``.
    """
    return Spike06dResult(
        sub_06d_2=sub_2,
        sub_06d_1=sub_1,
        sub_06d_3=sub_3,
        overall_passed=bool(sub_2.passed),
    )
