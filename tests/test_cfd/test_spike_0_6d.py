"""Tests for ``fanopt.cfd.spike_0_6d`` — the three Tier-1 quantitative-sanity
counter-checks (H10 supplement, 2026-05-14 addition).

Sub-spike 0.6d.1 + 0.6d.2 are gating; 0.6d.3 is advisory. The aggregator's
``overall_passed`` flag derives from sub_1 AND sub_2 only.

Spec reference: docs/report-final.md §Phase 0 Spike 0.6d.
Lock reference: Round-9 HIGH-12 (= C12) for the unsteady MACH lock.
"""

from __future__ import annotations

import io
import math

import pytest

from fanopt.cfd.spike_0_6d import (
    MACH_UNSTEADY_LOCK,
    Tier1AddedMassResult,
    Tier1IncompResult,
    Tier1SymmetryDimensionalResult,
    analyze_spike_06d,
    check_added_mass,
    check_incompressible_cross,
    check_symmetry_dimensional,
    compute_added_mass_moment_closed_form_2d_plate,
    compute_dimensional_envelope,
    cycle_averages_and_peaks,
    parse_history_csv,
)

# ---- shared CSV helpers ----------------------------------------------------


def _sinusoidal_history_csv(
    *, n_cycles: int, samples_per_cycle: int, amplitude: float, mean_bias: float = 0.0
) -> str:
    """Build a synthetic SU2-style history.csv with one sinusoid in CFx."""
    out = io.StringIO()
    out.write("Time_Iter,Inner_Iter,CFx,CD\n")
    total = n_cycles * samples_per_cycle
    for i in range(total):
        phase = 2.0 * math.pi * (i % samples_per_cycle) / samples_per_cycle
        cfx = mean_bias + amplitude * math.sin(phase)
        out.write(f"{i},0,{cfx:.6f},0.05\n")
    return out.getvalue()


def _two_inner_iter_history_csv() -> str:
    """A history with TWO inner-iter rows per outer step; only LAST should be kept."""
    out = io.StringIO()
    out.write("Time_Iter,Inner_Iter,CFx,CD\n")
    for i in range(20):
        # First inner row -- wrong value
        out.write(f"{i},0,99.0,0.0\n")
        # Last inner row -- correct value
        out.write(f"{i},1,{math.sin(2.0 * math.pi * i / 10.0):.6f},0.05\n")
    return out.getvalue()


# ---- parse_history_csv -----------------------------------------------------


def test_parse_history_csv_handles_quoted_headers() -> None:
    text = '"Time_Iter","CFx","CD"\n0,0.1,0.01\n1,0.2,0.02\n'
    rows = parse_history_csv(text)
    assert len(rows) == 2
    assert rows[0]["cfx"] == pytest.approx(0.1)
    assert rows[1]["cd"] == pytest.approx(0.02)


def test_parse_history_csv_drops_non_numeric_cells() -> None:
    text = "Time_Iter,Note,CFx\n0,launch,0.1\n1,run,0.2\n"
    rows = parse_history_csv(text)
    assert all("note" not in r for r in rows)
    assert rows[0]["cfx"] == pytest.approx(0.1)


# ---- cycle_averages_and_peaks ---------------------------------------------


def test_cycle_averages_zero_for_pure_sinusoid() -> None:
    """A zero-mean sinusoid over an integer number of cycles -> avg ≈ 0."""
    csv_text = _sinusoidal_history_csv(n_cycles=5, samples_per_cycle=200, amplitude=1.0)
    avg, peak, n = cycle_averages_and_peaks(csv_text, n_cycles=5)
    assert abs(avg) < 1e-2
    assert peak == pytest.approx(1.0, abs=1e-2)
    assert n > 0


def test_cycle_averages_biased_sinusoid_reflects_mean() -> None:
    csv_text = _sinusoidal_history_csv(
        n_cycles=5, samples_per_cycle=200, amplitude=1.0, mean_bias=2.0
    )
    avg, peak, _ = cycle_averages_and_peaks(csv_text, n_cycles=5)
    assert avg == pytest.approx(2.0, abs=0.05)
    assert peak == pytest.approx(3.0, abs=0.05)  # 2.0 bias + 1.0 amplitude


def test_cycle_averages_collapses_multi_inner_iter_rows() -> None:
    """When SU2 emits multiple inner-iter rows per outer step, keep the LAST."""
    csv_text = _two_inner_iter_history_csv()
    avg, peak, _ = cycle_averages_and_peaks(csv_text, n_cycles=2)
    # If the first inner-iter rows (CFx=99) had bled through, peak would be ~99.
    assert peak < 5.0


def test_cycle_averages_history_shorter_than_n_cycles_uses_whole_window() -> None:
    """Business-logic branch: when the history has fewer rows than n_cycles
    (per_cycle == 0), the function falls back to the whole window rather than
    dropping everything. Covers the truncated/interrupted-run case (e.g., a
    Colab session that died mid-run leaving a 3-row history.csv)."""
    out = io.StringIO()
    out.write("Time_Iter,Inner_Iter,CFx,CD\n")
    for i in range(3):  # only 3 rows, but caller asks for 5 cycles
        out.write(f"{i},0,{0.5 * (i - 1):.3f},0.05\n")
    csv_text = out.getvalue()
    avg, peak, n = cycle_averages_and_peaks(csv_text, n_cycles=5)
    # All 3 rows are used (no cycle-1 discard possible with < n_cycles rows).
    assert n == 3
    # CFx values: -0.5, 0.0, +0.5 → avg = 0.0, peak = 0.5
    assert avg == pytest.approx(0.0, abs=1e-9)
    assert peak == pytest.approx(0.5, abs=1e-9)


# ---- 0.6d.1 symmetry + dimensional sanity ---------------------------------


def test_symmetry_passes_on_perfectly_periodic_force_trace() -> None:
    csv_text = _sinusoidal_history_csv(n_cycles=5, samples_per_cycle=200, amplitude=0.5)
    res = check_symmetry_dimensional(
        csv_text,
        n_cycles=5,
        mass_kg=0.05,
        omega_rad_per_s=12.5664,
        r_cm_m=0.1,
        envelope_geometry="test plate",
    )
    assert res.symmetry_passed is True


def test_symmetry_fails_on_biased_force_trace() -> None:
    """Bias_ratio ≈ 2.0 (matches the 2026-05-14 diagnostic finding) → fails."""
    csv_text = _sinusoidal_history_csv(
        n_cycles=5, samples_per_cycle=200, amplitude=1.0, mean_bias=2.0
    )
    res = check_symmetry_dimensional(
        csv_text,
        n_cycles=5,
        mass_kg=0.05,
        omega_rad_per_s=12.5664,
        r_cm_m=0.1,
        envelope_geometry="biased trace",
    )
    assert res.symmetry_passed is False
    assert res.symmetry_ratio > 0.5  # well above the 0.05 threshold


def test_dimensional_envelope_passes_within_order_of_magnitude() -> None:
    # m × ω² × r = 0.05 × 12.5664² × 0.1 ≈ 0.789 N. Use peak amplitude = 0.5 N
    # → log10(0.5 / 0.789) ≈ −0.2, well inside ±1.0.
    csv_text = _sinusoidal_history_csv(n_cycles=5, samples_per_cycle=200, amplitude=0.5)
    res = check_symmetry_dimensional(
        csv_text,
        n_cycles=5,
        mass_kg=0.05,
        omega_rad_per_s=12.5664,
        r_cm_m=0.1,
        envelope_geometry="test plate",
    )
    assert res.magnitude_passed is True
    assert abs(res.magnitude_ratio_log10) < 1.0


def test_dimensional_envelope_fails_when_off_by_4_orders() -> None:
    """The 2026-05-14 Cell-8 output had CL ≈ 10⁶ — 4+ orders too large."""
    csv_text = _sinusoidal_history_csv(n_cycles=5, samples_per_cycle=200, amplitude=1.0e6)
    res = check_symmetry_dimensional(
        csv_text,
        n_cycles=5,
        mass_kg=0.05,
        omega_rad_per_s=12.5664,
        r_cm_m=0.1,
        envelope_geometry="non-physical magnitude",
    )
    assert res.magnitude_passed is False
    assert res.magnitude_ratio_log10 > 5.0  # 6+ orders too large


def test_dimensional_envelope_uses_geometry_specific_mass_and_radius() -> None:
    """Envelope is computed for whatever mass + r are passed in (F4 fix 2026-05-14)."""
    envelope_v1_panel = compute_dimensional_envelope(
        mass_kg=0.005, omega_rad_per_s=12.5664, r_cm_m=0.1
    )
    envelope_naca_0012 = compute_dimensional_envelope(
        mass_kg=0.05, omega_rad_per_s=12.5664, r_cm_m=0.25
    )
    assert envelope_naca_0012 > envelope_v1_panel
    assert envelope_v1_panel == pytest.approx(0.005 * 12.5664**2 * 0.1, rel=1e-6)


def test_compute_dimensional_envelope_rejects_negative_inputs() -> None:
    with pytest.raises(ValueError):
        compute_dimensional_envelope(mass_kg=-1.0, omega_rad_per_s=1.0, r_cm_m=1.0)


# ---- 0.6d.2 added-mass closed form ----------------------------------------


def test_added_mass_closed_form_for_2d_plate_at_quarter_chord() -> None:
    """Sedov/Newman: I_a = π ρ b⁴ (1/8 + a²) per unit span."""
    chord = 1.0
    b = chord / 2.0
    a = -0.5  # quarter-chord
    rho = 1.225
    omega = 10.0
    theta_max = 0.1
    expected_I_a = math.pi * rho * b**4 * (1.0 / 8.0 + a**2)
    expected_M_peak = expected_I_a * omega**2 * theta_max
    actual = compute_added_mass_moment_closed_form_2d_plate(
        chord_m=chord,
        pivot_offset_normalized=a,
        pitching_omega_rad_per_s=omega,
        pitching_amplitude_rad=theta_max,
    )
    assert actual == pytest.approx(expected_M_peak, rel=1e-9)


def test_added_mass_closed_form_rejects_nonpositive_inputs() -> None:
    with pytest.raises(ValueError):
        compute_added_mass_moment_closed_form_2d_plate(
            chord_m=0.0,
            pivot_offset_normalized=-0.5,
            pitching_omega_rad_per_s=10.0,
            pitching_amplitude_rad=0.1,
        )


def test_added_mass_passes_when_su2_within_15pct_of_closed_form() -> None:
    closed_form = compute_added_mass_moment_closed_form_2d_plate(
        chord_m=1.0,
        pivot_offset_normalized=-0.5,
        pitching_omega_rad_per_s=10.0,
        pitching_amplitude_rad=0.1,
    )
    su2_within = closed_form * 1.10  # +10%
    res = check_added_mass(
        su2_moment_peak=su2_within,
        chord_m=1.0,
        pivot_offset_normalized=-0.5,
        pitching_omega_rad_per_s=10.0,
        pitching_amplitude_rad=0.1,
    )
    assert res.passed is True
    assert abs(res.relative_error) < 0.15


def test_added_mass_fails_outside_15pct() -> None:
    closed_form = compute_added_mass_moment_closed_form_2d_plate(
        chord_m=1.0,
        pivot_offset_normalized=-0.5,
        pitching_omega_rad_per_s=10.0,
        pitching_amplitude_rad=0.1,
    )
    su2_off = closed_form * 1.30  # +30%, well outside ±15%
    res = check_added_mass(
        su2_moment_peak=su2_off,
        chord_m=1.0,
        pivot_offset_normalized=-0.5,
        pitching_omega_rad_per_s=10.0,
        pitching_amplitude_rad=0.1,
    )
    assert res.passed is False
    assert res.relative_error == pytest.approx(0.30, rel=1e-2)


# ---- 0.6d.3 incompressible cross-check (advisory) -------------------------


def test_incompressible_cross_check_passes_within_20pct() -> None:
    res = check_incompressible_cross(
        compressible_force_cycle_avg=0.50,
        incompressible_force_cycle_avg=0.55,  # +10%
    )
    assert res.passed is True
    assert res.relative_error < 0.20


def test_incompressible_cross_check_logged_as_advisory_on_fail() -> None:
    """Failure doesn't raise; aggregator ignores it for marker decisions."""
    res = check_incompressible_cross(
        compressible_force_cycle_avg=0.50,
        incompressible_force_cycle_avg=0.80,  # +60%, well outside ±20%
    )
    assert res.passed is False
    # The check itself returns a result; only the aggregator interprets it.
    assert isinstance(res, Tier1IncompResult)


# ---- aggregator (Spike06dResult) ------------------------------------------


def _passing_sub_1() -> Tier1SymmetryDimensionalResult:
    csv_text = _sinusoidal_history_csv(n_cycles=5, samples_per_cycle=200, amplitude=0.5)
    return check_symmetry_dimensional(
        csv_text,
        n_cycles=5,
        mass_kg=0.05,
        omega_rad_per_s=12.5664,
        r_cm_m=0.1,
        envelope_geometry="passing fixture",
    )


def _failing_sub_1() -> Tier1SymmetryDimensionalResult:
    csv_text = _sinusoidal_history_csv(
        n_cycles=5, samples_per_cycle=200, amplitude=1.0, mean_bias=2.0
    )
    return check_symmetry_dimensional(
        csv_text,
        n_cycles=5,
        mass_kg=0.05,
        omega_rad_per_s=12.5664,
        r_cm_m=0.1,
        envelope_geometry="failing fixture",
    )


def _passing_sub_2() -> Tier1AddedMassResult:
    closed = compute_added_mass_moment_closed_form_2d_plate(
        chord_m=1.0,
        pivot_offset_normalized=-0.5,
        pitching_omega_rad_per_s=10.0,
        pitching_amplitude_rad=0.1,
    )
    return check_added_mass(
        su2_moment_peak=closed * 1.05,
        chord_m=1.0,
        pivot_offset_normalized=-0.5,
        pitching_omega_rad_per_s=10.0,
        pitching_amplitude_rad=0.1,
    )


def _failing_sub_2() -> Tier1AddedMassResult:
    closed = compute_added_mass_moment_closed_form_2d_plate(
        chord_m=1.0,
        pivot_offset_normalized=-0.5,
        pitching_omega_rad_per_s=10.0,
        pitching_amplitude_rad=0.1,
    )
    return check_added_mass(
        su2_moment_peak=closed * 1.40,
        chord_m=1.0,
        pivot_offset_normalized=-0.5,
        pitching_omega_rad_per_s=10.0,
        pitching_amplitude_rad=0.1,
    )


def test_aggregator_pass_requires_sub_1_AND_sub_2_only() -> None:
    """sub_3 FAIL doesn't block the gate (advisory). sub_1 + sub_2 PASS → gate PASS."""
    failing_sub_3 = check_incompressible_cross(
        compressible_force_cycle_avg=0.5, incompressible_force_cycle_avg=0.9
    )
    assert failing_sub_3.passed is False  # confirm fixture
    result = analyze_spike_06d(_passing_sub_1(), _passing_sub_2(), failing_sub_3)
    assert result.overall_passed is True
    assert result.sub_06d_3 is failing_sub_3


def test_aggregator_fail_when_either_gating_subspike_fails() -> None:
    failing_then_passing = analyze_spike_06d(_failing_sub_1(), _passing_sub_2())
    passing_then_failing = analyze_spike_06d(_passing_sub_1(), _failing_sub_2())
    assert failing_then_passing.overall_passed is False
    assert passing_then_failing.overall_passed is False


def test_aggregator_handles_missing_sub_3_gracefully() -> None:
    result = analyze_spike_06d(_passing_sub_1(), _passing_sub_2())
    assert result.sub_06d_3 is None
    assert result.overall_passed is True


# ---- MACH lock sanity check (cross-module) --------------------------------


def test_mach_unsteady_lock_matches_h12() -> None:
    """Spike 0.6d inherits the same MACH=1e-9 lock as Spike 0.6c (H12)."""
    assert MACH_UNSTEADY_LOCK == 1e-9
