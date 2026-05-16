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
    AddedMassProjection,
    Tier1AddedMassResult,
    Tier1IncompResult,
    Tier1SymmetryDimensionalResult,
    analyze_spike_06d,
    check_added_mass_freq_consistency,
    check_incompressible_cross,
    check_symmetry_dimensional,
    compute_added_mass_moment_closed_form_2d_plate,
    compute_dimensional_envelope,
    cycle_averages_and_peaks,
    parse_history_csv,
    recover_added_mass_projection,
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


# ---- 0.6d.2 added-mass frequency-consistency gate (2026-05-15 redesign) ---


def _moment_history_csv(
    *,
    omega: float,
    theta_max: float,
    ia_nondim: float,
    drag_nondim: float = 0.0,
    n_cycles: int = 5,
    samples_per_cycle: int = 200,
) -> str:
    """Synthesize an SU2-style moment history.

    The pure added-mass moment coefficient is ``K·sin(φ)`` where
    ``K = ia_nondim·ω²·θ_max`` (since ``M_am = -I_a·θ̈ ∝ +sin(ωt)``); the
    optional ``drag_nondim·cos(φ)`` term is the velocity-in-phase
    (damping/drag) component the projector must NOT count as added mass.
    """
    out = io.StringIO()
    out.write("Time_Iter,Inner_Iter,CMz,CD\n")
    k_am = ia_nondim * omega**2 * theta_max
    total = n_cycles * samples_per_cycle
    for i in range(total):
        phi = 2.0 * math.pi * (i % samples_per_cycle) / samples_per_cycle
        cm = k_am * math.sin(phi) + drag_nondim * math.cos(phi)
        out.write(f"{i},0,{cm:.8e},0.05\n")
    return out.getvalue()


def test_recover_added_mass_projection_recovers_known_coefficient() -> None:
    """The Fourier projector recovers the planted I_a (nondim) exactly."""
    proj = recover_added_mass_projection(
        _moment_history_csv(omega=6.2832, theta_max=0.1745, ia_nondim=0.037),
        omega_rad_per_s=6.2832,
        pitching_amplitude_rad=0.1745,
        n_cycles=5,
    )
    assert isinstance(proj, AddedMassProjection)
    assert proj.recovered_ia_nondim == pytest.approx(0.037, rel=1e-3)


def test_recover_added_mass_projection_ignores_drag_component() -> None:
    """A pure cos(φ) (damping) signal contributes ~0 to the added-mass projection."""
    proj = recover_added_mass_projection(
        _moment_history_csv(omega=6.2832, theta_max=0.1745, ia_nondim=0.0, drag_nondim=5.0),
        omega_rad_per_s=6.2832,
        pitching_amplitude_rad=0.1745,
        n_cycles=5,
    )
    assert abs(proj.a_sin) < 1e-6  # no added-mass content
    assert abs(proj.a_cos) > 1.0  # drag IS present, in the cos projection


def test_added_mass_passes_when_frequency_consistent() -> None:
    """Same planted I_a at ω and 2ω → recovered I_a agrees → GATE PASS.

    This is the normalization-invariant Phase-4 de-risk: I_a is a pure
    geometric/fluid constant; equal recovery at two frequencies is the
    signature of physically-faithful added-mass behaviour.
    """
    th = 0.1745
    p1 = recover_added_mass_projection(
        _moment_history_csv(omega=6.2832, theta_max=th, ia_nondim=0.037),
        omega_rad_per_s=6.2832,
        pitching_amplitude_rad=th,
        n_cycles=5,
    )
    p2 = recover_added_mass_projection(
        _moment_history_csv(omega=12.5664, theta_max=th, ia_nondim=0.037),
        omega_rad_per_s=12.5664,
        pitching_amplitude_rad=th,
        n_cycles=5,
    )
    res = check_added_mass_freq_consistency(p1, p2, chord_m=1.0, pivot_offset_normalized=-0.5)
    assert res.passed is True
    assert res.freq_consistency_passed is True
    assert res.freq_consistency_rel_diff < 0.25


def test_added_mass_fails_when_frequency_inconsistent() -> None:
    """Frequency-DEPENDENT recovered I_a (simulated numerical distortion) →
    GATE FAIL. This is exactly the Phase-4-invalidating failure mode the
    redesigned gate exists to catch."""
    th = 0.1745
    p1 = recover_added_mass_projection(
        _moment_history_csv(omega=6.2832, theta_max=th, ia_nondim=0.037),
        omega_rad_per_s=6.2832,
        pitching_amplitude_rad=th,
        n_cycles=5,
    )
    p2 = recover_added_mass_projection(
        _moment_history_csv(omega=12.5664, theta_max=th, ia_nondim=0.10),
        omega_rad_per_s=12.5664,
        pitching_amplitude_rad=th,
        n_cycles=5,
    )
    res = check_added_mass_freq_consistency(p1, p2, chord_m=1.0, pivot_offset_normalized=-0.5)
    assert res.passed is False
    assert res.freq_consistency_rel_diff > 0.25


def test_added_mass_closed_form_comparison_is_advisory_only() -> None:
    """The closed-form magnitude check is reported but never gates.

    Frequency-consistent runs PASS the gate even when the closed-form
    factor is off (the absolute nondim convention is Phase 5's job).
    """
    th = 0.1745
    # Both runs share an I_a that does NOT match the closed-form nondim
    # value — but they ARE mutually consistent, so the gate must still pass.
    p1 = recover_added_mass_projection(
        _moment_history_csv(omega=6.2832, theta_max=th, ia_nondim=0.50),
        omega_rad_per_s=6.2832,
        pitching_amplitude_rad=th,
        n_cycles=5,
    )
    p2 = recover_added_mass_projection(
        _moment_history_csv(omega=12.5664, theta_max=th, ia_nondim=0.50),
        omega_rad_per_s=12.5664,
        pitching_amplitude_rad=th,
        n_cycles=5,
    )
    res = check_added_mass_freq_consistency(p1, p2, chord_m=1.0, pivot_offset_normalized=-0.5)
    assert res.passed is True  # gate keys on freq-consistency ONLY
    assert res.closed_form_advisory_ok is False  # advisory flag can fail...
    assert res.passed is True  # ...without affecting the gate


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


def _advisory_sub_1(*, passing: bool) -> Tier1SymmetryDimensionalResult:
    if passing:
        csv_text = _sinusoidal_history_csv(n_cycles=5, samples_per_cycle=200, amplitude=0.5)
    else:
        csv_text = _sinusoidal_history_csv(
            n_cycles=5, samples_per_cycle=200, amplitude=1.0, mean_bias=2.0
        )
    return check_symmetry_dimensional(
        csv_text,
        n_cycles=5,
        mass_kg=0.05,
        omega_rad_per_s=12.5664,
        r_cm_m=0.1,
        envelope_geometry="advisory fixture",
    )


def _gating_sub_2(*, consistent: bool) -> Tier1AddedMassResult:
    th = 0.1745
    p1 = recover_added_mass_projection(
        _moment_history_csv(omega=6.2832, theta_max=th, ia_nondim=0.037),
        omega_rad_per_s=6.2832,
        pitching_amplitude_rad=th,
        n_cycles=5,
    )
    ia2 = 0.037 if consistent else 0.10
    p2 = recover_added_mass_projection(
        _moment_history_csv(omega=12.5664, theta_max=th, ia_nondim=ia2),
        omega_rad_per_s=12.5664,
        pitching_amplitude_rad=th,
        n_cycles=5,
    )
    return check_added_mass_freq_consistency(p1, p2, chord_m=1.0, pivot_offset_normalized=-0.5)


def test_aggregator_gates_on_sub_2_only_sub_1_and_sub_3_advisory() -> None:
    """Redesign (2026-05-15): the gate is sub_2's frequency-consistency
    ONLY. sub_1 FAIL + sub_3 FAIL must NOT block when sub_2 passes."""
    failing_sub_1 = _advisory_sub_1(passing=False)
    failing_sub_3 = check_incompressible_cross(
        compressible_force_cycle_avg=0.5, incompressible_force_cycle_avg=0.9
    )
    assert failing_sub_1.passed is False  # confirm fixtures
    assert failing_sub_3.passed is False
    result = analyze_spike_06d(_gating_sub_2(consistent=True), failing_sub_1, failing_sub_3)
    assert result.overall_passed is True  # sub_2 passed → gate open
    assert result.sub_06d_1 is failing_sub_1  # still recorded for Phase 5
    assert result.sub_06d_3 is failing_sub_3


def test_aggregator_fails_only_when_sub_2_fails() -> None:
    """Even with sub_1 PASSing, a sub_2 frequency-inconsistency closes the gate."""
    passing_sub_1 = _advisory_sub_1(passing=True)
    assert passing_sub_1.passed is True
    result = analyze_spike_06d(_gating_sub_2(consistent=False), passing_sub_1)
    assert result.overall_passed is False


def test_aggregator_works_with_sub_2_alone() -> None:
    """sub_1 / sub_3 are optional (advisory) — sub_2 alone determines the gate."""
    result = analyze_spike_06d(_gating_sub_2(consistent=True))
    assert result.sub_06d_1 is None
    assert result.sub_06d_3 is None
    assert result.overall_passed is True


# ---- MACH lock sanity check (cross-module) --------------------------------


def test_mach_unsteady_lock_matches_h12() -> None:
    """Spike 0.6d inherits the same MACH=1e-9 lock as Spike 0.6c (H12)."""
    assert MACH_UNSTEADY_LOCK == 1e-9
