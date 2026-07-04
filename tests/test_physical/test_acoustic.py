"""Tests for fanopt.physical.acoustic (known-tone signals → acoustic signature)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fanopt.physical.acoustic import (
    P_REF_PA,
    AcousticTrace,
    amplitude_spectrum,
    analyze_acoustic_trace,
    find_tonal_peaks,
    load_acoustic_csv,
    overall_spl_db,
    spectral_centroid_hz,
)

_FS = 8000.0
_DUR = 1.0  # 1 s at 8 kHz → 1 Hz bins, tones land on exact bins


def _tone(freq_amp: list[tuple[float, float]]) -> AcousticTrace:
    t = np.arange(0.0, _DUR, 1.0 / _FS)
    x = np.zeros_like(t)
    for f, a in freq_amp:
        x += a * np.sin(2.0 * math.pi * f * t)
    return AcousticTrace(sample_rate_hz=_FS, pressure_pa=x)


# --- overall SPL ---


def test_overall_spl_matches_rms_of_tone():
    a = 0.5
    spl = overall_spl_db(_tone([(250.0, a)]))
    assert spl == pytest.approx(20.0 * math.log10((a / math.sqrt(2)) / P_REF_PA), rel=1e-6)


def test_overall_spl_of_silence_is_neg_inf():
    assert overall_spl_db(AcousticTrace(_FS, np.zeros(1000))) == float("-inf")


# --- amplitude spectrum ---


def test_amplitude_spectrum_recovers_tone_amplitude():
    freqs, amp = amplitude_spectrum(_tone([(250.0, 0.5)]), window="none")
    i = int(np.argmax(amp))
    assert freqs[i] == pytest.approx(250.0)
    assert amp[i] == pytest.approx(0.5, rel=1e-6)


def test_amplitude_spectrum_hann_recovers_bin_aligned_tone():
    freqs, amp = amplitude_spectrum(_tone([(250.0, 0.5)]), window="hann")
    assert amp[int(np.argmax(amp))] == pytest.approx(0.5, rel=1e-3)


def test_amplitude_spectrum_rejects_unknown_window():
    with pytest.raises(ValueError, match="unknown window"):
        amplitude_spectrum(_tone([(250.0, 0.5)]), window="blackman")


# --- centroid + dominant ---


def test_spectral_centroid_of_single_tone_is_that_tone():
    freqs, amp = amplitude_spectrum(_tone([(300.0, 1.0)]), window="none")
    assert spectral_centroid_hz(freqs, amp) == pytest.approx(300.0, abs=1.0)


def test_centroid_of_empty_spectrum_is_zero():
    assert spectral_centroid_hz(np.array([1.0, 2.0]), np.array([0.0, 0.0])) == 0.0


def test_dominant_frequency_is_the_loudest_tone():
    res = analyze_acoustic_trace(_tone([(250.0, 1.0), (500.0, 0.3)]))
    assert res.dominant_frequency_hz == pytest.approx(250.0)


# --- tonal peaks ---


def test_two_tones_give_two_peaks_strongest_first():
    freqs, amp = amplitude_spectrum(_tone([(250.0, 1.0), (500.0, 0.5)]), window="none")
    peaks = find_tonal_peaks(freqs, amp)
    assert peaks[0].frequency_hz == pytest.approx(250.0)
    assert {round(p.frequency_hz) for p in peaks[:2]} == {250, 500}


def test_high_prominence_threshold_suppresses_peaks():
    freqs, amp = amplitude_spectrum(_tone([(250.0, 1.0)]), window="none")
    assert find_tonal_peaks(freqs, amp, prominence_db=500.0) == []


def test_max_peaks_caps_returned_count():
    freqs, amp = amplitude_spectrum(
        _tone([(200.0, 1.0), (400.0, 0.9), (600.0, 0.8), (800.0, 0.7)]), window="none"
    )
    assert len(find_tonal_peaks(freqs, amp, max_peaks=2)) == 2


# --- blade-pass tone ---


def test_blade_pass_level_reads_the_tone_at_that_frequency():
    # blade_count 10 × f_wave 2 Hz would be 20 Hz; use a resolvable 400 Hz tone here.
    res = analyze_acoustic_trace(_tone([(400.0, 0.4)]), blade_pass_frequency_hz=400.0)
    expected = 20.0 * math.log10((0.4 / math.sqrt(2)) / P_REF_PA)
    assert res.blade_pass_level_db == pytest.approx(expected, rel=1e-3)


def test_blade_pass_rejects_nonpositive_frequency():
    with pytest.raises(ValueError, match="blade_pass_frequency_hz"):
        analyze_acoustic_trace(_tone([(250.0, 0.5)]), blade_pass_frequency_hz=0.0)


def test_analyze_without_blade_pass_leaves_fields_none():
    res = analyze_acoustic_trace(_tone([(250.0, 0.5)]))
    assert res.blade_pass_frequency_hz is None
    assert res.blade_pass_level_db is None


# --- CSV loader ---


def _write_csv(tmp_path, rows: str):
    p = tmp_path / "mic.csv"
    p.write_text(rows, encoding="utf-8")
    return p


def test_load_csv_derives_sample_rate(tmp_path):
    t = np.arange(0.0, 0.02, 1.0 / _FS)
    lines = ["# mic @ 300 mm", "t_s,pressure_pa"] + [f"{ti},{0.1 * math.sin(ti)}" for ti in t]
    trace = load_acoustic_csv(_write_csv(tmp_path, "\n".join(lines)))
    assert trace.sample_rate_hz == pytest.approx(_FS, rel=1e-6)
    assert trace.n_samples == t.size


def test_load_csv_missing_column_raises(tmp_path):
    with pytest.raises(ValueError, match="pressure_pa"):
        load_acoustic_csv(_write_csv(tmp_path, "t_s,foo\n0.0,1.0\n0.1,2.0\n"))


def test_load_csv_too_short_raises(tmp_path):
    rows = "t_s,pressure_pa\n" + "\n".join(f"{i * 0.1},{i}" for i in range(4))
    with pytest.raises(ValueError, match="too short"):
        load_acoustic_csv(_write_csv(tmp_path, rows))


def test_load_csv_non_monotonic_time_raises(tmp_path):
    rows = "t_s,pressure_pa\n" + "\n".join(f"{0.1 if i == 10 else i * 0.001},{i}" for i in range(20))
    with pytest.raises(ValueError, match="monotonically"):
        load_acoustic_csv(_write_csv(tmp_path, rows))


def test_load_csv_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="no rows"):
        load_acoustic_csv(_write_csv(tmp_path, "t_s,pressure_pa\n"))
