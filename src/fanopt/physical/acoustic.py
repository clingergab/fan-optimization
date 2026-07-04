"""Microphone recordings → acoustic signature (SPL, spectrum, tonal peaks).

Phase-6 acoustic FFT analyzer for the microphone-at-300 mm measurement of the
assembled fan during waving (`docs/report-final.md` §Phase 6). Reduces a raw
pressure recording to the numbers the design comparison actually uses:

- **overall SPL** (dB re 20 µPa) — loudness,
- **dominant frequency** + **spectral centroid** — where the energy sits,
- **tonal peaks** — narrowband tones standing above the broadband floor, and
- the **blade-pass tone** level, when the blade-pass frequency is supplied
  (``blade_count × f_wave``) — the tonal fingerprint of the corrugation.

Pure-numpy (rfft); the only external boundary is ``load_acoustic_csv``. Spectrum
uses a Hann window for real (non-periodic) recordings; overall SPL is the exact
time-domain RMS, independent of any window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "P_REF_PA",
    "AcousticTrace",
    "TonalPeak",
    "AcousticResult",
    "load_acoustic_csv",
    "overall_spl_db",
    "amplitude_spectrum",
    "spectral_centroid_hz",
    "find_tonal_peaks",
    "analyze_acoustic_trace",
]

# Reference sound pressure — 0 dB SPL in air.
P_REF_PA: float = 20e-6


@dataclass(frozen=True)
class AcousticTrace:
    """One microphone recording: uniformly-sampled pressure (Pa) + sample rate."""

    sample_rate_hz: float
    pressure_pa: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.pressure_pa.size)

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate_hz


@dataclass(frozen=True)
class TonalPeak:
    """A narrowband tone standing above the broadband floor."""

    frequency_hz: float
    level_db: float  # SPL of the tonal component
    prominence_db: float  # amount it stands above the local broadband floor


@dataclass(frozen=True)
class AcousticResult:
    """Reduced acoustic signature of one recording."""

    spl_db: float
    dominant_frequency_hz: float
    spectral_centroid_hz: float
    tonal_peaks: tuple[TonalPeak, ...]
    blade_pass_frequency_hz: float | None = None
    blade_pass_level_db: float | None = None
    meta: dict[str, float] = field(default_factory=dict)


def load_acoustic_csv(path: Path | str) -> AcousticTrace:
    """Load ``t_s, pressure_pa`` from CSV; derive the sample rate from the timebase.

    Tolerant about column order. Lines starting with ``#`` and blank lines are
    skipped. Requires a near-uniform timebase (real ADC recordings are).
    """
    path = Path(path)
    raw: list[list[str]] = []
    header: list[str] | None = None
    with path.open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            cols = [c.strip() for c in s.split(",")]
            if header is None:
                header = cols
                continue
            raw.append(cols)
    if header is None or not raw:
        raise ValueError(f"{path}: no rows found")

    cols_lower = [h.lower() for h in header]
    try:
        t_idx = cols_lower.index("t_s")
        p_idx = cols_lower.index("pressure_pa")
    except ValueError as e:
        raise ValueError(f"{path}: need 't_s' and 'pressure_pa' columns ({header})") from e

    t = np.asarray([float(r[t_idx]) for r in raw], dtype=float)
    pressure = np.asarray([float(r[p_idx]) for r in raw], dtype=float)
    if t.size < 16:
        raise ValueError(f"{path}: only {t.size} samples — too short for a spectrum")
    if not np.all(np.diff(t) > 0):
        raise ValueError(f"{path}: t_s is not monotonically increasing")

    dt = np.diff(t)
    sample_rate = 1.0 / float(np.median(dt))
    return AcousticTrace(sample_rate_hz=sample_rate, pressure_pa=pressure)


def overall_spl_db(trace: AcousticTrace) -> float:
    """Overall sound pressure level (dB re 20 µPa) from the time-domain RMS."""
    rms = float(np.sqrt(np.mean(np.square(trace.pressure_pa))))
    if rms <= 0.0:
        return float("-inf")
    return 20.0 * np.log10(rms / P_REF_PA)


def amplitude_spectrum(trace: AcousticTrace, *, window: str = "hann") -> tuple[np.ndarray, np.ndarray]:
    """One-sided amplitude spectrum: ``(freqs_hz, amplitude_pa)``.

    Amplitude is coherent-gain-corrected so a pure tone of amplitude ``A``
    resolves to ``≈ A`` at its bin. ``window`` is ``"hann"`` (leakage-robust,
    for real recordings) or ``"none"`` (exact for periodic/bin-aligned tones).
    """
    x = np.asarray(trace.pressure_pa, dtype=float)
    x = x - x.mean()  # drop DC so it can't masquerade as a tone
    n = x.size
    if window == "hann":
        w = np.hanning(n)
    elif window == "none":
        w = np.ones(n)
    else:
        raise ValueError(f"unknown window {window!r}; use 'hann' or 'none'")
    spectrum = np.fft.rfft(x * w)
    freqs = np.fft.rfftfreq(n, d=1.0 / trace.sample_rate_hz)
    amp = 2.0 * np.abs(spectrum) / np.sum(w)
    return freqs, amp


def spectral_centroid_hz(freqs: np.ndarray, amp: np.ndarray) -> float:
    """Amplitude-weighted mean frequency (the spectrum's centre of mass)."""
    total = float(np.sum(amp))
    if total <= 0.0:
        return 0.0
    return float(np.sum(freqs * amp) / total)


def _amp_to_spl_db(amp_pa: np.ndarray | float) -> np.ndarray:
    """Convert tonal amplitude (Pa peak) to SPL (dB re 20 µPa), RMS-referenced."""
    rms = np.asarray(amp_pa, dtype=float) / np.sqrt(2.0)
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.maximum(rms, 1e-30) / P_REF_PA)


def find_tonal_peaks(
    freqs: np.ndarray,
    amp: np.ndarray,
    *,
    prominence_db: float = 6.0,
    max_peaks: int = 8,
) -> list[TonalPeak]:
    """Local-maximum bins standing ``prominence_db`` above the broadband floor.

    The floor is the median bin level (a robust broadband estimate). Peaks are
    returned strongest-first, capped at ``max_peaks``.
    """
    level = _amp_to_spl_db(amp)
    floor = float(np.median(level))
    peaks: list[TonalPeak] = []
    for i in range(1, level.size - 1):
        if level[i] >= level[i - 1] and level[i] > level[i + 1]:
            prom = level[i] - floor
            if prom >= prominence_db:
                peaks.append(TonalPeak(float(freqs[i]), float(level[i]), float(prom)))
    peaks.sort(key=lambda p: p.level_db, reverse=True)
    return peaks[:max_peaks]


def analyze_acoustic_trace(
    trace: AcousticTrace,
    *,
    blade_pass_frequency_hz: float | None = None,
    prominence_db: float = 6.0,
) -> AcousticResult:
    """Full reduction: SPL, dominant frequency, centroid, tonal peaks, blade-pass."""
    freqs, amp = amplitude_spectrum(trace)
    dominant = float(freqs[int(np.argmax(amp))])
    centroid = spectral_centroid_hz(freqs, amp)
    peaks = find_tonal_peaks(freqs, amp, prominence_db=prominence_db)

    bp_level: float | None = None
    if blade_pass_frequency_hz is not None:
        if blade_pass_frequency_hz <= 0:
            raise ValueError("blade_pass_frequency_hz must be > 0")
        idx = int(np.argmin(np.abs(freqs - blade_pass_frequency_hz)))
        bp_level = float(_amp_to_spl_db(amp[idx]))

    return AcousticResult(
        spl_db=overall_spl_db(trace),
        dominant_frequency_hz=dominant,
        spectral_centroid_hz=centroid,
        tonal_peaks=tuple(peaks),
        blade_pass_frequency_hz=blade_pass_frequency_hz,
        blade_pass_level_db=bp_level,
        meta={"duration_s": trace.duration_s, "sample_rate_hz": trace.sample_rate_hz},
    )
