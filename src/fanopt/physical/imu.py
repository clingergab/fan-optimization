"""IMU traces → angular-work-per-cycle.

Implements Spike 0.3 (`docs/spike_0_3_protocol.md`,
`docs/plan_R11.md §Phase 0 Spike 0.3`) and the Phase-6 step-77 reuse:

  W_cycle = ∫_0^T |I_wrist · ω · (dω/dt)| dt

The **rectified** absolute-value form is intentional. For ideal periodic
SHM the signed integral evaluates to zero (the kinetic energy oscillates
between extrema, no net work). The operator's wrist actually expends
muscle in both accelerating and decelerating phases, so the absolute
integral is what "work the wrist does per cycle" means physically. Over
ideal SHM `W_cycle_rectified ≈ 2 · I · ω_max²` per cycle, which is the
quantity the Pareto objective in §6.4 implicitly normalizes against.

This module also extracts kinematic sanity values from the IMU trace —
f_wave, ω_max, θ_max — that the runner can compare against the locked
spec (2 Hz, 8.8 rad/s, 0.7 rad) to flag operator-cadence drift.

Reference:
- Spec: `docs/plan_R11.md §Phase 0 Spike 0.3`, `§Phase 5 step 77`
- Protocol: `docs/spike_0_3_protocol.md`
- I_wrist source: Spike 0.2 (`src/fanopt/physical/inertia.py`)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# np.trapezoid is numpy 2.0+; np.trapz exists on 1.x. Pick whichever's there
# so the module works on both pinned environments.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

__all__ = [
    "F_WAVE_TARGET_HZ",
    "OMEGA_MAX_TARGET_RAD_PER_S",
    "THETA_MAX_TARGET_RAD",
    "F_WAVE_TOL",
    "OMEGA_MAX_TOL",
    "THETA_MAX_TOL",
    "IMUTrace",
    "IMUResult",
    "load_imu_csv",
    "compute_w_cycle",
    "analyze_imu_trace",
]

# Locked spec targets (§3.2.3 kinematics).
F_WAVE_TARGET_HZ: float = 2.0
OMEGA_MAX_TARGET_RAD_PER_S: float = 8.8
THETA_MAX_TARGET_RAD: float = 0.6981  # 40°

# Sanity-band tolerances (relative). Not hard fails — warnings only.
F_WAVE_TOL: float = 0.10
OMEGA_MAX_TOL: float = 0.15
THETA_MAX_TOL: float = 0.20


@dataclass(frozen=True)
class IMUTrace:
    """One IMU recording — time, angle, angular velocity.

    `theta_rad` and `omega_rad_per_s` are about the **+y wrist axis** (§6.4
    coordinate convention). If the IMU was mounted on a different axis, the
    quantities below are not what `W_cycle` expects.
    """

    t_s: np.ndarray
    theta_rad: np.ndarray
    omega_rad_per_s: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.t_s.size)

    @property
    def duration_s(self) -> float:
        return float(self.t_s[-1] - self.t_s[0]) if self.t_s.size > 1 else 0.0

    @property
    def sample_rate_hz(self) -> float:
        if self.t_s.size < 2:
            return 0.0  # pragma: no cover  (load_imu_csv rejects size<8 already)
        return (self.n_samples - 1) / self.duration_s


@dataclass(frozen=True)
class IMUResult:
    """Per-trace analysis output."""

    W_cycle_J: float
    """Mean rectified-power integral per cycle (one cycle ≈ 1 / f_wave)."""

    n_cycles_integrated: float
    """Number of cycles spanned by the trace (duration × f_wave)."""

    f_wave_Hz: float
    """Dominant frequency in ω(t); typically near 2 Hz."""

    omega_max_rad_per_s: float
    """Peak |ω|."""

    theta_max_rad: float
    """Peak |θ|; if θ wasn't recorded, this is reconstructed from ω."""

    f_wave_ok: bool
    omega_max_ok: bool
    theta_max_ok: bool

    sanity_ok: bool
    """All three sanity bands passed."""


def load_imu_csv(path: Path | str) -> IMUTrace:
    """Load `t_s, theta_rad, omega_rad_per_s` from CSV.

    Tolerant about column order. `theta_rad` is optional — if missing, it is
    reconstructed from ω via cumulative trapezoidal integration. `omega_rad_per_s`
    is required (it carries the kinematic content W_cycle integrates).

    Lines starting with '#' and blank lines are skipped.
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
    except ValueError as e:
        raise ValueError(f"{path}: missing 't_s' column ({header})") from e
    try:
        omega_idx = cols_lower.index("omega_rad_per_s")
    except ValueError as e:
        raise ValueError(f"{path}: missing 'omega_rad_per_s' column ({header})") from e
    theta_idx = cols_lower.index("theta_rad") if "theta_rad" in cols_lower else None

    t = np.asarray([float(r[t_idx]) for r in raw], dtype=float)
    omega = np.asarray([float(r[omega_idx]) for r in raw], dtype=float)
    if theta_idx is not None:
        theta = np.asarray([float(r[theta_idx]) for r in raw], dtype=float)
    else:
        # ∫ω dt; constant of integration set so θ has zero mean (pure oscillation).
        theta = _cumulative_trapezoid(omega, t)
        theta = theta - theta.mean()

    if t.size < 8:
        raise ValueError(
            f"{path}: only {t.size} samples — IMU trace must cover ≥1 cycle at " f">= 100 Hz."
        )
    if not np.all(np.diff(t) > 0):
        raise ValueError(f"{path}: t_s is not monotonically increasing")

    return IMUTrace(t_s=t, theta_rad=theta, omega_rad_per_s=omega)


def compute_w_cycle(trace: IMUTrace, I_wrist_kgm2: float) -> float:
    """Mean rectified work per cycle.

    Returns `(1 / n_cycles) · ∫_0^T |I · ω · (dω/dt)| dt`.

    `n_cycles` comes from the FFT-derived f_wave (so partial cycles at the
    end of the recording don't bias the per-cycle average).
    """
    if I_wrist_kgm2 <= 0:
        raise ValueError(f"I_wrist must be > 0, got {I_wrist_kgm2}")
    omega = trace.omega_rad_per_s
    t = trace.t_s
    domega_dt = np.gradient(omega, t)
    abs_power = np.abs(I_wrist_kgm2 * omega * domega_dt)
    total_work = float(_trapezoid(abs_power, t))
    f_wave = _dominant_frequency(omega, t)
    n_cycles = max(trace.duration_s * f_wave, 1.0)
    return total_work / n_cycles


def analyze_imu_trace(trace: IMUTrace, I_wrist_kgm2: float) -> IMUResult:
    """Compute W_cycle + kinematic sanity flags for one IMU recording."""
    omega = trace.omega_rad_per_s
    theta = trace.theta_rad
    t = trace.t_s

    f_wave = _dominant_frequency(omega, t)
    omega_max = float(np.max(np.abs(omega)))
    theta_max = float(np.max(np.abs(theta)))

    domega_dt = np.gradient(omega, t)
    abs_power = np.abs(I_wrist_kgm2 * omega * domega_dt)
    total_work = float(_trapezoid(abs_power, t))
    n_cycles = max(trace.duration_s * f_wave, 1.0)
    W_cycle = total_work / n_cycles

    f_ok = abs(f_wave - F_WAVE_TARGET_HZ) / F_WAVE_TARGET_HZ < F_WAVE_TOL
    o_ok = abs(omega_max - OMEGA_MAX_TARGET_RAD_PER_S) / OMEGA_MAX_TARGET_RAD_PER_S < OMEGA_MAX_TOL
    th_ok = abs(theta_max - THETA_MAX_TARGET_RAD) / THETA_MAX_TARGET_RAD < THETA_MAX_TOL

    return IMUResult(
        W_cycle_J=W_cycle,
        n_cycles_integrated=n_cycles,
        f_wave_Hz=f_wave,
        omega_max_rad_per_s=omega_max,
        theta_max_rad=theta_max,
        f_wave_ok=f_ok,
        omega_max_ok=o_ok,
        theta_max_ok=th_ok,
        sanity_ok=f_ok and o_ok and th_ok,
    )


# ---------- internal --------------------------------------------------------


def _dominant_frequency(signal: np.ndarray, t: np.ndarray) -> float:
    """FFT-based dominant frequency. Removes DC; assumes ~uniform dt.

    Returns the frequency (Hz) with maximum spectral magnitude in the
    positive-frequency band. Falls back to zero-crossing rate if the FFT
    can't find a clean peak (signal too short or too noisy).
    """
    n = signal.size
    if n < 8:
        return 0.0
    dt_mean = float(np.mean(np.diff(t)))
    if dt_mean <= 0:
        return 0.0
    s = signal - signal.mean()
    spectrum = np.fft.rfft(s)
    freqs = np.fft.rfftfreq(n, d=dt_mean)
    mag = np.abs(spectrum)
    if mag.size < 2:  # pragma: no cover  (unreachable: n<8 guard above blocks it)
        return 0.0
    peak_idx = int(np.argmax(mag[1:]) + 1)
    f_peak = float(freqs[peak_idx])
    if f_peak > 0:
        return f_peak
    # Fallback: zero-crossing rate. Unreachable in practice — peak_idx ≥ 1 always
    # gives freqs[peak_idx] > 0 since dt_mean > 0 by the guard above. Kept as
    # paranoia for any future code that calls _dominant_frequency with weirder
    # inputs.
    zc = int(np.sum(np.diff(np.signbit(s)) != 0))  # pragma: no cover
    duration = float(t[-1] - t[0])  # pragma: no cover
    return zc / (2.0 * duration) if duration > 0 else 0.0  # pragma: no cover


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integration, returning an array the same length as y."""
    if y.size != x.size:
        raise ValueError(f"y, x size mismatch: {y.size} vs {x.size}")
    dx = np.diff(x)
    y_mid = 0.5 * (y[1:] + y[:-1])
    out = np.empty_like(y)
    out[0] = 0.0
    out[1:] = np.cumsum(y_mid * dx)
    return out
