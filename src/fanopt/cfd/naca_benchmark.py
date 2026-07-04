"""Oscillating NACA-0012 solver-validation benchmark (Spike 0.6c.2, SU2 side).

Validates SU2's unsteady laminar physics on the canonical pitching-airfoil case
(NACA 0012, Re ~ 40k, reduced frequency k ~ 0.55, ±10° about the quarter chord)
so we can trust the Tier-1 unsteady numerics before betting the campaign on them.
Per run: build the airfoil mesh → render the pitching cfg → run SU2 → reduce the
lift/drag history to ``C_L,max``, ``C_d,mean`` and the ``C_L``–α hysteresis-loop
area.

This is the SU2 half of the cross-solver gate; the PASS/FAIL comparison against
PyFR p=3 (Colab G4 GPU — Round-9 HIGH-11) and against published dynamic-stall
data is a separate step and needs reference numbers this module deliberately
does not invent. Everything here runs locally on CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fanopt.cfd.airfoil_mesh import (
    AIRFOIL_MARKER,
    FARFIELD_MARKER,
    AirfoilMeshParams,
    AirfoilMeshResult,
    build_airfoil_mesh,
)
from fanopt.cfd.airfoil_shapes import airfoil_polyline
from fanopt.cfd.configs import render_benchmark_cfg
from fanopt.cfd.parsers import parse_su2_unsteady_force_series
from fanopt.cfd.phase3 import find_su2, run_su2

__all__ = [
    "BenchmarkConfig",
    "BenchmarkMetrics",
    "pitch_angle_series",
    "benchmark_metrics",
    "prepare_benchmark_case",
    "run_benchmark",
]

GAMMA_AIR = 1.4
R_SPECIFIC_AIR_J_KGK = 287.05  # dry-air specific gas constant

MESH_NAME = "airfoil.su2"
CFG_NAME = "benchmark.cfg"

_CL_CANDIDATES = ("CL", "CLift", "C_L")
_CD_CANDIDATES = ("CD", "CDrag", "C_D")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Physical + numerical knobs for one oscillating-NACA-0012 run.

    The pitching angular frequency is derived from the reduced frequency
    ``k = ω c / (2 U∞)`` so the case is specified the way the literature reports
    it. ``motion_origin_frac`` places the pitch axis at that fraction of chord
    (0.25 = quarter chord).
    """

    mach_number: float = 0.05
    freestream_temperature_k: float = 300.0
    chord_m: float = 1.0
    reynolds_number: float = 40000.0
    reduced_frequency_k: float = 0.55
    pitch_amplitude_deg: float = 10.0
    motion_origin_frac: float = 0.25
    n_cycles: int = 5
    steps_per_cycle: int = 200
    inner_iter: int = 100
    n_airfoil_points: int = 120
    mesh_params: AirfoilMeshParams = field(default_factory=AirfoilMeshParams)

    def __post_init__(self) -> None:
        if not 0.0 < self.mach_number < 1.0:
            raise ValueError("mach_number must be in (0, 1)")
        if min(self.reynolds_number, self.reduced_frequency_k, self.chord_m) <= 0:
            raise ValueError("reynolds_number, reduced_frequency_k, chord_m must be > 0")
        if self.pitch_amplitude_deg <= 0:
            raise ValueError("pitch_amplitude_deg must be > 0")
        if not 0.0 <= self.motion_origin_frac <= 1.0:
            raise ValueError("motion_origin_frac must be in [0, 1]")
        if min(self.n_cycles, self.steps_per_cycle, self.inner_iter) < 1:
            raise ValueError("n_cycles, steps_per_cycle, inner_iter must be >= 1")

    @property
    def freestream_velocity_ms(self) -> float:
        speed_of_sound = math.sqrt(GAMMA_AIR * R_SPECIFIC_AIR_J_KGK * self.freestream_temperature_k)
        return self.mach_number * speed_of_sound

    @property
    def pitching_omega_rad_s(self) -> float:
        return 2.0 * self.freestream_velocity_ms * self.reduced_frequency_k / self.chord_m

    @property
    def pitch_amplitude_rad(self) -> float:
        return math.radians(self.pitch_amplitude_deg)

    @property
    def motion_origin_x_m(self) -> float:
        return self.motion_origin_frac * self.chord_m

    @property
    def period_s(self) -> float:
        return 2.0 * math.pi / self.pitching_omega_rad_s

    @property
    def time_step_s(self) -> float:
        return self.period_s / self.steps_per_cycle

    @property
    def max_time_s(self) -> float:
        return self.n_cycles * self.period_s

    @property
    def time_iter(self) -> int:
        return self.n_cycles * self.steps_per_cycle


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Reduced observables from one benchmark run (for comparison to reference data)."""

    c_l_max: float
    c_d_mean: float
    hysteresis_area: float  # ∮ C_L dα over the last full cycle (rad·C_L)
    alpha_at_cl_max_deg: float
    n_cycles_used: int


def pitch_angle_series(cfg: BenchmarkConfig) -> np.ndarray:
    """Prescribed pitch angle α(t) = A·sin(ωt) at each outer time step (radians).

    Matches SU2's ``RIGID_MOTION`` pitching (zero phase). Step ``k`` (0-based)
    sits at physical time ``(k+1)·dt`` — the first written history row is the end
    of the first sub-step, not t = 0.
    """
    k = np.arange(1, cfg.time_iter + 1, dtype=float)
    t = k * cfg.time_step_s
    return cfg.pitch_amplitude_rad * np.sin(cfg.pitching_omega_rad_s * t)


def _loop_area(x: np.ndarray, y: np.ndarray) -> float:
    """Signed shoelace area of the closed loop traced by ``(x, y)`` (abs value)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def benchmark_metrics(
    cl: np.ndarray,
    cd: np.ndarray,
    alpha_rad: np.ndarray,
    *,
    steps_per_cycle: int,
    n_discard: int = 1,
) -> BenchmarkMetrics:
    """Reduce lift/drag/α series to C_L,max, C_d,mean and the hysteresis-loop area.

    Discards the first ``n_discard`` cycles (startup transient). ``C_L,max`` and
    ``C_d,mean`` are taken over the remaining whole cycles; the hysteresis area is
    the C_L–α loop of the **last** full cycle. Series must be equal length and
    span at least ``n_discard + 1`` whole cycles.
    """
    cl = np.asarray(cl, dtype=float)
    cd = np.asarray(cd, dtype=float)
    alpha_rad = np.asarray(alpha_rad, dtype=float)
    if not (cl.shape == cd.shape == alpha_rad.shape):
        raise ValueError(f"cl/cd/alpha length mismatch: {cl.shape} {cd.shape} {alpha_rad.shape}")
    n_total_cycles = cl.size // steps_per_cycle
    if n_total_cycles <= n_discard:
        raise ValueError(
            f"need > {n_discard} whole cycles ({steps_per_cycle} steps each); "
            f"got {cl.size} steps = {n_total_cycles} cycles"
        )
    start = n_discard * steps_per_cycle
    end = n_total_cycles * steps_per_cycle
    cl_u, cd_u, a_u = cl[start:end], cd[start:end], alpha_rad[start:end]
    i_max = int(np.argmax(cl_u))
    last = slice(end - steps_per_cycle, end)
    return BenchmarkMetrics(
        c_l_max=float(cl_u[i_max]),
        c_d_mean=float(np.mean(cd_u)),
        hysteresis_area=_loop_area(alpha_rad[last], cl[last]),
        alpha_at_cl_max_deg=math.degrees(float(a_u[i_max])),
        n_cycles_used=n_total_cycles - n_discard,
    )


def prepare_benchmark_case(cfg: BenchmarkConfig, workdir: Path) -> AirfoilMeshResult:
    """Build the airfoil mesh and render the pitching cfg into ``workdir``."""
    workdir.mkdir(parents=True, exist_ok=True)
    poly = np.array(airfoil_polyline(cfg.n_airfoil_points, chord=cfg.chord_m), dtype=float)
    mesh = build_airfoil_mesh(
        poly,
        cfg.mesh_params,
        workdir / MESH_NAME,
        chord_m=cfg.chord_m,
        motion_origin_x_m=cfg.motion_origin_x_m,
    )
    cfg_text = render_benchmark_cfg(
        mesh_filename=MESH_NAME,
        marker_airfoil=AIRFOIL_MARKER,
        marker_farfield=FARFIELD_MARKER,
        reynolds_number=cfg.reynolds_number,
        reynolds_length=cfg.chord_m,
        pitching_omega_z=cfg.pitching_omega_rad_s,
        pitching_ampl_deg=cfg.pitch_amplitude_deg,
        motion_origin_x=cfg.motion_origin_x_m,
        time_step=cfg.time_step_s,
        max_time=cfg.max_time_s,
        time_iter=cfg.time_iter,
        mach_number=cfg.mach_number,
        freestream_temperature=cfg.freestream_temperature_k,
        inner_iter=cfg.inner_iter,
    )
    (workdir / CFG_NAME).write_text(cfg_text, encoding="utf-8")
    return mesh


def run_benchmark(
    cfg: BenchmarkConfig, workdir: Path, *, su2_bin: str | None = None, n_discard: int = 1
) -> BenchmarkMetrics:
    """Full benchmark: mesh + cfg + SU2 run + reduction. Runs SU2 as a subprocess."""
    su2 = su2_bin or find_su2()
    if su2 is None:
        raise RuntimeError("SU2_CFD not found (set $SU2_RUN or put SU2_CFD on PATH)")
    prepare_benchmark_case(cfg, workdir)
    hist = run_su2(CFG_NAME, workdir, su2)
    cl = parse_su2_unsteady_force_series(hist, force_candidates=_CL_CANDIDATES)
    cd = parse_su2_unsteady_force_series(hist, force_candidates=_CD_CANDIDATES)
    alpha = pitch_angle_series(cfg)
    n = min(cl.size, cd.size, alpha.size)
    return benchmark_metrics(
        cl[:n], cd[:n], alpha[:n], steps_per_cycle=cfg.steps_per_cycle, n_discard=n_discard
    )
