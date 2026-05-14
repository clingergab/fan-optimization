"""Jinja2 renderer for SU2 .cfg files (steady + unsteady).

Renders the templates under `configs/su2/*.cfg.j2` from a typed parameter
dict. Maintains the cross-tier vs tier-specific separation locked in plan
§9.4.1: MACH is **tier-specific** (0.0064 for steady tiers, 1e-9 for the
unsteady tier per the Round-9 HIGH-12 lock), so the cross-tier dict does
NOT carry MACH.

Public API:
    render_unsteady_cfg(params)  -> str  # Tier 1 (3D unsteady)
    render_steady_cfg(params)    -> str  # Tier 0 (3D steady)
    render_slice_steady_cfg(params) -> str  # Tier -1 (2D slice)
    render_benchmark_cfg(params) -> str  # Spike 0.6c.2 NACA 0012

    CROSS_TIER (dict)   - keys shared across all tiers
    TIER_SPECIFIC (dict) - {tier: {key: locked_value}} — MACH belongs here

Spec reference: docs/plan_R11.md §9.4.1.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import jinja2

from fanopt.geometry.schema import (
    F_WAVE_HZ,
    L_WRIST_TO_TIP_M,
    OMEGA_SHM_RAD_PER_S,
    PITCHING_AMPL_VEC,
    PITCHING_OMEGA_VEC,
    THETA_MAX_RAD,
)

__all__ = [
    "CROSS_TIER",
    "TIER_SPECIFIC",
    "MACH_UNSTEADY",
    "MACH_STEADY",
    "REYNOLDS_NUMBER_GLOBAL",
    "CFD_TEMPLATES_DIR",
    "render_unsteady_cfg",
    "render_steady_cfg",
    "render_benchmark_cfg",
    "TemplateRenderError",
]


# ---------------------------------------------------------------------------
# Locked tier-specific constants — plan §9.4.1
# ---------------------------------------------------------------------------

MACH_UNSTEADY: float = 1e-9
"""Round-9 HIGH-12 / C12 lock — unsteady cfg uses near-zero Mach."""

MACH_STEADY: float = 0.0064
"""Steady tiers (-1 / 0) use V_tip as freestream → Mach ≈ 2.20 m/s / 340 m/s."""

REYNOLDS_NUMBER_GLOBAL: float = 37000.0
"""Re_global at L = L_wrist_to_tip per §3.2.3 H8 symbol table."""


CROSS_TIER: dict[str, Any] = {
    "solver": "NAVIER_STOKES",
    "kind_turb_model": "NONE",
    "math_problem": "DIRECT",
    "reynolds_number": REYNOLDS_NUMBER_GLOBAL,
    "reynolds_length": L_WRIST_TO_TIP_M,
    "freestream_temperature": 300.0,
    "freestream_pressure": 101325.0,
}
"""Cross-tier locks. Critically does NOT include MACH (tier-specific) — per
the Round-9 HIGH-12 retired-phrase guard, anyone adding MACH here trips it."""


TIER_SPECIFIC: dict[int, dict[str, Any]] = {
    -1: {  # 2D steady slice
        "mach_number": MACH_STEADY,
        "time_domain": "NO",
    },
    0: {  # 3D steady
        "mach_number": MACH_STEADY,
        "time_domain": "NO",
    },
    1: {  # 3D unsteady — Round-9 HIGH-12 / C12 lock
        "mach_number": MACH_UNSTEADY,
        "time_domain": "YES",
        "time_marching": "DUAL_TIME_STEPPING-2ND_ORDER",
        "low_mach_prec": "YES",
        "grid_movement": "RIGID_MOTION",
        "pitching_omega_y": PITCHING_OMEGA_VEC[1],  # NEGATIVE — C11 sign lock
        "pitching_ampl_y": PITCHING_AMPL_VEC[1],  # POSITIVE 0.6981 rad
    },
}
"""Tier-specific locks. MACH lives here, NOT in CROSS_TIER."""


CFD_TEMPLATES_DIR: Path = (
    Path(__file__).resolve().parents[3] / "configs" / "su2"
)


class TemplateRenderError(Exception):
    """Raised when a template can't be rendered or a required variable is missing."""


def _env() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(CFD_TEMPLATES_DIR),
        autoescape=False,
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
    )


def _unsteady_defaults(*, n_cycles: int = 5, inner_iter: int = 100) -> dict[str, Any]:
    """Compute the Tier-1 time-stepping defaults from locked kinematics.

    Plan §0 + §3.2.3 lock:
      - dt = T_cycle / 200 = 2.5 ms
      - max_time = n_cycles * T_cycle
      - time_iter = n_cycles * 200
    """
    T = 1.0 / F_WAVE_HZ
    dt = T / 200.0
    return {
        "time_step": dt,
        "max_time": n_cycles * T,
        "time_iter": n_cycles * 200,
        "inner_iter": inner_iter,
    }


def render_unsteady_cfg(
    *,
    mesh_filename: str,
    marker_fan: str = "FAN",
    marker_farfield: str = "FARFIELD",
    reynolds_number: float = REYNOLDS_NUMBER_GLOBAL,
    reynolds_length: float = L_WRIST_TO_TIP_M,
    pitching_omega_y: float | None = None,
    pitching_ampl_y: float | None = None,
    motion_origin_x: float = 0.0,
    motion_origin_y: float = 0.0,
    motion_origin_z: float = 0.0,
    n_cycles: int = 5,
    inner_iter: int = 100,
    cfl_number: float = 1.0,
) -> str:
    """Render `fan3d_unsteady.cfg.j2` with the locked Tier-1 numerics.

    Default pitching_omega_y / pitching_ampl_y come from the schema's C11
    lock (negative-y omega, positive-y amplitude). Overriding is allowed
    only for benchmarks; production runs MUST use the default.

    The template ships the Round-9 HIGH-12 fallback path
    (`REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE`) because SU2 v8.0.1
    rejects the primary path (`FREESTREAM_OPTION = FREESTREAM_VELOCITY`)
    at parse time. The two paths are numerically equivalent for
    compressible-zero-flow runs.
    """
    if pitching_omega_y is None:
        pitching_omega_y = PITCHING_OMEGA_VEC[1]
    if pitching_ampl_y is None:
        pitching_ampl_y = PITCHING_AMPL_VEC[1]

    # Defensive: enforce Round-9 HIGH-12 sign on production omega.
    # Operators benchmarking other reference cases can override but a
    # positive y-omega here would silently flip the productive stroke.
    if pitching_omega_y > 0:
        raise TemplateRenderError(
            f"pitching_omega_y must be ≤ 0 per C11 / Round-9 HIGH-12 sign "
            f"lock (right-hand-rule on productive stroke). Got "
            f"{pitching_omega_y}. The locked value is "
            f"{PITCHING_OMEGA_VEC[1]:.4f} rad/s."
        )

    timestep_defaults = _unsteady_defaults(n_cycles=n_cycles, inner_iter=inner_iter)

    env = _env()
    try:
        tpl = env.get_template("fan3d_unsteady.cfg.j2")
    except jinja2.TemplateNotFound as e:
        raise TemplateRenderError(f"template not found: {e}") from e

    try:
        return tpl.render(
            mesh_filename=mesh_filename,
            marker_fan=marker_fan,
            marker_farfield=marker_farfield,
            reynolds_number=reynolds_number,
            reynolds_length=reynolds_length,
            pitching_omega_y=pitching_omega_y,
            pitching_ampl_y=pitching_ampl_y,
            motion_origin_x=motion_origin_x,
            motion_origin_y=motion_origin_y,
            motion_origin_z=motion_origin_z,
            cfl_number=cfl_number,
            **timestep_defaults,
        )
    except jinja2.UndefinedError as e:
        raise TemplateRenderError(f"missing template variable: {e}") from e


def render_steady_cfg(
    *,
    mesh_filename: str,
    marker_fan: str = "FAN",
    marker_farfield: str = "FARFIELD",
    reynolds_number: float = REYNOLDS_NUMBER_GLOBAL,
    reynolds_length: float = L_WRIST_TO_TIP_M,
    freestream_direction: tuple[float, float, float] = (0.0, 0.0, -1.0),
    cfl_number: float = 5.0,
) -> str:
    """Render `fan3d_steady.cfg.j2` (Tier 0 — 3D steady).

    Steady tiers use V_tip as freestream, MACH = 0.0064. `freestream_direction`
    defaults to the C2 PRODUCTIVE direction (-z).
    """
    env = _env()
    try:
        tpl = env.get_template("fan3d_steady.cfg.j2")
    except jinja2.TemplateNotFound as e:
        raise TemplateRenderError(f"template not found: {e}") from e
    try:
        return tpl.render(
            mesh_filename=mesh_filename,
            marker_fan=marker_fan,
            marker_farfield=marker_farfield,
            reynolds_number=reynolds_number,
            reynolds_length=reynolds_length,
            freestream_direction_x=freestream_direction[0],
            freestream_direction_y=freestream_direction[1],
            freestream_direction_z=freestream_direction[2],
            cfl_number=cfl_number,
            mach_number=MACH_STEADY,
        )
    except jinja2.UndefinedError as e:
        raise TemplateRenderError(f"missing template variable: {e}") from e


def render_benchmark_cfg(
    *,
    mesh_filename: str,
    marker_airfoil: str,
    marker_farfield: str,
    reynolds_number: float,
    reynolds_length: float,
    pitching_omega_y: float,
    pitching_ampl_y: float,
    motion_origin_x: float,
    time_step: float,
    max_time: float,
    time_iter: int,
    inner_iter: int = 100,
    cfl_number: float = 1.0,
) -> str:
    """Render the Spike 0.6c.2 NACA 0012 benchmark template.

    Uses the same Round-9 HIGH-12 fallback freestream syntax as the
    Tier-1 production cfg (REF_DIMENSIONALIZATION + reference state),
    since SU2 v8.0.1 rejects the primary FREESTREAM_VELOCITY directive.
    """
    env = _env()
    try:
        tpl = env.get_template("oscillating_airfoil_benchmark.cfg.j2")
    except jinja2.TemplateNotFound as e:
        raise TemplateRenderError(f"template not found: {e}") from e
    try:
        return tpl.render(
            mesh_filename=mesh_filename,
            marker_airfoil=marker_airfoil,
            marker_farfield=marker_farfield,
            reynolds_number=reynolds_number,
            reynolds_length=reynolds_length,
            pitching_omega_y=pitching_omega_y,
            pitching_ampl_y=pitching_ampl_y,
            motion_origin_x=motion_origin_x,
            time_step=time_step,
            max_time=max_time,
            time_iter=time_iter,
            inner_iter=inner_iter,
            cfl_number=cfl_number,
        )
    except jinja2.UndefinedError as e:
        raise TemplateRenderError(f"missing template variable: {e}") from e
