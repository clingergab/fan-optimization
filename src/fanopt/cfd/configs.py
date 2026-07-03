"""Jinja2 renderer for SU2 .cfg files (steady + unsteady).

Renders the templates under `configs/su2/*.cfg.j2` from a typed parameter
dict. Maintains the cross-tier vs tier-specific separation locked in plan
§9.4.1: MACH is **tier-specific** (0.0064 for steady tiers, 1e-9 for the
unsteady tier per the Round-9 HIGH-12 lock), so the cross-tier dict does
NOT carry MACH.

Public API:
    render_unsteady_cfg(params)     -> str  # Tier 1 (3D unsteady, MACH=1e-9)
    render_steady_cfg(params)       -> str  # Tier 0 (3D steady, MACH=0.0064)
    render_slice_steady_cfg(params) -> str  # Tier -1 (2D mid-radius slice)
    render_benchmark_cfg(params)    -> str  # wind-tunnel NACA 0012 (Phase 5 prep)
    render_thin_plate_2d_pitching_cfg(params) -> str  # Spike 0.6d.2 (H10 supplement)

    CROSS_TIER (dict)    - keys shared across all tiers
    TIER_SPECIFIC (dict) - {tier: {key: locked_value}} — MACH belongs here

Spec reference: docs/plan_R11.md §9.4.1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jinja2

from fanopt.geometry.schema import (
    F_WAVE_HZ,
    L_WRIST_TO_TIP_M,
    PITCHING_AMPL_VEC,
    PITCHING_OMEGA_VEC,
)

__all__ = [
    "CROSS_TIER",
    "TIER_SPECIFIC",
    "MACH_UNSTEADY",
    "MACH_STEADY",
    "REYNOLDS_NUMBER_GLOBAL",
    "CFD_TEMPLATES_DIR",
    "FREESTREAM_DIRECTION_2D_PRODUCTIVE",
    "FREESTREAM_DIRECTION_2D_RETURN",
    "render_unsteady_cfg",
    "render_steady_cfg",
    "render_slice_steady_cfg",
    "render_benchmark_cfg",
    "render_thin_plate_2d_pitching_cfg",
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


CFD_TEMPLATES_DIR: Path = Path(__file__).resolve().parents[3] / "configs" / "su2"


# 2D slice-frame C2 sign-locked freestream directions. The 3D
# FREESTREAM_PRODUCTIVE = (0, 0, -1) lock (blade frame) projects to the
# slice's chord-aligned 2D frame as (-1, 0): air flows -x past the
# stationary slice that is, in reality, being swept +x toward the user.
FREESTREAM_DIRECTION_2D_PRODUCTIVE: tuple[float, float] = (-1.0, 0.0)
FREESTREAM_DIRECTION_2D_RETURN: tuple[float, float] = (+1.0, 0.0)


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


def render_slice_steady_cfg(
    *,
    mesh_filename: str,
    marker_fan: str = "FAN",
    marker_farfield: str = "FARFIELD",
    marker_cascade: str | None = None,
    reynolds_number: float = REYNOLDS_NUMBER_GLOBAL,
    reynolds_length: float = L_WRIST_TO_TIP_M,
    freestream_direction: tuple[float, float] = FREESTREAM_DIRECTION_2D_PRODUCTIVE,
    cfl_number: float = 5.0,
) -> str:
    """Render `slice_steady.cfg.j2` (Tier -1 — 2D mid-radius slice).

    Tier -1 is the Phase 4 architecture-bandit screening tier. The 2D
    cross-section at r = r_mid uses the same MACH = 0.0064 lock as
    Tier 0 (steady tiers, per the Round-9 HIGH-12 tier-specific MACH
    placement — MACH lives in TIER_SPECIFIC[-1] and TIER_SPECIFIC[0],
    NOT in CROSS_TIER).

    `freestream_direction` is a 2D vector in the slice's chord-aligned
    frame. Defaults to ``FREESTREAM_DIRECTION_2D_PRODUCTIVE = (-1, 0)``
    (C2 productive-stroke convention). Pass
    ``FREESTREAM_DIRECTION_2D_RETURN = (+1, 0)`` for the return-stroke
    half of the two-eval delta.
    """
    if len(freestream_direction) != 2:
        raise TemplateRenderError(
            f"freestream_direction must be a 2-vector for the 2D slice cfg; "
            f"got {len(freestream_direction)}-vector {freestream_direction}"
        )
    env = _env()
    try:
        tpl = env.get_template("slice_steady.cfg.j2")
    except jinja2.TemplateNotFound as e:
        raise TemplateRenderError(f"template not found: {e}") from e
    try:
        return tpl.render(
            mesh_filename=mesh_filename,
            marker_fan=marker_fan,
            marker_farfield=marker_farfield,
            marker_cascade=marker_cascade,
            reynolds_number=reynolds_number,
            reynolds_length=reynolds_length,
            freestream_direction_x=freestream_direction[0],
            freestream_direction_y=freestream_direction[1],
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
    mach_number: float = 0.05,
    freestream_temperature: float = 300.0,
    freestream_pressure: float = 101325.0,
    inner_iter: int = 100,
    cfl_number: float = 1.0,
) -> str:
    """Render the wind-tunnel-frame NACA 0012 oscillating-airfoil template.

    Phase-5 prep deliverable. Renders a conventional wind-tunnel-frame
    cfg (MACH > 0, freestream ON, airfoil pitches in place) suitable
    for validation against published low-Re pitching references. NOT
    a Tier-1 lock-equivalence template — the MACH = 1e-9 unsteady lock
    (Round-9 HIGH-12 / C12) intentionally does NOT apply here.

    Default ``mach_number = 0.05`` is the conventional "low enough to
    be effectively incompressible, high enough that the compressible
    solver stays well conditioned with LOW_MACH_PREC = YES" choice.
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
            mach_number=mach_number,
            freestream_temperature=freestream_temperature,
            freestream_pressure=freestream_pressure,
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


def render_thin_plate_2d_pitching_cfg(
    *,
    mesh_filename: str,
    marker_plate: str,
    marker_farfield: str,
    pitching_omega_z: float,
    pitching_ampl_z: float,
    motion_origin_x: float,
    time_step: float,
    max_time: float,
    time_iter: int,
    reynolds_number: float = 40000.0,
    reynolds_length: float = 1.0,
    inner_iter: int = 100,
    cfl_number: float = 1.0,
) -> str:
    """Render the Spike 0.6d.2 2D thin-plate pitching template.

    Mirrors the production Tier-1 unsteady numerics (Round-9 HIGH-12 / C12:
    ``MACH=1e-9`` + ``REF_DIMENSIONALIZATION=FREESTREAM_PRESS_EQ_ONE`` +
    ``LOW_MACH_PREC=YES`` + ``DUAL_TIME_STEPPING-2ND_ORDER`` +
    ``GRID_MOVEMENT=RIGID_MOTION``) on a 2D thin-plate geometry where the
    Sedov/Newman closed-form added-mass moment is known. The H10 supplement
    spike (0.6d.2) compares SU2's inviscid-phase pitching moment against
    that closed form within ±15%.

    The 2D thin-plate cfg intentionally does NOT take a ``mach_number``
    parameter — it must use the production unsteady lock to be a faithful
    test of the production numerics. The C11 literal y-sign-lock applies to
    the 3D production cfg (``fan3d_unsteady.cfg.j2``); on a 2D x-y mesh the
    physically meaningful pitch axis is z, so the analog lock here is
    ``pitching_omega_z`` must be negative on the productive stroke.
    """
    if pitching_omega_z > 0:
        raise TemplateRenderError(
            "pitching_omega_z must be ≤ 0 per the C11 analog "
            "(2D x-y mesh pitch axis is z; negative on productive stroke)"
        )
    env = _env()
    try:
        tpl = env.get_template("thin_plate_2d_pitching.cfg.j2")
    except jinja2.TemplateNotFound as e:
        raise TemplateRenderError(f"template not found: {e}") from e
    try:
        return tpl.render(
            mesh_filename=mesh_filename,
            marker_plate=marker_plate,
            marker_farfield=marker_farfield,
            reynolds_number=reynolds_number,
            reynolds_length=reynolds_length,
            pitching_omega_z=pitching_omega_z,
            pitching_ampl_z=pitching_ampl_z,
            motion_origin_x=motion_origin_x,
            time_step=time_step,
            max_time=max_time,
            time_iter=time_iter,
            inner_iter=inner_iter,
            cfl_number=cfl_number,
        )
    except jinja2.UndefinedError as e:
        raise TemplateRenderError(f"missing template variable: {e}") from e
