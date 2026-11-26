"""As-printed blade mass, inertia about the wrist and the inertial effort of waving, under one print recipe.

The campaign ledger's ``mass_kg`` is a solid blade at whatever thickness the search tried; the fan is
printed thinner, with walls and a sparse core. Here every design is put on one recipe: its thickness
field is scaled about the mid-surface to a common area-mean (true) wall, and each piece of the sheet is
priced with a wall + core areal density fitted to slicer weights (:func:`fit_print_mass_model`).

The fan is the blade repeated ``blade_count`` times about the pin (z axis), spread evenly about +x; the
wrist axis is +y through ``(−wrist_offset_m, 0, 0)``. For the locked sinusoidal stroke
``θ = θ_max sin ωt`` the inertial torque does ``I θ_max² ω²`` of positive work per cycle (speeding the
fan up twice) and takes the same back braking it: push-only = ``I θ_max² ω²``, push + brake = twice that.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares

from fanopt.cfd.correlation import kendall_tau, spearman_rho
from fanopt.geometry.blade import BLADE_ROOT_RADIUS_M, RIB_TIP_RADIUS_M, BladeParams, half_width_at
from fanopt.geometry.blade_cad import surface_top_bottom_m

__all__ = [
    "BladeSheet",
    "PrintMassModel",
    "blade_sheet",
    "mean_thickness_m",
    "scaled_to_mean_thickness",
    "areal_density_kg_m2",
    "blade_mass_kg",
    "fit_print_mass_model",
    "fan_inertia_wrist_kgm2",
    "fan_com_wrist_m",
    "inertial_work_per_cycle_j",
    "design_effort",
    "rerank_summary",
]


@dataclass(frozen=True)
class BladeSheet:
    """One blade as cells: planform centre ``(x, y)``, mid-surface ``z``, true wall, surface area."""

    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    thickness_m: np.ndarray
    area_m2: np.ndarray


@dataclass(frozen=True)
class PrintMassModel:
    """Areal density ``ρ · (min(t, wall) + core · max(t − wall, 0))`` plus a per-blade ``base_kg``.

    ``wall_m`` is the solid shell through the wall (both faces together, incl. any gap fill);
    ``core_fraction`` the effective fill of the rest; ``base_kg`` what the sheet leaves out (hub, boss,
    edge walls) and sits at the pin.
    """

    wall_m: float
    core_fraction: float
    base_kg: float
    rho_kg_m3: float


def blade_sheet(params: BladeParams, *, n_radial: int = 80, n_tangential: int = 24) -> BladeSheet:
    """Sample the blade's height field on a planform grid; wall and area are measured on the surface."""
    dr = (RIB_TIP_RADIUS_M - BLADE_ROOT_RADIUS_M) / n_radial
    r = BLADE_ROOT_RADIUS_M + dr * (np.arange(n_radial) + 0.5)
    v = -1.0 + (2.0 / n_tangential) * (np.arange(n_tangential) + 0.5)
    w = np.array([half_width_at(x) for x in r])
    mid = np.empty((n_radial, n_tangential))
    t_vert = np.empty_like(mid)
    for i, (x, wi) in enumerate(zip(r, w, strict=True)):
        for j, vj in enumerate(v):
            top, bot = surface_top_bottom_m(params, float(x), float(vj * wi))  # type: ignore[misc]
            mid[i, j] = (top + bot) / 2.0
            t_vert[i, j] = top - bot
    dw = np.gradient(w, r)
    dm_dr = np.gradient(mid, r, axis=0)
    dm_dv = np.gradient(mid, v, axis=1)
    # y = v·w(r): convert the (r, v) gradient to the planform (x, y) slope.
    dm_dx = dm_dr - dm_dv * (v[None, :] * dw[:, None] / w[:, None])
    dm_dy = dm_dv / w[:, None]
    cos = 1.0 / np.sqrt(1.0 + dm_dx**2 + dm_dy**2)
    # The wall lies along the surface, which is longer than its planform by 1/cos on a slope.
    area = (dr * 2.0 * w / n_tangential)[:, None] / cos
    return BladeSheet(
        x_m=np.repeat(r, n_tangential),
        y_m=(w[:, None] * v[None, :]).ravel(),
        z_m=mid.ravel(),
        thickness_m=(t_vert * cos).ravel(),
        area_m2=area.ravel(),
    )


def mean_thickness_m(sheet: BladeSheet) -> float:
    """Area-weighted mean true wall."""
    return float((sheet.thickness_m * sheet.area_m2).sum() / sheet.area_m2.sum())


def scaled_to_mean_thickness(sheet: BladeSheet, target_m: float) -> BladeSheet:
    """``sheet`` thinned or thickened about its mid-surface to an area-mean wall of ``target_m``."""
    if target_m <= 0.0:
        raise ValueError(f"target thickness must be positive, got {target_m}")
    return replace(sheet, thickness_m=sheet.thickness_m * (target_m / mean_thickness_m(sheet)))


def areal_density_kg_m2(thickness_m: np.ndarray, model: PrintMassModel) -> np.ndarray:
    """Printed mass per unit area of a wall of ``thickness_m``."""
    t = np.asarray(thickness_m, dtype=float)
    solid = np.minimum(t, model.wall_m) + model.core_fraction * np.maximum(t - model.wall_m, 0.0)
    return model.rho_kg_m3 * solid


def _cell_mass_kg(sheet: BladeSheet, model: PrintMassModel) -> np.ndarray:
    return areal_density_kg_m2(sheet.thickness_m, model) * sheet.area_m2


def blade_mass_kg(sheet: BladeSheet, model: PrintMassModel) -> float:
    """One printed blade."""
    return model.base_kg + float(_cell_mass_kg(sheet, model).sum())


def fit_print_mass_model(
    sheets: Sequence[BladeSheet], masses_kg: Sequence[float], *, rho_kg_m3: float
) -> PrintMassModel:
    """Least-squares fit of wall, core fraction and base mass to weighed (or sliced) blades."""
    if len(sheets) != len(masses_kg):
        raise ValueError(f"{len(sheets)} sheets but {len(masses_kg)} masses")
    if len(sheets) < 3:
        raise ValueError(f"need at least 3 blades to fit 3 parameters, got {len(sheets)}")
    target = np.asarray(masses_kg, dtype=float)

    def residual(p: np.ndarray) -> np.ndarray:
        m = PrintMassModel(wall_m=p[0], core_fraction=p[1], base_kg=p[2], rho_kg_m3=rho_kg_m3)
        return np.array([blade_mass_kg(s, m) for s in sheets]) - target

    t_mean = float(np.mean([mean_thickness_m(s) for s in sheets]))
    fit = least_squares(
        residual,
        x0=np.array([0.5 * t_mean, 0.15, 0.0]),
        bounds=(np.array([0.0, 0.0, 0.0]), np.array([10.0 * t_mean, 1.0, float(target.max())])),
    )
    return PrintMassModel(
        wall_m=float(fit.x[0]),
        core_fraction=float(fit.x[1]),
        base_kg=float(fit.x[2]),
        rho_kg_m3=rho_kg_m3,
    )


def _blade_angles_rad(blade_count: int, pitch_rad: float) -> np.ndarray:
    return (np.arange(blade_count) - (blade_count - 1) / 2.0) * pitch_rad


def _fan_points(
    sheet: BladeSheet, blade_count: int, pitch_rad: float, wrist_offset_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Lever along the wrist's x (``x + offset``) and z for every cell of every blade, shape (n, cells)."""
    phi = _blade_angles_rad(blade_count, pitch_rad)[:, None]
    lever = sheet.x_m[None, :] * np.cos(phi) - sheet.y_m[None, :] * np.sin(phi) + wrist_offset_m
    z = np.broadcast_to(sheet.z_m[None, :], lever.shape)
    return lever, z


def fan_inertia_wrist_kgm2(
    sheet: BladeSheet,
    model: PrintMassModel,
    *,
    blade_count: int,
    pitch_rad: float,
    wrist_offset_m: float,
) -> float:
    """Moment of inertia of the open fan about the wrist axis; each blade's base mass sits at the pin."""
    lever, z = _fan_points(sheet, blade_count, pitch_rad, wrist_offset_m)
    m = _cell_mass_kg(sheet, model)[None, :]
    return float((m * (lever**2 + z**2)).sum() + blade_count * model.base_kg * wrist_offset_m**2)


def fan_com_wrist_m(
    sheet: BladeSheet,
    model: PrintMassModel,
    *,
    blade_count: int,
    pitch_rad: float,
    wrist_offset_m: float,
) -> float:
    """Distance of the open fan's centre of mass from the wrist, along the handle (+x)."""
    lever, _ = _fan_points(sheet, blade_count, pitch_rad, wrist_offset_m)
    m = _cell_mass_kg(sheet, model)[None, :]
    total = float(m.sum()) * blade_count + blade_count * model.base_kg
    return (float((m * lever).sum()) + blade_count * model.base_kg * wrist_offset_m) / total


def inertial_work_per_cycle_j(
    inertia_kgm2: float, theta_max_rad: float, omega_rad_s: float
) -> tuple[float, float]:
    """``(push_only, push_and_brake)`` work per sinusoidal cycle from inertia alone."""
    push = inertia_kgm2 * theta_max_rad**2 * omega_rad_s**2
    return push, 2.0 * push


def design_effort(
    params: BladeParams,
    model: PrintMassModel,
    *,
    recipe_thickness_m: float,
    blade_count: int,
    pitch_rad: float,
    wrist_offset_m: float,
    theta_max_rad: float,
    omega_rad_s: float,
) -> dict[str, float]:
    """Mass, inertia, centre of mass and inertial work per cycle of one design on the print recipe."""
    sheet = scaled_to_mean_thickness(blade_sheet(params), recipe_thickness_m)
    fan = {"blade_count": blade_count, "pitch_rad": pitch_rad, "wrist_offset_m": wrist_offset_m}
    inertia = fan_inertia_wrist_kgm2(sheet, model, **fan)
    push, push_brake = inertial_work_per_cycle_j(inertia, theta_max_rad, omega_rad_s)
    blade = blade_mass_kg(sheet, model)
    return {
        "blade_mass_kg": blade,
        "fan_mass_kg": blade * blade_count,
        "inertia_wrist_kgm2": inertia,
        "com_wrist_m": fan_com_wrist_m(sheet, model, **fan),
        "work_push_j": push,
        "work_push_brake_j": push_brake,
    }


def rerank_summary(
    score: np.ndarray, effort: np.ndarray, *, top_k: Sequence[int] = (10, 50)
) -> dict[str, object]:
    """How far ranking by ``score / effort`` moves from ranking by ``score`` alone.

    Spearman and Kendall between the two orderings, and the share of each top-k kept.
    """
    score = np.asarray(score, dtype=float)
    ratio = score / np.asarray(effort, dtype=float)
    by_score, by_ratio = np.argsort(-score), np.argsort(-ratio)
    return {
        "n": int(score.size),
        "spearman": spearman_rho(score, ratio),
        "kendall": kendall_tau(score, ratio),
        "top_k_kept": {
            str(k): len(set(by_score[:k]) & set(by_ratio[:k])) / float(k) for k in top_k
        },
        "best_by_score": int(by_score[0]),
        "best_by_ratio": int(by_ratio[0]),
    }
