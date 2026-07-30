"""3D whole-fan aero objective for the ADR-0004 redo — the BO's per-design evaluation.

Replaces the wave-blind 2D-slice objectives (``bo/objective.py``, ``bo/blade_objective.py``).
Builds the real 3D solid (so it **sees the rib meridian**), runs unsteady SU2, extracts
per-blade CFz thrust, and scales to **whole-fan wind = per-blade × blade_count** (audit B3:
what a person feels is the whole fan, so ``blade_count`` becomes a real total-wind-vs-mass
trade instead of an aero-inert variable). The thrust ``metric`` — cycle-mean vs rectified-peak
— is a knob resolved by the N1 discriminator. Isolated-blade × N; periodic-cascade inter-blade
interaction is a deferred fidelity upgrade (ADR-0004).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fanopt.bo.blade_codec import decode
from fanopt.cfd.blade_aero_3d import evaluate_blade_aero_3d
from fanopt.cfd.phase5 import VerifyConfig
from fanopt.geometry.blade import (
    RIB_TIP_RADIUS_M,
    BladeParams,
    estimate_mass_kg,
    feasible,
    half_width_at,
    panel_radial_stations,
)
from fanopt.geometry.blade_cad import fold_collision_clear
from fanopt.geometry.schema import E_PETG_XY_PA
from fanopt.utils.ledger import design_hash

__all__ = [
    "THRUST_METRICS",
    "AERO_PRESSURE_PA",
    "blade_panel_deflection_m",
    "whole_fan_j_fan",
    "Blade3DObjective",
]

THRUST_METRICS = ("mean", "peak")

AERO_PRESSURE_PA: float = 10.0  # §9.2 distributed-pressure reference
_CLAMPED_PLATE_ALPHA: float = 0.0138  # max-deflection coefficient, clamped rectangular plate


def blade_panel_deflection_m(params: BladeParams) -> float:
    """Panel-floppiness proxy (m) — area-weighted mean clamped-plate deflection over the grid.

    Per node ``δ ∝ α · p · width(r)⁴ / (E · t(r,v)³)`` with ``width(r) = 2·half_width_at(r)`` the
    local **trapezoid** tangential width (NOT the retired ``r·Δθ`` pie-slice span), area-weighted
    (width × radial strip over the 22 cm :data:`RIB_TIP_RADIUS_M` blade) and averaged over the whole
    ``panel_thickness_m`` grid — so it responds to the entire thickness field and the width growth
    (wider outer nodes flex more), not just the single thinnest node. A soft Pareto proxy; real
    structure is Phase-2 TO's job. Live: called by :class:`Blade3DObjective` (ADR-0004 3D path).
    """
    stations = panel_radial_stations()
    dr = RIB_TIP_RADIUS_M / (len(stations) - 1)  # radial strip; BLADE_ROOT_RADIUS_M = 0
    total_defl = 0.0
    total_area = 0.0
    for r, th_row in zip(stations, params.panel_thickness_m):
        width = max(2.0 * half_width_at(r), 1e-6)  # trapezoid full tangential width at r
        area = width * dr  # radial-strip area weight
        for t in th_row:
            total_defl += area * _CLAMPED_PLATE_ALPHA * AERO_PRESSURE_PA * width**4 / (E_PETG_XY_PA * t**3)
            total_area += area
    return total_defl / total_area


def whole_fan_j_fan(per_blade_thrust: float, blade_count: int) -> float:
    """Whole-fan wind = ``per_blade_thrust × blade_count``.

    A person feels the whole deployed fan, so ``blade_count`` trades total wind against total
    mass (both scale ~linearly with N) — a real aero/mass lever, versus the prior objective
    where blade_count was aero-inert (the CFD meshes one isolated blade; audit B3). Inter-blade
    cascade interaction is a deferred fidelity upgrade (ADR-0004).
    """
    return per_blade_thrust * blade_count


@dataclass(frozen=True)
class Blade3DObjective:
    """Picklable ``vector → (whole_fan_J_fan, mass, deflection)`` for the aero-first redo.

    ``metric`` selects the per-blade thrust reduction: ``"mean"`` (cycle-mean CFz — the spec's
    directed momentum flux) or ``"peak"`` (rectified per-cycle peak). **The N1 discriminator
    (2026-07-26) selected ``"mean"``:** a dished scoop gives a clearly positive cycle-mean CFz
    (net wind) while a flat/symmetric blade gives ≈0 — physically correct and a clean
    discriminator, whereas ``peak`` is ~equal across designs (dominated by instantaneous stroke
    force, not net wind). Infeasible / diverged designs return ``NaN`` (the backbone sanitizes
    them to a dominated penalty).
    """

    out_dir: Path
    su2_bin: str | None = None
    diag_dir: Path | None = None  # persistent (e.g. Drive) for markers; default = out_dir
    metric: str = "mean"
    cfg: VerifyConfig = field(default_factory=VerifyConfig)

    def __post_init__(self) -> None:
        if self.metric not in THRUST_METRICS:
            raise ValueError(f"metric must be one of {THRUST_METRICS}, got {self.metric!r}")

    def __call__(self, vector: np.ndarray) -> tuple[float, float, float]:
        params = decode(vector)
        h = design_hash(params.to_dict())
        workdir = self.out_dir / "designs" / h
        diagdir = (self.diag_dir or self.out_dir) / "designs" / h
        nan = float("nan")
        try:
            if not feasible(params):
                diagdir.mkdir(parents=True, exist_ok=True)
                (diagdir / "INFEASIBLE.txt").write_text(
                    f"infeasible: {params.to_dict()}\n", encoding="utf-8"
                )
                return (nan, nan, nan)
            # Authoritative CAD fold backstop (~seconds) BEFORE the minutes-long CFD: even if a
            # corner slips past the analytic feasible() proxy, never evaluate an un-foldable design.
            if not fold_collision_clear(params):
                diagdir.mkdir(parents=True, exist_ok=True)
                (diagdir / "UNFOLDABLE.txt").write_text(
                    f"CAD fold gate: adjacent blades collide when folded: {params.to_dict()}\n",
                    encoding="utf-8",
                )
                return (nan, nan, nan)
            res = evaluate_blade_aero_3d(params, workdir, cfg=self.cfg, su2_bin=self.su2_bin)
            per_blade = res.j_fan_mean if self.metric == "mean" else res.j_fan_peak
            if diagdir != workdir and workdir.exists():
                shutil.copytree(workdir, diagdir, dirs_exist_ok=True)
            return (
                float(whole_fan_j_fan(per_blade, params.blade_count)),
                float(estimate_mass_kg(params)),
                float(blade_panel_deflection_m(params)),
            )
        except Exception as exc:  # fault isolation: a bad design is penalized, not fatal
            diagdir.mkdir(parents=True, exist_ok=True)
            (diagdir / "FAILED.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            return (nan, nan, nan)
