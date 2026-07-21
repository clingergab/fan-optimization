"""Picklable BO objective for the redesigned blade: vector → (J_fan, mass, deflection).

The aero-first V1 objective the campaign optimizes. Decodes a design vector with the new
:mod:`~fanopt.bo.blade_codec`, and returns the three raw objectives the multi-objective
BO consumes (signs `(+1, -1, -1)` — **maximize** wind, **minimize** mass + deflection):

- ``J_fan`` — net momentum flux from the 2D-slice CFD (:func:`evaluate_blade_aero`, SU2).
- ``mass`` — the whole-fan mass estimate (cheap analytic proxy).
- ``deflection`` — panel tip deflection under aero pressure (cheap clamped-plate analytic).

Infeasible geometry (won't fold / over-mass / panel pokes past the rib) is penalized with
a non-finite objective **without** spending SU2 on it. Any hard failure (degenerate mesh,
SU2 divergence) is caught and penalized too — a bad design must not kill a long campaign.
Picklable (a frozen dataclass of ``Path``/``str``/scalars) so a ``ProcessPoolExecutor``
can ship it to parallel workers.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fanopt.bo.blade_codec import decode
from fanopt.cfd.blade_aero import evaluate_blade_aero
from fanopt.geometry.blade import (
    BladeParams,
    estimate_mass_kg,
    feasible,
)
from fanopt.geometry.schema import (
    E_PETG_XY_PA,
    HUB_RADIUS_M,
    INTER_BLADE_ANGLE_RAD,
    L_RIB_M,
)
from fanopt.utils.ledger import design_hash

__all__ = [
    "OBJECTIVE_NAMES",
    "AERO_PRESSURE_PA",
    "blade_panel_deflection_m",
    "BladeObjective",
]

OBJECTIVE_NAMES: tuple[str, str, str] = ("j_fan", "mass_kg", "deflection_m")
AERO_PRESSURE_PA: float = 10.0  # §9.2 distributed-pressure reference
_CLAMPED_PLATE_ALPHA: float = 0.0138  # max-deflection coefficient, clamped rectangular plate


def blade_panel_deflection_m(params: BladeParams) -> float:
    """Analytic panel tip deflection (m) — a clamped plate under uniform aero pressure.

    ``δ_max = α · p · b⁴ / (E · t³)`` with ``b`` the mid-radius tangential span and ``t``
    the nominal panel thickness. A cheap structural proxy (the panel is rib-supported and
    stiff, so this is small); it gives the optimizer a "thinner → more deflection" gradient
    without an FE solve. Full FE / shell fidelity is a V1.5 refinement.
    """
    r_mid = HUB_RADIUS_M + 0.5 * L_RIB_M
    span = max(r_mid * INTER_BLADE_ANGLE_RAD, 1e-6)
    t = min(v for row in params.panel_thickness_m for v in row)  # thinnest node governs
    return _CLAMPED_PLATE_ALPHA * AERO_PRESSURE_PA * span**4 / (E_PETG_XY_PA * t**3)


@dataclass(frozen=True)
class BladeObjective:
    """Picklable ``vector → (J_fan, mass, deflection)`` for the aero-first campaign.

    Each design gets a stable per-hash workdir under ``out_dir/designs`` so a resumed
    campaign reuses prior CFD output.
    """

    out_dir: Path
    su2_bin: str | None = None
    diag_dir: Path | None = None  # persistent (e.g. Drive) for failure markers; default = out_dir
    radial_u: float = 0.5
    n_panels: int = 5
    n_samples: int = 28
    n_cycles: int = 5
    inner_iter: int = 50
    steps_per_cycle: int = 200

    def __call__(self, vector: np.ndarray) -> tuple[float, float, float]:
        params = decode(vector)
        h = design_hash(params.to_dict())
        workdir = self.out_dir / "designs" / h  # SU2 scratch (may be ephemeral)
        # Failure markers go to a persistent dir so a killed session stays debuggable.
        diagdir = (self.diag_dir or self.out_dir) / "designs" / h
        nan = float("nan")
        try:
            if not feasible(params):
                # Won't fold / over-mass / panel pokes past the rib: penalize, skip SU2.
                diagdir.mkdir(parents=True, exist_ok=True)
                (diagdir / "INFEASIBLE.txt").write_text(
                    f"infeasible: {params.to_dict()}\n", encoding="utf-8"
                )
                return (nan, nan, nan)
            res = evaluate_blade_aero(
                params,
                workdir,
                su2_bin=self.su2_bin,
                radial_u=self.radial_u,
                n_panels=self.n_panels,
                n_samples=self.n_samples,
                n_cycles=self.n_cycles,
                inner_iter=self.inner_iter,
                steps_per_cycle=self.steps_per_cycle,
            )
            # Persist the (small) CFD output — mesh, cfgs, history.csv, final field — to the
            # persistent dir so a killed session keeps every result, not just the ledger.
            if diagdir != workdir and workdir.exists():
                shutil.copytree(workdir, diagdir, dirs_exist_ok=True)
            return (
                float(res.j_fan),
                float(estimate_mass_kg(params)),
                float(blade_panel_deflection_m(params)),
            )
        except Exception as exc:  # fault isolation: a bad design is penalized, not fatal
            diagdir.mkdir(parents=True, exist_ok=True)
            (diagdir / "FAILED.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            return (nan, nan, nan)
