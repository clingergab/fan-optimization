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
from fanopt.bo.blade_objective import blade_panel_deflection_m
from fanopt.cfd.blade_aero_3d import evaluate_blade_aero_3d
from fanopt.cfd.phase5 import VerifyConfig
from fanopt.geometry.blade import estimate_mass_kg, feasible
from fanopt.utils.ledger import design_hash

__all__ = ["THRUST_METRICS", "whole_fan_j_fan", "Blade3DObjective"]

THRUST_METRICS = ("mean", "peak")


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
