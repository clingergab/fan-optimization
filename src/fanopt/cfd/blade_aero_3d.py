"""3D unsteady aero evaluation for the BO **objective** (whole-fan wind).

The redo's per-design evaluation, reusing the corrected 3D verify pipeline
(:func:`fanopt.cfd.blade_verify.prepare_blade_verification_case` → SU2 → CFz thrust). Two
things distinguish it from the retired 2D slice and from ``phase5.extract_j_fan_3d``:

- it builds the real 3D solid, so it **sees the rib meridian wave** (the 2D slice did not);
- it returns BOTH the cycle-mean and the rectified-peak CFz thrust, so the BO can select the
  metric empirically (the "N1" question — a rigid blade's symmetric pitch can cancel the mean
  to ≈0 while the peak stays large).

This is the per-blade thrust; whole-fan scaling (× blade_count) lives in the objective. The
force axis is CFz (thrust) — the 2D-tuned CFx default is wrong in 3D (audit B1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fanopt.cfd.blade_verify import prepare_blade_verification_case
from fanopt.cfd.j_fan import STEPS_PER_CYCLE, reduce_cycles
from fanopt.cfd.parsers import _THRUST_Z_CANDIDATES, parse_su2_unsteady_force_series
from fanopt.cfd.phase3 import find_su2, run_su2
from fanopt.cfd.phase5 import CFG_NAME, VerifyConfig
from fanopt.geometry.blade import BladeParams

__all__ = ["Blade3DAeroResult", "evaluate_blade_aero_3d"]


@dataclass(frozen=True)
class Blade3DAeroResult:
    """Per-blade 3D thrust, both reductions (the BO picks one via the ``metric`` knob)."""

    j_fan_mean: float  # cycle-mean CFz thrust (spec's directed momentum flux)
    j_fan_peak: float  # rectified per-cycle peak CFz (robust when the mean cancels)
    n_nodes: float


def evaluate_blade_aero_3d(
    params: BladeParams,
    workdir: str | Path,
    *,
    cfg: VerifyConfig | None = None,
    su2_bin: str | None = None,
) -> Blade3DAeroResult:
    """Build → 3D-mesh → unsteady SU2 → per-blade CFz thrust (cycle-mean and peak).

    Raises if the run terminates short of ``cfg.n_cycles × STEPS_PER_CYCLE`` steps (diverged /
    killed) rather than reshaping a misaligned series into a laundered value (audit N2).
    """
    cfg = cfg or VerifyConfig()
    workdir = Path(workdir)
    su2 = su2_bin or find_su2()
    if su2 is None:
        raise RuntimeError("SU2_CFD not found (set $SU2_RUN or put SU2_CFD on PATH)")
    mesh = prepare_blade_verification_case(params, workdir, cfg)
    hist = run_su2(CFG_NAME, workdir, su2)
    series = parse_su2_unsteady_force_series(hist, force_candidates=_THRUST_Z_CANDIDATES)
    expected = cfg.n_cycles * STEPS_PER_CYCLE
    if series.size < expected:
        raise ValueError(
            f"{hist}: {series.size} time steps < expected {expected} "
            f"({cfg.n_cycles}×{STEPS_PER_CYCLE}) — run incomplete or diverged"
        )
    red = reduce_cycles(series[-expected:], steps_per_cycle=STEPS_PER_CYCLE, n_discard=1)
    return Blade3DAeroResult(
        j_fan_mean=red.j_fan, j_fan_peak=red.j_fan_peak, n_nodes=float(mesh.n_nodes)
    )
