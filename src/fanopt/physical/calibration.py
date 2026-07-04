"""Phase-6 model calibration — measured physical results vs predictions.

Consumes the printed top-k designs' bench measurements (IMU trace, anemometer
grid, microphone recording) and reduces each to the numbers that validate the
optimization's predictions (`docs/report-final.md` §Phase 6):

- **IMU** → ``W_cycle`` (J) + kinematic-cadence sanity — validates the ``I_wrist``
  objective the BO minimized.
- **Anemometer** → ``J_fan_proxy`` (N) — validates the CFD ``J_fan`` ranking. The
  measured proxy (Newtons) and the CFD ``J_fan`` (nondimensional) are **different
  units**, so they are compared by **rank** (Kendall τ across the designs), never
  by a fabricated conversion factor — the same rank-preservation test Phase 5 uses
  for slice-vs-3D.
- **Microphone** → SPL + tonal signature — for the blinded feel-test noise notes.

Every measurement is optional: a design with no bench data yet reduces to a row
of ``None``s, so this runs (and reports what's missing) before the whole set is
measured. Pure reduction around the tested :mod:`fanopt.physical` loaders; file
I/O stays at the :func:`reduce_design` boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fanopt.cfd.correlation import kendall_tau
from fanopt.physical.acoustic import analyze_acoustic_trace, load_acoustic_csv
from fanopt.physical.anemometer import analyze_anemometer_grid, load_anemometer_csv
from fanopt.physical.imu import F_WAVE_TARGET_HZ, analyze_imu_trace, load_imu_csv

__all__ = [
    "DesignMeasurement",
    "reduce_design",
    "calibrate",
]


@dataclass(frozen=True)
class DesignMeasurement:
    """One printed design's reduced bench measurements next to its predictions."""

    name: str
    predicted_i_wrist_kgm2: float | None = None
    predicted_j_fan_3d: float | None = None
    # IMU
    w_cycle_j: float | None = None
    f_wave_hz: float | None = None
    imu_sanity_ok: bool | None = None
    # Anemometer
    j_fan_proxy_n: float | None = None
    v_mean_m_per_s: float | None = None
    # Acoustic
    spl_db: float | None = None
    dominant_frequency_hz: float | None = None
    blade_pass_level_db: float | None = None
    n_measurements: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        return d


def reduce_design(
    name: str,
    *,
    predicted_i_wrist_kgm2: float | None = None,
    predicted_j_fan_3d: float | None = None,
    blade_count: int | None = None,
    imu_csv: Path | str | None = None,
    anemometer_csv: Path | str | None = None,
    acoustic_csv: Path | str | None = None,
) -> DesignMeasurement:
    """Reduce whichever bench measurements are present for one design.

    ``imu_csv`` needs ``predicted_i_wrist_kgm2`` (``W_cycle`` scales with it). If a
    path is ``None`` or missing, that channel is skipped (fields stay ``None``).
    ``blade_count`` sets the blade-pass tone (``blade_count × f_wave``) probed in
    the acoustic reduction.
    """
    fields: dict[str, Any] = {
        "predicted_i_wrist_kgm2": predicted_i_wrist_kgm2,
        "predicted_j_fan_3d": predicted_j_fan_3d,
    }
    n = 0

    if imu_csv is not None and Path(imu_csv).exists():
        if predicted_i_wrist_kgm2 is None:
            raise ValueError(f"{name}: imu_csv given but predicted_i_wrist_kgm2 is None")
        imu_res = analyze_imu_trace(load_imu_csv(imu_csv), predicted_i_wrist_kgm2)
        fields.update(
            w_cycle_j=imu_res.W_cycle_J,
            f_wave_hz=imu_res.f_wave_Hz,
            imu_sanity_ok=imu_res.sanity_ok,
        )
        n += 1

    if anemometer_csv is not None and Path(anemometer_csv).exists():
        anem_res = analyze_anemometer_grid(load_anemometer_csv(anemometer_csv))
        fields.update(
            j_fan_proxy_n=anem_res.J_fan_proxy_N, v_mean_m_per_s=anem_res.v_mean_grid_m_per_s
        )
        n += 1

    if acoustic_csv is not None and Path(acoustic_csv).exists():
        bp = blade_count * F_WAVE_TARGET_HZ if blade_count else None
        ac_res = analyze_acoustic_trace(load_acoustic_csv(acoustic_csv), blade_pass_frequency_hz=bp)
        fields.update(
            spl_db=ac_res.spl_db,
            dominant_frequency_hz=ac_res.dominant_frequency_hz,
            blade_pass_level_db=ac_res.blade_pass_level_db,
        )
        n += 1

    return DesignMeasurement(name=name, n_measurements=n, **fields)


def calibrate(designs: list[DesignMeasurement]) -> dict[str, Any]:
    """Cross-design calibration: does the measured J_fan preserve the CFD ranking?

    Kendall τ between predicted 3D ``J_fan`` and measured anemometer ``J_fan_proxy``
    over designs that have both (unit-free, so the scale mismatch is irrelevant).
    ``rank_preserved`` is ``None`` when fewer than two designs are jointly measured.
    """
    paired = [
        (d.predicted_j_fan_3d, d.j_fan_proxy_n)
        for d in designs
        if d.predicted_j_fan_3d is not None
        and d.j_fan_proxy_n is not None
        and np.isfinite(d.predicted_j_fan_3d)
        and np.isfinite(d.j_fan_proxy_n)
    ]
    n_imu = sum(1 for d in designs if d.w_cycle_j is not None)
    n_acoustic = sum(1 for d in designs if d.spl_db is not None)
    j_fan_rank: dict[str, Any]
    if len(paired) < 2:
        j_fan_rank = {"n": len(paired), "rank_preserved": None}
    else:
        pred = np.array([p[0] for p in paired], dtype=float)
        meas = np.array([p[1] for p in paired], dtype=float)
        tau = kendall_tau(pred, meas)
        j_fan_rank = {"n": len(paired), "kendall_tau": tau, "rank_preserved": bool(tau > 0.0)}

    return {
        "n_designs": len(designs),
        "n_with_imu": n_imu,
        "n_with_anemometer": len(paired),
        "n_with_acoustic": n_acoustic,
        "j_fan_rank": j_fan_rank,
        "designs": [d.to_dict() for d in designs],
    }
