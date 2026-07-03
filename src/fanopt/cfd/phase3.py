"""Phase 3 — steady↔unsteady 2D-slice correlation gate (report-final.md §Phase 3).

For a sweep of designs, run the cheap **steady** slice (CD) and the true
**2D-unsteady** slice, and correlate steady CD against the unsteady **RMS
loading amplitude**. A high R² means the steady proxy is a trustworthy screening
fidelity (and, per plan_v1_slim_latest.md §4, that adjoint-on-the-proxy is
viable). The RMS — not the cycle-mean — is the discriminator: the mean net
momentum flux (J_fan) is ~0 for the *symmetric* baseline panels (stroke forces
cancel), whereas the RMS amplitude is nonzero, converges in ~1 cycle, and is
what steady CD actually predicts. J_fan (mean) remains the final ASO objective;
the optimizer's job is to find asymmetric shapes that rectify it nonzero.

Pure-Python orchestration around tested pieces (``mesh_2d_slice``, ``configs``,
``parsers``, ``j_fan.reduce_cycles``, ``correlation``). The only side effect is
the SU2 subprocess in :func:`run_su2` — everything else is testable offline.
SU2 runs **locally** (macOS binary via Rosetta) or on Colab; both work.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fanopt.cfd.configs import render_slice_steady_cfg, render_slice_unsteady_cfg
from fanopt.cfd.correlation import CorrelationResult, correlate
from fanopt.cfd.j_fan import reduce_cycles
from fanopt.cfd.mesh_2d_slice import (
    CASCADE_WALL_MARKER,
    FAN_SURFACE_MARKER,
    FARFIELD_MARKER,
    SliceMeshParams,
    baseline_cascade_polygons,
    build_cascade_slice_mesh,
)
from fanopt.cfd.parsers import parse_su2_history_thrust, parse_su2_unsteady_force_series

__all__ = [
    "DesignPoint",
    "DesignResult",
    "sweep_designs",
    "prepare_design_case",
    "extract_steady_drag",
    "extract_unsteady_mean",
    "extract_unsteady_rms",
    "find_su2",
    "run_su2",
    "run_correlation_sweep",
]

MESH_NAME = "slice.su2"
_STEADY_CD = ("CD", "CDrag", "C_D")

# Demo unsteady settings — reduced from the locked 5 cycles / 200 steps so a
# full local sweep finishes in ~minutes. Production ranking uses 5 / 200.
_UNSTEADY_CYCLES = 3
_UNSTEADY_INNER = 30
_UNSTEADY_STEPS_PER_CYCLE = 40


@dataclass(frozen=True)
class DesignPoint:
    """One sweep design — parameters of the baseline-family cross-section."""

    name: str
    n_blades: int
    panel_thickness_m: float
    panel_gap_m: float


@dataclass(frozen=True)
class DesignResult:
    name: str
    steady_cd: float
    unsteady_mean: float  # net momentum flux (J_fan) — ~0 for symmetric baselines
    unsteady_rms: float  # loading-magnitude amplitude — the screening discriminator
    meta: dict[str, float] = field(default_factory=dict)


def sweep_designs() -> list[DesignPoint]:
    """A small spread of designs varying blade count + corrugation thickness +
    gap, so both steady and unsteady drag range across the set."""
    return [
        DesignPoint("b3_t22", 3, 0.0022, 0.005),
        DesignPoint("b3_t38", 3, 0.0038, 0.005),
        DesignPoint("b4_t30", 4, 0.0030, 0.005),
        DesignPoint("b5_t22", 5, 0.0022, 0.004),
        DesignPoint("b5_t38", 5, 0.0038, 0.004),
        DesignPoint("b4_t30_wide", 4, 0.0030, 0.010),
    ]


def _cross_section(d: DesignPoint) -> list[np.ndarray]:
    return baseline_cascade_polygons(
        n_blades=d.n_blades, panel_thickness_m=d.panel_thickness_m, panel_gap_m=d.panel_gap_m
    )


def prepare_design_case(d: DesignPoint, workdir: Path) -> dict[str, object]:
    """Mesh the design + render its steady (productive) and unsteady cfgs."""
    workdir.mkdir(parents=True, exist_ok=True)
    mesh = build_cascade_slice_mesh(_cross_section(d), SliceMeshParams(), workdir / MESH_NAME)
    steady = render_slice_steady_cfg(
        mesh_filename=MESH_NAME,
        marker_fan=FAN_SURFACE_MARKER,
        marker_farfield=FARFIELD_MARKER,
        marker_cascade=CASCADE_WALL_MARKER,
    )
    unsteady = render_slice_unsteady_cfg(
        mesh_filename=MESH_NAME,
        marker_fan=FAN_SURFACE_MARKER,
        marker_farfield=FARFIELD_MARKER,
        marker_cascade=CASCADE_WALL_MARKER,
        n_cycles=_UNSTEADY_CYCLES,
        inner_iter=_UNSTEADY_INNER,
        steps_per_cycle=_UNSTEADY_STEPS_PER_CYCLE,
    )
    (workdir / "steady.cfg").write_text(steady, encoding="utf-8")
    (workdir / "unsteady.cfg").write_text(unsteady, encoding="utf-8")
    return {"mesh": str(workdir / MESH_NAME), "n_nodes": mesh.n_nodes}


def extract_steady_drag(history_csv: Path) -> float:
    """Steady CD (streamwise drag) from a steady slice history.csv."""
    return parse_su2_history_thrust(history_csv, thrust_candidates=_STEADY_CD)


def extract_unsteady_mean(history_csv: Path, *, n_cycles: int = _UNSTEADY_CYCLES) -> float:
    """Cycle-averaged plunging force (unsteady J_fan proxy) from an unsteady run.

    Uses ``reduce_cycles`` over the retained cycles (discard cycle 1 transient).
    ``steps_per_cycle`` is inferred from the series length / n_cycles.
    """
    series = parse_su2_unsteady_force_series(history_csv)
    steps_per_cycle = series.size // n_cycles
    if steps_per_cycle < 1:
        raise ValueError(f"series too short ({series.size}) for {n_cycles} cycles")
    usable = series[: steps_per_cycle * n_cycles]
    return reduce_cycles(usable, steps_per_cycle=steps_per_cycle, n_discard=1).j_fan


def extract_unsteady_rms(
    history_csv: Path, *, n_cycles: int = _UNSTEADY_CYCLES, n_discard: int = 1
) -> float:
    """RMS amplitude of the plunging force over the retained cycles.

    The **screening discriminator** for the Phase 3 correlation gate. Unlike the
    cycle-MEAN (:func:`extract_unsteady_mean`, the net-thrust objective J_fan
    which is ~0 for *symmetric* baseline panels — up-stroke and down-stroke
    forces cancel), the RMS measures aerodynamic loading magnitude: nonzero and
    stable even for symmetric designs, converged within ~1 cycle. Steady CD
    predicts it well, so it validates the cheap tier as a screen
    (plan_v1_slim_latest.md §Phase 3).
    """
    series = parse_su2_unsteady_force_series(history_csv)
    steps_per_cycle = series.size // n_cycles
    if steps_per_cycle < 1:
        raise ValueError(f"series too short ({series.size}) for {n_cycles} cycles")
    retained = series[: steps_per_cycle * n_cycles].reshape(n_cycles, steps_per_cycle)[n_discard:]
    return float(np.sqrt(np.mean(retained**2)))


def find_su2() -> str | None:
    """Locate SU2_CFD: $SU2_RUN/SU2_CFD, then PATH."""
    run = os.environ.get("SU2_RUN")
    if run:
        cand = Path(run) / "SU2_CFD"
        if cand.exists():
            return str(cand)
    return shutil.which("SU2_CFD")


def run_su2(cfg: str, workdir: Path, su2_bin: str) -> Path:
    """Run SU2 on ``cfg`` in ``workdir``; return the produced history.csv.

    Raises RuntimeError with the log tail on failure (so errors surface).
    """
    log = workdir / (cfg + ".log")
    env = dict(os.environ, SU2_RUN=str(Path(su2_bin).parent))
    with open(log, "w") as f:
        r = subprocess.run(
            [su2_bin, cfg], cwd=str(workdir), stdout=f, stderr=subprocess.STDOUT, env=env
        )
    if r.returncode != 0:
        tail = "\n".join(log.read_text().splitlines()[-40:])
        raise RuntimeError(f"SU2 failed on {cfg} (exit {r.returncode}):\n{tail}")
    hist = sorted(workdir.glob("history*.csv"), key=lambda p: p.stat().st_mtime)
    if not hist:
        raise RuntimeError(f"SU2 produced no history.csv for {cfg}; see {log}")
    return hist[-1]


def run_correlation_sweep(
    workdir: Path, *, designs: list[DesignPoint] | None = None, su2_bin: str | None = None
) -> tuple[CorrelationResult, list[DesignResult]]:
    """Full sweep: per design, run steady + unsteady locally and correlate."""
    designs = designs or sweep_designs()
    su2_bin = su2_bin or find_su2()
    if su2_bin is None:
        raise RuntimeError("SU2_CFD not found (set $SU2_RUN or put SU2_CFD on PATH)")

    results: list[DesignResult] = []
    for d in designs:
        d_dir = workdir / d.name
        prepare_design_case(d, d_dir)
        steady_hist = run_su2("steady.cfg", d_dir, su2_bin)
        cd = extract_steady_drag(steady_hist)
        # unsteady run writes its own history; move steady's aside first
        steady_hist.rename(d_dir / "history_steady.csv")
        unsteady_hist = run_su2("unsteady.cfg", d_dir, su2_bin)
        mean = extract_unsteady_mean(unsteady_hist)
        rms = extract_unsteady_rms(unsteady_hist)
        results.append(DesignResult(d.name, cd, mean, rms, meta={"n_blades": float(d.n_blades)}))

    # Correlate steady CD against the RMS loading amplitude (the screening
    # discriminator) — NOT the cycle-mean, which is ~0 for symmetric baselines.
    steady = np.array([r.steady_cd for r in results])
    unsteady = np.array([r.unsteady_rms for r in results])
    return correlate(steady, unsteady), results
