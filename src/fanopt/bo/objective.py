"""Phase 4 objective spine — design vector → objectives (V1-slim Stage 2).

The single evaluation the BO loop drives: decode a design vector, build its 2D
mid-radius cascade slice, run the true unsteady SU2 slice, and reduce it to
``J_fan`` (the net directed momentum flux — the ASO objective, *maximized*; the
cycle-MEAN, not the Phase-3 screening RMS). The other two Pareto axes —
``I_wrist`` (rotational inertia, minimized) and the structural objective
(minimized) — are **injected callables** so this module stays CFD-focused and
testable without CadQuery/FEniCSx, and so the structural objective's definition
can be settled without churning this spine.

CFD side effects (gmsh mesh + SU2 subprocess) reuse the tested
:mod:`fanopt.cfd.phase3` helpers. SU2 runs locally (macOS binary via Rosetta) or
on Colab. The demo unsteady resolution mirrors Phase 3 so a single local eval
finishes in ~minutes; production ASO ranking uses the locked 5 cycles / 200 steps
(§9.4).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fanopt.bo.codec import decode
from fanopt.cfd.configs import render_slice_unsteady_cfg
from fanopt.cfd.mesh_2d_slice import (
    CASCADE_WALL_MARKER,
    FAN_SURFACE_MARKER,
    FARFIELD_MARKER,
    SliceMeshParams,
    SliceMeshResult,
    build_cascade_slice_mesh,
)
from fanopt.cfd.panel_slice import panel_slice_polygons
from fanopt.cfd.phase3 import extract_unsteady_mean, find_su2, run_su2
from fanopt.geometry.envelope import Layer1Params

__all__ = [
    "SliceEvalConfig",
    "PRODUCTION_EVAL_CFG",
    "ObjectiveResult",
    "prepare_slice_case",
    "evaluate_j_fan",
    "evaluate_design",
]

MESH_NAME = "slice.su2"
UNSTEADY_CFG = "unsteady.cfg"

# Demo unsteady resolution (matches phase3); production uses 5 / 200 (§9.4 lock).
_DEMO_CYCLES = 3
_DEMO_INNER = 30
_DEMO_STEPS_PER_CYCLE = 40


@dataclass(frozen=True)
class SliceEvalConfig:
    """Knobs for one slice evaluation (slice station, cascade size, unsteady res)."""

    radial_u: float = 0.5
    n_panels: int = 5
    n_samples: int = 24
    n_cycles: int = _DEMO_CYCLES
    inner_iter: int = _DEMO_INNER
    steps_per_cycle: int = _DEMO_STEPS_PER_CYCLE


# Frozen shared default so it can sit in argument defaults (ruff B008-safe).
_DEFAULT_EVAL_CFG = SliceEvalConfig()

# Production ASO ranking resolution: the §9.4 lock is 5 cycles at dt = T/200.
# Phase-4 V1 reports *relative* gain, so the (ω·dt)² added-mass bias (Spike 0.6d.2)
# is geometry-independent at fixed ω and cancels in ranking → T/200 is the V1
# production choice (T/400 is reserved for Phase-5 PyFR cross-solver work).
PRODUCTION_EVAL_CFG = SliceEvalConfig(n_cycles=5, inner_iter=50, steps_per_cycle=200)


@dataclass(frozen=True)
class ObjectiveResult:
    """The three Pareto objectives for one design (+ CFD metadata).

    ``j_fan`` is maximized; ``i_wrist_kgm2`` and ``structural`` are minimized.
    The latter two are ``None`` until their injected evaluators are supplied.
    """

    j_fan: float
    i_wrist_kgm2: float | None
    structural: float | None
    meta: dict[str, float] = field(default_factory=dict)


def prepare_slice_case(
    layer1: Layer1Params, workdir: Path, cfg: SliceEvalConfig = _DEFAULT_EVAL_CFG
) -> SliceMeshResult:
    """Mesh the design's Path A+ slice + render its unsteady cfg into ``workdir``."""
    workdir.mkdir(parents=True, exist_ok=True)
    polygons = panel_slice_polygons(
        layer1.thickness_field,
        radial_u=cfg.radial_u,
        n_panels=cfg.n_panels,
        n_samples=cfg.n_samples,
        camber_knots_m=layer1.camber_knots_m,
    )
    mesh = build_cascade_slice_mesh(polygons, SliceMeshParams(), workdir / MESH_NAME)
    unsteady = render_slice_unsteady_cfg(
        mesh_filename=MESH_NAME,
        marker_fan=FAN_SURFACE_MARKER,
        marker_farfield=FARFIELD_MARKER,
        marker_cascade=CASCADE_WALL_MARKER,
        n_cycles=cfg.n_cycles,
        inner_iter=cfg.inner_iter,
        steps_per_cycle=cfg.steps_per_cycle,
    )
    (workdir / UNSTEADY_CFG).write_text(unsteady, encoding="utf-8")
    return mesh


def evaluate_j_fan(
    vector: np.ndarray,
    workdir: Path,
    *,
    su2_bin: str | None = None,
    cfg: SliceEvalConfig = _DEFAULT_EVAL_CFG,
) -> tuple[float, dict[str, float]]:
    """Decode → slice → unsteady SU2 → ``J_fan`` (cycle-mean momentum flux).

    Returns ``(j_fan, meta)``. Raises ``RuntimeError`` if SU2 is not locatable.
    """
    layer1 = decode(vector)
    su2 = su2_bin or find_su2()
    if su2 is None:
        raise RuntimeError("SU2_CFD not found (set $SU2_RUN or put SU2_CFD on PATH)")
    mesh = prepare_slice_case(layer1, workdir, cfg)
    hist = run_su2(UNSTEADY_CFG, workdir, su2)
    j_fan = extract_unsteady_mean(hist, n_cycles=cfg.n_cycles)
    meta = {"n_nodes": float(mesh.n_nodes), "blade_count": float(layer1.blade_count)}
    return j_fan, meta


def evaluate_design(
    vector: np.ndarray,
    workdir: Path,
    *,
    su2_bin: str | None = None,
    cfg: SliceEvalConfig = _DEFAULT_EVAL_CFG,
    inertia_fn: Callable[[Layer1Params], float] | None = None,
    structural_fn: Callable[[Layer1Params], float] | None = None,
) -> ObjectiveResult:
    """Full 3-objective evaluation of one design vector.

    ``inertia_fn`` and ``structural_fn`` (injected) map the decoded
    :class:`Layer1Params` to ``I_wrist`` and the structural objective; when
    omitted those fields are ``None`` (CFD-only spine).
    """
    j_fan, meta = evaluate_j_fan(vector, workdir, su2_bin=su2_bin, cfg=cfg)
    layer1 = decode(vector)
    i_wrist = inertia_fn(layer1) if inertia_fn is not None else None
    structural = structural_fn(layer1) if structural_fn is not None else None
    return ObjectiveResult(j_fan=j_fan, i_wrist_kgm2=i_wrist, structural=structural, meta=meta)
