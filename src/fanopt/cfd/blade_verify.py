"""Phase 5 — 3D verification for the redesigned aero-first blade.

Re-evaluates the campaign's top Pareto designs with a **3D** unsteady CFD run to check that
the cheap 2D mid-radius-slice ranking (the :mod:`~fanopt.bo.blade_campaign` screen) survives
the 3D physics it omits (finite span, tip/root effects). Per design: decode the BO vector
with the new :mod:`~fanopt.bo.blade_codec` → build the both-face solid
(:func:`~fanopt.geometry.blade_cad.make_blade_solid`) → STEP → 3D volume mesh
(:mod:`~fanopt.cfd.mesh`) → 3D unsteady SU2 → canonical cycle-mean ``J_fan``. Then correlate
3D vs slice ``J_fan`` (Kendall τ > 0 means the slice preserved the ranking).

This is a thin adapter: the verify spine (process pool, checkpointing, ranking metrics) lives
in :mod:`~fanopt.cfd.phase5`; here we supply only the *design build* for the redesigned
geometry and a helper to pull the designs out of the campaign's ``pareto.json``. The blade's
radial extent runs along +x, so the fan stroke is a pitch about +y through the origin — the
default motion in :func:`~fanopt.cfd.configs.render_unsteady_cfg`, reused verbatim.

Per CLAUDE.md §4.1 this module imports cadquery unconditionally; environments without it fail
to import (tests skip at module load via ``find_spec``).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import cadquery as cq
import numpy as np

from fanopt.bo.blade_codec import decode
from fanopt.cfd.configs import render_unsteady_cfg
from fanopt.cfd.mesh import (
    FAN_SURFACE_MARKER,
    FARFIELD_MARKER,
    VolumeMeshResult,
    build_volume_mesh,
)
from fanopt.cfd.phase5 import VerifyConfig, VerifyResult, run_verification, verify_ranking
from fanopt.geometry.blade_cad import make_blade_solid
from fanopt.utils.ledger import design_hash

__all__ = [
    "STEP_NAME",
    "MESH_NAME",
    "CFG_NAME",
    "prepare_blade_verification_case",
    "designs_from_pareto",
    "verify_blades",
    "load_pareto",
]

STEP_NAME = "blade.step"
MESH_NAME = "blade.su2"
CFG_NAME = "verify.cfg"


def prepare_blade_verification_case(
    vector: np.ndarray, workdir: Path, cfg: VerifyConfig
) -> VolumeMeshResult:
    """Decode → build the redesigned blade → STEP → 3D mesh → write the unsteady cfg.

    The ``prepare_fn`` hook :func:`fanopt.cfd.phase5.run_verification` calls per design.
    Signature matches the spine's original codec-bound prep so it drops in directly.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    params = decode(vector)
    solid = make_blade_solid(params)
    step = workdir / STEP_NAME
    cq.exporters.export(solid, str(step))
    mesh = build_volume_mesh(step, cfg.mesh_params, workdir / MESH_NAME)
    unsteady = render_unsteady_cfg(
        mesh_filename=MESH_NAME,
        marker_fan=FAN_SURFACE_MARKER,
        marker_farfield=FARFIELD_MARKER,
        n_cycles=cfg.n_cycles,
        inner_iter=cfg.inner_iter,
        # J_fan only needs history.csv; write a single small restart once (SU2 requires >=1
        # OUTPUT_FILES entry) instead of ~240 MB of per-timestep field dumps.
        output_files="( RESTART )",
        output_wrt_freq=10_000_000,
    )
    (workdir / CFG_NAME).write_text(unsteady, encoding="utf-8")
    return mesh


def designs_from_pareto(
    pareto: list[dict[str, object]], top_k: int | None = None
) -> list[tuple[str, np.ndarray, float | None]]:
    """``pareto.json`` records → ``(name, vector, j_fan_slice)`` designs, best ``J_fan`` first.

    ``top_k`` keeps only the highest-``J_fan`` designs (all if ``None``). Each design's name is
    ``rank_hash`` — stable across reruns (the hash is the design's, the rank its slice order) so
    a resumed 3D verification reuses prior per-design scratch. ``j_fan_slice`` is the campaign's
    2D-slice ``J_fan``, carried through so the spine can correlate it against the 3D value.
    """
    ranked = sorted(pareto, key=lambda d: float(d["j_fan"]), reverse=True)  # type: ignore[arg-type]
    if top_k is not None:
        ranked = ranked[:top_k]
    designs: list[tuple[str, np.ndarray, float | None]] = []
    for rank, d in enumerate(ranked):
        vector = np.asarray(d["vector"], dtype=float)
        h = design_hash(d["params"])  # type: ignore[arg-type]
        designs.append((f"{rank:02d}_{h}", vector, float(d["j_fan"])))  # type: ignore[arg-type]
    return designs


def verify_blades(
    pareto: list[dict[str, object]],
    workdir: Path,
    *,
    top_k: int | None = None,
    cfg: VerifyConfig | None = None,
    su2_bin: str | None = None,
    n_workers: int = 1,
    progress: bool = False,
    on_result: Callable[[VerifyResult], None] | None = None,
) -> tuple[list[VerifyResult], dict[str, object]]:
    """3D-verify the campaign's top Pareto blades; return ``(results, ranking-metrics)``.

    Reuses the :mod:`~fanopt.cfd.phase5` spine with the redesigned-blade prep. ``results`` are
    per-design (3D ``J_fan`` beside slice ``J_fan``); the ranking dict is :func:`verify_ranking`'s
    slice-vs-3D agreement (Kendall τ, Spearman ρ, Pearson R², suspect flags).
    """
    designs = designs_from_pareto(pareto, top_k=top_k)
    results = run_verification(
        designs,
        Path(workdir),
        cfg=cfg or VerifyConfig(),
        su2_bin=su2_bin,
        n_workers=n_workers,
        progress=progress,
        on_result=on_result,
        prepare_fn=prepare_blade_verification_case,
    )
    return results, verify_ranking(results)


def load_pareto(path: Path) -> list[dict[str, object]]:
    """Read a campaign ``pareto.json`` (a list of design records)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
