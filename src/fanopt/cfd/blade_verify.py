"""Phase 5 / Stage 3.C — fine-tier 3D verification for the redesigned aero-first blade.

Re-evaluates the campaign's top designs (ranked by their **coarse** 3D ``J_fan``) at the **fine**
3D tier to check the cheap coarse ranking the campaign screened on survives the higher fidelity.
Per design (from its **absolute** ``BladeParams`` — decoded once from the campaign's stored vector,
so the verify is codec-range-independent): build the both-face solid
(:func:`~fanopt.geometry.blade_cad.make_blade_solid`) → STEP → 3D volume mesh
(:mod:`~fanopt.cfd.mesh`) → 3D unsteady SU2 → canonical cycle-mean ``J_fan``. Then correlate the
fine ``J_fan`` against the coarse ``J_fan`` (Kendall τ; ≥ 0.7 means the coarse tier preserved the
ranking, so the V1 pick comes from the fine ranking). Both tiers use the SAME 3D solver — ADR-0004
retired the 2D mid-radius slice — differing only in ``n_cycles`` / ``inner_iter``.

Thin adapter: the verify spine (process pool, checkpointing, ranking metrics) lives in
:mod:`~fanopt.cfd.phase5`; here we supply the *design build* for the redesigned geometry and helpers
to pull the top designs out of the distributed campaign's Drive shards (``evaluations_*.jsonl``,
which store each design's ``vector``) or a legacy ``pareto.json``. The blade's radial extent runs
along +x, so the fan stroke is a pitch about +y through the origin — the default motion in
:func:`~fanopt.cfd.configs.render_unsteady_cfg`, reused verbatim.

Per CLAUDE.md §4.1 this module imports cadquery unconditionally; environments without it fail
to import (tests skip at module load via ``find_spec``).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import cadquery as cq
import numpy as np

from fanopt.bo.blade_codec import N_DIMS, decode
from fanopt.cfd.configs import render_unsteady_cfg
from fanopt.cfd.mesh import (
    FAN_SURFACE_MARKER,
    FARFIELD_MARKER,
    VolumeMeshResult,
    build_volume_mesh,
)
from fanopt.cfd.phase5 import VerifyConfig, VerifyResult, run_verification, verify_ranking
from fanopt.geometry.blade import BladeParams
from fanopt.geometry.blade_cad import make_blade_solid
from fanopt.utils.ledger import design_hash

__all__ = [
    "STEP_NAME",
    "MESH_NAME",
    "CFG_NAME",
    "prepare_blade_verification_case",
    "designs_from_pareto",
    "load_campaign_rows",
    "top_designs_from_shards",
    "verify_blades",
    "load_pareto",
]

STEP_NAME = "blade.step"
MESH_NAME = "blade.su2"
CFG_NAME = "verify.cfg"


def prepare_blade_verification_case(
    params: BladeParams, workdir: Path, cfg: VerifyConfig
) -> VolumeMeshResult:
    """Build the redesigned blade → STEP → 3D mesh → write the unsteady cfg.

    Takes the design's **absolute** :class:`BladeParams` (not a BO vector), so the 3D verify
    is independent of the codec's search-space ranges — widening those for a later campaign
    can't retro-corrupt a verification of already-found designs. The ``prepare_fn`` hook
    :func:`fanopt.cfd.phase5.run_verification` calls per design.
    """
    workdir.mkdir(parents=True, exist_ok=True)
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


def _record_to_params(record: dict[str, object]) -> BladeParams:
    """Absolute :class:`BladeParams` from a design record — from ``params`` if present, else by
    decoding the stored ``vector`` (the distributed campaign shards carry only the vector)."""
    if "params" in record:
        return BladeParams.from_dict(record["params"])  # type: ignore[arg-type]
    if "vector" in record:
        return decode(np.asarray(record["vector"], dtype=float))
    raise KeyError("design record needs a 'params' dict or a 'vector'")


def designs_from_pareto(
    records: list[dict[str, object]], top_k: int | None = None
) -> list[tuple[str, BladeParams, float | None]]:
    """Design records → ``(name, params, j_fan_coarse)`` designs, best (coarse) ``J_fan`` first.

    Accepts either legacy ``pareto.json`` records (with a ``params`` dict) or the campaign's shard
    rows (with a ``vector``); each design's **absolute** :class:`BladeParams` is reconstructed so a
    verification is independent of the codec ranges the vector was encoded under. ``top_k`` keeps
    only the highest-``J_fan`` designs (all if ``None``). Each name is ``rank_hash`` — stable across
    reruns. ``j_fan_coarse`` is the campaign's coarse-tier ``J_fan``, carried through so the spine
    can correlate it against the fine 3D value.
    """
    ranked = sorted(records, key=lambda d: float(d["j_fan"]), reverse=True)  # type: ignore[arg-type]
    if top_k is not None:
        ranked = ranked[:top_k]
    designs: list[tuple[str, BladeParams, float | None]] = []
    for rank, d in enumerate(ranked):
        params = _record_to_params(d)
        h = design_hash(params.to_dict())
        designs.append((f"{rank:02d}_{h}", params, float(d["j_fan"])))
    return designs


def load_campaign_rows(shared_dir: Path | str) -> list[dict[str, object]]:
    """Read + dedup the distributed campaign's Drive shards (``evaluations_*.jsonl``), keeping only
    finite-``J_fan`` designs. Each surviving row carries the design ``vector`` and its coarse
    ``j_fan`` — the input the async campaign actually writes (no ``pareto.json``)."""
    by_hash: dict[object, dict[str, object]] = {}
    for shard in sorted(Path(shared_dir).glob("evaluations_*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            j = row.get("j_fan")
            if not isinstance(j, (int, float)) or not np.isfinite(j):
                continue  # skip NaN/failed (infeasible / diverged) designs
            vec = row.get("vector")
            # Mirror the campaign reader (distributed_campaign._iter_rows): drop a torn / clobbered
            # row whose vector isn't a length-N_DIMS list of numbers. Concurrent Drive shard merges
            # can produce such rows; without this guard a single bad vector crashes decode() OUTSIDE
            # the fault-isolated pool and aborts the whole multi-hour launch.
            if not isinstance(vec, list) or len(vec) != N_DIMS:
                continue
            try:
                [float(v) for v in vec]
            except (TypeError, ValueError):
                continue
            by_hash.setdefault(row.get("design_hash"), row)  # dedup across shards; first wins
    return list(by_hash.values())


def top_designs_from_shards(
    shared_dir: Path | str, top_k: int | None = None
) -> list[tuple[str, BladeParams, float | None]]:
    """Top-k campaign designs by coarse ``J_fan`` → ``(name, params, j_fan_coarse)``, best first.

    Reads the campaign's Drive shards directly (the async campaign writes no ``pareto.json``) and
    decodes each design's stored 33-D vector to its absolute :class:`BladeParams` for the fine verify.
    """
    return designs_from_pareto(load_campaign_rows(shared_dir), top_k=top_k)


def verify_blades(
    records: list[dict[str, object]],
    workdir: Path,
    *,
    top_k: int | None = None,
    cfg: VerifyConfig | None = None,
    su2_bin: str | None = None,
    n_workers: int = 1,
    progress: bool = False,
    on_result: Callable[[VerifyResult], None] | None = None,
    skip_hashes: set[str] | None = None,
) -> tuple[list[VerifyResult], dict[str, object]]:
    """Fine-tier verify the campaign's top blades; return ``(results, ranking-metrics)``.

    ``records`` are the campaign's top designs (shard rows with ``vector``, or legacy ``pareto.json``
    records with ``params``). Reuses the :mod:`~fanopt.cfd.phase5` spine with the redesigned-blade
    prep; ``cfg`` should be the FINE tier (:meth:`VerifyConfig.fine`) so this re-runs at higher
    fidelity than the coarse campaign screen. Each fine ``J_fan`` is scaled to **whole-fan**
    (× ``blade_count``, via ``_whole_fan_scale``) so it is the same quantity as the campaign's
    coarse ``j_fan``. ``skip_hashes`` (design_hash suffixes) are dropped from the top-k before
    verifying — the caller passes the already-done designs so a resumed run re-verifies only the
    rest. ``results`` are per-design (whole-fan fine ``J_fan`` beside the coarse ``J_fan``); the
    ranking dict is :func:`verify_ranking`'s coarse-vs-fine agreement (Kendall τ, Spearman ρ,
    Pearson R², failed-run flags).
    """
    designs = designs_from_pareto(records, top_k=top_k)
    if skip_hashes:  # resume: drop designs already verified (name is "{rank}_{design_hash}")
        designs = [d for d in designs if d[0].split("_", 1)[1] not in skip_hashes]
    results = run_verification(
        designs,
        Path(workdir),
        cfg=cfg or VerifyConfig.fine(),
        su2_bin=su2_bin,
        n_workers=n_workers,
        progress=progress,
        on_result=on_result,
        prepare_fn=prepare_blade_verification_case,
        scale_fn=_whole_fan_scale,  # per-blade CFz → whole-fan, matching the coarse campaign J_fan
    )
    return results, verify_ranking(results)


def _whole_fan_scale(params: object, per_blade_j: float) -> float:
    """Per-blade fine J_fan → whole-fan (× ``blade_count``), matching the campaign objective's
    ``blade_objective_3d.whole_fan_j_fan``. The campaign's coarse ``j_fan`` is whole-fan, so the
    fine value must be too for a like-for-like coarse-vs-fine comparison. Inlined (not imported)
    to avoid the blade_verify → blade_objective_3d → blade_aero_3d → blade_verify import cycle.
    Module-level (not a lambda) so ``_VerifyWorker`` stays picklable for the process pool."""
    return per_blade_j * params.blade_count  # type: ignore[attr-defined]


def load_pareto(path: Path) -> list[dict[str, object]]:
    """Read a campaign ``pareto.json`` (a list of design records)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
