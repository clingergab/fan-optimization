"""Phase 5 — high-fidelity 3D verification of the top Pareto designs.

Takes the campaign's top-k diverse designs and re-evaluates each with a **3D**
unsteady CFD run (a single V-unit blade in an external-flow domain — see
:mod:`fanopt.cfd.mesh`) to check that the cheap 2D mid-radius-slice ranking
survives the 3D physics it omits (finite span, tip/root effects). Per design:
decode → build the blade (CadQuery) → STEP → 3D volume mesh → 3D unsteady SU2 →
canonical cycle-mean ``J_fan``. Then correlate 3D vs slice ``J_fan`` (Kendall τ):
τ > 0 means the slice preserved the ranking.

Pure-Python orchestration around tested pieces; the only heavy side effect is the
SU2 subprocess (reused from :mod:`fanopt.cfd.phase3`). The 3D unsteady run is
expensive — a Colab job in practice — but geometry + meshing + cfg run locally.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import cadquery as cq
import numpy as np
from tqdm.auto import tqdm

from fanopt.bo.codec import decode
from fanopt.bo.inertia import NEUTRAL_LAYER4
from fanopt.cfd.configs import render_unsteady_cfg
from fanopt.cfd.correlation import kendall_tau, pearson_r2, spearman_rho
from fanopt.cfd.j_fan import reduce_cycles
from fanopt.cfd.mesh import (
    FAN_SURFACE_MARKER,
    FARFIELD_MARKER,
    VolumeMeshParams,
    VolumeMeshResult,
    build_volume_mesh,
)
from fanopt.cfd.parsers import parse_su2_unsteady_force_series
from fanopt.cfd.phase3 import find_su2, run_su2
from fanopt.geometry.assembly_cad import make_vunit_blade
from fanopt.geometry.fields import Layer2Params
from fanopt.geometry.generator import BladeDesignParams
from fanopt.geometry.primitives import Layer3Primitive

__all__ = [
    "VerifyConfig",
    "VerifyResult",
    "blade_params_from_vector",
    "prepare_verification_case",
    "extract_j_fan_3d",
    "run_verification",
    "verify_ranking",
]

STEP_NAME = "blade.step"
MESH_NAME = "blade.su2"
CFG_NAME = "verify.cfg"

# Demo unsteady resolution (3D is expensive); production ranking uses 5 / 100.
_DEMO_CYCLES = 3
_DEMO_INNER = 30


@dataclass(frozen=True)
class VerifyConfig:
    """Knobs for one 3D verification run."""

    n_cycles: int = _DEMO_CYCLES
    inner_iter: int = _DEMO_INNER
    mesh_params: VolumeMeshParams = field(default_factory=VolumeMeshParams)


@dataclass(frozen=True)
class VerifyResult:
    """One design's 3D J_fan next to its 2D-slice J_fan (from the campaign)."""

    name: str
    j_fan_3d: float
    j_fan_slice: float | None
    meta: dict[str, float] = field(default_factory=dict)


_DEFAULT_VERIFY_CFG = VerifyConfig()


def blade_params_from_vector(vector: np.ndarray) -> BladeDesignParams:
    """Decode a BO vector to a full single-blade design (neutral Layer 2/3/4)."""
    return BladeDesignParams(
        layer1=decode(vector),
        layer2=Layer2Params.all_inactive(),
        layer3=Layer3Primitive.absent(),
        layer4=NEUTRAL_LAYER4,
    )


def prepare_verification_case(
    vector: np.ndarray, workdir: Path, cfg: VerifyConfig = _DEFAULT_VERIFY_CFG
) -> VolumeMeshResult:
    """Build the blade, export STEP, 3D-mesh it, and render the unsteady cfg."""
    workdir.mkdir(parents=True, exist_ok=True)
    blade = make_vunit_blade(blade_params_from_vector(vector))
    step = workdir / STEP_NAME
    cq.exporters.export(blade, str(step))
    mesh = build_volume_mesh(step, cfg.mesh_params, workdir / MESH_NAME)
    unsteady = render_unsteady_cfg(
        mesh_filename=MESH_NAME,
        marker_fan=FAN_SURFACE_MARKER,
        marker_farfield=FARFIELD_MARKER,
        n_cycles=cfg.n_cycles,
        inner_iter=cfg.inner_iter,
    )
    (workdir / CFG_NAME).write_text(unsteady, encoding="utf-8")
    return mesh


def extract_j_fan_3d(history_csv: Path, *, n_cycles: int = _DEMO_CYCLES) -> float:
    """Cycle-mean 3D J_fan from an unsteady history.csv (discard cycle 1)."""
    series = parse_su2_unsteady_force_series(history_csv)
    steps_per_cycle = series.size // n_cycles
    if steps_per_cycle < 1:
        raise ValueError(f"series too short ({series.size}) for {n_cycles} cycles")
    usable = series[: steps_per_cycle * n_cycles]
    return reduce_cycles(usable, steps_per_cycle=steps_per_cycle, n_discard=1).j_fan


@dataclass(frozen=True)
class _VerifyWorker:
    """Picklable per-design 3D verification (for the process pool)."""

    workdir: Path
    cfg: VerifyConfig
    su2_bin: str
    prepare_fn: Callable[[np.ndarray, Path, VerifyConfig], VolumeMeshResult]

    def __call__(self, design: tuple[str, np.ndarray, float | None]) -> VerifyResult:
        name, vector, j_slice = design
        d_dir = self.workdir / name
        try:
            mesh = self.prepare_fn(vector, d_dir, self.cfg)
            hist = run_su2(CFG_NAME, d_dir, self.su2_bin)
            j3d = extract_j_fan_3d(hist, n_cycles=self.cfg.n_cycles)
            return VerifyResult(name, j3d, j_slice, meta={"n_nodes": float(mesh.n_nodes)})
        except Exception as exc:  # fault isolation: one bad design shouldn't sink the batch
            d_dir.mkdir(parents=True, exist_ok=True)
            (d_dir / "FAILED.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            return VerifyResult(name, float("nan"), j_slice, meta={"failed": 1.0})


def run_verification(
    designs: list[tuple[str, np.ndarray, float | None]],
    workdir: Path,
    *,
    cfg: VerifyConfig = _DEFAULT_VERIFY_CFG,
    su2_bin: str | None = None,
    n_workers: int = 1,
    progress: bool = False,
    on_result: Callable[[VerifyResult], None] | None = None,
    prepare_fn: Callable[
        [np.ndarray, Path, VerifyConfig], VolumeMeshResult
    ] = prepare_verification_case,
) -> list[VerifyResult]:
    """3D-verify each ``(name, vector, j_fan_slice)`` design; return the results.

    ``n_workers`` > 1 runs designs concurrently in **separate processes** (gmsh
    can't be threaded; each 3D SU2 run is single-core, so ``n_workers`` ≈ min(
    n_designs, cores) is the useful range). Order is preserved. ``progress`` shows
    a live ``tqdm`` bar over the designs (each 3D run takes a while). ``on_result``,
    if given, is called with each :class:`VerifyResult` as it completes — the caller
    uses it to checkpoint partial results so a mid-run crash/disconnect isn't total
    loss. ``prepare_fn`` builds one design's case (blade → STEP → 3D mesh → cfg); it
    defaults to the original codec-bound blade, and the redesigned aero-first blade
    passes :func:`fanopt.cfd.blade_verify.prepare_blade_verification_case` instead.
    """
    su2 = su2_bin or find_su2()
    if su2 is None:
        raise RuntimeError("SU2_CFD not found (set $SU2_RUN or put SU2_CFD on PATH)")
    worker = _VerifyWorker(workdir, cfg, su2, prepare_fn)
    bar = tqdm(total=len(designs), disable=not progress, desc="Phase 5 3D verify", unit="design")
    try:
        if n_workers > 1 and len(designs) > 1:
            out: list[VerifyResult | None] = [None] * len(designs)
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                fut_to_i = {pool.submit(worker, d): i for i, d in enumerate(designs)}
                for fut in as_completed(fut_to_i):
                    r = fut.result()
                    out[fut_to_i[fut]] = r
                    if on_result is not None:
                        on_result(r)
                    bar.update(1)
            return [r for r in out if r is not None]
        results: list[VerifyResult] = []
        for d in designs:
            r = worker(d)
            results.append(r)
            if on_result is not None:
                on_result(r)
            bar.update(1)
        return results
    finally:
        bar.close()


def _rank_metrics(pairs: list[tuple[float, float]]) -> dict[str, object]:
    """Kendall τ, Spearman ρ, Pearson R² over (slice, 3D) pairs (None if < 2)."""
    if len(pairs) < 2:
        return {"n": len(pairs), "kendall_tau": None, "spearman_rho": None, "pearson_r2": None}
    s = np.array([p[0] for p in pairs], dtype=float)
    c = np.array([p[1] for p in pairs], dtype=float)
    return {
        "n": len(pairs),
        "kendall_tau": kendall_tau(s, c),
        "spearman_rho": spearman_rho(s, c),
        "pearson_r2": pearson_r2(s, c),
    }


def verify_ranking(results: list[VerifyResult]) -> dict[str, object]:
    """Full-spectrum slice-vs-3D agreement — three metrics, with suspect designs flagged.

    A **suspect** design has a non-finite (failed 3D run) or **negative** 3D J_fan
    (net reverse thrust — physically invalid for a fan, i.e. a degenerate or
    non-converged run). Metrics are reported twice: over ``all_finite`` pairs
    (includes negative-J designs) and ``valid_only`` (excludes them), so a ranking
    that only "holds" because both fidelities agree a degenerate design is worst is
    exposed rather than hidden behind a single τ. Kept for back-compat: top-level
    ``n`` / ``kendall_tau`` / ``rank_preserved`` mirror the ``valid_only`` set.
    """
    with_slice = [r for r in results if r.j_fan_slice is not None]
    finite = [r for r in with_slice if np.isfinite(r.j_fan_3d)]
    suspect = [r for r in with_slice if not np.isfinite(r.j_fan_3d) or r.j_fan_3d < 0.0]
    valid = [r for r in finite if r.j_fan_3d >= 0.0]

    all_finite = _rank_metrics([(r.j_fan_slice, r.j_fan_3d) for r in finite])  # type: ignore[misc]
    valid_only = _rank_metrics([(r.j_fan_slice, r.j_fan_3d) for r in valid])  # type: ignore[misc]
    valid_tau = valid_only["kendall_tau"]
    rank_preserved = bool(valid_tau > 0.0) if isinstance(valid_tau, float) else None

    return {
        "n": valid_only["n"],
        "kendall_tau": valid_tau,
        "rank_preserved": rank_preserved,
        "n_suspect": len(suspect),
        "suspect_designs": [r.name for r in suspect],
        "all_finite": all_finite,
        "valid_only": valid_only,
        "pairs": [
            {
                "name": r.name,
                "j_fan_slice": r.j_fan_slice,
                "j_fan_3d": r.j_fan_3d if np.isfinite(r.j_fan_3d) else None,
                "suspect": not np.isfinite(r.j_fan_3d) or r.j_fan_3d < 0.0,
            }
            for r in with_slice
        ],
    }
