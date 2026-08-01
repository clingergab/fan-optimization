"""Phase 5 / Stage 3.C — high-fidelity verification of the campaign's top designs.

Takes the campaign's top-k designs (ranked by their **coarse-tier** 3D ``J_fan``) and re-evaluates
each at the **fine** 3D tier (more cycles / inner-iterations) to check that the cheap coarse ranking
the campaign screened on survives the higher-fidelity physics. Per design: build the blade (CadQuery)
→ STEP → 3D volume mesh → 3D unsteady SU2 → canonical cycle-mean ``J_fan``. Then correlate the fine
``J_fan`` against the coarse ``J_fan`` (Kendall τ): high τ means the coarse tier the campaign used
preserved the ranking, so the top-by-coarse designs really are the top-by-fine designs — the V1 pick
comes from the fine ranking.

Both tiers are the **same 3D unsteady solver** (ADR-0004 retired the old 2D mid-radius slice); the
only difference is temporal fidelity (``n_cycles`` / ``inner_iter``). Pure-Python orchestration around
tested pieces; the heavy side effect is the SU2 subprocess. The fine run is expensive — a Colab job
in practice — but geometry + meshing + cfg run locally.
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
from fanopt.cfd.j_fan import STEPS_PER_CYCLE, reduce_cycles
from fanopt.cfd.mesh import (
    FAN_SURFACE_MARKER,
    FARFIELD_MARKER,
    VolumeMeshParams,
    VolumeMeshResult,
    build_volume_mesh,
)
from fanopt.cfd.parsers import _THRUST_Z_CANDIDATES, parse_su2_unsteady_force_series
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

# Coarse campaign tier (cheap screening). Stage 3.C re-verifies the winners at the FINE tier below.
_DEMO_CYCLES = 3
_DEMO_INNER = 30

# Fine verification tier (Stage 3.C): more cycles + inner-iterations than the coarse campaign screen.
FINE_CYCLES = 5
FINE_INNER = 60


@dataclass(frozen=True)
class VerifyConfig:
    """Knobs for one 3D verification run. Defaults to the coarse tier; use :meth:`fine` for 3.C."""

    n_cycles: int = _DEMO_CYCLES
    inner_iter: int = _DEMO_INNER
    mesh_params: VolumeMeshParams = field(default_factory=VolumeMeshParams)

    @classmethod
    def fine(cls) -> VerifyConfig:
        """The Stage-3.C fine tier (``FINE_CYCLES`` / ``FINE_INNER``)."""
        return cls(n_cycles=FINE_CYCLES, inner_iter=FINE_INNER)


@dataclass(frozen=True)
class VerifyResult:
    """One design's FINE 3D J_fan next to its COARSE J_fan (the campaign's screening value)."""

    name: str
    j_fan_3d: float
    j_fan_coarse: float | None
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


def extract_j_fan_3d(
    history_csv: Path, *, n_cycles: int = _DEMO_CYCLES, steps_per_cycle: int = STEPS_PER_CYCLE
) -> float:
    """Cycle-mean 3D J_fan from an unsteady history.csv (discard cycle 1).

    The 3D user-ward thrust is +z (CFz): the blade spans +x, pitches about +y, and pushes
    air in ±z. The parser's DEFAULT force column is CFx-first (a legacy of the retired 2D slice),
    WRONG for the 3D blade — so CFz is forced here, matching the campaign objective
    (:mod:`fanopt.cfd.blade_aero_3d` uses the same ``_THRUST_Z_CANDIDATES``). The period
    is the ``dt = T/200`` lock (:data:`STEPS_PER_CYCLE`), NOT inferred from ``series.size``:
    a diverged / early-terminated run is caught and raised instead of silently reshaped into
    misaligned cycles that would launder a garbage value into the ranking.
    """
    series = parse_su2_unsteady_force_series(history_csv, force_candidates=_THRUST_Z_CANDIDATES)
    expected = n_cycles * steps_per_cycle
    if series.size < expected:
        raise ValueError(
            f"{history_csv}: {series.size} time steps < expected {expected} "
            f"({n_cycles}×{steps_per_cycle}) — run incomplete or diverged"
        )
    return reduce_cycles(series[-expected:], steps_per_cycle=steps_per_cycle, n_discard=1).j_fan


@dataclass(frozen=True)
class _VerifyWorker:
    """Picklable per-design 3D verification (for the process pool)."""

    workdir: Path
    cfg: VerifyConfig
    su2_bin: str
    prepare_fn: Callable[[np.ndarray, Path, VerifyConfig], VolumeMeshResult]
    scale_fn: Callable[[object, float], float] | None = None

    def __call__(self, design: tuple[str, np.ndarray, float | None]) -> VerifyResult:
        name, design_input, j_coarse = design
        d_dir = self.workdir / name
        try:
            mesh = self.prepare_fn(design_input, d_dir, self.cfg)
            hist = run_su2(CFG_NAME, d_dir, self.su2_bin)
            j3d = extract_j_fan_3d(hist, n_cycles=self.cfg.n_cycles)
            # extract_j_fan_3d is per-blade; scale_fn (e.g. whole-fan × blade_count) makes it the
            # SAME quantity as j_coarse so the coarse-vs-fine correlation compares like with like.
            if self.scale_fn is not None:
                j3d = self.scale_fn(design_input, j3d)
            return VerifyResult(name, j3d, j_coarse, meta={"n_nodes": float(mesh.n_nodes)})
        except Exception as exc:  # fault isolation: one bad design shouldn't sink the batch
            d_dir.mkdir(parents=True, exist_ok=True)
            (d_dir / "FAILED.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            return VerifyResult(name, float("nan"), j_coarse, meta={"failed": 1.0})


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
    scale_fn: Callable[[object, float], float] | None = None,
) -> list[VerifyResult]:
    """Fine-tier verify each ``(name, design_input, j_fan_coarse)`` design; return the results.

    ``n_workers`` > 1 runs designs concurrently in **separate processes** (gmsh
    can't be threaded; each 3D SU2 run is single-core, so ``n_workers`` ≈ min(
    n_designs, cores) is the useful range). Order is preserved. ``progress`` shows
    a live ``tqdm`` bar over the designs (each 3D run takes a while). ``on_result``,
    if given, is called with each :class:`VerifyResult` as it completes — the caller
    uses it to checkpoint partial results so a mid-run crash/disconnect isn't total
    loss. ``prepare_fn`` builds one design's case (blade → STEP → 3D mesh → cfg); it
    defaults to the original codec-bound blade, and the redesigned aero-first blade
    passes :func:`fanopt.cfd.blade_verify.prepare_blade_verification_case` instead.
    ``scale_fn(design_input, per_blade_j) -> j`` post-scales each fine J_fan (e.g. the
    blade path's whole-fan × blade_count) so it matches the coarse ``j_fan_coarse``; ``None``
    leaves the raw per-blade value. Applied inside the worker, so the checkpoint sees the
    scaled value too.
    """
    su2 = su2_bin or find_su2()
    if su2 is None:
        raise RuntimeError("SU2_CFD not found (set $SU2_RUN or put SU2_CFD on PATH)")
    worker = _VerifyWorker(workdir, cfg, su2, prepare_fn, scale_fn)
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
    """Kendall τ, Spearman ρ, Pearson R² over (coarse, fine) pairs (None if < 2)."""
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
    """Coarse-vs-fine ranking agreement — Kendall τ, Spearman ρ, Pearson R² — with failures flagged.

    A **suspect** design is one whose fine 3D run **failed** (non-finite J_fan — diverged /
    early-terminated), which carries no comparable value. A **negative** fine J_fan is NOT suspect:
    in the MACH=1e-9 body-in-still-air regime a design that produces net force in the non-productive
    direction is a legitimate (bad) result, and — critically — a coarse-positive → fine-negative flip
    is exactly the fidelity failure this check exists to catch, so it MUST stay in the correlation
    rather than be excused. τ therefore spans every design with a finite fine value, including
    negatives. ``rank_preserved`` uses a τ ≥ 0.7 bar (the same threshold as the Stage-2 gate), so a
    weak-but-positive τ doesn't read as "preserved". ``valid_only`` == ``all_finite`` (kept for
    back-compat; negatives are no longer excluded).
    """
    with_coarse = [r for r in results if r.j_fan_coarse is not None]
    finite = [r for r in with_coarse if np.isfinite(r.j_fan_3d)]
    suspect = [r for r in with_coarse if not np.isfinite(r.j_fan_3d)]

    metrics = _rank_metrics([(r.j_fan_coarse, r.j_fan_3d) for r in finite])  # type: ignore[misc]
    tau = metrics["kendall_tau"]
    rank_preserved = bool(tau >= 0.7) if isinstance(tau, float) else None

    return {
        "n": metrics["n"],
        "kendall_tau": tau,
        "rank_preserved": rank_preserved,
        "n_suspect": len(suspect),
        "suspect_designs": [r.name for r in suspect],
        "all_finite": metrics,
        "valid_only": metrics,  # back-compat alias; negatives are no longer excluded
        "pairs": [
            {
                "name": r.name,
                "j_fan_coarse": r.j_fan_coarse,
                "j_fan_3d": r.j_fan_3d if np.isfinite(r.j_fan_3d) else None,
                "suspect": not np.isfinite(r.j_fan_3d),
            }
            for r in with_coarse
        ],
    }
