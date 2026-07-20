"""Aero-first BO campaign for the redesigned blade (V1).

The Phase-4 campaign loop for the new blade: a Sobol design-of-experiments seed, then
qLogNEHVI + TuRBO iterations over the 18-var :mod:`~fanopt.bo.blade_codec`, maximizing
``J_fan`` while minimizing mass + deflection. Reuses the kept, codec-agnostic
:mod:`~fanopt.bo.backbone` (GP fit, acquisition, trust region, hypervolume) — this module
only wires the loop: DoE, evaluate (optionally across processes), normalize, propose,
checkpoint + JSONL ledger every iteration so a killed Colab session resumes.

The objective is **injected** (``objective_fn: vector -> (J_fan, mass, deflection)``) so
this module carries no gmsh/SU2 dependency and is testable with a cheap synthetic
objective; the real CFD objective is :class:`~fanopt.bo.blade_objective.BladeObjective`,
wired in ``scripts/run_phase4_aero.py``.
"""

from __future__ import annotations

import datetime as _dt
import json
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats.qmc import Sobol
from tqdm.auto import tqdm

from fanopt.bo.backbone import (
    OBJECTIVE_SIGNS,
    TrustRegionState,
    apply_objective_norm,
    fit_gp,
    hypervolume,
    infer_reference_point,
    normalize_objectives,
    pareto_mask,
    propose_candidates,
    sanitize_objectives,
    to_maximization,
)
from fanopt.bo.blade_codec import N_DIMS, bounds, clip_to_bounds, decode, encode
from fanopt.geometry.blade import BladeParams
from fanopt.utils.ledger import design_hash

__all__ = [
    "CampaignConfig",
    "CampaignState",
    "ObjectiveFn",
    "CHECKPOINT_NAME",
    "LEDGER_NAME",
    "sobol_doe",
    "diverse_fallback_designs",
    "pareto_designs",
    "run_campaign",
]

ObjectiveFn = Callable[[np.ndarray], tuple[float, float, float]]

CHECKPOINT_NAME = "checkpoint.npz"
LEDGER_NAME = "evaluations.jsonl"


@dataclass(frozen=True)
class CampaignConfig:
    """Campaign knobs."""

    n_init: int = 8
    n_iterations: int = 20
    batch_size: int = 1
    seed: int = 0
    stall_patience: int = 5
    use_trust_region: bool = True
    num_restarts: int = 8
    raw_samples: int = 128
    mc_samples: int = 128
    n_workers: int = 1


_DEFAULT_CFG = CampaignConfig()


@dataclass
class CampaignState:
    """Evolving campaign state (checkpointed each iteration)."""

    x: np.ndarray  # (n, N_DIMS)
    y_raw: np.ndarray  # (n, 3) — (J_fan, mass, deflection)
    tr: TrustRegionState
    iteration: int = 0
    stall_counter: int = 0
    used_fallback: int = 0


def sobol_doe(n: int, seed: int = 0) -> np.ndarray:
    """``n`` Sobol design vectors mapped into the blade search box (categoricals valid)."""
    low, high = bounds()
    unit = Sobol(d=N_DIMS, seed=seed).random(n)
    scaled = low + unit * (high - low)
    return np.array([clip_to_bounds(v) for v in scaled])


def _blade(grid, *, blade_count=10, t_hub=0.0025, t_tip=0.0035, panel=0.0015) -> BladeParams:
    return BladeParams(
        blade_count=blade_count,
        rib_bow_mid_m=0.010,
        rib_bow_tip_m=0.020,
        t_rib_hub_m=t_hub,
        t_rib_tip_m=t_tip,
        panel_offsets_m=grid,
        panel_thickness_nom_m=panel,
    )


def diverse_fallback_designs() -> list[np.ndarray]:
    """Structurally-diverse seed vectors for the BO-stall fallback.

    Spans the reachable aero archetypes — flat baseline, cambered, base→tip zigzag, thin,
    thick — across blade counts, so a stalled optimizer explores rather than exploits.
    """
    flat = tuple((0.0, 0.0, 0.0) for _ in range(4))
    camber = ((0.0004, 0.0009, 0.0004), (0.0005, 0.0011, 0.0005),
              (0.0006, 0.0013, 0.0006), (0.0007, 0.0015, 0.0007))
    zig = ((0.0008, -0.0008, 0.0008), (0.0009, -0.0009, 0.0009),
           (0.0010, -0.0010, 0.0010), (0.0011, -0.0011, 0.0011))
    designs = [
        _blade(flat, blade_count=10),
        _blade(camber, blade_count=10),
        _blade(zig, blade_count=8, t_hub=0.003, t_tip=0.004),
        _blade(flat, blade_count=8, panel=0.0012),  # thin/light
        _blade(camber, blade_count=12, t_hub=0.003, t_tip=0.0045),  # thick/stiff
    ]
    return [encode(p) for p in designs]


def _sanitize_yraw(y_raw: np.ndarray) -> np.ndarray:
    signs = np.asarray(OBJECTIVE_SIGNS)
    return sanitize_objectives(to_maximization(y_raw)) * signs


def _append_ledger(
    ledger_path: Path,
    vector: np.ndarray,
    objectives: tuple[float, float, float],
    *,
    iteration: int,
    source: str,
    wall_time_s: float,
) -> None:
    params = decode(vector)
    j_fan, mass, deflection = objectives
    row = {
        "design_hash": design_hash(params.to_dict()),
        "iteration": iteration,
        "source": source,
        "wall_time_s": wall_time_s,
        "timestamp_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "j_fan": float(j_fan),
        "mass_kg": float(mass),
        "deflection_m": float(deflection),
        "blade_count": params.blade_count,
    }
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _evaluate_batch(
    objective_fn: ObjectiveFn,
    ledger_path: Path,
    batch: np.ndarray,
    *,
    iteration: int,
    source: str,
    n_workers: int,
    on_eval: Callable[[], None] | None,
) -> np.ndarray:
    """Evaluate a batch (optionally across processes) and log each to the ledger."""
    n = len(batch)
    ys: list[tuple[float, float, float] | None] = [None] * n
    times: list[float] = [0.0] * n
    if n_workers > 1 and n > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            fut_to_i = {pool.submit(objective_fn, batch[i]): i for i in range(n)}
            for fut in as_completed(fut_to_i):
                ys[fut_to_i[fut]] = fut.result()
                if on_eval is not None:
                    on_eval()
    else:
        for i in range(n):
            t0 = time.perf_counter()
            ys[i] = objective_fn(batch[i])
            times[i] = time.perf_counter() - t0
            if on_eval is not None:
                on_eval()
    out: list[tuple[float, float, float]] = []
    for i in range(n):
        y = ys[i]
        assert y is not None
        _append_ledger(ledger_path, batch[i], y, iteration=iteration, source=source,
                       wall_time_s=times[i])
        out.append(y)
    return np.atleast_2d(np.asarray(out, dtype=float))


def _save_checkpoint(path: Path, state: CampaignState) -> None:
    np.savez(
        path, x=state.x, y_raw=state.y_raw, iteration=state.iteration,
        stall_counter=state.stall_counter, used_fallback=state.used_fallback,
        tr_length=state.tr.length, tr_success=state.tr.success_counter,
        tr_failure=state.tr.failure_counter, tr_batch=state.tr.batch_size,
    )


def _load_checkpoint(path: Path) -> CampaignState:
    d = np.load(path)
    tr = TrustRegionState(dim=N_DIMS, batch_size=int(d["tr_batch"]), length=float(d["tr_length"]))
    tr.success_counter = int(d["tr_success"])
    tr.failure_counter = int(d["tr_failure"])
    return CampaignState(
        x=d["x"], y_raw=d["y_raw"], tr=tr, iteration=int(d["iteration"]),
        stall_counter=int(d["stall_counter"]), used_fallback=int(d["used_fallback"]),
    )


def pareto_designs(state: CampaignState) -> list[dict[str, object]]:
    """The non-dominated designs (vectors + raw objectives + decoded params) found so far."""
    mask = pareto_mask(to_maximization(state.y_raw))
    out: list[dict[str, object]] = []
    for i in np.where(mask)[0]:
        j_fan, mass, deflection = state.y_raw[i]
        out.append({
            "vector": state.x[i].tolist(),
            "j_fan": float(j_fan),
            "mass_kg": float(mass),
            "deflection_m": float(deflection),
            "params": decode(state.x[i]).to_dict(),
        })
    return out


def run_campaign(
    objective_fn: ObjectiveFn,
    out_dir: Path,
    cfg: CampaignConfig = _DEFAULT_CFG,
    *,
    resume: bool = True,
    progress: bool = False,
) -> CampaignState:
    """Run (or resume) the aero-first BO campaign, persisting to ``out_dir``.

    Returns the final :class:`CampaignState`; the Pareto set is :func:`pareto_designs`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / LEDGER_NAME
    ckpt = out_dir / CHECKPOINT_NAME
    low, high = bounds()

    total = cfg.n_init + cfg.n_iterations * cfg.batch_size
    bar = tqdm(total=total, disable=not progress, desc="Phase 4 aero BO", unit="eval")
    try:
        if resume and ckpt.exists():
            state = _load_checkpoint(ckpt)
            bar.update(int(state.x.shape[0]))
        else:
            x0 = sobol_doe(cfg.n_init, cfg.seed)
            y0 = _evaluate_batch(objective_fn, ledger_path, x0, iteration=0, source="sobol",
                                 n_workers=cfg.n_workers, on_eval=lambda: bar.update(1))
            state = CampaignState(
                x=x0, y_raw=_sanitize_yraw(y0),
                tr=TrustRegionState(dim=N_DIMS, batch_size=cfg.batch_size),
            )
            _save_checkpoint(ckpt, state)

        fallback = deque(diverse_fallback_designs())

        while state.iteration < cfg.n_iterations:
            y_norm, loc, scale = normalize_objectives(to_maximization(state.y_raw))
            ref = infer_reference_point(y_norm)
            hv_before = hypervolume(y_norm, ref)

            stalled = cfg.stall_patience > 0 and state.stall_counter >= cfg.stall_patience
            if stalled and fallback:
                batch = np.atleast_2d(clip_to_bounds(fallback.popleft()))
                source = "fallback"
                state.used_fallback += 1
            else:
                model = fit_gp(state.x, y_norm, low, high)
                proposed = propose_candidates(
                    model, state.x, y_norm, low, high, ref, batch_size=cfg.batch_size,
                    tr_state=state.tr if cfg.use_trust_region else None,
                    num_restarts=cfg.num_restarts, raw_samples=cfg.raw_samples,
                    mc_samples=cfg.mc_samples,
                )
                batch = np.array([clip_to_bounds(c) for c in proposed])
                source = "bo"

            state.iteration += 1
            t_batch = time.perf_counter()
            y_new = _evaluate_batch(objective_fn, ledger_path, batch, iteration=state.iteration,
                                    source=source, n_workers=cfg.n_workers,
                                    on_eval=lambda: bar.update(1))
            # Real batch throughput — the unambiguous parallelism signal: a batch of B
            # should finish in ≈ one eval-time if parallel, ≈ B·eval-time if serial.
            batch_s = time.perf_counter() - t_batch
            evals_per_min = len(batch) / max(batch_s / 60.0, 1e-9)
            state.x = np.vstack([state.x, batch])
            state.y_raw = _sanitize_yraw(np.vstack([state.y_raw, y_new]))

            hv_after = hypervolume(
                apply_objective_norm(to_maximization(state.y_raw), loc, scale), ref
            )
            improved = hv_after > hv_before + 1e-12
            if cfg.use_trust_region:
                state.tr.update(improved)
            state.stall_counter = 0 if improved else state.stall_counter + 1
            _save_checkpoint(ckpt, state)
            bar.set_postfix(
                iter=state.iteration,
                best_J_fan=f"{state.y_raw[:, 0].max():.2e}",
                batch=f"{len(batch)}/{batch_s:.0f}s",
                ev_per_min=f"{evals_per_min:.1f}",
            )
    finally:
        bar.close()
    return state
