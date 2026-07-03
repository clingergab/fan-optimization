"""Phase 4 BO campaign orchestration (V1-slim Stage 2).

Drives the relay's expensive stage: a Sobol design-of-experiments seed, then
qLogNEHVI + TuRBO iterations, persisting every evaluation to a JSONL ledger and
checkpointing after each step so a killed Colab session resumes where it left
off. A **BO-stall fallback** (the Spike-0.7c V1 substitute) injects hand-picked
structurally-diverse designs when the hypervolume plateaus.

The objective is **injected** (``objective_fn: vector -> (J_fan, I_wrist,
structural)``) so this module carries no gmsh/CadQuery dependency and is testable
with a cheap synthetic objective; the real CFD objective is wired in
``scripts/run_phase4_bo.py``.
"""

from __future__ import annotations

import datetime as _dt
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
    TrustRegionState,
    fit_gp,
    hypervolume,
    infer_reference_point,
    pareto_mask,
    propose_candidates,
    to_maximization,
)
from fanopt.bo.codec import N_DIMS, bounds, clip_to_bounds, decode, encode
from fanopt.cfd.config_hash import CROSS_TIER_CONFIG_HASH
from fanopt.geometry.envelope import Layer1Params, ThicknessGridField
from fanopt.utils.ledger import SCHEMA_VERSION, LedgerRow, Status, Tier, append_row, design_hash

__all__ = [
    "CampaignConfig",
    "CampaignState",
    "ObjectiveFn",
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
    stall_patience: int = 5  # consecutive no-HV-gain iters before the diverse fallback
    use_trust_region: bool = True
    num_restarts: int = 8
    raw_samples: int = 128
    mc_samples: int = 128
    n_workers: int = 1  # parallel CFD processes (DoE + each batch); ≈ n_cores on Colab


# Frozen shared default so it can sit in an argument default (ruff B008-safe).
_DEFAULT_CAMPAIGN_CFG = CampaignConfig()


@dataclass
class CampaignState:
    """Evolving campaign state (checkpointed each iteration)."""

    x: np.ndarray  # (n, N_DIMS) evaluated design vectors
    y_raw: np.ndarray  # (n, 3) raw objectives (J_fan, I_wrist, structural)
    tr: TrustRegionState
    iteration: int = 0
    stall_counter: int = 0
    used_fallback: int = 0


def sobol_doe(n: int, seed: int = 0) -> np.ndarray:
    """``n`` Sobol design vectors mapped into the search box (categoricals valid)."""
    low, high = bounds()
    unit = Sobol(d=N_DIMS, seed=seed).random(n)
    scaled = low + unit * (high - low)
    return np.array([clip_to_bounds(v) for v in scaled])


def _layer1(field_: ThicknessGridField, *, blade_count: int, camber: float = 0.001) -> Layer1Params:
    return Layer1Params(
        blade_count=blade_count,
        camber_knots_m=(camber, camber, camber),
        twist_knots_rad=(0.0, 0.0),
        thickness_field=field_,
        edge_profile="rounded",
        fourier_le_amplitudes=(0.0, 0.0, 0.0),
        fourier_te_amplitudes=(0.0, 0.0, 0.0),
    )


def diverse_fallback_designs() -> list[np.ndarray]:
    """Structurally-diverse seed vectors for the BO-stall fallback (Spike 0.7c V1).

    Spans the codec's reachable archetypes — flat baseline, thin, thick, heavily
    corrugated, tangentially-asymmetric, high-camber — across blade counts.
    """
    grid_mid = ThicknessGridField.uniform(0.0030).grid_m
    asym = tuple(tuple(0.0022 + 0.0015 * (j / 5.0) for j in range(6)) for _ in range(3))
    designs = [
        _layer1(ThicknessGridField.uniform(0.0030), blade_count=10),  # flat baseline
        _layer1(ThicknessGridField.uniform(0.0022), blade_count=8),  # thin/light
        _layer1(ThicknessGridField.uniform(0.0038), blade_count=12),  # thick/stiff
        _layer1(
            ThicknessGridField(
                grid_m=grid_mid, corrugation_amplitude_m=0.0008, corrugation_wavelength=0.25
            ),
            blade_count=10,
        ),  # heavily corrugated
        _layer1(ThicknessGridField(grid_m=asym), blade_count=10),  # tangentially asymmetric
        _layer1(ThicknessGridField.uniform(0.0030), blade_count=12, camber=0.004),  # high camber
    ]
    return [encode(p) for p in designs]


def _persist_eval(
    ledger_path: Path,
    vector: np.ndarray,
    objectives: tuple[float, float, float],
    *,
    iteration: int,
    wall_time_s: float,
    source: str,
) -> None:
    """Append one evaluation to the JSONL ledger as a real :class:`LedgerRow`."""
    j_fan, i_wrist, structural = objectives
    params = decode(vector).to_dict()
    row = LedgerRow(
        schema_version=SCHEMA_VERSION,
        design_hash=design_hash(params),
        tier=Tier.TIER_MINUS_ONE,  # 2D slice fidelity
        status=Status.OK,
        wall_time_s=wall_time_s,
        timestamp_iso=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        config_hash=CROSS_TIER_CONFIG_HASH,
        J_fan=float(j_fan),
        I_wrist_kgm2=float(i_wrist),
        params={"iteration": iteration, "source": source, "structural_m": float(structural)},
    )
    append_row(ledger_path, row)


def _save_checkpoint(path: Path, state: CampaignState) -> None:
    np.savez(
        path,
        x=state.x,
        y_raw=state.y_raw,
        iteration=state.iteration,
        stall_counter=state.stall_counter,
        used_fallback=state.used_fallback,
        tr_length=state.tr.length,
        tr_success=state.tr.success_counter,
        tr_failure=state.tr.failure_counter,
        tr_batch=state.tr.batch_size,
    )


def _load_checkpoint(path: Path) -> CampaignState:
    d = np.load(path)
    tr = TrustRegionState(dim=N_DIMS, batch_size=int(d["tr_batch"]), length=float(d["tr_length"]))
    tr.success_counter = int(d["tr_success"])
    tr.failure_counter = int(d["tr_failure"])
    return CampaignState(
        x=d["x"],
        y_raw=d["y_raw"],
        tr=tr,
        iteration=int(d["iteration"]),
        stall_counter=int(d["stall_counter"]),
        used_fallback=int(d["used_fallback"]),
    )


def pareto_designs(state: CampaignState) -> list[dict[str, object]]:
    """The non-dominated designs (vectors + raw objectives) found so far."""
    mask = pareto_mask(to_maximization(state.y_raw))
    out: list[dict[str, object]] = []
    for i in np.where(mask)[0]:
        j_fan, i_wrist, structural = state.y_raw[i]
        out.append(
            {
                "vector": state.x[i].tolist(),
                "j_fan": float(j_fan),
                "i_wrist_kgm2": float(i_wrist),
                "structural_m": float(structural),
                "blade_count": decode(state.x[i]).blade_count,
            }
        )
    return out


def _evaluate_batch(
    objective_fn: ObjectiveFn,
    ledger_path: Path,
    batch: np.ndarray,
    *,
    iteration: int,
    source: str,
    n_workers: int = 1,
    on_eval: Callable[[], None] | None = None,
) -> np.ndarray:
    """Evaluate a batch (optionally across ``n_workers`` processes) and log each.

    **Processes, not threads:** gmsh installs a main-thread-only signal handler
    and keeps a single global model, so the CFD objective cannot run in threads.
    A process pool isolates each worker; the objective must therefore be picklable
    (:class:`fanopt.bo.cfd_objective.CfdObjective`). Results are written to the
    ledger in batch order; ``on_eval`` fires per completed eval (in the main
    process, via ``as_completed``) for a live progress bar.
    """
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
        assert y is not None  # every slot filled by the loop above
        _persist_eval(
            ledger_path, batch[i], y, iteration=iteration, wall_time_s=times[i], source=source
        )
        out.append(y)
    return np.atleast_2d(np.asarray(out, dtype=float))


def run_campaign(
    objective_fn: ObjectiveFn,
    out_dir: Path,
    cfg: CampaignConfig = _DEFAULT_CAMPAIGN_CFG,
    *,
    resume: bool = True,
    progress: bool = False,
) -> CampaignState:
    """Run (or resume) the BO campaign, persisting to ``out_dir``.

    ``progress`` shows a live ``tqdm`` bar (notebook widget or terminal text) over
    the total evaluations, with the best ``J_fan`` so far. Returns the final
    :class:`CampaignState`; the Pareto set is :func:`pareto_designs` of it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / LEDGER_NAME
    ckpt = out_dir / CHECKPOINT_NAME
    low, high = bounds()

    total_evals = cfg.n_init + cfg.n_iterations * cfg.batch_size
    bar = tqdm(total=total_evals, disable=not progress, desc="Phase 4 BO", unit="eval")
    try:
        if resume and ckpt.exists():
            state = _load_checkpoint(ckpt)
            bar.update(int(state.x.shape[0]))  # already-completed evals
        else:
            x0 = sobol_doe(cfg.n_init, cfg.seed)
            y0 = _evaluate_batch(
                objective_fn,
                ledger_path,
                x0,
                iteration=0,
                source="sobol",
                n_workers=cfg.n_workers,
                on_eval=lambda: bar.update(1),
            )
            state = CampaignState(
                x=x0, y_raw=y0, tr=TrustRegionState(dim=N_DIMS, batch_size=cfg.batch_size)
            )
            _save_checkpoint(ckpt, state)

        fallback = deque(diverse_fallback_designs())

        while state.iteration < cfg.n_iterations:
            y_max = to_maximization(state.y_raw)
            ref = infer_reference_point(y_max)
            hv_before = hypervolume(y_max, ref)

            stalled = cfg.stall_patience > 0 and state.stall_counter >= cfg.stall_patience
            if stalled and fallback:
                batch = np.atleast_2d(clip_to_bounds(fallback.popleft()))
                source = "fallback"
                state.used_fallback += 1
            else:
                model = fit_gp(state.x, y_max, low, high)
                proposed = propose_candidates(
                    model,
                    state.x,
                    y_max,
                    low,
                    high,
                    ref,
                    batch_size=cfg.batch_size,
                    tr_state=state.tr if cfg.use_trust_region else None,
                    num_restarts=cfg.num_restarts,
                    raw_samples=cfg.raw_samples,
                    mc_samples=cfg.mc_samples,
                )
                batch = np.array([clip_to_bounds(c) for c in proposed])
                source = "bo"

            state.iteration += 1
            y_new = _evaluate_batch(
                objective_fn,
                ledger_path,
                batch,
                iteration=state.iteration,
                source=source,
                n_workers=cfg.n_workers,
                on_eval=lambda: bar.update(1),
            )
            state.x = np.vstack([state.x, batch])
            state.y_raw = np.vstack([state.y_raw, y_new])

            hv_after = hypervolume(to_maximization(state.y_raw), ref)
            improved = hv_after > hv_before + 1e-12
            if cfg.use_trust_region:
                state.tr.update(improved)
            state.stall_counter = 0 if improved else state.stall_counter + 1
            _save_checkpoint(ckpt, state)
            bar.set_postfix(iter=state.iteration, best_J_fan=f"{state.y_raw[:, 0].max():.2e}")
    finally:
        bar.close()

    return state
