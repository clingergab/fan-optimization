"""Async shared-ledger distributed BO for the aero-first campaign (Stage 3).

Multiple Colab sessions optimize the SAME objective concurrently **without communicating** —
they share only a Drive directory. Each session, every iteration:

1. reads ALL evaluations from the shared ledger (every session's per-session shard),
2. refits the GP on the **combined** data (shared learning, not N blind runs),
3. proposes a batch, **seeded per session** so sessions diverge instead of duplicating,
4. **claims** each design via an atomic marker file (so two sessions never run the same one),
5. evaluates the claimed designs and appends them to **its own** ledger shard.

**Robustness for a multi-day run on Drive:**
- *Per-session shards* (``evaluations_<id>.jsonl``): a session only ever appends to its own file,
  so concurrent writes on Drive can't clobber each other's rows. ``read_ledger`` globs + dedups
  all shards, so a design that slips through the claim (Drive's create-exclusive is not perfectly
  atomic across VMs) is deduped on read — at worst a wasted eval, never lost or double-counted data.
- *Stale-claim reclaim + DoE mop-up*: a claim marker carries a timestamp; a claim older than
  ``claim_ttl_seconds`` (its session died mid-eval — the expected Colab-drop failure) can be
  stolen, and once a session finishes its own cold-start slice it mops up ANY un-evaluated DoE
  point. Together these keep cold-start from deadlocking as long as ≥1 session survives — a
  permanently-departed session's DoE slice is finished by the survivors once its claims go stale.

The ledger IS the state — a session joins/resumes just by reading it; no per-session checkpoint.
Cold-start: the first ``n_init`` evals are a shared Sobol DoE, sliced round-robin across sessions.
Reuses the codec-agnostic :mod:`~fanopt.bo.backbone`; the objective
(``vector -> (J_fan, mass, deflection)`` in the *raw* frame — J_fan maximized, mass/deflection
minimized) is injected, so this module carries no CFD dependency and is tested synthetically.
"""

from __future__ import annotations

import datetime as _dt
import json
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats.qmc import Sobol

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
from fanopt.bo.blade_codec import N_DIMS, bounds, clip_to_bounds, decode
from fanopt.utils.ledger import design_hash

__all__ = [
    "DistributedConfig",
    "LEDGER_GLOB",
    "CLAIMS_DIR",
    "shard_path",
    "read_ledger",
    "claim_designs",
    "append_eval",
    "pareto_from_ledger",
    "run_distributed_session",
]

LEDGER_GLOB = "evaluations_*.jsonl"
CLAIMS_DIR = "claims"

ObjectiveFn = Callable[[np.ndarray], tuple[float, float, float]]


@dataclass(frozen=True)
class DistributedConfig:
    """Knobs for one distributed session (identical across sessions except the id/index)."""

    total_budget: int = 300  # stop when the SHARED ledger reaches this many unique evaluations
    n_init: int = 16  # shared Sobol DoE size (cold-start), sliced round-robin across sessions
    batch_size: int = 8  # designs a session proposes+claims per iteration
    seed: int = 0  # shared Sobol seed (same across sessions so the DoE is one sequence)
    num_restarts: int = 8
    raw_samples: int = 128
    mc_samples: int = 128
    n_workers: int = 1
    use_trust_region: bool = True
    poll_seconds: float = 5.0  # sleep when there's nothing to claim (another session took it)
    claim_ttl_seconds: float = (
        # Steal a claim older than this (its session died). MUST exceed the worst-case eval wall
        # time or a live claim gets stolen → a redundant eval (deduped on read, but wasted). 6h
        # clears the ~4h high-fi CFD tier; lower it for a fast coarse tier to recover from drops sooner.
        21600.0
    )


_DEFAULT_CFG = DistributedConfig()


def shard_path(shared_dir: Path | str, session_id: str) -> Path:
    """This session's private ledger shard — it only ever appends here (no cross-session clobber)."""
    return Path(shared_dir) / f"evaluations_{session_id}.jsonl"


def _iter_rows(shared_dir: Path):
    for shard in sorted(Path(shared_dir).glob(LEDGER_GLOB)):
        for line in shard.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                vec = [float(v) for v in r["vector"]]
                if len(vec) != N_DIMS:
                    continue  # schema drift / clobbered merge — drop, don't crash the read (L1)
                yield str(r["design_hash"]), vec, [
                    float(r["j_fan"]),
                    float(r["mass_kg"]),
                    float(r["deflection_m"]),
                ]
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                continue  # a torn concurrent write; the row lands on the next read


def read_ledger(shared_dir: Path | str) -> tuple[np.ndarray, np.ndarray, set[str]]:
    """``(x (n,N_DIMS), y_raw (n,3), hashes)`` deduped across every session shard in ``shared_dir``.

    Deduped by ``design_hash`` (first occurrence wins), so a design that two sessions raced past the
    claim appears once — ``len(x) == len(hashes)`` always. Missing dir → empty. This is the whole
    campaign state; no checkpoint needed.
    """
    xs: list[list[float]] = []
    ys: list[list[float]] = []
    hashes: set[str] = set()
    if Path(shared_dir).exists():
        for h, vec, y in _iter_rows(shared_dir):
            if h in hashes:
                continue
            hashes.add(h)
            xs.append(vec)
            ys.append(y)
    x = np.array(xs, dtype=float).reshape(-1, N_DIMS)
    y = np.array(ys, dtype=float).reshape(-1, 3)
    return x, y, hashes


def _claim_one(claims_dir: Path, h: str, ttl_seconds: float) -> bool:
    """Atomically claim design hash ``h``; steal it if the existing claim is older than ``ttl``.

    ``True`` iff we hold the claim. A stale marker means its session died mid-eval (Colab drop) —
    stealing it lets the design (esp. a cold-start DoE point) be evaluated instead of deadlocking.
    """
    claims_dir.mkdir(parents=True, exist_ok=True)
    marker = claims_dir / f"{h}.claim"
    try:
        with open(marker, "x", encoding="utf-8") as f:  # exclusive create
            f.write(_dt.datetime.now(_dt.timezone.utc).isoformat())
        return True
    except FileExistsError:
        try:
            age = time.time() - marker.stat().st_mtime
        except FileNotFoundError:
            return False  # vanished under us; another session owns it now
        if age <= ttl_seconds:
            return False  # a live claim — leave it
        try:  # stale: steal it (owner likely dead)
            marker.unlink()
            with open(marker, "x", encoding="utf-8") as f:
                f.write(_dt.datetime.now(_dt.timezone.utc).isoformat())
            return True
        except (FileNotFoundError, FileExistsError):
            return False  # another session stole it first


def claim_designs(
    claims_dir: Path,
    ledger_hashes: set[str],
    batch: np.ndarray,
    ttl_seconds: float = _DEFAULT_CFG.claim_ttl_seconds,
) -> tuple[np.ndarray, list[str]]:
    """Keep only batch designs not already evaluated or actively claimed; claim them.

    Returns ``(claimed_vectors, claimed_hashes)``. Skips a design already in the ledger and one with
    a live claim marker; steals a stale claim (dead session). So no two live sessions run the same one.
    """
    keep: list[np.ndarray] = []
    keep_h: list[str] = []
    for v in np.atleast_2d(batch):
        h = design_hash(decode(v).to_dict())
        if h in ledger_hashes or h in keep_h:
            continue
        if _claim_one(Path(claims_dir), h, ttl_seconds):
            keep.append(v)
            keep_h.append(h)
    x = np.array(keep, dtype=float).reshape(-1, N_DIMS)
    return x, keep_h


def append_eval(
    shard: Path, vector: np.ndarray, y: tuple[float, float, float], *, session_id: str, source: str
) -> None:
    """Append one evaluation to this session's shard (with its vector, so any session reconstructs x)."""
    params = decode(vector)
    row = {
        "design_hash": design_hash(params.to_dict()),
        "vector": [float(v) for v in np.asarray(vector, dtype=float)],
        "session_id": session_id,
        "source": source,
        "timestamp_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "j_fan": float(y[0]),
        "mass_kg": float(y[1]),
        "deflection_m": float(y[2]),
        "blade_count": params.blade_count,
    }
    with open(shard, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _y_max(y_raw: np.ndarray) -> np.ndarray:
    """Raw objectives → sanitized **maximization** frame (all columns higher = better) for the GP."""
    return sanitize_objectives(to_maximization(y_raw))


def _sanitize_yraw(y_raw: np.ndarray) -> np.ndarray:
    """Raw objectives, NaN-sanitized to a dominated penalty, kept in the raw (return) frame."""
    return sanitize_objectives(to_maximization(y_raw)) * np.asarray(OBJECTIVE_SIGNS)


def _sobol_doe(n: int, seed: int) -> np.ndarray:
    low, high = bounds()
    unit = Sobol(d=N_DIMS, seed=seed).random(n)
    return np.array([clip_to_bounds(v) for v in (low + unit * (high - low))])


def _safe_eval(objective_fn: ObjectiveFn, v: np.ndarray) -> tuple[float, float, float]:
    try:
        return objective_fn(v)
    except Exception:  # one bad eval must not kill the session (would compound orphan claims)
        nan = float("nan")
        return (nan, nan, nan)


def _evaluate(objective_fn: ObjectiveFn, batch: np.ndarray, n_workers: int) -> list[tuple]:
    if n_workers > 1 and len(batch) > 1:
        out: list = [None] * len(batch)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_safe_eval, objective_fn, batch[i]): i for i in range(len(batch))}
            for fut in as_completed(futs):
                out[futs[fut]] = fut.result()
        return out
    return [_safe_eval(objective_fn, v) for v in batch]


def pareto_from_ledger(shared_dir: Path | str) -> list[dict[str, object]]:
    """Non-dominated designs across ALL sessions' evaluations in the shared ledger.

    Non-finite rows (diverged/failed CFD → NaN objectives, which ``_safe_eval`` persists) are
    dropped first — a failed design is not a candidate, and NaN is neither dominated nor dominating
    so it would otherwise be reported as a spurious Pareto point.
    """
    x, y_raw, _ = read_ledger(shared_dir)
    finite = np.isfinite(y_raw).all(axis=1) if len(y_raw) else np.zeros(0, dtype=bool)
    x, y_raw = x[finite], y_raw[finite]
    if len(x) == 0:
        return []
    mask = pareto_mask(to_maximization(y_raw))
    out: list[dict[str, object]] = []
    for i in np.where(mask)[0]:
        out.append(
            {
                "vector": x[i].tolist(),
                "j_fan": float(y_raw[i, 0]),
                "mass_kg": float(y_raw[i, 1]),
                "deflection_m": float(y_raw[i, 2]),
                "params": decode(x[i]).to_dict(),
            }
        )
    return out


def run_distributed_session(
    objective_fn: ObjectiveFn,
    shared_dir: Path | str,
    cfg: DistributedConfig = _DEFAULT_CFG,
    *,
    session_id: str,
    session_index: int = 0,
    n_sessions: int = 1,
    max_iters: int = 10_000,
    on_batch: Callable[[int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one async session until the SHARED ledger reaches ``cfg.total_budget`` unique evals.

    ``session_index`` / ``n_sessions`` slice the cold-start Sobol DoE round-robin so sessions don't
    duplicate it; in the BO phase each session seeds the acquisition by ``(session_index, ledger
    size)`` so it proposes *different* candidates from the same GP. Returns the final combined
    ``(x, y_raw)`` (raw frame). Raises ``ValueError`` on an out-of-range ``session_index``.
    """
    if n_sessions < 1 or not (0 <= session_index < n_sessions):
        raise ValueError(
            f"session_index must be in [0, {n_sessions}); got {session_index} with n_sessions={n_sessions}"
        )
    shared = Path(shared_dir)
    shared.mkdir(parents=True, exist_ok=True)
    shard = shard_path(shared, session_id)
    claims = shared / CLAIMS_DIR
    low, high = bounds()
    tr = TrustRegionState(dim=N_DIMS, batch_size=cfg.batch_size)
    doe = _sobol_doe(cfg.n_init, cfg.seed)

    for _ in range(max_iters):
        x, y_raw, hashes = read_ledger(shared)
        if len(x) >= cfg.total_budget:
            break

        bo_ctx: tuple | None = None
        if len(x) < cfg.n_init:
            # Cold-start DoE: this session owns Sobol points i where i % n_sessions == index;
            # propose the next ones NOT yet in the shared ledger, so it advances through the DoE
            # instead of re-proposing an already-evaluated batch (which would stall on claims).
            idxs = [i for i in range(cfg.n_init) if i % n_sessions == session_index]
            fresh = [doe[i] for i in idxs if design_hash(decode(doe[i]).to_dict()) not in hashes]
            if not fresh:
                # Own slice done but ledger < n_init: mop up ANY un-evaluated DoE point, so a
                # permanently-departed session's slice can't deadlock cold-start. claim_designs
                # skips points a live session still holds and steals only stale (dead) claims, so
                # this never double-runs live work — it just lets survivors finish the DoE.
                fresh = [
                    doe[i]
                    for i in range(cfg.n_init)
                    if design_hash(decode(doe[i]).to_dict()) not in hashes
                ]
            if not fresh:
                if cfg.poll_seconds > 0:  # whole DoE claimed/evaluated; wait for it to land
                    time.sleep(cfg.poll_seconds)
                continue
            proposed = np.array(fresh[: cfg.batch_size])
            source = "sobol"
        else:
            y_max = _y_max(y_raw)
            y_norm, loc, scale = normalize_objectives(y_max)
            ref = infer_reference_point(y_norm)
            hv_before = hypervolume(y_norm, ref)
            torch.manual_seed((session_index + 1) * 100_003 + len(x))  # per-session diversity
            model = fit_gp(x, y_norm, low, high)
            cand = propose_candidates(
                model,
                x,
                y_norm,
                low,
                high,
                ref,
                batch_size=cfg.batch_size,
                tr_state=tr if cfg.use_trust_region else None,
                num_restarts=cfg.num_restarts,
                raw_samples=cfg.raw_samples,
                mc_samples=cfg.mc_samples,
            )
            proposed = np.array([clip_to_bounds(c) for c in cand])
            source = "bo"
            bo_ctx = (loc, scale, ref, hv_before)

        claimed, _ = claim_designs(claims, hashes, proposed, cfg.claim_ttl_seconds)
        if len(claimed) == 0:
            if cfg.poll_seconds > 0:
                time.sleep(cfg.poll_seconds)  # another session took these; re-read and retry
            continue

        ys = _evaluate(objective_fn, claimed, cfg.n_workers)
        for v, yy in zip(claimed, ys, strict=True):
            append_eval(shard, v, yy, session_id=session_id, source=source)

        if bo_ctx is not None and cfg.use_trust_region:  # advance TuRBO (shrink/grow) on progress
            loc, scale, ref, hv_before = bo_ctx
            y_all = _y_max(np.vstack([y_raw, np.array(ys, dtype=float)]))
            hv_after = hypervolume(apply_objective_norm(y_all, loc, scale), ref)
            tr.update(hv_after > hv_before + 1e-12)

        if on_batch is not None:
            on_batch(len(claimed))

    x, y_raw, _ = read_ledger(shared)
    return x, _sanitize_yraw(y_raw) if len(y_raw) else y_raw
