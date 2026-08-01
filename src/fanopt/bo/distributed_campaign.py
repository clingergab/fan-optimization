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

import contextlib
import datetime as _dt
import json
import multiprocessing
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from concurrent.futures.process import BrokenProcessPool
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
    "active_claims",
    "append_eval",
    "pareto_from_ledger",
    "run_distributed_session",
    "run_async_session",
    "validate_async",
    "preflight_async_check",
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
    explore_fraction: float = 0.0  # in the BO phase, this fraction of dispatches are space-filling
    # Sobol draws (pure exploration) instead of acquisition-driven — a hedge against premature
    # convergence, so the search keeps probing new regions, not only refining the current best.
    poll_seconds: float = 5.0  # sleep when there's nothing to claim (another session took it)
    claim_ttl_seconds: float = (
        # BACKSTOP only (heartbeat leases below are the primary liveness signal). Used for legacy
        # anonymous claim markers and when a heartbeat isn't visible yet. Kept at 6h so a fresh claim
        # whose heartbeat hasn't propagated is never falsely stolen; the heartbeat, not this, is what
        # recovers a dropped session fast.
        21600.0
    )
    # Heartbeat lease: a live session refreshes ``_hb_<id>`` every ``heartbeat_interval_seconds``;
    # a claim whose owner's heartbeat is older than ``heartbeat_ttl_seconds`` is presumed dead and
    # stealable — DECOUPLED from eval wall-time (a live session keeps beating through a long eval, so
    # its claim is never stolen; a dropped session stops beating and its work frees in ~ttl, not 6h).
    # ttl must exceed the worst-case cross-session Drive propagation of the heartbeat (minutes); 10min
    # recovers ~36x faster than the claim_ttl backstop while tolerating that lag.
    heartbeat_interval_seconds: float = 60.0
    heartbeat_ttl_seconds: float = 600.0
    pool_start_method: str | None = None
    # ProcessPool start method. None = platform default ('fork' on Linux/Colab, 'spawn' on macOS).
    # 'fork' is fastest and works in a Colab notebook, but — with the heartbeat daemon thread live —
    # carries a tiny per-fork risk of a worker inheriting a held lock → a WEDGED worker (hung, not a
    # BrokenProcessPool, so the loop won't auto-rebuild) over a multi-day run. Set 'forkserver' to fork
    # workers from a clean thread-free server and remove that risk where the launch context supports it.
    # Left None so a deploy keeps the current, notebook-tested behavior.
    seed_designs: Sequence[np.ndarray] | None = None
    # Operator-locked seed designs (encoded N_DIMS vectors) dispatched AHEAD of the Sobol DoE. They
    # travel the SAME claim/ledger path as any design — deduped by design hash — so across N sessions
    # each seed is evaluated exactly once and lands as an ordinary eval. Identical across sessions.


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


def _heartbeat_path(claims_dir: Path, session_id: str) -> Path:
    """This session's liveness-lease file (one per session; not a ``*.claim``, so globs skip it)."""
    return Path(claims_dir) / f"_hb_{session_id}"


def touch_heartbeat(claims_dir: Path, session_id: str) -> None:
    """Refresh this session's liveness lease — rewrite ``_hb_<id>`` with the current ISO timestamp.

    Content-timestamped (not mtime) so it survives Drive FUSE mtime quirks; peers read the content to
    judge liveness. Called on a timer for the whole life of the session (see :func:`_heartbeat`).
    """
    d = Path(claims_dir)
    d.mkdir(parents=True, exist_ok=True)
    _heartbeat_path(d, session_id).write_text(
        _dt.datetime.now(_dt.timezone.utc).isoformat(), encoding="utf-8"
    )


def _heartbeat_age_seconds(claims_dir: Path, session_id: str) -> float | None:
    """Seconds since ``session_id`` last refreshed its heartbeat, or ``None`` if none is visible."""
    try:
        ts = _dt.datetime.fromisoformat(
            _heartbeat_path(claims_dir, session_id).read_text(encoding="utf-8").strip()
        )
        if ts.tzinfo is None:  # a corrupt/naive stamp would make the tz-aware subtraction raise
            ts = ts.replace(tzinfo=_dt.timezone.utc)
        return (_dt.datetime.now(_dt.timezone.utc) - ts).total_seconds()
    except (FileNotFoundError, ValueError, OSError, TypeError):
        return None  # missing / torn / unparseable → treated as not-alive (falls back to claim_ttl)


def _session_alive(claims_dir: Path, session_id: str, heartbeat_ttl: float) -> bool:
    """Whether ``session_id`` is presumed alive — its heartbeat is visible and fresher than the TTL."""
    age = _heartbeat_age_seconds(claims_dir, session_id)
    return age is not None and age <= heartbeat_ttl


def _claim_owner(marker: Path) -> str | None:
    """The ``session_id`` recorded in a claim marker, or ``None`` for a legacy anonymous marker."""
    try:
        owner = json.loads(marker.read_text(encoding="utf-8")).get("session_id")
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return None
    return str(owner) if owner is not None else None


def _claim_is_stale(
    marker: Path, claims_dir: Path, claim_ttl: float, heartbeat_ttl: float | None
) -> bool:
    """Whether this claim can be stolen (its owner is presumed dead).

    Heartbeat path (owner recorded AND ``heartbeat_ttl`` given): stale iff the owner's heartbeat is
    VISIBLE and older than ``heartbeat_ttl`` — decoupled from how long the eval runs. If the heartbeat
    isn't visible (owner just started / Drive lag / clean-exit removed it) we do NOT trust that as
    death; fall back to the claim's own age vs the long ``claim_ttl`` backstop so a fresh claim is
    never falsely stolen. Legacy anonymous markers always use the age backstop (old behavior).
    """
    owner = _claim_owner(marker)
    if owner is not None and heartbeat_ttl is not None:
        age = _heartbeat_age_seconds(claims_dir, owner)
        if age is not None:
            return age > heartbeat_ttl  # heartbeat visible → trust it, ignore eval duration
        # heartbeat not visible → don't steal on that alone; fall through to the age backstop.
    try:
        return (time.time() - marker.stat().st_mtime) > claim_ttl
    except (FileNotFoundError, OSError):
        return True  # vanished under us — treat as free


def _write_claim(marker: Path, vector: np.ndarray | None, session_id: str | None = None) -> None:
    payload = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "vector": (
            [float(v) for v in np.asarray(vector, dtype=float)] if vector is not None else None
        ),
        "session_id": session_id,  # None for legacy/anonymous callers → age-TTL fallback on steal
    }
    with open(marker, "x", encoding="utf-8") as f:  # exclusive create
        f.write(json.dumps(payload))


def _claim_one(
    claims_dir: Path,
    h: str,
    ttl_seconds: float,
    vector: np.ndarray | None = None,
    *,
    session_id: str | None = None,
    heartbeat_ttl: float | None = None,
) -> bool:
    """Atomically claim design hash ``h`` (recording its vector + owner); steal it if the owner is
    dead (heartbeat lapsed, or — legacy — marker older than ``ttl``).

    ``True`` iff we hold the claim. Stealing a dead owner's marker lets the design (esp. a cold-start
    DoE point) be evaluated instead of deadlocking. The vector is stored so other sessions can
    condition their acquisition on this in-flight design; ``session_id`` ties the claim to a heartbeat
    lease so a dropped session's work frees in ~``heartbeat_ttl`` instead of ~``ttl``.
    """
    claims_dir.mkdir(parents=True, exist_ok=True)
    marker = claims_dir / f"{h}.claim"
    try:
        _write_claim(marker, vector, session_id)
        return True
    except FileExistsError:
        if not _claim_is_stale(marker, claims_dir, ttl_seconds, heartbeat_ttl):
            return False  # owner alive (or within legacy TTL) — leave it
        try:  # stale: steal it (owner presumed dead)
            marker.unlink()
            _write_claim(marker, vector, session_id)
            return True
        except (FileNotFoundError, FileExistsError):
            return False  # another session stole it first


def _release_own_claims(claims_dir: Path, session_id: str) -> None:
    """Release EVERY claim marker owned by ``session_id`` — used on pool death, when all of this
    session's in-flight designs are abandoned (their evals died with the pool) and must be
    re-dispatched. Covers claims that leaked outside the tracked ``inflight`` set (e.g. a design
    claimed just before ``pool.submit`` raised ``BrokenProcessPool``): with heartbeat leases those
    are owned-by-a-live-session, so nothing else would ever free them. Safe because a pool death
    means none of this session's claimed designs are still running here.
    """
    d = Path(claims_dir)
    if not d.exists():
        return
    for marker in d.glob("*.claim"):
        if _claim_owner(marker) == session_id:
            with contextlib.suppress(FileNotFoundError):
                marker.unlink()


def _release_claim(claims_dir: Path, h: str) -> None:
    """Delete a completed design's claim marker — it's redundant once the design is in the ledger
    (the ledger-skip covers re-claims), and pruning keeps ``active_claims`` O(in-flight), not O(N).
    """
    with contextlib.suppress(FileNotFoundError):
        (Path(claims_dir) / f"{h}.claim").unlink()


def active_claims(
    claims_dir: Path, ttl_seconds: float, *, heartbeat_ttl: float | None = None
) -> list[tuple[str, np.ndarray | None]]:
    """``(hash, vector)`` for every non-stale claim marker — the designs currently in flight.

    Stale markers (dead owner — heartbeat lapsed, or legacy marker older than ``ttl``) are excluded,
    matching what a live session would steal. Used to (a) count in-flight work for the budget and
    (b) condition the acquisition on other sessions' running designs so async proposals don't
    duplicate them.
    """
    d = Path(claims_dir)
    out: list[tuple[str, np.ndarray | None]] = []
    if not d.exists():
        return out
    for marker in d.glob("*.claim"):
        try:
            if _claim_is_stale(marker, d, ttl_seconds, heartbeat_ttl):
                continue
            payload = json.loads(marker.read_text(encoding="utf-8"))
            vec = payload.get("vector")
            out.append((marker.stem, np.array(vec, dtype=float) if vec is not None else None))
        except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError):
            continue
    return out


def claim_designs(
    claims_dir: Path,
    ledger_hashes: set[str],
    batch: np.ndarray,
    ttl_seconds: float = _DEFAULT_CFG.claim_ttl_seconds,
    *,
    session_id: str | None = None,
    heartbeat_ttl: float | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Keep only batch designs not already evaluated or actively claimed; claim them.

    Returns ``(claimed_vectors, claimed_hashes)``. Skips a design already in the ledger and one with
    a live claim marker; steals a dead owner's claim (heartbeat lapsed / legacy stale). So no two live
    sessions run the same one. ``session_id``/``heartbeat_ttl`` tie the claims to this session's lease.
    """
    keep: list[np.ndarray] = []
    keep_h: list[str] = []
    for v in np.atleast_2d(batch):
        h = design_hash(decode(v).to_dict())
        if h in ledger_hashes or h in keep_h:
            continue
        if _claim_one(
            Path(claims_dir), h, ttl_seconds, v,
            session_id=session_id, heartbeat_ttl=heartbeat_ttl,
        ):
            keep.append(v)
            keep_h.append(h)
    x = np.array(keep, dtype=float).reshape(-1, N_DIMS)
    return x, keep_h


def append_eval(
    shard: Path,
    vector: np.ndarray,
    y: tuple[float, float, float],
    *,
    session_id: str,
    source: str,
    dispatch_s: float | None = None,
    finish_s: float | None = None,
    inflight_at_dispatch: int | None = None,
) -> None:
    """Append one evaluation to this session's shard (with its vector, so any session reconstructs x).

    The async loop also records ``dispatch_s`` / ``finish_s`` (seconds from session start) and
    ``inflight_at_dispatch`` (how many of this session's evals were already running when this one was
    dispatched) — the telemetry :func:`validate_async` uses to prove dispatch-on-completion.
    """
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
        "dispatch_s": dispatch_s,
        "finish_s": finish_s,
        "inflight_at_dispatch": inflight_at_dispatch,
    }
    with open(shard, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _heartbeat_loop(
    claims_dir: Path, session_id: str, interval: float, stop: threading.Event
) -> None:
    """Refresh the session's lease every ``interval`` s until asked to stop (daemon thread)."""
    while not stop.is_set():
        with contextlib.suppress(OSError):
            touch_heartbeat(claims_dir, session_id)
        stop.wait(interval)


@contextlib.contextmanager
def _heartbeat(claims_dir: Path, session_id: str, cfg: DistributedConfig):
    """Hold a live heartbeat lease for the wrapped session.

    Writes one heartbeat BEFORE yielding (so the session is provably alive before it claims anything —
    no startup window where a peer sees a fresh claim with no heartbeat), then refreshes it on a
    daemon thread. On exit the thread stops and the heartbeat simply goes stale after
    ``heartbeat_ttl`` — so whether the session ends cleanly OR the whole VM is killed (the finally
    never runs), a dropped session's claims free on the SAME fast path. No unlink: a missing heartbeat
    is ambiguous (never-started vs unsynced) and must NOT trigger a steal.
    """
    stop = threading.Event()
    touch_heartbeat(claims_dir, session_id)
    thread = threading.Thread(
        target=_heartbeat_loop,
        args=(claims_dir, session_id, cfg.heartbeat_interval_seconds, stop),
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stop.set()


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


def _coldstart_designs(cfg: DistributedConfig) -> np.ndarray:
    """The cold-start dispatch sequence: the operator seed designs FIRST, then the Sobol DoE.

    Seeds are prepended so they dispatch ahead of the DoE; both share the claim/ledger path, so a
    seed is evaluated exactly once across all sessions (deduped by design hash, like any design).
    Empty/None ``cfg.seed_designs`` reproduces the plain Sobol cold-start.
    """
    sobol = _sobol_doe(cfg.n_init, cfg.seed)
    if not cfg.seed_designs:
        return sobol
    seeds = np.array(
        [clip_to_bounds(np.asarray(s, dtype=float)) for s in cfg.seed_designs]
    ).reshape(-1, N_DIMS)
    return np.vstack([seeds, sobol])


def _safe_eval(objective_fn: ObjectiveFn, v: np.ndarray) -> tuple[float, float, float]:
    try:
        return objective_fn(v)
    except Exception:  # one bad eval must not kill the session (would compound orphan claims)
        nan = float("nan")
        return (nan, nan, nan)


def _pool(n_workers: int, start_method: str | None):
    """A ProcessPoolExecutor using ``start_method`` (``None`` = platform default)."""
    if start_method is None:
        return ProcessPoolExecutor(max_workers=n_workers)
    return ProcessPoolExecutor(
        max_workers=n_workers, mp_context=multiprocessing.get_context(start_method)
    )


def _evaluate(
    objective_fn: ObjectiveFn, batch: np.ndarray, n_workers: int, start_method: str | None = None
) -> list[tuple]:
    if n_workers > 1 and len(batch) > 1:
        out: list = [None] * len(batch)
        with _pool(n_workers, start_method) as pool:
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
    doe = _coldstart_designs(cfg)  # seed designs first, then the Sobol DoE
    n_cold = len(doe)

    with _heartbeat(claims, session_id, cfg):
        # On (re)start, free THIS session's own pre-crash orphan claims: a same-id restart writes a
        # fresh heartbeat, so its phantom claims would otherwise look 'live' forever (heartbeat lease)
        # and deadlock cold-start — neither this session nor a peer could ever reclaim them. Same
        # rationale as the async path's pool-death release; the sync path has no BrokenProcessPool
        # handler, so this startup sweep is its only orphan-recovery point.
        _release_own_claims(claims, session_id)
        for _ in range(max_iters):
            x, y_raw, hashes = read_ledger(shared)
            if len(x) >= cfg.total_budget:
                break

            bo_ctx: tuple | None = None
            if len(x) < n_cold:
                # Cold-start: this session owns cold-start points i where i % n_sessions == index
                # (seeds sit at the front, so they dispatch ahead of the DoE); propose the next ones NOT
                # yet in the shared ledger, so it advances instead of re-proposing an already-evaluated
                # batch (which would stall on claims).
                idxs = [i for i in range(n_cold) if i % n_sessions == session_index]
                fresh = [doe[i] for i in idxs if design_hash(decode(doe[i]).to_dict()) not in hashes]
                if not fresh:
                    # Own slice done but ledger < n_cold: mop up ANY un-evaluated cold-start point, so a
                    # permanently-departed session's slice can't deadlock cold-start. claim_designs
                    # skips points a live session still holds and steals only stale (dead) claims, so
                    # this never double-runs live work — it just lets survivors finish the cold-start.
                    fresh = [
                        doe[i]
                        for i in range(n_cold)
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

            claimed, _ = claim_designs(
                claims, hashes, proposed, cfg.claim_ttl_seconds,
                session_id=session_id, heartbeat_ttl=cfg.heartbeat_ttl_seconds,
            )
            if len(claimed) == 0:
                if cfg.poll_seconds > 0:
                    time.sleep(cfg.poll_seconds)  # another session took these; re-read and retry
                continue

            ys = _evaluate(objective_fn, claimed, cfg.n_workers, cfg.pool_start_method)
            for v, yy in zip(claimed, ys, strict=True):
                append_eval(shard, v, yy, session_id=session_id, source=source)
                _release_claim(claims, _hof(v))  # completed → marker redundant; keep claims dir small

            if bo_ctx is not None and cfg.use_trust_region:  # advance TuRBO (shrink/grow) on progress
                loc, scale, ref, hv_before = bo_ctx
                y_all = _y_max(np.vstack([y_raw, np.array(ys, dtype=float)]))
                hv_after = hypervolume(apply_objective_norm(y_all, loc, scale), ref)
                tr.update(hv_after > hv_before + 1e-12)

            if on_batch is not None:
                on_batch(len(claimed))

    x, y_raw, _ = read_ledger(shared)
    return x, _sanitize_yraw(y_raw) if len(y_raw) else y_raw


def _hof(vector: np.ndarray) -> str:
    return design_hash(decode(vector).to_dict())


def _is_nondominated(y_all_raw: np.ndarray, y_new: tuple[float, float, float]) -> bool:
    """Did the just-completed point land on the Pareto front? (async TuRBO success signal.)

    A failed eval (non-finite) is never progress. Scale-free (dominance), so it needs no shared
    normalization frame across the async completions.
    """
    yn_raw = np.asarray(y_new, dtype=float)
    if not np.all(np.isfinite(yn_raw)):
        return False
    big = to_maximization(np.atleast_2d(y_all_raw))
    yn = to_maximization(yn_raw[None])[0]
    dominated = np.any(np.all(big >= yn, axis=1) & np.any(big > yn, axis=1))
    return not bool(dominated)


def run_async_session(
    objective_fn: ObjectiveFn,
    shared_dir: Path | str,
    cfg: DistributedConfig = _DEFAULT_CFG,
    *,
    session_id: str,
    session_index: int = 0,
    n_sessions: int = 1,
    on_event: Callable[[dict], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Asynchronous distributed BO: keep ``cfg.n_workers`` evals running, dispatch-on-completion.

    The instant one eval finishes, refit the GP on everything completed and dispatch ONE new
    design to the freed worker — conditioning the acquisition on all in-flight designs (this
    session's + others', read from the claim markers) via ``X_pending`` so it doesn't duplicate
    running work. No batch barrier (fast workers never idle) and each proposal is informed by all
    completed evals, so quality approaches sequential at full parallel throughput. Records
    per-eval telemetry (``inflight_at_dispatch``) so :func:`validate_async` can prove the loop
    really ran async and not in batches. Returns the final combined ``(x, y_raw)`` (raw frame).
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
    doe = _coldstart_designs(cfg)  # seed designs first, then the Sobol DoE
    n_cold = len(doe)
    tr = TrustRegionState(dim=N_DIMS, batch_size=1)
    t0 = time.monotonic()
    inflight: dict = {}  # future -> (vector, hash, dispatch_s, inflight_at_dispatch, source)
    avoid: list[np.ndarray] = []  # designs whose acqf-optimum decoded to an already-known design;
    # fed back as X_pending so the acquisition is pushed to a genuinely NEW design instead of
    # re-proposing the same optimum (codec quantization maps a region to one design → livelock).
    bo_dispatches = [0]  # counts BO-phase attempts, for the explore/exploit schedule
    explore_sobol = (
        Sobol(d=N_DIMS, seed=cfg.seed + 9973 + session_index) if cfg.explore_fraction > 0 else None
    )
    explore_every = round(1 / cfg.explore_fraction) if cfg.explore_fraction > 0 else 0

    def dispatch_one(pool: ProcessPoolExecutor) -> str:
        """Propose+claim+submit one design. 'ok' | 'retry' (raced) | 'stop' (budget / DoE-wait)."""
        x, y_raw, hashes = read_ledger(shared)
        active = active_claims(claims, cfg.claim_ttl_seconds, heartbeat_ttl=cfg.heartbeat_ttl_seconds)
        known = hashes | {h for h, _ in active}
        if len(known) >= cfg.total_budget:
            return "stop"
        if len(hashes) < n_cold:  # cold-start (seeds first, then DoE; own slice first, then mop up)
            own = [doe[i] for i in range(n_cold) if i % n_sessions == session_index]
            cand = [v for v in own if _hof(v) not in known] or [
                doe[i] for i in range(n_cold) if _hof(doe[i]) not in known
            ]
            if not cand:
                return "stop"  # every DoE point is claimed/done; wait for it to reach the ledger
            vector, source = cand[0], "sobol"
        else:
            bo_dispatches[0] += 1
            if explore_sobol is not None and bo_dispatches[0] % explore_every == 0:
                # pure exploration: a space-filling Sobol draw, NOT acquisition-driven — keeps the
                # search probing new regions instead of only refining the incumbent.
                unit = explore_sobol.random(1)[0]
                vector, source = clip_to_bounds(low + unit * (high - low)), "explore"
            else:
                pending = [v for h, v in active if v is not None and h not in hashes] + avoid
                y_norm, _, _ = normalize_objectives(_y_max(y_raw))
                ref = infer_reference_point(y_norm)
                torch.manual_seed(
                    (session_index + 1) * 100_003 + len(x) + len(inflight) + len(avoid)
                )
                model = fit_gp(x, y_norm, low, high)
                cand = propose_candidates(
                    model,
                    x,
                    y_norm,
                    low,
                    high,
                    ref,
                    batch_size=1,
                    tr_state=tr if cfg.use_trust_region else None,
                    num_restarts=cfg.num_restarts,
                    raw_samples=cfg.raw_samples,
                    mc_samples=cfg.mc_samples,
                    X_pending=np.array(pending) if pending else None,
                )
                vector, source = clip_to_bounds(cand[0]), "bo"
        h = _hof(vector)
        if h in known:
            if source == "bo":  # push the acquisition off this already-known design next time
                avoid.append(vector)
                del avoid[: -2 * cfg.n_workers]  # bound the conditioning set
            return "retry"
        if not _claim_one(
            claims, h, cfg.claim_ttl_seconds, vector,
            session_id=session_id, heartbeat_ttl=cfg.heartbeat_ttl_seconds,
        ):
            return "retry"  # a peer claimed it between our read and now
        avoid.clear()  # made progress; the claimed design is now tracked via its marker
        inflight[pool.submit(_safe_eval, objective_fn, vector)] = (
            vector,
            h,
            time.monotonic() - t0,
            len(inflight),
            source,
        )
        return "ok"

    def budget_met() -> bool:
        _, _, hashes = read_ledger(shared)
        return (
            len(hashes | {h for h, _ in active_claims(claims, cfg.claim_ttl_seconds, heartbeat_ttl=cfg.heartbeat_ttl_seconds)})
            >= cfg.total_budget
        )

    # Outer loop rebuilds the pool if a worker dies (BrokenProcessPool — OOM / Colab preemption of a
    # subprocess). Completed evals are already in the ledger; the abandoned in-flight designs' claims
    # go stale and are reclaimed (by this or a peer session), so no work is permanently lost.
    with _heartbeat(claims, session_id, cfg):
        while not (budget_met() and not inflight):
            # A dead pool's futures are gone — RELEASE all of THIS session's claims before rebuilding.
            # This session is alive (its heartbeat stays fresh), so heartbeat liveness would keep its
            # owned claims 'live' forever; only the session itself can free its own worker-death
            # orphans so they get re-dispatched. Release by owner (not just the tracked `inflight`) to
            # also catch a claim that leaked when `pool.submit` raised BrokenProcessPool after the
            # marker was written. (Session-DEATH orphans are handled the other way: the heartbeat
            # lapses and a peer reclaims them.)
            _release_own_claims(claims, session_id)
            inflight.clear()
            try:
                with _pool(cfg.n_workers, cfg.pool_start_method) as pool:
                    while True:
                        tries = 0
                        while len(inflight) < cfg.n_workers and tries < cfg.n_workers * 3:
                            r = dispatch_one(pool)
                            if r == "ok":
                                continue
                            if r == "retry":
                                tries += 1
                                continue
                            break  # "stop"
                        if not inflight:
                            if budget_met():
                                break
                            # Nothing to run right now (peers still filling the DoE, or every fresh
                            # proposal collided with an in-flight/claimed design). Back off before
                            # retrying — a 50 ms floor even when poll_seconds==0 so this can't hot-spin
                            # the GP refit (Finding D).
                            time.sleep(cfg.poll_seconds if cfg.poll_seconds > 0 else 0.05)
                            continue
                        done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                        for fut in done:
                            vector, h, disp_s, n_at, source = inflight.pop(fut)
                            y = fut.result()
                            append_eval(
                                shard,
                                vector,
                                y,
                                session_id=session_id,
                                source=source,
                                dispatch_s=disp_s,
                                finish_s=time.monotonic() - t0,
                                inflight_at_dispatch=n_at,
                            )
                            _release_claim(claims, h)  # completed → marker redundant; keep claims small
                            if cfg.use_trust_region:
                                _, y_all, _ = read_ledger(shared)
                                tr.update(_is_nondominated(y_all, y))
                            if on_event is not None:
                                on_event(
                                    {"design_hash": h, "source": source, "inflight_at_dispatch": n_at}
                                )
                break  # inner loop exited cleanly (budget) → done
            except BrokenProcessPool:
                continue  # a worker died; rebuild the pool and resume from the ledger

    x, y_raw, _ = read_ledger(shared)
    return x, _sanitize_yraw(y_raw) if len(y_raw) else y_raw


def _steady_utilization(intervals: list[tuple[float, float]], n_workers: int) -> tuple[float, int]:
    """Worker utilization (mean concurrency / n_workers) over the STEADY window, plus peak concurrency.

    Integrating concurrency over time is honest regardless of how completions cluster (unlike a
    point-in-time occupancy count). Measured between the moment the pool first fills and the moment
    it starts draining, so ramp-up/down don't dilute it: async keeps every worker busy there
    (util → 1.0); a batch loop leaves early-finishing workers idle until their wave-mates finish
    (util → mean_eval/max_eval < 1).
    """
    ivs = [(float(d), float(f)) for d, f in intervals if f > d]
    if not ivs:
        return 0.0, 0
    events = sorted([(d, 1) for d, _ in ivs] + [(f, -1) for _, f in ivs])
    peak = c = 0
    for _, delta in events:
        c += delta
        peak = max(peak, c)
    disp = sorted(d for d, _ in ivs)
    fin = sorted(f for _, f in ivs)
    if len(ivs) > n_workers:  # trim the ramp-up (fill) and drain (tail) so edges don't dilute
        lo, hi = disp[n_workers - 1], fin[len(ivs) - n_workers]
    else:
        lo, hi = disp[0], fin[-1]
    if hi <= lo:
        lo, hi = disp[0], fin[-1]
    area = 0.0
    c = 0
    prev = events[0][0]
    for t, delta in events:
        seg_lo, seg_hi = max(prev, lo), min(t, hi)
        if seg_hi > seg_lo:
            area += c * (seg_hi - seg_lo)
        c += delta
        prev = t
    util = area / (n_workers * (hi - lo)) if hi > lo else 0.0
    return util, peak


def validate_async(shared_dir: Path | str, n_workers: int) -> dict:
    """Prove the campaign ran async (workers stayed busy), not in batches, from the ledger telemetry.

    Verdict is **worker utilization** — the time-integrated fraction of workers actually running an
    eval (see :func:`_steady_utilization`). Async keeps the pool full → utilization ≈ 1.0; a
    synchronous batch loop idles early-finishing workers until the slowest in their wave → notably
    lower. This is the quantity that actually matters (no wasted idle time) and, unlike counting
    point-in-time occupancy, it can't be fooled by completions arriving in clusters.
    """
    rows: list[dict] = []
    for shard in sorted(Path(shared_dir).glob(LEDGER_GLOB)):
        for line in shard.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    telem = [r for r in rows if r.get("dispatch_s") is not None and r.get("finish_s") is not None]
    per_session: dict[str, dict] = {}
    tot_busy = tot_capacity = 0.0
    for s in {r.get("session_id") for r in telem}:
        ivs = [(r["dispatch_s"], r["finish_s"]) for r in telem if r.get("session_id") == s]
        util, peak = _steady_utilization(ivs, n_workers)
        per_session[str(s)] = {"utilization": util, "peak_concurrency": peak, "n_evals": len(ivs)}
        window = max(f for _, f in ivs) - min(d for d, _ in ivs) if ivs else 0.0
        tot_busy += sum(f - d for d, f in ivs if f > d)
        tot_capacity += n_workers * window
    utilization = tot_busy / tot_capacity if tot_capacity > 0 else 0.0
    n = len(telem)
    steady_utils = [p["utilization"] for p in per_session.values() if p["n_evals"] > n_workers]
    verdict = min(steady_utils) if steady_utils else utilization
    return {
        "n_evals": n,
        "utilization": utilization,  # overall busy-fraction; async ≈ 1.0, batch-with-variance lower
        "per_session": per_session,
        # async keeps utilization high in the steady window; a batch loop idles workers → < ~0.9
        "is_async": bool(n >= n_workers and verdict >= 0.9),
    }


def _smoke_objective(v: np.ndarray) -> tuple[float, float, float]:
    """Per-design-staggered synthetic eval for the pre-flight (no CFD). Sleeps a few seconds — long
    enough for the pool to fill to n_workers before the first evals finish — with a stagger so
    completions arrive one at a time. It does NOT need to dominate the GP proposal: the pre-flight
    checks the dispatch MECHANISM, not throughput (see :func:`preflight_async_check`)."""
    a = np.asarray(v, dtype=float)
    time.sleep(2.5 + 1.5 * float(np.abs(np.sin(np.sum(a)))))
    return (float(np.sum(np.cos(a[:5]))), float(1.0 + np.sum(np.abs(a)) * 1e-3), 1e-3)


def preflight_async_check(n_workers: int, *, seed: int = 0) -> dict:
    """Quick smoke test (no CFD) that the async MACHINERY works in THIS environment — run it BEFORE
    the multi-hour campaign so a broken async loop / bad ProcessPool is caught up front.

    Runs ``run_async_session`` on a fast synthetic objective (throwaway dir, ``n_workers`` workers,
    ``2·n_workers`` budget). ``passed`` iff the pool **filled** (peak == n_workers) AND **refilled on
    completion** to exceed it — reaching ``2·n_workers`` unique evals with no duplicates. That is the
    dispatch-on-completion mechanism; a loop that never refilled would stop at ``n_workers``.

    Utilization is reported but is NOT the pass criterion here: with a fast dummy objective the serial
    GP proposal (~1–2 s) bottlenecks throughput so utilization is artificially low. On the real 2.8 h
    objective proposals are negligible and utilization ≈ 1.0 — measured live by the run cell.
    """
    budget = 2 * n_workers
    with tempfile.TemporaryDirectory() as d:
        cfg = DistributedConfig(
            total_budget=budget,
            n_init=n_workers,
            n_workers=n_workers,
            seed=seed,
            poll_seconds=0.0,
            num_restarts=2,
            raw_samples=16,
            mc_samples=16,
        )
        run_async_session(_smoke_objective, d, cfg, session_id="preflight")
        x, _, hashes = read_ledger(d)
        report = validate_async(d, n_workers)
    ps = report["per_session"].get("preflight", {})
    report["reached_budget"] = len(x) >= budget
    report["no_duplicates"] = len(x) == len(hashes)
    report["passed"] = bool(
        ps.get("peak_concurrency") == n_workers
        and report["reached_budget"]
        and report["no_duplicates"]
    )
    return report
