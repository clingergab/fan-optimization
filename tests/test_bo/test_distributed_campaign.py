"""Tests for the async shared-ledger distributed BO (Stage 3). Requires botorch (the [bo] extra)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time

import numpy as np
import pytest

if importlib.util.find_spec("botorch") is None:
    pytest.skip("botorch (the [bo] extra) required", allow_module_level=True)

from fanopt.bo import distributed_campaign as dc
from fanopt.bo.blade_campaign import stage3_seed_designs
from fanopt.bo.blade_codec import N_DIMS, bounds, clip_to_bounds, decode
from fanopt.bo.distributed_campaign import (
    DistributedConfig,
    active_claims,
    append_eval,
    claim_designs,
    pareto_from_ledger,
    preflight_async_check,
    read_ledger,
    run_async_session,
    run_distributed_session,
    shard_path,
    validate_async,
)
from fanopt.utils.ledger import design_hash


def _vec(seed: int) -> np.ndarray:
    low, high = bounds()
    rng = np.random.default_rng(seed)
    return clip_to_bounds(low + rng.random(N_DIMS) * (high - low))


def _synthetic(v: np.ndarray) -> tuple[float, float, float]:
    # finite, varied objective — not meaningful, just exercises the loop
    return (float(np.sum(np.cos(v[:5]))), float(1.0 + np.sum(np.abs(v)) * 1e-3), 1e-3)


def _staggered_synthetic(v: np.ndarray) -> tuple[float, float, float]:
    # Varied per-design sleep so completions arrive one at a time, AND long enough to dominate the
    # GP proposal (~1s) — mirroring the real regime (eval 2.8h >> proposal 2s). Only then does a
    # correct async loop keep the pool full (inflight_at_dispatch == n_workers-1). With eval faster
    # than the proposal the pool would drain no matter how the loop is written.
    h = int(hashlib.md5(np.asarray(v, dtype=float).tobytes()).hexdigest(), 16) % 100
    time.sleep(2.5 + 2.5 * (h / 100.0))
    return _synthetic(v)


# Trimmed acquisition: fast enough (~0.4s) to be small vs the deliberately slow synthetic eval
# (~3.75s), but NOT so degenerate (too-few raw_samples) that it repeatedly proposes the same
# codec-quantized design and churns. Mirrors the real regime (eval 2.8h >> proposal 2s).
_FAST_ACQ = dict(num_restarts=3, raw_samples=48, mc_samples=16)


def _worker_killer_once(v: np.ndarray) -> tuple[float, float, float]:
    # Hard-kill this worker on the first design ever evaluated (flag file), breaking the pool
    # (BrokenProcessPool) — simulates an OOM / Colab subprocess preemption. Later calls succeed.
    flag = os.environ.get("ASYNC_KILL_FLAG")
    if flag and not os.path.exists(flag):
        open(flag, "w").close()
        os._exit(1)
    return _synthetic(v)


def _raw_row_count(shared) -> int:
    return sum(
        1
        for shard in shared.glob(dc.LEDGER_GLOB)
        for ln in shard.read_text().splitlines()
        if ln.strip()
    )


def _ledger_hashes(shared) -> list[str]:
    return [
        json.loads(ln)["design_hash"]
        for shard in shared.glob(dc.LEDGER_GLOB)
        for ln in shard.read_text().splitlines()
        if ln.strip()
    ]


def _seed_hashes() -> list[str]:
    return [design_hash(decode(s).to_dict()) for s in stage3_seed_designs()]


# --- ledger round-trip / robustness ---


def test_append_and_read_ledger_roundtrip(tmp_path):
    append_eval(shard_path(tmp_path, "s0"), _vec(1), (2.0, 0.1, 1e-3), session_id="s0", source="bo")
    x, y, hashes = read_ledger(tmp_path)
    assert x.shape == (1, N_DIMS)
    assert y[0].tolist() == [2.0, 0.1, 1e-3]
    assert len(hashes) == 1


def test_read_ledger_missing_dir_is_empty(tmp_path):
    x, y, hashes = read_ledger(tmp_path / "nope")
    assert x.shape == (0, N_DIMS) and y.shape == (0, 3) and hashes == set()


def test_read_ledger_skips_torn_line(tmp_path):
    append_eval(shard_path(tmp_path, "s"), _vec(2), (1.0, 0.1, 1e-3), session_id="s", source="bo")
    with open(shard_path(tmp_path, "s"), "a", encoding="utf-8") as f:
        f.write('{"vector": [0.1, 0.2  <-- torn concurrent write\n')  # malformed
    x, _, _ = read_ledger(tmp_path)
    assert x.shape == (1, N_DIMS)  # torn line skipped, good row kept


def test_read_ledger_dedups_the_same_design_across_shards(tmp_path):
    # Drive's create-exclusive can race, letting two sessions evaluate one design; read dedups it.
    v = _vec(5)
    append_eval(shard_path(tmp_path, "A"), v, (1.0, 0.1, 1e-3), session_id="A", source="bo")
    append_eval(shard_path(tmp_path, "B"), v, (1.0, 0.1, 1e-3), session_id="B", source="bo")
    x, _, hashes = read_ledger(tmp_path)
    assert len(x) == 1 and len(hashes) == 1  # the duplicate design is counted once


def test_read_ledger_drops_wrong_length_vector(tmp_path):
    shard = shard_path(tmp_path, "s")
    append_eval(shard, _vec(6), (1.0, 0.1, 1e-3), session_id="s", source="bo")
    with open(shard, "a", encoding="utf-8") as f:  # valid JSON, wrong-length vector (schema drift)
        f.write(
            json.dumps(
                {
                    "design_hash": "x",
                    "vector": [0.0, 1.0],
                    "j_fan": 1.0,
                    "mass_kg": 0.1,
                    "deflection_m": 1e-3,
                }
            )
            + "\n"
        )
    x, _, _ = read_ledger(tmp_path)
    assert x.shape == (1, N_DIMS)  # bad row dropped, not a crash


# --- claim coordination ---


def test_claim_is_exclusive(tmp_path):
    batch = np.array([_vec(10), _vec(11), _vec(12)])
    got1, _ = claim_designs(tmp_path / "claims", set(), batch)
    got2, _ = claim_designs(tmp_path / "claims", set(), batch)  # same batch, already claimed
    assert len(got1) == 3 and len(got2) == 0  # no design claimed twice


def test_claim_skips_designs_already_in_ledger(tmp_path):
    v = _vec(20)
    already = {design_hash(decode(v).to_dict())}
    got, _ = claim_designs(tmp_path / "claims", already, np.array([v, _vec(21)]))
    assert len(got) == 1  # the already-evaluated design is skipped


def test_fresh_claim_is_not_stolen(tmp_path):
    claims = tmp_path / "claims"
    batch = np.array([_vec(30)])
    claim_designs(claims, set(), batch, ttl_seconds=10_000)
    got, _ = claim_designs(claims, set(), batch, ttl_seconds=10_000)  # still fresh
    assert len(got) == 0  # a live claim is respected


def test_stale_claim_is_reclaimed(tmp_path):
    # A session that died mid-eval (Colab drop) leaves a claim; after the TTL another session may
    # steal it, so an orphaned cold-start DoE point can't deadlock the whole campaign (H1).
    claims = tmp_path / "claims"
    v = _vec(31)
    claim_designs(claims, set(), np.array([v]), ttl_seconds=10_000)
    marker = claims / f"{design_hash(decode(v).to_dict())}.claim"
    old = os.stat(marker).st_mtime - 3600  # age it an hour
    os.utime(marker, (old, old))
    got, _ = claim_designs(claims, set(), np.array([v]), ttl_seconds=60)  # TTL=60s → stale
    assert len(got) == 1  # reclaimed


# --- heartbeat leases: liveness decoupled from eval wall-time (fast reclaim of a dropped session) ---


def _hb_aged(claims, session_id, age_s):
    """Write ``session_id``'s heartbeat as if last refreshed ``age_s`` seconds ago (simulate a drop)."""
    claims.mkdir(parents=True, exist_ok=True)
    ts = dc._dt.datetime.now(dc._dt.timezone.utc) - dc._dt.timedelta(seconds=age_s)
    dc._heartbeat_path(claims, session_id).write_text(ts.isoformat(), encoding="utf-8")


def _h(v):
    return design_hash(decode(v).to_dict())


def test_touch_heartbeat_makes_session_alive(tmp_path):
    dc.touch_heartbeat(tmp_path / "claims", "s0")
    assert dc._session_alive(tmp_path / "claims", "s0", heartbeat_ttl=600)


def test_session_dead_when_heartbeat_lapses(tmp_path):
    _hb_aged(tmp_path / "claims", "s0", age_s=1000)  # last beat 1000s ago, ttl 600
    assert not dc._session_alive(tmp_path / "claims", "s0", heartbeat_ttl=600)


def test_session_dead_when_no_heartbeat_file(tmp_path):
    assert not dc._session_alive(tmp_path / "claims", "never", heartbeat_ttl=600)


def test_live_owner_claim_not_stolen_even_when_marker_is_ancient(tmp_path):
    # Headline property: a live session mid-long-eval keeps its claim no matter how old the marker
    # is — liveness is the heartbeat, not the claim age (so a >6h eval is never falsely stolen).
    claims = tmp_path / "claims"
    v = _vec(40)
    dc.touch_heartbeat(claims, "owner")
    assert dc._claim_one(claims, _h(v), 60.0, v, session_id="owner", heartbeat_ttl=600)
    marker = claims / f"{_h(v)}.claim"
    old = os.stat(marker).st_mtime - 100_000  # marker ancient, far past any claim_ttl
    os.utime(marker, (old, old))
    assert dc._claim_one(claims, _h(v), 60.0, v, session_id="peer", heartbeat_ttl=600) is False


def test_dead_owner_claim_reclaimed_fast_even_when_marker_is_fresh(tmp_path):
    # Recovery property: a dropped session's claim frees on the heartbeat TTL (minutes), NOT the 6h
    # claim_ttl backstop — even though the marker itself is brand new.
    claims = tmp_path / "claims"
    v = _vec(41)
    _hb_aged(claims, "dead", age_s=1000)  # owner last beat 1000s ago → dead
    assert dc._claim_one(claims, _h(v), 21_600.0, v, session_id="dead", heartbeat_ttl=600)
    # marker fresh + claim_ttl 6h ⇒ legacy would NOT steal; heartbeat says dead ⇒ stolen
    assert dc._claim_one(claims, _h(v), 21_600.0, v, session_id="peer", heartbeat_ttl=600) is True


def test_owner_claim_falls_back_to_ttl_when_heartbeat_not_visible(tmp_path):
    # If the owner's heartbeat isn't visible (just started / Drive lag), a fresh claim must NOT be
    # stolen on that alone — fall back to the long claim_ttl backstop; but a truly old one still frees.
    claims = tmp_path / "claims"
    v = _vec(42)
    assert dc._claim_one(claims, _h(v), 21_600.0, v, session_id="owner", heartbeat_ttl=600)  # no hb
    assert dc._claim_one(claims, _h(v), 21_600.0, v, session_id="peer", heartbeat_ttl=600) is False
    marker = claims / f"{_h(v)}.claim"
    old = os.stat(marker).st_mtime - 21_601  # age it past claim_ttl
    os.utime(marker, (old, old))
    assert dc._claim_one(claims, _h(v), 21_600.0, v, session_id="peer", heartbeat_ttl=600) is True


def test_active_claims_excludes_dead_owner_includes_live(tmp_path):
    claims = tmp_path / "claims"
    v_live, v_dead = _vec(43), _vec(44)
    dc.touch_heartbeat(claims, "live")
    _hb_aged(claims, "dead", age_s=1000)
    dc._claim_one(claims, _h(v_live), 21_600.0, v_live, session_id="live", heartbeat_ttl=600)
    dc._claim_one(claims, _h(v_dead), 21_600.0, v_dead, session_id="dead", heartbeat_ttl=600)
    got = {h for h, _ in active_claims(claims, ttl_seconds=21_600, heartbeat_ttl=600)}
    assert _h(v_live) in got and _h(v_dead) not in got  # live in flight; dead owner's is free


def test_legacy_anonymous_marker_still_uses_age_ttl(tmp_path):
    # Backward-compat / rolling upgrade: a marker with no session_id (old-code session) keeps the
    # age-TTL behavior even when a new-code peer passes heartbeat_ttl.
    claims = tmp_path / "claims"
    v = _vec(45)
    claims.mkdir(parents=True, exist_ok=True)
    dc._write_claim(claims / f"{_h(v)}.claim", v, session_id=None)  # anonymous (old code)
    assert dc._claim_one(claims, _h(v), 10_000.0, v, session_id="peer", heartbeat_ttl=600) is False
    marker = claims / f"{_h(v)}.claim"
    old = os.stat(marker).st_mtime - 20_000
    os.utime(marker, (old, old))
    assert dc._claim_one(claims, _h(v), 10_000.0, v, session_id="peer", heartbeat_ttl=600) is True


def test_heartbeat_context_manager_beats_on_entry_and_refreshes(tmp_path):
    claims = tmp_path / "claims"
    cfg = DistributedConfig(heartbeat_interval_seconds=0.05, heartbeat_ttl_seconds=600)
    with dc._heartbeat(claims, "s0", cfg):
        assert dc._session_alive(claims, "s0", heartbeat_ttl=600)  # written before the body runs
        first = dc._heartbeat_path(claims, "s0").read_text()
        for _ in range(40):  # poll for a refresh (avoid a fixed-sleep flake)
            time.sleep(0.05)
            if dc._heartbeat_path(claims, "s0").read_text() != first:
                break
        assert dc._heartbeat_path(claims, "s0").read_text() != first  # the daemon refreshed it
    assert dc._heartbeat_path(claims, "s0").exists()  # not deleted on exit → goes stale by TTL


def test_release_own_claims_frees_only_this_sessions_markers(tmp_path):
    claims = tmp_path / "claims"
    va, vb = _vec(50), _vec(51)
    dc.touch_heartbeat(claims, "s0"); dc.touch_heartbeat(claims, "other")
    dc._claim_one(claims, _h(va), 21_600.0, va, session_id="s0", heartbeat_ttl=600)
    dc._claim_one(claims, _h(vb), 21_600.0, vb, session_id="other", heartbeat_ttl=600)
    dc._release_own_claims(claims, "s0")
    assert not (claims / f"{_h(va)}.claim").exists()  # own orphan freed
    assert (claims / f"{_h(vb)}.claim").exists()  # a peer's claim is untouched


def test_same_id_restart_orphan_is_undeadlockable_without_release(tmp_path):
    # A same-id restart writes a FRESH heartbeat, so its pre-crash claims look 'live' forever — neither
    # it nor a peer can steal them (the sync-path deadlock the release fix addresses).
    claims = tmp_path / "claims"
    v = _vec(52)
    dc.touch_heartbeat(claims, "s0")  # restart: fresh heartbeat
    dc._claim_one(claims, _h(v), 21_600.0, v, session_id="s0", heartbeat_ttl=600)  # phantom pre-crash claim
    assert dc._claim_one(claims, _h(v), 21_600.0, v, session_id="peer", heartbeat_ttl=600) is False
    dc._release_own_claims(claims, "s0")  # the fix: session frees its own orphans on (re)start
    assert dc._claim_one(claims, _h(v), 21_600.0, v, session_id="s0", heartbeat_ttl=600) is True


def test_corrupt_heartbeat_does_not_crash(tmp_path):
    # A naive/unparseable heartbeat must NOT raise (that would kill a live session mid-claim).
    claims = tmp_path / "claims"; claims.mkdir(parents=True)
    dc._heartbeat_path(claims, "naive").write_text("2020-01-01T00:00:00", encoding="utf-8")  # tz-naive
    assert dc._heartbeat_age_seconds(claims, "naive") is not None  # coerced to utc, no TypeError
    dc._heartbeat_path(claims, "junk").write_text("not-a-timestamp", encoding="utf-8")
    assert dc._heartbeat_age_seconds(claims, "junk") is None  # unparseable → not-alive


def test_pool_honors_start_method(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    for method in (None, "spawn", "forkserver"):
        ex = dc._pool(2, method)  # constructing doesn't start workers → cheap
        assert isinstance(ex, ProcessPoolExecutor)
        ex.shutdown()


# --- config validation ---


@pytest.mark.parametrize("index,n", [(3, 3), (-1, 2), (0, 0)])
def test_bad_session_index_raises(tmp_path, index, n):
    with pytest.raises(ValueError):
        run_distributed_session(
            _synthetic, tmp_path, session_id="s", session_index=index, n_sessions=n, max_iters=1
        )


# --- correctness of the optimization frame (C1) ---


def test_bo_fits_gp_in_maximization_frame_lighter_is_better(tmp_path, monkeypatch):
    # Seed n_init designs with varied mass, then run one BO iteration. The array handed to the GP
    # (via normalize_objectives) MUST rank the lightest design highest in the mass column — i.e. the
    # fit is in the maximization frame. If the raw frame leaked through, heavier would rank higher.
    shard = shard_path(tmp_path, "s")
    for i in range(8):
        append_eval(shard, _vec(i), (1.0, 0.02 + 0.01 * i, 1e-3), session_id="s", source="sobol")
    captured = {}
    real = dc.normalize_objectives

    def spy(y_max):
        captured["y"] = np.array(y_max)
        return real(y_max)

    monkeypatch.setattr(dc, "normalize_objectives", spy)
    cfg = DistributedConfig(total_budget=99, n_init=8, batch_size=2, poll_seconds=0.0)
    run_distributed_session(_synthetic, tmp_path, cfg, session_id="s", max_iters=1)
    y = captured["y"]  # mass column: lightest seeded design (row 0) must be the column-max
    assert y[:, 1].argmax() == 0


# --- end to end ---


def test_single_session_reaches_budget_no_duplicates(tmp_path):
    cfg = DistributedConfig(total_budget=12, n_init=6, batch_size=4, poll_seconds=0.0)
    run_distributed_session(_synthetic, tmp_path, cfg, session_id="s0", max_iters=50)
    x, y, hashes = read_ledger(tmp_path)
    assert len(x) >= cfg.total_budget
    assert _raw_row_count(tmp_path) == len(hashes)  # nothing evaluated twice
    assert np.isfinite(y).all()


def test_two_sessions_share_ledger_without_duplicating(tmp_path):
    # Interleave two sessions (1 iteration each) until the shared budget is met; the claim +
    # ledger-skip logic must ensure NO design is physically evaluated by both (raw rows == unique).
    cfg = DistributedConfig(total_budget=16, n_init=8, batch_size=4, poll_seconds=0.0)
    for _ in range(60):
        if len(read_ledger(tmp_path)[0]) >= cfg.total_budget:
            break
        run_distributed_session(
            _synthetic, tmp_path, cfg, session_id="A", session_index=0, n_sessions=2, max_iters=1
        )
        run_distributed_session(
            _synthetic, tmp_path, cfg, session_id="B", session_index=1, n_sessions=2, max_iters=1
        )
    x, _, hashes = read_ledger(tmp_path)
    assert len(x) >= cfg.total_budget
    assert _raw_row_count(tmp_path) == len(hashes)  # coordination held — no design run twice
    sessions = {s.name.split("_")[1].split(".")[0] for s in tmp_path.glob(dc.LEDGER_GLOB)}
    assert sessions == {"A", "B"}  # both sessions contributed


def test_parallel_eval_and_on_batch_callback(tmp_path):
    cfg = DistributedConfig(total_budget=8, n_init=8, batch_size=4, poll_seconds=0.0, n_workers=2)
    seen: list[int] = []
    run_distributed_session(
        _synthetic, tmp_path, cfg, session_id="s0", max_iters=20, on_batch=seen.append
    )
    x, _, _ = read_ledger(tmp_path)
    assert len(x) >= cfg.total_budget
    assert sum(seen) == len(x)  # on_batch reported every appended design


# --- async: claim vectors + validator ---


def test_active_claims_returns_vectors_and_excludes_stale(tmp_path):
    claims = tmp_path / "claims"
    v_live, v_stale = _vec(40), _vec(41)
    claim_designs(claims, set(), np.array([v_live]), ttl_seconds=10_000)
    claim_designs(claims, set(), np.array([v_stale]), ttl_seconds=10_000)
    stale_marker = claims / f"{design_hash(decode(v_stale).to_dict())}.claim"
    old = os.stat(stale_marker).st_mtime - 3600
    os.utime(stale_marker, (old, old))
    got = active_claims(claims, ttl_seconds=60)
    assert len(got) == 1  # stale one excluded
    assert np.allclose(got[0][1], v_live)  # vector recovered for X_pending conditioning


def _write_interval_rows(tmp_path, intervals):
    shard = shard_path(tmp_path, "s")
    with shard.open("a") as f:
        for d, fin in intervals:
            f.write(
                json.dumps({"session_id": "s", "dispatch_s": float(d), "finish_s": float(fin)})
                + "\n"
            )


def test_validate_async_flags_batch_signature(tmp_path):
    # 3 back-to-back waves of 4: each wave dispatched at once, finishing staggered → early workers
    # idle until their wave-mates finish → utilization ~0.67, below the async bar. Negative control.
    batch = [(4 * k, 4 * k + j) for k in range(3) for j in (1, 2, 3, 4)]
    _write_interval_rows(tmp_path, batch)
    v = validate_async(tmp_path, n_workers=4)
    assert v["is_async"] is False and v["utilization"] < 0.9


def test_validate_async_confirms_async_signature(tmp_path):
    # 4 workers continuously busy (back-to-back unit evals) → no idle → utilization ~1.0.
    asyncish = [(k, k + 1) for k in range(4) for _ in range(4)]
    _write_interval_rows(tmp_path, asyncish)
    v = validate_async(tmp_path, n_workers=4)
    assert v["is_async"] is True and v["utilization"] > 0.9


def test_preflight_async_check_passes():
    # Smoke test (no CFD) that the async machinery fills + refills the pool in THIS environment —
    # run before a multi-hour campaign. Passes on the mechanism (peak==n_workers, refilled to
    # 2*n_workers, no dups), NOT on smoke utilization (confounded by proposal cost with fast evals).
    r = preflight_async_check(n_workers=4)
    assert r["passed"] is True
    assert r["per_session"]["preflight"]["peak_concurrency"] == 4
    assert r["reached_budget"] and r["no_duplicates"]


# --- async: end-to-end behavior ---


def test_async_dispatches_on_completion_not_in_batches(tmp_path):
    # The headline guarantee: a freed worker gets new work immediately (no batch barrier), so the
    # workers stay busy — utilization near 1.0. A batch loop would idle early finishers (~0.67).
    cfg = DistributedConfig(total_budget=12, n_init=4, n_workers=4, poll_seconds=0.0, **_FAST_ACQ)
    run_async_session(_staggered_synthetic, tmp_path, cfg, session_id="s0")
    v = validate_async(tmp_path, n_workers=4)
    # Steady-window utilization ~0.84 here — decisively above the batch signature (~0.67), proving
    # dispatch-on-completion. In this fast-test regime the ~0.5s proposal (vs ~3.75s eval) keeps it
    # under the 0.9 is_async bar; the real campaign (2.8h evals) reaches ~1.0 → is_async True. The
    # 0.9 threshold itself is proven by the two control tests above.
    assert v["per_session"]["s0"]["utilization"] > 0.75
    assert v["per_session"]["s0"]["peak_concurrency"] == 4  # pool filled


def test_async_reaches_budget_no_duplicates(tmp_path):
    cfg = DistributedConfig(total_budget=12, n_init=4, n_workers=4, poll_seconds=0.0, **_FAST_ACQ)
    run_async_session(_staggered_synthetic, tmp_path, cfg, session_id="s0")
    x, y, hashes = read_ledger(tmp_path)
    assert len(x) >= cfg.total_budget
    assert len(hashes) == len(x) and np.isfinite(y).all()


def test_async_bad_session_index_raises(tmp_path):
    with pytest.raises(ValueError):
        run_async_session(_synthetic, tmp_path, session_id="s", session_index=2, n_sessions=2)


def test_async_explore_fraction_injects_exploration(tmp_path):
    # With explore_fraction>0, some BO-phase dispatches are space-filling Sobol (source="explore"),
    # so the search keeps probing new regions instead of only refining the incumbent.
    for i in range(8):
        append_eval(
            shard_path(tmp_path, "s"), _vec(i), (1.0, 0.1, 1e-3), session_id="s", source="sobol"
        )
    cfg = DistributedConfig(
        total_budget=16, n_init=8, n_workers=2, poll_seconds=0.0, explore_fraction=0.5, **_FAST_ACQ
    )
    run_async_session(_synthetic, tmp_path, cfg, session_id="s")
    srcs = [
        json.loads(ln)["source"]
        for ln in shard_path(tmp_path, "s").read_text().splitlines()
        if ln.strip()
    ]
    assert "explore" in srcs  # exploration was interleaved with BO


def test_async_recovers_from_worker_death(tmp_path, monkeypatch):
    # A worker dies mid-eval (BrokenProcessPool). The session must rebuild the pool and still reach
    # budget — the abandoned design's claim goes stale (short TTL) and is reclaimed. (Finding B.)
    monkeypatch.setenv("ASYNC_KILL_FLAG", str(tmp_path / "kill.flag"))
    cfg = DistributedConfig(
        total_budget=8, n_init=4, n_workers=4, poll_seconds=0.05, claim_ttl_seconds=1.0, **_FAST_ACQ
    )
    run_async_session(_worker_killer_once, tmp_path, cfg, session_id="s0")
    x, _, hashes = read_ledger(tmp_path)
    assert len(x) >= cfg.total_budget  # rebuilt after the death and completed
    assert len(hashes) == len(x)


def test_async_conditions_acquisition_on_other_sessions_inflight(tmp_path, monkeypatch):
    # Seed n_init completed + one in-flight claim from a "peer" session; the first BO dispatch must
    # pass that peer's in-flight vector as X_pending so it doesn't duplicate running work.
    for i in range(6):
        append_eval(
            shard_path(tmp_path, "s"), _vec(i), (1.0 + i, 0.1, 1e-3), session_id="s", source="sobol"
        )
    peer = _vec(99)
    claim_designs(tmp_path / "claims", set(), np.array([peer]), ttl_seconds=10_000)
    seen: list = []
    real = dc.propose_candidates
    monkeypatch.setattr(
        dc,
        "propose_candidates",
        lambda *a, **k: (seen.append(k.get("X_pending")), real(*a, **k))[1],
    )
    cfg = DistributedConfig(total_budget=8, n_init=6, n_workers=1, poll_seconds=0.0)
    run_async_session(_synthetic, tmp_path, cfg, session_id="s", session_index=0, n_sessions=1)
    pend = [p for p in seen if p is not None]
    assert pend and any(np.allclose(peer, row) for p in pend for row in np.atleast_2d(p))


def test_pareto_from_ledger(tmp_path):
    shard = shard_path(tmp_path, "s")
    append_eval(shard, _vec(1), (1.0, 0.5, 1e-3), session_id="s", source="bo")
    append_eval(shard, _vec(2), (2.0, 0.5, 1e-3), session_id="s", source="bo")  # dominates #1
    front = pareto_from_ledger(tmp_path)
    assert any(d["j_fan"] == 2.0 for d in front)


def test_pareto_drops_failed_nan_designs(tmp_path):
    # A diverged CFD row (NaN) is neither dominated nor dominating; it must NOT be reported as a
    # spurious Pareto design (regression: L3 now persists NaN rows instead of raising).
    nan = float("nan")
    shard = shard_path(tmp_path, "s")
    append_eval(shard, _vec(1), (2.0, 0.5, 1e-3), session_id="s", source="bo")
    append_eval(shard, _vec(2), (nan, nan, nan), session_id="s", source="bo")
    front = pareto_from_ledger(tmp_path)
    assert len(front) == 1 and front[0]["j_fan"] == 2.0
    assert all(np.isfinite(d["j_fan"]) for d in front)


def test_trust_region_is_updated_during_bo(tmp_path, monkeypatch):
    # M1: the TuRBO trust region must actually advance (shrink/grow) — not sit frozen. Spy on the
    # class update to prove the BO branch calls it.
    for i in range(8):
        append_eval(
            shard_path(tmp_path, "s"), _vec(i), (1.0 + i, 0.1, 1e-3), session_id="s", source="sobol"
        )
    calls: list[bool] = []
    real = dc.TrustRegionState.update
    monkeypatch.setattr(
        dc.TrustRegionState,
        "update",
        lambda self, improved: (calls.append(improved), real(self, improved))[1],
    )
    cfg = DistributedConfig(total_budget=12, n_init=8, batch_size=2, poll_seconds=0.0)
    run_distributed_session(_synthetic, tmp_path, cfg, session_id="s", max_iters=5)
    assert len(calls) >= 1  # TR advanced at least once in the BO phase


# --- seed injection (Stage-3 trapezoid cold rerun) ---


def test_coldstart_prepends_seed_designs():
    # Seeds sit at the FRONT of the cold-start sequence (dispatched ahead of the Sobol DoE); the
    # DoE keeps its full n_init size, so the cold-start grows by exactly the seed count.
    seeds = stage3_seed_designs()
    cfg = DistributedConfig(n_init=6, seed_designs=seeds)
    cold = dc._coldstart_designs(cfg)
    assert len(cold) == 6 + len(seeds)
    for i, s in enumerate(seeds):
        assert np.allclose(cold[i], clip_to_bounds(np.asarray(s, dtype=float)))


def test_coldstart_without_seeds_is_plain_sobol():
    cfg = DistributedConfig(n_init=6)  # seed_designs defaults to None
    assert len(dc._coldstart_designs(cfg)) == 6


def test_injected_seeds_land_exactly_once_single_session(tmp_path):
    seeds = stage3_seed_designs()
    cfg = DistributedConfig(
        total_budget=12, n_init=6, batch_size=4, poll_seconds=0.0, seed_designs=seeds
    )
    run_distributed_session(_synthetic, tmp_path, cfg, session_id="s0", max_iters=50)
    rows = _ledger_hashes(tmp_path)
    for h in _seed_hashes():
        assert rows.count(h) == 1  # each seed dispatched, evaluated, and NOT duplicated


def test_injected_seeds_land_once_across_two_sessions(tmp_path):
    # Two interleaved sessions both carry the same seed_designs; the claim/ledger dedup must ensure
    # each seed is physically evaluated exactly once (not once per session) — the headline guarantee.
    seeds = stage3_seed_designs()
    cfg = DistributedConfig(
        total_budget=16, n_init=8, batch_size=4, poll_seconds=0.0, seed_designs=seeds
    )
    for _ in range(60):
        if len(read_ledger(tmp_path)[0]) >= cfg.total_budget:
            break
        run_distributed_session(
            _synthetic, tmp_path, cfg, session_id="A", session_index=0, n_sessions=2, max_iters=1
        )
        run_distributed_session(
            _synthetic, tmp_path, cfg, session_id="B", session_index=1, n_sessions=2, max_iters=1
        )
    rows = _ledger_hashes(tmp_path)
    for h in _seed_hashes():
        assert rows.count(h) == 1  # deduped across sessions — no seed run twice
    assert _raw_row_count(tmp_path) == len(read_ledger(tmp_path)[2])  # nothing run twice at all


def test_async_injects_seeds_exactly_once(tmp_path):
    seeds = stage3_seed_designs()
    cfg = DistributedConfig(
        total_budget=8, n_init=4, n_workers=2, poll_seconds=0.0, seed_designs=seeds, **_FAST_ACQ
    )
    run_async_session(_synthetic, tmp_path, cfg, session_id="s0")
    rows = _ledger_hashes(tmp_path)
    for h in _seed_hashes():
        assert rows.count(h) == 1  # each seed evaluated once through the async claim/ledger path


def test_doe_completes_despite_a_departed_session_slice(tmp_path):
    # Session 0 of a declared 2 owns only even DoE indices. Without DoE mop-up the odd indices
    # (session 1 never launched / died) would never be evaluated → cold-start deadlocks and the
    # run never reaches BO. Mop-up lets the survivor finish the whole DoE.
    cfg = DistributedConfig(total_budget=12, n_init=8, batch_size=4, poll_seconds=0.0)
    run_distributed_session(
        _synthetic, tmp_path, cfg, session_id="only", session_index=0, n_sessions=2, max_iters=60
    )
    x, _, _ = read_ledger(tmp_path)
    assert len(x) >= cfg.total_budget  # DoE finished + BO ran despite the missing session-1 slice
