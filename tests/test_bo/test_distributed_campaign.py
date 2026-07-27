"""Tests for the async shared-ledger distributed BO (Stage 3). Requires botorch (the [bo] extra)."""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pytest

if importlib.util.find_spec("botorch") is None:
    pytest.skip("botorch (the [bo] extra) required", allow_module_level=True)

from fanopt.bo import distributed_campaign as dc
from fanopt.bo.blade_codec import N_DIMS, bounds, clip_to_bounds, decode
from fanopt.bo.distributed_campaign import (
    DistributedConfig,
    append_eval,
    claim_designs,
    pareto_from_ledger,
    read_ledger,
    run_distributed_session,
    shard_path,
)
from fanopt.utils.ledger import design_hash


def _vec(seed: int) -> np.ndarray:
    low, high = bounds()
    rng = np.random.default_rng(seed)
    return clip_to_bounds(low + rng.random(N_DIMS) * (high - low))


def _synthetic(v: np.ndarray) -> tuple[float, float, float]:
    # finite, varied objective — not meaningful, just exercises the loop
    return (float(np.sum(np.cos(v[:5]))), float(1.0 + np.sum(np.abs(v)) * 1e-3), 1e-3)


def _raw_row_count(shared) -> int:
    return sum(
        1
        for shard in shared.glob(dc.LEDGER_GLOB)
        for ln in shard.read_text().splitlines()
        if ln.strip()
    )


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
