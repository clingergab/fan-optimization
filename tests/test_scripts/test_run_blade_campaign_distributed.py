"""Tests for scripts/run_blade_campaign_distributed.py. Requires botorch (the [bo] extra)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

if importlib.util.find_spec("botorch") is None:
    pytest.skip("botorch (the [bo] extra) required", allow_module_level=True)

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import run_blade_campaign_distributed as script  # noqa: E402
from fanopt.bo.blade_campaign import stage3_seed_designs  # noqa: E402
from fanopt.bo.blade_codec import decode  # noqa: E402
from fanopt.bo.distributed_campaign import read_ledger  # noqa: E402
from fanopt.cfd.phase5 import VerifyConfig  # noqa: E402
from fanopt.utils.ledger import design_hash  # noqa: E402


def _synthetic(v: np.ndarray) -> tuple[float, float, float]:
    return (float(np.sum(np.cos(v[:5]))), float(1.0 + np.sum(np.abs(v)) * 1e-3), 1e-3)


def test_verify_config_overrides():
    args = script.build_parser().parse_args(
        ["--shared-dir", "/tmp/x", "--session-id", "s", "--n-cycles", "7", "--inner-iter", "45"]
    )
    cfg = script._verify_config(args)
    assert cfg.n_cycles == 7 and cfg.inner_iter == 45


def test_verify_config_defaults_when_unset():
    args = script.build_parser().parse_args(["--shared-dir", "/tmp/x", "--session-id", "s"])
    cfg = script._verify_config(args)
    assert cfg.n_cycles == VerifyConfig().n_cycles and cfg.inner_iter == VerifyConfig().inner_iter


def test_main_wires_injected_objective_into_the_loop(tmp_path):
    rc = script.main(
        [
            "--shared-dir",
            str(tmp_path),
            "--session-id",
            "s0",
            "--budget",
            "8",
            "--n-init",
            "8",
            "--batch-size",
            "4",
            "--poll-seconds",
            "0",
        ],
        objective_fn=_synthetic,
    )
    assert rc == 0
    x, _, hashes = read_ledger(tmp_path)
    assert len(x) >= 8
    assert len(hashes) == len(x)  # no duplicate designs


def test_inject_seeds_flag_dispatches_both_seeds(tmp_path):
    # --inject-seeds must wire the two operator-locked seeds into the campaign so they land in the
    # ledger (ahead of the Sobol DoE). Sync loop for determinism; synthetic objective (no CFD).
    rc = script.main(
        [
            "--shared-dir", str(tmp_path), "--session-id", "s0",
            "--budget", "10", "--n-init", "4", "--batch-size", "4",
            "--poll-seconds", "0", "--sync", "--inject-seeds",
        ],
        objective_fn=_synthetic,
    )
    assert rc == 0
    _, _, hashes = read_ledger(tmp_path)
    seed_hashes = {design_hash(decode(s).to_dict()) for s in stage3_seed_designs()}
    assert seed_hashes <= hashes  # both seeds were dispatched and landed


def test_no_inject_seeds_flag_leaves_seeds_out(tmp_path):
    rc = script.main(
        [
            "--shared-dir", str(tmp_path), "--session-id", "s0",
            "--budget", "8", "--n-init", "8", "--batch-size", "4",
            "--poll-seconds", "0", "--sync",
        ],
        objective_fn=_synthetic,
    )
    assert rc == 0
    _, _, hashes = read_ledger(tmp_path)
    seed_hashes = {design_hash(decode(s).to_dict()) for s in stage3_seed_designs()}
    assert not (seed_hashes & hashes)  # no seeds injected → the DoE is pure Sobol


def test_main_errors_without_su2_and_no_injected_objective(tmp_path, monkeypatch):
    monkeypatch.setattr(script, "find_su2", lambda: None)
    with pytest.raises(SystemExit):
        script.main(["--shared-dir", str(tmp_path), "--session-id", "s0", "--su2-bin", ""])
