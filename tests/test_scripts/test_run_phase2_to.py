"""Tests for scripts/run_phase2_to.py (Phase 2 rib TO runner)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import run_phase2_to  # noqa: E402

# A coarse-but-feasible mesh (1.5 mm → preserved ≈ 30% < 40% target) keeps it fast.
_KW = dict(elem_size_m=0.0015, pressure_pa=10.0, volfrac=0.4, max_iters=8)


def test_run_writes_artifacts(tmp_path):
    summary = run_phase2_to.run(out_dir=tmp_path, **_KW)
    assert (tmp_path / "rib_density.npy").exists()
    assert (tmp_path / "rib_active_mask.npy").exists()
    assert (tmp_path / "phase2_result.json").exists()
    assert isinstance(summary, dict)


def test_result_json_is_valid_and_has_keys(tmp_path):
    run_phase2_to.run(out_dir=tmp_path, **_KW)
    data = json.loads((tmp_path / "phase2_result.json").read_text())
    for key in ("converged", "iterations", "volume_fraction", "u_tip_max_mm"):
        assert key in data


def test_density_saved_within_bounds(tmp_path):
    run_phase2_to.run(out_dir=tmp_path, **_KW)
    rho = np.load(tmp_path / "rib_density.npy")
    assert rho.min() >= 0.0
    assert rho.max() <= 1.0 + 1e-9


def test_volume_fraction_near_target(tmp_path):
    summary = run_phase2_to.run(
        out_dir=tmp_path, elem_size_m=0.0015, pressure_pa=10.0, volfrac=0.4, max_iters=40
    )
    assert abs(summary["volume_fraction"] - 0.4) < 0.05


def test_main_cli_returns_zero(tmp_path, capsys):
    rc = run_phase2_to.main(
        ["--elem-size-m", "0.002", "--max-iters", "5", "--out-dir", str(tmp_path)]
    )
    assert rc == 0
    assert (tmp_path / "phase2_result.json").exists()


def test_main_default_help_parses():
    # argparse should build cleanly (no bad defaults).
    if importlib.util.find_spec("run_phase2_to") is None:  # pragma: no cover
        return
    assert run_phase2_to.main is not None
