"""Tests for scripts/analyze_phase4.py (campaign → analysis.json + Pareto plot)."""

from __future__ import annotations

import json

import numpy as np

import analyze_phase4 as script
from fanopt.bo.codec import encode
from fanopt.bo.results import CHECKPOINT_NAME
from fanopt.geometry.envelope import Layer1Params, ThicknessGridField


def _vec(blade_count: int, thickness: float) -> np.ndarray:
    return encode(
        Layer1Params(
            blade_count=blade_count,
            camber_knots_m=(0.001, 0.001, 0.001),
            twist_knots_rad=(0.0, 0.0),
            thickness_field=ThicknessGridField.uniform(thickness),
            edge_profile="rounded",
            fourier_le_amplitudes=(0.0, 0.0, 0.0),
            fourier_te_amplitudes=(0.0, 0.0, 0.0),
        )
    )


def _fake_campaign(tmp_path):
    x = np.array([_vec(8, 0.0022), _vec(10, 0.003), _vec(12, 0.0038)])
    y_raw = np.array([[2.0, 0.004, 0.001], [1.5, 0.005, 0.0009], [1.0, 0.006, 0.0008]])
    np.savez(tmp_path / CHECKPOINT_NAME, x=x, y_raw=y_raw, iteration=1)
    return tmp_path


def test_main_writes_analysis_and_plot(tmp_path):
    _fake_campaign(tmp_path)
    rc = script.main(["--out-dir", str(tmp_path), "--top-k", "2"])
    assert rc == 0
    summary = json.loads((tmp_path / "analysis.json").read_text())
    assert summary["n_pareto"] == 3
    assert len(summary["top_k_diverse"]) == 2
    assert (tmp_path / "pareto.png").exists()


def test_no_plot_skips_png(tmp_path):
    _fake_campaign(tmp_path)
    rc = script.main(["--out-dir", str(tmp_path), "--no-plot"])
    assert rc == 0
    assert (tmp_path / "analysis.json").exists()
    assert not (tmp_path / "pareto.png").exists()


def test_run_returns_summary(tmp_path):
    _fake_campaign(tmp_path)
    summary = script.run(out_dir=tmp_path, top_k=3, plot=False)
    assert summary["n_evaluations"] == 3
    assert all("blade_count" in d for d in summary["pareto"])
