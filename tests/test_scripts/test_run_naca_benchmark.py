"""Tests for scripts/run_naca_benchmark.py (mesh+SU2 boundary mocked)."""

from __future__ import annotations

import json

import run_naca_benchmark as script
from fanopt.cfd.naca_benchmark import BenchmarkMetrics

_FAKE = BenchmarkMetrics(
    c_l_max=1.23, c_d_mean=0.045, hysteresis_area=0.31, alpha_at_cl_max_deg=8.5, n_cycles_used=4
)


def test_run_writes_metrics_json(tmp_path, monkeypatch):
    monkeypatch.setattr(script, "run_benchmark", lambda cfg, workdir, su2_bin=None: _FAKE)
    metrics = script.run(
        workdir=tmp_path,
        reynolds_number=40000.0,
        reduced_frequency_k=0.55,
        pitch_amplitude_deg=10.0,
        n_cycles=5,
        steps_per_cycle=200,
    )
    assert metrics.c_l_max == 1.23
    written = json.loads((tmp_path / "metrics.json").read_text())
    assert written["c_d_mean"] == 0.045
    assert written["n_cycles_used"] == 4


def test_main_returns_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(script, "run_benchmark", lambda cfg, workdir, su2_bin=None: _FAKE)
    rc = script.main(["--workdir", str(tmp_path), "--re", "30000", "--cycles", "3"])
    assert rc == 0
    assert (tmp_path / "metrics.json").exists()
