"""Tests for scripts/run_phase3_correlation.py.

The pure summary builder is unit-tested directly; the SU2-running happy path is
exercised by the local integration sweep. The no-SU2 guard is tested at the real
boundary (env + PATH), not by mocking project internals.
"""

from __future__ import annotations

import importlib.util
import json

import pytest

if importlib.util.find_spec("gmsh") is None:  # pragma: no cover - env-dependent
    pytest.skip("gmsh not installed", allow_module_level=True)

import run_phase3_correlation as script
from fanopt.cfd import phase3
from fanopt.cfd.correlation import CorrelationResult
from fanopt.cfd.phase3 import DesignResult


def _corr(passed: bool = True) -> CorrelationResult:
    return CorrelationResult(
        r2=0.82, pearson_r=0.905, kendall_tau=0.7, n=6, passed=passed, meta={"threshold": 0.4}
    )


def test_summary_shape_and_rounding():
    results = [DesignResult("b3_t22", 1.07, -1.9e11, 4.6e13, meta={"n_blades": 3.0})]
    out = script._summary(_corr(), results)
    assert out["metric"] == "steady_cd_vs_unsteady_rms"
    assert out["r2"] == 0.82
    assert out["passed"] is True
    assert out["threshold"] == 0.4
    assert out["designs"][0]["name"] == "b3_t22"
    assert out["designs"][0]["unsteady_rms"] == 4.6e13
    assert out["designs"][0]["meta"]["n_blades"] == 3.0


def test_summary_is_json_serializable():
    out = script._summary(_corr(passed=False), [DesignResult("d", 1.0, 2.0, 3.0)])
    text = json.dumps(out)  # must not raise
    assert "passed" in text


def test_run_raises_without_su2(tmp_path, monkeypatch):
    # Real no-SU2 boundary: no $SU2_RUN and nothing named SU2_CFD on PATH.
    monkeypatch.delenv("SU2_RUN", raising=False)
    monkeypatch.setattr(phase3.shutil, "which", lambda _n: None)
    with pytest.raises(RuntimeError, match="SU2_CFD not found"):
        script.run(out_dir=tmp_path, su2_bin=None)


def test_main_writes_correlation_json_on_success(tmp_path, monkeypatch):
    # Substitute the SU2-driving sweep (the subprocess/filesystem boundary) so
    # the CLI wiring — arg parsing → run() → JSON write → exit 0 — is exercised.
    def fake_sweep(workdir, *, designs=None, su2_bin=None):
        return _corr(), [DesignResult("b3_t22", 1.07, -1.9e11, 4.6e13)]

    monkeypatch.setattr(script, "run_correlation_sweep", fake_sweep)
    rc = script.main(["--out-dir", str(tmp_path)])
    assert rc == 0
    written = json.loads((tmp_path / "correlation.json").read_text())
    assert written["passed"] is True
    assert written["designs"][0]["name"] == "b3_t22"
