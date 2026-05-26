"""Tests for scripts/smoke_test.py — end-to-end JSON -> properties."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

if importlib.util.find_spec("cadquery") is None:
    pytest.skip("cadquery not installed", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import smoke_test
from fanopt.geometry.envelope import Layer1Params
from fanopt.geometry.fields import Layer2Params
from fanopt.geometry.generator import BladeDesignParams
from fanopt.geometry.manufacturability import Layer4Params
from fanopt.geometry.primitives import Layer3Primitive


def _design() -> BladeDesignParams:
    return BladeDesignParams(
        layer1=Layer1Params(
            blade_count=8,
            camber_knots_m=(0.0, 0.002, 0.001),
            twist_knots_rad=(0.0, 0.0),
            thickness_knots_m=(0.0030, 0.0028, 0.0026),
            edge_profile="rounded",
            fourier_le_amplitudes=(0.0, 0.0, 0.0),
            fourier_te_amplitudes=(0.0, 0.0, 0.0),
        ),
        layer2=Layer2Params.all_inactive(),
        layer3=Layer3Primitive.absent(),
        layer4=Layer4Params(
            print_orientation="flat",
            layer_height_m=0.0002,
            click_chamfer_angle_deg=45.0,
            click_detent_size_m=0.0004,
            click_design_clearance_m=0.00018,
        ),
    )


def test_compute_smoke_summary_returns_full_payload() -> None:
    summary = smoke_test.compute_smoke_summary(_design())
    expected_keys = {
        "status",
        "blade_count",
        "blade_mass_kg",
        "total_mass_kg",
        "total_mass_under_cap",
        "mass_cap_kg",
        "centre_of_mass_m",
        "i_wrist_kgm2",
        "manufacturability_score",
        "manufacturability_passed",
        "critical_failures",
        "pending_cadquery",
    }
    assert expected_keys.issubset(summary.keys())


def test_compute_smoke_summary_reports_mass_cap_status() -> None:
    """The test canonical design is heavy (thickness at schema max);
    verify the cap-check field exists and is correctly typed."""
    summary = smoke_test.compute_smoke_summary(_design())
    assert isinstance(summary["total_mass_under_cap"], bool)
    assert summary["mass_cap_kg"] == 0.100
    assert summary["total_mass_kg"] > 0.0


def test_compute_smoke_summary_emits_finite_i_wrist() -> None:
    import math

    summary = smoke_test.compute_smoke_summary(_design())
    assert math.isfinite(summary["i_wrist_kgm2"])
    assert summary["i_wrist_kgm2"] > 0.0


def test_main_emits_json_to_stdout(tmp_path: Path, capsys) -> None:
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps(_design().to_dict()))

    rc = smoke_test.main(["--params", str(params_path)])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["blade_count"] == 8
    assert "i_wrist_kgm2" in payload


def test_main_writes_to_out_path(tmp_path: Path) -> None:
    params_path = tmp_path / "params.json"
    out_path = tmp_path / "summary.json"
    params_path.write_text(json.dumps(_design().to_dict()))

    rc = smoke_test.main(["--params", str(params_path), "--out", str(out_path)])
    assert rc == 0
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["blade_count"] == 8


def test_main_returns_2_on_missing_params(tmp_path: Path) -> None:
    rc = smoke_test.main(["--params", str(tmp_path / "nope.json")])
    assert rc == 2


def test_main_returns_2_on_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json {{")
    rc = smoke_test.main(["--params", str(bad)])
    assert rc == 2


def test_main_returns_1_on_manufacturability_fail(tmp_path: Path, monkeypatch) -> None:
    """Force a manufacturability failure by monkey-patching the filter."""
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps(_design().to_dict()))

    def fake_summary(_design):
        return {
            "status": "mfg_rejected",
            "blade_count": 8,
            "blade_mass_kg": 0.005,
            "total_mass_kg": 0.040,
            "total_mass_under_cap": True,
            "mass_cap_kg": 0.100,
            "centre_of_mass_m": [0.1, 0.0, 0.002],
            "i_wrist_kgm2": 1e-4,
            "manufacturability_score": 0.0,
            "manufacturability_passed": False,
            "critical_failures": ["3"],
            "pending_cadquery": [],
        }

    monkeypatch.setattr(smoke_test, "compute_smoke_summary", fake_summary)
    rc = smoke_test.main(["--params", str(params_path)])
    assert rc == 1
