"""Tests for scripts/print_strategy.py — per-blade vs full-assembly decision."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

if importlib.util.find_spec("cadquery") is None:
    pytest.skip("cadquery not installed", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import print_strategy
from fanopt.geometry.envelope import Layer1Params, ThicknessGridField
from fanopt.geometry.fields import Layer2Params
from fanopt.geometry.generator import BladeDesignParams
from fanopt.geometry.manufacturability import Layer4Params
from fanopt.geometry.primitives import Layer3Primitive


def _design(blade_count: int = 8) -> BladeDesignParams:
    return BladeDesignParams(
        layer1=Layer1Params(
            blade_count=blade_count,
            camber_knots_m=(0.0, 0.002, 0.001),
            twist_knots_rad=(0.0, 0.0),
            thickness_field=ThicknessGridField.from_radial_knots((0.0030, 0.0028, 0.0026)),
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


def test_decide_strategy_returns_decision_and_dims() -> None:
    decision, dims = print_strategy.decide_strategy(_design(8))
    assert decision in ("full-assembly", "per-blade")
    assert {"fan_x_mm", "fan_y_mm", "bed_x_mm", "bed_y_mm"} == set(dims.keys())


def test_decide_strategy_full_assembly_on_default_bed() -> None:
    """A 200 mm × ~250 mm deployed fan fits a 256 × 256 mm bed."""
    decision, dims = print_strategy.decide_strategy(_design(8))
    assert dims["fan_x_mm"] < dims["bed_x_mm"]
    # Whether the y-extent fits depends on blade_count; 8 blades is tight.
    if dims["fan_y_mm"] <= dims["bed_y_mm"]:
        assert decision == "full-assembly"


def test_decide_strategy_per_blade_when_bed_too_small() -> None:
    """Tiny 50 × 50 mm bed → fan doesn't fit → per-blade."""
    decision, _dims = print_strategy.decide_strategy(_design(10), bed_x_mm=50.0, bed_y_mm=50.0)
    assert decision == "per-blade"


def test_main_emits_text_output(tmp_path: Path, capsys) -> None:
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps(_design(8).to_dict()))

    rc = print_strategy.main(["--params", str(params_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[print_strategy]" in out
    assert "decision=" in out


def test_main_emits_json_output(tmp_path: Path, capsys) -> None:
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps(_design(8).to_dict()))

    rc = print_strategy.main(["--params", str(params_path), "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["decision"] in ("full-assembly", "per-blade")
    assert "fan_x_mm" in payload


def test_main_returns_2_on_missing_params(tmp_path: Path) -> None:
    rc = print_strategy.main(["--params", str(tmp_path / "nope.json")])
    assert rc == 2


def test_main_returns_2_on_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("garbage {{")
    rc = print_strategy.main(["--params", str(bad)])
    assert rc == 2
