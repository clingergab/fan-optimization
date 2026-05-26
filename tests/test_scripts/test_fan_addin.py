"""Tests for scripts/fan_addin.py — params.json -> STL + STEP export."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

if importlib.util.find_spec("cadquery") is None:
    pytest.skip("cadquery not installed", allow_module_level=True)

# scripts/ is on sys.path via pyproject conftest; import as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import fan_addin
from fanopt.geometry.envelope import Layer1Params
from fanopt.geometry.fields import Layer2Params
from fanopt.geometry.generator import BladeDesignParams
from fanopt.geometry.manufacturability import Layer4Params
from fanopt.geometry.primitives import Layer3Primitive


def _canonical_design_dict(blade_count: int = 8) -> dict:
    """Return a from_dict-compatible JSON-ready design with smaller
    blade_count so the test runs faster."""
    return BladeDesignParams(
        layer1=Layer1Params(
            blade_count=blade_count,
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
    ).to_dict()


def test_main_exports_per_blade_stl_and_assembly_step(tmp_path: Path) -> None:
    """End-to-end happy path: write a params.json, run main(), confirm
    N STLs and one STEP land in out-dir."""
    params_path = tmp_path / "params.json"
    out_dir = tmp_path / "exports"
    n = 8
    params_path.write_text(json.dumps(_canonical_design_dict(blade_count=n)))

    rc = fan_addin.main(["--params", str(params_path), "--out-dir", str(out_dir)])
    assert rc == 0

    for i in range(n):
        stl = out_dir / f"blade_{i}.stl"
        assert stl.exists(), f"blade_{i}.stl missing"
        assert stl.stat().st_size > 1000
    step = out_dir / "deployed_fan.step"
    assert step.exists()
    assert step.stat().st_size > 1000


def test_main_returns_2_on_missing_params_file(tmp_path: Path) -> None:
    rc = fan_addin.main(["--params", str(tmp_path / "nope.json"), "--out-dir", str(tmp_path / "out")])
    assert rc == 2


def test_main_returns_2_on_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json at all {{{")
    rc = fan_addin.main(["--params", str(bad), "--out-dir", str(tmp_path / "out")])
    assert rc == 2


def test_main_returns_2_on_schema_violation(tmp_path: Path) -> None:
    """A JSON that parses but fails BladeDesignParams.from_dict raises."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"layer1": {}}))  # missing required fields
    rc = fan_addin.main(["--params", str(bad), "--out-dir", str(tmp_path / "out")])
    assert rc == 2
