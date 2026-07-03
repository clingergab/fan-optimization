"""Tests for fanopt.bo.architecture (V1-slim architecture enumeration + growth gate)."""

from __future__ import annotations

import pytest

from fanopt.bo import architecture as arch
from fanopt.bo.codec import decode, encode
from fanopt.geometry.envelope import Layer1Params, ThicknessGridField
from fanopt.geometry.schema import BLADE_COUNTS


def test_default_enum_loads_and_enumerates():
    space = arch.load_architecture_space()
    combos = space.enumerate()
    assert space.count == len(combos)
    assert space.count == 6  # 3 blade_count × 2 print_orientation


def test_enumeration_covers_cartesian_product():
    space = arch.load_architecture_space()
    combos = space.enumerate()
    blade_counts = {c["blade_count"] for c in combos}
    orientations = {c["print_orientation"] for c in combos}
    assert blade_counts == set(BLADE_COUNTS)
    assert orientations == {"flat", "edge"}


def test_blade_count_axis_matches_codec_categorical():
    space = arch.load_architecture_space()
    # The enumeration's blade_count values must be exactly what the codec searches.
    assert tuple(space.axes["blade_count"]) == BLADE_COUNTS
    for bc in space.axes["blade_count"]:
        p = Layer1Params(
            blade_count=bc,
            camber_knots_m=(0.001, 0.002, 0.001),
            twist_knots_rad=(0.05, -0.05),
            thickness_field=ThicknessGridField.uniform(),
            edge_profile="rounded",
            fourier_le_amplitudes=(0.0, 0.0, 0.0),
            fourier_te_amplitudes=(0.0, 0.0, 0.0),
        )
        assert decode(encode(p)).blade_count == bc


def test_growth_gate_ceiling_uses_min_floor():
    # Small spaces are governed by the 50 floor, not the 1.1× factor.
    assert arch.growth_gate_ceiling(6) == 50
    assert arch.growth_gate_ceiling(100) == 110


def test_growth_gate_allows_within_ceiling():
    assert arch.check_growth_gate(6, 6) is True
    assert arch.check_growth_gate(100, 110) is True


def test_growth_gate_blocks_over_ceiling():
    assert arch.check_growth_gate(100, 111) is False


def test_load_rejects_unknown_axis_value(tmp_path):
    bad = tmp_path / "arch.yaml"
    bad.write_text("axes:\n  blade_count: [8, 14]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        arch.load_architecture_space(bad)


def test_load_rejects_empty_axes(tmp_path):
    bad = tmp_path / "arch.yaml"
    bad.write_text("version: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no 'axes'"):
        arch.load_architecture_space(bad)


def test_load_rejects_empty_axis(tmp_path):
    bad = tmp_path / "arch.yaml"
    bad.write_text("axes:\n  blade_count: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        arch.load_architecture_space(bad)
