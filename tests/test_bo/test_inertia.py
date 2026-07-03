"""Tests for fanopt.bo.inertia (I_wrist objective). Requires CadQuery."""

from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("cadquery") is None:  # pragma: no cover - env-dependent
    pytest.skip("cadquery not installed", allow_module_level=True)

from fanopt.bo.inertia import fan_i_wrist_kgm2
from fanopt.geometry.envelope import Layer1Params, ThicknessGridField


def _layer1(blade_count: int) -> Layer1Params:
    return Layer1Params(
        blade_count=blade_count,
        camber_knots_m=(0.001, 0.002, 0.001),
        twist_knots_rad=(0.05, -0.05),
        thickness_field=ThicknessGridField.uniform(0.003),
        edge_profile="rounded",
        fourier_le_amplitudes=(0.0, 0.0, 0.0),
        fourier_te_amplitudes=(0.0, 0.0, 0.0),
    )


def test_i_wrist_is_positive_finite():
    i_wrist = fan_i_wrist_kgm2(_layer1(8))
    assert i_wrist > 0.0
    assert i_wrist < 1.0  # sanity: a ~100 g fan at ~0.2 m ⇒ O(1e-3) kg·m²


def test_more_blades_more_inertia():
    assert fan_i_wrist_kgm2(_layer1(12)) > fan_i_wrist_kgm2(_layer1(8))
