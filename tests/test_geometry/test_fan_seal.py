"""Tests for deployed-fan sealing analysis + rib thinning (fanopt.geometry.fan_seal)."""

from __future__ import annotations

import numpy as np
import pytest

from fanopt.bo.blade_codec import N_DIMS, decode
from fanopt.geometry.blade import RIB_THICKNESS_RANGE_M, layer_spacing_m
from fanopt.geometry.fan_seal import deployed_panel_gap_m, panel_thickness_range_m, thin_ribs

_P = decode(np.full(N_DIMS, 0.5))


def test_panel_thickness_range_ordered():
    lo, hi = panel_thickness_range_m(_P)
    assert 0 < lo <= hi


def test_thin_ribs_reduces_rib_thickness():
    thinned = thin_ribs(_P, panel_thickness_range_m(_P)[1])
    assert thinned.t_rib_hub_m < _P.t_rib_hub_m
    assert thinned.t_rib_tip_m == thinned.t_rib_hub_m


def test_thin_ribs_never_increases():
    # A target above the current rib must not grow it (thinning only).
    thinned = thin_ribs(_P, RIB_THICKNESS_RANGE_M[1])
    assert thinned.t_rib_hub_m == pytest.approx(_P.t_rib_hub_m)


def test_thin_ribs_floors_at_panel_thickness():
    # Rib must still contain the panel: a target below the thickest panel clamps up to it.
    thinned = thin_ribs(_P, 0.0005)  # 0.5 mm, below any panel
    assert thinned.t_rib_hub_m == pytest.approx(panel_thickness_range_m(_P)[1])


def test_thinning_ribs_shrinks_layer_spacing_and_gap():
    thinned = thin_ribs(_P, panel_thickness_range_m(_P)[1])
    assert layer_spacing_m(thinned) < layer_spacing_m(_P)  # blades sit closer
    assert deployed_panel_gap_m(thinned) < deployed_panel_gap_m(_P)  # smaller wind slot


def test_deployed_panel_gap_matches_definition():
    assert deployed_panel_gap_m(_P) == pytest.approx(
        layer_spacing_m(_P) - panel_thickness_range_m(_P)[0]
    )
