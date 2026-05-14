"""Unit tests for fanopt.geometry.envelope (Layer 1 BO design parameters)."""
from __future__ import annotations

import math

import pytest

from fanopt.geometry.envelope import (
    CAMBER_RANGE_M,
    EDGE_PROFILES,
    FOURIER_AMPLITUDE_RELATIVE_MAX,
    TWIST_RANGE_RAD,
    Layer1Params,
)
from fanopt.geometry.schema import (
    BLADE_COUNTS,
    PANEL_THICKNESS_MAX_M,
    PANEL_THICKNESS_MIN_M,
)


def _canonical_kwargs() -> dict:
    """Return a valid Layer1Params kwargs dict for derivative tests."""
    return dict(
        blade_count=10,
        camber_knots_m=(0.001, 0.002, 0.001),
        twist_knots_rad=(0.0, 0.0),
        thickness_knots_m=(0.0030, 0.0028, 0.0026),
        edge_profile="rounded",
        fourier_le_amplitudes=(0.05, 0.0, 0.0),
        fourier_te_amplitudes=(0.0, 0.05, 0.0),
    )


def test_canonical_design_validates() -> None:
    """A reasonable default design constructs without error."""
    p = Layer1Params(**_canonical_kwargs())
    assert p.blade_count == 10
    assert p.edge_profile == "rounded"


def test_blade_count_rejects_14() -> None:
    """14 is MED-10-retired per Round 7. BLADE_COUNTS = (8, 10, 12)."""
    kw = _canonical_kwargs()
    kw["blade_count"] = 14
    with pytest.raises(ValueError, match="blade_count"):
        Layer1Params(**kw)


def test_blade_count_accepts_all_three_locked_values() -> None:
    for bc in BLADE_COUNTS:
        kw = _canonical_kwargs()
        kw["blade_count"] = bc
        p = Layer1Params(**kw)
        assert p.blade_count == bc


def test_camber_above_5mm_fails() -> None:
    kw = _canonical_kwargs()
    kw["camber_knots_m"] = (0.001, 0.006, 0.001)  # 6 mm > 5 mm cap
    with pytest.raises(ValueError, match="camber_knots_m"):
        Layer1Params(**kw)


def test_camber_below_zero_fails() -> None:
    kw = _canonical_kwargs()
    kw["camber_knots_m"] = (0.001, -0.001, 0.001)
    with pytest.raises(ValueError, match="camber_knots_m"):
        Layer1Params(**kw)


def test_camber_accepts_3_or_4_knots() -> None:
    for n in (3, 4):
        kw = _canonical_kwargs()
        kw["camber_knots_m"] = tuple(0.001 for _ in range(n))
        Layer1Params(**kw)


def test_camber_rejects_2_knots() -> None:
    kw = _canonical_kwargs()
    kw["camber_knots_m"] = (0.001, 0.002)
    with pytest.raises(ValueError, match="camber_knots_m"):
        Layer1Params(**kw)


def test_twist_outside_range_fails() -> None:
    kw = _canonical_kwargs()
    kw["twist_knots_rad"] = (0.0, math.radians(15.0))  # 15° > 10° cap
    with pytest.raises(ValueError, match="twist_knots_rad"):
        Layer1Params(**kw)


def test_thickness_below_min_fails() -> None:
    kw = _canonical_kwargs()
    kw["thickness_knots_m"] = (0.001, 0.003, 0.003)  # 1 mm < 2.2 mm floor
    with pytest.raises(ValueError, match="thickness_knots_m"):
        Layer1Params(**kw)


def test_thickness_above_max_fails() -> None:
    kw = _canonical_kwargs()
    kw["thickness_knots_m"] = (0.003, 0.005, 0.003)  # 5 mm > 3.8 mm cap
    with pytest.raises(ValueError, match="thickness_knots_m"):
        Layer1Params(**kw)


def test_thickness_requires_exactly_three_knots() -> None:
    kw = _canonical_kwargs()
    kw["thickness_knots_m"] = (0.003, 0.003)  # 2 knots, not 3
    with pytest.raises(ValueError, match="thickness_knots_m"):
        Layer1Params(**kw)


def test_thickness_at_locked_bounds_passes() -> None:
    """The locked PANEL_THICKNESS_MIN/MAX_M are inclusive bounds."""
    kw = _canonical_kwargs()
    kw["thickness_knots_m"] = (
        PANEL_THICKNESS_MIN_M,
        PANEL_THICKNESS_MAX_M,
        PANEL_THICKNESS_MIN_M,
    )
    Layer1Params(**kw)


def test_edge_profile_unknown_fails() -> None:
    kw = _canonical_kwargs()
    kw["edge_profile"] = "thicc-bois"
    with pytest.raises(ValueError, match="edge_profile"):
        Layer1Params(**kw)


def test_edge_profile_all_locked_values_pass() -> None:
    for ep in EDGE_PROFILES:
        kw = _canonical_kwargs()
        kw["edge_profile"] = ep
        Layer1Params(**kw)


def test_fourier_amplitude_over_15pct_fails() -> None:
    kw = _canonical_kwargs()
    kw["fourier_le_amplitudes"] = (0.20, 0.0, 0.0)  # 20% > 15%
    with pytest.raises(ValueError, match="fourier_le_amplitudes"):
        Layer1Params(**kw)


def test_fourier_negative_amplitude_at_bound_passes() -> None:
    """The ±15% bound is symmetric; -0.15 is allowed."""
    kw = _canonical_kwargs()
    kw["fourier_te_amplitudes"] = (-FOURIER_AMPLITUDE_RELATIVE_MAX, 0.0, 0.0)
    Layer1Params(**kw)


def test_fourier_requires_exactly_three_harmonics() -> None:
    kw = _canonical_kwargs()
    kw["fourier_le_amplitudes"] = (0.05, 0.0)  # k=1,2 only
    with pytest.raises(ValueError, match="fourier_le_amplitudes"):
        Layer1Params(**kw)


def test_round_trip_to_from_dict() -> None:
    p = Layer1Params(**_canonical_kwargs())
    d = p.to_dict()
    recovered = Layer1Params.from_dict(d)
    assert recovered == p


def test_camber_at_5mm_exactly_passes() -> None:
    """CAMBER_RANGE_M upper bound is inclusive."""
    kw = _canonical_kwargs()
    kw["camber_knots_m"] = (0.0, CAMBER_RANGE_M[1], 0.0)
    Layer1Params(**kw)


def test_twist_at_negative_10deg_exactly_passes() -> None:
    kw = _canonical_kwargs()
    kw["twist_knots_rad"] = (TWIST_RANGE_RAD[0], TWIST_RANGE_RAD[1])
    Layer1Params(**kw)
