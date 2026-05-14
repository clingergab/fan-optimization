"""Unit tests for fanopt.geometry.primitives (Layer 3 BO design parameters)."""

from __future__ import annotations

import math

import pytest

from fanopt.geometry.primitives import (
    PRIMITIVE_MARGIN_FROM_EDGE_M,
    PRIMITIVE_MAX_FRACTION_OF_ENVELOPE,
    PRIMITIVE_MIN_DIMENSION_M,
    PRIMITIVE_TYPES,
    Layer3Primitive,
)

# 5 cm × 5 cm × 5 mm envelope — comfortable test box.
ENV = (0.050, 0.050, 0.005)


def _canonical_kwargs() -> dict:
    return dict(
        present=True,
        shape_type="slot",
        polarity="subtract",
        position_x_m=0.025,
        position_y_m=0.025,
        position_z_m=0.0025,
        size_x_m=0.005,
        size_y_m=0.005,
        size_z_m=0.001,
        rotation_x_rad=0.0,
        rotation_y_rad=0.0,
        rotation_z_rad=0.0,
        local_envelope_xyz_m=ENV,
    )


def test_absent_canonical() -> None:
    p = Layer3Primitive.absent()
    assert p.present is False


def test_present_canonical_validates() -> None:
    p = Layer3Primitive(**_canonical_kwargs())
    assert p.present is True
    assert p.shape_type == "slot"


def test_absent_skips_validation_of_other_fields() -> None:
    """A primitive with present=False and bad sizes still constructs."""
    p = Layer3Primitive(present=False, size_x_m=-99.0)
    assert p.present is False


def test_present_without_envelope_fails() -> None:
    kw = _canonical_kwargs()
    kw["local_envelope_xyz_m"] = None
    with pytest.raises(ValueError, match="local_envelope_xyz_m must be set"):
        Layer3Primitive(**kw)


def test_present_with_zero_envelope_fails() -> None:
    kw = _canonical_kwargs()
    kw["local_envelope_xyz_m"] = (0.0, 0.05, 0.005)
    with pytest.raises(ValueError, match="components must be > 0"):
        Layer3Primitive(**kw)


def test_unknown_shape_type_fails() -> None:
    kw = _canonical_kwargs()
    kw["shape_type"] = "torus"
    with pytest.raises(ValueError, match="shape_type"):
        Layer3Primitive(**kw)


def test_all_locked_shape_types_pass() -> None:
    for st in PRIMITIVE_TYPES:
        kw = _canonical_kwargs()
        kw["shape_type"] = st
        Layer3Primitive(**kw)


def test_position_violates_edge_margin_fails() -> None:
    """Position within 1 mm of the envelope edge violates the margin."""
    kw = _canonical_kwargs()
    kw["position_x_m"] = 0.0005  # 0.5 mm from edge < 1 mm margin
    with pytest.raises(ValueError, match="position_x_m"):
        Layer3Primitive(**kw)


def test_position_at_margin_exactly_passes() -> None:
    """The margin is inclusive."""
    kw = _canonical_kwargs()
    kw["position_x_m"] = PRIMITIVE_MARGIN_FROM_EDGE_M
    Layer3Primitive(**kw)


def test_position_at_far_edge_minus_margin_passes() -> None:
    kw = _canonical_kwargs()
    kw["position_y_m"] = ENV[1] - PRIMITIVE_MARGIN_FROM_EDGE_M
    Layer3Primitive(**kw)


def test_size_below_min_fails() -> None:
    kw = _canonical_kwargs()
    kw["size_z_m"] = PRIMITIVE_MIN_DIMENSION_M / 2
    with pytest.raises(ValueError, match="size_z_m"):
        Layer3Primitive(**kw)


def test_size_above_30pct_envelope_fails() -> None:
    """size_x ≤ 30% of envelope_x = 30% × 0.05 = 0.015 m."""
    kw = _canonical_kwargs()
    kw["size_x_m"] = PRIMITIVE_MAX_FRACTION_OF_ENVELOPE * ENV[0] + 0.0001
    with pytest.raises(ValueError, match="size_x_m"):
        Layer3Primitive(**kw)


def test_size_at_30pct_envelope_exactly_passes() -> None:
    kw = _canonical_kwargs()
    kw["size_x_m"] = PRIMITIVE_MAX_FRACTION_OF_ENVELOPE * ENV[0]
    Layer3Primitive(**kw)


def test_rotation_above_pi_fails() -> None:
    kw = _canonical_kwargs()
    kw["rotation_z_rad"] = math.pi + 0.01
    with pytest.raises(ValueError, match="rotation_z_rad"):
        Layer3Primitive(**kw)


def test_round_trip_to_from_dict_present() -> None:
    p = Layer3Primitive(**_canonical_kwargs())
    d = p.to_dict()
    recovered = Layer3Primitive.from_dict(d)
    assert recovered == p


def test_round_trip_absent() -> None:
    p = Layer3Primitive.absent()
    recovered = Layer3Primitive.from_dict(p.to_dict())
    assert recovered == p


def test_polarity_add_subtract_both_pass() -> None:
    for pol in ("add", "subtract"):
        kw = _canonical_kwargs()
        kw["polarity"] = pol
        Layer3Primitive(**kw)


def test_unknown_polarity_fails() -> None:
    """polarity must be 'add' or 'subtract' — no other strings."""
    kw = _canonical_kwargs()
    kw["polarity"] = "neutral"
    with pytest.raises(ValueError, match="polarity"):
        Layer3Primitive(**kw)
