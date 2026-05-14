"""Unit tests for fanopt.geometry.manufacturability (Layer 4 BO params)."""
from __future__ import annotations

import pytest

from fanopt.geometry.manufacturability import (
    CLICK_CHAMFER_ANGLE_RANGE_DEG,
    CLICK_DESIGN_CLEARANCE_RANGE_M,
    LAYER_HEIGHTS_M,
    PRINT_ORIENTATIONS,
    Layer4Params,
)
from fanopt.geometry.schema import DETENT_RADIUS_RANGE_M


def _canonical_kwargs() -> dict:
    return dict(
        print_orientation="flat",
        layer_height_m=0.0002,
        click_chamfer_angle_deg=45.0,
        click_detent_size_m=0.0004,
        click_design_clearance_m=0.00018,
    )


def test_canonical_validates() -> None:
    p = Layer4Params(**_canonical_kwargs())
    assert p.print_orientation == "flat"


def test_unknown_print_orientation_fails() -> None:
    kw = _canonical_kwargs()
    kw["print_orientation"] = "diagonal"
    with pytest.raises(ValueError, match="print_orientation"):
        Layer4Params(**kw)


def test_all_locked_print_orientations_pass() -> None:
    for po in PRINT_ORIENTATIONS:
        kw = _canonical_kwargs()
        kw["print_orientation"] = po
        Layer4Params(**kw)


def test_layer_height_not_in_locked_set_fails() -> None:
    """Layer height is a discrete categorical."""
    kw = _canonical_kwargs()
    kw["layer_height_m"] = 0.00025  # 0.25 mm — not in the locked set
    with pytest.raises(ValueError, match="layer_height_m"):
        Layer4Params(**kw)


def test_all_locked_layer_heights_pass() -> None:
    for lh in LAYER_HEIGHTS_M:
        kw = _canonical_kwargs()
        kw["layer_height_m"] = lh
        Layer4Params(**kw)


def test_chamfer_angle_below_30deg_fails() -> None:
    kw = _canonical_kwargs()
    kw["click_chamfer_angle_deg"] = 25.0
    with pytest.raises(ValueError, match="click_chamfer_angle_deg"):
        Layer4Params(**kw)


def test_chamfer_angle_above_60deg_fails() -> None:
    kw = _canonical_kwargs()
    kw["click_chamfer_angle_deg"] = 65.0
    with pytest.raises(ValueError, match="click_chamfer_angle_deg"):
        Layer4Params(**kw)


def test_chamfer_angle_at_bounds_inclusive() -> None:
    lo, hi = CLICK_CHAMFER_ANGLE_RANGE_DEG
    for ang in (lo, hi):
        kw = _canonical_kwargs()
        kw["click_chamfer_angle_deg"] = ang
        Layer4Params(**kw)


def test_detent_size_below_locked_range_fails() -> None:
    kw = _canonical_kwargs()
    kw["click_detent_size_m"] = 0.0001  # 0.1 mm < 0.3 mm floor
    with pytest.raises(ValueError, match="click_detent_size_m"):
        Layer4Params(**kw)


def test_detent_size_at_locked_bounds_pass() -> None:
    lo, hi = DETENT_RADIUS_RANGE_M
    for d in (lo, hi):
        kw = _canonical_kwargs()
        kw["click_detent_size_m"] = d
        Layer4Params(**kw)


def test_design_clearance_below_range_fails() -> None:
    kw = _canonical_kwargs()
    kw["click_design_clearance_m"] = 0.00010
    with pytest.raises(ValueError, match="click_design_clearance_m"):
        Layer4Params(**kw)


def test_design_clearance_at_bounds_pass() -> None:
    lo, hi = CLICK_DESIGN_CLEARANCE_RANGE_M
    for c in (lo, hi):
        kw = _canonical_kwargs()
        kw["click_design_clearance_m"] = c
        Layer4Params(**kw)


def test_round_trip_to_from_dict() -> None:
    p = Layer4Params(**_canonical_kwargs())
    d = p.to_dict()
    recovered = Layer4Params.from_dict(d)
    assert recovered == p
