"""Geometry Layer 4 — manufacturing + click features (BO parameters).

Implements the Layer 4 BO design-parameter schema per plan §6.2.1
(`docs/report-final.md` §6.2.1 + §3.1.2b + §7.4.3). Layer 4 covers the
manufacturing categoricals (print orientation, layer height) and the
panel-edge click parameters (chamfer angle, detent size, design
clearance). ~3-5 vars.

Per plan locks:

- ``print_orientation``: ``flat`` (rib-flat, the §7.4.3 default), ``edge``,
  or ``custom-angle``. Triggers the plano-convex camber constraint on
  Layer 1 when ``= flat`` (enforced by :class:`BladeDesignParams`).
- ``layer_height_m``: {0.1, 0.15, 0.2} mm.
- ``click_chamfer_angle_deg``: 30-60°. Schema's
  ``CLICK_CHAMFER_ANGLE_DEG`` locks the nominal at 45°; Layer 4 permits
  the 30-60° BO band.
- ``click_detent_size_m``: 0.3-0.5 mm radius. Uses ``DETENT_RADIUS_RANGE_M``.
- ``click_design_clearance_m``: 0.15-0.20 mm per mating surface.

This module ships the Layer 4 parameter dataclass. The §9.7.3
manufacturability *filter* (downstream check on generated geometry) is
a separate function that lands in Phase 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fanopt.geometry.schema import DETENT_RADIUS_RANGE_M

__all__ = [
    "PRINT_ORIENTATIONS",
    "LAYER_HEIGHTS_M",
    "CLICK_CHAMFER_ANGLE_RANGE_DEG",
    "CLICK_DESIGN_CLEARANCE_RANGE_M",
    "Layer4Params",
]


PRINT_ORIENTATIONS: tuple[str, ...] = ("flat", "edge", "custom-angle")
"""Plan §6.2.1 + §7.4.3. ``flat`` (= rib-flat) is the default and triggers
plano-convex camber on Layer 1."""

LAYER_HEIGHTS_M: tuple[float, ...] = (0.0001, 0.00015, 0.0002)
"""{0.1, 0.15, 0.2} mm slicer setting — discrete categorical."""

CLICK_CHAMFER_ANGLE_RANGE_DEG: tuple[float, float] = (30.0, 60.0)
"""Plan §6.2.1: '30-60°'."""

CLICK_DESIGN_CLEARANCE_RANGE_M: tuple[float, float] = (0.00015, 0.00020)
"""0.15-0.20 mm per mating surface."""


@dataclass(frozen=True)
class Layer4Params:
    """Layer 4 design parameters — manufacturing + click features."""

    print_orientation: str
    layer_height_m: float
    click_chamfer_angle_deg: float
    click_detent_size_m: float
    click_design_clearance_m: float

    def __post_init__(self) -> None:
        if self.print_orientation not in PRINT_ORIENTATIONS:
            raise ValueError(
                f"print_orientation must be one of {PRINT_ORIENTATIONS}, "
                f"got {self.print_orientation!r}"
            )
        if not any(
            abs(self.layer_height_m - lh) < 1e-9 for lh in LAYER_HEIGHTS_M
        ):
            raise ValueError(
                f"layer_height_m = {self.layer_height_m} must match one of "
                f"{LAYER_HEIGHTS_M} (mm: 0.1 / 0.15 / 0.2)"
            )
        ang_lo, ang_hi = CLICK_CHAMFER_ANGLE_RANGE_DEG
        if not (ang_lo <= self.click_chamfer_angle_deg <= ang_hi):
            raise ValueError(
                f"click_chamfer_angle_deg = {self.click_chamfer_angle_deg} "
                f"outside range [{ang_lo}, {ang_hi}]"
            )
        det_lo, det_hi = DETENT_RADIUS_RANGE_M
        if not (det_lo <= self.click_detent_size_m <= det_hi):
            raise ValueError(
                f"click_detent_size_m = {self.click_detent_size_m} outside "
                f"locked range {DETENT_RADIUS_RANGE_M}"
            )
        cl_lo, cl_hi = CLICK_DESIGN_CLEARANCE_RANGE_M
        if not (cl_lo <= self.click_design_clearance_m <= cl_hi):
            raise ValueError(
                f"click_design_clearance_m = {self.click_design_clearance_m} "
                f"outside range [{cl_lo}, {cl_hi}]"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "print_orientation": self.print_orientation,
            "layer_height_m": self.layer_height_m,
            "click_chamfer_angle_deg": self.click_chamfer_angle_deg,
            "click_detent_size_m": self.click_detent_size_m,
            "click_design_clearance_m": self.click_design_clearance_m,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Layer4Params":
        return cls(
            print_orientation=str(d["print_orientation"]),
            layer_height_m=float(d["layer_height_m"]),
            click_chamfer_angle_deg=float(d["click_chamfer_angle_deg"]),
            click_detent_size_m=float(d["click_detent_size_m"]),
            click_design_clearance_m=float(d["click_design_clearance_m"]),
        )
