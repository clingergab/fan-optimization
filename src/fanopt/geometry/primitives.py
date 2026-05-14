"""Geometry Layer 3 — capped 0-1 independent primitive.

Implements the Layer 3 BO design-parameter schema per plan §6.2.1
(`docs/report-final.md` §6.2.1 + §3.2). Layer 3 carries at most one
primitive feature per design (capped to preserve dimensionality budget;
primitives are NOT the primary topology mechanism — Layer 2 fields are).

Per plan §9.7: Layer 3 is the **only** generator step that may fail at
CadQuery time (Boolean of an arbitrarily-placed primitive against the
Layer-2-carved envelope can produce degenerate geometry). The generator
wraps it in try/except; on failure the prior-step geometry is returned.
This module ships only the schema; the apply-primitive function lands
in Phase 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "PRIMITIVE_TYPES",
    "POLARITY_OPTIONS",
    "PRIMITIVE_MARGIN_FROM_EDGE_M",
    "PRIMITIVE_MIN_DIMENSION_M",
    "PRIMITIVE_MAX_FRACTION_OF_ENVELOPE",
    "PRIMITIVE_ROTATION_RANGE_RAD",
    "Layer3Primitive",
]


PRIMITIVE_TYPES: tuple[str, ...] = ("slot", "ellipsoid", "wedge")
"""Plan §6.2.1: restricted to these three for Boolean reliability."""

POLARITY_OPTIONS: tuple[str, ...] = ("add", "subtract")

PRIMITIVE_MARGIN_FROM_EDGE_M: float = 0.001
"""Plan §6.2.1: 'constrained ≥1 mm from envelope edges'."""

PRIMITIVE_MIN_DIMENSION_M: float = 0.0008
"""Plan §6.2.1: 'each ≥0.8 mm'."""

PRIMITIVE_MAX_FRACTION_OF_ENVELOPE: float = 0.30
"""Plan §6.2.1: 'each ≤30% local envelope'. Applied component-wise."""

PRIMITIVE_ROTATION_RANGE_RAD: tuple[float, float] = (-math.pi, math.pi)


@dataclass(frozen=True)
class Layer3Primitive:
    """Layer 3 design parameters — capped 0-1 independent primitive.

    When ``present = False`` the other fields are ignored. The generator
    skips Layer 3 entirely in that case.

    Position is expressed in panel-local coordinates (m). Sizes are the
    primitive's principal-axis half-extents (m). Rotations are extrinsic
    Euler angles (rad).

    The ``local_envelope_xyz_m`` field carries the panel-local envelope
    bounding-box used for the ≤30% fraction check. It is required when
    the primitive is present (no envelope = no way to enforce the bound).
    """

    present: bool
    shape_type: str = "slot"
    polarity: str = "subtract"
    position_x_m: float = 0.0
    position_y_m: float = 0.0
    position_z_m: float = 0.0
    size_x_m: float = PRIMITIVE_MIN_DIMENSION_M
    size_y_m: float = PRIMITIVE_MIN_DIMENSION_M
    size_z_m: float = PRIMITIVE_MIN_DIMENSION_M
    rotation_x_rad: float = 0.0
    rotation_y_rad: float = 0.0
    rotation_z_rad: float = 0.0
    local_envelope_xyz_m: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not self.present:
            return
        if self.shape_type not in PRIMITIVE_TYPES:
            raise ValueError(
                f"Layer3Primitive.shape_type must be one of {PRIMITIVE_TYPES}, "
                f"got {self.shape_type!r}"
            )
        if self.polarity not in POLARITY_OPTIONS:
            raise ValueError(
                f"Layer3Primitive.polarity must be one of {POLARITY_OPTIONS}, "
                f"got {self.polarity!r}"
            )
        if self.local_envelope_xyz_m is None:
            raise ValueError(
                "Layer3Primitive.local_envelope_xyz_m must be set when "
                "present=True (required for the ≤30% envelope fraction check)"
            )
        env_x, env_y, env_z = self.local_envelope_xyz_m
        if env_x <= 0 or env_y <= 0 or env_z <= 0:
            raise ValueError(
                f"Layer3Primitive.local_envelope_xyz_m components must be > 0, "
                f"got {self.local_envelope_xyz_m}"
            )
        margin = PRIMITIVE_MARGIN_FROM_EDGE_M
        for axis, pos, env in (
            ("x", self.position_x_m, env_x),
            ("y", self.position_y_m, env_y),
            ("z", self.position_z_m, env_z),
        ):
            if not (margin <= pos <= env - margin):
                raise ValueError(
                    f"Layer3Primitive.position_{axis}_m = {pos} violates "
                    f"{margin}-m margin from envelope [0, {env}]"
                )
        for axis, size, env in (
            ("x", self.size_x_m, env_x),
            ("y", self.size_y_m, env_y),
            ("z", self.size_z_m, env_z),
        ):
            if size < PRIMITIVE_MIN_DIMENSION_M:
                raise ValueError(
                    f"Layer3Primitive.size_{axis}_m = {size} below minimum "
                    f"{PRIMITIVE_MIN_DIMENSION_M} m"
                )
            if size > PRIMITIVE_MAX_FRACTION_OF_ENVELOPE * env:
                raise ValueError(
                    f"Layer3Primitive.size_{axis}_m = {size} exceeds "
                    f"{PRIMITIVE_MAX_FRACTION_OF_ENVELOPE * 100:.0f}% of local "
                    f"envelope {env} m"
                )
        lo, hi = PRIMITIVE_ROTATION_RANGE_RAD
        for axis, rot in (
            ("x", self.rotation_x_rad),
            ("y", self.rotation_y_rad),
            ("z", self.rotation_z_rad),
        ):
            if not (lo <= rot <= hi):
                raise ValueError(
                    f"Layer3Primitive.rotation_{axis}_rad = {rot} outside " f"range [{lo}, {hi}]"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "shape_type": self.shape_type,
            "polarity": self.polarity,
            "position_x_m": self.position_x_m,
            "position_y_m": self.position_y_m,
            "position_z_m": self.position_z_m,
            "size_x_m": self.size_x_m,
            "size_y_m": self.size_y_m,
            "size_z_m": self.size_z_m,
            "rotation_x_rad": self.rotation_x_rad,
            "rotation_y_rad": self.rotation_y_rad,
            "rotation_z_rad": self.rotation_z_rad,
            "local_envelope_xyz_m": (
                list(self.local_envelope_xyz_m) if self.local_envelope_xyz_m is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Layer3Primitive:
        env = d.get("local_envelope_xyz_m")
        return cls(
            present=bool(d["present"]),
            shape_type=str(d.get("shape_type", "slot")),
            polarity=str(d.get("polarity", "subtract")),
            position_x_m=float(d.get("position_x_m", 0.0)),
            position_y_m=float(d.get("position_y_m", 0.0)),
            position_z_m=float(d.get("position_z_m", 0.0)),
            size_x_m=float(d.get("size_x_m", PRIMITIVE_MIN_DIMENSION_M)),
            size_y_m=float(d.get("size_y_m", PRIMITIVE_MIN_DIMENSION_M)),
            size_z_m=float(d.get("size_z_m", PRIMITIVE_MIN_DIMENSION_M)),
            rotation_x_rad=float(d.get("rotation_x_rad", 0.0)),
            rotation_y_rad=float(d.get("rotation_y_rad", 0.0)),
            rotation_z_rad=float(d.get("rotation_z_rad", 0.0)),
            local_envelope_xyz_m=(
                tuple(float(v) for v in env) if env is not None else None  # type: ignore[arg-type]
            ),
        )

    @classmethod
    def absent(cls) -> Layer3Primitive:
        """Canonical 'no primitive' instance (Layer 3 inactive)."""
        return cls(present=False)
