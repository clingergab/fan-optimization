"""Geometry Layer 2 — macro-pattern + procedural-math fields.

Implements the Layer 2 BO design-parameter schema per plan §6.2.1
(`docs/report-final.md` §6.2.1 + §3.2). Layer 2 covers the 5-field
library {louver, texture, edge, noise, TPMS}; 0-3 active per design.

Per the plan: *all Layer 2 fields are safe-by-construction* — parameter
ranges mathematically guarantee features stay within the envelope
(≥1 mm margin, ≥0.8 mm minimum feature) and produce coherent CadQuery
geometry. The schema's bounds reflect those guarantees; the actual
CadQuery field functions land in Phase 1.

Architecture-bandit constraint (plan §6.2.2 + H1 lock): at most 3 fields
active at once. The H1 lock further restricts {noise, TPMS} to not
co-active — enforced at the architecture-enumeration level, not at this
dataclass level. This module enforces only the cardinality bound.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "MAX_ACTIVE_FIELDS",
    "MIN_FEATURE_SIZE_M",
    "LOUVER_COUNT_RANGE",
    "LOUVER_ANGLE_RANGE_RAD",
    "LOUVER_WIDTH_RANGE_M",
    "LOUVER_SPACING_PROFILES",
    "POLARITY_OPTIONS",
    "TEXTURE_TYPES",
    "TEXTURE_DENSITY_RANGE_PER_CM2",
    "TEXTURE_SIZE_RANGE_M",
    "TEXTURE_ORIENTATION_RANGE_RAD",
    "EDGE_FEATURE_TYPES",
    "EDGE_FEATURE_DEPTH_RANGE_M",
    "EDGE_APPLICATION_OPTIONS",
    "EDGE_FEATURE_COUNT_RANGE",
    "NOISE_SCALE_RANGE",
    "NOISE_ROTATION_RANGE_RAD",
    "NOISE_THRESHOLD_RETENTION_MIN",
    "TPMS_LATTICE_TYPES",
    "TPMS_CELL_SIZE_MIN_M",
    "TPMS_THICKNESS_GRADIENT_RANGE",
    "LouverField",
    "TextureField",
    "EdgeFeatureField",
    "NoiseField",
    "TpmsField",
    "Layer2Params",
]


MAX_ACTIVE_FIELDS: int = 3
"""Plan §6.2.1: 'The optimizer activates 0-3 fields per design'."""

MIN_FEATURE_SIZE_M: float = 0.0008
"""0.8 mm minimum feature size — safe-by-construction floor."""

POLARITY_OPTIONS: tuple[str, ...] = ("add", "subtract")

LOUVER_COUNT_RANGE: tuple[int, int] = (3, 12)
LOUVER_ANGLE_RANGE_RAD: tuple[float, float] = (
    -math.radians(60.0),
    math.radians(60.0),
)
LOUVER_WIDTH_RANGE_M: tuple[float, float] = (0.0005, 0.003)
LOUVER_SPACING_PROFILES: tuple[str, ...] = (
    "uniform",
    "clustered-at-tip",
    "gradient-toward-LE",
)

TEXTURE_TYPES: tuple[str, ...] = ("dimple", "ridge", "bump")
TEXTURE_DENSITY_RANGE_PER_CM2: tuple[float, float] = (0.5, 25.0)
TEXTURE_SIZE_RANGE_M: tuple[float, float] = (0.0005, 0.003)
TEXTURE_ORIENTATION_RANGE_RAD: tuple[float, float] = (
    -math.radians(90.0),
    math.radians(90.0),
)

EDGE_FEATURE_TYPES: tuple[str, ...] = ("serration", "scallop", "smooth-fade")
EDGE_FEATURE_COUNT_RANGE: tuple[int, int] = (3, 24)
EDGE_FEATURE_DEPTH_RANGE_M: tuple[float, float] = (0.0005, 0.003)
EDGE_APPLICATION_OPTIONS: tuple[str, ...] = ("LE", "TE", "both")

NOISE_SCALE_RANGE: tuple[float, float] = (1.0, 50.0)
NOISE_ROTATION_RANGE_RAD: tuple[float, float] = (
    -math.radians(90.0),
    math.radians(90.0),
)
NOISE_THRESHOLD_RETENTION_MIN: float = 0.40
"""Plan: 'threshold value (constrained to retain ≥40% material)'."""

TPMS_LATTICE_TYPES: tuple[str, ...] = ("gyroid", "schwarz-diamond", "off")
TPMS_CELL_SIZE_MIN_M: float = 3.0 * MIN_FEATURE_SIZE_M
"""Plan §6.2.1: 'cell size (≥3× min feature size)' → ≥ 2.4 mm."""
TPMS_THICKNESS_GRADIENT_RANGE: tuple[float, float] = (-0.5, 0.5)


def _check_range(name: str, value: float, lo: float, hi: float) -> None:
    if not (lo <= value <= hi):
        raise ValueError(f"{name} = {value} outside range [{lo}, {hi}]")


def _check_choice(name: str, value: str, options: tuple[str, ...]) -> None:
    if value not in options:
        raise ValueError(f"{name} must be one of {options}, got {value!r}")


@dataclass(frozen=True)
class LouverField:
    """Louver cuts/ribs — directional design family for asymmetric drag."""

    active: bool
    count: int = 6
    angle_rad: float = 0.0
    width_m: float = 0.001
    spacing_profile: str = "uniform"
    polarity: str = "subtract"

    def __post_init__(self) -> None:
        if not self.active:
            return
        c_lo, c_hi = LOUVER_COUNT_RANGE
        if not (c_lo <= self.count <= c_hi):
            raise ValueError(
                f"LouverField.count = {self.count} outside range [{c_lo}, {c_hi}]"
            )
        _check_range("LouverField.angle_rad", self.angle_rad, *LOUVER_ANGLE_RANGE_RAD)
        _check_range("LouverField.width_m", self.width_m, *LOUVER_WIDTH_RANGE_M)
        _check_choice(
            "LouverField.spacing_profile",
            self.spacing_profile,
            LOUVER_SPACING_PROFILES,
        )
        _check_choice("LouverField.polarity", self.polarity, POLARITY_OPTIONS)


@dataclass(frozen=True)
class TextureField:
    """Distributed boundary-layer features (dimples / ridges / bumps)."""

    active: bool
    feature_type: str = "dimple"
    density_per_cm2: float = 5.0
    size_m: float = 0.001
    orientation_rad: float = 0.0
    polarity: str = "subtract"

    def __post_init__(self) -> None:
        if not self.active:
            return
        _check_choice("TextureField.feature_type", self.feature_type, TEXTURE_TYPES)
        _check_range(
            "TextureField.density_per_cm2",
            self.density_per_cm2,
            *TEXTURE_DENSITY_RANGE_PER_CM2,
        )
        _check_range("TextureField.size_m", self.size_m, *TEXTURE_SIZE_RANGE_M)
        _check_range(
            "TextureField.orientation_rad",
            self.orientation_rad,
            *TEXTURE_ORIENTATION_RANGE_RAD,
        )
        _check_choice("TextureField.polarity", self.polarity, POLARITY_OPTIONS)


@dataclass(frozen=True)
class EdgeFeatureField:
    """Leading / trailing-edge shapers (serrations, scallops, smooth fades)."""

    active: bool
    feature_type: str = "serration"
    count: int = 8
    depth_m: float = 0.001
    application: str = "TE"

    def __post_init__(self) -> None:
        if not self.active:
            return
        _check_choice(
            "EdgeFeatureField.feature_type", self.feature_type, EDGE_FEATURE_TYPES
        )
        c_lo, c_hi = EDGE_FEATURE_COUNT_RANGE
        if not (c_lo <= self.count <= c_hi):
            raise ValueError(
                f"EdgeFeatureField.count = {self.count} outside range "
                f"[{c_lo}, {c_hi}]"
            )
        _check_range(
            "EdgeFeatureField.depth_m", self.depth_m, *EDGE_FEATURE_DEPTH_RANGE_M
        )
        _check_choice(
            "EdgeFeatureField.application",
            self.application,
            EDGE_APPLICATION_OPTIONS,
        )


@dataclass(frozen=True)
class NoiseField:
    """Perlin/Simplex noise-threshold field for organic topology."""

    active: bool
    x_scale: float = 10.0
    y_scale: float = 10.0
    rotation_rad: float = 0.0
    x_offset: float = 0.0
    threshold_retention: float = 0.6

    def __post_init__(self) -> None:
        if not self.active:
            return
        _check_range("NoiseField.x_scale", self.x_scale, *NOISE_SCALE_RANGE)
        _check_range("NoiseField.y_scale", self.y_scale, *NOISE_SCALE_RANGE)
        _check_range(
            "NoiseField.rotation_rad", self.rotation_rad, *NOISE_ROTATION_RANGE_RAD
        )
        if self.threshold_retention < NOISE_THRESHOLD_RETENTION_MIN:
            raise ValueError(
                f"NoiseField.threshold_retention = {self.threshold_retention} "
                f"below floor {NOISE_THRESHOLD_RETENTION_MIN} (plan: "
                f"'≥40% material retained')"
            )
        if self.threshold_retention > 1.0:
            raise ValueError(
                f"NoiseField.threshold_retention = {self.threshold_retention} "
                f"above 1.0 (must be a fraction)"
            )


@dataclass(frozen=True)
class TpmsField:
    """Triply Periodic Minimal Surface — directional through-blade porosity."""

    active: bool
    lattice_type: str = "gyroid"
    cell_size_m: float = TPMS_CELL_SIZE_MIN_M
    rotation_x_rad: float = 0.0
    rotation_y_rad: float = 0.0
    rotation_z_rad: float = 0.0
    thickness_gradient: float = 0.0

    def __post_init__(self) -> None:
        if not self.active:
            return
        _check_choice(
            "TpmsField.lattice_type", self.lattice_type, TPMS_LATTICE_TYPES
        )
        if self.cell_size_m < TPMS_CELL_SIZE_MIN_M:
            raise ValueError(
                f"TpmsField.cell_size_m = {self.cell_size_m} below floor "
                f"{TPMS_CELL_SIZE_MIN_M} (plan: ≥3× min feature size)"
            )
        _check_range(
            "TpmsField.thickness_gradient",
            self.thickness_gradient,
            *TPMS_THICKNESS_GRADIENT_RANGE,
        )


@dataclass(frozen=True)
class Layer2Params:
    """Layer 2 design parameters — the 5-field library aggregator.

    Holds one instance of each of the 5 field types. The ≤3-active
    cardinality bound is enforced at construction. Other cross-field
    constraints (e.g., the H1 lock forbidding noise+TPMS co-activation
    at the architecture-enumeration level) are NOT enforced here.
    """

    louver: LouverField
    texture: TextureField
    edge: EdgeFeatureField
    noise: NoiseField
    tpms: TpmsField

    def __post_init__(self) -> None:
        active_count = self.active_count()
        if active_count > MAX_ACTIVE_FIELDS:
            raise ValueError(
                f"Layer 2 has {active_count} active fields, exceeds "
                f"MAX_ACTIVE_FIELDS = {MAX_ACTIVE_FIELDS} (plan §6.2.1)"
            )

    def active_count(self) -> int:
        return sum(
            1 for f in (self.louver, self.texture, self.edge, self.noise, self.tpms)
            if f.active
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "louver": asdict(self.louver),
            "texture": asdict(self.texture),
            "edge": asdict(self.edge),
            "noise": asdict(self.noise),
            "tpms": asdict(self.tpms),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Layer2Params":
        return cls(
            louver=LouverField(**d["louver"]),
            texture=TextureField(**d["texture"]),
            edge=EdgeFeatureField(**d["edge"]),
            noise=NoiseField(**d["noise"]),
            tpms=TpmsField(**d["tpms"]),
        )

    @classmethod
    def all_inactive(cls) -> "Layer2Params":
        """Canonical 'no Layer 2 features' design (zero active fields)."""
        return cls(
            louver=LouverField(active=False),
            texture=TextureField(active=False),
            edge=EdgeFeatureField(active=False),
            noise=NoiseField(active=False),
            tpms=TpmsField(active=False),
        )
