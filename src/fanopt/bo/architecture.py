"""Phase 4 categorical architecture enumeration (V1-slim).

Loads ``configs/architecture_enumeration.yaml`` — the discrete design axes the
Stage-2 BO searches (V1-slim searches categoricals directly via the codec; the
R11 architecture bandit is retired). Provides the Cartesian enumeration and the
§9.4.1 growth gate (a new count must satisfy ``count <= max(prev*factor,
min_ceiling)``) so the space can't silently balloon.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from fanopt.geometry.manufacturability import PRINT_ORIENTATIONS
from fanopt.geometry.schema import BLADE_COUNTS

__all__ = [
    "DEFAULT_ENUM_PATH",
    "ArchitectureSpace",
    "load_architecture_space",
    "growth_gate_ceiling",
    "check_growth_gate",
]

DEFAULT_ENUM_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "architecture_enumeration.yaml"
)

_KNOWN_AXIS_VALUES: dict[str, tuple[Any, ...]] = {
    "blade_count": BLADE_COUNTS,
    "print_orientation": PRINT_ORIENTATIONS,
}


@dataclass(frozen=True)
class ArchitectureSpace:
    """The discrete architecture axes + growth-gate parameters."""

    axes: dict[str, tuple[Any, ...]]
    growth_factor: float
    growth_min_ceiling: int

    def enumerate(self) -> list[dict[str, Any]]:
        """Cartesian product of the axes as a list of architecture dicts."""
        keys = list(self.axes)
        return [dict(zip(keys, combo, strict=True)) for combo in product(*self.axes.values())]

    @property
    def count(self) -> int:
        n = 1
        for values in self.axes.values():
            n *= len(values)
        return n


def load_architecture_space(path: str | Path = DEFAULT_ENUM_PATH) -> ArchitectureSpace:
    """Parse + validate the enumeration YAML.

    Validates each axis against its known allowed values (``blade_count`` ⊆
    :data:`BLADE_COUNTS`, ``print_orientation`` ⊆ :data:`PRINT_ORIENTATIONS`) so
    the config can't drift from the locked schema.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    raw_axes = data.get("axes")
    if not raw_axes:
        raise ValueError(f"{path}: no 'axes' block")
    axes: dict[str, tuple[Any, ...]] = {}
    for name, values in raw_axes.items():
        if not values:
            raise ValueError(f"{path}: axis {name!r} is empty")
        allowed = _KNOWN_AXIS_VALUES.get(name)
        if allowed is not None:
            unknown = [v for v in values if v not in allowed]
            if unknown:
                raise ValueError(f"{path}: axis {name!r} has values {unknown} outside {allowed}")
        axes[name] = tuple(values)
    gate = data.get("growth_gate", {})
    return ArchitectureSpace(
        axes=axes,
        growth_factor=float(gate.get("factor", 1.1)),
        growth_min_ceiling=int(gate.get("min_ceiling", 50)),
    )


def growth_gate_ceiling(prev_count: int, *, factor: float = 1.1, min_ceiling: int = 50) -> int:
    """The largest allowed new count given the previous one (§9.4.1)."""
    return max(int(prev_count * factor), min_ceiling)


def check_growth_gate(
    prev_count: int, new_count: int, *, factor: float = 1.1, min_ceiling: int = 50
) -> bool:
    """True iff ``new_count`` is within the §9.4.1 growth ceiling."""
    return new_count <= growth_gate_ceiling(prev_count, factor=factor, min_ceiling=min_ceiling)
