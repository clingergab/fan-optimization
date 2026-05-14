"""Top-level blade-design parameter aggregator.

Combines the four layer dataclasses (Layer 1 envelope, Layer 2 fields,
Layer 3 primitive, Layer 4 manufacturing) into a single
:class:`BladeDesignParams` instance. Validates per-layer bounds AND the
cross-layer constraints the plan calls out:

- **Plano-convex camber under rib-flat** (plan §0 row 47, §7.4.3): when
  ``Layer4.print_orientation == "flat"``, Layer 1's camber must be
  applied to the top face only. Schema-level: we enforce that camber
  values are non-negative under the flat orientation (a sufficient
  condition; the actual top-vs-bottom assignment is the generator's job).

The CadQuery orchestration pipeline (``run_generator``) lands in
Phase 1; this module ships only the parameter schema + validators.
Reference: plan §9.7.1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fanopt.geometry.envelope import Layer1Params
from fanopt.geometry.fields import Layer2Params
from fanopt.geometry.manufacturability import Layer4Params
from fanopt.geometry.primitives import Layer3Primitive

__all__ = ["BladeDesignParams"]


@dataclass(frozen=True)
class BladeDesignParams:
    """One blade's full design parameter set across all 4 layers.

    Each layer validates its own bounds in ``__post_init__``. This
    aggregator additionally enforces cross-layer constraints.
    """

    layer1: Layer1Params
    layer2: Layer2Params
    layer3: Layer3Primitive
    layer4: Layer4Params

    def __post_init__(self) -> None:
        if self.layer4.print_orientation == "flat":
            for i, v in enumerate(self.layer1.camber_knots_m):
                if v < 0:
                    raise ValueError(
                        f"BladeDesignParams: plano-convex constraint violated — "
                        f"camber_knots_m[{i}] = {v} < 0 under "
                        f"print_orientation='flat' (rib-flat). The bottom "
                        f"face is planar; camber values must be non-negative."
                    )

    def to_dict(self) -> dict[str, Any]:
        """Full JSON-friendly serialisation across all four layers."""
        return {
            "layer1": self.layer1.to_dict(),
            "layer2": self.layer2.to_dict(),
            "layer3": self.layer3.to_dict(),
            "layer4": self.layer4.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BladeDesignParams":
        return cls(
            layer1=Layer1Params.from_dict(d["layer1"]),
            layer2=Layer2Params.from_dict(d["layer2"]),
            layer3=Layer3Primitive.from_dict(d["layer3"]),
            layer4=Layer4Params.from_dict(d["layer4"]),
        )
