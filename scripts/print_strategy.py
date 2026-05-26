"""Decide whether a deployed fan fits the print bed in one piece.

CLI wrapper around the deploy-fan bounding-box check. Reads
``params.json``, builds the deployed fan, compares the x-y bounding
box against the configured print bed, and outputs either
``"full-assembly"`` (one print) or ``"per-blade"`` (N separate prints).

Default bed: 256 × 256 mm (Bambu Lab X1 / Prusa MK4). Override via
``--bed-x-mm`` / ``--bed-y-mm``.

Usage::

    python3 scripts/print_strategy.py --params data/design.json
    python3 scripts/print_strategy.py --params data/design.json \\
        --bed-x-mm 220 --bed-y-mm 220

Exit codes:

* ``0`` — success
* ``2`` — params.json read / schema validation error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fanopt.geometry.fan_assembly import deploy_fan
from fanopt.geometry.generator import BladeDesignParams

__all__ = ["main", "decide_strategy"]


DEFAULT_BED_X_MM: float = 256.0
DEFAULT_BED_Y_MM: float = 256.0


def decide_strategy(
    design: BladeDesignParams,
    bed_x_mm: float = DEFAULT_BED_X_MM,
    bed_y_mm: float = DEFAULT_BED_Y_MM,
) -> tuple[str, dict[str, float]]:
    """Return ``(decision, {"fan_x_mm", "fan_y_mm", "bed_x_mm", "bed_y_mm"})``.

    ``decision`` is ``"full-assembly"`` if the deployed fan's x-y
    bounding box fits inside the bed, else ``"per-blade"``.
    """
    assembly, _ = deploy_fan(design)
    bb = assembly.val().BoundingBox()
    fan_x_m = bb.xmax - bb.xmin
    fan_y_m = bb.ymax - bb.ymin
    fan_x_mm = fan_x_m * 1000.0
    fan_y_mm = fan_y_m * 1000.0
    fits = fan_x_mm <= bed_x_mm and fan_y_mm <= bed_y_mm
    return (
        "full-assembly" if fits else "per-blade",
        {
            "fan_x_mm": fan_x_mm,
            "fan_y_mm": fan_y_mm,
            "bed_x_mm": bed_x_mm,
            "bed_y_mm": bed_y_mm,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True, help="params.json input")
    parser.add_argument("--bed-x-mm", type=float, default=DEFAULT_BED_X_MM)
    parser.add_argument("--bed-y-mm", type=float, default=DEFAULT_BED_Y_MM)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)

    try:
        design = BladeDesignParams.from_dict(json.loads(args.params.read_text()))
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"[print_strategy] failed to load params: {e}", file=sys.stderr)
        return 2

    decision, dims = decide_strategy(design, args.bed_x_mm, args.bed_y_mm)
    if args.json:
        print(json.dumps({"decision": decision, **dims}, indent=2))
    else:
        print(
            f"[print_strategy] decision={decision}  "
            f"fan={dims['fan_x_mm']:.0f}×{dims['fan_y_mm']:.0f} mm  "
            f"bed={dims['bed_x_mm']:.0f}×{dims['bed_y_mm']:.0f} mm"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
