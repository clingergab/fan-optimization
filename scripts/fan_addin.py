"""Headless fan export — params.json -> per-blade STLs + deployed-fan STEP.

CLI wrapper around ``fanopt.geometry.assembly_cad.make_vunit_blade`` +
``fanopt.geometry.fan_assembly.deploy_fan``. The original plan envisioned
this as a Fusion 360 add-in (hence the script name); the CadQuery-based
implementation produces the same artifacts without a Fusion dependency
and is the V1 path.

Usage::

    python3 scripts/fan_addin.py \\
        --params data/design.json \\
        --out-dir data/exports/

Outputs into ``out-dir``:

* ``blade_0.stl`` ... ``blade_{N-1}.stl`` — per-blade meshes for the slicer.
* ``deployed_fan.step`` — full-assembly STEP for downstream CAD review.

Exit codes:

* ``0`` — success
* ``2`` — params.json read / schema validation error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cadquery as cq

from fanopt.geometry.fan_assembly import deploy_fan
from fanopt.geometry.generator import BladeDesignParams

__all__ = ["main"]


def _load_design(params_path: Path) -> BladeDesignParams:
    raw = json.loads(params_path.read_text())
    return BladeDesignParams.from_dict(raw)


def _export(
    design: BladeDesignParams,
    out_dir: Path,
) -> tuple[list[Path], Path]:
    """Write per-blade STLs + deployed-fan STEP into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    assembly, per_blade = deploy_fan(design)
    stl_paths: list[Path] = []
    for i, blade in enumerate(per_blade):
        path = out_dir / f"blade_{i}.stl"
        cq.exporters.export(blade, str(path))
        stl_paths.append(path)
    step_path = out_dir / "deployed_fan.step"
    cq.exporters.export(assembly, str(step_path))
    return stl_paths, step_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True, help="params.json input")
    parser.add_argument("--out-dir", type=Path, required=True, help="export directory")
    args = parser.parse_args(argv)

    try:
        design = _load_design(args.params)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"[fan_addin] failed to load params: {e}", file=sys.stderr)
        return 2

    stl_paths, step_path = _export(design, args.out_dir)
    print(f"[fan_addin] wrote {len(stl_paths)} STLs + {step_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
