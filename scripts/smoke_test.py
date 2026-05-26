"""End-to-end roundtrip smoke test — JSON design -> geometry -> properties.

Reads ``params.json``, builds one V-unit blade, computes mass / centre
of mass / wrist-axis inertia / manufacturability score. Emits a JSON
summary plus the locked ``I_wrist_kgm2`` value the deferred Spike 0.2
V2 cross-check consumes (per ``docs/phase_logs/phase_0_signoff.md``).

This is the Phase-1 smoke test. The downstream Gmsh + SU2 + J_fan
roundtrip is a Phase-2/Phase-4 wiring; this script stops at the
physical-property summary that is sufficient to validate the
geometry pipeline end-to-end.

Usage::

    python3 scripts/smoke_test.py --params data/design.json
    python3 scripts/smoke_test.py --params data/design.json \\
        --out data/smoke_summary.json

Exit codes:

* ``0`` — success, manufacturability passed
* ``1`` — geometry produced but manufacturability score < 0.5
* ``2`` — params.json read / schema validation error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fanopt.geometry.assembly_cad import make_vunit_blade
from fanopt.geometry.fan_assembly import (
    compute_centre_of_mass,
    compute_i_wrist_kgm2,
    compute_mass_kg,
)
from fanopt.geometry.generator import BladeDesignParams
from fanopt.geometry.generator_cad import generate_blade_cad
from fanopt.geometry.schema import MAX_TOTAL_MASS_KG

__all__ = ["main", "compute_smoke_summary"]


def compute_smoke_summary(design: BladeDesignParams) -> dict[str, object]:
    """Build the full smoke-test payload for a validated design."""
    result, _panel = generate_blade_cad(design)
    blade = make_vunit_blade(design)
    mass = compute_mass_kg(blade)
    com = compute_centre_of_mass(blade)
    i_wrist = compute_i_wrist_kgm2(blade)
    n = design.layer1.blade_count
    total_mass = mass * n
    return {
        "status": result.status.value,
        "blade_count": n,
        "blade_mass_kg": mass,
        "total_mass_kg": total_mass,
        "total_mass_under_cap": total_mass <= MAX_TOTAL_MASS_KG,
        "mass_cap_kg": MAX_TOTAL_MASS_KG,
        "centre_of_mass_m": list(com),
        "i_wrist_kgm2": i_wrist,
        "manufacturability_score": result.manufacturability.score,
        "manufacturability_passed": result.manufacturability.passed,
        "critical_failures": list(result.manufacturability.critical_failures),
        "pending_cadquery": list(result.manufacturability.pending_cadquery),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True, help="params.json input")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional JSON output path (default: stdout)",
    )
    args = parser.parse_args(argv)

    try:
        design = BladeDesignParams.from_dict(json.loads(args.params.read_text()))
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"[smoke_test] failed to load params: {e}", file=sys.stderr)
        return 2

    summary = compute_smoke_summary(design)
    text = json.dumps(summary, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"[smoke_test] wrote {args.out}")
    else:
        print(text)

    return 0 if summary["manufacturability_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
