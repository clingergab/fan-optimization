#!/usr/bin/env python3
"""Put N copies of a single-blade Bambu project on one plate, with brim only at a few ears per blade.

The blades lie in parallel rows (each standing on its edge, as in the source project), centred on the
bed region both H2C nozzles reach; each gets ``--ears`` small round brims along its bed edge (both ends
plus evenly between) instead of a full brim. The output is a new project; it never overwrites.

    python3 scripts/make_print_plate.py \\
        --project print_ready/P2C_blade_00-FLAT-CONTROL_2026-11-25_..._PLA-15pct.3mf \\
        --copies 12 --out print_ready/H2C_plate12_00-FLAT-CONTROL_2026-11-26_brim-ears.3mf
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from fanopt.geometry.bambu_3mf import plate_centre_mm
from fanopt.geometry.print_plate import (
    footprint_ear_points_mm,
    object_vertices_mm,
    row_positions_mm,
    write_plate,
)


def run(
    *, project: Path, out: Path, copies: int, pitch_mm: float, ears: int, ear_radius_mm: float,
    inset_mm: float,
) -> dict:  # fmt: skip
    """Write the plate and return a short report."""
    with zipfile.ZipFile(project) as z:
        cfg = json.loads(z.read("Metadata/project_settings.config"))
    points = footprint_ear_points_mm(object_vertices_mm(project), ears, inset_mm=inset_mm)
    positions = row_positions_mm(copies, pitch_mm, plate_centre_mm(cfg))
    write_plate(project, out, positions, ear_points_mm=points, ear_radius_mm=ear_radius_mm)
    return {
        "out": str(out),
        "copies": copies,
        "row_pitch_mm": pitch_mm,
        "ear_radius_mm": ear_radius_mm,
        "ear_points_object_mm": points.round(2).tolist(),
        "row_y_mm": [round(y, 2) for _, y in positions],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", type=Path, required=True, help="single-blade .3mf project")
    ap.add_argument("--out", type=Path, required=True, help="new plate project (.3mf)")
    ap.add_argument("--copies", type=int, default=12)
    ap.add_argument("--pitch-mm", type=float, default=16.0, help="distance between rows")
    ap.add_argument("--ears", type=int, default=3, help="brim ears per blade (ends + between)")
    ap.add_argument("--ear-radius-mm", type=float, default=5.0)
    ap.add_argument(
        "--inset-mm", type=float, default=2.0, help="end ears this far in from the ends"
    )
    a = ap.parse_args(argv)
    report = run(
        project=a.project, out=a.out, copies=a.copies, pitch_mm=a.pitch_mm, ears=a.ears,
        ear_radius_mm=a.ear_radius_mm, inset_mm=a.inset_mm,
    )  # fmt: skip
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
