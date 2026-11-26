#!/usr/bin/env python3
"""Re-rank the V1 campaign by wind per inertial effort, every design on one print recipe (V2 plan W1).

Each valid design is thinned about its mid-surface to the recipe's mean wall (by default the printed
blade 00: its design wall × 0.8), priced with the calibrated mass model (``calibrate_print_mass.py``)
and given the fan's inertia about the wrist and the inertial work of the locked ±40°, 2 Hz stroke. Writes
one CSV row per design and a JSON summary comparing the ranking by ``j_fan / W`` with ``j_fan`` alone,
and with the solid ledger mass the campaign stored.

    python3 scripts/rerank_v1_effort.py --shards print_ready/drive_recovery/campaign_trapezoid \\
        --calibration data/v2_w1/print_mass_calibration.json \\
        --out-csv data/v2_w1/v1_inertial_effort.csv --out-json data/v2_w1/v1_inertial_effort_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from fanopt.bo.blade_codec import decode
from fanopt.cfd.blade_verify import load_campaign_rows
from fanopt.geometry.schema import (
    D_HANDLE_M,
    INTER_BLADE_ANGLE_RAD,
    MAX_R_COM_WRIST_M,
    MAX_TOTAL_MASS_KG,
    OMEGA_SHM_RAD_PER_S,
    THETA_MAX_RAD,
)
from fanopt.physical.print_effort import (
    PrintMassModel,
    blade_sheet,
    design_effort,
    mean_thickness_m,
    rerank_summary,
)

PRINTED_00_HASH = "ffe19f9e3e7fa81ba847bbe8"
PRINTED_THICKNESS_SCALE = 0.8  # the operator's 20 % thinning (print_guide §4c)
BLADE_COUNT = 12


def _stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
        "cv": float(values.std() / values.mean()),
    }


def run(
    *,
    shards: Path,
    calibration: Path,
    out_csv: Path,
    out_json: Path,
    recipe_design: str = PRINTED_00_HASH,
    recipe_scale: float = PRINTED_THICKNESS_SCALE,
    pitch_rad: float = INTER_BLADE_ANGLE_RAD,
) -> dict:
    """Effort for every valid design, written as CSV + JSON summary; returns the summary."""
    m = json.loads(calibration.read_text())["model"]
    model = PrintMassModel(
        wall_m=m["wall_mm"] * 1e-3,
        core_fraction=m["core_fraction"],
        base_kg=m["base_g"] * 1e-3,
        rho_kg_m3=m["rho_kg_m3"],
    )
    rows = [
        r
        for r in load_campaign_rows(shards)
        if r.get("j_fan") is not None and math.isfinite(float(r["j_fan"]))
    ]
    rows = list({str(r["design_hash"]): r for r in rows}.values())
    params = {str(r["design_hash"]): decode(np.asarray(r["vector"], dtype=float)) for r in rows}
    recipe_m = mean_thickness_m(blade_sheet(params[recipe_design])) * recipe_scale
    kin = {
        "blade_count": BLADE_COUNT, "pitch_rad": pitch_rad, "wrist_offset_m": D_HANDLE_M,
        "theta_max_rad": THETA_MAX_RAD, "omega_rad_s": OMEGA_SHM_RAD_PER_S,
    }  # fmt: skip
    table = []
    for r in rows:
        e = design_effort(params[str(r["design_hash"])], model, recipe_thickness_m=recipe_m, **kin)
        table.append(
            {
                "design_hash": r["design_hash"],
                "source": r.get("source"),
                "j_fan": float(r["j_fan"]),
                "ledger_solid_mass_kg": float(r["mass_kg"]),
                **e,
                "j_fan_per_push_work": float(r["j_fan"]) / e["work_push_j"],
            }
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    j = np.array([t["j_fan"] for t in table])
    work = np.array([t["work_push_j"] for t in table])
    fan_mass = np.array([t["fan_mass_kg"] for t in table])
    com = np.array([t["com_wrist_m"] for t in table])
    hashes = [t["design_hash"] for t in table]
    by_work = rerank_summary(j, work)
    by_ledger = rerank_summary(j, np.array([t["ledger_solid_mass_kg"] for t in table]))
    for s in (by_work, by_ledger):
        s["best_by_score"] = hashes[int(s["best_by_score"])]
        s["best_by_ratio"] = hashes[int(s["best_by_ratio"])]
    summary = {
        "recipe": {
            "mean_wall_mm": recipe_m * 1e3,
            "from": f"design {recipe_design} mean wall × {recipe_scale}",
            "mass_model": m,
            "inside": "slicer infill, no TO hollow",
        },
        "kinematics": {
            "theta_max_deg": math.degrees(THETA_MAX_RAD),
            "frequency_hz": OMEGA_SHM_RAD_PER_S / (2 * math.pi),
            "wrist_to_pin_m": D_HANDLE_M,
            "open_pitch_deg": math.degrees(pitch_rad),
            "blade_count": BLADE_COUNT,
        },
        "fan_mass_kg": _stats(fan_mass),
        "inertia_wrist_kgm2": _stats(np.array([t["inertia_wrist_kgm2"] for t in table])),
        "work_push_j_per_cycle": _stats(work),
        "com_wrist_m": _stats(com),
        "over_mass_cap": int((fan_mass > MAX_TOTAL_MASS_KG).sum()),
        "over_com_cap": int((com > MAX_R_COM_WRIST_M).sum()),
        "rerank_vs_j_fan": {"by_inertial_work": by_work, "by_ledger_solid_mass": by_ledger},
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shards", type=Path, required=True, help="dir with evaluations_*.jsonl")
    ap.add_argument(
        "--calibration", type=Path, required=True, help="calibrate_print_mass.py report"
    )
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument(
        "--recipe-design", default=PRINTED_00_HASH, help="design whose wall sets the recipe"
    )
    ap.add_argument("--recipe-scale", type=float, default=PRINTED_THICKNESS_SCALE)
    ap.add_argument(
        "--open-pitch-deg", type=float, default=math.degrees(INTER_BLADE_ANGLE_RAD),
        help="angle between neighbouring open blades",
    )  # fmt: skip
    a = ap.parse_args(argv)
    summary = run(
        shards=a.shards, calibration=a.calibration, out_csv=a.out_csv, out_json=a.out_json,
        recipe_design=a.recipe_design, recipe_scale=a.recipe_scale,
        pitch_rad=math.radians(a.open_pitch_deg),
    )  # fmt: skip
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
