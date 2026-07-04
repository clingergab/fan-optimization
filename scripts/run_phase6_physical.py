#!/usr/bin/env python
"""Phase 6 — reduce bench measurements of the printed designs vs predictions.

Reads the Phase-6 ``recommended.json`` (the printed top-k designs + their
predicted ``I_wrist`` / 3D ``J_fan``) and, per design, looks for bench data under
``<measurements>/<design>/{imu,anemometer,acoustic}.csv`` (design = ``b{blades}_i{index}``).
Reduces each present measurement and writes ``physical_results.json`` with the
measured-vs-predicted rows + the cross-design J_fan rank calibration.

    python scripts/run_phase6_physical.py \\
        --recommended data/phase6_recommend/recommended.json \\
        --measurements data/phase6_measurements

Missing measurements are fine — designs not yet measured report ``None`` and are
listed as pending. No measured values are invented.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fanopt.physical.calibration import DesignMeasurement, calibrate, reduce_design


def _design_name(entry: dict[str, Any]) -> str:
    return f"b{entry['blade_count']}_i{entry['index']}"


def run(*, recommended_path: Path, measurements_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Reduce every recommended design's measurements; write ``physical_results.json``."""
    recommended = json.loads(recommended_path.read_text(encoding="utf-8"))
    designs: list[DesignMeasurement] = []
    for entry in recommended.get("recommended", []):
        name = _design_name(entry)
        d_dir = measurements_dir / name
        designs.append(
            reduce_design(
                name,
                predicted_i_wrist_kgm2=entry.get("i_wrist_kgm2"),
                predicted_j_fan_3d=entry.get("j_fan_3d"),
                blade_count=entry.get("blade_count"),
                imu_csv=d_dir / "imu.csv",
                anemometer_csv=d_dir / "anemometer.csv",
                acoustic_csv=d_dir / "acoustic.csv",
            )
        )
    report = calibrate(designs)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "physical_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recommended", type=Path, default=Path("data/phase6_recommend/recommended.json")
    )
    parser.add_argument("--measurements", type=Path, default=Path("data/phase6_measurements"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/phase6_recommend"))
    args = parser.parse_args(argv)

    report = run(
        recommended_path=args.recommended,
        measurements_dir=args.measurements,
        out_dir=args.out_dir,
    )
    rank = report["j_fan_rank"]
    print(
        f"[phase6] {report['n_designs']} designs | measured: "
        f"imu={report['n_with_imu']} anemometer={report['n_with_anemometer']} "
        f"acoustic={report['n_with_acoustic']}"
    )
    print(f"  J_fan rank vs prediction: {rank}")
    for d in report["designs"]:
        pending = "" if d["n_measurements"] == 3 else f"  [{3 - d['n_measurements']} pending]"
        wc = f"{d['w_cycle_j']:.4f}J" if d["w_cycle_j"] is not None else "—"
        jf = f"{d['j_fan_proxy_n']:.4f}N" if d["j_fan_proxy_n"] is not None else "—"
        spl = f"{d['spl_db']:.1f}dB" if d["spl_db"] is not None else "—"
        print(f"  {d['name']}: W_cycle={wc}  J_fan_proxy={jf}  SPL={spl}{pending}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
