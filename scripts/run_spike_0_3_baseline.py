#!/usr/bin/env python
"""Spike 0.3 baseline runner — IMU + anemometer → J_fan_proxy / W_cycle.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.3; protocol in
docs/spike_0_3_protocol.md.

Reads:
  --imu          one or more IMU CSVs (t_s, theta_rad, omega_rad_per_s)
                 — typically 5 trials × 10 cycles each
  --anemometer   a 9-row CSV from the 3×3 anemometer grid (L8 lock)
                 columns: point, x_m, y_m, z_m, v_mean_m_per_s,
                          v_peak_m_per_s, notes
  --inertia      data/spike_0_2/results.json — supplies I_wrist_kgm2

Computes:
  - J_fan_proxy from the 9-point grid mean velocity × plane area
  - W_cycle (rectified power integral) from each IMU trace, then the
    mean across trials
  - J_fan_proxy / W_cycle, the canonical Phase-6 baseline

Sanity-checks IMU-derived kinematics against the locked spec (2 Hz,
8.8 rad/s, 0.7 rad) and warns (does NOT fail the run) if any drift > tol.

Writes baseline.json with everything Phase 6 needs to compare against.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fanopt.physical.anemometer import (
    PLANE_AREA_M2,
    RHO_AIR_KG_PER_M3,
    analyze_anemometer_grid,
    load_anemometer_csv,
)
from fanopt.physical.imu import (
    analyze_imu_trace,
    load_imu_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INERTIA_JSON = REPO_ROOT / "data" / "spike_0_2" / "results.json"
DEFAULT_OUT = REPO_ROOT / "data" / "spike_0_3" / "baseline.json"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imu",
        type=Path,
        nargs="+",
        required=True,
        help="One or more IMU CSV files, one per trial.",
    )
    parser.add_argument(
        "--anemometer",
        type=Path,
        required=True,
        help="9-row anemometer grid CSV (L8 lock 3×3 plane).",
    )
    parser.add_argument(
        "--inertia",
        type=Path,
        default=DEFAULT_INERTIA_JSON,
        help="Spike 0.2 results.json (supplies I_wrist_kgm2).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Where to write baseline.json (default: %(default)s).",
    )
    parser.add_argument(
        "--rho-air",
        type=float,
        default=RHO_AIR_KG_PER_M3,
        help="Air density override (kg/m³). Use if your bench is at altitude "
        "or non-standard temperature. Default: %(default)s.",
    )
    return parser.parse_args(argv)


def _load_inertia(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(
            f"{path}: Spike 0.2 results.json not found — run Spike 0.2 first "
            f"(scripts/spike_0_2_analyze.py)."
        )
    payload = json.loads(path.read_text())
    try:
        I_wrist = float(payload["result"]["I_wrist_kgm2"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(
            f"{path}: cannot read result.I_wrist_kgm2 — "
            f"is this a Spike 0.2 results.json?"
        ) from e
    passed = payload.get("result", {}).get("passed")
    return I_wrist, passed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ---- inputs --------------------------------------------------------
    try:
        I_wrist_kgm2, spike02_passed = _load_inertia(args.inertia)
    except (FileNotFoundError, ValueError) as e:
        print(f"[spike_0_3] inertia error: {e}", file=sys.stderr)
        return 2

    try:
        anem_grid = load_anemometer_csv(args.anemometer)
    except (FileNotFoundError, ValueError) as e:
        print(f"[spike_0_3] anemometer error: {e}", file=sys.stderr)
        return 2

    imu_traces = []
    for p in args.imu:
        try:
            imu_traces.append((p, load_imu_csv(p)))
        except (FileNotFoundError, ValueError) as e:
            print(f"[spike_0_3] IMU error ({p}): {e}", file=sys.stderr)
            return 2

    # ---- analysis ------------------------------------------------------
    anem_result = analyze_anemometer_grid(
        anem_grid,
        rho_air_kg_per_m3=args.rho_air,
        A_plane_m2=PLANE_AREA_M2,
    )

    imu_results = [(p, analyze_imu_trace(t, I_wrist_kgm2)) for (p, t) in imu_traces]

    W_cycle_per_trial = [r.W_cycle_J for (_, r) in imu_results]
    W_cycle_mean = statistics.fmean(W_cycle_per_trial)
    W_cycle_std = (
        statistics.stdev(W_cycle_per_trial) if len(W_cycle_per_trial) > 1 else 0.0
    )

    # Apples-to-apples ratio: peak proxy / mean per-cycle work
    J_per_W = anem_result.J_fan_proxy_N / W_cycle_mean if W_cycle_mean > 0 else float("nan")
    J_peak_per_W = (
        anem_result.J_fan_proxy_peak_N / W_cycle_mean
        if (anem_result.J_fan_proxy_peak_N is not None and W_cycle_mean > 0)
        else None
    )

    sanity_all_ok = all(r.sanity_ok for (_, r) in imu_results)
    trial_consistency_ok = (
        (W_cycle_std / W_cycle_mean) < 0.20 if W_cycle_mean > 0 else False
    )

    # ---- payload -------------------------------------------------------
    payload: dict[str, Any] = {
        "spec_reference": "docs/plan_R11.md §Phase 0 Spike 0.3 (+ L8 lock)",
        "inputs": {
            "imu_files": [str(p) for (p, _) in imu_traces],
            "anemometer_file": str(args.anemometer),
            "inertia_file": str(args.inertia),
            "I_wrist_kgm2": I_wrist_kgm2,
            "spike_0_2_passed": spike02_passed,
            "rho_air_kg_per_m3": args.rho_air,
            "A_plane_m2": PLANE_AREA_M2,
        },
        "anemometer": {
            **asdict(anem_result),
            "labels": list(anem_grid.labels),
            "v_mean_per_point": list(anem_grid.v_mean_m_per_s),
            "v_peak_per_point": (
                list(anem_grid.v_peak_m_per_s) if anem_grid.v_peak_m_per_s else None
            ),
        },
        "imu": {
            "n_trials": len(imu_results),
            "per_trial": [
                {
                    "file": str(p),
                    **asdict(r),
                }
                for (p, r) in imu_results
            ],
            "W_cycle_J_mean": W_cycle_mean,
            "W_cycle_J_std": W_cycle_std,
            "trial_consistency_ok": trial_consistency_ok,
            "kinematic_sanity_all_ok": sanity_all_ok,
        },
        "baseline": {
            "J_fan_proxy_N": anem_result.J_fan_proxy_N,
            "J_fan_proxy_peak_N": anem_result.J_fan_proxy_peak_N,
            "W_cycle_J": W_cycle_mean,
            "J_per_W": J_per_W,
            "J_peak_per_W": J_peak_per_W,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    _print_summary(payload, args.out)
    # The runner doesn't fail on sanity warnings — it just flags them.
    return 0


def _print_summary(payload: dict, out_path: Path) -> None:
    p = payload
    a = p["anemometer"]
    im = p["imu"]
    b = p["baseline"]
    print(f"[spike_0_3] I_wrist     = {p['inputs']['I_wrist_kgm2']:.6e} kg·m²  "
          f"(Spike 0.2 passed: {p['inputs']['spike_0_2_passed']})")
    print(f"[spike_0_3] ⟨v⟩_grid    = {a['v_mean_grid_m_per_s']:.3f} m/s  "
          f"(std {a['v_mean_grid_std_m_per_s']:.3f})")
    print(f"[spike_0_3] J_fan_proxy = {a['J_fan_proxy_N']:.3e} N "
          f"(rho={p['inputs']['rho_air_kg_per_m3']}, A={p['inputs']['A_plane_m2']} m²)")
    print(f"[spike_0_3] W_cycle     = {im['W_cycle_J_mean']:.3e} J  "
          f"(std {im['W_cycle_J_std']:.3e}, "
          f"trial-consistency-OK: {im['trial_consistency_ok']})")
    print(f"[spike_0_3] J/W         = {b['J_per_W']:.3e} N/J  ← canonical baseline")
    if b["J_peak_per_W"] is not None:
        print(f"[spike_0_3] J_peak/W    = {b['J_peak_per_W']:.3e} N/J  (peak-velocity proxy)")
    print(f"[spike_0_3] kinematics sanity (per trial):")
    for entry in im["per_trial"]:
        flags = (
            ("f" if entry["f_wave_ok"] else "F") +
            ("ω" if entry["omega_max_ok"] else "Ω") +
            ("θ" if entry["theta_max_ok"] else "Θ")
        )
        print(
            f"  {Path(entry['file']).name:30s}  "
            f"f={entry['f_wave_Hz']:.2f} Hz  "
            f"ω_max={entry['omega_max_rad_per_s']:.2f}  "
            f"θ_max={entry['theta_max_rad']:.2f}  "
            f"[{flags}]"
        )
    if not im["kinematic_sanity_all_ok"]:
        print("[spike_0_3] WARNING: at least one trial drifted outside the kinematic "
              "sanity band — re-shoot if the wandering trial dominates W_cycle.")
    print(f"[spike_0_3] wrote      {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
