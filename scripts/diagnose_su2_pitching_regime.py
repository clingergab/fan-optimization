#!/usr/bin/env python
"""Diagnose the force-regime of a SU2 pitching-airfoil run.

Spike 0.6c.2 diagnostic (added 2026-05-14 after the first Cell 8 run
produced non-physical CL values). Goal: determine whether the SU2
history shows **wind-tunnel-like** force behavior (CL ∝ α, in phase),
**added-mass-dominated** behavior (CL ∝ d²α/dt², leading α by ~90°),
or **non-physical bias** (cycle-mean CL significantly non-zero — neither
regime).

Reads the full per-outer-iter SU2 history.csv (not the per-cycle
aggregates), reconstructs the prescribed pitching angle
``α(t) = θ_max · sin(ω · t)``, and computes:

* Cycle-mean CL and CD (must be ≈ 0 for symmetric pitching about α=0°
  regardless of regime).
* Cycle amplitude of CL.
* Cross-correlation phase lag of CL vs α (signed degrees, range
  ``(-180, 180]``).
* Bias ratio = ``|cycle-mean| / cycle-amplitude``.

Outputs a console summary + a JSON metrics file + (optionally) a PNG
plot of CL(t), CD(t), and α(t) overlaid.

Classification thresholds:

* **WIND-TUNNEL-LIKE:** |phase lag| < 30°. CL roughly in phase with α
  (lift follows angle of attack to first order).
* **ADDED-MASS DOMINANCE:** 60° < |phase lag| < 120°. CL leads or lags
  α by ~90° (force ∝ derivative of α).
* **ANTI-PHASE:** |phase lag| > 150°. CL is 180° out of phase with α
  (rare; usually indicates a sign error in the cfg).
* **INTERMEDIATE:** anything else.

Independently:

* **NON-PHYSICAL BIAS:** ``bias_ratio > 0.2``. Cycle-mean CL is > 20% of
  cycle amplitude — symmetric pitching of a symmetric body about α=0°
  should produce a near-zero cycle-mean force regardless of regime, so
  a large bias indicates a numerical artifact, sign error, or
  asymmetric geometry/motion bug.

Spec reference: ``docs/phase_logs/spike_0_6c.md`` (2026-05-14 appendix
when the diagnostic runs).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 — Agg backend must be set first
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import parse_su2_history_to_cycles as parse_cli  # noqa: E402

# Classification thresholds (degrees of phase lag).
WIND_TUNNEL_MAX_LAG_DEG: float = 30.0
ADDED_MASS_MIN_LAG_DEG: float = 60.0
ADDED_MASS_MAX_LAG_DEG: float = 120.0
ANTI_PHASE_MIN_LAG_DEG: float = 150.0

# Bias threshold: |cycle-mean| / cycle-amplitude exceeding this is flagged
# as non-physical regardless of phase classification.
BIAS_NON_PHYSICAL_THRESHOLD: float = 0.2


def _normalise_lag_degrees(lag_deg: float) -> float:
    """Normalise a phase lag to the range ``(-180, 180]``."""
    while lag_deg > 180.0:
        lag_deg -= 360.0
    while lag_deg <= -180.0:
        lag_deg += 360.0
    return lag_deg


def _phase_lag_via_xcorr(
    cl_kept: np.ndarray, alpha_kept: np.ndarray, dt: float, omega: float
) -> float:
    """Cross-correlation phase lag of CL relative to α, in signed degrees.

    Positive lag = CL trails α by that many degrees.
    Negative lag = CL leads α by that many degrees.

    Both signals are mean-centered and unit-norm before correlation so
    the lag estimate is amplitude-independent.
    """
    if len(cl_kept) < 2 or len(alpha_kept) < 2:
        return 0.0
    cl_centered = cl_kept - cl_kept.mean()
    alpha_centered = alpha_kept - alpha_kept.mean()
    cl_std = cl_centered.std()
    alpha_std = alpha_centered.std()
    if cl_std < 1e-30 or alpha_std < 1e-30:
        return 0.0
    cl_norm = cl_centered / cl_std
    alpha_norm = alpha_centered / alpha_std
    xcorr = np.correlate(cl_norm, alpha_norm, mode="full")
    lag_samples = int(xcorr.argmax() - (len(alpha_norm) - 1))
    lag_seconds = lag_samples * dt
    period = 2.0 * math.pi / omega
    lag_fraction = lag_seconds / period
    return _normalise_lag_degrees(lag_fraction * 360.0)


def _classify(bias_ratio: float, phase_lag_deg: float) -> tuple[str, list[str]]:
    """Return (primary_label, list_of_findings).

    ``primary_label`` is the regime classification; ``findings`` is a
    list of human-readable diagnostic statements (always includes the
    primary label; may include a NON-PHYSICAL BIAS warning).
    """
    findings: list[str] = []
    if bias_ratio > BIAS_NON_PHYSICAL_THRESHOLD:
        findings.append(
            f"NON-PHYSICAL BIAS: |cycle-mean|/amplitude = {bias_ratio:.3f} "
            f"exceeds threshold {BIAS_NON_PHYSICAL_THRESHOLD}. A symmetric "
            "airfoil pitching about α=0° should produce ≈ zero cycle-mean "
            "force regardless of regime — bias suggests a numerical "
            "artifact, sign error, or asymmetric geometry/motion."
        )

    abs_lag = abs(phase_lag_deg)
    if abs_lag < WIND_TUNNEL_MAX_LAG_DEG:
        label = "WIND_TUNNEL_LIKE"
        findings.append(
            f"WIND-TUNNEL-LIKE: |phase lag| = {abs_lag:.1f}° < "
            f"{WIND_TUNNEL_MAX_LAG_DEG}°. CL roughly in phase with α."
        )
    elif ADDED_MASS_MIN_LAG_DEG < abs_lag < ADDED_MASS_MAX_LAG_DEG:
        label = "ADDED_MASS_DOMINANCE"
        findings.append(
            f"ADDED-MASS DOMINANCE: |phase lag| = {abs_lag:.1f}° in the "
            f"[{ADDED_MASS_MIN_LAG_DEG}°, {ADDED_MASS_MAX_LAG_DEG}°] band. "
            "CL leads/lags α by ~90° — characteristic of force ∝ derivative "
            "of α (moving body in still air)."
        )
    elif abs_lag > ANTI_PHASE_MIN_LAG_DEG:
        label = "ANTI_PHASE"
        findings.append(
            f"ANTI-PHASE: |phase lag| = {abs_lag:.1f}° > "
            f"{ANTI_PHASE_MIN_LAG_DEG}°. CL is ~180° out of phase with α — "
            "usually indicates a sign error in PITCHING_OMEGA or motion axis."
        )
    else:
        label = "INTERMEDIATE"
        findings.append(
            f"INTERMEDIATE: phase lag = {phase_lag_deg:+.1f}° falls outside "
            "the clean wind-tunnel / added-mass / anti-phase bands. Force "
            "regime is a mix; classification is ambiguous."
        )

    return label, findings


def diagnose_history(
    history_path: Path,
    *,
    theta_max_rad: float,
    omega_shm_rad_per_s: float,
    n_cycles: int = 5,
) -> dict[str, Any]:
    """Run the diagnostic on a SU2 history.csv and return metrics + classification.

    Reads the history via :mod:`parse_su2_history_to_cycles` (re-using
    column detection + multi-inner-iter collapse). Reconstructs ``α(t)``
    from the prescribed pitching kinematics. Computes cycle-mean CL,
    cycle amplitude, cross-correlation phase lag, and bias ratio over
    the kept cycles (cycle 0 discarded as initial transient).
    """
    rows, colmap = parse_cli._read_history(history_path)
    per_iter = parse_cli._per_outer_iter(rows)
    n_total = len(per_iter)
    if n_total < n_cycles * 2:
        raise ValueError(
            f"Need at least {n_cycles * 2} outer iters for diagnostic (≥ 2 "
            f"per cycle so phase analysis is meaningful); got {n_total}."
        )

    # Reconstruct time axis. Prefer recorded 'time' if available; else
    # assume dt = T/n_per per the production cfg.
    if all("time" in r for r in per_iter):
        t = np.array([r["time"] for r in per_iter])
    else:
        period = 2.0 * math.pi / omega_shm_rad_per_s
        n_per_cycle = n_total // n_cycles
        dt = period / n_per_cycle
        t = np.arange(n_total) * dt

    cl = np.array([r["cl"] for r in per_iter])
    cd = np.array([r["cd"] for r in per_iter])
    alpha = theta_max_rad * np.sin(omega_shm_rad_per_s * t)

    # Discard cycle 0 (initial transient) — match the production
    # spike_0_6c analyzer's BENCHMARK_CYCLES_DISCARD = 1.
    n_per_cycle = n_total // n_cycles
    kept_slice = slice(n_per_cycle, None)
    t_kept = t[kept_slice]
    cl_kept = cl[kept_slice]
    cd_kept = cd[kept_slice]
    alpha_kept = alpha[kept_slice]

    cl_mean = float(np.mean(cl_kept))
    cd_mean = float(np.mean(cd_kept))
    cl_amplitude = float((np.max(cl_kept) - np.min(cl_kept)) / 2.0)
    bias_ratio = abs(cl_mean) / cl_amplitude if cl_amplitude > 0.0 else float("inf")

    dt_eff = float(t_kept[1] - t_kept[0]) if len(t_kept) > 1 else 1.0
    phase_lag_deg = _phase_lag_via_xcorr(cl_kept, alpha_kept, dt_eff, omega_shm_rad_per_s)

    label, findings = _classify(bias_ratio, phase_lag_deg)

    return {
        "history_path": str(history_path),
        "n_outer_iters_total": n_total,
        "n_outer_iters_per_cycle": int(n_per_cycle),
        "n_cycles_kept": int(n_cycles - 1),
        "columns_detected": colmap,
        "metrics": {
            "cl_mean_kept_cycles": cl_mean,
            "cd_mean_kept_cycles": cd_mean,
            "cl_amplitude_kept_cycles": cl_amplitude,
            "bias_ratio_mean_over_amplitude": bias_ratio,
            "phase_lag_cl_vs_alpha_degrees": phase_lag_deg,
        },
        "classification": label,
        "findings": findings,
        "kinematic_inputs": {
            "theta_max_rad": theta_max_rad,
            "omega_shm_rad_per_s": omega_shm_rad_per_s,
            "n_cycles": n_cycles,
        },
    }


def plot_traces(
    history_path: Path,
    *,
    theta_max_rad: float,
    omega_shm_rad_per_s: float,
    n_cycles: int,
    out_png: Path,
) -> Path:
    """Plot CL(t), CD(t), and α(t) overlaid; save PNG to ``out_png``."""
    rows, _colmap = parse_cli._read_history(history_path)
    per_iter = parse_cli._per_outer_iter(rows)
    n_total = len(per_iter)

    if all("time" in r for r in per_iter):
        t = np.array([r["time"] for r in per_iter])
    else:
        period = 2.0 * math.pi / omega_shm_rad_per_s
        n_per_cycle = n_total // n_cycles
        dt = period / n_per_cycle
        t = np.arange(n_total) * dt

    cl = np.array([r["cl"] for r in per_iter])
    cd = np.array([r["cd"] for r in per_iter])
    alpha_deg = np.degrees(theta_max_rad * np.sin(omega_shm_rad_per_s * t))

    fig, (ax_cl, ax_cd, ax_alpha) = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    ax_cl.plot(t, cl, "b-", linewidth=0.8)
    ax_cl.set_ylabel("CL")
    ax_cl.grid(True, alpha=0.3)
    ax_cl.set_title(f"SU2 pitching-airfoil diagnostic — {history_path.name}")

    ax_cd.plot(t, cd, "r-", linewidth=0.8)
    ax_cd.set_ylabel("CD")
    ax_cd.grid(True, alpha=0.3)

    ax_alpha.plot(t, alpha_deg, "k-", linewidth=0.8)
    ax_alpha.set_ylabel("α(t) [deg]")
    ax_alpha.set_xlabel("Time [s]")
    ax_alpha.grid(True, alpha=0.3)

    # Mark cycle boundaries.
    n_per_cycle = n_total // n_cycles
    period = (t[n_per_cycle] - t[0]) if n_per_cycle < n_total else 1.0
    for k in range(1, n_cycles + 1):
        for ax in (ax_cl, ax_cd, ax_alpha):
            ax.axvline(k * period, color="gray", linestyle=":", linewidth=0.5)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def _print_report(result: dict[str, Any]) -> None:
    print("=== SU2 pitching-airfoil regime diagnostic ===")
    print(f"history          {result['history_path']}")
    print(f"n_outer_iters    {result['n_outer_iters_total']}")
    print(f"per_cycle        {result['n_outer_iters_per_cycle']}")
    print(f"cycles_kept      {result['n_cycles_kept']}")
    print(f"columns_used     {result['columns_detected']}")
    print()
    m = result["metrics"]
    print("--- kept-cycle metrics ---")
    print(f"cl_mean          {m['cl_mean_kept_cycles']:+.4e}")
    print(f"cd_mean          {m['cd_mean_kept_cycles']:+.4e}")
    print(f"cl_amplitude     {m['cl_amplitude_kept_cycles']:.4e}")
    print(f"bias_ratio       {m['bias_ratio_mean_over_amplitude']:.3f}")
    print(f"phase_lag_deg    {m['phase_lag_cl_vs_alpha_degrees']:+.1f}°")
    print()
    print(f"CLASSIFICATION   {result['classification']}")
    print()
    print("--- findings ---")
    for f in result["findings"]:
        print(f"  • {f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history", type=Path, required=True, help="SU2 history.csv path.")
    p.add_argument(
        "--theta-max-rad",
        type=float,
        default=math.radians(10.0),
        help="Pitching amplitude in rad. Default: 10° = 0.1745 rad.",
    )
    p.add_argument(
        "--omega-shm-rad-per-s",
        type=float,
        required=True,
        help="SHM angular frequency in rad/s (PITCHING_OMEGA magnitude).",
    )
    p.add_argument(
        "--n-cycles",
        type=int,
        default=5,
        help="Number of cycles in the run. Default 5 (Spike 0.6c.2 canonical).",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Path to write the metrics JSON. Default: alongside --history.",
    )
    p.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help="Path to write the trace plot. Default: alongside --history.",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting (still writes JSON + prints report).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.history.exists():
        print(f"[diagnose] history not found: {args.history}", file=sys.stderr)
        return 2

    out_json = args.out_json or args.history.parent / "regime_diagnostic.json"
    out_png = args.out_png or args.history.parent / "regime_diagnostic.png"

    result = diagnose_history(
        args.history,
        theta_max_rad=abs(args.theta_max_rad),
        omega_shm_rad_per_s=abs(args.omega_shm_rad_per_s),
        n_cycles=args.n_cycles,
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    _print_report(result)
    print(f"\n[diagnose] metrics JSON  {out_json}")

    if not args.no_plot:
        plot_traces(
            args.history,
            theta_max_rad=abs(args.theta_max_rad),
            omega_shm_rad_per_s=abs(args.omega_shm_rad_per_s),
            n_cycles=args.n_cycles,
            out_png=out_png,
        )
        print(f"[diagnose] trace plot    {out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
