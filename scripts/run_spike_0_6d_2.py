#!/usr/bin/env python
"""Spike 0.6d.2 — 2D thin-plate added-mass frequency-consistency (GATING).

Spec reference: docs/report-final.md §Phase 0 Spike 0.6d (sub-spike 0.6d.2);
protocol in docs/spike_0_6d_protocol.md. Redesigned 2026-05-15 (see
docs/phase_logs/phase_0_signoff.md Note 3).

**Gate (normalization-invariant Phase-4 de-risk):** run the 2D thin-plate
at TWO pitching frequencies (ω₁, ω₂) with the same plate / pivot / θ_max,
Fourier-project each moment-coefficient trace onto the added-mass (sin φ)
basis, and require the recovered added-mass coefficient
``I_a = a_sin/(ω²·θ_max)`` to AGREE between the two runs. ``I_a`` is a
pure geometric/fluid constant — frequency-independence is the signature
of physically-faithful added-mass behaviour, and the comparison is
SU2-against-SU2 so the FREESTREAM_PRESS_EQ_ONE q_ref cancels.

A Sedov/Newman closed-form magnitude comparison is also computed but is
**advisory** (its absolute scale needs SU2's exact reference-state
handling, which is Phase 5 step 62.5's job).

Inputs:

* ``--history-csv-f1`` / ``--omega-f1`` — first run's history.csv + ω₁.
* ``--history-csv-f2`` / ``--omega-f2`` — second run's history.csv + ω₂.
* ``--pitching-amplitude-rad`` — θ_max (same for both runs).
* ``--chord-m`` / ``--pivot-offset-normalized`` — plate geometry (for the
  advisory closed-form only).
* ``--n-cycles`` — cycles each run executed.

Outputs:

* ``data/spike_0_6d/sub_2_result.json`` — full Tier1AddedMassResult.
* ``data/spike_0_6d/sub_2.PASS`` or ``sub_2.FAIL`` marker.

Exit codes: ``0`` pass, ``1`` fail, ``2`` input error.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from fanopt.cfd.spike_0_6d import (
    MACH_UNSTEADY_LOCK,
    check_added_mass_freq_consistency,
    recover_added_mass_projection,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "data" / "spike_0_6d"
DEFAULT_RESULT_JSON = DEFAULT_DIR / "sub_2_result.json"
DEFAULT_MARKER_DIR = DEFAULT_DIR

# CMz is the pitching moment for the 2D thin-plate cfg (pitches about z;
# see configs/su2/thin_plate_2d_pitching.cfg.j2). CMy is identically zero
# for a 2D x-y mesh because forces are in-plane (no z-component, no
# z-extent → ∫(Fz·dx − Fx·dz) = 0), so it must NOT be probed first.
MOMENT_COLUMN_CANDIDATES = ("cmz", "cm", "cmoment", "cmy")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history-csv-f1", type=Path, required=True)
    p.add_argument("--omega-f1", type=float, required=True)
    p.add_argument("--history-csv-f2", type=Path, required=True)
    p.add_argument("--omega-f2", type=float, required=True)
    p.add_argument("--pitching-amplitude-rad", type=float, required=True)
    p.add_argument("--chord-m", type=float, required=True)
    p.add_argument(
        "--pivot-offset-normalized",
        type=float,
        default=-0.5,
        help="`a = (x_pivot - c/2) / (c/2)`. -0.5 at quarter-chord (default).",
    )
    p.add_argument("--n-cycles", type=int, default=5)
    p.add_argument(
        "--fluid-density-kg-per-m3",
        type=float,
        default=1.225,
        help="Freestream density for the advisory closed-form. Default: sea-level air.",
    )
    p.add_argument(
        "--freq-consistency-tol",
        type=float,
        default=0.25,
        help="GATING: max relative I_a difference between the two runs. Default 0.25.",
    )
    p.add_argument(
        "--closed-form-factor-tol",
        type=float,
        default=2.0,
        help="ADVISORY: closed-form factor band. Default 2.0× (not gating).",
    )
    p.add_argument("--result-json", type=Path, default=DEFAULT_RESULT_JSON)
    p.add_argument("--marker-dir", type=Path, default=DEFAULT_MARKER_DIR)
    return p.parse_args(argv)


def _read(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return text if text.strip() else None


def _write_marker(marker_dir: Path, passed: bool) -> Path:
    marker_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("sub_2.PASS", "sub_2.FAIL"):
        sp = marker_dir / stale
        if sp.exists():
            sp.unlink()
    marker = marker_dir / ("sub_2.PASS" if passed else "sub_2.FAIL")
    marker.write_text("")
    return marker


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    text_f1 = _read(args.history_csv_f1)
    text_f2 = _read(args.history_csv_f2)
    if text_f1 is None:
        print(f"[spike_0_6d.2] f1 history.csv unreadable: {args.history_csv_f1}", file=sys.stderr)
        return 2
    if text_f2 is None:
        print(f"[spike_0_6d.2] f2 history.csv unreadable: {args.history_csv_f2}", file=sys.stderr)
        return 2

    proj_f1 = recover_added_mass_projection(
        text_f1,
        omega_rad_per_s=args.omega_f1,
        pitching_amplitude_rad=args.pitching_amplitude_rad,
        n_cycles=args.n_cycles,
        moment_column_candidates=MOMENT_COLUMN_CANDIDATES,
        history_path=str(args.history_csv_f1),
    )
    proj_f2 = recover_added_mass_projection(
        text_f2,
        omega_rad_per_s=args.omega_f2,
        pitching_amplitude_rad=args.pitching_amplitude_rad,
        n_cycles=args.n_cycles,
        moment_column_candidates=MOMENT_COLUMN_CANDIDATES,
        history_path=str(args.history_csv_f2),
    )
    if proj_f1.a_sin == 0.0 and proj_f2.a_sin == 0.0:
        print(
            f"[spike_0_6d.2] no moment column {MOMENT_COLUMN_CANDIDATES} in either "
            "history.csv (added-mass projection is identically zero)",
            file=sys.stderr,
        )
        return 2

    result = check_added_mass_freq_consistency(
        proj_f1,
        proj_f2,
        chord_m=args.chord_m,
        pivot_offset_normalized=args.pivot_offset_normalized,
        fluid_density_kg_per_m3=args.fluid_density_kg_per_m3,
        freq_consistency_tol=args.freq_consistency_tol,
        closed_form_factor_tol=args.closed_form_factor_tol,
    )

    payload = {
        "spec_reference": "docs/report-final.md §Phase 0 Spike 0.6d (sub-spike 0.6d.2)",
        "lock_reference": "H10 supplement; redesigned 2026-05-15 (freq-consistency gate)",
        "mach_unsteady_lock": MACH_UNSTEADY_LOCK,
        "gate_note": (
            "GATE = freq_consistency_passed (normalization-invariant). "
            "closed_form_advisory_ok is reported but does NOT gate — absolute "
            "nondim scale is Phase 5 step 62.5's job."
        ),
        "projection_f1": asdict(proj_f1),
        "projection_f2": asdict(proj_f2),
        "result": asdict(result),
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(payload, indent=2) + "\n")
    marker = _write_marker(args.marker_dir, result.passed)

    print(
        f"[spike_0_6d.2] f1: ω={result.omega_f1_rad_per_s}  Ia_nondim={result.recovered_ia_nondim_f1:+.6e}"
    )
    print(
        f"[spike_0_6d.2] f2: ω={result.omega_f2_rad_per_s}  Ia_nondim={result.recovered_ia_nondim_f2:+.6e}"
    )
    print(
        f"[spike_0_6d.2] freq-consistency rel_diff = {result.freq_consistency_rel_diff:.4f} "
        f"(tol {result.freq_consistency_tol}) -> {'PASS' if result.freq_consistency_passed else 'FAIL'}"
    )
    print(
        f"[spike_0_6d.2] closed-form (ADVISORY): factor={result.closed_form_factor_f1:.4f} "
        f"ok={result.closed_form_advisory_ok} (does NOT gate)"
    )
    print(
        f"[spike_0_6d.2] drag/added-mass ratio f1 (advisory diagnostic) = "
        f"{result.drag_to_added_mass_ratio_f1:.4f}"
    )
    print(f"[spike_0_6d.2] OVERALL = {'PASS' if result.passed else 'FAIL'} -> {marker}")
    print(f"[spike_0_6d.2] result JSON = {args.result_json}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
