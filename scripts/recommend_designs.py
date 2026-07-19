#!/usr/bin/env python
"""Recommend the designs to print for Phase 6 (Pareto + 3D verification).

Consolidates the Phase-4 campaign's top-k structurally-diverse Pareto designs with
the Phase-5 3D ``verification.json`` (if available) into one ``recommended.json``:
the 3-5 designs to print for the blinded A/B feel test, with their 2D-slice and 3D
``J_fan`` side by side. Works before verification finishes (3D fields blank).

    python scripts/recommend_designs.py --campaign-dir data/phase4_bo \\
        --verification data/phase5_verify/verification.json --top-k 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fanopt.bo.results import recommend


def run(
    *,
    campaign_dir: Path,
    out_dir: Path,
    top_k: int,
    verification_path: Path | None = None,
) -> dict[str, Any]:
    """Write ``recommended.json`` to ``out_dir``; return the summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = recommend(campaign_dir, top_k=top_k, verification_path=verification_path)
    (out_dir / "recommended.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=Path("data/phase4_bo"))
    parser.add_argument(
        "--verification", type=Path, default=None, help="Phase-5 verification.json (optional)"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/phase6_recommend"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)

    summary = run(
        campaign_dir=args.campaign_dir,
        out_dir=args.out_dir,
        top_k=args.top_k,
        verification_path=args.verification,
    )
    print(
        f"[recommend] {len(summary['recommended'])} designs to print "
        f"(top-{summary['top_k']} of {summary['n_pareto']} Pareto) | "
        f"verification={summary['verification']} | {summary['n_verified']} 3D-verified"
    )

    ranked = summary["ranked"]
    if ranked:
        # Full ranked table of every verified design (by 3D J_fan) — ★ = print pick.
        print(f"\n  all {len(ranked)} verified designs, ranked by 3D J_fan:")
        print(f"  {'':2}{'design':12}{'J_fan_3D':>12}{'J_fan_2D':>12}{'I_wrist':>11}"
              f"{'defl_mm':>9}  status")
        for rk in ranked:
            star = "★ " if rk["recommended_for_print"] else "  "
            j3d = f"{rk['j_fan_3d']:.3e}" if rk["j_fan_3d"] is not None else "—"
            j2d = f"{rk['j_fan_slice']:.3e}" if rk["j_fan_slice"] is not None else "—"
            iw = f"{rk['i_wrist_kgm2']:.3e}" if rk["i_wrist_kgm2"] is not None else "—"
            defl = f"{rk['structural_m'] * 1000:.3f}" if rk["structural_m"] is not None else "—"
            status = "SUSPECT" if rk["suspect"] else "verified"
            print(f"  {star}{rk['name']:<11}{j3d:>12}{j2d:>12}"
                  f"{iw:>11}{defl:>9}  {status}")
        print("\n  ★ = recommended for the Phase-6 blinded A/B print set (top-k structurally diverse).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
