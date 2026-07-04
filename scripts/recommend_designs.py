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
    for d in summary["recommended"]:
        j3d = f"{d['j_fan_3d']:.3e}" if d["j_fan_3d"] is not None else "—"
        tag = " (3D-verified)" if d["verified"] else ""
        print(
            f"  b{d['blade_count']} {d['edge_profile']} (i{d['index']}): "
            f"J_fan_2D={d['j_fan_slice']:.3e} J_fan_3D={j3d} "
            f"I_wrist={d['i_wrist_kgm2']:.3e} defl={d['structural_m'] * 1000:.3f}mm{tag}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
