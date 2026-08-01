#!/usr/bin/env python
"""Recommend the designs to print for Phase 6 (campaign Pareto + 3D verification).

Consolidates the campaign's top-k structurally-diverse Pareto designs with the Stage-3.C 3D
``verification.json`` (if available) into one ``recommended.json``: the 3-5 designs to print for the
blinded A/B feel test, with their coarse and fine (whole-fan) 3D ``J_fan`` side by side. Works before
verification finishes (fine fields blank).

Two campaign formats:
  * ``--shared-dir`` — the Stage-3 distributed campaign's Drive shards (``evaluations_*.jsonl``); the
    current aero-first blade campaign. Verified designs join the shards by ``design_hash``.
  * ``--campaign-dir`` — a legacy Phase-4 ``checkpoint.npz`` campaign (integer-indexed designs).

    python scripts/recommend_designs.py --shared-dir /content/drive/MyDrive/fanopt/campaign_trapezoid \\
        --verification data/phase5_verify_blade/verification.json --top-k 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fanopt.bo.blade_results import recommend_blades
from fanopt.bo.results import recommend


def run(
    *,
    out_dir: Path,
    top_k: int,
    campaign_dir: Path | None = None,
    shared_dir: Path | None = None,
    verification_path: Path | None = None,
) -> dict[str, Any]:
    """Write ``recommended.json`` to ``out_dir``; return the summary.

    Reads the Stage-3 shard campaign (``shared_dir``) or a legacy Phase-4 checkpoint (``campaign_dir``).
    """
    if shared_dir is not None:
        summary = recommend_blades(shared_dir, top_k=top_k, verification_path=verification_path)
    elif campaign_dir is not None:
        summary = recommend(campaign_dir, top_k=top_k, verification_path=verification_path)
    else:
        raise ValueError("provide shared_dir (Stage-3 shards) or campaign_dir (legacy checkpoint)")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "recommended.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _fmt(v: object, spec: str, scale: float = 1.0) -> str:
    """Format a possibly-``None`` metric ('—' when absent)."""
    return format(v * scale, spec) if isinstance(v, (int, float)) else "—"


def _print_blade_table(summary: dict[str, Any]) -> None:
    ranked = summary["ranked"]
    if not ranked:
        return
    print(f"\n  all {len(ranked)} verified designs, ranked by fine (whole-fan) 3D J_fan:")
    print(f"  {'':2}{'design':14}{'rib':8}{'J_fan_fine':>12}{'J_fan_crs':>12}"
          f"{'mass_g':>8}{'defl_mm':>9}  status")
    for rk in ranked:
        star = "★ " if rk["recommended_for_print"] else "  "
        status = "SUSPECT" if rk["suspect"] else "verified"
        name = str(rk["name"])[:13]  # {rank}_{hash} names are long — truncate to the column
        print(f"  {star}{name:<14}{str(rk['rib_mode'] or '—'):8}"
              f"{_fmt(rk['j_fan_3d'], '.3e'):>12}{_fmt(rk['j_fan_coarse'], '.3e'):>12}"
              f"{_fmt(rk['mass_kg'], '.1f', 1000):>8}{_fmt(rk['deflection_m'], '.3f', 1000):>9}"
              f"  {status}")
    print("\n  ★ = recommended for the Phase-6 blinded A/B print set (top-k structurally diverse).")


def _print_legacy_table(summary: dict[str, Any]) -> None:
    ranked = summary["ranked"]
    if not ranked:
        return
    print(f"\n  all {len(ranked)} verified designs, ranked by 3D J_fan:")
    print(f"  {'':2}{'design':12}{'J_fan_fine':>12}{'J_fan_crs':>12}{'I_wrist':>11}"
          f"{'defl_mm':>9}  status")
    for rk in ranked:
        star = "★ " if rk["recommended_for_print"] else "  "
        status = "SUSPECT" if rk["suspect"] else "verified"
        print(f"  {star}{str(rk['name']):<11}{_fmt(rk['j_fan_3d'], '.3e'):>12}"
              f"{_fmt(rk['j_fan_coarse'], '.3e'):>12}{_fmt(rk['i_wrist_kgm2'], '.3e'):>11}"
              f"{_fmt(rk['structural_m'], '.3f', 1000):>9}  {status}")
    print("\n  ★ = recommended for the Phase-6 blinded A/B print set (top-k structurally diverse).")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shared-dir", type=Path, default=None,
        help="Stage-3 distributed campaign folder with evaluations_*.jsonl shards.",
    )
    parser.add_argument(
        "--campaign-dir", type=Path, default=None, help="Legacy Phase-4 checkpoint.npz campaign dir.",
    )
    parser.add_argument(
        "--verification", type=Path, default=None, help="Stage-3.C verification.json (optional)"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/phase6_recommend"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)
    if args.shared_dir is None and args.campaign_dir is None:
        parser.error("provide --shared-dir (Stage-3 shards) or --campaign-dir (legacy checkpoint)")

    summary = run(
        out_dir=args.out_dir,
        top_k=args.top_k,
        campaign_dir=args.campaign_dir,
        shared_dir=args.shared_dir,
        verification_path=args.verification,
    )
    print(
        f"[recommend] {len(summary['recommended'])} designs to print "
        f"(top-{summary['top_k']} of {summary['n_pareto']} Pareto) | "
        f"verification={summary['verification']} | {summary['n_verified']} 3D-verified"
    )
    if args.shared_dir is not None:
        _print_blade_table(summary)
    else:
        _print_legacy_table(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
