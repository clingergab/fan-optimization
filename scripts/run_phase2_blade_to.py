#!/usr/bin/env python3
"""Per-design 3D SIMP topology optimization over the top-N fine-verified blades.

Resolves the top-N designs by **fine 3D** ``J_fan`` (joining a Stage-3.C ``verification.json``
to the campaign shards), then runs the frozen-aero-skin SIMP TO on each design's own solid and
writes the carved density fields + a ``summary.json``. This is the V1 structural stage: the
aero winners come in, mass-trimmed structurally-screened blades come out, and 3 of them are
chosen to print.

Thin wrapper — the resolver lives in :mod:`fanopt.cfd.blade_verify` and the TO batch in
:mod:`fanopt.topopt.blade_topopt`.

    python3 scripts/run_phase2_blade_to.py \
        --shared-dir data/campaign_trapezoid \
        --verification data/phase5_verify_blade/verification.json \
        --out-dir data/phase2_blade_to --top-k 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fanopt.cfd.blade_verify import top_verified_designs
from fanopt.topopt.blade_fea_mesh import FeaMeshParams
from fanopt.topopt.blade_topopt import DEFAULT_VOLFRAC, run_blade_to_batch


def run(
    *,
    shared_dir: Path,
    verification: Path,
    out_dir: Path,
    top_k: int,
    volfrac: float,
    max_iters: int,
    mesh_size_m: float,
    skin_thickness_m: float | None,
    progress: bool = True,
) -> dict:
    """Resolve the top-N verified designs and run the per-design TO batch."""
    designs = top_verified_designs(shared_dir, verification, top_k=top_k)
    if progress:
        print(f"Resolved {len(designs)} designs by fine J_fan from {verification}")
        for name, _params, j3d in designs:
            print(f"  {name}  J_fan_3d={j3d:.4g}")

    def _log(name, res):
        if progress:
            print(
                f"  [TO] {name}: removed {res.volume_removed_frac * 100:.1f}%  "
                f"mass {res.mass_kg * 1e3:.1f} g  u_tip {res.u_tip_max_m * 1e3:.3f} mm  "
                f"VM {res.max_von_mises_pa / 1e6:.2f} MPa  ({res.iterations} it)"
            )

    mesh_params = FeaMeshParams(mesh_size_m=mesh_size_m)
    kwargs: dict = {}
    if skin_thickness_m is not None:
        kwargs["skin_thickness_m"] = skin_thickness_m
    return run_blade_to_batch(
        [(name, params) for name, params, _ in designs],
        out_dir,
        mesh_params=mesh_params,
        volfrac=volfrac,
        max_iters=max_iters,
        on_result=_log,
        **kwargs,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-dir", type=Path, required=True, help="campaign shard dir")
    parser.add_argument("--verification", type=Path, required=True, help="Stage-3.C verification.json")
    parser.add_argument("--out-dir", type=Path, default=Path("data/phase2_blade_to"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--volfrac", type=float, default=DEFAULT_VOLFRAC)
    parser.add_argument("--max-iters", type=int, default=40)
    parser.add_argument("--mesh-size-m", type=float, default=FeaMeshParams().mesh_size_m)
    parser.add_argument("--skin-thickness-m", type=float, default=None)
    args = parser.parse_args(argv)

    summary = run(
        shared_dir=args.shared_dir,
        verification=args.verification,
        out_dir=args.out_dir,
        top_k=args.top_k,
        volfrac=args.volfrac,
        max_iters=args.max_iters,
        mesh_size_m=args.mesh_size_m,
        skin_thickness_m=args.skin_thickness_m,
    )
    print(
        f"\nDone: {summary['n_succeeded']}/{summary['n_designs']} designs TO'd -> "
        f"{args.out_dir}/summary.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
