#!/usr/bin/env python3
"""Per-design 3D SIMP topology optimization over the top-N fine-verified blades.

Resolves the top-N designs by **fine 3D** ``J_fan`` (joining a Stage-3.C ``verification.json``
to the campaign shards), then runs the frozen-aero-skin SIMP TO on each design's own solid and
writes the carved density fields + a ``summary.json``. This is the V1 structural stage: the
aero winners come in, mass-trimmed structurally-screened blades come out, and 3 of them are
chosen to print.

By default runs in **screened** mode: for each design it finds the most aggressive (lowest-
volfrac) TO that still passes the structural screen (tip deflection < limit AND von Mises <
yield/FOS across all four load cases) — "remove as much material as possible while keeping
integrity", decided per design by the measured screen. ``--no-screen`` runs a single fixed
``--volfrac`` instead.

Thin wrapper — the resolver lives in :mod:`fanopt.cfd.blade_verify` and the TO in
:mod:`fanopt.topopt.blade_topopt`.

    python3 scripts/run_phase2_blade_to.py \
        --shared-dir data/campaign_trapezoid \
        --verification data/phase5_verify_blade/verification.json \
        --out-dir data/phase2_blade_to --top-k 10 --mesh-size-m 0.001
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

from fanopt.cfd.blade_verify import top_verified_designs
from fanopt.geometry.schema import SIGMA_Y_PETG_Z_PA
from fanopt.topopt.blade_fea_mesh import FeaMeshParams
from fanopt.topopt.blade_topopt import (
    DEFAULT_STRESS_FOS,
    DEFAULT_U_TIP_LIMIT_M,
    DEFAULT_VOLFRAC,
    DEFAULT_VOLFRAC_LADDER,
    run_blade_to_batch,
    topology_optimize_blade,
    topology_optimize_blade_screened,
)


def run(
    *,
    shared_dir: Path,
    verification: Path,
    out_dir: Path,
    top_k: int,
    max_iters: int,
    mesh_size_m: float,
    skin_thickness_m: float | None,
    n_workers: int = 1,
    screen: bool = True,
    volfrac: float = DEFAULT_VOLFRAC,
    volfrac_ladder: tuple[float, ...] = DEFAULT_VOLFRAC_LADDER,
    u_tip_limit_m: float = DEFAULT_U_TIP_LIMIT_M,
    stress_fos: float = DEFAULT_STRESS_FOS,
    progress: bool = True,
) -> dict:
    """Resolve the top-N verified designs and run the per-design TO batch."""
    designs = top_verified_designs(shared_dir, verification, top_k=top_k)
    if progress:
        mode = f"screened (ladder {volfrac_ladder}, FOS {stress_fos})" if screen else f"fixed volfrac {volfrac}"
        print(f"Resolved {len(designs)} designs by fine J_fan from {verification} | mode: {mode}")
        for name, _params, j3d in designs:
            print(f"  {name}  J_fan_3d={j3d:.4g}")

    def _log(name, res):
        if progress:
            vf = res.meta.get("accepted_volfrac")
            passed = res.meta.get("screen_passed")
            tag = "" if vf is None else f"  volfrac {vf} {'PASS' if passed else 'FAIL-screen'}"
            print(
                f"  [TO] {name}: removed {res.volume_removed_frac * 100:.1f}%  "
                f"mass {res.mass_kg * 1e3:.1f} g  u_tip {res.u_tip_max_m * 1e3:.3f} mm  "
                f"VM {res.max_von_mises_pa / 1e6:.2f} MPa{tag}"
            )

    if screen:
        optimize = functools.partial(
            topology_optimize_blade_screened,
            volfrac_ladder=volfrac_ladder,
            u_tip_limit_m=u_tip_limit_m,
            sigma_allow_pa=SIGMA_Y_PETG_Z_PA / stress_fos,
        )
    else:
        optimize = topology_optimize_blade

    kwargs: dict = {"optimize": optimize}
    if skin_thickness_m is not None:
        kwargs["skin_thickness_m"] = skin_thickness_m
    return run_blade_to_batch(
        [(name, params) for name, params, _ in designs],
        out_dir,
        mesh_params=FeaMeshParams(mesh_size_m=mesh_size_m),
        volfrac=volfrac,
        max_iters=max_iters,
        n_workers=n_workers,
        on_result=_log,
        **kwargs,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-dir", type=Path, required=True, help="campaign shard dir")
    parser.add_argument("--verification", type=Path, required=True, help="Stage-3.C verification.json")
    parser.add_argument("--out-dir", type=Path, default=Path("data/phase2_blade_to"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-iters", type=int, default=40)
    parser.add_argument("--mesh-size-m", type=float, default=FeaMeshParams().mesh_size_m)
    parser.add_argument("--skin-thickness-m", type=float, default=None)
    parser.add_argument("--n-workers", type=int, default=1,
                        help="designs to run in parallel (RAM-bound: ~18 GB/design at 0.8mm, ~57 GB at 0.6mm)")
    parser.add_argument("--no-screen", dest="screen", action="store_false",
                        help="run a single fixed --volfrac instead of the screened ladder search")
    parser.add_argument("--volfrac", type=float, default=DEFAULT_VOLFRAC, help="fixed-mode volfrac")
    parser.add_argument("--volfrac-ladder", type=float, nargs="+", default=list(DEFAULT_VOLFRAC_LADDER),
                        help="screened-mode volfracs, most-aggressive first")
    parser.add_argument("--u-tip-limit-mm", type=float, default=DEFAULT_U_TIP_LIMIT_M * 1e3)
    parser.add_argument("--stress-fos", type=float, default=DEFAULT_STRESS_FOS)
    args = parser.parse_args(argv)

    summary = run(
        shared_dir=args.shared_dir,
        verification=args.verification,
        out_dir=args.out_dir,
        top_k=args.top_k,
        max_iters=args.max_iters,
        mesh_size_m=args.mesh_size_m,
        skin_thickness_m=args.skin_thickness_m,
        n_workers=args.n_workers,
        screen=args.screen,
        volfrac=args.volfrac,
        volfrac_ladder=tuple(args.volfrac_ladder),
        u_tip_limit_m=args.u_tip_limit_mm / 1e3,
        stress_fos=args.stress_fos,
    )
    n_pass = sum(1 for r in summary["designs"] if r.get("screen_passed"))
    print(
        f"\nDone: {summary['n_succeeded']}/{summary['n_designs']} designs TO'd "
        f"({n_pass} passed the screen) -> {args.out_dir}/summary.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
