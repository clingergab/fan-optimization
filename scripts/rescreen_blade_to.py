#!/usr/bin/env python3
"""Re-screen already-optimized blade designs from their saved density fields — no re-optimization.

Loads the carved density fields a prior ``run_phase2_blade_to`` wrote to ``--out-dir`` and re-runs
the final structural screen once per design (one factorization vs the full TO's ~160), emitting the
**honest as-printed** readouts: carved-mass tip deflection, all-element AND solid-only von Mises, and
the per-load breakdown. Use it to get the updated screen numbers (carved-mass inertial, solid-only
stress) without paying for the optimization again — e.g. to check whether a high-stress design is a
gray-element artifact.

Mesh size and skin thickness are read from the original run's ``<out-dir>/summary.json`` so the
rebuilt mesh matches the saved density (override only if you know they differ). Designs are resolved
by fine 3D ``J_fan`` (same ranking as the TO), defaulting to the top 4 print candidates.

    python3 scripts/rescreen_blade_to.py \
        --shared-dir data/campaign_trapezoid \
        --verification data/stage3_verify_blade/verification.json \
        --out-dir data/stage4_blade_to --top-k 4 --n-workers 4

    ``--out-dir`` MUST be the same folder the TO run wrote its ``<name>_density.npy`` fields to
    (``stage4_blade_to`` in the notebooks) — pointing elsewhere re-screens 0 designs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fanopt.cfd.blade_verify import top_verified_designs
from fanopt.geometry.schema import SIGMA_Y_PETG_Z_PA
from fanopt.topopt.blade_fea_mesh import FeaMeshParams
from fanopt.topopt.blade_topopt import DEFAULT_STRESS_FOS, rescreen_blade_to_batch


def _run_params(out_dir: Path, mesh_size_m: float | None, skin_thickness_m: float | None):
    """Resolve mesh/skin from the original run's summary.json unless explicitly overridden."""
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        mesh_size_m = prior.get("mesh_size_m", mesh_size_m) if mesh_size_m is None else mesh_size_m
        skin_thickness_m = prior.get("skin_thickness_m", skin_thickness_m) if skin_thickness_m is None else skin_thickness_m
    if mesh_size_m is None or skin_thickness_m is None:
        raise SystemExit(
            "mesh_size_m / skin_thickness_m not found in summary.json; pass --mesh-size-m and --skin-thickness-m"
        )
    return mesh_size_m, skin_thickness_m


def run(
    *,
    shared_dir: Path,
    verification: Path,
    out_dir: Path,
    top_k: int,
    n_workers: int,
    mesh_size_m: float | None,
    skin_thickness_m: float | None,
    stress_fos: float,
    progress: bool = True,
) -> dict:
    """Resolve the top-N designs and re-screen their saved density fields."""
    mesh_size_m, skin_thickness_m = _run_params(out_dir, mesh_size_m, skin_thickness_m)
    designs = top_verified_designs(shared_dir, verification, top_k=top_k)
    sigma_allow_mpa = SIGMA_Y_PETG_Z_PA / stress_fos / 1e6
    if progress:
        print(
            f"Re-screening {len(designs)} designs from {out_dir} "
            f"(mesh {mesh_size_m * 1e3:.2f} mm, skin {skin_thickness_m * 1e3:.2f} mm, "
            f"σ_allow {sigma_allow_mpa:.1f} MPa, {n_workers} workers)"
        )

    def _log(name, res):
        if progress:
            drift = res.meta.get("remap_max_dist_m", 0.0)
            drift_tag = f"  [remap drift {drift * 1e3:.2f} mm]" if drift and drift > 1e-9 else ""
            print(
                f"  [rescreen] {name}: mass {res.mass_kg * 1e3:.1f} g  "
                f"u_tip {res.u_tip_max_m * 1e3:.3f} mm  "
                f"VM {res.max_von_mises_pa / 1e6:.2f} MPa "
                f"(solid-only {res.max_von_mises_solid_pa / 1e6:.2f} MPa, allow {sigma_allow_mpa:.1f}){drift_tag}"
            )
            by_load = "  ".join(f"{k} {v * 1e3:.3f}" for k, v in res.u_tip_by_load_m.items())
            print(f"             u_tip(mm) by load: {by_load}")

    summary = rescreen_blade_to_batch(
        [(name, params) for name, params, _ in designs],
        out_dir,
        mesh_params=FeaMeshParams(mesh_size_m=mesh_size_m),
        skin_thickness_m=skin_thickness_m,
        n_workers=n_workers,
        on_result=_log,
        j_fan_by_name={name: j3d for name, _params, j3d in designs},
    )
    if progress and designs and summary["n_succeeded"] == 0:
        # Every design failed — almost always a wrong --out-dir (no <name>_density.npy there).
        print(
            f"\n!! 0/{len(designs)} re-screened. Check that {out_dir} holds the TO run's <name>_density.npy "
            f"files (this must be the SAME --out-dir the TO wrote to)."
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-dir", type=Path, required=True, help="campaign shard dir")
    parser.add_argument("--verification", type=Path, required=True, help="Stage-3.C verification.json")
    parser.add_argument("--out-dir", type=Path, default=Path("data/stage4_blade_to"),
                        help="dir holding the saved <name>_density.npy from the TO run")
    parser.add_argument("--top-k", type=int, default=4, help="print candidates to re-screen (by fine J_fan)")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="designs to re-screen in parallel (RAM-bound: ~10-20 GB/design at 0.6mm)")
    parser.add_argument("--mesh-size-m", type=float, default=None, help="override; default from summary.json")
    parser.add_argument("--skin-thickness-m", type=float, default=None, help="override; default from summary.json")
    parser.add_argument("--stress-fos", type=float, default=DEFAULT_STRESS_FOS)
    args = parser.parse_args(argv)

    summary = run(
        shared_dir=args.shared_dir,
        verification=args.verification,
        out_dir=args.out_dir,
        top_k=args.top_k,
        n_workers=args.n_workers,
        mesh_size_m=args.mesh_size_m,
        skin_thickness_m=args.skin_thickness_m,
        stress_fos=args.stress_fos,
    )
    print(
        f"\nDone: re-screened {summary['n_succeeded']}/{summary['n_designs']} designs "
        f"-> {args.out_dir}/rescreen_summary.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
