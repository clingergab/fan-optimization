#!/usr/bin/env python
"""Run the aero-first Phase-4 BO campaign for the redesigned blade.

Thin CLI wrapper: build the CFD objective (:class:`BladeObjective`) + campaign config,
run (or resume) the BO loop, and write the Pareto set to ``<out-dir>/pareto.json``.
Designed to be launched from the Colab full-phase notebook or the terminal:

    python scripts/run_phase4_aero.py --out-dir data/phase4_aero \\
        --n-init 16 --n-iterations 60 --n-workers 4 --su2-bin $SU2_RUN/SU2_CFD
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fanopt.bo.blade_campaign import CampaignConfig, CampaignState, pareto_designs, run_campaign
from fanopt.bo.blade_objective import BladeObjective


def run(
    out_dir: Path,
    *,
    su2_bin: str | None = None,
    cfg: CampaignConfig | None = None,
    radial_u: float = 0.5,
    n_panels: int = 5,
    n_cycles: int = 5,
    scratch_dir: Path | None = None,
    objective_fn=None,
    progress: bool = False,
) -> CampaignState:
    """Build the objective (or use an injected one) and run/resume the campaign.

    Campaign *state* (``checkpoint.npz`` + ledger + ``pareto.json``) is small and
    essential — persist it via ``out_dir`` (e.g. Google Drive, so a disconnect resumes).
    The per-design SU2 *scratch* (thousands of files/eval, regenerable, never reused)
    goes to ``scratch_dir`` if given (e.g. a local ephemeral disk), else ``out_dir``.
    """
    out_dir = Path(out_dir)
    obj = objective_fn or BladeObjective(
        out_dir=Path(scratch_dir) if scratch_dir is not None else out_dir,
        su2_bin=su2_bin, radial_u=radial_u, n_panels=n_panels, n_cycles=n_cycles,
    )
    state = run_campaign(obj, out_dir, cfg or CampaignConfig(), progress=progress)
    (out_dir / "pareto.json").write_text(
        json.dumps(pareto_designs(state), indent=2), encoding="utf-8"
    )
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/phase4_aero"))
    parser.add_argument(
        "--scratch-dir", type=Path, default=None,
        help="local/ephemeral dir for SU2 per-design scratch (else --out-dir)",
    )
    parser.add_argument(
        "--su2-bin", type=str, default=None, help="SU2_CFD path (else $SU2_RUN/PATH)"
    )
    parser.add_argument("--n-init", type=int, default=16)
    parser.add_argument("--n-iterations", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--radial-u", type=float, default=0.5)
    parser.add_argument("--n-panels", type=int, default=5)
    parser.add_argument("--n-cycles", type=int, default=5)
    args = parser.parse_args(argv)

    cfg = CampaignConfig(
        n_init=args.n_init,
        n_iterations=args.n_iterations,
        batch_size=args.batch_size,
        seed=args.seed,
        n_workers=args.n_workers,
    )
    state = run(
        args.out_dir,
        su2_bin=args.su2_bin,
        cfg=cfg,
        radial_u=args.radial_u,
        n_panels=args.n_panels,
        n_cycles=args.n_cycles,
        scratch_dir=args.scratch_dir,
        progress=True,
    )
    pareto = pareto_designs(state)
    print(
        f"[phase4-aero] {state.x.shape[0]} evals | {len(pareto)} Pareto designs | "
        f"best J_fan = {state.y_raw[:, 0].max():.3e} | wrote {args.out_dir / 'pareto.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
