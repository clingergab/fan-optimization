#!/usr/bin/env python
"""Stage 3.C — fine-tier 3D verification of the aero-first campaign's top designs.

Reads the finished distributed Stage-3 campaign's Drive shards (``evaluations_*.jsonl`` under
``--shared-dir``, e.g. the ``campaign_trapezoid`` folder — the async campaign writes NO
``pareto.json``), picks the top-k designs by their **coarse** ``J_fan``, decodes each stored 33-D
vector to its blade, and re-evaluates it at the **fine** 3D tier (default 5 cycles / 60 inner-iter,
vs the campaign's coarse 3/30). Then checks the coarse ranking the campaign screened on survives the
higher fidelity (Kendall τ between coarse and fine ``J_fan``; ≥ 0.7 ⇒ the top-by-coarse designs are
the top-by-fine designs → the V1 pick comes from the fine ranking). Writes ``verification.json``,
rewritten after each design so a mid-run disconnect keeps progress.

    export SU2_RUN="$HOME/su2-local/extracted/bin"
    python scripts/run_phase5_verify_blade.py \\
        --shared-dir /content/drive/MyDrive/fanopt/campaign_trapezoid --top-k 10 --workers 10

Fine runs are expensive (~10-12h each) but parallelize across designs (``--workers`` ≈ min(top_k,
cores)); geometry + meshing + cfg are local. Pass ``--pareto`` instead for a legacy pareto.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from fanopt.cfd.blade_verify import load_campaign_rows, load_pareto, verify_blades
from fanopt.cfd.phase5 import FINE_CYCLES, FINE_INNER, VerifyConfig, VerifyResult, verify_ranking


def _write_verification(path: Path, summary: dict[str, Any]) -> None:
    """Write ``verification.json`` ATOMICALLY: dump to a ``.tmp`` sibling, then ``os.replace`` it in.

    An in-place ``write_text`` truncates the file first, so a crash mid-write (a Colab drop over the
    ~day-long run) would leave a half-written file that a resumed run can't parse — losing every
    completed design. Writing the full temp then swapping means the live file is always a complete
    prior version or the complete new one; if the swap itself is interrupted, the intact ``.tmp`` is
    recovered by :func:`_prior_results`.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _summary(results: list[VerifyResult]) -> dict[str, Any]:
    return {
        "ranking": verify_ranking(results),
        "designs": [
            {
                "name": r.name,
                "j_fan_3d": r.j_fan_3d,
                "j_fan_coarse": r.j_fan_coarse,
                "n_nodes": r.meta.get("n_nodes"),
            }
            for r in results
        ],
    }


def _prior_results(verification_path: Path) -> tuple[list[VerifyResult], set[str]]:
    """Already-verified designs (finite fine ``J_fan``) from an existing ``verification.json``, as
    ``(results, done_hashes)`` — so a resumed run keeps them and re-verifies only the rest. Only
    finite results are kept; a prior failed/non-finite run is retried (its hash is not in the skip
    set). ``done_hashes`` are the ``design_hash`` suffix of the ``{rank}_{hash}`` design names.
    """
    # Try the live file, then the ``.tmp`` an interrupted atomic swap may have left — and NEVER crash
    # on a corrupt/half-written file (that would abort the whole resume). A parse failure falls back
    # to the next candidate, and if none parses we return empty (re-verify all — wasteful but safe),
    # rather than raising and forcing the operator to delete everything.
    for candidate in (verification_path, verification_path.with_name(verification_path.name + ".tmp")):
        try:
            designs = json.loads(candidate.read_text(encoding="utf-8")).get("designs", [])
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue
        prior: list[VerifyResult] = []
        done: set[str] = set()
        for d in designs:
            jf3 = d.get("j_fan_3d")
            if isinstance(jf3, (int, float)) and math.isfinite(jf3):
                done.add(str(d.get("name", "")).split("_", 1)[-1])
                prior.append(
                    VerifyResult(
                        name=str(d["name"]),
                        j_fan_3d=float(jf3),
                        j_fan_coarse=d.get("j_fan_coarse"),
                        meta={"n_nodes": float(d.get("n_nodes") or 0.0)},
                    )
                )
        return prior, done
    return [], set()


def run(
    *,
    out_dir: Path,
    top_k: int | None,
    shared_dir: Path | None = None,
    pareto_path: Path | None = None,
    su2_bin: str | None = None,
    cfg: VerifyConfig | None = None,
    n_workers: int = 1,
    progress: bool = True,
    resume: bool = True,
) -> dict[str, object]:
    """Fine-verify the top-k blades and write ``verification.json``; return the summary.

    Reads the campaign's top designs from ``shared_dir`` (Drive shards) or ``pareto_path`` (legacy).
    ``resume`` (default): keep the finite results already in ``out_dir/verification.json`` and
    re-verify only the not-yet-done designs — so a Colab drop during the multi-hour fine run resumes
    instead of restarting from scratch. Each design is also checkpointed as it completes.
    """
    top_k = top_k or None  # 0 (or None) → verify ALL designs, matching the CLI's "0 = all"
    if shared_dir is not None:
        records = load_campaign_rows(shared_dir)
    elif pareto_path is not None:
        records = load_pareto(pareto_path)
    else:
        raise ValueError("provide shared_dir (campaign shards) or pareto_path")
    if not records:
        raise ValueError(
            "no finite-J_fan designs found — is the campaign folder correct / has it produced evals?"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    prior, skip = _prior_results(out_dir / "verification.json") if resume else ([], set())
    done: list[VerifyResult] = list(prior)  # resumed run keeps the already-verified designs

    def _checkpoint(r: VerifyResult) -> None:
        done.append(r)
        _write_verification(out_dir / "verification.json", _summary(done))

    verify_blades(
        records,
        out_dir,
        top_k=top_k,
        cfg=cfg or VerifyConfig.fine(),
        su2_bin=su2_bin,
        n_workers=n_workers,
        progress=progress,
        on_result=_checkpoint,
        skip_hashes=skip,
    )
    summary = _summary(done)  # prior (resumed) + newly-verified
    _write_verification(out_dir / "verification.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shared-dir", type=Path, default=None,
        help="Campaign folder with evaluations_*.jsonl shards (the async campaign's Drive output).",
    )
    parser.add_argument(
        "--pareto", type=Path, default=None, help="Legacy pareto.json (alternative to --shared-dir)."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/phase5_verify_blade"))
    parser.add_argument(
        "--top-k", type=int, default=10, help="Top designs by coarse J_fan (0 = all)."
    )
    parser.add_argument(
        "--n-cycles", type=int, default=FINE_CYCLES, help=f"Fine-tier cycles (default {FINE_CYCLES})."
    )
    parser.add_argument(
        "--inner-iter", type=int, default=FINE_INNER, help=f"Fine-tier inner-iter (default {FINE_INNER})."
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Parallel designs (processes); ~ min(top_k, cores)."
    )
    parser.add_argument("--su2-bin", default=None, help="Path to SU2_CFD (default: $SU2_RUN/PATH)")
    parser.add_argument("--no-progress", action="store_true", help="Disable the tqdm progress bar.")
    args = parser.parse_args(argv)
    if args.shared_dir is None and args.pareto is None:
        parser.error("provide --shared-dir (campaign shards) or --pareto (legacy pareto.json)")

    summary = run(
        out_dir=args.out_dir,
        top_k=args.top_k,  # run() normalizes 0 → None (all)
        shared_dir=args.shared_dir,
        pareto_path=args.pareto,
        su2_bin=args.su2_bin,
        cfg=VerifyConfig(n_cycles=args.n_cycles, inner_iter=args.inner_iter),
        n_workers=args.workers,
        progress=not args.no_progress,
    )
    ranking: dict[str, Any] = summary["ranking"]  # type: ignore[assignment]
    print(json.dumps(ranking, indent=2))
    valid = ranking["valid_only"]
    suspects = ranking["suspect_designs"]
    print(
        f"[phase5-blade] verified {len(summary['designs'])} designs → "  # type: ignore[arg-type]
        f"rank_preserved={ranking['rank_preserved']} "
        f"(valid n={valid['n']}: τ={valid['kendall_tau']}, ρ={valid['spearman_rho']}, "
        f"R²={valid['pearson_r2']})"
    )
    if suspects:
        print(
            f"[phase5-blade] {ranking['n_suspect']} suspect "
            f"(negative/failed 3D J_fan): {suspects}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
