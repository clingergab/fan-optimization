"""Post-hoc analysis of a blade-campaign ledger (pooled across shards / folders).

Pure/numpy-only (no botorch) so it runs anywhere. Reads the JSONL shards a campaign wrote,
dedups by ``design_hash``, and reports objective stats, the optimization progression (running-best
J_fan in evaluation order — the signal for "is it actually learning or random?"), the Pareto
front, and the best designs. Also surfaces how the data is split across folders/shards, which is
how we detect the Drive duplicate-folder coordination failure.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

__all__ = ["load_rows", "pareto_indices", "campaign_report", "find_shards"]


def load_rows(shard_files: list[str | Path]) -> list[dict]:
    """Parse every JSONL row from the given shard files (skips blank / torn lines)."""
    rows: list[dict] = []
    for f in shard_files:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def pareto_indices(objectives: np.ndarray) -> list[int]:
    """Indices of the non-dominated rows. ``objectives`` columns = (J_fan↑, mass↓, deflection↓)."""
    if len(objectives) == 0:
        return []
    m = objectives.copy()
    m[:, 1:] = -m[:, 1:]  # to a pure-maximization frame
    keep: list[int] = []
    for i in range(len(m)):
        dominated = np.any(np.all(m >= m[i], axis=1) & np.any(m > m[i], axis=1))
        if not dominated:
            keep.append(i)
    return keep


def campaign_report(shard_files: list[str | Path], *, top_k: int = 10) -> dict:
    """Pool + dedup the shards and summarize the campaign (see module docstring)."""
    rows = load_rows(shard_files)
    by_hash: dict[str, dict] = {}
    for r in rows:
        by_hash.setdefault(r.get("design_hash", id(r)), r)
    uniq = list(by_hash.values())
    finite = [
        r for r in uniq if isinstance(r.get("j_fan"), (int | float)) and np.isfinite(r["j_fan"])
    ]

    report: dict = {
        "total_rows": len(rows),
        "unique_designs": len(uniq),
        "finite": len(finite),
        "failed_nan": len(uniq) - len(finite),
        "duplicate_rows": len(rows) - len(uniq),
        "sources": {
            s: sum(1 for r in finite if r.get("source") == s)
            for s in sorted({r.get("source") for r in finite})
        },
    }
    if not finite:
        return report

    j = np.array([r["j_fan"] for r in finite])
    mass = np.array([r["mass_kg"] for r in finite])
    defl = np.array([r.get("deflection_m", float("nan")) for r in finite])
    report["j_fan"] = {"min": float(j.min()), "mean": float(j.mean()), "max": float(j.max())}
    report["mass_g"] = {
        "min": float(mass.min() * 1e3),
        "mean": float(mass.mean() * 1e3),
        "max": float(mass.max() * 1e3),
    }

    # progression: running-best J_fan in EVALUATION ORDER (by timestamp) — is it learning or flat?
    order = sorted(range(len(finite)), key=lambda i: finite[i].get("timestamp_iso", ""))
    running_best: list[float] = []
    best = -np.inf
    improvements = 0
    for i in order:
        if j[i] > best:
            best = float(j[i])
            improvements += 1
        running_best.append(best)
    report["progression"] = {
        "running_best": running_best,  # by evaluation order
        "n_new_bests": improvements,
        "first_best": running_best[0] if running_best else None,
        "final_best": running_best[-1] if running_best else None,
    }
    # did BO beat the random DoE? (best among sobol vs best among bo)
    sob = [r["j_fan"] for r in finite if r.get("source") == "sobol"]
    bo = [r["j_fan"] for r in finite if r.get("source") == "bo"]
    report["sobol_best"] = max(sob) if sob else None
    report["bo_best"] = max(bo) if bo else None

    obj = np.column_stack([j, mass, defl])
    pf = pareto_indices(obj)
    report["pareto_count"] = len(pf)
    report["pareto"] = sorted(
        (
            {
                "j_fan": float(j[i]),
                "mass_g": float(mass[i] * 1e3),
                "deflection_mm": float(defl[i] * 1e3),
                "blade_count": finite[i].get("blade_count"),
                "design_hash": finite[i].get("design_hash", "")[:8],
            }
            for i in pf
        ),
        key=lambda d: -d["j_fan"],
    )
    report["top_by_j_fan"] = sorted(
        (
            {
                "j_fan": float(j[i]),
                "mass_g": float(mass[i] * 1e3),
                "source": finite[i].get("source"),
                "design_hash": finite[i].get("design_hash", "")[:8],
            }
            for i in range(len(finite))
        ),
        key=lambda d: -d["j_fan"],
    )[:top_k]
    return report


def find_shards(root: str | Path, pattern: str = "blade_campaign*") -> dict[str, list[str]]:
    """Map each campaign folder under ``root`` to its ledger shards — exposes duplicate folders."""
    out: dict[str, list[str]] = {}
    for d in sorted(glob.glob(str(Path(root) / pattern))):
        shards = sorted(glob.glob(str(Path(d) / "evaluations_*.jsonl")))
        if shards:
            out[d] = shards
    return out
