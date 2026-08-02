"""Stage-3 (distributed-shard campaign) V1-pick consolidator — the shard-keyed counterpart of
:mod:`fanopt.bo.results`.

The Stage-3 async campaign writes per-session Drive shards (``evaluations_*.jsonl`` — each row a
33-D ``vector`` + raw objectives ``(J_fan↑, mass↓, deflection↓)`` + a ``design_hash``), NOT the
Phase-4 ``checkpoint.npz`` that :mod:`fanopt.bo.results` reads, and identifies designs by
``design_hash`` rather than an integer campaign index. This module reads those shards →
non-dominated Pareto front → top-k structurally-diverse picks → joins the Stage-3.C
``verification.json`` (fine 3D ``J_fan``, whose design names are ``{rank:02d}_{design_hash}``) **by
design_hash** → the ranked V1-print recommendation.

Kept cadquery/torch-free (like :mod:`fanopt.bo.results`, whose generic non-dominated sort + diversity
spread it reuses) so the pick can be computed anywhere the shards land — no CFD/BO deps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fanopt.bo.blade_codec import N_DIMS, bounds, decode
from fanopt.bo.results import non_dominated_mask, select_diverse

__all__ = [
    "SHARD_GLOB",
    "load_shard_rows",
    "blade_pareto_designs",
    "recommend_blades",
]

SHARD_GLOB = "evaluations_*.jsonl"
# The Stage-3 objective 3-tuple's maximize/minimize signs: (J_fan↑, mass↓, deflection↓), matching
# blade_objective_3d's return order. Defined locally (not imported from results.OBJECTIVE_SIGNS,
# whose tuple is labelled for the Phase-4 J_fan/I_wrist/structural objectives) so re-signing that
# campaign can never silently invert the blade Pareto — the two just happen to be (+1,-1,-1) today.
_BLADE_OBJECTIVE_SIGNS: tuple[float, float, float] = (1.0, -1.0, -1.0)


def _rib_mode(uniform: bool) -> str:
    return "uniform" if uniform else "ribbed"


def load_shard_rows(shared_dir: Path | str) -> list[dict[str, Any]]:
    """Read + dedup the campaign's Drive shards, keeping only well-formed, finite-objective rows.

    Mirrors ``distributed_campaign._iter_rows``: drops a row unless it has a length-``N_DIMS`` numeric
    ``vector`` and finite ``j_fan`` / ``mass_kg`` / ``deflection_m`` — so a torn/clobbered concurrent
    Drive write (schema drift, short vector, missing objective) can't poison the Pareto or crash a
    downstream ``decode``. Deduped by ``design_hash`` (first occurrence wins).
    """
    by_hash: dict[object, dict[str, Any]] = {}
    for shard in sorted(Path(shared_dir).glob(SHARD_GLOB)):
        for line in shard.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            vec = row.get("vector")
            if not isinstance(vec, list) or len(vec) != N_DIMS:
                continue
            try:
                [float(v) for v in vec]
                objectives = [float(row["j_fan"]), float(row["mass_kg"]), float(row["deflection_m"])]
            except (KeyError, TypeError, ValueError):
                continue
            if not all(np.isfinite(objectives)):
                continue  # a failed/infeasible design (NaN penalty) — not a real candidate
            by_hash.setdefault(row.get("design_hash"), row)
    return list(by_hash.values())


def blade_pareto_designs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Non-dominated shard designs over ``(J_fan↑, mass↓, deflection↓)``, most-airflow first.

    Each design carries the decoded blade summary (rib mode, blade_count, rib thickness), its raw
    objectives, its ``vector``, and its ``design_hash`` (the join key to the 3.C verification). The
    ``index`` is the Pareto rank position (0 = most airflow), a stable display id since the shard
    campaign has no integer campaign index.
    """
    if not rows:
        return []
    x = np.array([[float(v) for v in r["vector"]] for r in rows], dtype=float)
    y_raw = np.array(
        [[float(r["j_fan"]), float(r["mass_kg"]), float(r["deflection_m"])] for r in rows],
        dtype=float,
    )
    y_max = y_raw * np.asarray(_BLADE_OBJECTIVE_SIGNS)
    idx = np.where(non_dominated_mask(y_max))[0]
    idx = idx[np.argsort(-y_raw[idx, 0])]  # J_fan descending
    out: list[dict[str, Any]] = []
    for rank, i in enumerate(idx):
        j_fan, mass_kg, deflection_m = y_raw[i]
        params = decode(x[i])
        out.append(
            {
                "index": rank,
                "design_hash": rows[i].get("design_hash"),
                "vector": x[i].tolist(),
                "j_fan": float(j_fan),
                "mass_kg": float(mass_kg),
                "deflection_m": float(deflection_m),
                "blade_count": params.blade_count,
                "rib_mode": _rib_mode(params.uniform),
            }
        )
    return out


def _verification_by_hash(verification_path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Map a Stage-3.C ``verification.json``'s designs by ``design_hash`` (name ``{rank}_{hash}``)."""
    if verification_path is None or not Path(verification_path).exists():
        return {}
    ver = json.loads(Path(verification_path).read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for d in ver.get("designs", []):
        name = str(d.get("name", ""))
        if "_" not in name:  # 3.C names are "{rank:02d}_{design_hash}"; skip anything else
            continue
        out[name.split("_", 1)[1]] = d
    return out


def _finite_j3d(vr: dict[str, Any] | None) -> float | None:
    raw = vr.get("j_fan_3d") if vr else None
    return float(raw) if isinstance(raw, (int, float)) and bool(np.isfinite(raw)) else None


def recommend_blades(
    shared_dir: str | Path,
    *,
    top_k: int = 3,
    verification_path: str | Path | None = None,
) -> dict[str, Any]:
    """Consolidate the shard campaign's Pareto + the 3.C ``verification.json`` into the V1 print pick.

    Reads the campaign shards under ``shared_dir``, builds the ``(J_fan↑, mass↓, deflection↓)`` Pareto,
    picks the ``top_k`` structurally-diverse designs (the print set), and — if a ``verification.json``
    is given — joins each design's **fine whole-fan 3D** ``J_fan`` **by design_hash** (matching the
    ``{rank}_{hash}`` verify names). Returns ``recommended`` (the diverse print picks) and ``ranked``
    (every 3D-verified design, best fine ``J_fan`` first, with mass/deflection context) — the same
    shape :mod:`fanopt.bo.results`'s ``recommend`` emits so one CLI printer serves both campaigns.
    A negative fine ``J_fan`` is a real (bad) result → ``verified``; only a failed/non-finite run is
    ``suspect``.
    """
    rows = load_shard_rows(shared_dir)
    pareto = blade_pareto_designs(rows)
    ver_by_hash = _verification_by_hash(verification_path)

    x_pareto = (
        np.array([[float(v) for v in d["vector"]] for d in pareto], dtype=float)
        if pareto
        else np.zeros((0, N_DIMS))
    )
    diverse_pos = set(select_diverse(x_pareto, list(range(len(pareto))), top_k, span_bounds=bounds()))
    diverse_hashes = {pareto[i]["design_hash"] for i in diverse_pos}

    recommended: list[dict[str, Any]] = []
    for d in pareto:
        if d["index"] not in diverse_pos:
            continue
        vr = ver_by_hash.get(d["design_hash"])
        j3d = _finite_j3d(vr)
        recommended.append(
            {
                "index": d["index"],
                "design_hash": d["design_hash"],
                "blade_count": d["blade_count"],
                "rib_mode": d["rib_mode"],
                "j_fan_coarse": d["j_fan"],
                "j_fan_3d": j3d,
                "mass_kg": d["mass_kg"],
                "deflection_m": d["deflection_m"],
                "verified": j3d is not None,
                "vector": d["vector"],
            }
        )

    # Every 3D-verified design, ranked by the fine whole-fan J_fan (finite first, failed last), with
    # its campaign metrics joined by hash — the full picture, not just the print-k.
    by_hash = {d["design_hash"]: d for d in pareto}
    all_rows_by_hash = {r.get("design_hash"): r for r in rows}
    ranked: list[dict[str, Any]] = []
    for h, vr in ver_by_hash.items():
        p = by_hash.get(h)
        shard = all_rows_by_hash.get(h)
        params = decode(np.asarray(shard["vector"], dtype=float)) if shard else None
        j3d = _finite_j3d(vr)
        ranked.append(
            {
                "name": vr.get("name", h),
                "design_hash": h,
                "blade_count": (p or {}).get("blade_count")
                or (params.blade_count if params else None),
                "rib_mode": (p or {}).get("rib_mode")
                or (_rib_mode(params.uniform) if params else None),
                "j_fan_coarse": vr.get("j_fan_coarse", (shard or {}).get("j_fan")),
                "j_fan_3d": j3d,
                "mass_kg": (p or {}).get("mass_kg") or (shard or {}).get("mass_kg"),
                "deflection_m": (p or {}).get("deflection_m") or (shard or {}).get("deflection_m"),
                "verified": j3d is not None,  # finite fine value (negative is a real, bad result)
                "suspect": j3d is None,  # only a failed/non-finite fine run is suspect
                "recommended_for_print": h in diverse_hashes,
            }
        )
    ranked.sort(key=lambda d: (d["j_fan_3d"] is not None, d["j_fan_3d"] or 0.0), reverse=True)
    if ver_by_hash:
        # Verification is in → promote the top-`top_k` by FINE whole-fan J_fan (the aero truth this
        # stage exists to establish), NOT the pre-verify coarse-Pareto diversity: these are the
        # designs to carry into TO. (Diversity is the criterion for the FINAL print set, after TO.)
        promote = set([r["design_hash"] for r in ranked if r["verified"]][:top_k])
        for r in ranked:
            r["recommended_for_print"] = r["design_hash"] in promote

    return {
        "top_k": top_k,
        "n_evaluations": len(rows),
        "n_pareto": len(pareto),
        "verification": "present" if ver_by_hash else "absent",
        "n_verified": sum(1 for r in ranked if r["verified"]),
        "recommended": recommended,
        "ranked": ranked,
    }
