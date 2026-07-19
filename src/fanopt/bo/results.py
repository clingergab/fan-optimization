"""Phase 4 results analysis — campaign checkpoint/ledger → Pareto + top-k diverse.

Reads a finished (or in-progress) campaign directory and reconstructs the Pareto
front over the three objectives (J_fan ↑, I_wrist ↓, panel deflection ↓), then
picks the ``k`` most **structurally diverse** Pareto designs for Phase-5 printing
(greedy max-min spread in the normalized design vector — the "don't print 3
variations of one shape" rule). Botorch-free (numpy non-dominated sort) so results
can be analyzed anywhere the checkpoint lands.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fanopt.bo.codec import bounds, decode
from fanopt.utils.ledger import read_rows

__all__ = [
    "OBJECTIVE_SIGNS",
    "CampaignData",
    "non_dominated_mask",
    "load_campaign",
    "pareto_designs",
    "select_diverse",
    "analyze",
    "recommend",
]

# (J_fan, I_wrist, structural): +1 maximize, -1 minimize (matches backbone).
OBJECTIVE_SIGNS: tuple[float, float, float] = (1.0, -1.0, -1.0)
# Campaign artifact names (match fanopt.bo.orchestration; kept local so results
# analysis carries no botorch dependency).
CHECKPOINT_NAME = "checkpoint.npz"
LEDGER_NAME = "evaluations.jsonl"


@dataclass(frozen=True)
class CampaignData:
    """Everything a campaign dir holds: design vectors, raw objectives, ledger rows."""

    x: np.ndarray  # (n, n_dims)
    y_raw: np.ndarray  # (n, 3) — (J_fan, I_wrist, structural)
    ledger_rows: list[dict]


def non_dominated_mask(y_max: np.ndarray) -> np.ndarray:
    """Boolean mask of Pareto-optimal rows (all objectives *maximized*)."""
    y = np.atleast_2d(np.asarray(y_max, dtype=float))
    n = y.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i: j >= i on every objective and strictly better on one.
            if np.all(y[j] >= y[i]) and np.any(y[j] > y[i]):
                dominated[i] = True
                break
    return ~dominated


def load_campaign(out_dir: str | Path) -> CampaignData:
    """Load the checkpoint (design vectors + objectives) + JSONL ledger."""
    out_dir = Path(out_dir)
    ckpt = np.load(out_dir / CHECKPOINT_NAME)
    ledger_path = out_dir / LEDGER_NAME
    rows = read_rows(ledger_path) if ledger_path.exists() else []
    return CampaignData(x=ckpt["x"], y_raw=ckpt["y_raw"], ledger_rows=rows)


def pareto_designs(x: np.ndarray, y_raw: np.ndarray) -> list[dict[str, Any]]:
    """Non-dominated designs with decoded params + objectives (most-airflow first)."""
    y_max = np.asarray(y_raw, dtype=float) * np.asarray(OBJECTIVE_SIGNS)
    idx = np.where(non_dominated_mask(y_max))[0]
    idx = idx[np.argsort(-y_raw[idx, 0])]  # J_fan descending
    out: list[dict[str, Any]] = []
    for i in idx:
        j_fan, i_wrist, structural = y_raw[i]
        params = decode(x[i])
        out.append(
            {
                "index": int(i),
                "vector": x[i].tolist(),
                "j_fan": float(j_fan),
                "i_wrist_kgm2": float(i_wrist),
                "structural_m": float(structural),
                "blade_count": params.blade_count,
                "edge_profile": params.edge_profile,
            }
        )
    return out


def select_diverse(x: np.ndarray, indices: list[int], k: int) -> list[int]:
    """Pick ``k`` maximally-spread designs from ``indices`` (greedy max-min).

    Distances are in the bounds-normalized vector space, so no single axis (e.g.
    thickness scale) dominates the spread. Seeds from the most off-centre design.
    """
    if k <= 0 or not indices:
        return []
    if len(indices) <= k:
        return list(indices)
    low, high = bounds()
    span = np.where(high > low, high - low, 1.0)
    xn = (x[indices] - low) / span
    centre = xn.mean(axis=0)
    chosen = [int(np.argmax(np.linalg.norm(xn - centre, axis=1)))]
    while len(chosen) < k:
        dmin = np.min(np.stack([np.linalg.norm(xn - xn[c], axis=1) for c in chosen]), axis=0)
        dmin[chosen] = -1.0
        chosen.append(int(np.argmax(dmin)))
    return [indices[c] for c in chosen]


def analyze(out_dir: str | Path, *, top_k: int = 3) -> dict[str, Any]:
    """Full campaign summary: Pareto front + the ``top_k`` diverse print picks."""
    data = load_campaign(out_dir)
    pareto = pareto_designs(data.x, data.y_raw)
    y_max = np.asarray(data.y_raw, dtype=float) * np.asarray(OBJECTIVE_SIGNS)
    pareto_idx = [int(i) for i in np.where(non_dominated_mask(y_max))[0]]
    diverse_idx = set(select_diverse(data.x, pareto_idx, top_k))
    for d in pareto:
        d["diverse_pick"] = d["index"] in diverse_idx
    return {
        "n_evaluations": int(data.x.shape[0]),
        "n_pareto": len(pareto),
        "top_k_diverse": [d for d in pareto if d["diverse_pick"]],
        "pareto": pareto,
    }


def _verification_by_index(verification_path: str | Path | None) -> dict[int, dict[str, Any]]:
    """Map a Phase-5 ``verification.json``'s designs by campaign index (name ``b{n}_i{idx}``)."""
    if verification_path is None or not Path(verification_path).exists():
        return {}
    ver = json.loads(Path(verification_path).read_text(encoding="utf-8"))
    out: dict[int, dict[str, Any]] = {}
    for d in ver.get("designs", []):
        try:
            idx = int(str(d["name"]).split("_i")[-1])
        except (KeyError, ValueError):  # pragma: no cover - malformed name
            continue
        out[idx] = d
    return out


def recommend(
    out_dir: str | Path, *, top_k: int = 3, verification_path: str | Path | None = None
) -> dict[str, Any]:
    """Consolidated print recommendation: top-k diverse Pareto + 3D verification.

    Merges the Phase-4 Pareto (``analyze``'s structurally-diverse picks) with a
    Phase-5 ``verification.json`` (if present) so each recommended design shows its
    2D-slice **and** 3D ``J_fan`` side by side, flagged ``verified``. The 3–5
    designs to print for the Phase-6 blinded A/B test. Works before verification
    exists (``verification`` = ``"absent"``, 3D fields ``None``).
    """
    summary = analyze(out_dir, top_k=top_k)
    ver_by_idx = _verification_by_index(verification_path)
    diverse_idx = {int(r["index"]) for r in summary["top_k_diverse"]}
    pareto_by_idx = {int(d["index"]): d for d in summary["pareto"]}

    recommended: list[dict[str, Any]] = []
    for r in summary["top_k_diverse"]:
        vr = ver_by_idx.get(int(r["index"]))
        j3d = vr.get("j_fan_3d") if vr else None
        recommended.append(
            {
                "index": r["index"],
                "blade_count": r["blade_count"],
                "edge_profile": r["edge_profile"],
                "j_fan_slice": r["j_fan"],
                "j_fan_3d": j3d,
                "i_wrist_kgm2": r["i_wrist_kgm2"],
                "structural_m": r["structural_m"],
                "verified": vr is not None and j3d is not None and bool(np.isfinite(j3d)),
                "vector": r["vector"],
            }
        )

    # Every verified design, ranked by the high-fidelity 3D J_fan (valid first,
    # suspects/failed last), so the full picture is visible — not just the print-3.
    ranked: list[dict[str, Any]] = []
    for idx, vr in ver_by_idx.items():
        p = pareto_by_idx.get(idx, {})
        raw = vr.get("j_fan_3d")
        jf3 = float(raw) if isinstance(raw, int | float) and bool(np.isfinite(raw)) else None
        ranked.append(
            {
                "index": idx,
                "name": vr.get("name", f"i{idx}"),
                "blade_count": p.get("blade_count"),
                "edge_profile": p.get("edge_profile"),
                "j_fan_slice": vr.get("j_fan_slice", p.get("j_fan")),
                "j_fan_3d": jf3,
                "i_wrist_kgm2": p.get("i_wrist_kgm2"),
                "structural_m": p.get("structural_m"),
                "verified": jf3 is not None and jf3 > 0,
                "suspect": jf3 is None or jf3 <= 0,  # failed 3D run or net reverse thrust
                "recommended_for_print": idx in diverse_idx,
            }
        )
    ranked.sort(key=lambda d: (d["j_fan_3d"] is not None, d["j_fan_3d"] or 0.0), reverse=True)

    return {
        "top_k": top_k,
        "n_pareto": summary["n_pareto"],
        "verification": "present" if ver_by_idx else "absent",
        "n_verified": sum(1 for d in recommended if d["verified"]),
        "recommended": recommended,
        "ranked": ranked,
    }
