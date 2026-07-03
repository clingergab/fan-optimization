"""Phase 4 results analysis — campaign checkpoint/ledger → Pareto + top-k diverse.

Reads a finished (or in-progress) campaign directory and reconstructs the Pareto
front over the three objectives (J_fan ↑, I_wrist ↓, panel deflection ↓), then
picks the ``k`` most **structurally diverse** Pareto designs for Phase-5 printing
(greedy max-min spread in the normalized design vector — the "don't print 3
variations of one shape" rule). Botorch-free (numpy non-dominated sort) so results
can be analyzed anywhere the checkpoint lands.
"""

from __future__ import annotations

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
