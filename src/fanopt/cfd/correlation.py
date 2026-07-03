"""Steady↔unsteady correlation gate (report-final.md §Phase 3).

Compares the cheap steady drag proxy against the true 2D-unsteady result across
a set of designs and reports the correlation. The gate: if the steady proxy
predicts the unsteady ranking well (R² ≥ threshold), the cheap tier is
trustworthy for screening; in V1-Slim this number also informs the
adjoint-vs-BO backbone choice (plan_v1_slim_latest.md §4).

Correlation is **scale-invariant** (Pearson R² is invariant to linear scaling;
Kendall τ is rank-based) — which matters because the steady proxy (MACH=0.0064,
CD ~ O(10)) and the unsteady result (MACH=1e-9, force ~ O(1e14) under the
FREESTREAM_PRESS_EQ_ONE convention) live on wildly different scales.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "R2_RETENTION_THRESHOLD",
    "CorrelationResult",
    "pearson_r2",
    "kendall_tau",
    "correlate",
]

R2_RETENTION_THRESHOLD: float = 0.4
"""§Phase 3: R² ≥ 0.4 retains the cheap steady tier as a screening fidelity."""


@dataclass(frozen=True)
class CorrelationResult:
    r2: float
    pearson_r: float
    kendall_tau: float
    n: int
    passed: bool  # r2 >= R2_RETENTION_THRESHOLD
    meta: dict[str, float]


def pearson_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Coefficient of determination R² = (Pearson r)²."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"x, y must be equal-length 1D arrays; got {x.shape}, {y.shape}")
    if x.size < 2:
        raise ValueError("need at least 2 points to correlate")
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0  # a constant series has no linear correlation
    r = float(np.corrcoef(x, y)[0, 1])
    return r * r


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall rank correlation τ (scale-free ranking agreement)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        raise ValueError("need at least 2 points to correlate")
    tau = stats.kendalltau(x, y).correlation
    return 0.0 if np.isnan(tau) else float(tau)


def correlate(
    steady: np.ndarray, unsteady: np.ndarray, *, threshold: float = R2_RETENTION_THRESHOLD
) -> CorrelationResult:
    """Correlate per-design steady proxy vs unsteady result → R², τ, pass/fail."""
    steady = np.asarray(steady, dtype=float)
    unsteady = np.asarray(unsteady, dtype=float)
    r2 = pearson_r2(steady, unsteady)
    r = float(np.corrcoef(steady, unsteady)[0, 1]) if np.std(steady) and np.std(unsteady) else 0.0
    tau = kendall_tau(steady, unsteady)
    return CorrelationResult(
        r2=r2,
        pearson_r=r,
        kendall_tau=tau,
        n=int(steady.size),
        passed=r2 >= threshold,
        meta={"threshold": threshold},
    )
