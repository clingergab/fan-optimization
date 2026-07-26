"""Stage-2 validation analyses for the ADR-0004 optimization redo.

Two cheap de-risking checks that gate the expensive BO campaign:

- **Fidelity study (2a):** does a *cheaper* coarse-3D evaluation rank designs the same way the
  expensive fine-3D does? If yes, the campaign runs the cheap tier and confirms with the fine
  one — the single biggest speed lever. :func:`ranking_agreement` (Kendall τ / Spearman ρ).
- **Shape-space headroom probe (2b):** does 3D wind vary *enough* across designs to be worth a
  week of optimizing? A near-constant plateau means optimizing buys little (and the old
  "~0.8–1.1T plateau" was measured on the wrong CFx axis, so it must be re-established on the
  corrected CFz). :func:`headroom`.

Pure numpy + the existing rank-correlation helpers; the CFD that produces the inputs runs in
the harness scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fanopt.cfd.correlation import kendall_tau, spearman_rho

__all__ = ["FidelityAgreement", "ranking_agreement", "Headroom", "headroom"]


@dataclass(frozen=True)
class FidelityAgreement:
    """Does coarse-3D preserve the fine-3D ranking? (paired over the same designs)."""

    n: int
    kendall_tau: float | None
    spearman_rho: float | None
    ranking_preserved: bool | None


def ranking_agreement(
    coarse: np.ndarray, fine: np.ndarray, *, tau_threshold: float = 0.7
) -> FidelityAgreement:
    """Kendall τ / Spearman ρ between coarse- and fine-3D values over the same designs.

    ``ranking_preserved`` is ``True`` iff ``τ ≥ tau_threshold`` — i.e. the cheap tier orders
    designs closely enough to drive the BO and let the fine tier confirm. Non-finite pairs
    (a diverged eval at either fidelity) are dropped; ``None`` if fewer than 2 pairs remain.
    """
    c = np.asarray(coarse, dtype=float)
    f = np.asarray(fine, dtype=float)
    mask = np.isfinite(c) & np.isfinite(f)
    c, f = c[mask], f[mask]
    if c.size < 2:
        return FidelityAgreement(int(c.size), None, None, None)
    tau = float(kendall_tau(c, f))
    return FidelityAgreement(int(c.size), tau, float(spearman_rho(c, f)), bool(tau >= tau_threshold))


@dataclass(frozen=True)
class Headroom:
    """Does 3D wind vary enough across designs to justify optimizing?"""

    n: int
    mean: float
    cv: float  # coefficient of variation std/|mean|
    range_frac: float  # (max − min)/|mean|
    has_headroom: bool


def headroom(j_fans: np.ndarray, *, cv_threshold: float = 0.1) -> Headroom:
    """Spread of finite 3D-wind values; ``has_headroom`` iff ``CV ≥ cv_threshold``.

    A near-constant plateau (small CV) means the objective is design-insensitive and a long
    optimization buys little — surface that *before* spending the compute, not after.
    """
    v = np.asarray([x for x in np.asarray(j_fans, dtype=float) if np.isfinite(x)], dtype=float)
    if v.size < 2:
        return Headroom(int(v.size), float(v.mean()) if v.size else 0.0, 0.0, 0.0, False)
    mean = float(v.mean())
    denom = abs(mean) if mean != 0.0 else 1.0
    cv = float(v.std() / denom)
    range_frac = float((v.max() - v.min()) / denom)
    return Headroom(int(v.size), mean, cv, range_frac, bool(cv >= cv_threshold))
