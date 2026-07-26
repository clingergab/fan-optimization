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
    cv: float  # std / largest-magnitude value (scale-stable spread)
    range_frac: float  # (max − min) / largest-magnitude value
    has_headroom: bool


def headroom(j_fans: np.ndarray, *, spread_threshold: float = 0.15) -> Headroom:
    """Relative spread of finite 3D-wind values; ``has_headroom`` iff ``range_frac ≥ threshold``.

    The objective is the **signed** cycle-mean CFz (N1: flat ≈ −4e10, scoop ≈ +1.4e11), so a
    probe set can straddle zero and the mean can cancel to ≈0 — dividing by ``|mean|`` (the old
    form) then makes CV explode and falsely flag a flat plateau as having headroom. Instead we
    scale by the **largest-magnitude value**, which cannot cancel: ``range_frac = ptp / max|v|``
    measures whether the best design is meaningfully different from the worst. A near-constant
    plateau → small ``range_frac`` → no headroom, on the correct axis, before spending compute.

    (This measures *relative variation* — that a gradient exists to climb. Whether the best
    design's absolute wind clears the flat-panel baseline is the separate V1 baseline check.)
    """
    v = np.asarray([x for x in np.asarray(j_fans, dtype=float) if np.isfinite(x)], dtype=float)
    if v.size < 2:
        return Headroom(int(v.size), float(v.mean()) if v.size else 0.0, 0.0, 0.0, False)
    mean = float(v.mean())
    scale = max(abs(float(v.max())), abs(float(v.min())), 1e-30)
    cv = float(v.std() / scale)
    range_frac = float((v.max() - v.min()) / scale)
    return Headroom(int(v.size), mean, cv, range_frac, bool(range_frac >= spread_threshold))
