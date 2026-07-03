"""Phase 4 BO backbone — multi-objective qLogNEHVI + TuRBO (V1-slim single-fidelity).

The Stage-2 optimizer engine. Fits a multi-output GP over the observed designs and
proposes the next batch by maximizing the noisy expected hypervolume improvement
(qLogNEHVI) inside a TuRBO trust region. Three objectives — ``J_fan`` (maximize),
``I_wrist`` (minimize), panel deflection (minimize) — are converted to a common
maximization frame via :data:`OBJECTIVE_SIGNS`. Everything runs in the unit cube
(bounds-normalized) so the trust-region length is dimensionless.

This replaces the R11 ``bo/`` stubs (multi-fidelity qMFKG + bandit-K): V1-slim is
single fidelity, so there is no tier promotion — categorical architecture choices
are searched directly through the codec's continuous relaxation. SAASBO
(fully-Bayesian sparse GP) is provided as a high-dimensional fallback.

Optional dependency: botorch/gpytorch/torch (the ``[bo]`` extra). Imported at the
top per CLAUDE.md §4.1 — importing this module without them fails cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.fit import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls import ExactMarginalLogLikelihood

__all__ = [
    "OBJECTIVE_SIGNS",
    "SAASBO_DIM_THRESHOLD",
    "TrustRegionState",
    "to_maximization",
    "infer_reference_point",
    "pareto_mask",
    "hypervolume",
    "fit_gp",
    "fit_saas_gp",
    "propose_candidates",
]

_DTYPE = torch.double

# (J_fan, I_wrist, structural): +1 maximize, -1 minimize → common max frame.
OBJECTIVE_SIGNS: tuple[float, float, float] = (1.0, -1.0, -1.0)

# At or above this dimensionality, prefer the SAASBO sparse-GP fallback.
SAASBO_DIM_THRESHOLD: int = 50


@dataclass
class TrustRegionState:
    """TuRBO trust-region length + success/failure counters (unit-cube side)."""

    dim: int
    batch_size: int = 1
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    success_counter: int = 0
    failure_counter: int = 0
    success_tol: int = 3
    failure_tol: int = 0
    restart_triggered: bool = False

    def __post_init__(self) -> None:
        if self.failure_tol <= 0:
            # TuRBO default: tolerate ~ceil(dim / batch) consecutive failures.
            self.failure_tol = int(np.ceil(self.dim / max(self.batch_size, 1)))

    def update(self, improved: bool) -> None:
        """Grow on a success streak, shrink on a failure streak (TuRBO rule)."""
        if improved:
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1
        if self.success_counter == self.success_tol:
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tol:
            self.length /= 2.0
            self.failure_counter = 0
        if self.length < self.length_min:
            self.restart_triggered = True


def to_maximization(y_raw: np.ndarray, signs: tuple[float, ...] = OBJECTIVE_SIGNS) -> np.ndarray:
    """Flip minimize objectives so every column is maximized (``y * signs``)."""
    y = np.atleast_2d(np.asarray(y_raw, dtype=float))
    if y.shape[1] != len(signs):
        raise ValueError(f"y has {y.shape[1]} objectives; expected {len(signs)}")
    return y * np.asarray(signs, dtype=float)


def infer_reference_point(y_max: np.ndarray, *, margin_frac: float = 0.1) -> np.ndarray:
    """A reference point dominated by all observations (min per objective − margin).

    ``margin_frac`` extends below the observed range so the worst points still
    contribute positive hypervolume.
    """
    y = np.atleast_2d(np.asarray(y_max, dtype=float))
    lo = y.min(axis=0)
    hi = y.max(axis=0)
    span = np.where(hi > lo, hi - lo, np.abs(lo) + 1.0)
    return lo - margin_frac * span


def pareto_mask(y_max: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows (all objectives maximized)."""
    t = torch.as_tensor(np.atleast_2d(y_max), dtype=_DTYPE)
    return is_non_dominated(t).cpu().numpy()


def hypervolume(y_max: np.ndarray, ref_point: np.ndarray) -> float:
    """Dominated hypervolume of the observations above ``ref_point``."""
    y = torch.as_tensor(np.atleast_2d(y_max), dtype=_DTYPE)
    ref = torch.as_tensor(np.asarray(ref_point), dtype=_DTYPE)
    pareto_y = y[is_non_dominated(y)]
    if pareto_y.numel() == 0:
        return 0.0
    return float(Hypervolume(ref_point=ref).compute(pareto_y))


def _normalize(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> torch.Tensor:
    xn = (np.atleast_2d(x) - low) / (high - low)
    return torch.as_tensor(xn, dtype=_DTYPE)


def fit_gp(x: np.ndarray, y_max: np.ndarray, low: np.ndarray, high: np.ndarray) -> SingleTaskGP:
    """Fit a multi-output ``SingleTaskGP`` on unit-cube inputs, standardized Y."""
    xn = _normalize(x, low, high)
    y = torch.as_tensor(np.atleast_2d(y_max), dtype=_DTYPE)
    model = SingleTaskGP(xn, y, outcome_transform=Standardize(m=y.shape[-1]))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def fit_saas_gp(
    x: np.ndarray,
    y_max: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    warmup_steps: int = 256,
    num_samples: int = 128,
    thinning: int = 16,
) -> ModelListGP:
    """SAASBO fully-Bayesian sparse-axis GP fallback for high-dimensional search.

    SAASBO GPs are single-output, so a multi-objective fit is a
    :class:`ModelListGP` of one NUTS-fit :class:`SaasFullyBayesianSingleTaskGP`
    per objective. Needs the ``fully_bayesian`` extra (NumPyro); import fails
    cleanly without it.
    """
    xn = _normalize(x, low, high)
    y = torch.as_tensor(np.atleast_2d(y_max), dtype=_DTYPE)
    models = []
    for i in range(y.shape[-1]):
        m_i = SaasFullyBayesianSingleTaskGP(xn, y[:, i : i + 1], outcome_transform=Standardize(m=1))
        fit_fully_bayesian_model_nuts(
            m_i,
            warmup_steps=warmup_steps,
            num_samples=num_samples,
            thinning=thinning,
            disable_progbar=True,
        )
        models.append(m_i)
    return ModelListGP(*models)


def _trust_region_bounds(x_max_norm: torch.Tensor, tr: TrustRegionState) -> torch.Tensor:
    """Unit-cube box of side ``tr.length`` centred on the incumbent, clamped to [0,1]."""
    half = tr.length / 2.0
    lo = torch.clamp(x_max_norm - half, 0.0, 1.0)
    hi = torch.clamp(x_max_norm + half, 0.0, 1.0)
    return torch.stack([lo, hi])


def propose_candidates(
    model: SingleTaskGP,
    x: np.ndarray,
    y_max: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    ref_point: np.ndarray,
    *,
    batch_size: int = 1,
    tr_state: TrustRegionState | None = None,
    num_restarts: int = 8,
    raw_samples: int = 128,
    mc_samples: int = 128,
    incumbent_objective: int = 0,
) -> np.ndarray:
    """Propose ``batch_size`` designs by maximizing qLogNEHVI (optionally in a TR).

    Returns candidates in the **original** parameter space (shape
    ``(batch_size, n_dims)``). With ``tr_state`` given, the acquisition is
    optimized inside a TuRBO trust region centred on the incumbent that maximizes
    ``incumbent_objective`` (default ``J_fan``); otherwise over the full box.
    """
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    xn = _normalize(x, low, high)
    y = np.atleast_2d(np.asarray(y_max, dtype=float))

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
    acqf = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=torch.as_tensor(np.asarray(ref_point), dtype=_DTYPE).tolist(),
        X_baseline=xn,
        sampler=sampler,
        prune_baseline=True,
    )

    if tr_state is not None:
        incumbent = xn[int(np.argmax(y[:, incumbent_objective]))]
        bounds = _trust_region_bounds(incumbent, tr_state)
    else:
        bounds = torch.stack(
            [torch.zeros(xn.shape[-1], dtype=_DTYPE), torch.ones(xn.shape[-1], dtype=_DTYPE)]
        )

    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    cand_norm = candidates.detach().cpu().numpy()
    return low + cand_norm * (high - low)
