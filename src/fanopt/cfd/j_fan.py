"""Canonical J_fan post-processor (locked spec, report-final.md §9.4).

Every solver in the stack (SU2 2D/3D steady, SU2 2D/3D unsteady, PyFR) routes
its output through this module so all phases and fidelities compare
apples-to-apples. Two modes, auto-detected on whether the input carries a time
dimension:

- **Unsteady** (Tier 1, verification): the time-integrated directed
  momentum-flux through the fixed analysis plane Σ,
  ``J_fan = <(1/T) ∫∫_Σ ρ0 · u_n · (u·t̂) dA>`` averaged over cycles 2..5.
- **Steady proxy** (Tier -1 / Tier 0 screening): the two-eval delta
  ``Drag_productive − Drag_return`` of the fan surface force projected on t̂.

The core computations are pure functions over numpy arrays; SU2/PyFR file I/O
lives in ``fanopt.cfd.parsers`` and hands validated arrays to these functions.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from fanopt.geometry.schema import T_CYCLE_S

__all__ = [
    "RHO_AIR_KG_PER_M3",
    "THRUST_DIR",
    "ANALYSIS_PLANE_DISTANCE_M",
    "ANALYSIS_PLANE_SIZE_M",
    "N_CYCLES_CANONICAL",
    "N_CYCLES_DISCARD",
    "STEPS_PER_CYCLE",
    "CYCLE_EXTEND_REL_THRESHOLD",
    "SteadyRun",
    "SteadyProxyResult",
    "UnsteadyResult",
    "plane_momentum_flux",
    "plane_flux_from_velocity",
    "reduce_cycles",
    "compute_j_fan_steady",
    "compute_j_fan",
]

_LOG = logging.getLogger(__name__)

# --- Locked J_fan-spec constants (report-final.md §9.4 objective spec) ---
RHO_AIR_KG_PER_M3: float = 1.225  # constant; flow Mach ~0.008, compressibility <0.01%
THRUST_DIR: tuple[float, float, float] = (0.0, 0.0, 1.0)  # t̂ = +ẑ (user-ward)
ANALYSIS_PLANE_DISTANCE_M: float = 0.300  # plane 300 mm forward of pivot along +z
ANALYSIS_PLANE_SIZE_M: float = 0.600  # 600 × 600 mm plane
N_CYCLES_CANONICAL: int = 5  # 5 total; integrate cycles 2..5
N_CYCLES_DISCARD: int = 1  # discard cycle 1 (transient)
STEPS_PER_CYCLE: int = 200  # dt = T/200 lock
CYCLE_EXTEND_REL_THRESHOLD: float = 0.05  # cycle-2 vs cycle-3 > 5% ⇒ extend to 8 cycles


@dataclass(frozen=True)
class SteadyRun:
    """One steady CFD eval: the fan surface force projected onto t̂.

    ``thrust`` is ``∫∫_S (p·n̂ + τ)·t̂ dA`` for this run (Newtons, or a
    consistent force coefficient). ``stroke`` is ``"productive"`` (freestream
    ``(0,0,-1)``) or ``"return"`` (``(0,0,+1)``).
    """

    thrust: float
    stroke: str
    design_hash: str = ""

    def __post_init__(self) -> None:
        if self.stroke not in ("productive", "return"):
            raise ValueError(f"stroke must be 'productive' or 'return', got {self.stroke!r}")


@dataclass(frozen=True)
class SteadyProxyResult:
    """Steady two-eval delta proxy (§9.4.1)."""

    j_fan_steady_proxy: float
    proxy_kind: str  # "delta" | "one_direction"
    drag_productive: float | None
    drag_return: float | None


@dataclass(frozen=True)
class UnsteadyResult:
    """Time-integrated J_fan over the retained cycles (§9.4)."""

    j_fan: float
    j_fan_peak: float
    j_fan_se: float
    per_cycle: tuple[float, ...]
    n_avg: int
    cycle2_vs_cycle3_rel_diff: float
    extend_recommended: bool
    meta: dict[str, float] = field(default_factory=dict)


def plane_flux_from_velocity(
    velocity: np.ndarray,
    area: np.ndarray,
    *,
    n_hat: Sequence[float] = THRUST_DIR,
    t_hat: Sequence[float] = THRUST_DIR,
    rho: float = RHO_AIR_KG_PER_M3,
) -> float:
    """Instantaneous plane integral ``∫∫_Σ ρ · u_n · (u·t̂) dA``.

    ``velocity`` is ``(N, 3)`` sample velocities on the plane; ``area`` is the
    ``(N,)`` per-sample area weight. ``u_n = u·n̂`` and ``u_t = u·t̂`` are kept
    as separate signed projections so the general (direction-sensitive) form is
    preserved even when ``n̂ == t̂``.
    """
    velocity = np.asarray(velocity, dtype=float)
    area = np.asarray(area, dtype=float)
    if velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ValueError(f"velocity must be (N, 3); got shape {velocity.shape}")
    if area.shape != (velocity.shape[0],):
        raise ValueError(f"area must be (N,) matching velocity rows; got {area.shape}")
    n = np.asarray(n_hat, dtype=float)
    t = np.asarray(t_hat, dtype=float)
    u_n = velocity @ n
    u_t = velocity @ t
    return plane_momentum_flux(u_n, u_t, area, rho=rho)


def plane_momentum_flux(
    u_n: np.ndarray,
    u_t: np.ndarray,
    area: np.ndarray,
    *,
    rho: float = RHO_AIR_KG_PER_M3,
) -> float:
    """Instantaneous ``ρ · Σ (u_n · u_t · dA)`` given precomputed projections."""
    u_n = np.asarray(u_n, dtype=float)
    u_t = np.asarray(u_t, dtype=float)
    area = np.asarray(area, dtype=float)
    if not (u_n.shape == u_t.shape == area.shape):
        raise ValueError(
            f"u_n, u_t, area must share shape; got {u_n.shape}, {u_t.shape}, {area.shape}"
        )
    return float(rho * np.sum(u_n * u_t * area))


def reduce_cycles(
    instantaneous_flux: np.ndarray,
    *,
    steps_per_cycle: int = STEPS_PER_CYCLE,
    n_discard: int = N_CYCLES_DISCARD,
    period_s: float = T_CYCLE_S,
) -> UnsteadyResult:
    """Cycle-reduce a per-time-step instantaneous plane-flux series into J_fan.

    ``instantaneous_flux[k]`` is the plane integral ``∫∫_Σ ρ·u_n·u_t dA`` at
    outer time step ``k``. Length must be an exact multiple of
    ``steps_per_cycle``. The first ``n_discard`` cycles are dropped as transient;
    each retained cycle's value is its time-average ``(1/T) ∫_cycle I dt``
    (= the mean over the cycle's samples, since ``dt = T/steps_per_cycle``).

    ``J_fan`` is the mean of the retained per-cycle values; ``j_fan_se =
    std/√n_avg`` (``n_avg = N_CYCLES − n_discard``). ``extend_recommended`` is
    set when the first two retained cycles differ by more than
    ``CYCLE_EXTEND_REL_THRESHOLD`` (§9.4 decision rule).
    """
    flux = np.asarray(instantaneous_flux, dtype=float)
    if flux.ndim != 1:
        raise ValueError(f"instantaneous_flux must be 1D; got shape {flux.shape}")
    if steps_per_cycle <= 0:
        raise ValueError(f"steps_per_cycle must be > 0; got {steps_per_cycle}")
    if period_s <= 0:
        raise ValueError(f"period_s must be > 0; got {period_s}")
    n_total = flux.size
    if n_total == 0 or n_total % steps_per_cycle != 0:
        raise ValueError(
            f"series length {n_total} is not a positive multiple of "
            f"steps_per_cycle {steps_per_cycle}"
        )
    n_cycles = n_total // steps_per_cycle
    if n_discard < 0 or n_discard >= n_cycles:
        raise ValueError(f"n_discard {n_discard} must be in [0, {n_cycles})")

    cycles = flux.reshape(n_cycles, steps_per_cycle)
    retained = cycles[n_discard:]
    # (1/T) ∫_cycle I dt over a full period is the cycle time-average.
    per_cycle = retained.mean(axis=1)
    per_cycle_peak = np.clip(retained, 0.0, None).mean(axis=1)

    n_avg = int(per_cycle.size)
    j_fan = float(per_cycle.mean())
    j_fan_peak = float(per_cycle_peak.mean())
    j_fan_se = float(per_cycle.std(ddof=1) / np.sqrt(n_avg)) if n_avg > 1 else 0.0

    if n_avg >= 2:
        c2, c3 = per_cycle[0], per_cycle[1]
        denom = (abs(c2) + abs(c3)) / 2.0
        rel = float(abs(c2 - c3) / denom) if denom > 0 else 0.0
    else:
        rel = 0.0
    extend = rel > CYCLE_EXTEND_REL_THRESHOLD

    return UnsteadyResult(
        j_fan=j_fan,
        j_fan_peak=j_fan_peak,
        j_fan_se=j_fan_se,
        per_cycle=tuple(float(v) for v in per_cycle),
        n_avg=n_avg,
        cycle2_vs_cycle3_rel_diff=rel,
        extend_recommended=extend,
        meta={"n_cycles_total": float(n_cycles), "n_discard": float(n_discard)},
    )


def compute_j_fan_steady(runs: Sequence[SteadyRun]) -> SteadyProxyResult:
    """Steady two-eval delta proxy (§9.4.1).

    When both a productive and a return run are present, returns the delta
    ``Drag_productive − Drag_return`` (``proxy_kind="delta"``). When only one is
    present, returns the one-direction value with a WARNING (debug/legacy path;
    production always emits both strokes).
    """
    if not runs:
        raise ValueError("compute_j_fan_steady requires at least one SteadyRun")
    productive = [r for r in runs if r.stroke == "productive"]
    returns = [r for r in runs if r.stroke == "return"]
    if len(productive) > 1 or len(returns) > 1:
        raise ValueError("expected at most one run per stroke for one design")

    if productive and returns:
        drag_p, drag_r = productive[0].thrust, returns[0].thrust
        return SteadyProxyResult(
            j_fan_steady_proxy=drag_p - drag_r,
            proxy_kind="delta",
            drag_productive=drag_p,
            drag_return=drag_r,
        )

    single = (productive or returns)[0]
    _LOG.warning(
        "compute_j_fan_steady: only the %r stroke present; returning one-direction "
        "proxy (production emits both strokes for the delta)",
        single.stroke,
    )
    return SteadyProxyResult(
        j_fan_steady_proxy=single.thrust,
        proxy_kind="one_direction",
        drag_productive=productive[0].thrust if productive else None,
        drag_return=returns[0].thrust if returns else None,
    )


def compute_j_fan(source: object, **kwargs: object) -> SteadyProxyResult | UnsteadyResult:
    """Auto-dispatch on whether the input carries a time dimension.

    - A :class:`SteadyRun` or sequence of them ⇒ steady two-eval proxy.
    - A 1D array-like of per-time-step plane flux ⇒ unsteady cycle reduction.
    """
    if isinstance(source, SteadyRun):
        return compute_j_fan_steady([source])
    if (
        isinstance(source, Sequence)
        and not isinstance(source, str | bytes)
        and len(source) > 0
        and all(isinstance(r, SteadyRun) for r in source)
    ):
        return compute_j_fan_steady(list(source))  # type: ignore[arg-type]
    if isinstance(source, str | bytes):
        raise TypeError(f"compute_j_fan does not accept {type(source).__name__} input")
    try:
        arr = np.asarray(source, dtype=float)
    except (ValueError, TypeError) as exc:
        raise TypeError(
            f"compute_j_fan could not read {type(source)!r} as a numeric flux series: {exc}"
        ) from exc
    if arr.ndim == 1:
        return reduce_cycles(arr, **kwargs)  # type: ignore[arg-type]
    raise TypeError(
        "compute_j_fan expects a SteadyRun sequence (steady proxy) or a 1D "
        f"instantaneous-flux series (unsteady); got {type(source)!r} shape "
        f"{getattr(arr, 'shape', None)}"
    )
