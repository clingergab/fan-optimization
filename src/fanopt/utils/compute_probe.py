"""Compute-budget probe and local-M3 sub-spike gates for Spike 0.6.

Implements Spike 0.6 (`docs/spike_0_6_protocol.md`,
`docs/plan_R11.md §Phase 0 Spike 0.6`).

Spike 0.6 has three measurement strands:

1. **Calibration probe** (parent spike, NOT a gate per spec § "Status"):
   record wall-time and compute-unit consumption for one representative 3D
   unsteady SU2 case (500K cells, 5 pitching cycles, dt = T/200) on a Colab
   Pro CPU instance and on a Colab Pro G4-class GPU node. The numbers feed
   the Phase 4 / Phase 5 budget estimate; they do not gate downstream work.

2. **Sub-spike 0.6a** (gate for local-M3 SU2 use): run one Tier-1 case
   end-to-end on the MacBook M3 (CadQuery -> Gmsh 2D corrugated slice ->
   SU2 2D steady -> `j_fan.py`). Pass iff wall-time <= 15 min AND
   `J_fan_steady_proxy` is finite. Fail flips `smoke_test.py` to a Colab
   Pro CPU session (fallback in `docs/spike_0_6_protocol.md`).

3. **Sub-spike 0.6b** (gate for Phase 5 step 59.5 / step 64.5 local FEA):
   run one FEniCSx (or CalculiX) static FEA case on the M3 -- a simple
   cantilever rib under a 5 N tip load. Pass iff wall-time <= 2 min AND
   the measured tip deflection matches the analytic Euler-Bernoulli
   `P L^3 / (3 E I)` within 5%. Fail moves step 64.5's combined-blade
   structural FEA to a Colab Pro CPU session.

The aggregator (`analyze_spike_06`) treats the parent spike as calibration:
`overall_passed = True` even when sub-spikes fail, but the per-sub-spike
pass flags are surfaced for reporting and CI.

References:
- Spec: `docs/plan_R11.md §Phase 0 Spike 0.6`
- Protocol: `docs/spike_0_6_protocol.md`
- Phase log: `docs/phase_logs/spike_0_6.md`
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "M3_SU2_WALL_TIME_GATE_S",
    "M3_FEA_WALL_TIME_GATE_S",
    "M3_FEA_TIP_DEFLECTION_TOLERANCE_PCT",
    "analytic_cantilever_tip_deflection",
    "ComputeBudgetEntry",
    "SubSpike06AResult",
    "SubSpike06BResult",
    "Spike06Result",
    "analyze_06a",
    "analyze_06b",
    "analyze_spike_06",
]


# ----- pass-criterion gates (locked from §Phase 0 Spike 0.6) -----------------

M3_SU2_WALL_TIME_GATE_S: float = 15 * 60
"""Sub-spike 0.6a gate: Tier-1 end-to-end pipeline must complete in <= 15 min
on the MacBook M3 to qualify the M3 for any local SU2 use."""

M3_FEA_WALL_TIME_GATE_S: float = 2 * 60
"""Sub-spike 0.6b gate: cantilever-rib FEA must complete in <= 2 min on the
M3 to qualify the M3 for Phase 5 step 59.5 / step 64.5 local FEA."""

M3_FEA_TIP_DEFLECTION_TOLERANCE_PCT: float = 5.0
"""Sub-spike 0.6b cross-check gate: measured tip deflection must match the
analytic Euler-Bernoulli `P L^3 / (3 E I)` within 5%."""


# ----- analytic reference ----------------------------------------------------


def analytic_cantilever_tip_deflection(
    P_N: float, L_m: float, E_Pa: float, I_m4: float
) -> float:
    """Euler-Bernoulli tip deflection of a cantilever under a tip point load.

    `delta = P * L^3 / (3 * E * I)`. Used by sub-spike 0.6b to cross-check
    the FEniCSx (or CalculiX) FEA result against an analytic value the
    operator can hand-compute on the bench.

    Parameters
    ----------
    P_N : tip point load, newtons.
    L_m : cantilever length, metres.
    E_Pa : Young's modulus, pascals.
    I_m4 : second moment of area about the bending axis, m^4. For a
        rectangular cross-section of width `b` and height `h` bending about
        the b-axis: `I = b * h^3 / 12`.

    Returns
    -------
    Tip deflection in metres, positive in the direction of the applied load.
    """
    if P_N <= 0:
        raise ValueError(f"P_N must be > 0, got {P_N}")
    if L_m <= 0:
        raise ValueError(f"L_m must be > 0, got {L_m}")
    if E_Pa <= 0:
        raise ValueError(f"E_Pa must be > 0, got {E_Pa}")
    if I_m4 <= 0:
        raise ValueError(f"I_m4 must be > 0, got {I_m4}")
    return P_N * (L_m ** 3) / (3.0 * E_Pa * I_m4)


# ----- compute-budget rows (calibration only — not a gate) -------------------


@dataclass(frozen=True)
class ComputeBudgetEntry:
    """One row of measured compute-budget data.

    Spike 0.6's calibration strand collects at least two rows -- Colab CPU
    3D unsteady and Colab GPU 3D unsteady -- but the aggregator accepts any
    number of rows so the operator can also record M3 timings, smoke-test
    repeats, or alternative Colab hardware (e.g., G4 vs A100).
    """

    platform: str
    """Free-form platform label, e.g., 'colab_pro_cpu', 'colab_pro_g4_gpu',
    'macbook_m3'. See `docs/spike_0_6_protocol.md` for the canonical set."""

    workload: str
    """Short workload identifier, e.g., '3d_unsteady_500k_5cycles_dtT200'."""

    wall_time_s: float
    """Wall-clock duration of one full evaluation, seconds."""

    cu_consumed: float | None
    """Colab compute units consumed (None when not applicable, e.g., M3)."""

    cells: int | None
    """Mesh cell count for the run (None when not a mesh-based workload)."""

    notes: str = ""
    """Free-form operator notes."""


# ----- sub-spike 0.6a (M3 SU2 Tier-1 end-to-end) ----------------------------


@dataclass(frozen=True)
class SubSpike06AResult:
    """Output of `analyze_06a`. Surfaces the gate status for the M3 SU2
    Tier-1 end-to-end pipeline (CadQuery -> Gmsh 2D -> SU2 2D steady ->
    `j_fan.py`).

    Pass criterion (gate for any local-M3 SU2 use): wall_time <= 15 min AND
    `J_fan_steady_proxy` is finite (NaN / inf -> fail).
    """

    m3_wall_time_s: float
    J_fan_steady_proxy: float | None
    pipeline_stages: dict[str, float]
    wall_time_passed: bool
    j_fan_finite: bool
    passed: bool


def analyze_06a(
    wall_time_s: float,
    J_fan_steady_proxy: float | None,
    stages: dict[str, float] | None = None,
) -> SubSpike06AResult:
    """Apply the Spike 0.6 sub-spike-0.6a gate to one M3 Tier-1 timing.

    Parameters
    ----------
    wall_time_s : total end-to-end wall-clock for the Tier-1 pipeline on
        the M3 (CadQuery + Gmsh + SU2 + `j_fan.py`).
    J_fan_steady_proxy : the steady-tier proxy J_fan emitted by `j_fan.py`
        for the smoke geometry. May be None or NaN if the pipeline crashed;
        a non-finite value fails the gate.
    stages : optional per-stage wall-time breakdown (CadQuery, Gmsh, SU2,
        `j_fan.py`) for the run log. Sum need not equal `wall_time_s` --
        there is overhead between stages (file I/O, etc.).

    Returns
    -------
    SubSpike06AResult with the gate status. `passed` is True iff
    `wall_time_s <= M3_SU2_WALL_TIME_GATE_S` AND `J_fan_steady_proxy` is
    finite.
    """
    if wall_time_s < 0:
        raise ValueError(f"wall_time_s must be >= 0, got {wall_time_s}")

    stages_clean: dict[str, float] = dict(stages) if stages is not None else {}

    wall_time_passed = wall_time_s <= M3_SU2_WALL_TIME_GATE_S
    j_fan_finite = J_fan_steady_proxy is not None and math.isfinite(J_fan_steady_proxy)
    passed = wall_time_passed and j_fan_finite

    return SubSpike06AResult(
        m3_wall_time_s=float(wall_time_s),
        J_fan_steady_proxy=(
            float(J_fan_steady_proxy) if J_fan_steady_proxy is not None else None
        ),
        pipeline_stages=stages_clean,
        wall_time_passed=wall_time_passed,
        j_fan_finite=j_fan_finite,
        passed=passed,
    )


# ----- sub-spike 0.6b (M3 FEA cantilever) -----------------------------------


@dataclass(frozen=True)
class SubSpike06BResult:
    """Output of `analyze_06b`. Surfaces the gate status for the M3 FEA
    cantilever cross-check.

    Two gates apply (both must pass):
    1. wall_time <= 2 min
    2. |measured - analytic| / analytic * 100 <= 5%
    """

    wall_time_s: float
    measured_tip_deflection_m: float
    analytic_tip_deflection_m: float
    tip_deflection_pct: float
    wall_time_passed: bool
    tip_deflection_passed: bool
    passed: bool


def analyze_06b(
    wall_time_s: float,
    measured_deflection_m: float,
    P_N: float,
    L_m: float,
    E_Pa: float,
    I_m4: float,
) -> SubSpike06BResult:
    """Apply the Spike 0.6 sub-spike-0.6b gates to one M3 FEA cantilever run.

    Computes the analytic Euler-Bernoulli tip deflection
    `P L^3 / (3 E I)`, compares to the FEA-measured deflection, and applies
    both the wall-time and tip-deflection-tolerance gates.

    Parameters
    ----------
    wall_time_s : FEA wall-clock, seconds.
    measured_deflection_m : tip deflection reported by FEniCSx (or CalculiX).
    P_N, L_m, E_Pa, I_m4 : geometry / load / material parameters of the
        cantilever; passed through to `analytic_cantilever_tip_deflection`.
    """
    if wall_time_s < 0:
        raise ValueError(f"wall_time_s must be >= 0, got {wall_time_s}")
    if not math.isfinite(measured_deflection_m):
        raise ValueError(
            f"measured_deflection_m must be finite, got {measured_deflection_m}"
        )

    analytic = analytic_cantilever_tip_deflection(P_N=P_N, L_m=L_m, E_Pa=E_Pa, I_m4=I_m4)
    pct = 100.0 * abs(measured_deflection_m - analytic) / analytic

    wall_time_passed = wall_time_s <= M3_FEA_WALL_TIME_GATE_S
    tip_deflection_passed = pct <= M3_FEA_TIP_DEFLECTION_TOLERANCE_PCT
    passed = wall_time_passed and tip_deflection_passed

    return SubSpike06BResult(
        wall_time_s=float(wall_time_s),
        measured_tip_deflection_m=float(measured_deflection_m),
        analytic_tip_deflection_m=float(analytic),
        tip_deflection_pct=float(pct),
        wall_time_passed=wall_time_passed,
        tip_deflection_passed=tip_deflection_passed,
        passed=passed,
    )


# ----- aggregate Spike 0.6 result -------------------------------------------


@dataclass(frozen=True)
class Spike06Result:
    """Aggregate roll-up of Spike 0.6: calibration probe + both sub-spikes.

    Per spec § "Status", Spike 0.6 is treated as calibration, NOT a gate, so
    `overall_passed` is always True. The sub-spike pass flags
    (`sub_06a.passed`, `sub_06b.passed`) carry the actual gate status for
    their respective downstream phases and are surfaced separately so the
    CLI / CI can branch on them.
    """

    budget_entries: tuple[ComputeBudgetEntry, ...]
    sub_06a: SubSpike06AResult | None
    sub_06b: SubSpike06BResult | None
    overall_passed: bool = field(default=True)


def analyze_spike_06(
    budget_entries: list[ComputeBudgetEntry] | tuple[ComputeBudgetEntry, ...],
    sub_06a: SubSpike06AResult | None,
    sub_06b: SubSpike06BResult | None,
) -> Spike06Result:
    """Roll up the calibration probe + the two sub-spike results.

    Calibration is informational — `overall_passed` is True regardless of
    the contents of `budget_entries` or the sub-spike pass flags. Callers
    that need to gate on sub-spikes should branch on
    `result.sub_06a.passed` / `result.sub_06b.passed` directly.
    """
    return Spike06Result(
        budget_entries=tuple(budget_entries),
        sub_06a=sub_06a,
        sub_06b=sub_06b,
        overall_passed=True,
    )
