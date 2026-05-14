"""Unit tests for fanopt.utils.compute_probe.

Validates the analytic cantilever formula and the two sub-spike gates
(0.6a wall-time / J_fan-finite; 0.6b wall-time / tip-deflection tolerance),
plus the aggregator's calibration-only roll-up semantics.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.6; protocol in
docs/spike_0_6_protocol.md.
"""
from __future__ import annotations

import math

import pytest

from fanopt.utils.compute_probe import (
    M3_FEA_TIP_DEFLECTION_TOLERANCE_PCT,
    M3_FEA_WALL_TIME_GATE_S,
    M3_SU2_WALL_TIME_GATE_S,
    ComputeBudgetEntry,
    Spike06Result,
    SubSpike06AResult,
    SubSpike06BResult,
    analytic_cantilever_tip_deflection,
    analyze_06a,
    analyze_06b,
    analyze_spike_06,
)


# ----- analytic cantilever -------------------------------------------------


def test_analytic_cantilever_pl3_3ei() -> None:
    """`delta = P L^3 / (3 E I)` — hand-computed value.

    Pick numbers that yield a round answer:
      P = 3 N, L = 1 m, E = 1 Pa, I = 1 m^4
      delta = 3 * 1 / (3 * 1 * 1) = 1.0 m exactly.
    Also exercise the realistic PETG cantilever in the runner:
      P = 5 N, L = 0.200 m, E = 1.3e9 Pa, I = b h^3 / 12 with
      b = 0.012, h = 0.002 -> I = 8e-12 m^4.
    """
    assert analytic_cantilever_tip_deflection(P_N=3.0, L_m=1.0, E_Pa=1.0, I_m4=1.0) == 1.0

    b, h = 0.012, 0.002
    I = b * h**3 / 12.0
    expected = 5.0 * (0.200**3) / (3.0 * 1.3e9 * I)
    measured = analytic_cantilever_tip_deflection(P_N=5.0, L_m=0.200, E_Pa=1.3e9, I_m4=I)
    assert math.isclose(measured, expected, rel_tol=1e-12)


def test_analytic_cantilever_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="P_N"):
        analytic_cantilever_tip_deflection(P_N=0.0, L_m=1.0, E_Pa=1.0, I_m4=1.0)
    with pytest.raises(ValueError, match="L_m"):
        analytic_cantilever_tip_deflection(P_N=1.0, L_m=0.0, E_Pa=1.0, I_m4=1.0)
    with pytest.raises(ValueError, match="E_Pa"):
        analytic_cantilever_tip_deflection(P_N=1.0, L_m=1.0, E_Pa=0.0, I_m4=1.0)
    with pytest.raises(ValueError, match="I_m4"):
        analytic_cantilever_tip_deflection(P_N=1.0, L_m=1.0, E_Pa=1.0, I_m4=-1e-9)


# ----- sub-spike 0.6a ------------------------------------------------------


def test_06a_pass_under_15_min() -> None:
    """Wall-time well under 15 min AND a finite J_fan -> pass."""
    res = analyze_06a(
        wall_time_s=10 * 60,
        J_fan_steady_proxy=0.123,
        stages={"cadquery": 5.0, "gmsh_2d": 10.0, "su2_2d_steady": 580.0, "j_fan": 5.0},
    )
    assert res.wall_time_passed is True
    assert res.j_fan_finite is True
    assert res.passed is True
    assert res.m3_wall_time_s == 600.0
    assert res.pipeline_stages["su2_2d_steady"] == 580.0


def test_06a_fail_over_15_min() -> None:
    """Wall-time above the 15-min gate -> fail even with a finite J_fan."""
    res = analyze_06a(
        wall_time_s=M3_SU2_WALL_TIME_GATE_S + 1.0,
        J_fan_steady_proxy=0.123,
    )
    assert res.wall_time_passed is False
    assert res.j_fan_finite is True
    assert res.passed is False


def test_06a_fail_if_jfan_nan() -> None:
    """NaN J_fan -> j_fan_finite=False -> fail even at low wall-time."""
    res = analyze_06a(wall_time_s=60.0, J_fan_steady_proxy=float("nan"))
    assert res.j_fan_finite is False
    assert res.passed is False

    # None J_fan (e.g., pipeline crashed before j_fan.py ran) -> fail.
    res_none = analyze_06a(wall_time_s=60.0, J_fan_steady_proxy=None)
    assert res_none.j_fan_finite is False
    assert res_none.passed is False

    # +inf also fails the finiteness check.
    res_inf = analyze_06a(wall_time_s=60.0, J_fan_steady_proxy=float("inf"))
    assert res_inf.j_fan_finite is False
    assert res_inf.passed is False


def test_06a_rejects_negative_wall_time() -> None:
    with pytest.raises(ValueError):
        analyze_06a(wall_time_s=-1.0, J_fan_steady_proxy=0.1)


def test_06a_stages_default_to_empty_dict() -> None:
    res = analyze_06a(wall_time_s=60.0, J_fan_steady_proxy=0.1)
    assert res.pipeline_stages == {}


# ----- sub-spike 0.6b ------------------------------------------------------


def test_06b_pass_within_5pct_tolerance() -> None:
    """Analytic 1e-3 m, measured 1.03e-3 m -> 3.0% -> pass.

    Pick E and I so that `P L^3 / (3 E I) = 1e-3 m` exactly:
      P = 3 N, L = 1 m, E = 1 Pa, I = 1000 m^4
      delta = 3 / (3 * 1000) = 1e-3 m
    Then measure 1.03e-3 m (3% over) -> within tolerance.
    """
    res = analyze_06b(
        wall_time_s=30.0,
        measured_deflection_m=1.03e-3,
        P_N=3.0,
        L_m=1.0,
        E_Pa=1.0,
        I_m4=1000.0,
    )
    assert math.isclose(res.analytic_tip_deflection_m, 1.0e-3, rel_tol=1e-12)
    assert math.isclose(res.tip_deflection_pct, 3.0, rel_tol=1e-9)
    assert res.tip_deflection_passed is True
    assert res.wall_time_passed is True
    assert res.passed is True


def test_06b_fail_if_over_5pct_off() -> None:
    """8% off the analytic value -> exceeds 5% tolerance -> fail."""
    res = analyze_06b(
        wall_time_s=30.0,
        measured_deflection_m=1.08e-3,
        P_N=3.0,
        L_m=1.0,
        E_Pa=1.0,
        I_m4=1000.0,
    )
    assert res.tip_deflection_pct > M3_FEA_TIP_DEFLECTION_TOLERANCE_PCT
    assert res.tip_deflection_passed is False
    assert res.wall_time_passed is True
    assert res.passed is False


def test_06b_fail_if_over_2min() -> None:
    """Wall-time above 2 min gate -> fail even with a tight tip-deflection."""
    res = analyze_06b(
        wall_time_s=M3_FEA_WALL_TIME_GATE_S + 1.0,
        measured_deflection_m=1.0e-3,
        P_N=3.0,
        L_m=1.0,
        E_Pa=1.0,
        I_m4=1000.0,
    )
    assert res.tip_deflection_passed is True
    assert res.wall_time_passed is False
    assert res.passed is False


def test_06b_rejects_nonfinite_measured() -> None:
    with pytest.raises(ValueError):
        analyze_06b(
            wall_time_s=30.0,
            measured_deflection_m=float("nan"),
            P_N=3.0,
            L_m=1.0,
            E_Pa=1.0,
            I_m4=1000.0,
        )


def test_06b_rejects_negative_wall_time() -> None:
    with pytest.raises(ValueError):
        analyze_06b(
            wall_time_s=-0.5,
            measured_deflection_m=1.0e-3,
            P_N=3.0,
            L_m=1.0,
            E_Pa=1.0,
            I_m4=1000.0,
        )


# ----- aggregator roll-up --------------------------------------------------


def test_spike_06_aggregates_correctly() -> None:
    """`analyze_spike_06` is calibration: overall_passed always True; sub-spike
    pass flags surface independently. Two budget rows roundtrip unchanged."""
    budget = [
        ComputeBudgetEntry(
            platform="colab_pro_cpu",
            workload="3d_unsteady_500k_5cycles_dtT200",
            wall_time_s=3600.0,
            cu_consumed=4.5,
            cells=500_000,
            notes="single-node, 8 vCPU",
        ),
        ComputeBudgetEntry(
            platform="colab_pro_g4_gpu",
            workload="3d_unsteady_500k_5cycles_dtT200",
            wall_time_s=900.0,
            cu_consumed=3.2,
            cells=500_000,
            notes="G4 instance, fp32",
        ),
    ]
    sub_a_pass = analyze_06a(wall_time_s=500.0, J_fan_steady_proxy=0.1)
    sub_b_fail = analyze_06b(
        wall_time_s=10.0,
        measured_deflection_m=1.5e-3,  # 50% off
        P_N=3.0,
        L_m=1.0,
        E_Pa=1.0,
        I_m4=1000.0,
    )

    result = analyze_spike_06(budget, sub_a_pass, sub_b_fail)
    assert isinstance(result, Spike06Result)
    assert result.overall_passed is True  # calibration — never fails
    assert len(result.budget_entries) == 2
    assert result.budget_entries[0].platform == "colab_pro_cpu"
    assert result.budget_entries[1].cu_consumed == pytest.approx(3.2)
    assert result.sub_06a is not None
    assert result.sub_06b is not None
    assert result.sub_06a.passed is True
    assert result.sub_06b.passed is False

    # Sub-spikes are optional — missing ones must not change overall_passed.
    skip_result = analyze_spike_06([], None, None)
    assert skip_result.overall_passed is True
    assert skip_result.sub_06a is None
    assert skip_result.sub_06b is None
    assert skip_result.budget_entries == ()


def test_subspike_dataclasses_frozen() -> None:
    """Sub-spike result dataclasses are frozen — protects audit log integrity."""
    res_a = analyze_06a(wall_time_s=60.0, J_fan_steady_proxy=0.1)
    res_b = analyze_06b(
        wall_time_s=30.0,
        measured_deflection_m=1.0e-3,
        P_N=3.0,
        L_m=1.0,
        E_Pa=1.0,
        I_m4=1000.0,
    )
    with pytest.raises((AttributeError, TypeError)):
        res_a.passed = False  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        res_b.passed = False  # type: ignore[misc]


def test_gates_locked_at_spec_values() -> None:
    """Constants match the spec literally (catch accidental edits)."""
    assert M3_SU2_WALL_TIME_GATE_S == 15 * 60
    assert M3_FEA_WALL_TIME_GATE_S == 2 * 60
    assert M3_FEA_TIP_DEFLECTION_TOLERANCE_PCT == 5.0


def test_subspike_result_types() -> None:
    """Sanity: analyze_06a / analyze_06b return the right dataclass types."""
    a = analyze_06a(wall_time_s=60.0, J_fan_steady_proxy=0.1)
    b = analyze_06b(
        wall_time_s=30.0,
        measured_deflection_m=1.0e-3,
        P_N=3.0,
        L_m=1.0,
        E_Pa=1.0,
        I_m4=1000.0,
    )
    assert isinstance(a, SubSpike06AResult)
    assert isinstance(b, SubSpike06BResult)
