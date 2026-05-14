"""Tests for fanopt.cfd.configs.

Validates the Tier-1 / Tier-0 / benchmark renderers + the cross-tier vs
tier-specific separation (HIGH-12 lock). The actual rendered config text
is checked for key invariants — MACH value, FREESTREAM_OPTION, C11 omega
sign — rather than full string equality (which would brittle-break on
formatting tweaks).
"""
from __future__ import annotations

import re

import pytest

from fanopt import cfd as fanopt_cfd  # for monkeypatch.setattr(cfg, ...)
from fanopt.cfd import configs as cfg
from fanopt.cfd.configs import (
    CFD_TEMPLATES_DIR,
    CROSS_TIER,
    MACH_STEADY,
    MACH_UNSTEADY,
    REYNOLDS_NUMBER_GLOBAL,
    TIER_SPECIFIC,
    TemplateRenderError,
    render_benchmark_cfg,
    render_steady_cfg,
    render_unsteady_cfg,
)


# ---- cross-tier / tier-specific separation (HIGH-12 lock) -----------------


def test_cross_tier_does_not_carry_mach() -> None:
    """Plan: 'CROSS_TIER dict does NOT carry MACH.' (HIGH-12 Round-9)"""
    assert "mach_number" not in CROSS_TIER
    assert "MACH_NUMBER" not in CROSS_TIER


def test_tier_specific_mach_values() -> None:
    """Steady tiers use 0.0064 (V_tip-based); unsteady tier uses 1e-9."""
    assert TIER_SPECIFIC[-1]["mach_number"] == MACH_STEADY
    assert TIER_SPECIFIC[0]["mach_number"] == MACH_STEADY
    assert TIER_SPECIFIC[1]["mach_number"] == MACH_UNSTEADY


def test_mach_unsteady_is_1e_minus_9() -> None:
    """Production Tier-1 lock — NOT 0.0064."""
    assert MACH_UNSTEADY == 1e-9


def test_mach_steady_is_0_0064() -> None:
    """Steady tiers use V_tip = 2.20 m/s freestream → Mach 0.0064."""
    assert MACH_STEADY == 0.0064


def test_tier_1_uses_dual_time_stepping() -> None:
    assert TIER_SPECIFIC[1]["time_marching"] == "DUAL_TIME_STEPPING-2ND_ORDER"
    assert TIER_SPECIFIC[1]["time_domain"] == "YES"


def test_tier_1_pitching_omega_y_negative() -> None:
    """C11 sign lock — right-hand-rule on productive stroke."""
    assert TIER_SPECIFIC[1]["pitching_omega_y"] < 0


def test_tier_1_pitching_ampl_y_positive() -> None:
    """θ_max amplitude is +y."""
    assert TIER_SPECIFIC[1]["pitching_ampl_y"] > 0


def test_steady_tiers_have_time_domain_no() -> None:
    """Steady ≠ unsteady — TIME_DOMAIN must be NO."""
    assert TIER_SPECIFIC[-1]["time_domain"] == "NO"
    assert TIER_SPECIFIC[0]["time_domain"] == "NO"


def test_global_reynolds_number_locked() -> None:
    assert REYNOLDS_NUMBER_GLOBAL == 37000.0


# ---- unsteady cfg renderer ------------------------------------------------


@pytest.fixture
def unsteady_render() -> str:
    return render_unsteady_cfg(mesh_filename="fan3d.su2")


def test_unsteady_renders_mach_1e_minus_9(unsteady_render: str) -> None:
    """HIGH-12 lock — render must set MACH_NUMBER to 1e-9, not 0.0064."""
    m = re.search(r"^MACH_NUMBER=\s*([\dEe.+-]+)", unsteady_render, re.MULTILINE)
    assert m is not None
    assert float(m.group(1)) == 1e-9


def test_unsteady_renders_freestream_velocity_option(unsteady_render: str) -> None:
    """HIGH-12 primary path — FREESTREAM_OPTION = FREESTREAM_VELOCITY."""
    assert "FREESTREAM_OPTION= FREESTREAM_VELOCITY" in unsteady_render


def test_unsteady_renders_low_mach_prec_yes(unsteady_render: str) -> None:
    assert re.search(r"^LOW_MACH_PREC=\s*YES", unsteady_render, re.MULTILINE)


def test_unsteady_renders_dual_time_stepping(unsteady_render: str) -> None:
    assert "DUAL_TIME_STEPPING-2ND_ORDER" in unsteady_render


def test_unsteady_renders_rigid_motion(unsteady_render: str) -> None:
    assert "GRID_MOVEMENT= RIGID_MOTION" in unsteady_render


def test_unsteady_renders_c11_negative_y_omega(unsteady_render: str) -> None:
    """C11 lock: middle component of PITCHING_OMEGA must be negative."""
    m = re.search(
        r"^PITCHING_OMEGA=\s*([\d.+-eE]+)\s+([\d.+-eE]+)\s+([\d.+-eE]+)",
        unsteady_render,
        re.MULTILINE,
    )
    assert m is not None, "PITCHING_OMEGA not found in rendered cfg"
    _, y, _ = (float(g) for g in m.groups())
    assert y < 0, f"y-component must be negative per C11 lock, got {y}"


def test_unsteady_renders_positive_y_amplitude(unsteady_render: str) -> None:
    m = re.search(
        r"^PITCHING_AMPL=\s*([\d.+-eE]+)\s+([\d.+-eE]+)\s+([\d.+-eE]+)",
        unsteady_render,
        re.MULTILINE,
    )
    assert m is not None
    _, y, _ = (float(g) for g in m.groups())
    assert y > 0


def test_unsteady_dt_is_T_over_200(unsteady_render: str) -> None:
    """Plan: dt = T/200 = 2.5 ms with T = 0.5 s."""
    m = re.search(r"^TIME_STEP=\s*([\d.eE+-]+)", unsteady_render, re.MULTILINE)
    assert m is not None
    dt = float(m.group(1))
    assert dt == pytest.approx(2.5e-3, rel=1e-6)


def test_unsteady_5_cycle_max_time(unsteady_render: str) -> None:
    m = re.search(r"^MAX_TIME=\s*([\d.eE+-]+)", unsteady_render, re.MULTILINE)
    assert m is not None
    assert float(m.group(1)) == pytest.approx(2.5, rel=1e-6)


def test_unsteady_1000_outer_steps(unsteady_render: str) -> None:
    m = re.search(r"^TIME_ITER=\s*(\d+)", unsteady_render, re.MULTILINE)
    assert m is not None
    assert int(m.group(1)) == 1000


def test_unsteady_renders_reynolds_global() -> None:
    out = render_unsteady_cfg(mesh_filename="x.su2")
    m = re.search(r"^REYNOLDS_NUMBER=\s*([\d.eE+-]+)", out, re.MULTILINE)
    assert m is not None
    assert float(m.group(1)) == REYNOLDS_NUMBER_GLOBAL


def test_unsteady_rejects_positive_y_omega() -> None:
    """C11 sign lock — operator override that flips the sign must fail."""
    with pytest.raises(TemplateRenderError, match="C11"):
        render_unsteady_cfg(
            mesh_filename="x.su2", pitching_omega_y=+12.5664
        )


def test_unsteady_allows_zero_omega_for_benchmark() -> None:
    """ω = 0 is permitted (for static-airfoil sanity checks)."""
    out = render_unsteady_cfg(mesh_filename="x.su2", pitching_omega_y=0.0)
    assert "PITCHING_OMEGA= 0.0 0.0 0.0" in out


def test_unsteady_n_cycles_propagates() -> None:
    out = render_unsteady_cfg(mesh_filename="x.su2", n_cycles=8)
    m = re.search(r"^TIME_ITER=\s*(\d+)", out, re.MULTILINE)
    assert int(m.group(1)) == 1600
    mt = re.search(r"^MAX_TIME=\s*([\d.eE+-]+)", out, re.MULTILINE)
    assert float(mt.group(1)) == pytest.approx(4.0, rel=1e-6)


# ---- benchmark cfg renderer ----------------------------------------------


# ---- steady cfg renderer --------------------------------------------------


def test_render_steady_default_renders() -> None:
    """Default render must succeed and emit MACH = 0.0064."""
    out = render_steady_cfg(mesh_filename="fan3d.su2")
    assert "MACH_NUMBER= 0.0064" in out


def test_render_steady_default_freestream_is_productive() -> None:
    """Default direction is C2 PRODUCTIVE (0, 0, -1)."""
    out = render_steady_cfg(mesh_filename="fan3d.su2")
    assert "FREESTREAM_DIRECTION= 0.0 0.0 -1.0" in out


def test_render_steady_return_stroke() -> None:
    """Operator can switch to RETURN (0, 0, +1) for the two-eval delta."""
    out = render_steady_cfg(
        mesh_filename="fan3d.su2",
        freestream_direction=(0.0, 0.0, +1.0),
    )
    assert "FREESTREAM_DIRECTION= 0.0 0.0 1.0" in out


def test_render_steady_time_domain_no() -> None:
    out = render_steady_cfg(mesh_filename="x.su2")
    assert "TIME_DOMAIN= NO" in out


def test_steady_template_file_exists() -> None:
    assert (CFD_TEMPLATES_DIR / "fan3d_steady.cfg.j2").exists()


# ---- error paths ----------------------------------------------------------


def test_render_unsteady_template_not_found(monkeypatch, tmp_path) -> None:
    """If the template file is missing, raise TemplateRenderError cleanly."""
    monkeypatch.setattr(cfg, "CFD_TEMPLATES_DIR", tmp_path)
    with pytest.raises(TemplateRenderError, match="template not found"):
        cfg.render_unsteady_cfg(mesh_filename="x.su2")


def test_render_steady_template_not_found(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cfg, "CFD_TEMPLATES_DIR", tmp_path)
    with pytest.raises(TemplateRenderError, match="template not found"):
        cfg.render_steady_cfg(mesh_filename="x.su2")


def test_render_benchmark_template_not_found(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cfg, "CFD_TEMPLATES_DIR", tmp_path)
    with pytest.raises(TemplateRenderError, match="template not found"):
        cfg.render_benchmark_cfg(
            mesh_filename="x.su2",
            marker_airfoil="A",
            marker_farfield="F",
            reynolds_number=40000,
            reynolds_length=1.0,
            pitching_omega_y=-3.0,
            pitching_ampl_y=0.1,
            motion_origin_x=0.25,
            time_step=0.001,
            max_time=5.0,
            time_iter=5000,
        )


def test_benchmark_renders_naca_0012_style() -> None:
    """Spike 0.6c.2 NACA 0012 template renders with the same HIGH-12 locks."""
    out = render_benchmark_cfg(
        mesh_filename="naca0012.su2",
        marker_airfoil="AIRFOIL",
        marker_farfield="FARFIELD",
        reynolds_number=40000,
        reynolds_length=1.0,
        pitching_omega_y=-3.45,  # negative — same C11 convention
        pitching_ampl_y=0.1745,
        motion_origin_x=0.25,
        time_step=0.001,
        max_time=5.0,
        time_iter=5000,
    )
    assert "MACH_NUMBER= 1e-9" in out
    assert "FREESTREAM_OPTION= FREESTREAM_VELOCITY" in out
    assert "DUAL_TIME_STEPPING-2ND_ORDER" in out
    assert "RIGID_MOTION" in out


def test_benchmark_marker_airfoil_propagates() -> None:
    out = render_benchmark_cfg(
        mesh_filename="x.su2",
        marker_airfoil="MY_AIRFOIL",
        marker_farfield="MY_FARFIELD",
        reynolds_number=40000,
        reynolds_length=1.0,
        pitching_omega_y=-3.0,
        pitching_ampl_y=0.1,
        motion_origin_x=0.25,
        time_step=0.001,
        max_time=5.0,
        time_iter=5000,
    )
    assert "MARKER_HEATFLUX= ( MY_AIRFOIL, 0.0 )" in out
    assert "MARKER_FAR= ( MY_FARFIELD )" in out


# ---- templates exist ------------------------------------------------------


def test_unsteady_template_file_exists() -> None:
    assert (CFD_TEMPLATES_DIR / "fan3d_unsteady.cfg.j2").exists()


def test_benchmark_template_file_exists() -> None:
    assert (CFD_TEMPLATES_DIR / "oscillating_airfoil_benchmark.cfg.j2").exists()
