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

from fanopt.cfd import configs as cfg
from fanopt.cfd.configs import (
    CFD_TEMPLATES_DIR,
    CROSS_TIER,
    FREESTREAM_DIRECTION_2D_PRODUCTIVE,
    FREESTREAM_DIRECTION_2D_RETURN,
    MACH_STEADY,
    MACH_UNSTEADY,
    REYNOLDS_NUMBER_GLOBAL,
    TIER_SPECIFIC,
    TemplateRenderError,
    render_benchmark_cfg,
    render_slice_steady_cfg,
    render_slice_unsteady_cfg,
    render_steady_cfg,
    render_thin_plate_2d_pitching_cfg,
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


def test_unsteady_renders_ref_dimensionalization_fallback(unsteady_render: str) -> None:
    """Round-9 HIGH-12 fallback path — REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE.

    The primary `FREESTREAM_OPTION = FREESTREAM_VELOCITY` path is unusable
    on SU2 v8.0.1 (parser rejects the directive), so the template ships
    the fallback path. The plan accepts either; this test pins the
    template to the working one.
    """
    assert "REF_DIMENSIONALIZATION= FREESTREAM_PRESS_EQ_ONE" in unsteady_render
    # The primary directive line must NOT appear as an SU2 directive
    # (would cause SU2 v8.0.1 parse error). Mentions in % comments are
    # fine — strip them before checking.
    directives_only = "\n".join(
        line for line in unsteady_render.splitlines() if not line.lstrip().startswith("%")
    )
    assert re.search(r"^\s*FREESTREAM_OPTION\s*=", directives_only, re.MULTILINE) is None


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
        render_unsteady_cfg(mesh_filename="x.su2", pitching_omega_y=+12.5664)


def test_unsteady_allows_zero_omega_for_benchmark() -> None:
    """ω = 0 is permitted (for static-airfoil sanity checks)."""
    out = render_unsteady_cfg(mesh_filename="x.su2", pitching_omega_y=0.0)
    assert "PITCHING_OMEGA= 0.0 0.0 0.0" in out


def test_unsteady_n_cycles_propagates() -> None:
    out = render_unsteady_cfg(mesh_filename="x.su2", n_cycles=8)
    m = re.search(r"^TIME_ITER=\s*(\d+)", out, re.MULTILINE)
    assert m is not None
    assert int(m.group(1)) == 1600
    mt = re.search(r"^MAX_TIME=\s*([\d.eE+-]+)", out, re.MULTILINE)
    assert mt is not None
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


def test_benchmark_renders_wind_tunnel_frame() -> None:
    """Wind-tunnel-frame NACA 0012 template renders with MACH > 0 + freestream ON.

    Phase-5 prep: NOT the Tier-1 MACH = 1e-9 lock. The whole point of
    the 2026-05-14 deferral is that the benchmark must run in the
    conventional wind-tunnel frame, not body-in-still-air.
    """
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
    assert "MACH_NUMBER= 0.05" in out
    # The body-in-still-air directive (Tier-1 production fallback) MUST
    # be absent — its presence is the conceptual bug the 2026-05-14
    # diagnostic invalidated.
    assert "REF_DIMENSIONALIZATION" not in out
    assert "DUAL_TIME_STEPPING-2ND_ORDER" in out
    assert "RIGID_MOTION" in out
    # Freestream defaults sane for low-Re aerodynamic flow.
    assert "FREESTREAM_TEMPERATURE= 300.0" in out
    assert "FREESTREAM_PRESSURE= 101325.0" in out
    assert "LOW_MACH_PREC= YES" in out


def test_benchmark_mach_number_parameterizable() -> None:
    """``mach_number`` parameter overrides the default 0.05."""
    out = render_benchmark_cfg(
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
        mach_number=0.1,
    )
    assert "MACH_NUMBER= 0.1" in out
    assert "MACH_NUMBER= 0.05" not in out


def test_benchmark_freestream_state_parameterizable() -> None:
    """Freestream temperature + pressure can be overridden for non-STP cases."""
    out = render_benchmark_cfg(
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
        freestream_temperature=288.15,
        freestream_pressure=95000.0,
    )
    assert "FREESTREAM_TEMPERATURE= 288.15" in out
    assert "FREESTREAM_PRESSURE= 95000.0" in out


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


# ---- render_slice_steady (Tier -1, 2D mid-radius slice) -------------------


def test_slice_steady_renders_mach_0_0064() -> None:
    """Tier -1 uses the same MACH lock as Tier 0 (steady tiers, V_tip-based)."""
    out = render_slice_steady_cfg(mesh_filename="slice.su2")
    assert "MACH_NUMBER= 0.0064" in out


def test_slice_steady_default_freestream_is_productive_2d() -> None:
    """Default freestream PRODUCTIVE = (-1, 0) => flow in -x => AOA 180 (C2)."""
    out = render_slice_steady_cfg(mesh_filename="slice.su2")
    assert "AOA= 180.0" in out


def test_slice_steady_return_stroke() -> None:
    """RETURN-stroke half of the two-eval delta: 2D = (+1, 0) => AOA 0."""
    out = render_slice_steady_cfg(
        mesh_filename="slice.su2",
        freestream_direction=FREESTREAM_DIRECTION_2D_RETURN,
    )
    assert "AOA= 0.0" in out


def test_slice_steady_time_domain_no() -> None:
    """Tier -1 is steady — no time integration."""
    out = render_slice_steady_cfg(mesh_filename="slice.su2")
    assert "TIME_DOMAIN= NO" in out


def test_slice_steady_uses_aoa_not_freestream_direction() -> None:
    """SU2 8.0.1 has no FREESTREAM_DIRECTION option — the 2D slice sets the flow
    direction via AOA, uses a valid FREESTREAM_OPTION, and low-Mach prec."""
    out = render_slice_steady_cfg(mesh_filename="slice.su2")
    assert re.search(r"^AOA=\s*\S+", out, re.MULTILINE) is not None
    assert re.search(r"^FREESTREAM_DIRECTION=", out, re.MULTILINE) is None
    assert "FREESTREAM_OPTION= TEMPERATURE_FS" in out
    assert "LOW_MACH_PREC= YES" in out


def test_slice_steady_rejects_3d_freestream() -> None:
    """A 3-vector freestream is a category error for the 2D slice cfg."""
    with pytest.raises(TemplateRenderError, match="2-vector"):
        render_slice_steady_cfg(
            mesh_filename="slice.su2",
            freestream_direction=(1.0, 0.0, 0.0),  # type: ignore[arg-type]
        )


def test_slice_steady_marker_propagation() -> None:
    out = render_slice_steady_cfg(
        mesh_filename="slice.su2",
        marker_fan="BLADE_2D",
        marker_farfield="FAR_2D",
    )
    assert "MARKER_HEATFLUX= ( BLADE_2D, 0.0 )" in out
    assert "MARKER_FAR= ( FAR_2D )" in out


def test_slice_steady_reynolds_default_matches_global() -> None:
    """Default Reynolds is the §3.2.3 global Re."""
    out = render_slice_steady_cfg(mesh_filename="slice.su2")
    assert f"REYNOLDS_NUMBER= {REYNOLDS_NUMBER_GLOBAL}" in out


def test_slice_steady_2d_freestream_constants_are_unit_vectors() -> None:
    """Sanity-check the locked 2D direction constants."""
    pr = FREESTREAM_DIRECTION_2D_PRODUCTIVE
    rt = FREESTREAM_DIRECTION_2D_RETURN
    assert len(pr) == 2 and len(rt) == 2
    assert pr[0] ** 2 + pr[1] ** 2 == 1.0
    assert rt[0] ** 2 + rt[1] ** 2 == 1.0
    # PRODUCTIVE and RETURN are opposite-direction.
    assert pr[0] == -rt[0]
    assert pr[1] == -rt[1]


def test_slice_steady_template_not_found(monkeypatch, tmp_path) -> None:
    """Missing template raises TemplateRenderError."""
    monkeypatch.setattr(cfg, "CFD_TEMPLATES_DIR", tmp_path)
    with pytest.raises(TemplateRenderError, match="template not found"):
        render_slice_steady_cfg(mesh_filename="slice.su2")


# ---- templates exist ------------------------------------------------------


def test_unsteady_template_file_exists() -> None:
    assert (CFD_TEMPLATES_DIR / "fan3d_unsteady.cfg.j2").exists()


def test_benchmark_template_file_exists() -> None:
    assert (CFD_TEMPLATES_DIR / "oscillating_airfoil_benchmark.cfg.j2").exists()


# ---- render_thin_plate_2d_pitching (Spike 0.6d.2 H10 supplement) ---------


@pytest.fixture
def thin_plate_render() -> str:
    return render_thin_plate_2d_pitching_cfg(
        mesh_filename="thin_plate_2d.su2",
        marker_plate="PLATE",
        marker_farfield="FARFIELD",
        pitching_omega_z=-12.5664,  # C11 analog for 2D x-y mesh (pitch about z)
        pitching_ampl_z=0.1745,
        motion_origin_x=0.25,
        time_step=2.5e-3,
        max_time=2.5,
        time_iter=1000,
    )


def test_thin_plate_2d_cfg_mirrors_production_tier1_numerics(
    thin_plate_render: str,
) -> None:
    """0.6d.2 cfg MUST match production Tier-1 numerics — that's the point."""
    assert "MACH_NUMBER= 1e-9" in thin_plate_render
    assert "REF_DIMENSIONALIZATION= FREESTREAM_PRESS_EQ_ONE" in thin_plate_render
    assert "LOW_MACH_PREC= YES" in thin_plate_render
    assert "DUAL_TIME_STEPPING-2ND_ORDER" in thin_plate_render
    assert "RIGID_MOTION" in thin_plate_render


def test_thin_plate_2d_cfg_pitching_about_quarter_chord(
    thin_plate_render: str,
) -> None:
    assert "MOTION_ORIGIN= 0.25 0.0 0.0" in thin_plate_render


def test_thin_plate_2d_cfg_omega_z_negative_per_c11_analog(
    thin_plate_render: str,
) -> None:
    """C11 analog for 2D x-y mesh — z-component of PITCHING_OMEGA must be negative."""
    m = re.search(
        r"^PITCHING_OMEGA=\s*([\d.+-eE]+)\s+([\d.+-eE]+)\s+([\d.+-eE]+)",
        thin_plate_render,
        re.MULTILINE,
    )
    assert m is not None
    _, _, z = (float(g) for g in m.groups())
    assert z < 0


def test_thin_plate_2d_cfg_renders_with_defaults() -> None:
    """Render should succeed with default reynolds/inner_iter/cfl."""
    out = render_thin_plate_2d_pitching_cfg(
        mesh_filename="x.su2",
        marker_plate="P",
        marker_farfield="F",
        pitching_omega_z=-10.0,
        pitching_ampl_z=0.1,
        motion_origin_x=0.25,
        time_step=1e-3,
        max_time=1.0,
        time_iter=1000,
    )
    assert "REYNOLDS_NUMBER= 40000" in out
    assert "INNER_ITER= 100" in out


def test_thin_plate_2d_cfg_rejects_positive_omega_per_c11_analog() -> None:
    """C11 analog — the renderer refuses to flip the z-component sign."""
    with pytest.raises(TemplateRenderError, match="C11"):
        render_thin_plate_2d_pitching_cfg(
            mesh_filename="x.su2",
            marker_plate="P",
            marker_farfield="F",
            pitching_omega_z=+12.5664,  # wrong sign
            pitching_ampl_z=0.1,
            motion_origin_x=0.25,
            time_step=1e-3,
            max_time=1.0,
            time_iter=1000,
        )


def test_thin_plate_2d_template_file_exists() -> None:
    assert (CFD_TEMPLATES_DIR / "thin_plate_2d_pitching.cfg.j2").exists()


def test_slice_steady_template_file_exists() -> None:
    assert (CFD_TEMPLATES_DIR / "slice_steady.cfg.j2").exists()


# ---- render_slice_unsteady (Tier 1, 2D plunging slice) --------------------


def test_slice_unsteady_renders_mach_1e_minus_9() -> None:
    out = render_slice_unsteady_cfg(mesh_filename="slice.su2")
    m = re.search(r"^MACH_NUMBER=\s*([\dEe.+-]+)", out, re.MULTILINE)
    assert m and float(m.group(1)) == pytest.approx(1e-9)


def test_slice_unsteady_is_dual_time_stepping() -> None:
    out = render_slice_unsteady_cfg(mesh_filename="slice.su2")
    assert "DUAL_TIME_STEPPING-2ND_ORDER" in out
    assert re.search(r"^TIME_DOMAIN=\s*YES", out, re.MULTILINE)


def test_slice_unsteady_plunging_omega_propagates() -> None:
    out = render_slice_unsteady_cfg(mesh_filename="slice.su2", plunging_omega=7.5)
    assert re.search(r"^PLUNGING_OMEGA=\s*7\.5\b", out, re.MULTILINE)


def test_slice_unsteady_rigid_motion() -> None:
    out = render_slice_unsteady_cfg(mesh_filename="slice.su2")
    assert re.search(r"^GRID_MOVEMENT=\s*RIGID_MOTION", out, re.MULTILINE)


def test_slice_unsteady_marker_propagation() -> None:
    out = render_slice_unsteady_cfg(
        mesh_filename="slice.su2", marker_fan="fan_surface", marker_cascade="cascade_wall"
    )
    assert "fan_surface" in out
    assert re.search(r"^MARKER_SYM=\s*\(\s*cascade_wall", out, re.MULTILINE)


def test_slice_unsteady_rejects_nonpositive_amplitude() -> None:
    with pytest.raises(TemplateRenderError, match="plunging_ampl must be > 0"):
        render_slice_unsteady_cfg(mesh_filename="slice.su2", plunging_ampl=0.0)


def test_slice_unsteady_template_file_exists() -> None:
    assert (CFD_TEMPLATES_DIR / "slice_unsteady.cfg.j2").exists()
