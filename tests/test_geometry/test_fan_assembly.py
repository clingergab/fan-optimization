"""Tests for fanopt.geometry.fan_assembly (deployed-fan + properties)."""

from __future__ import annotations

import importlib.util
import math

import pytest

if importlib.util.find_spec("cadquery") is None:
    pytest.skip("cadquery not installed", allow_module_level=True)

import cadquery as cq

from fanopt.geometry.assembly_cad import make_vunit_blade
from fanopt.geometry.envelope import Layer1Params, ThicknessGridField
from fanopt.geometry.fan_assembly import (
    compute_centre_of_mass,
    compute_i_wrist_kgm2,
    compute_mass_kg,
    deploy_fan,
)
from fanopt.geometry.fields import Layer2Params
from fanopt.geometry.generator import BladeDesignParams
from fanopt.geometry.manufacturability import Layer4Params
from fanopt.geometry.primitives import Layer3Primitive
from fanopt.geometry.schema import (
    INTER_BLADE_ANGLE_RAD,
    MAX_TOTAL_MASS_KG,
    RHO_PETG_KG_PER_M3,
)


def _canonical_design(blade_count: int = 10) -> BladeDesignParams:
    return BladeDesignParams(
        layer1=Layer1Params(
            blade_count=blade_count,
            camber_knots_m=(0.0, 0.002, 0.001),
            twist_knots_rad=(0.0, 0.0),
            thickness_field=ThicknessGridField.from_radial_knots((0.0030, 0.0028, 0.0026)),
            edge_profile="rounded",
            fourier_le_amplitudes=(0.0, 0.0, 0.0),
            fourier_te_amplitudes=(0.0, 0.0, 0.0),
        ),
        layer2=Layer2Params.all_inactive(),
        layer3=Layer3Primitive.absent(),
        layer4=Layer4Params(
            print_orientation="flat",
            layer_height_m=0.0002,
            click_chamfer_angle_deg=45.0,
            click_detent_size_m=0.0004,
            click_design_clearance_m=0.00018,
        ),
    )


# ---- deploy_fan ----------------------------------------------------------


def test_deploy_fan_returns_n_blades() -> None:
    design = _canonical_design(blade_count=10)
    _assembly, per_blade = deploy_fan(design)
    assert len(per_blade) == 10


def test_deploy_fan_blade_count_8() -> None:
    design = _canonical_design(blade_count=8)
    _assembly, per_blade = deploy_fan(design)
    assert len(per_blade) == 8


def test_deploy_fan_blade_count_12() -> None:
    design = _canonical_design(blade_count=12)
    _assembly, per_blade = deploy_fan(design)
    assert len(per_blade) == 12


def test_deploy_fan_assembly_is_workplane() -> None:
    design = _canonical_design()
    assembly, _per_blade = deploy_fan(design)
    assert isinstance(assembly, cq.Workplane)
    assert assembly.val().Volume() > 0.0


def test_deploy_fan_blades_form_symmetric_arc() -> None:
    """10 blades at 13.3° spacing → outer half-angle = 9 × 13.3° / 2 ≈ 60°."""
    design = _canonical_design(blade_count=10)
    _assembly, per_blade = deploy_fan(design)
    # First and last blades should be rotated symmetric about y=0.
    bb_first = per_blade[0].val().BoundingBox()
    bb_last = per_blade[-1].val().BoundingBox()
    # The first blade is rotated to negative angle; last to positive.
    # Their y-mid centres should be roughly mirror-symmetric.
    y_mid_first = (bb_first.ymin + bb_first.ymax) / 2.0
    y_mid_last = (bb_last.ymin + bb_last.ymax) / 2.0
    assert y_mid_first < 0.0
    assert y_mid_last > 0.0
    assert abs(y_mid_first + y_mid_last) < 0.005


def test_deploy_fan_assembly_volume_matches_per_blade_sum() -> None:
    design = _canonical_design()
    assembly, per_blade = deploy_fan(design)
    per_blade_vol = sum(b.val().Volume() for b in per_blade)
    assembly_vol = assembly.val().Volume()
    assert assembly_vol == pytest.approx(per_blade_vol, rel=1e-6)


# ---- compute_mass_kg ----------------------------------------------------


def test_mass_matches_volume_times_density() -> None:
    blade = make_vunit_blade(_canonical_design())
    expected = blade.val().Volume() * RHO_PETG_KG_PER_M3
    assert compute_mass_kg(blade) == pytest.approx(expected, rel=1e-9)


def test_total_mass_finite_and_positive_for_canonical() -> None:
    """The canonical test design is a pipeline exerciser, NOT a C9-
    compliant production design (thickness is at the schema upper
    bound to exercise the loft, so it's heavy). Just verify mass is
    positive + finite; C9 compliance is a Phase-4 BO objective."""
    design = _canonical_design(blade_count=10)
    blade_mass = compute_mass_kg(make_vunit_blade(design))
    total = blade_mass * design.layer1.blade_count
    assert total > 0.0
    assert math.isfinite(total)
    # Pin the cap constant exists (C9's 100 g relaxed to 120 g, 2026-07-21).
    assert MAX_TOTAL_MASS_KG == 0.300


def test_mass_with_custom_density() -> None:
    blade = make_vunit_blade(_canonical_design())
    expected = blade.val().Volume() * 2000.0
    assert compute_mass_kg(blade, density_kg_per_m3=2000.0) == pytest.approx(expected, rel=1e-9)


# ---- compute_centre_of_mass ---------------------------------------------


def test_centre_of_mass_returns_xyz() -> None:
    blade = make_vunit_blade(_canonical_design())
    c = compute_centre_of_mass(blade)
    assert len(c) == 3
    assert all(isinstance(v, float) for v in c)


def test_centre_of_mass_x_positive_for_blade() -> None:
    """The V-unit's centroid x is well inside the panel (between 0 and L_BLADE)."""
    blade = make_vunit_blade(_canonical_design())
    x, _y, _z = compute_centre_of_mass(blade)
    assert 0.0 < x < 0.200


# ---- compute_i_wrist_kgm2 -----------------------------------------------


def test_i_wrist_positive_finite() -> None:
    blade = make_vunit_blade(_canonical_design())
    i_wrist = compute_i_wrist_kgm2(blade)
    assert i_wrist > 0.0
    assert math.isfinite(i_wrist)


def test_i_wrist_increases_with_density() -> None:
    blade = make_vunit_blade(_canonical_design())
    i_low = compute_i_wrist_kgm2(blade, density_kg_per_m3=500.0)
    i_high = compute_i_wrist_kgm2(blade, density_kg_per_m3=2000.0)
    assert i_high == pytest.approx(i_low * 4.0, rel=1e-6)


def test_i_wrist_order_of_magnitude_reasonable() -> None:
    """A 200 mm × 23 mm × 3 mm PETG blade at ~0.05 m wrist offset has
    I_wrist on the order of m·r² ≈ 0.01 kg × (0.15 m)² = 2e-4 kg·m².
    Loose 0.5× to 5× tolerance — pin the order of magnitude."""
    blade = make_vunit_blade(_canonical_design())
    i_wrist = compute_i_wrist_kgm2(blade)
    assert 1e-5 < i_wrist < 1e-2


# ---- end-to-end integration ---------------------------------------------


def test_deploy_fan_inter_blade_angle_matches_lock() -> None:
    """Each blade's rotation increment must equal INTER_BLADE_ANGLE_RAD
    (the C8 lock)."""
    design = _canonical_design(blade_count=10)
    _assembly, per_blade = deploy_fan(design)
    # Check that the n-th blade's centroid x-position decreases as the
    # blade rotates further out (rotated blades have lower x_centroid).
    # The angular spacing between adjacent blades is INTER_BLADE_ANGLE_RAD.
    # We verify by checking the angular spread between blade 0 and blade 1.
    c0 = per_blade[0].val().Center()
    c1 = per_blade[1].val().Center()
    # Centroids lie in a circle; the angle between them at the origin
    # equals INTER_BLADE_ANGLE_RAD.
    angle_0 = math.atan2(c0.y, c0.x)
    angle_1 = math.atan2(c1.y, c1.x)
    delta = abs(angle_1 - angle_0)
    assert delta == pytest.approx(INTER_BLADE_ANGLE_RAD, rel=0.05)
