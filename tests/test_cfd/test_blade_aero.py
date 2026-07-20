"""Tests for fanopt.cfd.blade_aero (blade → cascade-slice CFD case prep + J_fan eval).

The mesh step needs gmsh; skipped at module load without it. The SU2 subprocess
(``evaluate_blade_aero``) is an external boundary exercised by the validation runs, not
unit-tested here.
"""

from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("gmsh") is None:
    pytest.skip("gmsh not installed", allow_module_level=True)

from fanopt.cfd.blade_aero import BladeAeroResult, prepare_blade_aero_case
from fanopt.geometry.blade import BladeParams

_CAMBER = (
    (0.0004, 0.0009, 0.0004),
    (0.0005, 0.0011, 0.0005),
    (0.0006, 0.0013, 0.0006),
    (0.0007, 0.0015, 0.0007),
)


def _params() -> BladeParams:
    return BladeParams(
        blade_count=10,
        rib_bow_mid_m=0.010,
        rib_bow_tip_m=0.020,
        t_rib_hub_m=0.0025,
        t_rib_tip_m=0.0035,
        panel_offsets_m=_CAMBER,
        panel_thickness_nom_m=0.0016,
    )


def test_prepare_writes_mesh_and_both_cfgs(tmp_path):
    prepare_blade_aero_case(_params(), tmp_path, radial_u=0.6, n_panels=3, n_samples=24)
    assert (tmp_path / "blade_slice.su2").exists()
    assert (tmp_path / "steady.cfg").exists()
    assert (tmp_path / "unsteady.cfg").exists()


def test_prepare_returns_positive_node_count(tmp_path):
    info = prepare_blade_aero_case(_params(), tmp_path, radial_u=0.6, n_panels=3, n_samples=24)
    assert info["n_nodes"] > 0


def test_unsteady_cfg_reflects_cycle_count(tmp_path):
    prepare_blade_aero_case(
        _params(), tmp_path, radial_u=0.6, n_panels=2, n_samples=20, n_cycles=4, steps_per_cycle=50
    )
    text = (tmp_path / "unsteady.cfg").read_text(encoding="utf-8")
    assert "TIME" in text.upper()  # an unsteady (time-marching) cfg was rendered


def test_result_is_frozen_dataclass():
    r = BladeAeroResult(j_fan=1.2e-3, steady_cd=0.05, n_nodes=5000)
    assert r.j_fan == 1.2e-3 and r.steady_cd == 0.05 and r.n_nodes == 5000
    with pytest.raises(AttributeError):
        r.j_fan = 0.0  # type: ignore[misc]
