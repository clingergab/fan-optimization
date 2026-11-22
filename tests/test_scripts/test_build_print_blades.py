"""Tests for scripts/build_print_blades.py (TO density → on-edge print STL + report + Bambu project).

End-to-end on a synthetic 1-design campaign at a coarse mesh and voxel pitch. Requires gmsh + CadQuery
(the TO mesh), scikit-image and manifold3d.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

for _dep in ("gmsh", "cadquery", "skimage", "manifold3d"):
    if importlib.util.find_spec(_dep) is None:  # pragma: no cover - env-dependent
        pytest.skip(f"{_dep} not installed", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import build_print_blades  # noqa: E402
from fanopt.bo.blade_codec import bounds, clip_to_bounds, decode  # noqa: E402
from fanopt.geometry.schema import RHO_PETG_KG_PER_M3, RHO_PLA_KG_PER_M3  # noqa: E402
from fanopt.utils.ledger import design_hash  # noqa: E402


def _build(tmp: Path, *extra: str, mesh_mm: str = "6") -> tuple[int, Path]:
    low, high = bounds()
    vec = clip_to_bounds(low + (high - low) * 0.5)
    h = design_hash(decode(vec).to_dict())
    (tmp / "evaluations_c0.jsonl").write_text(
        json.dumps({"design_hash": h, "vector": vec.tolist(), "j_fan": 1.0}) + "\n"
    )
    name = f"00_{h}"
    np.save(tmp / f"{name}_centroids.npy", np.zeros((1, 3)))
    np.save(tmp / f"{name}_density.npy", np.ones(1))  # every tet maps to this one solid sample
    out = tmp / "blade.stl"
    rc = build_print_blades.main([
        "--name", name, "--shards", str(tmp), "--to-dir", str(tmp), "--out", str(out),
        "--rib-width-mm", "1.5", "--mesh-size-mm", mesh_mm, "--pitch-mm", "1.0", *extra,
    ])  # fmt: skip
    return rc, out


def _report(out: Path) -> dict:
    return json.loads(out.with_suffix(".json").read_text())


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("print_blades")
    with zipfile.ZipFile(tmp / "template.3mf", "w") as z:
        z.writestr("3D/3dmodel.model", '<model xmlns="x"><resources/><build/></model>')
        z.writestr(
            "Metadata/project_settings.config",
            json.dumps(
                {
                    "sparse_infill_density": "15%",
                    "printable_area": ["0x0", "256x0", "256x256", "0x256"],
                }
            ),
        )
        z.writestr("Metadata/model_settings.config", "<config/>")
    # A 6 mm mesh is too coarse to measure the detent surfaces; the detent has its own fixture below.
    return _build(tmp, "--template", str(tmp / "template.3mf"), "--detent-seats", "0")


@pytest.fixture(scope="module")
def pla_outputs(tmp_path_factory):
    return _build(
        tmp_path_factory.mktemp("print_blades_pla"), "--material", "pla", "--detent-seats", "0"
    )


@pytest.fixture(scope="module")
def detent_outputs(tmp_path_factory):
    rc, out = _build(tmp_path_factory.mktemp("print_blades_detent"), mesh_mm="2")
    return rc, _report(out)


def test_main_returns_zero(outputs):
    assert outputs[0] == 0


def test_writes_the_stl(outputs):
    assert outputs[1].stat().st_size > 84


def test_report_records_mass_and_fold_stack(outputs):
    rep = json.loads(outputs[1].with_suffix(".json").read_text())
    assert rep["mass_g"] > 0 and rep["design_folded_stack_mm"] > 0


def test_fully_solid_density_needs_no_feature_material(outputs):
    rep = json.loads(outputs[1].with_suffix(".json").read_text())
    assert rep["voxel_cm3"]["rib"] == pytest.approx(0.0) and rep["voxel_cm3"][
        "tip_cap"
    ] == pytest.approx(0.0)


def test_project_prints_solid_without_supports(outputs):
    cfg = json.loads(
        zipfile.ZipFile(outputs[1].with_suffix(".3mf")).read("Metadata/project_settings.config")
    )
    assert (cfg["sparse_infill_density"], cfg["enable_support"]) == ("100%", "0")


def test_default_material_is_petg(outputs):
    rep = _report(outputs[1])
    assert rep["mass_g"] == pytest.approx(rep["volume_cm3"] * RHO_PETG_KG_PER_M3 / 1e3)


def test_pla_mass_uses_pla_density(pla_outputs):
    rep = _report(pla_outputs[1])
    assert (rep["material"], rep["mass_g"]) == (
        "pla",
        pytest.approx(rep["volume_cm3"] * RHO_PLA_KG_PER_M3 / 1e3),
    )


def test_pla_keeps_the_to_mass_in_petg(pla_outputs):
    rep = _report(pla_outputs[1])
    # density 1 everywhere → TO mass = whole domain at PETG density, regardless of --material
    assert rep["to_mass_g_on_rebuilt_mesh_petg"] == pytest.approx(
        rep["design_domain_cm3_exact_tets"] * RHO_PETG_KG_PER_M3 / 1e3
    )


def test_without_template_no_project_is_written(pla_outputs):
    assert (
        _report(pla_outputs[1])["project_3mf"] is None
        and not pla_outputs[1].with_suffix(".3mf").exists()
    )


def test_unknown_material_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        build_print_blades.main(["--name", "x", "--shards", ".", "--to-dir", ".", "--out", "x.stl",
                                 "--rib-width-mm", "1", "--material", "abs"])  # fmt: skip


def test_boss_is_exactly_the_design_layer_spacing(outputs):
    rep = _report(outputs[1])
    assert rep["boss_thickness_mm"] == pytest.approx(rep["design_layer_spacing_mm"], abs=1e-3)


def test_detent_can_be_turned_off(outputs):
    assert _report(outputs[1])["detent"] is None


def test_detent_build_succeeds(detent_outputs):
    assert detent_outputs[0] == 0


def test_detent_defaults_to_five_seats_in_the_deploy_direction(detent_outputs):
    d = detent_outputs[1]["detent"]
    assert (len(d["seats"]), d["deploy_direction"]) == (5, -1)


def test_detent_bumps_cross_the_gap_by_the_interference(detent_outputs):
    d = detent_outputs[1]["detent"]
    assert all(
        s["bump_height_mm"]
        == pytest.approx(max(s["gap_closed_mm"], s["gap_open_mm"]) + d["spec_mm"]["interference_m"])
        for s in d["seats"]
    )
