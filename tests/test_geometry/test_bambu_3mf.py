"""Tests for the Bambu project writer (fanopt.geometry.bambu_3mf)."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest

from fanopt.geometry.bambu_3mf import (
    model_settings_xml,
    object_model_xml,
    plate_centre_mm,
    resolve_bambu_preset,
    sliced_filament_g,
    with_project_settings,
    write_bambu_project,
)

_ROOT = (
    '<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" '
    'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
    'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021">\n'
    ' <metadata name="Title">t</metadata>\n <metadata name="Thumbnail_Middle">/Metadata/plate_1.png</metadata>\n'
    ' <resources>\n  <object id="1" type="model"><mesh><vertices>'
    '<vertex x="0" y="0" z="0"/></vertices><triangles/></mesh></object>\n </resources>\n'
    ' <build>\n  <item objectid="1" transform="1 0 0 0 1 0 0 0 1 0 0 0" printable="1"/>\n </build>\n</model>\n'
)
_TET_V = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=float)
_TET_F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
_P2S_BED = ["0x0", "256x0", "256x256", "0x256"]
_H2C = {
    "printable_area": ["0x0", "330x0", "330x320", "0x320"],
    "extruder_printable_area": ["0x0,325x0,325x320,0x320", "25x0,330x0,330x320,25x320"],
}
_STALE = (
    "Metadata/slice_info.config",
    "Metadata/plate_1.json",
    "Metadata/plate_1.png",
    "Metadata/plate_1.gcode",
    "Metadata/_rels/model_settings.config.rels",
    "Metadata/cut_information.xml",
    "Metadata/filament_sequence.json",
    "3D/Objects/object_9.model",
)


def _template(path, *, with_rels=False, bed=None, root=_ROOT):
    cfg = {"brim_type": "auto_brim", "keep": "me", "printable_area": _P2S_BED, **(bed or {})}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("3D/3dmodel.model", root)
        z.writestr("Metadata/project_settings.config", json.dumps(cfg))
        z.writestr("Metadata/model_settings.config", "<config/>")
        z.writestr("_rels/.rels", '<Relationships><Relationship Target="/Metadata/plate_1.png"/>')
        for name in _STALE:
            z.writestr(name, "stale")
        if with_rels:
            z.writestr("3D/_rels/3dmodel.model.rels", "<Relationships/>")
    return path


def _item_transform(zf):
    root = zf.read("3D/3dmodel.model").decode()
    return [float(t) for t in re.search(r'<item [^>]*transform="([^"]+)"', root).group(1).split()]


@pytest.fixture
def project(tmp_path):
    t = _template(tmp_path / "t.3mf", with_rels=True)
    out = write_bambu_project(
        t,
        tmp_path / "o.3mf",
        _TET_V + 5.0,
        _TET_F,
        name="blade.stl",
        settings={"brim_type": "outer_only"},
    )
    return zipfile.ZipFile(out)


def test_object_model_counts_vertices_and_triangles():
    xml = object_model_xml(_TET_V, _TET_F)
    assert xml.count("<vertex ") == 4 and xml.count("<triangle ") == 4


def test_object_model_is_valid_xml():
    ET.fromstring(object_model_xml(_TET_V, _TET_F))


def test_model_settings_names_the_object():
    assert 'value="blade.stl"' in model_settings_xml("blade.stl", 4, 1.0)


def test_project_applies_setting_overrides(project):
    cfg = json.loads(project.read("Metadata/project_settings.config"))
    assert cfg["brim_type"] == "outer_only"


def test_project_keeps_template_settings(project):
    cfg = json.loads(project.read("Metadata/project_settings.config"))
    assert cfg["keep"] == "me"


def test_project_drops_stale_slice_results_thumbnails_and_old_objects(project):
    assert not set(project.namelist()) & set(_STALE)


def test_project_package_rels_point_only_at_the_model(project):
    rels = project.read("_rels/.rels").decode()
    assert "/3D/3dmodel.model" in rels and "plate_1.png" not in rels


def test_project_root_drops_thumbnail_references(project):
    assert "Thumbnail_" not in project.read("3D/3dmodel.model").decode()


def test_project_has_no_duplicate_entries(project):
    names = project.namelist()
    assert len(names) == len(set(names))


def test_project_root_points_at_the_new_object(project):
    root = project.read("3D/3dmodel.model").decode()
    ET.fromstring(root)
    assert 'p:path="/3D/Objects/object_1.model"' in root and "xmlns:p=" in root


def test_project_places_the_part_on_the_bed(project):
    root = project.read("3D/3dmodel.model").decode()
    obj = project.read("3D/Objects/object_1.model").decode()
    lift = float(re.search(r'<item [^>]*transform="([^"]+)"', root).group(1).split()[-1])
    zmin = min(float(z) for z in re.findall(r'z="([-\d.]+)"', obj))
    assert zmin + lift == pytest.approx(0.0)


def test_project_centres_the_part_on_the_plate(project):
    assert _item_transform(project)[9:11] == [128.0, 128.0]


def test_project_centres_on_the_region_both_h2c_nozzles_reach(tmp_path):
    t = _template(tmp_path / "t.3mf", bed=_H2C)
    out = write_bambu_project(t, tmp_path / "o.3mf", _TET_V, _TET_F, name="b", settings={})
    assert _item_transform(zipfile.ZipFile(out))[9:11] == [175.0, 160.0]


def test_sliced_template_with_self_closed_root_places_the_object(tmp_path):
    # A sliced .gcode.3mf saves an empty model as <resources/> and <build/>.
    root = re.sub(r"<resources>.*</resources>", "<resources/>", _ROOT, flags=re.S)
    root = re.sub(r"<build>.*</build>", "<build/>", root, flags=re.S)
    t = _template(tmp_path / "t.3mf", root=root)
    out = write_bambu_project(t, tmp_path / "o.3mf", _TET_V, _TET_F, name="b", settings={})
    assert _item_transform(zipfile.ZipFile(out))[9:11] == [128.0, 128.0]


def test_project_explicit_plate_position_wins(tmp_path):
    out = write_bambu_project(
        _template(tmp_path / "t.3mf"), tmp_path / "o.3mf", _TET_V, _TET_F, name="b", settings={},
        plate_xy_mm=(50.0, 60.0),
    )  # fmt: skip
    assert _item_transform(zipfile.ZipFile(out))[9:11] == [50.0, 60.0]


def test_plate_centre_single_nozzle_is_bed_centre():
    assert plate_centre_mm({"printable_area": _P2S_BED}) == (128.0, 128.0)


def test_plate_centre_ignores_empty_extruder_areas():
    assert plate_centre_mm({"printable_area": _P2S_BED, "extruder_printable_area": [""]}) == (
        128.0,
        128.0,
    )


def _presets(root):
    (root / "machine").mkdir()
    (root / "machine" / "base.json").write_text(json.dumps({"a": 1, "b": 1, "from": "system"}))
    (root / "machine" / "mid.json").write_text(json.dumps({"inherits": "base", "b": 2, "c": 2}))
    (root / "machine" / "leaf.json").write_text(json.dumps({"inherits": "mid", "c": 3}))
    return resolve_bambu_preset(root, "machine", "leaf")


def test_resolve_preset_child_values_win(tmp_path):
    d = _presets(tmp_path)
    assert {k: d[k] for k in "abc"} == {"a": 1, "b": 2, "c": 3}


def test_resolve_preset_is_standalone_and_named(tmp_path):
    d = _presets(tmp_path)
    assert "inherits" not in d and d["name"] == "leaf" and d["from"] == "system"


def test_project_without_template_rels_still_links_the_object(tmp_path):
    out = write_bambu_project(
        _template(tmp_path / "t.3mf"), tmp_path / "o.3mf", _TET_V, _TET_F, name="b", settings={}
    )
    rels = zipfile.ZipFile(out).read("3D/_rels/3dmodel.model.rels").decode()
    assert "/3D/Objects/object_1.model" in rels


def test_with_project_settings_applies_overrides(tmp_path):
    out = with_project_settings(
        _template(tmp_path / "t.3mf"), tmp_path / "o.3mf", {"brim_type": "no_brim"}
    )
    cfg = json.loads(zipfile.ZipFile(out).read("Metadata/project_settings.config"))
    assert (cfg["brim_type"], cfg["keep"]) == ("no_brim", "me")


def test_with_project_settings_keeps_every_other_entry(tmp_path):
    src = _template(tmp_path / "t.3mf")
    out = with_project_settings(src, tmp_path / "o.3mf", {})
    assert zipfile.ZipFile(out).namelist() == zipfile.ZipFile(src).namelist()


def _result(code=0, grams=(1.5, 2.0)):
    plates = [{"filaments": [{"main_used_g": g}]} for g in grams]
    return {"return_code": code, "error_string": "x", "sliced_plates": plates}


def test_sliced_filament_sums_every_plate():
    assert sliced_filament_g(_result()) == pytest.approx(3.5)


def test_sliced_filament_rejects_a_failed_slice():
    with pytest.raises(ValueError, match="slice failed"):
        sliced_filament_g(_result(code=250))
