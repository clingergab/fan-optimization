"""Tests for fanopt.geometry.print_plate (N copies of one project on a plate, brim only at ears)."""

from __future__ import annotations

import json
import re
import zipfile

import numpy as np
import pytest

from fanopt.geometry.bambu_3mf import write_bambu_project
from fanopt.geometry.print_plate import (
    footprint_ear_points_mm,
    object_vertices_mm,
    row_positions_mm,
    write_plate,
)

_ROOT = (
    '<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" '
    'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n <resources/>\n <build/>\n</model>\n'
)
# A bar standing on its long thin edge: 200 mm long (x), 3 mm thick (y), 40 mm tall (z).
_BAR_V = np.array(
    [[x, y, z] for x in (0.0, 200.0) for y in (0.0, 3.0) for z in (0.0, 40.0)], dtype=float
)
_BAR_F = np.array(
    [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
     [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
)  # fmt: skip


@pytest.fixture
def project(tmp_path):
    t = tmp_path / "t.3mf"
    with zipfile.ZipFile(t, "w") as z:
        z.writestr("3D/3dmodel.model", _ROOT)
        z.writestr(
            "Metadata/project_settings.config",
            json.dumps({"brim_type": "auto_brim", "printable_area": ["0x0", "330x0", "330x320", "0x320"]}),
        )  # fmt: skip
        z.writestr("Metadata/model_settings.config", "<config/>")
    return write_bambu_project(t, tmp_path / "p.3mf", _BAR_V, _BAR_F, name="bar.stl", settings={})


def _plate(project, tmp_path, n=3):
    pts = footprint_ear_points_mm(object_vertices_mm(project), 3, inset_mm=2.0)
    return write_plate(
        project, tmp_path / "plate.3mf", row_positions_mm(n, 16.0, (175.0, 160.0)),
        ear_points_mm=pts, ear_radius_mm=5.0,
    )  # fmt: skip


def _read(path, name):
    with zipfile.ZipFile(path) as z:
        return z.read(name).decode()


def test_vertices_round_trip_through_the_project(project):
    v = object_vertices_mm(project)
    assert np.ptp(v, axis=0) == pytest.approx([200.0, 3.0, 40.0])


def test_ears_sit_at_both_ends_and_between(project):
    pts = footprint_ear_points_mm(object_vertices_mm(project), 3, inset_mm=2.0)
    assert pts[:, 0] == pytest.approx([-98.0, 0.0, 98.0])


def test_ears_are_centred_across_the_edge_at_the_lowest_z(project):
    v = object_vertices_mm(project)
    pts = footprint_ear_points_mm(v, 3, inset_mm=2.0)
    assert np.allclose(pts[:, 1], 0.0) and np.allclose(pts[:, 2], v[:, 2].min())


def test_a_single_ear_goes_in_the_middle(project):
    pts = footprint_ear_points_mm(object_vertices_mm(project), 1, inset_mm=2.0)
    assert pts[0, 0] == pytest.approx(0.0)


def test_rows_are_centred_on_the_plate():
    pos = row_positions_mm(4, 10.0, (100.0, 50.0))
    assert pos == [(100.0, 35.0), (100.0, 45.0), (100.0, 55.0), (100.0, 65.0)]


def test_plate_has_one_build_item_per_copy(project, tmp_path):
    root = _read(_plate(project, tmp_path, n=5), "3D/3dmodel.model")
    assert len(re.findall(r"<item\b", root)) == 5


def test_copies_keep_the_height_and_get_their_row(project, tmp_path):
    root = _read(_plate(project, tmp_path), "3D/3dmodel.model")
    tfs = [t.split() for t in re.findall(r'<item [^>]*transform="([^"]+)"', root)]
    assert [float(t[10]) for t in tfs] == pytest.approx([144.0, 160.0, 176.0]) and [
        float(t[11]) for t in tfs
    ] == pytest.approx([20.0] * 3)


def test_copies_have_distinct_ids(project, tmp_path):
    root = _read(_plate(project, tmp_path), "3D/3dmodel.model")
    uuids = re.findall(r'<item [^>]*p:UUID="([^"]+)"', root)
    assert len(set(uuids)) == 3


def test_model_settings_list_every_instance(project, tmp_path):
    cfg = _read(_plate(project, tmp_path), "Metadata/model_settings.config")
    ids = re.findall(r'key="instance_id" value="(\d+)"', cfg)
    asm = re.findall(r'<assemble_item [^>]*instance_id="(\d+)"', cfg)
    assert ids == ["0", "1", "2"] and asm == ["0", "1", "2"]


def test_brim_is_switched_to_ears(project, tmp_path):
    cfg = json.loads(_read(_plate(project, tmp_path), "Metadata/project_settings.config"))
    assert cfg["brim_type"] == "brim_ears"


def test_ear_file_has_five_values_per_ear(project, tmp_path):
    txt = _read(_plate(project, tmp_path), "Metadata/brim_ear_points.txt")
    head, body = txt.splitlines()
    vals = body.split("|")[1].split()
    assert head == "brim_points_format_version=1" and len(vals) == 15 and vals[3] == "5.000000"


def test_refuses_to_overwrite(project, tmp_path):
    out = _plate(project, tmp_path)
    with pytest.raises(FileExistsError):
        write_plate(project, out, [(0.0, 0.0)], ear_points_mm=np.zeros((1, 3)), ear_radius_mm=5.0)


def test_rebuilding_from_a_plate_replaces_its_ears(project, tmp_path):
    first = _plate(project, tmp_path)
    again = write_plate(
        first,
        tmp_path / "again.3mf",
        [(0.0, 0.0)],
        ear_points_mm=np.zeros((1, 3)),
        ear_radius_mm=4.0,
    )
    with zipfile.ZipFile(again) as z:
        assert z.namelist().count("Metadata/brim_ear_points.txt") == 1


def test_project_without_a_build_item_is_rejected(project, tmp_path):
    bare = tmp_path / "bare.3mf"
    with zipfile.ZipFile(project) as zin, zipfile.ZipFile(bare, "w") as z:
        for i in zin.infolist():
            data = zin.read(i.filename)
            if i.filename == "3D/3dmodel.model":
                data = re.sub(rb"<item\b[^>]*/>", b"", data)
            z.writestr(i, data)
    with pytest.raises(ValueError, match="build item"):
        write_plate(
            bare,
            tmp_path / "x.3mf",
            [(0.0, 0.0)],
            ear_points_mm=np.zeros((1, 3)),
            ear_radius_mm=5.0,
        )


def test_project_without_an_instance_is_rejected(project, tmp_path):
    bare = tmp_path / "bare.3mf"
    with zipfile.ZipFile(project) as zin, zipfile.ZipFile(bare, "w") as z:
        for i in zin.infolist():
            data = zin.read(i.filename)
            if i.filename == "Metadata/model_settings.config":
                data = b"<config/>"
            z.writestr(i, data)
    with pytest.raises(ValueError, match="instance"):
        write_plate(
            bare,
            tmp_path / "x.3mf",
            [(0.0, 0.0)],
            ear_points_mm=np.zeros((1, 3)),
            ear_radius_mm=5.0,
        )
