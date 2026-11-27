"""Tests for scripts/make_print_plate.py (N blades on one plate with brim ears)."""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import make_print_plate  # noqa: E402
from fanopt.geometry.bambu_3mf import write_bambu_project  # noqa: E402

_V = np.array([[x, y, z] for x in (0.0, 200.0) for y in (0.0, 3.0) for z in (0.0, 40.0)])
_F = np.array(
    [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
     [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
)  # fmt: skip
_H2C = {
    "brim_type": "auto_brim",
    "printable_area": ["0x0", "330x0", "330x320", "0x320"],
    "extruder_printable_area": ["0x0,325x0,325x320,0x320", "25x0,330x0,330x320,25x320"],
}


@pytest.fixture
def project(tmp_path):
    t = tmp_path / "t.3mf"
    with zipfile.ZipFile(t, "w") as z:
        z.writestr("3D/3dmodel.model", '<model xmlns="x">\n <resources/>\n <build/>\n</model>\n')
        z.writestr("Metadata/project_settings.config", json.dumps(_H2C))
        z.writestr("Metadata/model_settings.config", "<config/>")
    return write_bambu_project(t, tmp_path / "p.3mf", _V, _F, name="bar.stl", settings={})


def test_rows_are_centred_where_both_nozzles_reach(project, tmp_path):
    rep = make_print_plate.run(
        project=project, out=tmp_path / "o.3mf", copies=2, pitch_mm=16.0, ears=3,
        ear_radius_mm=5.0, inset_mm=2.0,
    )  # fmt: skip
    root = zipfile.ZipFile(tmp_path / "o.3mf").read("3D/3dmodel.model").decode()
    xs = [float(t.split()[9]) for t in re.findall(r'<item [^>]*transform="([^"]+)"', root)]
    assert rep["row_y_mm"] == [152.0, 168.0] and xs == pytest.approx([175.0, 175.0])


def test_main_writes_twelve_copies(project, tmp_path, capsys):
    rc = make_print_plate.main(["--project", str(project), "--out", str(tmp_path / "o.3mf")])
    rep = json.loads(capsys.readouterr().out)
    assert rc == 0 and rep["copies"] == 12 and len(rep["ear_points_object_mm"]) == 3
