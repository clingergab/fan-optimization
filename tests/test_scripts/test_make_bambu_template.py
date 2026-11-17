"""Tests for scripts/make_bambu_template.py (Bambu system presets → exported template project).

The Bambu Studio executable is the external boundary: ``subprocess.run`` is replaced by a fake that
records the call and writes the export where Bambu Studio would.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import make_bambu_template  # noqa: E402


def _profiles(root: Path) -> Path:
    for kind, leaf in (("machine", "M"), ("process", "P"), ("filament", "F")):
        (root / kind).mkdir(parents=True)
        (root / kind / "base.json").write_text(json.dumps({"from": "system", "kind": kind}))
        (root / kind / f"{leaf}.json").write_text(json.dumps({"inherits": "base", "leaf": leaf}))
    return root


class _FakeBambu:
    def __init__(self, returncode: int = 0, export: bool = True):
        self.returncode, self.export, self.calls = returncode, export, []

    def __call__(self, cmd, **kw):
        loaded = {}
        for flag in ("--load-settings", "--load-filaments"):
            for f in cmd[cmd.index(flag) + 1].split(";"):
                loaded[Path(f).stem] = json.loads(Path(f).read_text())
        self.calls.append({"cmd": cmd, "kw": kw, "loaded": loaded})
        if self.export:
            out_dir = Path(cmd[cmd.index("--outputdir") + 1])
            (out_dir / cmd[cmd.index("--export-3mf") + 1]).write_text("project")
        return subprocess.CompletedProcess(cmd, self.returncode, stdout="log", stderr="")


@pytest.fixture
def call(tmp_path, monkeypatch):
    fake = _FakeBambu()
    monkeypatch.setattr(make_bambu_template.subprocess, "run", fake)
    rc = make_bambu_template.main([
        "--model", str(tmp_path / "m.stl"), "--out", str(tmp_path / "o" / "t.3mf"),
        "--machine", "M", "--process", "P", "--filament", "F",
        "--bambu", str(tmp_path / "bambu"), "--profiles", str(_profiles(tmp_path / "prof")),
    ])  # fmt: skip
    return rc, fake.calls[0], tmp_path


def test_main_returns_zero(call):
    assert call[0] == 0


def test_writes_the_exported_project_to_out(call):
    assert (call[2] / "o" / "t.3mf").read_text() == "project"


def test_loads_each_preset_flattened(call):
    loaded = call[1]["loaded"]
    assert [loaded[k]["leaf"] for k in ("machine", "process", "filament")] == [
        "M",
        "P",
        "F",
    ] and all(loaded[k]["from"] == "system" and "inherits" not in loaded[k] for k in loaded)


def test_export_name_is_bare_because_bambu_prefixes_outputdir(call):
    cmd = call[1]["cmd"]
    assert "/" not in cmd[cmd.index("--export-3mf") + 1]


def test_runs_in_the_scratch_dir_so_result_json_does_not_litter(call):
    cmd = call[1]["cmd"]
    assert call[1]["kw"]["cwd"] == Path(cmd[cmd.index("--outputdir") + 1])


def test_model_path_is_absolute(call):
    assert Path(call[1]["cmd"][-1]).is_absolute()


@pytest.mark.parametrize("fake", [_FakeBambu(returncode=1), _FakeBambu(export=False)])
def test_failed_export_raises(tmp_path, monkeypatch, fake):
    monkeypatch.setattr(make_bambu_template.subprocess, "run", fake)
    with pytest.raises(RuntimeError, match="export failed"):
        make_bambu_template.run(
            model=tmp_path / "m.stl", out=tmp_path / "t.3mf", machine="M", process="P", filament="F",
            bambu=tmp_path / "bambu", profiles=_profiles(tmp_path / "prof"),
        )  # fmt: skip
