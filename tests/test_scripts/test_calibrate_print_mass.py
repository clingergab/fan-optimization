"""Tests for scripts/calibrate_print_mass.py (slice real print blades → fit the mass model).

The blade build and Bambu Studio are external processes: both are replaced at the ``subprocess.run``
boundary by a fake that "slices" each blade to the mass a known model gives it, so the fit must recover
that model.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import calibrate_print_mass  # noqa: E402
from fanopt.bo.blade_codec import bounds, clip_to_bounds, decode  # noqa: E402
from fanopt.physical.print_effort import (  # noqa: E402
    PrintMassModel,
    blade_mass_kg,
    blade_sheet,
)
from fanopt.utils.ledger import design_hash  # noqa: E402

_TRUE = PrintMassModel(wall_m=1.2e-3, core_fraction=0.2, base_kg=0.003, rho_kg_m3=1240.0)


def _write_shards(d: Path) -> list[str]:
    low, high = bounds()
    rng = np.random.default_rng(3)
    rows, hashes = [], []
    for k in range(4):
        vec = clip_to_bounds(low + (high - low) * rng.uniform(0.2, 0.8, low.size))
        h = design_hash(decode(vec).to_dict())
        hashes.append(h)
        rows.append({"design_hash": h, "vector": vec.tolist(), "j_fan": float(k)})
    rows.append({"design_hash": "nan", "vector": rows[0]["vector"], "j_fan": float("nan")})
    (d / "evaluations_c0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return hashes


class _Fake:
    """Stands in for the build (writes the .3mf) and the slicer (writes result.json)."""

    def __init__(self):
        self.calls = 0
        self.scale: dict[Path, float] = {}

    def __call__(self, cmd, **kwargs):
        self.calls += 1
        if "--thickness-scale" in cmd:
            out = Path(cmd[cmd.index("--out") + 1])
            self.scale[out.parent] = float(cmd[cmd.index("--thickness-scale") + 1])
            out.with_suffix(".3mf").write_text("model")
            out.with_suffix(".json").write_text(
                json.dumps({"thickness_scale": self.scale[out.parent]})
            )
            return None
        d = Path(cmd[cmd.index("--outputdir") + 1])
        row = json.loads((d / "evaluations_cal.jsonl").read_text())
        sheet = blade_sheet(decode(np.asarray(row["vector"])))
        thinned = type(sheet)(
            sheet.x_m, sheet.y_m, sheet.z_m, sheet.thickness_m * self.scale[d], sheet.area_m2
        )
        grams = blade_mass_kg(thinned, _TRUE) * 1e3
        plate = {"filaments": [{"main_used_g": grams}]}
        (d / "result.json").write_text(
            json.dumps({"return_code": 0, "error_string": "Success.", "sliced_plates": [plate]})
        )
        return None


@pytest.fixture
def setup(tmp_path, monkeypatch):
    hashes = _write_shards(tmp_path)
    with zipfile.ZipFile(tmp_path / "t.3mf", "w") as z:
        z.writestr(
            "Metadata/project_settings.config",
            json.dumps({"brim_type": "auto_brim", "filament_density": ["1.24"]}),
        )
    fake = _Fake()
    monkeypatch.setattr(calibrate_print_mass.subprocess, "run", fake)
    return tmp_path, hashes, fake


def _run(tmp: Path, include: list[str]) -> dict:
    return calibrate_print_mass.run(
        shards=tmp, template=tmp / "t.3mf", work=tmp / "work", out=tmp / "cal.json",
        include=include, n_spread=2, targets_mm=[1.0, 1.6, 2.2, 2.8], jobs=2,
        bambu=Path("bambu"), python="python",
    )  # fmt: skip


def test_fit_recovers_the_model_the_slices_came_from(setup):
    tmp, hashes, _ = setup
    m = _run(tmp, [hashes[0]])["model"]
    assert (m["wall_mm"], m["core_fraction"], m["base_g"]) == pytest.approx(
        (1.2, 0.2, 3.0), rel=1e-3
    )


def test_fit_uses_the_template_filament_density(setup):
    tmp, hashes, _ = setup
    assert _run(tmp, [hashes[0]])["model"]["rho_kg_m3"] == pytest.approx(1240.0)


def test_leave_one_out_error_is_reported(setup):
    tmp, hashes, _ = setup
    assert _run(tmp, [hashes[0]])["loo_rmse_g"] == pytest.approx(0.0, abs=1e-3)


def test_report_is_written(setup):
    tmp, hashes, _ = setup
    _run(tmp, [hashes[0]])
    assert json.loads((tmp / "cal.json").read_text())["samples"]


def test_included_design_is_sampled(setup):
    tmp, hashes, _ = setup
    samples = _run(tmp, [hashes[1]])["samples"]
    assert hashes[1] in {s["design_hash"] for s in samples}


def test_brim_is_turned_off_in_the_slicing_template(setup):
    tmp, hashes, _ = setup
    _run(tmp, [hashes[0]])
    cfg = zipfile.ZipFile(tmp / "work" / "template_no_brim.3mf").read(
        "Metadata/project_settings.config"
    )
    assert json.loads(cfg)["brim_type"] == "no_brim"


def test_a_rerun_reuses_the_slices(setup):
    tmp, hashes, fake = setup
    _run(tmp, [hashes[0]])
    before = fake.calls
    _run(tmp, [hashes[0]])
    assert fake.calls == before


def test_slicer_output_dir_is_absolute(setup, monkeypatch):
    tmp, hashes, fake = setup
    monkeypatch.chdir(tmp)
    seen = []

    def spy(cmd, **kwargs):
        if "--outputdir" in cmd:
            seen.append(cmd[cmd.index("--outputdir") + 1])
        return fake(cmd, **kwargs)

    monkeypatch.setattr(calibrate_print_mass.subprocess, "run", spy)
    calibrate_print_mass.run(
        shards=Path("."), template=Path("t.3mf"), work=Path("work"), out=Path("cal.json"),
        include=[hashes[0]], n_spread=2, targets_mm=[1.0, 1.6, 2.2, 2.8], jobs=1,
        bambu=Path("bambu"), python="python",
    )  # fmt: skip
    assert seen and all(Path(s).is_absolute() for s in seen)


def test_a_reused_build_is_fitted_at_the_thinning_it_was_built_with(setup):
    tmp, hashes, _ = setup
    first = _run(tmp, [hashes[0]])["samples"][0]
    d = tmp / "work" / f"{first['design_hash']}_{first['target_mean_wall_mm']:.2f}mm"
    (d / "blade.json").write_text(json.dumps({"thickness_scale": 0.5}))  # built at another thinning
    again = _run(tmp, [hashes[0]])["samples"][0]
    assert again["built_mean_wall_mm"] == pytest.approx(0.5 * first["design_mean_wall_mm"])


def test_failed_designs_are_not_candidates(tmp_path):
    _write_shards(tmp_path)
    assert "nan" not in calibrate_print_mass.valid_rows(tmp_path)


def test_spread_picks_take_both_ends():
    picks = calibrate_print_mass.spread_picks({"a": 3.0, "b": 1.0, "c": 2.0, "d": 5.0}, 2)
    assert picks == ["b", "d"]


def test_main_prints_the_fit(setup, capsys):
    tmp, hashes, _ = setup
    rc = calibrate_print_mass.main([
        "--shards", str(tmp), "--template", str(tmp / "t.3mf"), "--work", str(tmp / "work"),
        "--out", str(tmp / "cal.json"), "--include", hashes[0], "--n-spread", "2",
        "--targets-mm", "1.0", "1.6", "2.2", "2.8",
    ])  # fmt: skip
    assert rc == 0 and "wall_mm" in capsys.readouterr().out
