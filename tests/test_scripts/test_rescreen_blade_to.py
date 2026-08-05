"""Tests for scripts/rescreen_blade_to.py (re-screen saved density fields, no re-optimization).

Runs the TO once at a coarse mesh to produce the saved density fields, then exercises the re-screen
path end-to-end. Requires the [topopt] extra.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

for _dep in ("skfem", "gmsh", "cadquery"):
    if importlib.util.find_spec(_dep) is None:  # pragma: no cover - env-dependent
        pytest.skip(f"{_dep} not installed", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import rescreen_blade_to  # noqa: E402
import run_phase2_blade_to  # noqa: E402

from fanopt.bo.blade_codec import bounds, clip_to_bounds, decode  # noqa: E402
from fanopt.utils.ledger import design_hash  # noqa: E402


def _campaign(tmp_path: Path, fracs_and_j3d):
    """Write a synthetic campaign shard + verification.json for the given (frac, j_fan_3d)s."""
    low, high = bounds()
    rows, ver = [], []
    for rank, (frac, j3d) in enumerate(fracs_and_j3d):
        vec = clip_to_bounds(low + (high - low) * frac)
        h = design_hash(decode(vec).to_dict())
        rows.append({"design_hash": h, "vector": vec.tolist(), "j_fan": float(rank)})
        ver.append({"name": f"{rank:02d}_{h}", "j_fan_3d": j3d})
    (tmp_path / "evaluations_c0.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    v = tmp_path / "verification.json"
    v.write_text(json.dumps({"designs": ver}), encoding="utf-8")
    return v


def _make_to_run(tmp_path: Path):
    """Produce the saved density fields (a coarse 1-design TO run) that the re-screen consumes."""
    ver = _campaign(tmp_path, [(0.5, 3.0)])
    out = tmp_path / "to_out"
    run_phase2_blade_to.run(
        shared_dir=tmp_path, verification=ver, out_dir=out, progress=False,
        top_k=1, volfrac=0.5, max_iters=1, mesh_size_m=0.006, skin_thickness_m=None, screen=False,
    )
    return ver, out


def test_rescreen_run_writes_summary_and_sidecar(tmp_path):
    ver, out = _make_to_run(tmp_path)
    summary = rescreen_blade_to.run(
        shared_dir=tmp_path, verification=ver, out_dir=out, top_k=1, n_workers=1,
        mesh_size_m=None, skin_thickness_m=None, stress_fos=2.0, progress=False,
    )
    assert summary["n_succeeded"] == 1 and summary["rescreened"] is True
    assert (out / "rescreen_summary.json").exists()
    rec = summary["designs"][0]
    assert (out / f"{rec['name']}_rescreen.json").exists()
    assert "max_von_mises_solid_mpa" in rec  # solid-only stress is reported


def test_rescreen_reads_mesh_params_from_summary(tmp_path):
    # mesh/skin default from the original run's summary.json so the rebuilt mesh matches the density.
    _ver, out = _make_to_run(tmp_path)
    mesh_size, skin = rescreen_blade_to._run_params(out, None, None)
    assert mesh_size == pytest.approx(0.006)
    assert skin == pytest.approx(0.0012)  # batch default recorded when the TO run passed None


def test_rescreen_missing_params_without_summary_errors(tmp_path):
    with pytest.raises(SystemExit):
        rescreen_blade_to._run_params(tmp_path / "no_such_dir", None, None)


def test_rescreen_main_smoke(tmp_path):
    ver, out = _make_to_run(tmp_path)
    rc = rescreen_blade_to.main(
        [
            "--shared-dir", str(tmp_path),
            "--verification", str(ver),
            "--out-dir", str(out),
            "--top-k", "1",
            "--n-workers", "1",
        ]
    )
    assert rc == 0
    assert (out / "rescreen_summary.json").exists()
