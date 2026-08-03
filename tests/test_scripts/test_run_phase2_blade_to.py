"""Tests for scripts/run_phase2_blade_to.py (per-design 3D blade TO batch driver).

Exercises the full resolve → mesh → SIMP → write path end-to-end on a synthetic 1-design
campaign at a coarse mesh (so gmsh + scikit-fem run fast). Requires the [topopt] extra.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

for _dep in ("skfem", "gmsh", "cadquery"):
    if importlib.util.find_spec(_dep) is None:  # pragma: no cover - env-dependent
        pytest.skip(f"{_dep} not installed", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

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


_COARSE = dict(top_k=1, volfrac=0.5, max_iters=1, mesh_size_m=0.006, skin_thickness_m=None, screen=False)


def test_run_writes_summary_and_density(tmp_path):
    ver = _campaign(tmp_path, [(0.5, 3.0)])
    out = tmp_path / "to_out"
    summary = run_phase2_blade_to.run(
        shared_dir=tmp_path, verification=ver, out_dir=out, progress=False, **_COARSE
    )
    assert summary["n_succeeded"] == 1
    assert (out / "summary.json").exists()
    rec = summary["designs"][0]
    assert "error" not in rec
    npy = out / f"{rec['name']}_density.npy"
    assert npy.exists()
    assert np.load(npy).size > 0


def test_run_selects_top_k_by_fine_jfan(tmp_path):
    # Two designs; the higher fine J_fan must be the one TO'd when top_k=1.
    ver = _campaign(tmp_path, [(0.4, 1.0), (0.6, 9.0)])
    out = tmp_path / "to_out"
    summary = run_phase2_blade_to.run(
        shared_dir=tmp_path, verification=ver, out_dir=out, progress=False, **_COARSE
    )
    assert summary["n_designs"] == 1  # only the top-1 was run
    assert summary["designs"][0]["name"].startswith("01_")  # rank-01 had fine J_fan 9.0


def test_main_smoke(tmp_path):
    ver = _campaign(tmp_path, [(0.5, 3.0)])
    out = tmp_path / "to_out"
    rc = run_phase2_blade_to.main(
        [
            "--shared-dir", str(tmp_path),
            "--verification", str(ver),
            "--out-dir", str(out),
            "--top-k", "1",
            "--max-iters", "1",
            "--mesh-size-m", "0.006",
            "--no-screen",
        ]
    )
    assert rc == 0
    assert (out / "summary.json").exists()


def test_run_screened_mode_records_accepted_volfrac(tmp_path):
    # Screened ladder with relaxed limits -> passes at the first (most aggressive) rung.
    ver = _campaign(tmp_path, [(0.5, 3.0)])
    out = tmp_path / "to_out"
    summary = run_phase2_blade_to.run(
        shared_dir=tmp_path, verification=ver, out_dir=out, progress=False,
        top_k=1, max_iters=1, mesh_size_m=0.006, skin_thickness_m=None,
        screen=True, volfrac_ladder=(0.4, 0.6), u_tip_limit_m=1.0, stress_fos=1e-6,
    )
    rec = summary["designs"][0]
    assert rec["accepted_volfrac"] == 0.4  # most-aggressive rung accepted
    assert rec["screen_passed"] is True
    assert rec["ladder_trace"][0]["volfrac"] == 0.4
