"""Tests for scripts/run_phase5_verify_blade.py.

The 3D-verifying run (verify_blades) is the expensive boundary and is mocked; the CLI
wiring — pareto load → verify → verification.json → exit 0 — is exercised. Requires gmsh +
cadquery (the blade_verify import pulls them).
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

if importlib.util.find_spec("gmsh") is None:  # pragma: no cover - env-dependent
    pytest.skip("gmsh not installed", allow_module_level=True)
if importlib.util.find_spec("cadquery") is None:  # pragma: no cover - env-dependent
    pytest.skip("cadquery not installed", allow_module_level=True)

import run_phase5_verify_blade as script
from fanopt.bo.blade_codec import bounds, clip_to_bounds, decode
from fanopt.cfd.phase5 import FINE_CYCLES, FINE_INNER, VerifyResult
from fanopt.utils.ledger import design_hash


def _vec(frac: float) -> np.ndarray:
    low, high = bounds()
    return clip_to_bounds(low + (high - low) * frac)


def _fake_verify_blades(records, out_dir, *, top_k=None, on_result=None, **kw):
    from fanopt.cfd.phase5 import verify_ranking

    ranked = sorted(records, key=lambda e: -float(e["j_fan"]))[:top_k] if top_k else records
    results = [
        VerifyResult(f"{i:02d}", j_fan_3d=float(i + 1), j_fan_coarse=float(e["j_fan"]),
                     meta={"n_nodes": 100.0})
        for i, e in enumerate(ranked)
    ]
    for r in results:
        if on_result is not None:
            on_result(r)
    return results, verify_ranking(results)


def _fake_shards(tmp_path):
    d = tmp_path / "campaign_trapezoid"
    d.mkdir()
    rows = [
        {"design_hash": design_hash(decode(_vec(f)).to_dict()), "vector": _vec(f).tolist(),
         "j_fan": j, "mass_kg": 0.05, "deflection_m": 1e-4}
        for f, j in [(0.3, 1.0), (0.6, 3.0), (0.5, 2.0)]
    ]
    (d / "evaluations_colab-0.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return d


def test_main_reads_campaign_shards(tmp_path, monkeypatch):
    d = _fake_shards(tmp_path)
    monkeypatch.setattr(script, "verify_blades", _fake_verify_blades)
    rc = script.main(["--shared-dir", str(d), "--out-dir", str(tmp_path / "out"), "--top-k", "2"])
    assert rc == 0
    v = json.loads((tmp_path / "out" / "verification.json").read_text())
    assert len(v["designs"]) == 2 and all("j_fan_coarse" in x for x in v["designs"])


def test_main_defaults_to_fine_tier_and_overrides(tmp_path, monkeypatch):
    d = _fake_shards(tmp_path)
    seen = {}

    def capture(records, out_dir, *, cfg=None, **kw):
        seen["cfg"] = cfg
        return _fake_verify_blades(records, out_dir, **kw)

    monkeypatch.setattr(script, "verify_blades", capture)
    script.main(["--shared-dir", str(d), "--out-dir", str(tmp_path / "o1"), "--top-k", "1"])
    assert (seen["cfg"].n_cycles, seen["cfg"].inner_iter) == (FINE_CYCLES, FINE_INNER)  # default = fine
    script.main(["--shared-dir", str(d), "--out-dir", str(tmp_path / "o2"), "--top-k", "1",
                 "--n-cycles", "7", "--inner-iter", "80"])
    assert (seen["cfg"].n_cycles, seen["cfg"].inner_iter) == (7, 80)  # overridable


def test_run_top_k_zero_verifies_all_designs(tmp_path, monkeypatch):
    # The CLI help says "0 = all"; run() must honor that for programmatic callers too (0 → None),
    # not verify zero designs (ranked[:0] == []).
    d = _fake_shards(tmp_path)
    monkeypatch.setattr(script, "verify_blades", _fake_verify_blades)
    summary = script.run(out_dir=tmp_path / "out", top_k=0, shared_dir=d)
    assert len(summary["designs"]) == 3  # all three shard designs, not zero


def test_run_resume_skips_already_verified_and_merges(tmp_path, monkeypatch):
    # A Colab drop during the multi-hour fine run must resume: keep the finite results already in
    # verification.json and re-verify ONLY the not-yet-done designs.
    d = _fake_shards(tmp_path)  # coarse J_fan: _vec(0.6)=3.0 (top), _vec(0.5)=2.0, _vec(0.3)=1.0
    out = tmp_path / "out"
    out.mkdir()
    top_hash = design_hash(decode(_vec(0.6)).to_dict())
    (out / "verification.json").write_text(  # pre-seed the top design as already verified
        json.dumps({"ranking": {}, "designs": [
            {"name": f"00_{top_hash}", "j_fan_3d": 9.9e12, "j_fan_coarse": 3.0, "n_nodes": 100.0}]}),
        encoding="utf-8",
    )
    seen = {}

    def fake(records, out_dir, *, top_k=None, on_result=None, skip_hashes=None, **kw):
        from fanopt.cfd.phase5 import verify_ranking

        seen["skip"] = skip_hashes
        ranked = sorted(records, key=lambda e: -float(e["j_fan"]))[:top_k] if top_k else records
        results = []
        for i, e in enumerate(ranked):
            name = f"{i:02d}_{design_hash(decode(np.asarray(e['vector'], dtype=float)).to_dict())}"
            if skip_hashes and name.split("_", 1)[1] in skip_hashes:
                continue  # honor resume-skip like the real verify_blades
            r = VerifyResult(name, j_fan_3d=float(i + 1), j_fan_coarse=float(e["j_fan"]),
                             meta={"n_nodes": 100.0})
            results.append(r)
            if on_result is not None:
                on_result(r)
        return results, verify_ranking(results)

    monkeypatch.setattr(script, "verify_blades", fake)
    summary = script.run(out_dir=out, top_k=3, shared_dir=d)
    assert top_hash in seen["skip"]  # the done design was passed to be skipped, not re-run
    by_name = {x["name"]: x for x in summary["designs"]}
    assert len(by_name) == 3  # 1 prior (kept) + 2 newly verified
    assert by_name[f"00_{top_hash}"]["j_fan_3d"] == 9.9e12  # prior result preserved, not recomputed


def test_main_requires_shared_dir_or_pareto(tmp_path):
    with pytest.raises(SystemExit):  # argparse error → SystemExit
        script.main(["--out-dir", str(tmp_path / "out")])


def _fake_pareto(tmp_path):
    pareto = [
        {"vector": _vec(f).tolist(), "j_fan": j, "mass_kg": 0.05,
         "deflection_m": 1e-4, "params": decode(_vec(f)).to_dict()}
        for f, j in [(0.3, 1.0), (0.6, 3.0), (0.5, 2.0)]
    ]
    p = tmp_path / "pareto.json"
    p.write_text(json.dumps(pareto), encoding="utf-8")
    return p


def test_main_writes_verification_json(tmp_path, monkeypatch):
    p = _fake_pareto(tmp_path)

    def fake_verify_blades(pareto, out_dir, *, top_k=None, on_result=None, **kw):
        results = [
            VerifyResult(f"{i:02d}", j_fan_3d=float(i + 1), j_fan_coarse=float(e["j_fan"]),
                         meta={"n_nodes": 100.0})
            for i, e in enumerate(pareto[:top_k] if top_k else pareto)
        ]
        for r in results:
            if on_result is not None:
                on_result(r)
        from fanopt.cfd.phase5 import verify_ranking

        return results, verify_ranking(results)

    monkeypatch.setattr(script, "verify_blades", fake_verify_blades)
    rc = script.main(["--pareto", str(p), "--out-dir", str(tmp_path / "out"), "--top-k", "2"])
    assert rc == 0
    v = json.loads((tmp_path / "out" / "verification.json").read_text())
    assert "ranking" in v
    assert len(v["designs"]) == 2
    assert all("j_fan_3d" in d and "j_fan_coarse" in d for d in v["designs"])


def test_run_checkpoints_after_each_design(tmp_path, monkeypatch):
    p = _fake_pareto(tmp_path)
    out = tmp_path / "out"
    seen_counts = []

    def fake_verify_blades(pareto, out_dir, *, top_k=None, on_result=None, **kw):
        results = []
        for i, e in enumerate(pareto[:top_k] if top_k else pareto):
            r = VerifyResult(f"{i:02d}", j_fan_3d=float(i + 1), j_fan_coarse=float(e["j_fan"]),
                             meta={"n_nodes": 100.0})
            results.append(r)
            if on_result is not None:
                on_result(r)
                seen_counts.append(
                    len(json.loads((out_dir / "verification.json").read_text())["designs"])
                )
        return results, {}

    monkeypatch.setattr(script, "verify_blades", fake_verify_blades)
    script.run(pareto_path=p, out_dir=out, top_k=2)
    assert seen_counts == [1, 2]  # incremental write after each design
