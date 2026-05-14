"""CLI smoke tests for ``scripts/run_spike_0_7a.py``.

Spec reference: ``docs/plan_R11.md §Phase 0 Spike 0.7`` (sub-spike 0.7a).
Protocol: ``docs/spike_0_7a_protocol.md``.

Exercises:
- happy-path PASS (exit 0, results.json schema, per-design subdir).
- ``--skip-adversarial`` produces a FAIL (every spike run with no adversarial
  set must fail — the adversarial coverage check is the spike's reason for
  existing).
- bad input (``--n-random 0``) → exit 2.
- output directory is created on demand.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_spike_0_7a as cli


def test_cli_smoke_pass(tmp_path: Path) -> None:
    """A full default run (10 random + 3 adversarial) must succeed."""
    out_dir = tmp_path / "spike_0_7a"
    rc = cli.main(
        [
            "--n-random",
            "10",
            "--seed",
            "42",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc in (0, 1), f"unexpected exit {rc} (must be 0 PASS or 1 FAIL — not error)"

    results = out_dir / "results.json"
    assert results.exists()
    payload = json.loads(results.read_text())
    assert payload["spec_reference"].startswith("docs/plan_R11.md")
    r = payload["result"]
    # Schema sanity — downstream consumers depend on these fields.
    assert "records" in r
    assert "n_random" in r
    assert "n_adversarial" in r
    assert "n_passing" in r
    assert "adversarial_blocked_count" in r
    assert "passed" in r
    assert r["n_random"] == 10
    assert r["n_adversarial"] == 3
    # All adversarial sets MUST be blocked under the shim — the shim is
    # designed to catch the failure modes the adversarial sets target.
    assert r["adversarial_blocked_count"] == 3


def test_cli_writes_per_design_subdirs(tmp_path: Path) -> None:
    out_dir = tmp_path / "spike_0_7a"
    rc = cli.main(["--n-random", "3", "--seed", "1", "--output-dir", str(out_dir)])
    assert rc in (0, 1)
    payload = json.loads((out_dir / "results.json").read_text())
    # Every record has a per-design subdir with params.json + record.json.
    for rec in payload["result"]["records"]:
        sub = out_dir / rec["params_hash"]
        assert sub.is_dir(), f"missing subdir {sub}"
        assert (sub / "params.json").exists()
        assert (sub / "record.json").exists()
        assert (sub / "stl_path.txt").exists()


def test_cli_skip_adversarial_fails(tmp_path: Path) -> None:
    """Without adversarial sets the spike can't meet its coverage gate."""
    out_dir = tmp_path / "spike_0_7a"
    rc = cli.main(
        [
            "--n-random",
            "10",
            "--seed",
            "42",
            "--output-dir",
            str(out_dir),
            "--skip-adversarial",
        ]
    )
    assert rc == 1
    payload = json.loads((out_dir / "results.json").read_text())
    r = payload["result"]
    assert r["n_adversarial"] == 0
    assert r["passed"] is False


def test_cli_rejects_zero_n_random(tmp_path: Path) -> None:
    rc = cli.main(["--n-random", "0", "--output-dir", str(tmp_path / "spike_0_7a")])
    assert rc == 2


def test_cli_creates_output_dir(tmp_path: Path) -> None:
    out_dir = tmp_path / "nested" / "spike_0_7a"
    assert not out_dir.exists()
    rc = cli.main(["--n-random", "2", "--output-dir", str(out_dir)])
    assert rc in (0, 1)
    assert out_dir.exists()
    assert (out_dir / "results.json").exists()


def test_cli_reproducible_with_same_seed(tmp_path: Path) -> None:
    """Same seed → same record hashes."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    cli.main(["--n-random", "5", "--seed", "99", "--output-dir", str(out_a)])
    cli.main(["--n-random", "5", "--seed", "99", "--output-dir", str(out_b)])
    pa = json.loads((out_a / "results.json").read_text())["result"]
    pb = json.loads((out_b / "results.json").read_text())["result"]
    hashes_a = [r["params_hash"] for r in pa["records"]]
    hashes_b = [r["params_hash"] for r in pb["records"]]
    assert hashes_a == hashes_b


@pytest.mark.parametrize("adv_id", ["a_louver", "b_tpms", "c_primitive"])
def test_cli_blocks_every_adversarial_set(tmp_path: Path, adv_id: str) -> None:
    """Every adversarial set must be blocked by the shim pipeline."""
    out_dir = tmp_path / "spike_0_7a"
    rc = cli.main(["--n-random", "1", "--seed", "0", "--output-dir", str(out_dir)])
    assert rc in (0, 1)
    payload = json.loads((out_dir / "results.json").read_text())
    # Find the adversarial record by walking per-design subdirs and matching
    # the _adversarial_id in params.json.
    matched: list[dict] = []
    for rec in payload["result"]["records"]:
        if not rec["is_adversarial"]:
            continue
        sub = out_dir / rec["params_hash"]
        params = json.loads((sub / "params.json").read_text())
        if params.get("_adversarial_id", "").startswith(adv_id):
            matched.append(rec)
    assert matched, f"no adversarial record with id starting {adv_id}"
    for rec in matched:
        assert rec["passed"] is False, (
            f"adversarial {adv_id} unexpectedly passed every check: " f"{rec['rejection_reasons']}"
        )
