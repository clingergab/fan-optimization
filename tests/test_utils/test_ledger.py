"""Tests for fanopt.utils.ledger.

Validates the design_hash deterministic-rounding property (the spec's
canonical regression case from §6.2.5), the LedgerRow schema, JSONL
round-trip, and the dedupe-by-design-hash merger helper.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from fanopt.utils.ledger import (
    DEFAULT_HASH_PRECISION,
    SCHEMA_VERSION,
    FailureCode,
    LedgerRow,
    Status,
    Tier,
    append_row,
    dedupe_by_design_hash,
    design_hash,
    read_rows,
)

# ---- design_hash ----------------------------------------------------------


def test_design_hash_deterministic_round_off_canonical_case() -> None:
    """Plan §6.2.5 required regression: {x: 0.1+0.2} == {x: 0.3}."""
    assert design_hash({"x": 0.1 + 0.2}) == design_hash({"x": 0.3})


def test_design_hash_nested_dict_round_off() -> None:
    """Plan §6.2.5 required regression: nested dicts also round."""
    assert design_hash({"nested": {"x": 0.1 + 0.2}}) == design_hash({"nested": {"x": 0.3}})


def test_design_hash_handles_lists_and_tuples_equivalently() -> None:
    """list and tuple serialize the same way under canonical JSON."""
    assert design_hash({"v": [0.1 + 0.2, 1.0]}) == design_hash({"v": (0.3, 1.0)})


def test_design_hash_handles_numpy_scalars() -> None:
    """np.float64 must round the same as Python float."""
    py = design_hash({"x": 0.3})
    npy = design_hash({"x": np.float64(0.3)})
    assert py == npy


def test_design_hash_distinguishes_unrelated_params() -> None:
    assert design_hash({"x": 0.3}) != design_hash({"x": 0.4})
    assert design_hash({"x": 0.3}) != design_hash({"y": 0.3})


def test_design_hash_24_hex_chars() -> None:
    """blake2b/12-byte digest → 24 hex chars."""
    h = design_hash({"x": 1.0})
    assert len(h) == 24
    assert all(c in "0123456789abcdef" for c in h)


def test_design_hash_sorted_keys_irrelevant_at_input() -> None:
    """{'a': 1, 'b': 2} and {'b': 2, 'a': 1} are the same dict — same hash."""
    assert design_hash({"a": 1, "b": 2}) == design_hash({"b": 2, "a": 1})


def test_design_hash_precision_param() -> None:
    """Coarser precision collapses near-equal values."""
    h_fine = design_hash({"x": 0.123456}, precision=6)
    h_coarse = design_hash({"x": 0.123457}, precision=4)
    h_coarse_2 = design_hash({"x": 0.123459}, precision=4)
    assert h_coarse == h_coarse_2  # both round to 0.1235
    assert h_fine != h_coarse_2


def test_design_hash_numpy_integer_serializable() -> None:
    """np.int64 must json-serialize without TypeError."""
    h = design_hash({"n": np.int64(5)})
    assert isinstance(h, str)


# ---- LedgerRow ------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row(**overrides):
    defaults = dict(
        schema_version=SCHEMA_VERSION,
        design_hash="a" * 24,
        tier=Tier.TIER_ZERO,
        status=Status.OK,
        wall_time_s=1.0,
        timestamp_iso=_now_iso(),
    )
    defaults.update(overrides)
    return LedgerRow(**defaults)


def test_minimal_row_validates() -> None:
    row = _row()
    d = row.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["tier"] == 0
    assert d["status"] == "ok"
    assert d["failure_code"] is None
    assert d["retry_count"] == 0


def test_row_with_failure_code_serializes_enum_value() -> None:
    row = _row(
        status=Status.FAILED,
        failure_code=FailureCode.CFD_DIVERGED,
        retriable=True,
        retry_count=1,
    )
    d = row.to_dict()
    assert d["status"] == "failed"
    assert d["failure_code"] == "cfd_diverged"
    assert d["retriable"] is True
    assert d["retry_count"] == 1


def test_row_rejects_empty_design_hash() -> None:
    with pytest.raises(ValueError, match="design_hash"):
        _row(design_hash="").to_dict()


def test_row_rejects_empty_timestamp() -> None:
    with pytest.raises(ValueError, match="timestamp_iso"):
        _row(timestamp_iso="").to_dict()


def test_row_rejects_negative_wall_time() -> None:
    with pytest.raises(ValueError, match="wall_time_s"):
        _row(wall_time_s=-0.001).to_dict()


def test_row_rejects_negative_retry_count() -> None:
    with pytest.raises(ValueError, match="retry_count"):
        _row(retry_count=-1).to_dict()


def test_row_preserves_composite_key_fields() -> None:
    row = _row(
        physics_hash="b" * 24,
        config_hash="c" * 24,
        material_hash="d" * 24,
        geometry_hash="e" * 24,
        run_direction="productive",
    )
    d = row.to_dict()
    assert d["physics_hash"] == "b" * 24
    assert d["run_direction"] == "productive"


def test_row_preserves_artifacts_and_params() -> None:
    row = _row(
        params={"blade_count": 10, "panel_thickness_t0": 0.003},
        artifacts={"stl": "designs/abc/blade.stl"},
    )
    d = row.to_dict()
    assert d["params"]["blade_count"] == 10
    assert d["artifacts"]["stl"] == "designs/abc/blade.stl"


# ---- JSONL I/O ------------------------------------------------------------


def test_append_and_read_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "results.jsonl"
    rows = [
        _row(design_hash="aaaa" + "0" * 20, wall_time_s=1.5),
        _row(design_hash="bbbb" + "0" * 20, wall_time_s=2.5, J_fan=0.05),
    ]
    for r in rows:
        append_row(p, r)
    loaded = read_rows(p)
    assert len(loaded) == 2
    assert loaded[0]["design_hash"] == "aaaa" + "0" * 20
    assert loaded[1]["J_fan"] == 0.05


def test_append_creates_parent_dir(tmp_path: Path) -> None:
    p = tmp_path / "sessions" / "s1" / "results.jsonl"
    append_row(p, _row())
    assert p.exists()


def test_read_rows_returns_empty_when_missing(tmp_path: Path) -> None:
    assert read_rows(tmp_path / "does_not_exist.jsonl") == []


def test_read_rows_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "with_blanks.jsonl"
    line = json.dumps(_row().to_dict(), sort_keys=True)
    p.write_text(f"\n{line}\n\n{line}\n\n")
    rows = read_rows(p)
    assert len(rows) == 2


def test_read_rows_raises_on_malformed_json(tmp_path: Path) -> None:
    p = tmp_path / "broken.jsonl"
    p.write_text("not valid json\n")
    with pytest.raises(ValueError, match="malformed JSONL"):
        read_rows(p)


def test_jsonl_lines_are_sorted_keys(tmp_path: Path) -> None:
    """Producer must emit sorted-key JSON for cross-platform reproducibility."""
    p = tmp_path / "results.jsonl"
    append_row(p, _row())
    line = p.read_text().strip()
    parsed = json.loads(line)
    re_dumped = json.dumps(parsed, sort_keys=True)
    assert line == re_dumped


# ---- dedupe_by_design_hash ----------------------------------------------


def test_dedupe_keeps_most_recent_per_hash() -> None:
    """Highest timestamp_iso wins."""
    h = "a" * 24
    rows = [
        {"design_hash": h, "timestamp_iso": "2025-01-01T00:00:00", "wall_time_s": 1.0},
        {"design_hash": h, "timestamp_iso": "2025-06-01T00:00:00", "wall_time_s": 2.0},
        {"design_hash": h, "timestamp_iso": "2025-03-01T00:00:00", "wall_time_s": 3.0},
    ]
    out = dedupe_by_design_hash(rows)
    assert len(out) == 1
    assert out[0]["timestamp_iso"] == "2025-06-01T00:00:00"


def test_dedupe_preserves_distinct_hashes() -> None:
    rows = [
        {"design_hash": "a" * 24, "timestamp_iso": "2025-01-01T00:00:00"},
        {"design_hash": "b" * 24, "timestamp_iso": "2025-01-01T00:00:00"},
    ]
    out = dedupe_by_design_hash(rows)
    assert {r["design_hash"] for r in out} == {"a" * 24, "b" * 24}


def test_dedupe_skips_rows_without_design_hash() -> None:
    rows = [
        {"design_hash": "a" * 24, "timestamp_iso": "2025-01-01"},
        {"timestamp_iso": "2025-01-01"},  # no design_hash
    ]
    out = dedupe_by_design_hash(rows)
    assert len(out) == 1


def test_dedupe_empty_input() -> None:
    assert dedupe_by_design_hash([]) == []


# ---- module-level constants -----------------------------------------------


def test_schema_version_is_2() -> None:
    """v2 includes composite-key fields per the H16 Round-9 lock."""
    assert SCHEMA_VERSION == 2


def test_default_hash_precision_is_6() -> None:
    assert DEFAULT_HASH_PRECISION == 6
