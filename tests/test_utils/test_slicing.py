"""Tests for fanopt.utils.slicing.

Validates round-robin assignment, pointer + slice file I/O, and the
dead-session rebalance procedure.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fanopt.utils.slicing import (
    POINTER_FILENAME,
    SLICE_FILENAME_TEMPLATE,
    SliceAssignment,
    load_pointer_version,
    read_assignment,
    rebalance_dead_session,
    round_robin_assign,
    write_assignment,
    write_pointer_version,
)


# ---- round_robin_assign ---------------------------------------------------


def test_round_robin_distributes_evenly() -> None:
    hashes = ["h0", "h1", "h2", "h3", "h4", "h5"]
    assn = round_robin_assign(hashes, ["s0", "s1", "s2"], version=1)
    assert assn.hashes_for("s0") == ("h0", "h3")
    assert assn.hashes_for("s1") == ("h1", "h4")
    assert assn.hashes_for("s2") == ("h2", "h5")


def test_round_robin_handles_remainder() -> None:
    """5 hashes / 2 sessions → s0 gets 3, s1 gets 2."""
    assn = round_robin_assign(
        ["a", "b", "c", "d", "e"], ["s0", "s1"], version=1
    )
    assert assn.hashes_for("s0") == ("a", "c", "e")
    assert assn.hashes_for("s1") == ("b", "d")


def test_round_robin_empty_hashes_gives_empty_slices() -> None:
    assn = round_robin_assign([], ["s0", "s1"], version=1)
    assert assn.hashes_for("s0") == ()
    assert assn.hashes_for("s1") == ()
    assert assn.all_hashes == ()


def test_round_robin_rejects_empty_session_list() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        round_robin_assign(["a"], [], version=1)


def test_round_robin_rejects_duplicate_session_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        round_robin_assign(["a"], ["s0", "s0"], version=1)


def test_round_robin_version_propagates() -> None:
    assn = round_robin_assign(["h"], ["s0"], version=7)
    assert assn.version == 7


def test_hashes_for_unknown_session_returns_empty() -> None:
    assn = round_robin_assign(["a"], ["s0"], version=1)
    assert assn.hashes_for("not_a_session") == ()


# ---- write / read assignment ----------------------------------------------


def test_write_read_assignment_round_trip(tmp_path: Path) -> None:
    assn = round_robin_assign(["a", "b", "c"], ["s0", "s1"], version=3)
    write_assignment(tmp_path, assn)
    loaded = read_assignment(tmp_path, version=3)
    assert loaded.version == 3
    assert loaded.by_session == assn.by_session


def test_write_assignment_filename_format(tmp_path: Path) -> None:
    assn = round_robin_assign(["a"], ["s0"], version=42)
    path = write_assignment(tmp_path, assn)
    assert path.name == SLICE_FILENAME_TEMPLATE.format(version=42)
    assert path.name == "slice_assignments_v42.json"


def test_write_assignment_creates_parent(tmp_path: Path) -> None:
    deep = tmp_path / "phase4" / "assignments"
    assn = round_robin_assign(["a"], ["s0"], version=1)
    write_assignment(deep, assn)
    assert (deep / "slice_assignments_v1.json").exists()


def test_read_assignment_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_assignment(tmp_path, version=999)


def test_written_assignment_is_sorted_json(tmp_path: Path) -> None:
    assn = round_robin_assign(["a"], ["s0"], version=1)
    p = write_assignment(tmp_path, assn)
    raw = p.read_text()
    parsed = json.loads(raw)
    assert parsed["version"] == 1
    assert parsed["by_session"]["s0"] == ["a"]


# ---- pointer file --------------------------------------------------------


def test_pointer_starts_at_zero_when_absent(tmp_path: Path) -> None:
    assert load_pointer_version(tmp_path) == 0


def test_pointer_round_trip(tmp_path: Path) -> None:
    write_pointer_version(tmp_path, 5)
    assert load_pointer_version(tmp_path) == 5


def test_pointer_filename_is_next_batch_txt(tmp_path: Path) -> None:
    write_pointer_version(tmp_path, 1)
    assert (tmp_path / POINTER_FILENAME).exists()
    assert POINTER_FILENAME == "next_batch.txt"


def test_pointer_rejects_negative(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="≥ 0"):
        write_pointer_version(tmp_path, -1)


def test_pointer_overwrites_on_bump(tmp_path: Path) -> None:
    write_pointer_version(tmp_path, 1)
    write_pointer_version(tmp_path, 7)
    assert load_pointer_version(tmp_path) == 7


# ---- rebalance_dead_session -----------------------------------------------


def test_rebalance_redistributes_unfinished_hashes() -> None:
    current = round_robin_assign(
        ["a", "b", "c", "d"], ["s0", "s1", "s2"], version=1
    )
    # s2 was assigned 'c'; it goes dead before completing.
    # s0 completed 'a'; s1 completed 'b'.
    new = rebalance_dead_session(
        current,
        dead_session_id="s2",
        completed_hashes_by_session={"s0": ["a"], "s1": ["b"]},
        new_version=2,
    )
    assert new.version == 2
    assert "s2" not in new.by_session
    # s0's unfinished was 'd' (since 'a' done); s1's was empty (since 'b' done);
    # s2's unfinished was 'c'. Round-robin onto [s0, s1]: 'c' → s0.
    assert "c" in new.hashes_for("s0")
    assert "d" in new.hashes_for("s0")


def test_rebalance_dead_session_unknown_raises() -> None:
    current = round_robin_assign(["a"], ["s0"], version=1)
    with pytest.raises(ValueError, match="not in current assignment"):
        rebalance_dead_session(
            current,
            dead_session_id="ghost",
            completed_hashes_by_session={},
            new_version=2,
        )


def test_rebalance_fails_when_no_survivors() -> None:
    current = round_robin_assign(["a"], ["s0"], version=1)
    with pytest.raises(ValueError, match="no surviving sessions"):
        rebalance_dead_session(
            current,
            dead_session_id="s0",
            completed_hashes_by_session={},
            new_version=2,
        )


def test_rebalance_preserves_completed_skipped() -> None:
    """Already-completed hashes are NOT redistributed."""
    current = round_robin_assign(
        ["a", "b", "c"], ["s0", "s1", "s2"], version=1
    )
    new = rebalance_dead_session(
        current,
        dead_session_id="s2",
        completed_hashes_by_session={"s0": [], "s1": [], "s2": ["c"]},
        new_version=2,
    )
    # 'c' was already done before s2 died — not redistributed.
    assert "c" not in new.hashes_for("s0")
    assert "c" not in new.hashes_for("s1")


def test_all_hashes_property_flattens_in_session_order() -> None:
    assn = SliceAssignment(
        version=1,
        by_session={"s0": ("a", "b"), "s1": ("c",)},
    )
    assert assn.all_hashes == ("a", "b", "c")
