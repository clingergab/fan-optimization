"""Tests for fanopt.utils.drive_io.

Validates .done / .heartbeat / .claim markers, atomic claim semantics,
heartbeat staleness detection, and composite-key marker naming.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from fanopt.utils.drive_io import (
    CLAIM_REAP_AGE_S,
    HEARTBEAT_STALE_AGE_S,
    composite_marker_name,
    done_marker_exists,
    done_marker_path,
    is_heartbeat_stale,
    release_claim,
    try_claim,
    write_done_marker,
    write_heartbeat,
)

# ---- done markers ---------------------------------------------------------


def test_done_marker_path_format(tmp_path: Path) -> None:
    p = done_marker_path(tmp_path, "abc" + "0" * 21)
    assert p.parent == tmp_path
    assert p.name == "abc" + "0" * 21 + ".done"


def test_write_done_marker_creates_file(tmp_path: Path) -> None:
    p = write_done_marker(tmp_path, "deadbeef" + "0" * 16)
    assert p.exists()
    assert done_marker_exists(tmp_path, "deadbeef" + "0" * 16)


def test_write_done_marker_idempotent(tmp_path: Path) -> None:
    write_done_marker(tmp_path, "h" * 24, contents="run1")
    write_done_marker(tmp_path, "h" * 24, contents="run2")
    assert done_marker_path(tmp_path, "h" * 24).read_text() == "run2"


def test_write_done_marker_creates_parent_dir(tmp_path: Path) -> None:
    deep = tmp_path / "designs" / "abc"
    write_done_marker(deep, "h" * 24)
    assert (deep / (("h" * 24) + ".done")).exists()


def test_done_marker_exists_false_when_absent(tmp_path: Path) -> None:
    assert done_marker_exists(tmp_path, "missing" + "0" * 17) is False


# ---- composite key marker -------------------------------------------------


def test_composite_marker_name_format() -> None:
    name = composite_marker_name(
        config_hash="c" * 24,
        physics_hash="p" * 24,
        material_hash="m" * 24,
        geometry_hash="g" * 24,
        fidelity=1,
        direction="productive",
    )
    assert name == "cccccccc-pppppppp-mmmmmmmm-gggggggg-1-productive.done"


def test_composite_marker_name_default_direction() -> None:
    name = composite_marker_name(
        config_hash="a" * 24,
        physics_hash="b" * 24,
        material_hash="c" * 24,
        geometry_hash="d" * 24,
        fidelity=0,
    )
    assert name.endswith("-0-default.done")


# ---- heartbeats -----------------------------------------------------------


def test_write_heartbeat_creates_file(tmp_path: Path) -> None:
    p = write_heartbeat(tmp_path)
    assert p.exists()
    assert p.name == ".heartbeat"


def test_heartbeat_fresh_when_just_written(tmp_path: Path) -> None:
    write_heartbeat(tmp_path)
    assert is_heartbeat_stale(tmp_path) is False


def test_heartbeat_stale_when_absent(tmp_path: Path) -> None:
    assert is_heartbeat_stale(tmp_path) is True


def test_heartbeat_stale_when_old(tmp_path: Path) -> None:
    """Set mtime far in the past; staleness must fire."""
    p = write_heartbeat(tmp_path)
    old_t = time.time() - HEARTBEAT_STALE_AGE_S - 60
    os.utime(p, (old_t, old_t))
    assert is_heartbeat_stale(tmp_path) is True


def test_heartbeat_custom_max_age(tmp_path: Path) -> None:
    p = write_heartbeat(tmp_path)
    # 100 s ago — fresh under default, stale under max_age=60.
    old_t = time.time() - 100
    os.utime(p, (old_t, old_t))
    assert is_heartbeat_stale(tmp_path, max_age_s=60) is True
    assert is_heartbeat_stale(tmp_path, max_age_s=300) is False


# ---- claim ----------------------------------------------------------------


def test_try_claim_succeeds_when_unclaimed(tmp_path: Path) -> None:
    p = try_claim(tmp_path, "h" * 24, session_id="s1")
    assert p is not None
    assert p.exists()
    assert "s1_" in p.read_text()


def test_try_claim_fails_when_already_claimed(tmp_path: Path) -> None:
    """Second claim attempt must return None, NOT raise."""
    first = try_claim(tmp_path, "h" * 24, session_id="s1")
    assert first is not None
    second = try_claim(tmp_path, "h" * 24, session_id="s2")
    assert second is None
    # First session's content must not be overwritten.
    assert "s1_" in first.read_text()


def test_release_claim_removes_file(tmp_path: Path) -> None:
    try_claim(tmp_path, "h" * 24, session_id="s1")
    removed = release_claim(tmp_path, "h" * 24)
    assert removed is True
    # Can re-claim after release.
    assert try_claim(tmp_path, "h" * 24, session_id="s2") is not None


def test_release_claim_returns_false_when_no_claim(tmp_path: Path) -> None:
    assert release_claim(tmp_path, "h" * 24) is False


def test_try_claim_writes_session_and_timestamp(tmp_path: Path) -> None:
    p = try_claim(tmp_path, "h" * 24, session_id="sessionA")
    content = p.read_text()
    assert content.startswith("sessionA_")


# ---- module constants ----------------------------------------------------


def test_claim_reap_age_15_min() -> None:
    """Per plan, orphaned claims age out at 15 minutes."""
    assert CLAIM_REAP_AGE_S == 900


def test_heartbeat_stale_age_15_min() -> None:
    """Watchdog flags sessions dead at 15 min."""
    assert HEARTBEAT_STALE_AGE_S == 900
