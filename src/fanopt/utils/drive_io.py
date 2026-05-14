"""Drive I/O — .done / .heartbeat / .claim markers, atomic claim, heartbeats.

Implements the marker scheme from plan §Phase 4 step 51 + the cross-
session claim protocol from §Phase 4 step 55. Phase 0 spike code uses
the simpler `done_marker_path` / `write_done_marker` pair; the atomic
O_CREAT|O_EXCL `claim` path is for Phase 4 cross-session orchestration.

**Composite-key marker (H16 Round-9 lock):**

    designs/{design_hash}/runs/
        {config_hash[:8]}-{physics_hash[:8]}-{material_hash[:8]}-{geometry_hash[:8]}-{fidelity}-{direction}.done

The 4-part hash prefix lets the orchestrator skip re-evaluation when the
composite key matches. The §Phase 4 step 51 migration script renames
legacy 6-tuple markers to the new path on first launch.

Phase 0 producers (Spike 0.2-0.7) only need the simpler
`write_done_marker(out_dir, design_hash)` form — composite-key markers
are wired up by Phase 4.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "CLAIM_REAP_AGE_S",
    "HEARTBEAT_STALE_AGE_S",
    "done_marker_path",
    "write_done_marker",
    "done_marker_exists",
    "composite_marker_name",
    "heartbeat_path",
    "write_heartbeat",
    "is_heartbeat_stale",
    "claim_path",
    "try_claim",
    "release_claim",
    "ClaimError",
]


CLAIM_REAP_AGE_S: int = 900
"""Per plan §Phase 4 cross-session claim section: orphaned `.claim` files
older than 15 minutes (900 s) are reaped if the session is no longer in
m3_active_sessions.txt. Matches the watchdog stop threshold."""

HEARTBEAT_STALE_AGE_S: int = 900
"""Sessions with heartbeat mtime older than 15 min are flagged dead."""


class ClaimError(Exception):
    """Raised when a session tries to claim a design that another session
    already holds. The orchestrator must skip and move on; the holder's
    JSONL row will appear in the merger pass."""


def done_marker_path(out_dir: Path | str, design_hash: str) -> Path:
    """Simple-form marker path: `{out_dir}/{design_hash}.done`.

    Used by Phase 0 spikes that don't need composite-key dedup.
    """
    return Path(out_dir) / f"{design_hash}.done"


def composite_marker_name(
    *,
    config_hash: str,
    physics_hash: str,
    material_hash: str,
    geometry_hash: str,
    fidelity: int,
    direction: str = "default",
) -> str:
    """Composite-key marker filename per H16 7-tuple lock.

    Hash prefixes are the first 8 hex chars of each blake2b/12 hash;
    fidelity is the integer tier; direction is the run direction
    ('productive' / 'return' / 'default').
    """
    return (
        f"{config_hash[:8]}-{physics_hash[:8]}-{material_hash[:8]}-"
        f"{geometry_hash[:8]}-{fidelity}-{direction}.done"
    )


def write_done_marker(
    out_dir: Path | str,
    design_hash: str,
    *,
    contents: str = "",
) -> Path:
    """Atomically write a `.done` marker for a successfully-completed
    evaluation. Idempotent — overwrites if already present."""
    path = done_marker_path(out_dir, design_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path


def done_marker_exists(out_dir: Path | str, design_hash: str) -> bool:
    """Check whether a `.done` marker exists for `design_hash` in `out_dir`."""
    return done_marker_path(out_dir, design_hash).exists()


def heartbeat_path(session_dir: Path | str) -> Path:
    return Path(session_dir) / ".heartbeat"


def write_heartbeat(session_dir: Path | str) -> Path:
    """Write a heartbeat with the current UTC timestamp.

    The presence + mtime of the heartbeat file are what the watchdog
    monitors; content is the ISO timestamp for human inspection.
    """
    path = heartbeat_path(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    path.write_text(now, encoding="utf-8")
    return path


def is_heartbeat_stale(
    session_dir: Path | str,
    *,
    max_age_s: int = HEARTBEAT_STALE_AGE_S,
) -> bool:
    """True if no heartbeat exists or its mtime is older than max_age_s.

    Per plan §Phase 4 step 78 telemetry: sessions with stale heartbeat
    are flagged 'dead' (Colab disconnect, OOM, or browser tab closed).
    """
    path = heartbeat_path(session_dir)
    if not path.exists():
        return True
    age_s = time.time() - path.stat().st_mtime
    return age_s > max_age_s


def claim_path(out_dir: Path | str, design_hash: str) -> Path:
    return Path(out_dir) / f"{design_hash}.claim"


def try_claim(
    out_dir: Path | str,
    design_hash: str,
    *,
    session_id: str,
) -> Path | None:
    """Attempt atomic O_CREAT|O_EXCL claim on `{out_dir}/{design_hash}.claim`.

    Returns the claim path on success, None if another session holds it.
    The claim file is written with content `{session_id}_{utc_iso}` so the
    Drive consistency re-read confirmation step (see plan §Phase 4 step 55
    M6 lock) can detect race losers.

    The 30 s post-claim sleep + re-read confirmation is NOT done here —
    it's the caller's responsibility, since the appropriate sleep depends
    on the observed Drive write-to-visibility latency.
    """
    path = claim_path(out_dir, design_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(path), flags)
    except FileExistsError:
        return None
    try:
        stamp = datetime.now(timezone.utc).isoformat()
        os.write(fd, f"{session_id}_{stamp}".encode())
    finally:
        os.close(fd)
    return path


def release_claim(out_dir: Path | str, design_hash: str) -> bool:
    """Release a previously-acquired claim. Returns True if removed, False
    if no claim existed."""
    path = claim_path(out_dir, design_hash)
    if not path.exists():
        return False
    path.unlink()
    return True
