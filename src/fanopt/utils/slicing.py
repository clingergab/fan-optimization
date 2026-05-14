"""Round-robin slice assignment + rebalance for Phase 4 cross-session BO.

Implements `slice_assignments_v{N}.json` per plan §Phase 4 step 48 + §12.1.
Each BO step writes a new version of the assignment file; sessions poll
`next_batch.txt` (or the equivalent pointer) for the current version and
read their slice from `slice_assignments_v{N}.json`.

**Design choice:** assignment is purely deterministic and pre-sliced on
the M3 (single-writer barrier — only the M3 writes assignment files).
Sessions never race to claim slices because the assignment file gives
them an explicit list; the cross-session `.claim` mechanism in
`drive_io.try_claim` is a safety net for hashes that appear in two slices
(rebalance race).

**Rebalance:** when a session goes dead (`drive_io.is_heartbeat_stale`),
the M3 calls `rebalance_dead_session` which produces the next assignment
version with the dead session's remaining hashes redistributed round-
robin across the survivors.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "SLICE_FILENAME_TEMPLATE",
    "POINTER_FILENAME",
    "SliceAssignment",
    "round_robin_assign",
    "write_assignment",
    "read_assignment",
    "load_pointer_version",
    "write_pointer_version",
    "rebalance_dead_session",
]


SLICE_FILENAME_TEMPLATE: str = "slice_assignments_v{version}.json"
POINTER_FILENAME: str = "next_batch.txt"
"""Plain-text pointer containing the current slice-assignment version
(integer). Plan §Phase 4 step 48 calls this `next_batch.txt` and the
versioned file `slice_assignments_v{N}.json`."""


@dataclass(frozen=True)
class SliceAssignment:
    """Map of session_id → list of design_hashes that session should run.

    `version` is the slice version; sessions read `POINTER_FILENAME` to
    discover the latest version, then load `SLICE_FILENAME_TEMPLATE.format(
    version=version)` to get their own slice.
    """

    version: int
    by_session: Mapping[str, tuple[str, ...]]

    @property
    def all_hashes(self) -> tuple[str, ...]:
        """Flatten every session's hashes (preserves insertion order per
        session, sessions in dict insertion order)."""
        out: list[str] = []
        for hashes in self.by_session.values():
            out.extend(hashes)
        return tuple(out)

    def hashes_for(self, session_id: str) -> tuple[str, ...]:
        return tuple(self.by_session.get(session_id, ()))


def round_robin_assign(
    design_hashes: Sequence[str],
    session_ids: Sequence[str],
    *,
    version: int,
) -> SliceAssignment:
    """Deal `design_hashes` round-robin to `session_ids`.

    With N hashes and S sessions: session i gets hashes [i, i+S, i+2S, …].
    Stable wrt session order. Empty session list → ValueError. Empty hash
    list → all sessions get empty tuples.
    """
    if not session_ids:
        raise ValueError("session_ids must be non-empty")
    if len(set(session_ids)) != len(session_ids):
        raise ValueError(f"duplicate session_ids: {session_ids}")
    by_session: dict[str, list[str]] = {sid: [] for sid in session_ids}
    n_sessions = len(session_ids)
    for i, h in enumerate(design_hashes):
        by_session[session_ids[i % n_sessions]].append(h)
    return SliceAssignment(
        version=version,
        by_session={sid: tuple(by_session[sid]) for sid in session_ids},
    )


def _slice_path(drive_dir: Path | str, version: int) -> Path:
    return Path(drive_dir) / SLICE_FILENAME_TEMPLATE.format(version=version)


def _pointer_path(drive_dir: Path | str) -> Path:
    return Path(drive_dir) / POINTER_FILENAME


def write_assignment(
    drive_dir: Path | str,
    assignment: SliceAssignment,
) -> Path:
    """Write `assignment` to `slice_assignments_v{N}.json`. Does NOT bump
    the pointer — call `write_pointer_version` separately so the pointer
    bump is atomic with whatever else the M3 needs to swap (e.g.,
    parameter-box updates)."""
    path = _slice_path(drive_dir, assignment.version)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": assignment.version,
        "by_session": {sid: list(hashes) for sid, hashes in assignment.by_session.items()},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_assignment(
    drive_dir: Path | str,
    version: int,
) -> SliceAssignment:
    """Read `slice_assignments_v{N}.json`. Raises FileNotFoundError if
    missing — sessions usually call `load_pointer_version` first."""
    path = _slice_path(drive_dir, version)
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_session = {sid: tuple(hashes) for sid, hashes in payload.get("by_session", {}).items()}
    return SliceAssignment(version=int(payload["version"]), by_session=by_session)


def load_pointer_version(drive_dir: Path | str) -> int:
    """Read `next_batch.txt`. Returns 0 if the pointer doesn't exist
    (initial campaign state — no slices yet)."""
    path = _pointer_path(drive_dir)
    if not path.exists():
        return 0
    return int(path.read_text(encoding="utf-8").strip())


def write_pointer_version(drive_dir: Path | str, version: int) -> Path:
    """Bump `next_batch.txt` to `version`. Atomic single-writer (M3 only)."""
    if version < 0:
        raise ValueError(f"version must be ≥ 0, got {version}")
    path = _pointer_path(drive_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{version}\n", encoding="utf-8")
    return path


def rebalance_dead_session(
    current: SliceAssignment,
    *,
    dead_session_id: str,
    completed_hashes_by_session: Mapping[str, Sequence[str]],
    new_version: int,
) -> SliceAssignment:
    """Produce a new assignment where the dead session's unfinished hashes
    are redistributed round-robin across the survivors.

    A session's "unfinished" hashes are those in its current slice that
    don't appear in `completed_hashes_by_session[session_id]`. The dead
    session is removed entirely from the new assignment (it gets no new
    hashes). Survivor allocations carry forward their unfinished hashes
    plus a round-robin share of the dead session's unfinished ones.

    Spec reference: docs/plan_R11.md §Phase 4 step 78 telemetry +
    rebalance procedure.
    """
    if dead_session_id not in current.by_session:
        raise ValueError(
            f"dead session {dead_session_id!r} not in current assignment "
            f"(sessions: {list(current.by_session)})"
        )
    survivors = [sid for sid in current.by_session if sid != dead_session_id]
    if not survivors:
        raise ValueError(f"cannot rebalance {dead_session_id!r}: no surviving sessions")

    def _unfinished(sid: str) -> list[str]:
        done = set(completed_hashes_by_session.get(sid, ()))
        return [h for h in current.by_session.get(sid, ()) if h not in done]

    # Survivors carry their own unfinished hashes forward.
    new_by_session: dict[str, list[str]] = {sid: _unfinished(sid) for sid in survivors}
    # Dead session's leftovers get round-robin'd onto survivors.
    dead_leftover = _unfinished(dead_session_id)
    for i, h in enumerate(dead_leftover):
        new_by_session[survivors[i % len(survivors)]].append(h)

    return SliceAssignment(
        version=new_version,
        by_session={sid: tuple(new_by_session[sid]) for sid in survivors},
    )
