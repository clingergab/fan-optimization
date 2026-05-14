"""Drive/JSONL ledger for evaluation results.

One Pydantic-validated JSON object per line in `session_<id>/results.jsonl`.
UTF-8, sorted keys, deterministic float precision (6 digits). Every row
carries `schema_version: int` so future migrations are explicit.

Spec reference: docs/plan_R11.md §Phase 4 step 51 + §6.2.5 (design_hash).

The full schema includes downstream Phase 4 fields (Pareto objectives,
stress-test fields, CFD intermediates) that don't have producers yet. The
dataclass declares them with `Optional[...] = None` defaults so Phase 0
spike runners can write minimal rows now, and Phase 4 fills in the rest
without bumping schema_version.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_HASH_PRECISION",
    "Tier",
    "Status",
    "FailureCode",
    "LedgerRow",
    "design_hash",
    "append_row",
    "read_rows",
    "dedupe_by_design_hash",
]


SCHEMA_VERSION: int = 2
"""Current ledger schema version. Bump on every field add/rename. See plan
§Phase 4 step 51; v2 added composite-key fields per the Round-9 H16 lock."""

DEFAULT_HASH_PRECISION: int = 6
"""Number of decimal digits to round numeric leaves to in design_hash."""


class Tier(int, Enum):
    """CFD fidelity tier — matches the multi-fidelity GP fidelity column."""

    TIER_MINUS_ONE = -1  # 2D steady CFD slice
    TIER_ZERO = 0  # 3D steady CFD
    TIER_ONE = 1  # 3D unsteady CFD


class Status(str, Enum):
    """Final status of an evaluation row."""

    OK = "ok"
    RETRIED = "retried"
    FAILED = "failed"
    REJECTED_HARD_CONSTRAINT = "rejected_hard_constraint"
    REJECTED_PRE_CFD = "rejected_pre_cfd"


class FailureCode(str, Enum):
    """Enum of expected failure modes. None if status == OK.

    From plan §9.4.2 + §Phase 4 step 51. Add new codes here as failure
    modes are discovered downstream; do NOT use bare strings in producer
    code.
    """

    CONFIG_MISMATCH = "config_mismatch"
    FEA_BENDING = "fea_bending"
    FEA_TENSION = "fea_tension"
    FEA_BEARING = "fea_bearing"
    FEA_COMBINED_TIP_DEFL = "fea_combined_tip_defl"
    FEA_COMBINED_PEAK_STRESS = "fea_combined_peak_stress"
    FEA_COMBINED_TORSION = "fea_combined_torsion"
    FEA_STRESS_TEST_FAIL = "fea_stress_test_fail"
    MASS_CAP = "mass_cap"
    COM_CAP = "com_cap"
    MFG_SCORE = "mfg_score"
    CFD_DIVERGED = "cfd_diverged"
    CFD_CFL_EXCEEDED = "cfd_cfl_exceeded"
    GEOMETRY_FAILED = "geometry_failed"
    CRASH = "crash"


def _round_leaves(obj: Any, precision: int) -> Any:
    """Walk `obj`, rounding every numeric leaf to `precision` digits.

    Doubles like 0.1+0.2 = 0.30000000000000004 silently break cross-session
    dedup if they leak into json.dumps. This pre-walk prevents that.
    Lists and tuples both serialize as JSON arrays, so we coerce tuple→list
    to keep the canonical form stable.
    """
    if isinstance(obj, float | np.floating):
        return round(float(obj), precision)
    if isinstance(obj, dict):
        return {k: _round_leaves(v, precision) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_round_leaves(x, precision) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def design_hash(params: dict, precision: int = DEFAULT_HASH_PRECISION) -> str:
    """Stable 24-hex hash of a canonical parameter dict.

    Rounds every numeric leaf to `precision` digits BEFORE json.dumps so
    double-precision noise across sessions/platforms does not leak in.
    blake2b/12-byte digest → 24-hex-char primary key, short enough to be
    a directory name.

    Spec reference: docs/plan_R11.md §6.2.5.

    Round-trip property the orchestrator asserts on every read:
        design_hash(load_params(designs/{hash}/params.json)) == hash
    """
    rounded = _round_leaves(params, precision)
    canonical = json.dumps(rounded, sort_keys=True)
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=12).hexdigest()


@dataclass
class LedgerRow:
    """One evaluation row.

    Mandatory fields populate at row creation time. Most other fields are
    optional and filled in by downstream producers; `to_dict()` only
    enforces the mandatory set.

    The 7-tuple composite-key fields (design_hash + physics_hash +
    config_hash + material_hash + geometry_hash + fidelity + run_direction
    per the H16 Round-9 lock) are declared as Optional. Phase 0 spike
    runners typically only have `design_hash` and `tier`; Phase 4 fills
    the rest.
    """

    # Mandatory
    schema_version: int
    design_hash: str
    tier: Tier
    status: Status
    wall_time_s: float
    timestamp_iso: str

    # Composite-key fields (H16 Round-9 lock) — Phase 4 producers fill in
    physics_hash: str | None = None
    config_hash: str | None = None
    material_hash: str | None = None
    geometry_hash: str | None = None
    run_direction: str | None = None  # 'productive' / 'return' for steady probes

    # Outcome
    failure_code: FailureCode | None = None
    retriable: bool | None = None
    retry_count: int = 0
    retry_history: list[str] = field(default_factory=list)

    # CFD intermediates
    cfl_max: float | None = None
    J_fan: float | None = None
    J_fan_delta: float | None = None
    J_fan_cycle_variance: float | None = None

    # Geometry / mass
    m_total_kg: float | None = None
    r_CoM_wrist_m: float | None = None
    I_wrist_kgm2: float | None = None

    # Stress-test fields
    stress_test_fail: bool | None = None

    # Free-form params snapshot (small dict; large blobs live on Drive)
    params: dict = field(default_factory=dict)

    # Producer-attached artifact paths (CFD outputs, STL files, etc.)
    artifacts: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-stable dict.

        Validates mandatory fields are non-empty / non-default. Floats are
        NOT rounded here — that only happens inside `design_hash`.
        """
        if not self.design_hash:
            raise ValueError("design_hash is required and must be non-empty")
        if not self.timestamp_iso:
            raise ValueError("timestamp_iso is required (use datetime.utcnow().isoformat())")
        if self.wall_time_s < 0:
            raise ValueError(f"wall_time_s must be ≥ 0, got {self.wall_time_s}")
        if self.retry_count < 0:
            raise ValueError(f"retry_count must be ≥ 0, got {self.retry_count}")
        return {
            "schema_version": self.schema_version,
            "design_hash": self.design_hash,
            "tier": int(self.tier),
            "status": self.status.value,
            "wall_time_s": self.wall_time_s,
            "timestamp_iso": self.timestamp_iso,
            "physics_hash": self.physics_hash,
            "config_hash": self.config_hash,
            "material_hash": self.material_hash,
            "geometry_hash": self.geometry_hash,
            "run_direction": self.run_direction,
            "failure_code": (self.failure_code.value if self.failure_code is not None else None),
            "retriable": self.retriable,
            "retry_count": self.retry_count,
            "retry_history": list(self.retry_history),
            "cfl_max": self.cfl_max,
            "J_fan": self.J_fan,
            "J_fan_delta": self.J_fan_delta,
            "J_fan_cycle_variance": self.J_fan_cycle_variance,
            "m_total_kg": self.m_total_kg,
            "r_CoM_wrist_m": self.r_CoM_wrist_m,
            "I_wrist_kgm2": self.I_wrist_kgm2,
            "stress_test_fail": self.stress_test_fail,
            "params": dict(self.params),
            "artifacts": dict(self.artifacts),
        }


def append_row(jsonl_path: Path | str, row: LedgerRow) -> None:
    """Append `row` to `jsonl_path` as one UTF-8 JSON line with sorted keys."""
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row.to_dict(), sort_keys=True, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_rows(jsonl_path: Path | str) -> list[dict[str, Any]]:
    """Read every JSONL row from `jsonl_path`. Skips blank lines.

    Returns raw dicts; producers can re-validate via LedgerRow if they
    need the typed view. Raises on malformed JSON.
    """
    path = Path(jsonl_path)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: malformed JSONL: {e}") from e
    return out


def dedupe_by_design_hash(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the most-recent row per design_hash (by timestamp_iso).

    The merger pass at Phase 4 step 56 dedupes cross-session JSONL rows
    on design_hash. Composite-key dedup (H16 7-tuple lock) is the Phase 4
    producer's responsibility; this helper is the simpler form for
    spike-level summaries.
    """
    by_hash: dict[str, dict[str, Any]] = {}
    for row in rows:
        h = row.get("design_hash")
        if not h:
            continue
        ts = row.get("timestamp_iso", "")
        existing = by_hash.get(h)
        if existing is None or ts >= existing.get("timestamp_iso", ""):
            by_hash[h] = row
    return list(by_hash.values())
