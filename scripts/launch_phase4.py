#!/usr/bin/env python
"""Phase 0 → Phase 4 handoff: create the `phase4-launch` git tag.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.6c (H10 lock) + §Phase 0
Spike 0.6d (2026-05-14 H10 supplement) + §6.2.1 architecture-bandit
growth-gate (the `+10%` rule that activates after this tag exists).

**The gate this script enforces (V1, post-2026-05-14 dual-gate):**

The script requires BOTH of the following markers before creating the
`phase4-launch` git tag:

- `data/spike_0_6c/PASS` — written by `run_spike_0_6c.py` iff Sub-spike
  0.6c.1 (Tier-1 cfg sanity) passes. Sub-spike 0.6c.2 was deferred to
  Phase 5 step 62.5 on 2026-05-14 (see `docs/phase_logs/spike_0_6c.md`
  Note 1).
- `data/spike_0_6d/PASS` — written by `run_spike_0_6d.py` iff Sub-spike
  0.6d.2's two-frequency added-mass consistency check passes (the sole,
  normalization-invariant gate after the 2026-05-15 redesign). Sub-spike
  0.6d.1 (symmetry/dimensional) and 0.6d.3 (incompressible) are advisory
  and do NOT gate (see `docs/phase_logs/phase_0_signoff.md` Notes 2-3).

Without BOTH markers, Phase 4's Tier-1 numerics lack independent
quantitative evidence and the 1000-h Phase 4 stop-rule budget cannot
legitimately start counting.

**Side effects:**
- Creates the lightweight `phase4-launch` git tag at HEAD.
- The plan calls out a `.git/hooks/pre-commit` architecture-bandit growth
  gate that is INERT before this tag exists. After this tag, the hook
  reads `HEAD~1`'s `configs/architecture_enumeration.yaml` and enforces
  the +10% ceiling. This script does NOT install the hook itself — that
  ships separately under `.git/hooks/`.

**Usage:**

    # Check whether the gate would pass (no side effects):
    python scripts/launch_phase4.py --check

    # Actually create the tag (refuses if marker absent or tag exists):
    python scripts/launch_phase4.py

    # Override (used only by recovery procedures, requires explicit flag):
    python scripts/launch_phase4.py --force
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKER_06C = REPO_ROOT / "data" / "spike_0_6c" / "PASS"
DEFAULT_MARKER_06D = REPO_ROOT / "data" / "spike_0_6d" / "PASS"
DEFAULT_MARKER = DEFAULT_MARKER_06C  # back-compat alias for the original `--marker`
PHASE4_TAG = "phase4-launch"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--marker",
        type=Path,
        default=DEFAULT_MARKER_06C,
        help="Path to the Spike 0.6c PASS marker (default: %(default)s).",
    )
    parser.add_argument(
        "--marker-06d",
        type=Path,
        default=DEFAULT_MARKER_06D,
        help=(
            "Path to the Spike 0.6d PASS marker (H10 supplement; "
            "2026-05-14 addition). Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check gate status without creating the tag. Exit 0 if gate "
        "passes, exit 1 if gate fails. Side-effect-free.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass the marker check. Use only for recovery (e.g., the "
        "marker was lost but the spike provably passed in CI). The "
        "operator is responsible for documenting the override in the "
        "spike's phase log.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repo root (used for git operations). Default: %(default)s.",
    )
    return parser.parse_args(argv)


def _check_single_marker(marker: Path, *, spike_label: str, runner_hint: str) -> tuple[bool, str]:
    """Return (passed, reason) for one marker path.

    `passed=True` iff the marker file exists. Markers are empty sentinel
    files written by each spike's aggregator; existence is the gate, content
    is not parsed.
    """
    if not marker.exists():
        return False, f"{spike_label} marker not found at {marker}. {runner_hint}"
    if marker.is_dir():
        return False, f"{marker} exists but is a directory; expected a file."
    return True, f"{spike_label} marker present at {marker}."


def _check_marker(marker_06c: Path, marker_06d: Path) -> tuple[bool, str]:
    """Return (passed, reason) for the dual-gate (0.6c AND 0.6d).

    Phase 4 launch is gated on BOTH markers per the 2026-05-14 plan revision.
    The order is intentional: we check 0.6c first (the V1-original gate),
    then 0.6d (the H10 supplement). The first failure short-circuits — its
    message guides the operator to the right runner.
    """
    runner_06c = (
        "Run `python scripts/run_spike_0_6c.py` to write the marker "
        "iff Sub-spike 0.6c.1 (Tier-1 cfg sanity) passes. Sub-spike "
        "0.6c.2 was deferred to Phase 5 step 62.5 on 2026-05-14; see "
        "docs/phase_logs/spike_0_6c.md Note 1."
    )
    runner_06d = (
        "Run `python scripts/run_spike_0_6d.py` to write the marker "
        "iff Sub-spike 0.6d.2's two-frequency added-mass consistency "
        "check passes (the sole gate post-2026-05-15 redesign). "
        "Sub-spikes 0.6d.1 + 0.6d.3 are advisory and do NOT gate. See "
        "docs/phase_logs/phase_0_signoff.md Note 2 for the rationale."
    )

    ok_c, reason_c = _check_single_marker(
        marker_06c, spike_label="Spike 0.6c PASS", runner_hint=runner_06c
    )
    if not ok_c:
        return False, reason_c
    ok_d, reason_d = _check_single_marker(
        marker_06d, spike_label="Spike 0.6d PASS", runner_hint=runner_06d
    )
    if not ok_d:
        return False, reason_d
    return True, f"Both markers present: 0.6c={marker_06c}; 0.6d={marker_06d}."


def _tag_exists(repo_root: Path, tag: str) -> bool:
    """True iff `tag` already exists in the repo."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "tag", "--list", tag],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and tag in result.stdout.split()


def _create_tag(repo_root: Path, tag: str) -> tuple[bool, str]:
    """Create a lightweight git tag at HEAD."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "tag", tag],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False, (
            result.stderr.strip()
            or result.stdout.strip()
            or f"git tag {tag} failed (rc={result.returncode})"
        )
    return True, f"Created git tag `{tag}` at HEAD."


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Gate check.
    if args.force:
        gate_passed = True
        gate_reason = "FORCED — marker check bypassed via --force."
        print(f"[launch_phase4] WARNING: {gate_reason}", file=sys.stderr)
    else:
        gate_passed, gate_reason = _check_marker(args.marker, args.marker_06d)

    print(f"[launch_phase4] gate: {'PASS' if gate_passed else 'FAIL'}  {gate_reason}")

    if args.check:
        # Side-effect-free path.
        return 0 if gate_passed else 1

    if not gate_passed:
        print(
            "[launch_phase4] refusing to create tag. Run with --check to "
            "audit gate status, or --force only for documented recovery.",
            file=sys.stderr,
        )
        return 1

    if _tag_exists(args.repo_root, PHASE4_TAG):
        print(
            f"[launch_phase4] tag `{PHASE4_TAG}` already exists. Nothing to do.",
            file=sys.stderr,
        )
        return 0

    ok, msg = _create_tag(args.repo_root, PHASE4_TAG)
    print(f"[launch_phase4] {msg}", file=sys.stderr if not ok else sys.stdout)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
