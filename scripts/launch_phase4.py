#!/usr/bin/env python
"""Phase 0 → Phase 4 handoff: create the `phase4-launch` git tag.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.6c (H10 lock) + §6.2.1
architecture-bandit growth-gate (the `+10%` rule that activates after this
tag exists).

**The gate this script enforces:**
- `data/spike_0_6c/PASS` marker must exist before the `phase4-launch` git
  tag can be created. This marker is written by `run_spike_0_6c.py` IFF
  both sub-spikes (0.6c.1 Tier-1 cfg sanity + 0.6c.2 NACA 0012 benchmark)
  passed. Without this marker, Phase 4's Tier-1 numerics are unvalidated
  and the 1000-h Phase 4 stop-rule budget cannot legitimately start
  counting.

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
DEFAULT_MARKER = REPO_ROOT / "data" / "spike_0_6c" / "PASS"
PHASE4_TAG = "phase4-launch"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--marker",
        type=Path,
        default=DEFAULT_MARKER,
        help="Path to the Spike 0.6c PASS marker (default: %(default)s).",
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


def _check_marker(marker: Path) -> tuple[bool, str]:
    """Return (passed, reason).

    `passed=True` iff the marker file exists. The marker is written as an
    empty file by run_spike_0_6c.py; its existence is the gate. Content is
    not parsed.
    """
    if not marker.exists():
        return False, (
            f"Spike 0.6c PASS marker not found at {marker}. "
            "Run `python scripts/run_spike_0_6c.py` and confirm both "
            "sub-spikes pass before attempting Phase 4 launch."
        )
    if marker.is_dir():
        return False, f"{marker} exists but is a directory; expected a file."
    return True, f"Spike 0.6c PASS marker present at {marker}."


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
        gate_passed, gate_reason = _check_marker(args.marker)

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
