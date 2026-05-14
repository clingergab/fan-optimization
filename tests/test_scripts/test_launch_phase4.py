"""CLI tests for scripts/launch_phase4.py.

Validates the Phase 4 launch gate behavior: marker present/absent, --check
side-effect-freeness, --force override, tag idempotence, and exit codes.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.6c (H10 lock).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import launch_phase4 as cli

# ---- isolated git repo fixture --------------------------------------------


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """Create a tiny throwaway git repo so tag operations don't touch real state."""
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    # Need at least one commit before tagging.
    (repo / "README").write_text("test\n")
    subprocess.run(["git", "add", "README"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init", "--quiet"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return repo


# ---- gate-check (--check) --------------------------------------------------


def test_check_passes_when_marker_present(tmp_path: Path) -> None:
    marker = tmp_path / "PASS"
    marker.write_text("")
    rc = cli.main(["--check", "--marker", str(marker)])
    assert rc == 0


def test_check_fails_when_marker_absent(tmp_path: Path) -> None:
    marker = tmp_path / "PASS"  # not created
    rc = cli.main(["--check", "--marker", str(marker)])
    assert rc == 1


def test_check_fails_when_marker_is_directory(tmp_path: Path) -> None:
    """Edge case: someone creates a dir at the marker path."""
    marker = tmp_path / "PASS"
    marker.mkdir()
    rc = cli.main(["--check", "--marker", str(marker)])
    assert rc == 1


def test_check_does_not_create_tag(fake_repo: Path, tmp_path: Path) -> None:
    """--check is side-effect-free even when the gate passes."""
    marker = tmp_path / "PASS"
    marker.write_text("")
    cli.main(["--check", "--marker", str(marker), "--repo-root", str(fake_repo)])
    # Tag must NOT have been created.
    assert not _tag_exists(fake_repo, "phase4-launch")


# ---- actual tag creation ---------------------------------------------------


def test_creates_tag_when_marker_present(fake_repo: Path, tmp_path: Path) -> None:
    marker = tmp_path / "PASS"
    marker.write_text("")
    rc = cli.main(["--marker", str(marker), "--repo-root", str(fake_repo)])
    assert rc == 0
    assert _tag_exists(fake_repo, "phase4-launch")


def test_refuses_to_create_tag_when_marker_absent(fake_repo: Path, tmp_path: Path) -> None:
    marker = tmp_path / "PASS"  # not created
    rc = cli.main(["--marker", str(marker), "--repo-root", str(fake_repo)])
    assert rc == 1
    assert not _tag_exists(fake_repo, "phase4-launch")


def test_idempotent_when_tag_already_exists(fake_repo: Path, tmp_path: Path) -> None:
    """Second invocation should return 0 without erroring."""
    marker = tmp_path / "PASS"
    marker.write_text("")
    # First invocation creates the tag.
    rc1 = cli.main(["--marker", str(marker), "--repo-root", str(fake_repo)])
    assert rc1 == 0
    # Second invocation finds the tag and is a no-op.
    rc2 = cli.main(["--marker", str(marker), "--repo-root", str(fake_repo)])
    assert rc2 == 0


# ---- --force override ------------------------------------------------------


def test_force_bypasses_marker_check(fake_repo: Path, tmp_path: Path) -> None:
    """--force creates the tag even when the marker is absent."""
    marker = tmp_path / "PASS"  # not created
    rc = cli.main(["--force", "--marker", str(marker), "--repo-root", str(fake_repo)])
    assert rc == 0
    assert _tag_exists(fake_repo, "phase4-launch")


def test_force_with_check_still_reports_forced(tmp_path: Path) -> None:
    """--check + --force returns 0 (forced) without side effects."""
    marker = tmp_path / "PASS"  # not created
    rc = cli.main(["--check", "--force", "--marker", str(marker)])
    assert rc == 0


# ---- helpers ---------------------------------------------------------------


def _tag_exists(repo: Path, tag: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "tag", "--list", tag],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and tag in result.stdout.split()
