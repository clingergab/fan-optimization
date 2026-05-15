"""CLI tests for scripts/launch_phase4.py.

Validates the Phase 4 launch gate behavior: dual-marker (0.6c AND 0.6d, post-
2026-05-14), --check side-effect-freeness, --force override, tag idempotence,
and exit codes.

Spec reference: docs/report-final.md §Phase 0 Spike 0.6c (H10 lock) +
§Phase 0 Spike 0.6d (2026-05-14 H10 supplement).
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


def _both_markers(
    tmp_path: Path, *, create_06c: bool = True, create_06d: bool = True
) -> tuple[Path, Path]:
    """Return (marker_06c, marker_06d) paths under tmp_path; create either or both."""
    marker_06c = tmp_path / "spike_0_6c_PASS"
    marker_06d = tmp_path / "spike_0_6d_PASS"
    if create_06c:
        marker_06c.write_text("")
    if create_06d:
        marker_06d.write_text("")
    return marker_06c, marker_06d


# ---- gate-check (--check) --------------------------------------------------


def test_check_passes_when_both_markers_present(tmp_path: Path) -> None:
    """Post-2026-05-14 dual-gate: BOTH markers required."""
    marker_06c, marker_06d = _both_markers(tmp_path)
    rc = cli.main(["--check", "--marker", str(marker_06c), "--marker-06d", str(marker_06d)])
    assert rc == 0


def test_check_fails_when_0_6c_marker_absent(tmp_path: Path) -> None:
    marker_06c, marker_06d = _both_markers(tmp_path, create_06c=False)
    rc = cli.main(["--check", "--marker", str(marker_06c), "--marker-06d", str(marker_06d)])
    assert rc == 1


def test_check_fails_when_0_6d_marker_absent(tmp_path: Path) -> None:
    """The new dual-gate adds the 0.6d requirement (2026-05-14)."""
    marker_06c, marker_06d = _both_markers(tmp_path, create_06d=False)
    rc = cli.main(["--check", "--marker", str(marker_06c), "--marker-06d", str(marker_06d)])
    assert rc == 1


def test_check_fails_when_0_6c_marker_is_directory(tmp_path: Path) -> None:
    """Edge case: someone creates a dir at the 0.6c marker path."""
    marker_06c = tmp_path / "spike_0_6c_PASS"
    marker_06c.mkdir()
    marker_06d = tmp_path / "spike_0_6d_PASS"
    marker_06d.write_text("")
    rc = cli.main(["--check", "--marker", str(marker_06c), "--marker-06d", str(marker_06d)])
    assert rc == 1


def test_check_fails_when_0_6d_marker_is_directory(tmp_path: Path) -> None:
    """Symmetric edge case for the 0.6d marker."""
    marker_06c = tmp_path / "spike_0_6c_PASS"
    marker_06c.write_text("")
    marker_06d = tmp_path / "spike_0_6d_PASS"
    marker_06d.mkdir()
    rc = cli.main(["--check", "--marker", str(marker_06c), "--marker-06d", str(marker_06d)])
    assert rc == 1


def test_check_does_not_create_tag(fake_repo: Path, tmp_path: Path) -> None:
    """--check is side-effect-free even when the gate passes."""
    marker_06c, marker_06d = _both_markers(tmp_path)
    cli.main(
        [
            "--check",
            "--marker",
            str(marker_06c),
            "--marker-06d",
            str(marker_06d),
            "--repo-root",
            str(fake_repo),
        ]
    )
    assert not _tag_exists(fake_repo, "phase4-launch")


# ---- actual tag creation ---------------------------------------------------


def test_creates_tag_when_both_markers_present(fake_repo: Path, tmp_path: Path) -> None:
    marker_06c, marker_06d = _both_markers(tmp_path)
    rc = cli.main(
        [
            "--marker",
            str(marker_06c),
            "--marker-06d",
            str(marker_06d),
            "--repo-root",
            str(fake_repo),
        ]
    )
    assert rc == 0
    assert _tag_exists(fake_repo, "phase4-launch")


def test_refuses_to_create_tag_when_0_6c_absent(fake_repo: Path, tmp_path: Path) -> None:
    marker_06c, marker_06d = _both_markers(tmp_path, create_06c=False)
    rc = cli.main(
        [
            "--marker",
            str(marker_06c),
            "--marker-06d",
            str(marker_06d),
            "--repo-root",
            str(fake_repo),
        ]
    )
    assert rc == 1
    assert not _tag_exists(fake_repo, "phase4-launch")


def test_refuses_to_create_tag_when_0_6d_absent(fake_repo: Path, tmp_path: Path) -> None:
    """The 0.6d gate, added 2026-05-14, must be enforced by the tag-creation path too."""
    marker_06c, marker_06d = _both_markers(tmp_path, create_06d=False)
    rc = cli.main(
        [
            "--marker",
            str(marker_06c),
            "--marker-06d",
            str(marker_06d),
            "--repo-root",
            str(fake_repo),
        ]
    )
    assert rc == 1
    assert not _tag_exists(fake_repo, "phase4-launch")


def test_idempotent_when_tag_already_exists(fake_repo: Path, tmp_path: Path) -> None:
    """Second invocation should return 0 without erroring."""
    marker_06c, marker_06d = _both_markers(tmp_path)
    rc1 = cli.main(
        [
            "--marker",
            str(marker_06c),
            "--marker-06d",
            str(marker_06d),
            "--repo-root",
            str(fake_repo),
        ]
    )
    assert rc1 == 0
    rc2 = cli.main(
        [
            "--marker",
            str(marker_06c),
            "--marker-06d",
            str(marker_06d),
            "--repo-root",
            str(fake_repo),
        ]
    )
    assert rc2 == 0


# ---- --force override ------------------------------------------------------


def test_force_bypasses_both_markers(fake_repo: Path, tmp_path: Path) -> None:
    """--force creates the tag even when BOTH markers are absent."""
    marker_06c, marker_06d = _both_markers(tmp_path, create_06c=False, create_06d=False)
    rc = cli.main(
        [
            "--force",
            "--marker",
            str(marker_06c),
            "--marker-06d",
            str(marker_06d),
            "--repo-root",
            str(fake_repo),
        ]
    )
    assert rc == 0
    assert _tag_exists(fake_repo, "phase4-launch")


def test_force_with_check_still_reports_forced(tmp_path: Path) -> None:
    """--check + --force returns 0 (forced) without side effects, even without markers."""
    marker_06c, marker_06d = _both_markers(tmp_path, create_06c=False, create_06d=False)
    rc = cli.main(
        [
            "--check",
            "--force",
            "--marker",
            str(marker_06c),
            "--marker-06d",
            str(marker_06d),
        ]
    )
    assert rc == 0


# ---- error messages disambiguate which marker is missing ------------------


def test_error_message_names_0_6c_when_only_0_6c_absent(tmp_path: Path, capsys) -> None:
    """When only 0.6c is missing, the operator-facing message must point at 0.6c."""
    marker_06c, marker_06d = _both_markers(tmp_path, create_06c=False)
    cli.main(["--check", "--marker", str(marker_06c), "--marker-06d", str(marker_06d)])
    out = capsys.readouterr().out
    assert "0.6c" in out


def test_error_message_names_0_6d_when_only_0_6d_absent(tmp_path: Path, capsys) -> None:
    """When only 0.6d is missing, the operator-facing message must point at 0.6d."""
    marker_06c, marker_06d = _both_markers(tmp_path, create_06d=False)
    cli.main(["--check", "--marker", str(marker_06c), "--marker-06d", str(marker_06d)])
    out = capsys.readouterr().out
    assert "0.6d" in out


# ---- helpers ---------------------------------------------------------------


def _tag_exists(repo: Path, tag: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "tag", "--list", tag],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and tag in result.stdout.split()
