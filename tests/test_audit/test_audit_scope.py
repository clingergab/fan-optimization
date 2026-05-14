"""Regression guard: the retired-phrase gate scans code only, not the plan.

If someone re-adds `docs/report-final.md` to `_scan_paths()`, this test fails
loudly. The plan is author-managed; drift checks against it live outside the
default pytest gate (see test_no_stale_architecture_refs.py module docstring).
"""
from __future__ import annotations

from pathlib import Path

from .test_no_stale_architecture_refs import REPO_ROOT, _scan_paths


PLAN_PATHS = (
    REPO_ROOT / "docs" / "report-final.md",
    REPO_ROOT / "report-final.md",
)


def test_scan_paths_excludes_plan() -> None:
    """Plan markdown must not appear in the audit scan list."""
    scanned = {Path(p).resolve() for p in _scan_paths()}
    for plan in PLAN_PATHS:
        assert plan.resolve() not in scanned, (
            f"Audit scope drift: {plan} is back in _scan_paths(). The plan is "
            "author-managed; only `src/fanopt/**/*.py` should be scanned."
        )


def test_scan_paths_includes_production_code() -> None:
    """`src/fanopt/` Python must still be scanned (the gate's actual job)."""
    scanned = _scan_paths()
    assert scanned, "_scan_paths() returned empty — production code is unscanned."
    src_dir = (REPO_ROOT / "src" / "fanopt").resolve()
    assert any(Path(p).resolve().is_relative_to(src_dir) for p in scanned), (
        f"No paths under {src_dir} in _scan_paths() — the gate would never fire."
    )


def test_scan_paths_only_python_files() -> None:
    """No non-Python source files scanned (markdown, YAML, etc.)."""
    for p in _scan_paths():
        assert p.suffix == ".py", (
            f"Unexpected non-Python path in scan list: {p}. The gate is "
            "designed for code drift; markdown/YAML drift checks need a "
            "different tool."
        )
