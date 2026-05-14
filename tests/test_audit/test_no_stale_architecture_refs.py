"""Retired-phrase CI gate (Round-8 v2 meta lock; gate scaffold Round-9 HIGH-5 lock).

Reads docs/retired_phrases.yaml; for each entry, greps **production code under
src/fanopt/** for matches and asserts zero matches outside the entry's
allow-lists. This catches accidental re-introduction of retired architectural
phrasings in code — constants, comments, docstrings, helper text — where
drift typically happens.

**Scope:** production code only. The plan (`docs/report-final.md`) is
deliberately NOT scanned by this gate: it is author-managed and intentionally
contains historical references, CI-gate self-descriptions, and meta-discussion
of retired phrases that would generate noise rather than signal. A separate
manual review pass (or a future opt-in tool) is the appropriate venue for
plan-prose drift checks.

Allow-list logic:
  - allow_list_sections: skip if the matching line's nearest preceding
    markdown heading contains any of the allowed substrings (case-insensitive).
    Mostly inert under the code-only scope, kept so the catalog schema stays
    portable to any future plan-scanning tool.
  - allow_list_disclaimers: skip if the matching line itself contains any
    of the allowed disclaimer phrases (case-insensitive)

A match is a VIOLATION only if it passes neither allow-list check.

Runs as part of:
  - Default pytest suite (CI + local), every invocation
  - Every adversarial review round's Pass B (prose-vs-locks audit), where
    the plan-prose pass is performed manually against the same catalog
  - The `phase4-launch` Git tag pre-flight check
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


CATALOG_PATH = REPO_ROOT / "docs" / "retired_phrases.yaml"


def _scan_paths() -> list[Path]:
    """Return the list of files to scan.

    Scope is production code only — every `*.py` under `src/fanopt/`. The
    plan (`docs/report-final.md`) is intentionally excluded; see this
    module's docstring for the rationale.
    """
    paths: list[Path] = []
    src_dir = REPO_ROOT / "src" / "fanopt"
    if src_dir.exists():
        paths.extend(src_dir.rglob("*.py"))
    return paths


def _section_of_line(line_num: int, lines: list[str]) -> str:
    """Return the nearest preceding markdown heading text (lower-cased), or ''.

    Walks backwards from `line_num` (1-indexed) looking for a line starting
    with one or more '#'. Returns the heading text without the leading '#'s
    and surrounding whitespace.
    """
    if line_num < 1 or line_num > len(lines):
        return ""
    for i in range(line_num - 1, -1, -1):
        if lines[i].startswith("#"):
            return lines[i].lstrip("#").strip().lower()
    return ""


def _in_allowed_section(section: str, allow_sections: list[str]) -> bool:
    """True iff `section` contains any of the allow-list section substrings."""
    if not section:
        return False
    section_lower = section.lower()
    return any(allowed.lower() in section_lower for allowed in allow_sections)


def _has_allowed_disclaimer(line: str, allow_disclaimers: list[str]) -> bool:
    """True iff `line` contains any of the allow-list disclaimer substrings."""
    line_lower = line.lower()
    return any(d.lower() in line_lower for d in allow_disclaimers)


def _scan_file(path: Path, catalog: list[dict]) -> list[dict]:
    """Scan one file against all catalog entries; return list of violations."""
    if not path.exists():
        return []
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    lines = content.splitlines()
    is_markdown = path.suffix == ".md"
    violations: list[dict] = []

    for entry in catalog:
        try:
            rgx = re.compile(entry["pattern"], re.IGNORECASE | re.MULTILINE)
        except re.error as e:
            pytest.fail(f"retired_phrases.yaml: invalid regex {entry['pattern']!r}: {e}")
        allow_sections = entry.get("allow_list_sections", []) or []
        allow_disclaimers = entry.get("allow_list_disclaimers", []) or []

        for line_num, line in enumerate(lines, start=1):
            if not rgx.search(line):
                continue
            section = _section_of_line(line_num, lines) if is_markdown else ""
            if is_markdown and _in_allowed_section(section, allow_sections):
                continue
            if _has_allowed_disclaimer(line, allow_disclaimers):
                continue
            violations.append(
                {
                    "file": str(path.relative_to(REPO_ROOT)),
                    "line": line_num,
                    "section": section[:80],
                    "text": line.strip()[:200],
                    "pattern": entry["pattern"],
                    "retired_by": entry["retired_by"],
                }
            )

    return violations


def test_no_stale_architecture_refs() -> None:
    """Assert zero retired-phrase violations across spec + production code.

    On failure, the error message lists every violation with file:line,
    surrounding section, line text, the offending regex, and the lock that
    retired the phrase. The implementer fixes by either:
      (a) editing the prose to use the current locked terminology, or
      (b) adding the legitimate-use phrasing to allow_list_disclaimers in
          docs/retired_phrases.yaml.
    """
    if not CATALOG_PATH.exists():
        pytest.skip(
            f"retired_phrases catalog not found at {CATALOG_PATH}; "
            "create it per HIGH-5 Round-9 lock"
        )

    raw = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        catalog = raw.get("retired_phrases", [])
    elif isinstance(raw, list):
        catalog = raw
    else:
        pytest.fail(
            f"retired_phrases.yaml: expected list or dict-with-retired_phrases-key, "
            f"got {type(raw).__name__}"
        )

    if not catalog:
        pytest.skip("retired_phrases catalog is empty")

    all_violations: list[dict] = []
    for path in _scan_paths():
        all_violations.extend(_scan_file(path, catalog))

    if all_violations:
        msg = f"\n{len(all_violations)} retired-phrase violation(s):\n"
        for v in all_violations:
            msg += (
                f"\n  {v['file']}:{v['line']}  (section: {v['section']!r})\n"
                f"    Retired by: {v['retired_by']}\n"
                f"    Pattern:    {v['pattern']}\n"
                f"    Text:       {v['text']}\n"
            )
        msg += (
            "\nFix: edit the prose to use the current locked terminology, OR add a "
            "legitimate-use disclaimer to docs/retired_phrases.yaml allow_list_disclaimers.\n"
        )
        raise AssertionError(msg)


if __name__ == "__main__":
    test_no_stale_architecture_refs()
    print("OK: no retired-phrase violations.")
