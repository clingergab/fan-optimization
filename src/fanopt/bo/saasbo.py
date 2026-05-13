"""SAASBO fallback (≤500 inducing points).

Scaffolded per docs/plan_R11.md §12.1. Implementation lands in Phase 0 / 1
per the section referenced below. Until then, this module is intentionally
empty: importing it succeeds (keeps test_audit gates green) but invoking any
helper will raise NotImplementedError via the package-level guard.

Reference: docs/plan_R11.md §6.2.3
"""
from __future__ import annotations

__all__: list[str] = []
