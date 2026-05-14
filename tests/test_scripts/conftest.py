"""Shared fixtures + path setup for scripts/ tests.

`scripts/` is not a Python package — it's a flat directory of CLI entry
points imported via importlib. This conftest puts `scripts/` on sys.path so
tests can `import spike_0_2_analyze` directly and call `main(...)` with a
fake argv.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
