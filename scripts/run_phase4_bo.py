#!/usr/bin/env python
"""Phase 4 — full multi-fidelity BO loop — scaffolded per docs/plan_R11.md §Phase 4.

CLI entry point; real logic lands when the corresponding phase is implemented.
Run with `python scripts/run_phase4_bo.py --help` once flags land.
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    print("Phase 0 scaffold: run_phase4_bo.py is not yet implemented.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
