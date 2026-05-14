"""CLI tests for scripts/run_spike_0_6a.py.

Validates the M3-SU2-viability gate behavior:
- Without SU2 on PATH and without --dry-run → fail loud (exit 2).
- With --dry-run → exit 0, write placeholder finite J_fan.
- With SU2 on PATH but no real pipeline yet → exit 3 (Phase 1 dep gate).

The "fail loud" behavior is a deliberate fix to the silent-pass stub the
runner shipped with in earlier rounds.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import run_spike_0_6a as cli


@pytest.fixture
def out_csv(tmp_path: Path) -> Path:
    return tmp_path / "06a.csv"


# ---- dry-run path ---------------------------------------------------------


def test_dry_run_writes_placeholder_and_exits_zero(out_csv: Path) -> None:
    rc = cli.main(["--dry-run", "--out", str(out_csv)])
    assert rc == 0
    assert out_csv.exists()
    rows = out_csv.read_text().splitlines()
    # header + one data row
    assert len(rows) >= 2


def test_dry_run_row_has_finite_j_fan_value(out_csv: Path) -> None:
    cli.main(["--dry-run", "--out", str(out_csv)])
    body = out_csv.read_text()
    # Placeholder value 0.0 is the dry-run stand-in for finite J_fan.
    assert "0.000000e+00" in body or "0.0" in body


# ---- fail-loud paths ------------------------------------------------------


def test_fails_loud_when_su2_absent_and_not_dry_run(out_csv: Path) -> None:
    """Exit 2 (input error) when SU2 isn't installed and the operator
    didn't ask for --dry-run."""
    with patch.object(cli, "_su2_available", return_value=False):
        rc = cli.main(["--out", str(out_csv)])
    assert rc == 2
    # No CSV row written — the aggregator must see the slot as NOT RUN.
    assert not out_csv.exists()


def test_fails_loud_when_su2_present_but_pipeline_unwired(out_csv: Path) -> None:
    """Exit 3 — SU2 is available but the production CadQuery→Gmsh→SU2→j_fan
    pipeline isn't landed yet. Don't write a misleading placeholder row."""
    with patch.object(cli, "_su2_available", return_value=True):
        rc = cli.main(["--out", str(out_csv)])
    assert rc == 3
    assert not out_csv.exists()


# ---- defaults -------------------------------------------------------------


def test_default_output_path_resolves(out_csv: Path) -> None:
    """Argparse default should resolve to a path under data/spike_0_6/."""
    assert str(cli.DEFAULT_OUTPUT).endswith("data/spike_0_6/06a.csv")


def test_help_does_not_crash() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
