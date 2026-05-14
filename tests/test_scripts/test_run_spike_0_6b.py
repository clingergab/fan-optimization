"""CLI tests for scripts/run_spike_0_6b.py.

Validates the M3-FEA-viability gate behavior:
- Without FEniCSx and without --dry-run → fail loud (exit 2).
- With --dry-run → exit 0, write the analytic placeholder.
- Real (non-dry-run) solver path raises NotImplementedError until the
  real dolfinx-based cantilever code lands.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import run_spike_0_6b as cli


@pytest.fixture
def out_csv(tmp_path: Path) -> Path:
    return tmp_path / "06b.csv"


# ---- dry-run path ---------------------------------------------------------


def test_dry_run_writes_analytic_placeholder(out_csv: Path) -> None:
    rc = cli.main(["--dry-run", "--out", str(out_csv)])
    assert rc == 0
    assert out_csv.exists()


def test_dry_run_does_not_invoke_solver(out_csv: Path) -> None:
    """Real solver is NotImplementedError; dry-run must NOT hit it."""
    rc = cli.main(["--dry-run", "--out", str(out_csv)])
    assert rc == 0


# ---- fail-loud paths ------------------------------------------------------


def test_fails_loud_when_fenicsx_absent_and_not_dry_run(out_csv: Path) -> None:
    with patch.object(cli, "_fenicsx_available", return_value=False):
        rc = cli.main(["--out", str(out_csv)])
    assert rc == 2
    assert not out_csv.exists()


def test_real_solver_raises_not_implemented(out_csv: Path) -> None:
    """Non-dry-run with FEniCSx present must hit the stub solver and raise.

    This protects against future code re-introducing the silent-pass stub
    (which would silently return the analytic value and trivially pass
    the 5% gate without exercising any FEA).
    """
    with patch.object(cli, "_fenicsx_available", return_value=True):
        with pytest.raises(NotImplementedError, match="real solver"):
            cli.main(["--out", str(out_csv)])


# ---- analytic-only helpers ------------------------------------------------


def test_analytic_tip_deflection_positive_for_positive_load() -> None:
    """Sanity: pure-numerics helper returns a sensible number."""
    delta = cli._analytic_tip_deflection_m(
        P=5.0,
        L=0.200,
        E=1.30e9,
        I=cli._i_rect_m4(cli.B_M, cli.H_M),
    )
    assert delta > 0


def test_i_rect_m4_formula() -> None:
    """I = b·h^3 / 12."""
    out = cli._i_rect_m4(0.01, 0.002)
    expected = 0.01 * (0.002**3) / 12.0
    assert out == pytest.approx(expected, rel=1e-12)
