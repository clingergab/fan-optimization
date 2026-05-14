"""Unit tests for fanopt.physical.anemometer.

Validates the L8 9-point grid plane integral against analytic uniform-flow
and linear-gradient cases. Verifies the CSV loader enforces the locked grid
positions.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.3, §Phase 6 step 78 (L8 lock).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from fanopt.physical.anemometer import (
    GRID_POINTS_M,
    PLANE_AREA_M2,
    RHO_AIR_KG_PER_M3,
    AnemometerGrid,
    analyze_anemometer_grid,
    j_fan_proxy_from_grid,
    load_anemometer_csv,
)


# ----- analytic plane integrals ---------------------------------------------


def test_uniform_flow_gives_rho_v_A() -> None:
    """A spatially uniform 1 m/s flow should integrate to ρ · 1 · A."""
    grid = AnemometerGrid(
        labels=tuple(f"p{i + 1}" for i in range(9)),
        xy_m=GRID_POINTS_M,
        v_mean_m_per_s=(1.0,) * 9,
        v_peak_m_per_s=None,
    )
    res = analyze_anemometer_grid(grid)
    expected = RHO_AIR_KG_PER_M3 * 1.0 * PLANE_AREA_M2
    assert res.J_fan_proxy_N == pytest.approx(expected, rel=1e-12)
    assert res.v_mean_grid_m_per_s == pytest.approx(1.0, rel=1e-12)
    assert res.v_mean_grid_std_m_per_s == pytest.approx(0.0, abs=1e-12)


def test_linear_gradient_grid_averages_to_center_value() -> None:
    """For a symmetric grid and a flow that's an odd function in x, the
    spatial mean across the 9 points equals the center-point value.
    """
    # v(x) = 0.5 + 0.4·(x / 0.2)  → at the 3 x-values {-0.2, 0, +0.2}
    # gives {0.1, 0.5, 0.9}. With 3 y-rows: spatial mean = 0.5.
    v_per_x = {-0.2: 0.1, 0.0: 0.5, 0.2: 0.9}
    v = tuple(v_per_x[round(x, 6)] for (x, _y) in GRID_POINTS_M)
    grid = AnemometerGrid(
        labels=tuple(f"p{i + 1}" for i in range(9)),
        xy_m=GRID_POINTS_M,
        v_mean_m_per_s=v,
        v_peak_m_per_s=None,
    )
    res = analyze_anemometer_grid(grid)
    assert res.v_mean_grid_m_per_s == pytest.approx(0.5, rel=1e-12)


def test_peak_proxy_separate_from_mean_proxy() -> None:
    grid = AnemometerGrid(
        labels=tuple(f"p{i + 1}" for i in range(9)),
        xy_m=GRID_POINTS_M,
        v_mean_m_per_s=(0.5,) * 9,
        v_peak_m_per_s=(1.5,) * 9,
    )
    res = analyze_anemometer_grid(grid)
    assert res.J_fan_proxy_N == pytest.approx(
        RHO_AIR_KG_PER_M3 * 0.5 * PLANE_AREA_M2, rel=1e-12
    )
    assert res.J_fan_proxy_peak_N is not None
    assert res.J_fan_proxy_peak_N == pytest.approx(
        RHO_AIR_KG_PER_M3 * 1.5 * PLANE_AREA_M2, rel=1e-12
    )
    # Diagnostic ratio (peak vs mean) preserved.
    assert res.J_fan_proxy_peak_N / res.J_fan_proxy_N == pytest.approx(3.0, rel=1e-12)


# ----- j_fan_proxy_from_grid scalar -----------------------------------------


def test_scalar_proxy_with_overrides() -> None:
    J = j_fan_proxy_from_grid(2.0, rho_air_kg_per_m3=1.2, A_plane_m2=0.5)
    assert J == pytest.approx(1.2 * 2.0 * 0.5, rel=1e-12)


def test_scalar_proxy_defaults_match_locked_constants() -> None:
    J = j_fan_proxy_from_grid(1.0)
    assert J == pytest.approx(RHO_AIR_KG_PER_M3 * 1.0 * PLANE_AREA_M2, rel=1e-12)


# ----- CSV loader -----------------------------------------------------------


def _write_csv(path: Path, rows: list[tuple]) -> None:
    header = "point,x_m,y_m,z_m,v_mean_m_per_s,v_peak_m_per_s,notes\n"
    body = "\n".join(",".join(str(c) for c in r) for r in rows)
    path.write_text(header + body + "\n")


def _ok_rows() -> list[tuple]:
    return [
        (f"p{i + 1}", x, y, 0.3, 0.5, 1.0, "")
        for i, (x, y) in enumerate(GRID_POINTS_M)
    ]


def test_load_anemometer_csv_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "grid.csv"
    _write_csv(path, _ok_rows())
    grid = load_anemometer_csv(path)
    assert grid.labels == tuple(f"p{i + 1}" for i in range(9))
    assert grid.xy_m == GRID_POINTS_M
    assert grid.v_mean_m_per_s == (0.5,) * 9
    assert grid.v_peak_m_per_s == (1.0,) * 9


def test_load_anemometer_csv_rejects_wrong_grid(tmp_path: Path) -> None:
    """If the recorded (x, y) doesn't match the locked grid, load fails."""
    rows = _ok_rows()
    # Shift p5 (center) by 50 mm in x — way past the 5 mm tolerance.
    rows[4] = ("p5", 0.05, 0.0, 0.3, 0.5, 1.0, "")
    path = tmp_path / "shifted.csv"
    _write_csv(path, rows)
    with pytest.raises(ValueError, match="locked 3×3 grid"):
        load_anemometer_csv(path)


def test_load_anemometer_csv_rejects_wrong_count(tmp_path: Path) -> None:
    rows = _ok_rows()[:8]  # only 8 rows
    path = tmp_path / "short.csv"
    _write_csv(path, rows)
    with pytest.raises(ValueError, match="9 grid points"):
        load_anemometer_csv(path)


def test_load_anemometer_csv_tolerates_missing_peak(tmp_path: Path) -> None:
    """v_peak_m_per_s column present but values blank → grid.v_peak_m_per_s is None."""
    path = tmp_path / "no_peak.csv"
    rows = [
        (f"p{i + 1}", x, y, 0.3, 0.5, "", "")
        for i, (x, y) in enumerate(GRID_POINTS_M)
    ]
    _write_csv(path, rows)
    grid = load_anemometer_csv(path)
    assert grid.v_peak_m_per_s is None
    # Analyzer also handles None peak gracefully.
    res = analyze_anemometer_grid(grid)
    assert res.J_fan_proxy_peak_N is None
    assert res.J_fan_proxy_N == pytest.approx(
        RHO_AIR_KG_PER_M3 * 0.5 * PLANE_AREA_M2, rel=1e-12
    )


def test_load_anemometer_csv_rejects_empty_file(tmp_path: Path) -> None:
    """No data rows → ValueError."""
    path = tmp_path / "empty.csv"
    path.write_text("# comments only\n\n")
    with pytest.raises(ValueError, match="no rows found"):
        load_anemometer_csv(path)


def test_load_anemometer_csv_rejects_missing_required_column(tmp_path: Path) -> None:
    """Missing one of {point, x_m, y_m, v_mean_m_per_s} → ValueError on the
    first row that lacks it (caught row-by-row)."""
    # Drop the v_mean_m_per_s column entirely.
    path = tmp_path / "no_vmean.csv"
    header = "point,x_m,y_m,z_m,v_peak_m_per_s,notes"
    rows = [
        f"p{i + 1},{x},{y},0.3,1.0,"
        for i, (x, y) in enumerate(GRID_POINTS_M)
    ]
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="v_mean_m_per_s"):
        load_anemometer_csv(path)


def test_load_anemometer_csv_skips_comments_and_blanks(tmp_path: Path) -> None:
    """Comment lines and blank lines inside the file are silently skipped."""
    path = tmp_path / "with_comments.csv"
    header = "point,x_m,y_m,z_m,v_mean_m_per_s,v_peak_m_per_s,notes"
    body_lines = [
        f"p{i + 1},{x},{y},0.3,0.5,1.0,"
        for i, (x, y) in enumerate(GRID_POINTS_M)
    ]
    # Interleave with comment and blank lines.
    full = "\n".join(
        ["# operator notes line 1", "", header, "# mid-file comment", body_lines[0]]
        + body_lines[1:]
    )
    path.write_text(full + "\n")
    grid = load_anemometer_csv(path)
    assert grid.labels[0] == "p1"
    assert len(grid.labels) == 9


def test_load_anemometer_csv_accepts_grid_in_any_order(tmp_path: Path) -> None:
    """The grid-position check is order-insensitive within tol."""
    # Reverse the row order; the load should still succeed.
    rows = list(reversed(_ok_rows()))
    path = tmp_path / "reversed.csv"
    _write_csv(path, rows)
    grid = load_anemometer_csv(path)
    # Labels are preserved in CSV order (reversed); positions match GRID_POINTS_M
    # rearranged.
    assert set(grid.labels) == {f"p{i + 1}" for i in range(9)}
    assert set(grid.xy_m) == set(GRID_POINTS_M)
