"""Tests for SU2 output parsers (fanopt.cfd.parsers)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fanopt.cfd.j_fan import SteadyRun, plane_flux_from_velocity
from fanopt.cfd.parsers import (
    parse_su2_history_thrust,
    parse_su2_plane_flow_csv,
    parse_su2_unsteady_force_series,
    plane_flux_series_from_csvs,
    steady_run_from_history,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# --- parse_su2_history_thrust -------------------------------------------------


def test_history_thrust_reads_last_converged_row(tmp_path):
    p = _write(
        tmp_path / "history.csv",
        "Inner_Iter,CD,CFz\n0,0.1,0.5\n1,0.1,0.9\n2,0.1,1.234\n",
    )
    assert parse_su2_history_thrust(p) == pytest.approx(1.234)


def test_history_thrust_detects_alternate_column_name(tmp_path):
    p = _write(tmp_path / "history.csv", '"Iter","CZ"\n0,3.0\n1,4.5\n')
    assert parse_su2_history_thrust(p) == pytest.approx(4.5)


def test_history_thrust_missing_z_force_column(tmp_path):
    p = _write(tmp_path / "history.csv", "Iter,CD,CL\n0,0.1,0.2\n")
    with pytest.raises(ValueError, match="no recognized"):
        parse_su2_history_thrust(p)


def test_history_thrust_empty_file(tmp_path):
    p = _write(tmp_path / "history.csv", "")
    with pytest.raises(ValueError, match="empty CSV"):
        parse_su2_history_thrust(p)


def test_history_thrust_header_only(tmp_path):
    p = _write(tmp_path / "history.csv", "Iter,CFz\n")
    with pytest.raises(ValueError, match="no data rows"):
        parse_su2_history_thrust(p)


def test_history_thrust_malformed_final_row(tmp_path):
    p = _write(tmp_path / "history.csv", "Iter,CFz\n0,1.0\n1,not-a-number\n")
    with pytest.raises(ValueError, match="malformed final row"):
        parse_su2_history_thrust(p)


# --- steady_run_from_history --------------------------------------------------


def test_steady_run_from_history(tmp_path):
    p = _write(tmp_path / "history.csv", "Iter,CFz\n0,2.5\n")
    run = steady_run_from_history(p, stroke="productive", design_hash="abc")
    assert isinstance(run, SteadyRun)
    assert run.thrust == pytest.approx(2.5)
    assert run.stroke == "productive"
    assert run.design_hash == "abc"


# --- parse_su2_plane_flow_csv -------------------------------------------------


def _plane_csv(tmp_path: Path, name: str = "plane.csv") -> Path:
    return _write(
        tmp_path / name,
        "x,y,z,Velocity_x,Velocity_y,Velocity_z,Area\n"
        "0,0,0.3,0,0,2.0,1.0\n"
        "0.1,0,0.3,0,0,2.0,1.0\n",
    )


def test_plane_flow_parses_velocity_and_area(tmp_path):
    velocity, area = parse_su2_plane_flow_csv(_plane_csv(tmp_path))
    assert velocity.shape == (2, 3)
    assert area.tolist() == [1.0, 1.0]
    assert velocity[0].tolist() == [0.0, 0.0, 2.0]


def test_plane_flow_feeds_flux(tmp_path):
    velocity, area = parse_su2_plane_flow_csv(_plane_csv(tmp_path))
    # ρ · Σ w² dA = 1.225 · (4 + 4)
    assert plane_flux_from_velocity(velocity, area) == pytest.approx(1.225 * 8.0)


def test_plane_flow_missing_velocity_column(tmp_path):
    p = _write(tmp_path / "plane.csv", "x,y,z,Area\n0,0,0.3,1.0\n")
    with pytest.raises(ValueError, match="missing velocity column"):
        parse_su2_plane_flow_csv(p)


def test_plane_flow_missing_area_column(tmp_path):
    p = _write(
        tmp_path / "plane.csv",
        "x,y,z,Velocity_x,Velocity_y,Velocity_z\n0,0,0.3,0,0,2\n",
    )
    with pytest.raises(ValueError, match="no recognized area"):
        parse_su2_plane_flow_csv(p)


def test_plane_flow_header_only(tmp_path):
    p = _write(tmp_path / "plane.csv", "x,y,z,Velocity_x,Velocity_y,Velocity_z,Area\n")
    with pytest.raises(ValueError, match="no data rows"):
        parse_su2_plane_flow_csv(p)


def test_plane_flow_malformed_data(tmp_path):
    p = _write(
        tmp_path / "plane.csv",
        "x,y,z,Velocity_x,Velocity_y,Velocity_z,Area\n0,0,0.3,0,0,bad,1.0\n",
    )
    with pytest.raises(ValueError, match="malformed numeric data"):
        parse_su2_plane_flow_csv(p)


# --- plane_flux_series_from_csvs ----------------------------------------------


def test_plane_flux_series_from_csvs(tmp_path):
    p1 = _plane_csv(tmp_path, "t0.csv")
    p2 = _plane_csv(tmp_path, "t1.csv")
    series = plane_flux_series_from_csvs([p1, p2])
    assert series.shape == (2,)
    assert series[0] == pytest.approx(1.225 * 8.0)
    assert np.allclose(series, 1.225 * 8.0)


def test_plane_flux_series_requires_paths():
    with pytest.raises(ValueError, match="at least one path"):
        plane_flux_series_from_csvs([])


# --- parse_su2_unsteady_force_series ------------------------------------------


def test_unsteady_series_keeps_last_inner_iter_per_step(tmp_path):
    # 2 outer steps, 2 inner iters each; last inner-iter value wins per step.
    h = _write(
        tmp_path / "u.csv",
        "Time_Iter,Inner_Iter,CFx\n0,0,1.0\n0,1,2.0\n1,0,3.0\n1,1,4.0\n",
    )
    series = parse_su2_unsteady_force_series(h)
    assert np.allclose(series, [2.0, 4.0])


def test_unsteady_series_length_is_number_of_outer_steps(tmp_path):
    rows = "\n".join(f"{t},0,{t * 1.5}" for t in range(5))
    h = _write(tmp_path / "u.csv", "Time_Iter,Inner_Iter,CFx\n" + rows + "\n")
    assert parse_su2_unsteady_force_series(h).size == 5


def test_unsteady_series_missing_force_column_raises(tmp_path):
    h = _write(tmp_path / "u.csv", "Time_Iter,Inner_Iter,CL\n0,0,1.0\n")
    with pytest.raises(ValueError, match="no recognized unsteady force column"):
        parse_su2_unsteady_force_series(h)


def test_unsteady_series_missing_time_iter_raises(tmp_path):
    h = _write(tmp_path / "u.csv", "Inner_Iter,CFx\n0,1.0\n")
    with pytest.raises(ValueError, match="no Time_Iter column"):
        parse_su2_unsteady_force_series(h)


def test_unsteady_series_no_data_rows_raises(tmp_path):
    h = _write(tmp_path / "u.csv", "Time_Iter,Inner_Iter,CFx\n")
    with pytest.raises(ValueError, match="no data rows"):
        parse_su2_unsteady_force_series(h)
