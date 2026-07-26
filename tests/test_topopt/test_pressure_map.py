"""Tests for the CFD-pressure → FEA-skin mapping."""

from __future__ import annotations

import numpy as np
import pytest

from fanopt.topopt.pressure_map import load_surface_pressure, map_pressure_to_facets


def _write_csv(path, header, rows):
    lines = [header] + [",".join(str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_surface_pressure_parses_points_and_pressure(tmp_path):
    csv_path = tmp_path / "surface_flow.csv"
    _write_csv(csv_path, '"x","y","z","Pressure"', [(0.0, 1.0, 2.0, 100.0), (3.0, 4.0, 5.0, 200.0)])
    pts, press = load_surface_pressure(csv_path)
    assert pts.shape == (2, 3)
    assert press.tolist() == [100.0, 200.0]


def test_load_surface_pressure_case_insensitive_headers(tmp_path):
    csv_path = tmp_path / "s.csv"
    _write_csv(csv_path, "X, Y, Z, PRESSURE", [(0, 0, 0, 42.0)])
    _, press = load_surface_pressure(csv_path)
    assert press[0] == pytest.approx(42.0)


def test_load_surface_pressure_missing_column_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    _write_csv(csv_path, '"x","y","z"', [(0, 0, 0)])
    with pytest.raises(ValueError):
        load_surface_pressure(csv_path)


def test_load_empty_csv_raises(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_surface_pressure(csv_path)


def test_map_nearest_picks_closest_sample():
    surf = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    press = np.array([10.0, 20.0])
    facets = np.array([[0.1, 0, 0], [0.9, 0, 0]])
    assert map_pressure_to_facets(surf, press, facets).tolist() == [10.0, 20.0]


def test_map_reproduces_linear_field_at_sample_points():
    surf = np.array([[float(x), 0, 0] for x in range(6)])
    press = surf[:, 0] * 3.0  # p = 3x
    got = map_pressure_to_facets(surf, press, surf)
    assert np.allclose(got, press)


def test_map_idw_averages_between_two_equidistant_samples():
    surf = np.array([[0.0, 0, 0], [2.0, 0, 0]])
    press = np.array([0.0, 100.0])
    mid = np.array([[1.0, 0, 0]])
    assert map_pressure_to_facets(surf, press, mid, k=2)[0] == pytest.approx(50.0)
