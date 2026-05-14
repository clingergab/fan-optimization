"""Tests for scripts/parse_su2_history_to_cycles.py.

Validates SU2-history column detection, per-cycle aggregation, hysteresis
area calculation, and the end-to-end CLI on synthetic SU2 history files.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

import parse_su2_history_to_cycles as cli


# ---- column detection ----------------------------------------------------


def test_detect_column_exact_match() -> None:
    assert cli._detect_column(["Time_Iter", "CL", "CD"], ("CL",)) == "CL"


def test_detect_column_case_insensitive() -> None:
    assert cli._detect_column(["Time_Iter", "cl", "cd"], ("CL",)) == "cl"


def test_detect_column_first_match_wins() -> None:
    """If both CL and C_L appear, the first candidate that matches wins."""
    assert cli._detect_column(["CL", "C_L"], ("CL", "C_L")) == "CL"
    assert cli._detect_column(["C_L", "CL"], ("CL", "C_L")) == "CL"  # CL matches via second iter


def test_detect_column_no_match() -> None:
    assert cli._detect_column(["Iter", "Drag"], ("CL", "CLift")) is None


# ---- per-outer-iter collapse ---------------------------------------------


def test_per_outer_iter_last_inner_wins() -> None:
    """When SU2 emits multiple inner-iter rows per outer step, keep last."""
    rows = [
        {"time_iter": 0, "cl": 0.1, "cd": 0.01},
        {"time_iter": 0, "cl": 0.2, "cd": 0.02},  # last inner @ outer 0
        {"time_iter": 1, "cl": 0.3, "cd": 0.03},
    ]
    out = cli._per_outer_iter(rows)
    assert len(out) == 2
    assert out[0]["cl"] == 0.2
    assert out[1]["cl"] == 0.3


def test_per_outer_iter_sorts_by_time_iter() -> None:
    rows = [
        {"time_iter": 2, "cl": 0.3, "cd": 0.03},
        {"time_iter": 0, "cl": 0.1, "cd": 0.01},
        {"time_iter": 1, "cl": 0.2, "cd": 0.02},
    ]
    out = cli._per_outer_iter(rows)
    assert [int(r["time_iter"]) for r in out] == [0, 1, 2]


# ---- cycle splitting -----------------------------------------------------


def test_split_cycles_even_division() -> None:
    rows = [{"time_iter": i, "cl": 0.0, "cd": 0.0} for i in range(1000)]
    cycles = cli._split_cycles(rows, n_cycles=5)
    assert len(cycles) == 5
    assert all(len(c) == 200 for c in cycles)


def test_split_cycles_rejects_too_few_rows() -> None:
    rows = [{"time_iter": i, "cl": 0.0, "cd": 0.0} for i in range(3)]
    with pytest.raises(ValueError, match="need ≥ 5"):
        cli._split_cycles(rows, n_cycles=5)


def test_split_cycles_warns_on_remainder(capsys) -> None:
    """Non-divisible row count emits a stderr warning but does not raise."""
    rows = [{"time_iter": i, "cl": 0.0, "cd": 0.0} for i in range(1003)]
    cycles = cli._split_cycles(rows, n_cycles=5)
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert len(cycles) == 5
    assert all(len(c) == 200 for c in cycles)


def test_split_cycles_rejects_nonpositive_n() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        cli._split_cycles([{"time_iter": 0, "cl": 0.0, "cd": 0.0}], n_cycles=0)


# ---- hysteresis area -----------------------------------------------------


def test_hysteresis_area_zero_for_collinear_points() -> None:
    """A line has zero enclosed area (shoelace on collinear → 0)."""
    alphas = [0.0, 0.1, 0.2, 0.3]
    cls = [0.0, 0.5, 1.0, 1.5]  # straight line
    area = cli._hysteresis_area_shoelace(alphas, cls)
    assert area == pytest.approx(0.0, abs=1e-12)


def test_hysteresis_area_unit_square() -> None:
    """Unit square traced counterclockwise: |area| = 1."""
    alphas = [0.0, 1.0, 1.0, 0.0]
    cls = [0.0, 0.0, 1.0, 1.0]
    area = cli._hysteresis_area_shoelace(alphas, cls)
    assert area == pytest.approx(1.0, abs=1e-12)


def test_hysteresis_area_handles_short_traces() -> None:
    """< 3 points → 0 area (degenerate)."""
    assert cli._hysteresis_area_shoelace([0.0], [0.0]) == 0.0
    assert cli._hysteresis_area_shoelace([0.0, 1.0], [0.0, 1.0]) == 0.0


def test_hysteresis_area_unit_circle() -> None:
    """Discretized unit circle → enclosed area ≈ π."""
    n = 128
    alphas = [math.cos(2 * math.pi * i / n) for i in range(n)]
    cls = [math.sin(2 * math.pi * i / n) for i in range(n)]
    area = cli._hysteresis_area_shoelace(alphas, cls)
    assert area == pytest.approx(math.pi, rel=1e-2)


# ---- cycles_from_rows ----------------------------------------------------


def _sinusoidal_history(n_outer: int, n_cycles: int) -> list[dict[str, float]]:
    """Synthesize a SU2-like history: CL(t) = sin(2π · cycle_index), CD = 0.05."""
    rows = []
    for i in range(n_outer):
        # Phase advances 2π per cycle.
        phase = 2.0 * math.pi * (i / (n_outer // n_cycles))
        rows.append(
            {
                "time_iter": float(i),
                "cl": math.sin(phase),
                "cd": 0.05,
                "time": float(i) * 1.0e-3,
            }
        )
    return rows


def test_cycles_from_rows_produces_n_cycles() -> None:
    rows = _sinusoidal_history(n_outer=1000, n_cycles=5)
    cycles = cli.cycles_from_rows(
        rows, n_cycles=5, theta_max_rad=0.1745, omega_shm_rad_per_s=2.0
    )
    assert len(cycles) == 5


def test_cycles_from_rows_cl_max_and_min() -> None:
    rows = _sinusoidal_history(n_outer=1000, n_cycles=5)
    cycles = cli.cycles_from_rows(
        rows, n_cycles=5, theta_max_rad=0.1745, omega_shm_rad_per_s=2.0
    )
    # Each cycle covers one full sine period → max ≈ 1, min ≈ -1.
    for c in cycles:
        assert c["c_l_max"] == pytest.approx(1.0, abs=0.02)
        assert c["c_l_min"] == pytest.approx(-1.0, abs=0.02)


def test_cycles_from_rows_cd_mean_constant() -> None:
    rows = _sinusoidal_history(n_outer=1000, n_cycles=5)
    cycles = cli.cycles_from_rows(
        rows, n_cycles=5, theta_max_rad=0.1745, omega_shm_rad_per_s=2.0
    )
    for c in cycles:
        assert c["c_d_mean"] == pytest.approx(0.05, abs=1e-9)


def test_cycles_from_rows_explicit_alpha_column_used() -> None:
    """When alpha is in the history rows, it's used directly (no reconstruction)."""
    rows = []
    for i in range(1000):
        rows.append(
            {
                "time_iter": float(i),
                "cl": 0.1 * i % 1.0,
                "cd": 0.05,
                "time": float(i) * 1e-3,
                "alpha": 0.1745 * math.sin(2 * math.pi * i / 200),
            }
        )
    cycles = cli.cycles_from_rows(
        rows, n_cycles=5, theta_max_rad=0.1745, omega_shm_rad_per_s=2.0
    )
    assert len(cycles) == 5
    # Hysteresis area should be finite (we have real alpha + real CL).
    assert all(c["c_l_hysteresis_area"] >= 0.0 for c in cycles)


def test_cycles_from_rows_no_alpha_no_time_uses_index() -> None:
    """No alpha and no time → fall back to row-index x-axis."""
    rows = [{"time_iter": float(i), "cl": math.sin(0.1 * i), "cd": 0.05} for i in range(1000)]
    cycles = cli.cycles_from_rows(
        rows, n_cycles=5, theta_max_rad=0.1745, omega_shm_rad_per_s=2.0
    )
    # Doesn't raise; hysteresis area is computed against the index proxy.
    assert len(cycles) == 5


# ---- read_history / CLI ---------------------------------------------------


def _write_history(path: Path, header: list[str], rows: list[list[float]]) -> Path:
    path.write_text(
        ",".join(header)
        + "\n"
        + "\n".join(",".join(f"{v}" for v in r) for r in rows)
        + "\n"
    )
    return path


def test_read_history_basic(tmp_path: Path) -> None:
    p = _write_history(
        tmp_path / "history.csv",
        ["Time_Iter", "Cur_Time", "CL", "CD"],
        [[i, i * 0.001, math.sin(i * 0.1), 0.05] for i in range(10)],
    )
    rows, colmap = cli._read_history(p)
    assert len(rows) == 10
    assert colmap["cl"] == "CL"
    assert colmap["cd"] == "CD"


def test_read_history_with_aoa_column(tmp_path: Path) -> None:
    p = _write_history(
        tmp_path / "history.csv",
        ["Time_Iter", "AoA", "CL", "CD"],
        [[i, 0.1 * math.sin(i), math.sin(i), 0.05] for i in range(10)],
    )
    rows, colmap = cli._read_history(p)
    assert colmap["alpha"] == "AoA"
    assert all("alpha" in r for r in rows)


def test_read_history_errors_on_missing_cl(tmp_path: Path) -> None:
    p = _write_history(
        tmp_path / "history.csv",
        ["Time_Iter", "Drag"],
        [[i, 0.05] for i in range(3)],
    )
    with pytest.raises(ValueError, match="lift/drag columns"):
        cli._read_history(p)


def test_read_history_errors_on_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(ValueError, match="empty"):
        cli._read_history(p)


def test_read_history_errors_on_missing_time_iter(tmp_path: Path) -> None:
    p = _write_history(
        tmp_path / "history.csv",
        ["Step", "CL", "CD"],  # no time-iter-like column
        [[i, 0.0, 0.0] for i in range(3)],
    )
    with pytest.raises(ValueError, match="time-iter"):
        cli._read_history(p)


def test_cli_end_to_end(tmp_path: Path) -> None:
    """Synthesize a 5-cycle history, parse it, verify the measured.csv."""
    history = _write_history(
        tmp_path / "history.csv",
        ["Time_Iter", "Cur_Time", "CL", "CD"],
        [
            [
                i,
                i * 1e-3,
                math.sin(2 * math.pi * i / 200),
                0.05,
            ]
            for i in range(1000)
        ],
    )
    out = tmp_path / "measured.csv"
    rc = cli.main(
        [
            "--history", str(history),
            "--n-cycles", "5",
            "--omega-shm-rad-per-s", "31.4",  # arbitrary positive
            "--out", str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    # measured.csv should have header + 5 cycle rows.
    text = out.read_text().splitlines()
    assert text[0] == "cycle_index,c_l_max,c_l_min,c_d_mean,c_l_hysteresis_area"
    assert len(text) == 6


def test_cli_missing_history_exits_2(tmp_path: Path) -> None:
    rc = cli.main(
        [
            "--history", str(tmp_path / "nope.csv"),
            "--n-cycles", "5",
            "--omega-shm-rad-per-s", "31.4",
            "--out", str(tmp_path / "measured.csv"),
        ]
    )
    assert rc == 2


def test_cli_malformed_history_exits_2(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("Time_Iter,CL,CD\n0,abc,0.05\n")
    rc = cli.main(
        [
            "--history", str(p),
            "--n-cycles", "5",
            "--omega-shm-rad-per-s", "31.4",
            "--out", str(tmp_path / "measured.csv"),
        ]
    )
    assert rc == 2


def test_omega_sign_does_not_matter(tmp_path: Path) -> None:
    """C11 PITCHING_OMEGA is locked negative for the production fan cfg, but
    the benchmark cfg may use either sign — `abs(omega)` is applied so a
    negative value doesn't blow up the alpha reconstruction."""
    history = _write_history(
        tmp_path / "history.csv",
        ["Time_Iter", "Cur_Time", "CL", "CD"],
        [[i, i * 1e-3, math.sin(i), 0.05] for i in range(1000)],
    )
    rc = cli.main(
        [
            "--history", str(history),
            "--n-cycles", "5",
            "--omega-shm-rad-per-s", "-31.4",  # negative
            "--out", str(tmp_path / "measured.csv"),
        ]
    )
    assert rc == 0
