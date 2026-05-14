"""CLI tests for scripts/run_spike_0_5.py.

Exercises:
- happy-path PASS (3 blades, every metric CV < 5%, overall_passed = true).
- FAIL on J_fan CV ≥ 5% (one knob dominates the spread).
- input error → exit 2 (missing required column).

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.5; protocol in
docs/spike_0_5_protocol.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import run_spike_0_5 as cli


# ─────────────────────────────────────────────────────────────────────
# Canonical-input fixture
# ─────────────────────────────────────────────────────────────────────


_DIM_COLS = [f"d{i}_mm" for i in range(1, 11)]
_HEADER = (
    "blade_id,mass_g,"
    + ",".join(_DIM_COLS)
    + ",bend_deflection_mm,j_fan_proxy,notes"
)


def _row(
    blade_id: int,
    mass_g: float,
    dims: list[float],
    bend_mm: float,
    j_fan: float,
    notes: str = "",
) -> str:
    cells = [str(blade_id), f"{mass_g:.4f}"]
    cells.extend(f"{d:.4f}" for d in dims)
    cells.extend([f"{bend_mm:.4f}", f"{j_fan:.4f}", notes])
    return ",".join(cells)


def _write_measurements(
    path: Path,
    *,
    masses: list[float] | None = None,
    j_fans: list[float] | None = None,
    bends: list[float] | None = None,
    dims_per_blade: list[list[float]] | None = None,
) -> Path:
    """3-blade CSV with all metrics inside the 5% gate by default."""
    if masses is None:
        masses = [5.000, 5.010, 4.990]
    if j_fans is None:
        j_fans = [0.3500, 0.3510, 0.3490]
    if bends is None:
        bends = [1.200, 1.205, 1.195]
    if dims_per_blade is None:
        dims_per_blade = [[25.000] * 10, [25.010] * 10, [24.990] * 10]

    lines = [
        "# Spike 0.5 — single-blade fab-noise measurements (test fixture)",
        _HEADER,
    ]
    for i, (m, jf, bd, dims) in enumerate(
        zip(masses, j_fans, bends, dims_per_blade, strict=True), start=1
    ):
        lines.append(_row(i, m, dims, bd, jf, notes=""))
    path.write_text("\n".join(lines) + "\n")
    return path


# ─────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────


def test_cli_pass(tmp_path: Path) -> None:
    meas = _write_measurements(tmp_path / "measurements.csv")
    out = tmp_path / "results.json"

    rc = cli.main(["--measurements", str(meas), "--out", str(out)])

    assert rc == 0
    assert out.exists()
    payload = json.loads(out.read_text())
    r = payload["result"]
    assert r["overall_passed"] is True
    assert r["mass_cv"]["passed"] is True
    assert r["j_fan_cv"]["passed"] is True
    assert r["dimension_cv"]["passed"] is True
    assert r["bend_cv"]["passed"] is True

    # Schema sanity — Phase 6 / Drive-ledger consumer relies on these keys.
    assert payload["spec_reference"].startswith("docs/plan_R11.md")
    assert payload["gates"]["cv_gate_pct"] == 5.0
    assert payload["gates"]["n_blades_required"] == 3
    assert payload["inputs"]["n_blades"] == 3
    assert payload["inputs"]["dimension_columns"] == _DIM_COLS

    # Canonical top-level alias for the fabrication noise floor. Downstream
    # Phase 6 consumers compare J_fan deltas against this value; surface it
    # explicitly rather than burying it in result.j_fan_cv.cv_pct.
    assert "published_noise_floor_pct" in payload
    assert payload["published_noise_floor_pct"] == r["j_fan_cv"]["cv_pct"]


def test_cli_pass_with_extra_4th_blade(tmp_path: Path) -> None:
    """Spec floor is 3; a 4th copy for outlier diagnosis is permitted."""
    meas = _write_measurements(
        tmp_path / "measurements.csv",
        masses=[5.000, 5.010, 4.990, 5.005],
        j_fans=[0.3500, 0.3510, 0.3490, 0.3505],
        bends=[1.200, 1.205, 1.195, 1.198],
        dims_per_blade=[[25.000] * 10, [25.010] * 10, [24.990] * 10, [25.005] * 10],
    )
    out = tmp_path / "results.json"
    rc = cli.main(["--measurements", str(meas), "--out", str(out)])
    assert rc == 0
    r = json.loads(out.read_text())["result"]
    assert r["overall_passed"] is True
    assert len(r["per_blade"]) == 4


# ─────────────────────────────────────────────────────────────────────
# Failure paths
# ─────────────────────────────────────────────────────────────────────


def test_cli_fails_on_j_fan_cv_over_5pct(tmp_path: Path) -> None:
    """J_fan 0.30 / 0.40 / 0.50 → CV ≈ 25% → exit 1."""
    meas = _write_measurements(
        tmp_path / "measurements.csv",
        j_fans=[0.30, 0.40, 0.50],
    )
    out = tmp_path / "results.json"
    rc = cli.main(["--measurements", str(meas), "--out", str(out)])
    assert rc == 1
    r = json.loads(out.read_text())["result"]
    assert r["overall_passed"] is False
    assert r["j_fan_cv"]["passed"] is False
    assert r["j_fan_cv"]["cv_pct"] > 5.0


def test_cli_fails_on_mass_cv_over_5pct(tmp_path: Path) -> None:
    """Mass 4.5 / 5.0 / 5.5 → CV = 10% → exit 1; J_fan still tight."""
    meas = _write_measurements(
        tmp_path / "measurements.csv",
        masses=[4.50, 5.00, 5.50],
    )
    out = tmp_path / "results.json"
    rc = cli.main(["--measurements", str(meas), "--out", str(out)])
    assert rc == 1
    r = json.loads(out.read_text())["result"]
    assert r["mass_cv"]["passed"] is False
    assert r["j_fan_cv"]["passed"] is True
    assert r["overall_passed"] is False


# ─────────────────────────────────────────────────────────────────────
# Input error → exit 2
# ─────────────────────────────────────────────────────────────────────


def test_cli_missing_column_exits_2(tmp_path: Path) -> None:
    """Missing 'j_fan_proxy' column → exit 2."""
    bad_header = (
        "blade_id,mass_g,"
        + ",".join(_DIM_COLS)
        + ",bend_deflection_mm,notes"  # NO j_fan_proxy
    )
    rows = [
        bad_header,
        "1,5.000," + ",".join(["25.000"] * 10) + ",1.200,",
        "2,5.010," + ",".join(["25.010"] * 10) + ",1.205,",
        "3,4.990," + ",".join(["24.990"] * 10) + ",1.195,",
    ]
    meas = tmp_path / "measurements.csv"
    meas.write_text("\n".join(rows) + "\n")
    out = tmp_path / "results.json"
    rc = cli.main(["--measurements", str(meas), "--out", str(out)])
    assert rc == 2
    # No results file written on input error.
    assert not out.exists()


def test_cli_too_few_dimension_columns_exits_2(tmp_path: Path) -> None:
    """Only 9 dimension columns → exit 2 (spec floor is 10)."""
    nine_dims = [f"d{i}_mm" for i in range(1, 10)]
    header = (
        "blade_id,mass_g,"
        + ",".join(nine_dims)
        + ",bend_deflection_mm,j_fan_proxy,notes"
    )
    rows = [header]
    for i, m in enumerate([5.000, 5.010, 4.990], start=1):
        rows.append(
            f"{i},{m}," + ",".join(["25.000"] * 9) + ",1.200,0.3500,"
        )
    meas = tmp_path / "measurements.csv"
    meas.write_text("\n".join(rows) + "\n")
    rc = cli.main(["--measurements", str(meas), "--out", str(tmp_path / "results.json")])
    assert rc == 2


def test_cli_too_few_blades_exits_2(tmp_path: Path) -> None:
    """Only 2 blade rows → exit 2 per N_BLADES_REQUIRED."""
    lines = [_HEADER]
    for i, m in enumerate([5.00, 5.01], start=1):
        lines.append(_row(i, m, [25.0] * 10, 1.2, 0.35))
    meas = tmp_path / "measurements.csv"
    meas.write_text("\n".join(lines) + "\n")
    rc = cli.main(["--measurements", str(meas), "--out", str(tmp_path / "results.json")])
    assert rc == 2


def test_cli_missing_file_exits_2(tmp_path: Path) -> None:
    rc = cli.main(
        [
            "--measurements", str(tmp_path / "does_not_exist.csv"),
            "--out", str(tmp_path / "results.json"),
        ]
    )
    assert rc == 2


def test_cli_nonfloat_mass_exits_2(tmp_path: Path) -> None:
    lines = [_HEADER]
    lines.append("1,oops," + ",".join(["25.0"] * 10) + ",1.2,0.35,bad")
    lines.append(_row(2, 5.0, [25.0] * 10, 1.2, 0.35))
    lines.append(_row(3, 5.0, [25.0] * 10, 1.2, 0.35))
    meas = tmp_path / "measurements.csv"
    meas.write_text("\n".join(lines) + "\n")
    rc = cli.main(["--measurements", str(meas), "--out", str(tmp_path / "results.json")])
    assert rc == 2


def test_cli_empty_csv_exits_2(tmp_path: Path) -> None:
    """Header-only CSV → exit 2."""
    meas = tmp_path / "measurements.csv"
    meas.write_text(_HEADER + "\n")
    rc = cli.main(["--measurements", str(meas), "--out", str(tmp_path / "results.json")])
    assert rc == 2
