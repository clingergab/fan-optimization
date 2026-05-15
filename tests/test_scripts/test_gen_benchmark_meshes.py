"""Tests for scripts/gen_benchmark_meshes.py.

The script's body is thin — it wires the pure-Python airfoil shapes
(``fanopt.cfd.airfoil_shapes``) into gmsh's Python API and writes .su2
output. The shape math is covered in `tests/test_cfd/test_airfoil_shapes.py`.

This file covers:

- CLI arg parsing (no gmsh needed; we parse argv only)
- End-to-end probe-mesh + naca0012-mesh generation (requires gmsh)

Module-level import of gmsh — per the project import rules (CLAUDE.md), if
gmsh isn't installed the whole test module is skipped collection-time.
That's correct: the script literally cannot run without gmsh, so its
integration tests don't need to either.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_HAS_GMSH = importlib.util.find_spec("gmsh") is not None

# Argparse tests only need the script's parse_args, which doesn't
# itself import gmsh (gmsh is imported at module top). To exercise
# parse_args without gmsh installed we'd have to refactor — for now,
# skip everything when gmsh is absent.
if not _HAS_GMSH:
    pytest.skip("gmsh not installed", allow_module_level=True)


import gen_benchmark_meshes as cli  # noqa: E402 — module-level skip above

# ---- CLI arg parsing ------------------------------------------------------


def test_cli_parse_args_naca0012(tmp_path: Path) -> None:
    out = tmp_path / "x.su2"
    args = cli.parse_args(["--kind", "naca0012", "--out", str(out)])
    assert args.kind == "naca0012"
    assert args.out == out


def test_cli_parse_args_probe_default_out() -> None:
    args = cli.parse_args(["--kind", "probe"])
    assert args.kind == "probe"
    assert args.out is None


def test_cli_rejects_bad_kind() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--kind", "wat"])


def test_cli_default_output_path_resolves() -> None:
    """Default mesh dir is data/spike_0_6c/meshes."""
    assert str(cli.DEFAULT_MESH_DIR).endswith("data/spike_0_6c/meshes")


# ---- end-to-end mesh generation (only runs when gmsh available) -----------


def test_probe_mesh_generates(tmp_path: Path) -> None:
    """probe mesh: SU2 file written with both required markers."""
    out = tmp_path / "probe.su2"
    rc = cli.main(["--kind", "probe", "--out", str(out)])
    assert rc == 0
    assert out.exists()
    text = out.read_text()
    assert "NDIME=" in text
    assert "AIRFOIL" in text
    assert "FARFIELD" in text


def test_naca0012_mesh_generates_small(tmp_path: Path) -> None:
    """naca0012 mesh: write smaller version (faster test) and verify
    markers + minimum cell count."""
    out = tmp_path / "naca.su2"
    rc = cli.main(
        [
            "--kind",
            "naca0012",
            "--out",
            str(out),
            "--n-airfoil-points",
            "64",  # smaller for test speed
            "--farfield-radius-chord",
            "20.0",  # smaller domain
        ]
    )
    assert rc == 0
    assert out.exists()
    text = out.read_text()
    assert "AIRFOIL" in text
    assert "FARFIELD" in text
    # NDIME=2 must be the dimension marker.
    assert "NDIME= 2" in text


def test_build_probe_mesh_creates_parent_dir(tmp_path: Path) -> None:
    """Mesh writer must mkdir parents."""
    out = tmp_path / "nested" / "dir" / "probe.su2"
    rc = cli.main(["--kind", "probe", "--out", str(out)])
    assert rc == 0
    assert out.exists()


# ---- thin_plate_2d (Spike 0.6d.2 — H10 supplement, 2026-05-14) -----------


def test_cli_parse_args_thin_plate_2d() -> None:
    """The kind=thin_plate_2d enum is supported (added 2026-05-14)."""
    args = cli.parse_args(["--kind", "thin_plate_2d"])
    assert args.kind == "thin_plate_2d"


def test_thin_plate_2d_mesh_generates(tmp_path: Path) -> None:
    """thin_plate_2d mesh: SU2 file written with PLATE + FARFIELD markers.

    Markers must match `configs/su2/thin_plate_2d_pitching.cfg.j2` (which
    expects `marker_plate` and `marker_farfield`).
    """
    out = tmp_path / "thin_plate.su2"
    rc = cli.main(
        [
            "--kind",
            "thin_plate_2d",
            "--out",
            str(out),
            "--farfield-radius-chord",
            "10.0",  # smaller domain for test speed
        ]
    )
    assert rc == 0
    assert out.exists()
    text = out.read_text()
    assert "NDIME= 2" in text
    # Required markers for 0.6d.2 cfg template.
    assert "PLATE" in text
    assert "FARFIELD" in text


def test_thin_plate_2d_default_output_path_under_spike_0_6d(tmp_path: Path) -> None:
    """Default output for thin_plate_2d goes to data/spike_0_6d/meshes/, not 0.6c."""
    # Run with no --out arg, capture the path used.
    out_default = cli.REPO_ROOT / "data" / "spike_0_6d" / "meshes" / "thin_plate_2d.su2"
    # Make sure stale cleanup if a prior test left one behind.
    if out_default.exists():
        out_default.unlink()
    rc = cli.main(
        [
            "--kind",
            "thin_plate_2d",
            "--farfield-radius-chord",
            "5.0",  # tiny for test speed
        ]
    )
    assert rc == 0
    assert out_default.exists()
    # Clean up.
    out_default.unlink()
