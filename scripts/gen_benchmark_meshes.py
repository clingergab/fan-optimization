#!/usr/bin/env python
"""Generate SU2 meshes for Spike 0.6c — NACA 0012 benchmark + probe.

Two meshes:

**probe**: trivial 2D square with airfoil + farfield markers. Sole purpose
is to let Spike 0.6c.1 parse + launch SU2 for one outer time-step. The
mesh quality is irrelevant; the spike only checks that SU2's parser
accepts the locked Tier-1 cfg (Round-9 HIGH-12 numerics).

**naca0012**: 2D O-grid mesh around a NACA 0012 airfoil at Re ~40k
resolution. ~30-40k cells; sufficient for the ±15% benchmark gate at
k_reduced ~0.5-0.6 (Spike 0.6c.2). Uses the gmsh Python API so the mesh
is generated inside the Colab notebook without checking a 5 MB file
into the repo.

Markers (must match `oscillating_airfoil_benchmark.cfg.j2`):
  AIRFOIL   - viscous wall on the airfoil surface
  FARFIELD  - outer-boundary far-field

Usage (from the Colab notebook):

    python scripts/gen_benchmark_meshes.py \\
        --kind naca0012 \\
        --out data/spike_0_6c/meshes/naca0012.su2

    python scripts/gen_benchmark_meshes.py \\
        --kind probe \\
        --out data/spike_0_6c/meshes/probe.su2

Dependencies:
- ``gmsh`` Python wheel (``pip install gmsh``). Tested with gmsh 4.11+.

Importing this module requires gmsh on sys.path. Pure-Python NACA
shape utilities (no gmsh dep) live in ``src/fanopt/cfd/airfoil_shapes.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gmsh  # type: ignore[import-not-found]

from fanopt.cfd.airfoil_shapes import NACA0012_CHORD, airfoil_polyline

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MESH_DIR = REPO_ROOT / "data" / "spike_0_6c" / "meshes"


def build_naca0012_mesh(
    out_path: Path,
    *,
    chord: float = NACA0012_CHORD,
    farfield_radius_chord: float = 50.0,
    n_airfoil_points: int = 200,
    cells_circ: int = 240,
) -> Path:
    """Build a 2D O-grid mesh around a NACA 0012 airfoil and write .su2.

    `farfield_radius_chord` of 50 is standard for low-Re aero benchmarks.
    The distance-threshold field clusters cells near the wall for boundary
    layer resolution.
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay (2D)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)  # Quads
        gmsh.option.setNumber("Mesh.Smoothing", 10)
        gmsh.model.add("naca0012")

        # ---- Airfoil curve (closed polyline) ----
        pts = airfoil_polyline(n_airfoil_points, chord=chord)
        airfoil_tags = []
        for x, y in pts:
            airfoil_tags.append(gmsh.model.geo.addPoint(x, y, 0.0))
        airfoil_curve = gmsh.model.geo.addSpline(airfoil_tags + [airfoil_tags[0]])
        airfoil_loop = gmsh.model.geo.addCurveLoop([airfoil_curve])

        # ---- Far-field circle centered at the airfoil quarter-chord ----
        cx, cy = 0.25 * chord, 0.0
        ff_center = gmsh.model.geo.addPoint(cx, cy, 0.0)
        ff_radius = farfield_radius_chord * chord
        ff_pts = [
            gmsh.model.geo.addPoint(cx + ff_radius, cy, 0.0),
            gmsh.model.geo.addPoint(cx, cy + ff_radius, 0.0),
            gmsh.model.geo.addPoint(cx - ff_radius, cy, 0.0),
            gmsh.model.geo.addPoint(cx, cy - ff_radius, 0.0),
        ]
        ff_arcs = [
            gmsh.model.geo.addCircleArc(ff_pts[0], ff_center, ff_pts[1]),
            gmsh.model.geo.addCircleArc(ff_pts[1], ff_center, ff_pts[2]),
            gmsh.model.geo.addCircleArc(ff_pts[2], ff_center, ff_pts[3]),
            gmsh.model.geo.addCircleArc(ff_pts[3], ff_center, ff_pts[0]),
        ]
        ff_loop = gmsh.model.geo.addCurveLoop(ff_arcs)

        # ---- Surface = annulus between airfoil and far-field ----
        surface = gmsh.model.geo.addPlaneSurface([ff_loop, airfoil_loop])

        gmsh.model.geo.synchronize()

        # ---- Mesh sizing: Distance + Threshold field ----
        dist_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_field, "CurvesList", [airfoil_curve])
        gmsh.model.mesh.field.setNumber(dist_field, "Sampling", 200)

        thresh_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thresh_field, "InField", dist_field)
        gmsh.model.mesh.field.setNumber(thresh_field, "SizeMin", chord / cells_circ)
        gmsh.model.mesh.field.setNumber(thresh_field, "SizeMax", chord)
        gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", ff_radius * 0.2)

        gmsh.model.mesh.field.setAsBackgroundMesh(thresh_field)

        # ---- Physical groups (SU2 markers) ----
        gmsh.model.addPhysicalGroup(1, [airfoil_curve], name="AIRFOIL")
        gmsh.model.addPhysicalGroup(1, ff_arcs, name="FARFIELD")
        gmsh.model.addPhysicalGroup(2, [surface], name="FLOW")

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Netgen")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(out_path))
        return out_path
    finally:
        gmsh.finalize()


def build_probe_mesh(out_path: Path) -> Path:
    """Build a trivial 2D box with airfoil + farfield markers.

    The probe mesh's geometry is meaningless — it exists only so SU2's
    parser can validate the locked Tier-1 cfg in 1 outer step. The two
    markers (AIRFOIL, FARFIELD) match what the benchmark cfg expects.
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("probe")

        # Outer box, 2 × 2 square.
        p1 = gmsh.model.geo.addPoint(-1.0, -1.0, 0.0, 0.5)
        p2 = gmsh.model.geo.addPoint(1.0, -1.0, 0.0, 0.5)
        p3 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0, 0.5)
        p4 = gmsh.model.geo.addPoint(-1.0, 1.0, 0.0, 0.5)
        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)
        outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])

        # Inner "airfoil" — tiny square in the middle.
        q1 = gmsh.model.geo.addPoint(-0.1, -0.05, 0.0, 0.05)
        q2 = gmsh.model.geo.addPoint(0.1, -0.05, 0.0, 0.05)
        q3 = gmsh.model.geo.addPoint(0.1, 0.05, 0.0, 0.05)
        q4 = gmsh.model.geo.addPoint(-0.1, 0.05, 0.0, 0.05)
        m1 = gmsh.model.geo.addLine(q1, q2)
        m2 = gmsh.model.geo.addLine(q2, q3)
        m3 = gmsh.model.geo.addLine(q3, q4)
        m4 = gmsh.model.geo.addLine(q4, q1)
        inner_loop = gmsh.model.geo.addCurveLoop([m1, m2, m3, m4])

        surface = gmsh.model.geo.addPlaneSurface([outer_loop, inner_loop])

        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(1, [m1, m2, m3, m4], name="AIRFOIL")
        gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4], name="FARFIELD")
        gmsh.model.addPhysicalGroup(2, [surface], name="FLOW")

        gmsh.model.mesh.generate(2)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(out_path))
        return out_path
    finally:
        gmsh.finalize()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=("naca0012", "probe"),
        required=True,
        help="Which mesh to build.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output .su2 path. Defaults to "
            "data/spike_0_6c/meshes/<kind>.su2 under the repo root."
        ),
    )
    parser.add_argument(
        "--chord",
        type=float,
        default=NACA0012_CHORD,
        help="Airfoil chord (m). Default 1.0. NACA 0012 only.",
    )
    parser.add_argument(
        "--farfield-radius-chord",
        type=float,
        default=50.0,
        help="Far-field radius in multiples of chord. Default 50.",
    )
    parser.add_argument(
        "--n-airfoil-points",
        type=int,
        default=200,
        help="Number of points along the airfoil surface. Default 200.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out or (DEFAULT_MESH_DIR / f"{args.kind}.su2")

    if args.kind == "naca0012":
        path = build_naca0012_mesh(
            out,
            chord=args.chord,
            farfield_radius_chord=args.farfield_radius_chord,
            n_airfoil_points=args.n_airfoil_points,
        )
    else:
        path = build_probe_mesh(out)

    print(f"[gen_meshes] wrote {args.kind} mesh to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
