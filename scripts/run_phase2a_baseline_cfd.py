#!/usr/bin/env python
"""Phase 2a — baseline 2D-slice CFD for rib-TO aero loads (report-final.md §Phase 2a).

Thin orchestration around tested library code. Two steps:

1. ``prepare_baseline_case`` — build the baseline flat-panel cascade cross-section
   (``baseline_cascade_polygons``), mesh it (``mesh_2d_slice``), and render the
   productive + return steady SU2 cfgs. Needs ``gmsh``; runs anywhere.
2. ``extract_baseline_load`` — parse the two SU2 ``history.csv`` outputs into the
   steady two-eval drag-asymmetry proxy (``j_fan``) and a suggested distributed
   aero pressure for ``run_phase2_to.py --pressure-pa``. Needs nothing but the CSVs.

The **SU2 run itself** (between the two steps) needs Colab — see
``notebooks/colab_phase2a_baseline_cfd.ipynb``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from fanopt.cfd.configs import (
    FREESTREAM_DIRECTION_2D_PRODUCTIVE,
    FREESTREAM_DIRECTION_2D_RETURN,
    MACH_STEADY,
    render_slice_steady_cfg,
)
from fanopt.cfd.j_fan import compute_j_fan_steady
from fanopt.cfd.mesh_2d_slice import (
    CASCADE_WALL_MARKER,
    FAN_SURFACE_MARKER,
    FARFIELD_MARKER,
    SliceMeshParams,
    baseline_cascade_polygons,
    build_cascade_slice_mesh,
)
from fanopt.cfd.parsers import steady_run_from_history

MESH_NAME = "baseline_slice.su2"


def prepare_baseline_case(
    out_dir: Path, *, n_blades: int = 5, params: SliceMeshParams | None = None
) -> dict[str, object]:
    """Mesh the baseline flat-panel cascade slice + render both stroke cfgs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    polys = baseline_cascade_polygons(n_blades=n_blades)
    mesh = build_cascade_slice_mesh(polys, params or SliceMeshParams(), out_dir / MESH_NAME)

    def _cfg(direction: tuple[float, float]) -> str:
        return render_slice_steady_cfg(
            mesh_filename=MESH_NAME,
            marker_fan=FAN_SURFACE_MARKER,
            marker_farfield=FARFIELD_MARKER,
            marker_cascade=CASCADE_WALL_MARKER,
            freestream_direction=direction,
        )

    (out_dir / "productive.cfg").write_text(
        _cfg(FREESTREAM_DIRECTION_2D_PRODUCTIVE), encoding="utf-8"
    )
    (out_dir / "return.cfg").write_text(_cfg(FREESTREAM_DIRECTION_2D_RETURN), encoding="utf-8")
    return {
        "mesh": str(out_dir / MESH_NAME),
        "productive_cfg": str(out_dir / "productive.cfg"),
        "return_cfg": str(out_dir / "return.cfg"),
        "n_nodes": mesh.n_nodes,
        "n_blades": n_blades,
    }


# The 2D-slice streamwise force is the drag along the freestream (CD). NOT the
# out-of-plane z-force (CFz) — that is the 3D t̂=+z thrust and is identically 0
# in a 2D slice, which zeroed the whole load.
_STREAMWISE_2D_CANDIDATES = ("CD", "CDrag", "C_D", "Aero_CD")


def extract_baseline_load(
    productive_history: Path, return_history: Path, *, panel_chord_m: float = 0.045
) -> dict[str, object]:
    """Parse the two SU2 histories → steady drag-asymmetry proxy + a suggested
    distributed pressure (proxy force per unit chord) for the rib TO."""
    prod = steady_run_from_history(
        productive_history, stroke="productive", thrust_candidates=_STREAMWISE_2D_CANDIDATES
    )
    ret = steady_run_from_history(
        return_history, stroke="return", thrust_candidates=_STREAMWISE_2D_CANDIDATES
    )
    proxy = compute_j_fan_steady([prod, ret])
    # drag_* are the DIMENSIONLESS CD; the physical load scale is the dynamic
    # pressure q = ½ρV² at the slice (V = MACH·c). The panel sees ~Cp·q with
    # Cp ~ O(1), i.e. a few Pa — the analytic rib-TO load is already in range.
    q_dyn_pa = _dynamic_pressure_pa()
    return {
        "j_fan_steady_proxy": proxy.j_fan_steady_proxy,
        "cd_productive": proxy.drag_productive,
        "cd_return": proxy.drag_return,
        "dynamic_pressure_pa": q_dyn_pa,
        "suggested_pressure_pa": q_dyn_pa,  # load scale ~q; panel load ~Cp·q
    }


def _dynamic_pressure_pa(
    mach: float = MACH_STEADY, temp_k: float = 300.0, pressure_pa: float = 101325.0
) -> float:
    """q = ½ρV² at the steady slice (V = MACH·c, c=√(γRT), ρ=P/RT), standard air."""
    gamma, r_air = 1.4, 287.05
    c = math.sqrt(gamma * r_air * temp_k)
    v = mach * c
    rho = pressure_pa / (r_air * temp_k)
    return 0.5 * rho * v**2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="step", required=True)

    p_prep = sub.add_parser("prepare", help="mesh + render cfgs")
    p_prep.add_argument("--out-dir", type=Path, default=Path("data/phase2a_baseline_cfd"))
    p_prep.add_argument("--n-blades", type=int, default=5)

    p_ext = sub.add_parser("extract", help="parse SU2 histories -> baseline load")
    p_ext.add_argument("--productive-history", type=Path, required=True)
    p_ext.add_argument("--return-history", type=Path, required=True)
    p_ext.add_argument("--out", type=Path, default=Path("data/phase2a_baseline_cfd/load.json"))

    args = parser.parse_args(argv)
    if args.step == "prepare":
        manifest = prepare_baseline_case(args.out_dir, n_blades=args.n_blades)
        print(json.dumps(manifest, indent=2))
    else:
        load = extract_baseline_load(args.productive_history, args.return_history)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(load, indent=2), encoding="utf-8")
        print(json.dumps(load, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
