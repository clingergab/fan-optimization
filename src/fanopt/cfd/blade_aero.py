"""New solid blade → 2D cascade-slice CFD → J_fan (aero-first V1 objective spine).

The aero evaluation for the redesigned blade, and the thing the optimizer maximizes.
It reuses the Phase-3 slice pipeline verbatim — only the cross-section changes:

    BladeParams
      → blade_slice_polygons          (the solid-blade cascade cross-section)
      → build_cascade_slice_mesh      (2D cascade mesh, kept)
      → render_slice_{steady,unsteady}_cfg
      → SU2  (steady productive-stroke drag screen + unsteady plunging run)
      → J_fan  (net momentum flux over the plunge cycle) + steady-CD screen

`J_fan` (the cycle-mean plunging force / net momentum flux) is the primary aero
objective; the steady CD is the cheap screening proxy. Pure geometry/config generation
is separated from the SU2 subprocess so the former is testable without a solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fanopt.cfd.blade_slice import blade_slice_polygons
from fanopt.cfd.configs import render_slice_steady_cfg, render_slice_unsteady_cfg
from fanopt.cfd.mesh_2d_slice import (
    CASCADE_WALL_MARKER,
    FAN_SURFACE_MARKER,
    FARFIELD_MARKER,
    SliceMeshParams,
    build_cascade_slice_mesh,
)
from fanopt.cfd.phase3 import extract_steady_drag, extract_unsteady_mean, find_su2, run_su2
from fanopt.geometry.blade import BladeParams

__all__ = [
    "MESH_NAME",
    "BladeAeroResult",
    "prepare_blade_aero_case",
    "evaluate_blade_aero",
]

MESH_NAME = "blade_slice.su2"
_N_CYCLES = 5
_INNER_ITER = 50
_STEPS_PER_CYCLE = 200


@dataclass(frozen=True)
class BladeAeroResult:
    """Aero observables for one blade design."""

    j_fan: float  # net momentum flux over the plunge cycle — the primary objective
    steady_cd: float  # productive-stroke steady drag — the cheap screen
    n_nodes: int


def prepare_blade_aero_case(
    params: BladeParams,
    workdir: Path,
    *,
    radial_u: float = 0.5,
    n_panels: int = 5,
    n_samples: int = 32,
    mesh_params: SliceMeshParams | None = None,
    n_cycles: int = _N_CYCLES,
    inner_iter: int = _INNER_ITER,
    steps_per_cycle: int = _STEPS_PER_CYCLE,
) -> dict[str, Any]:
    """Mesh the blade slice + render its steady and unsteady cfgs into ``workdir``.

    Pure geometry/config generation (no SU2) — needs gmsh for the mesh. Returns the mesh
    path + node count.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    polys = blade_slice_polygons(
        params, radial_u=radial_u, n_panels=n_panels, n_samples=n_samples
    )
    mesh = build_cascade_slice_mesh(
        polys, mesh_params or SliceMeshParams(), workdir / MESH_NAME
    )
    steady = render_slice_steady_cfg(
        mesh_filename=MESH_NAME,
        marker_fan=FAN_SURFACE_MARKER,
        marker_farfield=FARFIELD_MARKER,
        marker_cascade=CASCADE_WALL_MARKER,
    )
    unsteady = render_slice_unsteady_cfg(
        mesh_filename=MESH_NAME,
        marker_fan=FAN_SURFACE_MARKER,
        marker_farfield=FARFIELD_MARKER,
        marker_cascade=CASCADE_WALL_MARKER,
        n_cycles=n_cycles,
        inner_iter=inner_iter,
        steps_per_cycle=steps_per_cycle,
    )
    (workdir / "steady.cfg").write_text(steady, encoding="utf-8")
    (workdir / "unsteady.cfg").write_text(unsteady, encoding="utf-8")
    return {"mesh": str(workdir / MESH_NAME), "n_nodes": mesh.n_nodes}


def evaluate_blade_aero(
    params: BladeParams,
    workdir: Path,
    *,
    su2_bin: str | None = None,
    radial_u: float = 0.5,
    n_panels: int = 5,
    n_samples: int = 32,
    mesh_params: SliceMeshParams | None = None,
    n_cycles: int = _N_CYCLES,
    inner_iter: int = _INNER_ITER,
    steps_per_cycle: int = _STEPS_PER_CYCLE,
) -> BladeAeroResult:
    """Full aero eval: prepare + run SU2 (steady + unsteady) + reduce to J_fan.

    Runs SU2 as a subprocess (the IO boundary). Raises if SU2 isn't found.
    """
    su2 = su2_bin or find_su2()
    if su2 is None:
        raise RuntimeError("SU2_CFD not found (set $SU2_RUN or pass su2_bin)")
    info = prepare_blade_aero_case(
        params,
        workdir,
        radial_u=radial_u,
        n_panels=n_panels,
        n_samples=n_samples,
        mesh_params=mesh_params,
        n_cycles=n_cycles,
        inner_iter=inner_iter,
        steps_per_cycle=steps_per_cycle,
    )
    steady_hist = run_su2("steady.cfg", workdir, su2)
    steady_cd = extract_steady_drag(steady_hist)
    unsteady_hist = run_su2("unsteady.cfg", workdir, su2)
    j_fan = extract_unsteady_mean(unsteady_hist, n_cycles=n_cycles)
    return BladeAeroResult(j_fan=j_fan, steady_cd=steady_cd, n_nodes=int(info["n_nodes"]))
