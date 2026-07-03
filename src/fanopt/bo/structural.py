"""Panel-stiffness structural objective — panel tip deflection (Phase 4 Pareto axis).

Injected as ``structural_fn`` into :func:`fanopt.bo.objective.evaluate_design`. A
Reissner-Mindlin plate clamped at the hub edge and loaded by a **fixed nominal**
aero pressure; the design's Path A+ thickness field sets the per-element bending
stiffness, so thicker / stiffer panels deflect less. This is *independent* of the
aero (J_fan) axis — it rewards a genuinely different design trait — which the
rib-under-CFD-pressure alternative would not (there ``u_tip ∝ pressure``, nearly
collinear with airflow).

Modeling scope: the plate carries the thickness-grid stiffening exactly and the
corrugation's stiffening partially (through the convexity of the ``t³`` bending
term — a corrugated field of the same mean thickness is stiffer). The full
geometric stiffening of a corrugated shell (material moved off the neutral
surface) would need a shell model and is a documented V1 approximation. Plate
size is fixed across designs so deflection differences isolate the thickness
field. Uses the tested :mod:`fanopt.topopt.plate_bending` primitives.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix

from fanopt.cfd.panel_slice import panel_layout_at_radius
from fanopt.geometry.envelope import Layer1Params, ThicknessGridField
from fanopt.geometry.schema import E_PETG_XY_PA, L_BLADE_M, NU_PETG
from fanopt.topopt.plate_bending import (
    DOF_PER_NODE,
    PlateMesh,
    element_stiffness_unit,
    solve_displacements,
)

__all__ = [
    "NOMINAL_PRESSURE_PA",
    "PANEL_NELX",
    "PANEL_NELY",
    "panel_tip_deflection_m",
]

NOMINAL_PRESSURE_PA: float = 10.0  # §9.2 distributed-pressure example (Phase 2a baseline)
PANEL_NELX: int = 20  # radial elements (hub → tip)
PANEL_NELY: int = 6  # tangential elements
_WIDTH_RADIAL_U: float = 0.5  # representative panel width taken at mid-radius


def _element_thickness_grid(field: ThicknessGridField, nelx: int, nely: int) -> np.ndarray:
    """Per-element panel thickness sampled at element centroids (``(nely, nelx)``)."""
    t = np.empty((nely, nelx), dtype=float)
    for ey in range(nely):
        v = 2.0 * (ey + 0.5) / nely - 1.0
        for ex in range(nelx):
            u = (ex + 0.5) / nelx
            t[ey, ex] = field.thickness_at(u, v)
    return t


def _assemble_varying_thickness(
    mesh: PlateMesh, thickness_e: np.ndarray, *, nu: float, e_pa: float
) -> csc_matrix:
    """Global stiffness for a solid panel with per-element thickness (E = ``e_pa``)."""
    n_e = mesh.nelx * mesh.nely
    rows = np.empty(n_e * 144, dtype=np.int64)
    cols = np.empty(n_e * 144, dtype=np.int64)
    vals = np.empty(n_e * 144, dtype=float)
    p = 0
    for ey in range(mesh.nely):
        for ex in range(mesh.nelx):
            ke = e_pa * element_stiffness_unit(nu, float(thickness_e[ey, ex]), mesh.dx, mesh.dy)
            edofs = mesh.element_dofs(ex, ey)
            rr, cc = np.meshgrid(edofs, edofs, indexing="ij")
            rows[p : p + 144] = rr.ravel()
            cols[p : p + 144] = cc.ravel()
            vals[p : p + 144] = ke.ravel()
            p += 144
    k = csc_matrix((vals, (rows, cols)), shape=(mesh.n_dofs, mesh.n_dofs))
    return (k + k.T) * 0.5


def _uniform_pressure_forces(mesh: PlateMesh, pressure_pa: float) -> np.ndarray:
    """Consistent transverse (w-DOF) nodal loads for a uniform pressure."""
    f = np.zeros(mesh.n_dofs)
    tributary = pressure_pa * mesh.dx * mesh.dy / 4.0
    for ey in range(mesh.nely):
        for ex in range(mesh.nelx):
            for nd in mesh.element_nodes(ex, ey):
                f[DOF_PER_NODE * nd] += tributary
    return f


def _root_clamped_dofs(mesh: PlateMesh) -> np.ndarray:
    """Fix (w, θx, θy) on the inner radial edge (``ex = 0`` nodes) — the hub anchor."""
    dofs: list[int] = []
    for iy in range(mesh.nely + 1):
        nid = mesh.node_id(0, iy)
        dofs += [DOF_PER_NODE * nid, DOF_PER_NODE * nid + 1, DOF_PER_NODE * nid + 2]
    return np.array(sorted(set(dofs)))


def panel_tip_deflection_m(
    layer1: Layer1Params,
    *,
    pressure_pa: float = NOMINAL_PRESSURE_PA,
    nelx: int = PANEL_NELX,
    nely: int = PANEL_NELY,
) -> float:
    """Max transverse deflection (m) of the panel under a fixed nominal pressure.

    The plate spans ``L_BLADE_M`` radially × the mid-radius panel width
    tangentially, clamped at the hub edge; per-element thickness comes from the
    design's :class:`ThicknessGridField`. Larger return ⇒ floppier panel.
    """
    if nelx < 1 or nely < 1:
        raise ValueError(f"nelx, nely must be >= 1; got {nelx}, {nely}")
    width_m = panel_layout_at_radius(_WIDTH_RADIAL_U).panel_width_m
    mesh = PlateMesh(nelx=nelx, nely=nely, dx=L_BLADE_M / nelx, dy=width_m / nely)
    thickness_e = _element_thickness_grid(layer1.thickness_field, nelx, nely)
    k = _assemble_varying_thickness(mesh, thickness_e, nu=NU_PETG, e_pa=E_PETG_XY_PA)
    f = _uniform_pressure_forces(mesh, pressure_pa)
    u = solve_displacements(k, f, _root_clamped_dofs(mesh))
    return float(np.abs(u[0::DOF_PER_NODE]).max())  # max |w| over all nodes
