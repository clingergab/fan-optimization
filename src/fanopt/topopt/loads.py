"""Rib topology-optimization problem setup — domain, BCs, preserved zones, loads
(report-final.md §3.1 / §9.2).

Builds the tapered rib design domain (radial 165 mm × 4→6 mm width, §3.1.2a
dual rib-band lock), the boundary conditions, the preserved zones (rib-panel
interface + fillet, ρ=1), and the multi-load-case pressure loads.

**BC modelling choice (documented):** §9.2 leaves the rib BC under-specified
("Claude writes the FE assembly"). The rib-panel interface + fillet are held as
**preserved zones** (ρ=1) per §3.1.2a. For the structural clamp we anchor the
rib at its **inner radial end** (x = HUB_RADIUS, the junction with the hub/boss
structural region) and load it with distributed aero pressure — the cantilever
model §9.2's own output description assumes ("full material near the root where
bending moment is highest, lightening toward the tip"). The clamp edge is a
config option so the interface-edge clamp can be selected later if Phase 2.5 FEA
prefers it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fanopt.geometry.schema import (
    E_PETG_XY_PA,
    L_RIB_M,
    NU_PETG,
    RIB_BASE_WIDTH_M,
    RIB_THICKNESS_M,
    RIB_TIP_WIDTH_M,
)
from fanopt.topopt.plate_bending import DOF_PER_NODE, PlateMesh

__all__ = [
    "SIMP_PENALTY",
    "EMIN_FACTOR",
    "DEFAULT_VOLFRAC",
    "DEFAULT_ELEM_SIZE_M",
    "DEFAULT_AERO_PRESSURE_PA",
    "LoadCase",
    "RibProblem",
    "build_rib_problem",
]

SIMP_PENALTY: float = 3.0  # §3.1.1
EMIN_FACTOR: float = 1e-9  # E_min = EMIN_FACTOR · E0 (§3.1.1)
DEFAULT_VOLFRAC: float = 0.40  # §3.1.1 (range 0.3–0.5)
DEFAULT_ELEM_SIZE_M: float = 0.0005  # 0.5 mm (§9.2)
DEFAULT_AERO_PRESSURE_PA: float = 10.0  # §9.2 distributed-pressure example


@dataclass(frozen=True)
class LoadCase:
    name: str
    forces: np.ndarray  # global force vector (n_dofs,)
    weight: float = 0.5


@dataclass(frozen=True)
class RibProblem:
    """Fully-specified rib SIMP problem consumed by the solver."""

    mesh: PlateMesh
    active: np.ndarray  # (nely, nelx) bool — inside the tapered rib envelope
    preserved: np.ndarray  # (nely, nelx) bool — ρ=1 clamp (interface + fillet)
    fixed_dofs: np.ndarray
    load_cases: list[LoadCase]
    e0: float = E_PETG_XY_PA
    nu: float = NU_PETG
    thickness_m: float = RIB_THICKNESS_M
    penal: float = SIMP_PENALTY
    volfrac: float = DEFAULT_VOLFRAC
    rmin_elems: float = 3.0  # r_min ≈ 1.5 mm at 0.5 mm elements
    meta: dict[str, float] = field(default_factory=dict)

    @property
    def free(self) -> np.ndarray:
        """Design elements the optimizer may change (active AND not preserved)."""
        return self.active & ~self.preserved


def _tapered_active_mask(nelx: int, nely: int, w_base_el: float, w_tip_el: float) -> np.ndarray:
    """Boolean (nely, nelx) mask of elements inside the tapered rib envelope,
    centered on the mid-width row, widening linearly root→tip."""
    active = np.zeros((nely, nelx), dtype=bool)
    for ex in range(nelx):
        frac = ex / max(nelx - 1, 1)
        w_local = w_base_el + (w_tip_el - w_base_el) * frac
        n_active = max(int(round(w_local)), 1)
        y0 = (nely - n_active) // 2
        active[y0 : y0 + n_active, ex] = True
    return active


def _pressure_forces(mesh: PlateMesh, active: np.ndarray, pressure_pa: float) -> np.ndarray:
    """Consistent nodal transverse (w-DOF) loads for a uniform pressure over the
    active elements: each element distributes ``p·dA/4`` to its 4 nodes."""
    f = np.zeros(mesh.n_dofs)
    tributary = pressure_pa * mesh.dx * mesh.dy / 4.0
    for ey in range(mesh.nely):
        for ex in range(mesh.nelx):
            if not active[ey, ex]:
                continue
            for nd in mesh.element_nodes(ex, ey):
                f[DOF_PER_NODE * nd] += tributary
    return f


def _root_fixed_dofs(mesh: PlateMesh, active: np.ndarray) -> np.ndarray:
    """Clamp (w, θx, θy = 0) the nodes on the inner radial edge (ex = 0 column)
    that belong to active elements — the rib's anchor at the hub."""
    dofs: list[int] = []
    active_cols0 = np.where(active[:, 0])[0]
    if active_cols0.size == 0:
        active_cols0 = np.where(active.any(axis=1))[0]  # pragma: no cover - degenerate mesh
    iy_lo, iy_hi = active_cols0.min(), active_cols0.max() + 1
    for iy in range(iy_lo, iy_hi + 1):
        nid = mesh.node_id(0, iy)
        dofs += [DOF_PER_NODE * nid, DOF_PER_NODE * nid + 1, DOF_PER_NODE * nid + 2]
    return np.array(sorted(set(dofs)))


def _interface_preserved_mask(active: np.ndarray) -> np.ndarray:
    """Rib-panel interface + fillet preserved zone (ρ=1): the panel-facing edge
    band — the lowest active row of each radial column (a 1-element fillet strip)."""
    preserved = np.zeros_like(active, dtype=bool)
    for ex in range(active.shape[1]):
        rows = np.where(active[:, ex])[0]
        if rows.size:
            preserved[rows.min(), ex] = True
    return preserved


def build_rib_problem(
    *,
    elem_size_m: float = DEFAULT_ELEM_SIZE_M,
    pressure_pa: float = DEFAULT_AERO_PRESSURE_PA,
    volfrac: float = DEFAULT_VOLFRAC,
    rmin_m: float = 0.0015,
) -> RibProblem:
    """Assemble the canonical rib SIMP problem from the locked geometry."""
    nelx = max(int(round(L_RIB_M / elem_size_m)), 2)
    nely = max(int(round(RIB_TIP_WIDTH_M / elem_size_m)), 2)
    mesh = PlateMesh(nelx=nelx, nely=nely, dx=elem_size_m, dy=elem_size_m)

    w_base_el = RIB_BASE_WIDTH_M / elem_size_m
    w_tip_el = RIB_TIP_WIDTH_M / elem_size_m
    active = _tapered_active_mask(nelx, nely, w_base_el, w_tip_el)
    preserved = _interface_preserved_mask(active)
    fixed = _root_fixed_dofs(mesh, active)

    # Productive (+p) and return (−p) strokes; equal weight (§3.1.1 w=0.5 each).
    f_prod = _pressure_forces(mesh, active, pressure_pa)
    load_cases = [
        LoadCase("productive", f_prod, 0.5),
        LoadCase("return", -f_prod, 0.5),
    ]
    return RibProblem(
        mesh=mesh,
        active=active,
        preserved=preserved,
        fixed_dofs=fixed,
        load_cases=load_cases,
        volfrac=volfrac,
        rmin_elems=rmin_m / elem_size_m,
        meta={
            "nelx": float(nelx),
            "nely": float(nely),
            "n_active": float(active.sum()),
            "pressure_pa": pressure_pa,
        },
    )
