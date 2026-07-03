"""Rib SIMP topology-optimization loop — OC update, density filter, multi-load
compliance minimization (report-final.md §3.1 / §9.2).

Ties together the Mindlin plate FE core (:mod:`plate_bending`), the SIMP
interpolation + filter + OC update (:mod:`simp`), and the rib problem setup
(:mod:`loads`). Pure numpy/scipy — runs locally on the M3, no FEniCSx/Colab.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fanopt.topopt.loads import EMIN_FACTOR, RibProblem
from fanopt.topopt.plate_bending import (
    DOF_PER_NODE,
    assemble_global_stiffness,
    element_stiffness_unit,
    solve_displacements,
)
from fanopt.topopt.simp import apply_filter, build_density_filter, oc_update, simp_modulus

__all__ = ["RibTOResult", "run_rib_topology_optimization"]


@dataclass(frozen=True)
class RibTOResult:
    density: np.ndarray  # (nely, nelx) physical density ρ̃
    compliance_history: tuple[float, ...]
    volume_fraction: float
    u_tip_max_m: float
    converged: bool
    iterations: int
    meta: dict[str, float] = field(default_factory=dict)


def _element_dof_table(problem: RibProblem) -> np.ndarray:
    """(n_e, 12) global-DOF indices for every element, row-major (ey, ex)."""
    mesh = problem.mesh
    table = np.empty((mesh.nely * mesh.nelx, 12), dtype=np.int64)
    for ey in range(mesh.nely):
        for ex in range(mesh.nelx):
            table[ey * mesh.nelx + ex] = mesh.element_dofs(ex, ey)
    return table


def run_rib_topology_optimization(
    problem: RibProblem,
    *,
    max_iters: int = 60,
    tol: float = 0.01,
    move: float = 0.2,
) -> RibTOResult:
    """Run the SIMP TO loop to convergence (or ``max_iters``).

    Convergence = max design-variable change < ``tol``. Returns the physical
    density field, per-iteration compliance, achieved volume fraction, and the
    peak tip deflection (for the §3.1 ``u_tip < 1 mm`` rigid-blade check).
    """
    mesh = problem.mesh
    active = problem.active
    free = problem.free
    e0, emin = problem.e0, EMIN_FACTOR * problem.e0
    penal = problem.penal
    ke_unit = element_stiffness_unit(problem.nu, problem.thickness_m, mesh.dx, mesh.dy)
    edof = _element_dof_table(problem)
    hs = build_density_filter(mesh.nely, mesh.nelx, problem.rmin_elems)

    # Design variables: free -> volfrac, preserved -> 1, inactive -> 0.
    x = np.zeros((mesh.nely, mesh.nelx))
    x[free] = problem.volfrac
    x[problem.preserved] = 1.0

    dv_drho = np.ones_like(x)  # ∂V/∂ρ̃ (unit element volumes)
    dv_dx = apply_filter(hs.T.tocsr(), dv_drho)

    history: list[float] = []
    converged = False
    iterations = 0
    rho = x.copy()
    for it in range(1, max_iters + 1):
        iterations = it
        rho = apply_filter(hs, x)
        e_elem = np.full((mesh.nely, mesh.nelx), emin)
        e_elem[active] = simp_modulus(rho[active], penal, e0, emin)

        k = assemble_global_stiffness(mesh, e_elem, problem.nu, problem.thickness_m)

        compliance = 0.0
        dc_drho = np.zeros_like(x)
        for lc in problem.load_cases:
            u = solve_displacements(k, lc.forces, problem.fixed_dofs)
            ue = u[edof]  # (n_e, 12)
            energy = np.einsum("ei,ij,ej->e", ue, ke_unit, ue).reshape(mesh.nely, mesh.nelx)
            compliance += lc.weight * float(lc.forces @ u)
            # dC/dρ̃ = −p ρ̃^(p−1)(E0−Emin)·energy ; only active elements vary.
            dc_drho[active] += (
                -lc.weight * penal * rho[active] ** (penal - 1.0) * (e0 - emin) * energy[active]
            )
        history.append(compliance)

        dc_dx = apply_filter(hs.T.tocsr(), dc_drho)
        x_new = oc_update(
            x, dc_dx, dv_dx, hs, volfrac=problem.volfrac, free=free, active=active, move=move
        )
        change = float(np.abs(x_new - x)[free].max()) if free.any() else 0.0
        x = x_new
        if change < tol:
            converged = True
            break

    rho = apply_filter(hs, x)
    # Peak transverse deflection under the productive stroke (u_tip check).
    e_elem = np.full((mesh.nely, mesh.nelx), emin)
    e_elem[active] = simp_modulus(rho[active], penal, e0, emin)
    k = assemble_global_stiffness(mesh, e_elem, problem.nu, problem.thickness_m)
    u_prod = solve_displacements(k, problem.load_cases[0].forces, problem.fixed_dofs)
    w_dofs = np.arange(mesh.n_nodes) * DOF_PER_NODE
    u_tip = float(np.abs(u_prod[w_dofs]).max())

    vol_frac = float(rho[active].sum() / active.sum()) if active.any() else 0.0
    return RibTOResult(
        density=rho,
        compliance_history=tuple(history),
        volume_fraction=vol_frac,
        u_tip_max_m=u_tip,
        converged=converged,
        iterations=iterations,
        meta=dict(problem.meta),
    )
