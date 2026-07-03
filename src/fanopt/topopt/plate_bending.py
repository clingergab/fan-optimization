"""Reissner-Mindlin plate-bending finite element (Q4) — the FE core for the
rib SIMP topology optimization (report-final.md §3.1 / §9.2).

Pure numpy/scipy (the DTU-TopOpt path, §9.2) — no FEniCSx dependency. A 4-node
bilinear plate element with 3 DOF/node ``[w, θx, θy]`` and **selective reduced
integration** (2×2 Gauss on bending, 1×1 on shear) to avoid shear locking on
the thin (2 mm) rib.

Kinematics (mid-surface normal rotations βx=θx, βy=θy):
- curvature   κ = [θx,x, θy,y, θx,y + θy,x]
- shear       γ = [w,x + θx, w,y + θy]

The element stiffness is linear in E, so ``Ke(E) = E · Ke_unit`` — exactly the
form SIMP needs (``E_eff(ρ) = E_min + ρ^p (E0 − E_min)``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

__all__ = [
    "SHEAR_CORRECTION",
    "DOF_PER_NODE",
    "PlateMesh",
    "element_stiffness_unit",
    "assemble_global_stiffness",
    "solve_displacements",
]

SHEAR_CORRECTION: float = 5.0 / 6.0  # k — Reissner-Mindlin shear correction factor
DOF_PER_NODE: int = 3  # [w, θx, θy]

# 2×2 Gauss points for the bending term (weights all 1).
_G = 1.0 / np.sqrt(3.0)
_GAUSS_2X2 = ((-_G, -_G), (_G, -_G), (_G, _G), (-_G, _G))


def _q4_derivs(
    xi: float, eta: float, dx: float, dy: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bilinear Q4 shape functions + x/y derivatives at natural coords (xi, eta).

    Element spans ``[-1, 1]²`` in (xi, eta) mapped to a ``dx × dy`` rectangle,
    so ``∂/∂x = (2/dx)·∂/∂xi`` and ``∂/∂y = (2/dy)·∂/∂eta``.
    """
    n = 0.25 * np.array(
        [(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]
    )
    dndxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    dndeta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return n, dndxi * (2.0 / dx), dndeta * (2.0 / dy)


def _bending_B(dndx: np.ndarray, dndy: np.ndarray) -> np.ndarray:
    """Curvature-displacement matrix Bb (3×12) — κ = Bb·d, d=[w,θx,θy]×4."""
    b = np.zeros((3, 12))
    for i in range(4):
        c = 3 * i
        b[0, c + 1] = dndx[i]  # κxx = θx,x
        b[1, c + 2] = dndy[i]  # κyy = θy,y
        b[2, c + 1] = dndy[i]  # κxy = θx,y + θy,x
        b[2, c + 2] = dndx[i]
    return b


def _shear_B(n: np.ndarray, dndx: np.ndarray, dndy: np.ndarray) -> np.ndarray:
    """Shear-strain matrix Bs (2×12) — γ = Bs·d."""
    b = np.zeros((2, 12))
    for i in range(4):
        c = 3 * i
        b[0, c] = dndx[i]  # γxz = w,x + θx
        b[0, c + 1] = n[i]
        b[1, c] = dndy[i]  # γyz = w,y + θy
        b[1, c + 2] = n[i]
    return b


def element_stiffness_unit(nu: float, t: float, dx: float, dy: float) -> np.ndarray:
    """12×12 element stiffness at **E = 1** (SIMP scales it by ``E_eff``).

    Bending integrated with 2×2 Gauss (full); shear with 1×1 (reduced) to keep
    the thin plate from locking.
    """
    area_j = (dx / 2.0) * (dy / 2.0)  # constant Jacobian determinant for a rectangle

    d0 = t**3 / (12.0 * (1.0 - nu**2))
    db = d0 * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]])
    gs = SHEAR_CORRECTION * (1.0 / (2.0 * (1.0 + nu))) * t  # k·G·t at E=1
    ds = gs * np.eye(2)

    ke = np.zeros((12, 12))
    for xi, eta in _GAUSS_2X2:  # full 2×2 on bending
        _, dndx, dndy = _q4_derivs(xi, eta, dx, dy)
        bb = _bending_B(dndx, dndy)
        ke += bb.T @ db @ bb * area_j
    n0, dndx0, dndy0 = _q4_derivs(0.0, 0.0, dx, dy)  # reduced 1×1 on shear
    bs = _shear_B(n0, dndx0, dndy0)
    ke += bs.T @ ds @ bs * (4.0 * area_j)
    return ke


@dataclass(frozen=True)
class PlateMesh:
    """Structured rectangular Q4 plate mesh (``nelx × nely`` elements)."""

    nelx: int
    nely: int
    dx: float
    dy: float

    @property
    def n_nodes(self) -> int:
        return (self.nelx + 1) * (self.nely + 1)

    @property
    def n_dofs(self) -> int:
        return DOF_PER_NODE * self.n_nodes

    def node_id(self, ix: int, iy: int) -> int:
        return iy * (self.nelx + 1) + ix

    def element_nodes(self, ex: int, ey: int) -> tuple[int, int, int, int]:
        """CCW node ids of element (ex, ey)."""
        return (
            self.node_id(ex, ey),
            self.node_id(ex + 1, ey),
            self.node_id(ex + 1, ey + 1),
            self.node_id(ex, ey + 1),
        )

    def element_dofs(self, ex: int, ey: int) -> np.ndarray:
        """The 12 global DOF indices of element (ex, ey)."""
        nodes = self.element_nodes(ex, ey)
        return np.array([DOF_PER_NODE * n + k for n in nodes for k in range(DOF_PER_NODE)])


def assemble_global_stiffness(
    mesh: PlateMesh, e_elem: np.ndarray, nu: float, t: float
) -> csc_matrix:
    """Assemble the global stiffness with per-element modulus ``e_elem`` (SIMP).

    ``e_elem`` is a ``(nely, nelx)`` array of effective moduli; ``Ke = E · Ke_unit``.
    """
    if e_elem.shape != (mesh.nely, mesh.nelx):
        raise ValueError(f"e_elem must be ({mesh.nely}, {mesh.nelx}); got {e_elem.shape}")
    ke_unit = element_stiffness_unit(nu, t, mesh.dx, mesh.dy)
    n_e = mesh.nelx * mesh.nely
    rows = np.empty(n_e * 144, dtype=np.int64)
    cols = np.empty(n_e * 144, dtype=np.int64)
    vals = np.empty(n_e * 144, dtype=float)
    p = 0
    for ey in range(mesh.nely):
        for ex in range(mesh.nelx):
            edofs = mesh.element_dofs(ex, ey)
            ke = e_elem[ey, ex] * ke_unit
            rr, cc = np.meshgrid(edofs, edofs, indexing="ij")
            rows[p : p + 144] = rr.ravel()
            cols[p : p + 144] = cc.ravel()
            vals[p : p + 144] = ke.ravel()
            p += 144
    k = csc_matrix((vals, (rows, cols)), shape=(mesh.n_dofs, mesh.n_dofs))
    return (k + k.T) * 0.5  # symmetrize against assembly round-off


def solve_displacements(k: csc_matrix, f: np.ndarray, fixed_dofs: np.ndarray) -> np.ndarray:
    """Solve ``K u = F`` with Dirichlet ``fixed_dofs`` (u = 0 there)."""
    n = k.shape[0]
    free = np.setdiff1d(np.arange(n), fixed_dofs)
    u = np.zeros(n)
    if free.size == 0:
        return u
    kff = k[free][:, free]
    u[free] = spsolve(kff.tocsc(), f[free])
    return u
