"""Linear-elastic FEA of the blade solid with per-element anisotropic PETG stiffness.

Assembles and solves ``K u = f`` on the tet mesh using scikit-fem, with each element
carrying its own rotated transversely-isotropic stiffness (the weak build axis follows
the blade's local width). Precomputes the strain-displacement operator and per-element
base stiffness in a reusable :class:`FeaModel` so the SIMP loop can re-assemble
``K(ρ) = Σ ρ_e^p K_e`` cheaply each iteration and read per-element strain energies (the
compliance sensitivity) and stresses (the yield / weak-Z constraint).

Requires scikit-fem (the ``[topopt]`` extra); the test module gates on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from skfem import Basis, BilinearForm, ElementTetP1, ElementVector, MeshTet, condense, solve
from skfem.helpers import sym_grad

from fanopt.topopt.material import rotate_stiffness

__all__ = [
    "FeaModel",
    "build_fea_model",
    "assemble_global_stiffness",
    "solve_displacements",
    "element_strains",
    "element_stresses",
    "element_strain_energies",
    "compliance",
]


@BilinearForm
def _anisotropic_stiffness(u, v, w):
    """Weak form ``∫ ε(u)ᵀ C ε(v)`` with per-element Voigt ``C`` from the field ``w['C']``."""
    eu, ev = sym_grad(u), sym_grad(v)
    su = (eu[0, 0], eu[1, 1], eu[2, 2], 2 * eu[1, 2], 2 * eu[0, 2], 2 * eu[0, 1])
    sv = (ev[0, 0], ev[1, 1], ev[2, 2], 2 * ev[1, 2], 2 * ev[0, 2], 2 * ev[0, 1])
    c = w["C"]
    out = 0.0
    for i in range(6):
        for j in range(6):
            out = out + c[i, j] * su[i] * sv[j]
    return out


@dataclass(frozen=True)
class FeaModel:
    """Reusable FEA operators for one mesh: basis, per-element B, volumes, base stiffness."""

    basis: Basis
    tets: np.ndarray  # (m, 4) int
    b_matrices: np.ndarray  # (m, 6, 12) strain-displacement
    volumes: np.ndarray  # (m,)
    base_stiffness: np.ndarray  # (m, 6, 6) per-element rotated C0
    elem_dofs: np.ndarray  # (m, 12) global dof indices, node-major [n0xyz, n1xyz, ...]
    support_dofs: np.ndarray  # (s,) clamped dof indices
    n_dofs: int
    meta: dict = field(default_factory=dict)


def _shape_gradients(nodes: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P1 tet shape-function gradients ``(m,4,3)`` and volumes ``(m,)`` (constant per tet)."""
    p = nodes[tets]  # (m, 4, 3)
    edges = p[:, 1:, :] - p[:, :1, :]  # (m, 3, 3): rows are p1−p0, p2−p0, p3−p0
    inv = np.linalg.inv(edges)
    grad123 = np.transpose(inv, (0, 2, 1))  # rows = ∇λ1, ∇λ2, ∇λ3
    grad0 = -grad123.sum(axis=1, keepdims=True)
    grads = np.concatenate([grad0, grad123], axis=1)  # (m, 4, 3)
    volumes = np.abs(np.linalg.det(edges)) / 6.0
    return grads, volumes


def _b_matrices(grads: np.ndarray) -> np.ndarray:
    """Strain-displacement ``B`` ``(m,6,12)`` from shape gradients (engineering shear)."""
    m = grads.shape[0]
    b = np.zeros((m, 6, 12))
    for a in range(4):
        gx, gy, gz = grads[:, a, 0], grads[:, a, 1], grads[:, a, 2]
        c0, c1, c2 = 3 * a, 3 * a + 1, 3 * a + 2
        b[:, 0, c0] = gx
        b[:, 1, c1] = gy
        b[:, 2, c2] = gz
        b[:, 3, c1] = gz
        b[:, 3, c2] = gy
        b[:, 4, c0] = gz
        b[:, 4, c2] = gx
        b[:, 5, c0] = gy
        b[:, 5, c1] = gx
    return b


def build_fea_model(
    nodes: np.ndarray,
    tets: np.ndarray,
    base_stiffness: np.ndarray,
    support_node_ids: np.ndarray,
    material_frames: np.ndarray | None = None,
) -> FeaModel:
    """Assemble reusable FEA operators; rotate the base stiffness into each element's frame.

    ``base_stiffness`` is the 6×6 material-frame stiffness (from
    :func:`fanopt.topopt.material.transversely_isotropic_stiffness`). With
    ``material_frames`` (per element, from :mod:`fanopt.topopt.design_domain`) each element
    gets ``rotate_stiffness(C, R_e)``; without them the same C is shared (global frame).
    """
    mesh = MeshTet(nodes.T, tets.T)
    basis = Basis(mesh, ElementVector(ElementTetP1()))
    grads, volumes = _shape_gradients(nodes, tets)
    b = _b_matrices(grads)
    m = len(tets)
    if material_frames is None:
        base = np.broadcast_to(base_stiffness, (m, 6, 6)).copy()
    else:
        base = np.stack([rotate_stiffness(base_stiffness, r) for r in material_frames])
    nd = basis.nodal_dofs  # (3, n_nodes)
    elem_dofs = np.transpose(nd[:, tets], (1, 2, 0)).reshape(m, 12)  # node-major [n0xyz, ...]
    support_dofs = nd[:, np.asarray(support_node_ids, dtype=int)].flatten()
    return FeaModel(
        basis=basis,
        tets=np.asarray(tets),
        b_matrices=b,
        volumes=volumes,
        base_stiffness=base,
        elem_dofs=elem_dofs,
        support_dofs=support_dofs,
        n_dofs=basis.N,
        meta={"n_nodes": float(len(nodes)), "n_tets": float(m)},
    )


def _element_stiffness_field(
    model: FeaModel, density: np.ndarray | None, penal: float
) -> np.ndarray:
    """Per-element SIMP-scaled stiffness as the ``(6,6,m,1)`` field the form expects."""
    c = model.base_stiffness
    if density is not None:
        c = c * (np.asarray(density) ** penal)[:, None, None]
    return np.moveaxis(c, 0, 2)[..., None]


def assemble_global_stiffness(
    model: FeaModel, density: np.ndarray | None = None, penal: float = 3.0
):
    """Global stiffness ``K``; with ``density`` applies SIMP scaling ``ρ_e^penal``."""
    return _anisotropic_stiffness.assemble(model.basis, C=_element_stiffness_field(model, density, penal))


def _force_vector(model: FeaModel, node_forces: np.ndarray) -> np.ndarray:
    """Scatter nodal forces ``(n_nodes,3)`` into a global load vector."""
    f = np.zeros(model.n_dofs)
    nd = model.basis.nodal_dofs
    for c in range(3):
        f[nd[c]] = node_forces[:, c]
    return f


def solve_displacements(model: FeaModel, node_forces: np.ndarray, k=None) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``K u = f`` with the hub clamped; return ``(u, f)``.

    ``k`` may be a prebuilt stiffness (e.g. the SIMP-scaled one) to avoid re-assembly;
    otherwise the full-density stiffness is assembled.
    """
    stiffness = assemble_global_stiffness(model) if k is None else k
    f = _force_vector(model, node_forces)
    u = solve(*condense(stiffness, f, D=model.support_dofs))
    return u, f


def element_strains(model: FeaModel, u: np.ndarray) -> np.ndarray:
    """Constant per-element Voigt strain ``ε = B u_e`` — shape ``(m, 6)``."""
    u_e = u[model.elem_dofs]  # (m, 12)
    return np.einsum("eij,ej->ei", model.b_matrices, u_e)


def element_stresses(model: FeaModel, u: np.ndarray) -> np.ndarray:
    """Per-element Voigt stress ``σ = C0 ε`` in the global frame — shape ``(m, 6)``.

    Uses the full-density material stiffness (physical stress in solid material), the
    input to the yield / weak-Z constraint on the final design.
    """
    eps = element_strains(model, u)
    return np.einsum("eij,ej->ei", model.base_stiffness, eps)


def element_strain_energies(model: FeaModel, u: np.ndarray) -> np.ndarray:
    """Per-element ``V_e · εᵀ C0 ε`` (= ``u_eᵀ K_e0 u_e``) — the compliance sensitivity base."""
    eps = element_strains(model, u)
    quad = np.einsum("ei,eij,ej->e", eps, model.base_stiffness, eps)
    return model.volumes * quad


def compliance(u: np.ndarray, f: np.ndarray) -> float:
    """Structural compliance ``fᵀu`` (lower = stiffer)."""
    return float(u @ f)
