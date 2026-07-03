"""SIMP material interpolation, density filter, and OC update for the rib TO
(report-final.md §3.1.1 / §3.1.4).

- ``simp_modulus`` — ``E_eff(ρ) = E_min + ρ^p (E0 − E_min)`` (§3.1.1).
- ``build_density_filter`` — row-normalized linear (cone) filter of radius
  ``r_min`` so the physical density ``ρ̃ = Hs · x`` is mesh-independent and
  checkerboard-free (§3.1.4).
- ``oc_update`` — optimality-criteria design update with a bisection on the
  Lagrange multiplier to hit the volume target, respecting the ``active`` /
  ``preserved`` masks (§9.2).

All arrays are ``(nely, nelx)`` unless noted; the filter works on the flattened
grid (C order).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

__all__ = [
    "simp_modulus",
    "build_density_filter",
    "apply_filter",
    "oc_update",
]


def simp_modulus(rho: np.ndarray, penal: float, e0: float, emin: float) -> np.ndarray:
    """``E_eff = E_min + ρ^p (E0 − E_min)`` (elementwise)."""
    return emin + rho**penal * (e0 - emin)


def build_density_filter(nely: int, nelx: int, rmin: float) -> csr_matrix:
    """Row-normalized cone filter ``Hs`` (``(n_e, n_e)`` sparse).

    ``ρ̃ = Hs · x``. Weight between elements = ``max(0, r_min − dist)`` on the
    element-center grid (unit element spacing), then each row normalized to sum
    1 so a uniform field is unchanged.
    """
    if rmin <= 0:
        raise ValueError(f"rmin must be > 0; got {rmin}")
    n_e = nelx * nely
    ceil_r = int(np.ceil(rmin)) - 1
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for ey in range(nely):
        for ex in range(nelx):
            e = ey * nelx + ex
            for jy in range(max(ey - ceil_r, 0), min(ey + ceil_r + 1, nely)):
                for jx in range(max(ex - ceil_r, 0), min(ex + ceil_r + 1, nelx)):
                    w = rmin - np.hypot(ex - jx, ey - jy)
                    if w > 0:
                        rows.append(e)
                        cols.append(jy * nelx + jx)
                        vals.append(w)
    h = csr_matrix((vals, (rows, cols)), shape=(n_e, n_e))
    row_sums = np.asarray(h.sum(axis=1)).ravel()
    inv = csr_matrix((1.0 / row_sums, (range(n_e), range(n_e))), shape=(n_e, n_e))
    return (inv @ h).tocsr()


def apply_filter(hs: csr_matrix, x: np.ndarray) -> np.ndarray:
    """Physical density ``ρ̃ = Hs · x`` (x is ``(nely, nelx)``; result same shape)."""
    shape = x.shape
    return (hs @ x.ravel()).reshape(shape)


def oc_update(
    x: np.ndarray,
    dc: np.ndarray,
    dv: np.ndarray,
    hs: csr_matrix,
    *,
    volfrac: float,
    free: np.ndarray,
    active: np.ndarray,
    move: float = 0.2,
    eta: float = 0.5,
) -> np.ndarray:
    """One optimality-criteria step on the design variables.

    ``dc``/``dv`` are ∂C/∂x and ∂V/∂x (already filter-chained). ``free`` is the
    boolean mask of updatable elements (active AND not preserved); ``active`` is
    the full design domain. Non-free elements keep their current value
    (preserved = 1, inactive = 0). The bisection finds the multiplier so the
    filtered volume ``Σ ρ̃`` over the **whole active domain** hits
    ``volfrac · n_active`` — the free elements absorb the preserved (ρ=1) mass.
    """
    xf = x[free]
    dcf = dc[free]
    dvf = dv[free]
    # OC scaling uses −dc/dv (both should drive material where compliance-sensitive).
    b = np.maximum(-dcf / np.maximum(dvf, 1e-12), 0.0)
    target = volfrac * active.sum()

    l1, l2 = 1e-9, 1e9
    x_new = x.copy()
    while (l2 - l1) / (l1 + l2) > 1e-6:
        lmid = 0.5 * (l1 + l2)
        xe = xf * (b / lmid) ** eta
        xe = np.clip(xe, np.maximum(xf - move, 0.0), np.minimum(xf + move, 1.0))
        x_new[free] = xe
        vol = apply_filter(hs, x_new)[active].sum()  # whole-domain physical volume
        if vol > target:
            l1 = lmid
        else:
            l2 = lmid
    return x_new
