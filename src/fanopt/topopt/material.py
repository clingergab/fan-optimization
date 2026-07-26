"""Orthotropic (transversely-isotropic) PETG constitutive model for structural TO.

FDM PETG is **not** isotropic: the printed layer (XY) plane is stiff, the build
direction (Z) is weak because it relies on inter-layer bonding. For this blade the
build/Z axis is aligned with the blade **width** (printed on its side), so the two
strong in-plane directions are base-to-tip (radial) and through-thickness. We model
that as **transversely isotropic**: the layer plane is a plane of isotropy, the build
axis is the weak special axis.

Voigt convention (engineering shear, ``γ = 2ε``), tensor order
``[11, 22, 33, 23, 13, 12]``: ``stress6 = C @ strain6`` with
``strain6 = [ε11, ε22, ε33, γ23, γ13, γ12]``. In the **material principal frame** the
weak build axis is axis 3 (index 2), so ``σ33`` is the inter-layer normal stress and
``σ23 / σ13`` are the inter-layer (delamination) shears — the failure modes FDM parts
actually die from. Stresses fed to the failure helpers must be expressed in this
material frame (rotate the FEA/global stress with :func:`rotate_stiffness`'s companion
rotation first).

Pure numpy — no FEA dependency; the elastic constants are injected as arguments so the
model is testable with synthetic values while the two genuinely-independent PETG
constants (inter-layer shear ``G_z`` and through-thickness Poisson ``ν_zp``) are sourced
from cited literature and land in :mod:`fanopt.geometry.schema`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "orthotropic_stiffness",
    "transversely_isotropic_stiffness",
    "isotropic_stiffness",
    "in_plane_shear_modulus",
    "stress_bond_matrix",
    "rotate_stiffness",
    "weak_axis_normal_stress",
    "interlaminar_shear",
    "failure_index",
]

# Voigt slot of the material-frame build (weak) axis normal stress σ33.
_WEAK_NORMAL_SLOT = 2
# Voigt slots of the two inter-layer (delamination) shear stresses σ23, σ13.
_INTERLAMINAR_SHEAR_SLOTS = (3, 4)


def in_plane_shear_modulus(e_p: float, nu_p: float) -> float:
    """In-plane shear modulus ``G = E / (2(1+ν))`` — derived, not an independent input.

    The layer plane is isotropic, so its shear modulus follows from the in-plane Young's
    modulus and Poisson ratio. Only the *inter-layer* shear ``G_z`` is independent.
    """
    return e_p / (2.0 * (1.0 + nu_p))


def orthotropic_stiffness(
    e1: float,
    e2: float,
    e3: float,
    nu12: float,
    nu13: float,
    nu23: float,
    g12: float,
    g13: float,
    g23: float,
) -> np.ndarray:
    """6×6 stiffness ``C`` from the 9 orthotropic engineering constants (Voigt frame).

    Builds the symmetric compliance ``S`` (``ν_ij/E_i = ν_ji/E_j`` enforced by
    construction) and inverts it. Raises if the result is not positive-definite — a
    thermodynamically inadmissible constant set (e.g. a Poisson ratio too large for the
    modulus ratio) is a caller error, not something to silently return.
    """
    s = np.zeros((6, 6), dtype=float)
    s[0, 0], s[1, 1], s[2, 2] = 1.0 / e1, 1.0 / e2, 1.0 / e3
    s[0, 1] = s[1, 0] = -nu12 / e1
    s[0, 2] = s[2, 0] = -nu13 / e1
    s[1, 2] = s[2, 1] = -nu23 / e2
    s[3, 3], s[4, 4], s[5, 5] = 1.0 / g23, 1.0 / g13, 1.0 / g12
    c = np.linalg.inv(s)
    if not np.all(np.linalg.eigvalsh(c) > 0.0):
        raise ValueError("inadmissible orthotropic constants: stiffness not positive-definite")
    return c


def transversely_isotropic_stiffness(
    e_p: float, e_z: float, nu_p: float, nu_zp: float, g_z: float
) -> np.ndarray:
    """6×6 stiffness for a transversely-isotropic material, weak axis = material axis 3.

    ``e_p, nu_p`` describe the isotropic layer plane (axes 1–2); ``e_z`` is the weak
    build-direction modulus (axis 3); ``nu_zp`` couples build and in-plane; ``g_z`` is the
    inter-layer shear. In-plane shear ``G12`` is derived (:func:`in_plane_shear_modulus`).
    """
    g_p = in_plane_shear_modulus(e_p, nu_p)
    return orthotropic_stiffness(
        e1=e_p, e2=e_p, e3=e_z, nu12=nu_p, nu13=nu_zp, nu23=nu_zp, g12=g_p, g13=g_z, g23=g_z
    )


def isotropic_stiffness(e: float, nu: float) -> np.ndarray:
    """6×6 isotropic stiffness — the ``E_p=E_z``, ``ν_p=ν_zp`` limit (for tests/fallback)."""
    g = in_plane_shear_modulus(e, nu)
    return orthotropic_stiffness(e, e, e, nu, nu, nu, g, g, g)


def stress_bond_matrix(rotation: np.ndarray) -> np.ndarray:
    """6×6 Bond stress-transformation ``M`` for a 3×3 rotation (rows = new axes in old).

    Maps material-frame stress to the rotated frame: ``σ' = M @ σ`` (Voigt order
    ``[11,22,33,23,13,12]``). Paired with :func:`rotate_stiffness` (``C' = M C Mᵀ``).
    """
    r = np.asarray(rotation, dtype=float)
    if r.shape != (3, 3):
        raise ValueError(f"rotation must be 3×3, got {r.shape}")
    (l1, m1, n1), (l2, m2, n2), (l3, m3, n3) = r
    return np.array(
        [
            [l1*l1, m1*m1, n1*n1, 2*m1*n1, 2*n1*l1, 2*l1*m1],
            [l2*l2, m2*m2, n2*n2, 2*m2*n2, 2*n2*l2, 2*l2*m2],
            [l3*l3, m3*m3, n3*n3, 2*m3*n3, 2*n3*l3, 2*l3*m3],
            [l2*l3, m2*m3, n2*n3, m2*n3+m3*n2, n2*l3+n3*l2, l2*m3+l3*m2],
            [l3*l1, m3*m1, n3*n1, m3*n1+m1*n3, n3*l1+n1*l3, l3*m1+l1*m3],
            [l1*l2, m1*m2, n1*n2, m1*n2+m2*n1, n1*l2+n2*l1, l1*m2+l2*m1],
        ],
        dtype=float,
    )


def rotate_stiffness(c: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Rotate a 6×6 stiffness into a new frame: ``C' = M C Mᵀ`` (``M`` = Bond matrix).

    Used to express the material-frame stiffness in the mesh/global frame per element
    (the weak build axis follows the blade's local width direction on the curved surface).
    """
    m = stress_bond_matrix(rotation)
    return m @ np.asarray(c, dtype=float) @ m.T


def weak_axis_normal_stress(stress6_material: np.ndarray) -> float:
    """Inter-layer normal stress ``σ33`` (build-direction, material frame).

    Tensile ``σ33`` pulls printed layers apart — the primary FDM failure driver. Input
    must already be in the material principal frame.
    """
    return float(np.asarray(stress6_material, dtype=float)[_WEAK_NORMAL_SLOT])


def interlaminar_shear(stress6_material: np.ndarray) -> float:
    """Resultant inter-layer shear ``√(σ23² + σ13²)`` (material frame) — delamination shear."""
    s = np.asarray(stress6_material, dtype=float)
    a, b = _INTERLAMINAR_SHEAR_SLOTS
    return float(np.hypot(s[a], s[b]))


def failure_index(
    stress6_material: np.ndarray,
    sigma_y_in_plane: float,
    sigma_y_weak: float,
    *,
    tau_interlaminar: float | None = None,
) -> float:
    """Max-stress failure index (≥ 1 ⇒ yields); anisotropic, material-frame stress.

    In-plane normal stresses (``σ11, σ22``) are checked against ``sigma_y_in_plane``, the
    weak build-direction normal (``σ33``) against the lower ``sigma_y_weak``. If an
    inter-layer shear strength ``tau_interlaminar`` is supplied, the delamination shear is
    included; the in-plane shear ``σ12`` is intentionally omitted (no in-plane shear-
    strength datum — a documented gap, not a silent assumption).
    """
    s = np.asarray(stress6_material, dtype=float)
    terms = [
        abs(s[0]) / sigma_y_in_plane,
        abs(s[1]) / sigma_y_in_plane,
        abs(s[_WEAK_NORMAL_SLOT]) / sigma_y_weak,
    ]
    if tau_interlaminar is not None:
        terms.append(interlaminar_shear(s) / tau_interlaminar)
    return max(terms)
