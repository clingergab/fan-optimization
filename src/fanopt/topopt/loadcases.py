"""The four locked structural load cases as FEA nodal-force vectors.

Productive stroke, return stroke, inertial, click engagement — each reduced to a nodal
force field ``(n_nodes, 3)`` the linear-elastic solver consumes. Aero strokes come from
the mapped CFD surface pressure (:mod:`fanopt.topopt.pressure_map`); the inertial case is
the body force at the stroke turning point (angular acceleration ``α_max`` about the
stroke axis, where the pitch velocity is momentarily zero); the click case is the worst-
case engagement force at the click region.

Kinematics locks are honored: the stroke/inertial axis follows ``PITCHING_OMEGA_VEC``
(**negative y** — C11 sign lock), and a wrist torque converts to tip force through
``L_WRIST_TO_TIP_M = 0.25 m`` (H8 lever-arm lock, NOT ``L_blade``).

Pure numpy — consumes only lower-level schema constants; no FEA dependency, so every case
is testable on a synthetic mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fanopt.geometry.schema import (
    ALPHA_MAX_RAD_PER_S2,
    L_WRIST_TO_TIP_M,
    PITCHING_OMEGA_VEC,
    RHO_PETG_KG_PER_M3,
)

__all__ = [
    "LoadCase",
    "default_angular_acceleration",
    "tip_force_from_wrist_torque",
    "pressure_surface_load",
    "productive_stroke_load",
    "return_stroke_load",
    "inertial_body_load",
    "click_load",
    "assemble_load_cases",
]


@dataclass(frozen=True)
class LoadCase:
    """One named load case: its nodal force field and a weight for multi-load aggregation."""

    name: str
    node_forces: np.ndarray  # (n_nodes, 3)
    weight: float = 1.0


def default_angular_acceleration() -> np.ndarray:
    """``α_max`` along the stroke axis — ``PITCHING_OMEGA_VEC`` direction (C11 negative-y)."""
    omega = np.asarray(PITCHING_OMEGA_VEC, dtype=float)
    return ALPHA_MAX_RAD_PER_S2 * omega / np.linalg.norm(omega)


def tip_force_from_wrist_torque(torque_nm: float) -> float:
    """Convert a wrist torque to an equivalent tip force via the H8 lever arm (0.25 m)."""
    return torque_nm / L_WRIST_TO_TIP_M


def pressure_surface_load(
    n_nodes: int,
    facet_node_ids: np.ndarray,
    facet_areas: np.ndarray,
    facet_normals: np.ndarray,
    facet_pressure: np.ndarray,
) -> np.ndarray:
    """Consistent nodal forces from a pressure field on triangular facets.

    Facet force is ``−p · n · A`` (pressure pushes against the outward normal ``n``),
    lumped equally to the facet's three nodes. ``facet_node_ids`` is ``(m, 3)`` int.
    """
    facet_force = (-np.asarray(facet_pressure)[:, None] * np.asarray(facet_normals)) * np.asarray(
        facet_areas
    )[:, None]
    out = np.zeros((n_nodes, 3), dtype=float)
    ids = np.asarray(facet_node_ids, dtype=int).reshape(-1)
    np.add.at(out, ids, np.repeat(facet_force / 3.0, 3, axis=0))
    return out


def productive_stroke_load(
    n_nodes: int,
    facet_node_ids: np.ndarray,
    facet_areas: np.ndarray,
    facet_normals: np.ndarray,
    facet_pressure: np.ndarray,
) -> np.ndarray:
    """Productive-stroke aero load — the mapped CFD pressure as consistent nodal forces."""
    return pressure_surface_load(n_nodes, facet_node_ids, facet_areas, facet_normals, facet_pressure)


def return_stroke_load(
    n_nodes: int,
    facet_node_ids: np.ndarray,
    facet_areas: np.ndarray,
    facet_normals: np.ndarray,
    facet_pressure: np.ndarray,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Return-stroke aero load — the productive pressure reversed (opposite face loaded).

    ``scale`` lets the caller down-weight the return stroke if the harvested return
    pressure is milder than the productive one; default is an equal-magnitude reversal.
    """
    return pressure_surface_load(
        n_nodes, facet_node_ids, facet_areas, facet_normals, -scale * np.asarray(facet_pressure)
    )


def inertial_body_load(
    n_nodes: int,
    element_node_ids: np.ndarray,
    element_volumes: np.ndarray,
    element_centroids: np.ndarray,
    *,
    element_density: np.ndarray | None = None,
    alpha_vec: np.ndarray | None = None,
    rho: float = RHO_PETG_KG_PER_M3,
) -> np.ndarray:
    """Body force ``ρ · (α × r)`` at the stroke turning point, lumped to element nodes.

    ``element_density`` is the SIMP density (∈ [0,1]) scaling each element's mass — pass it
    so the inertial load tracks the evolving design (solid skin + partly-void interior);
    ``None`` treats the blade as fully solid. ``element_node_ids`` is ``(e, 4)`` int.
    """
    alpha = default_angular_acceleration() if alpha_vec is None else np.asarray(alpha_vec, dtype=float)
    rho_e = np.ones(len(element_volumes)) if element_density is None else np.asarray(element_density)
    accel = np.cross(alpha, np.asarray(element_centroids, dtype=float))  # a = α × r
    elem_force = (rho * rho_e * np.asarray(element_volumes))[:, None] * accel
    out = np.zeros((n_nodes, 3), dtype=float)
    ids = np.asarray(element_node_ids, dtype=int).reshape(-1)
    np.add.at(out, ids, np.repeat(elem_force / 4.0, 4, axis=0))
    return out


def click_load(n_nodes: int, click_node_ids: np.ndarray, force_vec: np.ndarray) -> np.ndarray:
    """Total engagement force ``force_vec`` (N) spread equally over the click-region nodes."""
    out = np.zeros((n_nodes, 3), dtype=float)
    ids = np.asarray(click_node_ids, dtype=int)
    if ids.size:
        out[ids] += np.asarray(force_vec, dtype=float) / ids.size
    return out


def assemble_load_cases(
    n_nodes: int,
    facet_node_ids: np.ndarray,
    facet_areas: np.ndarray,
    facet_normals: np.ndarray,
    facet_pressure: np.ndarray,
    element_node_ids: np.ndarray,
    element_volumes: np.ndarray,
    element_centroids: np.ndarray,
    click_node_ids: np.ndarray,
    click_force_vec: np.ndarray,
    *,
    element_density: np.ndarray | None = None,
    return_scale: float = 1.0,
) -> list[LoadCase]:
    """Build all four locked load cases as :class:`LoadCase` records (order preserved)."""
    return [
        LoadCase(
            "productive_stroke",
            productive_stroke_load(n_nodes, facet_node_ids, facet_areas, facet_normals, facet_pressure),
        ),
        LoadCase(
            "return_stroke",
            return_stroke_load(
                n_nodes, facet_node_ids, facet_areas, facet_normals, facet_pressure, scale=return_scale
            ),
        ),
        LoadCase(
            "inertial",
            inertial_body_load(
                n_nodes,
                element_node_ids,
                element_volumes,
                element_centroids,
                element_density=element_density,
            ),
        ),
        LoadCase("click_engagement", click_load(n_nodes, click_node_ids, click_force_vec)),
    ]
