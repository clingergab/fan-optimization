"""Per-design 3D SIMP topology optimization of a solid blade.

This is the V1 structural stage for the aero-first solid blade (ADR-0003): TO runs on
**each** design's own 3D solid — the pre-redesign 2D representative-rib path
(:mod:`fanopt.topopt.solver`) does not apply here because every top design has a distinct
meridian/thickness and therefore a distinct structure. TO is free to remove material from
the rib and the panel interior/sides, but the **aero skin** (a frozen shell of fixed
thickness on the top/bottom panel faces) is held at ``ρ ≡ 1`` so the air-pushing surfaces
are never punched through — holes leak airflow.

Pipeline per design::

    BladeParams
      → build_blade_fea_mesh   (tet mesh + aero-skin / hub / click tags)
      → build_blade_to_problem (frozen skin, carvable design domain, FEA operators,
                                unstructured density filter, the four locked load cases)
      → run_blade_topology_optimization  (SIMP OC loop, multi-load compliance min)
      → BladeTOResult          (carved density field + volume removed, u_tip, max σ_VM)

The four load cases (:mod:`fanopt.topopt.loadcases`) are the locked productive / return /
inertial / click set; the inertial body force tracks the evolving density each iteration.
Compliance sensitivity is the standard SIMP ``dC/dρ = −p ρ^(p−1) uₑᵀKₑ⁰uₑ`` read from
:func:`fanopt.topopt.fea.element_strain_energies`.

Requires the ``[topopt]`` extra (gmsh + CadQuery for the mesh, scikit-fem for the FEA);
the test module gates on them.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from fanopt.geometry.blade import BladeParams
from fanopt.geometry.schema import (
    E_PETG_XY_PA,
    E_PETG_Z_PA,
    HUB_RADIUS_M,
    NU_PETG,
    RHO_PETG_KG_PER_M3,
    SIGMA_Y_PETG_Z_PA,
)
from fanopt.topopt.blade_fea_mesh import (
    BladeFeaMeshResult,
    FeaMeshParams,
    boundary_facets,
    build_blade_fea_mesh,
    classify_aero_skin,
    click_region_facets,
    facet_geometry,
)
from fanopt.topopt.design_domain import (
    DesignDomain,
    build_density_filter_unstructured,
    classify_frozen_skin,
    revolution_material_frames,
    tet_centroids_and_volumes,
)
from fanopt.topopt.fea import (
    assemble_global_stiffness,
    build_fea_model,
    compliance,
    element_strain_energies,
    element_stresses,
    solve_displacements_multi,
)
from fanopt.topopt.loadcases import assemble_load_cases
from fanopt.topopt.loads import DEFAULT_AERO_PRESSURE_PA, EMIN_FACTOR, SIMP_PENALTY
from fanopt.topopt.material import in_plane_shear_modulus, transversely_isotropic_stiffness
from fanopt.topopt.simp import oc_update_unstructured

__all__ = [
    "BladeTOProblem",
    "BladeTOResult",
    "von_mises",
    "build_blade_to_problem",
    "run_blade_topology_optimization",
    "topology_optimize_blade",
    "topology_optimize_blade_screened",
    "run_blade_to_batch",
    "DEFAULT_SKIN_THICKNESS_M",
    "DEFAULT_VOLFRAC_LADDER",
    "DEFAULT_U_TIP_LIMIT_M",
    "DEFAULT_STRESS_FOS",
    "DEFAULT_RMIN_M",
    "DEFAULT_CLICK_FORCE_N",
    "DEFAULT_VOLFRAC",
]

# Frozen aero-skin shell thickness — the air-pushing faces held solid. Sized to ~0.8× the
# production mesh (1.5 mm) so at least the surface element layer freezes; a thin panel is
# ≤ 2 layers so it stays fully solid, while a thick rib exposes a carvable core.
DEFAULT_SKIN_THICKNESS_M: float = 0.0012
# Density-filter radius (minimum carved feature size); ≈ one mesh edge at the 1.5 mm mesh.
DEFAULT_RMIN_M: float = 0.0020
# Canonical click-engagement force (α·m·r·N ≈ 0.275 N, report-final §Phase 2), applied
# tangentially over the tip click band — footprint is the documented radius-band approx.
DEFAULT_CLICK_FORCE_N: float = 0.275
# Design-region target volume fraction (SIMP retains this share of the carvable volume).
DEFAULT_VOLFRAC: float = 0.40
# Screened mode: volume fractions tried per design, most-aggressive (lowest) first. The search
# accepts the first (lowest → most material removed) that passes the structural screen.
DEFAULT_VOLFRAC_LADDER: tuple[float, ...] = (0.25, 0.35, 0.45, 0.60)
# Rigid-blade tip-deflection limit (report-final §3.1) — the stiffness half of the screen.
DEFAULT_U_TIP_LIMIT_M: float = 1.0e-3
# Safety factor applied to the PETG weak-axis yield for the stress half of the screen.
DEFAULT_STRESS_FOS: float = 2.0


@dataclass(frozen=True)
class BladeTOProblem:
    """A fully-assembled per-design 3D SIMP problem, ready for the OC loop.

    Bundles the reusable FEA operators (:class:`fanopt.topopt.fea.FeaModel`), the design
    domain (frozen aero skin vs carvable), the unstructured density filter, and the raw
    arrays the four load cases are (re)built from each iteration (the inertial case tracks
    the current density).
    """

    fea_model: object  # FeaModel
    domain: DesignDomain
    filter_matrix: object  # scipy csr_matrix
    n_nodes: int
    aero_facet_node_ids: np.ndarray  # (k, 3) int — productive/return load face
    aero_facet_areas: np.ndarray  # (k,)
    aero_facet_normals: np.ndarray  # (k, 3) outward
    aero_pressure_pa: np.ndarray  # (k,)
    element_node_ids: np.ndarray  # (m, 4) int — the tets, for the inertial body load
    click_node_ids: np.ndarray  # (c,) int
    click_force_vec: np.ndarray  # (3,)
    return_scale: float
    volfrac: float
    penal: float
    rho_min: float
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BladeTOResult:
    """Carved density field + structural readouts for one blade's TO."""

    density: np.ndarray  # (m,) physical element density ρ̃ (frozen skin = 1)
    compliance_history: tuple[float, ...]
    design_volume_fraction: float  # retained share of the CARVABLE region
    volume_removed_frac: float  # material removed vs the full solid (skin + interior)
    mass_kg: float  # post-TO single-blade mass (ρ_PETG · retained volume)
    u_tip_max_m: float  # worst nodal deflection over the four load cases
    max_von_mises_pa: float  # worst element σ_VM over the four load cases (all elements)
    converged: bool
    iterations: int
    meta: dict[str, float] = field(default_factory=dict)
    element_centroids: np.ndarray | None = None  # (m, 3) — saved so the render aligns to ρ̃


def von_mises(stress6: np.ndarray) -> np.ndarray:
    """Per-element von Mises stress from Voigt stress ``(m, 6)`` ``[xx,yy,zz,yz,xz,xy]``."""
    s = np.asarray(stress6, dtype=float)
    sxx, syy, szz, syz, sxz, sxy = s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5]
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (syz**2 + sxz**2 + sxy**2)
    )


def build_blade_to_problem(
    mesh: BladeFeaMeshResult,
    *,
    hub_radius_m: float = HUB_RADIUS_M,
    tip_radius_m: float | None = None,
    click_band_m: float = 0.010,
    skin_thickness_m: float = DEFAULT_SKIN_THICKNESS_M,
    rmin_m: float = DEFAULT_RMIN_M,
    aero_pressure_pa: float = DEFAULT_AERO_PRESSURE_PA,
    click_force_n: float = DEFAULT_CLICK_FORCE_N,
    return_scale: float = 1.0,
    volfrac: float = DEFAULT_VOLFRAC,
    penal: float = SIMP_PENALTY,
    shear_modulus_z_pa: float | None = None,
    use_revolution_frames: bool = True,
) -> BladeTOProblem:
    """Assemble the per-design SIMP problem from a tagged blade tet mesh.

    Recomputes boundary-facet geometry with **outward** normals (so the pressure load and
    aero-skin classification are orientation-correct), freezes the aero skin within
    ``skin_thickness_m`` of the top/bottom panel faces, and builds the anisotropic FEA
    model with per-element material frames following the surface of revolution.

    ``shear_modulus_z_pa`` defaults to the isotropic-equivalent interlaminar shear
    ``E_z / 2(1+ν)`` (a documented approximation — FDM interlayer shear is not an
    independent measured constant here); pass a measured value to override.
    """
    nodes, tets = mesh.nodes, mesh.tets
    tip_radius_m = FeaMeshParams().tip_radius_m if tip_radius_m is None else tip_radius_m

    centroids, volumes = tet_centroids_and_volumes(nodes, tets)

    # Outward-oriented boundary facets: needed for the pressure sign and skin tagging.
    facets, opps = boundary_facets(tets)
    centers, normals, areas = facet_geometry(nodes, facets, opps)
    top, bottom = classify_aero_skin(normals, centers, hub_radius_m)
    click_facets = click_region_facets(facets, centers, tip_radius_m, click_band_m)

    # Freeze a solid shell of thickness `skin_thickness_m` on ALL air-facing (top/bottom)
    # surfaces — the thin panel membrane stays fully solid (it is essentially all skin, so
    # never punctured: airflow preserved), while the thick rib rails and any thick-panel
    # interior keep only their outer shell and expose a carvable core. This uniform coating
    # + carve-the-interior model is aero-safe and printable; the rib/panel thickness ranges
    # overlap (2–12 mm vs 3–10 mm) so a per-element rib-vs-panel split is not well-defined.
    aero_face_nodes = np.unique(np.vstack([facets[top], facets[bottom]]))
    frozen = classify_frozen_skin(centroids, nodes[aero_face_nodes], skin_thickness_m)
    design = ~frozen

    frames = revolution_material_frames(centroids) if use_revolution_frames else None
    g_z = in_plane_shear_modulus(E_PETG_Z_PA, NU_PETG) if shear_modulus_z_pa is None else shear_modulus_z_pa
    base_c = transversely_isotropic_stiffness(E_PETG_XY_PA, E_PETG_Z_PA, NU_PETG, NU_PETG, g_z)
    model = build_fea_model(nodes, tets, base_c, mesh.support_node_ids, material_frames=frames)
    filter_matrix = build_density_filter_unstructured(centroids, rmin_m, volumes)

    domain = DesignDomain(
        element_centroids=centroids,
        element_volumes=volumes,
        frozen_mask=frozen,
        design_mask=design,
        support_node_ids=mesh.support_node_ids,
        material_frames=frames,
        meta={"n_frozen": float(frozen.sum()), "n_design": float(design.sum())},
    )
    # Void stiffness floor: ρ_min^penal · C == E_min · C keeps K SPD in fully-carved regions.
    rho_min = float(EMIN_FACTOR ** (1.0 / penal))
    return BladeTOProblem(
        fea_model=model,
        domain=domain,
        filter_matrix=filter_matrix,
        n_nodes=len(nodes),
        aero_facet_node_ids=facets[top],
        aero_facet_areas=areas[top],
        aero_facet_normals=normals[top],
        aero_pressure_pa=np.full(int(top.sum()), aero_pressure_pa),
        element_node_ids=np.asarray(tets, dtype=int),
        click_node_ids=np.unique(click_facets).astype(int),
        click_force_vec=np.array([0.0, click_force_n, 0.0]),
        return_scale=return_scale,
        volfrac=volfrac,
        penal=penal,
        rho_min=rho_min,
        meta={"n_nodes": float(len(nodes)), "n_tets": float(len(tets))},
    )


def _load_cases(problem: BladeTOProblem):
    """Build the four locked load cases at the FULL-SOLID mass (design-independent).

    The inertial body force is held at the full-density (solid-blade) mass rather than the
    evolving SIMP density. This is deliberate on two counts: (1) it keeps every load case
    independent of ρ, so the compliance sensitivity is the exact self-adjoint SIMP form
    ``dC/dρ = −p ρ^(p−1) uₑᵀKₑ⁰uₑ`` — a ρ-dependent load would need the extra adjoint term
    ``2 uᵀ ∂f/∂ρ`` (its omission flips the sign of many sensitivities when inertial load
    dominates); (2) the solid blade is the heaviest, so the inertial load is a conservative
    upper bound and the trimmed design (lighter, lower real inertial load) is only safer.
    """
    dom = problem.domain
    return assemble_load_cases(
        problem.n_nodes,
        problem.aero_facet_node_ids,
        problem.aero_facet_areas,
        problem.aero_facet_normals,
        problem.aero_pressure_pa,
        problem.element_node_ids,
        dom.element_volumes,
        dom.element_centroids,
        problem.click_node_ids,
        problem.click_force_vec,
        element_density=None,  # full-solid mass -> ρ-independent load -> exact sensitivity
        return_scale=problem.return_scale,
    )


def _physical_density(problem: BladeTOProblem, x: np.ndarray) -> np.ndarray:
    """Filtered density with the aero skin re-pinned to 1 and a void floor for SPD ``K``."""
    rho = problem.filter_matrix @ x
    rho[problem.domain.frozen_mask] = 1.0  # skin stays solid regardless of filter bleed
    return np.clip(rho, problem.rho_min, 1.0)


def run_blade_topology_optimization(
    problem: BladeTOProblem, *, max_iters: int = 40, tol: float = 0.01, move: float = 0.2
) -> BladeTOResult:
    """Run the multi-load SIMP OC loop to convergence (or ``max_iters``).

    Convergence = max design-variable change < ``tol``. Returns the carved physical density
    field, the compliance history, the retained/removed volume, single-blade mass, and the
    peak deflection + von Mises stress under the productive stroke (the §Phase-2 screening
    readouts; the binding structural certification is the §59.5 combined-blade gate).
    """
    dom = problem.domain
    model = problem.fea_model
    w = problem.filter_matrix
    volumes = dom.element_volumes
    frozen, design = dom.frozen_mask, dom.design_mask
    penal = problem.penal

    x = np.where(design, problem.volfrac, 0.0)
    x[frozen] = 1.0

    load_cases = _load_cases(problem)  # ρ-independent (full-solid inertial) -> built once
    dv_dx = w.T @ volumes  # filter-consistent volume sensitivity ∂V/∂x = Wᵀ V (top88 form)

    history: list[float] = []
    converged = False
    iterations = 0
    for it in range(1, max_iters + 1):
        iterations = it
        rho = _physical_density(problem, x)
        k = assemble_global_stiffness(model, density=rho, penal=penal)

        compliance_tot = 0.0
        dc_drho = np.zeros_like(x)
        # All four load cases share this iteration's K — factorize once, back-substitute each.
        solved = solve_displacements_multi(model, [lc.node_forces for lc in load_cases], k)
        for lc, (u, f) in zip(load_cases, solved):
            compliance_tot += lc.weight * compliance(u, f)
            energy = element_strain_energies(model, u)  # uₑᵀKₑ⁰uₑ (full-density base)
            dc_drho += -lc.weight * penal * rho ** (penal - 1.0) * energy
        history.append(compliance_tot)

        dc_drho[frozen] = 0.0  # frozen ρ is constant → no design sensitivity through it
        dc_dx = w.T @ dc_drho  # filter chain: dC/dx = Wᵀ dC/dρ̃
        x_new = oc_update_unstructured(
            x, dc_dx, volumes, w, volfrac=problem.volfrac, design_mask=design,
            move=move, volume_sensitivity=dv_dx,
        )
        change = float(np.abs(x_new - x)[design].max()) if design.any() else 0.0
        x = x_new
        if change < tol:
            converged = True
            break

    # Structural screen: worst deflection + peak von Mises over ALL FOUR load cases and ALL
    # elements (the frozen aero skin carries the extreme-fibre bending stress — excluding it
    # would hide the true peak; a single-load screen would miss the click/inertial worst case).
    rho = _physical_density(problem, x)
    k = assemble_global_stiffness(model, density=rho, penal=penal)
    u_tip = 0.0
    max_vm = 0.0
    for u, _ in solve_displacements_multi(model, [lc.node_forces for lc in load_cases], k):
        u_tip = max(u_tip, float(np.linalg.norm(u.reshape(-1, 3), axis=1).max()))
        max_vm = max(max_vm, float(von_mises(element_stresses(model, u)).max()))

    retained_vol = float((volumes * rho).sum())
    full_vol = float(volumes.sum())
    design_full = float(volumes[design].sum())
    design_retained = float((volumes * rho)[design].sum())
    return BladeTOResult(
        density=rho,
        compliance_history=tuple(history),
        design_volume_fraction=design_retained / design_full if design_full else 0.0,
        volume_removed_frac=1.0 - retained_vol / full_vol if full_vol else 0.0,
        mass_kg=RHO_PETG_KG_PER_M3 * retained_vol,
        u_tip_max_m=u_tip,
        max_von_mises_pa=max_vm,
        converged=converged,
        iterations=iterations,
        meta={**problem.meta, "n_design": float(design.sum()), "n_frozen": float(frozen.sum())},
        element_centroids=dom.element_centroids,
    )


def topology_optimize_blade(
    params: BladeParams,
    *,
    mesh_params: FeaMeshParams | None = None,
    volfrac: float = DEFAULT_VOLFRAC,
    max_iters: int = 40,
    tol: float = 0.01,
    skin_thickness_m: float = DEFAULT_SKIN_THICKNESS_M,
    rmin_m: float = DEFAULT_RMIN_M,
) -> BladeTOResult:
    """Mesh one blade design and run its 3D SIMP TO end-to-end (the per-design entry point).

    Thin composition of :func:`build_blade_fea_mesh` → :func:`build_blade_to_problem` →
    :func:`run_blade_topology_optimization` for a single :class:`BladeParams`. The batch
    driver (``scripts/run_phase2_blade_to.py``) calls this once per top-N design.
    """
    mp = mesh_params or FeaMeshParams()
    mesh = build_blade_fea_mesh(params, mesh_params=mp)
    problem = build_blade_to_problem(
        mesh,
        hub_radius_m=mp.hub_radius_m,
        tip_radius_m=mp.tip_radius_m,
        click_band_m=mp.click_band_m,
        skin_thickness_m=skin_thickness_m,
        rmin_m=rmin_m,
        volfrac=volfrac,
    )
    return run_blade_topology_optimization(problem, max_iters=max_iters, tol=tol)


def _passes_screen(res: BladeTOResult, u_tip_limit_m: float, sigma_allow_pa: float) -> bool:
    """Structural screen: worst tip deflection under the limit AND worst σ_VM under the allowable."""
    return res.u_tip_max_m <= u_tip_limit_m and res.max_von_mises_pa <= sigma_allow_pa


def _screen_ladder(run_at_volfrac, volfrac_ladder, u_tip_limit_m, sigma_allow_pa):
    """Try ``volfrac_ladder`` (most-aggressive/lowest first); accept the first that passes the screen.

    ``run_at_volfrac(vf) -> BladeTOResult``. Returns ``(result, accepted_volfrac, trace, passed)``.
    Assumes the screen is monotone in volfrac (more material → stiffer/lower stress → likelier to
    pass), so the first passing rung is the minimum-mass design that keeps integrity. If none pass,
    returns the safest (highest-volfrac) attempt with ``passed=False`` so the caller can flag it.
    """
    trace: list[dict] = []
    result = None
    for vf in volfrac_ladder:
        result = run_at_volfrac(vf)
        passed = _passes_screen(result, u_tip_limit_m, sigma_allow_pa)
        trace.append(
            {
                "volfrac": vf,
                "u_tip_mm": result.u_tip_max_m * 1e3,
                "vm_mpa": result.max_von_mises_pa / 1e6,
                "removed_pct": result.volume_removed_frac * 100.0,
                "passed": passed,
            }
        )
        if passed:
            return result, vf, trace, True
    return result, volfrac_ladder[-1], trace, False


def topology_optimize_blade_screened(
    params: BladeParams,
    *,
    mesh_params: FeaMeshParams | None = None,
    max_iters: int = 40,
    tol: float = 0.01,
    skin_thickness_m: float = DEFAULT_SKIN_THICKNESS_M,
    rmin_m: float = DEFAULT_RMIN_M,
    volfrac: float | None = None,  # accepted for batch-call compat; screened mode uses the ladder
    volfrac_ladder: tuple[float, ...] = DEFAULT_VOLFRAC_LADDER,
    u_tip_limit_m: float = DEFAULT_U_TIP_LIMIT_M,
    sigma_allow_pa: float | None = None,
) -> BladeTOResult:
    """Most aggressive (lowest-volfrac) TO that still passes the structural screen, per design.

    Meshes + assembles the FEA model **once**, then re-runs only the SIMP OC loop at each rung of
    ``volfrac_ladder`` (via :func:`dataclasses.replace` on the frozen problem — the big arrays are
    shared, not rebuilt), stopping at the first volfrac whose result passes ``u_tip < u_tip_limit_m``
    and ``σ_VM < yield/FOS``. This directly answers "remove as much material as possible while
    keeping structural integrity" — the acceptance is decided by the measured screen, not a guess.
    The accepted volfrac, whether the screen passed, and the full ladder trace are stored in
    ``result.meta``.
    """
    sigma_allow = SIGMA_Y_PETG_Z_PA / DEFAULT_STRESS_FOS if sigma_allow_pa is None else sigma_allow_pa
    mp = mesh_params or FeaMeshParams()
    mesh = build_blade_fea_mesh(params, mesh_params=mp)
    problem = build_blade_to_problem(
        mesh,
        hub_radius_m=mp.hub_radius_m,
        tip_radius_m=mp.tip_radius_m,
        click_band_m=mp.click_band_m,
        skin_thickness_m=skin_thickness_m,
        rmin_m=rmin_m,
        volfrac=volfrac_ladder[0],
    )

    def run_at(vf: float) -> BladeTOResult:
        return run_blade_topology_optimization(replace(problem, volfrac=vf), max_iters=max_iters, tol=tol)

    result, accepted_vf, trace, passed = _screen_ladder(
        run_at, volfrac_ladder, u_tip_limit_m, sigma_allow
    )
    return replace(
        result,
        meta={
            **result.meta,
            "accepted_volfrac": accepted_vf,
            "screen_passed": float(passed),
            "u_tip_limit_mm": u_tip_limit_m * 1e3,
            "sigma_allow_mpa": sigma_allow / 1e6,
            "ladder_trace": trace,
        },
    )


def _result_summary(name: str, params: BladeParams, res: BladeTOResult) -> dict:
    """JSON-friendly per-design record (the density field is written separately as .npy)."""
    return {
        "name": name,
        "design_hash": name.split("_", 1)[1] if "_" in name else name,
        "uniform": params.uniform,
        "volume_removed_frac": res.volume_removed_frac,
        "design_volume_fraction": res.design_volume_fraction,
        "mass_kg": res.mass_kg,
        "u_tip_max_mm": res.u_tip_max_m * 1e3,
        "max_von_mises_mpa": res.max_von_mises_pa / 1e6,
        "converged": res.converged,
        "iterations": res.iterations,
        "compliance_initial": res.compliance_history[0] if res.compliance_history else None,
        "compliance_final": res.compliance_history[-1] if res.compliance_history else None,
        "n_design": res.meta.get("n_design"),
        "n_frozen": res.meta.get("n_frozen"),
        # Present only in screened mode (topology_optimize_blade_screened).
        "accepted_volfrac": res.meta.get("accepted_volfrac"),
        "screen_passed": bool(res.meta["screen_passed"]) if "screen_passed" in res.meta else None,
        "ladder_trace": res.meta.get("ladder_trace"),
    }


def _optimize_one_design(
    name, params, out, mp, volfrac, max_iters, tol, skin_thickness_m, rmin_m, optimize
):
    """Run one design's TO and write its artifacts; return ``(name, record, result_or_None)``.

    Module-level (picklable) so it can run in a process-pool worker. Isolates any failure —
    mesh/solve error OR an I/O error — into an ``error`` record so one bad design never aborts
    the batch. The ``<name>.json`` sidecar is written only on success (so a failure is retried
    on the next resume).
    """
    try:
        res = optimize(
            params, mesh_params=mp, volfrac=volfrac, max_iters=max_iters, tol=tol,
            skin_thickness_m=skin_thickness_m, rmin_m=rmin_m,
        )
        np.save(out / f"{name}_density.npy", res.density)
        if res.element_centroids is not None:
            np.save(out / f"{name}_centroids.npy", res.element_centroids)
        rec = _result_summary(name, params, res)
        (out / f"{name}.json").write_text(json.dumps(rec), encoding="utf-8")
        return name, rec, res
    except Exception as exc:  # noqa: BLE001 - isolate a single bad design (incl. its I/O)
        return name, {"name": name, "error": f"{type(exc).__name__}: {exc}"}, None


def run_blade_to_batch(
    designs: Sequence[tuple[str, BladeParams]],
    out_dir: str | Path,
    *,
    mesh_params: FeaMeshParams | None = None,
    volfrac: float = DEFAULT_VOLFRAC,
    max_iters: int = 40,
    tol: float = 0.01,
    skin_thickness_m: float = DEFAULT_SKIN_THICKNESS_M,
    rmin_m: float = DEFAULT_RMIN_M,
    n_workers: int = 1,
    optimize: Callable[..., BladeTOResult] = topology_optimize_blade,
    on_result: Callable[[str, BladeTOResult], None] | None = None,
) -> dict:
    """Run per-design 3D TO over a batch of designs, writing artifacts under ``out_dir``.

    For each ``(name, params)`` writes ``<name>_density.npy`` (carved field), ``<name>_centroids.npy``
    (element centroids, so the render aligns without re-meshing), and a ``<name>.json`` sidecar.
    The aggregate ``summary.json`` is rewritten atomically **after every design**, so a crash
    mid-batch never loses completed work. **Resumable:** a design whose sidecar already exists is
    reloaded and skipped (delete the sidecar to force a re-run). A design that raises (mesh
    failure, singular solve, or a write error) is recorded with an ``error`` and skipped, so one
    bad blade never aborts the batch.

    ``n_workers > 1`` runs designs concurrently in a process pool (each design is independent).
    Concurrency is **RAM-bound**: every worker holds a full FEA factorization (~15 GB at a 0.8 mm
    mesh, ~30 GB at 0.6 mm), so set ``n_workers ≈ session_RAM / per_design_RAM`` — over-subscribing
    RAM will OOM the run. ``optimize`` must be picklable in parallel mode (a module-level function
    or a ``functools.partial`` of one — not a closure/lambda). ``on_result`` fires as each design
    finishes. Returns the aggregate summary dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mp = mesh_params or FeaMeshParams()

    def _write_summary(records: list[dict]) -> dict:
        summary = {
            "n_designs": len(designs),
            "n_succeeded": sum("error" not in r for r in records),
            "volfrac": volfrac,
            "skin_thickness_m": skin_thickness_m,
            "mesh_size_m": mp.mesh_size_m,
            "n_workers": n_workers,
            "designs": records,
        }
        tmp = out / "summary.json.tmp"
        tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        tmp.replace(out / "summary.json")
        return summary

    # Resume: reuse completed designs' sidecars; only the rest are (re)computed.
    records: list[dict] = []
    pending: list[tuple[str, BladeParams]] = []
    for name, params in designs:
        sidecar = out / f"{name}.json"
        if sidecar.exists():
            records.append(json.loads(sidecar.read_text(encoding="utf-8")))
        else:
            pending.append((name, params))
    summary = _write_summary(records)

    args = [
        (name, params, out, mp, volfrac, max_iters, tol, skin_thickness_m, rmin_m, optimize)
        for name, params in pending
    ]
    if n_workers > 1 and len(args) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_optimize_one_design, *a) for a in args]
            for future in as_completed(futures):
                name, rec, res = future.result()
                records.append(rec)
                summary = _write_summary(records)
                if res is not None and on_result is not None:
                    on_result(name, res)
    else:
        for a in args:
            name, rec, res = _optimize_one_design(*a)
            records.append(rec)
            summary = _write_summary(records)
            if res is not None and on_result is not None:
                on_result(name, res)
    return summary
