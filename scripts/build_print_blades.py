#!/usr/bin/env python3
"""Rebuild a Stage-4 TO blade as an on-edge printable STL from its saved density, without inflation.

Re-meshes the design's own TO tet mesh (same params + mesh size as the TO run), maps the saved per-tet
density onto it, and hands both to :func:`fanopt.geometry.print_blade.build_print_blade`. Writes the STL
(mm), a JSON report of every volume added or removed, the mass, wall-width and bed-contact checks, and —
given ``--template`` (the operator's own Bambu project) — a ready-to-slice .3mf beside the STL: same
printer/filament/process settings, supports off, 8 mm outer brim, no raft. ``--material`` sets the
filament density used for the reported mass; the .3mf keeps the template's filament profile.

    python3 scripts/build_print_blades.py --name 00_ffe19f9e3e7fa81ba847bbe8 --rib-width-mm 2.0 \\
        --shards print_ready/drive_recovery/campaign_trapezoid --to-dir print_ready/drive_recovery \\
        --out print_ready/P2C_blade_00_rebuilt.stl --template ~/Downloads/H2C_blade_00.3mf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from fanopt.bo.blade_codec import decode
from fanopt.cfd.blade_verify import load_campaign_rows
from fanopt.geometry.bambu_3mf import write_bambu_project
from fanopt.geometry.blade import folded_stack_height_m, layer_spacing_m
from fanopt.geometry.blade_detent import DEFAULT_DETENT_SPEC, DetentSpec, detent_geometry
from fanopt.geometry.print_blade import (
    bed_contact_fraction,
    build_print_blade,
    flat_boss_radius_m,
    layer_thin_fraction,
    mesh_volume,
)
from fanopt.geometry.rib_detent import DEFAULT_DEPLOY_DIRECTION, rib_detent_seats
from fanopt.geometry.schema import (
    HUB_RADIUS_M,
    PIVOT_BOSS_RADIUS_M,
    PIVOT_PIN_DIAMETER_M,
    RHO_PETG_KG_PER_M3,
    RHO_PLA_KG_PER_M3,
)
from fanopt.geometry.to_stl import write_binary_stl
from fanopt.topopt.blade_fea_mesh import (
    FeaMeshParams,
    boundary_facets,
    build_blade_fea_mesh,
    facet_geometry,
)
from fanopt.topopt.design_domain import tet_centroids_and_volumes

MATERIAL_DENSITY_KG_PER_M3: dict[str, float] = {
    "petg": RHO_PETG_KG_PER_M3,
    "pla": RHO_PLA_KG_PER_M3,
}

# On-edge with the whole LE on the bed: no overhang needs support; a brim steadies the tall thin wall.
# 100 % infill: the TO sized (and stress-screened) the blade as solid PETG, so print it solid.
PROJECT_SETTINGS: dict[str, object] = {
    "sparse_infill_density": "100%",
    "sparse_infill_pattern": "zig-zag",  # grid is rejected at 100 %; same pattern as the profile's solid infill
    "enable_support": "0",
    "raft_layers": "0",
    "brim_type": "outer_only",
    "brim_width": "8",
}


def run(
    *,
    name: str,
    shards: Path,
    to_dir: Path,
    out: Path,
    rib_width_m: float,
    mesh_size_m: float = 0.6e-3,
    skin_m: float = 0.5e-3,
    tip_cap_m: float = 0.6e-3,
    pitch_m: float = 1.0e-4,
    template: Path | None = None,
    material: str = "petg",
    detent_seats: int = 5,
    deploy_direction: int = DEFAULT_DEPLOY_DIRECTION,
    detent_spec: DetentSpec = DEFAULT_DETENT_SPEC,
) -> dict:
    """Build one blade's STL (+ .json report, + .3mf when ``template`` is given); returns the report."""
    rho_material = MATERIAL_DENSITY_KG_PER_M3[material]
    rows = {str(r["design_hash"]): r for r in load_campaign_rows(shards)}
    params = decode(np.asarray(rows[name.split("_", 1)[1]]["vector"], dtype=float))
    mesh = build_blade_fea_mesh(params, mesh_params=FeaMeshParams(mesh_size_m=mesh_size_m))
    centroids, volumes = tet_centroids_and_volumes(mesh.nodes, mesh.tets)
    saved_rho = np.load(to_dir / f"{name}_density.npy")
    remap, idx = cKDTree(np.load(to_dir / f"{name}_centroids.npy")).query(centroids)
    rho = saved_rho[idx]
    facets, opposite = boundary_facets(mesh.tets)
    centers, normals, _areas = facet_geometry(mesh.nodes, facets, opposite)
    spacing = layer_spacing_m(params)
    detent = None
    if detent_seats > 0:
        seats = rib_detent_seats(params, deploy_direction=deploy_direction, n_seats=detent_seats)
        detent = detent_geometry(seats, mesh.nodes, facets, spacing, detent_spec)

    blade = build_print_blade(
        mesh.nodes, mesh.tets, rho, centers, normals,
        hub_radius_m=HUB_RADIUS_M, bore_radius_m=PIVOT_PIN_DIAMETER_M / 2.0,
        rib_width_m=rib_width_m, tip_cap_m=tip_cap_m, skin_m=skin_m, pitch_m=pitch_m,
        boss_half_height_m=spacing / 2.0, boss_radius_m=PIVOT_BOSS_RADIUS_M,
        bump_spheres=None if detent is None else detent.bumps,
        recess_spheres=None if detent is None else detent.recesses,
    )  # fmt: skip
    boss_r = flat_boss_radius_m(mesh.nodes, spacing / 2.0)
    cad_verts = (blade.verts - blade.print_offset_m) @ blade.print_from_cad
    in_boss = np.hypot(cad_verts[:, 0], cad_verts[:, 1]) < boss_r - 0.5e-3
    write_binary_stl(blade.verts, blade.faces, out)
    project = None
    if template is not None:
        project = write_bambu_project(
            template,
            out.with_suffix(".3mf"),
            blade.verts * 1e3,
            blade.faces,
            name=out.name,
            settings=PROJECT_SETTINGS,
        )

    voxel_cm3 = pitch_m**3 * 1e6
    volume_m3 = mesh_volume(blade.verts, blade.faces)
    areas = blade.solid.sum(axis=(0, 1)).astype(float)
    thin = layer_thin_fraction(blade.solid, pitch_m, 0.4e-3)
    report = {
        "name": name,
        "stl": str(out),
        "project_3mf": None if project is None else str(project),
        "triangles": int(len(blade.faces)),
        "rib_width_mm": rib_width_m * 1e3,
        "skin_mm": skin_m * 1e3,
        "tip_cap_mm": tip_cap_m * 1e3,
        "voxel_pitch_mm": pitch_m * 1e3,
        "density_remap_mm": {
            q: float(np.percentile(remap, p) * 1e3)
            for q, p in (("p50", 50), ("p99", 99), ("max", 100))
        },
        # The TO's own mass bookkeeping is PETG; kept in PETG so it compares with the recorded run.
        "to_mass_g_on_rebuilt_mesh_petg": float((volumes * rho).sum() * RHO_PETG_KG_PER_M3 * 1e3),
        "to_solid_cm3_exact_tets": float((volumes * (rho >= 0.5)).sum() * 1e6),
        "design_domain_cm3_exact_tets": float(volumes.sum() * 1e6),
        "voxel_cm3": {k: v * voxel_cm3 for k, v in blade.voxel_counts.items()},
        "volume_cm3": volume_m3 * 1e6,
        "material": material,
        "material_density_kg_m3": rho_material,
        "mass_g": volume_m3 * rho_material * 1e3,
        "fan_mass_12_blades_g": 12 * volume_m3 * rho_material * 1e3,
        "design_folded_stack_mm": folded_stack_height_m(params) * 1e3,
        "design_layer_spacing_mm": spacing * 1e3,
        "boss_flat_radius_mm": boss_r * 1e3,
        "boss_thickness_mm": float(np.ptp(cad_verts[in_boss, 2]) * 1e3),
        "detent": (
            None
            if detent is None
            else {
                "deploy_direction": deploy_direction,
                "spec_mm": {k: v * 1e3 for k, v in vars(detent_spec).items()},
                "seats": detent.seats,
            }
        ),
        "bed_contact_fraction": bed_contact_fraction(blade.solid),
        "area_frac_thinner_than_0p4mm_conservative": float((thin * areas).sum() / areas.sum()),
        "print_from_cad": blade.print_from_cad.tolist(),
        "print_offset_mm": (blade.print_offset_m * 1e3).tolist(),
        "bbox_mm": [
            (blade.verts.min(axis=0) * 1e3).tolist(),
            (blade.verts.max(axis=0) * 1e3).tolist(),
        ],
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--name", required=True, help="TO design name, e.g. 00_ffe19f9e3e7fa81ba847bbe8"
    )
    ap.add_argument(
        "--shards", type=Path, required=True, help="dir with the campaign evaluations_*.jsonl"
    )
    ap.add_argument(
        "--to-dir", type=Path, required=True, help="dir with <name>_density.npy + _centroids.npy"
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output STL path (a .json report is written beside it)",
    )
    ap.add_argument("--rib-width-mm", type=float, required=True)
    ap.add_argument("--mesh-size-mm", type=float, default=0.6, help="must match the TO run")
    ap.add_argument(
        "--skin-mm", type=float, default=0.5, help="must match the TO run's frozen skin"
    )
    ap.add_argument("--tip-cap-mm", type=float, default=0.6)
    ap.add_argument("--pitch-mm", type=float, default=0.1)
    ap.add_argument(
        "--template", type=Path, default=None, help="Bambu .3mf whose settings the project copies"
    )
    ap.add_argument(
        "--material",
        choices=sorted(MATERIAL_DENSITY_KG_PER_M3),
        default="petg",
        help="filament for the reported mass (the .3mf filament profile is the template's)",
    )
    ap.add_argument(
        "--detent-seats", type=int, default=5, help="hold-detent seats per blade (0 = no detent)"
    )
    ap.add_argument(
        "--deploy-direction",
        type=int,
        choices=(-1, 1),
        default=DEFAULT_DEPLOY_DIRECTION,
        help="fan-out sense the fan is assembled to open in (sets which rib carries the bumps)",
    )
    a = ap.parse_args(argv)
    report = run(
        name=a.name, shards=a.shards, to_dir=a.to_dir, out=a.out, rib_width_m=a.rib_width_mm / 1e3,
        mesh_size_m=a.mesh_size_mm / 1e3, skin_m=a.skin_mm / 1e3, tip_cap_m=a.tip_cap_mm / 1e3,
        pitch_m=a.pitch_mm / 1e3, template=a.template, material=a.material,
        detent_seats=a.detent_seats, deploy_direction=a.deploy_direction,
    )  # fmt: skip
    print(json.dumps({k: v for k, v in report.items() if k not in ("print_from_cad",)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
