#!/usr/bin/env python3
"""Fit the as-printed blade mass model to Bambu Studio slices of real print blades.

Picks campaign designs spread over their mean wall (plus any named), builds each at several common
mean walls with ``build_print_blades.py`` (solid inside, no detent), slices it with the operator's own
settings (the printed ``.gcode.3mf`` as template, brim off so the brim isn't counted) and fits
:class:`fanopt.physical.print_effort.PrintMassModel` to the sliced filament weights. Resumable: a sample
whose slice result exists is not rebuilt.

    python3 scripts/calibrate_print_mass.py --shards print_ready/drive_recovery/campaign_trapezoid \\
        --template "print_ready/H2C_blade_00_..._PLA-15%infill.gcode.3mf" \\
        --include ffe19f9e3e7fa81ba847bbe8 --work print_ready/mass_calibration \\
        --out data/v2_w1/print_mass_calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from fanopt.bo.blade_codec import decode
from fanopt.cfd.blade_verify import load_campaign_rows
from fanopt.geometry.bambu_3mf import (
    filament_density_kg_m3,
    sliced_filament_g,
    with_project_settings,
)
from fanopt.physical.print_effort import (
    blade_mass_kg,
    blade_sheet,
    fit_print_mass_model,
    mean_thickness_m,
)

BAMBU_APP = Path("/Applications/BambuStudio.app/Contents/MacOS/BambuStudio")
BUILD_SCRIPT = Path(__file__).resolve().parent / "build_print_blades.py"


def valid_rows(shards: Path) -> dict[str, dict]:
    """Campaign rows with a finite ``j_fan``, by design hash."""
    return {
        str(r["design_hash"]): r
        for r in load_campaign_rows(shards)
        if r.get("j_fan") is not None and math.isfinite(float(r["j_fan"]))
    }


def spread_picks(values: dict[str, float], n: int) -> list[str]:
    """``n`` keys at evenly spaced quantiles of their values (lowest and highest included)."""
    keys = sorted(values, key=values.__getitem__)
    idx = np.unique(np.round(np.linspace(0, len(keys) - 1, n)).astype(int))
    return [keys[i] for i in idx]


def _sample_dir(work: Path, design: str, target_mm: float) -> Path:
    return work / f"{design}_{target_mm:.2f}mm"


def build_and_slice(
    *, design: str, row: dict, scale: float, d: Path, template: Path, bambu: Path, python: str
) -> tuple[float, float]:
    """``(sliced filament g, thinning the build used)`` for one solid print blade; reuses earlier results.

    A reused build keeps the thinning it was built with (read from its report), even if ``scale``
    would now differ.
    """
    d = d.resolve()  # the slicer runs with cwd=d, so a relative --outputdir would nest
    result = d / "result.json"
    if not result.exists():
        d.mkdir(parents=True, exist_ok=True)
        name = f"cal_{design}"
        (d / "evaluations_cal.jsonl").write_text(json.dumps(row) + "\n")
        np.save(d / f"{name}_centroids.npy", np.zeros((1, 3)))
        np.save(d / f"{name}_density.npy", np.ones(1))  # solid: every tet maps to one solid sample
        blade = d / "blade.stl"
        if not blade.with_suffix(".3mf").exists():
            subprocess.run(
                [
                    python, str(BUILD_SCRIPT), "--name", name, "--shards", str(d), "--to-dir", str(d),
                    "--out", str(blade), "--rib-width-mm", "2.0", "--thickness-scale", f"{scale:.6f}",
                    "--material", "pla", "--detent-seats", "0", "--template", str(template),
                    "--template-settings-only",
                ],
                check=True, capture_output=True, text=True,
            )  # fmt: skip
        # Bambu Studio writes result.json into --outputdir; a failed slice is reported there too.
        subprocess.run(
            [
                str(bambu), "--slice", "0", "--outputdir", str(d), "--export-3mf", "sliced.gcode.3mf",
                str(blade.with_suffix(".3mf")),
            ],
            check=False, capture_output=True, text=True, cwd=d,
        )  # fmt: skip
    used = float(json.loads((d / "blade.json").read_text())["thickness_scale"])
    return sliced_filament_g(json.loads(result.read_text())), used


def run(
    *,
    shards: Path,
    template: Path,
    work: Path,
    out: Path,
    include: list[str],
    n_spread: int,
    targets_mm: list[float],
    jobs: int,
    bambu: Path,
    python: str,
) -> dict:
    """Build + slice every (design, wall) sample, fit the model, write and return the report."""
    rows = valid_rows(shards)
    sheets = {h: blade_sheet(decode(np.asarray(r["vector"], dtype=float))) for h, r in rows.items()}
    means = {h: mean_thickness_m(s) for h, s in sheets.items()}
    designs = list(dict.fromkeys([*include, *spread_picks(means, n_spread)]))
    samples = [(h, t) for h in designs for t in targets_mm if t * 1e-3 <= means[h]]
    work.mkdir(parents=True, exist_ok=True)
    nobrim = with_project_settings(
        template, work / "template_no_brim.3mf", {"brim_type": "no_brim"}
    )

    def one(sample: tuple[str, float]) -> tuple[float, float]:
        h, t = sample
        return build_and_slice(
            design=h, row=rows[h], scale=t * 1e-3 / means[h], d=_sample_dir(work, h, t),
            template=nobrim, bambu=bambu, python=python,
        )  # fmt: skip

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(one, samples))
    grams = [g for g, _ in results]
    fit_sheets = [
        replace(sheets[h], thickness_m=sheets[h].thickness_m * used)
        for (h, _), (_, used) in zip(samples, results, strict=True)
    ]
    masses = [g * 1e-3 for g in grams]
    # The slicer weighs with the template's filament density (Bambu PLA Matte: 1.32 g/cm³), so fit with it.
    rho = filament_density_kg_m3(template)
    model = fit_print_mass_model(fit_sheets, masses, rho_kg_m3=rho)
    loo = []
    for i in range(len(samples)):
        rest = [j for j in range(len(samples)) if j != i]
        m = fit_print_mass_model(
            [fit_sheets[j] for j in rest], [masses[j] for j in rest], rho_kg_m3=rho
        )
        loo.append(blade_mass_kg(fit_sheets[i], m) * 1e3 - grams[i])
    pred = [blade_mass_kg(s, model) * 1e3 for s in fit_sheets]
    report = {
        "template": str(template),
        "brim": "off (no_brim) so the brim is not counted",
        "model": {
            "wall_mm": model.wall_m * 1e3,
            "core_fraction": model.core_fraction,
            "base_g": model.base_kg * 1e3,
            "rho_kg_m3": model.rho_kg_m3,
        },
        "rmse_g": float(np.sqrt(np.mean((np.array(pred) - np.array(grams)) ** 2))),
        "loo_rmse_g": float(np.sqrt(np.mean(np.array(loo) ** 2))),
        "loo_max_abs_g": float(np.max(np.abs(loo))),
        "samples": [
            {
                "design_hash": h, "target_mean_wall_mm": t, "design_mean_wall_mm": means[h] * 1e3,
                "built_mean_wall_mm": mean_thickness_m(s) * 1e3, "sliced_g": g, "model_g": p,
                "loo_error_g": e,
            }
            for (h, t), s, g, p, e in zip(samples, fit_sheets, grams, pred, loo, strict=True)
        ],  # fmt: skip
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shards", type=Path, required=True, help="dir with evaluations_*.jsonl")
    ap.add_argument("--template", type=Path, required=True, help="the printed .gcode.3mf")
    ap.add_argument("--work", type=Path, required=True, help="dir for the builds and slices")
    ap.add_argument("--out", type=Path, required=True, help="calibration report (.json)")
    ap.add_argument("--include", nargs="*", default=[], help="design hashes always sampled")
    ap.add_argument("--n-spread", type=int, default=5, help="designs spread over the mean wall")
    ap.add_argument(
        "--targets-mm", type=float, nargs="+", default=[1.6, 2.0, 2.5, 3.0],
        help="common mean walls to build each design at (only thinner than the design)",
    )  # fmt: skip
    ap.add_argument("--jobs", type=int, default=3, help="builds in parallel")
    ap.add_argument("--bambu", type=Path, default=BAMBU_APP, help="Bambu Studio executable")
    a = ap.parse_args(argv)
    report = run(
        shards=a.shards, template=a.template, work=a.work, out=a.out, include=a.include,
        n_spread=a.n_spread, targets_mm=a.targets_mm, jobs=a.jobs, bambu=a.bambu,
        python=sys.executable,
    )  # fmt: skip
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
