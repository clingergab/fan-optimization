"""Fusion 360 add-in — Spike 0.1: headless params.json → User Parameters → STL/STEP export.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.1
    "Can Fusion be driven headlessly from a Python script on macOS (read JSON,
    regenerate model, export per-blade STLs + STEP)?"

This file is BOTH a Fusion script (single-shot, invoked via the Run Script
menu or `--runScript` CLI flag) AND an installable add-in (Fusion calls
`run(context)` on load and `stop(context)` on unload). For Spike 0.1 we
exercise the script path: it's simpler and matches the spec's
"minimal Fusion Python add-in" wording.

Workflow:
    1. Read `params.json` from `$FANOPT_PARAMS` (env) or the per-OS default
       location. JSON schema is documented in `params.example.json`.
    2. For each entry in `user_parameters`, set the matching Fusion User
       Parameter expression. Unknown names are recorded but not fatal.
    3. Force a model recompute (Fusion auto-recomputes on expression-set,
       but we call `Design.computeAll()` explicitly for determinism).
    4. Export per-body STL files (one per `BRepBody`) and a single
       assembled-root-component STEP file to `output_dir`.
    5. Write a success marker `SPIKE_0_1_PASS.json` with the run summary;
       on exception, write `SPIKE_0_1_FAIL.json` with traceback so the
       headless driver can detect outcome without parsing Fusion's log.

The driver script `scripts/run_spike_0_1.py` polls for one of those two
markers to declare pass/fail.

After Spike 0.1 lands, this same file can grow into the §12.1-described
"multi-blade assembly view" (post-print viewer) — the parameter-application
+ export plumbing here is the same primitive that the viewer would extend.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import adsk.core
import adsk.fusion


DEFAULT_PARAMS_MACOS = Path.home() / "Projects" / "fan-optimization" / "configs" / "fusion" / "params.json"
DEFAULT_OUTPUT_DIR = Path.home() / "Projects" / "fan-optimization" / "data" / "meshes" / "spike_0_1"


def _resolve_params_path() -> Path:
    env = os.environ.get("FANOPT_PARAMS")
    if env:
        return Path(env).expanduser()
    return DEFAULT_PARAMS_MACOS


def _resolve_output_dir(params: dict) -> Path:
    explicit = params.get("output_dir")
    if explicit:
        return Path(explicit).expanduser()
    return DEFAULT_OUTPUT_DIR


def _apply_user_parameters(design: adsk.fusion.Design, requested: dict) -> tuple[list, list]:
    """Apply every requested User Parameter expression.

    Returns (applied, missing) where each element is (name, expression).
    `applied` succeeded; `missing` named a parameter that does not exist in
    the active document (recorded so the driver can flag schema drift).
    """
    user_params = design.userParameters
    applied: list[tuple[str, str]] = []
    missing: list[tuple[str, str]] = []
    for name, value in requested.items():
        if name.startswith("_"):
            continue
        expr = str(value)
        param = user_params.itemByName(name)
        if param is None:
            missing.append((name, expr))
            continue
        param.expression = expr
        applied.append((name, expr))
    return applied, missing


def _export_per_body_stl(
    design: adsk.fusion.Design,
    out_dir: Path,
    base_name: str,
    refinement: str,
) -> list[str]:
    """Export each top-level BRepBody under the root component as its own STL."""
    refinement_map = {
        "low": adsk.fusion.MeshRefinementSettings.MeshRefinementLow,
        "medium": adsk.fusion.MeshRefinementSettings.MeshRefinementMedium,
        "high": adsk.fusion.MeshRefinementSettings.MeshRefinementHigh,
    }
    mesh_setting = refinement_map.get(refinement.lower(), refinement_map["medium"])

    root = design.rootComponent
    export_mgr = design.exportManager
    written: list[str] = []
    bodies = list(root.bRepBodies)
    if not bodies:
        path = out_dir / f"{base_name}.stl"
        opts = export_mgr.createSTLExportOptions(root, str(path))
        opts.meshRefinement = mesh_setting
        export_mgr.execute(opts)
        written.append(str(path))
        return written

    for idx, body in enumerate(bodies):
        safe = body.name.replace("/", "_").replace(" ", "_") or f"body_{idx}"
        path = out_dir / f"{base_name}__{idx:02d}__{safe}.stl"
        opts = export_mgr.createSTLExportOptions(body, str(path))
        opts.meshRefinement = mesh_setting
        export_mgr.execute(opts)
        written.append(str(path))
    return written


def _export_assembled_step(design: adsk.fusion.Design, out_dir: Path, base_name: str) -> str:
    root = design.rootComponent
    export_mgr = design.exportManager
    path = out_dir / f"{base_name}.step"
    opts = export_mgr.createSTEPExportOptions(str(path), root)
    export_mgr.execute(opts)
    return str(path)


def _write_marker(out_dir: Path, name: str, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / name, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def run(context):  # noqa: ARG001 — Fusion entry-point signature
    """Spike 0.1 entry point. Runs once when the script/add-in starts."""
    ui = None
    out_dir = DEFAULT_OUTPUT_DIR
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface

        params_path = _resolve_params_path()
        if not params_path.exists():
            raise FileNotFoundError(
                f"params.json not found at {params_path}. "
                f"Set $FANOPT_PARAMS or copy params.example.json to that path."
            )
        with open(params_path) as f:
            params = json.load(f)

        out_dir = _resolve_output_dir(params)
        out_dir.mkdir(parents=True, exist_ok=True)

        product = app.activeProduct
        design = adsk.fusion.Design.cast(product)
        if design is None:
            raise RuntimeError(
                "No active Fusion Design document. Open the one-blade test file in "
                "Fusion BEFORE running this script (Fusion cannot open documents "
                "headlessly on macOS — see configs/fusion/README.md)."
            )

        applied, missing = _apply_user_parameters(
            design,
            params.get("user_parameters", {}),
        )
        design.computeAll()

        base_name = params.get("export_name", "blade_test")
        refinement = params.get("stl_refinement", "medium")
        export_step = bool(params.get("export_step", True))
        export_stl = bool(params.get("export_stl", True))

        stl_paths: list[str] = []
        step_path: str | None = None
        if export_stl:
            stl_paths = _export_per_body_stl(design, out_dir, base_name, refinement)
        if export_step:
            step_path = _export_assembled_step(design, out_dir, base_name)

        _write_marker(
            out_dir,
            "SPIKE_0_1_PASS.json",
            {
                "success": True,
                "params_path": str(params_path),
                "design_name": design.parentDocument.name,
                "applied_parameters": applied,
                "missing_parameters": missing,
                "stl_paths": stl_paths,
                "step_path": step_path,
                "output_dir": str(out_dir),
            },
        )

        if ui is not None:
            summary = (
                f"Spike 0.1 PASS\n\n"
                f"Applied {len(applied)} parameters; {len(missing)} missing.\n"
                f"STL files: {len(stl_paths)}\n"
                f"STEP file: {'yes' if step_path else 'no'}\n"
                f"Output: {out_dir}"
            )
            ui.messageBox(summary)

    except Exception as exc:  # noqa: BLE001 — capture everything for the driver
        tb = traceback.format_exc()
        try:
            _write_marker(
                out_dir,
                "SPIKE_0_1_FAIL.json",
                {
                    "success": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": tb,
                    "python_version": sys.version,
                    "params_path": str(_resolve_params_path()),
                },
            )
        except Exception:  # noqa: BLE001 — last-ditch; nothing else to do
            pass
        if ui is not None:
            ui.messageBox(f"Spike 0.1 FAIL:\n\n{tb}")


def stop(context):  # noqa: ARG001 — Fusion entry-point signature
    """No-op; Spike 0.1 is single-shot."""
    return None
