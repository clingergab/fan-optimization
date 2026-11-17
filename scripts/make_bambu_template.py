#!/usr/bin/env python3
"""Have Bambu Studio export a project (.3mf) for a printer / process / filament from its system presets.

The result is a template for :func:`fanopt.geometry.bambu_3mf.write_bambu_project` (``--template`` of
``build_print_blades.py``): Bambu Studio itself writes every printer-specific setting, so a printer with a
different layout (e.g. the two-nozzle H2C) gets a consistent project rather than values patched onto
another printer's. Any model works as ``--model``; the blade build replaces it.

    python3 scripts/make_bambu_template.py --model print_ready/P2C_blade_00_rebuilt.3mf \\
        --machine "Bambu Lab H2C 0.4 nozzle" --process "0.20mm Standard @BBL H2C" \\
        --filament "Generic PETG @BBL H2C 0.4 nozzle" --out print_ready/rebuild/H2C_petg_template.3mf
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from fanopt.geometry.bambu_3mf import resolve_bambu_preset

BAMBU_APP = Path("/Applications/BambuStudio.app/Contents/MacOS/BambuStudio")
BAMBU_PROFILES = Path("/Applications/BambuStudio.app/Contents/Resources/profiles/BBL")
_EXPORT_NAME = "template.3mf"


def run(
    *,
    model: Path,
    out: Path,
    machine: str,
    process: str,
    filament: str,
    bambu: Path = BAMBU_APP,
    profiles: Path = BAMBU_PROFILES,
) -> Path:
    """Flatten the three presets, slice ``model`` with them and keep Bambu Studio's exported project."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        paths = {}
        for kind, name in (("machine", machine), ("process", process), ("filament", filament)):
            paths[kind] = work / f"{kind}.json"
            paths[kind].write_text(json.dumps(resolve_bambu_preset(profiles, kind, name)))
        # Bambu Studio prefixes --export-3mf with --outputdir, so the export name must be bare.
        cmd = [
            str(bambu.resolve()),
            "--load-settings", f"{paths['machine']};{paths['process']}",
            "--load-filaments", str(paths["filament"]),
            "--slice", "1", "--outputdir", str(work), "--export-3mf", _EXPORT_NAME, str(model.resolve()),
        ]  # fmt: skip
        # cwd=work: Bambu Studio also drops a result.json into its working directory.
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=work)
        exported = work / _EXPORT_NAME
        if proc.returncode != 0 or not exported.exists():
            raise RuntimeError(
                f"Bambu Studio export failed ({proc.returncode}): {proc.stdout[-2000:]}"
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported), out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--model", type=Path, required=True, help="any STL/3MF to slice (replaced later)"
    )
    ap.add_argument("--out", type=Path, required=True, help="template .3mf to write")
    ap.add_argument("--machine", required=True, help='e.g. "Bambu Lab H2C 0.4 nozzle"')
    ap.add_argument("--process", required=True, help='e.g. "0.20mm Standard @BBL H2C"')
    ap.add_argument("--filament", required=True, help='e.g. "Generic PETG @BBL H2C 0.4 nozzle"')
    ap.add_argument("--bambu", type=Path, default=BAMBU_APP, help="Bambu Studio executable")
    ap.add_argument("--profiles", type=Path, default=BAMBU_PROFILES, help="Bambu system preset dir")
    a = ap.parse_args(argv)
    out = run(
        model=a.model, out=a.out, machine=a.machine, process=a.process, filament=a.filament,
        bambu=a.bambu, profiles=a.profiles,
    )  # fmt: skip
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
