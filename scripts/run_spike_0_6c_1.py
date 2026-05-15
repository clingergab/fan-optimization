#!/usr/bin/env python
"""Spike 0.6c.1 — Tier-1 cfg sanity check runner.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.6c (lines 1839-1844);
protocol in docs/spike_0_6c_protocol.md.

Renders the canonical Tier-1 unsteady cfg
(``configs/su2/fan3d_unsteady.cfg.j2``) and validates it against the
Round-9 HIGH-12 (= C12) lock:

* ``MACH_NUMBER = 1e-9``
* ``FREESTREAM_OPTION = FREESTREAM_VELOCITY`` (primary syntax)
  OR ``REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE`` (fallback)

**Outer-step evidence (V1, post-2026-05-14 bugfix).** The full pass
criterion requires SU2 to have completed ≥ 1 outer time-step using the
locked Tier-1 numerics. Evidence can come from any of three paths, in
priority order:

1. ``--su2-history-csv PATH`` — point to a history.csv from ANY prior
   successful SU2 run using the same Tier-1 numerics (MACH=1e-9 +
   FREESTREAM_PRESS_EQ_ONE). Each row counts as one completed outer
   step. This is the recovery path when a prior Colab run produced
   evidence but the runtime is gone — the Drive history.csv suffices.
2. ``--su2-log-file PATH`` — point to a captured SU2 stdout file from a
   prior run; the parser scans for ``Time_Iter:`` / ``Time Iter:``
   markers.
3. Fresh probe-mesh invocation — if ``SU2_CFD`` is on PATH AND
   ``--probe-mesh PATH`` is provided, the runner copies the probe mesh
   into a scratch dir, renders the cfg pointing at it, runs SU2 for ≤ 1
   outer step (TIME_ITER override), and parses the captured stdout.

If none of the above is supplied, the runner falls back to a cfg-only
parser sanity check and reports ``passed = False`` (outer-step gate
unsatisfied). **The pre-2026-05-14 behavior of silently invoking SU2
with a missing ``probe.su2`` was the cause of the May 2026
``outer_steps=0`` failure that triggered this rewrite.**

Output:

* ``data/spike_0_6c/sub_1_result.json`` — serialized
  ``Tier1CfgSanityResult``.
* ``data/spike_0_6c/sub_1.PASS`` or ``data/spike_0_6c/sub_1.FAIL`` —
  marker file consumed by the Spike 0.6c aggregator.

Exit codes:

* ``0`` — sub-spike passed (cfg parses, MACH lock satisfied, outer step
  evidence present).
* ``1`` — sub-spike failed (any of the above checks failed, or no
  outer-step evidence was provided and no probe mesh was supplied).
* ``2`` — input error (template missing, render failure, requested
  evidence file does not exist, etc.).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from fanopt.cfd.configs import render_unsteady_cfg
from fanopt.cfd.spike_0_6c import (
    MACH_UNSTEADY_LOCK,
    Tier1CfgSanityResult,
    check_tier1_cfg_sanity,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = REPO_ROOT / "configs" / "su2" / "fan3d_unsteady.cfg.j2"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "spike_0_6c"
DEFAULT_RESULT_JSON = DEFAULT_OUTPUT_DIR / "sub_1_result.json"


# ---- cfg rendering --------------------------------------------------------


def _render_tier1_cfg(template_path: Path, mesh_filename: str = "probe.su2") -> str:
    """Render the Tier-1 cfg via the production renderer.

    ``mesh_filename`` is what the rendered cfg's MESH_FILENAME line points
    at. The default ``"probe.su2"`` is used for the cfg-only parser path;
    when ``--probe-mesh`` is supplied, the caller passes its filename so
    SU2 can actually find the mesh next to the rendered cfg.
    ``template_path`` is currently informational — the renderer reads the
    canonical template under configs/su2/; the parameter is kept so the
    CLI can route to alternate templates in future runs.
    """
    del template_path  # informational only; renderer reads canonical template
    return render_unsteady_cfg(
        mesh_filename=mesh_filename,
        marker_fan="FAN",
        marker_farfield="FARFIELD",
    )


# ---- evidence loaders -----------------------------------------------------


def _read_history_outer_step_count(history_csv_path: Path) -> int:
    """Count completed outer steps from an SU2 history.csv (one row each).

    Each row of an SU2 history.csv corresponds to one completed outer time
    step. We count distinct ``Time_Iter`` values to avoid double-counting
    if SU2 emitted multiple inner-iter rows per outer step.
    """
    text = history_csv_path.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text), skipinitialspace=True)
    outer_iter_values: set[int] = set()
    row_count = 0
    for raw in reader:
        row_count += 1
        for k, v in raw.items():
            if k is None or v is None:
                continue
            if k.strip().strip('"').lower() in {"time_iter", "iter", "outer_iter"}:
                try:
                    outer_iter_values.add(int(float(str(v).strip().strip('"'))))
                except (TypeError, ValueError):
                    continue
                break
    return len(outer_iter_values) or row_count


def _su2_log_from_history(history_csv_path: Path) -> str:
    """Synthesize a fake SU2 stdout from a history.csv for the existing
    parser in `_count_completed_outer_steps`.

    The parser scans for ``Time_Iter: N`` patterns. We emit one such line
    per distinct outer step in the history.
    """
    n = _read_history_outer_step_count(history_csv_path)
    return "".join(f"Time_Iter: {i}\n" for i in range(max(n, 0)))


# ---- SU2 invocation (probe mesh) ------------------------------------------


def _su2_available() -> bool:
    return shutil.which("SU2_CFD") is not None


def _invoke_su2_one_step(cfg_text: str, work_dir: Path, probe_mesh: Path) -> str:
    """Invoke ``SU2_CFD`` for 1 outer time step on a probe cfg/mesh.

    Returns the captured stdout for ``_count_completed_outer_steps`` to
    consume. Requires ``SU2_CFD`` to be on PATH (caller's responsibility)
    AND a real ``probe_mesh`` file that gets copied into ``work_dir``.

    Note: this is a smoke invocation, not a converged solve. The cfg is
    written under ``work_dir/probe.cfg`` and the probe mesh is copied
    alongside it as ``work_dir/<probe_mesh.name>`` so SU2 can resolve
    MESH_FILENAME relative to the cfg.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = work_dir / "probe.cfg"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    mesh_dest = work_dir / probe_mesh.name
    if probe_mesh.resolve() != mesh_dest.resolve():
        shutil.copy(probe_mesh, mesh_dest)
    try:
        result = subprocess.run(
            ["SU2_CFD", str(cfg_path)],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return f"[probe] SU2_CFD failed: {e}\n"
    return (result.stdout or "") + (result.stderr or "")


# ---- I/O helpers ----------------------------------------------------------


def _write_result(out_path: Path, result: Tier1CfgSanityResult) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec_reference": "docs/plan_R11.md §Phase 0 Spike 0.6c (sub-spike 0.6c.1)",
        "lock_reference": "Round-9 HIGH-12 (= C12)",
        "mach_unsteady_lock": MACH_UNSTEADY_LOCK,
        "result": asdict(result),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_marker(out_dir: Path, passed: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear any stale marker of the opposite polarity.
    for stale in ("sub_1.PASS", "sub_1.FAIL"):
        stale_path = out_dir / stale
        if stale_path.exists():
            stale_path.unlink()
    marker = out_dir / ("sub_1.PASS" if passed else "sub_1.FAIL")
    marker.write_text("")
    return marker


# ---- CLI ------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help="Path to the Tier-1 unsteady cfg .j2 template. Default: %(default)s.",
    )
    p.add_argument(
        "--result-json",
        type=Path,
        default=DEFAULT_RESULT_JSON,
        help="Where to write the sub_1 result JSON. Default: %(default)s.",
    )
    p.add_argument(
        "--marker-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write the PASS / FAIL marker. Default: %(default)s.",
    )
    p.add_argument(
        "--probe-workdir",
        type=Path,
        default=None,
        help="Optional scratch dir for the SU2 probe invocation.",
    )
    p.add_argument(
        "--probe-mesh",
        type=Path,
        default=None,
        help=(
            "Path to a pre-built probe mesh (.su2). Required when invoking "
            "SU2 fresh — the runner copies this file next to the rendered "
            "cfg so SU2 can resolve MESH_FILENAME. Generate via "
            "`scripts/gen_benchmark_meshes.py --kind probe`."
        ),
    )
    p.add_argument(
        "--su2-log-file",
        type=Path,
        default=None,
        help=(
            "Path to a captured SU2 stdout from a prior successful run on "
            "the same Tier-1 numerics. Parsed for outer-step evidence; "
            "skips the fresh SU2 invocation entirely."
        ),
    )
    p.add_argument(
        "--su2-history-csv",
        type=Path,
        default=None,
        help=(
            "Path to a history.csv from a prior successful SU2 run with "
            "the same Tier-1 numerics (MACH=1e-9 + FREESTREAM_PRESS_EQ_ONE). "
            "Row count is used as outer-step evidence. The recovery path "
            "when a prior Colab run produced evidence but the runtime is "
            "gone — the Drive history.csv suffices, no Colab re-run needed."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Evidence resolution: prefer prior-run evidence over a fresh SU2 invocation.
    # Priority: --su2-history-csv > --su2-log-file > fresh probe invocation.
    su2_log: str | None = None
    evidence_source = "none (cfg-only fallback)"

    if args.su2_history_csv is not None:
        if not args.su2_history_csv.exists():
            print(
                f"[spike_0_6c_1] --su2-history-csv path does not exist: " f"{args.su2_history_csv}",
                file=sys.stderr,
            )
            return 2
        su2_log = _su2_log_from_history(args.su2_history_csv)
        evidence_source = f"prior-run history.csv ({args.su2_history_csv})"
    elif args.su2_log_file is not None:
        if not args.su2_log_file.exists():
            print(
                f"[spike_0_6c_1] --su2-log-file path does not exist: " f"{args.su2_log_file}",
                file=sys.stderr,
            )
            return 2
        su2_log = args.su2_log_file.read_text(encoding="utf-8")
        evidence_source = f"prior-run SU2 stdout ({args.su2_log_file})"
    elif args.probe_mesh is not None:
        if not args.probe_mesh.exists():
            print(
                f"[spike_0_6c_1] --probe-mesh path does not exist: {args.probe_mesh}",
                file=sys.stderr,
            )
            return 2
        if not _su2_available():
            print(
                "[spike_0_6c_1] --probe-mesh supplied but SU2_CFD not on PATH; "
                "cannot run fresh probe invocation",
                file=sys.stderr,
            )
            return 2
        probe_workdir = args.probe_workdir or (DEFAULT_OUTPUT_DIR / "probe")
        cfg_text = _render_tier1_cfg(args.template, mesh_filename=args.probe_mesh.name)
        su2_log = _invoke_su2_one_step(cfg_text, probe_workdir, args.probe_mesh)
        evidence_source = f"fresh SU2 probe ({probe_workdir})"

    # The cfg-only parser check always runs (regardless of evidence source).
    try:
        # Re-render with the default mesh placeholder if we didn't already render
        # for a probe invocation. The cfg-text is what gets parsed for the lock
        # invariants; the actual MESH_FILENAME doesn't affect that.
        if args.probe_mesh is None:
            cfg_text = _render_tier1_cfg(args.template)
    except FileNotFoundError as e:
        print(f"[spike_0_6c_1] render error: {e}", file=sys.stderr)
        return 2

    print(f"[spike_0_6c_1] outer-step evidence: {evidence_source}")

    result = check_tier1_cfg_sanity(
        cfg_text=cfg_text,
        su2_log=su2_log,
        cfg_path=str(args.template),
    )

    _write_result(args.result_json, result)
    marker = _write_marker(args.marker_dir, result.passed)

    print(f"[spike_0_6c_1] cfg_path             {result.cfg_path}")
    print(f"[spike_0_6c_1] parsed_ok            {result.parsed_ok}")
    print(
        f"[spike_0_6c_1] mach_value           {result.mach_value!r} "
        f"(lock = {MACH_UNSTEADY_LOCK!r})"
    )
    print(f"[spike_0_6c_1] freestream_option    {result.freestream_option!r}")
    print(f"[spike_0_6c_1] ref_dimensionaliz.   {result.ref_dimensionalization!r}")
    print(f"[spike_0_6c_1] outer_time_steps     {result.outer_time_steps_completed}")
    if result.error:
        print(f"[spike_0_6c_1] error                {result.error}")
    print(f"[spike_0_6c_1] passed               {result.passed}")
    print(f"[spike_0_6c_1] result_json          {args.result_json}")
    print(f"[spike_0_6c_1] marker               {marker}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
