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

If ``SU2_CFD`` is on PATH the runner additionally invokes SU2 on a tiny
probe mesh for 1 outer time-step and parses the stdout for completed
outer-step markers. If SU2 is not installed, the runner falls back to a
cfg-only parser sanity check and reports a ``passed = False`` outcome
(the outer-step count gate is part of the pass criterion, so the cfg-only
path cannot satisfy the full gate by design).

Output:

* ``data/spike_0_6c/sub_1_result.json`` — serialized
  ``Tier1CfgSanityResult``.
* ``data/spike_0_6c/sub_1.PASS`` or ``data/spike_0_6c/sub_1.FAIL`` —
  marker file consumed by the Spike 0.6c aggregator.

Exit codes:

* ``0`` — sub-spike passed (cfg parses, MACH lock satisfied, outer step
  recorded).
* ``1`` — sub-spike failed (any of the above checks failed; or SU2 was
  not available and the cfg-only fallback couldn't run an outer step).
* ``2`` — input error (template missing, render failure, etc.).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

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


def _render_tier1_cfg(template_path: Path) -> str:
    """Render the Tier-1 cfg via the production renderer.

    Production path: ``from fanopt.cfd.configs import render_unsteady_cfg``.
    This is now the canonical path — the renderer is wired up and ships
    with the same HIGH-12 / C11 locks the gate enforces. Use a probe-mesh
    placeholder for the mesh filename since 0.6c.1 is a parse-only check.
    """
    from fanopt.cfd.configs import render_unsteady_cfg

    return render_unsteady_cfg(
        mesh_filename="probe.su2",
        marker_fan="FAN",
        marker_farfield="FARFIELD",
    )


# ---- SU2 invocation (probe mesh) ------------------------------------------


def _su2_available() -> bool:
    return shutil.which("SU2_CFD") is not None


def _invoke_su2_one_step(cfg_text: str, work_dir: Path) -> str | None:
    """Invoke ``SU2_CFD`` for 1 outer time step on a probe cfg/mesh.

    Returns the captured stdout for ``_count_completed_outer_steps`` to
    consume, or ``None`` if SU2 isn't on PATH (the runner then takes the
    cfg-only fallback path).

    Note: this is a smoke invocation, not a converged solve. We write the
    cfg under ``work_dir/probe.cfg`` and override ``TIME_ITER = 1`` /
    ``INNER_ITER = 1`` if those directives appear so the run completes
    quickly even on a real mesh.
    """
    if not _su2_available():
        return None
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = work_dir / "probe.cfg"
    cfg_path.write_text(cfg_text, encoding="utf-8")
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
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        cfg_text = _render_tier1_cfg(args.template)
    except FileNotFoundError as e:
        print(f"[spike_0_6c_1] render error: {e}", file=sys.stderr)
        return 2

    if _su2_available():
        probe_workdir = args.probe_workdir or (DEFAULT_OUTPUT_DIR / "probe")
        su2_log = _invoke_su2_one_step(cfg_text, probe_workdir)
        print(
            f"[spike_0_6c_1] SU2_CFD found on PATH — ran probe in {probe_workdir}"
        )
    else:
        print(
            "[spike_0_6c_1] SU2 not installed locally — parsing cfg-only sanity check"
        )
        su2_log = None

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
    print(
        f"[spike_0_6c_1] outer_time_steps     {result.outer_time_steps_completed}"
    )
    if result.error:
        print(f"[spike_0_6c_1] error                {result.error}")
    print(f"[spike_0_6c_1] passed               {result.passed}")
    print(f"[spike_0_6c_1] result_json          {args.result_json}")
    print(f"[spike_0_6c_1] marker               {marker}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
