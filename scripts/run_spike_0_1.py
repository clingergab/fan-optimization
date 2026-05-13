#!/usr/bin/env python
"""Spike 0.1 driver — invoke Fusion headlessly and collect the result marker.

Spec reference: docs/plan_R11.md §Phase 0 Spike 0.1
Pass criterion: Fusion executes `configs/fusion/fan_script.py` against the
active document, applies User Parameters, and writes
`data/meshes/spike_0_1/SPIKE_0_1_PASS.json` within the timeout.

The driver does NOT open the Fusion document for you — Fusion's CLI on
macOS runs scripts against the already-active document. See
`configs/fusion/README.md` for the full procedure.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS_PATH = REPO_ROOT / "configs" / "fusion" / "params.json"
DEFAULT_SCRIPT_PATH = REPO_ROOT / "configs" / "fusion" / "fan_script.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "meshes" / "spike_0_1"
PASS_MARKER = "SPIKE_0_1_PASS.json"
FAIL_MARKER = "SPIKE_0_1_FAIL.json"

# Autodesk renamed the bundle from "Autodesk Fusion 360.app" to
# "Autodesk Fusion.app" in 2023+. Match both.
FUSION_BUNDLE_NAMES = ("Autodesk Fusion.app", "Autodesk Fusion 360.app")


def _discover_fusion_app() -> Path | None:
    """Return the first Autodesk Fusion .app bundle found in known macOS locations.

    Three install styles in the wild:
      1. Legacy: /Applications/Autodesk Fusion 360.app
      2. Rebranded legacy: /Applications/Autodesk Fusion.app
      3. Webdeploy (current default): ~/Library/Application Support/Autodesk/
         webdeploy/production/<hash>/Autodesk Fusion.app — the hash changes
         after each auto-update, so we glob the directory.
    """
    candidates: list[Path] = []
    for name in FUSION_BUNDLE_NAMES:
        candidates.append(Path("/Applications") / name)
    webdeploy = (
        Path.home()
        / "Library"
        / "Application Support"
        / "Autodesk"
        / "webdeploy"
        / "production"
    )
    if webdeploy.exists():
        for name in FUSION_BUNDLE_NAMES:
            candidates.extend(sorted(webdeploy.glob(f"*/{name}")))
    for c in candidates:
        if c.exists():
            return c
    return None


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fusion-app",
        type=Path,
        default=None,
        help=(
            "Path to Autodesk Fusion 360.app. Defaults to autodiscovery — "
            "first /Applications/Autodesk Fusion 360.app, then the Webdeploy "
            "install under ~/Library/Application Support/Autodesk/webdeploy/."
        ),
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=DEFAULT_PARAMS_PATH,
        help="params.json to feed the Fusion script (default: %(default)s)",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=DEFAULT_SCRIPT_PATH,
        help="Fusion script to execute (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where the script writes STL/STEP/marker (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Marker-poll timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--keep-stale-markers",
        action="store_true",
        help="Don't clear existing PASS/FAIL markers before launch (debug only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the launch command without executing",
    )
    return parser.parse_args(argv)


def log(msg: str) -> None:
    print(f"[spike_0_1] {msg}", file=sys.stderr)


def validate_prerequisites(args: argparse.Namespace) -> None:
    if sys.platform != "darwin":
        log("Spike 0.1 targets macOS; the --runScript path differs on Windows / Linux.")
    if args.fusion_app is None:
        discovered = _discover_fusion_app()
        if discovered is None:
            raise FileNotFoundError(
                "Autodesk Fusion 360.app not found. Searched:\n"
                "  /Applications/Autodesk Fusion 360.app\n"
                "  ~/Library/Application Support/Autodesk/webdeploy/production/*/Autodesk Fusion 360.app\n"
                "Install Fusion 360 or pass --fusion-app /path/to/Autodesk Fusion 360.app"
            )
        args.fusion_app = discovered
    elif not args.fusion_app.exists():
        raise FileNotFoundError(
            f"Fusion.app not found at {args.fusion_app}. "
            f"Install Fusion 360 or pass --fusion-app /path/to/Autodesk Fusion.app"
        )
    macos_dir = args.fusion_app / "Contents" / "MacOS"
    if not macos_dir.exists() or not any(macos_dir.iterdir()):
        raise FileNotFoundError(
            f"Fusion binary missing under {macos_dir}. Is this really a Fusion .app bundle?"
        )
    if not args.script.exists():
        raise FileNotFoundError(f"Fusion script not found at {args.script}")
    if not args.params.exists():
        example = args.params.with_name("params.example.json")
        hint = f"\n  cp {example} {args.params}" if example.exists() else ""
        raise FileNotFoundError(f"params.json not found at {args.params}{hint}")


def clear_stale_markers(out_dir: Path) -> None:
    for name in (PASS_MARKER, FAIL_MARKER):
        target = out_dir / name
        if target.exists():
            log(f"clearing stale marker {target.name}")
            target.unlink()


def launch_fusion(args: argparse.Namespace) -> subprocess.Popen | None:
    """Launch Fusion with --runScript via `open -ga` so it stays backgrounded."""
    cmd = [
        "/usr/bin/open",
        "-ga",
        str(args.fusion_app),
        "--args",
        "--runScript",
        str(args.script),
    ]
    env = os.environ.copy()
    env["FANOPT_PARAMS"] = str(args.params)
    env["FANOPT_OUTPUT_DIR"] = str(args.output_dir)

    log(f"Fusion: {args.fusion_app}")
    log(f"params:  {args.params}")
    log(f"script:  {args.script}")
    log(f"output:  {args.output_dir}")
    log("launch:  " + " ".join(cmd))

    if args.dry_run:
        log("dry-run: not executing")
        return None

    return subprocess.Popen(cmd, env=env)


def poll_for_marker(out_dir: Path, timeout: int) -> tuple[str, dict] | None:
    """Poll `out_dir` for either PASS or FAIL marker. Returns (kind, payload) or None."""
    deadline = time.monotonic() + timeout
    log(f"polling for marker (timeout {timeout} s)")
    while time.monotonic() < deadline:
        for name in (PASS_MARKER, FAIL_MARKER):
            target = out_dir / name
            if target.exists():
                try:
                    payload = json.loads(target.read_text())
                except json.JSONDecodeError:
                    time.sleep(0.5)  # mid-write; retry
                    continue
                return ("PASS" if name == PASS_MARKER else "FAIL"), payload
        time.sleep(1.0)
    return None


def report(kind: str, payload: dict, out_dir: Path) -> int:
    if kind == "PASS":
        stl_paths = payload.get("stl_paths", [])
        step_path = payload.get("step_path")
        missing = payload.get("missing_parameters", [])
        log(
            f"PASS — wrote {len(stl_paths)} STL "
            f"+ {'1' if step_path else '0'} STEP to {out_dir}"
        )
        if missing:
            log(f"  warning: {len(missing)} User Parameter(s) not found in active document:")
            for name, expr in missing:
                log(f"    - {name} = {expr}")
        return 0

    log("FAIL")
    log(f"  error: {payload.get('error', '(no message)')}")
    log(f"  type:  {payload.get('error_type', '(unknown)')}")
    tb = payload.get("traceback", "")
    if tb:
        sys.stderr.write("\n--- Fusion script traceback ---\n")
        sys.stderr.write(tb)
        sys.stderr.write("--- end traceback ---\n\n")
    return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_prerequisites(args)
    except FileNotFoundError as e:
        log(f"prerequisite check failed: {e}")
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.keep_stale_markers:
        clear_stale_markers(args.output_dir)

    proc = launch_fusion(args)
    if args.dry_run:
        return 0

    try:
        result = poll_for_marker(args.output_dir, timeout=args.timeout)
    finally:
        if proc is not None and proc.poll() is None:
            # `open -ga` returns quickly; the child is Fusion itself which we leave running.
            pass

    if result is None:
        log(f"timeout after {args.timeout} s without a marker.")
        log("  hint: open Fusion → Scripts and Add-Ins → run fan_script.py manually to see the error dialog.")
        return 3

    kind, payload = result
    return report(kind, payload, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
