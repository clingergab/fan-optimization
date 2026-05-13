# Spike 0.1 — Fusion headless add-in workflow on macOS

**Spec reference:** `docs/plan_R11.md §Phase 0 Spike 0.1` (lines 1785-1789)

**Question:** Can Fusion 360 be driven headlessly from a Python script on
macOS — read `params.json`, regenerate the model, export per-blade STL files
and a STEP file?

**Pass criterion (Spike 0.1):** Fusion launches via the CLI flag, executes
`fan_script.py` against an open one-blade test file, applies User Parameters,
and writes a `SPIKE_0_1_PASS.json` marker plus the requested STL / STEP files
without manual UI interaction.

**Fallback if it fails:** CadQuery becomes the sole geometry backend. Fusion
keeps a role as viewer + post-print manufacturable cleanup but never appears
in the optimization loop.

---

## What's here

| File | Purpose |
|---|---|
| `fan_script.py` | The Fusion script. `run(context)` reads `params.json`, applies User Parameters on the active document, exports STL (per body) + STEP, writes a PASS/FAIL marker. |
| `fan_script.manifest` | Add-in manifest. Use only if you install as an add-in rather than running as a one-shot script. |
| `params.example.json` | Example parameter file. Copy to `params.json` before running. |
| `../../scripts/run_spike_0_1.py` | Headless driver. Validates prerequisites, invokes Fusion with `--runScript`, polls for the marker, reports pass/fail. |
| `../../docs/phase_logs/spike_0_1.md` | Protocol log — fill in as you run. |

---

## Prerequisites

1. **Fusion 360 installed** at the default macOS location:
   `/Applications/Autodesk Fusion 360.app` (or the Fusion 360 admin install).
   The driver script accepts an override path via `--fusion-app`.
2. **A one-blade test file open in Fusion** with these User Parameters
   defined (Fusion → Modify → Change Parameters):
   - `L_blade`, `panel_thickness`, `rib_base_width`, `rib_tip_width`,
     `rib_thickness`, `hub_radius`, `rib_tip_taper`, `pivot_center_x`,
     `click_chamfer_dim`.

   If your test file uses different names, edit `params.example.json` (or
   `params.json` after you copy it) to match — unknown names are recorded
   but not fatal, so partial coverage still produces an export.

   *If you don't yet have a test file:* create a minimal one with a single
   sketched rectangle (`L_blade` × something) extruded by `panel_thickness`.
   Spike 0.1 only validates the workflow plumbing; the geometry detail
   matters in Spike 0.7 (generative-geometry sanity check), not here.

3. **Save the test file once before running** so Fusion has a parent
   document to compute against.

---

## How to run (3-minute procedure)

```bash
cd ~/Projects/fan-optimization

# 1. Copy the example params to the live name (only once)
cp configs/fusion/params.example.json configs/fusion/params.json

# 2. (Optional) edit configs/fusion/params.json with the parameter
#    values you want this run to test

# 3. Open your one-blade test file in Fusion 360 and make it the active
#    document. (Fusion cannot open documents headlessly on macOS — see
#    "Headless caveats" below.)

# 4. Run the driver
python scripts/run_spike_0_1.py

#    The driver:
#      a. confirms Fusion.app exists
#      b. confirms configs/fusion/params.json exists
#      c. clears any stale marker in data/meshes/spike_0_1/
#      d. invokes Fusion with `--runScript configs/fusion/fan_script.py`
#      e. polls for SPIKE_0_1_PASS.json or SPIKE_0_1_FAIL.json
#      f. prints the marker summary and exits with 0 (pass) / 1 (fail)
```

Expected output on success:

```
[spike_0_1] Fusion: /Applications/Autodesk Fusion 360.app/Contents/MacOS/Autodesk Fusion 360
[spike_0_1] params:  configs/fusion/params.json
[spike_0_1] launching Fusion --runScript ...
[spike_0_1] polling for marker (timeout 120 s)
[spike_0_1] PASS — wrote 1 STL + 1 STEP to data/meshes/spike_0_1/
```

---

## Headless caveats (the truth about macOS Fusion)

Fusion 360 on macOS is **not fully headless**. The CLI workflow:

1. Fusion's GUI launches (briefly, can be backgrounded with `open -g`).
2. Fusion loads whatever document is set as the active document — it does
   NOT auto-open a path passed as an argument.
3. The `--runScript` flag executes the named `.py` against the active
   document's Python API context.
4. The script writes its outputs to disk.

Consequences:

- **You must have the test document open in Fusion before you launch
  the driver.** That's the one piece of manual setup. Once it's open,
  the rest is scripted.
- A Fusion window will flash on screen. The driver uses `open -ga "..."
  --args --runScript ...` so it stays backgrounded, but it isn't true
  invisible-headless.
- If you log out / lock the screen during the run, Fusion may stall. Run
  the spike with the user session active.

If Fusion proves too brittle for unattended Phase 4 / Phase 5 batch use,
the spike fail-action is the locked fallback per `§Phase 0 Spike 0.1`:
**CadQuery becomes the sole geometry backend**, Fusion drops to
viewer/cleanup. The fallback path is pre-wired — `src/fanopt/geometry/`
already plans on CadQuery as primary per §3.2 / §9.7.

---

## AppleScript fallback (not implemented for the spike)

The spec lists "fall back to AppleScript UI automation if needed." That
path would script the Fusion GUI through `osascript`:

```applescript
tell application "Autodesk Fusion 360"
    activate
    -- click menu items, type into the parameter dialog, save, export
end tell
```

It's documented here for completeness but **not implemented in this
commit** — if `--runScript` works (the spec's primary path), AppleScript
adds complexity without benefit. Implement only if Spike 0.1's primary
path fails and the fallback decision is to retain Fusion as backend.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `SPIKE_0_1_FAIL.json: No active Fusion Design document` | The test file isn't open / isn't the active tab | Switch to the test file's tab in Fusion before launching the driver |
| `params.json not found` | Didn't copy the example | `cp configs/fusion/params.example.json configs/fusion/params.json` |
| `missing_parameters` in PASS marker is non-empty | Your User Parameter names differ from the keys in `params.json` | Either rename the Fusion User Parameters, or edit `params.json` to match |
| Driver times out after 120 s | Fusion is hung / your script crashed before reaching `_write_marker` | Open Fusion → Scripts and Add-Ins → Run `fan_script.py` manually; a dialog will show the traceback |
| `Authorization required` dialog from macOS | First-time launch consent | Click Allow once; persists |

---

## What to record in `docs/phase_logs/spike_0_1.md`

After the run, fill in:
- Pass / fail.
- Wall-clock time from driver launch to marker.
- Any `missing_parameters` (schema drift to fix before Phase 1).
- One-line manufacturability eyeball check on the exported STL.

A pass closes Spike 0.1 and unlocks the parallel Spike 0.2-0.5 workflow.
A fail triggers the fallback decision in the next phase-log entry.
