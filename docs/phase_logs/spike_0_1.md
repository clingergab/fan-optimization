# Spike 0.1 — Fusion headless add-in workflow on macOS

**Spec reference:** `docs/plan_R11.md §Phase 0 Spike 0.1` (lines 1785-1789).

**Question:** Can Fusion 360 be driven headlessly from a Python script on
macOS — read `params.json`, regenerate the model, export per-blade STLs +
STEP?

**Artifacts shipped with this spike:**
- `configs/fusion/fan_script.py` — the Fusion script
- `configs/fusion/fan_script.manifest` — add-in manifest (optional path)
- `configs/fusion/params.example.json` — schema reference
- `configs/fusion/README.md` — full test directions
- `scripts/run_spike_0_1.py` — headless driver

**Pass criterion:** `SPIKE_0_1_PASS.json` written by the script within the
driver's timeout, with at least one STL and one STEP output present and a
manufacturability eyeball check of the STL.

---

## Run log

| Field | Value |
|---|---|
| Date | _to be filled_ |
| Operator | _to be filled_ |
| Fusion version | _e.g., 2.0.20000_ |
| Test file name | _e.g., blade_test_v0.f3d_ |
| Driver invocation | `python scripts/run_spike_0_1.py` |
| Wall-clock (launch → marker) | _seconds_ |
| Result | _PASS / FAIL_ |
| `applied_parameters` count | _N_ |
| `missing_parameters` count | _N_ |
| STL files written | _list_ |
| STEP file written | _path_ |
| Manufacturability eyeball | _OK / issues described_ |

---

## Fallback decision (only if FAIL)

If Spike 0.1 fails the pass criterion, record here which fallback the
campaign takes per the spec:

- [ ] **CadQuery becomes the sole geometry backend.** Fusion drops to
      viewer / post-print cleanup. No further Fusion work in V1.
- [ ] **AppleScript UI automation retry.** Implement and re-run Spike 0.1
      before committing to the CadQuery-only fallback. (Only if there is
      a specific reason to keep Fusion in the loop, e.g., a feature
      CadQuery cannot reproduce.)

Decision rationale:

> _Fill in if needed._

---

## Findings (post-run)

> _What worked, what surprised you, what to address before Phase 1._

---

## Sign-off

- [ ] Pass criterion met.
- [ ] STL output inspected for manufacturability.
- [ ] `missing_parameters` (if any) reconciled — either the Fusion test
      file's parameter names were corrected, or `params.json` was
      updated, or the discrepancy is documented as intentional.
- [ ] This log committed to `docs/phase_logs/`.
- [ ] Spike 0.1 closed in the phase tracker.
