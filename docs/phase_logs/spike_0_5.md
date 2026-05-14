# Spike 0.5 — Single-blade fabrication-noise floor

**Spec reference:** `docs/plan_R11.md §Phase 0 Spike 0.5` (lines 1826-1831).

**Depends on:** Spike 0.4 (click-feature geometry validated on the same
printer + filament + slicer profile).

**Question:** Across three identical printed copies of a single
representative blade, each assembled into the baseline-10 fan with 9
unchanged Spike 0.3 baseline blades, is the coefficient of variation of
J_fan-proxy (and of mass, 10-point caliper dimensions, and three-point
bend deflection) below 5% of the mean?

**Procedure:** `docs/spike_0_5_protocol.md`.

**Artifacts shipped with this spike:**
- `docs/spike_0_5_protocol.md` — operator procedure.
- `src/fanopt/physical/fab_noise.py` — Spike 0.5 library (CV + per-metric
  + roll-up dataclasses).
- `scripts/run_spike_0_5.py` — CLI analyzer.
- `tests/test_physical/test_fab_noise.py` — analytic-known CV + gate tests.
- `tests/test_scripts/test_run_spike_0_5.py` — CLI smoke tests.
- `data/spike_0_5/measurements.template.csv` — input CSV template.

**Pass criterion:**
1. `j_fan_cv.cv_pct < 5%` across the three single-blade fans (the spec
   criterion).
2. `mass_cv.cv_pct`, `dimension_cv.cv_pct`, `bend_cv.cv_pct` each `< 5%`
   (per-metric breakdown so a metric-specific failure flags the right
   fabrication knob to tune).

---

## Run log

| Field | Value |
|---|---|
| Date | _to be filled_ |
| Operator | _to be filled_ |
| Printer | _e.g., Prusa MK4_ |
| Filament spool | _brand + batch ID_ |
| Slicer profile rev | _commit hash or filename_ |
| Blade geometry | _production click / V1 rib-tab fallback_ |
| Build plate config | _single plate × 3 / one plate per blade_ |
| Room temp / humidity (mean) | _°C / %RH_ |
| Spike 0.3 baseline rev | _commit hash for the 9 baseline blades_ |
| Reserved slot index | _e.g., slot 5_ |
| Blade 1 mass (g) | _to be filled_ |
| Blade 2 mass (g) | _to be filled_ |
| Blade 3 mass (g) | _to be filled_ |
| `mass_cv.cv_pct` | _to be filled_ |
| `dimension_cv.cv_pct` (aggregate of 10 points) | _to be filled_ |
| `bend_cv.cv_pct` | _to be filled_ |
| Blade 1 J_fan-proxy | _to be filled_ |
| Blade 2 J_fan-proxy | _to be filled_ |
| Blade 3 J_fan-proxy | _to be filled_ |
| `j_fan_cv.cv_pct` (the published floor) | _to be filled_ |
| `overall_passed` | _true / false_ |
| Drive / JSONL ledger row ID | _to be filled_ |

Analyzer invocation:

```
python scripts/run_spike_0_5.py \
  --measurements data/spike_0_5/measurements.csv \
  --out data/spike_0_5/results.json
```

---

## Diagnostics if any sub-gate fails

| Symptom | Likely cause | Fix |
|---|---|---|
| `mass_cv` ≥ 5% | Flow-rate drift between prints | Recalibrate slicer flow rate; re-print all three. |
| `dimension_cv` ≥ 5% | Linear / pressure advance under-tuned | Run slicer's linear-advance calibration tower; re-print. |
| `bend_cv` ≥ 5% | Inconsistent layer bonding (fan curve / first-layer squish) | Re-tune part-cooling fan curve; re-print. |
| `j_fan_cv` ≥ 5%, others pass | The 9 baseline blades drifted between runs | Re-seat each baseline blade with reference jig; re-run Step 4. |
| `j_fan_cv` ≥ 5%, others also fail | Genuine fabrication noise above spec | Tighten print process OR raise gain threshold per memo issue #16. |

---

## Findings (post-run)

> _Achieved CV per metric; whether the print process was tightened or the
> gain threshold was raised; anything that should propagate to the
> protocol doc, the analyzer, or the gain-vs-floor comparison rule in
> Phase 6._

---

## Sign-off

- [ ] Spike 0.4 closed; click-feature geometry validated.
- [ ] 3 single blades printed identically on the locked slicer profile.
- [ ] Per-blade mass / 10-point dimensions / bend deflection recorded.
- [ ] Each blade assembled into the baseline-10 fan; J_fan-proxy measured
      via Spike 0.3 protocol.
- [ ] `data/spike_0_5/measurements.csv` filled and committed.
- [ ] `data/spike_0_5/results.json` written; `overall_passed` recorded.
- [ ] Achieved `j_fan_cv.cv_pct` recorded in Drive / JSONL ledger.
- [ ] If `overall_passed = false`: mitigation decision logged
      (print-process tightening vs. memo issue #16 gain-threshold raise).
- [ ] This log committed to `docs/phase_logs/`.
- [ ] Spike 0.5 closed in `docs/phase_checklist.md`.
