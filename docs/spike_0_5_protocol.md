# Spike 0.5 — Single-blade fabrication-noise floor protocol

**Spec reference:** `docs/plan_R11.md §Phase 0 Spike 0.5` (lines ~1826-1831).

**Depends on:** Spike 0.4 (`docs/spike_0_4_protocol.md`) must be closed — the
click-feature chamfer + detent geometry, clearance, and engagement force are
validated on the same printer + filament + slicer profile this spike will use.
If Spike 0.4's force-balance gate auto-armed the V1 fallback
(`v1_lock_fallback_armed = true`), print the three test blades with the
rib-tab fallback geometry; otherwise use the production click chamfer.

**Why this exists.** Panel topology + airfoil optimization is projected to
lift `J_fan` by 15-30% over the Spike 0.3 flat-panel baseline. That gain must
clear the printer's part-to-part noise floor: if printing the *same* blade
three times produces 8% variation in J_fan-proxy, then a 15% claimed gain is
indistinguishable from fabrication noise on the low end. This spike measures
the floor — coefficient of variation of J_fan-proxy across three identical
copies of a single representative blade — and locks the achieved value into
the Drive / JSONL ledger so every subsequent J_fan delta is compared against
it.

**Why three single blades, not three full fans.** Printing three full fans
conflates blade-to-blade fabrication noise with assembly-to-assembly noise
(pivot-pin slop, click-feature seating, panel stack alignment). The
quantity of interest is the *fabrication* component alone, so we hold the
assembly constant: 9 unchanged Spike 0.3 baseline blades are reused across
all three runs, and only the slot-N blade varies. The three J_fan-proxy
measurements then differ only by the fabrication of that one printed blade.

---

## Apparatus

- **Calipers** — 0.01 mm resolution (digital preferred). Used to measure the
  10-point dimensional grid on each printed blade.
- **Jewelry scale** — 0.01 g resolution. Used to weigh each printed single
  blade.
- **Three-point bend rig** — two-support span (e.g., 150 mm), known mid-span
  load (e.g., 100 g calibrated mass), dial indicator or digital indicator at
  0.01 mm resolution. The rig reused across all three blades — same supports,
  same load, same indicator position — so the per-blade deflection is
  comparable.
- **Anemometer + IMU rig from Spike 0.3** — reuse the J_fan-proxy bench
  exactly as configured for the Spike 0.3 baseline measurement (9-point
  anemometer grid at 300 mm, IMU on the handle, same operator stroke
  profile). Re-running the Spike 0.3 protocol is the entire Step 4 below.
- **Spike 0.3 baseline fan disassembled** — 9 unchanged baseline blades
  retained on the pivot pin; one slot reserved for the test blade. Keep a
  labelled storage tray so the 9 baseline blades go back into the same
  positions across all three runs.

---

## Step 1 — Print three identical single blades

Three copies of one representative blade, printed:

- Same printer.
- Same filament spool (do not swap mid-batch).
- Same slicer profile (linear / pressure advance, flow rate, layer height,
  fan curve — all locked to the Spike 0.4 validated profile).
- Back-to-back, ideally on the same build plate, ideally in the same print
  position rotated through three identical bed locations or on a single
  triple-replicate plate.

Acceptance: visual inspection — no string-and-blob, no warp, no missed
layers. If any blade looks cosmetically off, print one more and discard
the bad copy. The three blades that go forward must be visually
indistinguishable.

The "representative blade" geometry: use the production click chamfer +
detent from the Spike 0.4-validated geometry. If Spike 0.4 auto-armed the
V1 rib-tab fallback, use the fallback geometry. **Do not** start
fabricating with a panel topology under active BO — this spike measures
the floor for the *baseline* blade so the floor is comparable across all
future J_fan deltas.

---

## Step 2 — Per-blade quality measurements

For each of the three blades:

1. **Mass.** Weigh on the jewelry scale, record to 0.01 g. Repeat once to
   confirm; if the two readings differ by > 0.02 g, zero the scale and
   re-weigh.
2. **10-point caliper grid.** Measure dimensions at 10 nominally identical
   anatomical positions. The 10 positions must be the same across all three
   blades — operator picks the convention up front (e.g., panel thickness
   at the 5 hub-side reference notches and 5 tip-side reference notches)
   and records it on the worksheet. Locking the convention is what makes
   per-point CV meaningful.
3. **Three-point bend.** Place the blade on the rig supports, apply the
   calibrated mid-span load, read the deflection at the indicator after
   30 s settling. Remove the load and confirm the blade returns to zero
   deflection (within 0.02 mm). Record the loaded deflection.

Worksheet rows (one row per blade):

```
blade_id, mass_g, d1_mm, d2_mm, ..., d10_mm, bend_deflection_mm, j_fan_proxy, notes
```

`j_fan_proxy` is left blank at this stage — filled in after Step 4.

---

## Step 3 — Assemble each blade into the baseline-10 fan

For each of the three blades (one at a time):

1. Disassemble the current 10-blade fan if assembled.
2. Take the 9 unchanged Spike 0.3 baseline blades from the storage tray and
   slot them back in their labelled positions.
3. Slot the **new printed blade** into the reserved slot (same slot index
   across all three runs — pick slot 5 or 6, mid-fan, to keep the symmetry
   roughly intact).
4. Re-pin the pivot.

Sanity check before each run: deploy / fold once by hand. The click feature
must engage cleanly at every adjacent pair. If the new blade fails to seat,
return to Spike 0.4 — the click clearance has drifted.

---

## Step 4 — Measure J_fan-proxy on each variant

Run the **Spike 0.3 J_fan-proxy protocol** (anemometer 9-point grid × plane
area, IMU-normalized) for each of the three variants. Hold every variable
fixed across all three runs:

- Same operator, same stroke profile (use the IMU envelope from the Spike
  0.3 baseline as the reference; the operator targets matching ω_max ±
  0.5 rad/s).
- Same room temperature and humidity (record both — if they drift > 2 °C
  / 10% RH between runs, pause and let the room re-equilibrate before
  continuing).
- Same anemometer + IMU rig setup, same 300 mm plane offset, same 9-point
  jig position.
- 3 strokes per variant; take the mean.

Record the per-variant `j_fan_proxy` value into the `j_fan_proxy` column of
the Step 2 worksheet. Final CSV should now have all four numerical columns
populated for every row.

---

## Step 5 — Run the analyzer

Fill `data/spike_0_5/measurements.csv` (one row per blade — see
`measurements.template.csv`) and run:

```
python scripts/run_spike_0_5.py \
  --measurements data/spike_0_5/measurements.csv \
  --out data/spike_0_5/results.json
```

The analyzer reports:

| Metric | Aggregation | Pass criterion |
|---|---|---|
| `mass_cv` | CV across the three printed-blade masses | `< 5%` |
| `dimension_cv` | per-point CV across blades, averaged across the 10 points | `< 5%` |
| `bend_cv` | CV across the three three-point bend deflections | `< 5%` |
| `j_fan_cv` | CV across the three assembled-fan J_fan-proxy values | `< 5%` |
| `overall_passed` | logical AND of the four above | `true` for PASS |

Exit codes:
- `0` — overall PASS (spike closes).
- `1` — overall FAIL (one or more metrics' CV ≥ 5%; see mitigation).
- `2` — input error (missing file / column / non-numeric value, < 3 blade
  rows, fewer than 10 dimension columns).

---

## Step 6 — Record the achieved CV in the Drive / JSONL ledger

Per the spec mitigation rule, **the achieved J_fan CV is the published
fabrication-noise floor**. Append a row to the Drive / JSONL ledger with:

- spike id (`spike_0_5`)
- date
- printer + filament + slicer-profile revision
- blade geometry (production click vs. V1 rib-tab fallback)
- achieved `j_fan_cv.cv_pct`
- achieved `mass_cv.cv_pct`, `dimension_cv.cv_pct`, `bend_cv.cv_pct`
- `overall_passed`

All subsequent J_fan deltas (Phase 4-6 Pareto top-3, Phase 6 verification
prints) must be compared against the recorded `j_fan_cv.cv_pct`. A claimed
gain whose magnitude does not clear `2 × j_fan_cv.cv_pct` is not
distinguishable from fabrication noise and must be re-measured with more
replicates or rejected.

---

## Mitigation if any CV ≥ 5%

The spec gives two decisions when the floor is too noisy:

### (a) Tighten the print process

If `mass_cv` or `dimension_cv` is the offender — i.e., the fabrication is
varying *physically* — return to the Spike 0.4 fallback tree and re-tune
slicer parameters:

- **Linear advance / pressure advance** — under-tuned linear advance gives
  inconsistent extrusion at corners, which shows up as per-point dimensional
  CV. Run the slicer's linear-advance calibration tower and update the
  profile.
- **Flow rate** — over- / under-extrusion gives mass CV proportional to
  flow-rate error. Re-calibrate against a 20 × 20 × 20 mm test cube.
- **Bed adhesion / first-layer squish** — drives warp at the tip, which
  shows up as bend-deflection CV (the printed neutral axis shifts).
- **Fan curve** — too much or too little part cooling drives layer-to-layer
  bonding variation, also visible as bend CV.

After re-tuning, re-print three new blades and re-run Steps 2-5.

### (b) Commit only to gains > 15% (memo issue #16)

If tightening the print process is not viable in the available time budget,
the floor is locked at the measured value and the gain threshold is raised.
The default 15% lower-bound of the projected J_fan-gain range becomes the
hard commit threshold: any Pareto candidate whose measured J_fan gain over
the Spike 0.3 baseline does not exceed 15% **plus the fabrication-noise
floor** is rejected.

Concretely: if `j_fan_cv.cv_pct = 8%`, then the effective publish threshold
becomes `15% + 8% = 23%` and any candidate below 23% gain is not committed.
File the decision in memo issue #16 with the achieved CV pinned alongside.

---

## Fallback decision tree

| Failing sub-gate | Likely cause | Fix |
|---|---|---|
| `mass_cv.passed = false` | Flow-rate drift between blades | Recalibrate slicer flow rate; re-print. |
| `dimension_cv.passed = false` | Linear / pressure advance under-tuned | Run slicer's linear-advance calibration; re-print. |
| `bend_cv.passed = false` | Inconsistent layer bonding (fan curve / first-layer) | Re-tune part-cooling fan curve; re-print. |
| `j_fan_cv.passed = false` (but others pass) | Assembly variation leaking through (the 9 baseline blades drifted between runs) | Re-seat each baseline blade with reference jig; re-run Step 4 only. |
| `j_fan_cv.passed = false` and others fail too | Genuine fabrication-noise floor exceeds spec | Tighten print process per (a) above, OR raise gain threshold per (b) (memo issue #16). |

---

## What this rig is reused for

- **Phase 6 verification (3-copy noise recheck):** every top-3 Pareto
  prototype is printed in triplicate and the same CV gate applies. A
  candidate that passes Spike 0.5's CV gate on its own geometry is
  certified for the J_fan delta measurement; a candidate that fails has
  its print process re-tuned before its J_fan is published.
- **The achieved `j_fan_cv.cv_pct` floor** is the divisor every subsequent
  J_fan delta divides by to compute "gain in units of noise floors". Phase
  6 step 79's `≥ 15% gain` check is implicitly `gain ≥ 15% AND
  gain ≥ 2 × j_fan_cv.cv_pct`.

---

## Sign-off

- [ ] Spike 0.4 closed; click-feature geometry validated.
- [ ] 3 single blades printed identically on locked slicer profile.
- [ ] Per-blade mass / 10-point dimensions / bend deflection recorded.
- [ ] Each blade assembled into the baseline-10 fan; J_fan-proxy measured
      via Spike 0.3 protocol.
- [ ] `data/spike_0_5/measurements.csv` filled.
- [ ] `data/spike_0_5/results.json` written; `overall_passed` confirmed.
- [ ] Achieved CVs recorded in Drive / JSONL ledger.
- [ ] If `overall_passed = false`: mitigation decision logged (print-process
      tightening vs. memo issue #16 gain-threshold raise).
- [ ] Phase log committed to `docs/phase_logs/spike_0_5.md`.
- [ ] Spike 0.5 closed in `docs/phase_checklist.md`.
