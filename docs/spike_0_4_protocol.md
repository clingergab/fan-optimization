# Spike 0.4 — Click-feature tolerance, cycle life, and V1-lock force-balance protocol

**Spec reference:** `docs/plan_R11.md §Phase 0 Spike 0.4` (lines ~1803-1825),
H6 lock (force balance), H8 lock (lever arm 0.25 m), C9 mass cap.

**Why this exists.** V1 of the fan deploys click features at the panel's outer
tangential edge at the tip (chamfer + detent). Two questions must be answered
before any panel-pivot prototype ships:

1. Does cumulative inter-blade friction at the deployed position resist the
   peak inertial back-drive force the wrist imparts at α_max?
   (**H6 lock**.) If not, a V1 fallback geometry — a printed rib-tab snap-fit
   on each guard blade — is auto-armed via
   `params.layer4.v1_lock_fallback_enabled`.
2. Are the click chamfer + detent geometry, clearance, and cycle life within
   spec on the actual printer + filament we will use?

The same spike covers both, because the click feature is the variable in
both gates.

## Axis lock — DO NOT SKIP

The click feature is on the **panel's outer tangential edge at the tip**, at
`(x = L_blade, y = ±panel_tangential_outer ≈ ±0.0225 m)`. It is NOT on a rib
face — the rib terminates at `L_blade − RIB_TIP_TAPER = 0.185 m`, 15 mm short
of the tip.

**H8 lever-arm lock:** when you convert wrist inertial torque to a tangential
force at the click feature, divide by **`L_wrist_to_tip = 0.25 m`** —
NOT by `0.20 m` (the pivot-to-tip distance). The wrist axis sits at
`d_handle = 0.05 m` proximal to the pivot pin (§0 row 27, §6.4), and the click
feature sits at `L_blade = 0.20 m` distal to the pivot. The two distances add:
`0.25 m`. Dividing by 0.20 m by mistake under-reports F_inertial by 20%, which
flips a failing force balance into a false pass.

---

## Apparatus

- **Force gauge:** range ≥ 10 N, resolution ≥ 0.1 N (digital force gauge or
  a calibrated spring scale).
- **Calipers + feeler gauge** for clearance measurement at the click feature.
- **Cycle counter:** manual tally + clicker, or scripted with a stepper if
  the rig is motorized.
- **Inspection rig:** calipers + magnifying loupe for detent geometry check
  every 100 cycles; calipers for adjacent-blade-tip alignment gap.
- **Reference inertia from Spike 0.2:** `I_wrist_kgm2`. Required input to the
  force-balance pass criterion — re-run Spike 0.2 first if it has not been
  measured for the current baseline fan.

---

## Step 1 — Print 2 single-blade test articles

Each test article is one printed blade (single-segment), ~200 mm × 25 mm ×
2 mm, with the **production click chamfer + detent on the facing panels'
outer tangential edges at the tip** (i.e., the mating geometry at
`x = L_blade`, `y = ±panel_tangential_outer ≈ ±0.0225 m`). Print both articles
on the same printer, same filament, same slicer profile that the prototype
will use.

Acceptance: visual inspection of the chamfer (0.5-1 mm corner bevel per the
HIGH-8 Round-9 Option A lock — not a full-panel-thickness face) and detent
shows clean walls, no string-and-blob, no warp at the tip.

---

## Step 2 — V1-lock force balance (H6 lock)

1. Assemble the baseline fan (10 blades, 9 inter-blade pairs at the click
   feature). Deploy to the full 133.3° spread (`blade_count × 13.3°`).
2. Apply the force gauge **tangentially at the panel's outer tangential edge
   at the tip** — i.e., at `(x = L_blade, y = ±panel_tangential_outer)` —
   pushing against the engaged detent in the direction that would back-drive
   the click.
3. Slowly increase force until the click feature is just on the verge of
   disengaging on each pair. Read the disengage force per pair on the gauge.
4. Repeat for each of the 9 inter-blade pairs. Sum the per-pair forces to
   get `F_friction_cumulative_N`.
5. Record in `data/spike_0_4/force_balance.csv`:

```
I_wrist_kgm2,F_friction_cumulative_N,notes
1.0e-3,1.55,from_spike_0_2_results;sum_of_9_pairs
```

The analyzer computes:
- `τ_inertial_peak = I_wrist · α_max` (α_max = 110 rad/s², H8/H6 lock).
- `F_inertial_at_click = τ_inertial_peak / 0.25 m` (H8 lever-arm lock).
- Pass iff `F_friction_cumulative ≥ 2 × F_inertial_at_click`.

If the gate fails, the V1 fallback rib-tab geometry is auto-armed
(`v1_lock_fallback_armed = true`). Downstream, this sets
`params.layer4.v1_lock_fallback_enabled = True`: a 3 mm × 5 mm × 1.5 mm
printed tab on each guard blade mates a 0.2 mm-deeper pocket on the adjacent
inner blade's outer rib at 133.3°, giving a 3-5 N radial snap-fit.

---

## Step 3 — As-printed clearance at the click feature

For each of the 9 inter-blade pairs (or each single-blade-test-article pair),
use the feeler gauge to measure the **clearance at the click feature**, per
mating surface. Target band: **0.15-0.20 mm**.

Record in `data/spike_0_4/clearance.csv`:

```
mating_surface,clearance_mm,notes
blade1_blade2_outer,0.17,clean
blade2_blade3_outer,0.18,clean
...
```

A measurement outside the band suggests slicer over- or under-extrusion at the
click feature. Fix: re-tune slicer flow rate or printer calibration; do NOT
patch by widening the spec tolerance.

---

## Step 4 — Click engagement force (low regime)

Using the deploy-from-folded motion (start at 0° spread, drive to 133.3°),
catch the click engagement at each pair on the force gauge. Target band:
**0.5-2.0 N**.

Record in `data/spike_0_4/engagement_force.csv` with `regime = "low"`:

```
trial,force_N,regime,notes
1,0.9,low,blade1_blade2
2,1.1,low,blade2_blade3
...
```

Repeat enough trials to cover every pair at least once; ideally each pair 3×
for a tight mean/std.

---

## Step 5 — 1000-cycle deploy/fold with every-100-cycle inspection

1. Deploy → fold → deploy. One full out-and-back is one cycle.
2. Every 100 cycles, pause and inspect the detent geometry under the loupe:
   - Any wear (rounded edges, scuffed walls)? Record `wear_observed = true`.
   - Any fracture (chipped detent, cracked chamfer)? Record
     `fracture = true`. Stop the run; the low-amplitude gate fails.
3. Measure adjacent-blade-tip alignment gap at full deployment **once at the
   end of the 1000-cycle run** (or earlier if alignment is visibly drifting).
   Record the worst-case gap variation in mm.

Record in `data/spike_0_4/cycle_inspections.csv`:

```
cycle,wear_observed,fracture,notes
100,false,false,clean
200,false,false,clean
...
1000,false,false,clean
```

If the detent fractures before 1000 cycles → low-amplitude gate fails →
upgrade to embedded neodymium magnetic catch (~20-40 g, within the 100 g
C9 mass cap). Re-run from Step 1 with the magnetic-catch geometry.

---

## Step 6 — High-amplitude stress segment (100 cycles at 1-4 N)

After the 1000-cycle low-amplitude run completes cleanly, run an additional
**100 cycles at ~2× the design-point engagement force** — operator drives the
deploy with extra force so the force gauge reads 1-4 N at engagement.

Append `regime = "high"` rows to the engagement-force CSV (same file as Step
4):

```
...
20,2.5,high,stress_seg_cycle_005
21,3.0,high,stress_seg_cycle_010
...
```

Pass: detent geometry intact after 100 high-amplitude cycles. If detent
fractures during this segment, record the cycle index in
`--high-amp-failure-cycle`.

---

## Step 7 — Run the analyzer

```
python scripts/run_spike_0_4.py \
  --force-balance data/spike_0_4/force_balance.csv \
  --clearance data/spike_0_4/clearance.csv \
  --engagement-force data/spike_0_4/engagement_force.csv \
  --cycle-inspections data/spike_0_4/cycle_inspections.csv \
  --alignment-gap-variation-mm <worst-case from Step 5> \
  --high-amp-completed \
  --out data/spike_0_4/results.json
```

If the high-amplitude segment fractured at, e.g., cycle 42, drop
`--high-amp-completed` and add `--high-amp-failure-cycle 42`.

The analyzer reports:

| Field | Meaning | Pass criterion |
|---|---|---|
| `force_balance.passed` | F_friction ≥ 2 × F_inertial_at_click | true |
| `force_balance.v1_lock_fallback_armed` | inverse of above | false on PASS |
| `clearance.passed` | every measurement in [0.15, 0.20] mm | true |
| `engagement_force.passed` (low) | every trial in [0.5, 2.0] N | true |
| `high_amp_engagement_force.passed` | every trial in [1.0, 4.0] N | true |
| `cycle_life.low_amp_passed` | 1000 cycles without fracture | true |
| `cycle_life.high_amp_passed` | 100 high-amp cycles completed without fracture | true |
| `cycle_life.alignment_passed` | gap variation < 1.0 mm | true |
| `overall_passed` | all of the above | true |

Exit codes:
- `0` — overall PASS (spike closes)
- `1` — overall FAIL (at least one sub-gate failed; consult fallback tree)
- `2` — input error (missing file / column / non-numeric value)

---

## Fallback decision tree

| Failing sub-gate | Likely cause | Fix |
|---|---|---|
| `force_balance.passed = false` | Cumulative friction insufficient | Auto-arm V1 rib-tab fallback (`params.layer4.v1_lock_fallback_enabled = True`); re-run the assembly with the snap-fit guard tabs. |
| `clearance.passed = false` | Slicer over- or under-extrusion at the click feature | Re-tune slicer flow rate or printer calibration. Do NOT widen the spec tolerance. |
| `engagement_force.passed = false` (low) | Detent geometry too tight or too slack | Adjust chamfer angle (0.5-1 mm bevel) or detent depth in the generator; re-print. |
| `cycle_life.low_amp_passed = false` (fracture in first 1000 cycles) | Detent material failure | Upgrade to embedded neodymium magnetic catch (20-40 g, within C9). |
| `cycle_life.high_amp_passed = false` | Detent fractures at high force | Same as above — magnetic catch. |
| `cycle_life.alignment_passed = false` | Adjacent-tip gap drifts | Tighten pivot-pin slip-fit; re-check Spike 0.1 pivot bushing tolerance. |

---

## What this rig is reused for

- **Phase 5** verification: every printed top-3 Pareto prototype gets one
  click-feature inspection pass at the deployed position before any CFD/PIV
  comparison.
- **`params.layer4.v1_lock_fallback_enabled`** binary flag: defaults to
  `False`; flipped to `True` if Spike 0.4's force balance fails. The BO
  outer architecture bandit sees this flag as fixed for V1.
