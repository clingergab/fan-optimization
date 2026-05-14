# Spike 0.4 — Click-feature tolerance, cycle life, and V1-lock force-balance

**Spec reference:** `docs/plan_R11.md §Phase 0 Spike 0.4` (lines 1803-1825);
H6 lock (force balance), H8 lock (lever arm 0.25 m), C9 mass cap.

**Question:** Does cumulative inter-blade friction at the deployed position
exceed `2 × F_inertial_at_click` (H6 force-balance pass), and does the click
chamfer + detent geometry print cleanly, engage at 0.5-2 N, and survive 1000
deploy/fold cycles plus a 100-cycle 1-4 N high-amplitude stress segment with
< 1 mm adjacent-tip alignment gap variation?

**Procedure:** `docs/spike_0_4_protocol.md`.

**Artifacts shipped with this spike:**
- `docs/spike_0_4_protocol.md` — operator procedure
- `src/fanopt/physical/click_rig.py` — Spike 0.4 library (force balance,
  clearance, engagement force, cycle life)
- `scripts/run_spike_0_4.py` — CLI analyzer
- `tests/test_physical/test_click_rig.py` — boundary + analytic-known gates
- `tests/test_scripts/test_run_spike_0_4.py` — CLI smoke tests
- `data/spike_0_4/*.csv` — input CSV templates

**Pass criterion:**
1. `F_friction_cumulative ≥ 2 × F_inertial_at_click`, where
   `F_inertial_at_click = (I_wrist · α_max) / L_wrist_to_tip` with
   `α_max = 110 rad/s²` and `L_wrist_to_tip = 0.25 m` (H8 lock).
2. Every per-mating-surface clearance in [0.15, 0.20] mm.
3. Every low-regime click engagement force in [0.5, 2.0] N.
4. 1000 deploy/fold cycles complete without detent fracture.
5. 100-cycle high-amplitude stress segment (1-4 N) completes without
   fracture.
6. Adjacent-blade-tip alignment gap variation < 1 mm.

---

## Run log

| Field | Value |
|---|---|
| Date | _to be filled_ |
| Operator | _to be filled_ |
| Test articles printed | _2 single-blade specimens, slicer profile rev_ |
| Printer + filament | _e.g., Prusa MK4 + PETG_ |
| I_wrist (from Spike 0.2) | _to be filled_ kg·m² |
| τ_inertial_peak | _I_wrist · 110_ N·m |
| F_inertial_at_click | _τ / 0.25_ N |
| F_friction_cumulative (sum of 9 pairs) | _to be filled_ N |
| Required friction (`2 × F_inertial`) | _to be filled_ N |
| Force-balance pass? | _true / false_ |
| V1 fallback armed? | _true / false_ |
| Clearance range (min, max, mean) | _to be filled_ mm |
| Out-of-band clearance count / total | _to be filled_ |
| Engagement force (low) mean ± std | _to be filled_ N |
| Out-of-band low-regime trials | _to be filled_ |
| Engagement force (high) mean ± std | _to be filled_ N |
| Cycle inspections logged | _to be filled_ |
| First wear cycle | _to be filled_ |
| First fracture cycle | _to be filled / none_ |
| High-amp segment completed? | _true / false_ |
| Alignment gap variation (worst case) | _to be filled_ mm |
| `overall_passed` | _true / false_ |

Analyzer invocation:

```
python scripts/run_spike_0_4.py \
  --force-balance data/spike_0_4/force_balance.csv \
  --clearance data/spike_0_4/clearance.csv \
  --engagement-force data/spike_0_4/engagement_force.csv \
  --cycle-inspections data/spike_0_4/cycle_inspections.csv \
  --alignment-gap-variation-mm <value> \
  --high-amp-completed \
  --out data/spike_0_4/results.json
```

---

## Diagnostics if any sub-gate fails

| Symptom | Likely cause | Fix |
|---|---|---|
| Force balance fails | Cumulative friction insufficient | V1 rib-tab fallback auto-arms; flip `params.layer4.v1_lock_fallback_enabled = True` |
| Clearance out-of-band | Slicer over- / under-extrusion | Re-tune slicer flow rate or printer calibration |
| Low-regime engagement out-of-band | Detent geometry too tight / slack | Adjust chamfer angle or detent depth; re-print |
| Detent fracture < 1000 cycles | Material failure | Upgrade to neodymium magnetic catch (20-40 g, within C9) |
| High-amp fracture | Same as above | Magnetic catch |
| Alignment gap ≥ 1 mm | Pivot-pin slop | Tighten pivot bushing tolerance (Spike 0.1) |

---

## Findings (post-run)

> _What worked, what surprised you, anything that should propagate to the
> protocol doc, the analyzer, or the click-feature generator._

---

## Sign-off

- [ ] 2 single-blade test articles printed at production slicer profile.
- [ ] V1-lock force balance measured; `F_friction_cumulative` recorded.
- [ ] Clearance measured for every mating surface; all in [0.15, 0.20] mm.
- [ ] Click engagement force (low regime) within [0.5, 2.0] N.
- [ ] 1000 deploy/fold cycles complete, inspections every 100 cycles logged.
- [ ] 100-cycle high-amplitude stress segment (1-4 N) complete.
- [ ] Adjacent-blade-tip alignment gap variation < 1 mm.
- [ ] `data/spike_0_4/results.json` committed (or pinned to Drive).
- [ ] If `v1_lock_fallback_armed = true`, `params.layer4.v1_lock_fallback_enabled` flipped.
- [ ] This log committed to `docs/phase_logs/`.
- [ ] Spike 0.4 closed in `docs/phase_checklist.md`.
