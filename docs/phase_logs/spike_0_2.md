# Spike 0.2 — Torsional-pendulum rotational-inertia protocol

**Spec reference:** `docs/plan_R11.md §Phase 0 Spike 0.2` (lines 1791–1796).

**Question:** Can we measure `I_wrist` about the +y wrist axis on the
assembled baseline fan with < 3% repeatability and within ±10% of the
generator-emitted `I_wrist_kgm2`?

**Procedure:** `docs/spike_0_2_protocol.md`.

**Artifacts shipped with this spike:**
- `docs/spike_0_2_protocol.md` — operator procedure
- `src/fanopt/physical/inertia.py` — I_wrist library
- `scripts/spike_0_2_analyze.py` — CLI analyzer
- `tests/test_physical/test_inertia.py` — analytic-rod / analytic-plate gates

**Pass criterion:**
1. Repeatability `std(I_wrist) / mean(I_wrist) × 100 < 3%` across 5 trials.
2. Cross-check `|I_meas − I_gen| / I_gen × 100 < 10%` vs. the generator-emitted
   `I_wrist_kgm2` for the Spike 0.3 baseline geometry.

---

## Run log

| Field | Value |
|---|---|
| Date | _to be filled_ |
| Operator | _to be filled_ |
| Rig type | _torsion wire / bifilar_ |
| Wire / string spec | _e.g., music wire 0.6 mm × 150 mm_ |
| Mount block | _print rev_ |
| Reference geometry | _e.g., 6061 Al rod, 8 mm × 120 mm_ |
| m_ref (kg) | _to be filled_ |
| L_ref (m) | _to be filled_ |
| I_ref (kg·m²) | _to be filled_ |
| T_ref (s, mean of 5) | _to be filled_ |
| κ (N·m / rad) | _to be filled_ |
| κ sanity-check geometry | _second reference + agreement %_ |
| Baseline fan rev | _commit hash or filename_ |
| Generator `I_wrist_kgm2` (from `smoke_test.py`) | _to be filled_ |
| T_osc trials (s) | _t1, t2, t3, t4, t5_ |
| Mean T_osc (s) | _to be filled_ |
| `I_wrist_kgm2` (measured) | _to be filled_ |
| Repeatability (%) | _to be filled_ |
| Cross-check (%) | _to be filled_ |
| `passed` | _true / false_ |

Analyzer invocation:

```
python scripts/spike_0_2_analyze.py \
  --calibration data/spike_0_2/calibration.csv \
  --measurements data/spike_0_2/measurements.csv \
  --generator-i-wrist <value> \
  --out data/spike_0_2/results.json
```

---

## Diagnostics if either gate fails

| Symptom | Likely cause | Fix |
|---|---|---|
| Repeatability > 3% | Mount slop | Re-print mount block tighter |
| Repeatability > 3% | Non-linear torsion at large amplitude | Drop amplitude to 3–5° |
| Repeatability > 3% | Damping (air drag, friction) | Lighter ref / longer wire |
| Cross-check > 10% | Axis convention bug | Verify +y wrist axis, NOT +x or +z |
| Cross-check > 10% | PETG actual density ≠ 1270 kg/m³ | Weigh single blade; pass `--rho-petg-override` |
| Cross-check > 10% | Mount block I_mount not subtracted | Measure empty-rig I; pass `--mount-i-wrist` |

---

## Findings (post-run)

> _What worked, what surprised you, anything that should propagate to the
> protocol doc, the analyzer, or the generator's `i_wrist_assembly`._

---

## Sign-off

- [ ] κ calibration repeatability < 1% on reference rod.
- [ ] κ sanity check against second reference within 2%.
- [ ] T_osc repeatability < 3% on baseline fan across 5 trials.
- [ ] Measured I_wrist within ±10% of generator-emitted value.
- [ ] `data/spike_0_2/results.json` committed (or pinned to Drive).
- [ ] Calibrated rig parked at a known location for Phase 6 reuse.
- [ ] This log committed to `docs/phase_logs/`.
- [ ] Spike 0.2 closed in `docs/phase_checklist.md`.
