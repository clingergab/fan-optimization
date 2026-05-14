# Spike 0.6 — Colab Pro compute-budget probe + M3 local-pipeline sub-spikes

**Spec reference:** `docs/plan_R11.md §Phase 0 Spike 0.6`.

**Question:**
1. What is the actual per-evaluation compute cost on Colab Pro for the
   Tier-1 (3D unsteady, ~500K cells, 5 pitching cycles at `dt = T/200`)
   workload — on a CPU runtime and on a G4-class GPU runtime?
2. Is the MacBook M3 usable for local SU2 (sub-spike 0.6a)?
3. Is the MacBook M3 usable for local FEA (sub-spike 0.6b)?

**Status note (per spec § "Status"):** Spike 0.6 is treated as
**calibration, not a gate**. The aggregator's `overall_passed` is always
True. Sub-spikes 0.6a and 0.6b ARE gates for their respective downstream
phases (local-M3 SU2 and local-M3 FEA) and are surfaced separately.

**Procedure:** `docs/spike_0_6_protocol.md`.

**Artifacts shipped with this spike:**
- `docs/spike_0_6_protocol.md` — operator procedure (calibration + sub-spikes)
- `src/fanopt/utils/compute_probe.py` — library (gates, analytic cantilever, dataclasses)
- `scripts/run_spike_0_6.py` — aggregator CLI
- `scripts/run_spike_0_6a.py` — M3 SU2 Tier-1 runner (gated on `SU2_CFD`)
- `scripts/run_spike_0_6b.py` — M3 FEA cantilever runner (gated on `dolfinx`)
- `tests/test_utils/test_compute_probe.py` — gate / analytic / aggregator tests
- `tests/test_scripts/test_run_spike_0_6.py` — CLI smoke tests
- `data/spike_0_6/budget.template.csv`,
  `data/spike_0_6/06a.template.csv`,
  `data/spike_0_6/06b.template.csv` — CSV input templates

**Pass criteria:**
1. **Calibration** — none (informational only).
2. **Sub-spike 0.6a** — Tier-1 end-to-end on the M3 in <= 15 min AND
   `J_fan_steady_proxy` finite.
3. **Sub-spike 0.6b** — cantilever FEA on the M3 in <= 2 min AND
   `|measured - analytic| / analytic * 100 <= 5%`.

---

## Run log

| Field | Value |
|---|---|
| Date | _to be filled_ |
| Operator | _to be filled_ |
| Colab account | _to be filled_ |
| Colab CPU runtime spec | _e.g., 8 vCPU, 51 GB RAM_ |
| Colab GPU runtime spec | _e.g., G4 95 GB_ |
| M3 spec | _e.g., MBP M3 Pro, 18 GB_ |
| Baseline geometry rev | _commit hash or filename_ |
| Mesh cell count (3D unsteady) | _to be filled_ |
| Colab CPU wall-time (s) | _to be filled_ |
| Colab CPU compute units | _to be filled_ |
| Colab GPU wall-time (s) | _to be filled_ |
| Colab GPU compute units | _to be filled_ |
| 0.6a SU2 installed locally? | _yes / no_ |
| 0.6a M3 Tier-1 wall-time (s) | _to be filled_ |
| 0.6a `J_fan_steady_proxy` | _to be filled_ |
| 0.6a `passed` | _true / false / skipped_ |
| 0.6b FEniCSx installed locally? | _yes / no_ |
| 0.6b M3 FEA wall-time (s) | _to be filled_ |
| 0.6b measured tip deflection (m) | _to be filled_ |
| 0.6b analytic tip deflection (m) | _to be filled_ |
| 0.6b `tip_deflection_pct` | _to be filled_ |
| 0.6b `passed` | _true / false / skipped_ |
| Aggregate `overall_passed` | true (calibration framing) |

Aggregator invocation:

```
python scripts/run_spike_0_6.py \
  --budget-csv data/spike_0_6/budget.csv \
  --06a-csv data/spike_0_6/06a.csv \
  --06b-csv data/spike_0_6/06b.csv \
  --out data/spike_0_6/results.json
```

---

## Diagnostics if either sub-spike fails

| Symptom | Likely cause | Fix |
|---|---|---|
| 0.6a wall-time > 15 min | M3 thermal throttling | Cooling pad; close other apps |
| 0.6a wall-time > 15 min | Mesh > 500K cells | Verify 2D slice cell count |
| 0.6a `J_fan_steady_proxy` NaN | SU2 didn't converge | Inspect history.csv; bump CFL / iters |
| 0.6a SU2 missing | M3 has no SU2 binary | Apply §06a fallback (Colab CPU smoke) |
| 0.6b wall-time > 2 min | Too-fine mesh | Coarsen 1D mesh to <= 100 elements |
| 0.6b tip-pct > 5% | Wrong cross-section in solver | Verify `I = b h^3 / 12` |
| 0.6b FEniCSx missing | conda env not set up | Apply §06b fallback (Colab CPU FEA) |

---

## Fallback decisions

- **If 0.6a fails / skipped:** shift `smoke_test.py` to a Colab Pro CPU
  session. M3 retains geometry / mesh-QC / Fusion / IMU roles.
- **If 0.6b fails / skipped:** move Phase 5 step 64.5's combined-blade
  structural FEA to a Colab Pro CPU session. M3 retains the same non-FEA
  roles.

---

## Findings (post-run)

> _What the numbers actually were, any thermal-throttling weirdness, any
> Colab quota surprises, anything that should propagate to the protocol
> doc, the runner scripts, or the Phase 4 / Phase 5 budget model._

---

## Sign-off

- [ ] Colab CPU 3D unsteady wall-time + CU recorded.
- [ ] Colab GPU G4 3D unsteady wall-time + CU recorded.
- [ ] Sub-spike 0.6a outcome recorded (PASS / FAIL / SKIPPED).
- [ ] Sub-spike 0.6b outcome recorded (PASS / FAIL / SKIPPED).
- [ ] Fallback decisions logged for any failed sub-spike.
- [ ] `data/spike_0_6/results.json` committed (or pinned to Drive).
- [ ] This log committed to `docs/phase_logs/`.
- [ ] Spike 0.6 closed in `docs/phase_checklist.md`.
