# Spike 0.6c — Tier-1 unsteady-config benchmark validation (H10 lock)

**Spec reference:** `docs/plan_R11.md §Phase 0 Spike 0.6c` (lines 1839-1844).

**Lock callouts:** H10 (Tier-1 cfg benchmark validation), Round-9 HIGH-12
(= C12, unsteady `MACH = 1e-9`).

**Questions:**

1. Does the canonical Tier-1 cfg (rendered from
   `configs/su2/fan3d_unsteady.cfg.j2`) parse and run for 1 outer time
   step on a probe mesh — i.e., is the Round-9 HIGH-12 lock actually
   honoured by the deployed cfg?
2. Does the locked numerics combination
   (compressible + low-Mach prec + RIGID_MOTION + near-zero-ambient +
   5-cycle dual-time-stepping) reproduce published unsteady lift/drag
   data for a NACA 0012 pitching at `k_reduced ≈ 0.55`, `Re ≈ 40k`,
   within ±15%?

**Status note (per spec line 1844):** Phase 4 launch is **gated on
Spike 0.6c passing**; `scripts/launch_phase4.py` refuses to create the
`phase4-launch` tag if `data/spike_0_6c/PASS` is absent. Both
sub-spikes must individually pass for the aggregator to write `PASS`.

**Procedure:** `docs/spike_0_6c_protocol.md`.

**Artifacts shipped with this spike:**

- `src/fanopt/cfd/spike_0_6c.py` — library (cfg sanity + benchmark analyzer)
- `configs/su2/oscillating_airfoil_benchmark.cfg.j2` — NACA 0012 benchmark cfg template
- `scripts/run_spike_0_6c_1.py` — sub-spike 0.6c.1 runner (cfg sanity)
- `scripts/run_spike_0_6c_2.py` — sub-spike 0.6c.2 runner (benchmark)
- `scripts/run_spike_0_6c.py` — aggregator + Phase 4 launch gate marker
- `docs/spike_0_6c_protocol.md` — operator procedure
- `tests/test_cfd/test_spike_0_6c.py` — library tests
- `tests/test_scripts/test_run_spike_0_6c.py` — CLI smoke tests
- `data/spike_0_6c/measured.template.csv` — measured-cycles CSV template
- `data/spike_0_6c/reference.template.json` — reference values JSON template

**Pass criteria:**

1. **Sub-spike 0.6c.1 (cfg sanity)** — rendered cfg parses;
   `mach_value == 1e-9` (Round-9 HIGH-12); EITHER
   `FREESTREAM_OPTION = FREESTREAM_VELOCITY` (primary) OR
   `REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE` (fallback);
   SU2 completes ≥ 1 outer time step on a probe mesh.
2. **Sub-spike 0.6c.2 (benchmark)** — every reported metric within ±15%
   of its published reference, integrated over the last 4 of 5 cycles
   (cycle 0 discarded as initial transient).
3. **Aggregate** — both sub-spikes individually pass; `PASS` marker
   present at `data/spike_0_6c/PASS`.

---

## Run log

| Field | Value |
|---|---|
| Date | _to be filled_ |
| Operator | _to be filled_ |
| SU2 build commit | _to be filled_ |
| Gmsh version | _to be filled_ |
| Compute platform | _e.g., Colab Pro CPU_ |
| SU2 installed locally? | _yes / no_ |
| Sub-spike 0.6c.1 `parsed_ok` | _to be filled_ |
| Sub-spike 0.6c.1 `mach_value` | _to be filled_ |
| Sub-spike 0.6c.1 `freestream_option` | _to be filled_ |
| Sub-spike 0.6c.1 `ref_dimensionalization` | _to be filled_ |
| Sub-spike 0.6c.1 `outer_time_steps_completed` | _to be filled_ |
| Sub-spike 0.6c.1 `passed` | _to be filled_ |
| Reference paper cited | _to be filled (e.g., McAlister/Carr 1978 fig 7c)_ |
| Sub-spike 0.6c.2 `k_reduced` | _to be filled_ |
| Sub-spike 0.6c.2 `reynolds` | _to be filled_ |
| Sub-spike 0.6c.2 cycles total | _to be filled (= 5)_ |
| Sub-spike 0.6c.2 `c_l_max` measured / ref / pct | _to be filled_ |
| Sub-spike 0.6c.2 `c_l_min` measured / ref / pct | _to be filled_ |
| Sub-spike 0.6c.2 `c_d_mean` measured / ref / pct | _to be filled_ |
| Sub-spike 0.6c.2 `c_l_hysteresis_area` measured / ref / pct | _to be filled_ |
| Sub-spike 0.6c.2 `passed` | _to be filled_ |
| Aggregator `overall_passed` | _to be filled_ |
| Phase 4 launch gate marker | _to be filled (PASS / FAIL)_ |

Runner invocations:

```
python scripts/run_spike_0_6c_1.py
python scripts/run_spike_0_6c_2.py \
  --measured data/spike_0_6c/measured.csv \
  --reference data/spike_0_6c/reference.json \
  --k-reduced 0.55 \
  --reynolds 40000 \
  --reference-source "McAlister/Carr UH110A 1978 — fig 7c"
python scripts/run_spike_0_6c.py
```

---

## Diagnostics if either sub-spike fails

| Symptom | Likely cause | Fix |
|---|---|---|
| 0.6c.1 `mach_value != 1e-9` | TIER_SPECIFIC[1] drift | Verify §9.4.1 TIER_SPECIFIC[1]; ensure CROSS_TIER doesn't carry MACH (Round-9 HIGH-12 / C12) |
| 0.6c.1 neither freestream syntax present | H10 lock missing | Add `FREESTREAM_OPTION = FREESTREAM_VELOCITY` or `REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE` |
| 0.6c.1 outer steps = 0 with SU2 | mesh missing or marker names mismatch | Inspect SU2 stdout in probe/ scratch dir |
| 0.6c.1 outer steps = 0 without SU2 | SU2 not installed | Install SU2 or run on Colab Pro CPU |
| 0.6c.2 drag metric high | mesh first-cell height | Halve first-cell height; verify boundary-layer y+ < 1 |
| 0.6c.2 lift peaks late | dt convergence | Halve `TIME_STEP`; verify cycle 2 vs cycle 3 < 5% |
| 0.6c.2 forces oscillate within cycle | inner-iter convergence | Bump `INNER_ITER` from 100 to 200 |
| 0.6c.2 hysteresis area very low | low-Mach prec coefficients | Tighten `MIN_ROE_TURKEL_PREC` / `MAX_ROE_TURKEL_PREC` band |
| 0.6c.2 all metrics scaled wrong | AMPL unit drift | Verify SU2 build commit pinned in `material_locks.SU2_COMMIT` |

---

## Fallback decisions

- **If 0.6c.1 fails:** investigate before running the benchmark. The
  cfg sanity check is a prerequisite for the benchmark — bad cfg means
  meaningless numbers. If the primary `FREESTREAM_VELOCITY` syntax is
  rejected by the deployed SU2 build, switch to the
  `REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE` fallback per the
  H10 / Round-9 HIGH-12 lock and record the working syntax here.
- **If 0.6c.2 fails:** investigate mesh quality, dt convergence,
  low-Mach prec coefficients, dual-time inner iterations BEFORE Phase 4
  launches. Do NOT silently proceed — the entire Tier 1 dataset (the
  only "true J_fan" tier) would rest on unvalidated numerics.

---

## Findings (post-run)

> _What the numbers actually were; whether the primary or fallback
> freestream syntax worked; any mesh-quality / convergence iterations
> you had to do; the chosen reference paper; final ±% per metric._

---

## Sign-off

- [ ] Sub-spike 0.6c.1 outcome recorded (PASS / FAIL).
- [ ] Working freestream syntax recorded (primary FREESTREAM_VELOCITY
      OR fallback REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE).
- [ ] Sub-spike 0.6c.2 outcome recorded with cited reference paper.
- [ ] All four NACA 0012 metrics' ±% recorded.
- [ ] Aggregate `overall_passed` recorded.
- [ ] `data/spike_0_6c/PASS` marker present iff aggregate passed.
- [ ] `data/spike_0_6c/results.json` committed (or pinned to Drive).
- [ ] This log committed to `docs/phase_logs/`.
- [ ] Spike 0.6c closed in `docs/phase_checklist.md`.
- [ ] Phase 4 launch unblocked (or fail-action documented).
