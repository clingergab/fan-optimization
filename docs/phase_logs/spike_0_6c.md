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
   5-cycle dual-time-stepping) produce an **internally consistent**
   pitching-airfoil solution for NACA 0012 at `k_reduced ≈ 0.55`,
   `Re ≈ 40k`, ±10° about α=0°? (See V1 decision below — literature-
   comparison gate retired.)

**Status note (per spec line 1844):** Phase 4 launch is **gated on
Spike 0.6c passing**; `scripts/launch_phase4.py` refuses to create the
`phase4-launch` tag if `data/spike_0_6c/PASS` is absent. Both
sub-spikes must individually pass for the aggregator to write `PASS`.

**Procedure:** `docs/spike_0_6c_protocol.md`.

---

## V1 decision — literature-comparison gate retired (2026-05-13)

The early draft protocol enumerated a ±15% literature-comparison gate
on four metrics (`c_l_max`, `c_l_min`, `c_d_mean`, `c_l_hysteresis_area`)
against a hand-typed `NACA0012_REFERENCE` dict claiming representative
McAlister/Carr UH110A + Anderson DB values.

A targeted literature survey (2026-05) confirmed:

* **The operating point (Re=40k, k=0.55, ±10°, mean α=0°, c/4 pivot)
  is in a published-data gap** between the well-studied low-Re/low-k
  attached-pitching regime (Kim & Chang 2013 at Re=48k, k=0.1, ±6°)
  and the moderate-Re/high-k dynamic-stall regime (MDPI Aerospace
  12(6) 457 2025 at Re=66k; McCroskey/Carr at Re~10⁶). The closest
  neighbors disagree on amplitude, k, **and** Re simultaneously.
* **The hand-typed placeholder values are likely ~2× too high on
  C_L_max** (those numbers belong to the high-Re McAlister/Carr regime,
  not Re=40k).
* **Inter-study scatter on C_L_max at this Re is ≥25%** between
  published studies that *do* overlap the regime. The ±15% gate is
  tighter than the scatter floor — it would either false-fail valid
  numerics or false-pass invalid numerics at the operator's whim.

**Decision:** retire the literature-comparison gate. Replace with two
internal-consistency gates that validate SU2 is solving its own
equations consistently, without pretending to ground-truth at an
operating point nobody has published:

1. **Convergence gate** — across kept cycles 1–4, the relative range
   of `c_l_max`, `c_l_min`, `c_d_mean` is each < 2%
   (`CONVERGENCE_TOLERANCE_PCT`). Catches dt-too-large, mesh-too-coarse,
   inner-iter-not-converged.
2. **C_L symmetry gate** —
   `|⟨c_l_max⟩ + ⟨c_l_min⟩| / max(|⟨c_l_max⟩|, |⟨c_l_min⟩|) < 5%`
   (`SYMMETRY_TOLERANCE_PCT`). Required by NACA 0012 geometric symmetry
   combined with mean α = 0°. Catches sign-flipped `PITCHING_OMEGA`,
   motion-origin error, amplitude-unit drift.

`c_l_hysteresis_area` is logged as a diagnostic in
`BenchmarkResult.diagnostic_hysteresis_area_mean` but **not gated** — at
k=0.55 the loop is near sign-inversion (researcher notes loop direction
inverts somewhere in 0.25 < k < 0.5), so a relative-range gate on a
near-zero quantity is numerically unstable.

**Cross-solver gate deferred to Phase 5.** Quantitative validation of
SU2's NACA 0012 numerics against a second open solver (PyFR is already
provisioned in the Phase 5 verification budget) is the right venue for
"are SU2's numbers physically correct, or just self-consistent?" The
researcher-recommended acceptance bound (for Phase 5 implementation):
SU2 vs PyFR within ±20% on C_L_max, ±25% on C_d_mean, hysteresis-loop-
sign-matches, area-within-2×.

**Code artifacts of this decision:**

* `src/fanopt/cfd/spike_0_6c.py` — `CONVERGENCE_TOLERANCE_PCT`,
  `SYMMETRY_TOLERANCE_PCT`, `CONVERGENCE_METRICS`, `ConvergenceCheck`,
  `SymmetryCheck`, `check_convergence`, `check_symmetry`.
  Removed: `NACA0012_REFERENCE`, `BENCHMARK_TOLERANCE_PCT`,
  `BenchmarkComparison`, `compare_cycle_to_reference`,
  `all_metrics_within_15pct` (all entered in
  `docs/retired_phrases.yaml`).
* `scripts/run_spike_0_6c_2.py` — dropped `--reference` and
  `--reference-source` CLI flags; new payload schema reflects
  convergence + symmetry records.
* `docs/spike_0_6c_protocol.md` §2.5 + §Reference data — rewritten.

**Researcher reference (2026-05):**

| Study | Re | k | Amplitude | Mean α | Closest mismatch |
|---|---|---|---|---|---|
| Kim & Chang 2013 (Aerospace Sci. Tech.) | 23k–48k | 0.1 | ±6° | 0° | k off by 5×; amp off by 1.7× |
| MDPI Aerospace 12(6) 457 (2025) | 66k | 0.094–0.628 | varies | varies | Re off by 1.65× |
| AIAA 2007-4555 | 10⁴–10⁵ | 0.02–0.12 | 5°–10° | 0°/10° | k off by ~5× |
| Kurtulus 2019 (IJMAV) | 1k | various | ±1° | 0°–60° | Re off by 40× |
| McCroskey/McAlister/Carr NASA-TM 1978/82 | 10⁶ | 0.05–0.25 | 5°–10° | 5°–15° | Re off by 25× |

Nearest in any single dimension; none match all four simultaneously.

---

## Artifacts shipped with this spike

- `src/fanopt/cfd/spike_0_6c.py` — library (cfg sanity + benchmark analyzer)
- `configs/su2/oscillating_airfoil_benchmark.cfg.j2` — NACA 0012 benchmark cfg template
- `scripts/run_spike_0_6c_1.py` — sub-spike 0.6c.1 runner (cfg sanity)
- `scripts/run_spike_0_6c_2.py` — sub-spike 0.6c.2 runner (internal-consistency gates)
- `scripts/run_spike_0_6c.py` — aggregator + Phase 4 launch gate marker
- `scripts/parse_su2_history_to_cycles.py` — SU2 `history.csv` → per-cycle `measured.csv`
- `docs/spike_0_6c_protocol.md` — operator procedure
- `tests/test_cfd/test_spike_0_6c.py` — library tests
- `tests/test_scripts/test_run_spike_0_6c.py` — CLI smoke tests
- `tests/test_scripts/test_parse_su2_history.py` — parser unit + end-to-end pipeline tests
- `notebooks/colab_spike_0_6c.ipynb` — Colab runbook (Cells 1–11)
- `data/spike_0_6c/measured.template.csv` — measured-cycles CSV template

---

## Pass criteria

1. **Sub-spike 0.6c.1 (cfg sanity)** — rendered cfg parses;
   `mach_value == 1e-9` (Round-9 HIGH-12); EITHER
   `FREESTREAM_OPTION = FREESTREAM_VELOCITY` (primary) OR
   `REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE` (fallback);
   SU2 completes ≥ 1 outer time step on a probe mesh.
2. **Sub-spike 0.6c.2 (internal consistency)** — both gates clear:
   * **Convergence** (`c_l_max`, `c_l_min`, `c_d_mean`): relative
     range across kept cycles 1–4 < 2% each.
   * **C_L symmetry**:
     `|⟨c_l_max⟩ + ⟨c_l_min⟩| / max(|⟨c_l_max⟩|, |⟨c_l_min⟩|) < 5%`.
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
| Sub-spike 0.6c.2 `k_reduced` | _to be filled (= 0.55 nominal)_ |
| Sub-spike 0.6c.2 `reynolds` | _to be filled (= 40000 nominal)_ |
| Sub-spike 0.6c.2 cycles total | _to be filled (= 5)_ |
| Sub-spike 0.6c.2 `c_l_max` mean / range% | _to be filled_ |
| Sub-spike 0.6c.2 `c_l_min` mean / range% | _to be filled_ |
| Sub-spike 0.6c.2 `c_d_mean` mean / range% | _to be filled_ |
| Sub-spike 0.6c.2 convergence_passed | _to be filled_ |
| Sub-spike 0.6c.2 symmetry asymmetry% | _to be filled_ |
| Sub-spike 0.6c.2 symmetry_passed | _to be filled_ |
| Sub-spike 0.6c.2 diagnostic ⟨hysteresis area⟩ | _to be filled (not gated; Phase 5 cross-solver check)_ |
| Sub-spike 0.6c.2 `passed` | _to be filled_ |
| Aggregator `overall_passed` | _to be filled_ |
| Phase 4 launch gate marker | _to be filled (PASS / FAIL)_ |

Runner invocations:

```
python scripts/run_spike_0_6c_1.py
python scripts/parse_su2_history_to_cycles.py \
  --history data/spike_0_6c/sub_2_run/history.csv \
  --n-cycles 5 \
  --omega-shm-rad-per-s <PITCHING_OMEGA magnitude from cfg> \
  --out data/spike_0_6c/measured.csv
python scripts/run_spike_0_6c_2.py \
  --measured data/spike_0_6c/measured.csv \
  --k-reduced 0.55 \
  --reynolds 40000
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
| 0.6c.2 convergence on `c_l_max` (range ≥ 2%) | dt too large; cycles haven't settled | Halve `TIME_STEP`; re-run; if still divergent, add more cycles |
| 0.6c.2 convergence on `c_d_mean` | inner-iter convergence | Bump `INNER_ITER` from 100 to 200; verify residuals reach `CONV_RESIDUAL_MINVAL` |
| 0.6c.2 convergence on all three metrics | mesh quality / first-cell height | Halve first-cell height; verify boundary-layer y+ < 1 |
| 0.6c.2 symmetry (asymmetry ≥ 5%) | sign error on `PITCHING_OMEGA_Y` or motion origin | Verify `PITCHING_OMEGA_VEC` is negative-y (Round-9 HIGH-12 / C11 lock); verify `motion_origin_x` = quarter-chord |
| 0.6c.2 symmetry, all metrics scaled wrong | AMPL unit drift (rad vs deg) | Verify SU2 build commit pinned in `material_locks.SU2_COMMIT` |

---

## Fallback decisions

- **If 0.6c.1 fails:** investigate before running the benchmark. The
  cfg sanity check is a prerequisite for the benchmark — bad cfg means
  meaningless numbers. If the primary `FREESTREAM_VELOCITY` syntax is
  rejected by the deployed SU2 build, switch to the
  `REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE` fallback per the
  H10 / Round-9 HIGH-12 lock and record the working syntax here.
  *(The shipped template now uses the fallback by default — SU2 v8.0.1
  rejects the primary syntax at parse time.)*
- **If 0.6c.2 fails:** investigate per the decision tree above BEFORE
  Phase 4 launches. Do NOT silently proceed — the entire Tier 1 dataset
  (the only "true J_fan" tier) would rest on unvalidated numerics.

---

## Findings (post-run)

> _What the numbers actually were; which freestream syntax worked; any
> mesh-quality / convergence iterations you had to do; the actual
> per-metric range% and asymmetry%; whether the diagnostic hysteresis
> mean is in the qualitatively-expected range (~0.05–0.3 per the
> researcher's order-of-magnitude bound)._

---

## Sign-off

- [ ] Sub-spike 0.6c.1 outcome recorded (PASS / FAIL).
- [ ] Working freestream syntax recorded (primary FREESTREAM_VELOCITY
      OR fallback REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE).
- [ ] Sub-spike 0.6c.2 outcome recorded.
- [ ] Convergence range% per metric recorded; symmetry asymmetry%
      recorded; diagnostic ⟨hysteresis area⟩ recorded.
- [ ] Aggregate `overall_passed` recorded.
- [ ] `data/spike_0_6c/PASS` marker present iff aggregate passed.
- [ ] `data/spike_0_6c/results.json` committed (or pinned to Drive).
- [ ] This log committed to `docs/phase_logs/`.
- [ ] Spike 0.6c closed in `docs/phase_checklist.md`.
- [ ] Phase 4 launch unblocked (or fail-action documented).
- [ ] Phase 5 PyFR cross-solver work-item logged (the researcher-
      recommended SU2-vs-PyFR validation that the V1 internal-
      consistency gates explicitly defer to Phase 5).
