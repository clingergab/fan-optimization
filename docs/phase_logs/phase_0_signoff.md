# Phase 0 Signoff + V1/V2 Scope Decision (2026-05-13)

This document records the authoritative scope decision separating V1 (build a fan that demonstrably feels meaningfully better than the printed baseline) from V2 (quantify the gain with hardware-instrumented measurements). It is the canonical reference for why four Phase-0 spikes are marked deferred in their per-spike `data/spike_0_X/deferral.json` sentinels.

## Decision

**V1 is a sim-only + qualitative-feel project.** Hardware-instrumented measurement (anemometer, IMU, torsion pendulum) is deferred to V2.

The non-negotiable kept-in-V1 spike is **Spike 0.6c.1** — the Tier-1 unsteady cfg sanity check. Sub-spike 0.6c.2 (NACA 0012 numerical-consistency benchmark) was deferred to Phase 5 on 2026-05-14 after the regime diagnostic confirmed the production-faithful cfg can't be validated against any published wind-tunnel reference in the same frame; see Note 1 below. Sub-spike 0.6c.1 confirms the production Tier-1 cfg parses + launches under the deployed SU2 build — that's the minimum sanity bar before Phase 4 spends compute on it. Numerical-correctness validation (the SU2-vs-PyFR cross-solver gate) lives in Phase 5; until that completes, Tier-1 evaluations are trusted only as "syntactically valid + solver-launchable" with sim-vs-sim relative deltas defensible because both sides use the same numerics.

## What V1 ships

A printable, manufacturability-validated fan design that subjectively feels better than the printed flat-panel baseline. Outputs:

1. A printable fan from the Phase 5 top-3 Pareto candidates (operator picks the winner by hands-on feel).
2. A JSONL ledger of simulated metrics for every design tried during Phase 4 BO.
3. A documented Phase-6 qualitative test record (blinded A/B feel test results).

## What V1 explicitly does NOT ship

- A `J_fan / W_cycle` ratio with bench-measured I_wrist.
- A quantitative `J_fan_measured` value at 300 mm.
- A 3-copy fabrication-noise CV.
- A Sobol-vs-BO iso-compute head-to-head.

These deliverables live in V2. The plan's "≥15% improvement vs. Spike 0.3 baseline" reporting target is suspended for V1; V1 reports sim-vs-sim relative gain (the optimized design's simulated `J_fan` vs. the baseline's simulated `J_fan`) and the operator's qualitative comparison.

## Deferred spikes (sentinel files + plan §13 V2 backlog entries)

| Spike | Sentinel | V1 substitute |
|---|---|---|
| 0.2 (torsional pendulum I_wrist) | `data/spike_0_2/deferral.json` | Analytic `i_wrist_assembly` from §6.4 generator; safety factor in Spike 0.4 bumped 2× → 3× via `--i-wrist-analytic` flag. |
| 0.3 (anemometer + IMU baseline) | `data/spike_0_3/deferral.json` | Phase 2a baseline CFD as the sim-side baseline + Phase 6 blinded A/B feel test of printed top-3. |
| 0.5 (3-copy fab-noise CV) | `data/spike_0_5/deferral.json` | Print one top candidate twice; compare by feel. |
| 0.7c (Sobol-vs-BO iso-compute) | `data/spike_0_7c/deferral.json` | BO-stall fallback: switch to hand-picked diverse candidates if Phase 4 stalls. |

Full V2 specifications (revisit triggers, acceptance criteria) live in `docs/V2_backlog.md` under "Deferred Phase-0 spikes".

## Spikes still required for V1

| Spike | Status | What it gates |
|---|---|---|
| 0.0 scaffolding | ✅ done | repo + CI |
| 0.1 Fusion headless | ✅ done | geometry export pipeline |
| 0.4 click rig + V1 lock force balance | ✅ implemented (uses analytic I_wrist) | click feature integrity + V1 lock decision |
| 0.6 Colab compute probe | ⚠️ aggregator solid; runners stub | calibration only, not a gate |
| 0.6a/0.6b M3 SU2/FEA viability | ⚠️ stub-only; needs real SU2/FEniCSx integration | local M3 vs Colab decision for downstream FEA work |
| **0.6c.1 Tier-1 cfg sanity** | `launch_phase4.py` wired (done); 0.6c.2 deferred to Phase 5 on 2026-05-14 — see Note 1; sub_1 still needs a Colab run that captures SU2 stdout to write `sub_1.PASS` | **gates Phase 4 launch tag — sub_1.PASS suffices** |
| 0.7a generative geometry sanity | shim only; Phase 1 will land the real generator | sanity check on parameter-space coverage |
| 0.7b BO infra sanity | implemented (uses synthetic objective) | confirms BO machinery doesn't crash at 37-46D |

## V1 cheap mitigations adopted at the same decision

These are the "do this anyway" items that compensate for the deferred spikes without adding hardware cost. None requires anything specialized.

1. **Diverse Phase 5 print candidates.** The top-3 prints must span Layer 2 archetypes (one louver-heavy, one TPMS-heavy, one near-baseline). Mitigates the "BO exploits a sim artifact" failure mode by ensuring at least one printed design is structurally robust.

2. **Print one top candidate twice.** Same-design sanity check at Phase 6 substitutes for Spike 0.5 fab-noise CV. If the two prints feel different, the design comparison is contaminated and the operator must flag before declaring ship-ready.

3. **Blinded A/B Phase 6 protocol.**
   - Operator has someone else hand them fans without naming them.
   - Stopwatch-paced 20 strokes at a consistent metronome cadence (2 Hz target).
   - Score each fan 1-5 on (airflow felt / weight / sound / aesthetics) separately.
   - Repeat the protocol on a different day.
   - Catches the worst confirmation-bias problems for $0.

4. **Pace consistency.** Use a stopwatch and count strokes to keep the waving cadence consistent across designs. A 20% pace difference between trials dominates any airflow signal.

5. **BO-stall fallback.** If Phase 4 Tier-0 best-J_fan stalls for 20 consecutive acquisitions, switch to hand-picked candidates rather than burning more compute.

## Risks accepted at decision time

| Risk | Why we accept | Mitigation if it bites |
|---|---|---|
| BO finds a sim artifact that doesn't survive printing | Mitigated by print-3-diverse rule + qualitative-feel veto | Pick a different Pareto candidate; document as V2 sim-fidelity issue |
| Sub-15% gains lost in print noise | Mitigated by print-twice sanity check; only matters if real gains are small | V2 fab-noise measurement (Spike 0.5) |
| BO is no better than random search | Mitigated by BO-stall fallback; cost is upper-bounded by Phase 4 1300-h ceiling | V2 Sobol-vs-BO baseline (Spike 0.7c) |
| Confirmation bias in Phase 6 feel test | Mitigated by blinded A/B + repeat-on-different-day | V2 quantitative measurement |
| Analytic I_wrist is wrong | Mitigated by 3× safety factor on V1 lock force balance | V2 Spike 0.2 with torsional pendulum |

## Test-suite state at signoff

- 518 tests, all green (as of 2026-05-13 post-0.6c.2-revision).
- New deferral-aware paths exercised: `run_spike_0_4.py --i-wrist-analytic <value> --f-friction-cumulative-n <value>` honors 3× safety factor and skips the Spike-0.2 cross-check.
- All other V1-required spike runners pass their gates either canonically (clearance, engagement-force, cycle-life) or under dry-run plumbing exercise (0.6a/0.6b).
- End-to-end pipeline test (`test_pipeline_parse_then_analyzer_yields_pass`) exercises the full Cell-8-output path: synthesized SU2 history.csv → `parse_su2_history_to_cycles` → `run_spike_0_6c_2` → PASS marker. De-risks the post-Cell-8 path without burning Colab hours.

## Note 1 — Spike 0.6c.2 deferred to Phase 5 (2026-05-14, supersedes 2026-05-13 revision)

The Cell 8 SU2 run on Colab produced non-physical CL values (~10⁶, c_l_min positive throughout cycles 1–4). The 2026-05-14 regime diagnostic (`scripts/diagnose_su2_pitching_regime.py`) confirmed the root cause: the production-faithful MACH=1e-9 cfg produces **body-in-still-air added-mass/quadratic-drag forces**, NOT wind-tunnel-like aerodynamic lift. CL oscillates at 2× the prescribed pitching frequency with bias ratio 2.234 — both characteristic of body-in-still-air physics. The 2026-05-13 internal-consistency revision (convergence + symmetry gates) implicitly assumed wind-tunnel physics and is therefore conceptually unsound for the production cfg's regime.

**Decision:** drop sub-spike 0.6c.2 from the Phase 4 launch gate entirely. Phase 4 launch gates on sub-spike 0.6c.1 (cfg sanity) only. Quantitative cross-solver validation (SU2 vs PyFR on a wind-tunnel-frame benchmark) is the **Phase 5** home for absolute-accuracy validation; PyFR is already provisioned in the Phase 5 budget (Round-9 HIGH-11 G4 GPU lock).

**Code-side cleanup:** `src/fanopt/cfd/spike_0_6c.py` stripped to sub_1 only (convergence/symmetry/benchmark code removed); `scripts/run_spike_0_6c_2.py` deleted; `scripts/run_spike_0_6c.py` simplified to read sub_1 only; retired identifiers entered in `docs/retired_phrases.yaml`.

Full evidence trail, diagnostic output, and Phase 5 deliverable specification: `docs/phase_logs/spike_0_6c.md` → "V1 decision — Spike 0.6c.2 deferred to Phase 5 (2026-05-14)". The earlier 2026-05-13 internal-consistency revision is retained in the same doc for traceability but is itself superseded.

This revision is tactical (under V1 scope) and does not change the V1/V2 split documented above. It DOES change the Phase 4 launch criterion: sub_1.PASS suffices.

## Note 2 — Spike 0.6d added as Tier-1 quantitative-sanity counter-check set (2026-05-14, complements Note 1)

After the Note 1 deferral, the V1 confidence picture had Phase 4 about to burn ~1300 GPU-hours on Tier-1 evaluations with **no independent quantitative check** on SU2's body-in-still-air response at MACH=1e-9. The 2026-05-13 internal-consistency gate had been the planned compensation; Note 1 retired it. To prevent a repeat-mode "we have only consistency evidence" failure, a new Phase-0 spike — **Spike 0.6d** — lands three lightweight independent counter-checks before Phase 4 launches.

**Sub-spikes:**

| Sub-spike | What it tests | Pass criterion | Gating? | Cost |
|---|---|---|---|---|
| 0.6d.1 | Cycle-averaged force symmetry + dimensional-force magnitude vs analytic envelope | `\|F_cycle_avg\| < 0.05 × F_cycle_peak`; `F_cycle_peak ∈ [0.02, 2.0] N` | Yes | ~1–2 h Colab CPU |
| 0.6d.2 | 2D thin-plate added-mass coefficient vs Sedov/Newman closed-form | SU2 inviscid-phase moment within ±15% of analytic | Yes | ~2–4 h Colab CPU |
| 0.6d.3 | SU2 incompressible-mode (`INC_NAVIER_STOKES`) vs compressible-with-MACH=1e-9 | Dimensional forces within ±20% | **Advisory only** | ~2–3 h Colab CPU |

**Phase 4 launch gate (post-2026-05-14):** `data/spike_0_6c/PASS` (= 0.6c.1) **AND** `data/spike_0_6d/PASS` (= 0.6d.1 ∧ 0.6d.2). 0.6d.3 is logged but does not affect either marker. `scripts/launch_phase4.py` is updated to require both marker files.

**Independent-codebase cross-check moved to Phase 5 step 62.5:** OpenFOAM `pimpleFoam` incompressible joins SU2 + PyFR as a 3rd solver in the Phase 5 published-reference benchmark (Sane & Dickinson 2002 robotic-flapper or equivalent body-in-still-air case from the insect-flight / Morison-equation literature). This is the absolute-accuracy answer; 0.6d is the cheap-evidence Phase-0 prelude that ensures Phase 4 ranking isn't being driven by solver artifacts.

**Phase 4 in-flight follow-on (plan step 56.5; cost re-scoped 2026-05-14 per F3 fix):** Phase 4 carries two monitoring rules sized to fit inside the 1000-h stop-rule headroom — (a) baseline-regression every N=100 BO acquisitions at **Tier 0** (NOT Tier 1; 30-90 min/eval; ~5-15 h cumulative across campaign); (b) MACH-perturbation rank-stability at Phase-4 midpoint, **10 representative geometries** (NOT 20) at MACH=1e-7 (~30-60 h one-shot). Combined ~35-75 h, counted against the 1000-h stop rule.

**Confidence assessment after 0.6d + step 62.5:**

- Tier 0 (analytic / closed-form): 0.6d.1 + 0.6d.2 give order-of-magnitude + closed-form independent evidence in Phase 0.
- Tier 1 (cross-solver same regime): 0.6d.3 (SU2 incompressible) advisory in Phase 0; Phase 5 step 62.5 SU2 ↔ PyFR ↔ OpenFOAM 3-codebase comparison.
- Tier 2 (published reference): Phase 5 step 62.5 — regime-appropriate published data (body-in-still-air, not wind-tunnel-frame).
- Tier 3 (hardware): V2 (Spike 0.2 torsion pendulum); unchanged from prior signoff.

This restores roughly the confidence level we'd have had if the 2026-05-13 internal-consistency gate had been validating actual physics — and via independent channels, not consistency loops. The work is ~3 days my effort + ~5–9 h Colab CPU; calendar impact on V1 ship is ~1 week. The full plan-edit set documenting this lives in `docs/proposed_plan_edits_0_6d.md` (proposed → authorized → applied 2026-05-14).

**Superseded in part by Note 3 (2026-05-15)** — the first live Colab run of Spike 0.6d showed the 0.6d.1 gate design was unsound; the gate was rebuilt around 0.6d.2 only. The Tier-0/1/2/3 confidence layering above still holds, but the Phase-0 Tier-0 evidence is now "frequency-consistent added-mass recovery" (0.6d.2), not the 0.6d.1 symmetry/envelope checks.

## Note 3 — Spike 0.6d gate redesigned to added-mass frequency-consistency (2026-05-15, supersedes Note 2's gate design)

The first live Colab run produced (correctly recovered) `data/spike_0_6c/PASS` via the 0.6c.1 history.csv-evidence path, then ran 0.6d.1 on the deferred 0.6c.2 benchmark history.csv. 0.6d.1 FAILed — and inspection showed the **FAIL was not a clean signal; the gate design was flawed**:

1. **Wrong dataset / circular.** 0.6d.1 re-analysed the deferred 0.6c.2 benchmark output — the exact data the 2026-05-14 diagnostic already classified as added-mass-dominated. Its FAIL re-confirms the deferral rationale; it is not independent new evidence about the production numerics.
2. **Nondimensionalisation conflation.** The magnitude check compared SU2's `CFx` (nondimensional under `REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE`, `q_ref = 1`) against a dimensional Newton envelope (`m·ω²·r`). The ~6-order gap is dominated by the nondim convention, not proven non-physical output. The check cannot distinguish "garbage" from "sane forces under a `q_ref=1` convention."
3. **Symmetry criterion ill-posed for a fan.** A working fan produces NET force by design (asymmetric productive/return stroke); a near-zero-cycle-mean criterion would reject the very physics the project depends on.

**Decision (option "A-lite"):** demote 0.6d.1 + 0.6d.3 to **advisory** (recorded for Phase 5, not gating) and rebuild the gate on a single **normalization-invariant, parameter-free falsification test** — Sub-spike 0.6d.2, redesigned:

- Run the 2D thin-plate at TWO pitching frequencies (ω₁, ω₂), same plate / pivot / θ_max.
- Fourier-project each moment-coefficient trace onto the added-mass (sin φ) basis; recover `I_a = a_sin/(ω²·θ_max)`.
- **Gate:** `|I_a(ω₁) − I_a(ω₂)| / mean < 0.25`. `I_a = πρb⁴(1/8+a²)` is frequency-independent by construction; the comparison is SU2-vs-SU2 so the fixed `q_ref` cancels — fully normalization-invariant. A frequency-dependent recovered `I_a` falsifies the MACH=1e-9 + low-Mach-prec numerics *before* Phase 4 spends ~1300 GPU-hours optimizing a wrong objective.
- The Sedov/Newman closed-form magnitude comparison is computed but **advisory** (absolute scale needs SU2's exact reference-state handling = Phase 5 step 62.5).

**Why this is worth the ~1–2 extra days:** the cost asymmetry — if the 2.234 bias-ratio anomaly (normalization-invariant, unexplained) reflects a numerical artifact, catching it now costs ~days; missing it costs a full Phase-4 redo (~weeks of compute). The redesigned 0.6d.2 degrades gracefully: even if the advisory magnitude stays ambiguous, the frequency-consistency verdict is unambiguous.

**Honest note:** the original 0.6d design (Note 2) passed its synthetic unit tests but its *criteria* did not survive the real nondimensionalisation + physics. The unit tests tested the implementation against the criteria; they could not catch mis-specified criteria. The redesigned 0.6d.2 is validated the same way (synthetic projection in/out) — its soundness rests on the normalization-invariance argument above, which the first live run will confirm or falsify.

Phase 4 gate (post-2026-05-15): `data/spike_0_6c/PASS` (0.6c.1) **AND** `data/spike_0_6d/PASS` (0.6d.2 frequency-consistency only).

## Sign-off

- [x] Spike 0.2 deferral sentinel committed (`data/spike_0_2/deferral.json`).
- [x] Spike 0.3 deferral sentinel committed.
- [x] Spike 0.5 deferral sentinel committed.
- [x] Spike 0.7c deferral sentinel committed.
- [x] `run_spike_0_4.py` accepts analytic I_wrist + 3× safety factor.
- [x] V2 backlog updated with revisit triggers + acceptance criteria.
- [x] Plan (`docs/report-final.md`) annotated with the deferral note (additive, locked decisions untouched).
- [x] `docs/phase_checklist.md` updated to reflect new V1 scope.
- [x] Spike 0.3 kitchen-scale appendix added (optional V2 upgrade path).
- [ ] **Action items remaining for operator before Phase 4 launch** (current as of 2026-05-21, reflecting Note 3 redesign):
  - [ ] **0.6c.1 PASS marker** — `notebooks/colab_spike_0_6d.ipynb` Cell 5b recovers this via the `--su2-history-csv` evidence path against the existing Drive history.csv (no stdout-capture issue; the runner now accepts a prior SU2 history.csv as evidence of cfg-launch sanity). Cell 5b runs the 0.6c aggregator → writes `data/spike_0_6c/PASS`.
  - [ ] **0.6d.2 PASS marker (the gate)** — Cell 7 renders the 2D thin-plate cfg, runs SU2 at ω₁ and ω₂ on the same plate/pivot/θ_max, feeds both history.csv files to `scripts/run_spike_0_6d_2.py`. Pass criterion: `|I_a(ω₁) − I_a(ω₂)| / mean < 0.25`.
  - [ ] **Aggregator + dual-gate check** — Cell 9 runs `scripts/run_spike_0_6d.py --sub-2-json …` → writes `data/spike_0_6d/PASS` iff sub_2 passed; then `scripts/launch_phase4.py --check` must return 0 with both `data/spike_0_6c/PASS` AND `data/spike_0_6d/PASS` present.
  - [ ] (Optional, advisory) Cell 6 — `scripts/run_spike_0_6d_1.py` against the existing Cell-8 history.csv for the Phase-5 record. Per Note 3 this does NOT gate.
  - [ ] (Optional, advisory) Cell 8 — `scripts/run_spike_0_6d_3.py` incompressible-mode cross-check. Per Note 3 this does NOT gate; skip in V1.
  - [ ] (Optional) Cell 10 — PAT-push the PASS markers + `results.json` files back to `main` for traceability.

  Notes on the 0.6c notebook (`notebooks/colab_spike_0_6c.ipynb`): no longer required for V1. Cells 8–10 (NACA 0012 benchmark + analyzer) are DEFERRED-TO-PHASE-5; Cell 7 is superseded by the 0.6d notebook's Cell 5b history.csv-evidence recovery path. The 0.6c notebook is retained for Phase-5 cross-solver use.
