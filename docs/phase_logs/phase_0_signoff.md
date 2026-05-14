# Phase 0 Signoff + V1/V2 Scope Decision (2026-05-13)

This document records the authoritative scope decision separating V1 (build a fan that demonstrably feels meaningfully better than the printed baseline) from V2 (quantify the gain with hardware-instrumented measurements). It is the canonical reference for why four Phase-0 spikes are marked deferred in their per-spike `data/spike_0_X/deferral.json` sentinels.

## Decision

**V1 is a sim-only + qualitative-feel project.** Hardware-instrumented measurement (anemometer, IMU, torsion pendulum) is deferred to V2.

The non-negotiable kept-in-V1 spike is **Spike 0.6c** — the CFD numerics validation against a published benchmark. Without 0.6c, the sim-side V1 is unmoored from physical reality. With 0.6c, fan-to-fan deltas of >15% in simulation are defensibly real.

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
| **0.6c Tier-1 cfg + benchmark validation** | needs `scripts/launch_phase4.py` wired (done) and the benchmark to actually run | **gates Phase 4 launch tag — must complete before Phase 4** |
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

- 480+ tests, all green.
- New deferral-aware paths exercised: `run_spike_0_4.py --i-wrist-analytic <value> --f-friction-cumulative-n <value>` honors 3× safety factor and skips the Spike-0.2 cross-check.
- All other V1-required spike runners pass their gates either canonically (clearance, engagement-force, cycle-life) or under dry-run plumbing exercise (0.6a/0.6b).

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
- [ ] **Action items remaining for operator before Phase 1 launch:**
  - [ ] Run Spike 0.6c when SU2 is installed on Colab Pro CPU (~12 h).
  - [ ] Confirm `scripts/launch_phase4.py --check` returns 0 once 0.6c PASSes.
