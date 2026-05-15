# Proposed plan edits — Spike 0.6d + Phase 5 published-ref benchmark

**Status:** DRAFT for line-by-line authorization. None of these edits land in `docs/report-final.md` until the user explicitly approves each one.

**Context:** Per the 2026-05-XX confidence assessment after the Spike 0.6c regime diagnostic, we agreed (this session) to:

1. Add a new Phase-0 spike **0.6d** with three sub-spikes (two gating, one advisory) that provide independent quantitative-sanity evidence on the production Tier-1 cfg before Phase 4 launches.
2. Replace the deferred Spike 0.6c.2 published-benchmark validation with a regime-appropriate version in Phase 5, bundled with the existing PyFR cross-solver work. OpenFOAM `pimpleFoam` joins as a third independent-codebase cross-check.
3. Leave operational Phase-4 monitoring (baseline-regression every N=50 acquisitions, MACH-perturbation rank-stability check) to the protocol doc, NOT the plan — minimal plan-text disruption.

The plan-edit philosophy is **purely additive**. None of the proposed edits modify an architectural lock (C-series, H-series, MED-series). The H10 Phase-4-launch-gate area gains a second marker file alongside the existing `data/spike_0_6c/PASS`; the existing gate is preserved exactly.

---

## Edit 1 — INSERT new Spike 0.6d block after current line 1844

**Type:** Pure addition. Inserts a new spike block between the existing Spike 0.6c block (lines 1839–1844) and the existing Spike 0.7 block (line 1846).

**Touches locked area:** H10 (Phase-4-launch gate). The proposed edit ADDS a second gate marker; it does NOT remove or modify the existing 0.6c marker. Both must PASS for Phase 4 to launch — the existing gate is strictly preserved.

**Insert (verbatim) immediately after line 1844:**

```text

**Spike 0.6d: Tier-1 quantitative-sanity counter-checks (H10 lock supplement — gates Phase 4 launch alongside 0.6c.1)**

- Motivation: The 2026-05-XX Spike 0.6c regime diagnostic (`docs/phase_logs/spike_0_6c.md` Note 1) confirmed the production Tier-1 cfg simulates body-in-still-air physics correctly but cannot be validated against published wind-tunnel reference data — the regimes differ, and Sub-spike 0.6c.2 was deferred to Phase 5 on that basis. To prevent Phase 4 from burning ~1300 GPU-hours on a Tier-1 solver output we have only consistency evidence for, three lightweight independent checks land before Phase 4 launches. The objective is to convert "we have no independent quantitative check on SU2 at MACH=1e-9" into "we have order-of-magnitude + closed-form + same-solver-incompressible-mode cross-evidence" — the absolute-accuracy claim still lives in Phase 5 (cross-solver + published-reference, see step 62.5).
- **Sub-spike 0.6d.1 — Symmetry + dimensional-force sanity** (gating): on the existing Spike 0.6c Cell 8 SU2 history.csv plus one fresh production-Tier-1 run on the flat-plate baseline geometry, verify (a) cycle-averaged F, M within tolerance of zero (physical symmetry of periodic pitching), (b) dimensional cycle-peak force lies within ±1 order of magnitude of the analytic added-mass + quadratic-drag envelope `m_panel × ω² r ≈ 0.2 N`. **Pass criterion:** (a) `|F_cycle_avg| < 0.05 × F_cycle_peak`; (b) dimensional `F_cycle_peak ∈ [0.02, 2.0] N` for the V1 flat-panel reference. **Cost:** ~1–2 h Colab CPU. **Implementation:** `src/fanopt/cfd/spike_0_6d.py` + `scripts/run_spike_0_6d_1.py`; reuses the existing `diagnose_su2_pitching_regime.py` infrastructure.
- **Sub-spike 0.6d.2 — 2D flat-plate added-mass analytic check** (gating): render a 2D thin-plate pitching cfg matching the production Tier-1 numerics (MACH=1e-9, RIGID_MOTION, dual-time-stepping, same dt and inner-iter count) and compare SU2's inviscid-phase pitching moment coefficient about the pivot against the closed-form Sedov/Newman added-mass moment `M_added = -m_a I_pitch θ̈`. **Pass criterion:** SU2 cycle-peak inviscid-phase moment within ±15% of the closed-form added-mass prediction. **Cost:** ~2–4 h Colab CPU. **Implementation:** new 2D thin-plate cfg template `configs/su2/thin_plate_2d_pitching.cfg.j2` + renderer in `src/fanopt/cfd/configs.py` + `scripts/run_spike_0_6d_2.py`.
- **Sub-spike 0.6d.3 — SU2 incompressible-mode cross-check** (advisory, NOT gating): duplicate the production Tier-1 cfg with `SOLVER= INC_NAVIER_STOKES`, same mesh + motion + pitching schedule. **Pass criterion (advisory):** dimensional cycle-averaged forces agree with compressible-mode-with-MACH=1e-9 output within ±20%. **Cost:** ~2–3 h Colab CPU. **Why advisory not gating:** same-solver cross-check is weaker independence evidence than cross-codebase. Failure here is documented as a Phase-5 investigation item but does not block Phase 4.
- **Fail action:** if 0.6d.1 or 0.6d.2 fails, investigate the production Tier-1 cfg (low-Mach preconditioner coefficients, dual-time inner-iter count, dt convergence, possibly a finer mesh) before Phase 4 launches. Phase 4 launch is **gated on Spike 0.6c.1 PASS AND Spike 0.6d.1 PASS AND Spike 0.6d.2 PASS**; `scripts/launch_phase4.py` refuses to create the `phase4-launch` tag unless `data/spike_0_6c/PASS` AND `data/spike_0_6d/PASS` markers are both present. The H10 lock's existing `data/spike_0_6c/PASS` requirement is preserved; the 0.6d marker is an additive supplement.
```

**Rationale:** Pure addition; the new spike has its own marker file in its own directory (`data/spike_0_6d/`); the existing H10 lock area is supplemented, not modified. Effort: ~3 days my work + ~6 hours Colab compute total across all three sub-spikes.

---

## Edit 2 — UPDATE line 1844 fail-action sentence (additive clause)

**Type:** Substantive edit to a line that documents the H10 lock. Touches a locked area.

**Touches locked area:** H10 (Phase-4-launch gate). The proposed edit clarifies that the existing gate is now one of two gates; it does NOT remove or weaken the existing gate.

**Current text (line 1844, verbatim):**

```text
- **Fail action:** if the benchmark misses by >15%, investigate (mesh quality, dt convergence, low-Mach prec coefficients, dual-time inner iterations) before Phase 4 launches; do NOT silently proceed. Phase 4 launch is **gated on Spike 0.6c passing**; `scripts/launch_phase4.py` refuses to create the `phase4-launch` tag if `phase0/spike_0_6c/PASS` marker is absent.
```

**Proposed text:**

```text
- **Fail action:** if the benchmark misses by >15%, investigate (mesh quality, dt convergence, low-Mach prec coefficients, dual-time inner iterations) before Phase 4 launches; do NOT silently proceed. Phase 4 launch is **gated on Spike 0.6c.1 passing** (the V1 reduced 0.6c gate; 0.6c.2 deferred to Phase 5 per 2026-05-XX decision, see Spike 0.6d for compensating quantitative-sanity checks); `scripts/launch_phase4.py` refuses to create the `phase4-launch` tag if `data/spike_0_6c/PASS` is absent **or if `data/spike_0_6d/PASS` is absent** (the 0.6d gate is additive to the existing 0.6c gate, not a replacement). Marker path `phase0/spike_0_6c/PASS` in earlier plan text resolves to `data/spike_0_6c/PASS` per the as-built repository layout.
```

**Rationale:** Reflects the operational reality already established: B's work moved 0.6c.2 to Phase 5 and reduced the 0.6c gate to 0.6c.1 alone; A's work landed a Phase-5-prep wind-tunnel-frame cfg. This edit makes the additive 0.6d gate explicit in the plan and reconciles the `phase0/spike_0_6c/PASS` vs `data/spike_0_6c/PASS` path divergence between plan text and code.

---

## Edit 3 — UPDATE line 3775 V1 required-spikes list

**Type:** Substantive edit to a sentence summarizing V1 scope. Additive (adds 0.6d to a list).

**Touches locked area:** None. This sentence summarizes the V1/V2 split (which was authored in the 2026-05-13 signoff decision, not an architectural lock).

**Current text (line 3775, verbatim):**

```text
**Spikes still required for V1:** 0.0 (done), 0.1 (done), 0.4, 0.6, 0.6a/0.6b (gates), **0.6c (non-negotiable — Phase 4 launch gate)**, 0.7a, 0.7b. The 15-30% gain target in §0 row 35 is **suspended for V1**; V1 reports sim-vs-sim relative gain plus the operator's qualitative feel comparison. The numerical gain target re-enters in V2 once a quantitative baseline exists.
```

**Proposed text:**

```text
**Spikes still required for V1:** 0.0 (done), 0.1 (done), 0.4, 0.6, 0.6a/0.6b (gates), **0.6c.1 (non-negotiable — Phase 4 launch gate; 0.6c.2 deferred to Phase 5 per 2026-05-XX decision)**, **0.6d.1 + 0.6d.2 (Tier-1 quantitative-sanity counter-checks; gate Phase 4 launch alongside 0.6c.1; 0.6d.3 advisory)**, 0.7a, 0.7b. The 15-30% gain target in §0 row 35 is **suspended for V1**; V1 reports sim-vs-sim relative gain plus the operator's qualitative feel comparison. The numerical gain target re-enters in V2 once a quantitative baseline exists.
```

**Rationale:** Inline reconciliation of the V1 spike list with the 2026-05-13 deferral + this session's 0.6d addition. No scope change; just accuracy.

---

## Edit 4 — INSERT new row in Appendix B compute-time table (after line 3936)

**Type:** Pure addition.

**Touches locked area:** None (operational accounting).

**Insert (verbatim) as a new row immediately after the line-3936 Spike-0.6c row:**

```text
| **Spike 0.6d — Tier-1 quantitative-sanity counter-checks (H10 supplement)** | Colab Pro CPU | **~5–9 hours total** (1–2 h 0.6d.1 + 2–4 h 0.6d.2 + 2–3 h 0.6d.3 advisory) | Phase 0; gates Phase 4 launch alongside 0.6c.1. NOT booked against the 1000-h Phase 4 stop rule. |
```

**Rationale:** Matches the format and accounting convention of the existing Spike-0.6c row. Phase-0 compute, doesn't draw against the 1000-h Phase-4 stop rule.

---

## Edit 5 — INSERT new Phase 5 step 62.5 (after current line 2348)

**Type:** Pure addition. Inserts between existing step 62 (PyFR top-3 verification) and existing step 63 (final ranking).

**Touches locked area:** None directly. The Phase-5 PyFR-on-G4 hardware lock (HIGH-11) is preserved; this step ADDS another item to Phase 5's deliverables, no hardware path change.

**Insert (verbatim) immediately after current line 2348 (the existing step 62 bullet that ends with "...needing investigation (mesh refinement, time-step reduction, or solver re-tuning)."):**

```text
62.5. **Body-in-still-air published-reference benchmark** (Phase-5 absolute-accuracy validation; replaces 2026-05-XX-deferred Sub-spike 0.6c.2):
 - **Reference case:** pick a published body-in-still-air pitching/flapping case with full force-trace data at Re ~ 10³–10⁴. Default candidates: Sane & Dickinson 2002 robotic-flapper protocol; Dickinson/Ellington/Birch insect-flight datasets; Sarpkaya plate-in-oscillating-flow tabulated coefficients (Morison-equation literature). Pick is documented in `docs/phase_logs/phase_5_signoff.md` at Phase-5 launch.
 - **cfg:** render a body-in-still-air cfg matched to the reference's kinematics + Reynolds via the wind-tunnel-frame template (`configs/su2/oscillating_airfoil_benchmark.cfg.j2`), with `MACH_NUMBER` set per the production Tier-1 lock and a prescribed body motion matching the reference. The template generalizes (freestream → 0 + prescribed motion ON) without a separate cfg.
 - **Three-solver comparison:** run the reference case through (a) SU2 compressible with MACH=1e-9 + low-Mach prec (production Tier-1 numerics), (b) PyFR p=3 on Colab Pro G4 GPU (existing HIGH-11 lock), (c) **OpenFOAM `pimpleFoam` incompressible** on Colab Pro CPU (independent codebase; native incompressible, no low-Mach preconditioning involved — strongest available evidence on whether SU2's MACH=1e-9 trick is quantitatively faithful).
 - **Pass criterion:** SU2 and PyFR each match published cycle-averaged forces within ±15%; SU2 ↔ PyFR mutual agreement within ±10%; OpenFOAM agreement with SU2 within ±20% (advisory — disagreement flagged for investigation, not blocking).
 - **Cost:** SU2 ~6–12 h CPU; PyFR ~2–4 h GPU; OpenFOAM ~6–10 h CPU. Total Phase-5 compute addition ~14–26 h.
 - **Fail action:** if SU2 misses published cycle-averaged forces by >15%, Phase 5 reranking flags absolute-accuracy as compromised and the V1 ship decision falls back entirely on the Phase-6 qualitative blinded A/B feel test (which is already the V1 ship criterion — so V1 is not blocked, but V2 quantitative claims downstream of Phase 5 are gated on resolving this).
```

**Rationale:** Replaces the deferred 0.6c.2 framing with a regime-appropriate target. OpenFOAM as a 3rd solver per this session's decision — strongest available evidence on the low-Mach trick. Bundles cleanly with existing Phase-5 PyFR work; no hardware lock changes. The fail action explicitly preserves V1 ship criterion.

---

## Summary table of proposed edits

| # | Where | Type | Touches lock? | Lines added | Lines modified |
|---|---|---|---|---|---|
| 1 | After §Phase 0 line 1844 | Pure addition | H10 (additive only) | ~10 | 0 |
| 2 | Line 1844 (in-place) | Substantive | H10 (additive only) | 0 | 1 |
| 3 | Line 3775 (in-place) | Substantive | None | 0 | 1 |
| 4 | After line 3936 | Pure addition | None | 1 | 0 |
| 5 | After line 2348 | Pure addition | None | ~8 | 0 |

Net plan-text impact: **~19 lines added, 2 lines modified, 0 lines removed**. Zero architectural locks modified (only additive supplements).

---

## Companion non-plan additions (no plan-edit authorization needed)

After plan edits 1–5 are authorized, these companion docs will be created/updated directly (additive, established pattern):

- **`docs/spike_0_6d_protocol.md`** (NEW) — protocol mirroring `spike_0_6c_protocol.md` structure. Documents the three sub-spikes' inputs/outputs/pass criteria/fail actions in operator-runnable detail. Includes the recipes for running each on Colab.
- **`docs/phase_logs/phase_0_signoff.md`** (UPDATE) — add Note 2 documenting the 0.6d countermeasure spike + its rationale, linked to Note 1 (the 2026-05-XX 0.6c.2 deferral).
- **`docs/phase_logs/spike_0_6d.md`** (NEW) — empty stub for the eventual Spike 0.6d findings log; mirrors `spike_0_6c.md` pattern.
- **`docs/phase_checklist.md`** (UPDATE) — add Spike 0.6d rows after the 0.6c rows.
- **`scripts/launch_phase4.py`** (UPDATE) — gate now requires both `data/spike_0_6c/PASS` AND `data/spike_0_6d/PASS`; companion test updates.

These follow the same additive pattern used for B's 0.6c.2 deferral work (Note 1 in phase_0_signoff.md). They do NOT require plan-edit authorization since they're operational docs, not the plan source-of-truth.

---

## Phase-4 in-flight monitoring (deferred to protocol, NOT plan)

Per this session's scoping discussion, the operational Phase-4 monitoring items DO NOT go in the plan:

- **Baseline-regression every N=50 BO acquisitions:** re-run flat-plate baseline through SU2, log `J_fan`, flag drift > ±10% across checkpoints.
- **Rank-stability under MACH perturbation at Phase 4 midpoint:** re-run 20 representative geometries at MACH=1e-7, verify Pareto ranking stability.

These will land in `docs/phase4_runbook.md` (NEW) or as additions to the existing Phase 4 protocol section in the architecture-bandit protocol. They're operational guidance, not plan-architectural items — and adding them to the plan would bloat its scope without value.

If the user prefers these in the plan, that's a sixth proposed edit; flag explicitly.

---

## Implementation effort estimate (post-authorization)

| Work item | My effort | Compute cost | Calendar |
|---|---|---|---|
| Plan edits 1–5 (authorized one at a time) | ~30 min total | none | minutes |
| `spike_0_6d_protocol.md` + `phase_logs/spike_0_6d.md` stub + `phase_0_signoff.md` Note 2 + `phase_checklist.md` rows | ~2 hours | none | half day |
| `src/fanopt/cfd/spike_0_6d.py` + `Tier1SymmetryDimensionalResult` + `Tier1AddedMassResult` + `Tier1IncompResult` dataclasses + analyzers + tests | ~1 day | none | 1 day |
| `configs/su2/thin_plate_2d_pitching.cfg.j2` + renderer + tests | ~half day | none | half day |
| `scripts/run_spike_0_6d_1.py` + `run_spike_0_6d_2.py` + `run_spike_0_6d_3.py` + `run_spike_0_6d.py` aggregator + tests | ~1 day | none | 1 day |
| `scripts/launch_phase4.py` extension + tests | ~1 hour | none | hours |
| 0.6d.1 + 0.6d.2 + 0.6d.3 Colab runs | (user time) | ~5–9 h CPU | half-day calendar |
| Final §7.1 verification pass + commit | ~30 min | none | minutes |
| **TOTAL** | **~3 days my work** | **~5–9 h Colab CPU** | **~1 week calendar** |

Phase 5 step 62.5 implementation is in-Phase-5 work, not in-this-session work.

---

## Decision needed before any plan edit lands

Per CLAUDE.md §7.2 ("Claude does not edit the plan except when the user explicitly authorizes a specific additive edit"), each of the five proposed plan edits needs explicit go-ahead. The user can:

- **Approve all five at once** — fastest; we proceed to implementation.
- **Approve a subset** — e.g., approve 1, 4, 5 (pure additions); defer 2, 3 (in-place substantive edits) pending discussion.
- **Reject any specific edit** — flag the concern, redraft as needed.
- **Defer the whole thing** — implement only the non-plan-edit companion files (protocol, signoff Note 2, checklist) and leave the plan untouched; the plan text will be stale relative to operational reality but the V1 path is still complete.
