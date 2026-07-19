# V2 Backlog

Canonical, expanded V2 plan. The in-spec summary lives at `../report-final.md` §13; this file is the authoritative location for V2 work descriptions, triggers, and acceptance criteria.

---

## Deferred Phase-0 spikes (V1 scope pivot, 2026-05-13)

These spikes were originally Phase 0 deliverables. They are deferred to V2 to keep V1 free of specialized measurement hardware purchases. The V1 substitute approach lives next to each one. Decision rationale: `docs/phase_logs/phase_0_signoff.md`. Per-spike sentinels: `data/spike_0_{2,3,5,7c}/deferral.json`.

### Spike 0.2 — Torsional-pendulum I_wrist measurement

**Why deferred:** torsion-wire rig + reference rod + sub-mm caliper measurements are research-grade rigor for a personal project. The plan's `J_fan / W_cycle` ratio is the binding artifact, and V1 substitutes a simpler unit (see Spike 0.3 below).

**V1 substitute:** analytic `I_wrist_kgm2` from the §6.4 generator (`i_wrist_assembly`). The Spike 0.4 force balance consumes the analytic value via the new `--i-wrist-analytic <float>` + `--f-friction-cumulative-n <float>` flags on `scripts/run_spike_0_4.py`, with the safety factor bumped 2× → 3× to absorb the unverified-inertia uncertainty.

**Revisit trigger:** V1 ships a fan that subjectively feels meaningfully better than the printed baseline AND the operator wants to quantify the improvement.

**V2 acceptance:** repeatability < 3% across 5 trials; cross-check vs the analytic `i_wrist_assembly` value within ±10% (per the original Spike 0.2 protocol). If the cross-check fails, the analytic value used during V1's Spike-0.4 force balance was wrong and the rib-tab fallback might have been required but wasn't armed — Phase 6 needs a retest.

### Spike 0.3 — Anemometer + IMU baseline measurement

**Why deferred:** anemometer + 9-point grid + dedicated IMU is hardware the operator does not own and will not purchase for V1.

**V1 substitute:** two co-baselines, both sim-side. **(a)** Phase 2a baseline CFD on the flat-panel 10-blade design — emits the simulated `J_fan` that every optimized design's simulated `J_fan` is compared against (sim-vs-sim relative gain). **(b)** Phase 6 qualitative blinded A/B feel-test of printed top-3 designs vs. the printed baseline. A blinded protocol (operator hands fans without naming them; stopwatch-paced 20 strokes; 1-5 score on each of airflow / weight / sound / aesthetics) is the recommended V1 reporting form.

**Revisit trigger:** V1 ships a fan that subjectively feels better and the operator wants quantitative confirmation. Three V2 upgrade paths in order of cheapness:
1. **Kitchen scale + cardboard target** (~$0, ~15 min protocol) — see `docs/spike_0_3_protocol.md` Appendix A.
2. **Phyphox phone IMU** (free; phone already owned) — `src/fanopt/physical/imu.py` already reads CSVs in the right format.
3. The original anemometer + IMU rig per `docs/spike_0_3_protocol.md` body.

**V2 acceptance:** any V2 path must produce a `J_fan` baseline that V1's printed top-3 can be compared against. The ≥15% gain target only applies once a measured baseline exists; until then, V1 reports sim-vs-sim deltas.

### Spike 0.5 — Single-blade fabrication-noise CV

**Why deferred:** 3-copy CV requires printing 3 nominally-identical blades, instrumenting each, then measuring J_fan across three otherwise-identical assemblies. This is the same hardware-instrumentation cost as Spike 0.3 plus three extra print runs.

**V1 substitute:** print one V1 top candidate **twice** (same design, same printer, same settings) as a same-design sanity check at Phase 6. Compare by feel. If the two prints feel meaningfully different, the print-noise floor is wider than the V1 design-gain target and the V1 design comparison is contaminated — flag and discuss with the operator before declaring V1 ship-ready. No formal CV computation in V1.

**Revisit trigger:** V1 quantitative metrics matter (kitchen scale or anemometer). The 3-copy CV gates whether sub-15% deltas are real.

**V2 acceptance:** as originally specified — CV < 5% across the three measured fans.

### Spike 0.7c — Sobol-vs-BO iso-compute baseline

**Why deferred:** the 430 h Phase-0 budget is sized for an honest BO-vs-baseline head-to-head. For V1, the operator commits to BO without the formal validation.

**V1 substitute:** BO-stall fallback. If Phase 4 Tier-0 best-J_fan does not improve over 20 consecutive acquisitions within an architecture, the orchestrator switches to hand-picked diverse candidates rather than burning more compute. Diverse-candidate rule: one near-baseline, one louver-heavy, one TPMS-heavy, one high-camber, one asymmetric — span Layer 2 archetypes rather than 5 variations of one shape.

**Revisit trigger:** V1 BO observably stalls AND the operator wants to know whether BO is fundamentally outperforming Sobol on this objective. Without the trigger, V2 may simply skip this entirely.

**V2 acceptance:** as originally specified — BO best-J_fan ≥ Sobol best-J_fan by ≥ 5% on at least 2 of 3 budgets {30, 100, 300} h.

### V1 → V2 cheap mitigations adopted at decision time

Documented here so they don't get lost between rounds:

- **Diverse Phase 5 print candidates.** The top-3 printed designs must span Layer 2 archetypes (not 3 variations of one shape). Mitigates BO-exploits-sim-artifact failure mode.
- **Print one top candidate twice.** Same-design sanity check at Phase 6 substitutes for Spike 0.5 fab-noise CV.
- **Blinded A/B in Phase 6.** Operator hands fans without naming them; stopwatch-paced 20 strokes; 1-5 score on airflow / weight / sound / aesthetics. Repeat on a different day. Free; ~20 min per comparison.

---

## Triggered items (V1 failure → V2 in-scope)

Each item has a triggering condition that fires at Phase 6 wrap-up (or earlier if the trigger is a Phase 4 diagnostic). If the trigger fires, the V2 effort begins with the corresponding entry as its first deliverable.

### V2 designed lock mechanism

**Trigger:** Phase 6 testing shows the fan unlocks under sustained 2 Hz waving (the H6 V1 force balance passes in Spike 0.4 but fails in practice).

**V1 fallback:** the rib-tab fallback (`params.layer4.v1_lock_fallback_enabled`) is armed conditionally if Spike 0.4 force balance fails. V2 supersedes the rib-tab with a designed lock.

**V2 scope:**
- Locked-cantilever snap engaging the guard rib outer face at the deployed angle (120° default).
- Magnetic-catch alternative (embedded N52 neodymium discs, 2 mm × 1 mm; ~1 g per pair, ~20 g across 10 blades; within the 60 g mass budget).
- Cycle-life test: ≥10,000 deploy/fold cycles without functional degradation.

**Acceptance:** 30-min sustained 2 Hz waving session at 40° amplitude with no inter-pair friction-driven unlock event.

### Centrifugal Filter 4 (re-introduce a real Filter 4)

**Trigger:** Phase 6 testing reveals fatigue failures at the pivot driven by centrifugal pull under aggressive waving (the kind the canonical Filter 2 cyclic check misses).

**V1 status:** Filter 3 is a deprecated pass-through stub. Centrifugal load is covered by the one-shot Phase-2 dynamic-load assertion (`α_peak · m_rib · r_tip · N_blades < 0.1 · click_detent_allowable`), not per-design.

**V2 scope:**
- Proper per-design Filter 4 with the correct kinematics: cyclic tangential reaction at click detent + centrifugal stress at the boss, both computed with **H8 wrist-to-tip lever arms** (`L_wrist_to_tip = 0.25 m`, NOT `L_blade = 0.20 m`).
- σ_centrifugal at boss = `m_blade · ω_blade_max² · r_boss / (2 · A_boss)` evaluated at the boss radius (7 mm); compared against the §10.1 bearing allowable (2.00 MPa Z-direction binds).

**Acceptance:** Filter 4 rejects designs that fail the Phase 6 centrifugal fatigue criterion, on the cohort of Phase 4 designs evaluated by the time the trigger fires.

### Alternative MFBO architectures for TPMS/noise

**Trigger:** L7 empirical-bias diagnostic fires often in Phase 4 (mean `|Δ_TPMS / mean_J_fan_tier0| > 0.30` across the first 100 Tier-0 evals).

**V1 status:** TPMS/noise architectures use the 0.3/0.7 reweighting compromise (Tier 0 weight 0.7, Tier -1 weight 0.3) for promotion decisions.

**V2 scope (pick (a) or (b) based on Phase 4 diagnostics):**

(a) **Disable multi-fidelity GP for TPMS/noise architectures** — run Tier-0-only single-fidelity GP per affected architecture. Removes the Tier -1 ↔ Tier 0 correlation kernel where it's known to be mis-specified.

(b) **Treat Tier -1 as a separate cheap-feature input** — concatenate `J_fan_tier_minus_1_proxy` as an extra GP input dimension rather than a fidelity tag. Preserves the cheap-screening signal without forcing the multi-fidelity kernel.

**Acceptance:** affected architectures' Tier 0 → Tier 1 Spearman ρ² improves by ≥ 0.1 over the V1 0.3/0.7 reweighting baseline.

---

## V1.5 — Staggered AO↔TO co-optimization (computational-only, no new hardware)

**Origin:** operator design discussion 2026-07-18. V1 runs aero optimization (AO,
Phase 4 BO) and rib topology optimization (TO, Phase 2 SIMP) **sequentially** and
decoupled, with the §59.5 combined-blade FEA gate as a one-way verification (it
*rejects* under-built designs but doesn't *feed back*). This item closes the loop
with a **staggered (block Gauss-Seidel) AO↔TO iteration** — alternate AO and TO,
passing loads/mass between them, until the coupled objectives converge.

**Why this is V1.5, not V2:** it is **purely computational** — every objective is
simulated (J_fan from SU2, I_wrist analytic, compliance/stress from FEA). It needs
**none** of the deferred V2 measurement hardware (Spikes 0.2/0.3/0.5). The
epistemic status is identical to V1: optimize in sim, validate by the printed
blinded A/B feel-test.

**Prerequisites — both solvers already exist and have run:**
- AO loop ✅ — Phase 4 machinery (`fanopt.bo.orchestration`, 208-eval campaign).
- TO loop ✅ — Phase 2 rib SIMP (`fanopt.topopt.{simp,plate_bending,loads,solver}`,
  landed `8565212`; converged, −71.6% compliance at volfrac 0.4).
- Missing = **coupling orchestration only** (a `scripts/run_staggered_mdo.py` +
  the live load-passing wiring). Phase 2a already does CFD→structural-load
  extraction (`loads.py`), so the AO→TO direction has precedent.

**Architecture sketch (`scripts/run_staggered_mdo.py`):**
1. Seed from the V1 Phase-4 Pareto winner(s).
2. **AO → TO:** extract the winning design's SU2 pressure field → map to the rib
   structural load → run rib SIMP TO.
3. **TO → AO:** the TO'd rib updates **mass → I_wrist** (already a BO objective)
   and support stiffness; re-run a *bounded* Phase-4 AO around the current design.
4. Repeat until ΔJ_fan and ΔI_wrist between passes fall below tolerance
   (≈ 2–4 outer iterations expected).

**Coupling channels + honest expected payoff:**
- **mass → I_wrist** (TO shaves rib material → better wrist-feel at equal airflow):
  the main, cleanly-captured win.
- **aero-pressure → rib load:** weak (few-Pa aero vs ~10–25× larger inertial/click
  loads); marginal.
- **panel-compliance → as-loaded aero shape** (the panel flexes 5–15 mm under aero,
  §3.1 note): the *biggest* coupling, but it needs a **static-deflection step**
  (deflect panel under aero, re-mesh, re-run) — the V1 "No FSI" lock (§2.3) is
  relaxed here. Still 100% computational, no hardware; it's the extra machinery
  that makes the loop worth doing.

**Cost:** each outer pass ≈ one bounded AO campaign (hours–1 day on an 8-core Colab
CPU) + a TO solve (~30 min). 2–4 passes ≈ a few days, CPU-only. GPU only becomes
relevant if an **ML-surrogate TO** is added to accelerate the inner loop — exactly
the scenario `report-final.md` §6.3 (ML-for-TO) flags as worthwhile *only* under
iterative TO↔ASO coupling.

**Relationship to V2 / V3:** V1.5 is the lightweight precursor. V2's queued
**Winkler-foundation BC** (§13.3 / `report-final.md` §3.1 rib-panel BC note)
captures the panel-compliance coupling *without* a full loop. Full **monolithic
MDO** (coupled SU2↔FEniCSx adjoints, simultaneous aero+topology) is the rigorous
end-state — see Out-of-scope (V3+/research) below.

**Trigger:** V1 ships (printed + feel-tested) and the operator wants a
tighter-coupled design without buying measurement hardware.

**Acceptance:** the staggered loop converges (Δobjectives < tol), and the
co-optimized design Pareto-dominates the V1 sequential winner on (J_fan, I_wrist)
while still passing the §59.5 combined-blade structural gate.

---

## ML-driven TO + AO (research track — V2/V3)

**Origin:** operator direction 2026-07-18 — push the design space with genuinely
**ML-based** topology *and* aero optimization (surrogate + generative), not just
deterministic SIMP + GP-BO. This is the ambitious end-state; V1.5's staggered
AO↔TO loop is the harness it plugs into (same objectives, params schema, §59.5
gate).

**Why it's compelling here:** the CFD is the binding cost (~85 min/eval; the
208-eval Phase-4 campaign took ~23 h). A trained aero surrogate that predicts
J_fan in milliseconds turns a multi-day campaign into minutes, unlocking orders
of magnitude more design exploration and making generative search tractable.

**North star — escape the parameterization ceiling (operator insight 2026-07-18):**
V1's 35-variable codec is an **expert-priored simplification chosen for compute
tractability** (GP-BO degrades past ~40 dims; each CFD eval is expensive), *not* a
fundamental limit. It can only find the best fan **expressible in that hand-picked
basis** — `report-final.md` §7 is explicit: "BO searches *within* them but cannot
invent a 6th field type or a primitive shape outside the library." So V1/V1.5 are
optimal *within the box*; the deepest payoff of the ML route is **removing the box**,
because cheap surrogate evals + GPU make a far larger space searchable. Progression
of design freedom:
- (a) **Higher-dimensional parameterization** — more Fourier modes / fields / finer
  control (hundreds of dims: intractable for GP-BO, fine for a neural surrogate +
  gradient or generative search).
- (b) **Free-form representations** — neural implicit fields (a network *is* the
  shape) or voxel/mesh-level TO (thousands of density variables): no hand-picked
  basis, arbitrary topology.
- (c) **Generative latent spaces** — diffusion / GAN / VAE that learn a manifold of
  valid designs from data and generate novel topologies the codec cannot express.

**What still binds (NOT simplifications to remove):** the architectural +
manufacturing locks — panel-pivot architecture, m < 100 g, single-material PETG
printability, the click mechanism, blade-count range, the kinematic load cases.
These **define a valid, buildable fan**; free-form search explores shapes *within*
them, and the physics gate (§59.5 + real SU2/FEA) verifies buildability. Distinguish
a **compute-driven basis simplification** (escape it) from a **product-defining
constraint** (keep it).

**The hard part = data + GPU (be honest about this):** ML TO/AO needs thousands of
(design → response) pairs from SIMP/SU2. V1 has a *seed* (208 aero evals + 1 rib
TO), not a training set, and the fan is bespoke (no off-the-shelf dataset). **Data
generation is the dominant cost** and is GPU/compute-heavy. Mitigations: active
learning (sample only where the surrogate is uncertain), transfer learning, and
physics-informed operators (PINN / DeepONet / Fourier Neural Operator) that need
less data. **Physics stays ground truth** — the surrogate proposes; SU2/FEA + the
§59.5 gate dispose. Never ship an unverified ML output.

**Staged path (highest ROI first):**
1. **ML aero surrogate** — biggest win (CFD is the bottleneck). Train a model
   (CNN/GNN on the flow field, or an FNO for the PDE operator) on the accumulated
   SU2 evals to predict J_fan / pressure. Bootstrap from the existing Phase-4
   ledger (`evaluations.jsonl` + design vectors) via active learning; replace most
   SU2 calls in the BO inner loop, keep periodic SU2 spot-checks.
2. **ML TO surrogate** — CNN/FNO predicting the rib compliance/stress field, to
   accelerate the V1.5 staggered inner loop (the §6.3 "iterative TO↔ASO" case).
   GPU re-entry point on the structural side.
3. **Generative design** — VAE/GAN/diffusion over rib topologies + panel shapes
   that *generate* near-optimal candidates from loads/BCs, seeding the physics
   verifier instead of blind BO/SIMP starts. Hardest to constrain to feasibility;
   best as a candidate-generator feeding Filter 2 + §59.5.
4. **End-to-end differentiable / neural-operator MDO** — the research end-state:
   differentiable aero + structural surrogates enabling gradient-based *monolithic*
   coupled MDO, or RL/generative agents over the joint design space.

**First concrete step:** an `src/fanopt/ml/` + notebook prototype — an aero
surrogate trained on the Phase-4 ledger, validated against held-out SU2 evals, with
an active-learning acquisition. **Success:** predicts J_fan within the CFD noise
floor on held-out designs and cuts SU2 calls ≥5× in a re-run campaign. This is
where GPU finally matters across the whole pipeline (surrogate training +
generative inference), not only PyFR.

---

## Optional (V1-complete, V2-improves)

Items where V1 ships a working solution but V2 has a clear path to a better one. No triggering condition required; V2 picks these up as time permits.

### Mid-Phase-4 rib re-tune

**Current V1 spec:** Phase 2 rib SIMP TO runs once with a smooth-baseline panel placeholder.

**V2 scope:** re-trigger Phase 2 rib SIMP TO every K Phase-4 architecture promotions, conditioned on the panel topology the architecture-bandit is actually selecting. Timing: re-tune happens after the first K promotions complete (≈ Phase 4 month 1).

**Cost:** ~3-5 additional Phase 2 SIMP solves × ~30 min each = 1.5-2.5 hours per re-tune; per-architecture-class, not per-design.

### Textured-PEI bed-surface portability

**Current V1 spec:** §3.2.4 / M13 lock smooth-PEI (Bambu Cool Plate Super Tack AP05, Ra ≤ 5 µm).

**V2 scope:** document the §3.2.4 wall-roughness calibration procedure so users on other bed surfaces (Prusa textured PEI Ra 10-30 µm, Anycubic frosted PEI Ra 15-25 µm) can re-derive the roughness-model parameters. Includes a portable Phase 0 sub-spike that measures the bed-contact face Ra in-situ and refits the calibration coefficients.

### Asymmetric-stroke physics in J_fan

**Current V1 spec:** J_fan is symmetric in time (integrates over full cycles).

**V2 scope:** explore an asymmetric weighting `J_fan_biased = w_p · J_productive_half + (1 − w_p) · J_return_half` with `w_p` measured from IMU during Phase 6. Would change the optimization target away from the parachute baseline more aggressively (the symmetric metric rewards equal forward/backward drag; the biased metric rewards productive-stroke drag specifically).

### `directional_asymmetry_score` functional-form refinement

**Current V1 spec (C6 lock):** starter form from §Phase 3 step 33:

```
directional_asymmetry_score(design) :=
    sum over Layer 2 louver fields:
        (louver_count) × |sin(louver_angle)| × (active flag)
    + |Fourier_LE_phase_offset − Fourier_TE_phase_offset| · 0.1
    + sum over Layer 3 primitives:
        (polarity_sign) × (primitive_size_relative_to_chord)
```

**V2 scope:** converge on the functional form that best predicts the Phase 6 IMU-measured J_fan spread across designs (richer signal than the Phase 0 3-radius calibration sample). Candidates:
- weighted sum of Layer 2 louver angles only,
- Fourier TE/LE phase difference only,
- integrated `|chord_z⁺(x) − chord_z⁻(x)|` over the planform (camber asymmetry),
- the starter sum-of-three form from V1.

Pick the candidate with highest R² on the Phase 6 dataset. The score is dimensionless; β carries dimensional scaling to J_fan units.

## Out-of-scope (V3+ or research)

Items that are not in the V1 or V2 roadmap, queued for either a V3 effort or a research follow-on.

- **Active electronic flow control** — embedded micro-blowers or piezo actuators in the panel cutouts. Adds power + control complexity; out of V2 scope.
- **Multi-DOF wrist motion** — current model assumes pure +y wrist rotation (flexion). Real waving has yaw + pitch + roll components. V3 could extend the SU2 unsteady cfg to support compound rotations.
- **Multi-material printing** — TPU membranes, dual-extruder panels. V1 explicitly rejects this (single-material PETG except the steel/brass pin); V2 stays single-material. V3 could revisit if multi-material AMS toolchains improve.
- **Adjoint-based aero shape optimization on the panel envelope** — V1 uses generative parametric design (4-layer hybrid); V3 could couple SU2 continuous adjoint to the Layer 1 envelope spline directly for a finer-grained gradient-based refinement of the top-1 Pareto design.
