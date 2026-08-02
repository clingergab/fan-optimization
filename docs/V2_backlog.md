# V2 Backlog

Canonical, expanded V2 plan. The in-spec summary lives at `../report-final.md` §13; this file is the authoritative location for V2 work descriptions, triggers, and acceptance criteria.

---

## Audit 2026-08 — toolchain & approach retrospective (V2 input)

A near-end-of-V1 clean-slate evaluation (5 domain deep-dives + 2 adversarial passes; the metric was
recomputed from raw SU2 data). Full record: **`docs/audit_2026-08_toolchain_and_approach.md`**. Headline:
the toolchain and the V1 *result* are sound (the campaign optimized a **validated** signed-rectification
signal, not noise); **none of the findings improve the existing V1 designs** — they are V2 input, plus one
cheap V1 validation correction. Findings map onto existing backlog sections:
- **Validation instrument must be directional** → Spike 0.3 above (⚠ correction added there). *The one V1-relevant item.*
- **Collapse the 12 panel-offset dims → 1-2 tangential scalars** (radial offsets redundant with the ±20mm
  meridian; tangential fold-locked) → "Deferred design-space relaxations" below.
- **Real 2-fidelity MF-BO (gated on coarse↔fine R²>0.75) · √D GP prior · Ax migration · feasibility-classifier
  for NaN failures** → "Alternative MFBO architectures" below.
- **CFD regime compressible@1e-9 → incompressible** (kills the stiffness cost + divergence) → touches the
  C12/HIGH-12 lock, so it is an **ADR decision** (`docs/adr/`), not a silent change.
- **Cheap analytic added-mass+drag surrogate as a screening FILTER** (not a ranker — the `gentle>deep`
  non-monotonicity would invert a quasi-steady model) → new V2 item; pairs with MF-BO.
- **Substrate**: shrink the campaign first (surrogate + fewer dims → fits one node/session); migrate to
  Modal / one-box+Postgres / gcsfuse-`ttl-secs=0` **only if** the campaign stays big.

---

## Deferred design-space relaxations (trapezoid redesign, 2026-07-29)

The trapezoid blade redesign (ADR-0005) freed several arbitrary pre-decisions so the BO can
choose rather than have shape baked in. **Relaxed in V1** (fold-safe, feasible-by-construction):
bipolar meridian `RIB_BOW_RANGE_M` `[0, +30] mm` → **`[−20, +20] mm`** (up-humps *and*
down-cups/scoops; ±20 not ±30 is the measured max that folds by construction at 12 blades under the
90 mm stack cap — see ADR-0005 / `blade.py`). This already doubles the meridian travel (40 mm p2p).

**Panel surface amplitude was NOT relaxed** (measured 2026-07-30, found fold-limited not starved):
the ribbed panel offset ceiling is ~1 mm (flat meridian, ~0.15–0.3 mm with a deep bow) and the
uniform ~2 mm, all set by the shared 12-blade / 90 mm fold budget — the meridian, rib thickness, and
panel offset draw on the SAME budget. The radial panel offset is a surface-of-revolution wave, so it
is *redundant* with the (now large, bipolar) meridian; the tangential (Way-2) component is genuinely
fold-limited (rotational asymmetry collides). The old ~3 % "starvation" is already resolved by the
redesign's thick ribs + bipolar meridian + 3 mm base + the 3–10 mm thickness grid. The only lever for
MORE ribbed-panel offset is to let it bulge past the rib (fold thicker) — a fold/mass/click trade,
deferred below.

**Deferred to V2** — also arbitrary, also fold-safe, but each **adds search dimensions**, so held
back to keep the V1 BO tractable. Revisit if the V1 run stalls or the operator wants a wider search:

- **More meridian knots** (`RIB_BOW_KNOT_COUNT` 5 → 7–9) — finer / higher-frequency waves; the
  current 5 caps the wave to ~2 oscillations. +2–4 dims.
- **Finer panel grid** (`PANEL_GRID_RADIAL_COUNT × PANEL_GRID_TANGENTIAL_COUNT` 4×3 → finer) —
  finer surface cups / texture (Way-2). +N dims per row/col added.
- **Non-linear / higher rib-thickness profile** — currently a 2-point (hub, tip) linear ramp,
  ≤12 mm. Let it vary non-linearly and/or thicker.
- **Rib width as a BO variable** — `RIB_BASE_WIDTH_M` / `RIB_TIP_WIDTH_M` were H12-locked at
  4/6 mm; the redesign already touches that lock, so exposing rib width to the optimizer is a
  natural V2 step.
- **Ribbed panel bulge past the rib** (drop the `panel ≤ rib` containment cap for BOTH the thickness
  grid and the mean-offset grid) — the surface-of-revolution + rib-rail window + panel-aware layer
  spacing mean an over-poke just makes a thicker blade that still folds; `containment_margin` is
  already a conservative proxy, not a hard fold limit. This is the ONLY lever that gives the ribbed
  panel more than its current ~1 mm fold-honest offset ceiling (uniform already bulges freely). Trade:
  a thicker folded bundle + more mass, and it inverts the `panel ≤ rib` click-chamfer clearance
  ordering (open ADR-0005 item) — so it needs a click-clearance re-check, not just a range bump.

Each deferred item is a one-line range/count change + fold re-verification; none is blocked on new
physics. Acceptance for any of them: the full random+extreme decode sample stays ~100 % CAD-fold-
clear and the seeds still hold.

### Analytic (boolean-free) fold gate — DONE in V1 (2026-07-30)

Implemented, not deferred. The in-loop gate is now the boolean-free analytic `fold_penetration_m`
(max fold interpenetration from the smooth surface-of-revolution height field), replacing the CAD
swept-volume boolean which the bipolar relaxation made untenable (it **hangs** on steep-zigzag +
checkerboard-offset solids and runs ~35–45 s on normal bipolar designs at the fine mesh a tight gate
needed). The analytic gate is ~40 ms, hang-proof, faceting-free, validated 100 % fold-clear over 512
Sobol designs, and restores full feasible-by-construction. The CAD boolean `fold_collision_volume_m3`
is kept as an offline deep-verify. See ADR-0005 "Analytic fold gate". *V2 follow-on:* vectorise the
gate (numpy over the swing×grid) if per-call cost ever matters, and fold the boss annulus in
explicitly if a future geometry lets the boss rise with the meridian.

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

**⚠ Audit 2026-08 correction (see `docs/audit_2026-08_toolchain_and_approach.md`) — the instrument must be DIRECTIONAL.** The optimized `J_fan` is the *signed cycle-mean* (net directed rectification: flat nets ≈0/slightly negative, a cupped blade nets positive *toward* the user). A plain/thermal anemometer reads *unsigned* speed magnitude (~RMS/peak), which the N1 data shows is ~equal across all designs — so it would fail to distinguish a good fan from the flat baseline. Priority therefore INVERTS: path **#1 (kitchen scale + light target)** measures net directed *force* and is the correctly-aligned cheap validator; a **vane** (propeller) anemometer read in **AVG mode** over a fixed fanning window is acceptable (it's directional); a **thermal/hot-wire** anemometer is NOT. The blinded feel test (Spike-0.3 V1 substitute) is itself aligned — a human feels the directed puff. Any V1 quantitative check should print the **flat baseline** and measure the *net* flat-vs-candidate gap.

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

### Geometry-validity filter (found in the Phase-5 K=6 re-run, 2026-07-19)

Phase-4's objective only evaluates the fast **2D mid-radius slice**, which cannot
see 3D self-intersection — so the BO pushed toward extreme high-J_fan designs whose
**full 3D blade self-intersects** (an invalid, un-printable, un-meshable solid).
Observed: of a 7-design Phase-5 top-k, **3 failed 3D meshing with self-intersection
errors** (`Invalid boundary mesh (overlapping facets)` ×2, `PLC Error: segment and
facet intersect` ×1), and they were the **highest-2D-J_fan Pareto picks** — the
optimizer was being rewarded for geometry that can't be built.

**Fix:** build the 3D CadQuery/OCC solid and test validity (`Shape.val().isValid()`
/ OCC `BRepCheck_Analyzer`) inside the Phase-4 objective (or a pre-filter), and
**penalize invalid geometry** — return the same dominated-penalty used for a
diverged CFD run — so the optimizer never proposes un-buildable shapes. **Cost:**
one CAD solid build per eval (adds geometry time, but catches the failure at
proposal time instead of wasting a ~85-min 3D verification on it). Benefits V1.5
(a cleaner, all-buildable Pareto) and is a **hard prerequisite for the ML/generative
route** — a generative model will propose far more invalid geometry than BO does.

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

### Effort-minimizing asymmetric two-face blade (operator insight 2026-07-19)

**Distinct from the weighting entry above:** that one re-weights the *objective*; this
one changes the *geometry*. A biconvex `)(` blade generates ≈ the same airflow on the
up-stroke and the down-stroke — but you only *want* airflow on the productive
(down) stroke. So a future blade could be **asymmetric between its two faces**: the
down-stroke-facing side shaped for high thrust, the up-stroke-facing side shaped
**aerodynamically clean (low drag)** so the return stroke costs less muscular effort.
Over a sustained few-minute continuous wave, that lowers **net work per cycle**.

**Why it fits cleanly here:** the project already carries *productive stroke* and
*return stroke* as two of the four locked kinematic load cases, and the binding
artifact is a **`J_fan / W_cycle` ratio** (airflow-per-effort), so "minimize effort
over a cycle" is already half-present. This item makes the **geometry** exploit that
asymmetry rather than only the objective weighting.

**What it needs (why it's not V1):** (a) an objective term for **net cycle work /
sustained-swing effort** (integrate the unsteady aero + inertial reaction torque over
a full up-down cycle, not just peak productive thrust), and (b) evaluating the blade
on **both stroke directions** per candidate (roughly doubles the unsteady-CFD cost).
Pairs naturally with the V1.5 staggered loop and the asymmetric-`J_fan` weighting above.

**Not foreclosed by the V1 lean codec:** the lean panel is already a *free both-face*
surface (base ± thickness), so it can represent an asymmetric `)(`-vs-clean profile
today — only the effort objective + dual-stroke evaluation are missing. So V1 keeps
this reachable without a parameterization change.

**Trigger:** V1/V1.5 ships and the operator wants to optimize sustained-waving effort,
not just per-stroke airflow.

### Compliance-based passive feathering — multi-material blade (operator insight 2026-07-24)

**The *material* route to effort asymmetry.** The two-face entry above chases lower net cycle
work via *geometry*; this pursues the same goal via *material compliance*: a rigid frame + a
panel that is **stiff on the productive stroke but flexes (feathers) on the return stroke**,
cutting return-stroke drag like a feathering oar. Concretely, **PETG rib frame** (rigid,
strong, carries the fold structure) **+ a tuned-TPU panel** — dual-material FDM.

**Why it is NOT "just make the panel floppy":** every V1 result favored *stiffness* — a rigid
panel transfers the swing into the air; a uniformly floppy one gives way and moves *less* air
(deflection ~0 in the winning designs). Uniform TPU flexes *both* directions, killing the push
too. The win only exists if the compliance is **directional/tuned** (graded stiffness, a hinge
line, anisotropic infill) so it feathers on the return but stays firm on the push.

**Interface bond is the weak point.** PETG (polyester) and TPU (polyurethane elastomer) have
mediocre FDM interlayer adhesion, and the rib↔panel seam takes *shear* every flex — the load
that peels a weak bond. Design a **mechanical interlock** (dovetail / interpenetrating fingers
/ lattice transition zone) so geometry holds them together; don't trust the chemical bond.
Needs dual-material hardware (AMS / dual extruder).

**Why it's heavy — needs FSI (this is what pushes it to V3):** a compliant panel *deforms
under the airflow, which changes the airflow* — aero and structure become **coupled
(fluid–structure interaction)**. V1's whole pipeline runs aero on a *rigid* geometry (the
"No FSI" lock, §2.3), so this needs FSI co-simulation (CFD↔FEA in a loop, or a monolithic FSI
solver) — machinery we don't have. It shares the **panel-compliance → as-loaded aero shape**
coupling channel with the V1.5 staggered loop's static-deflection step, but goes further
(dynamic, per-stroke). Design space also grows (which regions which material, graded-stiffness
maps, interface geometry).

**Classification:** multi-material → **V3** per the Out-of-scope entry below (V1 *and* V2 stay
single-material PETG). But it's the compliance sibling of the V2 effort-asymmetry entries and
worth prototyping the moment an FSI toolchain + dual-material FDM are both in hand.

**Trigger:** V1/V1.5 ships, the operator wants to attack sustained-swing effort via material
compliance, and a multi-material FDM toolchain is available.

### Porosity / vent sweet-spot for wind-per-effort (operator insight 2026-07-19)

V1's solid-surface parameterization forbids through-holes (they leak airflow, so a solid
paddle beats a vented one for *raw* wind). But the binding artifact is **`J_fan / W_cycle`
— wind per effort**, and there may be a **sweet spot** where small pores / slots shave
drag (hence swing effort) *faster* than they shave wind, improving the *ratio* even as
peak wind drops. A solid-surface V1 can never propose this; only a **free-form / topology-
varying** search (the ML free-form route below) can discover porous or louvered blades and
weigh the drag-reduction vs wind-loss tradeoff. Related to the directional-louver idea in
the effort-asymmetry entry above.

**Scope:** free-form aero optimization that allows the blade to open pores/slots, scored
on `J_fan / W_cycle` (not peak `J_fan`), with the physics verifier confirming the folded
+ printable result. Requires the V2 free-form representation + the CFD to resolve
flow-through-slot; not expressible in any V1 parameterization.

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

### Heterogeneous per-position blade shaping (operator insight 2026-07-26)

**Idea:** once we optimize a blade *as a unit of the whole fan* (the 2026-07-26 pivot moved the
objective to whole-fan wind via a 3D periodic-cascade × `blade_count` — see ADR-0004,
`docs/adr/0004-optimization-3d-objective-redo.md`), a further question opens: the end-of-sector blades have free
edges the packed middle blades don't, so a **heterogeneous fan** (different shapes per angular
position) could capture edge/interaction effects a uniform fan leaves on the table.

**Why it's deferred (measure-then-decide, V2.5+):** it breaks the cheap periodicity — different
blades means you can't simulate one and tile it, so you need a **full deployed-fan sim** (~N× the
mesh, ~5–15 h/eval → a BO loop becomes infeasible), plus N× the design space and N distinct
printed parts, for a likely **second-order** gain (end effects touch ~2 of 8–12 blades). **Do
not build speculatively.** The revisit trigger is a full-sector *verification* sim (needed anyway
for final validation): measure how much the end blades' optimal shape actually differs from the
middle. Negligible → dead; large → earns a V2.5+ effort. Relates to the `directional_asymmetry`
and `Asymmetric-stroke physics in J_fan` items above (both about squeezing more directed wind
from the same stroke).

## Out-of-scope (V3+ or research)

Items that are not in the V1 or V2 roadmap, queued for either a V3 effort or a research follow-on.

- **Active electronic flow control** — embedded micro-blowers or piezo actuators in the panel cutouts. Adds power + control complexity; out of V2 scope.
- **Multi-DOF wrist motion** — current model assumes pure +y wrist rotation (flexion). Real waving has yaw + pitch + roll components. V3 could extend the SU2 unsteady cfg to support compound rotations.
- **Multi-material printing** — TPU membranes, dual-extruder panels. V1 explicitly rejects this (single-material PETG except the steel/brass pin); V2 stays single-material. V3 could revisit if multi-material AMS toolchains improve. Most compelling driver: **Compliance-based passive feathering** (Optional section above) — a tuned-TPU panel as the *material* route to effort asymmetry; needs FSI tooling + a mechanical rib↔panel interlock (PETG/TPU bond is weak).
- **Adjoint-based aero shape optimization on the panel envelope** — V1 uses generative parametric design (4-layer hybrid); V3 could couple SU2 continuous adjoint to the Layer 1 envelope spline directly for a finer-grained gradient-based refinement of the top-1 Pareto design.
