# V2 plan — ML-driven wind-per-effort optimization (AO + TO together)

Ordered roadmap for V2, written at the end of V1 (2026-11-24) and revised after a readiness review
(2026-11-25). `V2_backlog.md` keeps the full item descriptions; this file says **what V2 is, where we
start, what must be decided first, and in what order the work runs**. The plan of record for V1 stays
`report-final.md`; design decisions go in `docs/adr/`.

**The 2026-11-25 revision in one line:** check the simulation against the printed fans and fix the
known gaps in the CFD run (wrist pivot, whole fan, physical units) *before* growing the design space;
free-form shapes, GPU TO and joint optimization wait until those checks pass.

---

## 0. Goal and scope

**V2 finds the blade design — thickness, shapes, curves, surfaces and inner structure — that maximizes
wind per unit effort over sustained waving (about a minute).** It uses ML surrogates and GPUs to search a
much larger design space than V1, and optimizes the aerodynamic shape (AO) and the inner structure (TO)
together instead of one after the other.

- **In:** a correct wind-per-effort objective; blade thickness as a real, widely explored variable; a larger
  shape space (more curve detail, possibly free-form); flex-aware aero; a stiffness/buckling-aware,
  infill-aware structure step; coupled AO+TO; ML surrogates + active learning; GPU compute.
- **Out (later):** lights in the blades (**V2.5**, `V2_backlog.md` "V2.5 — Lit blades"); multi-material /
  compliant panels and full FSI (V3); per-position (heterogeneous) blades unless a full-fan check says
  they matter.

## 1. Starting point (what V1 leaves us)

| Asset | State | Use in V2 |
|---|---|---|
| 3D CFD campaign (`print_ready/drive_recovery/campaign_trapezoid/evaluations_colab-*.jsonl`) | **838 runs, 805 designs**, 33-knob codec; 751 with a finite `j_fan`, 87 failed (NaN); 179 Sobol + 610 BO + 49 explore. BO's best 9.65e12 vs Sobol's 6.33e12, failures 32 % → 5 %; best found at run 377, nothing better after. Rigid single blade; **scalars only** (`j_fan`, solid `mass_kg`, analytic plate `deflection_m`) — no flow/pressure fields kept (confirm on Drive). Solid masses 349–598 g (median 413 g): every design is over the 300 g cap as a solid; none is the blade as printed | First surrogate + failure classifier; prior / validation set for the V1 corner of the V2 space; effort re-ranking (W1) |
| Stage-4 TO (`stage4_summary.json`, 00/02/03 densities) | **8 designs**, one setup, 40 iterations (not converged); all 8 failed the 1 mm tip screen and fell back to volume fraction 0.6 | Not training data; baseline to beat |
| Printed V1 fans (00, 02; PLA, 15 % grid infill, 20 % thinned; 00 ≈ 215 g sliced) | Feel test + measurements pending | **Ground truth** for the simulations |
| Flat control (`print_guide.md` §1a, built 2026-11-25) | Blade 00 unbent: same reach, true wall, TO hollow, detent and slicer settings; 17.69 g vs 17.88 g per blade sliced | The control that isolates the shape (W0) |
| Code | Codec, analytic fold gate, 3D CFD objective, distributed BO, per-design 3D SIMP, print path | Reused; extended per §3 |

**Gaps V1 leaves open** (each closed by a workstream below):

1. **No printed control until now.** V1's success test was a blinded comparison against a printed flat
   fan; 00 and 02 are 5 % apart in coarse `J_fan`, inside what the coarse tier resolves, so comparing them
   tests nothing. The flat control now exists (W0).
2. **One isolated blade × 12.** The real fan is an overlapping sheet; the whole-fan (periodic-cascade)
   sim was deferred in ADR-0004 (D8, W3).
3. **The blade pitches about the pin, not the wrist** (ADR-0004 N4, "fix before any V2 absolute
   quantification"). Effort is torque about the wrist, so this is now mandatory (D7, W2).
4. **`J_fan` has no physical units.** The unsteady runs use Mach 1e-9 with a "freestream pressure = 1"
   reference; values near 1e12 are scaled coefficients, not newtons (D9, W2).
5. **The printed blades were never simulated** (all searched designs are solid and over the cap; the
   prints are TO-hollowed, thinned and 15 % infill) (W1, W2).
6. **No fidelity number to test models against.** The CFD is deterministic (the one design run twice gave
   an identical value), so repeats measure nothing; the uncertainty that matters is coarse-vs-fine, and the
   ~15 % figure comes from four designs only (D4, W3).

**Thickness in V1:** 14 of the 33 knobs were thickness (rib hub/tip + a 4 × 3 panel grid), but boxed in:
panel 3–10 mm, rib 2–12 mm, panel ≤ rib, the 90 mm fold budget. The objectives could only trade it as
solid mass against an analytic plate-flex proxy at a placeholder 10 Pa; thickness barely moves the rigid
CFD wind. The printed blades (~2.5 mm after the TO and the operator's 20 % thinning) sit **below the
searched 3 mm floor** — the final thickness was a hand decision, not an optimum.

## 2. Decisions before generating V2 data

V1's costliest errors were objective errors (ADR-0004: a wave-blind 2D slice, then CFx instead of CFz).
So V2 settles the objective and the space **before** spending GPU time on data. Each decision is an ADR.

| # | Decision | Notes |
|---|---|---|
| **D1** | **Objective: wind ÷ effort per cycle** (`J_fan / W_cycle`) | `J_fan` = signed cycle-mean thrust (ADR-0004), in newtons (D9). `W_cycle` (new in sim) = work per cycle at the wrist (D7): inertial torque (blade mass × distance² from the wrist, from the as-printed mass distribution, W1) + aero torque (from the CFD moment about the wrist axis, W2). **Operator decision (2026-11-24):** compute both push-only `∫ max(τω, 0) dt` and push + brake `∫ |τω| dt`; decide only if they rank designs differently. **Revised 2026-11-25:** for pure inertia push + brake is exactly 2 × push-only (the energy put in speeding the fan up comes back out slowing it), and a constant factor cannot change a ranking; the two only differ once aero torque exists (the air helps brake). So the comparison runs in W2, not W1. Keep wind and effort as two Pareto objectives as well as the ratio. |
| **D2** | **Design space and representation** | **Operator decision (2026-11-24): two stages** — (1) an extended codec (finer meridian + finer panel grid, independent top and bottom faces), feasible by construction; then (2) free-form (neural implicit / SDF), seeded from the stage-1 best and required to beat it. **Two-face asymmetry is in V2.** **Revised 2026-11-25: the size of stage 1 is set in W4, after W0–W3 pass**, not up front — at hours per CFD run, hundreds of CFD dimensions are unaffordable until CFD is cheaper. Thickness floor from printing, not 3 mm (set by the W0 thickness series; expect ~1.5–2 mm); thickness × inside fill as one question; developed (arc) length held at 220 mm; rib width free (H12 relaxation). The 12 panel-offset knobs are **untested, not dead**: V1's containment held them to ~3 % of their range (00's offsets are all < 0.1 mm). On a ribbed blade they are pinned at the rib edges, so they make a cross-chord dish, which the meridian cannot; but a dish across the chord is not rotationally symmetric, so the fold stack still limits it (thinner blades free some budget). |
| **D3** | **CFD regime** — ADR-0006 incompressible (Proposed) | Expected 2–5× faster and no divergence class. For speed and stability; it does **not** block D1 (newtons need only the reference scaling, D9). Accept or reject in W2. |
| **D4** | **Fidelity ladder** | Coarse CFD is usable for training/screening only if coarse-vs-fine R² > 0.75 on a design spread (audit 2026-08). The coarse-vs-fine difference is also the error bar every surrogate is tested against (W5); the ~15 % figure comes from four designs, so W3 measures it on a wider spread before it is used as a threshold. Else single-fidelity. |
| **D5** | **Structure formulation** | `min mass s.t. stiffness, buckling λ ≥ λ_min, stress ≤ σ_allow` (not compliance at a fixed 0.6); infill-aware material (walls + sparse core with measured stiffness); clamp + skin-freeze at the boss (~9 mm), not the retired 20 mm band; run to convergence. **Revised 2026-11-25: first as a thickness-map optimization on a shell model** — a ~2.5 mm blade is almost all wall (15 % infill saved only ~10 %), so this captures most of the gain at a fraction of the cost; GPU 3D SIMP only if the shell result falls short. |
| **D6** | **Compute** | **A100 / G4-class GPUs available (operator, 2026-11-24).** GPU for surrogate training/inference and structure (port to a GPU array library); CFD stays CPU (SU2) unless a GPU solver passes the D4 check. Shrink the campaign first; move to a bigger substrate only if it stays large. |
| **D7** | **Pitch about the wrist** (new 2026-11-25) | The CFD motion pivots about the wrist, with the operating point (V_tip, Re) from the real lever (ADR-0004 N4). Required for the effort objective. |
| **D8** | **Whole fan, not one isolated blade** (new 2026-11-25) | A periodic-cascade (one blade in its fan context) or a sector sim, checked against a full-fan run on a handful of designs (ADR-0004's deferred whole-fan fix). |
| **D9** | **Physical units** (new 2026-11-25) | `J_fan` and the aero moment in newtons and newton-metres via the reference scaling, so aero torque and inertial torque add in one unit. |

## 3. Workstreams, in order

Each has an acceptance test; later streams start only when the earlier ones they depend on pass.
W0–W3 are the checks; W4 onward is the V2 search proper.

**W0 — V1 ground truth.** Blinded feel test of the **flat control** vs 00 (and 02), plus a store-bought fan
as a real-world benchmark (not a control: it differs in size, mass, material and stiffness at once);
**directional** wind on each (kitchen scale + light card, or a vane anemometer in AVG mode — not a
thermal/hot-wire anemometer, `V2_backlog.md` Spike 0.3 correction); fan weights; tip-deflection test
(clamp at the hub, known weight at the tip) at 15 % and 100 % infill; a thickness-series print (e.g.
80/70/60 %) through the same tests, for stiffness and to check real weights against the slicer.
Optional: one design the sim predicts blows the wrong way. *Done when* the numbers are recorded in
`docs/phase_logs/` and the flat-vs-00 gap is compared with the simulated one.

**W1 — Inertial effort for the V1 designs (no CFD).** A mass model under one common print recipe (the
operator's: ~2.5 mm, 2 walls, 15 % infill) — walls + sparse core — fitted to slicer weights across
designs and thicknesses (the slicer weighs a blade in seconds without printing) and checked against the
W0 printed weights; inertia about the wrist from the as-printed mass distribution, validated against the
analytic `i_wrist_assembly` and, if available, a phone-IMU swing. *Done when* `J_fan / W_inertial` is
computed for all 751 designs and its ranking is compared with `J_fan` alone (§3a hypothesis H1).
**Done 2026-11-25** (`docs/phase_logs/v2_w1_inertial_effort.md`): slicer-calibrated mass model (0.13 g
RMSE); fans 224–266 g on the recipe, inertial effort ±4 % across designs; the ranking barely moves
(Spearman 0.999, best design unchanged). The printed-fan weighing (W0) is still the final check.

**W2 — Rebuild the CFD run once.** Incompressible (D3), pitching about the wrist (D7), physical units
(D9), and saving rich outputs every run: surface pressure, **force + moment histories** (the moment gives
the aero part of effort — the only effort term that tells shapes apart under a common recipe), the mesh /
geometry. Re-run 10–20 V1 designs and the flat control. *Done when* the V1 ranking holds (or the change is
explained), the full `J_fan / W_cycle` is computed for those designs, and push-only vs push + brake (D1)
is compared.

**W3 — Whole-fan, flex and fidelity checks.** On a handful of designs: the whole-fan sim (D8) against the
isolated blade; a flex step (deform under the real aero load, re-run — `V2_backlog.md` "flex-blind BO")
across a stiffness spread, to measure how much flex reshuffles the ranking; coarse-vs-fine on a wider
spread (D4). Flex is the link between shape (wind) and thickness (stiffness), so it decides whether the two
can be optimized separately (H1). *Done when* each gap is measured and D3/D4/D8 are decided.

**W4 — Size the design space (D2).** Only after W0–W3: the stage-1 codec sized from what they show, with
the thickness changes, fold-safe by construction (analytic fold gate extended), 3D validity check in the
loop (`V2_backlog.md` "Geometry-validity filter" — a hard prerequisite for generative search). *Done when*
a random + extreme sample is ~100 % fold-clear and buildable.

**W5 — ML tooling on V1 data (non-gating).** `src/fanopt/ml/`: a `j_fan` surrogate on the 751 valid runs
(held-out test split, compared with the GP), a failure classifier on the 87 NaNs, and an active-learning
loop. A learning exercise: on 751 scalars in 33 dimensions it is not expected to beat the GP, so it gates
nothing and can run alongside W0–W4. *Done when* held-out error is reported against the coarse-vs-fine
difference (D4), not a repeat-run noise floor.

**W6 — V2 data generation.** Space-filling sample of the W4 space, then surrogate-guided active learning
(sample where the model is uncertain or promising), every run with W2 rich outputs and the full
objective. *Done when* the surrogate's held-out error stays within the coarse-vs-fine difference on fresh
runs across the V2 space.

**W7 — Structure rebuild (D5).** Thickness map on a shell model with stiffness + buckling + stress
constraints, infill-aware, boss clamp, converged; calibrated with the W0 tip-deflection data; GPU 3D SIMP
only if the shell result falls short. *Done when* it beats V1 on mass at equal measured stiffness.

**W8 — Coupled AO + structure.** Staggered first (`V2_backlog.md` V1.5 loop: AO → loads → structure →
mass/inertia + stiffness → AO, 2–4 passes), with surrogates in the inner loops; joint optimization over
shape + thickness + structure, and free-form (D2 stage 2), only once both surrogates are trusted. *Done
when* the loop converges and the result Pareto-dominates V1 on (wind, effort) with the structural gate
passed.

**W9 — Verify and print.** Physics (fine CFD + FEA) re-checks every candidate — never ship an unverified
ML output; print the top designs with the V1 print path; W0 tests again against the V1 fans and the flat
control.

### 3a. Working hypothesis H1 — shape and thickness split

Under a common print recipe, shape barely moves mass (curved 00 vs the flat control: 17.88 vs 17.69 g per
blade, ~1 %), and every design has the same planform, so the **inertial part of effort is nearly the same
for every shape**. If so, V2 splits into two nearly independent problems:

- **Shape** sets the wind **and** the aero part of effort — both from the same CFD run (W2's moment
  history).
- **Thickness and inside fill** set the inertial part of effort and the stiffness — a mass model plus a
  structural check, no CFD.

The link between them is **flex** (a thinner blade bends more, and bending changes the wind), measured in
W3. W1 tested the first half and it held: `J_fan / W_inertial` ranks the V1 designs almost the same
as `J_fan` (Spearman 0.999). W2 and W3 test the rest. If H1 holds, W6 can spend its CFD budget on shape alone and W7 can
size thickness without CFD in the loop.

## 4. What still binds

Product-defining constraints, not simplifications to remove: 12 blades (ADR-0005); 220 mm blade
(developed length, per D2); `MAX_TOTAL_MASS_KG` 0.300; `MAX_R_COM_WRIST_M` 0.160; the 90 mm folded stack;
the ±40°, 2 Hz stroke; a pin pivot through the boss (no hole through the rib); no through-holes in the
blade surface (they leak air — the porosity sweet-spot entry is the one deliberate exception to test in a
free-form search); single-material FDM printability; the open/closed hold detent.

## 5. Operator questions

Answered 2026-11-24 (recorded in §2):

1. ~~Does negative (braking) work count?~~ → compute push-only and push + brake; decide only if they
   disagree (D1). Revised 2026-11-25: the comparison is only informative once aero torque exists (W2).
2. ~~Free-form or extended codec first?~~ → extended codec first, then free-form seeded from its best (D2);
   its size is set in W4.
3. ~~Compute?~~ → A100 / G4-class GPUs; exact budget and substrate settled at W6 (D6).
4. ~~Asymmetry in V2?~~ → yes, two-face asymmetry is optimized in V2 (D2; `V2_backlog.md` "Effort-minimizing
   asymmetric two-face blade" moves into V2 scope).

Answered 2026-11-25: the review's order (ground truth → inertial effort → CFD rebuild → whole-fan/flex
check → size the space) replaces the original W0–W8 order; the flat control is 12 flat blades at the same
reach and true wall thickness (operator choice over matching the arc length).

Still open: whether the stroke itself becomes asymmetric in the objective (`V2_backlog.md` "Asymmetric-stroke
physics in J_fan") — needs the W0/W2 measurements first.

## 6. Backlog entries by workstream

| Workstream | `V2_backlog.md` entries |
|---|---|
| W0 | Spike 0.3 (directional correction), Spike 0.5, V1 fans at 15 % infill |
| W1 | Effort-minimizing asymmetric two-face blade (effort term), V1 fans at 15 % infill |
| W2 | ADR-0006, Asymmetric-stroke physics |
| W3 | Audit 2026-08, Rigid-aero ranking validity (flex-blind BO), Alternative MFBO architectures |
| W4 | Deferred design-space relaxations, Blade length = developed, Geometry-validity filter, Porosity sweet-spot |
| W5–W6 | ML-driven TO + AO (steps 1, 3) |
| W7 | TO objective stiffness-/buckling-blind, TO hub clamp, V1 fans at 15 % infill, ML step 2 |
| W8 | V1.5 staggered AO↔TO, ML step 4, Mid-Phase-4 rib re-tune |
