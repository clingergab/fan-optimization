# V2 plan — ML-driven wind-per-effort optimization (AO + TO together)

Ordered roadmap for V2, written at the end of V1 (2026-11-24). `V2_backlog.md` keeps the full item
descriptions; this file says **what V2 is, where we start, what must be decided first, and in what order
the work runs**. The plan of record for V1 stays `report-final.md`; design decisions go in `docs/adr/`.

---

## 0. Goal and scope

**V2 finds the blade design — thickness, shapes, curves, surfaces and inner structure — that maximizes
wind per unit effort over sustained waving (about a minute).** It uses ML surrogates and GPUs to search a
much larger design space than V1, and optimizes the aerodynamic shape (AO) and the inner structure (TO)
together instead of one after the other.

- **In:** a correct wind-per-effort objective; blade thickness as a real, widely explored variable; a larger
  shape space (more curve detail, possibly free-form); flex-aware aero; a stiffness/buckling-aware,
  infill-aware TO; coupled AO+TO; ML surrogates + active learning; GPU compute.
- **Out (later):** lights in the blades (**V2.5**, `V2_backlog.md` "V2.5 — Lit blades"); multi-material /
  compliant panels and full FSI (V3); per-position (heterogeneous) blades unless a full-fan check says
  they matter.

## 1. Starting point (what V1 leaves us)

| Asset | State | Use in V2 |
|---|---|---|
| 3D CFD campaign (`print_ready/drive_recovery/campaign_trapezoid/evaluations_colab-*.jsonl`) | **838 runs, 805 designs**, 33-knob codec; 751 with a finite `j_fan`, 87 failed (NaN); 179 Sobol + 610 BO + 49 explore; rigid blades; **scalars only** (`j_fan`, solid `mass_kg`, analytic plate `deflection_m`) — no flow/pressure fields kept (confirm on Drive) | First surrogate + failure classifier; prior / validation set for the V1 corner of the V2 space |
| Stage-4 TO (`stage4_summary.json`, 00/02/03 densities) | **8 designs**, one setup, 40 iterations (not converged); all 8 failed the 1 mm tip screen and fell back to volume fraction 0.6 | Not training data; baseline to beat |
| Printed V1 fans (00, 02; PLA, 15 % grid infill, 20 % thinned) | Feel test + measurements pending | **Ground truth** for the simulations |
| Code | Codec, analytic fold gate, 3D CFD objective, distributed BO, per-design 3D SIMP, print path | Reused; extended per §3 |

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
| **D1** | **Objective: wind ÷ effort per cycle** (`J_fan / W_cycle`) | `J_fan` = signed cycle-mean thrust (ADR-0004). `W_cycle` (new in sim) = work per cycle at the wrist: inertial torque (blade mass × distance² from the wrist, from the CAD mass distribution incl. TO/infill) + aero torque (from the CFD moment about the wrist axis). **Operator decision (2026-11-24):** compute both push-only `∫ max(τω, 0) dt` and push + brake `∫ |τω| dt` for every design; if they rank designs the same the choice is moot, if not decide then with the data. Keep wind and effort as two Pareto objectives as well as the ratio. |
| **D2** | **Design space and representation** | Thickness floor from printing, not 3 mm (set by the §3 W0 thickness-series print; expect ~1.5–2 mm); thickness × inside fill as one question; more meridian knots (5 → 7–9); finer panel grid; developed (arc) length held at 220 mm; rib width free (H12 relaxation). Thinner blades free fold budget → allow a larger meridian range. **Operator decision (2026-11-24): two stages** — (1) an extended codec (hundreds of knobs: finer meridian + finer panel grid, independent top and bottom faces), feasible by construction; then (2) free-form (neural implicit / SDF), seeded from the stage-1 best and required to beat it. **Two-face asymmetry is in V2** (free in both stages). |
| **D3** | **CFD regime** — ADR-0006 incompressible (Proposed) | Expected 2–5× faster and no divergence class; accept or reject at V2 kickoff. |
| **D4** | **Fidelity ladder** | Coarse CFD is usable for training/screening only if coarse-vs-fine R² > 0.75 on a design spread (audit 2026-08). Else single-fidelity. |
| **D5** | **TO formulation** | `min mass s.t. stiffness, buckling λ ≥ λ_min, stress ≤ σ_allow` (not compliance at a fixed 0.6); infill-aware material (walls + sparse core with measured stiffness); clamp + skin-freeze at the boss (~9 mm), not the retired 20 mm band; run to convergence. |
| **D6** | **Compute** | **A100 / G4-class GPUs available (operator, 2026-11-24).** GPU for surrogate training/inference and TO (port SIMP/FEA to a GPU array library); CFD stays CPU (SU2) unless a GPU solver passes the D4 correlation check. Shrink the campaign first; move to a bigger substrate only if it stays large. |

## 3. Workstreams, in order

Each has an acceptance test; later streams start only when the earlier ones they depend on pass.

**W0 — V1 wrap-up (ground truth).** Blinded feel test of 00 vs 02 vs a store-bought fan; **directional**
wind measurement (vane anemometer in AVG mode, or kitchen scale + light card — not a thermal/hot-wire
anemometer, `V2_backlog.md` Spike 0.3 correction); fan weights; tip-deflection test (clamp at the hub,
known weight at the tip) at 15 % and 100 % infill; a thickness-series print (e.g. 80/70/60 %) through the
same test. *Done when* the numbers are recorded in `docs/phase_logs/`.

**W1 — Objective (D1).** Simulated `W_cycle`: inertia about the wrist from the as-printed mass
distribution; aero moment from the CFD force/moment history. Validate the inertial part against the
analytic `i_wrist_assembly` and, if available, a phone-IMU swing. *Done when* `J_fan / W_cycle` is computed
per design and ranks the W0 fans in the measured order (or the disagreement is explained).

**W2 — Design space (D2).** New codec/representation with the thickness changes, fold-safe by
construction (analytic fold gate extended), 3D validity check in the loop (`V2_backlog.md`
"Geometry-validity filter" — a hard prerequisite for generative search). *Done when* a random + extreme
sample is ~100 % fold-clear and buildable.

**W3 — CFD pipeline.** Save rich outputs every run (surface pressure, force + moment histories, the mesh /
geometry) so each run trains field-level models; D3 regime; D4 fidelity check; a flex step (deform under
the real aero load, re-run — `V2_backlog.md` "flex-blind BO") run on a stiffness spread to measure how much
flex reshuffles the ranking. *Done when* a run produces the rich outputs and D3/D4 are decided.

**W4 — ML prototype on V1 data.** `src/fanopt/ml/`: a `j_fan` surrogate on the 751 valid runs (held-out
test split, compared with the GP), a failure classifier on the 87 NaNs, and an active-learning loop. It
proves the tooling before V2 data exists. *Done when* held-out error is within the CFD noise floor in the
V1 region (`V2_backlog.md` ML section, step 1).

**W5 — V2 data generation.** Space-filling sample of the W2 space, then surrogate-guided active learning
(sample where the model is uncertain or promising), every run with W3 rich outputs and the W1 objective.
*Done when* the surrogate holds its accuracy on fresh held-out runs across the V2 space.

**W6 — TO rebuild (D5).** Stiffness + buckling + stress constraints, infill-aware, boss clamp, converged,
on GPU; calibrated with the W0 tip-deflection data. Train a TO surrogate once enough solves exist.
*Done when* it re-runs the V1 designs to convergence and beats V1 on mass at equal measured stiffness.

**W7 — Coupled AO + TO.** Start staggered (`V2_backlog.md` V1.5 loop: AO → loads → TO → mass/inertia +
stiffness → AO, 2–4 passes), with surrogates in the inner loops; move toward joint optimization over
shape + thickness + structure once both surrogates are trusted. *Done when* the loop converges and the
result Pareto-dominates V1 on (wind, effort) with the structural gate passed.

**W8 — Verify and print.** Physics (fine CFD + FEA) re-checks every candidate — never ship an unverified
ML output; print the top designs with the V1 print path; W0 tests again against the V1 fans.

## 4. What still binds

Product-defining constraints, not simplifications to remove: 12 blades (ADR-0005); 220 mm blade
(developed length, per D2); `MAX_TOTAL_MASS_KG` 0.300; `MAX_R_COM_WRIST_M` 0.160; the 90 mm folded stack;
the ±40°, 2 Hz stroke; a pin pivot through the boss (no hole through the rib); no through-holes in the
blade surface (they leak air — the porosity sweet-spot entry is the one deliberate exception to test in a
free-form search); single-material FDM printability; the open/closed hold detent.

## 5. Operator questions

Answered 2026-11-24 (recorded in §2):

1. ~~Does negative (braking) work count?~~ → compute push-only and push + brake; decide only if they disagree (D1).
2. ~~Free-form or extended codec first?~~ → extended codec first, then free-form seeded from its best (D2).
3. ~~Compute?~~ → A100 / G4-class GPUs; exact budget and substrate settled at W5 (D6).
4. ~~Asymmetry in V2?~~ → yes, two-face asymmetry is optimized in V2 (D2; `V2_backlog.md` "Effort-minimizing
   asymmetric two-face blade" moves into V2 scope).

Still open: whether the stroke itself becomes asymmetric in the objective (`V2_backlog.md` "Asymmetric-stroke
physics in J_fan") — needs the W0/W1 measurements first.

## 6. Backlog entries by workstream

| Workstream | `V2_backlog.md` entries |
|---|---|
| W0 | Spike 0.3 (directional correction), Spike 0.5, V1 fans at 15 % infill |
| W1 | Effort-minimizing asymmetric two-face blade (effort term), Asymmetric-stroke physics |
| W2 | Deferred design-space relaxations, Blade length = developed, Geometry-validity filter, Porosity sweet-spot |
| W3 | Audit 2026-08, Rigid-aero ranking validity, Alternative MFBO architectures, ADR-0006 |
| W4–W5 | ML-driven TO + AO (steps 1, 3) |
| W6 | TO objective stiffness-/buckling-blind, TO hub clamp, V1 fans at 15 % infill, ML step 2 |
| W7 | V1.5 staggered AO↔TO, ML step 4, Mid-Phase-4 rib re-tune |
