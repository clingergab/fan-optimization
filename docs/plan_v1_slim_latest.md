# Plan V1-Slim — Zoned TO + ASO Fan Optimization (proposed revision)

**Status:** DRAFT for review. Does not modify `docs/report-final.md` (locked plan).
This is a proposed slimmed successor derived from the 2026-07-02 design conversation.
If accepted, we decide which locks in `report-final.md` it supersedes (with your
explicit authorization per CLAUDE.md §7.2) before any code changes.

**Date:** 2026-07-02
**Supersedes (proposed):** the R11 multi-fidelity-BO + generative-field-library
approach for V1 scope. Keeps the V1/V2 split from `phase_0_signoff.md`.

---

## 0. Why this revision exists

The R11 plan is engineering-rigorous but over-built for the V1 deliverable (a
printed fan judged by a blinded A/B feel test, reporting sim-vs-sim *relative*
gain). The optimization apparatus (37–46-D multi-fidelity BO over TPMS/Perlin
fields, 3-tier CFD, PyFR cross-solver) targets precision the deliverable cannot
perceive, on physics (separated, shedding, unsteady bluff-body flow) whose
absolute CFD accuracy is explicitly deferred. Almost none of Phases 2–6 is built
yet, so slimming now costs essentially no thrown-away work.

**Goal (user, 2026-07-02):** a *genuinely* topology- and aero-optimized fan —
real structural TO for the skeleton, real aero shape optimization for maximum
airflow, exposing the airflow-vs-weight trade-off — built with the least effort
that still delivers real optimization. Emergent shape (corrugation/faceting) is
wanted; emergent *porosity* is not (it leaks the air we're trying to push).

---

## 1. Locked design decisions (this revision)

| # | Decision | Rationale |
|---|----------|-----------|
| S1 | **Cut Layer-2 porosity fields** (TPMS gyroid, Perlin/Simplex noise). | Perforation leaks net push; counterproductive for a max-airflow objective; highest CAD-failure + mesh cost. |
| S2 | **Single working CFD fidelity for the search** + unsteady CFD only to *verify* the top 1–3. Drop the 3-tier stack + PyFR cross-solver to V2. | The search only needs correct *ranking*; a bluff-body regime ranks fine at one fidelity. PyFR verifies a metric we use only relatively — V2 material. |
| S3 | **Zoned optimization.** Structural TO acts on the **ribs only**; aero shape opt acts on the **panel only** (incl. panel-thickness lightweighting + emergent corrugation). | Zoning is the *mechanism* that guarantees TO never deletes wind-generating surface — it's spatially forbidden from the panel. More reliable than hoping a co-optimizer protects it. |
| S4 | **Aero backbone = Bayesian optimization, "Path A+"** — panel surface as a moderate free-form basis (coarse control-point grid / modal surface, ~25–40 vars) with an explicit corrugation family. | Robust on separated/shedding flow (black-box; no adjoint fragility); evaluates the true unsteady objective; native multi-objective Pareto + categoricals; rich enough to express emergent facets/zigzags. ~25–40 vars ≈ BO's practical ceiling. |
| S5 | **Adjoint (SU2 FFD) = optional conditional polish** on the winning 1–3 designs, not the backbone. Gated on: Phase-3 proxy fidelity high AND steady solver converges on those shapes. | Adjoint's shape-freedom edge is real but *fragile* on separated flow (steady base may not converge; separated-flow gradients are noisy) and proxy-bound. Bonus finisher, droppable with little loss. |
| S6 | **Multi-objective Pareto:** maximize airflow (J_fan); minimize inertia (I_wrist); minimize peak stress. Folded form factor retained as a soft check, not a 4th search axis unless cheap. | The airflow-vs-weight tension the user wants exposed. qNEHVI gives the whole front in one campaign. |
| S7 | **Cheap-loop-early, relay-expensive, measured-refinement** handshake between aero and structure (see §3). | Loops pay off where cheap+coupled; relays win where expensive+weakly-coupled. The rib↔aero coupling is weak/asymmetric, so the expensive stages relay once and refinement count is data-driven. |

**Kept from R11 unchanged:** V-unit blade architecture + all §0 architectural
locks (panel-pivot, C11 sign, click geometry, mass/CoM caps, PETG); the CadQuery
generator + schema; the JSONL ledger; the rib SIMP TO formulation; the two-eval
steady drag-asymmetry proxy; the Phase-6 blinded A/B feel test; the V1/V2 scope
split.

---

## 2. The optimization backbone (how the three methods divide the work)

Each method owns the regime it is strongest in — they are complements, not rivals:

- **Structural TO (SIMP, rib zone):** free-form *structural* material layout,
  driven by structural sensitivities. Cheap (minutes/solve), cached per
  structural context. Owns the emergent *skeleton*.
- **Bayesian optimization (panel zone, ~25–40 vars):** global search +
  categorical architecture choices (blade count, print orientation) +
  multi-objective Pareto, evaluated on the *true* (unsteady) airflow objective.
  Owns the emergent *aero shape* up to BO's dimensional ceiling. **Main engine.**
- **Adjoint shape opt (panel zone, optional):** high-dimensional *local* shape
  polish, gradient cost independent of dimension. Owns the *fine* shape detail
  BO's parameterization can't reach — but only if the steady proxy is trustworthy
  on these shapes.

Dimensionality is the clean dividing line: BO for the low-D global/architecture
decisions; adjoint (if used) for the high-D local detail.

---

## 3. Staged pipeline (the "relay")

```
Stage 0  Cheap coarse aero pass ── low-eval, low-fidelity, on a rough panel
         → rough pressure map + sensitivity to which knobs matter
         (CHEAP LOOP: informs Stage 1 loads AND the Stage 2 parameterization)
   │
Stage 1  Rib structural TO (SIMP) under Stage-0 loads + inertial + click cases
         → lightweight emergent rib. Cached by (architecture, orientation).
   │
Stage 2  BO Path A+ — panel shape search (rib fixed)
         → Pareto front of candidate panels (airflow vs inertia vs stress)
         [MAIN COMPUTE. Runs once, uninterrupted, to BO's exploitation regime.]
   │
Stage 3  (optional) Adjoint FFD polish on top 1–3 Pareto designs
         → fine local shape gains. Gated on Phase-3 proxy fidelity + convergence.
   │
Stage 4  Re-TO the winner(s) under their actual refined loads
         → final structure. MEASURE Pareto shift vs Stage 2.
         • shift small  → stop (coupling empirically weak; expected)
         • shift large  → one warm-started refinement loop, then re-measure
   │
Stage 5  Unsteady CFD verification on the final top 1–3
   │
Stage 6  Print top 1–3 (structurally diverse) + baseline → blinded A/B feel test
```

**Cost discipline:** the expensive breadth (hundreds of evals) happens once, in
Stage 2, at a moderate dimension. Stages 1, 3, 4 touch only a handful of
survivors. No optimizer runs inside another's loop. The only *loops* are the
cheap Stage-0 exploration and the evidence-gated Stage-4 refinement.

---

## 4. The proxy-fidelity gate (decides adjoint in or out)

Phase 3's steady↔unsteady correlation check (retained from R11) becomes the
switch for Stage 3:

- **High correlation + steady solver converges on winner shapes** → adjoint
  polish is trustworthy → run Stage 3.
- **Low correlation, or steady base flow won't converge (shedding)** → skip
  Stage 3; BO Path A+ on the true unsteady objective already carries the aero
  optimization. Lose little.

We do not commit the backbone to a proxy before measuring whether the proxy is
faithful.

> **Note added 2026-07-03 (Phase 3 first run — metric choice).** The correlation
> discriminator is the unsteady **RMS loading amplitude**, *not* the cycle-mean
> momentum flux. Empirically (6-design local SU2 sweep): steady `CD` vs unsteady
> **RMS** → R²=0.96, r=+0.98, τ=+0.73 (**PASS**); steady `CD` vs unsteady
> **mean** → R²=0.14, r=−0.37 (fail, noise). Reason: the mean net momentum flux
> (= `J_fan`, the fan's true objective) is ~0 for the *symmetric* baseline
> panels — up-stroke and down-stroke forces cancel — so it neither converges
> (206% cycle-to-cycle spread) nor discriminates designs; the RMS amplitude is
> nonzero, converges within ~1 cycle (<1.4% spread), and is what steady `CD`
> actually predicts. **`J_fan` remains the final ASO objective** (mean net
> airflow); the optimizer's job is to find asymmetric shapes that rectify it
> nonzero. This note documents the screening-metric choice only; it changes no
> locked decision. Implemented in `fanopt.cfd.phase3.extract_unsteady_rms`.

> **Note added 2026-07-03 (Phase 4 objective spine — objective realizations).**
> The Stage-2 objective evaluator (`fanopt.bo.objective.evaluate_design`, one
> design vector → three objectives) realizes the S6 Pareto axes as: **(1)**
> airflow = `J_fan`, the unsteady 2D-slice cycle-**mean** net momentum flux (the
> ASO objective — the RMS above is only Phase-3 screening); **(2)** inertia =
> total `I_wrist` of the deployed fan (`fanopt.bo.inertia`, CadQuery mass
> properties); **(3)** the structural axis = **panel-stiffness tip deflection**
> under a fixed nominal pressure (`fanopt.bo.structural`), *not* rib-`u_tip`
> under the design's CFD pressure. Reason: under compliance-min the rib topology
> is pressure-scale-invariant and `u_tip ∝ pressure`, so the CFD-pressure variant
> collapses to ≈linear in the aero load — nearly collinear with axis (1), giving a
> degenerate front. The panel-stiffness deflection instead depends on the design's
> Path A+ thickness field (thicker/corrugated → stiffer), an axis independent of
> airflow. It carries thickness-grid stiffening exactly and corrugation stiffening
> partially (via the `t³` bending convexity); the full corrugated-shell geometric
> stiffening is a documented V1 approximation (shell model = later upgrade). This
> is a V1 realization of the S6 "minimize peak stress" intent as a stiffness
> proxy; it changes no locked decision.

> **Note added 2026-07-03 (Phase 4 BO machinery + production dt).** The Stage-2
> optimizer landed: multi-objective **qLogNEHVI + TuRBO** trust-region BO
> (`fanopt.bo.backbone`), a Sobol-DoE → iterate → JSONL-ledger → checkpoint/resume
> campaign loop with the Spike-0.7c **diverse-design stall fallback**
> (`fanopt.bo.orchestration`), and `scripts/run_phase4_bo.py`. Categoricals are
> searched directly by the codec relaxation (no bandit); **SAASBO** is the
> high-D (>50-var) fallback. `configs/architecture_enumeration.yaml` **supersedes**
> the R11 §9.4.1 enumeration (3,240 combos / 20 Layer-2 profiles): V1-slim cut
> Layer-2 porosity and the bandit, so the space is just the discrete axes
> (blade_count × print_orientation) with the +10% growth gate retained. **Production
> unsteady resolution = 5 cycles, dt = T/200** (`PRODUCTION_EVAL_CFG`): the
> Spike-0.6d.2 (ω·dt)² added-mass bias is geometry-independent at fixed ω and
> cancels in the relative ranking V1 reports, so T/200 is chosen; T/400 is reserved
> for Phase-5 PyFR cross-solver work. Resolves the checklist's open dt decision.

> **Note added 2026-07-03 (Phase 4 search-space finalization + parallel/analysis).**
> The BO codec (`fanopt.bo.codec`) is finalized at **35 vars**: 18 thickness-grid
> + 4 corrugation + 3 camber + 2 twist + 6 Fourier LE/TE + edge_profile +
> blade_count. **Camber is now folded into the 2D slice** (`panel_slice`,
> `envelope.camber_height_at`) — the airfoil chord maps to the tangential span, so
> camber bows the slice top face faithfully to the CadQuery blade; it is the
> smooth directed-thrust lever alongside corrugation asymmetry. **Deferred by
> decision:** Layer-2 **louvers → V2** (their worth is a 2D-slice drag-asymmetry
> model that needs reference validation; corrugation + camber already break
> symmetry), and **print_orientation fixed at 'flat'** (V1 plano-convex baseline),
> which leaves `twist` geometrically inert (reserved for a future 'edge'
> orientation). Also landed: campaign **results/analysis** (`fanopt.bo.results` +
> `scripts/analyze_phase4.py` — Pareto + top-k structurally-diverse print picks)
> and **parallel CFD evaluation** — a **process** pool over the DoE/batch
> (`--workers` / `N_WORKERS`); threads are impossible because gmsh installs a
> main-thread-only signal handler + keeps a global model, so the objective is a
> picklable `CfdObjective` shipped to worker processes. Verified with real SU2
> (2 evals / 2 procs at 186% CPU). Set `BATCH_SIZE = N_WORKERS` to parallelize the
> BO loop (q-batch qNEHVI), not just the DoE. Changes no locked decision.

---

## 5. Re-scoped phases (delta vs `report-final.md` §8)

| Phase | R11 | V1-Slim | Change |
|-------|-----|---------|--------|
| 0 | Done (dual gate passed) | Unchanged | — |
| 1 | 4-layer generator (mostly done) | **✅ DONE (2026-07-03)** — Path A+ `ThicknessGridField` (3×6 grid + corrugation) integrated through `Layer1Params` → generator; Layer 2 pruned to the 3 non-porous families (noise/TPMS cut). 560 geometry/scripts tests green. | Emergent facets/corrugation available, flat baseline reachable (not forced). |
| 2a | Baseline 2D CFD for loads | **= Stage 0 cheap aero pass** | Reframed as the cheap early loop; also seeds parameterization. |
| 2 | Rib SIMP TO | **= Stage 1** | Unchanged formulation. |
| 2b | 37–46-D generative BO seed | **Folded into Stage 2** | Smaller design space (~25–40), non-porous. |
| 3 | 2D correlation gate | **= §4 proxy-fidelity gate** | Same run, now also gates adjoint. |
| 4 | 3-tier MF-BO + bandit + PyFR-adjacent | **= Stage 2 (single-fidelity BO) + optional Stage 3 adjoint** | Drop tiers 0/1 split + bandit's multi-fidelity machinery; single fidelity + verify. |
| 5 | PyFR p=3 cross-solver + FEA | **= Stage 4 re-TO + Stage 5 unsteady verify** | PyFR → V2. Keep combined-blade FEA sanity on winner. |
| 6 | Print + feel test | **= Stage 6** | Unchanged. |

Estimated compute: roughly **⅓ of R11's** Phase-4 budget (single fidelity, no
PyFR, ~25–40 vs ~46 dims, no bandit-tier overhead).

---

## 6. Open items to resolve before coding

1. **Confirm which `report-final.md` §0 locks this supersedes** (multi-fidelity
   CFD row; 4-objective→3-objective Pareto; Layer-2 field library; K/bandit
   locks). Requires explicit user authorization per CLAUDE.md §7.2.
2. **Choose the single search fidelity** (2D unsteady slice vs 3D steady
   two-eval proxy) — depends on Phase-3 correlation + per-eval cost.
3. ~~Design the Path A+ panel basis~~ — **✅ DONE, see §10** (control-point grid
   3×6 + corrugation family; implemented + integrated 2026-07-03).
4. **Decide V2 stretch:** true fluid/aero topology optimization (Brinkman /
   unsteady adjoint) as the "emergent airflow surface" investigation.

---

## 7. `report-final.md` §0 lock disposition

Every §0 locked decision, classified. **Physical locks (geometry, kinematics,
material, click, mass) are KEPT unchanged** — the slim plan does not alter the
fan itself. Only the *optimization / CFD / compute machinery* rows change. Per
CLAUDE.md §7.2 the lock rows in `report-final.md` are **not edited**; this table
is the authoritative record of what each row means for V1-Slim.

| §0 row | Disposition | Why |
|--------|-------------|-----|
| Fan architecture (V-unit click-mating) | **KEEP** | Physical architecture unchanged. |
| Per-blade structure (2 ribs + panel) | **KEEP** | Physical. |
| Folding mechanism (rigid rotation) | **KEEP** | Physical. |
| Inter-blade engagement (chamfer + detent) | **KEEP** | Physical. |
| Material (PETG + steel pin) | **KEEP** | Physical. |
| Optimization approach (4-layer, 5-field library) | **MODIFY** | Cut Layer-2 porosity fields (TPMS, noise) — S1. Add Path A+ panel basis + corrugation family — S4. Keep louver/texture/edge shape families + Layer 3 primitive. |
| TO scope (rib only) | **KEEP** | This *is* the zoning (S3). Unchanged and central. |
| Rib preserved zones | **KEEP** | Physical / structural. |
| Pivot architecture | **KEEP** | Physical. |
| Coordinate convention (C11 sign, axes) | **KEEP** | Physics-correctness lock. |
| Rotational-inertia reference axis (wrist +y) | **KEEP** | Feeds I_wrist objective — still used. |
| Mass constraint C9 (<100 g) | **KEEP** | Hard constraint retained. |
| CoM constraint (r_CoM ≤ 0.160 m) | **KEEP** | Hard constraint retained. |
| Compute budget (600–1300 h, K stop rule) | **SUPERSEDE** | Single fidelity + no bandit → ≈⅓ budget (S2). K/stop-rule machinery removed. |
| Compute hardware (incl. G4 GPU for PyFR) | **MODIFY** | PyFR → V2 (S2), so G4 GPU not needed for V1. M3 + Colab CPU remain. |
| Multi-objective Pareto (4 objectives) | **MODIFY** | → 3 objectives (airflow, I_wrist, peak stress). Folded form factor → soft check (S6). |
| Panel thickness range (2.2–3.8 mm, 3 knots) | **MODIFY** | Thickness spline generalized into the Path A+ surface basis; SI bounds (2.2–3.8 mm, folded-collision floor) retained. |
| Folded form factor target (≤50 mm) | **KEEP** (as soft check) | No longer a search axis; still a manufacturability check. |
| Realistic gain target (15–30% IMU-normalized) | **MODIFY** | IMU-normalized already suspended for V1 (`phase_0_signoff.md`); report sim-vs-sim relative gain + feel test. |
| CFD fidelity (multi-fidelity 3-tier) | **SUPERSEDE** | Single working fidelity for search + unsteady CFD to verify winners only (S2). |
| Steady-state proxy (two-eval delta) | **KEEP** | Actively used — Stage 0 cheap pass, candidate search fidelity, and adjoint driver (S5). |
| Cycle count for J_fan (5, extend to 8) | **KEEP** | Physics-correctness for the unsteady verification runs. |
| Stress-test load case (2.5× p, 2× α, √2 ω) | **KEEP** | Structural robustness gate retained. |
| Architecture promotion K (bandit) | **SUPERSEDE** | No multi-fidelity promotion under single fidelity. Categorical architecture (blade count etc.) still searched by BO, without the K/ρ² promotion machinery. |
| Inner-loop acquisition cap (35 TuRBO) | **KEEP** (retune) | TuRBO still the inner search; cap value revisited for the ~25–40-D space. |
| Hypervolume early-stop (4D) | **MODIFY** | → 3D HV early-stop; mechanism unchanged. |
| GP noise model (fixed-floor epistemic) | **KEEP** | Still a GP-based BO. |
| Lock mechanism (deferred V2) | **KEEP** | Unchanged. |
| Campaign tracker (Drive/JSONL) | **KEEP** | Infra reused as-is. |
| JSONL schema | **KEEP** (extend) | May add Path A+ / adjoint fields; bump `SCHEMA_VERSION`. |
| Compute target / stop rule | **SUPERSEDE** | Replaced by single-fidelity budget + evidence-gated refinement (S7). |

**`locks_index.md` machinery locks affected:** Section C (CFD constants — tier
structure collapses; the HIGH-12 MACH tier-specific *physics* lock is kept for
whatever tiers remain) and Section D (BO + campaign — K-promotion, 4D HV,
cost-tuple, multi-fidelity GP machinery superseded per S2/S4/S7). Section A
(architectural) and Section B (material) locks are entirely **KEPT**.

---

## 8. Decision log (why we shifted — for future reads)

Chronological record of the 2026-07-02 design conversation, so a future reader
understands *why* V1-Slim diverges from R11:

1. **Precision mismatch.** R11's apparatus targets precision (unvalidated
   absolute J_fan, 15–30% deltas) finer than the V1 deliverable (blinded feel
   test + unmeasured print-noise floor) can perceive. Confront it rather than
   accept it as risk.
2. **Physics is low-dimensional.** The airflow levers (camber/cupping, drag
   asymmetry, area, low inertia, stiffness) are ~6 intuitive knobs; a 46-D
   search over porosity fields is unlikely to beat a cambered/louvered scoop.
3. **Porosity leaks.** TPMS/noise cutouts fight a max-airflow objective →
   cut (S1). Corrugation/faceting (surface shaping, not holes) is kept and
   *encouraged* — it's aerodynamically plausible and is what the user wants
   the optimizer to discover.
4. **Zoning protects the wind surface.** Structural TO is spatially confined to
   the ribs; it *cannot* delete panel surface. This is a stronger guarantee than
   a co-optimizer "learning" to protect it (S3).
5. **BO vs. adjoint.** Adjoint offers more shape freedom but is fragile on
   separated/shedding flow (steady base may not converge; separated-flow
   gradients are noisy) and is proxy-bound. BO is robust, evaluates the true
   unsteady objective, and gives the Pareto front + categoricals natively →
   BO backbone (S4), adjoint as optional conditional polish (S5).
6. **Loop vs. relay.** Loop where cheap + coupled; relay where expensive +
   weakly coupled. Rib↔aero coupling is weak/asymmetric and BO is costly to
   restart → cheap loop early (Stage 0), relay the expensive stages once,
   refine only as the measured Pareto shift justifies (S7).
7. **Goal (user).** Real TO + real ASO producing a genuinely optimized fan with
   least effort; emergent structure and emergent (non-porous) aero shape; expose
   the airflow-vs-weight trade-off. Not a portfolio showcase, not purely
   empirical.

---

## 9. Execution & resourcing policy

**Where work runs:**

- **M3 local** = all dev + geometry generation (CadQuery) + the BO optimizer
  itself (GP fit + acquisition are cheap) + Gmsh meshing + all Python
  post-processing. Claude writes it; you run one command.
- **Colab CPU** = the CFD compute muscle. Regular CPU for the 2D search sweep;
  **high-RAM CPU for 3D unsteady verification.** No GPU needed for V1 (PyFR → V2).
- **Manual (you)** concentrates in two places: babysitting parallel Colab
  sessions during the BO campaign, and the physical print/assemble/click-tuning.
- **Fusion 360** is out of the loop — CadQuery generates all geometry
  programmatically; Fusion is an optional STL/STEP viewer only.

**Notebook policy (binding, = CLAUDE.md §6):** anything slow on the M3, or
needing high-RAM CPU / GPU, is a **thin notebook in `notebooks/`**. Notebooks
contain **no business logic** — every class/method lives in `src/fanopt/` (or
`scripts/`), is unit-tested, and is *imported* into the notebook. A method may
appear in a notebook only if it already has tests. Notebooks generated in VSCode
may run via the Colab extension where that genuinely simplifies things;
otherwise the operator runs them on Colab. The M3-vs-Colab boundary for SU2/FEA
(Spike 0.6a/0.6b) is **unconfirmed** and is resolved by the first CFD-layer
task.

---

## 10. Path A+ panel basis — committed spec (2026-07-02)

Decision: **control-point grid + corrugation family** (per §1 S4). Panel surface
parameterized over ``(u, v)`` = (normalized radial hub→tip ``[0,1]``, normalized
tangential across local panel width ``[-1,1]``). This is the fixed interface the
geometry code, schema, and 2D-slice extractor build against.

| Component | Spec | Vars |
|---|---|---|
| Thickness grid | 3 radial × 6 tangential control points, each ``t_ij ∈ [2.2, 3.8] mm``; **bilinear** interpolation → top-face height over the planar bottom | 18 |
| Corrugation family | ``a·sin(2π·(u·sinθ + v·cosθ)/λ + φ)`` added to the grid then clamped to [2.2,3.8]: amplitude ``a∈[0,0.8]mm``, wavelength ``λ``, phase ``φ``, orientation ``θ`` | 4 |
| Airfoil mean surface (kept Layer 1) | camber spline + twist | ~6 |
| Louvers (kept, non-porous) | angled surface slats (activation + angle + spacing) for drag asymmetry | ~4 |
| Manufacturing (Layer 4) | print orientation (categorical), layer height, chamfer/detent/clearance | ~4 |
| Fan macro | blade_count {8,10,12} categorical, blade_length (rib widths H12-locked) | ~2 |

**Total ≈ 35–40 vars** (TuRBO; SAASBO fallback at the high end).

**Locks satisfied by construction:** per-point thickness ∈ [2.2,3.8] → thickness
lock + folded-collision floor; planar bottom → plano-convex under rib-flat;
``PANEL_PIVOT_REGION`` control points pinned at boss thickness → boss preserved;
field on the panel only → rib (structural TO) zone untouched (the zoning
guarantee). Mass < 100 g and r_CoM stay hard constraints (checked per design).

**Supersedes** the old 3-knot ``panel_thickness`` spline (subsumed into the grid's
radial dimension). Cut Layer-2 porosity fields (TPMS/noise) per S1.

**Unblocks:** schema update (grid + corrugation params replacing the thickness
spline), the geometry generator (``fields_cad``/``generator_cad`` build the
thickness field surface), and the deployed-geometry → 2D mid-radius
cross-section extractor that feeds ``mesh_2d_slice.build_cascade_slice_mesh``.
