# Phase Checklist

Working tracker for project execution. Pulls from `docs/report-final.md` §Phase 0–6 and §13.
The plan is the spec; this is the to-do list. When in doubt, the plan wins.

Legend: `[x]` done · `[~]` in progress · `[ ]` not started · `[-]` skipped/N/A · `[→V2]` deferred to V2

## V1 / V2 scope pivot (2026-05-13)

Authoritative decision record: **`docs/phase_logs/phase_0_signoff.md`**.

Hardware-instrumented measurement (anemometer, IMU, torsion pendulum) is deferred to V2. V1 ships a printable optimized fan judged by qualitative blinded A/B feel test against the printed flat-panel baseline. Spike 0.6c (CFD numerics validation) is the **only non-negotiable kept-in-V1 spike** — without it, the sim-only V1 is unmoored from physical reality. Deferred-spike sentinels live at `data/spike_0_{2,3,5,7c}/deferral.json`. Full V2 specs in `docs/V2_backlog.md`.

---

## Phase 0 — Scaffolding + Risk Spikes + Environment (Week 1–2)

Dependencies flow forward: 0.0 → 0.1 (parallel) → 0.2 → 0.3 ; 0.4 → 0.5 ; 0.6 → 0.6c ;
0.7 is independent. Environment setup runs in parallel with the spikes.

### Step 0.0 — Project scaffolding ✅
- [x] Git repo created at `fan-optimization/`
- [x] Directory tree per §12.1 (`src/fanopt/{geometry,topopt,cfd,bo,physical,utils}/`, `tests/`, `scripts/`, `configs/`, `notebooks/`, `docs/`, `data/`)
- [x] `pyproject.toml`, `environment.yml`, `.pre-commit-config.yaml`, `.gitignore`, `README.md`, `CLAUDE.md`
- [x] CI scaffold passing on a brand-new repo

### Spike 0.1 — Fusion headless add-in workflow (macOS) ✅
- [x] Minimal Fusion Python add-in reads `params.json`, sets User Parameters, exports STL
- [x] Drive via `Fusion.app/Contents/MacOS/Fusion` (or AppleScript UI fallback)
- [x] Log outcome in `docs/phase_logs/spike_0_1.md`
- [-] Fallback path: CadQuery-only geometry backend (armed if 0.1 had failed)

### Spike 0.2 — Torsional-pendulum rotational-inertia protocol  `[→V2]` DEFERRED

**V1 substitute:** analytic `I_wrist_kgm2` from §6.4 generator. `scripts/run_spike_0_4.py --i-wrist-analytic <value> --f-friction-cumulative-n <value>` skips the Spike-0.2 cross-check and bumps the safety factor 2× → 3× to absorb unverified-inertia uncertainty. Sentinel: `data/spike_0_2/deferral.json`. V2 revisit when V1 ships a fan that feels meaningfully better than baseline.


**Why:** Phase 6 reports `J_fan_measured / W_cycle`; needs `I_wrist` about the **handle-grip +y axis** (NOT the pivot pin axis — they differ by `d_handle = 0.05 m`). Must exist before Spike 0.3 can evaluate IMU-normalized.

**Artifacts shipped (ready to use):**
- `docs/spike_0_2_protocol.md` — operator procedure (rig build, κ calibration, T_osc measurement)
- `docs/phase_logs/spike_0_2.md` — run log template
- `src/fanopt/physical/inertia.py` — I_wrist library (`i_wrist_from_period`, `analyze_trials`)
- `scripts/spike_0_2_analyze.py` — CLI analyzer (reads `calibration.csv` + `measurements.csv`)
- `tests/test_physical/test_inertia.py` — 17 unit tests (analytic rod, gates, mount subtraction)

**Operator tasks:**
- [ ] Build torsional pendulum: suspend assembled fan from a torsion wire/rod **attached at the handle wrist-grip point** (axis must coincide with §6.4 wrist axis)
- [ ] Calibrate torsion constant κ with a known reference mass at known radius → record in `data/spike_0_2/calibration.csv`
- [ ] Run 5 T_osc trials on the Spike 0.3 baseline fan → record in `data/spike_0_2/measurements.csv`
- [ ] Once Phase 1 lands, extend `smoke_test.py` to emit `I_wrist_kgm2` for the baseline (cross-check target)
- [ ] Run `python scripts/spike_0_2_analyze.py --generator-i-wrist <value> --out data/spike_0_2/results.json`
- [ ] **Pass:** repeatability < 3% across 5 measurements; agrees within ±10% of generator-emitted `I_wrist_kgm2`
- [ ] Fill in run log at `docs/phase_logs/spike_0_2.md` and commit

### Spike 0.3 — Baseline physical measurement *(depends on 0.2)*  `[→V2]` DEFERRED

**V1 substitute:** Two co-baselines — **(a)** Phase 2a baseline CFD on the flat-panel 10-blade design gives a simulated `J_fan` that every Phase 4 optimized design's simulated `J_fan` is compared against (sim-vs-sim relative gain); **(b)** Phase 6 blinded A/B feel test of printed top-3 designs vs. the printed baseline (stopwatch-paced 20 strokes; 1-5 scoring on airflow / weight / sound / aesthetics; repeat on a different day). Sentinel: `data/spike_0_3/deferral.json`. Optional V2 upgrade paths in order of cheapness: kitchen-scale + cardboard target (see `docs/spike_0_3_protocol.md` Appendix A) → Phyphox phone IMU → original anemometer rig.



**Artifacts shipped (ready to use):**
- `docs/spike_0_3_protocol.md` — operator procedure (baseline CAD, IMU setup, L8 9-point grid)
- `docs/phase_logs/spike_0_3.md` — run log template
- `src/fanopt/physical/imu.py` — `W_cycle = ∫|I·ω·dω/dt| dt` + kinematic sanity (f, ω, θ vs locked spec)
- `src/fanopt/physical/anemometer.py` — L8 3×3 grid plane integral
- `scripts/run_spike_0_3_baseline.py` — CLI runner (reads IMU CSVs + anemometer CSV + Spike 0.2 results.json)
- `tests/test_physical/test_imu_known_waveform.py` — 11 SHM/CSV tests
- `tests/test_physical/test_anemometer.py` — 10 grid-integration/loader tests

**Operator tasks (blocked until Phase 1 CAD lands for the baseline geometry):**
- [ ] Generate baseline CAD: 10-blade PETG fan, flat panels (no TO, no airfoil camber) via §9.7 generator at default JSON
- [ ] Print 10 baseline blades + pivot pin assembly
- [ ] Verify the Spike 0.2 I_wrist applies to this exact assembled fan (re-measure if any doubt)
- [ ] Set up IMU on handle (axis = +y wrist axis, ≥100 Hz sample rate)
- [ ] Build the anemometer 9-point reticle (3×3 @ 600×600 mm @ 300 mm, 200 mm pitch — L8 lock)
- [ ] Record 5 IMU trials × 10 cycles each, metronome 2 Hz → `data/spike_0_3/imu_trial{1..5}.csv`
- [ ] Record 9-point anemometer grid → `data/spike_0_3/anemometer_grid.csv`
- [ ] Run `python scripts/run_spike_0_3_baseline.py --imu ... --anemometer ... --inertia data/spike_0_2/results.json --out data/spike_0_3/baseline.json`
- [ ] Verify kinematic sanity (f≈2 Hz ±10%, ω_max≈8.8 ±15%, θ_max≈0.7 ±20%); re-shoot if warnings dominate W_cycle
- [ ] Commit the published `J_per_W` from `baseline.json` — this is the canonical Phase-6 baseline

### Spike 0.4 — Click-feature tolerance + 1000-cycle test + V1 lock force balance (H6)
**Two parts:** click cycle life **AND** the V1 force-balance gate that decides whether the rib-tab fallback arms.

**Click cycle life (test articles):**
- [ ] Print 2 single-blade test articles with mating click features at the **panel outer tangential edge** at `(x = L_blade, y = ±panel_tangential_outer ≈ ±0.0225 m)` — NOT on the rib
- [ ] Measure as-printed clearance (target 0.15–0.20 mm per mating surface; calipers + feeler)
- [ ] Measure click engagement force (target 0.5–2 N; force gauge at blade tip)
- [ ] Measure deployed-state alignment uniformity (visual)
- [ ] Run 1000 deploy/fold cycles; inspect detent every 100 cycles
- [ ] Add 100-cycle stress segment at ~2× design-point force (1–4 N); inspect for detent fracture / chamfer chipping / alignment drift
- [ ] **Pass:** clearance in band; force in band; no fracture or excessive wear after 1000+100 cycles; alignment gap variation < 1 mm

**V1 lock force balance (H6):**
- [ ] Measure `F_friction_cumulative` across 9 inter-blade pairs at deployed position (force gauge tangential at the panel tip)
- [ ] Compute `τ_inertial_peak = I_wrist · α_max` using Spike 0.2 I_wrist and α_max = 110 rad/s²
- [ ] Convert: `F_inertial_at_click = τ_inertial_peak / 0.25 m` (wrist-to-tip lever arm, NOT 0.20 m from pivot)
- [ ] **Pass:** `F_friction_cumulative ≥ 2 × F_inertial_at_click`
- [ ] **If fail:** arm `params.layer4.v1_lock_fallback_enabled = True` (printed rib-tab fallback)
- [ ] **If detent fracture/wear:** plan magnetic-catch upgrade (~20–40 g, still inside C9 100 g)

### Spike 0.5 — Single-blade fabrication-noise floor *(depends on 0.4)*  `[→V2]` DEFERRED

**V1 substitute:** Print one V1 top candidate **twice** (same design, same printer, same settings) as a same-design sanity check at Phase 6. Compare by feel. If two prints feel meaningfully different, the print-noise floor is wider than the V1 gain target and the design comparison is contaminated — flag and discuss before declaring V1 ship-ready. No formal CV computation. Sentinel: `data/spike_0_5/deferral.json`.


- [ ] Print **3 identical copies of a single representative blade** on same printer/settings (using 0.4-validated click geometry)
- [ ] Measure dimensional accuracy (calipers at 10 points each), mass (jewelry scale), 3-point bend deflection
- [ ] Assemble each into a 10-blade fan (1 new + 9 baseline); measure J_fan-proxy
- [ ] **Pass:** CV < 5% of mean across the three single-blade fan results
- [ ] **If CV > 5%:** tighten print process OR commit only to gains > 15% (issue #16); document achieved CV in the ledger as the floor

### Spike 0.6 — Colab Pro compute budget probe
- [ ] Run 1 representative 3D unsteady SU2 case (500K cells, 5 pitching cycles, dt=T/200) on Colab Pro CPU; record wall-time + compute-unit cost
- [ ] Repeat on Colab Pro G4-class GPU; record wall-time + compute-unit cost

**Sub-spike 0.6a — M3 local SU2 viability:**
- [ ] Run 1 Tier -1 case (CadQuery → Gmsh 2D corrugated slice → SU2 2D steady → `j_fan.py`) end-to-end on MacBook M3
- [ ] **Pass:** completes ≤ 15 min, finite `J_fan_steady_proxy`
- [ ] **If fail:** shift `smoke_test.py` to Colab Pro CPU

**Sub-spike 0.6b — M3 local FEA viability (gates §59.5 / step 64.5 locality):**
- [ ] Run 1 FEniCSx (or CalculiX) static FEA on M3: cantilever rib under 5 N tip load
- [ ] **Pass:** completes ≤ 2 min, matches analytic tip deflection within 5%
- [ ] **If fail:** move §59.5 combined-blade FEA to Colab Pro CPU (M3 keeps geometry/mesh-QC/Fusion/IMU)

### Spike 0.6c — Tier-1 unsteady-config benchmark validation (H10 — gates Phase 4 launch)
**Sub-spike 0.6c.1 — Tier-1 cfg sanity check (MUST run before benchmark):**
- [ ] Render locked Tier-1 cfg; run 1 inner iteration on a probe mesh
- [ ] **Pass:** SU2 launches + completes 1 outer time-step without parser error
- [ ] If `FREESTREAM_VELOCITY = (0, 0, 0.001)` is not valid in deployed SU2 build, fall back to `MACH_NUMBER = 1e-9` + `REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE`
- [ ] Document working syntax in §9.4.1 as the locked Tier-1 reference cfg

**Sub-spike 0.6c.2 — Published-benchmark validation:**
- [ ] Run NACA 0012 pitching about quarter-chord (k_reduced ≈ 0.5–0.6, Re ≈ 30k–50k) through the working Tier-1 cfg setup
- [ ] **Pass:** lift/drag coefficients within ±15% of published over 5 cycles (discard 1, integrate 4)
- [ ] **If fail (>15% miss):** investigate mesh, dt, low-Mach prec coefficients, dual-time inner iterations BEFORE Phase 4 launch
- [ ] Write `phase0/spike_0_6c/PASS` marker (Phase 4 launch tag is gated on this)

### Spike 0.7 — Generative-geometry + BO-infra sanity check
**Sub-spike 0.7a — Generative geometry sanity:**
- [ ] Run `generate_blade.py` with 10 random parameter sets drawn from JSON schema bounds
- [ ] Visually inspect each STL for manufacturability + reasonable topology
- [ ] Run manufacturability filter; confirm it rejects obviously-infeasible + accepts obviously-good designs
- [ ] Print 2 passing designs; confirm clean print + click features engage
- [ ] **Adversarial set** (≥3 hand-picked params): (a) Layer 2 louver clustered-at-tip pushing cuts toward outer-rib edge; (b) Layer 2 TPMS at min cell-size rotated through click region + ribs; (c) Layer 3 primitive at the bounds-edge of the ≥5 mm outer-rib click-region constraint
- [ ] **Pass:** click footprint bit-identical + rib material from Phase 2 TO bit-preserved on every blade (enforced by `tests/test_geometry/test_click_feature_preservation.py` + `test_panel_mask.py`)

**Sub-spike 0.7b — BO infra scaling:**
- [ ] Run 5–10 LHS samples through architecture bandit + TuRBO + multi-fidelity GP at 37–46 dims on synthetic objective values (no CFD)
- [ ] Calibrate `EPISTEMIC_NOISE_FLOOR` via replicate Tier-1 runs (same design, perturbed ICs); lock at `max(measured, 1e-6)` for GP `train_Yvar`
- [ ] **Pass:** GP fit time ≤ 60 s/iteration; architecture bandit promotes sensible K=4 (hard-coded for spike); TuRBO trust regions update correctly
- [ ] **If GP fit > 60 s consistently:** plan more-aggressive bandit OR sparse-GP variant

**Sub-spike 0.7c — Sobol/random-search baseline (Iso-Compute comparison):**  `[→V2]` DEFERRED

V1 substitute: BO-stall fallback. If Phase 4 Tier-0 best-J_fan does not improve over 20 consecutive acquisitions within an architecture, switch to hand-picked diverse candidates (one near-baseline, one louver-heavy, one TPMS-heavy, one high-camber, one asymmetric — span Layer 2 archetypes). Sentinel: `data/spike_0_7c/deferral.json`. V2 revisit if V1 BO observably stalls AND the operator wants to know whether BO is fundamentally outperforming Sobol.
- [-] ~~Run architecture-bandit infra with GP+acquisition replaced by uniform-random Sobol (50 samples at Tier -1)~~  V1: deferred
- [-] ~~Run 100 BO iterations under production GP+qMFKG config~~  V1: deferred
- [-] ~~Compare at budgets B ∈ {30, 100, 300} h cumulative compute~~  V1: deferred
- [-] ~~**Pass:** BO best-J_fan ≥ Sobol best-J_fan by ≥ 5% on at least 2 of 3 budgets~~  V1: deferred
- [ ] BO-stall fallback wired in Phase 4 orchestrator (V1 substitute)

### Environment setup (parallel with spikes)
- [~] Mac: conda env from `environment.yml` (no preCICE, no TPU, no QSST/XFoil — confirmed). SU2 install line in env.yml now correctly points to brew / conda-forge / source build (Round-7 retired the wrong PyPI `su2` package). **Still needed:** lockfile committed + refreshed weekly. ([ ] you must `conda-lock` on your machine and commit `conda-lock.yml`)
- [ ] Colab Pro: `notebooks/colab_phase4_runner.ipynb` template with SU2 + Gmsh + checkpointing wrapper; parallel-sessions config documented (currently 1-cell stub)
- [ ] Colab GPU (G4-class, 95 GB): PyFR install verified; CUDA toolkit pinned (Round-9 HIGH-11 — T4 will OOM at PyFR p=3). **No pin file committed yet.**
- [x] **Drive/JSONL ledger wired up:** `src/fanopt/utils/ledger.py` + `drive_io.py` + `slicing.py` now real. Pydantic-style `LedgerRow` dataclass with mandatory + optional Phase-4 fields, `design_hash` per §6.2.5 (round-trip property tested), `.done` / `.heartbeat` / `.claim` marker helpers, round-robin slice assignment + rebalance. 86 unit tests.

### Phase 0 user signoff (must complete before Phase 1 launches)
- [ ] J_fan metric (§9.4) locked
- [ ] Open Question #2 closed: Phase 2 TO loads come from Phase 3 baseline CFD-derived pressures
- [ ] Generative parametric design (§3.1.3, §9.7) locked as panel-topology approach (no density-based TO on the panel)

---

## Phase 1 — 4-Layer Generative Blade Geometry Pipeline (Week 3) — STUB

*(To be filled out when Phase 0 nears completion. Quick starting points:)*

- [ ] Fusion Python add-in `fan_addin.py`: reads `params.json` → regenerates 1 V-unit blade → 10 instances around pivot → per-blade STLs + deployed-fan STEP
- [ ] CadQuery fallback generator (`generate_fan_cq.py`) — always built (fast path for Phase 2b/Phase 4 inner loops)
- [ ] V-unit blade parameterization (`blade_geom.py`): 2 ribs (165 mm × 4–6 mm tapered × 2 mm) + 1 trapezoidal panel (full L_blade × tangential width per H13 formula × 2.2–3.8 mm at 3 control points; 12 mm-OD boss at pivot) + Option A click chamfer + detent at panel outer tangential edge
- [ ] JSON parameter schema (`src/fanopt/geometry/schema.py`) covers all 4 layers (~37–46 vars) + load-time validation
- [ ] Shared constants in `schema.py`: `HUB_RADIUS = 0.020 m`, `RIB_TIP_TAPER = 0.015 m`, `PIVOT_CENTER_X = 0.008 m`, `PANEL_PIVOT_REGION = CircularMask(...)`, `CLICK_FOOTPRINT_X_RANGE`, `CLICK_FOOTPRINT_Y_RANGE_PANEL_EDGE`
- [ ] Print-strategy decision script (`print_strategy.py`): per-blade vs. full-assembly based on bed size
- [ ] Roundtrip smoke test (`smoke_test.py`): JSON → geometry → Gmsh mesh → SU2 steady stub → J_fan; emits `I_wrist_kgm2` for Spike 0.2 cross-check

---

## Later phases — placeholders

- [ ] **Phase 2a** — Baseline 2D CFD slice for rib-TO loads (Week 3.5; ½ day; gates Phase 2 by producing `phase3_baseline.csv`)
- [ ] **Phase 1.9** — SIMP pre-baked-strip sanity check (Week 3.7; N=30 LHS; gates Phase 2 launch)
- [ ] **Phase 2** — Rib-only plate-bending TO (Week 4; runs after 1.9 + 2a)
- [ ] **Phase 2.5** — Rib-only fillet 3D static FEA re-check (Week 4.5)
- [ ] **Phase 2b** — Generative parametric panel optimization (Week 5–8)
- [ ] **Phase 3** — 2D CFD slice on rigid corrugated geometry (Week 6)
- [ ] **Phase 4** — Multi-fidelity BO on Colab Pro (Week 8–10)
- [ ] **Phase 5** — High-fidelity verification + PyFR cross-solver on top-3 (Week 11–12). **V1 note:** must print 3-5 *structurally-diverse* candidates spanning Layer 2 archetypes (one louver-heavy, one TPMS-heavy, one near-baseline, etc.) — NOT 5 variations of one shape. Mitigates "BO exploits a sim artifact" failure mode.
- [ ] **Phase 6** — Single-material PETG print + blinded A/B feel test (V1 scope; IMU + anemometer paths deferred to V2 per `docs/phase_logs/phase_0_signoff.md`). Includes: print one top candidate twice (fab-noise sanity), blinded A/B with stopwatch-paced 20 strokes, 1-5 scoring on airflow / weight / sound / aesthetics, repeat on a different day. Quantitative measurements re-enter at V2.
