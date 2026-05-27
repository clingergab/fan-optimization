# Phase Checklist

Working tracker for project execution. Pulls from `docs/report-final.md` §Phase 0–6 and §13.
The plan is the spec; this is the to-do list. When in doubt, the plan wins.

Legend: `[x]` done · `[~]` in progress · `[ ]` not started · `[-]` skipped/N/A · `[→V2]` deferred to V2 · `[→Phase 5]` deferred to Phase 5

## V1 / V2 scope pivot (2026-05-13)

Authoritative decision record: **`docs/phase_logs/phase_0_signoff.md`**.

Hardware-instrumented measurement (anemometer, IMU, torsion pendulum) is deferred to V2. V1 ships a printable optimized fan judged by qualitative blinded A/B feel test against the printed flat-panel baseline. Spike 0.6c (CFD numerics validation) is the **only non-negotiable kept-in-V1 spike** — without it, the sim-only V1 is unmoored from physical reality. Deferred-spike sentinels live at `data/spike_0_{2,3,5,7c}/deferral.json`. Full V2 specs in `docs/V2_backlog.md`.

## Spike 0.6c.2 deferred to Phase 5 (2026-05-14, supersedes 2026-05-13 revision)

Authoritative decision record: **`docs/phase_logs/spike_0_6c.md` → "V1 decision — Spike 0.6c.2 deferred to Phase 5 (2026-05-14)"**.

The 2026-05-14 regime diagnostic on Cell 8's SU2 history.csv confirmed the production-faithful MACH=1e-9 cfg produces body-in-still-air added-mass/quadratic-drag forces (CL at 2× the prescribed pitching frequency, bias ratio 2.234) — not wind-tunnel-like aerodynamic lift. The 2026-05-13 internal-consistency revision (convergence + symmetry gates) implicitly assumed wind-tunnel physics and is itself superseded. Sub-spike 0.6c.2 is now a **Phase 5 deliverable**: SU2 vs PyFR cross-solver agreement (C_L_max within ±20%, C_d_mean within ±25%, hysteresis loop sign matches + area within 2×). PyFR is already provisioned in the Phase 5 budget. Phase 4 launch gates on sub-spike 0.6c.1 only.

---

## Phase 0 — Scaffolding + Risk Spikes + Environment (Week 1–2)

Dependencies flow forward (per plan §Phase 0 dependency table): 0.0 → 0.1 (parallel) → 0.2 → 0.3 ; 0.4 → 0.5 ; 0.6 → 0.6c ; **0.7 is independent (no upstream dep)**. Environment setup runs in parallel with the spikes.

### What's still open in Phase 0 (as of 2026-05-13)

| Item | Status | Unblock path |
|---|---|---|
| Spike 0.2 / 0.3 / 0.5 / 0.7c | `[→V2]` | V1 substitutes wired; revisit only on V2 |
| Spike 0.4 (click cycle + V1 lock) | Operator-blocked | Code done; needs printed test articles + force gauge |
| Spike 0.6 main + 0.6a + 0.6b (compute probes) | Operator-blocked | Calibration, not a gate per plan; one M3 run + one Colab run each |
| Spike 0.6c.1 | `[~]` PASS recovered on Colab VM via the 0.6d notebook Cell 5b `--su2-history-csv` evidence path; awaiting push of `data/spike_0_6c/PASS` to `main` | 0 (recovery path; was: stdout-capture fix, now superseded) |
| Spike 0.6c.2 | `[→Phase 5]` DEFERRED 2026-05-14 — see plan §Phase 5 step 62.5 | Re-enters as 3-solver published-reference benchmark in Phase 5 |
| Spike 0.6d (2026-05-14 addition, redesigned 2026-05-15; 0.6d.2 freq-consistency = sole gate; 0.6d.1 + 0.6d.3 advisory) | `[~]` code on `main`; awaiting 2× Colab SU2 runs (in flight) | ~2–4 h Colab CPU |
| Spike 0.7a operator steps | Blocked on Phase 1 CadQuery generator | Land `make_outer_envelope` + 5 field functions + primitive function |
| Spike 0.7b | ✅ done | — |
| Environment: Mac `conda-lock.yml` | Operator action | One `conda-lock` invocation |
| Environment: `colab_phase4_runner.ipynb` | Defer | Phase 4 prep, not Phase 0 launch |
| Environment: PyFR install + CUDA pin | Defer | Phase 5 prep, not Phase 0 launch |
| Phase 0 signoff items | Administrative | Operator confirmation of plan-side locks |

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

### Spike 0.6c — Tier-1 unsteady-config sanity (V1: gates Phase 4 launch)

**Artifacts shipped (V1):**
- `src/fanopt/cfd/spike_0_6c.py` — library: cfg sanity + simplified aggregator (sub_1 only)
- `src/fanopt/cfd/configs.py` — `render_unsteady_cfg()` with the Round-9 HIGH-12 fallback path locked in by default
- `configs/su2/fan3d_unsteady.cfg.j2` — production Tier-1 template
- `scripts/run_spike_0_6c_1.py` / `run_spike_0_6c.py` — sub_1 runner + aggregator
- `scripts/parse_su2_history_to_cycles.py` — SU2 `history.csv` → per-cycle `measured.csv` (kept for Phase 5 cross-solver use)
- `scripts/diagnose_su2_pitching_regime.py` — regime classifier (added 2026-05-14 as part of the deferral evidence trail)
- `scripts/launch_phase4.py` — Phase 4 launch gate; reads `data/spike_0_6c/PASS`
- `notebooks/colab_spike_0_6c.ipynb` — Colab runbook (Cells 1–11 retained; cells 8–10 marked DEFERRED-TO-PHASE-5)
- `docs/spike_0_6c_protocol.md` + `docs/phase_logs/spike_0_6c.md` (full diagnostic addendum)

**Sub-spike 0.6c.1 — Tier-1 cfg sanity check** `[~]` Awaiting Colab `sub_1.PASS`
- [x] Render locked Tier-1 cfg; run 1 inner iteration on a probe mesh
- [x] Round-9 HIGH-12 fallback (`MACH_NUMBER = 1e-9` + `REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE`) shipped as the default — SU2 v8.0.1 rejects the primary `FREESTREAM_OPTION = FREESTREAM_VELOCITY` syntax at parse time
- [x] Round-9 HIGH-12 + C11 sign locks pinned by `tests/test_cfd/test_unsteady_freestream_consistency.py`
- [x] **0.6c.1 PASS recovered (2026-05-21)** on the Colab VM via the 0.6d notebook Cell 5b history.csv-evidence path: `python scripts/run_spike_0_6c_1.py --su2-history-csv <prior Drive history.csv>` writes `sub_1.PASS` from the existing 2026-05-14 SU2 Cell-8 run, no fresh SU2 invocation needed. The earlier `outer_steps=0` stdout-capture issue is moot under the evidence path. PASS marker still needs pushing to `main` (Cell 10 PAT push, optional).

**Sub-spike 0.6c.2 — NACA 0012 benchmark** `[→Phase 5]` DEFERRED 2026-05-14
- Removed from V1 scope after the 2026-05-14 regime diagnostic confirmed the moving-body-in-still-air cfg can't be validated against any wind-tunnel reference in the same frame. Full decision record + evidence: `docs/phase_logs/spike_0_6c.md`.
- Code-side removals (entered in `docs/retired_phrases.yaml`): `CONVERGENCE_TOLERANCE_PCT`, `SYMMETRY_TOLERANCE_PCT`, `CONVERGENCE_METRICS`, `ConvergenceCheck`, `SymmetryCheck`, `BenchmarkCycleData`, `BenchmarkResult`, `check_convergence`, `check_symmetry`, `analyze_benchmark`. Script `run_spike_0_6c_2.py` deleted.
- Re-enters as **Phase 5 step 62.5** (2026-05-14 addition): three-solver published-reference benchmark (SU2 + PyFR + OpenFOAM `pimpleFoam`) against a body-in-still-air-frame reference case (Sane & Dickinson 2002 / Morison-equation literature). Replaces the wind-tunnel-frame framing entirely. See plan §Phase 5 step 62.5.

### Spike 0.6d — Tier-1 added-mass frequency-consistency gate (2026-05-14 addition; redesigned 2026-05-15; gates Phase 4 launch alongside 0.6c.1)

**Why this spike exists:** After 0.6c.2's deferral, Phase 4 would have launched with no independent quantitative check on SU2's body-in-still-air response at MACH=1e-9. The gate was **redesigned 2026-05-15** after the first live run showed the original 0.6d.1-gating design was unsound (nondimensionalisation conflation + a symmetry criterion ill-posed for a net-work fan). The gate now rests on a single normalization-invariant falsification test (0.6d.2). Authoritative: §Phase 0 Spike 0.6d + `docs/phase_logs/phase_0_signoff.md` Note 3.

**Code status: DELIVERED on `main`** — implementation + tests landed (PRs #10, #13) plus the 2026-05-21 y→z pitching-axis follow-up for the 2D cfg; awaiting the Colab run.

**Sub-spike 0.6d.2 — 2D thin-plate added-mass frequency-consistency** `[~]` code done; awaiting Colab `(GATING — sole Phase-4 gate)`
- [x] `configs/su2/thin_plate_2d_pitching.cfg.j2` + `render_thin_plate_2d_pitching_cfg` (+ tests)
- [x] `recover_added_mass_projection` + `check_added_mass_freq_consistency` in `src/fanopt/cfd/spike_0_6d.py` (+ tests)
- [x] `scripts/run_spike_0_6d_2.py` — two-frequency CLI (`--history-csv-f1/-f2`, `--omega-f1/-f2`) (+ tests)
- [ ] **Colab:** generate the 2D plate mesh, run SU2 at ω₁ and ω₂ (same plate/pivot/θ_max), feed both history.csv to the runner
- [ ] **Pass (GATING):** `|I_a(ω₁) − I_a(ω₂)| / mean < 0.25` (normalization-invariant, parameter-free)

**Sub-spike 0.6d.1 — Symmetry + dimensional-force sanity** `[~]` code done `(ADVISORY — demoted 2026-05-15, does NOT gate)`
- [x] `check_symmetry_dimensional` + `scripts/run_spike_0_6d_1.py` (+ tests) — recorded for Phase 5; not a Phase-4 blocker

**Sub-spike 0.6d.3 — SU2 incompressible-mode cross-check** `[~]` code done `(ADVISORY, no marker)`
- [x] `check_incompressible_cross` + `scripts/run_spike_0_6d_3.py` (+ tests)

**Aggregator + Phase 4 gate**
- [x] `scripts/run_spike_0_6d.py` — `--sub-2-json` required (the gate); `--sub-1-json`/`--sub-3-json` optional advisory; writes `data/spike_0_6d/PASS` iff `sub_2.freq_consistency_passed`
- [x] `scripts/launch_phase4.py` requires both `data/spike_0_6c/PASS` AND `data/spike_0_6d/PASS` (logic unchanged; docstring updated for the new 0.6d gate semantics)
- [x] `tests/test_scripts/test_launch_phase4.py` dual-gate tests
- [ ] Run on Colab Pro CPU; ~2-4 h total (2× short 2D-plate SU2 runs)

### Spike 0.7 — Generative-geometry + BO-infra sanity check
**Sub-spike 0.7a — Generative geometry sanity:**
- [x] Adversarial-parameter library landed (`src/fanopt/geometry/spike_0_7a.py`): louver-clustered-at-tip, TPMS at min-cell rotated through click+rib, Layer 3 primitive at click-clearance bound — 3 hand-picked cases per spec
- [x] Library-level coverage: `tests/test_geometry/test_spike_0_7a.py` — parameter sampling, record dataclass, analyze gate, all four boolean checks
- [x] Click-feature preservation + panel-mask CI tests pass (`test_click_feature_preservation.py`, `test_panel_mask.py`)
- [ ] Run `generate_blade()` (now scaffolded) on the adversarial set + 10 random LHS draws once CadQuery generator helpers land in Phase 1
- [ ] Visually inspect each STL; print 2 passing designs; confirm click engagement
- [ ] **Pass:** click footprint bit-identical + rib material bit-preserved on every blade

**Sub-spike 0.7b — BO infra scaling** ✅
- [x] Synthetic-objective runner landed (`scripts/run_spike_0_7b.py` + `src/fanopt/bo/spike_0_7b.py`); per-iteration GP-fit timing + architecture-bandit screen + TuRBO TR state machine
- [x] **GP backend decision (2026-05-13):** botorch path retired from the spike to enforce CLAUDE.md §4.1 "no inline imports anywhere"; numpy-only RBF GP (Cholesky-solved exact GP) covers all three gates (60s fit time, K=4 promotions, TR shrink/grow). Production BO continues to use botorch via the `fanopt.bo.*` modules under the `[bo]` extras.
- [x] `EPISTEMIC_NOISE_FLOOR_DEFAULT = 1e-6` locked + `calibrate_epistemic_noise_floor` shipped (`src/fanopt/bo/spike_0_7b.py`)
- [x] All three gates exercised end-to-end on synthetic objective: GP fit ≤ 60 s, K = 4 promotions, TR shrinks-on-fail / grows-on-success
- [ ] Re-validate timing gate at full 37–46 dims once production GP backend is wired in Phase 4 (synthetic-objective check is the spike's V1 scope)

**Sub-spike 0.7c — Sobol/random-search baseline (Iso-Compute comparison):**  `[→V2]` DEFERRED

V1 substitute: BO-stall fallback. If Phase 4 Tier-0 best-J_fan does not improve over 20 consecutive acquisitions within an architecture, switch to hand-picked diverse candidates (one near-baseline, one louver-heavy, one TPMS-heavy, one high-camber, one asymmetric — span Layer 2 archetypes). Sentinel: `data/spike_0_7c/deferral.json`. V2 revisit if V1 BO observably stalls AND the operator wants to know whether BO is fundamentally outperforming Sobol.
- [-] ~~Run architecture-bandit infra with GP+acquisition replaced by uniform-random Sobol (50 samples at Tier -1)~~  V1: deferred
- [-] ~~Run 100 BO iterations under production GP+qMFKG config~~  V1: deferred
- [-] ~~Compare at budgets B ∈ {30, 100, 300} h cumulative compute~~  V1: deferred
- [-] ~~**Pass:** BO best-J_fan ≥ Sobol best-J_fan by ≥ 5% on at least 2 of 3 budgets~~  V1: deferred
- [ ] BO-stall fallback wired in Phase 4 orchestrator (V1 substitute)

### Environment setup (parallel with spikes)
- [~] Mac: conda env from `environment.yml` (no preCICE, no TPU, no QSST/XFoil — confirmed). SU2 install line in env.yml now correctly points to brew / conda-forge / source build (Round-7 retired the wrong PyPI `su2` package). **Still needed:** lockfile committed + refreshed weekly. ([ ] you must `conda-lock` on your machine and commit `conda-lock.yml`)
- [x] Colab Pro CPU: `notebooks/colab_spike_0_6c.ipynb` runbook for Spike 0.6c (Cells 1–11; SU2 install path with Drive cache + binary release + source-build fallback; gmsh native deps; `_restore_exec_bits` after Drive roundtrip)
- [ ] Colab Pro: `notebooks/colab_phase4_runner.ipynb` Phase 4 orchestration template with SU2 + Gmsh + checkpointing wrapper; parallel-sessions config documented (currently 1-cell stub)
- [ ] Colab GPU (G4-class, 95 GB): PyFR install verified; CUDA toolkit pinned (Round-9 HIGH-11 — T4 will OOM at PyFR p=3). **No pin file committed yet.**
- [x] **Drive/JSONL ledger wired up:** `src/fanopt/utils/ledger.py` + `drive_io.py` + `slicing.py` now real. Pydantic-style `LedgerRow` dataclass with mandatory + optional Phase-4 fields, `design_hash` per §6.2.5 (round-trip property tested), `.done` / `.heartbeat` / `.claim` marker helpers, round-robin slice assignment + rebalance.

### Quality + tooling (parallel with spikes)
- [x] Inline-import audit (CLAUDE.md §4.1) — **0 violations** across `src/`, `scripts/`, `tests/`. AST scanner in CI.
- [x] Retired-phrase audit gate (`tests/test_audit/test_no_stale_architecture_refs.py`) clean against the current catalog; scans `src/fanopt/**/*.py` only.
- [x] ruff + black + mypy sweep landed 2026-05-13: pyproject configured, 100 files reformatted, 114 ruff auto-fixes + 32 unsafe-fixes + 16 targeted fixes; B905 strict=True on 8 zip sites; pre-existing mypy errors documented (8 in deferred-spike fixture + drive_io.py, none new).
- [x] Coverage: aggregate 95% on `src/fanopt`; 100% on every module authored in today's work (envelope, fields, primitives, manufacturability, generator, schema).

### Phase 0 user signoff

These are administrative confirmations of plan-side locks the operator has already accepted (every item is locked in `docs/report-final.md`; the checklist box exists only to make signoff explicit). The dual Spike 0.6c + 0.6d PASS markers are the real gates.

- [ ] J_fan metric (§9.4) — already locked in the plan; signoff = "I read it and accept"
- [ ] Open Question #2 — already closed in the plan: Phase 2 TO loads come from Phase 2a baseline CFD-derived pressures (Phase 2a was added to break the Phase-3-circular dependency; the plan was updated)
- [ ] Generative parametric design (§3.1.3, §9.7) — already locked in the plan as the panel-topology approach (no density-based TO on the panel)
- [ ] **Spike 0.6c PASS marker present** (`data/spike_0_6c/PASS`; 0.6c.1 cfg sanity) — recovered on Colab VM 2026-05-21; awaiting push to `main`
- [ ] **Spike 0.6d PASS marker present** (`data/spike_0_6d/PASS`; 0.6d.2 freq-consistency) — in flight on Colab; the dual gate `launch_phase4.py --check` requires both markers

---

## Phase 1 — 4-Layer Generative Blade Geometry Pipeline (Week 3)

**Status (2026-05-13):** schema + orchestration scaffold landed; CadQuery generator + Fusion add-in are the remaining real-code pieces.

### Schema layer ✅ (lands the BO search-space contract)
- [x] Shared locked constants in `src/fanopt/geometry/schema.py`: `HUB_RADIUS_M = 0.020`, `RIB_TIP_TAPER_M = 0.015`, `PIVOT_CENTER_X_M = 0.008`, `PANEL_PIVOT_REGION = CircularMask(0.008, 0, 0.007)`, `CLICK_FOOTPRINT_X_RANGE_M`, `CLICK_FOOTPRINT_Y_RANGE_PANEL_EDGE`, full kinematics symbol table (`F_WAVE_HZ`, `OMEGA_BLADE_MAX_RAD_PER_S`, `THETA_MAX_RAD`, `V_TIP_M_PER_S`, `PITCHING_OMEGA_VEC` with C11 negative-y lock), PETG material constants
- [x] JSON parameter schema covers all 4 layers (~37–46 vars) with load-time + bounds validation:
  - `src/fanopt/geometry/envelope.py` — Layer 1 (`Layer1Params`): blade_count, camber/twist/thickness splines, edge profile, Fourier LE/TE amplitudes
  - `src/fanopt/geometry/fields.py` — Layer 2 (`Layer2Params` + `LouverField` / `TextureField` / `EdgeFeatureField` / `NoiseField` / `TpmsField`): the 5-field library + ≤3-active cardinality bound
  - `src/fanopt/geometry/primitives.py` — Layer 3 (`Layer3Primitive`): capped 0-1 primitive with margin / size / rotation bounds
  - `src/fanopt/geometry/manufacturability.py` — Layer 4 (`Layer4Params`): print orientation, layer height, click chamfer/detent/clearance
  - `src/fanopt/geometry/generator.py` — top-level (`BladeDesignParams`): nested aggregator with cross-layer plano-convex validation under rib-flat
- [x] `to_dict` / `from_dict` round-trip on every layer for BO ledger serialisation
- [x] 100% test coverage on every schema module + the aggregator

### Orchestration + manufacturability scaffold ✅ (lands the generator contract)
- [x] `generate_blade(params) -> GenerationResult`: deterministic Layer 1 → 2 → 3 → 4 application order; Layer 2 fixed sub-order TPMS → noise → louver → texture → edge; Layer 3 wrapped in try/except per plan §9.7
- [x] `panel_domain_mask_description(blade_count)`: serialised PANEL_PIVOT_REGION + CLICK_FOOTPRINT exclusion masks per §9.7.1 Step 0
- [x] `GenerationStatus` enum: OK | LAYER3_FAILED | MFG_REJECTED — status branches exercised by monkey-patched tests
- [x] `GeneratorVersion = "0.1.0-scaffold"` stamp so the orchestration contract is versioned independently of CadQuery helper swap-ins
- [x] §9.7.3 / §N7 14-row manufacturability filter protocol: `CheckSeverity` (critical/moderate/soft/hard_bound), `CheckStatus` (passed/failed/pending_cadquery), `run_manufacturability_filter(geometry_description)`, `_aggregate_score(checks)` extracted as testable scoring helper
- [x] Geometry-level checks (#1, #2, #3, #4, #5, #6, #8, #12, #13, #14) marked PENDING_CADQUERY — they wait on the CadQuery helpers below
- [x] Hard-parameter-bound checks (#7, #9, #10, #11) register as PASSED (upstream-enforced by the layer dataclasses)

### CadQuery generator (Phase 1 main work) ✅ landed 2026-05-21
- [x] CadQuery 2.7.0 installed locally + `import cadquery as cq` smoke OK
- [x] Layer 1 envelope — `src/fanopt/geometry/envelope_cad.py::make_outer_envelope` (the `_cad` split keeps `envelope.py` schema-only per CLAUDE.md §4.1)
- [x] Layer 2 fields — `src/fanopt/geometry/fields_cad.py::apply_{tpms,noise,louver,texture,edge_feature}_field` + `apply_layer2_fields` dispatcher in the locked TPMS → noise → louver → texture → edge order
- [x] Layer 3 primitive — `src/fanopt/geometry/primitives_cad.py::apply_primitive`, wrapped in try/except by `generator_cad` per plan §9.7
- [x] Manufacturability shape inspection — `src/fanopt/geometry/manufacturability_cad.py::run_manufacturability_filter_cad`: 8 of 10 prior PENDING_CADQUERY checks now real; #1 (per-feature size) + #8 (per-feature aspect) remain PENDING with documented Phase-2 deferral
- [x] V-unit blade composition — `src/fanopt/geometry/assembly_cad.py::make_vunit_blade`: panel (envelope ∘ Layer 2 ∘ Layer 3) + 2 ribs (H12 width taper) + pivot boss + Round-9 HIGH-8 Option A click chamfer + hemispherical detent
- [x] Deployed-fan composition + physical properties — `src/fanopt/geometry/fan_assembly.py::deploy_fan` / `compute_mass_kg` / `compute_centre_of_mass` / `compute_i_wrist_kgm2`
- [x] Phase 1 smoke run on representative designs — `docs/phase_logs/phase_1_smoke.md` (baseline + features_light pass; full-features 3-field-stack times out — Phase-2 implementation refinement)

### Fusion + ancillary tooling ✅ landed 2026-05-21
- [x] `scripts/fan_addin.py` — CadQuery-based replacement for the planned Fusion 360 add-in. Reads `params.json` → generates V-unit blade → N instances around pivot → per-blade STLs + deployed-fan STEP. Does NOT require Fusion 360.
- [x] `scripts/print_strategy.py` — per-blade vs full-assembly decision against configurable bed dimensions. Default 256 × 256 mm.
- [x] `scripts/smoke_test.py` — Phase-1 roundtrip: JSON → geometry → mass / centroid / I_wrist / manufacturability summary. Gmsh + SU2 + J_fan downstream wiring is a Phase 2/4 task; the current smoke stops at the physical-property summary.

---

## CFD pipeline infrastructure (shared across Phase 2a / Phase 3 / Phase 4)

Pre-built so the Phase-2a/3/4 steps don't have to build their own cfg renderers from scratch.

- [x] Tier -1 (2D mid-radius slice steady) cfg + renderer: `configs/su2/slice_steady.cfg.j2` + `render_slice_steady_cfg`. MACH=0.0064, 2-component freestream, `FREESTREAM_DIRECTION_2D_PRODUCTIVE / RETURN` C2 constants. **Consumed by:** Phase 2a baseline rib-TO load extraction, Phase 3 R²-correlation sweep, Phase 4 architecture-bandit screening.
- [x] Tier 0 (3D steady, full deployed fan) cfg + renderer: `render_steady_cfg`. **Consumed by:** Phase 4 architecture-bandit promotion (3D steady, ranking-only) + Phase 4 two-eval delta proxy (PRODUCTIVE / RETURN per C2 lock).
- [x] Tier 1 (3D unsteady) cfg + renderer: `render_unsteady_cfg` with the Round-9 HIGH-12 fallback default. **Consumed by:** Phase 4 BO inner loop on promoted architectures + Phase 5 verification.
- [ ] `configs/su2/slice_unsteady.cfg.j2` (Phase 3 inner-loop verification cfg) — currently an empty scaffold stub; lands in Phase 3 prep.
- [ ] `mesh_2d_slice.py` — Gmsh 2D slice mesh generator that Phase 2a / Phase 3 / Phase 4-Tier-(-1) all consume.
- [ ] `j_fan.py` — canonical J_fan post-processor per plan §9.4. Currently a Phase-1-stub module.
- [ ] Per-design config-hash assertion (§9.4.1) — enforces that the cross-tier locked numerics are identical per-design across tiers.

---

## Phase 2 — Rib SIMP topology optimization (Week 4)

Pre-Phase-2 sequence (gating order: 1.9 → 2a → 2):

- [ ] **Phase 2a** — Baseline 2D CFD slice for rib-TO loads (Week 3.5; ½ day; gates Phase 2 by producing `phase3_baseline.csv`). Runs the Tier -1 `slice_steady.cfg` pipeline once on the Phase 1 baseline flat-panel design. Requires Phase 1 baseline geometry + `mesh_2d_slice.py` + `j_fan.py`.
- [ ] **Phase 1.9** — SIMP pre-baked-strip sanity check (Week 3.7; N=30 LHS; gates Phase 2 launch). 30 SIMP solves × 2 branches; §N7 rejection-rate gate within +20pp, no element > 0.7·σ_allow in the deleted preserved zone.
- [ ] **Phase 2** — Rib-only plate-bending TO (Reissner-Mindlin, 2D; per §9.2 / §3.1); 4 locked load cases (productive, return, inertial, click engagement); FEniCSx.
- [ ] **Phase 2.5** — Rib-only fillet 3D static FEA re-check (Week 4.5; localized junction).

## Phase 2b / Phase 3 / Phase 4 — placeholders

- [ ] **Phase 2b** — Generative parametric panel optimization (Week 5–8); consumes the Phase 1 4-layer hybrid generator (now scaffolded above) once the CadQuery helpers land
- [ ] **Phase 3** — 2D CFD slice on rigid corrugated geometry (Week 6); roughness-model R²≥0.4 calibration against fully-resolved corrugation
- [ ] **Phase 4** — Multi-fidelity BO on Colab Pro (Week 8–10); blocked on `data/spike_0_6c/PASS` per `scripts/launch_phase4.py`. Dependencies still to wire:
  - [ ] `notebooks/colab_phase4_runner.ipynb` — full Phase 4 orchestration (currently a 1-cell stub)
  - [ ] `configs/architecture_enumeration.yaml` — 60–100 architectures after H1-locked pruning; pre-commit `+10%` growth gate
  - [ ] Per-design config-hash assertion (§9.4.1)
  - [ ] Production GP backend wired (botorch SingleTaskGP + qMFKG + qNEHVI + TuRBO); the Spike 0.7b numpy-only sanity check is V1 scope, not production

## Phase 5 — High-fidelity verification + PyFR cross-solver (Week 11–12)

- [ ] PyFR p=3 verification on top-3 Pareto candidates (Colab Pro G4 GPU; T4 will OOM per Round-9 HIGH-11)
- [ ] Step 59.5 combined-blade FEA on the full assembly (Phase 2 rib + Phase 2b real panel topology)
- [ ] **Spike 0.6c.2 cross-solver gate** (deferred from V1 internal-consistency revision): SU2 vs PyFR on the NACA 0012 case at (Re=40k, k=0.55, ±10°) — acceptance: C_L_max within ±20%, C_d_mean within ±25%, hysteresis loop sign matches + area within 2×. Documents the absolute-accuracy bound that the V1 internal-consistency gate alone does not establish.
- [ ] **V1 print-diversity rule:** print 3–5 *structurally-diverse* candidates spanning Layer 2 archetypes (one louver-heavy, one TPMS-heavy, one near-baseline, etc.) — NOT 5 variations of one shape. Mitigates "BO exploits a sim artifact" failure mode.

## Phase 6 — Single-material PETG print + blinded A/B feel test (V1 scope)

IMU + anemometer quantitative paths deferred to V2 per `docs/phase_logs/phase_0_signoff.md`.

- [ ] Print top-3 Pareto designs + one duplicate of a top candidate (Spike 0.5 V1 substitute — fab-noise sanity)
- [ ] Print the flat-panel baseline (Spike 0.3 V1 substitute)
- [ ] Blinded A/B protocol: operator hands fans without naming them; stopwatch-paced 20 strokes at 2 Hz metronome cadence; 1-5 scoring on airflow / weight / sound / aesthetics; repeat on a different day
- [ ] If two prints of the same design feel meaningfully different → flag print-noise floor > V1 gain target before declaring ship-ready
- [ ] Document final pick in `docs/phase_logs/phase_6_signoff.md`
