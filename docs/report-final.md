# Topology Optimization and Aerodynamic Shape Optimization for a 3D-Printed Folding Fan

## Comprehensive Design, Optimization, and Fabrication Guide

**Date:** 2026-05-12
**Revision:** Final spec — V-unit blade architecture, single-material PETG, click-mating, plate-bending rib TO, 4-layer hybrid panel parameterization (~37-46 vars), multi-fidelity BO. Full revision history in Appendix D.
**Scope:** Combined rib structural topology optimization (TO, plate-bending) and **4-layer hybrid generative panel design** (envelope + Fourier modulation, macro-pattern + procedural math fields including TPMS and Perlin/Simplex noise, capped 0-1 primitive, manufacturing categoricals) with ML surrogate modeling for a **fully 3D-printed single-material PETG folding fan with discrete V-unit blades**. ~37-46 design variables optimized via BoTorch (architecture bandit over categoricals + TuRBO over continuous + multi-fidelity GP over CFD tiers, ). **Honest framing:** this is an **expert-priored 4-layer parameterized design study with procedural-math fields for organic variation** — *not* topology optimization in the SIMP / level-set sense. Layer 1's Fourier modulation, Layer 2's 5 hand-picked field types {louver, texture, edge, noise-threshold, TPMS}, and Layer 3's primitive shape library {slot, ellipsoid, wedge} are all human priors; BO searches *within* them but cannot invent a 6th field type or a primitive shape outside the library. Output can produce slatted, lattice, scalloped, swiss-cheese, organic-noise, or solid panels — surprising-looking from inside the parameterization, but bounded by it. No constraint on visual symmetry. Blades click-mate at adjacent outer ribs to form a corrugated deployed surface; folding is pure rigid-body rotation about the shared pivot.

---

## 0. Locked Decisions

These are binding inputs to every downstream section; do not re-litigate.

| Decision | Value | Implication |
|----------|-------|-------------|
| **Fan architecture** | Discrete V-unit blades with **click-mating panel outer tangential edges** (item #3 panel-edge relocation; the click does NOT live on the rib face — see row 139 below for the full z-lap engagement spec, and §2.2 for the mechanism description) | Eliminates membrane folding; enables true per-blade airfoil optimization |
| **Per-blade structure** | 2 side ribs + 1 wind-generating panel, single rigid piece | One blade = one print, one structural unit; symmetry across blades is exact |
| **Folding mechanism** | Pure rotation about shared pivot; rigid-body blade nesting | No internal blade deformation, no living hinges, no FSI |
| **Inter-blade engagement** | Mating chamfer + detent (default); upgrade to magnetic catch only if cycle testing fails | Self-aligning, FDM-printable, no extra parts |
| **Material** | Single-material PETG except for the pivot pin (steel or brass, ~2.5 g) | Drops multi-material, TPU, AMS toolchanger complexity. The steel/brass pin handles the ~3.6 N·m click-engagement bending that PETG cannot. |
| **Optimization approach** | Four-layer hybrid parameterization: Layer 1 outer envelope + Fourier edge modulation (~14 vars); Layer 2 macro-pattern + procedural math fields with 5-field library (louver, texture, edge, noise-threshold, TPMS), 0-3 active per design (~15-20 vars); Layer 3 capped 0-1 independent primitive (~5-7 vars); Layer 4 manufacturing + click (~3-5 vars). Total ~37-46 vars. | Addresses Mr-Potato-Head alignment, asymmetric-drag access, and OpenCASCADE Boolean reliability. See §6.2.1, §6.2.4, §9.7. |
| **TO scope** | Rib structural TO only (2D plate-bending, Reissner-Mindlin); panel topology is fully driven by Phase 2b's generative parameterization | Density-based TO on the panel and aero-sensitivity-weighted TO were both evaluated and rejected; see §3.1. |
| **What stays preserved (hard constraints in rib TO)** | Rib-panel interface (the rib's junction at y = ±rib_center over the **rib radial band x ∈ [HUB_RADIUS, L_blade − RIB_TIP_TAPER] = [0.020, 0.185] m** per Architectural D / C7 lock) + rib-panel fillet (cyclic mechanical load — the panel-to-rib bending-moment + click-moment transfer flows through here). **NO click-feature footprint in the rib** — the click feature lives on the panel's outer tangential edge in the rib-absent tip region per the item #3 / row 45 relocation. The rib's preserved zones are interface + fillet only, and live only within the rib radial band; the inner 20 mm (HUB region) and outer 15 mm (click region) of each blade are panel-only with no rib material. | Mandatory mechanical constraints; nothing else. Pivot region preserved zone lives in the panel inside the HUB region — see §3.1.2 + below. The dual rib-band lock (C7 inner HUB_RADIUS + Architectural A outer RIB_TIP_TAPER) means the rib is a **bounded middle band**, not a full-length structural element. |
| **Pivot architecture** | Pivot pin runs in +z through the **panel** at y = 0; 10 panels stack on the pin. Ribs do NOT carry pivot holes (they sit at y = ±rib_center, where a single straight z-pin cannot pass through both ribs of one blade). Stack height ≈ 10 × panel_thickness + 2-4 mm spacers ≈ 22-42 mm. | Recovers the ≤50 mm folded-form-factor target. See §2.1, §2.3, §3.1.2, §3.1.5. |
| **Coordinate convention** | Pivot pin axis = world z (stacking direction). Blade planform in XY (200 mm radial along +x, 6-8 mm panel width along ±y). Blade thickness in z. **Wrist rotation axis = +y** (wrist-flexion hinge, perpendicular to the forearm direction +x). SU2 mesh built with **+z toward user**; the **PRODUCTIVE stroke** moves the fan in +z toward the user, which in stationary-fan CFD corresponds to air flowing in **−z** relative to the (stationary) fan: `FREESTREAM_PRODUCTIVE = (0, 0, -1)` / `FREESTREAM_RETURN = (0, 0, +1)` (NOT `AOA` directives — see §9.4). **PITCHING_OMEGA in the rendered SU2 cfg is `(0, -12.5664, 0)` — the NEGATIVE sign on the y-component is part of the locked convention, not just the magnitude** (C11 lock). PITCHING_AMPL = (0, +0.6981, 0). **Kinematic check (locked sign, right-hand-rule convention):** at the SHM instant where the fan is sweeping in +z (productive stroke), the angular velocity vector points in **−y** by the right-hand-rule (curl right-hand fingers from +x to +z; thumb points in −y). So `ω_blade_max = (0, -8.8, 0)` at this instant. For the blade tip at r = (0.25, 0, 0): `v_tip = ω_blade_max × r = (0, -8.8, 0) × (0.25, 0, 0) = (0, 0, +8.8·0.25) = (0, 0, +2.20 m/s)` — the broad face sweeps **in +z** (toward the user), which is the productive stroke. The relative air motion in the (stationary) fan frame is therefore **−z**, matching `FREESTREAM_PRODUCTIVE = (0, 0, -1)`. The earlier draft of this row had the forward proxy at `(0, 0, +1)` and used `ω = (0, +8.8, 0)` — both inverted relative to the right-hand-rule on a productive stroke. The C2 rename + sign correction propagate to §3.2.0, §9.4, §9.4.1, the JSONL schema, and the Jinja2 templates. | All locked CFD numerics (V_tip = 2.20 m/s, Re = 37000, k ≈ 0.57) derive from this convention. See §3.2.0 + §9.4.1. |
| **Rotational-inertia reference axis** | Wrist axis = +y through the handle-grip point at world origin. Pivot pin axis = +z, offset by d_handle = 0.05 m from the wrist axis along +x. CadQuery `i_wrist_about_y` (§6.4) and Spike 0.2 torsional pendulum both measure about the +y wrist axis. Units kg·m². | d_handle offsets the grip from the pivot so I_wrist becomes sensitive to mass placement (mass closer to grip → lower I_wrist); the 20-35% range it adds gives meaningful I_wrist sensitivity. |
| **Mass constraint (C9 relax — was 60 g)** | Total assembly mass m_total < **0.100 kg = 100 g** (hard constraint, not a Pareto objective). The 60 g earlier bound assumed rectangular panel; trapezoidal panel widening (item #35) brings per-blade solid-panel mass to ~20 g × 10 blades = 200 g, requiring ~75% Layer 2/3 carving to fit a 60 g bound — too narrow a feasibility envelope. The 100 g cap admits realistic Layer 2/3 carve rates (40-60%) while still rejecting absurd-mass solutions. | Protects against absurd-mass solutions that happen to be I_wrist-balanced; expands feasibility for trapezoidal panel under #35. |
| **CoM constraint** | r_CoM_wrist ≤ 0.160 m (= d_handle 0.05 m + 0.55·L_blade 0.20 m); same geometric meaning ("CoM no more than 55% of the way out the blade from the wrist") regardless of mass cap. The bound is **distribution-based, not mass-based**, so the C9 mass relaxation does not propagate to r_CoM. | Prevents tip-heavy solutions slipping past I_wrist by being inertia-balanced. |
| **Compute budget** | Colab Pro CPU sessions + Colab Pro GPU (PyFR only); ~600-1100 compute hours expected / 1300 h pessimistic / 300-600 h favorable for Phase 4. **Pre-commit stop rule (locked, single policy ):** at 1000 h with no convergence, K **stays at its current value** within the locked range {3, 4, 5} and the orchestrator force-grows the (Tier 0, Tier 1) overlap by scheduling additional Tier-1 evals on existing Tier-0 points until the K-decision threshold N ≥ 30 is satisfied. **The earlier "drop K from 4 to 2" rule is removed** — K = 2 is outside the locked {3, 4, 5} range, and the force-grow rule already handles the 1000-h deadlock without violating the K range. See §6.2.3. | The favorable case requires K=3 from Phase 3 ρ² ≥ 0.6 AND §6.3.1 prefilter cull rate at the high end. |
| **Compute hardware** | MacBook Pro M3 for local dev + geometry generation + step 59.5 structural FEA + Phase 6 post-processing; Colab Pro CPU (2-4 parallel sessions) for SU2 Tier -1 / Tier 0 / Tier 1; **Colab Pro G4 GPU (95 GB VRAM) for PyFR p=3 top-3 verification only** (T4 is insufficient per HIGH-11 Round-9 lock — 2.5M tets × 100 DOFs/element × fp64 + RK time-stepping buffers + flux storage = 14-18 GB working set; T4 16 GB OOMs, G4 has 5-6× headroom). Tier 1 (~1.5M cells) and Phase 5 verification (~2-3M cells) default to high-RAM CPU. | See Appendix B compute-time table for per-tier hardware and timing. |
| **Multi-objective Pareto** | 4 objectives: maximize J_fan; minimize I_wrist about the handle-grip +y axis; minimize peak pivot stress; minimize folded form factor. Top-3 by Pareto coverage (light corner / knee / heavy corner) are printed and validated in Phase 6. Hard constraints m_total < 100 g and r_CoM_wrist ≤ 0.160 m are NOT objectives. | I_wrist is what the wrist actually feels (τ = I·α); aligns the BO target with Phase 6 IMU `W_cycle = ∫I·ω·dω/dt dt`. |
| **Panel thickness range (4 named control points)** | **3 spline knots at `panel_thickness_t = (t0, t1, t2)`** in **2.2-3.8 mm**. Hard schema bounds: lower `panel_thickness_min ≥ rib_thickness + 2·chamfer_clearance = 2 + 0.2 = 2.2 mm`; upper `panel_thickness_max ≤ 2·rib_thickness − folded_clearance = 2·0.002 − 0.0002 = 3.8 mm`. Four named derivations: **`panel_thickness_pivot = t(pivot_center_x)`** (drives z-stack pitch in the assembled fan), **`panel_thickness_tip = t(L_blade)`** (drives **click-chamfer-depth budget per HIGH-8 Round-9 Option A lock** — the chamfer cuts 0.5-1 mm into the +z and −z corners at the panel's outer tangential edge, NOT a chamfer-z-extent spanning the full panel thickness; see row 139 + §2.2 for full geometry), **`panel_thickness_max = max(t0, t1, t2)`** (drives folded-collision floor), **`panel_thickness_mid = t(L_blade/2)`** (drives J_fan analytical proxies + §3.2.4 corrugation amplitude). The 2.2 mm lower bound gives the click chamfer 0.1 mm of Z-clearance on each side; the "crank" terminology used in earlier drafts referred to a rib-mounted Z-step that is retired under the item #3 panel-edge relocation. | Enables 3D-carved features within the folded-stack collision floor while preserving Z-room for click chamfers. Layer 2/3 carving is excluded from `PANEL_PIVOT_REGION = CircularMask(center=(0.008, 0), radius=0.007)`. The 4-name distinction is mechanical, not a Pareto-axis explosion: the same 3-knot spline produces all four derivations. |
| **Folded form factor target** | 22-42 mm stack thickness at 10 blades. | Recovers the ≤50 mm folded-form-factor target. |
| **Realistic gain target** | 15-30% IMU-normalized J_fan over Spike 0.3 baseline. | Per-blade structural TO 15-30% rib-mass reduction → I_wrist drop; generative panel design 10-25% J_fan; click-feature corrugation 3-8% J_fan. Rotational-inertia savings (not raw mass) flow into the angular-work-per-cycle denominator. |
| **CFD fidelity for optimization** | Multi-fidelity over Tier -1 (2D steady CFD slice, ~10 min/eval — two-eval delta proxy); Tier 0 (3D steady CFD, 30-90 min, ranking-only); Tier 1 (3D unsteady CFD, 3-6 h, true J_fan per §9.4 locked spec). Cost ratios (2, 10, 50). qMFKG over the three tiers. | See §6.2.2 + §6.2.3. |
| **Steady-state proxy** | Two-eval delta: run **PRODUCTIVE** stroke (`FREESTREAM_PRODUCTIVE = (0, 0, -1)`, air flowing -z past a stationary fan that's actually being swept in +z toward the user) and **RETURN** stroke (`FREESTREAM_RETURN = (0, 0, +1)`); `J_fan_steady_proxy = Drag_productive − Drag_return`. A productive louver (slats angled to grab air on the user-ward stroke and feather on the return) scores positive. Symmetric designs score ≈ 0. | The earlier draft labeled `FREESTREAM = (0, 0, +1)` as "forward" — that convention had productive louvers scoring negative and BO would have optimized the wrong sign. C2 rename fixes the inversion. See §9.4.1 Steady-State Proxy. |
| **Cycle count for J_fan integration** | 5 cycles canonical, discard cycle 1 as transient, integrate cycles 2-5 (n = 4 cycles averaged). If cycle-2 vs cycle-3 J_fan differs > 5%, extend to 8 cycles total (discard 2, average 6). **J_fan_se = std/√(n_avg) where n_avg = N_CYCLES − 1** (4 → SE = std/2 at canonical 5 cycles; 6 → SE = std/√6 at extended 8 cycles). **Stored as JSONL `J_fan_cycle_variance` diagnostic only** — NOT fed to the GP as `train_Yvar` (see GP noise model row below). | See §9.4 locked spec + §Phase 3 step 39 decision rule. |
| **Stress-test load case (Architectural C: ω scaling added)** | Add an occasional-peak static load to Phase 2 rib TO and to the Phase 5 step 59.5 combined-blade FEA: **2.5× design-point aero pressure + 2× α_max (220 rad/s²) + 1.41× ω_blade_max (12.4 rad/s vs canonical 8.8)**, applied simultaneously and statically. **Why scale ω:** for SHM at fixed θ_max, doubling α_max (force-driven swing) requires `ω_SHM` to scale by √2, so the peak instantaneous blade angular velocity `ω_blade_max = θ_max · ω_SHM` also scales by √2 ≈ 1.41×. Centrifugal load `F_c = m·ω²·r` therefore **doubles** under stress-test (2× from ω², matching the 2× from α — both stem from the same √2 ω_SHM scaling). The earlier "2.5× p_aero + 2× α_max" spec held ω at the canonical value, which under-estimated centrifugal bearing by 2× under stress-test. Pass criterion: peak σ_VM < 12.4 MPa nominal static (XY at K_tt = 2.42) and < 13.3 MPa (Z bearing), tip deflection < 5 mm, first bending mode > 10 Hz. The **click-detent assertion under stress-test** reads `α · m · r · N = 220 · 0.001 · 0.25 · 10 = 0.55 N` (2× canonical 0.275 N) — overshoots the 0.2-0.4 N budget by 40-175% and triggers a Spike 0.4 magnetic-upgrade decision if Phase 6 testing confirms the load profile. Cyclic per-mode allowables (tension 5.58 / bending 4.22 / bearing 2.00 MPa; §3.1.7) still bind the canonical load. | Represents "user puts their back into it once or twice per session." Designs that pass canonical but fail stress-test are dropped from the Pareto. |
| **Architecture promotion K** | K ∈ {3, 4, 5}. **K is determined by TWO data sources at different phases (H3 lock):** (1) **Seed at Phase 4 launch** comes from **Phase 3's 2D R²** (2D steady ↔ 2D unsteady on the slice): K=3 if Phase 3 R² ≥ 0.6, K=4 if 0.5 ≤ R² < 0.6, K=5 if R² < 0.5. (2) **Running update during Phase 4** uses **Spearman ρ² on the running (Tier 0, Tier 1) 3D overlap**, recomputed every 20 Tier-0 completions; K is updated once N ≥ 30 accumulates in the overlap. The two sources are distinct: Phase 3 R² is the 2D fidelity validity gate (R² ≥ 0.4 floor to retain Tier -1 at all), the Phase 4 running ρ² is the 3D K-decision driver. **N < 30 at the 1000-h trigger fallback:** if cumulative budget hits 1000 h with N < 30 in the overlap, K **stays at its current value** within {3, 4, 5} (no drop to 2) and the orchestrator force-grows the overlap by scheduling Tier-1 evaluations on existing Tier-0 points until N ≥ 30 (typical N=30-N_current Tier-1 evals at 50 cost-units each = ~50-150 extra compute hours). Once N ≥ 30 the K-decision fires normally. | Resolves the deadlock between "N ≥ 30 to decide K" and "1000 h to drop K". |
| **Inner-loop acquisition cap** | 35 TuRBO acquisition rounds per architecture (hard ceiling); early-stop (UCB-improvement < 3% over 5 iters) fires first on convergent architectures. | Worst-case bound; ~10-20% of architectures may marginally improve between 35-50 iterations; the K=4 hedge covers it. |
| **Hypervolume early-stop (4D Pareto)** | After every BO step append the 4D hypervolume; if `(HV[-1] − HV[-50]) / HV_baseline < 1e-4` (i.e., <0.01% of `HV_baseline` cumulative HV gain over the last 50 rounds), declare convergence and reallocate remaining budget to FEA gate top-50 + Phase 5 verification. **Hard floor: 500 rounds** before the check can fire (matches empirical MFBO convergence of 300-600 evaluations for 4D Pareto fronts). **HV_baseline = HV at round 200** (locked; fixed reference, NOT a moving baseline; reference points fixed at Phase 4 launch and never re-computed). The earlier draft used `dHV_per_round < 0.001 · hv_history[-50]` (moving baseline, ~5% per 50 rounds equivalent) which is permissive for 4D MFBO and can trigger early on transient stalls; tightened to `1e-4 · HV_baseline` (~0.01% per 50 rounds) and the floor harmonized to 500. | Threshold pre-registered in `configs/bo_convergence.yaml`. |
| **GP noise model** | **Fixed-floor epistemic noise** (NOT per-observation `train_Yvar`). Solver-noise floor is measured once via replicate Tier-1 runs (same design, perturbed initial conditions) in Spike 0.7b's pre-Phase-4 calibration and locked at `EPISTEMIC_NOISE_FLOOR = max(measured, 1e-6)` for the GP `train_Yvar` constant. **Per-design `J_fan_se` is still measured** and stored as `J_fan_cycle_variance` in the JSONL row for diagnostics, **but is NOT fed to the GP as noise** — physics-driven limit-cycle variance from vortex shedding (5-15% for shed-heavy designs) would otherwise tag the very designs we want to find as low-confidence and weaken the cross-fidelity correlation kernel exactly where asymmetric-drag wins. Tier-specific noise floors are allowed (Tier -1 has higher floor than Tier 1) but each is a *fixed scalar*, not per-observation. | Avoids treating shed-physics signal as measurement noise; see §6.2.3 code block + Spike 0.7b calibration step. |
| **Lock mechanism** | Active locking mechanism is **deferred to V2** (see `docs/V2_backlog.md`). V1 ships with a 3 mm outer-face reinforcement strip on each guard rib + click-feature friction. | If Phase 6 testing shows the fan unlocks under sustained 2 Hz waving, V2 lands a designed lock. |
| **Click-feature footprint (item #3 + rib taper-out + Architectural D HUB_RADIUS — dual rib-band lock)** | 5 × 5 mm patch at the **panel's outer tangential edge at the tip**, in the rib-absent outer band. The rib lives in `x ∈ [HUB_RADIUS, L_blade − RIB_TIP_TAPER] = [0.020, 0.185] m` only — **the inner 20 mm (HUB region) and outer 15 mm (click region) are panel-only**. The click chamfer + detent live in the outer 15 mm panel-only band. Shared constants `CLICK_FOOTPRINT_X_RANGE = (L_blade − 0.010, L_blade)` (last 10 mm — fully inside the rib-absent tip region) and `CLICK_FOOTPRINT_Y_RANGE_PANEL_EDGE = (panel_tangential_outer − 0.005, panel_tangential_outer)` where `panel_tangential_outer = panel_width(r=L_blade)/2 ≈ 0.0225 m` per the #35 widening lock. The chamfer face spans the full panel_thickness in z. Applied to every blade panel's outer tangential edge (inner blades + guard blades alike). **Two new shared constants** in `src/fanopt/geometry/schema.py`: `HUB_RADIUS = 0.020 m` (C7 lock, inner rib boundary) and `RIB_TIP_TAPER = 0.015 m` (Architectural A lock, outer rib boundary). | Panel-edge placement locks adjacent panels into the deployed corrugated surface; rib-mounted clicks were geometrically impossible (z-gap between adjacent ribs is 0.2-1.8 mm under the locked stacking convention) AND would be shielded by the rib's tangential extent even if z were not an issue. The dual rib-band locks make both ends of the blade panel-only: HUB region hosts the 12 mm boss + pivot hole (rib would conflict with boss at r = 0.05, panel_width 11.6 mm < 2·rib_base_width); click region hosts the lap chamfer. The rib is a structural middle band over the radial extent where it's geometrically and mechanically meaningful. |
| **Click design clearance** | 0.15-0.20 mm per mating surface. | Matches Spike 0.4 tolerance-validation range. |
| **Plano-convex envelope (rib-flat print)** | If `print_orientation = rib-flat` (the §7.4.3 default), the Layer 1 envelope is constrained to plano-convex: the bottom face (z = 0 in the build frame) is a single planar surface; all camber + thickness variation goes onto the top face. `deployed-V` orientation relaxes this. | Without the constraint, a 5 mm-camber blade printed rib-flat needs ~4500 mm² of support per blade and post-support Ra ≈ 100-300 µm breaks the §3.2.4 wall-roughness calibration. |
| **Campaign tracker** | Google Drive + per-session JSONL append + pre-sliced round-robin assignment via versioned `slice_assignments_v{N}.json` + content-hashed `current_slice_pointer.txt`. Single-writer GP-acquisition barrier on the M3. Cross-session O_CREAT\|O_EXCL claim sentinel + heartbeat watchdog. | See §Phase 4 step 48, §12.1. |
| **JSONL schema** | One Pydantic-validated row per evaluation with `schema_version`, `design_hash`, `physics_hash`, `material_hash`, `tier`, `status`, `failure_code`, params, mass/CoM/inertia/J_fan/J_fan_delta + per-eval intermediates, stress-test fields, `cfl_max`, retriable + retry_count. | See §Phase 4 step 51 for the full schema. |
| **Compute target line is a favorable case, not a hard ceiling** | Read with the stop rule: at 1000 h cumulative without promoted-arch convergence, K stays at its current value within {3, 4, 5} (force-grow overlap rule above) and the candidate pool narrows top-50 → top-20. **No "K ← max(2, K_required_by_ρ)" path** — K = 2 is outside the locked range. | See §6.2.3 "Honest compute budget." |

---

## Table of Contents

1. [Project Overview: The Folding Fan](#1-project-overview-the-folding-fan)
2. [Folding Fan Geometry and Engineering Constraints](#2-folding-fan-geometry-and-engineering-constraints)
3. [Key Equations and Physical Models](#3-key-equations-and-physical-models)
4. [Software Tools: Easiest Path Analysis](#4-software-tools-easiest-path-analysis)
5. [Claude Code Delegation Map](#5-claude-code-delegation-map)
6. [ML Surrogate Modeling: Core Workflow](#6-ml-surrogate-modeling-core-workflow)
7. [3D Printing Materials Selection](#7-3d-printing-materials-selection)
8. [Step-by-Step Project Execution Plan](#8-step-by-step-project-execution-plan)
9. [Tool Guides and Configuration](#9-tool-guides-and-configuration)
10. [Validation Approaches](#10-validation-approaches)
11. [References and Sources](#11-references-and-sources)
12. [Project Structure and Tooling](#12-project-structure-and-tooling)
13. [Future Work, V2 Backlog, and Baselines](#13-future-work-v2-backlog-and-baselines)

---

## 1. Project Overview: The Folding Fan

### 1.1 What We Are Designing
A folding fan consisting of **N discrete V-unit blades** (default N=10) stacked on a shared pivot pin. Each blade is a **single rigid PETG part** containing 2 side ribs and 1 wind-generating panel between them, printed in one piece. When deployed, adjacent blades' outer tangential panel edges mate via printed click features (chamfer + detent — per item #3 panel-edge relocation, the click lives on the panel edge, NOT on the rib), forming a quasi-continuous corrugated aerodynamic surface (rib ridges separating panel scoops). When folded, blades rotate about the pivot and nest into a V-shaped stack with total folded thickness **22-42 mm at the 10-blade default**, depending on the Pareto-chosen 2.2-3.8 mm panel thickness. Folded form factor is the 4th Pareto objective in §6.4.

> **Footnote (full BO range, MED-10 trimmed):** the architecture-bandit explores n_blades ∈ {8, 10, 12}, so the full Pareto-attainable folded-stack range across the BO is **18-46 mm** (= 8 × 2.2 mm at the light corner to 12 × 3.8 mm at the heavy corner; under the pivot = z convention each blade's Z-extent ≈ panel_thickness ∈ [2.2, 3.8] mm). The ≤50 mm target is recovered across the entire BO range under MED-10. The 14-blade option (would have given 14 × 3.8 = 53.2 mm and 186.2° spread) is removed for ergonomic infeasibility. The design the user picks from the 4D Pareto front determines where in the 18-46 mm range the printed fan lands.

This is fundamentally different from a paddle fan (uchiwa) AND from a traditional sensu/ogi fan:

- **Structural domain:** N rigid PETG blades + 1 pivot pin. Rib TO is one design domain (2D plate-bending via SIMP, §3.1); panel topology is a separate design domain handled by the 4-layer hybrid generative parameterization in Phase 2b (envelope + Fourier modulation + 5-field library of macro-pattern and procedural math fields + capped 0-1 primitive).
- **Aerodynamic domain:** A corrugated rigid surface formed by N panels separated by rib ridges. Surface geometry is driven by the 4-layer parameterization (Layer 1 outer envelope including Fourier-modulated leading/trailing edges; Layer 2 macro-pattern and procedural math fields including TPMS and Perlin/Simplex noise; Layer 3 capped 0-1 primitive). Applied identically to all blades by exact symmetry.
- **Assembly:** N single-material PETG blades + 1 **steel or brass pivot pin** (NOT PETG — per §0 + §2.3 the pivot pin carries a ~3.6 N·m bending moment that a 3 mm PETG pin would fail at σ ≈ 1360 MPa). No fabric, no TPU, no glue, no living hinges, no FSI.
- **Mechanism:** Pure rigid-body rotation of each blade about the pivot pin. Click features on the panel's outer tangential edges engage when adjacent blades reach the deployed angle (per item #3 panel-edge relocation; the rib does not carry click features). Locking mechanism on the guard blades holds the assembly open against any residual click-engagement friction.

### 1.2 Why This Is Harder Than a Paddle Fan
| Aspect | Paddle Fan (uchiwa) | Folding Fan (V-unit blades with 4-layer hybrid generative panel) |
|--------|---------------------|------------------------|
| TO design domain | Single continuous plate | One representative rib (2D plate-bending SIMP), applied to all 20 ribs across all blades by exact symmetry. Panel topology is generated by the 4-layer hybrid (not TO). |
| TO problem size | Large (500K+ elements for full blade) | Modest (one rib ~50-150K plate-bending elements); one TO solve per Pareto candidate |
| TO compute time | 2-4 hours per run | 5-30 minutes per rib |
| ASO design space | Planform, camber, thickness distribution | 4-layer hybrid (~37-46 vars): envelope + Fourier LE/TE, 5-field Layer 2 library (louver, texture, edge, noise-threshold, TPMS) with 0-3 active, capped 0-1 primitive, manufacturing |
| CFD complexity | Standard bluff body | Corrugated rigid surface with Boolean cutouts; CFD resolves rib ridges + Layer 2 features. **No FSI.** |
| Structural failure mode | Distributed bending of plate | Rib bending under aero + inertial load (primary); concentrated stress at the pivot hole (secondary); click-feature fatigue (tertiary) |
| Critical stress location | Blade root (uniform plate) | **Three hotspots under panel-pivot architecture, evaluated independently (no scalar superposition):** (1) **Panel pivot hole** (K_tt = 2.42 at d/w = 3/12 = 0.25 in the 12 mm boss; cyclic tension allowable 5.58 MPa, bearing 2.00 MPa Z-direction — **bearing binds first** for canonical loading). (2) **Rib-panel fillet** (K_t = 1.5; cyclic shear/bending where the rib joins the panel; allowable 9.00 MPa). (3) **Click detent at panel outer edge** (cyclic Z-shear from lap engagement; K_t = 3.0; allowable 2.00 MPa Z-floor — single mode; the prior "4.50 MPa XY" line referred to the rib-mounted click and is retired with the item #3 panel-edge relocation). See §3.1.5 K_t hotspot table for the full set. |
| Assembly | Single print | N single-material PETG print jobs (or one large multi-blade print) + 1 pivot pin |
| Print strategy | Single large flat print | Per-blade prints (default), OR full-assembly print if bed size permits |

### 1.3 The Two Optimization Problems
**Topology Optimization (TO, rib only):** For one representative rib (applied to all blades' ribs by exact symmetry; each blade has 2 identical ribs), determine the optimal 2D plate-bending topology via standard SIMP. TO is NOT applied to the panel — panel topology is generated by Phase 2b's 4-layer hybrid parameterization. Plate-bending (Reissner-Mindlin) captures out-of-plane bending under aerodynamic and inertial loading. See §3.1 and Phase 2. Loads come from CFD-derived pressures (closes Open Question #2). Two earlier drafts that did include panel TO (density-based via SU2+Brinkman; aero-sensitivity-weighted plate-bending) were rejected — see §0 and §3.1.

**Generative Parametric Panel Optimization (4-layer hybrid):** Determine the optimal blade panel topology + overall fan geometry. Design space (~37-46 variables, see §6.2.1 for full table):

- **Layer 1 — Outer envelope + Fourier modulation (~14 vars):** camber spline, twist, thickness profile (**2.2-3.8 mm at 3 control points**, hard schema bound), edge profile category, Fourier LE/TE harmonic amplitudes.
- **Layer 2 — Macro-pattern + procedural math fields (~15-20 vars active, 0-3 fields per design):** library of 5 fields — louver / texture / edge feature / **noise-threshold** / **TPMS**.
- **Layer 3 — Capped 0-1 independent primitive (~5-7 vars):** asymmetric point features (slot/ellipsoid/wedge).
- **Layer 4 — Manufacturing + click features (~3-5 vars):** layer height, print orientation, chamfer/detent/clearance.
- **Plus fan macro (~4 vars):** blade count ∈ {8, 10, 12} categorical (spread angle is **derived** from `blade_count × 13.3°` per C8 lock — NOT a free BO parameter; 14-blade trimmed per MED-10), blade length, base rib width, tip rib width.

Total: **~37-46 design variables**, down from a flat ~45-55 Boolean-primitive list in earlier drafts. The 4-layer structure addresses the "Mr. Potato Head" alignment problem, gives the optimizer direct access to asymmetric-drag design families (louvers), and reduces CAD failure rate from ~5% to <2% (safe-by-construction Layer 2). See §6.2.4 for the full rationale.

### 1.4 User Priorities

1. **Minimize cost** -- student budget; free tools strongly preferred.
2. **Minimize effort** -- where "effort" means the user's manual tool-driving and physical-task time, not compute spend or calendar weeks. Scripted/programmatic workflows preferred; Claude Code handles most of the coding.
3. **Maximize results** -- best achievable fan performance within the above constraints.

---

## 2. Folding Fan Geometry and Engineering Constraints

### 2.1 V-Unit Blade Fan Dimensions
| Parameter | Value | Notes |
|-----------|----------|-------|
| **Blade count** | 10 default; BO range {8, 10, 12} (MED-10 trim: 14 removed for ergonomic infeasibility — 186.2° past straight-line) | Each blade = 2 ribs + 1 panel; tuned in Phase 4 BO |
| **Blade length** | 200 mm | Rib + panel extend full length from pivot to tip |
| **Rib width (at base)** | 12 mm | Tapers to tip |
| **Rib width (at tip)** | 6 mm | Linear taper |
| **Rib thickness** | 2.0 mm | Constrained by FDM minimum feature size |
| **Panel width (between rib pair within a blade)** | 6-8 mm | Per-blade airfoil-shape design variable |
| **Panel thickness** | **2.2-3.8 mm at 3 control points** | Per-element thickness field is set by Phase 2b's Layer 1 envelope generator (§6.2.1, §9.7) and Pareto-traded against I_wrist / folded form factor in §6.4. Hard upper bound 3.8 mm prevents the Z-stacking collision under the pivot = z convention. No TO on the panel — Floor ≥0.4 mm at any local skin for FDM printability. |
| **Blade angular pitch (C8 lock)** | **13.3°** = centerline-to-centerline spacing AND blade tangential width (adjacent blades meet at their tangential edges; no inter-blade gap under the panel-widening lock #35). The earlier "120° fan spread" figure referred to 9 inter-centerline distances × 13.3° (the span from one END blade's centerline to the other END blade's centerline), excluding the half-pitch on each end. | The fan's actual angular extent is **10 × 13.3° = 133.3°** (full 10-blade coverage), not 120°. All downstream consumers (CFD cascade arc-length, J_fan plane sizing, mass distribution, folded silhouette) use 133.3° as the deployed extent. |
| **Deployed fan extent (C8 + MED-10 trim)** | **133.3°** at 10 blades default; BO range adjusts via blade-count axis only **(8 → 106.4°, 10 → 133.0°, 12 → 159.6°)**. The 14-blade option (186.2°, past straight-line ergonomics) is **removed from V1 BO range per MED-10** — a hand fan opened past 180° has end blades behind the wrist axis and is not a practical hand-fan posture. The "spread angle" is NOT a separate BO Pareto axis — it's a derived quantity from blade pitch (locked 13.3°) × blade count. | Earlier drafts treated spread as independent (90-150°) and included 14 blades; under panel-edge meeting at the pitch boundary, spread becomes derived; 14 blades produces 186.2° which is unergonomic and is trimmed from the BO range. |
| **Guard blades (outer 2)** | Same blade pattern + 3 mm outer-face reinforcement strip | Hold fan deployed against residual click-engagement force via the reinforced outer-face strip + click-feature friction; active locking mechanism is a V2 scope item — see `docs/V2_backlog.md` |
| **Pivot hole diameter** | 3 mm | **In the panel at y = 0**, one hole per blade panel (NOT one hole per rib). The pin is steel/brass. d/w = 3/12 = 0.25 in the locked 12 mm-OD boss (the panel is locally thickened to a 12 mm-OD circular boss centered on the pivot hole; without the boss, d/w = 3/8 = 0.375 against the 8 mm inter-rib width, but Layer 2/3 carving cannot reach into the boss per the §9.7.3 PANEL_PIVOT_REGION). |
| **Pivot stack height** | 10 panels × panel_thickness + 2-4 mm spacers ≈ **22-42 mm** | Recovers the ≤50 mm folded-form-factor target without collision. |
| **Pivot location (locked base-relative)** | `pivot_center_x = 0.008 m` from the rib base end (= panel pivot hole center). Boss spans `x ∈ [0.002, 0.014]` (12 mm-OD boss). Shared across all ribs of all blades. The PANEL_PIVOT_REGION circular mask is anchored at this `(pivot_center_x, 0)` and the §9.7 generator reads it as a single named constant — no other coordinate frame (pivot-center-origin, panel-start-origin) is used in production code. |
| **Blade tangential width derivation (item #35 lock)** | At each radial station `r ∈ [d_handle, d_handle + L_blade]`, the blade's tangential span is `tangential_pitch(r) = r · inter_blade_angle_rad = r · 0.232`. At the pivot (r = d_handle = 0.05 m): pitch = 11.6 mm. At the tip (r = 0.25 m): pitch = 58 mm. The blade fully spans its angular pitch: `tip_panel_width(r) = tangential_pitch(r) − 2 · rib_width(r) − 0.5 mm_gap`. At the tip with rib_width_tip = 6 mm: panel_width = 58 − 12 − 0.5 = 45.5 mm (≫ the 6-8 mm inter-rib width near the pivot). The earlier "6-8 mm panel width" was the value at the pivot end only; the panel widens with r so adjacent blades meet tangentially. **Per-blade panel** is a trapezoid (6-8 mm at base, ~45 mm at tip); CFD-relevant J_fan integrates over the trapezoidal area. The mass calc updates accordingly. (Without this widening, the user-reported geometric impossibility holds: 200 mm × 13.3° = 46 mm arc at tip vs the prior 20 mm blade tangential width — 26 mm gap that cannot be bridged by 0.5-1 mm chamfers.) |
| **Click feature: engagement architecture (Option A lock per HIGH-8 Round-9 — 45° chamfered butt joint with detent, friction + detent engagement, NO full-z chamfer face, NO Z-axis overlap)** | The click is a **45° bevel of 0.5-1 mm × 0.5-1 mm** at each panel's outer tangential corner: blade *i*'s panel carries a chamfer cut into its +z face at the outer corner (removing a triangular wedge); blade *i+1*'s panel carries the matching chamfer on its −z face at its outer corner. When the fan is deployed, the two 45° chamfer faces meet at the line `(y = ±panel_tangential_outer, z = z_i + panel_thickness/2)` and form a self-aligning 45° contact. **NO Z-axis overlap is required between adjacent panels** — the chamfered corners touch at a LINE (not a face), and a hemispherical detent bump (0.3-0.5 mm radius) on each chamfer face provides the click engagement via PETG flex during deployment. Force to disengage: 0.5-2 N per click. Engagement is by **friction + detent**, NOT positive Z-lock. **The "lap joint" / "chamfer spans the full panel_thickness" / "panel face overlaps" terminology used in earlier drafts is retired** — adjacent panels do not overlap in z; they meet at a 45° butt joint with detent. **Locked CI:** `tests/test_geometry/test_click_z_lap.py` asserts (a) the chamfer is a small 0.5-1 mm corner bevel (NOT a full-panel-thickness span), (b) chamfer angle is 45° within 1°, (c) no Z-axis overlap between adjacent panels, (d) the chamfer lives at the panel outer tangential edge at `(x = L_blade, y = ±panel_tangential_outer)`. The earlier `test_click_z_contained.py` is deleted in the same commit. | Resolves the geometric impossibility of the prior "Z-axis overlap" lap-joint language. Matches the rest of the spec's friction-engagement intent (Spike 0.4 force-balance criterion `F_friction ≥ 2 · F_inertial`). Fallback if Spike 0.4 shows friction is insufficient: embedded neodymium magnets at the same panel-edge location (~2 g per click pair, ~20 g for 10 blades — within C9 100 g budget). |
| **Click feature: mating chamfer (Option A lock per HIGH-8 Round-9 — small corner bevel, NOT full-z face)** | 45° angle, **0.5-1 mm × 0.5-1 mm bevel** at each panel's outer tangential corner — blade *i*'s panel: chamfer on its +z face at the outer corner; blade *i+1*'s panel: matching chamfer on its −z face at the outer corner. The chamfer is a corner bevel removing a small triangular wedge of material, NOT a face spanning the full panel_thickness. The two adjacent panels' chamfered corners contact at a 45° line at the deployed angle; detent bumps (0.3-0.5 mm radius hemispherical) on each chamfer face provide click engagement. | Chamfered butt joint with detent. Self-aligning during deployment via the 45° contact angle. |
| **Click feature: detent bump** | 0.3-0.5 mm radius hemispherical bump on the panel-edge chamfer face (top +z face of blade *i*'s panel outer tangential edge; bottom −z face of blade *i+1*'s) | Optional retention (graduated to magnetic if cycle test fails) |
| **Click feature: design clearance** | 0.15-0.20 mm per mating surface (z-direction lap clearance, set by FDM Z-axis tolerance) | Compensates for FDM ±0.1 mm tolerance |
| **Click feature: CI regression** | `tests/test_geometry/test_click_chamfer_face.py` — assert the chamfer normal vector on blade *i*'s **panel outer tangential edge** has a +z component (top panel face); blade *i+1*'s matching panel chamfer has a −z component (bottom panel face); both within numerical tolerance. AND `test_click_z_lap.py` (above) for the z-span across panel_thickness. | Catches (a) angular-facing chamfer drift; (b) accidental chamfer relocation back to the rib face. |
| **Panel x-start (panel-pivot architecture)** | `panel_x_start = 0.000 m` from the rib's pivot end | Under panel-pivot architecture the **panel CARRIES the pivot hole** at y = 0, so the panel must extend through `x = 0` (the pivot end) into the pivot region. The panel runs the **full** `x ∈ [0, L_blade] = [0, 0.200 m]`. At `x ∈ [0, 0.020 m]` the panel is locally thickened into the 12 mm-OD circular boss × `panel_thickness` centered at `(pivot_center_x, 0) = (0.008, 0)` (so the boss x-range is `[0.002, 0.014]` for a 12 mm-OD boss centered on the pivot pin at x = 8 mm). The ribs sit at `y = ±rib_center` along the full length. The CadQuery generator (§9.7.1) reads `panel_x_start = 0.0 m` and `pivot_center_x = 0.008 m`. Note: the earlier `panel_x_start = 0.020 m` value was inherited from the pre-panel-pivot rib-pivot architecture (where the panel had to stand off so the rib's pivot hole region was clear) — it is removed under the locked panel-pivot architecture. |
| **Pivot pin length** | ≥45 mm | Includes end terminations |
| **Folded form factor target** | **22-42 mm stack thickness** at 10 blades (= 10 × 2.2-3.8 mm panel-Z + 2-4 mm spacers/end caps; depends on Pareto-chosen panel thickness; see §6.4). The ≤50 mm target is recovered. | 10 blades × **2.2-3.8 mm**/blade (panel Z-extent under pivot = z; ribs are at +Y/−Y edges of the planform and share Z with the panel) — Pareto-trade vs J_fan / I_wrist |

The "rib width" and "panel width" above describe each rib's and panel's *planform-envelope* dimensions. The actual material distribution **within each rib** is determined by Phase 2's plate-bending SIMP TO . The **panel topology** is determined by Phase 2b's 4-layer hybrid generator (envelope + Fourier + macro-pattern/procedural-math fields + capped primitive), not by TO —

### 2.2 Rib Geometry Details

Each rib is a tapered beam **anchored along its full-length rib-panel y-edge** (NOT at a pivot). Under panel-pivot architecture the rib has no pivot hole; bending moments from the panel's aero pressure transfer into the rib along the rib-panel junction at y = ±rib_center. The cross-section at any point along the rib length is the TO design domain.

**Rib-panel BC choice (Phase 2 SIMP TO; Architectural B note):** the rib-panel interface is fixed as a **Dirichlet boundary condition** (all DOFs along the y-edge clamped to zero displacement) against the **smooth-baseline panel placeholder** that the Phase 2 SIMP TO uses internally. This is the **rigid-panel BC** — it assumes the panel is infinitely stiff and turns the rib into a wall-supported plate rather than a cantilever beam. **Limitation:** the real Phase 2b-generated panel (with Layer 2/3 cutouts — TPMS, louvers, noise-threshold) is more compliant than the placeholder, so the Phase 2 rib SIMP TO produces rib material distributions that are slightly **under-built** for the real load (the actual panel transfers less moment because it deflects, so the rib carries more load than the rigid-BC analysis predicted). **Why this is acceptable in V1:** the §59.5 Phase 5 combined-blade structural gate evaluates the full assembly (Phase 2 rib + Phase 2b real panel topology) under the canonical + stress-test loads; any rib design that's under-built for the real panel fails the §59.5 gate and is dropped from the Pareto. The §59.5 gate is the binding correctness check; Phase 2 SIMP TO is screening. **V2 upgrade (queued in §13.3):** replace the Dirichlet BC with an **elastic Winkler-foundation BC** where the rib's z-DOFs at the interface are coupled to a spring stiffness `k_panel` representing the panel's local compliance. Calibrate `k_panel` once from a Phase 0 FEA on a representative skeletal panel.

**Planform envelope (for TO; H12 + Architectural D locks):**
```
Width: w(x) = w_base + (w_tip - w_base) * ((x - HUB_RADIUS) / L_rib)
 w_base = 4 mm at x = HUB_RADIUS (narrow at root)
 w_tip  = 6 mm at x = L_blade - RIB_TIP_TAPER (wider at tip)
 HUB_RADIUS = 0.020 m, L_blade = 0.200 m, RIB_TIP_TAPER = 0.015 m
 Rib radial extent: x ∈ [HUB_RADIUS, L_blade - RIB_TIP_TAPER] = [0.020, 0.185] m
 Rib radial length: L_rib = L_blade - HUB_RADIUS - RIB_TIP_TAPER = 0.165 m

Thickness: h = 2.0 mm (uniform, constrained by FDM layer count)

The TO domain is the 2D tapered planform (length x width) over the rib
radial band only. The inner 20 mm (HUB region) and outer 15 mm (click region)
are panel-only with no rib material.

Up-taper direction (narrow root → wider tip): geometrically motivated by the
widening angular pitch (tangential space available for the rib grows with r);
structurally acceptable because the rib base at r = HUB_RADIUS = 0.020 m has
a short cantilever lever arm to the pivot, so root bending isn't as critical
as it would be for a full-length rib reaching r = 0.

At constant 2 mm thickness, TO determines which portions of the planform are
material vs. void (creating cutout patterns and lightening holes).
See §9.2 for why 2D plate-bending TO (Reissner-Mindlin) is preferred over
3D voxel or 2D plane-stress for this rib thickness.
```

**Pivot region (panel-pivot architecture):** 10 panels stack on the pivot pin along +z; each panel carries a **single 3 mm pivot hole centered at y = 0 on the rib-pair midplane**. The ribs do **NOT** carry pivot holes — under the locked convention (pivot pin = +z, ribs at y = ±rib_center) a single straight z-pin cannot pass through both ribs of one blade. The 8 mm inter-rib panel is locally thickened to a **12 mm-OD circular boss × panel_thickness** centered on the pivot hole (e/d = 2.0 satisfies the pin-tear-out minimum; see §6.3.1 Filter 2). See §2.1 dimensions table, §2.3 blade-assembly, and §3.1.2 preserved-zones for the full architecture spec.
- **The rib's preserved zone is the rib-panel interface** (rib's full-length junction with the panel at y = ±rib_center, where bending moments from aero pressure transfer between panel and rib). Plus the **rib-panel fillet**: a 2 mm × 2 mm cross-section strip running the full rib length at the fillet location, density-clamped to ρ = 1 in the rib SIMP TO; carries the ~0.18 N·m per-rib click-mate moment. §N7 check 15 requires fillet radius ≥ 1.0 mm.
- The pivot-pin-bearing stress concentration (K_tt = 2.42 tension at d/w = 0.25 in the 12 mm boss; K_t_bearing = 1.5 Z-direction) lives in the **panel** per §3.1.5; the rib itself sees its hotspots at the rib-panel fillet (K_t_fillet ≈ 1.5 for ≥ 1 mm fillet) and at slot ends / lock-mech corners (case-by-case K_t).
- The rib must transition from the rib-panel interface to the tapered body smoothly (§N7 check 15 — fillet ≥ 1.0 mm enforces this).
- **CAD-template rename map:** `rib_pivot_hole_diameter` → `panel_pivot_hole_diameter`; any "bottom-of-rib hole" geometry in the §9.7 CadQuery generator / Fusion templates is removed; the generator reads `PANEL_PIVOT_REGION` + drills one panel pivot hole per blade at y = 0 + extrudes the 12 mm-OD boss in `make_panel_solid`. A `tests/test_geometry/test_no_rib_pivot_hole.py` regression check asserts no `rib_pivot_hole_*` parameter remains in `src/fanopt/geometry/schema.py`.

**Guard blade (outer blade) differences:**
- Same airfoil geometry as inner blades but with a **3 mm outer-face reinforcement strip** on the outermost rib .
- **Two rib classes — `inner` and `guard` (paradox fix;.** A single SIMP solve cannot simultaneously enforce mirror symmetry across the inner↔guard interface AND a guard-only locking-mechanism preserved zone — the two constraints contradict. splits the rib TO into **two independent solves**: one for the inner-rib class (8 inner blades × 2 ribs each, mirror-symmetric within the class), one for the guard-rib class (2 guard blades × 2 ribs each, mirror-symmetric within the class but carrying the locking-mechanism preserved zone + the ~3 mm outer-face reinforcement strip). The two solves use independent forbidden-zone masks. Cost ~2× rib-TO time per design — negligible vs CFD. New JSONL schema field `rib_class ∈ {inner, guard}` groups results in post-hoc analysis.
- **V1 lock policy (per §0 lock):** V1 ships **without an active locking mechanism**. The 3 mm outer-face reinforcement strip + click-feature friction holds the fan deployed against the cumulative click-engagement friction across 9 inter-blade pairs during waving. An active locking mechanism is deferred to V2 (see `docs/V2_backlog.md`); V2 will land a designed lock only if Phase 6 testing shows the fan unlocks under sustained 2 Hz waving.

### 2.3 V-Unit Blade Assembly and Click Mechanism
The fan uses discrete one-piece rigid blades. Each blade is a structural unit; the deployed fan is a corrugated surface formed by the click-engagement of adjacent blades; the folded fan is a rigid-body nested stack.

**Blade structure:**

Each blade is a single rigid PETG part containing:
- Inner rib (closer to the next blade in the deployed-counter-clockwise direction): **165 mm × 4-6 mm tapered (narrow at root, wider at tip per H12) × 2 mm thick** (rib lives in `x ∈ [HUB_RADIUS, L_blade − RIB_TIP_TAPER] = [0.020, 0.185] m` per Architectural D / C7 lock; the inner 20 mm and outer 15 mm of the blade are panel-only).
- Outer rib (closer to the previous blade): **same dimensions: 165 mm × 4-6 mm tapered × 2 mm thick.** No click features on the rib — the click chamfer + detent live on the panel's outer tangential edge in the rib-absent tip region (x ∈ [0.185, 0.200]); the boss + pivot hole live in the rib-absent root region (x ∈ [0, 0.020]).
- Panel: **200 mm full L_blade × trapezoidal tangential width × 2.2-3.8 mm thickness at 3 control points** (H13 panel-width formula): `panel_width(r) = r · 0.232 − 2·rib_width(r) − 0.5 mm gap`, valid for `r ∈ [HUB_RADIUS, L_blade − RIB_TIP_TAPER] = [0.020, 0.185 m]` (the rib-present band). Outside this band the panel runs the full angular pitch `r · 0.232` with no ribs and no inter-rib gap (the inner 20 mm hosts the 12 mm boss; the outer 15 mm hosts the click chamfer at the panel's outer tangential edge). Locally widened to 12 mm-OD at the pivot boss centered on x = 0.008 m. Panel topology and porosity come from Phase 2b's 4-layer hybrid generator (§6.2.1, §9.7), not from SIMP TO. - **Panel pivot hole (— one hole per blade panel at y = 0):** 3 mm diameter through-hole at the panel's pivot-region center (x = 0.008 m, y = 0). The pivot pin runs through this hole and stacks 10 panels along +z. **The ribs do NOT carry pivot holes** — they are at y = ±rib_center (different y from the pin) and would not align onto a single straight z-pin even if they did. The panel pivot hole is the critical-stress location at K_tt = 2.42 (d/w = 0.25 in the 12 mm boss); cyclic-fatigue allowables 5.58 MPa tension / 4.22 MPa bending / 2.00 MPa bearing (Z) per §3.1.5 — bearing binds first under canonical loading. Panel-pivot region preserved zone (`PANEL_PIVOT_REGION = CircularMask(center=(0.008, 0), radius=0.007)` in `schema.py`): 7 mm-radius circular keep-out around the pivot pin centerline (covers the full 12 mm-OD boss + 1 mm clearance), excluded from Layer 2/3 generator carving by the §9.7.1 Step 0 panel-domain mask.
- All three parts joined into one rigid body. No internal joints, no living hinges, no flexible elements.

**Pivot assembly:**

**Architecture: pin-through-panel.** Under the locked convention (pivot pin = +z, ribs at y = ±rib_center), a single straight z-pin cannot pass through both ribs of one blade — they sit at *different* y positions. The pin passes through the **panel** at y = 0 instead, one pivot hole per blade panel; 10 panels stack on the pin. Consequences (propagated to §2.3, §3.1.2, §3.1.5, §3.1.7, §6.4):
 - **Pivot stack height: 10 panels × panel_thickness + spacers ≈ 22-42 mm.**
 - The **pivot hole is in the panel, not in the rib** (§3.1.2 preserved-zone moves from rib to panel).
 - §3.1.5 K_t: d/w = 3/12 = 0.25 (12 mm boss centered on the 3 mm pin hole) → **K_tt = 2.42** from the Peterson polynomial. Cyclic-fatigue tension allowable: 0.30 · 45 / 2.42 ≈ **5.58 MPa** nominal; bending 4.22 MPa; bearing 2.00 MPa (Z, σ_y_Z = 30 MPa).
 - The rib's stress concentration is the **rib-panel interface** (rib joins the panel along the rib's full length at y = ±rib_center). The §3.1 plate-bending TO is unchanged — the rib bends under aero pressure transmitted from the panel — but the SIMP preserved-zone mask anchors at the rib-panel interface, not at a rib pivot hole.

**Pivot pin material:** 3 mm steel or brass rod, length ≥45 mm. **Pin bending lever arm correction (item #33 lock):** the bending moment on the pin is `F_click_total · pin_z_extent`, NOT `F_click_total · L_blade`. The pin's transverse loading is the cumulative click-engagement reaction from all 10 panels stacked on it, applied along the pin's Z-extent (22-42 mm stack height depending on panel_thickness). Worst-case: pin = simply-supported beam between its end terminations, distributed transverse load = 9 inter-blade pair clicks × ~2 N peak each ≈ 18 N total, lever arm = stack_height / 2 ≈ 21 mm. Resulting peak bending moment `M_pin ≈ F · L_eff = 18 · 0.021 / 4 ≈ 0.09 N·m` (simply-supported under distributed load). For 3 mm pin: `σ_pin ≈ 32·M / (π·d³) = 32·0.09 / (π·0.003³) ≈ 34 MPa`. **A 3 mm PETG pin at 34 MPa is below the 45 MPa yield but well above the 13.5 MPa fatigue allowable (0.30·σ_y_XY); steel/brass at >200 MPa yield carries the load with >5× safety margin.** The earlier 3.6 N·m / 1360 MPa numbers used `L_blade = 200 mm` as the lever arm, which over-stated the pin bending by ~40× — the steel/brass requirement still binds at the corrected magnitude (PETG would fail in fatigue regardless), but the absolute number was incorrect. The §0 "single-material PETG throughout" rule reads as **"single-material PETG except the pivot pin (steel/brass, ~2.5 g, ~4% of mass budget)."** The non-PETG density of the pin is incorporated in the `i_wrist_assembly` mass-weighted CoM and I_wrist calculation (§6.4 `RHO_PIN ≈ 7850 kg/m³` for steel or 8500 kg/m³ for brass, vs PETG 1270 kg/m³).
3. The two outermost blades are the **guard blades** — same airfoil geometry but with a **3 mm outer-face reinforcement strip** on their outer ribs .
4. **Spacers are FORBIDDEN under the Z-lap click architecture (M18 lock):** the click chamfer + detent on the panel's outer tangential edge engages adjacent panels in a Z-lap with `chamfer_clearance = 0.1 mm per side` (§0 row 33). Any inter-panel Z-pitch addition (a 0.2 mm spacer doubles the chamfer-to-chamfer gap from 0.2 mm design clearance to 0.4 mm actual) would **fully disengage** the click chamfers. The pin stack runs through 10 panels directly under panel-pivot architecture; PETG-on-PETG slip at the panel-panel contact is acceptable at the loads involved. **If Spike 0.4 reveals friction issues during fold/deploy cycles**, the mitigation is (a) tightening the pivot-pin nut to increase axial preload (raises slip threshold), or (b) applying a dry PTFE lubricant to the panel-panel contact faces (reduces μ without altering Z-pitch). The earlier "0.2 mm printed PETG washers between adjacent panels" fallback is retired.

**Click-mating mechanism (inter-blade engagement) — see §0 row 139 panel-edge z-lap lock:**

Adjacent blades engage via **z-direction lap engagement at the panel's outer tangential edge** (NOT on the rib face — rib-mounted z-lap is **geometrically impossible** per §0 row 139: chamfers cannot bridge the 0.2-1.8 mm inter-rib z-gap while staying within the rib's ±1 mm z-extent. Earlier drafts described a rib-mounted "structural crank" — that architecture is retired and the crank concept is gone from V1.). The click is integral to the panel — no separate hardware required.

- **Mating chamfer (locked per item #3 panel-edge relocation + HIGH-8 Round-9 Option A lock):** 45° chamfer of **0.5-1 mm × 0.5-1 mm** at each panel's outer tangential corner. Blade *i*'s panel carries a chamfer cut into its +z face along the outer tangential edge (a small triangular wedge of material removed from the top-outer corner); blade *i+1*'s panel carries the matching chamfer on its −z face along the same tangential boundary (wedge removed from the bottom-outer corner). When the fan is deployed, blade *i+1*'s panel rotates over blade *i*'s panel; the two 45° chamfer faces meet at the line `(y = ±panel_tangential_outer, z = z_i + panel_thickness/2 = z_{i+1} − panel_thickness/2)` and contact at a 45° angle. **No Z-axis overlap is required** — adjacent panels' outer-edge corners touch at a single line; the engagement holds via the 45° contact friction plus the detent bump (next bullet). Self-aligning during deployment. The earlier "chamfer face spans the full panel_thickness in z" / "+z panel face overlaps blade i+1's −z panel face" terminology is retired (Option A lock).
- **Detent bump (default):** 0.3-0.5 mm radius hemispherical bump on the panel's outer tangential edge chamfer face, with matching depression on the mating panel's chamfer. Force to disengage ~0.5-2 N.
- **Design clearance:** `chamfer_clearance = 0.1 mm per side` per §0 row 33 (per-face design clearance). The 0.15-0.20 mm figure referenced in §3.2.2 is the per-face manufacturing clearance including FDM ±0.1 mm tolerance stack — see §3.2.2 line 519 for the full reconciliation.
- **CI gate:** `tests/test_geometry/test_click_z_lap.py` asserts every click chamfer spans `[z_i − ε, z_i + panel_thickness/2 + ε]` where ε is the 0.1 mm chamfer clearance.
- **Fallback if Spike 0.4 shows detent cycle life is insufficient:** upgrade to small embedded neodymium magnets (~1-2 g per click pair, ~20-40 g total for 10 blades; within the C9 100 g mass constraint).

**Folding behavior (no internal deformation):**

When the user closes the fan, each blade rotates about the pivot pin as a rigid body. There is no elastic bending, no spring-back energy stored, no living-hinge stress. The folded form factor is set by:
- **Per-blade Z-extent under convention (pivot = z, blade in XY, thickness in Z):** **2.2-3.8 mm** in z (the panel thickness; the 2 ribs of one blade are at +Y/−Y *edges* of the panel in the XY planform, NOT spaced in Z). The clamp ensures the per-blade Z-extent never exceeds 2·rib_thickness − 0.2 mm so adjacent panels don't collide when folded.
- **Folded stack:** **10 blades × 2.2-3.8 mm in z = 22-38 mm total folded thickness** + **~1-3 mm of end caps only (no inter-panel spacers per M18 lock)** ≈ **23-41 mm overall**. **The ≤50 mm target is recovered.** Folded form factor remains a 4th Pareto objective for designs that push panel toward 3.8 mm vs designs that stay near 2.2 mm.
- Click features disengage during folding (chamfer geometry permits clean separation).

**Engineering decisions logged here:**
- **Decision:** Each blade is one rigid PETG piece; no compliance, no FSI.
- **Reason:** A rigid printable corrugation is fabricable in one shot from a single PETG print, removes FSI from the design loop entirely, and eliminates spring-back failure modes. No compliant-panel sub-study.
- **Decision:** Click features are chamfer + detent printed integral to each **panel's outer tangential edge** (NOT on the rib — see panel-edge z-lap lock above), with magnetic upgrade gated on Spike 0.4.
- **Reason:** FDM-printable, self-aligning, no separate hardware. Magnetic upgrade is a known-good fallback if printed detent cycle life is inadequate.
- **Decision:** Guard-blade locking mechanism is **DEFERRED TO V2**. V1 relies on the 3 mm outer-face reinforcement strip + click-feature friction to hold the fan deployed under the residual click-engagement force during waving. If Phase 6 testing shows the fan unlocks under sustained 2 Hz waving, the V2 backlog (`docs/V2_backlog.md`) is the venue for the lock design.
- **Reason:** Cumulative click-engagement force across 9 inter-blade pairs is non-negligible during high-amplitude waving; a deferred V2 lock is the venue if Phase 6 testing fails.

### 2.4 Structural Loading on Each Blade
Each blade (treated as a single rigid unit comprising 2 ribs + 1 panel) experiences four load types.

**1. Aerodynamic pressure (distributed on the panel surface):**
- Direct surface pressure on the panel; not transmitted through any intermediate compliant layer.
- Each blade's panel carries the full local pressure over its area (no inter-blade load transfer, because adjacent blades are mechanically independent rigid bodies pinned only at the pivot and engaged only at the deployed click position).
- Pressure magnitude: 5-20 Pa at peak waving velocity (~2.5 m/s tip speed).
- **Aero pressure sourcing:** pressure profile comes from Phase 3 baseline CFD on the corrugated rigid geometry. Initial estimates (10 Pa uniform) bootstrap Phase 2 if Phase 3 has not yet completed; one iteration with CFD-derived loads is required if the initial estimate diverges by >10%.
- **Multi-load-case requirement:** Phase 2 rib TO must run **both** peak-positive (push-stroke) and peak-negative (return-stroke) pressure load cases. Pressures are not symmetric in magnitude (push stroke moves more air than return stroke), but the structural design must withstand both extremes within fatigue allowables.

**2. Inertial forces (distributed):**
- Each rib + panel mass oscillates about the pivot axis.
- **Corrected peak angular acceleration:** For SHM at 2 Hz with 40° amplitude, α_max = ω² · θ_max = (2π·2)² · 0.7 ≈ **110 rad/s²**.
- **Inertial loads use wrist-relative r about the +y WRIST axis, NOT the +z PIN axis** (consistent with §6.4's wrist-axis I_wrist). The wrist axis is +y through world origin; the pivot pin runs in +z through the panel at y = 0. **These are two different axes.** The inertial moment arm of a rib mass at `(x_rib, y_rib, z_rib)` under wrist (+y) rotation is `r_wrist = √(x_rib² + z_rib²)`. **Lock:** every `r·α·m` calculation in the structural chain reads `r` from the **wrist axis (+y)**, not from the pin axis (+z). The pin axis is offset by d_handle = 0.05 m in +x from the wrist axis (the pivot at (0.05, 0, 0) vs the wrist origin at (0,0,0)); the perpendicular distance from a rib mass element at (x_rib, y_rib, z_rib) to the +y wrist axis is √(x_rib² + z_rib²), dominated by x_rib (the radial blade length, 0.05-0.25 m) since blade thickness in z is small (≤3.8 mm). At the rib tip x_rib = 0.25 m, r_wrist ≈ 0.25 m; at the pivot end x_rib = 0.05 m, r_wrist ≈ 0.05 m = d_handle.
- **Peak rib-tip acceleration sanity table:** at α_max = 110 rad/s², `a_tip = α · r_wrist_tip = 110 · 0.25 = 27.5 m/s² = 2.8 g`. At the rib midspan (r ≈ 0.15 m): 16.5 m/s² = 1.7 g. At the rib base near the pivot (r ≈ 0.05 m = d_handle): 5.5 m/s² = 0.56 g. These values feed the Phase 2 inertial-load multi-load-case and are recorded in `material_locks.alpha_max_kinematics_sanity` for traceability.
- A 5-gram blade with centroid at 150 mm from the **wrist axis** (= d_handle 50 mm + half-blade 100 mm) experiences F = m·α·r = 0.005 · 110 · 0.15 ≈ **0.083 N** inertial force. A 2-gram rib at 150 mm from the wrist: ≈ **0.033 N**. (Earlier drafts reported 0.055 N / 0.022 N using r = 100 mm measured from the pivot pin, which under-sized the inertial load by ~33% because they omitted the d_handle = 50 mm offset. The Phase 2 `rib_to_solver.py` step 11 inertial-load build, the Phase 5 step 59.5 combined-blade FEA loads, and any other r·α·m calculation in the structural chain must use wrist-relative r.)
- Small compared to aerodynamic load but contributes to fatigue and is the dominant load source at the pivot during direction reversal.
- α_max ≈ 110 rad/s² sets the inertial load magnitudes; downstream values in Phase 2 TO load case (b) and the pivot fatigue calculation in §3.1.5 recompute from the corrected number.

**3. Pivot reaction (concentrated at base):**
- The pivot pin provides a fixed support (moment and shear reaction).
- All loads along the blade resolve to a reaction force and moment at the pivot.
- The 3 mm pivot hole in the panel at y = 0 (one per blade, NOT in the ribs — see §2.1 / §2.2 panel-pivot architecture) is the stress-concentration / failure-prone location.
- Peak bending moment at pivot: M = ∫ q(x) · x dx over the blade.

**4. Click-engagement force (distributed along the panel's outer tangential edge, cyclic; relocated from rib face per item #3):**
- Each time the fan deploys, the outer tangential edge of each blade's panel engages with the next blade's panel outer tangential edge via the click feature (the chamfer + detent live on the panel edge, not on the rib face — see §2.2 click-mating mechanism for the full chamfered-butt-joint + detent engagement spec per HIGH-8 Option A lock).
- Force magnitude: ~0.5-2 N per click event per inter-blade pair (depending on detent geometry; characterized in Spike 0.4).
- Cyclic frequency: once per deploy/fold cycle (~few times per day in typical use; not at the 2 Hz waving frequency).
- Adds a small fatigue-relevant load to the click-feature region of the outer **panel** (per the item #3 relocation; the click is on the panel's outer tangential edge at the tip, NOT on the rib). **Locked location:** `CLICK_FOOTPRINT_X_RANGE = (L_blade − 0.010, L_blade)` × `CLICK_FOOTPRINT_Y_RANGE_PANEL_EDGE` (see §3.1.2b). The Phase 2b 4-layer panel generator (§9.7 `make_panel_solid`) writes the chamfer + detent into this region; the §9.7.3 Check 7 exclusion band prevents Layer 2/3 carving from removing it. **Phase 2 rib TO does NOT carry a click preserved zone** per §3.1.2a — the rib has interface + fillet preserved zones only.

**5. Blade-blade aerodynamic interaction:**
- Adjacent blades shed wakes into each other during waving.
- Not a structural load per se; relevant only to CFD modeling (mesh must resolve inter-blade gaps; CFD post-processor includes all blade surface forces). See §2.5.
- Blade symmetry is exact (no membrane coupling), so per-blade TO requires no sensitivity check.

### 2.5 Deployed Fan Geometry for CFD
When deployed at **133.3°** with the default N=10 blades (C8 lock: 10 × 13.3° pitch — adjacent blades meet at their tangential edges; the earlier "120° spread" was the span between end-blade centerlines only, excluding the half-pitch on each end), the fan forms a corrugated rigid sector:
- 10 blades × 13.3° angular pitch = **133.3° total extent**. Each blade fully spans its 13.3° pitch tangentially; adjacent blades meet edge-to-edge (no inter-blade gap).
- 200 mm blade length; each blade has 2 ribs (~2 mm thick each) + 1 panel (2.2-3.8 mm) between them.
- Adjacent blades' outer tangential panel edges mate at click features when deployed (per item #3 panel-edge relocation); the deployed surface is a quasi-continuous corrugated geometry with rib ridges separating panel scoops.
- **No flexible membrane, no porous surface, no gaps to model.**

**CFD modeling approaches for the corrugated rigid geometry:**

1. **Calibrated roughness model (Phase 4 BO inner loop, ELIGIBLE LAYER 2 FIELDS ONLY):** treat the rib-ridge surface corrugation + Layer 2 {texture, edge feature} surface features as a wall-roughness term in the boundary layer. Calibrated against fully-resolved corrugation in the Phase 3 2D unsteady slice (R² ≥ 0.4 hard gate). **NOT applicable to** Layer 2 {louver, noise-threshold, TPMS} — these are **through-flow features** (slits, perforations, lattice channels) that the calibrated roughness model would silently smear into surface roughness, destroying the asymmetric-drag physics the optimizer is supposed to exploit.
2. **Resolved-through-flow mesh (Phase 4 + Phase 5; activates per-design):** for any design whose Layer 2 activation profile includes louver, noise-threshold, or TPMS, the Tier 0 and Tier 1 CFD configs **switch the mesh template** to fully resolve those features (~1.5-3 M cells). The per-design switch is determined at mesh-generation time: `if any(layer2.activations & {louver, noise, TPMS}): mesh = mesh_3d_resolved else: mesh = mesh_3d_roughness`. The two mesh modes share boundary-condition marker names per §9.6 so the SU2 cfg template is unchanged.
3. **Smooth-sector approximation (Phase 2b LHS seeding only):** treat the deployed fan as a smooth curved plate for the cheap initial samples. Replaced by approach 1 or 2 once the BO promotes designs into the inner loop.

**Layer 2 field classification (locked for Phase 4 mesh routing):**
| Field | Class | CFD treatment at Tier 0/1 |
|-------|-------|----------------------------|
| Louver | through-flow | resolved mesh |
| Noise threshold | through-flow | resolved mesh |
| TPMS | through-flow | resolved mesh |
| Texture | roughness-eligible | roughness model |
| Edge feature | roughness-eligible | roughness model |

**Blade-blade aerodynamic interaction:** the mesh must resolve inter-blade gaps where adjacent blades' wakes shed into each other. The CFD post-processor (canonical `j_fan.py`) integrates surface forces over all blades.

---

## 3. Key Equations and Physical Models

### 3.1 Structural Mechanics — Rib-Only Plate-Bending TO
**narrows TO scope to ribs only.** Panel topology is generated by Phase 2b's rich parametric Boolean-subtraction pipeline, not by SIMP density variables. Two earlier drafts that did try density-based TO on the panel were rejected (see §0 for the full rationale); the short version:

1. **Density-based TO with SU2+Brinkman penalization** was rejected because steady CFD cannot capture asymmetric drag (the unsteady mechanism that makes hand fans work), because Brinkman + adjoint AD is research-level effort, and because grey-fluid artifacts vanish on binarization.
2. **Aero-sensitivity-weighted plate-bending TO with 2.5D skin breakthroughs** was rejected because the linearization of J_fan around a baseline is only valid for small material perturbations (large TO changes — the whole point — exit the linear regime), and because mixing TO with aero on the panel entangled the structural and aerodynamic problems in a way BoTorch's multi-fidelity machinery cannot cleanly decouple.

The split: **rib TO via SIMP** (well-understood, no aero coupling) + **panel topology via parametric generative design** (BO finds the best subtraction pattern, BO sees real unsteady CFD physics).

**Formulation:** plate-bending (Reissner-Mindlin), not plane-stress. The rib's primary load is aerodynamic pressure normal to the panel surface, which produces out-of-plane bending of the rib. Plane-stress captures in-plane loading only and would minimize the wrong compliance metric. FEniCSx supports plate-bending via mixed function spaces (rotations + transverse displacement).

#### 3.1.1 SIMP Formulation for the Rib (Plate-Bending)

Each finite element in the **rib design domain only** (not the panel) is assigned a pseudo-density `rho_e` in [0, 1] (0 = void, 1 = solid material). The rib has constant thickness 2 mm; only the planform density distribution varies.

**Plate stiffness interpolation:**

```
D(rho_e) = E_eff(rho_e) · t^3 / (12 · (1 - nu^2))
E_eff(rho_e) = E_min + rho_e^p · (E_0 - E_min)
```

Where:
- `E_0` = 1300 MPa (FDM PETG XY); `E_min` = 1e-9 · E_0; `p` = 3 (SIMP penalty).
- `t` = 2 mm (rib thickness, constant).
- `nu` = 0.38; element area `v_e`; material density `ρ_mat` = 1.27 g/cm³.

**Compliance minimization (plate-bending, multi-load-case for push/return strokes):**

```
minimize: C(rho) = sum_{lc in {push, return}} w_lc · F_lc^T · u_lc
 = sum_{lc} w_lc · sum_e (rho_e^p · u_{e,lc}^T · K_{0,e}^{bend} · u_{e,lc})
subject to: K^{bend}(rho) · u_lc = F_lc (equilibrium, each load case)
 sum_e (rho_e · v_e) <= V* (volume constraint; V* ~ 0.4)
 0 < rho_min <= rho_e <= 1 (bounds)
 rho_e = 1 for rib-panel interface (preserved zone — full-length y = ±rib_center strip)
 rho_e = 1 for rib-panel fillet (preserved zone — 2×2 mm strip at the fillet)
 # NO rib click-footprint preserved zone — the click lives on the panel's outer
 # tangential edge per the item #3 relocation. The Phase 2b generator's
 # make_panel_solid step writes the chamfer + detent into the panel; the rib SIMP
 # TO does not allocate material for a click feature on the rib.
 sigma_vm(rho, u_lc) <= sigma_allow at rib-panel fillet (stress constraint, p-norm aggregated;
                                                        sigma_allow = 9.00 MPa from §10.1 K_t table)
 u_tip_max <= 0.005 · L = 1 mm at 200 mm (rigid-blade gating)
 feature size >= 0.8 mm (manufacturability; 2× nozzle diameter)
```

Where:
- `K^{bend}` = plate-bending stiffness matrix (Reissner-Mindlin element).
- `w_lc` = per-load-case weights (default w_push = w_return = 0.5).
- `V*` = target volume fraction (default 0.4; range 0.3-0.5).
- `u_tip_max` = blade tip deflection under combined peak load; constraint enforces rigid-blade assumption per . If violated, document the failure (have no Phase 2d escape route). **Necessary but not sufficient:** the Phase 2 rib-only check uses a "smooth-baseline panel placeholder" and does NOT see the §9.7 generator's Layer 2 cutouts (TPMS, noise, louvers) which carve the actual panel. A blade that passes Phase 2 u_tip_max < 1 mm with the solid placeholder can still flex 5-15 mm under aero load once Phase 2b's skeletal panel topology is applied. **The binding gate is the Phase 5 combined-blade structural check (step 59.5) on the top-3 Pareto designs.**

**Sensitivity (standard SIMP gradient, plate-bending):**

```
dC/d(rho_e) = -p · rho_e^(p-1) · sum_{lc} w_lc · u_{e,lc}^T · K_{0,e}^{bend} · u_{e,lc}
```

Filter radius `r_min` = 1.5 mm (Helmholtz-PDE filter). OC update for SIMP convergence.

#### 3.1.2 What Is Preserved (Mandatory Mechanical Constraints)

Two regions remain hard-preserved (`rho_e = 1`):

- **Pivot region (panel-pivot architecture):** **Circular boss footprint of radius `PIVOT_BOSS_RADIUS + clearance = 0.006 m + 0.001 m = 0.007 m` centered on `(pivot_center_x, 0) = (0.008 m, 0)` in the PANEL at y = 0** (the panel widens locally from the 6-8 mm inter-rib width to the 12 mm-OD circular boss × `panel_thickness`). Mechanically mandatory: the pivot hole is the highest-stress location in the *panel* (K_t ≈ **2.42** at d/w = 3/12 = 0.25 in the 12 mm boss from Peterson polynomial); any panel topology that removes material here violates the fatigue allowable. **Important — Phase 2 TO scope under the panel-pivot architecture lock:** the rib SIMP TO no longer carries a pivot preserved-zone (rib has no pivot hole); the rib's preserved zone is at the **rib-panel interface** (the rib's full-length junction with the panel at y = ±rib_center, where bending moments from aero pressure transfer between panel and rib). The panel is generated by the §9.7 Phase 2b 4-layer generator (NOT SIMP TO), so the panel pivot region is enforced by the §9.7.3 manufacturability filter as a hard parameter bound: `CLICK_FOOTPRINT_X_RANGE` already locks the click-feature X position (last 10 mm before L_blade); paired with `CLICK_FOOTPRINT_Y_RANGE_PANEL_EDGE` it defines the panel-edge click footprint per §9.7.3 Check 7 (the earlier "at the rib tip" phrasing is retired — the click feature is at the panel's outer tangential edge per the item #3 relocation). A new shared constant **`PANEL_PIVOT_REGION`** is added to `src/fanopt/geometry/schema.py` as a **circular mask**: `PANEL_PIVOT_REGION = CircularMask(center=(PIVOT_CENTER_X, 0), radius=PIVOT_BOSS_RADIUS + 0.001)` where `PIVOT_CENTER_X = 0.008 m` and `PIVOT_BOSS_RADIUS = 0.006 m` (so the keep-out radius is 7 mm from the pin centerline — covers the full 12 mm-OD boss plus a 1 mm clearance). Layer 2 fields (TPMS, noise, louver, texture, edge) and Layer 3 primitives are *both* clipped against `PANEL_PIVOT_REGION` in §9.7.1 Step 0 panel-domain mask BEFORE the Boolean subtraction — the pivot region is excluded from the carving domain just like the rib region is. **Why a circular mask (item #41 lock):** the prior rectangular `((0.0, 0.010), (-0.004, +0.004))` bound was 8 mm in y, narrower than the 12 mm boss (the boss extends to |y| = 6 mm). Layer 2/3 carving could legally cross into the boss annulus at |y| ∈ [4, 6] mm, exactly the region that delivers the e/d = 2.0 pin-tear-out margin. The circular mask covers the full boss + clearance regardless of orientation.
- **Rib-panel fillet (preserved zone — closes click-mate moment load path):** under the rib is no longer rigidly tied to the pivot (which moved to the panel at y = 0), so the **full click-mate moment (~0.18 N·m per rib)** flows through the **rib-panel fillet** at y = ±rib_center. adds the rib-panel junction as a preserved zone: a 2 mm × 2 mm cross-section strip running the full rib length at the fillet location, density-clamped to ρ = 1 in the rib SIMP TO. Backed by §N7 **check 15 — "rib-panel fillet radius ≥ 1.0 mm"** and a §59.5 combined-blade FEA assertion that peak σ_VM at the fillet < the -Material orientation-dependent cyclic allowable. The `material_locks` table specifies `K_t_fillet = 1.5` for a generous 1 mm fillet; tighter fillets (< 1 mm) trip the §N7 check and reject the design.
- **Click-feature footprint** . Two shared constants in `src/fanopt/geometry/schema.py`:
 - `CLICK_FOOTPRINT_X_RANGE = (L_blade - 0.010 m, L_blade)` — last 10 mm of rib length (gives 5 mm of slack at the inboard end of the patch).
 - `CLICK_FOOTPRINT_Y_RANGE_PANEL_EDGE = (panel_tangential_outer − 0.005, panel_tangential_outer)` — a 5 mm tangential band at the panel's outer edge at the tip, where `panel_tangential_outer = panel_width(r=L_blade)/2 ≈ 0.0225 m` per the #35 widening lock. (Item #3 relocation: the earlier `CLICK_FOOTPRINT_Y_RANGE = (−0.0025, +0.0025)` placed the click on the outer rib's centerline; that placement was geometrically impossible to engage adjacent blades — see the §0 click-architecture row.)

 The **5 × 5 mm cyclic-load patch** is the rectangle `CLICK_FOOTPRINT_X_RANGE × CLICK_FOOTPRINT_Y_RANGE_PANEL_EDGE` at the panel's outer tangential edge. Tip placement (not mid-blade) is mandatory: it locks adjacent panels into the deployed corrugated surface; a mid-blade click would leave the tips free. Mandatory: cyclic engagement load + dimensional tolerance band require continuous material here. **Shared constants:** the **CadQuery generator's `make_panel_solid` step** (§9.7) writes the click chamfer + detent into this region; the §9.7.3 Check 7 exclusion band reads `CLICK_FOOTPRINT_X_RANGE` AND `CLICK_FOOTPRINT_Y_RANGE_PANEL_EDGE` from `schema.py` so the click footprint and the schema-level Layer 2/3 bound can't drift apart. The FEniCSx rib TO solver no longer carries the click footprint as a preserved zone (per the §3.1.2a update — the click moved off the rib).

**§3.1.2 split (clarification for panel-pivot architecture):**

- **§3.1.2a — Rib TO preserved zones (apply to ribs only):** rib-panel interface (y-edge over the rib's radial band) and rib-panel fillet (2×2 mm strip). Both density-clamped to ρ = 1 in the rib SIMP TO. **C14 note:** the rib-panel fillet preserved zone is enforced **in 2D SIMP via density-clamping ρ = 1 over the 2×2 mm in-plane strip** — this preserves the material in the planform but does NOT evaluate the 3D fillet stress (which is a z-axis transition the 2D Reissner-Mindlin elements cannot resolve). The actual 3D fillet stress is evaluated separately in **Phase 2.5 (rib-only 3D static FEA on the localized junction)** and again in the **§59.5 combined-blade check** (full assembly with Layer 2/3 panel topology). The Phase 2 SIMP density-clamp is a *geometric* preserve, not a stress-correct check. **Rib radial band: x ∈ [HUB_RADIUS, L_blade − RIB_TIP_TAPER] = [0.020, 0.185 m]** per Architectural D (C7 inner) + Architectural A (RIB_TIP_TAPER outer). The rib has NO pivot hole under panel-pivot architecture, NO click-feature footprint under the item #3 fix, NO material at x < 0.020 m (the inner 20 mm is the HUB / panel-only boss region per C7), AND NO material at x > 0.185 m (the last 15 mm is the click / panel-only tip region per Architectural A). **Total rib radial extent: 165 mm** (was 200 / 185 mm). The dual rib-band lock means the panel runs the full L_blade = 200 mm radially with ribs flanking only the central 165 mm.
- **§3.1.2b — Panel generator preserved masks (apply to panel only):** **(1)** `PANEL_PIVOT_REGION = CircularMask(center=(0.008, 0), radius=0.007 m)` — the 12 mm-OD boss + 1 mm clearance. **(2)** `CLICK_FOOTPRINT_PANEL_EDGE_REGION` (NEW per item #3 fix) — the 5×5 mm patch at the panel's outer tangential edge at the tip, locked at `CLICK_FOOTPRINT_X_RANGE = (L_blade − 0.010, L_blade)` × `CLICK_FOOTPRINT_Y_RANGE_PANEL_EDGE = (panel_tangential_outer − 0.005, panel_tangential_outer)` where `panel_tangential_outer = panel_width(r=L_blade)/2 ≈ 0.0225 m` (half the tip panel width per the #35 widening lock). The panel between the ribs is NOT a TO domain — its topology is fully driven by Phase 2b's parametric generative design (4-layer hybrid: envelope + Fourier + macro-pattern/procedural-math fields + capped primitive); the panel-pivot mask AND the click-footprint mask are both enforced by §9.7.1 Step 0 panel-domain mask (Layer 2/3 cannot carve into either region).

The earlier closing line "These preservations apply to ribs only" was correct for §3.1.2a but incorrectly grouped the §3.1.2b panel-pivot mask under it. The split removes the contradiction.

#### 3.1.3 Panel Topology Generation
The panel is generated by a CadQuery script (`generate_blade.py`) executing five stages over a 4-layer parameterization:

1. **Layer 1 — Outer envelope:** `make_outer_envelope(camber, twist, thickness_2_to_5_mm, edge_profile, fourier_LE, fourier_TE)` → smooth solid block whose leading and trailing edges may carry Fourier-series modulation (k=1,2,3 harmonic amplitudes with fixed phases). Output silhouette can range from smooth airfoil to bat-wing, leaf-like, scalloped, or jagged.
2. **Layer 2 — Macro-pattern and procedural math fields:** 0-3 active fields per design, applied in fixed order to avoid order-dependent CAD problems: TPMS lattice → noise-threshold subtraction → louver cuts/ribs → surface texture (dimples/ridges/bumps) → edge serrations. The 5-field library gives the optimizer access to: directional design families (louvers — critical for asymmetric drag), boundary-layer manipulators (texture), aesthetic/noise-shaping (edge), organic emergent topology (Perlin/Simplex noise thresholding), and lattice mass-savings (gyroid or Schwarz-diamond TPMS). All Layer 2 fields are safe-by-construction; parameter ranges mathematically guarantee no CAD edge cases.
3. **Layer 3 — Capped 0-1 independent primitive:** for asymmetric point features that don't fit any pattern family. Wrapped in try/except (the only step where CAD failures can occur). Skip on failure.
4. **Layer 4 — Manufacturing categoricals:** print orientation, layer height, click chamfer/detent/clearance.
5. **§N7 manufacturability filter** (11 checks; score [0,1]): reject if score < 0.5; warn if 0.5-0.8; clean if ≥ 0.8.

The optimizer can produce: slatted/louvered designs (Layer 2 louver field), bone/coral/sponge-like organic topology (Layer 2 noise threshold), gyroid-lattice cutouts with directional density gradients (Layer 2 TPMS), bat-wing or leaf-like silhouettes (Layer 1 Fourier), traditional smooth airfoils (Layer 2 all inactive), or combinations. See §9.7 for the full generator architecture and Phase 2b for the BO setup.

#### 3.1.4 Density Filtering

```
rho_tilde_e = (sum_i H_ei * v_i * rho_i) / (sum_i H_ei * v_i)
```

Where `H_ei = max(0, r_min - dist(e, i))` with filter radius `r_min` = 1.5 to 3 times the element size.

#### 3.1.5 Stress at the Pivot (Critical Failure Location)

The pivot hole creates a stress concentration. For a circular hole of diameter `d` in a plate of width `w` under bending:

```
K_t = 3.0 - 3.13*(d/w) + 3.66*(d/w)^2 - 1.53*(d/w)^3
```

**The pivot hole is in the PANEL** at y = 0, NOT in the rib. The panel is locally thickened to a 12 mm-OD circular boss around the pin hole. For d = 3 mm pin in the **12 mm boss**: **d/w = 3/12 = 0.25; the tension Peterson polynomial gives K_tt = 2.42 — this is the locked value** (item #5 lock). Tension allowable: 0.30 · σ_y_XY / K_tt = 0.30 · 45 / 2.42 = **5.58 MPa nominal**. Bearing allowable: 0.10 · σ_y_Z / K_t_bearing = 0.10 · 30 / 1.5 = **2.00 MPa nominal** (Z-direction; binds for canonical baseline). Bending allowable: 0.30 · σ_y_XY / K_tb = 0.30 · 45 / 3.2 = **4.22 MPa nominal**. Static allowable (tension, SF=1.5): 45 / (1.5 · 2.42) ≈ **12.4 MPa nominal**. **Filter 2 thresholds:** §6.3.1 Filter 2 evaluates each mode independently against its own per-mode allowable (no scalar superposition); a design passes iff all three pass.

The peak stress at the pivot region:

```
sigma_max = K_t * sigma_nominal = K_t * (M * c) / I
```

Where M is the bending moment at the pivot, c is the distance from the neutral axis to the extreme fiber, and I is the second moment of area of the rib cross-section at the pivot (accounting for the hole).

**Fatigue at the pivot:** With **K_tt = 2.42 (panel-pivot, d/w = 0.25 in 12 mm boss)** and cyclic loading at 2 Hz, the panel pivot hole is the fatigue-critical location. Keep peak cyclic stress below 30% of yield / K_t in XY tension and 10% of σ_y_Z / K_t in Z bearing to ensure adequate fatigue life (see §3.1.7).

**K_t at non-pivot hotspots (NEW table — closes the "K_t evaluated at all hotspots" requirement from ):** Filter 2 and the §59.5 FEA gate evaluate `σ_eff` at every stress-concentration hotspot and reject on the **minimum** allowable across the hotspot set. Without numerical K_t values per hotspot the rule is unenforceable. specifies:

| Hotspot | Geometry | K_t | Mode | Reference / chart | Cyclic allowable |
|---------|----------|-----|------|-------------------|------------------|
| **Panel pivot hole — tension** | 3 mm hole in **12 mm boss** (panel-pivot architecture); tangential tension at hole equator | **K_tt = 2.42** | XY in-plane tension | Peterson §5.1 (hole in finite-width plate), d/w = 0.25 (3 mm hole in 12 mm boss; the 6-8 mm inter-rib panel is locally thickened to a 12 mm-OD boss centered on the pivot hole — Layer 2/3 cannot carve into the boss) | **5.58 MPa** (= 0.30·σ_y_XY/K_tt, σ_y_XY = 45 MPa, rib-flat print) |
| **Panel pivot hole — bending** | Same hole; chordwise bending peak at panel top/bottom skin | **K_tb = 3.2** (literature-spread upper bound, conservative) | XY in-plane bending | Peterson §5.3.1 (bending chart), d/w = 0.25 | **4.22 MPa** (= 0.30·σ_y_XY/K_tb) |
| **Panel pivot hole — bearing** | Pin-bore contact; radial loading through Z-thickness | **K_t_bearing = 1.5** | **Z-direction** (interlayer) | Peterson pin-bearing baseline; load transmitted through stacked print layers | **2.00 MPa** (= 0.10·σ_y_Z/K_t_bearing, σ_y_Z = 30 MPa per §10.1) |
| **Click detent at panel outer edge (item #3 lock — moved from rib tip; single Z-shear mode)** | 0.3-0.5 mm radius detent bump on the panel's outer tangential edge at the tip; chamfer face spans full panel_thickness in z. Loaded **only in Z-shear** during lap engagement (adjacent panel chamfers slide past each other along their z-span); there is no chordwise XY tension at the panel edge from click loading (the chordwise loads at the panel pivot are carried by the boss, not the tip). The earlier draft listed a separate 4.50 MPa XY mode here — that mode referred to the **rib-mounted** click and is retired with the panel-edge relocation. | **K_t = 3.0** (small-radius semicircular notch in panel edge; conservative) | Z-direction (interlayer) — FDM interlayer dominates because the chamfer face is printed in successive Z layers | Peterson §3.3 with Z-fatigue factor | **2.00 MPa** (= 0.20·σ_y_Z/K_t = 0.20·30/3.0; the 0.20 factor — not 0.10 — reflects that the click chamfer is loaded only during fold/deploy events (~few cycles/day) vs the continuous 2 Hz pin-bore bearing, so the fatigue knockdown is relaxed by 2× per the §3.1.5 rationale row) |
| **TPMS through-hole** | 1-2 mm characteristic opening in 2.2-3.8 mm panel; smoothly curved walls (lattice topology) | **K_t = 2.5** (smoother than a sharp drilled hole; gyroid/Schwarz-D curvature locally decreases concentration vs a circular hole) | XY tension | Peterson §5.1 with curvature correction | **5.40 MPa** |
| **Rib-panel fillet** | 1 mm radius fillet at rib-panel junction (§N7 check 15 requires ≥ 1 mm radius) | **K_t = 1.5** (large-radius fillet; lower bound) | Bending | Peterson §2.6 (fillet in stepped flat bar), r/h ≈ 0.5 with the 2 mm rib thickness | **9.00 MPa** |
| **Slot end (Layer 2 louver)** | Louver-slot end-cap with min(0.5 mm, slot_width/3) auto-fillet | **K_t = 2.0** (filleted slot end; the auto-fillet keeps this below the unfilleted slot's K_t ≈ 3.5-4.0) | Mixed | Peterson §2.6, r/h ≈ 0.25 with slot-width-dependent fillet | **6.75 MPa** |
| **Lock-mech corner** | Reserved for V2 — no V1 hotspot | N/A for V1 | — | N/A | N/A |

**Per-hotspot independent checks (no scalar superposition):** Filter 2 + §59.5 evaluate σ at each hotspot in its own mode and direction; a design passes iff **every** check passes. **Scalar superposition `σ_eff = K_tb·σ_bend + K_tt·σ_tens + σ_bearing` is wrong** — these stresses peak at three different physical points (panel skin chordwise / hole equator tangential / pin bore radial) and three different tensor directions, so a scalar sum over-rejects by ~2-3× and would distort the Pareto front. The three independent checks:
1. **Bending at panel skin:** `K_tb · σ_bending < σ_allow_bending = 4.22 MPa nominal`.
2. **Tension at hole equator:** `K_tt · σ_tension < σ_allow_tension = 5.58 MPa nominal` (at K_tt = 2.42 in 12 mm boss).
3. **Bearing at pin bore:** `σ_bearing < σ_allow_bearing = 2.00 MPa nominal` (Z-direction, σ_y_Z = 30 MPa).

Filter 2 returns `passed=True` iff all three pass; the failing-mode tag goes into `failure_code ∈ {fea_bending, fea_tension, fea_bearing}` so post-campaign analysis sees which mode dominates rejections.

**Binding hotspot for baseline (no TPMS, no aggressive louvers):** the **panel-pivot bearing check at 2.00 MPa** is the minimum allowable in the table and binds first under canonical loading (item #3 σ_y_Z = 30 MPa lock). For TPMS-heavy or louver-heavy designs, the **click detent Z-floor at 2.00 MPa** or the **bending mode at 4.22 MPa** can also bind if the dynamic-load assertion (below) is near its budget. The pre-CFD struct solver (`pre_cfd_struct_estimate.py`) reads the K_t + mode table from `material_locks.k_t_by_hotspot` at module import; CI test asserts every hotspot listed has a non-null (K_t, mode, allowable) tuple in the locked dict.

#### 3.1.6 Von Mises Stress Criterion

For a thin rib (plane stress approximation, sigma_z ~ 0):

```
sigma_vm = sqrt(sigma_x^2 + sigma_y^2 - sigma_x * sigma_y + 3 * tau_xy^2)
```

This simplification is appropriate for ribs with thickness 2 mm and width 6-12 mm.

**Typical yield strengths for 3D-printed materials:**
- PLA: ~50-60 MPa (brittle)
- PETG: ~40-50 MPa (ductile, good fatigue)
- Nylon (PA): ~40-85 MPa (excellent fatigue)
- PLA-CF: ~60-70 MPa (stiff but brittle)

#### 3.1.7 Fatigue Considerations for Rib Oscillatory Loading

A hand fan rib undergoes oscillatory loading:
- **Per session:** At 2 Hz for 5-15 minutes: 600-1,800 cycles per session.
- **Lifetime:** Daily use over months: 50,000-500,000 total cycles.

**Polymer fatigue for FDM parts:**
- Fatigue life depends on print parameters (raster angle, infill density, layer height).
- For FDM PETG, raster angle at 45 degrees outperforms 0 degrees at low stress amplitudes.
- Conservative design rule: keep peak cyclic stresses below 30% of yield strength **for XY-plane loading**. **Z-direction (interlayer) fatigue is materially worse** — FDM PETG Z-fatigue can drop to **0.10-0.15 · σ_y at 10⁵ cycles** per published data. The Layer 4 `print_orientation` BO variable directly determines which σ_y applies (XY-plane vs Z-direction loading on the highest-stress section). locks the **per-hotspot per-mode allowable** (NOT a single scalar min-formula):
 `σ_cyc_allowable[hotspot] = fatigue_factor[mode] · σ_y[orientation_of_load] / K_t[hotspot]`
 where each hotspot in the §3.1.5 table has a fixed (mode, K_t, allowable) tuple and the load orientation is fixed by physics (e.g., pin-bearing is **always** Z-direction regardless of print orientation because it goes through stacked print layers; panel skin tension under rib-flat print is XY). The earlier draft used a scalar `min(0.30·σ_y_XY, 0.20·σ_y_Z) / K_t` — this is conceptually wrong because XY and Z allowables apply to different physical stress directions at different hotspots; taking a min over both is meaningful only when the same hotspot's stress could swap orientation under a different print mode (and then only for that hotspot's bound). The pre-CFD Filter 2 + the §59.5 FEA gate both consume the per-hotspot allowable from `material_locks.k_t_by_hotspot` in `material_locks.py`; legacy `σ_cyc_allowable_by_orientation` is deprecated.
- **At the panel pivot hole with K_tt = 2.42 (panel-pivot, d/w = 0.25 in the 12 mm boss):** effective allowable nominal tension stress = 0.30 · 45 MPa / 2.42 ≈ **5.58 MPa nominal** (cyclic-fatigue). This is one of three independent checks at the panel pivot (tension 5.58 / bending 4.22 / bearing 2.00). The **bearing mode at 2.00 MPa binds first** under canonical loading because the bearing load goes through the stacked print layers and the Z-direction fatigue factor is half the XY factor. Production code reads each allowable from `material_locks.k_t_by_hotspot` keyed by hotspot name; the prior 5.97 MPa value (K_t = 2.26, d/w = 0.375 in 8 mm panel) is obsolete under the locked boss thickening.

#### 3.1.8 FDM Material Anisotropy

FDM-printed ribs exhibit 20-40% reduction in mechanical properties in the build direction (Z-axis) compared to in-plane (XY). Since ribs are printed flat on the build plate:
- Primary bending loads act in-plane (XY) -- the strong direction.
- The isotropic SIMP assumption is most valid for this print orientation.
- Verify with FEA post-optimization using orthotropic properties.

**FDM-printed PETG properties (100% infill, 0.2mm layer):**
- E_XY ~ 1300 MPa (NOT the datasheet value of 2100 MPa which is for injection-molded)
- E_Z ~ 1000 MPa
- nu = 0.38, density = 1270 kg/m^3

### 3.2 Aerodynamics -- Folding Fan in Oscillatory Motion

#### 3.2.0 Gesture being optimized
The optimization targets the **paddle / bluff-body slapping gesture** — *not* a wing-like pitching motion. The user holds the fan with the **broad panel face perpendicular to the swing direction** and sweeps the wrist back-and-forth at f = 2 Hz so that the broad face pushes air on the forward stroke and is dragged through air on the return stroke. The fan acts as an oscillating bluff body, not an oscillating airfoil.

**Coordinate convention (locked — pivot pin = +z, broad face = +z; the 10 PANELS stack on the pivot pin along z under panel-pivot architecture, each panel 2.2-3.8 mm thick in z. The 20 ribs (10 blades × 2 ribs/blade) sit at `y = ±rib_center` on either side of each panel's tangential midline; ribs do NOT stack on the pin — they're off-axis at different y positions, and a single straight z-pin cannot pass through both ribs of one blade per §0 row 25. Pivot pin location: through `(d_handle, 0, 0) = (0.05, 0, 0)`, parallel to z.
- **Blade planform = XY plane.** A blade at sector-center orientation extends in **+x (radial direction): 200 mm in x** (from pivot at x = 0.05 to tip at x = 0.25). Blade width direction = **±y: 6-8 mm at the pivot end, widening trapezoidally to ~45 mm at the tip** per the item #35 / H13 panel-widening lock (see §0 row 138 for the `panel_width(r) = r · 0.232 − 2·rib_width(r) − 0.5 mm` formula). The two ribs sit at `y = ±rib_center` on either side of the panel's tangential midline at each radial station; the inter-rib panel is 6-8 mm only near the pivot.
- **Blade thickness direction = z.** Blade is a flat plate with planform in XY and thickness in z. **The 2 ribs of one blade are at the +y and −y edges of the planform** (NOT spaced in z). Panel between them is at the same z as the rib midplane. Z-extent at panel midspan = `panel_thickness` (2.2-3.8 mm under ); z-extent at the rib edges in y = rib thickness (2 mm).
- **Wrist axis = +y** (wrist-flexion hinge, perpendicular to the forearm direction +x). The wrist-grip point is at world origin (0, 0, 0); the handle extends in +x to the pivot at (0.05, 0, 0). Sinusoidal wrist rotation about +y drives the gesture.
- **Swing direction at swing-center = ±z.** For a blade at (x, 0, 0) with x ∈ [0.05, 0.25], rotation about +y gives v = ω × r = (0, ω, 0) × (x, 0, 0) = (0, 0, -ω·x) — the broad panel face (normal ±z) sweeps in ±z, face-on to the swing direction at swing-center.
- **Broad panel face normal at swing-center = ±z** → **freestream direction in SU2 = ±z**.

**Implications for SU2 configs:**
- **Freestream direction (C2 lock):** the steady CFD configs use **explicit `FREESTREAM_PRODUCTIVE = (0, 0, -1)`** (air flows in −z relative to a stationary fan that's actually being swept in +z toward the user — the productive stroke) and **`FREESTREAM_RETURN = (0, 0, +1)`** (return stroke) — NOT `AOA = 0/180` directives. SU2's `AOA` convention measures the angle between freestream and the body x-axis and silently couples to the mesh's reference frame in version-dependent ways; explicit vectors are unambiguous. The `j_fan.py` post-processor's `t̂` integration matches the SU2 convention without an axis remap. **Implementation responsibility:** the Gmsh meshing scripts in `src/fanopt/cfd/mesh_*.py` write the +z-toward-user convention — blade radial direction along **+x**, wrist axis along **+y**, pivot pin axis along **+z**, blade width direction along **±y**, +z face is the user-ward face. **Required regression tests:** (1) `tests/test_cfd/test_mesh_streamwise_axis.py` — assert the `farfield`/inlet marker face normal is along the configured stroke direction. (2) `tests/test_cfd/test_freestream_direction.py` — run a steady case, sample the inlet face's velocity field, assert the vector matches the configured `FREESTREAM_PRODUCTIVE` / `FREESTREAM_RETURN` within numerical tolerance. Catches the case where SU2's AOA interpretation differs from the mesh's streamwise axis.
- **Rotation axis for the unsteady 3D config:** rotation is about the wrist axis +y — `PITCHING_OMEGA = (0, -ω_SHM, 0) = (0, -12.5664, 0) rad/s` (C11 sign lock: the NEGATIVE y-component is required by the right-hand-rule on the productive stroke — see §0 row 26). `MOTION_ORIGIN = (0, 0, 0)` at the wrist-grip point. Rotation is *of the entire fan about the wrist axis*, not per-blade pitching.
- **Steady CFD as a J_fan proxy + two-eval delta (C2 sign convention):** because the broad face is perpendicular to the freestream, steady CFD integrates the full pressure-drag force on the panel onto t̂. Run **two CFD evals per design** with explicit `FREESTREAM_DIRECTION`: **PRODUCTIVE = `(0, 0, -1)`** (air flows -z relative to a stationary fan that's actually being swept in +z toward the user) and **RETURN = `(0, 0, +1)`**. Emit `J_fan_steady_proxy = Drag_productive − Drag_return`. Symmetric designs score ≈ 0; asymmetric productive designs (louvers angled to grab air on the user-ward stroke) score positive. Without the delta, a one-direction steady proxy rewards solid-wall designs over asymmetric-drag designs and the BO converges on the parachute.

**Axis-convention lock:** **pivot/stacking = z, broad face = z, freestream = ±z, wrist = +y**. The §9.4.1 per-tier config-hash assertion includes a `PITCHING_OMEGA_AXIS = (0, 1, 0)` cross-tier constant so the CFD freestream and the structural pivot share one geometric frame.

#### 3.2.1 Flow Regime

- **Fan rib length (L):** ~0.20 m
- **Waving velocity (V):** 1-3 m/s (tip speed)
- **Reynolds number:** Re = V * L / nu = 10,000 - 40,000
- **Regime:** Low Reynolds number, unsteady, laminar to transitional

#### 3.2.2 Aerodynamic Differences: V-Unit Folding Fan vs. Solid Plate

The deployed fan (10 click-mated rigid PETG V-unit blades) differs from a solid plate in several ways that affect CFD:

1. **Rib ridges (one-sided under rib-flat, symmetric under deployed-V/edge per Architectural E / C10 lock):** under `print_orientation == 'rib-flat'` (the §7.4.3 default), the panel protrudes above the rib face by `(panel_thickness − rib_thickness) = 0.2-1.8 mm on +z only` (one-sided corrugation; panel and rib share a common z = 0 bottom face per the C10 lock; the −z face has panel and rib flush). Under `deployed-V` or `edge` orientations, midplane symmetry holds and the panel protrudes `(panel_thickness − rib_thickness)/2 = 0-0.9 mm per side` (symmetric corrugation). These create small spanwise corrugation features that trip the boundary layer, alter separation behavior, and contribute to the asymmetric drag the fan needs to generate net forward flux. The one-sided rib-flat case **amplifies** asymmetric drag (productive +z stroke exposes corrugation ridges; return −z stroke exposes the flush face). See §3.2.0 / Architectural E for the full conditional convention; the §3.2.4 wall-roughness calibration and the Phase 3 step 33/35 2D-slice geometry both consume the print-orientation-conditional amplitude.

2. **Click-mating seams between blades:** the inter-blade engagement is mechanical, not a continuous solid surface. Microscale gaps at click seams may permit small flow leakage. **H15 lock — reconciling the two clearance specs:** §0 row 33 locks `chamfer_clearance = 0.1 mm per side` (per-face design clearance between adjacent chamfer surfaces). The §7.4.1 figure of "0.15-0.20 mm design clearance" refers to **per-face manufacturing clearance** including FDM ±0.1 mm tolerance — i.e., the as-printed gap (with worst-case ±0.05 mm tolerance stack on each face) lands in the 0.15-0.20 mm band. The total chamfer-to-chamfer gap is therefore ≈ 2 × per-face clearance = 0.3-0.4 mm in the as-printed worst case (0.2 mm in the nominal design). CFD models this either explicitly (Phase 5 resolved-corrugated mesh) or implicitly as a wall-roughness correction (Phase 4 Tier 0/1 BO inner loop).

3. **Layer 2 generative cutouts (the emergent geometry):** depending on which Layer 2 fields the optimizer activates, the panel may have louver slits (designed flow leakage for asymmetric drag), noise-threshold cutouts (sponge/coral organic topology), TPMS through-blade channels (directional through-blade porosity), or surface dimples/ridges. These cut-throughs and surface features dominate the aerodynamic difference from a solid plate. The CFD must resolve them.

4. **No flexible-membrane physics:** the panels are rigid. No billowing, no flutter, no compliance-induced effects. The blade is a true rigid body for CFD purposes; flutter is not a concern.

**Fan performance metric -- directed momentum flux:**

```
eta_fan = (integral over one cycle of net momentum flux toward user dt) /
 (integral over one cycle of waving power input dt)
```

#### 3.2.3 Oscillating Motion Model

```
theta(t) = theta_max · sin(omega · t), omega = 2·pi·f
omega(t) = theta_max · omega · cos(omega · t)
alpha(t) = -theta_max · omega^2 · sin(omega · t)

omega_max = theta_max · omega
alpha_max = theta_max · omega^2 <-- peak angular acceleration
V_tip(t) = L · omega(t)
V_tip_max = L · theta_max · omega
```

For **L = 0.25 m (wrist-to-tip = d_handle 0.05 m + L_blade 0.20 m per §0 / §2.1)**, theta_max = 0.7 rad (40 deg), f = 2 Hz (omega = 4·pi rad/s):
- V_tip_max = 0.25 · 0.7 · 4·pi = **2.20 m/s**
- alpha_max = 0.7 · (4·pi)² = 0.7 · 157.9 ≈ **110 rad/s²**

**α_max:** ≈ 110 rad/s² for SHM at these parameters. All inertial-load magnitudes derive from this value (see §2.4 load case 2 and §3.1 plate-bending TO load specification).

**Reduced frequency:** k = π · f · c / V_tip_max ≈ **0.57** with c (mean panel chord) = 0.2 m, f = 2 Hz, V_tip_max = 2.20 m/s. The engineering conclusion is unchanged: k ≈ 0.5-0.6 places the flow firmly in the unsteady regime and motivates the multi-fidelity steady/unsteady split in Phase 4.

**Kinematics symbol table (H8 lock — every section uses these subscripted symbols; bare `ω` is forbidden in production code or new prose):**

| Symbol | Value | Meaning | Used at |
|--------|-------|---------|---------|
| `f_wave` | 2 Hz | Waving frequency | §3.2.3, §9.4 |
| `T_cycle` | 0.5 s | Waving period | §9.4 J_fan integration |
| `θ_max` | 0.6981 rad (40°) | SHM amplitude | §3.2.3, SU2 PITCHING_AMPL |
| `ω_SHM` | 2π·f = **12.566 rad/s** | SHM angular frequency of the pitching motion | SU2 PITCHING_OMEGA |
| `ω_blade_max` | θ_max · ω_SHM = **8.8 rad/s** | **Peak instantaneous blade angular velocity** | V_tip derivation, V_local(r), BL thickness |
| `α_max` | θ_max · ω_SHM² = **110 rad/s²** | Peak angular acceleration | §2.4 inertial loads, Filter 2 |
| `V_tip` | ω_blade_max · L_wrist-to-tip = **2.20 m/s** | Peak tip velocity (wrist axis to tip) | Re_global, Mach |
| `V_local(r)` | ω_blade_max · r_wrist | Local tangential velocity at wrist-axis radius r | §3.2.4 BL, C6 multi-radius slice |
| `Re_global` | 37000 | Re at L = 0.25 m wrist-to-tip | SU2 cfg Tier 0 / Tier 1 |
| `Re_local(r)` | V_local(r) · r / ν | Local Re | §3.2.4 BL thickness |
| `k_reduced` | π·f·c/V_tip ≈ 0.57 | Reduced frequency | §3.2.3 |
| `L_blade` | 0.20 m | Rib + panel radial extent from pivot to tip | Phase 2 TO domain |
| `L_wrist_to_tip` | 0.25 m = d_handle + L_blade | **Distance from wrist axis to tip; the canonical lever arm for τ → F conversions** | Pin bending, click-friction balance, §59.5 V_local_max |

**Lever-arm audit lock (H8 propagation):** any conversion that converts a torque to a tangential force at the click region (or anywhere outside the pivot region) MUST use `L_wrist_to_tip = 0.25 m`, NOT `L_blade = 0.20 m`. This failure mode appeared in: the pin-bending lever (§2.3, fixed); the H6 click-friction force balance (`F_inertial_at_click = τ_inertial_peak / 0.25`, not `/ 0.20`); the §59.5 V_local_max derivation. CI audit `tests/test_audit/test_lever_arm_uses_wrist_to_tip.py` greps prose for `/ 0.20` near torque-to-force conversion language; allow-listed only where the calculation truly is at the rib tip from the pivot (pure r-from-pivot, e.g., the rib-internal bending-moment integral).

#### 3.2.4 Aerodynamics of the Corrugated Rigid Surface
**Corrugation sign convention (Architectural E — print-orientation conditional):** under the locked axis frame (pivot pin = +z, blade planform in XY, blade thickness in z), the panel-rib z-relationship is **conditional on print_orientation**:

- **`print_orientation == 'rib-flat'` (the §7.4.3 default, plano-convex constraint):** panel and rib share a **common BOTTOM face at z = 0** (no midplane symmetry). Both extend UPWARD from z = 0; the panel reaches z = panel_thickness (2.2-3.8 mm); the rib reaches z = rib_thickness = 2 mm. **Panel protrudes above the rib by (panel_thickness − rib_thickness) = 0.2-1.8 mm on the +z side only**; the −z side has both flush at z = 0. **One-sided corrugation, not symmetric.** The earlier draft asserted midplane symmetry under rib-flat — that violates the plano-convex print constraint (rib bottom hovering 0.9 mm above the build plate at panel_thickness = 3.8 mm would need supports under the entire 200 mm rib length, exactly the failure plano-convex was meant to prevent).
- **`print_orientation == 'deployed-V'` or `'edge'` (non-functional bed contact; supports acceptable):** midplane symmetry holds — panel and rib share a common midplane, panel protrudes on BOTH +z and −z by (panel_thickness − rib_thickness)/2 = 0-0.9 mm per side.

At max panel thickness 3.8 mm: rib-flat protrusion = 1.8 mm on +z only (no −z protrusion); deployed-V/edge protrusion = 0.9 mm per side, 1.8 mm differential. The corrugation amplitude (= panel-rib differential) is 0.2-1.8 mm regardless of orientation, but **rib-flat produces one-sided corrugation that amplifies asymmetric drag** (productive +z stroke exposes ridges; return −z stroke exposes a flush face). Beneficial for J_fan; must be modeled in the CFD slice (Phase 3 step 33 / step 35). The §3.2.4 model, §3.2.5 surface-roughness calibration, and the 2D-slice geometry use the print-orientation-conditional sign convention.

**sign convention lock (closes the both-sides-of-midplane ambiguity):** the **upward-facing functional surface of the deployed fan is +z** (per : +z = streamwise = freestream-incoming direction; the user's hand on the handle holds the deployed fan with the +z face oriented toward the user's body, sweeping +z → −z and back). Under this convention:
- The **+z panel face** is the aero-functional face (the one the §3.2.5 wall-roughness calibration is anchored on; bed-down for plano-convex print orientation per §N7 check 14).
- The **panel protrudes ABOVE the rib by (panel_thickness − rib_thickness)/2 on the +z side** (the calibrated face is the one the CFD treats as the wind-pushing surface).
- The **panel also protrudes BELOW the rib by the same amount on the −z side**, but that face is the return-stroke face and is treated as a symmetric roughness term by the wall-model (no separate calibration).
- **Wall-normal outward unit vector at the +z panel face is +z** (asserted by the Phase 3 step 33 / regression test `tests/test_cfd/test_corrugation_direction.py` AND a new sub-assertion: `mesh.surface_marker("fan_surface").faces_with_normal_z_positive.area > 0.5 · total_fan_surface_area`).
- If the slicer's bridging tags and the CFD's wall-normals disagree on which side is +z, J_fan will integrate with the wrong sign and every Pareto front is wrong. The regression test catches this at mesh-generation time.

**When this matters:** Phase 3 step 33 generates the 2D slice from the §9.7 generator output and resolves the corrugation explicitly. If `mesh_2d_slice.py` is written from the §3.2.4 prose under the *pre-* "rib ridges above panel" framing, the resolved geometry will put the rib above the panel (the inverse of the geometry), and:
1. The Phase 3 R² correlation between the 2D-resolved-ridge result and the 2D-roughness-modeled result will be calibrated to the wrong physics.
2. The Phase 4 wall-roughness model parameters (per the last paragraph of this section) inherit that miscalibration.
3. Every Tier 0 / Tier 1 CFD run that uses the calibrated roughness model integrates J_fan over a wrong surface curvature.

**Why the fix lands here, not "queued for later":** the §3.2.4 prose is the *construction spec* that `mesh_2d_slice.py` reads from. If Phase 3 step 33 is implemented from the un-corrected prose, the bug compounds through every Phase 3/4/5 CFD run before anyone spots it. Inline-now closes the source-of-truth window.

**How the fix lands:**
- This section rewrites "rib ridges (1-1.5 mm protrusion above the panel surface)" → "**panel ridges (0-0.9 mm protrusion above each rib face) under **" (next paragraph, below).
- Phase 3 step 33 (`mesh_2d_slice.py`) gets a **mandatory regression test (`tests/test_cfd/test_corrugation_direction.py`, )** that asserts the resolved 2D-slice geometry has the panel above the rib at every x along the deployed sector (panel face z-coordinate ≥ rib face z-coordinate within each blade's planform at any sampled x). Test runs as part of Phase 0's CI before any Phase 3 evaluation dispatches.
- §3.2.5 surface-roughness calibration is updated to call out that the calibrated face is the **panel top face under (plano-convex constraint with rib-flat print orientation)**, not the rib face.
- The amendment row in §0 already documents the geometric basis; this section closes the aero-modeling consequence.

The deployed fan is a corrugated rigid surface: N panels (one per blade) separated by N+1 rib gaps (each blade contributes 2 ribs but adjacent blades share an inter-blade gap so the surface alternates **panel-ridge → rib-valley → panel-ridge → rib-valley**).

**Low-Re corrugated-surface drag (relevant references):**
- Selig low-Re airfoil database (UIUC LSATs): polar data for thin cambered airfoils at Re = 30k-100k overlaps the fan operating regime.
- NACA 0006-0012 family: thin symmetric and cambered baselines for the panel airfoil shape.
- Williamson, "Vortex dynamics in the cylinder wake": bluff-body vortex shedding at low Re; informs the rib-ridge wake interaction.

**Corrugation amplitude vs. boundary layer thickness (corrected direction):**

The aero-relevant question is whether the **panel ridges** are large enough to perturb the boundary layer (and thus the J_fan) or small enough to act like distributed surface roughness.
- **Panel protrusion above each rib face (print-orientation conditional per Architectural E):** under `rib-flat` (the §7.4.3 default), the panel protrudes by **(panel_thickness − rib_thickness) = 0.2-1.8 mm on +z only** (one-sided corrugation; panel and rib share a common bottom at z = 0). Under `deployed-V` or `edge`, the panel protrudes by (panel_thickness − rib_thickness)/2 = 0-0.9 mm on each side (midplane-symmetric corrugation). For a 2.2 mm panel: 0.2 mm (minimal corrugation); for the maximum 3.8 mm panel: 1.8 mm (rib-flat, one-sided) or 0.9 mm per side (deployed-V/edge). The corrugation amplitude scales with `panel_thickness − rib_thickness` and is a **Pareto-traded quantity** (Layer 1 thickness profile drives it). Rib-flat designs near the upper panel-thickness bound have ~1.8 mm one-sided corrugation, amplifying asymmetric drag between productive and return strokes.
- Boundary-layer thickness on the 200-mm panel chord (Blasius laminar estimate): δ ≈ 5·x / √Re_x. **At x = 0.2 m, Re_x = V·x/ν = 2.20·0.2/1.5e-5 ≈ 29,300** (NOT the §9.4 REYNOLDS_NUMBER = 37,000, which is at the REYNOLDS_LENGTH = 0.25 m wrist-to-tip reference; Re scales linearly with x, so Re_at_x = 37000 · 0.2/0.25 = 29,600). Then δ ≈ 5 · 0.2 / √29,600 ≈ **5.8 mm**.
- **Panel ridges (0-0.9 mm protrusion under )** are *smaller* than the ~5 mm boundary-layer thickness across the entire 2.2-3.8 mm panel-thickness range, so they act more like distributed roughness with intermittent separation than discrete bluff bodies. **Local-velocity caveat:** the 5.8 mm estimate uses V = V_tip = 2.20 m/s, which is the friendliest velocity on the blade. At panel midspan (r ≈ 0.15 m from wrist) the local velocity is V_mid = 1.32 m/s and the local BL thickness at x = 0.2 m is δ ≈ 5·0.2/√(1.225·1.32·0.2/1.5e-5) ≈ 5 · 0.2 / √21,560 ≈ 6.8 mm — slightly *larger* than the tip-velocity estimate (lower local Re → thicker laminar BL). The conclusion (BL >> panel ridge) holds with margin across the blade and across the panel-thickness range; the tip-velocity number is a useful round number and is conservative against ridge disruption (thinner BL is harder for ridges to fit inside).

**Phase 3 modeling implication:** the 2D unsteady CFD slice must explicitly include the rib ridges (not smear them into a roughness term) to capture the right separation behavior, but the 3D production CFD in Phase 4 may use a calibrated wall-roughness model to avoid the very fine mesh that ridges otherwise require. Calibration: compare the 2D-resolved-ridge result to a 2D-roughness-modeled result on the same baseline geometry; use the discrepancy to set the roughness model parameters.

**Print-frame ↔ Deployed-frame mapping (M15 lock — single canonical table; referenced by §6.2.1, §9.7, §N7 Check 14):**

```
Print frame (build plate at z = 0, build direction +z):
  - Bed-contact face: outward normal = −z (faces the bed)
  - Top of print: outward normal = +z (faces ambient air)

Deployed frame (user at +z relative to fan):
  - Aero-functional face (productive thrust direction; the user-ward face): outward normal = +z (toward user)
  - Return-stroke face: outward normal = −z (away from user)

Mapping under rib-flat print orientation (the §7.4.3 default):
  - Print bed-contact face → Deployed +z face (toward user; calibrated for wall roughness)
  - Print top → Deployed -z face (return stroke; symmetric roughness model)
```

This table appears once here and is cross-referenced from §6.2.1 (plano-convex camber spline), §9.7.1 (envelope construction), §N7 Check 14 (calibrated face). Existing prose updates to "see §3.2.4 print-deployed frame mapping" instead of redescribing.

#### 3.2.5 Surface Roughness Effects
FDM layer lines on the rigid PETG blade surface affect boundary-layer transition at these Reynolds numbers. **Bed surface locked (M13):** `material_locks.BED_SURFACE = "smooth_pei"` — specifically Bambu Lab Cool Plate Super Tack (P/N: AP05) or equivalent smooth-PEI surface with Ra ≤ 5 µm. The calibrated face's total Ra is the upper envelope of "layer-line texture ~5 µm" + "smooth-PEI ≤ 5 µm contribution from the bed contact" ≈ **Ra ≈ 5-10 µm** (tightened from the earlier 5-15 µm spec which conflated smooth and textured PEI). Textured PEI is a *different* surface (Ra 10-30 µm) and is NOT permitted under the locked calibration; switching the bed surface requires re-running the §3.2.4 wall-roughness calibration and is queued to `docs/V2_backlog.md`. CI test `tests/test_audit/test_bed_surface_locked.py` asserts the calibration metadata's `BED_SURFACE` field matches the lock.

Print orientation per blade (§7.4) determines whether the layer lines run along the panel chord (parallel to flow, minimal impact) or across it (perpendicular, potential boundary-layer trip). Print orientation is a categorical BO variable in Phase 4.

---

## 4. Software Tools: Easiest Path Analysis

### 4.1 Philosophy: Scripted Workflows Over GUI Tools

The user works with Claude Code (an AI coding assistant in the terminal). The ideal workflow maximizes what can be automated through Python scripts that Claude writes and the user runs. This fundamentally changes the tool selection criteria:

- **GUI tools (Fusion 360, FreeCAD GUI):** Require manual interaction; Claude cannot operate them. Every iteration requires the user to manually click through menus.
- **Scripted tools (CadQuery, PyTopo3D, SU2 config files, BoTorch):** Claude writes the scripts; the user runs them. Iteration is fast -- Claude modifies the script and the user re-runs.

### 4.2 Tool Comparison: Ranked by Ease of Scripted Execution

| Rank | Tool Stack | Cost | Effort to Learn | Effort to Execute | Script-ability | Claude Can Write It? | Notes |
|------|-----------|------|-----------------|-------------------|---------------|---------------------|-------|
| **1** | **CadQuery + 2D SIMP (DTU/FEniCS) + SU2 + BoTorch** | Free | Medium | **Low (scripted)** | **Full** | **Yes (all of it)** | Entire pipeline is Python scripts + config files. No GUI needed. 2D TO has correct formulation for thin ribs. |
| **2** | **CadQuery + BESO/CalculiX + SU2 + BoTorch** | Free | Medium | Medium | High | Yes (most) | BESO can be run headless via config files. CalculiX is CLI. |
| **3** | **CadQuery + Modified PyTopo3D + SU2 + BoTorch** | Free | Medium | Medium | High | Yes (with source mods) | Requires forking PyTopo3D to add custom BCs/loads. 3D TO only worthwhile for thicker ribs (4mm+). |
| **4** | **Fusion 360 + SU2 + BoTorch** | Free (edu) | **High (steep GUI learning curve)** | High (manual) | Low | No (Fusion is GUI-only) | User rejected this path. |

### 4.3 Recommended Stack: The Fully Scripted Pipeline

**Primary recommendation: CadQuery (geometry) + PyTopo3D (TO) + SU2 (CFD/ASO) + BoTorch (ML optimization)**

This stack is chosen because:

1. **CadQuery** -- Pure Python parametric CAD. Claude writes CadQuery scripts to generate rib geometry with any parameterization. Exports directly to STL/STEP. No GUI required. **Install via conda (recommended):** `conda install -c conda-forge cadquery`. Pip installation (`pip install cadquery`) requires pre-built OCP (OpenCASCADE) wheels that are only available for Python 3.9-3.12 on specific platforms; conda is the better-tested path. A conda environment is recommended for the entire project anyway since SU2 is also best installed via conda-forge.

2. **2D SIMP TO (primary) or modified PyTopo3D (alternative)** -- For rib topology optimization, the problem is reformulated as a **2D planform optimization** (see Section 9.2 for rationale). The primary TO tool is a 2D SIMP code based on the [DTU TopOpt Python codes](https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python) or the [FEniCS-based SIMP implementation](https://comet-fenics.readthedocs.io/en/latest/demo/topology_optimization/simp_topology_optimization.html), both of which support arbitrary boundary conditions and load definitions natively. Claude writes the complete problem setup. As an alternative, PyTopo3D (`pip install pytopo3d`) provides 3D SIMP on structured grids with STL import/export, but its API has **hardcoded boundary conditions** (standard cantilever beam only) and would require source code modification (~200-400 lines) to support custom BCs/loads for the rib problem. See Section 9.2 for details.

3. **SU2** -- Command-line CFD solver with built-in adjoint ASO. Config files are text-based -- Claude writes them. `conda install -c conda-forge su2` or build from source.

4. **BoTorch/GPyTorch** -- Python Bayesian optimization. Claude writes the entire optimization loop. `pip install botorch gpytorch`.

5. **Gmsh** -- Scriptable mesh generation. Python API available. Claude writes meshing scripts. `pip install gmsh`.

6. **CalculiX** (optional, for FEA verification) -- CLI-based FEA solver. Input files are text-based -- Claude writes them. Python wrappers available (pyccx, pycalculix).

### 4.4 Tool Details

#### CadQuery (Parametric Geometry Generation)

- **What it does:** Python library for creating parametric 3D CAD models. Based on OpenCASCADE (same kernel as FreeCAD).
- **Why it fits:** Fan ribs are simple extruded shapes with holes -- ideal for CadQuery. The entire rib parameterization (length, taper, thickness, pivot hole, cross-section profile) can be expressed in ~50 lines of Python.
- **Installation:** `conda install -c conda-forge cadquery` (recommended; pip requires pre-built OCP wheels only available for Python 3.9-3.12 on specific platforms)
- **Export:** STL, STEP, AMF, SVG
- **Claude can:** Write the complete parametric rib generator, modify parameters, generate variants for optimization studies.
- **Limitation:** No built-in visualization in headless mode. Use `cadquery-ocp` or export and view in a separate tool.

#### 2D SIMP TO (Primary TO Tool for Rib Planform Optimization)

- **What it does:** 2D plane-stress topology optimization using the SIMP method. Determines optimal material distribution (cutout patterns) in the rib's length-width plane at constant thickness.
- **Why it fits:** The rib TO problem is naturally 2D (see Section 9.2 for full rationale). A 2D formulation gives the optimizer full design freedom in the planform (hundreds of elements across the width) rather than being constrained to 4 voxels through a 2mm thickness in 3D.
- **Implementation options:**
 - **(a) DTU TopOpt Python codes** -- Educational 2D SIMP codes ([topopt.mek.dtu.dk](https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python)). Simple, self-contained, ~200 lines. Support arbitrary BCs and loads. Claude can modify directly.
 - **(b) FEniCS-based SIMP** -- The [comet-fenics TO tutorial](https://comet-fenics.readthedocs.io/en/latest/demo/topology_optimization/simp_topology_optimization.html) provides a clean 2D SIMP implementation with arbitrary BCs. More powerful (supports multi-load-case, stress constraints) but requires FEniCS installation. A [55-line FEniCS TO code](https://arxiv.org/abs/2012.08208) is also available.
 - **(c) ToPy** -- Python TO framework ([github.com/williamhunter/topy](https://github.com/williamhunter/topy)). Supports 2D and 3D problems with configurable BCs/loads via text config files.
- **Claude can:** Write the complete problem definition with custom BCs (rib's full-length y-edge fixity at the rib-panel interface — NOT pivot fixity, since the rib has no pivot hole under panel-pivot architecture) and loads (distributed pressure), run optimization, export results as coordinate arrays that feed into CadQuery for STL generation.
- **Limitation:** 2D formulation assumes constant rib thickness. Through-thickness features (if desired) require 3D TO.

#### PyTopo3D (Alternative: 3D TO with Source Modification)

- **What it does:** 3D SIMP topology optimization on structured voxel grids. `pip install pytopo3d` (Python 3.10+).
- **Actual API:** The real API is a single function `top3d(nelx, nely, nelz, volfrac, penal, rmin, disp_thres, obstacle_mask=None, use_gpu=False)` from `pytopo3d.core.optimizer`. There is NO class-based API, no `set_fixed_region`, no `set_distributed_load`, no `export_stl` method. The boundary conditions and loads are **hardcoded** to the standard cantilever beam benchmark (one face fixed, point load on opposite face).
- **To use for fan ribs:** Claude would need to **fork and modify** the PyTopo3D source code (~200-400 lines of changes) to support custom BCs and distributed loads. This is feasible since the code is pure Python and open-source, but it should be understood as a source-level modification, not a simple API call.
- **3D resolution problem:** At 2mm rib thickness with 0.5mm voxels, there are only 4 voxels through the thickness. This provides almost no design freedom in the thickness direction -- the optimizer can only produce 4 discrete thickness levels (25%, 50%, 75%, 100%), not true topology features like lightening holes (which require at least 3-4 voxels to form a void surrounded by material). This is why the 2D planform formulation is preferred.
- **Key features:** STL domain import, AM constraints (overhang angle), direct STL export, GPU acceleration.
- **Best use case:** If the rib is thickened to 4-5mm (allowing 8-10 voxels through thickness at 0.5mm resolution), 3D TO becomes meaningful. Otherwise, use 2D planform TO.

#### BESO + CalculiX (Alternative TO Path)

- **What it does:** BESO topology optimization driven by CalculiX FEA on unstructured meshes.
- **Why it fits:** Better mesh quality for thin ribs (tetrahedral elements conform to rib shape). Can be run fully headless via config files.
- **Claude can:** Write CalculiX input (.inp) files, BESO configuration files, mesh generation scripts via Gmsh Python API.
- **Limitation:** Post-processing pipeline (density field to STL) is manual and painful. See Section 8.3.1.

#### SU2 (Aerodynamic Analysis and Shape Optimization)

- **What it does:** CFD solver with built-in adjoint-based shape optimization. Handles incompressible and compressible flows, steady and unsteady.
- **Why it fits:** The only free tool that does proper adjoint-based ASO. Config files are text -- Claude writes them.
- **Claude can:** Write SU2 config files (hundreds of parameters), write mesh deformation configs, write FFD box definitions, set up the optimization pipeline, write post-processing scripts to parse results.
- **Limitation:** Steep learning curve for unsteady adjoint setup. Mesh generation (Gmsh) is the hardest part.

### 4.5 Head-to-Head: 2D SIMP vs. Modified PyTopo3D vs. BESO/CalculiX vs. Fusion 360

| Criterion | 2D SIMP (DTU/FEniCS) | Modified PyTopo3D | BESO/CalculiX | Fusion 360 |
|-----------|---------------------|-------------------|---------------|------------|
| **Cost** | Free | Free | Free | Free (with .edu) |
| **GUI required?** | No (pure Python) | No (pure Python) | Partially | Yes (GUI-only) |
| **Claude can write it?** | Yes -- 100% | Yes (with source mods) | Yes -- mostly | No |
| **Install effort** | `pip install scipy numpy` or FEniCS | `pip install pytopo3d` + fork | Multiple components | Register + download |
| **Custom BCs/loads** | **Yes (native)** | **No (requires source mod)** | Yes (CalculiX .inp) | Yes (GUI) |
| **TO formulation** | 2D plane-stress SIMP | 3D voxel SIMP | 3D BESO | Proprietary |
| **Thin rib handling** | **Excellent (natural 2D formulation)** | Poor (4 voxels through 2mm) | Good (tets conform) | Good |
| **Design freedom** | High (hundreds of elements across width) | Very low in thickness direction | High | High |
| **STL export** | Via CadQuery (density to geometry) | Direct (built-in) | Requires post-processing | Direct |
| **Best for** | **Thin rib planform optimization** | Thick parts (4mm+) | Complex 3D geometries | GUI users |

**Verdict:** 2D SIMP (DTU codes or FEniCS) is the correct formulation for a 2mm-thick fan rib. The rib is thin enough that through-thickness TO is meaningless (only 4 discrete levels possible), but planform optimization (where to place material in the length-width plane, creating cutout patterns and lightening holes) offers genuine design freedom. Claude writes the entire 2D SIMP script (~150-250 lines), and the optimized density field is converted to rib geometry via CadQuery.

For users who want 3D TO (e.g., for thicker ribs at 4-5mm), the modified PyTopo3D or BESO/CalculiX paths are viable but require more effort. All paths converge on SU2 + BoTorch for ASO and ML.

### 4.6 Hybrid Fusion + CadQuery

The pipeline maintains both backends for geometry: **Fusion 360 with a Python add-in** is the manufacturability path (parametric master file, manufacturable STL/STEP export, Fusion Simulation FEA), and **CadQuery** is the BO inner-loop path (sub-second regeneration for parameter sweeps and inner-loop geometry calls). Both backends read the same `params.json` schema, so any design is reproducible across either.

Manual steps that remain regardless of stack: visual inspection of Fusion regenerations, final manufacturable-geometry approval, physical assembly + pivot-pin insertion, anemometer + IMU + acoustic measurements. Everything else (TO, CFD, BO, FEA) is scripted from JSON.

---

## 5. Claude Code Delegation Map

### 5.1 What Claude Code Can Do

Claude Code is an AI coding assistant that runs in the terminal. It can write, modify, and help debug code. It CANNOT execute long-running simulations, interact with GUIs, or perform physical tasks. Here is a precise breakdown:

#### Tasks Claude Can Fully Automate (Write the Code, User Runs It)

| Task | Tool | What Claude Writes | Estimated Lines | Notes |
|------|------|-------------------|----------------|-------|
| **Parametric rib geometry** | CadQuery | Python script generating rib STL/STEP with all parameters | ~80-120 | Taper, thickness, pivot hole, cross-section profile |
| **Guard stick geometry** | CadQuery | Wider/thicker outer rib variants | ~60 | Variant of rib script |
| **Full fan assembly visualization** | CadQuery | Script placing all ribs at correct angles | ~100 | For visual verification |
| **TO problem setup** | 2D SIMP (DTU/FEniCS) | Python script defining 2D planform domain, loads, BCs, volume constraint | ~150-250 | Includes FE assembly, sensitivity filtering, OC update; load values from CFD results |
| **TO batch runner** | 2D SIMP + shell | Script running TO for multiple parameter sets | ~40 | For design space exploration |
| **CalculiX input files** | Text (.inp) | Complete FEA setup with nodes, elements, materials, loads | ~200-500 | If using BESO path instead |
| **BESO config files** | Text (.py) | Optimization parameters, domain selection, convergence criteria | ~50-80 | Headless execution |
| **Gmsh meshing scripts** | Gmsh Python API | 2D/3D mesh around fan geometry for CFD | ~100-200 | Boundary layer, refinement zones |
| **SU2 config files** | Text (.cfg) | Complete CFD setup: solver, BCs, objectives, FFD, adjoint | ~100-200 | Steady and unsteady variants |
| **SU2 optimization pipeline** | Shell + Python | Scripts to run SU2_CFD, SU2_CFD_AD, SU2_DOT, shape_optimization.py | ~30-50 | Wrapper scripts |
| **BoTorch optimization loop** | Python | Complete Bayesian optimization with GP surrogate | ~150-250 | Multi-fidelity, multi-objective |
| **CFD result parser** | Python | Parse SU2 history.csv, extract forces, pressures | ~50-80 | Feed into BoTorch |
| **FEA result parser** | Python | Parse CalculiX .frd/.dat files, extract stress/displacement | ~80-120 | Feed into TO or validation |
| **Post-processing / visualization** | Python (matplotlib, PyVista) | Plots of convergence, Pareto fronts, rib topology images | ~50-100 | Per visualization |
| **STL repair and smoothing** | Python (PyMeshLab, trimesh) | Automated mesh cleanup pipeline | ~60-100 | If using BESO path |
| **Multi-fidelity GP training** | Python (BoTorch) | GP fitting, cross-validation, uncertainty calibration | ~100-150 | |
| **Design of experiments** | Python (scipy.stats.qmc) | Latin Hypercube Sampling for CFD runs | ~30-40 | |
| **SfePy FEA scripts** | Python | Alternative FEA setup entirely in Python | ~100-200 | If avoiding CalculiX |

#### Tasks Claude Can Partially Automate

| Task | What Claude Does | What User Does |
|------|-----------------|---------------|
| **Mesh quality checking** | Writes script to compute mesh quality metrics | User reviews and decides if re-meshing is needed |
| **TO result interpretation** | Writes scripts to visualize and quantify TO results | User makes engineering judgment on which features to keep |
| **CFD validation** | Writes comparison scripts | User verifies results make physical sense |
| **Parameter tuning** | Suggests parameter ranges, writes sweep scripts | User evaluates results and decides next direction |
| **Debugging solver issues** | Analyzes error messages, suggests fixes | User applies fixes and re-runs |

#### Tasks the User MUST Do Manually

| Task | Why Claude Cannot Do It |
|------|------------------------|
| **Multi-material 3D printing** | Physical task requiring a dual-extruder/AMS printer |
| **Pivot assembly** | Physical: inserting pivot pin through the printed assembly, peening rivet or fastening nut |
| **Physical testing** | Waving the fan, anemometer measurements, IMU recording, smoke visualization |
| **Material testing** | Printing and testing tensile/bend specimens (PETG only); Spike 0.4 click-feature 1000-cycle test |
| **Visual inspection** | Checking print quality on each PETG blade, click-feature engagement, pivot fit, folded form factor |
| **Subjective evaluation** | How the fan feels to wave, comfort, aesthetics |

### 5.2 Recommended Workflow: Claude Writes, User Runs

The ideal workflow maximizes Claude's contribution:

```
Phase 0: Scaffold repo (Step 0.0) + Risk Spikes 0.1-0.7 + 0.6c + Baseline (Week 1-2)
 Claude writes: Fusion add-in smoke test, click-feature 1000-cycle test rig,
 Colab compute probe, J_fan post-processor, single-blade fab-noise
 study, torsional-pendulum protocol, generative-geometry sanity-check
 script (Spike 0.7a), BO-infrastructure scaling sanity check (Spike 0.7b)
 User runs: spike scripts (7 spikes); reports go/no-go for each
 User does: prints baseline 10-blade SOLID-panel fan;
 anemometer + IMU measurement (using Spike 0.2 inertia rig);
 click-feature cycle test on 2 single-blade test articles;
 3-copy single-blade fab-noise CV;
 generative-geometry sanity print (2 of 10 random parameter sets)

Phase 1: 4-Layer Generative Blade Geometry Pipeline (Week 3)
 Claude writes: CadQuery generative blade generator (§9.7) —
 make_outer_envelope + apply_boolean_subtractions +
 add_surface_features + manufacturability_check + export_stl;
 JSON schema (~37-46 vars for the 4-layer hybrid) with load-time validation;
 Fusion add-in for multi-blade assembly view
 User runs: python generate_blade.py --params baseline.json --emit-stl
 python schema_validator.py
 User does: visual inspection of one generated blade in Fusion

Phase 2: Rib-Only Plate-Bending TO (Week 4)
 Claude writes: FEniCSx 2D plate-bending SIMP solver (Reissner-Mindlin) —
 ribs only, NOT panel; stress-constrained pivot; preserved
 click-feature footprint; multi-load-case; rigid-blade gate check
 User runs: python aero_loads.py (pulls Phase 3 baseline CFD pressures)
 python rib_to_solver.py (5-30 min per rib)
 python density_to_rib.py --> rib.dxf
 python verify_rib.py (Fusion Sim)
 User does: prints one optimized rib; static deflection vs FEA

Phase 2b: 4-Layer Hybrid Generative Optimization Seed (Week 5-8)
 Claude writes: generative parametric panel optimization infrastructure —
 2D STEADY CFD configs (Tier -1, ), 3D steady CFD ranking
 (Tier 0), 3D unsteady CFD (Tier 1); BO seed loop with
 ~30 LHS Tier -1 + ~30 LHS Tier 0 + ~5 LHS Tier 1
 per representative architecture
 User runs: python mf_panel_bo.py --seed (runs LHS on Colab Pro)
 User does: hand off seed dataset to Phase 4

Phase 3: 2D CFD Slice (Week 6; runs in parallel with Phase 2/2b)
 Claude writes: 2D Gmsh (resolves corrugated surface + Boolean subtractions),
 SU2 unsteady configs (locked dt=T/200, 5 cyc), canonical
 j_fan.py, steady-vs-unsteady correlation analysis,
 dt and cycle independence checks
 User runs: SU2 on 8-12 designs (rigid corrugated, NO FSI)
 python steady_unsteady_corr.py --> mf_prior.json
 Check R² >= 0.4 hard gate; otherwise drop steady fidelity

Phase 4: Multi-Fidelity BO at ~37-46D on Colab Pro (Week 8-11)
 Claude writes: 3D Gmsh (rigid corrugated + subtractions), SU2 3D configs with
 checkpointing, Colab orchestrator with parallel sessions,
 architecture bandit (K=3-5 data-driven) + continuous TuRBO inner
 loop, SAASBO alternative with ≤500 inducing points,
 Drive/JSONL ledger, Pareto-to-Fusion regenerator
 User runs: python mf_bo_turbo.py --init
 2-4 parallel colab_runner.ipynb sessions
 python mf_bo_turbo.py --iterate
 python pareto_to_fusion.py (top-3 in Fusion: light/knee/heavy)

Phase 5: Verification + PyFR Cross-Solver on Top-3 (Week 11-12)
 Claude writes: high-fidelity Gmsh (fully resolved corrugated + subtractions),
 SU2 verification configs, PyFR top-3 verification, verified
 reranking
 User runs: SU2 verification on top-3
 python pyfr_top3.py (Colab GPU)
 python rerank_verified.py --> final_designs.json (top 3)

Phase 6: Single-Material Print TOP-3 + IMU + Acoustic Validation (Week 12-13)
 User does: per-blade PETG print of each of 3 Pareto designs,
 assembles each, pivot stack mechanical check (N2),
 folded form factor verification (N3),
 3-copy single-blade fab-noise recheck against Spike 0.5,
 IMU instrumentation, anemometer at 300 mm,
 angular work per cycle for each design,
 click-feature long-cycle test,
 acoustic measurement (microphone at 300 mm),
 drop test (light)
 Claude writes: IMU CSV post-processor, acoustic FFT analyzer,
 model-calibration scripts triggered by physical results
 User chooses: preferred design from the 3 based on measured numbers + subjective
 feel
```

### 5.3 Estimated Scripting Effort for Claude
For the full project, Claude would write approximately (organized under the §12 package layout):

| Location | Content | Approx. lines |
|----------|---------|---------------|
| `src/fanopt/geometry/` | Layer 1 envelope + Fourier; Layer 2 5-field library (louver, texture, edge, noise, TPMS); Layer 3 capped primitive; §N7 manufacturability filter; generator orchestration; JSON schema validator | ~800-1200 |
| `src/fanopt/topopt/` | Reissner-Mindlin plate-bending element + assembly; SIMP material interpolation + Helmholtz filter; OC update loop; multi-load-case (push/return/inertial/click) | ~400-600 |
| `src/fanopt/cfd/` | Gmsh wrappers (2D slice + 3D corrugated); SU2 .cfg Jinja2 template generators; subprocess + Colab-checkpointing runner; **canonical `j_fan.py`** (§9.4 locked spec); SU2 history.csv + VTU parsers | ~700-1000 |
| `src/fanopt/bo/` | `SingleTaskMultiFidelityGP` + `qMFKG`; architecture bandit (combined Tier -1/0 promotion); TuRBO trust regions; SAASBO fallback; `qNoisyExpectedHypervolumeImprovement` 4D Pareto; orchestration | ~600-900 |
| `src/fanopt/physical/` | IMU angular-work-per-cycle; acoustic FFT; anemometer J_fan extraction | ~150-250 |
| `src/fanopt/utils/` | **Drive/JSONL ledger** — per-session JSONL append, marker files, heartbeat, round-robin slicing, listing cache; Colab session helpers; structured logging | **~150-200** (down from ~200-300; no SQLAlchemy) |
| `tests/` | pytest test suite (geometry golden, plate-bending cantilever benchmark, j_fan synthetic, BO Branin, IMU known waveform, acoustic known tones) | ~600-1000 |
| `scripts/run_*.py` | Per-spike and per-phase CLI entry points | ~200-300 |
| `notebooks/` | ≤50-line orchestrators (Colab Phase 4 runner, Pareto analysis, geometry inspection, physical results) | ~150-200 |
| `configs/su2/*.cfg.j2` | 4 Jinja2 SU2 templates (2D steady, 2D unsteady, 3D steady, 3D unsteady) | ~400-600 (config text) |
| `configs/fusion/fan_script.py` | Fusion add-in for multi-blade assembly view | ~200-300 |
| **Total** | | **~4,400-6,650 lines** |

Compared with earlier estimates of 3,500-5,000 lines, the increase reflects the 4-layer generative geometry expansion (more files in `src/fanopt/geometry/`) plus the proper test pyramid (~600-1000 lines of pytest code). Notebooks are kept thin per §12.2.

This is well within Claude Code's capabilities. The user's primary effort is running the scripts, interpreting results, performing physical tasks, and approving Fusion regenerations.

---

## 6. ML Surrogate Modeling: Core Workflow

### 6.1 Why ML Surrogates Are Important

**ML for ASO (Aerodynamic Shape Optimization):** A single CFD simulation of the deployed fan takes 30 minutes to 4 hours. Exploring the full design space (rib count, spread angle, camber profile, rib curvature) via direct CFD would require thousands of simulations. An ML surrogate replaces the CFD solver with a model that predicts aerodynamic performance in milliseconds.

**ML for TO (Topology Optimization):** For a single rib, TO runs in minutes to hours via PyTopo3D or BESO. ML acceleration is less critical here because the rib TO is a relatively small problem (compared to a full fan blade). However, if exploring many rib cross-section variants or coupling TO with ASO iteratively, ML can accelerate the inner loop.

**Recommendation:** ML-for-ASO via BoTorch is the core ML component. ML-for-TO is an optional extension for this project.

### 6.2 ML for ASO (Folding Fan Parameters)

#### 6.2.1 Design Parameters
The panel parameterization is a 4-layer hybrid (~37-46 vars), restructured from an earlier flat ~45-55-var Boolean-subtraction list to address three reviewed concerns: BO can't reliably converge correlated independent primitives ("Mr. Potato Head"), rigid fans need direct access to asymmetric-drag design families (louvers), and OpenCASCADE Boolean operations on arbitrarily-placed primitives fail too often for automated BO loops.

**Layer 1 — Outer envelope + Fourier modulation (~14 vars)**

| Parameter | Count | Range | Description |
|-----------|-------|-------|-------------|
| ~~Fan macro: spread angle~~ DELETED per C8 lock | — | — | **Spread angle is NOT a BO parameter** — it's derived from `blade_count × 13.3°` per C8 lock (§0 row 133). Values: 8 → 106.4°, 10 → 133.0°, 12 → 159.6° (14 trimmed per MED-10). Re-introducing spread_angle as a free parameter would violate the C8 lock and undermine the panel-edge meeting at the pitch boundary (item #35); future BO modifications must respect this. |
| Fan macro: blade count | 1 | {8, 10, 12} (MED-10 trim: 14 removed for ergonomic infeasibility — 186.2° past straight-line) categorical | Architecture-bandit variable |
| Fan macro: blade length | 0 (pinned) | **200 mm (pinned, not a BO variable)** | The locked SU2 numerics (V_tip = 2.20 m/s, REYNOLDS_NUMBER = 37000, k = 0.57, the §9.4.1 config-hash assertion, the §3.2.4 BL thickness, and §Phase 3 V_mid = 1.32 m/s) all assume L_blade = 200 mm exactly. If the BO moved L_blade off 200 mm, every Re/V/k value would be wrong AND the per-tier config-hash assertion would fail. L_blade is pinned at 200 mm and is not a BO variable. Re-introducing it would require re-deriving every locked CFD number per-design and rewriting the assertion to check per-design — out of scope for V1. |
| ~~Rib base width~~ DELETED per H12 lock (HIGH-10 Round-9) | — | — | **H12 locked constant: `RIB_BASE_WIDTH_M = 0.004 m = 4 mm`** (see GEOMETRY_LOCKS). NOT a BO parameter. The earlier 8-15 mm BO range would force `panel_width(r) = r·0.232 − 2·rib_base − 0.5 mm` negative at the rib base: at rib_base = 8 mm, `panel_width(r=0.07) = 16.24 − 16 − 0.5 = −0.26 mm` — CAD generator crashes. The H12 lock at 4 mm gives `panel_width(root) = 16.24 − 8 − 0.5 = 7.74 mm` (positive). |
| ~~Rib tip width~~ DELETED per H12 lock (HIGH-10 Round-9) | — | — | **H12 locked constant: `RIB_TIP_WIDTH_M = 0.006 m = 6 mm`**. NOT a BO parameter. The H12 UP-taper (narrow root → wider tip) is geometrically motivated by the widening angular pitch and structurally acceptable because the rib base at `r = HUB_RADIUS = 0.020 m` has a short cantilever lever arm. |
| Chordwise camber spline knots (plano-convex constraint) | 3-4 | 0-5 mm, **applied to the top face only when `print_orientation = rib-flat` (the §7.4.3 default)**; both faces may curve only when `print_orientation = deployed-V` | Panel camber. **Plano-convex:** with rib-flat + camber > 0, a non-plano-convex envelope would put the bottom face up to 5 mm above the build plate at chord midpoint → ~4500 mm² support footprint per blade × 10 blades, and the post-support-removal Ra ≈ 100-300 µm would break the §3.2.4 wall-roughness calibration. constrains the bottom face (z = 0 in the build frame) to be a single planar surface and applies camber + thickness variation to the top face only. Lost design freedom: S-curve and double-cambered envelopes; preserved: symmetric and single-camber airfoils — which is what flow physics at k ≈ 0.57 cares about. CadQuery construction: "extrude the planform from a flat base, then loft a top-face spline over the 3 camber control points." The constraint relaxes for `print_orientation = deployed-V` (lower-priority print mode per §7.4.3). |
| Twist distribution knots | 2-3 | -10° to +10° | Spanwise twist |
| **Thickness profile control points** | 3 | **2.2-3.8 mm** | Per-element thickness; enables 3D-carved features within the folded-stack collision floor. **Direct lever on `I_wrist` Pareto objective (§6.4):** thickness near the blade tip enters I_wrist with an r² weight (where r ≈ L_blade + d_handle ≈ 0.25 m at the tip), so thickness gradients across the 3 control points directly trade J_fan against rotational inertia. The optimizer will typically push the tip control point toward the lower end of 2.2-3.8 mm unless the tip thickness materially improves J_fan. Hard upper bound 3.8 mm = 2·rib_thickness − 0.2 mm folded clearance. |
| Edge profile category | 1 | {sharp, rounded, mildly-serrated} categorical | Default edge shape family |
| **Fourier LE harmonic amplitudes ** | 3 | bounded so envelope stays within ±15% of mean | k=1, 2, 3 amplitudes; phases fixed at 0, π/3, 2π/3 |
| **Fourier TE harmonic amplitudes ** | 3 | bounded so envelope stays within ±15% of mean | k=1, 2, 3 amplitudes; phases fixed |

Fourier modulation lets the blade silhouette evolve into bat-wing, leaf-like, scalloped, or jagged outer shapes rather than being constrained to a smooth airfoil. Aerodynamically meaningful: 3rd harmonic on TE produces owl-wing-style serrations that reduce vortex-shedding noise; asymmetric LE creates directional drag bias.

**Layer 2 — Macro-pattern + procedural math fields (~15-20 vars active, 0-3 fields per design)**

The optimizer activates 0-3 fields per design from a library of FIVE field types. Each field has a binary activation flag (categorical, handled by the architecture bandit). Inactive fields contribute nothing to the geometry.

| Field type | Vars per field | Description |
|------------|----------------|-------------|
| **(a) Louver** | ~5-6 | Cut/rib count (3-12, rounded), cut/rib angle (±60°), cut/rib width (0.5-3 mm), spacing distribution categorical {uniform, clustered-at-tip, gradient-toward-LE}, polarity {subtract=slits, add=ribs}. Directional design family for asymmetric drag. |
| **(b) Texture** | ~5 | Feature type categorical {dimple, ridge, bump}, density (features/cm²), characteristic size (0.5-3 mm), orientation angle (±90°), polarity {add, subtract}. Distributed boundary-layer features. |
| **(c) Edge feature** | ~3-4 | Feature type categorical {serration, scallop, smooth-fade}, count (rounded integer), depth (0.5-3 mm), application categorical {LE, TE, both}. Trailing/leading-edge shapers. |
| **(d) Noise threshold ** | ~5 | X-scale, Y-scale, rotation (±90°), X-offset, threshold value (constrained to retain ≥40% material). 2D Perlin or Simplex noise stamped through the blade; voxels where noise > threshold are subtracted. Produces bone/coral/sponge-like organic topology. |
| **(e) TPMS ** | ~5 | Lattice type categorical {gyroid, schwarz-diamond, off}, cell size (≥3× min feature size), X/Y/Z rotation, thickness gradient. Triply Periodic Minimal Surface for **directional through-blade porosity** (redefinition from the earlier "internal-infill" reading). The TPMS surface intersects the panel envelope, **creating directional through-cuts; outer skin NOT preserved where TPMS is active.** Self-supporting (no overhangs >45° anywhere in the lattice). |

**All Layer 2 fields are safe-by-construction:** parameter ranges mathematically guarantee features stay within the envelope (≥1 mm margin from edges, ≥0.8 mm minimum feature size, etc.) and produce coherent CadQuery geometry without try/except.

**Layer 3 — Capped 0-1 independent primitive (~5-7 vars)**

For asymmetric point features that don't fit any pattern family. Capped at 0-1 primitives per design to preserve dimensionality budget; primitives are NOT the primary topology mechanism .

| Parameter | Count | Range | Description |
|-----------|-------|-------|-------------|
| Number of primitives | 1 | {0, 1} categorical | 0 = no primitive |
| If active: shape type | 1 | {slot, ellipsoid, wedge} categorical | (smaller shape library than the earlier {sphere, ellipsoid, slot, cylinder}) |
| If active: polarity | 1 | {add, subtract} categorical | |
| If active: position xyz | 3 | constrained ≥1 mm from envelope edges | |
| If active: size (1-3 dims) | 1-3 | each ≥0.8 mm, ≤30% local envelope | |
| If active: rotation | 1-3 | full 6-DOF | |

Wrapped in try/except in the §9.7 pipeline — the only step where CAD failures can occur. Failure → log + skip the primitive, return geometry from prior steps.

**Layer 4 — Manufacturing + click features (~3-5 vars)**

| Parameter | Count | Range | Description |
|-----------|-------|-------|-------------|
| Print orientation per blade | 1 | categorical {flat, edge, custom-angle} | FDM anisotropy direction |
| Layer height | 1 | categorical {0.1 mm, 0.15 mm, 0.2 mm} | Slicer setting |
| Click chamfer angle | 1 | 30-60° | Mating-face chamfer |
| Click detent size | 1 | 0.3-0.5 mm radius | 0 if magnetic upgrade |
| Click design clearance | 1 | **0.15-0.20 mm** | Per mating surface; matches the Spike 0.4 tolerance-validation range (§2.1 and §7.4.1 also lock 0.15-0.20 mm). |

**Total: ~37-46 design variables, depending on which Layer 2 fields are active.** Down from an earlier 45-55. Bounded by architecture bandit + TuRBO + multi-fidelity GP capability.

**Discrete handling:** blade count, Layer 2 field activation flags (5 binary), Layer 2 per-field categoricals, Layer 3 primitive count/type/polarity, edge profile, print orientation, layer height are categorical/discrete. The architecture bandit (§6.2.2) handles them; the bandit promotes a fixed K=4 architectures from **Tier -1 2D-steady-CFD screening** to inner-loop 3D CFD. With 0-3 Layer 2 fields active per architecture, the inner-loop continuous subspace is typically ~20-30 dims.

**Dimensionality implications:** ~37-46 total dims (with ~20-30 in the continuous-only inner subspace per architecture). Spike 0.7b validates GP fits at this scale complete in ≤60s/iteration before Phase 4 launches. SAASBO with ≤500 inducing points is the fallback if TuRBO under-explores.

#### 6.2.2 Recommended Model: Architecture Outer Loop + TuRBO Continuous Inner Loop

Naïve TuRBO with `MixedSingleTaskGP` and naïve SAASBO have known failure modes at this scale; the architecture below revises them.

**Why the naïve TuRBO + MixedSingleTaskGP combination is wrong:** TuRBO's trust regions are continuous hyperrectangles that assume local correlation. Categorical jumps (rib count 12 → 15, surface feature type "none" → "vortex generator") break that local-correlation assumption: changing rib count requires an entirely different CFD mesh and the function value can change discontinuously. Forcing categorical variables into a trust region produces a trust region that is meaningless for those dimensions.

**Why naïve SAASBO at large N stalls:** Fully-Bayesian GPs (SAASBO uses NUTS over the GP hyperparameters) are O(N³) per Cholesky step and NUTS draws hundreds of samples. At N=2000 training points, NUTS won't finish in reasonable time. Practical limit for `SaasFullyBayesianSingleTaskGP` is ~500 points before sampling stalls. stays well under this limit — Tier -1 is **2D steady CFD at 30 evals/architecture**; the ≤500-inducing-point cap is preserved as a defensive measure if SAASBO is exercised on a larger pooled dataset.

**Architecture:**

1. **Outer loop: categorical "architecture" search .** Categorical/discrete variables — blade count ∈ {8, 10, 12} (MED-10 trim: 14 removed for ergonomic infeasibility — 186.2° past straight-line), edge profile category ∈ {sharp, rounded, mildly-serrated}, blade print orientation ∈ {flat, edge, custom-angle}, layer height ∈ {0.1 / 0.15 / 0.2 mm}, plus the 5 Layer 2 field activation flags (each binary, with max 3 active), plus Layer 2 per-field categoricals (louver polarity / texture type / edge feature type / TPMS lattice type), plus Layer 3 primitive presence + type — are enumerated into a set of *architectures*. With the 4-layer structure the architecture space is on the order of **~40-120 distinct architectures after the pruning rule below**. Each architecture has its own CFD mesh and its own continuous-only design space.

 **Architecture-count pruning rule (must hold before Phase 4 launches; enforced in `architecture_enumerate.py`):** the full Cartesian product (blade_count × edge_profile × print_orientation × layer_height × C(5,0)+C(5,1)+C(5,2)+C(5,3) Layer 2 activation profiles × per-field categoricals × Layer 3 primitive presence/type) is in the **10⁴-10⁵** range, which would blow the Tier -1 compute budget (a week+ at 30 evals × 5 min). To stay at ~80-130 architectures the enumerator applies:
 - **(a) Layer 2 activation profiles widened to 20 explicit combinations (H1 lock — earlier draft capped at 5, but louver `count_min=3`, noise `material_retention≥0.40` and TPMS `cell_size_min` cannot decay to inactive via continuous-near-zero, so 3 of 5 fields are categorically unreachable except via the activation flag).** Starting from the full 32 = 2⁵ combinations: drop the 6 combinations at {4-active, 5-active} (no Layer 2 signal would be tractable — too many fields interacting); drop the 4 combinations containing both {noise, TPMS} (both 3D-coherent, both suppressed at Tier -1 → no Layer 2 signal at screening); optionally drop weak singletons {texture-only, edge-only} (low expected design value per §6.2.4 rationale). Result: 32 − 6 − 4 − 2 = **20 production Layer 2 activation profiles**, committed in `configs/architecture_enumeration.yaml`. Pre-commit gate asserts the file's profile count = 20.
 - **(b) Layer 2 per-field categoricals treated as inner-loop categoricals**, not architecture-level. Louver polarity / texture type / edge feature type / TPMS lattice type each become a 1-hot inner-loop variable handled by TuRBO's continuous relaxation (or the architecture-bandit can re-shard within an architecture if the inner-loop GP shows clear sub-cluster separation).
 - **(c) Layer 3 primitive type** is the only inner-loop categorical when Layer 3 primitive presence = 1.
 - **(d) Concrete enumeration (H1 retuned + MED-10 trim):** with **3 blade counts (MED-10 trim: 14 removed)** × 3 edge profiles × 3 print orientations × 3 layer heights × **20 Layer 2 activation profiles** × 2 Layer 3 primitive presence = **3,240 raw combinations** — pruned to **60-100** by collapsing weakly-distinguishing axes (fix print_orientation per blade_count by Spike 0.4 bed-size decision; enumerate only layer_height ∈ {0.15, 0.2}). If the post-prune count exceeds 150, the budget guard at (e) blocks Phase 4 launch until pruning is tightened (e.g., freeze edge_profile or further reduce layer_height options). The exact enumeration plan is committed to `configs/architecture_enumeration.yaml` *before* Phase 4 begins and is validated against the Tier -1 compute budget (≤3,600 evals total at 10 min each per two-eval delta cost). **Pre-commit hook:** `.git/hooks/pre-commit` refuses commits to `configs/architecture_enumeration.yaml` that increase the architecture count by >10% over `HEAD` without an explicit `--allow-arch-bandit-growth` flag. **Inert before Phase 4 launch:** the 10% growth gate is **disabled until the `phase4-launch` Git tag is created** (`git tag phase4-launch` runs from the Phase 0 → Phase 1 handoff script in `scripts/launch_phase4.py`). Before that tag exists, the hook returns 0 unconditionally because the enumeration is still being authored and pre-launch growth is expected. After the tag exists, the hook reads `HEAD~1`'s enumeration count and enforces the +10% ceiling. The hook also re-runs the cost estimate and warns if the new enumeration exceeds the §0 compute-budget upper bound. Architecture-count budget guard: `len ≤ max(current_count × 1.1, 50)`.
 - **(e) Budget guard:** if the enumerator produces >150 architectures, Phase 4 launch is blocked until pruning is tightened or the Tier -1 budget is renegotiated. Phase 4's launch script (`scripts/run_phase4_bo.py --init`) asserts `len(architectures) <= 150` and refuses to start otherwise.
2. **Inner loop per architecture: continuous TuRBO.** Inside each fixed architecture, run standard continuous TuRBO over the **~20-30 continuous dimensions** (Layer 1 envelope + Fourier continuous + Layer 2 active-field continuous params + Layer 3 primitive continuous params if active + Layer 4 click continuous + fan-macro continuous). 3-5 parallel trust regions; standard `SingleTaskMultiFidelityGP` (no Mixed wrapper needed).
3. **Bandit allocation across architectures .** With ~40-120 categorical combinations (blade count × Layer 2 field activations × Layer 2 per-field categoricals × Layer 3 primitive presence × edge profile × print orientation × layer height), brute-force enumeration at high fidelity is wasteful. Run a UCB / Thompson-sampling bandit: each architecture gets initial **2D-steady-CFD screening (Tier -1, ) at 30 evals per architecture**; the bandit then runs **~5 Tier-0 (3D steady) evals per candidate before promotion**; combined Tier -1 + Tier 0 rank score promotes K=4 architectures to inner-loop CFD with Tier 0 + Tier 1.

**Bandit hyperparameter defaults (Phase 4, + amendments):**

| Stage | Hyperparameter | Default value | Notes |
|-------|---------------|---------------|-------|
| **Tier -1 screening ** | Evaluations per architecture | **30** (, down from 200) | LHS in the architecture's continuous subspace; 2D steady CFD is fast enough at this budget |
| **Tier 0 pre-promotion** | Evaluations per candidate before bandit decision | **~5 Tier-0 evals per promoted candidate** | : promotion uses Tier -1 + Tier 0 *combined*, not Tier -1 alone, to avoid the "Tier -1 2D-CFD porous-penalty bias" |
| Tier -1 + Tier 0 → Tier 1 promotion | Promote top-K by combined-rank score | **K = 4 default; data-driven adjustment from the Phase 3 R² gate:** if Phase 3 (steady ↔ unsteady) R² **≥ 0.6** → K = 3 (cheap fidelity tracks the truth well; K = 4 hedge is partially wasted); if **0.5 ≤ R² < 0.6** → K = 4 (default); if **R² near the 0.4 floor** → K = 5 (cheap fidelity is barely informative; hedge harder). | Saves ~20-25% Phase 4 compute when the screening fidelity is trustworthy; preserves the hedge when it isn't. ( fixed K = 4; makes it data-driven without dropping below the original hedge floor.) |
| Acquisition cap per architecture | TuRBO inner-loop iterations | **35 hard ceiling (, down from 50)** with the existing early-stop (UCB-improvement < 3% over 5 iters) still firing first | The 50-cap only bit on architectures that never converged within 50 iters; lowering to 35 saves compute on those without affecting the early-stop majority. Risk: ~10-20% of architectures in 20-30D continuous spaces are still marginally improving between iters 35-50; usually sub-1% absolute J_fan gains. The K = 4 hedge covers the "next-best instead of best" case. |
| Inner-loop Tier 0/1 allocation per promoted architecture | Evaluations | ~30 Tier 0 + ~10 Tier 1 + adaptive | Stop early if UCB-improvement < 2% over 10 consecutive iterations |
| Top-3 Pareto promotion to Phase 5 | Top-K by Pareto coverage (light/knee/heavy) | 3 designs | Hard cap (compute-driven) |
| UCB confidence width | β | 2.0 | Standard for UCB1 |

Defaults are configurable in `params.json`; Claude tunes them once Phase 4 runs reveal the actual Tier -1 / Tier 0 / Tier 1 correlation strength.

**— Combined Tier -1 + Tier 0 promotion (explicit):** Architecture promotion decisions weight Tier -1 mean rank and Tier 0 mean rank equally, not Tier -1 alone. Each promoted-candidate architecture gets ~5 Tier-0 evals before the bandit decision is finalized. **Known limitation — Tier -1 2D-CFD "porous-penalty bias":** the 2D steady CFD slice penalizes designs with many Boolean cutouts more harshly than the true 3D unsteady physics would, because in 2D the air "leaks through" with no opportunity for the wake to reattach over the next pitching cycle. Combining Tier -1 with Tier 0 (3D steady, full geometry) before promotion mitigates this bias. The 2D-unsteady fidelity is reserved as an inner-loop verification fidelity (Phase 3 slice + ad-hoc verification calls), not as part of the architecture-screening Tier -1.

**— Tier-1-restricted geometry asymmetry (explicit; ties to §Phase 3 step 33's `generate_blade_for_2d_slice` guard):** For architectures with Layer 2 TPMS or noise-threshold active, Tier -1 evaluates a geometry that has **those fields suppressed** (the 2D slice would be topologically disconnected otherwise — see Phase 3 step 33). Tier 0 / Tier 1 evaluate the full geometry. The multi-fidelity GP correlation is therefore weaker for TPMS-active or noise-active architectures: Tier -1 systematically over-predicts J_fan for these designs (no through-blade leakage in the screening fidelity). **GP-kernel concession:** the multi-fidelity GP's correlation kernel assumes the low-fidelity surrogate is a *noisy version of the truth on the same design*. For TPMS/noise architectures, Tier -1 evaluates a *different geometry* than Tier 0/1, so this assumption is violated — the multi-fidelity GP is mis-specified on these architectures. **Per-architecture promotion reweighting** (Tier 0 weight 0.7 / Tier -1 weight 0.3, vs. default 0.5/0.5 for 2D-slice-coherent architectures) reduces Tier -1's influence on promotion decisions but does *not* fix the kernel.

**Empirical-bias diagnostic (L7 lock — validates the 0.3/0.7 reweighting against actual data):** within the first 100 Tier-0 evaluations completed in Phase 4, compute `Δ_TPMS = mean(J_fan_tier0_TPMS) − mean(J_fan_tier_minus_1_TPMS_with_TPMS_suppressed)` and `Δ_noise` analogously. Log to `phase4/diagnostics/tier_minus_1_bias_<date>.json`. **Action rule:** if `|Δ_TPMS / mean_J_fan_tier0| > 0.30` or `|Δ_noise / mean_J_fan_tier0| > 0.30`, the 0.3/0.7 reweighting is insufficient; the orchestrator switches the affected TPMS/noise architectures to **Tier-0-only promotion (no Tier -1 input)** and writes the decision to `phase4/diagnostics/PROMOTION_REWEIGHT_<date>.json`. Makes the reweighting empirically validated rather than assumed.

Two cleaner alternatives are documented but not adopted (both V2 candidates): (a) **disable multi-fidelity GP** for TPMS/noise architectures and run Tier-0-only (single-fidelity), or (b) **treat Tier -1 as a separate cheap-feature input** rather than a fidelity column (e.g., concatenate `J_fan_tier_minus_1_proxy` as an extra GP input dimension instead of a fidelity tag). The chosen design uses the reweighted-promotion compromise + L7 empirical bias check because it preserves the existing multi-fidelity infrastructure with a one-line change to `architecture_bandit.py`; the rank-correlation diagnostic surfaces residual mis-specification for TPMS/noise architectures (expect their Tier-(-1) ↔ Tier-0 ρ to be the lowest in the campaign).

```python
# Outer loop: bandit over architectures + combined Tier-1/Tier-0 promotion + inner continuous TuRBO.
# (Schematic; full implementation is in mf_bo_turbo.py.)

architectures = list(product(
 blade_counts=[8, 10, 12]  # MED-10 trim: 14 removed,
 edge_profile_category=['sharp', 'rounded', 'mildly-serrated'],
 print_orientation=['flat', 'edge', 'custom-angle'],
 layer_height=[0.1, 0.15, 0.2],
 layer2_field_activations=[...combinations with ≤3 active flags from 5-field library...],
 layer3_primitive_count=[0, 1],
 # plus per-field categoricals (louver polarity, texture type, etc.)
))

# Per-architecture SIMP rib-TO cache: SIMP outputs depend on `print_orientation` (which sets
# the orientation-dependent K_t and σ_allow tables; see §3.1.7). When the architecture bandit
# switches print_orientation, the SIMP TO must re-run for the new orientation — the cache key
# is (architecture_id, design_seed). The orchestrator looks up SIMP results per-architecture
# from this cache before dispatching CFD; a cache miss triggers a fresh SIMP solve.
simp_cache = {} # keyed by (arch_id, design_seed) → (rib_density_field, peak_stress, ...)

def get_rib_for_arch_and_design(arch, design_seed):
 key = (arch_id(arch), design_seed)
 if key not in simp_cache:
 # Pass the orientation-dependent allowable (σ_y_XY for rib-flat, σ_y_Z for edge/custom)
 # so SIMP minimizes compliance subject to the correct fatigue ceiling.
 simp_cache[key] = run_simp_rib_to(arch=arch, design_seed=design_seed)
 return simp_cache[key]

# Stage 1: cheap 2D-STEADY CFD screening (Tier -1) on every architecture (30 evals each).
tier_minus1_scores = {a: tier_minus1_screen(a, n=30) for a in architectures}

# Stage 2: rank by Tier -1 mean, take top ~8 candidates, run ~5 Tier-0 evals each.
top_candidates = top_k_by_mean(tier_minus1_scores, k=8)
tier0_pre_promotion = {a: tier0_eval(a, n=5) for a in top_candidates}

# Stage 3: combined-rank promotion to K=4 architectures for inner-loop CFD BO.
combined_scores = combine_ranks(tier_minus1_scores, tier0_pre_promotion, weights=(0.5, 0.5))
K_promoted = phase3_k_decision(phase3_r2) # data-driven K: 5 if R²<0.5, 4 if 0.5≤R²<0.6, 3 if R²≥0.6
promoted = top_k(combined_scores, k=K_promoted)
inner_bo_results = {a: turbo_continuous_with_mf(a, tiers=[0, 1], budget=40) for a in promoted}

# Stage 4: top-3 by Pareto coverage (light/knee/heavy) advance to Phase 5 verification.
finalists = pareto_top3_coverage(inner_bo_results)
```

**Why this is better:**
- No mixed-variable trust regions; categorical jumps live at the outer loop where they are evaluated as independent campaigns rather than local moves.
- Each architecture has its own GP fit on its own (modest) number of training points -- well within ARD-Matern's comfort zone.
- Tier -1 (**2D steady CFD**) data is used at the *architecture-screening* tier and at the prior-mean for the GP within each architecture, not as raw training points for SAASBO. (The 2D-unsteady fidelity is reserved as inner-loop verification, not the screening tier.)

**SAASBO fallback (revised):** SAASBO remains an option as the per-architecture inner-loop model *if* TuRBO struggles. When used, **downsample Tier -1 data to ≤500 inducing points** before feeding into `SaasFullyBayesianSingleTaskGP`. The downsampling can be uniform or biased toward high-J_fan regions (importance sampling).

**Engineering decisions logged here:**

- **Decision:** outer loop bandit + inner continuous TuRBO replaces single-monolithic TuRBO + MixedSingleTaskGP.
- **Reason:** mixed-variable trust regions break local-correlation assumptions at categorical jumps; this is a known TuRBO failure mode for categorical+continuous design spaces.
- **Decision:** Tier -1 **2D-steady-CFD** data feeds the architecture bandit (cheap screening at 30 evals/arch) and the per-architecture GP prior mean. combines Tier -1 + Tier 0 rank scores for the promotion decision.
- **Reason:** fully-Bayesian SAASBO at N=500+ training points stalls NUTS; capping at ≤500 inducing points keeps NUTS feasible.

**Why GP-family at all (not neural network):** Realistic per-architecture budget is ~30 steady + ~10 unsteady evaluations. GPs (with or without the SAASBO prior) remain correct for small datasets and integrate with multi-fidelity natively. Neural networks would underfit this regime.

#### 6.2.3 Multi-Fidelity Bayesian Optimization Loop

**Why multi-fidelity:** At reduced frequency **k ≈ 0.57**, the flow is firmly unsteady. Published studies on oscillating flat plates at similar reduced frequencies show that quasi-steady models can overpredict peak forces by 30-50% and, critically, can **change the relative ranking** of different geometries (because unsteady effects like dynamic stall and leading-edge vortex formation interact differently with different planforms and camber profiles). Running the entire BO budget on steady-state CFD alone risks optimizing against a proxy that does not preserve design rankings.

**Solution:** Use multi-fidelity BO with BoTorch's `SingleTaskMultiFidelityGP`, mixing cheap steady-state evaluations (fidelity=0) with expensive unsteady evaluations (fidelity=1). This is the textbook use case for multi-fidelity BO. The GP learns the correlation between steady and unsteady results and allocates the budget intelligently.

Claude writes this entire loop as a Python script:

```python
# Claude writes this script; user runs it
# Multi-fidelity BO: steady-state (cheap) + unsteady (expensive) CFD
import torch
from botorch.models import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.optim import optimize_acqf_mixed
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc
import subprocess, json

N_DIMS = 42 # ~37-46 dims (Layer 1 envelope + Fourier + Layer 2 macro/procedural fields + Layer 3 capped primitive + Layer 4 manufacturing + fan macro)
N_TIER_MINUS1_INIT = 30 # : 2D STEADY CFD screening
N_STEADY_INIT = 30 # Initial 3D steady CFD LHS samples (ranking-only fidelity)
N_UNSTEADY_INIT = 5 # Initial 3D unsteady CFD samples (true J_fan)
N_BO_ITERS = 50 # Multi-fidelity BO iterations (TuRBO-paced)

# Three-tier fidelity column: -1 = 2D STEADY CFD slice, 0 = 3D steady CFD (ranking-only),
# 1 = 3D unsteady CFD (true J_fan). QSST analytical removed in because it cannot
# represent the generative topology cleanly (Boolean subtractions break the strip decomposition).
bounds = torch.tensor(
 [[...] + [-1.0], # lower bounds + fidelity lower
 [...] + [ 1.0]], # upper bounds + fidelity upper
 dtype=torch.double
)
target_fidelities = {N_DIMS: 1.0} # optimize at highest fidelity (3D unsteady)

def run_eval(params, fidelity):
 """Run a fan evaluation at specified fidelity. Claude writes this wrapper."""
 # fidelity=-1: 2D STEADY CFD slice (~5 min per eval)
 # fidelity= 0: 3D steady CFD (~30-90 min on Colab Pro 8 vCPU; ranking only)
 # fidelity= 1: 3D unsteady CFD, 5 cycles, dt=T/200 (~3-6 hours on Colab Pro)
 # 1. Generate multi-material fan geometry from params via Fusion add-in or CadQuery
 # 2. Mesh with Gmsh
 # 3. Run SU2_CFD at the appropriate fidelity
 # 4. Parse results with canonical j_fan.py (Section 9.4 locked spec)
 # 5. Return J_fan (and J_fan_peak as secondary)
 ...

from config.cost import COST_TUPLE # canonical (Tier-1, Tier-0, Tier+1) costs, see §6.2.2

class Tier(IntEnum):
 TIER_MINUS_ONE = -1 # 2D steady CFD
 TIER_ZERO = 0 # 3D steady CFD
 TIER_ONE = 1 # 3D unsteady CFD

def cost_model_evaluate(tier: Tier) -> float:
 """Three-tier cost model: TIER_MINUS_ONE = 2, TIER_ZERO = 10, TIER_ONE = 50 (default COST_TUPLE).
 Tier lookup is by named enum, NOT by Python negative indexing — `evaluate(-1)` previously
 looked like 'last element' under standard semantics but meant 'Tier -1' (Tier.TIER_MINUS_ONE)
 which is a different value. Explicit enum lookup removes the ambiguity."""
 # COST_TUPLE = (cost_at_TIER_MINUS_ONE, cost_at_TIER_ZERO, cost_at_TIER_ONE)
 # Index by tier offset from TIER_MINUS_ONE so the negative tier index resolves correctly.
 return COST_TUPLE[int(tier) - int(Tier.TIER_MINUS_ONE)]

def cost_model(X):
 """Batched version: map a fidelity column in {-1, 0, 1} to its COST_TUPLE entry."""
 fidelity = X[..., -1]
 return torch.where(
 fidelity < -0.5, torch.tensor(float(COST_TUPLE[0])), # Tier -1 (2D steady)
 torch.where(fidelity < 0.5,
 torch.tensor(float(COST_TUPLE[1])), # Tier 0 (3D steady)
 torch.tensor(float(COST_TUPLE[2]))) # Tier 1 (3D unsteady)
 )

# CI assertion: cost_model_evaluate(Tier.TIER_MINUS_ONE) == COST_TUPLE[0] == 2.0
# Replaces the prior `cost_model.evaluate(-1).item() == 2.0` which was ambiguous about
# whether `-1` meant Python negative-index (last element = 50.0) or Tier.TIER_MINUS_ONE
# (= 2.0). Always use the enum.

# Generate initial training data across all three fidelity tiers.
# Per-architecture seed budget: 30 2D-steady (Tier -1) + 30 3D-steady (Tier 0) + 5 3D-unsteady (Tier 1).
sampler = qmc.LatinHypercube(d=N_DIMS)

# 2D steady (fidelity = -1): cheap exploration
X_2d = torch.tensor(sampler.random(n=N_TIER_MINUS1_INIT), dtype=torch.double)
X_2d = bounds[0, :-1] + (bounds[1, :-1] - bounds[0, :-1]) * X_2d
X_2d = torch.cat([X_2d, torch.full((N_TIER_MINUS1_INIT, 1), -1.0)], dim=-1) # fid=-1 (2D steady CFD)

# 3D steady CFD (fidelity = 0): ranking-only; independent LHS
X_steady = torch.tensor(sampler.random(n=N_STEADY_INIT), dtype=torch.double)
X_steady = bounds[0, :-1] + (bounds[1, :-1] - bounds[0, :-1]) * X_steady
X_steady = torch.cat([X_steady, torch.zeros(N_STEADY_INIT, 1)], dim=-1) # fid=0

# 3D unsteady CFD (fidelity = 1): true J_fan; subset of steady designs, re-tagged
unsteady_indices = torch.randperm(N_STEADY_INIT)[:N_UNSTEADY_INIT]
X_unsteady = X_steady[unsteady_indices].clone()
X_unsteady[:, -1] = 1.0 # fid=1

X_init = torch.cat([X_2d, X_steady, X_unsteady])
Y_init = torch.tensor(
 [run_eval(x[:-1], x[-1].item()) for x in X_init],
 dtype=torch.double
).unsqueeze(-1)

# Multi-fidelity BO loop. production: wrap this in a TuRBO trust-region loop
# with 3-5 parallel regions, per architecture from the outer-loop bandit.
X_train, Y_train = X_init, Y_init
for iteration in range(N_BO_ITERS):
 # FIXED-FLOOR EPISTEMIC NOISE (§0 locked). Per-observation `J_fan_se` is
 # stored as the JSONL `J_fan_cycle_variance` diagnostic but is NOT passed as
 # train_Yvar — physics-driven limit-cycle variance from vortex shedding
 # (5-15% for shed-heavy designs) would otherwise tag the very designs we
 # want to find as low-confidence and weaken the cross-fidelity correlation
 # kernel exactly where asymmetric-drag wins. The GP uses a per-tier scalar
 # epistemic-noise floor calibrated once in Spike 0.7b's replicate-Tier-1
 # runs and locked at EPISTEMIC_NOISE_FLOOR = max(measured, 1e-6).
 train_Yvar_floor = torch.full_like(Y_train, EPISTEMIC_NOISE_FLOOR) # scalar, NOT per-eval J_fan_se
 gp = SingleTaskMultiFidelityGP(
 X_train, Y_train,
 train_Yvar=train_Yvar_floor, # fixed-floor (NOT heteroscedastic per-obs)
 outcome_transform=Standardize(m=1),
 data_fidelities=[N_DIMS], # last column is fidelity
 )
 mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
 fit_gpytorch_mll(mll)

 cost_utility = InverseCostWeightedUtility(cost_model=cost_model)
 qMFKG = qMultiFidelityKnowledgeGradient(
 model=gp,
 current_value=gp.posterior(
 X_train[X_train[:, -1] == 1.0] # value at target fidelity
 ).mean.max() if (X_train[:, -1] == 1.0).any() else torch.tensor(0.0),
 cost_aware_utility=cost_utility,
 project=lambda X: project_to_target_fidelity(X, target_fidelities),
 )

 # Optimize over design params + discrete fidelity choice across all three tiers.
 candidate, acq_value = optimize_acqf_mixed(
 qMFKG, bounds=bounds, q=1,
 fixed_features_list=[{N_DIMS: -1.0}, {N_DIMS: 0.0}, {N_DIMS: 1.0}],
 num_restarts=20, raw_samples=512,
 )

 fidelity = candidate[0, -1].item()
 new_y = run_eval(candidate[0, :-1], fidelity)
 X_train = torch.cat([X_train, candidate])
 Y_train = torch.cat([Y_train, torch.tensor([[new_y]], dtype=torch.double)])

 n_2d = (X_train[:, -1] == -1.0).sum().item()
 n_steady = (X_train[:, -1] == 0.0).sum().item()
 n_unsteady= (X_train[:, -1] == 1.0).sum().item()
 print(f"Iter {iteration}: best = {Y_train.max().item():.4f}, "
 f"fidelity={fidelity:+.0f}, "
 f"total: {n_2d} 2D + {n_steady} 3D-steady + {n_unsteady} 3D-unsteady")

# Extract best at target fidelity (3D unsteady).
mask_hf = X_train[:, -1] == 1.0
best_idx = Y_train[mask_hf].argmax()
best_params = X_train[mask_hf][best_idx, :-1]
```

**Stop-criteria priority ladder (H5 lock — explicit precedence; orchestrator evaluates these at every BO step in order):**

1. **Hypervolume plateau** (per the §6.4 HV early-stop spec): `(HV[-1] − HV[-50]) / HV_baseline < 1e-4` after round 500. Action: declare convergence, reallocate remaining budget to FEA gate top-50 + Phase 5.
2. **1000-hour cumulative compute trigger** (force-grow rule): K stays at its current value within {3, 4, 5}; orchestrator force-grows the (Tier 0, Tier 1) overlap until N ≥ 30 for the K-decision. Action: continue Phase 4 BO at the updated K — do NOT terminate.
3. **1300-hour pessimistic ceiling**: hard stop on Phase 4 BO regardless of HV plateau. Action: terminate Phase 4, hand off the current best Pareto-front candidates to Phase 5.

**If criteria #1 and #2 fire simultaneously, #1 takes precedence** (convergence declared, reallocation triggered, force-grow skipped). **If #2 fires and force-grow completes (N ≥ 30 satisfied) but BO has not declared convergence by #3, terminate at #3 with the current Pareto front and do NOT extend.** Implemented in `scripts/run_phase4_bo.py --iterate` as a priority-check at each iteration.

**Budget allocation (+ hardening + rebaseline post-cost-doubling):** Three-tier multi-fidelity GP plus `qMultiFidelityKnowledgeGradient` allocates across 2D-steady / 3D-steady / 3D-unsteady with **cost ratios (2, 10, 50) per (was (1, 5, 50); two-eval delta doubled Tier -1 and Tier 0)**. Architecture-bandit Tier -1 screening: ~80-130 architectures × 30 evals = ~2400-3900 cheap 2D-steady evals. Per promoted architecture (K=3-5): ~30 Tier 0 + ~5 Tier 1 seed + ~35 acquisition rounds (hard ceiling; mixed across tiers per the cost model; early-stop fires first on convergent architectures). Total ~3000-4500 evaluations across all architectures. **Honest compute budget:** post-the back-of-envelope arithmetic is 200-600 h Tier -1 screening + 240 h Tier-0 pre-promo + 90 h Tier-1 baseline + 165 h acquisitions ≈ **895 h baseline at K=4 with optimistic assumptions** (no retries, no I/O wait, no GP retrain stalls). Realistic **600-1100 h expected / 1300 h pessimistic**. data-driven K and acquisition cap claw back ~25-30%; §6.3.1 prefilters cut another ~30-50% of Tier -1 minutes. **stop rule:** if cumulative wall-clock > **1000 h** before promoted-arch convergence (measured by early-stop or qNEHVI hypervolume plateau over 20 acquisitions), **K stays at its current value within {3, 4, 5}** and the orchestrator force-grows the (Tier 0, Tier 1) overlap until the K-decision threshold N ≥ 30 is satisfied — the "drop K to 2" rule from an earlier draft is removed because K = 2 is outside the locked range. Compute target line in §0 should be read as the favorable case, not a hard ceiling. 25% contingency added to all phase-budget line items.

#### 6.2.4 Hybrid Parameterization Rationale
The parameterization uses a four-layer hybrid structure rather than the earlier single-mechanism Boolean-subtraction approach. This is a deliberate engineering trade-off addressing three reviewed concerns:

1. **"Mr. Potato Head" alignment problem.** The earlier ~4-6 independent 6-DOF primitives are statistically unlikely to converge to *aligned* patterns under Bayesian optimization. The GP surrogate would need to learn correlation between many independent variables — chicken-and-egg in correlated-feature discovery. Result: lumpy swiss-cheese designs rather than coherent emergent topologies (slats, lattices, etc.).
2. **Asymmetric drag is critical for rigid fans.** With single-material rigid PETG (no FSI), the only mechanism for asymmetric drag — the physics that makes hand fans work — is angled passive directional features (louvers). The parameterization should give the optimizer direct access to this design family rather than hoping it discovers alignment by chance.
3. **OpenCASCADE Boolean reliability.** Independent 6-DOF primitives frequently produce CAD edge cases (zero-thickness walls, tangent intersections) that crash CadQuery. In automated BO loops, CAD failures create cliffs in the surrogate model that ruin acquisition functions. Safe-by-construction parameterizations are more robust.

A fourth observation drove the addition of procedural math generators (Layer 2 (d) and (e)): **human-designed pattern families** (louvers, dimples, serrations) **produce optimized versions of those families but cannot produce genuinely novel topology**. Procedural math generators (Perlin/Simplex noise, TPMS lattices, Fourier-series boundaries) are mathematical primitives that yield organic shapes nobody designed, expanding the design space toward truly emergent topology.

**The four layers:**

- **Layer 1 (envelope + Fourier modulation, ~14 vars):** establishes overall blade silhouette. The Fourier-series leading/trailing edges allow the blade outline to evolve into organic, leaf-like, or bat-wing shapes rather than being constrained to a smooth airfoil.
- **Layer 2 (macro-pattern + procedural math fields, ~15-20 vars active):** the bulk topology mechanism. Five field types are available; the optimizer activates 0-3 per design. **Macro-pattern fields** (louvers, textures, edge features) give the optimizer direct access to aerodynamically-important and statistically-tractable design families. **Procedural math fields** (noise thresholding, TPMS) give access to organic emergent topology that no human pre-designed. Each field is safe-by-construction.
- **Layer 3 (capped 0-1 independent primitive, ~5-7 vars):** preserves a small space for asymmetric point features that don't fit any pattern family. Cannot dominate the topology but can introduce point-feature surprises (an asymmetric ridge, a specifically-placed slot) that the library cannot produce.
- **Layer 4 (manufacturing categoricals, ~3-5 vars):** print orientation, layer height, and click-feature geometry parameters.

**The hybrid avoids both failure modes:**

- *Pure independent primitives (~50 vars, default):* statistically unlikely to find aligned patterns under BO; high CAD failure rate.
- *Pure macro-patterns (~10 vars):* gives up emergent design freedom that motivates the project.

**Design surprise expectation :** the procedural math fields are the primary source of organic-looking, "grown-not-engineered" output *within* the 4-layer parameterization. Noise thresholding produces bone-like, coral-like, sponge-like patterns. TPMS produces gyroid-lattice cutouts with directional density gradients. Fourier outer boundaries produce bat-wing or leaf-like silhouettes. In combination, these can produce designs that *look* genuinely grown or evolved rather than engineered. **What this is NOT:** SIMP / level-set topology optimization that can invent geometry outside its parameterization — those approaches are rejected (§3.1 §0 rationale). The 4-layer parameterization gives more design freedom than a flat Boolean-primitive list but is still expert-priored at the field-library level.

**Note on TPMS aero benefit:** at the panel thicknesses in scope (2.2-3.8 mm), TPMS produces **directional through-blade porosity rather than internal infill** — the lattice cell pitch (≥3× min feature size = ~2.4 mm) is comparable to the panel thickness, so there isn't enough material depth for a "buried" lattice with intact skins. The optimizer determines whether through-blade porosity helps or hurts J_fan via CFD evaluation; designs using TPMS are not preferred a priori. The aero impact may be very different from "internal mass savings with intact aerodynamic skin" — the lattice creates literal flow channels that can either leak unhelpfully or route flow directionally.

#### 6.2.5 Deterministic Design Hashing
<a id="6.2.5"></a>

Every candidate design is identified by a deterministic content hash of its canonical parameter dict. Same parameter dict → same hash; cross-session dedup is one `os.path.exists` call on Drive.

```python
import json, hashlib

def _round_leaves(obj, precision: int):
 """Walk the params dict and round every numeric leaf BEFORE json.dumps.
 Required because json.dumps(default=...) is only invoked for objects JSON
 can't serialize natively — Python float is JSON-native, so a default= lambda
 is silently dead code for numbers and would let raw 17-digit float reprs
 leak into the hash and silently break cross-session dedup."""
 if isinstance(obj, (float, np.floating)):  # also catches np.float64 etc.
 return round(float(obj), precision)
 if isinstance(obj, dict):
 return {k: _round_leaves(v, precision) for k, v in obj.items()}
 if isinstance(obj, (list, tuple)):
 return [_round_leaves(x, precision) for x in obj]
 return obj # int / str / bool / None pass through

def design_hash(params: dict, precision: int = 6) -> str:
 """Stable hash of a canonical parameter dict; rounds every numeric leaf to
 `precision` digits to absorb double-precision noise across sessions/platforms
 BEFORE json.dumps sees them. blake2b/12-byte digest → 24-hex-char primary
 key, short enough to be a directory name."""
 rounded = _round_leaves(params, precision)
 canonical = json.dumps(rounded, sort_keys=True)
 return hashlib.blake2b(canonical.encode("utf-8"), digest_size=12).hexdigest()
```

**Required regression test (`tests/test_utils/test_design_hash.py`):** assert `design_hash({"x": 0.1 + 0.2}) == design_hash({"x": 0.3})` (the JSON-native float would have leaked `0.30000000000000004` without the leaf-walk and produced different hashes). Also assert recursive equality on `{"nested": {"x": 0.1+0.2}}` vs `{"nested": {"x": 0.3}}`.

Used everywhere a design is identified: SQLite-replacement JSONL rows (§Phase 4 step 51), Drive directories (`designs/{hash}/`), cache lookups, BO acquisition de-duplication, Pareto-front identification, top-3 selection. **Round-trip property:** `design_hash(load_params(designs/{hash}/params.json)) == hash` — asserted by the orchestrator on every read; mismatch indicates corruption or a JSON-precision drift bug.

### 6.3 ML for TO (Optional Extension)

#### 6.3.1 Pre-CFD Prefilters
Two cheap, deterministic prefilters run between BO candidate proposal and CFD-tier dispatch. Together they remove ~30-50% of candidates that would have failed downstream anyway, freeing that compute for productive evaluations.

**Filter 1 — Pre-CFD hard-constraint filter (+ -Soft):** compute `m_total`, `r_CoM_wrist`, and the §9.7.3 manufacturability score from the generated geometry only — no CFD. **-Soft soft penalty (parallels ):** a design with `mfg_score = 0.51` and one with `0.95` are indistinguishable to the GP under a pure boolean Filter 1 (both pass with `mfg ≥ 0.5`), so BO keeps proposing borderline-manufacturable designs. adds a **soft penalty** in the 0.5-0.8 band: designs feed into the GP as `J_fan_observed − α · max(0, 0.8 − mfg_score)² · J_fan_scale` with **α ≈ 0.25**. Hard reject (no GP observation) still applies at `mfg_score < 0.5`. Same pattern for the structural filters. Reject hard any candidate violating:
- `m_total < 0.100 kg` (§6.4 mass cap)
- `r_CoM_wrist ≤ 0.160 m` (§6.4 CoM cap; computed via `r_com_assembly` position-of-CoM, NOT radius of gyration — see §6.4 code).
- `manufacturability_score ≥ 0.5` (§9.7.3 §N7 filter).

Rejected candidates write a JSONL record with `status="rejected_hard_constraint"` and the list of violating constraints; no CFD tier is dispatched.

**Filter 2 — Pre-CFD structural prefilter:** a closed-form rib-bending + tip-deflection estimate (beam theory on the V-unit outer rib at peak aero + inertial load) gates obviously-too-flexible designs before they reach CFD. The screen drives each of the three panel-pivot modes (tension/bending/bearing) independently from the §3.1.5 K_t hotspot table — no scalar superposition. **Reference pressure for Filter 2 (M3 lock):** `p_aero_reference = 10 Pa` — a **fixed canonical baseline from §2.4**, NOT a per-design CFD pressure (Filter 2 runs pre-CFD, deterministic from geometry + canonical loads). The 2.5× stress-test multiplier scales this fixed 10 Pa reference. The `material_locks.P_AERO_REFERENCE_PA = 10.0` constant is the single source of truth for Filter 2; the §59.5 gate uses a different reference (`p_stagnation_peak = ½ρV²_local_max ≈ 3 Pa`, per H9 lock) because §59.5 has access to per-design Tier-1 history while Filter 2 does not. Reject if any of:
- (canonical, rib-panel fillet bending) estimated peak fillet-bending stress > **0.6 × 9.00 = 5.40 MPa nominal** (60% of the §10.1 rib-panel fillet cyclic allowable), OR
- (canonical, panel-pivot tension) estimated peak tension at the hole equator > **0.6 × 5.58 = 3.35 MPa nominal** (60% of the §10.1 panel-pivot tension allowable at K_tt = 2.42 boss), OR
- (canonical, panel-pivot bearing) estimated bearing stress at pin bore > **0.6 × 2.00 = 1.20 MPa nominal** (60% of the §10.1 Z-direction bearing allowable; this is the binding mode under canonical loading), OR
- **(stress-test, rib-panel fillet) estimated fillet peak σ_VM under 2.5× p_aero + 2× α_max > 0.6 × 12.40 = 7.44 MPa nominal** (60% of the §10.1 static SF-1.5 allowable at K_t = 2.42), OR
- estimated tip deflection > **0.6 mm** (60% of the §3.1.1 1 mm rigid-blade bound; the prior 1.0 mm threshold was a copy-paste error).

Implementation: a ~50-line closed-form beam solver on the rib's planform extracted from the generator's STL (treat the rib as a tapered Euler-Bernoulli beam loaded by the panel's per-blade aero force and the m·α·r_wrist inertial load). Result fields `pre_cfd_stress_estimate_mpa`, `pre_cfd_tip_defl_mm`, `pre_cfd_struct_ok` go onto the JSONL record. Rejected candidates write `designs/{hash}/.rejected` and `status="rejected_pre_cfd_struct"`.

**Why both:** Filter 1 catches geometry violations the optimizer should have respected but didn't; Filter 2 catches geometry that's mechanically valid but would fail the §59.5 combined-blade FEA gate downstream. Phase 5 step 59.5 still runs on the top-3 — these prefilters reduce wasted Tier -1/0/1 CFD minutes on candidates that 59.5 would have killed anyway.

**Filter 3 — DEPRECATED PASS-THROUGH.** The original Filter 3 argued plate-bending was blind to centrifugal pull *along the rib axis*, citing ω_max from `PITCHING_OMEGA = 12.566 rad/s` as a spin rate. That's wrong on two counts: (1) `PITCHING_OMEGA` is the SHM angular frequency (2π·f for f = 2 Hz), not a steady spin rate; the *instantaneous* peak blade angular velocity is ω_blade_max = θ_max · ω_SHM = 0.7 · 12.566 = 8.8 rad/s. (2) Under wrist-axis = +y, the rib runs in +x, so rotation about +y produces tangential velocity perpendicular to the rib axis, not along it. The real concern is the cyclic tangential reaction at the **click detent** (~22-27 m/s² tangential at the **panel outer edge** at the tip — moved from the rib tip per the item #3 relocation; **0.275 N reaction per blade** under canonical loading using the H8-corrected L_wrist_to_tip = 0.25 m lever arm, not the stale 0.22 N value computed with L_blade = 0.20 m) — that load is covered by the one-shot Phase-2 dynamic-load assertion (see §6.3.1 above), not by a per-design Filter 3. Filter 3 stays as a **deprecated pass-through stub** that returns `passed=True` with `failure_code=None` so existing `filter[N]` indexed code paths don't shift; future filters take index 4+. The `rejected_pre_cfd_centrifugal` failure code is removed from §9.4.2; the `pre_cfd_centrifugal_ok` field is removed from the JSONL schema.

**load-combination lock (clarifies Filter 2's "inertial" term):** the canonical load is **peak-positive aero pressure + hand-swing inertial (m·α·r_wrist, wrist-relative) + centrifugal (m·ω²·r_wrist)**. Coriolis and tangential (ω̇) are negligible at f = 2 Hz and are dropped. The stress-test load uses the same combination scaled by 2.5× / 2× / 2× respectively. **Filter 2 covers the full structural envelope** (Filter 3 is a deprecated pass-through stub — see Filter 3 paragraph below; the §6.3.1 click-detent dynamic-load assertion runs once at Phase 2 entry, not per design).

**K_t hotspot tightening (H4 — explicit hotspot list, NOT a rib-axis proximity criterion):** the earlier "any K_t feature within 3 mm of the rib axis" criterion missed the binding hotspot — the panel-pivot bearing at y = 0 sits more than 3 mm from any rib axis (ribs at y = ±rib_center ≈ ±5 mm) and would have escaped tightening under the proximity rule. **Lock:** Filter 2 applies the 0.40× tightening at every entry in this explicit hotspot list:

1. Panel pivot hole (all three modes: tension at K_tt = 2.42, bending at K_tb = 3.2, bearing at K_t_bearing = 1.5) — at y = 0
2. Click detent at panel outer edge (item #3 relocation) — at the click footprint (per CLICK_FOOTPRINT_X_RANGE × CLICK_FOOTPRINT_Y_RANGE_PANEL_EDGE)
3. Rib-panel fillet — at y = ±rib_center, full rib length
4. Slot ends (Layer 2 louver) — wherever the louver field generates cuts
5. TPMS through-holes — wherever the TPMS field generates perforations
6. Layer 3 primitive boundaries — wherever the primitive exists

Per-hotspot evaluation: Filter 2 passes iff every hotspot's per-mode allowable is satisfied at the 0.40× threshold; the 0.60× headroom applies only away from any of the listed hotspots. Resulting per-mode hotspot-tightened thresholds: bending 1.69 MPa / tension 2.23 MPa / bearing **0.80 MPa** (the binding one; 0.40 × 2.00 MPa). The closed-form rib-beam solver in `pre_cfd_struct_estimate.py` enumerates hotspots from the design's STL (Layer 2 fields + Layer 3 primitive activations are read from `params.json`) and applies per-hotspot thresholds.

**soft-penalty path (avoids invisible-boundary problem):** designs that fail Filter 2 / Filter 3 by less than 1.0× σ_y are NOT hard-rejected — instead they enter the GP as **`J_fan_observed + soft_penalty(stress − 0.6·σ_y)`** with a quadratic penalty proportional to the overshoot. Hard-reject only beyond 1.0× σ_y. This keeps the GP informed near the constraint boundary where the best designs live; without it, BO clusters proposals near the boundary and sees no gradient because rejected designs vanish from the training set.

**Ordering convention (+ + + locked):** parameter-box bounds → §9.7.3 manufacturability → Filter 1 (mass + r_CoM_wrist) → Filter 2 (closed-form bending+inertial+bearing+tear-out structural, multi-mode expansion) → ~~Filter 3 (centrifugal pivot stress, )~~ **Filter 3 DEPRECATED-PASS-THROUGH** (dropped the gate; keeps the slot number so existing `filter[N]` indexed code paths don't shift — a deprecated-stub `Filter 3` always returns `passed=True` with `failure_code=None`; future filters that need a new index take Filter 4+) → CFD-tier dispatch. The first failure short-circuits; cheap checks first. Filter 1 + Filter 2 soft-reject before they hard-reject; the hard-reject envelope is at 1.0× σ_y, not at the 0.6× / 0.35× screening thresholds.

**Filter 2 panel-pivot stress checks (four independent modes, no scalar sum) — relabeled to separate axial-shear (Z, from m·α·r tangential reaction) from transverse bearing (+x, from centrifugal m·ω²·r):** under panel-pivot architecture, the pivot hole sees four distinct stress modes at four different physical locations / directions. Each is checked separately against its own allowable; Filter 2 passes iff **all four** pass (see §3.1.5 K_t table for the (K_t, mode, allowable) tuples):

1. **Bending at panel skin (chordwise, XY):** beam-theory peak bending stress at the panel top/bottom skin from aero+inertial loading. `K_tb · σ_bending < 4.22 MPa nominal` (canonical) / `< 0.40 · 4.22 = 1.69 MPa` (hotspot-tightened). K_tb = **3.2** locked (literature spread 2.7-3.2; conservative upper bound).
2. **Tension at hole equator (tangential, XY):** Peterson tension solution at the hole circumference. `K_tt · σ_tension < 5.58 MPa nominal` (canonical) / `< 0.40 · 5.58 = 2.23 MPa` (hotspot). K_tt = **2.42** (Peterson polynomial, d/w = 0.25 in 12 mm boss).
3. **Axial shear on pin / panel-interlayer fatigue (Z-direction; from m·α·r tangential reaction; H11 per-panel correction):** the `m_blade · α_max · r_wrist` reaction force acts in the ±z swing direction. Since the pin axis is also +z (per §0), this force is **along** the pin axis — it doesn't press transversely into the pin (that's "bearing", item 4 below); instead it shears the panel material along its print Z-direction at the pin-panel interface. This is an FDM **interlayer fatigue** mode. **Per-panel correction (H11):** each panel-pin interface sees only the load from its own blade's inertia; the pin reacts m_total at its end terminations, but the panel-pin contact patch sees `m_blade ≈ m_total / N_blades ≈ 0.006 kg` (not m_total). `τ_axial_shear = F_axial_per_panel / (π · d_pin · t_panel)` with `F_axial_per_panel = m_blade · α_max · r_wrist ≈ 0.006 · 110 · 0.15 ≈ 0.10 N` per cycle and `d_pin = 3 mm`, `t_panel = 2 mm` → `τ_axial_shear ≈ 0.0053 MPa` canonical (an order of magnitude smaller than the earlier m_total-based 0.053 MPa). Allowable: `τ_axial_shear < 2.00 MPa nominal` (Z-direction interlayer; `0.10·σ_y_Z/K_t = 0.10·30/1.5` factor; FDM Z-fatigue floor is the binding mechanism). Stress-test (per Architectural C: 2× α_max combined with 1.41× ω) doubles the centrifugal contribution to mode 4 but leaves mode 3 at 2× the canonical (axial shear scales linearly with α, not ω).
4. **Centrifugal bearing on pin (transverse, radial in +x — actual bearing mode; H11 per-panel correction):** the `m_blade · ω_blade_max² · r_wrist` centrifugal reaction acts in +x, pressing the pin transversely into the panel hole. **Per-panel correction (H11):** as in mode 3, each panel-pin interface sees only its own blade's centrifugal contribution. `σ_bearing = F_centrifugal_per_panel / (d_pin · t_panel)` with `F_centrifugal_per_panel = m_blade · ω_blade_max² · r_wrist ≈ 0.006 · 8.8² · 0.15 ≈ 0.07 N` per cycle; for d_pin = 3 mm, t_panel = 2 mm → `σ_bearing ≈ 0.012 MPa` canonical (an order of magnitude smaller than the earlier m_total-based 0.12 MPa; far below allowable, explicit for completeness). Allowable: `σ_bearing < 2.00 MPa nominal` (transverse bearing on the boss; K_t_bearing = 1.5; `0.10·σ_y_Z/K_t_bearing` factor). The 4-mode separation makes the pre-amendment "Z-direction" label honest: modes 3 and 4 both end up Z-fatigue-limited but for different physical reasons (axial shear vs transverse bearing). The H11 per-panel correction doesn't change pass/fail (both modes still far below 2.00 MPa allowable) but documents the correct physics: per-panel inertial loads scale with m_blade, not m_total.

**Tear-out resolved by locked boss thickening:** with the original 8 mm panel-pivot width the pin-to-edge distance was e = 4 mm, ratio e/d = 1.33, below the `e/d ≥ 2.0` minimum for pin-loaded plates. **Locked V1 fix: local boss thickening** — `make_panel_solid` in §9.7 generates a **circular boss of OD = 12 mm × panel_thickness** centered on the pivot hole (e_boss = 6 mm, e/d_boss = 2.0). The boss is part of the panel-domain mask exclusion (Layer 2/3 cannot carve into the circular 7 mm-radius PANEL_PIVOT_REGION). The §3.1.5 K_t at the boss uses d/w = 3/12 = 0.25 → **K_tt = 2.42 (locked)** with tension allowable 5.58 MPa; the bearing mode at 2.00 MPa still binds first.

**Click-detent dynamic-load assertion (closes the dropped-Filter-3 rationale, one-shot at Phase 2 entry; H8 lever-arm corrected):** `α_peak · m_rib · r_tip · N_blades < 0.1 · click_detent_allowable`. **Numerically (using L_wrist_to_tip = 0.25 m per H8, NOT L_blade = 0.20 m — the click detent sits at the panel outer edge at x = L_blade, which is at d_handle + L_blade = 0.25 m from the wrist axis):** `110 · 0.001 · 0.25 · 10 = 0.275 N` tangential reaction at the click detent under canonical loading. Spike 0.4 measures `click_detent_allowable` (1000-cycle force-to-detent-fracture floor, target ~2-4 N); the 0.1× safety factor gives a 0.2-0.4 N budget. The 0.275 N reaction sits at **69-138% of the 0.2-0.4 N budget** (i.e., overshoots the lower bound by 38% and lands at the high end of the budget envelope at the upper bound) — designs near the upper m_rib bound need the assertion to fire. Written as a static check in `phase2_dynamic_load_assert.py` and runs once at Phase 2 entry — replaces the prior per-design Filter 3 (which used PITCHING_OMEGA as a spin rate; the correct dynamic load is the cyclic tangential reaction at the click detent, not a centrifugal pull along the rib axis). Filter 3 stays as a deprecated pass-through stub to preserve `filter[N]` indexing. (The earlier duplicate paragraph that used `r_tip = 0.20 m` and reported 0.22 N is deleted; the H8 lever-arm lock applies wrist-to-tip distance to all torque-to-force conversions outside the pivot region.)

**Known limitation — post-hoc filter, not constrained acquisition:** Filter 1 and Filter 2 run **after** the BO proposes a candidate; the GP itself doesn't know about them. Over many iterations, the acquisition function (qMFKG / qNEHVI) can converge on regions where the prefilters reject everything, wasting acquisition budget on infeasible proposals. The Phase 4 step 56 orchestrator monitors the **prefilter rejection rate** — if rejection rate > 20% over 5 consecutive BO steps, switch to **constrained EI / feasibility-weighted acquisition** with a separate GP modeling the manufacturability + mass + r_CoM_wrist score as a constraint (standard BoTorch pattern: `ConstrainedExpectedImprovement` / `qNoisyExpectedHypervolumeImprovement` with `constraints=...`). The post-hoc filter approach works for low rejection rates (the dominant expected regime); the constrained-acquisition fallback handles the degraded regime without aborting the campaign. The switch is logged in `phase4/diagnostics/constrained_acq_<date>.json` for post-campaign analysis.

For rib TO, the problem is small enough (a single thin beam) that ML acceleration is unnecessary for a first project. A single 2D SIMP run on a rib discretized at 0.5 mm elements (400 x 24 = ~9,600 elements in the planform) converges in seconds to minutes.

If exploring many rib variants (different cross-section profiles, taper ratios, load cases), DL4TO can train a neural network to predict near-optimal rib topologies instantly. See Section 9.5 for details.

### 6.4 Multi-Objective Optimization
A fourth Pareto objective (folded form factor) exposes the bulk-vs-compactness trade-off introduced by the expanded 2.2-3.8 mm thickness range.

**Four Pareto objectives :**
1. **Maximize J_fan** (per locked §9.4 spec — full-cycle directed momentum flux through 600×600 mm plane at 300 mm).
2. **Minimize rotational inertia I_wrist about the wrist (handle-grip) axis** (wrist axis = **+y**, pivot pin axis = **+z** (stacking direction); wrist axis ≠ pivot pin axis — see §0 coordinate-convention row). Computed as `I_wrist = ∫ r_perp² dm` over the full open-fan assembly, where `r_perp = √(x² + z²)` is the perpendicular distance from the **wrist (+y) axis through the handle grip point at world origin** to each mass element. **Units:** I_wrist in **kg·m²** (SI throughout per the §0 units lock).

**Mass-distribution for I_wrist:** under panel-pivot architecture, the pivot pin runs in +z through one hole per blade panel at y = 0 (NOT through two holes per blade in the ribs). The pin's mass (steel/brass, ρ ≈ 7850-8500 kg/m³, ~2.5 g over ≥45 mm length) is centered on the +z axis through the pivot point (d_handle, 0, 0); its perpendicular distance to the wrist (+y) axis is √(d_handle² + z²), so for the pin's center at z = 0 the contribution is d_handle ≈ 0.05 m, growing with |z| over the pin's ≥45 mm length. The `i_wrist_assembly` call iterates over `[*blades, spine, pin]` with the steel/brass density applied to the `pin` solid (separate from the PETG density applied to blades + spine). The rib pivot holes are removed (the rib has no pivot hole; the hole is in the panel at y = 0).

 **Implementation:** use OpenCASCADE's built-in `GProp_GProps::MomentOfInertia(gp_Ax1)` — it takes a `gp_Ax1` (point + direction) and returns the scalar moment about that axis, handling the parallel-axis reduction + body/world rotation internally so we never roll our own. Regression tests guard `matrixOfInertia` against (1) unit conversion, (2) body-frame vs world-frame, and (3) CadQuery API-surface bugs.

 `src/fanopt/geometry/generator.py`:

 ```python
 from OCP.BRepGProp import BRepGProp
 from OCP.GProp import GProp_GProps
 from OCP.gp import gp_Ax1, gp_Pnt, gp_Dir

 # per-part density table (was buggy: uniform RHO_PETG ignored -Pin steel/brass pin)
 MATERIAL_DENSITIES_KGM3 = {
 "petg": 1270.0, # PETG (SI; §0 units lock; ρ_y_XY = 45 MPa, ρ_y_Z lower)
 "steel": 7850.0, # plain carbon steel pivot pin
 "brass": 8500.0, # alternative pivot pin material
 }
 D_HANDLE_M = 0.050 # m; wrist-grip ↔ pivot-pin offset along the handle axis

 def _volume_props(solid):
 """Run OCP's volume-properties pass once and return the GProp_GProps object,
 which exposes Mass [= volume at ρ=1], CentreOfMass, and MomentOfInertia(axis)."""
 props = GProp_GProps()  # instantiate; was bare class reference
 BRepGProp.VolumeProperties_s(solid.wrapped, props)
 return props

 def i_wrist_about_y(solid, material, wrist_origin_m=(0.0, 0.0, 0.0)):
 """Inertia about the wrist (+y) axis through wrist_origin_m for ONE solid.
 Takes the solid's material tag and looks up the matching density so steel/brass
 pivot pin (6.2-6.7× PETG density) is not silently miscounted. Geometry MUST be in
 metres. Returns I in kg·m²."""
 rho = MATERIAL_DENSITIES_KGM3[material]
 props = _volume_props(solid)
 wrist_axis = gp_Ax1(gp_Pnt(*wrist_origin_m), gp_Dir(0, 1, 0)) # wrist axis = +y
 return props.MomentOfInertia(wrist_axis) * rho # length⁵ at ρ=1 × ρ → kg·m²

 def i_wrist_assembly(parts, wrist_origin_m):
 """Total I_wrist (kg·m²) over a heterogeneous assembly.
 `parts` is a list of `(solid, material_tag)` tuples — e.g.
 `[(blade_1, "petg"), ..., (blade_10, "petg"), (spine, "petg"), (pin, "steel")]`.
 Per-part density lookup; pin material tagged from BOM."""
 return sum(i_wrist_about_y(s, mat, wrist_origin_m) for (s, mat) in parts)

 # CI assertion (tests/test_geometry/test_inertia_heterogeneous.py):
 # Assemble a 10-PETG-blade + steel-pin reference model in CadQuery, export to STEP,
 # run OCC `BRepGProp` mass-properties on the STEP file → `M_ref` (kg).
 # Compute `M_calc = Σ ρᵢ · Vᵢ` from the generator's per-part (volume_m3, material_tag)
 # output using the same MATERIAL_DENSITIES_KGM3 the generator uses internally.
 # Assert `|M_calc − M_ref| / M_ref < 0.001` (0.1% threshold, catches a stray
 # uniform-density fallback or a missing-pin path).
 # The 5-15% I_wrist drift mentioned in earlier reviews is unverified — actual ΔI_wrist
 # from heterogeneous density depends on the pin geometry; the CI threshold is set at
 # the 0.1% floor where any drift indicates a bug, not at a claimed Pareto-error level.

 def r_com_assembly(parts, wrist_origin_m):
 """Perpendicular distance from the **+y wrist axis** to the assembly center of
 mass (m). The wrist axis runs along +y through `wrist_origin_m`, so the
 perpendicular distance from a point (x, y, z) to that axis is **√(x² + z²)**
 — NOT √(x² + y²). The earlier draft used the XY-projection, which is the
 perpendicular distance to the +z axis (the pivot pin), not the +y wrist
 axis. The bug is invisible to the uniform-rod-along-x test (z = 0 makes
 the two formulas equal) but corrupts I_wrist Pareto ranking once mass
 spreads in z (panel thickness 2.2-3.8 mm).

 This is **position-of-CoM**, NOT radius of gyration. The intent of the §6.4
 constraint r_CoM_wrist ≤ 0.160 m is "the assembly's centre of mass is at most
 55% of the way out the blade, measured from the wrist axis." That is a
 position-of-CoM bound, computed as the perpendicular distance from the
 assembled CoM to the +y wrist axis.

 `parts` is the same `[(solid, material_tag), ...]` list as i_wrist_assembly
 so the steel/brass pin's 6-7× density is correctly weighted."""
 total_m, weighted_dx, weighted_dz = 0.0, 0.0, 0.0
 for (s, material) in parts:
 rho = MATERIAL_DENSITIES_KGM3[material]  # per-part density lookup
 props = _volume_props(s)
 com = props.CentreOfMass # world-frame, m
 m_s = props.Mass * rho # Mass at ρ=1 returns m³; ×ρ → kg
 weighted_dx += m_s * (com.X - wrist_origin_m[0])
 weighted_dz += m_s * (com.Z - wrist_origin_m[2])  # perpendicular to +y axis = √(x²+z²)
 total_m += m_s
 x_com = weighted_dx / total_m
 z_com = weighted_dz / total_m
 return (x_com * x_com + z_com * z_com) ** 0.5
 ```

 The generator emits `I_wrist_kgm2`, `r_CoM_wrist_m`, and `m_total_kg` into `params.json` from a single OCP volume-properties pass per solid. **Locked convention (matches §0 / §9.4):** wrist origin is at world (0, 0, 0); pivot is at `(D_HANDLE_M, 0, 0) = (0.05, 0, 0)`. The generator and the meshing scripts both place the wrist at the world origin — there is no conditional case where the pivot lives at world origin. `I_wrist = I_pivot + m_total · d_handle²` falls out of OCP's parallel-axis reduction automatically.

 **Required regression tests (must all pass before BoTorch sees `I_wrist`):**

 ```python
 # tests/test_geometry/test_inertia.py

 def test_uniform_rod_about_wrist_y():
 """Uniform PETG rod 0.200 × 0.010 × 0.005 m extending in +x, centered at x=0.100,
 with its end at the wrist origin (0,0,0). I about the wrist +y axis through origin:
 I_wrist_y = ∫(x² + z²)ρ dV = m·(L_x²/12 + L_z²/12 + x_cm²)
 = 0.0127·(0.200²/12 + 0.005²/12 + 0.100²)
 = 0.0127·(0.003333 + 2.083e-6 + 0.01)
 = 0.0127·0.013336 ≈ 1.694e-4 kg·m²
 (Coincidentally the same numerical value as the prior +z-axis test because
 the rod is thin in z — the parallel-axis term dominates and is identical
 about y or z for this geometry. The test discriminates the +y axis from
 the +z axis only when paired with `test_rotated_plate_*` below, which
 exercises body/world rotation handling.)"""
 rod = cq.Workplane("XY").box(0.200, 0.010, 0.005).val
 rod = rod.translate((0.100, 0, 0)) # pivot end at the wrist origin
 I = i_wrist_about_y(rod, material="petg", wrist_origin_m=(0.0, 0.0, 0.0))
 assert abs(I - 1.694e-4) < 5e-7, f"got {I}, expected ~1.694e-4 kg·m²"

 def test_rotated_plate_about_x_axis_then_iwrist_y():
 """Thin plate rotated 30° about world x (NOT y), then I about world y computed
 by OCP. Rotating about x mixes I_yy_body and I_zz_body into world I_yy, so
 this test actually exercises the body/world rotation handling (rotating about
 y would leave I_yy invariant and pass under either correct or buggy code)."""
 plate = cq.Workplane("XY").box(0.020, 0.100, 0.001).val
 plate = plate.rotate((0, 0, 0), (1, 0, 0), 30.0) # rotate about x
 I = i_wrist_about_y(plate, material="petg", wrist_origin_m=(0.0, 0.0, 0.0))
 # Analytic: I_yy_world after x-rotation by θ = I_yy_body·cos²θ + I_zz_body·sin²θ
 m = 1270.0 * 0.020 * 0.100 * 0.001
 I_yy_body = (m / 12.0) * (0.020**2 + 0.001**2)
 I_zz_body = (m / 12.0) * (0.020**2 + 0.100**2)
 theta = np.radians(30.0)
 I_expected = I_yy_body * np.cos(theta)**2 + I_zz_body * np.sin(theta)**2
 assert abs(I - I_expected) < 1e-9, f"got {I}, expected {I_expected:.3e} kg·m²"
 ```

 Any code path that fails either test must not reach BoTorch — the I_wrist Pareto objective is meaningless without unit lock + correct body/world handling. The rod test verifies the SI unit lock and parallel-axis reduction about the **+y wrist axis** (matching §6.4 / §0); the rotated-plate test (about x, not y — y-rotation leaves I_yy invariant and would silently pass under a body-frame-only implementation) verifies the body/world rotation handling that the OCP `MomentOfInertia(axis)` path gives us for free. The earlier draft of these tests called `i_wrist_about_z`, which doesn't exist in §6.4 (the locked function is `i_wrist_about_y`) — the rename also recomputes the expected analytic values for the +y axis.

 **r_com_assembly XZ-projection discriminator (closes C1):**

 ```python
 # tests/test_geometry/test_r_com_xz_projection.py
 def test_r_com_assembly_xz_projection():
     """Discriminator test: a point mass at (0.1, 0.0, 0.05) is at perpendicular distance
     √(0.01 + 0.0025) ≈ 0.1118 m from the +y wrist axis through the origin. The XY-projection
     (the prior buggy formula) returns 0.1 m. The XZ-projection (the locked formula) returns
     0.1118 m. This test FAILS under the XY formula and PASSES under XZ, discriminating
     them where the uniform-rod-along-x test cannot (z = 0 for the rod makes both formulas
     return the same value)."""
     # Small cube acting as a near-point mass at (0.1, 0.0, 0.05) m
     block = cq.Workplane("XY").box(0.001, 0.001, 0.001).val
     block = block.translate((0.1, 0.0, 0.05))
     r = r_com_assembly([(block, "petg")], wrist_origin_m=(0.0, 0.0, 0.0))
     assert abs(r - 0.1118) < 1e-3, f"got {r}, expected ≈ 0.1118 m (perpendicular to +y axis)"
     # The XY-projection would return 0.1 m exactly; the test fails on that.
 ```

 **Why inertia (about wrist), not mass — peak-effort framing ():** Felt effort during waving has two distinct components: (1) **peak torque the wrist must produce**, which sets the moment of maximum perceived heaviness, and (2) **integrated work per cycle**, which sets metabolic load over time. The peak torque is `τ_peak ≈ I_wrist · α_max + τ_aero_peak` — the inertial contribution scales linearly with I_wrist and dominates at the SHM turning points where α is maximum and v is zero. The cycle work `W_cycle = ∫ τ_aero · ω dt` is dominated by **aerodynamic resistance**: the inertial component `∫ I_wrist · ω · (dω/dt) dt = [½ · I_wrist · ω²]₀ᵀ` integrates to **zero over a complete cycle** because kinetic energy returns to its starting value at the end of each period. **I_wrist is therefore the right Pareto objective for peak felt effort** — what users describe as "how heavy does it feel" at the turning points and during direction reversal. The Phase 6 IMU measures the full τ·ω signal; the BO-optimizable inertial component (`I_wrist`) sets the peak amplitude of the torque envelope, while the aero component (out of structural control, dominated by J_fan) sets the cycle integral. Two 30 g fans with identical mass but different mass distributions feel completely different at the turning points — tip-heavy is exhausting because `I_wrist · α_max` is large, hub-heavy is effortless because `I_wrist` is small. Optimizing on mass alone lets BO cheat by pushing material outboard "for free." I_wrist captures the lever the optimizer can actually pull on. (Earlier draft claimed W_cycle alignment with `∫ I·ω·(dω/dt) dt`; that integral is identically zero over a complete cycle — the alignment claim was mathematically wrong. C4 reframes the rationale to peak-felt-effort, which is what I_wrist actually captures.) **Compute cost:** identical to mass — both are mass-distribution integrals over the same CAD body, so no Tier-affecting impact on the architecture bandit.
3. **Minimize peak pivot stress** (across all 10 panel pivot holes in the assembly; under panel-pivot architecture the ribs do not carry pivot holes — see §3.1.2 / §0 row 25). The peak pivot stress is the maximum σ_VM at the panel pivot bore across all 10 panels, evaluated against the §3.1.5 K_t hotspot table per-mode allowables (5.58 MPa tension at K_tt = 2.42 / 4.22 MPa bending at K_tb = 3.2 / 2.00 MPa bearing Z-direction at K_t_bearing = 1.5).
4. **Minimize folded form factor ** = Σ per-blade-thickness across all 10 blades; reflects the bulk-vs-compactness trade-off. With panel thickness 2.2-3.8 mm × 10 blades + 2-4 mm spacers, the folded stack ranges 22-42 mm at 10 blades (covering the ≤50 mm target across the full BO range). Making it an explicit Pareto objective lets the optimizer expose the trade-off rather than hide it; user picks the Pareto point.

**Hard constraints :**
- Total fan mass `m_total < 100 g` (hard constraint retained — protects against absurd-mass solutions that happen to be inertia-balanced).
- **Center-of-mass radius `r_CoM_wrist ≤ d_handle + 0.55 · L_blade = 0.160 m`** . The original "55% of the way out the blade" intent is preserved by *adding* the d_handle offset to the threshold when the axis switches from pivot to wrist. **Sanity check (H14, corrected kinematics per MED-3 Round-9 lock — earlier draft applied the angular-spread factor to the static handle portion, which is geometrically wrong because blades pivot about the pin at world x = 0.05 m, not about the wrist origin):**

 Under the trapezoidal panel widening (b = 5 mm at pivot end, B = 45 mm at tip), the per-blade centroid in the **blade frame (from the pin)** is:

   `r_centroid_from_pin = L_blade · (b + 2·B) / (3·(b + B)) = 0.20 · (5 + 90) / (3·50) = 0.20 · 0.633 = 0.127 m`

 For a blade at angle θ_i from the centerline, the centroid in **world coordinates** is at:

   `(x_centroid_i, y_centroid_i) = (d_handle + r_centroid_from_pin · cos θ_i, r_centroid_from_pin · sin θ_i)`
   `                            = (0.05 + 0.127 · cos θ_i,  0.127 · sin θ_i)`

 The blade radiates from the **PIN** at world x = d_handle = 0.05 m, NOT from the wrist (world origin); only the post-pin segment fans angularly. The static handle portion (x ∈ [0, 0.05]) does NOT fan.

 With the 133.3° angular-spread factor `mean(cos θ_i) ≈ 0.75`:

   `x_CoM_blades = 0.05 + 0.127 · 0.75 = 0.145 m`

 With the handle/pin contribution at ~0.025 m (~10 g of handle/pin / 100 g total = 10% mass weight per C9 lock):

   `full assembly r_CoM ≈ 0.90 · 0.145 + 0.10 · 0.025 ≈ 0.133 m`

 Margin against the 0.160 m bound: `(0.160 − 0.133)/0.160 = ` **17%** (was 31% under pre-trapezoidal assumption; was ~19% under the earlier incorrect kinematics that applied the spread factor to the handle portion — that 19% number is retired per MED-3 Round-9 lock). Bound is still feasible but headroom is tighter than the earlier 19% claim suggested. The corrected 0.160 m bound is computed from `r_com_assembly(..., wrist_origin_m=(0, 0, 0))` (wrist at world origin per the locked convention); shares data with the I_wrist call.
- Peak panel-pivot stress in all three modes (tension ≤ 5.58 MPa at K_tt = 2.42, bending ≤ 4.22 MPa at K_tb = 3.2, bearing ≤ 2.00 MPa Z-direction at K_t_bearing = 1.5; per §10.1). - Manufacturability score (§N7) ≥ 0.5

**BO algorithm (two-stage MFMO; ):** BoTorch doesn't ship a single `qMultiFidelityMultiObjectiveEHVI`, and naively pairing qMFKG (single-objective multi-fidelity) with qNEHVI (multi-objective single-fidelity) is underspecified. The campaign uses a **two-stage acquisition**:

1. **Stage 1 — fidelity learning for J_fan only:** `qMultiFidelityKnowledgeGradient` runs **only on the J_fan objective** (the one objective with CFD fidelity tiers Tier -1 / Tier 0 / Tier 1). qMFKG decides which fidelity to evaluate J_fan at next, based on the cost model and the J_fan posterior. The other three Pareto objectives (I_wrist, peak_pivot_stress, folded_form_factor) have NO CFD fidelity — they are deterministic geometry quantities (I_wrist, folded_form) or have their own non-CFD fidelity chain (stress: Filter 2 closed-form / §59.5 FEA / Phase 6 measurement).
2. **Stage 2 — Pareto selection at the target fidelity:** `qNoisyExpectedHypervolumeImprovement` (or `qLogNoisyExpectedHypervolumeImprovement`) runs over `(J_fan_at_target_fidelity, I_wrist, peak_pivot_stress, folded_form_factor)`. The J_fan posterior at target fidelity is computed from the multi-fidelity GP (so qMFKG's fidelity-learning compounds into qNEHVI's Pareto selection). The other three objectives use their own single-objective GPs / deterministic functions.

**Per-objective surrogate models (NOT a single ModelListGP over all 4):**
- **J_model:** `SingleTaskMultiFidelityGP` over (params, fidelity); only objective with real CFD fidelity.
- **I_wrist_model:** deterministic function within a fixed architecture (computed from the CadQuery geometry via §6.4 `i_wrist_assembly` using the architecture's cached rib SIMP output); identical across the three CFD fidelities since I_wrist is geometry-only. (The geometry changes when the architecture-bandit switches architecture — different blade count, different rib SIMP output — so the "identical at every fidelity" claim is true for CFD fidelity tiers Tier -1/0/1 but NOT across architectures.) In the qNEHVI step, treated as a `DeterministicModel` (BoTorch supports this directly) OR a single-fidelity GP with `train_Yvar = 1e-10` so the posterior collapses to a point estimate.
- **folded_form_model:** deterministic (geometry-only, same treatment as I_wrist).
- **stress_model:** its own multi-fidelity chain (Filter 2 closed-form pre-screen / §59.5 FEA gate / Phase 6 measurement). For the BO inner loop, use the Filter 2 closed-form estimate as a single-fidelity GP — §59.5 is too expensive to call inside the BO loop and only runs on top-3 at handoff.

The four per-objective posteriors are stitched together inside qNEHVI via a `ModelListGP` (BoTorch's pattern for heterogeneous posteriors). Top 3 designs by Pareto coverage (light corner / knee / heavy corner) are printed and validated in Phase 6.

**hypervolume-plateau early-stop (NEW):** four-objective Pareto fronts typically want O(100) non-dominated points for a useful frontier; reaching that from 600-1100 noisy evaluations under tier weighting is feasible but tight. Without a stop check, the BO will exhaust the full compute budget even after the Pareto front stops improving. adds a **hypervolume monotone-improvement check** to §Phase 4 step 56 `--iterate` orchestrator:

```
# After each BO step:
hv_history.append(compute_hypervolume(pareto_front, ref_point))
HV_baseline = hv_history[200] if len(hv_history) > 200 else None  # fixed reference at round 200
HV_FLOOR_ROUNDS = 500     # locked hard floor — see §0
HV_TIGHTNESS    = 1e-4    # cumulative 50-round HV gain / HV_baseline threshold
if len(hv_history) >= HV_FLOOR_ROUNDS and HV_baseline is not None:
 fifty_round_gain = (hv_history[-1] - hv_history[-50]) / HV_baseline
 if fifty_round_gain < HV_TIGHTNESS: # ~0.01% over 50 rounds
 log_convergence(f"hypervolume plateau detected at round {len(hv_history)}")
 declare_phase4_converged()
 # remaining compute budget reallocated to FEA gate top-50 (fallback ladder)
 break
```

**Pre-registered thresholds (locked in `configs/bo_convergence.yaml`):** `HV_FLOOR_ROUNDS = 500`, `HV_BASELINE_ROUND = 200` (fixed reference, NOT moving), `HV_TIGHTNESS = 1e-4`. **Compute saved:** if the plateau fires at round 600 instead of running to 1100, ~500 wall-clock hours of compute moves to FEA gate top-50 (fallback ladder gets a larger candidate pool) and Phase 5 verification.

J_fan_peak is retained in the **Drive/JSONL ledger** as a non-Pareto auxiliary; not part of the 4D objective vector but available for post-hoc analysis.

BoTorch supports multi-objective BO via `qNoisyExpectedHypervolumeImprovement`:

```python
from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement
from botorch.models import ModelListGP

# 4D Pareto: (J_fan, I_wrist, peak_pivot_stress, folded_form_factor).
# Y_inertia (about the wrist axis) — see §6.4 rationale.
gp_jfan = SingleTaskGP(X, Y_jfan)
gp_inertia = SingleTaskGP(X, Y_inertia) # I_wrist (kg·m²) about the handle-grip wrist axis
gp_stress = SingleTaskGP(X, Y_stress)
gp_form = SingleTaskGP(X, Y_form_factor)
model = ModelListGP(gp_jfan, gp_inertia, gp_stress, gp_form)

ref_point = torch.tensor([
 Y_jfan.min - 0.1,
 -(Y_inertia.max + 0.1), # minimize ⇒ negate
 -(Y_stress.max + 0.1),
 -(Y_form_factor.max + 0.1),
])
qNEHVI = qNoisyExpectedHypervolumeImprovement(
 model=model, ref_point=ref_point, X_baseline=X,
)
```

---

## 7. 3D Printing Materials Selection

### 7.1 Material Properties for Rib Printing

| Property | PLA | PETG | PA (Nylon) | PLA-CF |
|----------|-----|------|------------|--------|
| **Tensile Strength (MPa)** | 50-60 | 40-50 | 40-85 | 60-70 |
| **Young's Modulus, FDM XY (GPa)** | 2.5-3.2 | 1.1-1.5 | 0.8-1.5 | 3.5-5.5 |
| **Elongation at Break (%)** | 3-6 | 15-25 | 30-100+ | 2-4 |
| **Fatigue Resistance** | Poor | Good | Excellent | Poor |
| **Print Difficulty** | Easy | Easy-Medium | Hard | Medium |
| **Cost ($/kg)** | $15-25 | $20-30 | $30-50 | $30-50 |

### 7.2 Material Recommendations for Folding Fan Ribs

#### Best Overall: PETG

- Good balance of stiffness and flexibility for thin rib structures.
- Excellent fatigue resistance -- critical for the pivot stress concentration.
- Easy to print, no enclosure needed.
- The slight flexibility (compared to PLA) is actually beneficial: it prevents brittle fracture at the pivot hole under cyclic loading.

#### Best for Prototyping: PLA

- Easiest to print, cheapest.
- Sufficient for evaluating geometry, fit, and pivot mechanism.
- NOT recommended for final fan due to brittleness and poor fatigue life at the pivot.

#### NOT Recommended: PLA-CF

- Extremely brittle (2-4% elongation). The pivot hole stress concentration combined with cyclic loading will cause rapid fatigue failure.

### 7.3 Print Settings for Thin Fan Ribs

| Parameter | PLA (prototype) | PETG (final) |
|-----------|-----------------|--------------|
| **Layer Height** | 0.15-0.2 mm | 0.15-0.2 mm |
| **Wall Count** | 3-4 (most of rib is walls at 2 mm thickness) | 3-4 |
| **Infill** | N/A (rib is too thin for infill; all walls) | N/A |
| **Print Speed** | 50-60 mm/s | 40-50 mm/s |
| **Orientation** | Flat on bed | Flat on bed |
| **Nozzle** | 0.4 mm (or 0.3 mm for finer features) | 0.4 mm |

**Note on rib printing:** At 2 mm thickness and 0.4 mm nozzle, each rib is approximately 5 perimeters wide. There is no room for infill -- the rib is essentially solid perimeters. This is structurally ideal because perimeter walls are stronger than infill patterns.

**Print orientation:** Ribs MUST be printed flat. This aligns the strong (XY) material direction with the primary bending loads during waving.

### 7.4 Single-Material PETG Print Considerations for V-Unit Blades
The project uses single-material PETG (no TPU membrane, no multi-material). This section covers print considerations for the V-unit blade architecture; §7.1-§7.3 above cover general PETG properties.

#### 7.4.1 Click Feature Print Considerations

The mating chamfer + detent click features on each blade's panel outer tangential edge at the tip (per item #3 panel-edge relocation + HIGH-8 Option A lock) are the most print-tolerance-sensitive geometry.

| Parameter | Recommendation |
|-----------|----------------|
| Design clearance per mating surface | **0.15-0.20 mm** to absorb FDM ±0.1 mm tolerance |
| Chamfer angle | 45° (printable without supports at typical print orientations) |
| Chamfer overlap depth | 0.5-1 mm |
| Detent bump radius | 0.3-0.5 mm (test smaller bumps in Spike 0.4 first) |
| Layer height in click feature zone | ≤0.15 mm (finer than the 0.2 mm default elsewhere) for better dimensional resolution at the detent surfaces |
| Wall count in click feature zone | 3+ to ensure detent geometry is fully formed (not infilled) |
| Print orientation effect | If the blade prints rib-flat (chamfer face perpendicular to build plate), the chamfer prints cleanly. If the blade prints in deployed-V orientation, the chamfer face is at an angle and may need supports. **Phase 1 print-orientation categorical includes this consideration.** |

**Spike 0.4 validates:** the as-printed clearance, click engagement force, deployed-state alignment, and 1000-cycle fatigue life. If clearance is too tight, blades seize; if too loose, click does not retain.

#### 7.4.2 Per-Blade Print Settings (PETG)

print settings for one V-unit blade (rib + panel + rib as one body):

| Parameter | Default | Notes |
|-----------|---------|-------|
| Layer height | 0.15-0.20 mm | 0.15 in click-feature zone, 0.20 elsewhere |
| Wall count | 3-4 | Most of the rib volume is walls at 2 mm thickness |

| Panel infill density | **Generator-determined:** the §9.7 generator produces the panel geometry directly (possibly with Layer 2 cutouts, porosity, TPMS lattices, louver slits, etc. — the panel is NOT always a solid body). The slicer prints whatever the STL encloses. Set walls = 12+ for full perimeter coverage on a 5 mm panel; the slicer's "infill density" parameter is moot because the generator already produced the solid/voided regions in the STL. | No SIMP density on the panel. |
| Print speed | 40-50 mm/s | Standard PETG; no TPU-specific slowdown |
| Nozzle temp | 230-245 °C (material-specific) | Standard PETG |
| Bed temp | 70-80 °C | Standard PETG |
| Cooling | 30-50% fan | Helps thin click features hold shape |

#### 7.4.3 Print-Orientation Choice (Per-Blade)

Each blade can be printed in one of two orientations; this is a categorical Phase 4 BO variable.

| Orientation | Pros | Cons |
|-------------|------|------|
| **Rib-flat** (blade lying flat on build plate, panel parallel to bed) | Layer lines run along the rib length → aligned with primary bending load (strong direction). Click features print without supports. Best dimensional accuracy on the panel surface that faces the user. | Per-blade footprint is largest (200 × ~25 mm); only ~10 blades fit on a 256 mm bed at a time. |
| **Deployed-V** (blade printing in its V cross-section, with the V opening upward) | Smaller per-blade footprint; more blades fit per bed. Layer lines run across the rib length → mixed alignment with bending load. | Click feature faces are angled; may need brim or small supports. Lower panel-surface accuracy in places. |

Default: **rib-flat** for best aero surface quality. The deployed-V option is available to the BO if the surface-finish penalty turns out to be small.

#### 7.4.4 Full-Assembly Print Option (vs Per-Blade Print)

If the user's printer bed is large enough (≥360 × 210 mm), all 10 blades can be printed in one job in the deployed-V configuration, with the entire assembly already in its deployed geometry. This eliminates the per-blade print overhead.

| Strategy | Wall-clock | Per-blade QC | Bed size required |
|----------|------------|--------------|-------------------|
| Per-blade prints (default) | ~10 × 1-2 h = 10-20 h | Inspect each blade before assembly | 256 × 256 mm bed adequate |
| Full-assembly print | ~8-12 h continuous | One-shot; if any blade fails, restart entire print | ~360 × 210 mm bed required |

The decision is made in Phase 0 / Phase 1 based on the user's available printer (per Spike 0.4 bed-size check).

---

## 8. Step-by-Step Project Execution Plan

### Timeline Expectations

**Realistic timeline (4-layer hybrid generative design):** **~10-13 weeks**.

**Time breakdown:**
- Phase 0 (Step 0.0 scaffolding + Risk spikes 0.1-0.7 + environment + baseline): 1-2 weeks
- Phase 1 (Generative parametric blade geometry pipeline): 1 week
- **Phase 2a (Baseline 2D CFD slice for rib-TO loads, ): ~½ day, Week 3.5** — runs the §Phase 3 slice pipeline once on the Spike 0.3 baseline; emits `phase3_baseline.csv` so Phase 2 doesn't crash on the missing-file dependency
- Phase 2 (Rib-only plate-bending TO): 1 week — runs after Phase 2a
- Phase 2b (Generative parametric optimization seeding + LHS): 3-4 weeks (seeds the multi-fidelity GP for Phase 4)
- Phase 3 (2D steady CFD slice, rigid corrugated, no FSI; serves as Tier -1 of MF stack; 2D unsteady is preserved as inner-loop verification fidelity, not the screening tier): 1 week
- Phase 4 (Multi-fidelity BO at **~37-46D** on Colab Pro): 3-4 weeks (compute-dominated; 300-600 hours over 2-4 weeks with parallel Colab sessions)
- Phase 5 (High-fidelity verification + PyFR on top-3): 1-2 weeks
- Phase 6 (Single-material PETG print of top-3 + IMU + acoustic measurement): 1-2 weeks

Phase 2c and Phase 2d are removed in (see §0).

### Phase 0: Project Scaffolding + Risk Spikes + Environment (Week 1-2)

Phase 0 runs Step 0.0 + Spikes 0.1-0.7 + 0.6c (no TPU/preCICE/FSI spikes; sequential and dependency-respecting). Each spike is a go/no-go that determines the downstream plan; fallback paths are pre-specified.

**Execution order (depends-on chain flows forward, not backward):**

| New # | Old # | Description | Depends on |
|-------|-------|-------------|------------|
| 0.0 | | Project scaffolding (Git repo + CI) | — |
| 0.1 | 0.1 | Fusion headless add-in workflow | — |
| 0.2 | 0.8 | Torsional-pendulum rotational-inertia protocol | — |
| 0.3 | 0.4 | Baseline physical measurement (10-blade flat-panel fan) | 0.2 (needs inertia rig) |
| 0.4 | 0.1c | Click-feature tolerance + 1000-cycle test | — |
| 0.5 | 0.5 | Single-blade fab-noise floor (3-copy CV) | 0.4 (needs click features) |
| 0.6 | 0.3 | Colab Pro compute budget probe | — |
| 0.6c | (H10) | Tier-1 unsteady-config sanity (V1: 0.6c.1 only; 0.6c.2 deferred to Phase 5 step 62.5 per 2026-05-14 decision) | 0.6 (Colab compute budget probe) |
| 0.6d | (H10 supplement; 2026-05-14 addition) | Tier-1 quantitative-sanity counter-checks (0.6d.1 + 0.6d.2 gating, 0.6d.3 advisory) | 0.6c.1 (cfg sanity must clear before quant checks); may reuse 0.6c Cell-6 NACA 0012 mesh |
| 0.7 | 0.9 | Generative-geometry + BO-infra sanity check | — |


**Step 0.0 : Project scaffolding (~1-2 hours, runs FIRST before any spike).**

Before any spike runs, create the `fan-optimization/` Git repository with the full directory tree from §12. This ensures every spike script, test, and config file lands in the right place from the start rather than being moved into structure later. Full deliverables list in the original §12.5 description — preserved at the end of this section under "Step 0.0 detail" for backward reference.

**Spike 0.1: Fusion headless add-in workflow on macOS**

- Question: Can Fusion be driven headlessly from a Python script on macOS (read JSON, regenerate model, export per-blade STLs + STEP)?
- Approach: Minimal Fusion Python add-in reads `params.json`, sets User Parameters on a one-blade test file, exports STL. Drive via `Fusion.app/Contents/MacOS/Fusion`; fall back to AppleScript UI automation if needed.
- **Fallback:** CadQuery becomes the sole geometry backend; Fusion is viewer/manufacturable-cleanup only.

**Spike 0.2: Torsional-pendulum rotational-inertia protocol**

- Motivation: Phase 6 reports `J_fan_measured / angular_work_per_cycle`, requiring the assembled fan's rotational inertia **I_wrist about the handle-grip wrist axis** (same axis the §6.4 BO Pareto objective uses; **not** the pivot pin axis, which is offset by d_handle ≈ 50 mm). This rig must exist before Spike 0.3 (baseline measurement) can be evaluated in IMU-normalized form.
- Approach: Build a simple torsional pendulum -- suspend the assembled fan from a calibrated torsion wire or thin rod **attached at the handle's wrist-grip point** (locking the rotation axis to coincide with §6.4's wrist axis, NOT the pivot pin). Measure natural oscillation period T_osc. Compute I_wrist = κ · (T_osc / 2π)², where κ is the torsion constant (calibrate once with a known reference mass at a known radius). Alternative: bifilar pendulum, again suspended from the handle wrist-grip point.
- **Output:** Measurement protocol + a calibrated rig that produces I_wrist_design in <10 minutes. Used by Spike 0.3 (baseline) and Phase 6 (each printed design).
- **Pass criterion:** Inertia measurement repeatability < 3% across 5 measurements of the same fan; **cross-check against generator-emitted `I_wrist_kgm2` (§6.4) agrees within ±10%** for the Spike 0.3 baseline (any larger disagreement indicates an axis-convention or density-mismatch bug that must be resolved before Phase 6). **Toolchain note :** the Spike 0.3 baseline geometry (10-blade flat-panel fan with uniform-thickness Layer 1 and all Layer 2/3 disabled) **must be routed through the same §9.7 generator + §6.4 `i_wrist_assembly` call site** that Phase 4 designs use, so the pendulum measurement compares against the same SI-locked, axis-correct computation. Phase 1 Step 6 `smoke_test.py` is extended to emit `I_wrist_kgm2` for the baseline so Spike 0.2 has a target value to compare against.

**Spike 0.3: Baseline physical measurement** *(depends on Spike 0.2)*

- Print a 10-blade baseline fan in PETG with simple flat panels (no TO, no airfoil camber). Measure peak and average airflow at 300 mm with a handheld anemometer + IMU-instrumented handle over 5 trials of 10 cycles each, metronome-paced at 2 Hz. Compute IMU-normalized angular work per cycle using the rotational inertia measured in Spike 0.2.
- **Reports both `J_fan_proxy` (anemometer) and `J_fan_proxy / W_cycle` (IMU-normalized).** The IMU-normalized number is the canonical baseline that all optimized designs must beat.

**Spike 0.4: Click-feature tolerance, cycle life, AND V1 lock force balance (H6 lock)**

**V1-lock force balance (Phase 0 measurement; the only in-V1 path that prevents the fan unlocking under sustained 2 Hz waving):**

Measure cumulative click-engagement friction across 9 inter-blade pairs at the deployed position (force gauge applied tangentially at the **panel's outer tangential edge at the tip** — where the click chamfer + detent live per the item #3 panel-edge relocation; specifically at `(x = L_blade, y = ±panel_tangential_outer)`. The rib does NOT reach this x — it terminates at `L_blade − RIB_TIP_TAPER = 0.185 m` per Architectural A. Record `F_friction_cumulative`). Compute the peak inertial torque the click features must hold against:
- `τ_inertial_peak = I_wrist · α_max` (using the Spike 0.2 measured I_wrist and α_max = 110 rad/s²)
- Convert to tangential force **at the click location** using the wrist-to-tip lever arm: `F_inertial_at_click = τ_inertial_peak / L_wrist_to_tip = τ_inertial_peak / 0.25 m` (H8 lever-arm lock — the click is at the **panel's outer tangential edge at the tip**, which sits at `d_handle + L_blade = 0.25 m` from the wrist axis, NOT `/ 0.20 m` from the pivot)
- **Pass criterion:** `F_friction_cumulative ≥ 2 × F_inertial_at_click` (factor-of-2 safety margin so the fan stays deployed under nominal waving)

**V1 fallback geometry (auto-armed if Spike 0.4 force balance fails):** a **printed rib-tab** on each guard blade — a 3 mm × 5 mm × 1.5 mm tab on the guard's outer face at the deployment-angle position, mating a 0.2 mm-deeper pocket on the adjacent inner blade's outer rib. Snap-fit engagement at 133.3° deployed extent; deliberate 3-5 N radial disengagement force. Added to `params.layer4.v1_lock_fallback_enabled` as a binary flag, default `False` (off unless Spike 0.4 force balance triggers it). `docs/V2_backlog.md` notes the V2 designed lock supersedes V1 rib-tab.

- Question: Does the printer produce click features that engage reliably without seizing, and survive the deploy/fold cycle life requirement?
- Approach: Print 2 single-blade test articles (each ~200 mm × 25 mm × 2 mm) with mating chamfer + detent on **their facing panels' outer tangential edges at the tip** — per item #3 panel-edge relocation; the chamfer does NOT live on the rib face. The test article geometry must match the production-blade geometry at the click region (panel outer tangential edge at `x = L_blade`, `y = ±panel_tangential_outer ≈ ±0.0225 m`) so the friction measurement is representative. Assemble on a temporary pivot pin. Measure:
 - As-printed clearance at the click feature (target 0.15-0.20 mm per mating surface; calipers + feeler gauge).
 - Click engagement force (deploy from folded; force gauge at the blade tip; target 0.5-2 N).
 - Deployed-state alignment (visual; gap between adjacent blade tips should be uniform).
 - Cycle to 1000 deploy/fold cycles; inspect detent geometry for wear or fracture after every 100 cycles.
 - **high-amplitude stress segment (after the 1000-cycle low-amplitude run):** add **100 cycles at ~2× design-point engagement force** (force-gauge at the blade tip; target 1-4 N, vs the canonical 0.5-2 N). Inspect for detent fracture, chamfer chipping, or alignment drift between the high-amplitude segment and the resumed low-amplitude state. Pass criterion: detent geometry intact after the 100 high-amplitude cycles. If the click features fracture under occasional peak engagement, the magnetic-upgrade fallback (per the Spike 0.4 fallback path elsewhere in this section) is triggered.
- **Pass criterion:** Clearance within target band; engagement force within target band; no detent fracture or excessive wear after 1000 cycles; alignment gap variation < 1 mm.
- **Fallback if it fails:**
 - Detent fracture or excessive wear → upgrade to embedded neodymium magnetic catch (adds ~20-40 g to the assembly, still within 100 g constraint (C9 lock)).
 - Clearance out of tolerance → re-tune slicer settings or invest in printer calibration; may require linear advance / pressure advance tuning.

**Spike 0.5: Fabrication-noise floor for single-blade prints** *(depends on Spike 0.4)*

- Motivation: The projected 15-30% J_fan gain must clear the printer's part-to-part noise floor.
- Approach: Print **three identical copies of a single representative blade** on the same printer with the same settings (three single blades, not three full fans, because the blade-to-blade variation is the actual quantity of interest). Requires Spike 0.4's validated click-feature geometry. Measure each blade's dimensional accuracy (calipers at 10 points), mass (jewelry scale), and three-point bend deflection under a known load. Assemble each into a 10-copy of the baseline fan (one new blade + 9 unchanged baseline blades) and measure J_fan-proxy.
- **Pass criterion:** CV < **5%** of mean across the three single-blade fan results.
- **Mitigation if CV > 5%:** Tighten print process or commit only to gains >15% (memo issue #16). Document the achieved CV in the **Drive/JSONL ledger**; all subsequent J_fan deltas must be compared against this floor.

**Spike 0.6: Colab Pro compute budget probe**

- Question: What is the actual per-evaluation compute cost on Colab Pro for the planned workloads, and is the MacBook M3 usable for local SU2 and local FEA at all?
- Approach: Run one representative 3D unsteady SU2 case (500K cells, 5 pitching cycles, dt=T/200) on a Colab Pro CPU instance and on a Colab Pro G4-class GPU node; record wall-time and compute-unit consumption. **Sub-spike 0.6a :** run one Tier -1 case (CadQuery → Gmsh 2D corrugated slice → SU2 2D steady → `j_fan.py`) end-to-end on the MacBook M3. **Pass criterion:** completes in ≤15 min and produces a finite J_fan_steady_proxy. This is the gate for the M3 being usable for any local SU2 (Phase 1 smoke test, single-eval debugging). **Sub-spike 0.6b :** run one representative FEniCSx (or CalculiX) static FEA case on the M3 — a simple cantilever rib under a 5 N tip load. **Pass criterion:** completes in ≤2 min and matches the analytic tip deflection within 5%. This is the gate for **Phase 5 step 59.5 / step 64.5 running locally on the M3**. **Fallback if 0.6a fails** (e.g., conda-forge SU2 won't build cleanly on ARM64, or it builds but throws on AD-enabled binaries): shift `smoke_test.py` to a Colab Pro CPU session. **Fallback if 0.6b fails** (FEniCSx/CalculiX ARM64 build issues — historically correlated with SU2's ARM64 status): **move step 64.5's combined-blade structural FEA to a Colab Pro CPU session** (still cheap — 5 min × 3 designs × 2 load cases = ~30 min of Colab CPU; the assertion + design-rotate logic in step 64.5 runs from the Mac driver, only the FEA itself runs on Colab). The M3 retains geometry/mesh-QC/Fusion/IMU roles regardless.
- **Status:** Treated as calibration, not a gate (memo §3: compute is not a constraint). Useful to confirm before Phase 4 launches.

**Spike 0.6c: Tier-1 unsteady-config benchmark validation (H10 lock — gates Phase 4 launch)**

- Motivation: the compressible-with-low-Mach-prec + RIGID_MOTION + near-zero-ambient + 5-cycle-dual-time-stepping numerics combination is locked on engineering judgment. With no benchmark validation, the entire Tier 1 dataset (the only "true J_fan" tier) rests on unvalidated numerics — a silent error in any of the locked numerics would propagate through every Phase 4/5 Tier-1 result.
- **Sub-spike 0.6c.1 — Tier 1 cfg sanity check** (must run BEFORE the benchmark): render the locked Tier 1 cfg, attempt to parse + run for **1 inner iteration** on a probe mesh. **Pass criterion:** SU2 launches and completes 1 outer time-step without parser error. If `FREESTREAM_VELOCITY = (0, 0, 0.001)` (the H10 near-zero-ambient spec from the Tier 1 cfg lock — H6 is the V1 lock force balance, a different item) is NOT a valid compressible-solver directive in the deployed SU2 build, replace with the alternative compressible-zero-flow trick: `MACH_NUMBER = 1e-9` plus explicit `REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE` + reference state. **Document the working syntax in §9.4.1 as the locked Tier 1 reference cfg.**
- **Sub-spike 0.6c.2 — Published-benchmark validation: DEFERRED TO PHASE 5 (2026-05-14 decision).** The original Phase-0 framing — run a published oscillating airfoil case through the working Tier 1 cfg setup, reference case NACA 0012 pitching about quarter-chord at `k_reduced ≈ 0.5-0.6`, low Reynolds (~30k-50k) — was demonstrated unsound by the 2026-05-14 regime diagnostic (`scripts/diagnose_su2_pitching_regime.py`): the production Tier-1 numerics simulate body-in-still-air added-mass / quadratic-drag physics correctly but cannot be validated against published *wind-tunnel-frame* references (different regime; CL bias ratio 2.234, phase lag +88.2°, CL at 2× the prescribed pitching frequency — all consistent with body-in-still-air, all inconsistent with the wind-tunnel reference data). The full evidence trail is in `docs/phase_logs/spike_0_6c.md` Note 1; the V1/V2 decision is in `docs/phase_logs/phase_0_signoff.md` Note 1. The published-reference validation moves to **Phase 5 step 62.5** as a body-in-still-air-frame benchmark with full force-trace data from the insect-flight / Morison-equation literature, run through SU2 + PyFR + OpenFOAM (3-solver cross-check). To compensate for the lost Phase-0 absolute-accuracy evidence, **Spike 0.6d** (below) lands three lightweight Phase-0 quantitative-sanity counter-checks. Phase 4 launch is no longer gated on 0.6c.2 (see updated fail-action below); 0.6c.1 + 0.6d.1 + 0.6d.2 are the V1 gate set.
- **Fail action (V1 gate set, post-2026-05-14):** Phase 4 launch is **gated on Spike 0.6c.1 PASS AND Spike 0.6d.1 PASS AND Spike 0.6d.2 PASS** (0.6c.2 deferred to Phase 5 per Note above; 0.6d.3 is advisory, NOT gating). `scripts/launch_phase4.py` refuses to create the `phase4-launch` tag unless both `data/spike_0_6c/PASS` AND `data/spike_0_6d/PASS` markers are present. (Marker path `phase0/spike_0_6c/PASS` in earlier plan text resolves to `data/spike_0_6c/PASS` per the as-built repository layout.) If any gating sub-spike fails, investigate the production Tier-1 cfg (low-Mach preconditioner coefficients, dual-time inner-iter count, dt convergence, mesh quality) before Phase 4 launches; do NOT silently proceed.

**Spike 0.6d: Tier-1 quantitative-sanity counter-checks (2026-05-14 addition; H10 lock supplement — gates Phase 4 launch alongside 0.6c.1)**

- Motivation: the 2026-05-14 Spike 0.6c regime diagnostic (see Sub-spike 0.6c.2 deferral note above) confirmed the production Tier-1 cfg simulates body-in-still-air physics correctly but cannot be validated against published wind-tunnel references — the regimes differ. Sub-spike 0.6c.2 was deferred to Phase 5 on that basis. To prevent Phase 4 from burning ~1300 GPU-hours on a Tier-1 solver output we have only consistency evidence for, three lightweight independent checks land before Phase 4 launches. The objective is to convert "we have no independent quantitative check on SU2 at MACH=1e-9" into "we have order-of-magnitude + closed-form + same-solver-incompressible-mode cross-evidence" — the absolute-accuracy claim still lives in Phase 5 step 62.5 (cross-solver + published-reference).
- **Sub-spike 0.6d.1 — Symmetry + dimensional-force sanity** (gating): on the existing Spike 0.6c Cell 8 SU2 history.csv (NACA 0012 mesh generated by Cell 6 of the 0.6c Colab notebook — already on Drive at `gdrive/fan-optimization/spike_0_6c/sub_2_run/history.csv`; **reused here for cost-efficiency** since the V1 flat-panel-baseline mesh isn't generated until Phase 4 step 46), verify (a) cycle-averaged F, M within tolerance of zero (physical symmetry of periodic pitching), (b) dimensional cycle-peak force lies within ±1 order of magnitude of the analytic added-mass + quadratic-drag envelope computed **for the NACA 0012 geometry the run was performed on** (`F_envelope = m_NACA0012 × ω² × r_cm` where mass + r_cm are derived from the airfoil's mesh-defined geometry). **Pass criterion:** (a) `|F_cycle_avg| < 0.05 × F_cycle_peak`; (b) dimensional `F_cycle_peak` within ±1 order of magnitude of `F_envelope`. **Note:** the envelope is regime-appropriate, not geometry-specific — the body-in-still-air added-mass scaling holds for any rigid pitching body. The V1 panel-specific envelope is *also* checked in step 56.5(a) once the Phase-4 mesh exists. **Cost:** ~1–2 h Colab CPU (one fresh Tier-1 run on the same NACA 0012 mesh for the dimensional cross-check; the symmetry check reuses existing Cell 8 history.csv at zero compute). **Implementation:** `src/fanopt/cfd/spike_0_6d.py` + `scripts/run_spike_0_6d_1.py`; reuses the existing `diagnose_su2_pitching_regime.py` infrastructure for the symmetry analysis; new analytic-envelope helper for the magnitude check.
- **Sub-spike 0.6d.2 — 2D flat-plate added-mass analytic check** (gating): render a 2D thin-plate pitching cfg matching the production Tier-1 numerics (MACH=1e-9, RIGID_MOTION, dual-time-stepping, same dt and inner-iter count) and compare SU2's inviscid-phase pitching moment coefficient about the pivot against the closed-form Sedov/Newman added-mass moment `M_added = -m_a I_pitch θ̈`. **Pass criterion:** SU2 cycle-peak inviscid-phase moment within ±15% of the closed-form added-mass prediction. **Cost:** ~2–4 h Colab CPU. **Implementation:** new 2D thin-plate cfg template `configs/su2/thin_plate_2d_pitching.cfg.j2` + renderer in `src/fanopt/cfd/configs.py` + `scripts/run_spike_0_6d_2.py`.
- **Sub-spike 0.6d.3 — SU2 incompressible-mode cross-check** (advisory, NOT gating): duplicate the production Tier-1 cfg with `SOLVER= INC_NAVIER_STOKES`, same mesh + motion + pitching schedule. **Pass criterion (advisory):** dimensional cycle-averaged forces agree with compressible-mode-with-MACH=1e-9 output within ±20%. **Cost:** ~2–3 h Colab CPU. **Why advisory not gating:** same-solver cross-check is weaker independence evidence than cross-codebase. Failure here is documented as a Phase-5 investigation item but does not block Phase 4. The independent-codebase cross-check moves to Phase 5 step 62.5 via OpenFOAM `pimpleFoam`.
- **Aggregate marker file:** `scripts/run_spike_0_6d.py` reads `sub_1` + `sub_2` (+ optional `sub_3` advisory) result JSONs and writes `data/spike_0_6d/PASS` iff sub_1 AND sub_2 both pass; `sub_3` advisory result is logged but does not affect the marker.
- **V1/V2 boundary:** like Spike 0.6c, Spike 0.6d is V1-scope and gates Phase 4. It does NOT depend on any hardware-instrumented measurement (consistent with the 2026-05-13 V1/V2 split documented in `docs/phase_logs/phase_0_signoff.md`). Phase 5 step 62.5 absorbs the absolute-accuracy validation work.

**Spike 0.7: Generative-geometry + BO-infra sanity check**

- Motivation (N7, N8): the rich parametric generative design (Phase 2b) produces ~37-46-D outputs that pass through a CadQuery 4-layer pipeline + manufacturability filter. Two risks: (a) the CadQuery generator may produce degenerate geometry from some parameter combinations (overlapping primitives, tangent intersections, very small overlaps); (b) the BO infrastructure (architecture bandit + multi-fidelity GP + TuRBO) may not fit and propose at this dimensionality within reasonable time. Both need validation before Phase 4 commits the full ~300-600-hour compute budget.
- **Sub-spike 0.7a — Generative geometry sanity check:**
 - Run `generate_blade.py` with 10 random parameter sets drawn from the JSON-schema bounds.
 - Visually inspect each output STL for manufacturability and reasonable topology.
 - Manually run the manufacturability filter; verify it rejects the obviously infeasible designs and accepts the obviously good ones.
 - Print 2 of the passing designs on the user's printer; confirm they print without failures and that click features engage correctly.
 - **Adversarial parameter-set sub-clause :** include ≥3 hand-picked adversarial parameter sets that would have invaded the click-feature footprint under the old (guard-only) Check 7, AND would have punched through rib material under a no-panel-mask generator. Specifically: (a) Layer 2 louver with clustered-at-tip spacing pushing cuts toward outer-rib edge; (b) Layer 2 TPMS at minimum cell size with rotation aligned to put through-cuts at the click region AND across the ribs; (c) Layer 3 primitive positioned at the bounds-edge of the new "≥5 mm from outer-rib click region" constraint. Run on inner blades (not just guards). Pass criteria: (i) click-feature footprint is bit-for-bit intact on every blade, (ii) the rib material from the Phase 2 TO output is bit-for-bit preserved (no Boolean subtraction reaches into the rib region — enforced by the §9.7.1 panel-domain invariant). Companion regression tests: `tests/test_geometry/test_click_feature_preservation.py` and `tests/test_geometry/test_panel_mask.py`.
- **Sub-spike 0.7b — BO infrastructure scaling sanity check:**
 - Run 5-10 LHS samples through the full BO infrastructure (architecture bandit + TuRBO + multi-fidelity GP) at the ~37-46 dimensions on synthetic objective values (no CFD).
 - Verify GP fit time per iteration ≤ 60 seconds; verify the architecture bandit promotes a sensible **K_promoted set** (the production K is data-driven per based on Phase 3's R²; Spike 0.7b runs in Phase 0 *before* Phase 3 finishes, so the spike uses **K = 4 hard-coded** for the synthetic-objective sanity check — production K is determined later from Phase 3's measured R²); verify TuRBO trust regions update correctly.
 - If GP fit time exceeds 60s consistently, plan to make the architecture bandit more aggressive (more time per architecture screening, fewer architectures promoted) or use a sparse-GP variant.
- **Sub-spike 0.7c — Sobol/random-search baseline (one-day exercise):**
 - Run the same architecture-bandit infrastructure with the GP+acquisition replaced by uniform-random Sobol sampling over the same parameter box. 50 Sobol samples evaluated at Tier -1.
 - Run 100 BO iterations on the same architecture set under the production GP+qMFKG configuration.
 - **Pass criterion (fixed-budget Iso-Compute comparison):** given equal CFD budgets `B ∈ {30, 100, 300} hours`, BO's best-J_fan exceeds Sobol's best-J_fan by **≥ 5% on at least 2 of the 3 budgets**. **Budget accounting — locked rules:** (1) "Hours" = **cumulative tier-(-1) + tier-0 + tier-1 compute including Sobol seed runs**; the 1000-h Phase-4 stop rule (§6.2.3) uses the same accounting, so Sobol-seed time is fully counted against the campaign budget — there is no "free" Sobol allowance. (2) **Budgets run serially with results gating next**: the B=30 comparison must complete before B=100 starts (so the 100-h GP uses the 30-h seed data); the B=100 must complete before B=300 starts. This makes the GP fit time monotonically realistic — at B=30 the GP has only seed data; at B=300 it has full multi-fidelity training. (3) Wall-clock vs cumulative-compute: Colab Pro CPU runs 2-4 parallel sessions, so 100 h of cumulative compute is ~25-50 h of wall-clock; the comparison axis is **cumulative compute** (matches the §6.2.3 stop rule), not wall-clock. The 30 + 100 + 300 = **430 h total** Sobol-vs-BO experiment is fully booked against the 1000-h Phase 4 stop-rule budget. **If BO doesn't beat Sobol on 2 of 3 fixed budgets:** **fall back to SAASBO inner-loop** (with the ≤500-inducing-point cap from §6.2.2) OR **fix the architecture set** (reduce dimensionality by collapsing Layer 2 categoricals — e.g., pin Layer 2 activation profile to a single combination — and re-run). The choice between SAASBO and architecture-set-reduction depends on which sub-axis the GP-fit-time exceeds 60 s for; both fallbacks are pre-specified in Spike 0.7b.
 - Sobol seed runs live at `gdrive/fan-optimization/phase0/sobol_seed/results.jsonl` and **double as the GP seed set for Phase 4** (so the day isn't wasted even if BO does win cleanly).
 - **Budget allocation (H7 lock):** Spike 0.7c's **430 h is booked under Phase 0**, NOT against Phase 4. The 1000-h Phase 4 stop rule starts counting at `git tag phase4-launch` (created by `scripts/launch_phase4.py`), not at Phase 0 beginning; Spike 0.7c compute is accounted under Phase 0 and does not count against the Phase 4 stop rule. Without this split, the 430 h of Sobol seed evaluations would consume 43% of the Phase 4 1000-h budget before Phase 4 launches — leaving only ~570 h for the BO inner loop, incompatible with the 600-1100 h expected range. The Sobol seed data reuses as Phase 4 GP initialization without re-running.
- **Pass criterion (all three sub-spikes):** sanity-check geometries print and engage; BO infrastructure fits and proposes in ≤ 60s/iteration; BO beats Sobol by ≥ 2σ.
- **Fallback if 0.7a fails:** tighten the JSON schema bounds (e.g., disallow primitives that are very close to each other; cap primitive sizes); add more aggressive manufacturability filter rules.
- **Fallback if 0.7b fails:** switch from TuRBO to SAASBO with ≤500 inducing points (slower per-iteration but more reliable at high D); or reduce the design space by fixing some categoricals upfront (e.g., fix Layer 2 activation profile to a 2-field combination like {louver, TPMS}, freeing 3 binary activation dims, and/or pin Layer 3 primitive presence to 1, freeing 1 binary).

**User signoff items in Phase 0:**

1. J_fan metric (§9.4) is locked.
2. Open Question #2 is closed: Phase 2 TO loads come from Phase 3 baseline CFD-derived pressures.
3. Generative parametric design (§3.1.3, §9.7) is the locked panel-topology approach; no density-based TO on the panel.

**Step 0.0 detail block — DELETED per M8 (was a duplicate of the Step 0.0 description above; risked drift).** The canonical Step 0.0 description lives in the main prose earlier in this section; §12.5 cross-references it via "see §Phase 0 Step 0.0".

**Environment setup (in parallel with the spikes):**

- Mac: conda env from `environment.yml`. **No preCICE, no TPU, no QSST dependencies (XFoil dropped ).** Conda lockfile is committed and refreshed weekly.
- Colab Pro: notebook template (`notebooks/colab_phase4_runner.ipynb` stub) with SU2, Gmsh, checkpointing wrapper; parallel-sessions configuration documented.
- Colab GPU (G4-class): PyFR install verified; CUDA toolkit pinned.
- **Drive/JSONL ledger** (`src/fanopt/utils/ledger.py` + `drive_io.py` + `slicing.py`): one JSONL line per evaluation per session in `session_<id>/results.jsonl`; fidelity (-1/0/1 for **2D steady / 3D steady / 3D unsteady**, amendment), status, wall-time, all output metrics, artifact paths; full schema in Phase 4 step 51. Resume-safe via `.done` markers; cross-session dedup via `design_hash` (§6.2.5); pre-sliced round-robin assignment via `slice_assignments_v{N}.json` (single-writer barrier on the M3 — no atomic claim needed because sessions never write the same file).

**Outcome:** Project repository scaffolded with passing CI. All eight spikes (**0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6c, 0.7**) resolved with clear fallback paths. Working environments on Mac, Colab CPU, and Colab GPU. Locked J_fan metric. Closed Open Question #2. Baseline IMU-normalized physical measurement + rotational-inertia protocol + click-feature tolerance characterization + single-blade fabrication-noise floor + generative-geometry + BO-infrastructure validation + Tier-1 cfg sanity + benchmark validation in hand.

### Phase 1: 4-Layer Generative Blade Geometry Pipeline (Week 3)

Phase 1 produces single-material PETG blades via the **4-layer hybrid generative blade generator** (§9.7) — split across `src/fanopt/geometry/{envelope,fields,primitives,manufacturability,generator,schema}.py` per the §12 package layout. JSON schema covers ~37-46 design variables (the full Phase 2b/Phase 4 design space).

**Claude writes (generative pipeline, per §12 package layout):**

1. Fusion Python add-in (`fan_addin.py`): reads `params.json`, sets User Parameters on the master Fusion file, regenerates one V-unit blade body (2 ribs + 1 panel + click features), instantiates N=10 copies around the pivot axis, exports **one PETG STL per blade** plus a deployed-fan STEP. No multi-material; no per-material STL split.
2. CadQuery fallback generator (`generate_fan_cq.py`) — built always, both as a backup and as the fast path for Phase 2b/Phase 4 inner loops.
3. **V-unit blade parameterization** (`blade_geom.py`): generates a single blade body with:
 - 2 side ribs with tapered planform (12 mm base → 6 mm tip × 200 mm × 2 mm thick).
 - 1 panel between the ribs (6-8 mm wide × 200 mm long × **2.2-3.8 mm at 3 control points** — thickness profile is a Layer 1 design variable, not a TO output; locally widened to 12 mm-OD boss at the pivot region).
 - Click features on each outer rib (45° chamfer + 0.3-0.5 mm detent bump, with 0.15-0.20 mm design clearance).
 - Optional surface features on the panel (vortex generators, dimples, scallops) -- categorical from JSON.
 - Camber, twist, and edge profile of the panel from the per-blade airfoil JSON parameters.
4. **JSON parameter schema** (`src/fanopt/geometry/schema.py`) with all 4-layer hybrid design variables (full table in §6.2.1):
 - **Layer 1 — Outer envelope + Fourier modulation (~13 vars):** fan macro (blade count ∈ {8, 10, 12} categorical — spread angle derived from `blade_count × 13.3°` per C8 lock, NOT a free parameter; 14-blade trimmed per MED-10), blade length, base/tip rib width), camber spline (3-4), twist (2-3), thickness profile (3, **2.2-3.8 mm**), edge profile categorical, Fourier LE/TE harmonic amplitudes (3+3).
 - **Layer 2 — Macro-pattern + procedural math fields (~15-20 vars active, 0-3 of 5):** louver, texture, edge feature, noise threshold , TPMS . Each has an activation flag + per-field continuous params.
 - **Layer 3 — Capped 0-1 independent primitive (~5-7 vars):** primitive count {0, 1}; if active: type {slot/ellipsoid/wedge}, polarity, position xyz, size 1-3 dims, rotation 1-3 angles.
 - **Layer 4 — Manufacturing + click features (~3-5 vars):** print orientation, layer height, chamfer angle, detent size, design clearance.
 - **Total: ~37-46 variables **, all with bounds and types declared. Load-time validation rejects overlapping primitives, Fourier amplitudes outside ±15% envelope, TPMS cell size <3× min feature, etc.
5. Print-strategy decision script (`print_strategy.py`): given the user's bed dimensions (from Spike 0.4 bed-size check) and the JSON-defined fan size, choose **per-blade prints** (default; ≤256 mm bed) or **full-assembly print** (≥360 mm bed). Emits the appropriate STL layout.
6. Roundtrip smoke test (`smoke_test.py`): JSON → geometry → Gmsh mesh → SU2 steady stub → J_fan metric value. Confirms the pipeline is end-to-end before any optimization scaffolding is built.

**User runs:**

7. `python orchestrator.py --params baseline.json --emit-stl` → produces per-blade PETG STLs (×10) or one full-assembly STL.
8. Visual inspection in Fusion (or STEP viewer): confirm blade count, spread angle, pivot alignment, click feature geometry on the panel's outer tangential edges (per item #3 panel-edge relocation; NOT on the rib), and panel surface features (if any) look correct.
9. `python smoke_test.py --params baseline.json` → confirm a numeric J_fan value comes out the far end.
10. Slice the STLs in the user's slicer; confirm wall counts, infill (or TO geometry passthrough), and print orientation.

**Outcome:** Reproducible JSON → single-material PETG geometry → mesh → J_fan pipeline with both Fusion and CadQuery backends. Each design is fully described by one JSON file. Print-strategy decision (per-blade vs full-assembly) locked.

**phase ordering:** Phase 2 (rib TO) runs first; Phase 2b (generative parametric optimization on the panel) runs after Phase 2 because Phase 2b needs the optimized rib geometry to know the panel's available envelope. Phase 2a/2c/2d are not used (aero-sensitivity-weighted TO and compliant-panel sub-study were rejected — see §0 and §3.1 for the rationale).

### Phase 2a: Baseline 2D CFD Slice for Rib-TO Loads (Week 3.5, NEW — runs before Phase 2)

**Causality fix:** the earlier draft had Phase 2 (Week 4) read `phase3_baseline.csv`, which Phase 3 didn't produce until Week 6 — a circular dependency that would have crashed `aero_loads.py --cfd phase3_baseline.csv` at step 16. splits a half-day baseline CFD run out of Phase 3 and runs it *before* Phase 2 starts, so the file exists when Phase 2 needs it.

**Scope:** runs the Phase 3 2D-slice pipeline (`mesh_2d_slice.py` → SU2 `slice_steady.cfg` → `j_fan.py` + surface-pressure export) **once** on the **Spike 0.3 baseline flat-panel design** (no TO, no airfoil camber — the simplest blade geometry). Emits `phase3_baseline.csv` with the steady surface-pressure distribution. ~30-60 min of Colab CPU; runs the day after Spike 0.3 baseline-design CAD is committed.

**Outcome of Phase 2a:** `phase3_baseline.csv` exists on Drive before Phase 2's `aero_loads.py` runs; Phase 2 has no Phase-3-completion dependency. The remaining Phase 3 work (R² correlation sweep over 8-12 design points + dt/cycle independence checks) runs as Phase 3 currently does in Week 6 and *re-uses* the same `slice_steady.cfg` pipeline, so no duplicated tooling.

### Phase 1.9: SIMP Pre-baked-Strip Sanity Check (Week 3.7, NEW — gates Phase 2 launch)

**Purpose:** verify that pre-baking the guard rib's 3 mm outer-face reinforcement strip into the SIMP preserved zone (the `guard` rib class per §2.1 / §3.1.2) does NOT spike the §N7 manufacturability rejection rate. **Why this gates Phase 2:** Phase 2 IS the production SIMP run; if pre-baking the strip causes >20% of designs to fail §N7 (e.g., the strip's edge interacts with a Layer 2 louver field and trips the min-feature-size check), reactive validation after the fact wastes ~60-100 compute hours. A 10-design micro-pass before Phase 2 launches catches the failure mode for ~1 compute hour.

**Method (H2 lock — bumped from N=10 to N=30):** run **N = 30** SIMP solves per branch (with-strip-baked vs. without-strip-baked) on a Latin-hypercube sample of the parameter space at the Phase 2 settings. **Why N=30:** at N=10 with a 5% baseline rejection rate, the binomial std for the rate-difference is ≈ 10pp, giving the 20pp threshold only ~2σ power — marginal detection. At N=30 the std drops to ≈ 5.6pp and the 20pp threshold becomes ~3.5σ, statistically sufficient. For each solve: run the full §N7 manufacturability filter chain on the output STL and tally pass/fail counts. **Cost:** 30 solves × 5-30 min/each × 2 branches ≈ **5-30 compute hours** total (was 1-10 h at N=10). Within Phase 1.9's intended scope.

**Pass criterion (gates Phase 2 launch):**
1. With-strip-baked §N7 rejection rate ≤ without-strip-baked rate + **20 percentage points** (e.g., 5% → 25% is allowed; 5% → 30% blocks Phase 2).
2. **No element** in the deleted preserved zone (where the strip would have been, in the without-strip branch) exceeds **0.7 · σ_allow** at the worst-case load. (If a worst-case stress spike >70% of allowable appears in the deleted region, pre-baking the strip is mechanically necessary and the rejection-rate gate is binding.)

**Output:** `phase1_9/simp_strip_sanity_<date>.json` with per-design pass/fail tags + peak-stress map; commit to repo before `git tag phase2-launch` runs. Phase 2 launch script `scripts/launch_phase2.py` reads the JSON and refuses to launch unless both pass criteria are satisfied.

### Phase 2: Rib-Only Plate-Bending TO (Week 4; runs after Phase 1.9 + Phase 2a)

Narrows TO to ribs only. Panel topology is fully handled by Phase 2b's generative parameterization. See §3.1 for the formulation. **Pre-launch gate:** Phase 1.9 SIMP sanity check (above) must have passed.

**Closes Open Question #2 (aero-loads-from-CFD coupling):** TO loads come from Phase 2a's baseline CFD-derived pressures. If the assumed-pressure result and the CFD-pressure result differ by >10%, re-run once with CFD pressures.

**Claude writes:**

11. FEniCSx 2D plate-bending SIMP solver (`rib_to_solver.py`):
 - Design domain: one representative rib planform (tapered, 200 mm × 12 mm base / 6 mm tip × 2 mm thick).
 - 2D Reissner-Mindlin plate elements with bending DOFs.
 - Multi-load-case: (a) peak-positive aerodynamic pressure on panel transmitted to rib (push stroke, from Phase 2a baseline CFD); (b) peak-negative pressure (return stroke); (c) inertial load: **F_i = m_i · α_max · r_i_from_wrist** with **α_max = 110 rad/s²** and **r_i measured from the wrist axis** (= d_handle 0.05 m + position-along-blade from pivot); see §2.4 load case 2 for the axis convention ; (d) click-engagement reaction force; **(e) stress-test occasional-peak load (static, not fatigue-cycled; Architectural C ω scaling): 2.5× peak-positive aero pressure (≈ 50 Pa vs canonical 20 Pa) AND 2× α_max (220 rad/s² vs canonical 110 rad/s²) AND 1.41× ω_blade_max (12.4 rad/s vs canonical 8.8 rad/s) applied simultaneously** — the 1.41× ω scaling doubles centrifugal `F_c = m·ω²·r` matching the 2× α_max scaling (both stem from the same √2 ω_SHM ramp under force-driven swing). Pass criteria (rib-panel fillet, panel-pivot tension, and panel-pivot bearing — three independent checks from §10.1): canonical-cyclic ≤ 9.00 / 5.58 / 2.00 MPa nominal respectively (the bearing mode at 2.00 MPa binds first under canonical loading); stress-test static ≤ 20 / 12.4 / 13.3 MPa nominal. If a design passes the (a)-(d) cyclic checks but fails the (e) static check, the rib TO solver outputs `stress_test_fail=true` and the design is dropped from the Pareto by the §6.3.1 prefilter.
 - **update: rib SIMP TO no longer carries a pivot preserved-zone (the rib has no pivot hole under panel-pivot architecture).** Stress-constrained preserved zone in the rib is now the **rib-panel interface** (the rib's full-length junction with the panel at y = ±rib_center, p-norm aggregated; allowable σ from `material_locks.k_t_by_hotspot["rib_panel_fillet"]` = 9.00 MPa nominal cyclic for K_t_fillet = 1.5 with rib-flat XY-mode loading; the §3.1 TO solver tightens if the rib-panel geometry dictates higher K_t). The **panel** pivot hole (K_tt = 2.42, per-mode allowables 5.58 / 4.22 / 2.00 MPa tension/bending/bearing) is enforced by the §9.7.3 manufacturability filter as a hard `PANEL_PIVOT_REGION` keep-out (no Layer 2/3 carving in the 7 mm-radius circular region centered on the pivot pin) — NOT by the rib TO solver.
 - Preserved zones (panel-pivot architecture): rib-panel interface + rib-panel fillet + click-feature footprint (NO rib pivot hole — pivot is in the panel at y = 0).
 - Volume fraction target: 0.4 (range 0.3-0.5, tunable via JSON).
 - Filter radius: 1.5 mm (Helmholtz-PDE filter).
 - Material: orthotropic PETG (E_XY = 1300 MPa, E_Z = 1000 MPa, nu = 0.38, density 1.27 g/cm³).
12. **Rigid-blade gating check** integrated into `rib_to_solver.py`:
 - After TO convergence, compute the assembled blade's tip deflection under combined peak load (rib TO output + Phase 2b smooth-baseline panel placeholder).
 - **Pass criterion:** u_tip_max < 0.005 · L = 1 mm at 200 mm.
 - **Fail criterion:** u_tip_max ≥ 1 mm. has no Phase 2d escape; document the failure and either tighten the rib volfrac, stiffen the panel baseline thickness in Phase 1, or accept the limitation.
13. **Aero-loads-from-CFD coupling** (`aero_loads.py`):
 - Reads Phase 3 baseline 2D-slice CFD pressure distribution (steady push + steady return).
 - Maps the 2D pressure profile to the rib via tributary width.
 - If the resulting load magnitude differs from the initial 10 Pa estimate by >10%, re-run Phase 2 with the CFD-derived load.
14. Density field → manufacturable rib geometry (`density_to_rib.py`):
 - Threshold at rho = 0.5 (Gaussian-smoothed).
 - Marching squares contour extraction; Douglas-Peucker simplification (**epsilon = 0.05 mm**, fix; was 0.2 mm which can shift vertices by up to the click-feature clearance size and silently destroy snap geometry). **Click-feature polylines are tagged `preserve_vertices=True` and skipped by the DP pass entirely.** Required unit test asserts the click male/female bounding boxes are bit-identical before and after simplification.
 - Exports a DXF that the CadQuery blade generator (§9.7) imports as the rib cross-section sketch.
15. Fusion Simulation setup (`verify_rib.py`): orthotropic PETG, mesh with 0.3 mm element size at pivot, modal analysis for first 10 modes (first bending mode > 10 Hz = 5× waving frequency).

**User runs:**

16. `python aero_loads.py --cfd phase3_baseline.csv` → `loads.json`.
17. `python rib_to_solver.py --loads loads.json` → `density_rib.npy` (5-30 minutes at plate-bending resolution).
18. Check `u_tip_max` in solver log; document any failure.
19. `python density_to_rib.py` → `rib.dxf` (consumed by §9.7 generator in Phase 2b).
20. `python verify_rib.py` (Fusion Simulation) — confirm stress < allowable, modal > 10 Hz, tip deflection < 1 mm.
21. Print one isolated rib in PETG; static deflection vs FEA prediction.

**Outcome:** One TO-optimized rib design (applied to all blades by exact symmetry; each blade has 2 identical ribs). Rib geometry feeds into Phase 2b's blade generator. Open Question #2 closed.

#### 8.2.1 Post-Processing Rib TO Results (Phase 2 utility)

*(This subsection was previously labelled §8.3.1 and lived inside Phase 4 by mistake. It is a Phase 2 utility — the density-to-STL conversion for the rib SIMP output — so it now lives here, between Phase 2's Outcome and the start of Phase 2b. A `<a id="8.3.1"></a>` anchor is kept so older cross-references resolve.)*

<a id="8.3.1"></a>

The 2D SIMP optimization produces a density field (numpy array). Converting this to a printable STL requires:

1. **Threshold** the density field at rho = 0.5 to create a binary material/void map.
2. **Extract contours** using `matplotlib.contour` or `skimage.measure.find_contours` to get smooth boundary curves.
3. **Generate 3D rib geometry** via CadQuery: extrude the material regions to 2 mm thickness, add the pivot hole, and smooth sharp corners.
4. **Verify** minimum feature sizes meet printability requirements (minimum wall thickness >= 2 * nozzle_diameter = 0.8 mm for a 0.4 mm nozzle).
5. **Light smoothing** (optional): Claude writes a PyMeshLab script for Taubin smoothing on the final STL.

Claude writes the entire post-processing pipeline as a single Python script (~80-120 lines).

If using the BESO/CalculiX path instead, the full post-processing pipeline applies:
1. Density thresholding (rho = 0.5 cutoff)
2. Marching cubes surface extraction (ParaView or scikit-image)
3. Mesh smoothing (MeshLab/PyMeshLab)
4. Mesh repair (close holes, fix non-manifold edges)
5. Verification FEA on the smoothed geometry

Claude can write scripts for all of steps 1-4 using Python libraries (scikit-image, PyMeshLab, trimesh).

### Phase 2.5: Rib-Only Fillet FEA Re-check (Week 4.5, NEW — closes the rib-panel-junction stress-path gap; rib-only scope, not §59.5)

**Scope:** The §3.1.2 preserved-zone update moves the rib's primary preserved zone from the deleted rib-pivot hole to the **rib-panel fillet** (the rib's full-length junction with the panel at y = ±rib_center, carrying the bending-moment transfer from panel to rib). The Phase 2 SIMP solve runs once per rib class (inner + guard = 2 SIMP solves). If either SIMP run produced a rib geometry where the fillet zone is partially carved out by the SIMP density variable, the design passed Phase 2's compliance objective but loaded a stress path that wasn't constrained.

**Method (rib-only 3D static FEA, NOT the Phase 2 2D plate-bending solver — C14 lock):** the rib-panel fillet is a **3D Z-axis feature** — a 1 mm radius transition between the 2 mm rib (in z) and the 2.2-3.8 mm panel (in z) — that **cannot be resolved by the Phase 2 Reissner-Mindlin 2D plate-bending solver** (no z-direction nodes; the fillet stress lives in the z-dimension). The earlier "re-run the canonical-load profile Phase 2 used" was a category error: applying the Phase 2 2D solver to a 3D feature either crashes (no DOFs in z for the fillet element) or silently returns the closest in-plane stress (not the fillet stress the §3.1.5 K_t = 1.5 / σ_allow = 9.00 MPa check assumes).

**Fix:** for **each Phase 2 rib class** (inner + guard; **2 solves total**, not "every design in any campaign"), generate a **localized 3D mesh** of the rib-panel junction (rib extruded in z to its 2 mm thickness; smooth-baseline panel extruded to its nominal 3 mm thickness; **1 mm fillet radius CAD-modeled at the junction**). Run a 3D static FEA in **FEniCSx 3D** (or **CalculiX**) — same toolchain as §59.5 step 59.5, but on a localized slice (not the full combined blade). Apply the canonical-load profile Phase 2 used (peak-positive aero pressure + inertial body load at α_max). Extract peak σ_VM at the fillet surface; compare against the §3.1.5 K_t table allowable (9.00 MPa for rib-flat orientation, 1 mm fillet). Phase 2b's Layer 2/3 generative cutouts don't exist yet at Week 4.5; the rib-only check is appropriate scope — the full §59.5 combined-blade FEA is deferred to Phase 5 step 59.5 where the top-3 Phase 4 designs HAVE Layer 2/3 panel topology to evaluate.

**Cost (MED-8 merge):** ~3-5 min per rib class × 2 classes = **~6-10 min total** (one-shot, not per-design). Runs on the MacBook M3 alongside the Phase 2 SIMP solve. The 3D extrusion mesh is small enough that CalculiX terminates well within the M3's single-eval budget. **Re-runs (if invalidated by the 5.40 MPa threshold):** ~30 min per rib SIMP re-solve. Runs once at the Phase 2 → Phase 2b handoff. **Fallback (Sub-spike 0.6b validates the toolchain):** if M3's FEniCSx/CalculiX build has issues, Phase 2.5 dispatches to Colab CPU per the same fallback path as step 64.5. **Historical note:** an earlier draft cited "5 min × 100-300 designs = 8-25 hours" — that was a copy-paste of the §59.5 scope and conflated Phase 2 (which produces 2 rib classes, not hundreds of designs) with Phase 4 (where hundreds of designs exist).

**Invalidation rule:** any rib-class result that fails to constrain the fillet AND whose Phase 2.5 re-check shows σ_VM_fillet > 0.6 · 9.00 = **5.40 MPa nominal** is **invalidated** and the corresponding rib SIMP solve is re-run with the fillet zone pre-baked as a preserved zone per §3.1.2. Designs that pass the 5.40 MPa threshold are retained with a `phase2_5_revalidated=true` field in the SIMP cache metadata.

*(Duplicate Cost block deleted per MED-8 — single canonical Cost section above.)*

### Phase 2b: Generative Parametric Panel Optimization (Week 5-8; ~37-46 design variables in 4-layer hybrid)

The central panel-optimization step. Instead of density-based TO on the panel (rejected, see §3.1), each blade's panel is generated by a CadQuery script that combines an airfoil-shaped envelope with Boolean subtraction primitives and optional surface features. BoTorch optimizes the **~37-46 parameters (4-layer hybrid)** of that generator over multi-fidelity CFD.

**Why this works where density-based TO didn't:** the output geometry is always a clean STL (no grey-fluid artifacts), unsteady CFD runs on the actual binarized geometry (proper physics for asymmetric drag), the topology emerges from the BO landscape rather than being human-prescribed, and manufacturability is checked by the CadQuery generator before any CFD compute is spent.

**Three CFD fidelity tiers (update — QSST dropped):**

| Fidelity | Model | Wall-time | Use case |
|----------|-------|-----------|----------|
| **-1** | 2D steady CFD slice (corrugated cross-section, SU2 compressible + low-Mach prec; ) | ~5 min on Colab | Architecture-bandit screening at 30 evals/architecture; J_fan computed via §9.4 steady-state proxy (surface-force integration) |
| **0** | 3D steady CFD (full corrugated geometry, SU2 compressible + low-Mach prec) | 30-90 min on Colab | **Relative ranking only** among same-topology designs; NOT trustworthy as absolute J_fan |
| **1** | 3D unsteady CFD (full corrugated geometry, SU2 compressible + low-Mach prec, pitching, dt=T/200, 5 cycles) | 3-6 hours on Colab | True J_fan via canonical `j_fan.py` |

Cost ratios approximately **(2, 10, 50)** for (Tier -1, Tier 0, Tier 1). **Canonical location: `config.cost.COST_TUPLE = (2.0, 10.0, 50.0)`** in `src/fanopt/config/cost.py`. Every BoTorch `InverseCostWeightedUtility` construction in the campaign imports `COST_TUPLE` from this single source — there is no second declaration anywhere in the code, the SU2 templates, or the test fixtures. A regression test `tests/test_bo/test_cost_tuple_single_source.py` greps the codebase for stray `(2, 10, 50)` literals and fails if any are found outside `config/cost.py`.

**Cost-tuple drift monitoring (Colab wallclock variance is large; static lock would over-recommend Tier -1):** the M3 orchestrator reads moving-median `wallclock_s` from `merged/all_results.jsonl` per tier every BO step. If observed median diverges from configured cost by >25% over the last 20 evals per tier, the M3 updates `COST_TUPLE` from the static `(2.0, 10.0, 50.0)` to the observed ratio (normalized so the new Tier -1 cost stays ≥ 2.0 floor and Tier 1 stays = 50.0 anchor), writes the update to `phase4/diagnostics/cost_model_drift_<date>.json`, and reloads `cost_model`. Floor/ceiling sanity: Tier -1 ∈ [2.0, 10.0], Tier 0 ∈ [10.0, 50.0], Tier 1 fixed at 50.0. **Why:** Colab Pro wallclock routinely drifts 2-3× under throttling; without monitoring, qMFKG would still treat Tier -1 as cheap and starve the higher tiers while burning the 1000-h budget faster than planned. The §Phase 4 step 56 orchestrator runs the drift check as part of the per-BO-step diagnostic suite (alongside rank-correlation).

**Why QSST is removed:** QSST works for parameterized airfoil shapes (radial-strip decomposition). It cannot cleanly represent the topologically-emergent geometry from Boolean subtractions (slats, swiss-cheese, scalloped edges) — the strip decomposition breaks down when the cross-section has holes or interruptions.

**Design parameter inventory (4-layer hybrid, ~37-46 vars total, locked):** see §6.2.1 for the full table and §6.2.4 for the rationale. The summary:

- **Layer 1 — Outer envelope + Fourier modulation (~14 vars):** camber spline (3-4), twist (2-3), thickness profile (3 control points, **2.2-3.8 mm**), edge profile categorical {sharp, rounded, mildly-serrated}, Fourier LE harmonic amplitudes k=1,2,3 (3 vars, fixed phases), Fourier TE harmonic amplitudes k=1,2,3 (3 vars, fixed phases).
- **Layer 2 — Macro-pattern + procedural math fields (~15-20 vars active, 0-3 fields per design):** library of 5 field types — (a) louver, (b) texture, (c) edge feature, (d) **noise threshold **, (e) **TPMS **. Each field's activation is a categorical flag (architecture-bandit variable); the field's continuous params are optimized only when active.
- **Layer 3 — Capped 0-1 independent primitive (~5-7 vars):** number of primitives categorical {0, 1}; if active, shape {slot, ellipsoid, wedge}, polarity {add, subtract}, position xyz, size, rotation.
- **Layer 4 — Manufacturing + click features (~3-5 vars):** print orientation, layer height, chamfer angle, detent size, clearance.

Plus fan-macro continuous (~4 vars): blade count ∈ {8, 10, 12} categorical (spread angle derived per C8 lock; 14-blade trimmed per MED-10), blade length, base rib width, tip rib width.

**Total: ~37-46 design variables** (down from 45-55; reduction comes from capping primitives at 0-1 and folding the earlier serration/dimple top-level vars into Layer 2's `edge feature` and `texture` fields). At this scale the architecture bandit + multi-fidelity GP + TuRBO infrastructure (§6.2.2) handles the load with GP fits in ~30-60 seconds per iteration (validated in Spike 0.7b).

**Claude writes (Phase 2b deliverables; the BO infrastructure itself is shared with Phase 4):**

22. **Generative blade generator (`generate_blade.py`)** — described in full at §9.7. Five-step pipeline:
 - Step 1: `make_outer_envelope(...)` — camber + twist + thickness (2.2-3.8 mm) + edge profile + Fourier LE/TE.
 - Step 2: `apply_layer2_fields(envelope, active_fields)` — TPMS → noise threshold → louver → texture → edge serrations, all safe-by-construction.
 - Step 3: `try: apply_layer3_primitive(geom, primitive) except OpenCASCADE: skip`.
 - Step 4: `manufacturability_check(geom)` — 11 checks, score [0, 1] (see §N7 / §9.7.3).
 - Step 5: `export_stl` if mfg_score ≥ 0.5; else return None (BO penalty).
23. JSON parameter schema validator (`schema_validator.py`):
 - Validates at load time; rejects invalid combinations with clear error messages.
 - Reduces BO budget waste on infeasible designs.
 - **Implementation table (M10 lock — explicit per-rule):**

 | Rejection rule | Implementation | Cost |
 |----------------|----------------|------|
 | Fourier amplitude > ±15% envelope | Math check on `params.layer1.fourier_*` magnitudes | <1 ms |
 | TPMS cell size < 3× min feature | Math check on `params.layer2.tpms.cell_size_mm / min_feature_size` | <1 ms |
 | Noise threshold < 0.4 material retention | Math check on `params.layer2.noise.threshold` | <1 ms |
 | Layer 3 primitive position < 1 mm from envelope edge | Distance check on `primitive.position − envelope.bounds` | <1 ms |
 | Layer 3 primitive disconnects blade | CadQuery `Workplane.cut(primitive).solids()` count check (≥ 1 solid after subtraction) | ~10-100 ms |

 The disconnect check is the only CAD-level operation. Total validator runtime: ~20-200 ms per design, negligible vs CFD.
24. 2D slice mesh generator (`mesh_2d_corrugated.py`):
 - Cross-section through the deployed corrugated fan at 50% blade radius.
 - Resolves rib ridges, panel scoops, and Boolean-subtraction cutouts explicitly.
 - Structured boundary layer; far-field.
25. SU2 2D STEADY config (`configs/su2/slice_steady.cfg.j2`) — Tier -1 of the multi-fidelity stack. SU2 2D unsteady config (`configs/su2/slice_unsteady.cfg.j2`) is preserved as inner-loop verification fidelity, not the screening tier.
26. SU2 3D steady config (`fan3d_steady.cfg`) — Tier 0; ranking-only flag carried in the **Drive/JSONL ledger row**.
27. SU2 3D unsteady config (`fan3d_unsteady.cfg`) — Tier 1; true J_fan.
28. Phase 2b-only BO seed loop (`mf_panel_bo.py`):
 - Runs ~30 LHS at Tier -1 (2D steady, ) + ~30 LHS at Tier 0 + ~5 LHS at Tier 1 per representative architecture to seed the multi-fidelity GP before Phase 4 kicks in.
 - Validates that the architecture bandit + TuRBO infrastructure (§6.2.2) actually fits and proposes at **~37-46D ** within the Spike 0.6 compute budget.

**User runs:**

29. `python schema_validator.py --params test.json` → confirm CadQuery generator handles a variety of valid + invalid parameter sets.
30. `python generate_blade.py --params baseline.json` → produces one blade STL; visual inspection.
31. `python mf_panel_bo.py --seed` → seed LHS samples for the multi-fidelity GP.
32. Hand off seed dataset to Phase 4 for the full BO campaign.

**Outcome of Phase 2b:** Generative blade generator validated; multi-fidelity GP seeded with **~65 LHS samples per representative architecture** (30 Tier -1 2D-steady + 30 Tier 0 3D-steady + 5 Tier 1 3D-unsteady) across a small set of representative architectures. The full Phase 4 architecture-bandit screening then runs ~1,200-3,600 Tier -1 evals across all ~40-120 enumerated architectures. Phase 2b does NOT produce a "final design" — it produces the optimization infrastructure that Phase 4 then runs to convergence.

### Phase 3: 2D CFD Slice on Rigid Corrugated Geometry (Week 6)

With the architecture amendment: the 2D **steady** CFD slice is now **Tier -1 of the multi-fidelity stack** (used for cheap architecture screening). The 2D **unsteady** CFD slice is preserved as an **inner-loop verification fidelity** used at the BO inner loop on a small number of designs to anchor the steady-unsteady correlation. No FSI. Both 2D variants validate the rigid corrugated geometry (with Boolean cutouts) before the 3D BO commits compute; the 2D-unsteady runs also supply the CFD-derived pressure profile for Phase 2 rib TO loads.

**Claude writes:**

33. 2D cross-section geometry generator (`mesh_2d_slice.py`):
 - **Tier-1-restricted geometry (must hold for every 2D slice):** the slice is generated from a **Tier-(-1)-restricted variant of the §9.7 generator** that disables Layer 2 TPMS and Layer 2 noise-threshold fields. **Why:** TPMS and noise-threshold produce 3D-coherent porosity; a mathematical 2D slice through a Schwarz-D / gyroid lattice or a Perlin/Simplex noise field produces ~30-50 *disconnected* solid islands per blade. Gmsh's structured boundary-layer mesher requires a single closed wire per surface and throws on disconnected-wire topology; even with a tetrahedral fallback, SU2's wall-roughness + low-Mach prec assumes each solid surface bounds a coherent fluid region, so "air flowing between islands that are 3D-connected" is physically meaningless. The implementation is a one-line guard in the generator:

 ```python
 def generate_blade_for_2d_slice(params):
 """Tier-(-1) variant: skip 3D-coherent fields (TPMS, noise threshold) since
 their 2D slice would be topologically disconnected and crash Gmsh."""
 params_2d = copy.deepcopy(params)
 params_2d["layer2"]["tpms"]["active"] = False
 params_2d["layer2"]["noise_threshold"]["active"] = False
 return generate_blade(params_2d)
 ```

 Layer 2 {louver, texture, edge feature} are **2D-slice-coherent** (louver cuts pass through the panel thickness as clean parallel slits; texture maps to surface roughness; edge features modify the LE/TE outline) and remain active in the Tier-(-1) slice. Layer 1 envelope + Fourier modulation is always active. The 3D-coherent fields (TPMS, noise) are retained in the design's parameter set and are evaluated at Tier 0 (3D steady) and Tier 1 (3D unsteady) where the geometry is fully resolved.

 - Companion regression test (`tests/test_cfd/test_2d_slice_meshability.py`): generate a blade with TPMS active and confirm (i) `generate_blade_for_2d_slice(params)` returns a topology that is a single connected component when sliced at 50% radius; (ii) `generate_blade(params)` (full 3D) retains TPMS in the volume; (iii) Gmsh successfully meshes the 2D slice without `Gmsh_error`.
 - **Cross-section through the deployed fan at mid-radius (50% of L_blade = 100 mm from the pivot, 150 mm from the wrist axis), unwrapped into a linear cascade (NOT a Cartesian plane intersection of the 133.3° deployed sector, which gives an elliptical mess of variable spacing).** Extract N=10 blade cross-sections at r = 0.10 m from the pivot, array them along the **swing-tangent direction** with arc-length spacing `r_pivot · Δθ ≈ 0.10 m × 0.232 rad ≈ 23 mm` between adjacent blades (Δθ = blade angular pitch = 13.3° ≈ 0.232 rad per §2.5 / C8 lock; deployed extent = 10 × 13.3° = 133.3°). The result is a stationary 2D cascade: N=10 corrugated blade cross-sections in a uniform inlet flow. **Inlet flow direction: parallel to the swing-tangent direction** (i.e., the cascade axis), magnitude V_mid = 1.32 m/s — NOT normal to the cascade. For steady CFD the moving fan becomes a stationary cascade in this inlet flow.
 - **Radial-correction calibration (closes C6, single-radius bias):** the mid-radius slice uses a uniform `V_mid = 1.32 m/s` derived from `ω_blade_max · r_mid_wrist = 8.8 · 0.15`. Real radial velocity varies ~33% across the deployed sector (V(r) at root r_wrist = 0.05 m is `8.8 · 0.05 = 0.44 m/s` vs V(tip r_wrist = 0.25 m) = 2.20 m/s). A single-radius Tier -1 ranking can mis-rank designs whose asymmetric-drag mechanism varies radially (e.g., tip-clustered louvers vs root-clustered louvers). To preserve the architecture-bandit-calibrated coverage at 30 LHS samples × 1 mid-radius per architecture (≈ 1200-3600 evals total), add a **one-time Phase 0 radial-correction calibration**: 15-20 designs spanning low/medium/high directional-asymmetry parameters, each evaluated at all three radii `r_wrist ∈ {0.10, 0.15, 0.20} m`. Fit a linear correction `J_fan_corrected = J_fan_at_mid + β · directional_asymmetry_score(design)` where β comes from the calibration fit. Phase 4 Tier -1 ranks use `J_fan_corrected` instead of raw `J_fan_at_mid`. Compute: ~50 h one-time in Phase 0 (alongside Spike 0.7b). No Phase 4 budget impact. Calibration model stored at `gdrive/fan-optimization/phase0/radial_correction_v1.json` and consumed by `architecture_bandit.py` at promotion time. **Escalation if R² < 0.70:** if the linear fit explains less than 70% of the 3-radius variance, escalate to **N=18 LHS × 3 radii per architecture** as fallback (heavier; uses ~360-1080 h against the 1300 h pessimistic ceiling and triggers a recompute of slice_size and the architecture-count budget).

 - **`directional_asymmetry_score(design)` definition (starter form; functional form calibrated empirically):**

 ```
 directional_asymmetry_score(design) :=
     sum over Layer 2 louver fields:
         (louver_count) × |sin(louver_angle)| × (active flag)
     + |Fourier_LE_phase_offset − Fourier_TE_phase_offset| · 0.1
     + sum over Layer 3 primitives:
         (polarity_sign) × (primitive_size_relative_to_chord)
 ```

 The exact functional form is calibrated empirically — Phase 0 fits ~3-5 candidate forms and picks the one with the highest R² in the 3-radius spread. Candidates (also queued for V2 refinement, see §13): (a) the starter form above; (b) weighted sum of Layer 2 louver angles only; (c) Fourier TE/LE phase difference only; (d) integrated `|chord_z⁺(x) − chord_z⁻(x)|` over the planform (camber asymmetry). The **score is dimensionless**; `β` carries the dimensional scaling to J_fan units (N · score⁻¹). The score MUST be deterministic from `params.json` (no randomness, no CFD dependency) so it can be computed pre-CFD at slice dispatch time. The Phase 0 calibration artifact `radial_correction_v1.json` records both the chosen functional form (which of (a)-(d)) and the fitted β.
 - **Z-squash for the 2D mesh (deployed-surface modeling, not as-stacked geometry):** the 10 blade cross-sections are placed at a **common z = 0** in the 2D mesh. The pivot-stack z-offset (each blade sits at z_i = i · panel_thickness in the assembled fan) is **discarded** for the 2D cascade because the cascade models the deployed corrugated surface as a single z = 0 plane — the physical user experience is "the fan surface that pushes air", not "the stacked pivot column". Without the squash, the 2D mesh either fails to generate (planar 2D solver cannot represent z-offsets) or produces non-corrugated open-Venetian-blind physics with through-gap leakage that the real corrugation does not have. **Regression test (`tests/test_cfd/test_2d_slice_z_squash.py`):** load the 3D source assembly (pre-squash) AND the 2D `slice.su2` (post-squash); assert that the 3D source has `max(z) − min(z) > panel_thickness` (i.e., the source IS z-stacked, not already squashed) AND that the 2D output has `max(z) − min(z) < 1e-6 m` (the squash actually happened). Testing only the post-squash slice (as the earlier draft did) is trivially satisfied by any 2D mesh — the test had to assert the squash-from-3D-source transformation, not a property that holds by definition for any planar mesh. The test runs as part of Phase 0's CI before any Phase 3 evaluation dispatches.
 - **Unsteady-physics convention (`oscillating-inflow + stationary-blades`):** For a rigid click-locked deployed assembly, individual blades do NOT pitch relative to the assembly; the entire fan rotates about the wrist axis with sinusoidally-varying angular velocity. In the moving-cascade reference frame this becomes **stationary blades in a sinusoidally-varying inlet flow** `V_inlet(t) = V_mid · sin(2π·f·t)` with f = 2 Hz, V_mid = 1.32 m/s. (The SU2 unsteady config drives the inlet velocity via a `MARKER_INLET` time-varying BC, not via `GRID_MOVEMENT = RIGID_MOTION` on the cascade.)
 - Resolves the corrugated surface: N panels (cambered airfoils) separated by N+1 rib ridges.
 - Inter-blade gap geometry explicitly captured.
 - Gmsh script with structured boundary layer (first cell ~0.05 mm, growth 1.15, ~15 layers).
34. SU2 2D steady configuration (`slice_steady.cfg`):
 - Compressible Navier-Stokes + low-Mach preconditioning (consistent with the 3D campaign).
 - Laminar flow. **Re at the mid-radius 2D slice (step-33 mid-radius consistency)**: "mid-radius" per step 33 means **50% of L_blade = 100 mm from the pivot = 150 mm from the wrist axis**. Use the proportional derivation: V_mid / V_tip = (150 mm wrist-radius) / (250 mm wrist-to-tip) = 0.60, so V_mid = 0.60 · 2.20 m/s ≈ **1.32 m/s**. (Note: ω in this project is the SHM angular frequency 2π·f = 4π rad/s of the pitching motion, not the instantaneous blade angular velocity. The maximum instantaneous blade angular velocity is ω_blade_max = θ_max · ω_SHM = 0.7 · 4π ≈ 8.8 rad/s; V_tip_max = ω_blade_max · L_wrist-to-tip = 8.8 · 0.25 = 2.20 m/s, which is consistent.) With chord c = 0.2 m and real-air ν = 1.5e-5 m²/s, **Re_mid = V·c/ν ≈ 18,000**. SU2 `REYNOLDS_NUMBER` for the slice config is set to **18000**.
 - Outputs surface pressure distribution for use by Phase 2's (rib-only TO) aero-loads-from-CFD coupling.
35. SU2 2D unsteady configuration (`slice_unsteady.cfg`):
 - Compressible solver with low-Mach preconditioning.
 - **Sinusoidally-varying inlet velocity at f = 2 Hz, V_inlet(t) = V_mid · sin(2π·f·t) with V_mid = 1.32 m/s; blades stationary (no `GRID_MOVEMENT`)** — matches the oscillating-inflow + stationary-blades convention locked in step 33. (Note: this differs from the 3D unsteady config in §9.4 which keeps `GRID_MOVEMENT = RIGID_MOTION` about the wrist axis for the full-fan rotation; the two configs model different physics in different reference frames — the moving-cascade frame for the 2D slice, the lab frame for the 3D fan — both internally consistent.)
 - **5 pitching cycles, dual time-stepping 2nd order, dt = T/200 = 2.5 ms** (per locked J_fan spec in §9.4).
 - Discard cycle 1; integrate J_fan over cycles 2-5.
36. Canonical J_fan post-processor (`j_fan.py`):
 - Implements the locked §9.4 spec: `(1/T) ∫ ∫_Σ ρ_0 · u_n · (u · t̂) dA dt` over cycles 2-5 with `ρ_0 = 1.225 kg/m³`, plane Σ = 600 × 600 mm at 300 mm forward of pivot.
 - Also reports `J_fan_peak = (1/T) ∫ max(J(t), 0) dt`.
 - Used by every phase that produces a J_fan number (all CFD in Phase 3/4/5, Phase 6 physical measurement post-processing).
37. Multi-fidelity correlation analysis (`steady_unsteady_corr.py`):
 - Sweeps 8-12 design points; runs each through steady and unsteady CFD on the rigid corrugated geometry.
 - Fits a linear model: J_unsteady = a + b · J_steady + ε.
 - Reports R² and residual scale; these seed the Phase 4 multi-fidelity GP prior.
 - **Hard gate:** if R² < 0.4 between steady and unsteady, drop steady from Phase 4 and run unsteady-only.
38. Time-step independence check (`dt_independence.py`): re-runs the baseline at dt = T/100 and dt = T/400; confirms J_fan changes by <2% between dt = T/200 and dt = T/400. If not, refine.
39. Cycle-convergence check (`cycle_independence.py`): runs the baseline for **7 cycles** at k ≈ 0.57; verifies cycles 4-5 agree with cycles 6-7 within 5%. **Decision rule:** **cycle 1 is always discarded as transient** (per §9.4). After that, compute the **per-cycle directed-momentum-flux integral** `J_fan_cycle_k = (1/T) ∫_{(k-1)T}^{kT} [∫_Σ ρ_0 · u_n · (u · t̂) dA] dt` for k = 2..N — this is the §9.4.1 directed-momentum-flux integral evaluated on the kth cycle, NOT a per-cycle time-averaged thrust. The two quantities differ at the cycle boundary; the directed-momentum-flux integral is the canonical J_fan and must be used everywhere.

**Cycle-extension trigger (with denominator floor):**

```
J_scale_floor = 0.10 * J_fan_baseline_magnitude  # 10% of Spike 0.3 baseline; stored in locked-params
rel = abs(J_cycle2 - J_cycle3) / max(abs(J_cycle2), abs(J_cycle3), J_scale_floor)
if rel > 0.05:
    n_cycles = 8  # extend; discard 2, average 6
else:
    n_cycles = 5  # canonical; discard 1, average 4
```

The `J_scale_floor` denominator clamp matters for **symmetric designs near the baseline** (e.g., the parachute baseline, untilted Fourier-only): when `J_fan_cycle3 ≈ 0`, the unprotected ratio `|J2 − J3| / J3` diverges and the extension fires falsely. The floor caps `rel` at a finite value when both cycles report near-zero thrust.

The convective wake from cycle 1 sets the rationale for cycle 2 vs cycle 3 disagreement: convective timescale `L/V ≈ 0.25 / 1.32 ≈ 0.19 s` vs cycle T = 0.5 s → ~38% of one cycle, non-trivial at k ≈ 0.57.

**Standard-error parameterization on the chosen n_cycles:** with `n_avg = n_cycles − 1` (cycle 1 always discarded), `J_fan_se = std / √n_avg`. At canonical 5 cycles: SE = std/√4 = std/2. At extended 8 cycles: SE = std/√6 ≈ std/2.45 (NOT std/√4 — the extension averages 6 samples, not 4). The §9.4.1 config-hash assertion is parameterized on `n_cycles`; the JSONL `J_fan_se` field stores the value computed with the actual `n_avg` for that run. Per §0, this SE is **diagnostic only** and is not fed to the GP.

**User runs:**

40. `python mesh_2d_slice.py --params baseline.json` → `slice.su2` (Mac, <1 min).
41. `SU2_CFD slice_steady.cfg` (~5 min on Mac per case).
42. `SU2_CFD slice_unsteady.cfg` (~30-60 min on Mac per case).
43. Sweep 8-12 design points across panel camber, rib spacing, and blade count; collect (J_steady, J_unsteady) pairs.
44. `python steady_unsteady_corr.py` → `mf_prior.json` with the correlation. Check R² ≥ 0.4 hard gate.
45. `python dt_independence.py` and `python cycle_independence.py` → confirm time-step and cycle counts; finalize Phase 4 numerics.

**Outcome:** Quantified steady↔unsteady correlation for the rigid-corrugated geometry. Validated meshing, J_fan post-processor, and time-step/cycle-count choices. CFD-derived pressure profile available for Phase 2's rib-only TO aero-loads coupling.

### Phase 4: Multi-Fidelity BO on Colab Pro (Week 8-10)

The BO infrastructure scales to the 4-layer hybrid design space: **~37-46 dims total** (Layer 1 envelope + Fourier ~14 vars; Layer 2 macro-pattern + procedural math fields with 0-3 active out of 5-field library ~15-20 vars; Layer 3 capped 0-1 primitive ~5-7 vars; Layer 4 manufacturing ~3-5 vars; plus ~5 fan-macro vars). Continuous-only inner subspace per architecture is ~20-30 dims after the architecture bandit fixes the categoricals.

**Three-tier multi-fidelity:**

| Fidelity | Model | Cost (Colab) | Allocation strategy |
|----------|-------|--------------|--------------------|
| **-1** | 2D steady CFD (corrugated cross-section slice; ) | ~5 min on Colab | Architecture-bandit screening at 30 evals/architecture |
| **0** | 3D steady CFD (rigid corrugated geometry, SU2 compressible + low-Mach prec) | 30-90 min | Mid-fidelity ranking; uses combined Tier -1 + Tier 0 score for promotion |
| **1** | 3D unsteady CFD (rigid corrugated geometry, SU2 compressible + low-Mach prec, dt=T/200, 5 cycles) | 3-6 hours | High-fidelity ranking (true J_fan via canonical `j_fan.py`) |

QSST analytical fidelity is dropped — cannot represent generative Boolean-subtraction topology. 2D unsteady CFD is preserved as an **inner-loop verification fidelity** (Phase 3 + ad-hoc calls), not as the screening tier. No FSI tier in baseline (Phase 2d was dropped ).

**Claude writes:**

46. 3D Gmsh meshing script (`mesh_3d_fan.py`):
 - Reads `params.json`; meshes the deployed corrugated fan (10 blades, each with 2 ribs + 1 panel).
 - Resolves rib ridges explicitly OR applies the calibrated roughness model from §3.2.4 (calibration runs during Phase 3).
 - Structured boundary layer, ~500K cells for steady, ~1.5M for unsteady.
 - Downstream analysis plane fixed at 300 mm forward of pivot (matches locked J_fan spec).
47. SU2 3D configurations:
 - `fan3d_steady.cfg` (compressible + low-Mach prec, steady).
 - `fan3d_unsteady.cfg` (compressible + low-Mach prec, pitching, 5 cycles, dual time-stepping, dt = T/200 = 2.5 ms).
 - Both write checkpoint restarts every 100 iterations for Colab session-limit survival.
48. **Colab orchestrator + Drive/JSONL ledger** (`colab_runner.ipynb` + `src/fanopt/utils/ledger.py` + `src/fanopt/utils/drive_io.py` + `src/fanopt/utils/slicing.py`) **— storage architecture:**
 - **No SQLite, no atomic claim.** Each session (`SESSION_ID` hard-coded in notebook cell 1) reads `gdrive/fan-optimization/phase4/slices/slice_assignments_v{N}.json[SESSION_ID]`, iterates the assigned design hashes in order, and writes only to its own `session_<id>/results.jsonl` — never parallel-writes the same file. Cross-session dedup: each session skips any hash with an existing `designs/{hash}/.done` marker.
 - **Per-design loop on a session:** pull design JSON → check composite-key dedup (below) → run §6.3.1 prefilters (Filter 1 + Filter 2) → if rejected, append a `rejected_*` JSONL row + write `.rejected` marker, skip; → otherwise generate mesh → run SU2 (with §9.4.1 config-hash assertion; checkpoint every 100 inner iters) → compute J_fan + J_fan_se via `j_fan.py` → write `designs/{hash}/run_meta_{tier}_{session}_{ts}.json` sidecar → append result row to `session_<id>/results.jsonl` → on completion write the per-(config × physics × material × fidelity × direction) `.done` marker.
 - **Composite-key dedup (cache-poisoning fix; H16 7-tuple lock):** dedup key is **`(design_hash, physics_hash, config_hash, material_hash, geometry_hash, fidelity, run_direction)`** — was 6-tuple pre-H16, now 7-tuple with `geometry_hash` added. A bare `design_hash` covers only the canonical parameter dict; if the CFD config, motion spec, material constants, or **locked geometry constants** (HUB_RADIUS, RIB_TIP_TAPER, etc.) change, the same `design_hash` already has a stale `.done` marker from a prior run, and the orchestrator would skip re-evaluation and train the GP on stale data. The composite key catches every such change. **Marker path:** `designs/{design_hash}/runs/{config_hash[:8]}-{physics_hash[:8]}-{material_hash[:8]}-{geometry_hash[:8]}-{fidelity}-{direction}.done` — multiple `.done` markers per design, one per (config × physics × material × geometry × fidelity × direction) combination. **Migration on first composite-key launch:** the orchestrator scans existing `.done` markers (legacy 6-tuple or pre-composite format), recomputes their composite key against the CURRENT (config, materials, geometry, motion) snapshot; markers whose composite matches are renamed to the new path; markers whose composite doesn't match are quarantined to `designs/{design_hash}/runs/stale/` for human inspection and possible re-run. Migration is one-shot and logged to `phase4/diagnostics/composite_key_migration_<date>.json`.
 - **Buffered append (Drive doesn't support true append).** `drive_io.append_jsonl` buffers **N=5 records in memory** and flushes every 5 completions or every **30 s**, whichever first. The 30 s timer caps crash-resume granularity. **Crash-recovery sentinel:** on dispatch, the session writes a one-line `designs/{hash}/.in_progress` sentinel; on successful flush, the sentinel is deleted. Session-restart logic scans for orphaned `.in_progress` markers on Drive and re-queues those hashes — designs that crashed mid-eval don't silently vanish from the campaign. **Dead-session detection latency (documented):** the 30 s flush window + 60 s heartbeat cadence gives a worst-case staleness of **30 + 60 = 90 s** between a session's last successful flush and its heartbeat reflecting the crash. The 15-min (900 s) watchdog catches it eventually, so the 90 s gap is bounded loss and does NOT require tightening — it's documented here so post-mortem analysis doesn't blame "missing" 30 s of work on a bug.
 - **Cross-session claim (locked: local lock files, not UUID re-read).** If a slicing bug puts the same hash in two `slice_assignments` rows, the per-session sentinels (namespaced under each session's directory) won't conflict; both sessions would run, write rows, and the M3 would silently dedupe one later — wasting CFD. **Spec:** each session attempts `open("designs/{hash}/.claim", "x")` (O_CREAT|O_EXCL via Python `"x"` mode); on failure another session has the claim → skip the hash. **Cleanup path:** the session that holds the claim must call `claim_release(hash)` on successful flush (deletes `.claim`); on session crash, the orphaned `.claim` is reaped together with the orphaned `.in_progress` by the same session-restart sweep — both are removed if the session is no longer in `m3_active_sessions.txt` AND the orphaned-marker age exceeds `CLAIM_REAP_AGE_S = 900` (15 min, matches the watchdog stop threshold). The `O_CREAT|O_EXCL` race window on Drive (eventual consistency can let two sessions both succeed if they race within ~5 s in the optimistic case, up to several minutes in the worst case) is closed by a **post-claim sleep + re-read confirmation**: after `open("x")` succeeds, sleep **30 s** (M6 lock — bumped from 5 s; Drive's published 30 s − few minute consistency window exceeds the prior 5 s budget), then `re-read .claim` and assert the file's content matches `<session>_<utc>`; if it doesn't (another session also "won" the race and overwrote), the higher session id yields and deletes its claim. **Latency monitoring:** every BO step, M3 logs write-to-visibility latency for `slice_assignments_v{N}.json` (M3 write timestamp vs first-read timestamp from a Colab session) to `phase4/diagnostics/drive_latency_<date>.json`. If observed P99 exceeds 25 s over 5 consecutive steps, the orchestrator bumps the post-claim sleep to 60 s and logs the change. **Recovery semantics (if both fail to yield):** if two sessions race-claim despite the 30 s window (eventual-consistency window exceeded), the merger pass dedupes by `(design_hash, physics_hash)` composite — the redundant Tier -1 eval is discarded (cheap), the redundant Tier 0/1 eval is kept (extra MF GP training data; low harm).
 - **Slice-assignment versioning + atomic two-version pointer.** The M3 publishes a new assignment every BO iteration; Drive's eventual-consistency window (30 s - few minutes) means a session can read an OLD or partially-synced assignment after a new one is published. **Spec:** the M3 writes immutable **`slice_assignments_v{N}.json`** files (never overwritten) plus a small **`current_slice_pointer.txt`** that sessions poll every 5 min. **Two-version pointer format (locked for V1):** `"{N} <sha256-of-v{N}-file> {N-1} <sha256-of-v{N-1}-file>"` (single line). The two-version format supports SHA-rollback: if v{N}'s SHA fails to validate after the propagation window, sessions fall back to v{N-1} (already validated on the previous publish). **First-publish bootstrap (N = 0):** the pointer reads `"0 <sha-v0> -1 0000…0000"` (zero-sha sentinel); sessions special-case N < 1 to skip the rollback validation. **No migration path needed** — no pointer files have been written yet, so the two-version format is the initial spec. **Publish order:** (a) M3 writes `slice_assignments_v{N+1}.json.tmp`, fsync, atomic-renames to `slice_assignments_v{N+1}.json`; (b) M3 waits `DRIVE_PROPAGATION_GUARD_SEC ≈ 60s`; (c) M3 writes the new pointer (`"{N+1} <sha-vN+1> {N} <sha-vN>"`) via .tmp + atomic-rename. Sessions parse the pointer, fetch the named version file, validate the SHA before consuming; on SHA mismatch retry with exponential backoff up to 5 min, then fall back to v{N-1}. **M3 single-writer / heartbeat:** M3 writes `m3_heartbeat.txt` every 60 s; **auto-resume / watchdog rule (single threshold, no gap):** sessions monitor `heartbeat_age_s` and apply: `age ≤ 5·60 = 300 s` → resume dispatch; `5·60 < age ≤ 15·60 = 900 s` → held-paused (no new dispatch, finish in-flight work, do not release credits); `age > 900 s` → stopped, credits released, M3 considered failed (manual failover SOP at `docs/m3_failover_sop.md`). `caffeinate -dimsu` in the M3 launch script prevents Mac sleep. **Fallback Sobol pool (retrain starvation):** `phase4/fallback_slice.json` holds a pre-baked pool of safe Tier -1 Sobol candidates; if `pointer_age > slice_life_estimate` (i.e., M3 retrain is taking longer than the current slice drains), sessions consume from the fallback pool and log `slice_starvation`.
 - **slice-size formula (locked hardware).** `slice_size = retrain_interval_min · sessions · parallelism / mean_tier_min`. With Tier -1 at 10 min/eval, the **locked 2-4 parallel Colab Pro sessions × 5 parallel SU2 slots per session, and a 30 min retrain cadence**: `30 · 2 · 5 / 10 = 30` (lower bound) → `30 · 4 · 5 / 10 = 60` (upper bound). **Locked slice_size = 30-60 designs per slice** (was 90 in an earlier draft that used 6 sessions; the 6-session number over-allocates by 50% under the locked 2-4-session hardware). Recompute when the tier mix shifts (e.g., Tier 0 promotion to Tier 1, K-drop from 4 to 2). Stored as `slice_size` in the per-iteration `slice_assignments_v{N}.json` so the value is hashed and traceable.
 - **Marker files:** `session_<id>/.heartbeat` touched every 60 s; `session_<id>/.done` written when the slice is flushed. Orchestrator waits for all `.done` markers before launching the merger pass; missing heartbeat > 15 min signals a dead session for manual re-slicing.
 - **Single-writer BO acquisition barrier (M3 only).** The Mac orchestrator (Phase 4 step 54 below) is the only process that retrains the GP, runs `qMultiFidelityKnowledgeGradient` / `qNoisyExpectedHypervolumeImprovement`, slices the new candidates round-robin into `slice_assignments_v{N}.json`, and bumps `next_batch.txt`. Sessions poll `next_batch.txt` every 5 minutes. Keeping the GP single-writer eliminates the only place coordination would be required.
 - **hardware routing (explicit; HIGH-11 Round-9 lock):** **Colab Pro CPU sessions for all SU2 tiers** (Tier -1 2D steady, Tier 0 3D steady, Tier 1 3D unsteady — none of these use the GPU); **Colab Pro G4 GPU (95 GB VRAM) reserved for Phase 5 PyFR p=3 top-3 verification only** (T4 16 GB OOMs at p=3 on 2-3M cells; G4 95 GB has 5-6× headroom). The notebook template selects the runtime per session — CPU (standard or high-RAM for Tier 0/1 if mesh > 1 M cells) for screening sessions and G4 GPU only when running the Phase 5 PyFR session.
 - **Drive caveats (operational rules):** (i) eventual consistency — never use Drive as a sync primitive (a file written by session A may take 30s-several-minutes to appear to session B; sequencing comes from `slice_assignments_v{N}.json` written once at campaign start, not from inter-session "wait until file X exists" polling); (ii) small-file I/O is slow (~50× slower than reading one big JSONL) — `designs/{hash}/` subdirectories are used only for top-100 candidates that need richer artifacts; bulk data is JSONL; (iii) API rate-limit ~10k req/hr — don't call `drive.files.list` per-candidate; the orchestrator caches directory listings for the duration of each BO step.
49. **Two-level optimization loop** (`src/fanopt/bo/orchestration.py`) — see §6.2.2 for full rationale:
 - **Outer loop: architecture bandit.** Enumerate categorical/discrete variable combinations: blade count {8, 10, 12} (MED-10 trim: 14 removed for ergonomic infeasibility — 186.2° past straight-line) × Layer 2 field activation flags (≤3 active from 5-field library) × Layer 2 per-field categoricals (louver polarity, texture type, edge feature type, TPMS lattice type) × Layer 3 primitive presence + type × edge profile category × print orientation × layer height. Total **~40-120 architectures** depending on enumeration depth. **Cheap 2D-steady-CFD screening (Tier -1, ) on every architecture at 30 evals each.**
 - **combined promotion:** the bandit promotes K=4 fixed architectures by **combined-rank score** (Tier -1 mean rank + Tier 0 mean rank, equally weighted), not Tier -1 alone. Each candidate gets ~5 Tier-0 evals before the decision to mitigate the Tier -1 2D-CFD porous-penalty bias.
 - **Inner loop per promoted architecture: continuous TuRBO.** Inside each fixed architecture, run TuRBO over **~20-30 continuous dimensions**. 3-5 parallel trust regions. `SingleTaskMultiFidelityGP` with three fidelity tiers {2D steady (Tier -1), 3D steady (Tier 0), 3D unsteady (Tier 1)}.
 - `qMultiFidelityKnowledgeGradient` with cost model **(2D steady=2, 3D steady=10, 3D unsteady=50)** at the inner loop, sourced from `config.cost.COST_TUPLE`.
 - Multi-objective: `qNoisyExpectedHypervolumeImprovement` over **4 objectives **: (J_fan, **I_wrist** [about handle-grip wrist axis; ). m_total < 0.100 kg and r_CoM_wrist ≤ 0.160 m (= d_handle 0.05 m + 0.55·L_blade 0.20 m) are hard constraints (§6.4), not objectives.
 - **Hard mass constraint:** total assembly mass < 100 g (C9 lock per §0; not an objective).
 - LHS init per promoted architecture: ~30 Tier 0 + ~5 Tier 1.
 - Inner-loop budget per architecture: **~35 acquisition rounds (hard cap; early-stop fires first when UCB-improvement < 3% over 5 iters; earlier draft said 50)**.
 - Total compute target: ~40-120 architectures × 30 Tier -1 evals (~3-6 hours total) + **K_promoted × ~35-40 Tier 0/1 evals (data-driven K, default K=4)**.
50. **SAASBO alternative inner-loop model** (`src/fanopt/bo/saasbo.py`):
 - Drop-in replacement for the inner TuRBO GP if TuRBO under-explores.
 - `SaasFullyBayesianSingleTaskGP`; downsampling cap of **500 inducing points** to keep NUTS feasible.
 - Tier -1 2D-steady-CFD data feeds the GP prior mean (not raw training points).
51. **Drive/JSONL ledger schema (`src/fanopt/utils/ledger.py`):** one Pydantic-validated JSON object per line in `session_<id>/results.jsonl`. UTF-8, sorted keys, deterministic float precision (6 digits). **Every row carries `schema_version: int` (current = 2, -Vsn);** `schema_migrations.py` upgrades v{i} → v{i+1} in-memory on load so a merged JSONL containing pre-amendment rows still validates against the current Pydantic model. CI test fixtures live at `tests/fixtures/ledger/v{i}/` for every prior version. Bump the version on every field add/rename. Schema (current v2):
 ```
 {
 "design_hash": "<24-hex blake2b/12, §6.2.5>",
 "session_id": "<SESSION_ID from notebook cell 1>",
 "timestamp_utc": "<ISO-8601>",
 "tier": -1 | 0 | 1,
 "status": "ok" | "rejected_*" | "failed_*" | "running",
 "failure_code": "<§9.4.2 enum or null>",
 "detail": "<free-text, ≤500 chars>",
 "params": {...canonical parameter dict...},
 "m_total_kg": <float>,
 "r_CoM_wrist_m": <float, position-of-CoM from wrist axis>,
 "I_wrist_kgm2": <float, about handle-grip wrist axis>,
 "mfg_score": <float ∈ [0,1]>,
 "pre_cfd_stress_estimate_mpa": <float, §6.3.1 Filter 2>,
 "pre_cfd_tip_defl_mm": <float, §6.3.1 Filter 2>,
 "pre_cfd_struct_ok": <bool>,
 "J_fan_se": <Optional[float], std/√(n_cycles−1) = std/2 at canonical 5 cycles; NULL for Tier -1 / Tier 0; DIAGNOSTIC ONLY — stored as `J_fan_cycle_variance`-equivalent SE; NOT fed to the GP per §0 fixed-floor lock>,
 "J_fan_peak": <Optional[float], N — secondary metric; NULL for Tier -1 / Tier 0>,
 "peak_pivot_stress_mpa": <float, nominal at K_t = 2.42>,
 "folded_form_factor_m": <float>,
 "wallclock_s": <float>,
 "host": "<platform.node + platform.machine>",
 "su2_commit": "<git rev>",
 "fanopt_commit": "<git rev>",
 "config_hash": "<§9.4.1 hash of resolved SU2 cfg>",
 "physics_hash": "<blake2b(config_hash || materials_blob || motion_blob); strict superset of CROSS_TIER ∪ TIER_SPECIFIC ∪ MATERIAL_LOCKS ∪ MOTION_SPEC per §9.4.1>",
 "material_hash": "<blake2b(MATERIAL_LOCKS); covers (RHO_PETG, RHO_PIN, E_PETG_XY, E_PETG_Z, σ_y_XY, σ_y_Z, K_t_pivot_tension, K_t_pivot_bending, K_t_bearing, fatigue_factors); §10.1 + §3.1.5>",
 "geometry_hash": "<blake2b(GEOMETRY_LOCKS); H16 + Round-8 HIGH-4 locks — covers 18 entries: (HUB_RADIUS_M, RIB_TIP_TAPER_M, L_BLADE_M, D_HANDLE_M, RIB_THICKNESS_M, RIB_BASE_WIDTH_M, RIB_TIP_WIDTH_M, INTER_BLADE_ANGLE_RAD, PIVOT_BOSS_RADIUS_M, PIVOT_CENTER_X_M, CLICK_FOOTPRINT_X_RANGE_M, PANEL_PIVOT_REGION_RADIUS_M, CHAMFER_CLEARANCE_M, PANEL_THICKNESS_MIN_M, PANEL_THICKNESS_MAX_M, FILLET_RADIUS_M, CLICK_FOOTPRINT_Y_RANGE_PANEL_EDGE_M_AT_TIP, CLICK_DETENT_BUMP_RADIUS_M); §0 + §3.1.2 + §9.7>",
 "schema_version": 2,
 "rib_class": "<: inner | guard>",
 "pre_cfd_stress_test_passed": <bool, true if all three Filter-2 modes pass under stress-test load>,
 "pre_cfd_stress_test_spec_hash": "<blake2b/12 hash of (p_aero_factor, alpha_factor, σ_y_XY, σ_y_Z, K_t hotspot table) at filter time — catches a stress-test reference that drifts vs the materials hash>",
 "pre_cfd_stress_test_max_stress_mpa": <float, max σ_eff across the three Filter-2 modes under stress-test load>,
 "pre_cfd_stress_test_allowable_mpa": <float, lower of (canonical-cyclic, static-SF) allowable; pass condition is max_stress < allowable>,
 "pre_cfd_stress_test_load": {"p_aero_factor": 2.5, "alpha_factor": 2.0},
 "J_fan_productive": "<Optional[float], N — FREESTREAM_PRODUCTIVE=(0,0,-1) drag; NULL for Tier 1 unsteady>",
 "J_fan_return":     "<Optional[float], N — FREESTREAM_RETURN=(0,0,+1) drag; NULL for Tier 1 unsteady>",
 "J_fan_delta":      "<Optional[float], N — (J_fan_productive − J_fan_return) for Tier -1/0; NULL for Tier 1 (matches J_fan directly)>",
 "J_fan":            "<float, N — required all tiers; for Tier 1 (3D unsteady) the time-integrated metric per §9.4 over cycles 2-N; for Tier -1 / Tier 0 (steady) set to J_fan_delta for downstream BO consumption>",
 "retry_count": <int, ; max 2 before non-retriable>,
 "retry_history": ["<list[str] of prior retry attempts and their failure_codes, -retry pattern mining>"],
 "cfl_max": <float, max Courant number across the run; assert < 10 in post-processing>,
 "stress_test_fail_mode": "<split: fea_combined_tip_defl | fea_combined_peak_stress | fea_combined_torsion | null>"
 }
 ```
 Reads concat all `session_*/results.jsonl` (a few-MB total even at 5k evals) → Pandas DataFrame → dedupe by `design_hash` (keep most-recent timestamp). Resume-safe: re-running a session re-reads its slice and skips any `design_hash` with a `.done` marker. Aggregated `merged/all_results.jsonl` is rewritten end-to-end by the merger pass after each BO step.
52. Best-of-Pareto regenerator (`scripts/run_phase5_verify.py`):
 - Identifies the **top 3 designs** on the 4D Pareto front (J_fan, **I_wrist** [about the handle-grip wrist axis], stress, folded_form_factor) using the light/knee/heavy coverage rule (N9).
 - Regenerates each in Fusion at full quality for Phase 5 verification.

**User runs:**

53. Upload Phase 3's `mf_prior.json` and Phase 2b's seed dataset to Colab Pro.
54. `python scripts/run_phase4_bo.py --init` (Mac): generate ~40-120 × 30 = ~1200-3600 Tier -1 (2D steady) screening evals + per-promoted-architecture LHS seeds; **write `gdrive/fan-optimization/phase4/slices/slice_assignments_v{N}.json`** with round-robin interleave (`session_k` gets hashes `[k, k+K, k+2K, ...]`, NOT contiguous blocks — keeps "easy" and "hard" parameter regions evenly distributed across hardware).
55. Open `notebooks/colab_phase4_runner.ipynb` in 2-4 parallel Colab Pro sessions; each notebook hard-codes its `SESSION_ID` in cell 1 (new session = copy template, change SESSION_ID, rerun slice-assignment). Each session reads its assigned slice from `slice_assignments_v{N}.json`, skips any hash with a `.done` marker, and appends results to its own `session_<id>/results.jsonl`. Reconnect and restart when sessions die — the `.done` markers handle resume.
56. `python scripts/run_phase4_bo.py --iterate` (Mac, single-writer barrier): for every BO step, (a) read all `session_*/results.jsonl` into a Pandas DataFrame; (b) dedupe by `design_hash`; (c) retrain `SingleTaskMultiFidelityGP` with **fixed-floor `train_Yvar = EPISTEMIC_NOISE_FLOOR` scalar per tier** (NOT per-observation; §0 + §6.2.3 lock — per-design `J_fan_se` is stored as `J_fan_cycle_variance` diagnostic but never reaches the GP); (d) run `qMultiFidelityKnowledgeGradient` for Tier -1/0 and `qNoisyExpectedHypervolumeImprovement` for the 4D Pareto promotion; (e) **run §6.3.1 prefilters on proposed candidates** before slicing; (f) round-robin-slice survivors into the next `slice_assignments_v{N}.json`; (g) bump `next_batch.txt` (sessions poll every 5 min). **rank-correlation diagnostic + action rule:** every N=20 Tier-0 completions, compute Spearman ρ on the (Tier -1, Tier 0) overlap and (Tier 0, Tier 1) overlap from the merged JSONL; write `phase4/diagnostics/rank_corr_<date>.json`. **If either ρ < 0.4:** the orchestrator **freezes multi-fidelity** for the affected tier pair and runs **pure Tier 1 (no Tier -1 / Tier 0 promotion) on the next 10 acquisition rounds within the current architecture** (disambiguation — "10 architecture decisions" was interpreted as "acquisition rounds within the current architecture" (~30-60 h), not "architecture-level promotions" (900-1800 h)). After those 10 complete, recompute ρ on the updated overlap. **Hysteresis:** recover only when ρ has been **above 0.5 for two consecutive 20-eval windows** (prevents 0.4-0.5 band oscillation that would thrash the freeze/resume cycle). If ρ is still < 0.4 after the 10 forced Tier-1 rounds, **pause the campaign and surface to the human** (write `phase4/diagnostics/PAUSE_ρ_below_floor.txt` + send PushNotification if configured). **per-architecture-class threshold:** for 2D-slice-coherent architectures (Layer 2 ⊆ {louver, texture, edge}) the threshold is **ρ < 0.4**; for 2D-slice-incoherent architectures (Layer 2 ∋ TPMS or noise) the threshold drops to **ρ < 0.2** because the multi-fidelity GP kernel is *known* to be mis-specified for those architectures and the ρ < 0.4 floor would fire routinely with no diagnostic value. **** the R² used in the K-decision is **Spearman ρ² on the (Tier 0, Tier 1) overlap** (Spearman because nonlinearity makes Pearson misleading); the (Tier -1, Tier 0) ρ from this same diagnostic feeds the architecture-bandit weighting , not the K-decision. Pre-register the threshold table in `configs/k_decision.yaml`.
56.5. **In-flight Tier-1 trust monitoring (2026-05-14 addition; Spike-0.6d follow-on):** in compensation for Phase 0's reduced absolute-accuracy evidence (0.6c.2 → Phase 5 per `docs/phase_logs/spike_0_6c.md` Note 1), Phase 4 carries two monitoring rules sized to fit inside the 1000-h stop-rule headroom. **(a) Baseline-regression every N=100 BO acquisitions, at Tier 0 (NOT Tier 1):** re-run the flat-panel baseline geometry through the **Tier-0** production cfg (steady, 30-90 min/eval); log `J_fan_steady_proxy` and report drift. The Tier-0 choice is deliberate — drift in the steady proxy is enough to detect orchestrator / mesh / cfg non-determinism; Tier-1 drift would be the gold standard but would 6-12× the cost. **Pass criterion:** drift `< ±10%` across checkpoints. **Fail action:** flag in `phase4/diagnostics/baseline_regression_<date>.json`; pause campaign if drift > ±25% (indicates a substantive bug). **Cost:** ~10-15 baseline reruns × Tier-0 cost (30-90 min) = **~5-15 h cumulative** across the full Phase 4 campaign. Counted against the 1000-h stop rule. **(b) MACH-perturbation rank-stability at midpoint:** at the Phase-4 midpoint (~500 h cumulative or end of architecture round 3, whichever first), re-run **10 representative geometries** spanning the current Pareto front at `MACH_NUMBER = 1e-7` (still low-Mach regime; different low-Mach preconditioner conditioning). **Pass criterion:** Pareto ranking is stable (Spearman ρ ≥ 0.8 between MACH=1e-9 and MACH=1e-7 J_fan rankings on those 10 geometries); absolute J_fan values may differ by up to ±15% without alarm. **Fail action:** if ρ < 0.8, pause the campaign and surface to the human (`phase4/diagnostics/PAUSE_mach_rank_unstable.txt`); the rank-instability indicates Phase-4 ranking is being driven by the low-Mach preconditioner choice itself, not the underlying geometry. **Cost:** 10 extra Tier-1 evals × 3-6 h = **~30-60 h one-shot**. Counted against the 1000-h stop rule. **Combined cost of (a) + (b):** ~35-75 h total — within the 1000-h stop-rule budget (expected campaign 600-1100 h; favorable case has 400+ h headroom).
57. Total wall-clock: **2-4 weeks** of intermittent Colab use (**rebaseline: ~600-1100 h expected / 1300 h pessimistic**; favorable case ~300-600 h applies only with K=3 + §6.3.1 prefilter cull rate at the high end of expected; see §6.2.3 "Honest compute budget"). **stop rule:** if cumulative > 1000 h before convergence, K stays at its current value within {3, 4, 5} and the orchestrator force-grows the (Tier 0, Tier 1) overlap (no K-drop-to-2 path; K = 2 is outside the locked range).
58. `python scripts/run_phase4_bo.py --pareto-top3` → top-3 Fusion files for Phase 5.
59. **Phase 4 → Phase 5 handoff** (`scripts/run_phase4_bo.py --freeze`): commit the merged Drive JSONL (`phase4/merged/all_results.jsonl`, ), top-3 design JSONs, Pareto front plot, and the multi-fidelity GP state to the project repo under `results/phase4_<date>/`; tag the Git commit `phase4-frozen-<date>`; queue Phase 5 verification jobs against the frozen top-3 set. Asserts that all three designs satisfy the §6.4 hard constraints (**m_total < 0.100 kg** = 100 g, **r_CoM_wrist ≤ 0.160 m**, **panel-pivot per-mode cyclic allowables: tension ≤ 5.58 MPa, bending ≤ 4.22 MPa, bearing (Z) ≤ 2.00 MPa nominal** [panel-pivot architecture, K_tt = 2.42 in 12 mm boss, σ_y_Z = 30 MPa per §10.1], **manufacturability ≥ 0.5**) and refuses to hand off otherwise — a violation triggers a Pareto-front re-prune and re-rank rather than letting an infeasible design enter Phase 5.

**Outcome:** Pareto-front set of optimized rigid-blade designs across **~37-46 dimensions** (4-layer hybrid), multi-fidelity-calibrated across **2D steady / 3D steady / 3D unsteady**. Top-3 candidates (light corner / knee / heavy corner of the 4D Pareto front per N9) regenerated in Fusion at full quality and ready for Phase 5 verification. Phase 4 → Phase 5 handoff is committed and tagged in Git (step 59).

### Phase 5: High-Fidelity Verification + PyFR Cross-Solver on Top-3 (Week 11-12)

PyFR runs on the **top 3 Pareto designs**, not a single final design, so that numerical artefacts in any one solver are caught.

**Claude writes:**

**59.5. Combined-blade structural verification gate :** For each of the top-3 Pareto designs, run a fast static FEA on the **generated** STL (rib TO output + Phase 2b panel topology, glued at the rib-panel interface) before any SU2/PyFR CFD work begins.

 - Solver: FEniCSx (or CalculiX) with orthotropic PETG (E_XY = 1300 MPa, E_Z = 1000 MPa, ν = 0.38, ρ = 1.27 g/cm³).
 - **Loads (uniform pressure at the analytic stagnation peak, H9 lock + MED-4 Round-9 disclosure lock):** uniform static pressure `p_uniform = p_stagnation_peak` applied to the panel surface, where `p_stagnation_peak = ½ · ρ_0 · V_local_max²` with `V_local_max = ω_blade_max · L_wrist_to_tip = 8.8 · 0.25 = 2.20 m/s` and `ρ_0 = 1.225 kg/m³`, giving `p_stagnation_peak ≈ 3.0 Pa`.

 **Important — implicit conservatism factor from uniform-vs-r² approximation:** real pressure scales as `p(r) = p_peak · (r/L)²` (centrifugal-pressure profile under rotation), so applying `p_peak` uniformly overstates the true root bending moment by `∫₀^L p_peak · r dr / ∫₀^L p_peak · (r/L)² · r dr = (L²/2) / (L²/4) = 2×`. This factor stacks with the explicit 2.5× stress-test multiplier:

   **Total bending-moment conservatism vs nominal physical loading: 2 × 2.5 = 5×**

 This is intentional stacked over-conservatism. A design that passes FEA at this loading level has substantial real-world margin: any failure point at the 5× load is operating ~5× below its actual cyclic capacity in canonical use.

 **Stress-test load** applies the 2.5× multiplier on top of `p_uniform`: `p_stress_test = 2.5 · p_stagnation_peak ≈ 7.5 Pa` applied uniformly.

 Inertial body load at α_max = 110 rad/s² (wrist-relative r, per §2.4). Click-engagement reaction force on the panel's outer tangential edge (per item #3 panel-edge relocation; HIGH-8 Option A chamfered butt joint).

 **Alternative (deferred to V2 if margin pressure emerges in Phase 5):** apply the actual `p(r) = p_peak · (r/L)²` radial distribution in FEA. Drops the implicit 2× factor and gives 2.5×-only margin against canonical loading. Requires §59.5 FEA load function to support radial pressure profiles (current implementation supports only uniform loads). Keep current uniform-load approach for V1; revisit if Phase 5 reveals stress-margin pressure.

 **Rationale:** the earlier draft used `p_uniform = F_peak / A_panel = (max_t J(t)) / A_panel`, which is the spatial average of pressure over the panel. Local pressure peaks at stagnation points are typically 2-5× the spatial average; the 2.5× stress-test multiplier does NOT substitute for spatial-distribution conservatism (they target different sources of margin). Using analytic stagnation pressure `½ρV²` as the uniform load is conservative against the spatial peak distribution. Auto-mapping a transient SU2 pressure field from a CFD tet mesh onto a fresh FEniCSx/CalculiX mesh is the kind of node-matching exercise that adds days to the gate without buying screening signal; uniform stagnation-peak loading is the right trade. Cost stays at ~5 min/design.
 - Pass criteria (all three must hold under the CANONICAL load case):
 1. **Tip deflection < 5 mm** under combined peak load (preserves the rigid-blade assumption that justifies the no-FSI CFD baseline).
 2. **Peak von Mises stress at the rib-panel fillet < 9.00 MPa nominal** AND **peak panel-pivot tension < 5.58 MPa nominal** AND **peak panel-pivot bearing (Z) < 2.00 MPa nominal** (per-mode independent checks from §10.1; the bearing mode binds first under canonical loading because the Z-direction fatigue factor is half the XY factor and σ_y_Z = 30 MPa).
 3. **First bending mode > 10 Hz** (5× the 2 Hz waving frequency).
 - **stress-test re-evaluation (occasional-peak load case; Architectural C ω scaling):** after the canonical pass, re-run the FEA at **2.5× p_uniform AND 2× α_max AND 1.41× ω_blade_max** (static, not fatigue-cycled — represents "user puts their back into it once or twice per session"). The 1.41× ω scaling doubles the centrifugal load (`F_c = m·ω²·r`); earlier spec omitted this and under-estimated stress-test centrifugal by 2×. Pass criteria for the stress-test load:
 1. Tip deflection < 5 mm (unchanged threshold; tip-deflection-driving load grows roughly 2-3× under stress-test, so a design with margin against canonical 5 mm may still fail this).
 2. **Peak σ_VM at the rib-panel fillet < 12.4 MPa nominal** AND **peak panel-pivot tension < 12.4 MPa nominal** (§10.1 static SF = 1.5 at K_t = 2.42 → 45/(1.5·2.42) ≈ 12.4 MPa) AND **peak panel-pivot bearing (Z) < 13.3 MPa nominal** (σ_y_Z/(SF·K_t_bearing) = 30/(1.5·1.5)). Replaces the cyclic-fatigue limits because stress-test is occasional-static, not fatigue.
 3. — (modal-criterion line removed; first bending mode is load-independent and already gated under the canonical case; re-asserting it under the stress-test load is at best a discretization-stability sanity check, not a stress-test criterion.)
 - **Fail action:** drop the design and use the next Pareto-front candidate. The §6.4 `qNEHVI` ranker re-ranks against surviving neighbors. **Fallback ladder if all top-3 fail the gate:** try **top-10** (relax the K=3 hard cap), then **top-50** (which costs ~250 min of FEA at the gate cost of ~5 min/design, still cheap). If none of the top-50 pass canonical + stress-test, **pause the campaign and surface to the human** — the §6.3.1 Filter 2 prefilter is systematically under-rejecting (Filter 3 is a deprecated pass-through stub, so the entire structural-prefilter rejection burden falls on Filter 2), and the BO is sampling a region where physical designs simply can't survive the §59.5 gate. Write `phase5/diagnostics/PAUSE_top50_fea_all_fail.txt` + PushNotification. A canonical-pass / stress-test-fail design is structurally underbuilt for occasional max-effort use and is dropped at the same gate as a canonical fail. Step 60 onwards proceeds only on designs that pass both the canonical and the stress-test load cases. **Failure-mode tagging:** each failing design records `failure_code ∈ {fea_combined_tip_defl, fea_combined_peak_stress, fea_combined_torsion, fea_stress_test_fail}` so post-campaign analysis can update the pre-CFD screen with the actual failure-driving mode rather than the assumed bending mode (the canonical Filter 2 only models bending compliance — if the FEA gate consistently rejects on torsion or buckling, the screen needs a new term).
 - **Cost:** ~5 minutes per design × 3 designs × 2 load cases (canonical + stress-test) = ~30 minutes total — cheap relative to step 60+'s 6-12 hour SU2 verification per design.
 - **Why this exists :** the Phase 2 rib u_tip check uses a smooth-baseline panel placeholder and is blind to Phase 2b's Layer 2 cutouts (§3.1.1 update). Phase 4/5 CFD verification doesn't see structural deflection (it assumes rigid). Without step 59.5, Phase 6 (after we've printed) is the first time a TPMS-heavy or noise-heavy design's excessive flex shows up — too late.
 - Associated regression test: `tests/test_topopt/test_combined_blade_stiffness.py` (synthetic rib + skeletal-panel example; asserts the gate correctly fails the example designed to be too flexible and passes the example designed to be stiff).

60. High-fidelity 3D Gmsh meshing for top-3 designs (`mesh_3d_verify.py`):
 - Reads top-3 design JSONs from Phase 4.
 - Generates **fully resolved corrugated geometry** (rib ridges and panel scoops explicitly meshed; no roughness model).
 - Finer boundary layer (~30 layers, first cell 0.02 mm) and higher overall cell count (~2-3 M cells) for verification accuracy.
61. SU2 unsteady verification configs (`fan3d_verify.cfg`):
 - Same compressible + low-Mach prec as Phase 4, with the higher-fidelity mesh.
 - 5 pitching cycles, dt = T/200; same J_fan post-processor.
62. **PyFR top-3 verification** (`pyfr_top3.py`):
 - High-order (p=3) discontinuous Galerkin.
 - GPU-native on Colab Pro G4 node.
 - Single 5-cycle unsteady run **per top-3 design**.
 - Compares J_fan against SU2's verification result for each design; flags any discrepancy >15% as needing investigation (mesh refinement, time-step reduction, or solver re-tuning).
62.5. **Body-in-still-air published-reference benchmark (2026-05-14 addition; replaces 2026-05-14-deferred Sub-spike 0.6c.2)** (`scripts/run_phase5_published_benchmark.py`):
 - **Purpose:** absolute-accuracy validation of the production Tier-1 cfg against published reference data in the *same regime* the fan operates in (body-in-still-air pitching, not wind-tunnel-frame). This step replaces the deferred Spike 0.6c.2 framing with a regime-appropriate target; see `docs/phase_logs/spike_0_6c.md` Note 1 for the deferral evidence trail.
 - **Reference case:** pick a published body-in-still-air pitching/flapping case with full force-trace data at Re ~ 10³–10⁴. Default candidates: Sane & Dickinson 2002 robotic-flapper protocol; Dickinson/Ellington/Birch insect-flight datasets; Sarpkaya plate-in-oscillating-flow tabulated coefficients (Morison-equation literature). Pick is documented in `docs/phase_logs/phase_5_signoff.md` at Phase-5 launch.
 - **cfg:** render a body-in-still-air cfg matched to the reference's kinematics + Reynolds via the wind-tunnel-frame template (`configs/su2/oscillating_airfoil_benchmark.cfg.j2` after 2026-05-14 rewrite), with `MACH_NUMBER` set per the production Tier-1 lock and a prescribed body motion matching the reference. The template generalizes (freestream → 0 + prescribed motion ON) without a separate cfg.
 - **Three-solver comparison:** run the reference case through (a) SU2 compressible with MACH=1e-9 + low-Mach prec (production Tier-1 numerics), (b) PyFR p=3 on Colab Pro G4 GPU (existing HIGH-11 lock; reuses the step-62 infrastructure), (c) **OpenFOAM `pimpleFoam` incompressible** on Colab Pro CPU (independent codebase; native incompressible, no low-Mach preconditioning involved — strongest available evidence on whether SU2's MACH=1e-9 trick is quantitatively faithful).
 - **Pass criterion:** SU2 and PyFR each match published cycle-averaged forces within ±15%; SU2 ↔ PyFR mutual agreement within ±10%; OpenFOAM agreement with SU2 within ±20% (advisory — disagreement flagged for investigation, not blocking).
 - **Cost:** SU2 ~6–12 h CPU; PyFR ~2–4 h GPU; OpenFOAM ~6–10 h CPU. Total Phase-5 compute addition ~14–26 h.
 - **Fail action:** if SU2 misses published cycle-averaged forces by >15%, Phase 5 reranking flags absolute-accuracy as compromised and the V1 ship decision falls back entirely on the Phase-6 qualitative blinded A/B feel test (which is already the V1 ship criterion — V1 is not blocked, but downstream V2 quantitative claims are gated on resolving this).
63. Final ranking script (`rerank_verified.py`):
 - For each top-3 design, computes the J_fan agreement between SU2 (verification) and PyFR.
 - Selects the **single final design** based on:
 - Highest J_fan_unsteady on the verified mesh (primary).
 - SU2↔PyFR agreement within 15% (gate; if any top-3 fails this gate, drop it and use the next-best).
 - Mass constraint < 100 g (C9 lock).
 - Fab-noise headroom: predicted J_fan gain over baseline must be ≥ 2× the Spike 0.5 fab-noise CV (so the gain is measurably distinguishable from print variation).
 - **Step 62.5 absolute-accuracy gate (2026-05-14 addition):** if step 62.5 flagged absolute-accuracy as compromised (SU2 missed published cycle-averaged forces by >15%), the final design is still selected on the criteria above for V1 ship purposes (V1 ships on qualitative-feel anyway), BUT any quantitative gain claim downstream of Phase 5 is suspended pending resolution. `rerank_verified.py` writes `results/phase4_<date>/step_62_5_status.json` with the absolute-accuracy flag; if compromised, the final design JSON is marked `quantitative_claims_gated: true`.
64. Final Fusion package (`finalize.py`):
 - Final manufacturable single-material PETG STL per blade (or full-assembly STL).
 - Final STEP for documentation.

**User runs:**

**64.5. `python scripts/verify_combined_blade.py --top3 results/phase4_<date>/top3.json` on the MacBook M3 (~5 min per design × 3 designs × 2 load cases = ~30 min total).** Runs the step 59.5 structural gate locally before any Colab compute commits. For each failure: fetch the next Pareto candidate from the Phase 4 frozen results, re-run, repeat until all three pass. Commit the verified-top-3 set as `results/phase4_<date>/top3_verified.json`. **Steps 65+ use this verified set, not the original Phase 4 top-3.** Skipping this step risks burning 6-12 hours of Colab SU2 verification on a skeletal-panel design that flexes >5 mm at peak load.

65. Upload **top3_verified.json** (NOT the unverified Phase 4 output) to Colab Pro.
66. Run SU2 verification on all 3 verified designs (~6-12 hours per design on Colab Pro, parallelizable).
67. `python pyfr_top3.py` on Colab Pro GPU for the top-3 verified designs (~2-4 hours per design).
68. `python rerank_verified.py` → final design JSON.
69. Visually inspect the final geometry in Fusion. Approve manufacturable STL set.

**Outcome:** Single final design selected from the top-3 with **structural-gate clearance (step 59.5 / 64.5)** + cross-solver CFD verification (SU2 vs. PyFR). Manufacturable single-material PETG STL set ready for Phase 6 printing.

### Phase 6: Single-Material PETG Print + IMU Validation + Acoustic Measurement (Week 12-13)

Single-material PETG throughout (no multi-material, no TPU, no AMS tool changes). **Acoustic measurement** captures the vortex-shedding tone of the corrugated rigid surface. The mechanical pivot-stack and folded-form-factor checks are gating verifications, not pass-fail nice-to-haves.

**User does (Claude writes any helper scripts on request):**

70. **Multi-blade PETG print of the final design.** Strategy decided in Phase 1 based on bed size:
 - **Per-blade prints (default, 256 mm bed):** print 10 blades sequentially (~1-2 h each), inspect each before assembly. Total wall-clock 10-20 h.
 - **Full-assembly print (≥360 mm bed):** print all 10 blades in deployed configuration in one job (~8-12 h continuous). If any blade fails, restart entire print.
 Print settings: 0.2 mm layer height (0.15 in click feature zone), 3-4 walls; **panel printed as whatever the §9.7 generator produced** — solid, louvered, TPMS-perforated, noise-cut, or any Layer-2 combination thereof. Slicer's infill density is moot (the generator already produced the solid/voided regions in the STL); set walls = 12+ on a 5 mm panel for full perimeter coverage of any solid regions. PETG at recommended temps.
71. Print or source pivot pin: 3 mm brass rod ≥45 mm long. **No spacers between adjacent panels** (M18 lock — any Z-pitch addition disengages the panel-edge click chamfers per §2.3). The pin stack runs through 10 panels directly under panel-pivot architecture.
72. **Pivot stack mechanical verification:** assemble all 10 panels onto the pivot pin. Measure pin bending under transverse load (apply 5 N at the blade tips, measure pin deflection at mid-pin). Verify deflection < 1 mm. Check rib-to-rib friction by deploying/folding manually; force should be ~2-5 N at the handle.
73. **Folded form factor verification:** fold the assembled fan completely; measure stack thickness with calipers. **Target: matches the §6.4 Pareto-chosen folded form factor for the design under test** (22-42 mm range at 10 blades depending on the Pareto-chosen 2.2-3.8 mm panel thickness). If measured stack thickness exceeds the design-specified value by >5 mm, investigate per-blade thickness and recheck the §9.7 generator output against the JSON `panel_thickness` control points.
74. Final assembly: pivot pin through all 10 panel pivot holes; peen or use a nut. Verify click-feature engagement on all 9 inter-blade pairs.
75. **3-copy fabrication-noise recheck (M17 lock — composition explicit):** isolates variance to the newly-printed test blade. Procedure: **(1)** keep 9 of the original Phase 4-print blades from the top-3 Pareto design under test — the same blades used in steps 70-74 (NOT fresh prints of those blades). **(2)** Print 3 fresh copies of the same blade design (one blade per print run). **(3)** Install each fresh copy in **slot N (default: slot 5 — the middle of the deployed sector — chosen for symmetry; same slot for all 3 measurements)** of the otherwise-original 10-blade assembly; measure J_fan_measured / W_cycle. The variance across the 3 measurements isolates the newly-printed blade's fabrication noise. Run the Spike 0.5 protocol again with the tightened CV < 5% target. If CV has risen above 5%, document and investigate.

**Physical validation:**

76. **IMU instrumentation:** attach a phone (or strapdown gyro) to the fan handle. Record full waving kinematics (angular position θ(t), angular velocity ω(t)) for each test.
77. Compute angular work per cycle: `W_cycle = ∫_0^T I_wrist · ω · dω/dt dt`, where **I_wrist** is the rotational inertia measured about the **handle-grip wrist axis** (the same axis §6.4's BO Pareto objective uses) via the Spike 0.2 torsional-pendulum rig (one inertia measurement per design and per copy in the 3-copy noise recheck). Cross-check: I_wrist_measured should agree with `I_wrist_kgm2` emitted by the generator (§6.4) within the 3% repeatability bound from Spike 0.2; any disagreement >10% indicates either a CAD-density-vs-actual-PETG-density mismatch or a generator-frame-vs-pendulum-frame axis-convention bug and must be resolved before the W_cycle comparison is published.
78. Anemometer testing **at 300 mm matching the J_fan plane location** (§9.4). **L8 lock — 9-point grid (NOT single point):** the handheld anemometer measures point velocity, but the §9.4 spec integrates over the 600×600 mm plane. To approximate the plane integral, take **9 point measurements arranged in a 3×3 grid covering the 600×600 mm plane at 300 mm** (point spacing 200 mm, centered on the pivot's +z axis). Each point is averaged over 10 cycles, metronome-paced at 2 Hz. The mean of the 9 averaged values approximates the plane-integrated velocity; multiply by the plane area (0.36 m²) to get a proxy for J_fan_measured. The proxy is a coarse approximation (3×3 grid undersamples the spatial distribution) but bounds the variance from a single-point measurement and matches the J_fan spec's plane convention better than one point. **Budget:** 9 points × 10 cycles × 30 s/cycle = 45 min per design (within Phase 6's 1-2 week window). Report `J_fan_measured / W_cycle` -- the apples-to-apples efficiency metric.
79. Compare to **Spike 0.3 baseline** `J_fan_measured / W_cycle`. **Target: >15% improvement.**
80. **Kinematics validation:** plot the IMU-recorded θ(t), ω(t) waveforms and compare to the pure-pitching SHM model used in CFD. If real waving has noticeable translation (the user's wrist moves the pivot point during the wave), document and consider feeding the actual kinematics back into a Phase 4 sensitivity run.
81. Static deflection test: clamp a blade at the pivot, hang a 100 g weight from the tip, measure deflection with calipers. Compare to FEA prediction (agree within 10%). Verifies the rigid-blade assumption.
82. Pivot fatigue test: wave continuously for 30 minutes (~3600 cycles) at metronome pace. **Inspect the pivot region on all 10 panel pivot holes under magnification for cracks** (under panel-pivot architecture, the 10 panels carry the pivot hole; the ribs sit off-axis at `y = ±rib_center` and have no pivot hole — see §0 row 25 and §3.1.2). **aggressive-effort stress segment (after the 30-minute metronome-paced run):** add a **5-minute aggressive-effort segment** — instruct the user to wave the fan as hard as they can comfortably manage one-handed (subject is encouraged to alternate brief bursts at higher amplitude/frequency rather than sustain the peak for 5 minutes continuously). **Inspect for crack initiation at each panel's pivot hole + at the rib-panel interface on each blade** after the segment. **Pass criterion:** no visible cracks after the aggressive segment. A pass here validates the stress-test design margin against real-world occasional peak loading.
83. **Click-feature long-cycle test (extends Spike 0.4):** open/close the fan in stages — 100, 500, 1000, and 3000 cycles — inspecting click-feature engagement quality at each stage. Record the cycle count at first sign of detent wear or alignment drift. **Target: ≥3000 cycles without functional failure.**
84. Smoke visualization: incense stick at 150 mm; record video; look for directed airflow pattern, vortex shedding from rib ridges, and any unexpected flow features.
85. **Acoustic measurement:** microphone at 300 mm (matching the J_fan plane), 10-second recordings during waving. Compute the dominant vortex-shedding frequency from FFT and compare to the CFD-predicted frequency. **L13 Nyquist disclosure:** CFD time-step `dt = T/200 = 2.5 ms` has a Nyquist limit of **200 Hz**. The geometry's estimated shedding frequency `Strouhal × V_tip / rib_thickness ≈ 0.21 × 2.20 / 0.003 ≈ 147 Hz` falls within this limit. If physical acoustic measurement reveals a dominant shedding frequency **> 200 Hz**, the CFD has under-resolved the relevant physics and Phase 4's calibrated wall-roughness model has missed a contribution; document the gap and consider a Phase 5 verification at finer `dt` (e.g., T/400) on top-3 designs to reconcile. The corrugated surface may produce different shedding tones than a smooth surface; characterize this for documentation. A particularly loud tone (>50 dBA above background at the shedding frequency) is an aesthetic-quality concern that may motivate a Pareto re-weighting.

**Telemetry — session-health panel:**

A small post-processor (`scripts/session_health.py`) reads the mtimes of `phase4/session_*/.heartbeat` markers and prints a status table: session ID, hardware, last heartbeat (mtime − now), evals completed, current tier, slice progress percentage, any `.failed_*` counts in the assigned slice. Flag any session with stale heartbeat > 15 min as **dead** (suggests Colab disconnect, kernel OOM, or browser tab closed). Manual remediation: re-slice the dead session's remaining hashes round-robin across the live sessions via `python scripts/rebalance_slice.py --dead session_X` (writes a new `slice_assignments_v{N}.json` and bumps `next_batch.txt`). Expected 0-2 rebalances per 2-3 week campaign.

**Feedback to model:**

- Any discrepancy >20% between Phase 5 verified J_fan and IMU-normalized measurement triggers a model-calibration pass. Claude updates the relevant model parameters (FDM material moduli, mesh resolution at the rib ridges and click features, SU2 wall-roughness model for the corrugated surface, low-Mach preconditioning coefficients); user re-runs the affected phase as needed.
- Any click-feature failure within 1000 cycles triggers a click-feature redesign: larger detent radius, magnetic upgrade per Spike 0.4 fallback, or adjusted clearance.
- The 3-copy fab-noise recheck against Spike 0.5 catches print-process drift between the calibration phase and the final part.
- Measured folded form factor differs from the design's §6.4 Pareto-chosen value by >5 mm → investigate the §9.7 generator output vs. as-printed thickness (matches step 73).

**Outcome:** Validated, characterized, single-material PETG folding fan with discrete V-unit blades. Measurable performance gain over baseline, calibrated against IMU-normalized angular work, with quantified manufacturing variability, fatigue characteristics, and acoustic signature.

---

## 9. Tool Guides and Configuration

### 9.1 CadQuery -- Parametric Rib Generation

#### Installation

```bash
# Recommended: conda (better-tested, handles OCP binary dependency automatically)
conda install -c conda-forge cadquery

# Alternative: pip (requires pre-built OCP wheels; only Python 3.9-3.12)
# pip install cadquery
# May fail on some platforms (notably macOS Apple Silicon with certain Python versions)

# For visualization (optional):
pip install cadquery-ocp
```

#### Example: Parametric Fan Rib (Claude Would Write This)

```python
"""
Parametric folding fan rib generator.
Claude Code writes this; user runs it to generate STL files.
"""
import cadquery as cq
import math
import argparse

HUB_RADIUS_M = 0.020     # C7 / Architectural D: inner rib boundary (HUB / boss region is panel-only)
RIB_TIP_TAPER_M = 0.015  # Architectural A: outer rib boundary (click region is panel-only)
L_BLADE_M = 0.200        # blade full radial extent
L_RIB_M = L_BLADE_M - HUB_RADIUS_M - RIB_TIP_TAPER_M  # = 0.165 m

def generate_rib(
 rib_radial_length=L_RIB_M,   # m, rib radial length 165 mm (C7 + Architectural A bands)
 rib_x_start=HUB_RADIUS_M,    # m, rib starts at HUB_RADIUS = 0.020 m (NOT at x = 0)
 base_width=0.004,    # m, width at rib root x = HUB_RADIUS (H12 narrow-at-root taper)
 tip_width=0.006,     # m, width at rib tip x = L_blade − RIB_TIP_TAPER (H12 wider-at-tip)
 thickness=0.002,     # m, rib thickness (FDM Z-extent)
 camber=0.0,          # m, maximum camber (curvature) at mid-span
):
 # HIGH-9 Round-9 lock: NO pivot-hole parameters in this signature.
 # The pivot pin runs through the panel at y = 0 per the panel-pivot architecture
 # (§0 row 25 + §3.1.2). The rib radial extent is [HUB_RADIUS, L_blade − RIB_TIP_TAPER]
 # = [0.020, 0.185] m — entirely outside the pivot's blade-frame x = 0.008 m position.
 # The earlier pivot_hole_dia + pivot_offset parameters violated this lock and were
 # caught by the test_no_rib_pivot_hole.py CI gate (extended to scan §9.1 code blocks
 # per HIGH-9 Round-9 lock).
 #
 # NOTE: ALL geometry passed through this module is in SI units (meters; C9 lock).
 # CadQuery is unit-agnostic; mixing mm and m silently corrupts I_wrist by (1e-3)⁵ = 10⁻¹⁵.
 # The `tests/test_geometry/test_units_meters.py` CI gate enforces this lock.
 """Generate a single fan rib as a CadQuery Workplane object.

 Returned geometry lives in x ∈ [rib_x_start, rib_x_start + rib_radial_length]
 = [HUB_RADIUS_M, L_BLADE_M − RIB_TIP_TAPER_M] = [0.020, 0.185] m. The inner 20 mm
 (HUB region, panel-only — hosts the 12 mm boss + pivot hole IN THE PANEL, not the rib)
 and outer 15 mm (click region, panel-only — hosts the chamfer + detent on the panel
 outer tangential edge) are emitted by make_panel_solid, NOT by this function.

 NO PIVOT HOLE is drilled in the rib (HIGH-9 Round-9 lock; panel-pivot architecture)."""

 # Define rib outline points (tapered trapezoid)
 half_base = base_width / 2
 half_tip = tip_width / 2

 # Create 2D profile with H12 UP-taper (narrow root → wider tip)
 points = [
 (rib_x_start, -half_base),                       # root, bottom edge (at x = HUB_RADIUS)
 (rib_x_start + rib_radial_length, -half_tip),    # tip, bottom edge (at x = L_blade − RIB_TIP_TAPER)
 (rib_x_start + rib_radial_length, half_tip),     # tip, top edge
 (rib_x_start, half_base),                        # root, top edge
 ]

 # Create the rib body — extruded prism, NO pivot hole (HIGH-9 lock)
 rib = (
 cq.Workplane("XY")
 .polyline(points)
 .close()
 .extrude(thickness)
 )

 # Add camber (optional curvature along the length)
 # For camber > 0, we would use a lofted or swept profile
 # This simple version uses a flat rib; camber is added in a more
 # advanced version using spline-based sweep

 return rib

def generate_guard_stick(
 length=L_RIB_M,      # m, guard stick radial length 165 mm (same as rib — HIGH-9 lock)
 rib_x_start=HUB_RADIUS_M,  # m, guard starts at HUB_RADIUS = 0.020 m
 width=0.018,         # m, guard width (wider than inner ribs)
 thickness=0.003,     # m, guard thickness (thicker than inner ribs for stiffness)
):
 """Generate a guard stick (outer rib, wider and thicker; NO pivot hole per HIGH-9 lock).

 The guard stick has NO pivot hole — under panel-pivot architecture, the pivot pin runs
 through the panel at y = 0; the guard stick sits at the outer y-position and engages
 the fan at its angular extent. The earlier pivot_hole_dia + pivot_offset parameters
 violated the panel-pivot architecture lock and were caught by HIGH-9 Round-9."""
 guard = (
 cq.Workplane("XY")
 .center(rib_x_start + length / 2, 0)
 .rect(length, width)
 .extrude(thickness)
 )
 return guard

def main:
 parser = argparse.ArgumentParser(description="Generate folding fan ribs")
 parser.add_argument("--rib-count", type=int, default=15)
 parser.add_argument("--length", type=float, default=200.0)
 parser.add_argument("--base-width", type=float, default=12.0)
 parser.add_argument("--tip-width", type=float, default=6.0)
 parser.add_argument("--thickness", type=float, default=2.0)
 parser.add_argument("--output", type=str, default="ribs/")
 args = parser.parse_args

 import os
 os.makedirs(args.output, exist_ok=True)

 # Generate inner ribs
 n_inner = args.rib_count - 2 # subtract 2 guard sticks
 for i in range(n_inner):
 rib = generate_rib(
 length=args.length,
 base_width=args.base_width,
 tip_width=args.tip_width,
 thickness=args.thickness,
 )
 cq.exporters.export(rib, f"{args.output}/rib_{i+1:02d}.stl")
 print(f"Generated rib_{i+1:02d}.stl")

 # Generate guard sticks
 for side in ["left", "right"]:
 guard = generate_guard_stick(length=args.length)
 cq.exporters.export(guard, f"{args.output}/guard_{side}.stl")
 print(f"Generated guard_{side}.stl")

 print(f"Total: {n_inner} inner ribs + 2 guard sticks")

if __name__ == "__main__":
 main
```

### 9.2 Rib Topology Optimization -- 2D Plate-Bending Formulation
**Note:** §9.2. The rib TO uses **2D plate-bending (Reissner-Mindlin)**. See §3.1 for the full formulation.

#### Why 2D Plate-Bending, Not 3D Voxel or 2D Plane-Stress

The rib is 200 mm long, 6-12 mm wide, and 2 mm thick. Two formulation choices to consider:

1. **3D voxel TO** with 0.5 mm voxels yields only **4 voxels through the 2 mm thickness**. With 4 discrete levels, the optimizer can only choose 25%, 50%, 75%, or 100% thickness at each planform location — this is thickness optimization with 4 levels, not real topology optimization. Features like lightening holes require at least 3-4 voxels to form (one void surrounded by material on each side), which consumes the entire thickness. **Rejected** because the resolution-to-thickness ratio is too poor for genuine topological freedom.
2. **2D plane-stress** treats the rib as having constant 2 mm thickness and optimizes material distribution in the length-width plane. At 0.5 mm element size, the rib has 400 × 24 = 9,600 elements with full design freedom in both directions. But: plane-stress models in-plane loading only and does NOT capture out-of-plane bending. The rib's primary load (aerodynamic pressure normal to the panel surface) produces out-of-plane bending; plane-stress would minimize the wrong compliance metric. **Rejected** as a physical-correctness failure.
3. **2D plate-bending (Reissner-Mindlin):** retains the 2D 9,600-element design freedom of plane-stress but uses plate-bending finite elements with bending DOFs (rotations + transverse displacement). Captures the out-of-plane bending physics correctly. FEniCSx supports this directly via mixed function spaces. **This is the chosen formulation.**

This is the correct formulation for thin-walled structures whose primary load is transverse pressure rather than in-plane membrane loading.

#### Installation (DTU TopOpt Python Path)

```bash
# The DTU TopOpt codes are self-contained Python scripts using only NumPy/SciPy.
# No special installation needed beyond:
pip install numpy scipy matplotlib

# Download the DTU TopOpt Python code from:
# https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python
```

#### Alternative Installation (FEniCS Path)

```bash
# FEniCS provides more advanced TO capabilities (multi-load, stress constraints)
# but has a heavier installation:
conda install -c conda-forge fenics-dolfinx
# Or: pip install fenics-dolfinx (on supported platforms)

# FEniCS SIMP tutorial:
# https://comet-fenics.readthedocs.io/en/latest/demo/topology_optimization/simp_topology_optimization.html
```

#### Example: 2D Rib Planform TO (Claude Would Write This)

```python
"""
2D topology optimization of a fan rib planform using SIMP.
Claude Code writes this; user runs it.

**This pedagogical example uses 2D plane-stress for clarity** (matches the
classic DTU TopOpt Python codes it's adapted from). Production code
uses 2D Reissner-Mindlin plate-bending per §3.1 — see `src/fanopt/topopt/
plate_bending.py` for the bending element + assembly used in Phase 2.
The plane-stress version below is retained for didactic comparison; do
NOT use it directly for the rib TO.

The rib has constant 2mm thickness; TO determines where material is
placed within the tapered planform envelope.
"""
import numpy as np
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# --- Problem parameters (NOTE: pedagogical DTU TopOpt block — mm-units for clarity
#     against the published 88-line code; production code in src/fanopt/topopt/
#     uses the SI/meters lock per C9 + the C7 HUB_RADIUS / Architectural A
#     RIB_TIP_TAPER rib radial band per C13 lock) ---
HUB_RADIUS_MM = 20.0      # C7: inner rib boundary; rib starts here, not at x = 0
RIB_TIP_TAPER_MM = 15.0   # Architectural A: outer rib boundary
L_BLADE_MM = 200.0        # blade full radial extent
L_rib = L_BLADE_MM - HUB_RADIUS_MM - RIB_TIP_TAPER_MM  # = 165 mm rib radial length
w_base = 4.0 # mm, width at rib root x = HUB_RADIUS (H12 narrow-at-root)
w_tip = 6.0 # mm, width at rib tip x = L_blade − RIB_TIP_TAPER (H12 wider-at-tip)
thickness = 2.0 # mm, constant rib thickness (used for plane-stress stiffness)
volfrac = 0.40 # volume fraction target
penal = 3.0 # SIMP penalization exponent
rmin = 1.5 # filter radius in mm
E0 = 1300.0 # MPa, FDM PETG in-plane modulus
nu = 0.38 # Poisson's ratio

# --- Discretization ---
elem_size = 0.5 # mm
nelx = int(L_rib / elem_size) # 330 elements along rib radial length (165 mm / 0.5 mm)
nely_max = int(w_tip / elem_size) # 12 elements max width (use the LARGER tip width since H12 is UP-tapered)

# Create tapered domain mask: which elements are inside the rib envelope
active = np.zeros((nely_max, nelx), dtype=bool)
for ix in range(nelx):
 x_frac = ix / nelx
 w_local = w_base - (w_base - w_tip) * x_frac
 n_active = int(w_local / elem_size)
 y_start = (nely_max - n_active) // 2
 active[y_start:y_start + n_active, ix] = True

n_active_elems = active.sum
print(f"Active elements: {n_active_elems} (of {nelx * nely_max} total)")

# --- Preserved regions (panel-pivot architecture: NO pivot hole in rib; C15 diagonal BC fix) ---
# Rib-panel interface (the rib's inner y-edge over the rib radial band) + rib-panel fillet
# are density-clamped to 1.0. The legacy "pivot region" + "click-feature footprint" preserved
# zones are removed (panel-pivot architecture moves them to the panel; rib has neither).
preserved = np.zeros_like(active, dtype=bool)

# C15 lock: under the trapezoidal panel widening (item #35), the rib-panel interface
# is a DIAGONAL curve, not a single Cartesian row. The interface follows
#     y_interface(x) = sign · panel_width(x) / 2     (sign = +1 inner, -1 outer rib)
# where panel_width(x) = r(x) · INTER_BLADE_ANGLE_RAD − 2·rib_width(x) − 0.5 mm gap.
# Hardcoding `inner_edge_row_idx = 0` (an earlier draft) clamps a horizontal line
# at y = 0, cutting diagonally THROUGH the rib material — the SIMP TO would converge
# to a meaningless optimum with BCs internal to the rib body.

def get_rib_panel_interface_mask(active_mask, x_grid_m, y_grid_m,
                                 panel_width_fn, rib_side="inner"):
    """Identify the diagonal rib-panel interface in the 2D planform mesh.
    Returns a boolean mask of grid cells along the interface; these are
    (a) density-clamped to ρ = 1 (preserved-zone enforcement) AND
    (b) have their transverse-displacement + rotation DOFs fixed (plate-bending BC)."""
    sign = +1 if rib_side == "inner" else -1
    interface_mask = np.zeros_like(active_mask, dtype=bool)
    for x_idx, x_m in enumerate(x_grid_m):
        if HUB_RADIUS_M <= x_m <= L_BLADE_M - RIB_TIP_TAPER_M:
            y_interface = sign * panel_width_fn(x_m) / 2
            y_idx = np.argmin(np.abs(y_grid_m - y_interface))
            interface_mask[x_idx, y_idx] = True
    return interface_mask

inner_edge_mask = get_rib_panel_interface_mask(
    active, x_grid_m, y_grid_m, panel_width_fn, rib_side="inner"
)
preserved[inner_edge_mask] = True  # density-clamp ρ = 1 along the DIAGONAL interface

# Rib-panel fillet strip (2 mm × 2 mm cross-section over the rib radial band) is
# density-clamped at ρ = 1 in the planform; the actual 3D fillet stress is evaluated
# in Phase 2.5 + §59.5 (C14 lock — 2D Reissner-Mindlin cannot resolve the z-axis fillet).

# --- Boundary conditions: fix transverse displacement + rotations along the
#     DIAGONAL rib-panel interface mask (NOT a single Cartesian row) ---
# Under panel-pivot architecture + trapezoidal panel, the rib's anchor is the
# diagonal rib-panel y-edge; the Dirichlet clamp follows the same curve as the
# preserved-zone mask above.
fixed_dofs = dofs_at_mask(inner_edge_mask, dof_components=("u_z", "θ_x", "θ_y"))

# --- Loads: distributed pressure on one face ---
# Aerodynamic pressure: 10 Pa over tributary width ~15 mm
# Force per unit length on rib: 10 Pa * 15 mm = 0.15 N/m = 0.00015 N/mm
# Applied as nodal forces on the top edge of the 2D domain.

# --- Core SIMP optimization loop (OC update) ---
# [Claude writes the full FE assembly, sensitivity computation, filtering,
# and optimality criteria update here -- approximately 100-150 lines.
# The structure follows the DTU 88-line code adapted for:
# (1) non-rectangular (tapered) domain via the 'active' mask,
# (2) preserved regions via the 'preserved' mask,
# (3) distributed load instead of point load.]

# --- Output ---
# Save optimized density field
# np.save("optimized_density.npy", rho)
# Visualize: plot density field showing cutout pattern
# plt.imshow(rho, cmap='gray_r', origin='lower')
# plt.title("Optimized rib planform (black=material, white=void)")
# plt.savefig("optimized_rib_planform.png", dpi=200)
```

**What the output looks like:** The 2D TO produces a density field showing which parts of the rib planform are material (density near 1.0) and which are void (density near 0.0). The result is a rib with cutout patterns -- full material near the pivot where bending moment is highest, lightening holes or truss-like structures in the mid-span and tip where loads are lower. A post-processing script (Claude writes) converts this density field to a 3D rib geometry via CadQuery by extruding the material regions to 2 mm thickness.

#### Alternative: PyTopo3D with Source Modification (for thicker ribs)

If you increase rib thickness to 4-5 mm (yielding 8-10 voxels through thickness), 3D TO via PyTopo3D becomes meaningful. However, note that the actual PyTopo3D API is:

```python
from pytopo3d.core.optimizer import top3d

# This is the REAL API -- a single function call, not a class.
# BCs and loads are HARDCODED to the standard cantilever benchmark.
result = top3d(
 nelx=400, nely=24, nelz=10, # 10 voxels through 5mm at 0.5mm resolution
 volfrac=0.4,
 penal=3.0,
 rmin=3.0,
 disp_thres=0.5,
 obstacle_mask=None, # optional: binary mask for forbidden regions
 use_gpu=False,
)
```

To use this for fan ribs, Claude would need to **fork the PyTopo3D repository** and modify `pytopo3d/core/optimizer.py` to:
1. Accept custom fixed-DOF arrays instead of hardcoded left-face fixity (the rib's anchor is the rib-panel y-edge under panel-pivot architecture, NOT a left-face pivot).
2. Accept custom force vectors instead of the hardcoded point load.
3. Add a preserved-region mask (density fixed to 1.0 at the rib-panel interface + rib-panel fillet + click-feature footprint — NOT a "pivot area").

Estimated modification effort: ~200-400 lines of Python changes. This is feasible but should be understood as a source-level fork, not standard API usage.

### 9.3 CalculiX -- FEA Verification (Claude Writes .inp Files)

#### Example: Rib Stress Verification

```
** CalculiX input file for rib stress verification
** Claude Code generates this file; user runs: ccx rib_verify
**
*HEADING
Fan rib FEA verification - orthotropic PETG
**
*INCLUDE, INPUT=rib_mesh.inp
** (mesh generated by Gmsh, exported as CalculiX format)
**
** Material: FDM-printed PETG (orthotropic)
*MATERIAL, NAME=PETG_FDM
*ELASTIC, TYPE=ENGINEERING CONSTANTS
1300.0, 1300.0, 1000.0, 0.38, 0.38, 0.38, 500.0, 500.0,
385.0
*DENSITY
1.27e-9
**
*SOLID SECTION, ELSET=ALL_ELEMENTS, MATERIAL=PETG_FDM
**
** Boundary conditions: fix rib-panel interface (full-length y-edge under panel-pivot architecture)
** NOT a "pivot node set" — the rib has no pivot hole under panel-pivot architecture.
*BOUNDARY
RIB_PANEL_INTERFACE_NODES, 1, 3, 0.0
**
** Loads: distributed pressure on top face
*DLOAD
TOP_FACE, P2, 0.010
** (10 Pa pressure, positive = into surface)
**
** Analysis step
*STEP
*STATIC
*NODE FILE
U
*EL FILE
S
*END STEP
```

### 9.4 SU2 -- CFD for Deployed Fan

#### 9.4.1 — SU2 Config-Hash Assertion + Per-Run Metadata Sidecars

Every SU2 run (Tier -1 / Tier 0 / Tier 1 / Phase 5 verification) opens with a config-hash assertion against the §0 locked-decisions snapshot:

```python
# At the top of every SU2 launcher. The 2D slice (Tier -1) has different Re
# from the 3D tiers (18000 vs 37000), and only the unsteady tiers have a
# TIME_STEP / PITCHING_OMEGA — a single cross-tier dict would fail on every
# Tier-(-1) launch. Split into cross-tier constants + tier-specific values.
locked = json.load(open("gdrive/fan-optimization/locked_decisions.json"))

CROSS_TIER = {
 # Every key that the freestream / kinematics / materials chain reads from must be here
 # so any config edit fail-closes instead of silently producing zero-motion or wrong-axis runs.
 # NOTE: REYNOLDS_LENGTH moved to TIER_SPECIFIC — Tier -1's 2D slice uses c = 0.2 m chord,
 # Tier 0/1 use the L = 0.25 m wrist-to-tip reference. A single cross-tier value made
 # Tier -1's effective viscosity ~22% high (V·REYNOLDS_LENGTH/REYNOLDS_NUMBER mismatch).
 # NOTE: MACH_NUMBER also moved to TIER_SPECIFIC under the HIGH-12 Round-9 lock — Tier
 # -1 / Tier 0 use the steady-proxy MACH = V_tip/c = 0.0064 (V_tip as freestream, body
 # stationary), while Tier 1 unsteady uses MACH = 1e-9 with FREESTREAM_OPTION =
 # FREESTREAM_VELOCITY override (body moves via GRID_MOVEMENT; ambient near-zero).
 # A single cross-tier MACH would fire `config_mismatch` on every Tier-1 run.
 "j_fan_plane_m": 0.300,
 "j_fan_plane_size": 0.600,
 "MOTION_ORIGIN": (0.0, 0.0, 0.0), # wrist-grip at world origin (cross-tier; relevant when grid is moving in Tier 1)
 # PITCHING_*_AXIS moved to TIER_SPECIFIC[1] (L5 lock) — Tier -1 / Tier 0 have no pitching motion.
 "FREESTREAM_PRODUCTIVE": (0.0, 0.0, -1.0), # productive stroke: fan sweeps +z toward user → air flows -z in fan frame (C2)
 "FREESTREAM_RETURN": (0.0, 0.0, +1.0), # return stroke (C2)
 "V_TIP_MPS": 2.20,
 "T_CYCLE_S": 0.5,
 "ALPHA_MAX_RADS2": 110.0,
 "K_REDUCED": 0.57,
 # Material constants (full set; consumed by Filter 2, §59.5, J_fan post-processor):
 "RHO_PETG_KGM3": 1270.0,
 "RHO_PIN_KGM3": 7850.0, # steel; brass = 8500 is a documented variant
 "E_PETG_XY_MPA": 1300.0,
 "E_PETG_Z_MPA": 1000.0,
 "SIGMA_Y_PETG_XY_MPA": 45.0,
 "SIGMA_Y_PETG_Z_MPA": 30.0, # §10.1 lock — FDM anisotropy 20-40% reduction from σ_y_XY; conservative middle
 "K_T_PIVOT_TENSION": 2.42, # d/w = 0.25 in 12 mm boss (panel-pivot architecture)
 "K_T_PIVOT_BENDING": 3.2, # literature upper bound
 "K_T_BEARING": 1.5, # pin-bore baseline
}
TIER_SPECIFIC = {
 -1: {"REYNOLDS_NUMBER": 18000.0, "REYNOLDS_LENGTH": 0.20, "MACH_NUMBER": 0.0064},  # 2D slice at mid-radius chord c = 0.2 m; no pitching; steady proxy MACH = V_tip/c
 0: {"REYNOLDS_NUMBER": 37000.0, "REYNOLDS_LENGTH": 0.25, "MACH_NUMBER": 0.0064},  # 3D steady; wrist-to-tip L = 0.25 m; no pitching; steady proxy MACH = V_tip/c
 1: {"REYNOLDS_NUMBER": 37000.0, "REYNOLDS_LENGTH": 0.25,  # 3D unsteady; wrist-to-tip L
 "MACH_NUMBER": 1e-9,  # HIGH-12 Round-9 lock: near-zero Mach paired with FREESTREAM_OPTION = FREESTREAM_VELOCITY override; body motion via GRID_MOVEMENT = RIGID_MOTION, NOT via MACH-based tailwind.
 "TIME_STEP": 0.0025,
 "MAX_TIME": 2.5,
 "TIME_ITER": 1000,
 "PITCHING_OMEGA_AXIS": (0.0, 1.0, 0.0),  # +y wrist axis (L5: tier-specific to Tier 1)
 "PITCHING_AMPL_AXIS": (0.0, 1.0, 0.0),   # MUST match OMEGA axis or zero-motion (L5: tier-specific)
 "PITCHING_OMEGA_SIGNED_Y_RADS": -12.5664,  # C11 SIGN LOCK: NEGATIVE — rendered cfg emits (0, -12.5664, 0). Right-hand-rule on productive stroke requires the negative sign. Including the SIGN in the hash so any drift to (0, +12.5664, 0) triggers config_hash mismatch.
 "PITCHING_OMEGA_MAG_RADS": 12.5664,  # magnitude only; the SIGN lives in PITCHING_OMEGA_SIGNED_Y_RADS above.
 "PITCHING_AMPL_RAD": 0.6981, # 40° = 0.6981 rad
 "N_CYCLES": 5}, # bump to 8 if §Phase 3 step 39 rule fires
}
asserted = {**CROSS_TIER, **TIER_SPECIFIC[tier]}
for k, v in asserted.items:
 if abs(config[k] - v) > 1e-9:
 write_failure_record(failure_code="config_mismatch", detail=f"{k}: got {config[k]}, locked {v}")
 sys.exit(1)
```

**Four-hash separation (H16 lock — geometry_hash added):** `config_hash`, `material_hash`, `geometry_hash`, and `physics_hash` are distinct and related by construction:
- **`config_hash := H(resolved SU2 .cfg file)`** — captures the literal SU2-resolved key/value config text (CROSS_TIER ∪ TIER_SPECIFIC CFD-config values as written into the .cfg). Changes whenever any CFD-config key in the assertion changes.
- **`material_hash := H(MATERIAL_LOCKS)`** — captures the locked material table (`RHO_PETG`, `RHO_PIN`, `E_PETG_XY`, `E_PETG_Z`, `σ_y_XY`, `σ_y_Z`, full K_t hotspot table, fatigue factors). Changes whenever §10.1 / `material_locks.py` changes.
- **`geometry_hash := H(GEOMETRY_LOCKS)`** — captures the locked geometry constants (H16 lock). The dict:
 ```python
 GEOMETRY_LOCKS = {
     "HUB_RADIUS_M": 0.020,
     "RIB_TIP_TAPER_M": 0.015,
     "L_BLADE_M": 0.200,
     "D_HANDLE_M": 0.050,
     "RIB_THICKNESS_M": 0.002,
     "RIB_BASE_WIDTH_M": 0.004,
     "RIB_TIP_WIDTH_M": 0.006,
     "INTER_BLADE_ANGLE_RAD": 0.232,
     "PIVOT_BOSS_RADIUS_M": 0.006,
     "PIVOT_CENTER_X_M": 0.008,
     "CLICK_FOOTPRINT_X_RANGE_M": (0.190, 0.200),
     "PANEL_PIVOT_REGION_RADIUS_M": 0.007,
     "CHAMFER_CLEARANCE_M": 0.0001,
     "PANEL_THICKNESS_MIN_M": 0.0022,
     "PANEL_THICKNESS_MAX_M": 0.0038,
     # HIGH-4 (Round 8) extensions — three constants locked elsewhere in the spec but missing from the initial dict:
     "FILLET_RADIUS_M": 0.001,                                    # 1 mm rib-panel fillet (§N7 check 15 lower bound; drives §3.1.5 K_t_fillet = 1.5)
     "CLICK_FOOTPRINT_Y_RANGE_PANEL_EDGE_M_AT_TIP": (0.0175, 0.0225),  # 5 mm half-width tangential band at the tip (§0 row 45)
     "CLICK_DETENT_BUMP_RADIUS_M": 0.0004,                        # midpoint of locked 0.3-0.5 mm range per §3.1.5; drives K_t = 3.0
 }
 ```
 Changes whenever any of these locked constants are edited (e.g., a future re-tune of HUB_RADIUS or RIB_TIP_TAPER). Without `geometry_hash` the cache-poisoning fix (M6) doesn't catch geometry edits — stale `.done` markers from before the change would silently mark a run as already-done, training the GP on data computed against the wrong geometry.
- **`physics_hash := blake2b(config_hash || material_hash || geometry_hash || motion_blob)`** — a strict superset that mechanically depends on all three. **The dependency is enforced:** if any constituent changes, `physics_hash` MUST change. The constructor `compute_physics_hash(config_hash, material_hash, geometry_hash, motion_blob)` is the single canonical entry point.

**CI assertion (`tests/test_utils/test_hash_dependency.py`):** asserts `physics_hash` recomputed from `(config_hash, material_hash, geometry_hash, motion_blob)` matches the value stored in each JSONL row. Catches the failure mode where a config-or-material-or-geometry edit lands but the cached `physics_hash` doesn't recompute — the dedup composite-key would silently treat the run as already-done and the GP would train on stale results.

The §9.4.1 assertion above covers all four hashes: every key it asserts contributes to at least one of them. The hash-tuple `(config_hash, material_hash, geometry_hash, physics_hash)` is stored in every JSONL row. **Composite-key dedup is now a 7-tuple:** `(design_hash, physics_hash, config_hash, material_hash, geometry_hash, fidelity, run_direction)` — was 6-tuple pre-H16. **`.done` marker path** is similarly updated: `designs/{dh}/runs/{ch[:8]}-{ph[:8]}-{mh[:8]}-{gh[:8]}-{fid}-{dir}.done` (adds `gh` for geometry-hash short-prefix). **Legacy-marker migration:** the orchestrator re-scans pre-H16 markers, recomputes `geometry_hash` against the CURRENT GEOMETRY_LOCKS snapshot; markers whose geometry-hash matches the current snapshot are renamed to the new 7-tuple path; markers whose geometry-hash doesn't match are quarantined for human inspection (same pattern as the M6 material-hash migration).

Every successful run emits a metadata sidecar `designs/{hash}/run_meta_{tier}_{session}_{ts}.json` capturing: resolved SU2 config (post-template-render), Git commit of `fanopt`, SU2 binary commit + build flags, host (`platform.node` + `platform.machine`), wall-clock seconds, design hash, locked-decisions hash. This catches the silent-config-drift failure mode where a Colab kernel restart or a copied notebook poisons the GP training set with subtly inconsistent objective values — every J_fan in the GP is traceable to the exact config + binary that produced it.

#### 9.4.2 Failure-Code Taxonomy
Every failed evaluation is tagged with a canonical failure code from this finite enum. Lets post-campaign diagnostics separate "BO proposed bad points" from "infrastructure broke" (the failure-code field is also indexed by the rank-correlation diagnostics in Phase 4 step 56). NOTE: failures are NOT fed to the GP as noise — per §0 the GP uses a fixed-floor `EPISTEMIC_NOISE_FLOOR` scalar; failure codes feed diagnostic dashboards only.

| `failure_code` | Meaning |
|----------------|---------|
| `mesh_fail` | Gmsh raised; geometry was invalid for meshing (e.g., disconnected wires in a 2D slice — see Tier-1 guard). |
| `solver_diverged` | SU2 history shows residuals not converging within `ITER` budget. |
| `config_mismatch` | The §9.4.1 config-hash assertion fired; run aborted before solve. |
| `oom` | Out-of-memory; Colab kernel crashed. Auto-rescheduled to high-RAM instance. |
| `timeout_wall` | Wall-clock budget exceeded; run killed externally. |
| `timeout_iter` | SU2 hit `ITER` (steady) or `TIME_ITER` (unsteady) without reaching `CONV_RESIDUAL_MINVAL`. |
| `nan_result` | J_fan or any downstream metric came back NaN / Inf. |
| `rejected_hard_constraint` | §6.3.1 Filter 1: m_total / r_CoM_wrist / manufacturability bound violated. |
| `rejected_pre_cfd_struct` | §6.3.1 Filter 2: closed-form rib stress or tip deflection bound violated. |
| `manufacturability_fail` | §9.7.3 score < 0.5 (Critical-failure → score 0 paths). |
| `unknown` | Catch-all; triggers post-mortem inspection. |

**retriable column :**

| `failure_code` | `retriable` | Retry limit | Notes |
|----------------|-------------|-------------|-------|
| `oom` | YES | 2 | Re-dispatch to a high-RAM CPU instance on first retry; second retry → high-RAM with reduced mesh refinement. |
| `timeout_wall` | YES (first only) | 1 | Re-dispatch with a checkpoint resume + extended wall-clock; subsequent timeouts → non-retriable. |
| `config_mismatch` | YES (after auto-correct) | 1 | The §9.4.1 assertion fired; auto-correct script re-renders the config from the locked-decisions snapshot and re-dispatches once. |
| `unknown` | YES | 1 | Re-dispatch once; if it fails again with another `unknown`, treat as non-retriable. |
| `mesh_fail` | NO | — | Geometry was invalid for meshing. Re-dispatching produces the same failure. |
| `solver_diverged` | NO | — | The SU2 residual history showed real divergence; not a transient failure. |
| `nan_result` | NO | — | Numerical instability in `j_fan.py`; not a transient infrastructure failure. |
| `rejected_hard_constraint` | NO | — | §6.3.1 Filter 1 deterministic rejection; re-dispatching changes nothing. |
| `rejected_pre_cfd_struct` | NO | — | §6.3.1 Filter 2 deterministic rejection. |
| **`cfl_excursion`** | YES | **1** (per `failure_codes_config["cfl_excursion"]["max_retries"]` in `src/fanopt/cfd/failure_codes.py`) | `cfl_max ≥ 10` from `j_fan.py` post-processing. **First failure → retry once with `CFL_NUMBER` halved** (param_modifier: `halve_cfl` — e.g., 10 → 5 in the SU2 config); record the achieved `cfl_max` in `retry_history` and increment `retry_count`. **Second failure (after halved CFL) → demote to `solver_diverged` (non-retriable, hard reject).** The CFL = 10 threshold is **conservative for SU2 dual-time-stepping with inner iterations** (SU2 docs note CFL > 10 is acceptable when inner residuals converge in ≤ ITER_INNER, but CFL > 10 frequently indicates the time-step is too large for the local time-scale; halving on first excursion costs ~2× wallclock vs the alternative of silently retraining on a non-converged solution). Failure-code config canonical location: `failure_codes_config = {"cfl_excursion": {"max_retries": 1, "param_modifier": "halve_cfl"}}` in `src/fanopt/cfd/failure_codes.py`; CI test asserts the config is the single source of truth. |
| `manufacturability_fail` | NO | — | §9.7.3 score < 0.5; geometry-driven, not transient. |
| `timeout_iter` | NO | — | The solver hit ITER without converging; mesh / numerics issue, not transient. |
| `fea_combined_tip_defl` | NO | — | §59.5 gate failure; design is structurally underbuilt. |
| `fea_combined_peak_stress` | NO | — | §59.5 gate failure; design exceeds the §10.1 per-mode allowable (canonical: fillet 9.00 / pivot tension 5.58 / pivot bearing 2.00 MPa cyclic) or the static SF=1.5 allowable (stress-test: fillet 12.4 / pivot tension 12.4 / pivot bearing 13.3 MPa). |
| `fea_combined_torsion` | NO | — | §59.5 gate failure; first torsional mode < 10 Hz or torsional buckling detected. |
| **`transient_infra` ** | YES | 1 | Drive 503, Colab disconnect mid-eval, Gmsh OOM that succeeds on retry — once-only transient infrastructure failures. Single retry with a clean working directory; if it fails again with another `transient_infra`, demote to `unknown`. Records `retry_history` in the JSONL row for pattern mining (if the same hash hits `transient_infra` 3 sessions in a row across the campaign, it's not transient and should be flagged for inspection). |

**Beyond the retry limit:** the failure is fed to the GP as a **high-noise NaN observation** with `J_fan = nan` + `J_fan_se = inf`. The GP knows the design was seen but doesn't trust the value — this prevents the same hash from being re-proposed (`design_hash` dedup) without poisoning the GP training set with infrastructure noise.

Stored as JSONL fields (`status`, `failure_code`, `detail`) plus a `designs/{hash}/.failed_{code}` marker for cheap glob-level filtering.


#### 9.4.1 Canonical Steady-State Configuration

The project uses compressible Navier-Stokes with low-Mach preconditioning at Mach ~0.008 (sidesteps issue #193) and computes J_fan via the canonical `j_fan.py` post-processor against the locked spec below. The SU2 config does not set an `OBJECTIVE_FUNCTION` -- the metric is post-processed from the volume/surface output.

```ini
% SU2 config: compressible + low-Mach preconditioning, steady
SOLVER= NAVIER_STOKES
KIND_TURB_MODEL= NONE
%
MACH_NUMBER= 0.0064 % V_tip = 2.20 m/s at wrist-to-tip L = 0.25 m, Mach = 2.20/343
FREESTREAM_DIRECTION= 0.0 0.0 -1.0 % PRODUCTIVE stroke (C2 lock): the fan sweeps in +z toward the user, so in stationary-fan CFD the air flows in -z relative to the (stationary) fan. SU2's AOA directive is intentionally NOT used here because its interpretation is version- and build-dependent (rotates between body x-axis and freestream rather than between the mesh's streamwise direction); an explicit vector is unambiguous. For the RETURN stroke run, set FREESTREAM_DIRECTION= 0.0 0.0 +1.0. The §9.4.1 config-hash assertion includes both `FREESTREAM_PRODUCTIVE` and `FREESTREAM_RETURN` as cross-tier constants.
FREESTREAM_TEMPERATURE= 300.0
FREESTREAM_PRESSURE= 101325.0
REYNOLDS_NUMBER= 37000.0 % Re = ρVL/μ = 1.225·2.20·0.25/1.8e-5 ≈ 37k (wrist-to-tip = 0.25 m).
REYNOLDS_LENGTH= 0.25
%
LOW_MACH_PREC= YES
MIN_ROE_TURKEL_PREC= 0.01
MAX_ROE_TURKEL_PREC= 0.2
%
MESH_FILENAME= fan_deployed.su2
MARKER_HEATFLUX= ( fan_surface, 0.0 )
MARKER_FAR= ( farfield )
%
% NO OBJECTIVE_FUNCTION line: J_fan is computed by j_fan.py from the volume/surface output
% per the locked spec below.
MARKER_ANALYZE= ( downstream_plane ) % the 600x600 mm plane at 300 mm forward
MARKER_MONITORING= ( fan_surface )
%
NUM_METHOD_GRAD= GREEN_GAUSS
CFL_NUMBER= 10.0
ITER= 5000
CONV_RESIDUAL_MINVAL= -8
%
OUTPUT_FILES= RESTART, PARAVIEW
TABULAR_OUTPUT= CSV
HISTORY_OUTPUT= ITER, RMS_RES, AERO_COEFF
```

Unsteady variant: see "Unsteady Configuration (Advanced)" below for the pitching-motion configuration used for Phase 3/4 unsteady runs.

#### Objective Function for a Fan: Locked J_fan Spec

**Critical note:** `DRAG` is the wrong objective. A fan's purpose is to push air. `SURFACE_TOTAL_PRESSURE` is also wrong (total pressure is conserved along streamlines in inviscid flow; its surface integral does not represent net momentum delivered to the user). The project locks a custom directed-momentum-flux integral as the single canonical metric.

**Locked J_fan specification (user-approved):**

```
J_fan = (1/T) ∫_{cycle 2 to 5} [ ∫∫_Σ ρ_0 · u_n(x, t) · (u(x, t) · t̂) dA ] dt

where:
 Σ = analysis plane, 600 mm × 600 mm, centered on the pivot,
 located 300 mm forward of the fan pivot along +z, normal = +t̂
 t̂ = +ẑ (the user-ward direction; positive J_fan = air moving in +z
 toward the user. The t̂ definition is decoupled from the FREESTREAM
 sign convention by construction: t̂ is a geometric direction in the
 world frame fixed at Phase 4 launch, while FREESTREAM is a per-eval
 cfg directive that is +z or -z depending on which stroke is being
 simulated. The earlier draft tied t̂ to "FREESTREAM_DIRECTION=(0,0,1)
 forward proxy"; under the C2 sign correction the productive stroke
 freestream is (0,0,-1), so coupling t̂ to FREESTREAM_DIRECTION sign
 created a stale cross-reference. NOT "from pivot toward user" —
 that anatomical phrasing is ambiguous because the user's hand is
 at -x (the wrist axis) but the user's torso is at +z; the +ẑ lock
 removes the ambiguity.)
 n̂ = plane normal = +t̂ (so u_n = u · t̂ for this plane orientation)
 u_n = u(x, t) · n̂ (velocity component normal to plane)
 ρ_0 = 1.225 kg/m³ (constant; flow Mach ~0.008, compressibility <0.01%)
 T = waving period = 0.5 s at 2 Hz
 Δt = T/200 = 2.5 ms (time-step independence verified in Phase 3)
 Cycles = 5 total, integrate cycles 2-5 only (discard cycle 1 transient)

Reported units: J_fan in N (effective thrust toward user)
Sign convention: positive J_fan = air moving toward user
```

**Secondary metric** (reported alongside J_fan as a second Pareto objective):

```
J_fan_peak = (1/T) ∫_{cycle 2 to 5} max(J(t), 0) dt
```

Captures "feel-able peak airflow" for designs the user might subjectively prefer even when net J_fan is slightly lower.

**Why this spec (design choice rationale):**

| Choice | Spec | Rationale (memo §4.5) |
|--------|------|----------------------|
| Plane geometry | Single plane perpendicular to user direction | Matches how a person feels a fan; easiest to mesh, compare, interpret |
| Plane distance | 300 mm | Typical arm-to-face distance; matches Phase 6 anemometer protocol |
| Plane size | Fixed 600 × 600 mm | Big enough to avoid clipping any geometry; reproducible |
| Integrand | Directed momentum flux ρ · u_n · (u · t̂) | "Push toward user" physical meaning |
| Density | Constant ρ = 1.225 kg/m³ | Mach <0.01; same across SU2/PyFR for direct comparison |
| Cycle reduction | Full-cycle net (not RMS, not positive-only) | Captures the asymmetric-drag mechanism that actually makes hand fans work; symmetric designs correctly score ~0 |
| Cycles | 5, discard 1 | Standard for periodic flows; verified empirically in Phase 3 |
| Time step | T/200 | Captures vortex shedding from rib ridges (V-unit corrugated geometry) without compute explosion |

**Implementation:** A single canonical post-processor (`src/fanopt/cfd/j_fan.py`) implements both the unsteady time-integrated metric above and the steady-state proxy defined immediately below. Every solver in the stack (SU2 2D steady, SU2 2D unsteady, SU2 3D steady, SU2 3D unsteady, PyFR) routes through `j_fan.py`, which auto-detects whether the input contains a time dimension and applies the appropriate formula. This guarantees apples-to-apples comparison across all phases and fidelities up to the steady-vs-unsteady correlation modeled by the multi-fidelity GP. QSST and FSI are no longer used; see §0 for the rationale.

#### Steady-State Proxy (Tier -1 and Tier 0; addition)

For steady CFD outputs (no time dimension) — Tier -1 2D steady and Tier 0 3D steady — `j_fan.py` cannot compute the time-integrated metric above because there is no `t` to integrate over and no cycles to discard. Instead, it computes the **axial surface force on the fan projected onto the thrust direction**:

```
J_fan_steady_proxy = ∫∫_{S_fan} [ p(x) · n̂_surface(x) + τ(x) ] · t̂ dA

where:
 S_fan = fan surface (all blades, all panels, all ribs)
 n̂_surface = outward unit normal at each surface element
 t̂ = thrust direction (unit vector from pivot toward the user)
 p = static pressure
 τ = viscous shear-stress vector at the surface
 Mach = 0.0064 (set by the prescribed body velocity = peak waving tip speed 2.20 m/s at wrist-to-tip L = 0.25 m)
 Sign = positive ⇒ thrust toward user (consistent with the unsteady spec's convention)
```

This is just the standard surface-force integration SU2 already provides via `MARKER_MONITORING`; `j_fan.py` wraps it with a sign convention + projection onto t̂. The integrand has natural physical meaning: it is **the thrust the fan would produce if held stationary in equivalent flow**.

**two-eval delta protocol (mandatory for Tier -1 and Tier 0):** the one-direction proxy above rewards the *magnitude* of pressure drag, which a solid-wall design maxes out. A louvered / TPMS-perforated design with great *asymmetric* drag — the actual physics that drives a hand fan — has lower one-direction drag because the leakage reduces the total force. The proxy converges on the parachute. Fix this by running **two CFD evals per design** with explicit `FREESTREAM_DIRECTION`: **PRODUCTIVE = `(0, 0, -1)`** (air flowing -z relative to a stationary fan that's actually being swept in +z toward the user) and **RETURN = `(0, 0, +1)`** (the return stroke). (SU2's `AOA` directive is intentionally not used — its interpretation is version-dependent and silently couples to the body's reference frame; an explicit direction vector is unambiguous.) The reported proxy is the **delta**:

```
J_fan_steady_proxy = (J_proxy at FREESTREAM_PRODUCTIVE = (0, 0, -1))
                   − (J_proxy at FREESTREAM_RETURN    = (0, 0, +1))
                   = Drag_productive − Drag_return
```

Symmetric designs (solid panel, symmetric Fourier outline) score ≈ 0 because productive and return drags are equal. Asymmetric designs (louvers angled to catch air on the productive stroke and feather on the return, TPMS with directional flow channels biased toward the user-ward stroke, noise-threshold cutouts that bias the flow) score **positive** in the productive direction and **negative** in the return direction. The delta tracks unsteady J_fan much better than either one-direction value: expected Phase 3 R² jumps from ~0.4-0.5 (one-direction) to ~0.7-0.85 (delta).

**Sign-discriminator test (`tests/test_cfd/test_steady_proxy_sign.py`):** generate a 45° slatted louver oriented to scoop air on the productive (+z body sweep) stroke; run both steady evals; assert `J_fan_steady_proxy > 0`. The test fails under the earlier (inverted) convention where productive louvers scored negative.

`j_fan.py` auto-detection: when two steady runs with opposite `FREESTREAM_DIRECTION` vectors exist for the same `design_hash` (one with `(0, 0, -1)` = PRODUCTIVE and one with `(0, 0, +1)` = RETURN in the run_meta sidecar), the post-processor returns the delta. If only one is present, it returns the one-direction proxy with a `proxy_kind = "one_direction"` flag and a `WARNING` log entry — that path exists for legacy / debug use only; production Tier -1 + Tier 0 always emit both directions.

SU2 configs `slice_steady.cfg.j2` and `fan3d_steady.cfg.j2` are rendered twice per design (one Jinja2 var `stroke ∈ {productive, return}`; was `aoa_deg ∈ {0.0, 180.0}` in the inverted draft — renamed per C2 to make the productive/return mapping unambiguous in the template); the orchestrator dispatches both evals as a pair and waits for both before computing the delta.

Cost impact: Tier -1 doubles (~5 → ~10 min/eval); Tier 0 doubles (~30-90 → ~60-180 min/eval). Net Phase 4 budget impact is roughly flat because the R² boost lets 's data-driven K drop to 3, saving ~150 Tier-1 hours that net out the ~100-200 added at Tier-0. The promotion weighting (50/50 Tier -1 / Tier 0) is queued for post-Phase-3 re-tune (see `docs/phase_logs/post_phase3_decisions.md`); with the better proxy, Tier -1 may be a reliable standalone signal and the Tier-0 pre-promotion evals may be skippable.

**The steady proxy is NOT equal to the unsteady J_fan.** It misses vortex shedding, asymmetric drag from direction reversal, and the starting-vortex / wake-reversal mechanisms that actually power a hand fan. But it correlates with the unsteady J_fan — and the multi-fidelity GP (§6.2.3) is specifically designed to learn this correlation via the `SingleTaskMultiFidelityGP` fidelity column. That is exactly what makes the steady proxy useful as Tier -1 / Tier 0 screening despite its physical incompleteness.

**Why surface-force rather than steady plane-integral (∫∫_Σ ρ · u_n · (u · t̂) dA without time integration):**
1. SU2 outputs surface forces directly per iteration via `MARKER_MONITORING` — no post-processing required.
2. The surface-force formulation matches what SU2 produces natively, so the `j_fan.py` wrapper for steady inputs is just a sign convention + projection onto t̂.
3. The plane-integral approach without time integration would require selecting an arbitrary instant of the steady solution (any instant gives the same result, but the convention is unclear in unsteady context).

**Auto-detection in `j_fan.py`:** the post-processor inspects the SU2 output for time-dimension keys (e.g., `TimeIter`, multiple history rows per design). Absent ⇒ steady proxy via surface-force integration. Present ⇒ time-integrated metric per the §9.4 main spec.

The unsteady Tier 1 path still uses the full §9.4 time-integrated metric; nothing changes for unsteady CFD.

**Legacy alternatives (do not use for primary optimization):**

- **`SURFACE_TOTAL_PRESSURE`** — rough proxy, retained only for early-development sanity checks.
- **Multi-objective LIFT + DRAG** — works with SU2 built-ins but is a poor approximation of the full integral.

#### Unsteady Configuration (Advanced)

For unsteady simulation with pitching motion, use the compressible solver with low-Mach preconditioning (the incompressible solver has known issues with pitching motion):

```ini
SOLVER= NAVIER_STOKES
KIND_TURB_MODEL= NONE
%
MACH_NUMBER= 1e-9 % HIGH-12 Round-9 lock: NEAR-ZERO Mach. Unsteady physics is "moving body in still air" (NOT moving body in V_tip-magnitude tailwind). MACH × c_ref = 1e-9 × 343 = ~3e-7 m/s, well below FREESTREAM_VELOCITY's 1 mm/s.
FREESTREAM_OPTION= FREESTREAM_VELOCITY % HIGH-12 Round-9 lock: explicit override — solver uses FREESTREAM_VELOCITY vector, NOT MACH-based freestream. Resolves the SU2-default-FREESTREAM_OPTION=TEMPERATURE_FS ambiguity that would otherwise have computed freestream as MACH × c_ref = 2.20 m/s = V_tip, producing zero net body-vs-ambient relative velocity. Fallback if SU2 build rejects this directive: MACH_NUMBER = 1e-9 + REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE (Spike 0.6c.1 verifies syntax — see line 1821).
FREESTREAM_VELOCITY= 0.0 0.0 0.001 % 1 mm/s in +z — kept numerically nonzero for Riemann far-field stability; ≪ V_tip = 2.20 m/s by 2000×. Body-vs-ambient frame separation handled by GRID_MOVEMENT = RIGID_MOTION block below.
% FREESTREAM_DIRECTION is NOT specified — under FREESTREAM_OPTION = FREESTREAM_VELOCITY override, direction comes from the velocity vector. Setting both causes SU2-version-dependent silent-axis-coupling. The earlier draft's `FREESTREAM_DIRECTION = 0.0 0.0 1.0` is retired (it conflicted with the steady cfg's −1.0 productive-stroke sign convention AND introduced ambiguity vs FREESTREAM_VELOCITY).
% Sign convention (C2 lock): body sweeps in +z during productive stroke per the RIGID_MOTION block below. Ambient freestream is in +z (1 mm/s, near-zero).
FREESTREAM_TEMPERATURE= 300.0
FREESTREAM_PRESSURE= 101325.0
REYNOLDS_NUMBER= 37000.0 % see steady config above
REYNOLDS_LENGTH= 0.25
%
LOW_MACH_PREC= YES
MIN_ROE_TURKEL_PREC= 0.01
MAX_ROE_TURKEL_PREC= 0.2
%
TIME_DOMAIN= YES
TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER
TIME_STEP= 0.0025 % T/200 at f=2 Hz (T=0.5 s)
MAX_TIME= 2.5 % 5 cycles × 0.5 s/cycle at 2 Hz
TIME_ITER= 1000 % 5 × 200 = 1000 inner-time steps
INNER_ITER= 100
%
GRID_MOVEMENT= RIGID_MOTION
MOTION_ORIGIN= 0.0 0.0 0.0 % Coord convention (LOCKED, no alternates): SU2 mesh places the WRIST axis at world origin (0,0,0); the pivot pin is at (d_handle, 0, 0) = (0.05, 0, 0); blades extend from x = 0.05 to x = 0.25. MOTION_ORIGIN = (0,0,0) rotates the fan about the WRIST axis (matches V_tip = 2.20 m/s, Re = 37000, k = 0.57, and the §9.4.1 config-hash assertion). The Gmsh meshing scripts (§9.6 / §Phase 4 step 46) MUST write the mesh with the wrist at world origin — no pivot-at-origin alternative; the assertion + the §3.2.0 axis-convention CI test fail-close on any drift.
PITCHING_OMEGA= 0.0 -12.5664 0.0 % C11 SIGN LOCK: NEGATIVE y-component. Wrist rotation about +y axis (wrist-flexion hinge perpendicular to the +x forearm direction); magnitude 2·π·f = 12.5664 rad/s. The NEGATIVE sign is part of the locked convention (NOT a typo): per §0 row 26 right-hand-rule, `ω_blade_max = (0, -8.8, 0)` at the productive-stroke instant produces v_tip in +z (toward user). Emitting (0, +12.5664, 0) would run the simulation with the swing direction inverted (tip moves in -z, the return stroke, not productive). The Jinja2 template hard-codes the negative sign; do NOT replace with `abs(PITCHING_OMEGA_MAG_RADS)` or render from the unsigned magnitude in `material_locks`.
PITCHING_AMPL= 0.0 0.6981 0.0 % 40° = 0.6981 rad amplitude about +y (wrist axis). SU2 pairs each axis: θ_i(t) = AMPL_i·sin(OMEGA_i·t), so AMPL and OMEGA must share the same nonzero axis or the mesh does not rotate. **CI test (physical motion, not axis identity):** load rendered .cfg, run SU2 for 1 cycle on a probe mesh. Two assertions sampled at DIFFERENT times because SHM has v=0 when |θ|=θ_max and v=peak when θ=0: (a) at **t = T/4** (peak displacement), tip z-displacement ≈ r_perp · sin(θ_max) = 0.25 · sin(40°) ≈ 0.16 m (and tip velocity magnitude ≈ 0 — verifies the sinusoid is at its turning point); (b) at **t = 0** (zero displacement, peak velocity), tip velocity vector is along -z with magnitude ≈ ω_blade_max · r_tip = 8.8 · 0.25 ≈ 2.20 m/s (verifies the SHM derivative + correct axis); (c) the pivot point stays fixed at (0.05, 0, 0). Sampling velocity at t=T/4 (as the earlier draft did) yields v=0 by construction and the test passes any axis direction within tolerance — the t=0 sample is the discriminative one. **AMPL unit lock:** SU2 builds vary on whether `PITCHING_AMPL` is radians or degrees; the campaign pins the SU2 build commit in `material_locks.SU2_COMMIT` and adds a standalone SU2 amplitude-assertion test (a steady run with θ_max applied as a fixed rotation; assert the resulting tip displacement matches sin(40°) · r_tip within 1%). If the deployed SU2 build interprets AMPL as degrees, the assertion fires and the campaign halts before the GP poisons.
```

**Unsteady-config locked numerics (must match the §9.4 J_fan spec):** f = 2 Hz waving frequency, T = 0.5 s cycle period, dt = T/200 = 2.5 ms, 5 cycles, MAX_TIME = 2.5 s, TIME_ITER = 1000 outer steps, PITCHING_OMEGA = 2π·f = 12.5664 rad/s. The earlier draft (TIME_STEP = 0.001, MAX_TIME = 1.0, TIME_ITER = 1100, PITCHING_OMEGA = 6.2832) implemented 1 Hz with dt = T/500 — that is not the J_fan canonical spec and would have produced the wrong CFD. If you copy this config block into Phase 4/5 runs, use the values above, not earlier historical numbers.

**Body-motion vs. ambient-freestream separation (Tier 1 true-unsteady):** Tier 1 models a moving body in still air (real hand-fan physics) — NOT a moving body in a V_tip-scale freestream. Per the HIGH-12 Round-9 lock, the unsteady cfg uses `MACH_NUMBER = 1e-9` (near-zero) together with `FREESTREAM_OPTION = FREESTREAM_VELOCITY` (explicit override). The override forces SU2 to take the ambient velocity directly from the `FREESTREAM_VELOCITY = (0.0, 0.0, 0.001)` vector (1 mm/s in +z; numerically nonzero to keep the Riemann far-field well-conditioned but ≪ V_tip = 2.20 m/s by 2000×) rather than reconstructing it from `MACH × c_ref` under SU2's default `FREESTREAM_OPTION = TEMPERATURE_FS`. Without the override, the default path would compute freestream as `MACH × c_ref` — e.g., `0.0064 × 343 ≈ 2.20 m/s = V_tip` under the earlier draft's `MACH = 0.0064` — producing zero net body-vs-ambient relative velocity and physically nonsensical CFD. The earlier-draft phrasing "KEEPS MACH_NUMBER = 0.0064 for SU2's nondimensionalization" is retired by HIGH-12 (see `docs/retired_phrases.yaml`); thermo and viscosity nondimensionalization under the override path is anchored by `FREESTREAM_TEMPERATURE` and `FREESTREAM_PRESSURE`, not by a V_tip-scale Mach. The `GRID_MOVEMENT = RIGID_MOTION` block provides the body's pitching motion; the measurement plane stays fixed in the world frame at z = 0.300 m. **CI gate:** `tests/test_cfd/test_unsteady_freestream_consistency.py` asserts the four HIGH-12 invariants on the rendered .cfg (FREESTREAM_OPTION override or REF_DIMENSIONALIZATION fallback; MACH ≤ 1e-6; |FREESTREAM_VELOCITY| < 0.01 · V_tip; FREESTREAM_DIRECTION unset or consistent).

**Tier 1 cfg sanity-check + fallback syntax (H10 / cleanup lock):** before any Tier-1-numerics quantitative-sanity work runs, **Spike 0.6c.1** must verify that the rendered cfg parses + runs for 1 inner iteration. If the deployed SU2 build does NOT accept `FREESTREAM_VELOCITY = (0, 0, 0.001)` as a compressible-solver directive (the directive's availability is version-dependent), replace with the locked alternative: `MACH_NUMBER = 1e-9` + explicit `REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE` + an explicit reference state. The Spike 0.6c.1 output documents which of the two syntaxes the build accepts; the working syntax becomes the locked Tier 1 reference cfg in §9.4.1 (and the Phase-0 quantitative-sanity counter-checks — Spike 0.6d.1 dimensional / 0.6d.2 added-mass / 0.6d.3 incompressible cross-mode — use that working syntax to test the numerics). **Spike 0.6c.2 published-benchmark validation, originally referenced in this lock as the absolute-accuracy gate, is DEFERRED to Phase 5 step 62.5** per the 2026-05-14 regime diagnostic (full evidence trail in `docs/phase_logs/spike_0_6c.md` Note 1); the body-in-still-air published-reference benchmark at step 62.5 (SU2 + PyFR + OpenFOAM 3-solver) absorbs that absolute-accuracy work in a regime-appropriate frame.

The **steady-proxy two-eval delta (Tier -1 + Tier 0)** is a DIFFERENT mode: stationary geometry in a ±z freestream of magnitude V_tip — those configs keep the full freestream and have no `GRID_MOVEMENT`. Both modes are internally consistent; mixing them (both freestream AND grid motion at V_tip-equivalent magnitudes) would double-count the relative velocity. Document the distinction in `configs/su2/README.md`.

**Steady-state proxy limitations:** The reduced frequency **k ≈ 0.57** places this firmly in the unsteady regime. At this k, added mass effects, dynamic stall, and leading-edge vortex formation significantly alter force histories compared to quasi-steady predictions. Published studies on oscillating flat plates show quasi-steady models can overpredict peak forces by 30-50% and, critically, change the **relative ranking** of different geometries. **Do not run the entire BO budget on steady-state alone.** Use the multi-fidelity BO approach described in Section 6.2.3, which mixes cheap steady-state evaluations with expensive unsteady ones via `SingleTaskMultiFidelityGP`. This lets the GP learn the steady-to-unsteady correlation and allocate budget intelligently. As a minimum, validate the final top-3 designs through unsteady CFD before committing to printing.

### 9.5 BoTorch -- Bayesian Optimization

#### Installation

```bash
pip install botorch gpytorch torch scipy matplotlib
```

Claude writes the complete multi-fidelity BO loop as described in §6.2.3. Key parameters (Phase 4):

- `num_restarts`: 20 (for acquisition function optimization)
- `raw_samples`: 512 (initial candidates for acquisition optimization)
- **Initial samples per promoted architecture:** ~30 2D-steady CFD (Tier -1) + ~30 3D steady CFD (Tier 0) + ~5 3D unsteady CFD (Tier 1). Across all 40-120 architectures, Tier -1 screening = ~3600 evals; only **K_promoted ∈ {3, 4, 5} (data-driven; default K = 4 when Phase 3 R² is in [0.5, 0.6))** architectures promoted to Tier 0/1.
- BO iterations: **~35 acquisition rounds per promoted architecture (hard cap; was 50; early-stop fires first when UCB-improvement < 3% over 5 iters)**, allocated across the three tiers by `qMultiFidelityKnowledgeGradient`.
- Compute target: **~300-600 hours favorable / 500-800 expected on Colab Pro** (see §6.2.3 "Honest compute budget" for the breakdown; pessimistic case can reach ~900 hours; the 300-600 number is the favorable case with data-driven K and reduced acquisition cap).
- Surrogate: `SingleTaskMultiFidelityGP` with fidelity column over {-1, 0, 1}; wrapped in TuRBO trust regions for the ~20-30-D continuous-only design space per architecture (categoricals handled by the §6.2.2 outer-loop bandit); `SaasFullyBayesianSingleTaskGP` (≤500 inducing points) as the SAASBO alternative.
- Acquisition: `qMultiFidelityKnowledgeGradient` with three-tier cost model .
- Discrete handling: outer-loop architecture bandit over blade count ∈ {8, 10, 12} (MED-10 trim: 14 removed for ergonomic infeasibility — 186.2° past straight-line), edge profile, print orientation, layer height, Layer 2 field activations + per-field categoricals, Layer 3 primitive presence + type.
- Multi-objective: `qNoisyExpectedHypervolumeImprovement` over **4 objectives** (J_fan, **I_wrist** [about handle-grip wrist axis], peak_pivot_stress, folded_form_factor). m_total < 0.100 kg and r_CoM_wrist ≤ 0.160 m (= d_handle 0.05 m + 0.55·L_blade 0.20 m) are §6.4 hard constraints, not Pareto objectives.
- Surrogate validation: NRMSE < 15% on leave-one-out cross-validation at target fidelity (3D unsteady).

### 9.6 Gmsh -- Meshing for CFD

#### Installation

```bash
pip install gmsh
```

Claude writes Gmsh Python scripts for meshing the deployed fan geometry. Key challenges for a folding fan:

- **Sector geometry:** The deployed fan is a sector of a circle, not a rectangle.
- **Curved surface:** If ribs are cambered, the fan surface is doubly curved.
- **Boundary layer mesh:** Need structured layers near the fan surface (first cell height ~0.2 mm, growth ratio 1.15).
- **Downstream analysis plane:** Define as an internal boundary at 1.5x fan radius downstream.

**Locked boundary-condition marker names (must match §9.4 SU2 configs).** Gmsh writes physical-group names exactly as below; SU2's `MARKER_*` directives reference these strings verbatim. A mismatch produces silent BC dropouts that Claude Code would otherwise have to debug from solver-divergence symptoms.

| Marker | Gmsh physical-group name | Used by SU2 directive | Geometry |
|--------|--------------------------|-----------------------|----------|
| Fan body | `fan_surface` | `MARKER_HEATFLUX`, `MARKER_MONITORING` | Union of all rib + panel surfaces across the 10 blades |
| Far-field | `farfield` | `MARKER_FAR` | Outer box (3D) or outer arc (2D slice) — Riemann far-field |
| Downstream plane | `downstream_plane` | `MARKER_ANALYZE` | Internal flat plane **0.300 m forward of the pivot along +z** (centered on the pivot at (0.05, 0, 0.300) under the locked convention; the wrist is at (0,0,0), pivot at (0.05, 0, 0), so the plane centerline is at x=0.05 — note the pivot-vs-wrist distinction is geometrically tolerable for a 600×600 mm plane but the spec lock is **pivot-centered**), sized 0.600 × 0.600 m, matches the locked §9.4 J_fan plane |
| **Cascade slip wall (— 2D slice only)** | `cascade_wall` | `MARKER_SYM` | Top + bottom boundaries of the unwrapped linear cascade in the 2D slice. **Forces freestream mass flow through the inter-blade gaps** instead of routing around the cascade's outer edges (which would leave the middle 4-6 blades in a stagnation zone with near-zero mass flow and falsely penalize Layer 2 porosity placed in those blades). Companion regression test `tests/test_cfd/test_cascade_mass_flow.py`: assert mass flow at midspan ≈ mass flow at top/bottom inlets within 10%. Only used in the 2D slice (`slice_steady.cfg.j2` + `slice_unsteady.cfg.j2`); the 3D config doesn't need it (the deployed fan has actual open sides). |

The Gmsh meshing scripts in `src/fanopt/cfd/mesh_*.py` write these names via `gmsh.model.addPhysicalGroup(..., name=...)`. The §9.4.1 config-hash assertion includes a `MARKER_NAMES` cross-tier check that asserts all three names appear in the SU2 config file unaltered.

Budget 1-2 weeks for mesh generation even with Claude writing the scripts, because mesh quality tuning requires iteration.

### 9.7 Generative Blade Geometry Generator

The CadQuery generator is the centerpiece of the panel optimization pipeline. It implements the 4-layer hybrid parameterization (§6.2.1) and outputs a single PETG STL per blade plus a manufacturability score in [0, 1] consumed by the BO loop. Under the §12 package layout, the generator is split across `src/fanopt/geometry/` modules — `envelope.py` (Layer 1), `fields.py` (Layer 2), `primitives.py` (Layer 3), `manufacturability.py` (§N7 filter), `generator.py` (orchestration via `generate_blade(params)`), and `schema.py` (JSON validation). The pseudocode below shows the orchestration logic that lives in `generator.py`.

#### 9.7.1 Generator Orchestration (`src/fanopt/geometry/generator.py`)

```
generate_blade(params: dict) -> (stl: Optional[str], mfg_score: float)
 ├── Step 0 :
 │ ribs = load_to_optimized_ribs(params.rib_dxf) # Phase 2 density_to_rib.py output
 │ # plano-convex constraint: when print_orientation == 'rib-flat',
 │ # make_outer_envelope applies camber + thickness variation to the TOP face only;
 │ # bottom face is a flat planar extrusion of the planform. For 'deployed-V'
 │ # orientation, both faces may curve. The envelope generator reads
 │ # params.layer4.print_orientation to switch construction modes:
 │ # rib-flat → extrude flat base, loft top-face spline over 3 camber knots
 │ # deployed-V → loft both faces (symmetric or asymmetric camber allowed)
 │ full_envelope = make_outer_envelope(params.layer1, params.layer4.print_orientation)
 │ # Build a 3D mask of the rib region by extruding the rib top-face wires
 │ # through ENOUGH Z height to fully envelop the envelope's Z extent (not just
 │ # panel_thickness_max). With max camber (5 mm) + max thickness (5 mm) the
 │ # envelope's Z-extent can reach ~10 mm; a 5 mm-tall rib mask would leave
 │ # envelope material above/below the rib planform region, and Layer 2/3 ops
 │ # would carve unintended geometry there.
 │ # CONVENTION : the blade is constructed with Z = panel-thickness
 │ # direction (perpendicular to the rib+panel planform). ribs.faces(">Z").wires
 │ # returns the rib's planform outline; for any other construction axis swap to
 │ # the corresponding face selector AND swap the bounding-box span axis below.
 │ env_bbox = full_envelope.val.BoundingBox
 │ z_span = env_bbox.zlen * 2.0 # 2× the envelope Z-extent
 │ rib_top_wires = ribs.faces(">Z").wires
 │ rib_mask_3d = (cq.Workplane.add(rib_top_wires)
 │ .extrude(z_span, both=True)) # ±z; spans envelope fully
 │ panel_domain = full_envelope.cut(rib_mask_3d) # panel region only;
 │ # ribs excluded from the
 │ # Layer 2/3 subtraction domain
 │ # Regression test: tests/test_geometry/test_rib_mask_zheight.py constructs an
 │ # envelope with max camber + max thickness and asserts (panel_domain ∩ rib_xy_region)
 │ # has zero volume across the full envelope Z-extent.
 │
 ├── Step 1: panel_solid = make_panel_solid(panel_domain, params.layer1.thickness, params.layer4.click)
 │ └── camber spline + twist + thickness profile (2.2-3.8 mm) +
 │ edge profile categorical + Fourier LE/TE modulation, ALL CONSTRAINED TO panel_domain
 │ → always produces valid geometry (smooth parametric + Fourier; no CAD edge cases)
 │ → **dual rib-band lock (C7 HUB_RADIUS + Architectural A RIB_TIP_TAPER):** the generator's
 │ rib-generation step (load_to_optimized_ribs at Step 0) emits rib material ONLY for
 │ `x ∈ [HUB_RADIUS, L_blade − RIB_TIP_TAPER] = [0.020, 0.185 m]`. The inner 20 mm
 │ (x ∈ [0, 0.020], the HUB / boss region) AND the outer 15 mm (x ∈ [0.185, 0.200],
 │ the click region) are panel-only — no rib material on either side of the panel — so
 │ the boss is unobstructed by ribs AND the panel's outer tangential edge is fully
 │ exposed for the click chamfer. Total rib radial extent: 165 mm.
 │ → **click chamfer + detent on panel outer tangential edge (item #3 lock — added to make_panel_solid):**
 │ at `CLICK_FOOTPRINT_X_RANGE = (L_blade − 0.010, L_blade)` × `CLICK_FOOTPRINT_Y_RANGE_PANEL_EDGE`,
 │ extrude a 45° chamfer on the +z face (blade i side; mates with blade i+1's −z chamfer)
 │ AND on the −z face (blade i side; mates with blade i−1's +z chamfer). The chamfer face
 │ spans the full panel_thickness in z so the lap joint forms naturally at the panel-panel
 │ boundary. Detent bump (0.3-0.5 mm radius) embossed on the +z chamfer face; matching
 │ depression on the −z chamfer face. The rib's outer face stays flat (no click features
 │ on the rib) — the chamfer and detent live entirely on the panel envelope, and the
 │ rib doesn't reach this x-range at all (rib terminates at x = 0.185 m).
 │ → **CI regression** `tests/test_geometry/test_rib_taper_out.py`: assert
 │ `blade.outermost_y(x=L_blade) == panel_tangential_outer` (no rib material at the tip)
 │ AND `blade.outermost_y(x=L_blade − 0.020) == rib_outer_y` (rib still present 20 mm
 │ inboard of the tip). Catches accidental drift back to a full-length rib.
 │ → **boss thickening at pivot:** at x ∈ [pivot_center_x − boss_radius, pivot_center_x + boss_radius],
 │ panel widens from `panel_width(x)` (6-8 mm) to 2·PIVOT_BOSS_RADIUS = 12 mm via a
 │ cosine taper: `width_local(x) = panel_width(x) + (2·PIVOT_BOSS_RADIUS − panel_width(x)) · 
 │ smoothstep(0, boss_radius, |x − pivot_center_x|)`. The rib root y-position stays at
 │ `±(panel_width(x)/2 + rib_width_base/2)` — the boss bulges into the panel-interior
 │ region, NOT into the rib region; rib roots are NOT shifted by the boss.
 │ → **regression test** (`tests/test_geometry/test_boss_inside_inter_rib.py`): generate a baseline
 │ blade; assert (a) the boss bounding-circle of radius PIVOT_BOSS_RADIUS is fully contained
 │ within the inter-rib region (panel solid); (b) `(boss ∩ rib_solid).Volume == 0`; (c) the boss
 │ merges via small fillet at the panel-boss boundary (no zero-thickness wall).
 │
 ├── Step 2: carved_panel = apply_layer2_fields(panel_solid, params.layer2)
 │ ├── if TPMS active: generate TPMS surface → intersect with panel_solid → subtract
 │ ├── if noise threshold active: sample 2D Perlin/Simplex noise → mask zeroed within 5 mm
 │ │ of every outer-rib region → intersect with panel_solid → subtract
 │ ├── if louver active: generate parallel cuts/ribs at angle/spacing → clip to panel_solid →
 │ │ subtract or union
 │ ├── if texture active: generate dimple/ridge/bump array → clip to panel_solid → subtract or union
 │ └── if edge feature active: apply serrations/scallops to LE/TE (panel_solid only)
 │ → all fields safe-by-construction (parameter bounds guarantee no CAD failures)
 │ → every subtraction operand is intersected with panel_solid before the Boolean op,
 │ so Layer 2 can NEVER reach into rib material
 │
 ├── Step 3: try: carved_panel = apply_layer3_primitive(carved_panel, params.primitive)
 │ except OpenCASCADE_error: log + skip
 │ └── 0 or 1 independent primitive (slot/ellipsoid/wedge); 6-DOF placement,
 │ clipped to panel_solid AND ≥5 mm from every outer-rib click region
 │ → ONLY step where CAD failures can occur; CAD-safe via try/except
 │
 ├── Step 4: blade = ribs ∪ carved_panel
 │ └── TO-optimized rib material is union-ed back into the blade in its original form;
 │ Layer 2/3 never touched it, so the rib structural integrity is preserved by construction
 │
 ├── Step 5: mfg_score = manufacturability_check(blade) [see §9.7.3]
 │
 └── Step 6: if mfg_score >= 0.5: export STL; else: return None
```

**Panel-domain invariant (hard structural guarantee):** All Layer 2 and Layer 3 Boolean operations are intersected with the panel domain (`full_envelope.cut(rib_mask_3d)` per Step 0 above; `rib_mask_3d` is built by extruding `ribs.faces(">Z").wires` through `panel_thickness_max`) before being applied to the blade body. The TO-optimized rib material is union-ed back into the blade *after* the Layer 2/3 pipeline completes — **Layer 2/3 fields cannot reach into rib material under any parameter combination**, including the adversarial sets exercised in Spike 0.7a. This is distinct from the §9.7.3 Check 7 click-feature exclusion (which is positional, enforced at JSON-schema time); the panel-domain mask is a generator-pipeline invariant enforced at the Boolean-op level on every evaluation.

**Why this exists :** Without the panel-domain mask, "envelope" in the Layer 2 pipeline is the whole blade (2 ribs + 1 panel as one body, per §2.3). A blind Boolean subtraction punches through everything in its path — TO-optimized rib material, the §3.1.2 preserved pivot region (a TPMS through-cut at d/w ≈ 0.25 around the 3 mm hole would push K_t well past 2.42 and break the 5.58 MPa fatigue allowable), and the click-feature footprint (Check 7 covers positional exclusion of *primitive locations* but cannot stop a TPMS *surface* that happens to intersect the click region). The Phase 5 step 59.5 combined-blade structural gate would catch the rib-destruction failure on top-3 designs, but only after Phase 4 has burned hundreds of compute-hours evaluating architectures whose ribs were being silently destroyed every evaluation. The panel-domain mask kills the failure mode at the generator level.

**Required regression test (`tests/test_geometry/test_panel_mask.py`):** generate a blade with adversarial Layer 2 parameters that would punch through the ribs without the mask (e.g., TPMS with cell size = min and rotation aligned with the rib long axis; noise threshold at the lower-bound material-retention threshold and X/Y scale set so noise cells overlap rib material). Compute `volume_diff = ribs_input.Volume − (output_blade ∩ ribs_input.bounding_box).Volume` and assert it equals zero to numerical precision (the rib material is bit-for-bit preserved). Any code path that fails this test must not reach BoTorch — every Phase 4 evaluation is silently broken until it passes.

**CAD safety guarantees:**
- Layer 1 envelope: never fails (smooth parametric + Fourier construction).
- Layer 2 macro-pattern fields (louver, texture, edge): never fail (procedural with mathematical bounds).
- Layer 2 procedural math fields (noise threshold, TPMS): never fail (deterministic mathematical operations).
- Layer 3 independent primitive: may fail; handled by try/except + skip.
- **Total CAD failure rate target: <2% of generated designs** (improved from ~5% in earlier flat-primitive drafts because most independent primitives removed).

**BO failure handling:**
- If geometry is None: BO sees mass=∞, J_fan=0, manufacturability=0 — design strongly penalized but doesn't crash the optimization loop.
- Failed designs are recorded in the **Drive/JSONL ledger** with the matching §9.4.2 `failure_code` and a `designs/{hash}/.failed_<code>` marker for post-hoc analysis; consistent failure patterns flag bugs in the generator.

#### 9.7.2 Layer 2 Field Implementation Notes

**Order is fixed** to avoid order-dependent CAD problems: TPMS first (operates on bulk interior), then noise threshold (volume cutouts), then louver (planar cuts/ribs), then texture (surface features), then edge serrations (boundary modifications).

**TPMS field:** generated from analytical equation (gyroid: `sin x cos y + sin y cos z + sin z cos x = 0`; Schwarz-D: `sin x sin y sin z + sin x cos y cos z + cos x sin y cos z + cos x cos y sin z = 0`). Scaled by cell size, rotated by user angles, sampled on a voxel grid through the blade interior, **multiplied by an LE/TE keep-out ramp (+ -Transition-Re widening):** `mask(x) = smoothstep(0, TPMS_LE_PROTECT, dist_to_LE(x)) * smoothstep(0, TPMS_TE_PROTECT, dist_to_TE(x))`. **Baseline:** `TPMS_LE_PROTECT = max(5 mm, 0.05·chord)` and `TPMS_TE_PROTECT = max(3 mm, 0.03·chord)`. **Transition-Re widening:** if any TPMS perforation would lie within a region where the local `Re_x = v_local · x / ν < 1e4`, **widen `TPMS_LE_PROTECT` to `max(10 mm, 0.10·chord)`** for that design. **`v_local` at filter time:** the local tangential velocity at the perforation's radial position depends on instantaneous angular velocity; under PITCHING oscillation, the time-peak (NOT the time-average) is the correct screening reference because the worst case is what trips transition during the high-speed half of the cycle. Compute `v_local = ω_blade_max · r_centroid` where `ω_blade_max = θ_max · ω_SHM = 0.7 · 12.566 = 8.8 rad/s` (the worst-case pitching peak from the SU2 motion spec) and `r_centroid` is the radial distance from the wrist axis to the perforation centroid (in metres). For a perforation at the blade tip (`r_centroid ≈ 0.25 m`), `v_local ≈ 2.20 m/s`; at the panel midspan (`r ≈ 0.15 m`), `v_local ≈ 1.32 m/s`. The filter applies `Re_x < 1e4` using this time-peak velocity, so the widened-protect region is the conservative envelope. The generator checks each design's TPMS perforations against this rule at parameter-validation time and applies the wider protect distance per-design where needed. Without this widening, perforations near the LE trip transition at x_trans ≈ 0.03 m, BL thickens ~3×, and the smooth-roughness assumption that the §3.2.4 wall-roughness calibration is built on fails. Both `tpms_le_protect_mm` and `tpms_te_protect_mm` are part of the locked-params dict (hashed by §6.2.5 design_hash + §9.4.1 config-hash assertion). After masking: converted to STL via marching cubes, subtracted from envelope. Outer skin preserved by the min-feature-size constraint in §N7 (panel skin must remain ≥0.4 mm).

**Noise threshold field:** 2D Perlin or Simplex noise sampled at a panel point grid; noise scaled by user X/Y-scale and rotated by user angle; voxels where noise > threshold are removed via Boolean subtraction. Threshold is hard-bounded to retain ≥40% material (no skeletal designs). 2D-only (panel prints flat), so no overhang concerns.

**Louver field:** parallel cuts at user angle/spacing/width, with optional clustered-at-tip or gradient-toward-LE spacing. Polarity flag: subtract = slits (air-permeable), union = ribs (drag-enhancing protrusions). All cuts/ribs guaranteed ≥1 mm from envelope edge. **auto-fillet:** every louver slot end-cap is automatically filleted with radius `min(0.5 mm, slot_width / 3)` before the Boolean op. Rationale: sharp slot ends are stress concentrators well above the K_t ≈ 2.42 baseline (§3.1.3) and would silently fail the Phase 5 step 59.5 combined-blade FEA gate on a substantial fraction of louvered designs. Geometry-side only — no parameter exposed to BO and no storage impact.

**Texture field:** distributed dimple/ridge/bump array at user density and characteristic size, applied across the panel surface with user orientation angle. Polarity flag controls add/subtract.

**Edge feature field:** modifies LE/TE with serrations or scallops at user count and depth. Application categorical selects LE-only, TE-only, or both.

#### 9.7.3 Manufacturability Filter (§N7)

<a id="N7"></a>
<!-- §N7 = this manufacturability-filter section; the "N7" shorthand is used throughout the document. -->

The manufacturability filter has **11 checks** to handle the new field types and stricter edge cases.

| Check | Threshold | Polarity |
|-------|-----------|----------|
| 1. Minimum feature size | All features ≥ 0.8 mm (= 2× nozzle for 0.4 mm). Applies to wall thicknesses, void widths, primitive sizes, TPMS cell dimensions. | Moderate failure (-0.3) |
| 2. Overhang angle | Overhangs ≤ 45° from vertical without support, OR ≤ 60° with bridging acceptable. TPMS surfaces are inherently self-supporting and pass automatically. | Moderate failure (-0.3) |
| 3. Connectivity | Blade must be a single connected component (no floating bits). | **Critical failure → score 0** |
| 4. Bridging limits | Any horizontal section longer than 8 mm in any FDM orientation is flagged. | Soft failure (-0.1) |
| 5. Internal voids | Any fully-enclosed void without an exit path ≥ 1 mm is flagged unprintable. **TPMS lattices exempt** . | **Critical failure → score 0** (with TPMS exemption) |
| **1b. TPMS-induced thin sections** | Thin sections (<0.8 mm) created by TPMS surface intersecting the envelope are **exempt from check #1** if structurally sound (TPMS surfaces are mathematically smooth and self-supporting; thin sections at the lattice/skin interface are an unavoidable feature of through-blade-porosity TPMS, not a manufacturability defect) | No penalty if TPMS is the source |
| 6. Edge clearance | All features must be ≥ 1 mm from blade outer envelope edge. Layer 2 fields guarantee by construction; Layer 3 primitives enforce via position constraints. | **Critical failure → score 0** |

| 7. Click feature clearance (MED-10 panel-edge update) | A **5 mm exclusion band around the click-feature footprint on every blade's panel outer tangential edge** (mating chamfer + detent surfaces) is preserved — no Layer 2 or 3 features may extend into this region. **Location:** the footprint is at the outer tip of each panel's tangential edge, `x ∈ CLICK_FOOTPRINT_X_RANGE = (L_blade − 0.010 m, L_blade)` × `CLICK_FOOTPRINT_Y_RANGE_PANEL_EDGE` per item #3 panel-edge relocation — a single shared pair of constants in `src/fanopt/geometry/schema.py` read by both the §9.7 generator's `make_panel_solid` step (writes the chamfer + detent into the panel) and this check, so the two can't drift apart. The rib does NOT reach this x-range (rib terminates at `L_blade − RIB_TIP_TAPER = 0.185 m` per Architectural A); the entire click region is panel-only. **Applies to inner blades and guard blades alike**. Enforced as a hard parameter bound at JSON-schema time in `src/fanopt/geometry/schema.py`: (a) Layer 2 louver field — spacing distribution + position bounds cap at `CLICK_FOOTPRINT_X_RANGE[0] − 5 mm` on the panel-edge-facing edge; (b) Layer 2 noise threshold field — sampling mask is zeroed within 5 mm of the `CLICK_FOOTPRINT_X_RANGE` region on every panel's outer tangential edge; (c) Layer 2 TPMS field — cell pitch + edge-falloff envelope keeps through-blade porosity away from the `CLICK_FOOTPRINT_X_RANGE` region on every panel; (d) Layer 3 primitive — position constraint tightened from "≥1 mm from envelope edges" to "≥5 mm from `CLICK_FOOTPRINT_X_RANGE`". | Hard parameter bound (no penalty step) |
| 8. Aspect ratio | No feature with aspect ratio > 20:1 (prevents thin protrusions that fail printing). | Soft failure (-0.1) |
| 9. Noise field threshold range | Hard-bounded to retain ≥ 40% material (parameter bound). | Bound (no penalty step) |
| 10. TPMS cell size minimum | Cell size ≥ 3× minimum feature size (parameter bound). | Bound (no penalty step) |
| 11. Fourier amplitude maximum | Envelope stays within ±15% of mean dimensions (parameter bound). | Bound (no penalty step) |
| **12. Layer-adhesion Z-thin-section flag ** | Any solid feature thinner than 1.5× layer height (0.3 mm at 0.2 layer-height; 0.225 mm at 0.15) along the Z (build) direction is flagged: PETG inter-layer adhesion is the weakest link, and a thin Z-section is at risk of delamination under -Material's Z-direction cyclic load. **Hard fail if more than 5% of the design's panel volume is Z-thin** (catches TPMS skin-interface thin walls that should have been caught by check #1b but slipped through under the TPMS exemption). | Moderate failure (-0.3) |
| **13. Warpage proxy ** | Large planar faces × aspect ratio: a face with bounding-box aspect > 8:1 AND area > 1000 mm² is flagged. PETG warps along the long axis of large flat sections during cooling; an 8:1 aspect-ratio panel face is geometrically prone to potato-chip warpage. (TPMS / noise-cutout designs naturally break up large planar areas and pass this check automatically.) | Soft failure (-0.1) |
| **14. Support-scar location on functional surfaces (M1 reworded)** | The plano-convex constraint (when `print_orientation == 'rib-flat'`) puts the flat face on the build plate. Check 14 asserts that **the panel's bottom face (in the print frame, the bed-contact face) has its outward normal pointing in −z** (i.e., the face's surface normal points into the bed). The calibrated face for the §3.2.4 wall-roughness model is therefore the bed-contact face — no support scars, achieving Ra ≈ 5-10 µm from layer-line texture only. If the generator produces geometry where the calibrated face would need support, raise a critical failure (post-support-removal Ra ≈ 100-300 µm breaks the calibration). (Earlier draft said "bottom face faces +z in the print frame", which is contradictory — a bed-contact face has its outward normal pointing DOWN into the bed, not UP to +z.) | **Critical failure → score 0** |
| **1b extension — TPMS exemption paired with FEA half-allowable ** | Check #1b exempts TPMS-induced thin sections (<0.8 mm) from check #1 because the lattice/skin interface inherently has them. -N7-1b pairs the exemption with a **FEA half-allowable** at the §59.5 combined-blade gate: any FEA element whose underlying STL geometry has wall thickness < 0.8 mm uses **σ_allowable = 0.5 · σ_allowable_normal** (single-perimeter PETG has ~50% the tensile strength of multi-perimeter; TPMS thin walls usually print as single perimeters). Without this pairing, check #1b lets thin-walled TPMS designs into Phase 5 and the §59.5 gate rejects them anyway — wasted Tier-1/0/Phase-5 compute. | No penalty step here (the half-allowable lives in §59.5, not §9.7.3); the §9.7.3 row exists for traceability of the linkage. |

**Slicer-clipping note (caveat to check #1b):** the exemption from check #1 for TPMS-induced thin sections assumes the slicer can faithfully reproduce knife-edge geometry at the lattice/skin interface. In practice, most slicers (Bambu Studio, PrusaSlicer, Cura) silently clip features thinner than 0.5-1 layer-width to zero or to a single extrusion line. This can produce a G-code geometry that differs visibly from the input STL in the TPMS-affected regions, and the discrepancy compounds with the CFD wall-shape sensitivity at the rib-ridge / panel interface. The baseline accepts this risk because the CAD STL is the canonical geometry both the CFD and the manufacturability filter operate on; if Phase 6 physical measurements reveal a significant prediction-vs-measurement gap on TPMS-heavy designs, the .A fallback below is the response.

**.A Fallback — slicer-aware pre-clipping pass (only if Phase 6 reveals TPMS clipping bias):**

If Phase 6 physical measurements show a systematic prediction-vs-measurement gap for TPMS-heavy designs (>15% J_fan delta, with visible discrepancy between sliced G-code and CAD STL traced to clipped knife-edge geometry), upgrade the manufacturability filter to do its own pre-clipping pass before CFD evaluation:

1. **Detection:** scan the STL for local thickness < 0.4 mm using a signed-distance-field or shell-thickness check (e.g., `trimesh.proximity.thickness` or a custom raycast pass).
2. **Clipping:** for each below-threshold region, either snap the surface to the nearest skin face (collapse the thin taper to zero) or delete the region (remove the feature entirely). Pick per-region based on local connectivity.
3. **Re-validation:** re-run §N7 checks on the clipped geometry. If it now fails check #3 (connectivity) or check #5 (internal voids), the cleanup created a new failure mode and the design is rejected.
4. **CFD re-eval:** affected Pareto-front designs are re-evaluated through CFD with the clipped STL. Update the multi-fidelity GP with corrected J_fan values. Mark these designs in the **Drive/JSONL ledger** with a `clipping_corrected = true` field on the JSONL row.

.A is **off by default**; it activates only if Phase 6 evidence justifies the additional pipeline complexity and the additional CFD re-evaluation budget.

**Scoring:** manufacturability_score in [0, 1].
- **1.0** = no issues.
- **0.5-0.8** = printable but suboptimal (warning logged).
- **≥ 0.8** = clean design.
- **< 0.5** = infeasible (design rejected, BO sees J_fan=0 + mass=∞).

**Failure penalty types:**
- **Critical failures** (#3, #5, #6): immediate score = 0.
- **Moderate failures** (#1, #2): each subtracts 0.3 from the running score.
- **Soft failures** (#4, #8): each subtracts 0.1.
- **Hard parameter bounds** (#7, #9, #10, #11): no penalty step; enforced upstream in the JSON schema so violations never reach the filter.

#### 9.7.4 Installation Notes

```bash
# CadQuery already in the environment lockfile
conda install -c conda-forge cadquery

# Additions:
pip install noise # Perlin / Simplex noise generation
pip install trimesh # marching cubes for TPMS STL extraction
# scikit-image (used by connectivity post-processor)
```

---

## 10. Validation Approaches

### 10.1 Computational Validation

#### 10.1.0 Material constants (locked)

The full material-property table below is the canonical source for every K_t allowable, every Filter 2 / §59.5 stress check, and the `material_hash` cross-tier assertion. Production code reads these from `src/fanopt/material_locks.py`; the K_t hotspot table in §3.1.5 references them by symbol.

| Symbol | Value | Notes |
|--------|-------|-------|
| `RHO_PETG` | 1270 kg/m³ | FDM PETG, solid infill (slicer ~95-100% wall coverage in our parts) |
| `RHO_PIN` | 7850 kg/m³ (steel) or 8500 kg/m³ (brass) | Pin material is a build choice; both are within the m_total = 100 g budget |
| `E_PETG_XY` | 1300 MPa | Young's modulus, in-plane (FDM published) |
| `E_PETG_Z` | 1000 MPa | Young's modulus, Z-direction (interlayer) |
| **`σ_y_XY`** | **45 MPa** | Tensile yield, XY in-plane loading |
| **`σ_y_Z`** | **30 MPa** | Tensile yield, Z-direction (interlayer). **FDM PETG Z is 20-40% below XY per §3.1.8 anisotropy data → 27-36 MPa range; conservative middle = 30 MPa.** (Earlier draft used 55 MPa, the upper-end published value, which contradicted the plan's own anisotropy claim and produced non-conservative Z allowables.) |
| `K_t_bearing` | 1.5 | Pin-bore baseline (Peterson) |
| `K_t_pivot_tension` | **2.42** | Peterson polynomial, **d/w = 3/12 = 0.25 in the 12 mm boss** (). The earlier draft listed d/w = 0.375 (3 mm hole in 8 mm panel, no boss) → K_t = 2.26, which applies to the un-thickened inter-rib panel; under the locked boss thickening (§6.3.1 Filter 2 / §9.7) the boss geometry binds at d/w = 0.25. |
| `K_t_pivot_bending` | 3.2 | Conservative upper bound of literature range 2.7-3.2 |
| `K_t_click_detent` | 3.0 | §3.1.5 hotspot table |
| Fatigue factor XY | 0.30 | Standard FDM PETG XY cyclic floor at 10⁵ cycles |
| Fatigue factor Z (cyclic) | 0.20 | FDM PETG interlayer cyclic floor |
| Fatigue factor Z (bearing) | 0.10 | Bearing-mode interlayer (conservative; bearing concentrates load on Z-stacked layers) |

**Derived allowables (consistent with the §3.1.5 K_t table; all values nominal cyclic):**
- Panel-pivot tension (XY, at boss): `0.30 · 45 / 2.42 = 5.58 MPa`
- Panel-pivot bending (XY): `0.30 · 45 / 3.2 = 4.22 MPa`
- Panel-pivot bearing (Z): `0.10 · 30 / 1.5 = 2.00 MPa` — **the 0.10 factor (vs 0.20 for click detent)** reflects that pin-bore bearing contact cycles at the 2 Hz waving frequency (~7000 cycles/hour, ~50k-500k lifetime cycles), so the FDM Z-fatigue knockdown is at its most conservative.
- Click detent (any orientation, Z-binding floor): `0.20 · 30 / 3.0 = 2.00 MPa` — **the 0.20 factor (vs 0.10 for bearing)** reflects that the chamfer face is loaded **only during fold/deploy events** (~few cycles/day, ~few thousand lifetime cycles), so the Z-fatigue knockdown can be relaxed by 2× vs the continuously-cycled pin-bore.
- Rib-panel fillet (rib-flat, XY): `0.30 · 45 / 1.5 = 9.00 MPa`

**Factor-rationale lock:** the two Z-direction allowables that both land at 2.00 MPa nominal use *different* fatigue knockdown factors (0.10 vs 0.20) because the duty cycles differ by ~1000×. The coincidence that both end at 2.00 MPa is numerical, not designed: pin-bore K_t = 1.5 with 0.10 factor matches click-detent K_t = 3.0 with 0.20 factor under σ_y_Z = 30 MPa. Treat as independent allowables; do not collapse to a single "Z-floor at 2.00 MPa" rule because the duty-cycle premise that justifies each factor is different.

**Binding hotspot under canonical load: panel-pivot bearing at 2.00 MPa nominal** (replaces the prior 3.67 MPa value which used the non-conservative σ_y_Z = 55 MPa).

#### FEA Stress Verification (Per Rib)

Claude writes the verification script. Key checks:

1. Peak von Mises stress at **panel pivot hole** (— pivot is in the panel, not the rib) < min(static-SF limit, cyclic-fatigue limit) across all three modes. For PETG (σ_y_XY = 45 MPa, σ_y_Z = 30 MPa) at **K_tt = 2.42 (d/w = 3/12 = 0.25 in the 12 mm boss)**:
 - **Static check (tension, SF = 1.5):** allowable = 45 / (1.5 · 2.42) ≈ **12.4 MPa** nominal stress at the panel pivot hole.
 - **Static check (bearing, Z):** allowable = 30 / (1.5 · 1.5) ≈ **13.3 MPa** nominal stress (bearing is in the Z direction; the K_t_bearing = 1.5 is smaller than K_tt because the bearing load is distributed through pin-bore contact rather than concentrated at a hole equator).
 - **Cyclic-fatigue check (§3.1.7; binding constraint):** three independent per-mode allowables apply (per §3.1.5 K_t hotspot table) — **tension 5.58 MPa, bending 4.22 MPa, bearing 2.00 MPa**. For a 2 Hz hand fan with 50k-500k lifetime cycles the fatigue limit binds; the **bearing mode at 2.00 MPa is the lowest** and therefore the canonical binding constraint. **The §6.4 hard constraint reads "all three per-mode allowables".** A verification run that only checks one mode would pass designs that violate another; always check all three.
 - **Rib-panel interface check:** under panel-pivot architecture, the rib no longer carries the pivot hole; the rib's structural critical point is at the **rib-panel interface** (where bending moments from aero pressure transfer between panel and rib). Verification: peak von Mises at the rib-panel fillet < per-hotspot allowable from the §3.1.5 K_t table. For a "rib-flat" print orientation with K_t_fillet = 1.5: cyclic allowable = 0.30 · 45 / 1.5 = **9.00 MPa nominal**.
2. Maximum rib tip deflection < 5 mm under peak aerodynamic + inertial load.
3. Use orthotropic material properties (E_XY = 1300 MPa, E_Z = 1000 MPa for FDM PETG).

#### Modal Analysis (Resonance Check)

Compute first 5-10 natural frequencies of an individual rib (cantilevered at pivot):
- First bending mode must be > 10 Hz (5x the 2 Hz waving frequency).
- For a 200 mm PETG rib at 2 mm thickness: f_1 is estimated at 15-30 Hz (safe margin).
- A TO-optimized rib with reduced mass may have a different first mode -- verify.

#### CFD Verification

1. Mesh independence study: coarse, medium, fine meshes. Quantity of interest should change <2% between medium and fine.
2. Steady vs. unsteady validation: run 3 designs through both to verify ranking preservation.
3. Gap leakage assessment: run one case with resolved rib-gap geometry to quantify leakage penalty.

### 10.2 Physical Validation

#### Anemometer Testing

- Equipment: handheld anemometer ($15-30).
- Protocol: fan mounted at fixed waving rhythm (use metronome), anemometer at 300 mm, record peak and average velocity over 10 cycles, repeat 5 times.
- Compare optimized vs. baseline fan at same waving effort.

#### Structural Testing

- **Static deflection:** Clamp rib at pivot, hang 50-200g weight from tip, measure deflection. Compare to FEA prediction.
- **Pivot fatigue:** Wave the fan continuously for 30 minutes (~3600 cycles). Inspect pivot region for cracks.
- **Assembly test:** Open/close the fan 100 times. Check for rib binding, spacer wear, **click-feature engagement quality**, inter-blade alignment in deployed state. Extended to ≥3000 cycles in Phase 6 step 83.

#### Smoke Visualization

- Incense stick at 100-200 mm from fan face.
- Wave fan, record with video.
- Look for: directed airflow pattern, vortex shedding from rib tips, gap leakage effects.

---

## 11. References and Sources

### Folding Fan Design and Geometry

1. [Japanese Folding Fan (Sensu) -- KA-CHO-FU-GETSU](https://kcfg-japan.com/blogs/blogs/japanese-folding-fan-sensu) -- Traditional sensu construction and dimensions.
2. [SensuOgi (folding fan) -- Japanese Wiki Corpus](https://www.japanesewiki.com/culture/SensuOgi%20(folding%20fan).html) -- Historical rib counts and spread angles.
3. [Fan Rib Collection and Folding Fan Making Tools -- MakerWorld](https://makerworld.com/en/models/1517149-fan-rib-collection-folding-fan-making-tools) -- 3D-printed fan rib designs.
4. [Collapsible Hand Fan (print in place) -- Printables](https://www.printables.com/model/944878-collapsible-hand-fan-print-in-place) -- 3D-printed folding fan pivot mechanism.
5. [Print-in-Place Hand Fan -- Printables](https://www.printables.com/model/132278-print-in-place-hand-fan) -- Alternative 3D-printed fan design with internal stops.
6. [3D-Printed Chinese Folding Fan -- Thingiverse](https://www.thingiverse.com/thing:673741) -- No-assembly-required 3D-printed folding fan.
7. [Fan Size Guide -- Ibasen (Edo fan maker)](https://www.ibasen.co.jp/en/pages/fan-size) -- Traditional fan size standards.
8. [Design and Construction of a Support for a Folding Fan -- AIC](https://cool.culturalheritage.org/coolaic/sg/bpg/annual/v05/bp05-04.html) -- Fan structural analysis for conservation.

### Parametric CAD and Scripted Geometry

9. [CadQuery GitHub -- Python Parametric CAD](https://github.com/CadQuery/cadquery) -- CadQuery source and documentation.
10. [CadQuery Documentation](https://cadquery.readthedocs.io/en/latest/intro.html) -- API reference and tutorials.

### Topology Optimization

11. [PyTopo3D: A Python Framework for 3D SIMP-based Topology Optimization (arXiv 2025)](https://arxiv.org/abs/2504.05604) -- PyTopo3D paper with benchmarks.
12. [PyTopo3D GitHub](https://github.com/jihoonkim888/PyTopo3D) -- Source code and examples.
13. [PyTopo3D on PyPI](https://pypi.org/project/pytopo3d/) -- pip installation.
14. [ToPy: Topology Optimization using Python (GitHub)](https://github.com/williamhunter/topy) -- Alternative Python TO framework.
15. [DTU TopOpt: Topology Optimization Codes in Python](https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python) -- Educational TO codes.
16. [BESO for CalculiX (GitHub)](https://github.com/calculix/beso) -- BESO topology optimization with CalculiX.
17. [TopOpt Python Library](https://pypi.org/project/topopt/) -- SIMP with MMA solver.
18. [DL4TO GitHub Repository](https://github.com/dl4to/dl4to) -- ML-accelerated TO library.
19. [DL4TO Documentation](https://dl4to.github.io/dl4to/) -- API reference and tutorials.

### FEA Tools

20. [CalculiX: A Three-Dimensional Structural Finite Element Program](https://www.calculix.de/) -- CalculiX solver.
21. [pycalculix -- Python Library for CalculiX (PyPI)](https://pypi.org/project/pycalculix/) -- Python automation for CalculiX.
22. [pyccx -- Python Framework for CalculiX (GitHub)](https://github.com/drlukeparry/pyccx) -- Alternative Python CalculiX wrapper.
23. [SfePy: Simple Finite Elements in Python](https://sfepy.org/) -- Pure Python FEA.
24. [SfePy Gallery of Examples](http://sfepy.org/gallery/) -- Beam and elasticity examples.

### Aerodynamic Shape Optimization

25. [SU2: Multiphysics Simulation and Design Software](https://su2code.github.io/) -- Official SU2 website.
26. [SU2 Tutorial Collection](https://su2code.github.io/tutorials/home/) -- Step-by-step tutorials.
27. [SU2 Unsteady Shape Optimization Tutorial](https://su2code.github.io/tutorials/Unsteady_Shape_Opt_NACA0012/) -- Unsteady adjoint-based ASO.
28. [SU2 GitHub Issue #193](https://github.com/su2code/SU2/issues/193) -- Incompressible solver pitching motion bug.
29. [Gmsh Tutorial Collection](https://gmsh.info/doc/texinfo/gmsh.html#Tutorial) -- Meshing tutorials.
30. [SimScale: CFD Simulation Software](https://www.simscale.com/product/cfd/) -- Cloud-based CFD for validation.

### ML Surrogate Modeling

31. [BoTorch: Bayesian Optimization in PyTorch](https://botorch.org/docs/overview/) -- Official documentation.
32. [BoTorch Tutorials](https://botorch.org/tutorials/) -- GP and BO examples.
33. [GPyTorch Models in BoTorch](https://botorch.org/docs/models/) -- GP model reference.
34. [BoTorch Multi-Objective BO Tutorial](https://botorch.org/docs/tutorials/multi_objective_bo/) -- qEHVI and qNEHVI.
35. [Ax: Adaptive Experimentation Platform](https://ax.dev/tutorials/) -- Higher-level BO wrapper.

### 3D Printing Materials

36. [PETG vs PLA vs ABS: Strength Comparison (Ultimaker)](https://ultimaker.com/learn/petg-vs-pla-vs-abs-3d-printing-strength-comparison/) -- Material properties.
37. [Experimental and Numerical Analysis for FDM PETG (PMC 7600181)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7600181/) -- FDM PETG modulus values 1117-1330 MPa.
38. [Tensile and Fatigue Analysis of 3D-Printed PETG (ResearchGate)](https://www.researchgate.net/publication/332001021_Tensile_and_Fatigue_Analysis_of_3D-Printed_Polyethylene_Terephthalate_Glycol) -- PETG fatigue data.
39. [Effect of 3D Printing Parameters on Fatigue Properties (MDPI 2023)](https://www.mdpi.com/2076-3417/13/2/904) -- Print parameter effects on fatigue.
40. [Material Anisotropy in Additively Manufactured Polymers (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8512748/) -- AM polymer anisotropy review.

### Aerodynamic Physics

41. [Drag on Oscillating Flat Plates at Low Reynolds Numbers (Cambridge Core)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/drag-on-oscillating-flat-plates-in-liquids-at-low-reynolds-numbers/4A6222AC968750F25984BD2538E5DCDA) -- Experimental data for oscillating plates.
42. [Drag Coefficient -- Wikipedia](https://en.wikipedia.org/wiki/Drag_coefficient) -- Reference for flat plate C_D values.

### TO Post-Processing

43. [Surface Smoothing for Topological Optimized 3D Models (Springer 2021)](https://link.springer.com/article/10.1007/s00158-021-03027-6) -- Smoothing methods.
44. [Smooth Geometry Extraction from SIMP via SDF (arXiv 2025)](https://arxiv.org/html/2512.06976v1) -- Modern SDF-based approach.

### Multi-Fidelity Bayesian Optimization

45. [BoTorch Multi-Fidelity BO with Knowledge Gradient](https://botorch.org/docs/tutorials/multi_fidelity_bo/) -- Official BoTorch tutorial for continuous multi-fidelity BO.
46. [BoTorch Discrete Multi-Fidelity BO](https://botorch.org/docs/tutorials/discrete_multi_fidelity_bo/) -- Discrete fidelity levels (steady vs. unsteady).
47. [SingleTaskMultiFidelityGP API Reference](https://botorch.readthedocs.io/en/latest/_modules/botorch/models/gp_regression_fidelity.html) -- Model documentation.

### 2D Topology Optimization

48. [DTU TopOpt Python Codes](https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python) -- Educational 2D SIMP codes in Python.
49. [FEniCS SIMP TO Tutorial (comet-fenics)](https://comet-fenics.readthedocs.io/en/latest/demo/topology_optimization/simp_topology_optimization.html) -- FEniCS-based SIMP with arbitrary BCs.
50. [A 55-line code for large-scale parallel TO in 2D and 3D (arXiv 2020)](https://arxiv.org/abs/2012.08208) -- Compact FEniCS TO implementation.
51. [FEniCSx Topology Optimization Guide (Medium, 2024)](https://medium.com/@abolfazl.dmg/topology-optimization-with-fenicsx-a-step-by-step-guide-b603a237dd61) -- Step-by-step FEniCSx TO tutorial.
52. [Sigmund 99-Line Code FEniCSx Rewrite (GitHub)](https://github.com/floating-gates/Sigmund---A-99-Line-Topology-Optimization-Code-Written-in-MATLAB---FEniCSx-rewrite) -- Python rewrite of the classic 99-line code.

### SU2 Capabilities and Limitations

53. [SU2 Boundary Conditions Documentation](https://su2code.github.io/docs_v7/Markers-and-BC/) -- Official marker/BC reference (no porous media).
54. [SU2 Porous Media Discussion (CFD-Online)](https://www.cfd-online.com/Forums/su2/240454-su2-porous-media-porous-jump-model.html) -- Confirms porous media is NOT supported.

### CadQuery Installation

55. [CadQuery Installation Documentation](https://cadquery.readthedocs.io/en/latest/installation.html) -- Conda recommended over pip.
56. [OCP Build System (GitHub)](https://github.com/CadQuery/ocp-build-system) -- OCP wheel build infrastructure.

---

## 12. Project Structure and Tooling

The optimization codebase uses a single Git repository (`fan-optimization/`); the scaffolding is created in Phase 0 before any optimization code is written so that all subsequent modules go in the right places from the start.

### 12.0 Naming Locks + CI Gates

**Naming locks (all production code reads these names; CI fails on drift):**

| Symbol | Canonical name | Notes |
|--------|----------------|-------|
| Click feature | `click_detent` (e.g. `click_detent_force_n`, `click_detent_radius_mm`, `K_t_click_detent`) | The bare word `detent` is **reserved for the V2-deferred locking detent** (Spike 0.4 fallback, see §2.1 / §2.2 / `docs/V2_backlog.md`). A regex retirement of `\bdetent\b` would kill click features — DO NOT do that. The locking-detent retirement regex is restricted to `\b(?:locking[_ ]?detent|lock[_ ]?detent)\b`. |
| Cost tuple | `config.cost.COST_TUPLE` | Single source for `(2.0, 10.0, 50.0)` — no stray literal `(2, 10, 50)` outside `config/cost.py`. |
| Material constants | `src/fanopt/material_locks.py` | Single source for σ_y_XY, σ_y_Z, ρ, E_XY, E_Z, K_t hotspot table. |
| Hashes | `(design_hash, config_hash, physics_hash, material_hash)` | Distinct definitions in §6.2.5 + §9.4.1; dedup uses the composite key. |
| Material orientation tag | `print_orientation ∈ {rib-flat, edge, custom-angle}` (NOT `flat` / `edge` — qualifying prefix `rib-` removes the ambiguity with `print_orientation = edge`). |

**CI gates (run on every push; block the merge on failure):**

| Gate | Test file | What it asserts |
|------|-----------|-----------------|
| **Dimension audit** | `tests/test_audit/test_no_legacy_dims.py` | Greps the doc + the schema + the cfg templates for legacy stack-height values; **regex set** (asserts empty match): `\b[5-9][0-9]\b\s*(?:[-–—]|to)\s*\b[5-9][0-9]\b\s*mm`, `\bbetween\s+\d+\s+and\s+\d+\s+mm\b`, `\$[5-9][0-9]\s*[-–—]\s*[5-9][0-9]\$\s*\\?mm`, plus thin-space variant `[  ]`. **CI gate fails on a non-empty match set** (the deliverable is the empty result, not the regex itself). Manually clear figure captions and code docstrings before running. |
| **Click-detent preservation** | `tests/test_audit/test_click_detent_count.py` | Greps `click[_ ]detent` across the codebase + the doc; asserts the count is exactly the baseline locked in `tests/fixtures/click_detent_baseline.txt` (drifts up = a new click feature was added; drifts down = the retirement regex over-reached). |
| **No rib pivot hole** | `tests/test_geometry/test_no_rib_pivot_hole.py` | Greps `src/fanopt/geometry/schema.py` for any `rib_pivot_hole_*` parameter (must be empty under the locked panel-pivot architecture). |
| **Cost tuple single source** | `tests/test_bo/test_cost_tuple_single_source.py` | Greps the repo for stray `(2, 10, 50)` literals outside `config/cost.py` AND greps for legacy `(1, 5, 50)` literals anywhere outside `docs/` revision-history sections. The doubled cost-tuple lock makes both regexes part of the gate — the (1,5,50) regex catches stale references that the (2,10,50) regex would miss. |
| **Hotspot table completeness** | `tests/test_material/test_k_t_by_hotspot_complete.py` | Asserts every hotspot in the §3.1.5 table has a non-null (K_t, mode, allowable) tuple in `material_locks.k_t_by_hotspot`. |
| **Inertia heterogeneous** | `tests/test_geometry/test_inertia_heterogeneous.py` | Σ ρᵢ·Vᵢ vs OCC `BRepGProp` mass-properties on the assembled STEP; threshold `< 0.1%`. |
| **SU2 freestream** | `tests/test_cfd/test_freestream_direction.py` | Sample inlet-face velocity, assert vector matches `FREESTREAM_DIRECTION` within numerical tolerance. |
| **SU2 physical motion** | `tests/test_cfd/test_pitching_physical_motion.py` | Run 1 cycle: assert (a) tip z-displacement ≈ **+0.16 m** at t=T/4 (productive-stroke turning point, tip swept toward user); (b) tip velocity magnitude ≈ 2.20 m/s **in +z** at t=0 (peak-velocity instant on the productive stroke, where SHM derivative is maximum); (c) pivot stays fixed. **Sign-convention note (M20):** the +0.16 m and +z-velocity signs assume the spec's right-handed-about-(-y) angular velocity convention per §0 row 26 (PITCHING_OMEGA in the rendered SU2 cfg is `(0, -12.5664, 0)` per C11 sign lock). Under standard right-handed-about-(+y) convention, the displacement would be −0.16 m and the velocity −z — that result indicates the cfg was rendered with the WRONG sign (a missing negative). **Promoted to pre-merge gate (C11 lock)** so the sign bug is caught before any SU2 minutes are consumed; the gate runs on a probe mesh in ≤30 s of CI time, not at full Tier-1 cost. |
| **SU2 AMPL unit** | `tests/test_cfd/test_su2_ampl_unit.py` | Steady run with `PITCHING_AMPL = (0, 0.6981, 0)` applied as fixed rotation; assert tip displacement matches sin(40°) · r_tip ≈ 0.161 m within 1%. Catches the SU2-version-dependent radians-vs-degrees AMPL interpretation. |
| **2D slice z-squash** | `tests/test_cfd/test_2d_slice_z_squash.py` | All blade cross-sections in `slice.su2` share z within 1e-6 m. |
| **V1 lock retired** | `tests/test_audit/test_no_mandatory_lock.py` | Greps the doc + the schema for "Locking mechanism is mandatory" outside Appendix D revision history; asserts empty match. Catches stale rib-tab-lock prose that contradicts the V2-deferred lock policy. |
| **Pivot pin material** | `tests/test_audit/test_pivot_pin_material.py` | Greps the doc for "PETG pivot pin" or "brass or PETG pivot" outside Appendix D revision history; asserts empty match. Catches the failure mode where a builder follows stale §1.1 text and prints a PETG pin that fails on first deploy. |
| **Geometry units = meters (SI)** | `tests/test_geometry/test_units_meters.py` | Generates a baseline blade via the production CadQuery generator; asserts the assembly bounding-box matches expected SI extents: `0.18 < x_extent < 0.22 m`, `0.04 < y_extent < 0.06 m`, `0.002 < z_extent < 0.006 m`. An off-by-1000 mm error (the silent CadQuery unit-conversion drift) fails the test. Required because CadQuery is unit-agnostic and `i_wrist_about_y` would otherwise return values off by (1e-3)⁵ = 10⁻¹⁵, silently corrupting the I_wrist Pareto axis. The `src/fanopt/geometry/generator.py` module-level docstring locks: "ALL geometry passed through this module is in SI units (meters). Any CadQuery operation that introduces mm-valued literals MUST convert to meters before returning." |
| **r_com XZ-projection** | `tests/test_geometry/test_r_com_xz_projection.py` | Place a near-point mass at (0.1, 0.0, 0.05) m and assert `r_com_assembly ≈ 0.112 m` (not 0.1 m). Discriminates the XZ-projection (correct, perpendicular to +y wrist axis) from the XY-projection (the C1 bug). |
| **Steady-proxy sign** | `tests/test_cfd/test_steady_proxy_sign.py` | 45° slatted louver oriented to grab air on the productive (+z body sweep) stroke; assert `J_fan_steady_proxy > 0`. Catches the C2-inverted convention where productive louvers scored negative. |
| **JSONL Tier-1 nullable** | `tests/test_utils/test_jsonl_tier1_nullable.py` | Construct a Tier 1 row with `J_fan_productive = J_fan_return = J_fan_delta = J_fan_se = J_fan_peak = None`; assert Pydantic validation passes. Construct a Tier -1 row with these fields populated and `J_fan_se = J_fan_peak = None`; assert validation also passes. |
| **No heteroscedastic GP refs** | `tests/test_audit/test_no_heteroscedastic.py` | Grep codebase + doc (excluding Appendix D revision history + this gate's row) for forbidden patterns: `heteroscedastic`, `J_fan_se²`, `J_fan_se ** 2`, `train_Yvar = J_fan_se`, `train_Yvar=Y_var`. Asserts empty match. Catches stale per-observation noise refs that contradict the §0 fixed-floor lock. |
| **H8 lever-arm audit** (item 19; paired with item 7 Filter 2 fix) | `tests/test_audit/test_lever_arm_uses_wrist_to_tip.py` | Greps prose + code for `/ 0.20` or `r_tip = 0.20` near torque-to-force or tangential-reaction language; allow-listed only where the calc is truly from-pivot (rib-internal bending integrals, which legitimately reference L_blade = 0.20 m). Catches stale propagation of pivot-to-tip when wrist-to-tip (0.25 m) is the correct lever. |
| **C2 sign convention** (item 20; paired with items 2 + 3 FREESTREAM cleanup) | `tests/test_audit/test_no_forward_backward_freestream.py` | Greps doc + cfg templates for `(forward)` or `(backward)` within 200 chars of `FREESTREAM_DIRECTION`, `FREESTREAM_PRODUCTIVE`, or `FREESTREAM_RETURN`, excluding Appendix D revision-history and §0 row-26 historical notes. Asserts empty match. Catches stale pre-C2 convention prose. |
| **Decimal-number typo guard** (item 21; paired with item 9 typo fix) | `tests/test_audit/test_no_triple_decimal.py` | Greps doc + schema for the pattern `\d\.\d\.\d\s*[-–—]\s*\d` (three-dot decimal adjacent to a range hyphen). Asserts empty match. Catches `2.2.2-3.8` copy-paste typos and similar triple-decimal artifacts that escape the main dimension-audit regex. Pairs with the dimension-audit gate to form a more complete numeric-format sanity check. |
| **Rib taper-out at tip** (Architectural A; paired with item #3 panel-edge click) | `tests/test_geometry/test_rib_taper_out.py` | Asserts `blade.outermost_y(x=L_blade) == panel_tangential_outer` (no rib material at the tip; the last 15 mm is panel-only) AND `blade.outermost_y(x=L_blade − 0.020) == rib_outer_y` (rib still present 20 mm inboard of the tip). Catches accidental drift back to a full-length rib that would shield the panel-edge click feature. |
| **Rib radial extent (HUB + tip taper bounds)** (Architectural D; C7 lock) | `tests/test_geometry/test_rib_radial_extent.py` | Asserts `rib.x_min == HUB_RADIUS = 0.020 m` (no rib material in the inner 20 mm HUB / boss region) AND `rib.x_max == L_blade − RIB_TIP_TAPER = 0.185 m` (no rib material in the outer 15 mm click region). Catches accidental drift to a full-radial-length rib that would conflict with the 12 mm boss at the root OR shield the click at the tip. |
| **Rib code matches locks** (C13; paired with C7 + H12 + C9 SI lock) | `tests/test_geometry/test_rib_code_matches_locks.py` | Imports `generate_rib` and asserts the emitted geometry has `length == L_RIB_M = 0.165 m ± 1e-6`, `base_width == 0.004 m` at root, `tip_width == 0.006 m` at tip, ALL in SI meters. Catches drift from any future direct edit of the §9.1 / §9.2 / `src/fanopt/geometry/generator.py` code that re-introduces mm units or pre-Round-5 dimensions. |
| **Rib-panel BC follows diagonal interface** (C15; paired with #35 trapezoidal panel + H13) | `tests/test_topopt/test_rib_bc_diagonal.py` | Builds a representative rib mesh, calls `get_rib_panel_interface_mask(rib_side="inner")`, asserts (a) the mask follows `y_interface(x) = panel_width(x)/2` within one grid cell, (b) the mask cells are NOT all in a single Cartesian row (the diagonal must traverse multiple rows from root to tip), (c) the fixed-DOF count equals the expected interface-cell count along the rib radial band. Catches drift back to the pre-Round-7 hardcoded `inner_edge_row_idx = 0` Cartesian BC that would put SIMP BCs inside the rib body instead of along its panel-facing edge. |
| **No lock-namespace collision** (HIGH-11; paired with the Round-5 → Round-1-continuation rename + Round-8 LOW-12 round-internal absorption) | `tests/test_audit/test_no_lock_namespace_collision.py` | Greps the doc for both label styles `\bC-[0-9]+\b` (Round 5 dashed) AND `\bC[0-9]+\b` (Round 1 no-dash) and asserts the dashed style yields zero matches outside Appendix D revision history. Same regex pair for `H-` vs `H` and `M-` vs `M`. **Round-8 LOW-12 extension:** also forbids bare round-internal labels (`\bCRIT-[0-9]+\b`, `\bHIGH-[0-9]+\b`, `\bMED-[0-9]+\b`, `\bLOW-[0-9]+\b`) in production prose; allowed only inside Appendix D revision history and explicit "Round-N CRIT-N" prefixed citations. Catches accidental re-introduction of the C-1..5 / H-6..9 / M-11..13 dashed namespace AND drift back to bare CRIT-N round-internal labels (which collide across review rounds). |
| **Retired phrases (architecture drift catcher)** (Round-8 v2 meta lock; paired with `docs/retired_phrases.yaml`) | `tests/test_audit/test_no_stale_architecture_refs.py` | Reads `docs/retired_phrases.yaml` — a catalog of architectural phrases retired across review rounds (e.g., "60 g mass cap", "120° spread angle", "{8, 10, 12, 14} blade counts", "PITCHING_OMEGA = 0.0 12.5664 0.0", "outer ribs mate", "crank"). For each entry, greps the doc + the schema for the pattern and fails on any match outside the entry's `allow_list` (revision history, CI-gate descriptions that reference the forbidden phrase, explicit "earlier draft" historical notes). One-time investment that pays back permanently: every future review round adds its retired phrases to `retired_phrases.yaml` and the gate then catches re-introduction automatically. |
| **Rib-bottom coplanar with panel-bottom (rib-flat only)** (Architectural E; C10 lock) | `tests/test_geometry/test_rib_bottom_coplanar_panel_bottom.py` | For any `print_orientation == 'rib-flat'` design, asserts `abs(rib.z_min − panel.z_min) < 1e-6 m` (panel and rib share a common bottom face at z = 0; one-sided corrugation +z only). Skips for `deployed-V` / `edge` orientations (midplane symmetry holds there). Catches accidental drift back to the symmetric-midplane convention that would put the rib bottom hovering 0.9 mm above the build plate and require unprintable support material under the full rib length. |

### 12.1 Repository Layout

```
fan-optimization/
├── pyproject.toml # project metadata, dependencies, tool config
├── environment.yml # conda env (needed for CadQuery, FEniCSx, SU2)
├── README.md # how to set up + run
├── CLAUDE.md # project context for Claude Code
├── .gitignore # excludes data/, results/, .venv, etc.
├── .pre-commit-config.yaml # ruff + black + mypy hooks
│
├── src/
│ └── fanopt/ # the main package
│ ├── __init__.py
│ ├── geometry/ # CadQuery generator + manufacturability
│ │ ├── envelope.py # Layer 1: outer envelope + Fourier LE/TE
│ │ ├── fields.py # Layer 2: louver, texture, edge, noise, TPMS
│ │ ├── primitives.py # Layer 3: capped 0-1 independent primitive
│ │ ├── manufacturability.py # §N7 11-check filter
│ │ ├── generator.py # orchestration: generate_blade(params)
│ │ └── schema.py # JSON schema + load-time validation
│ ├── topopt/ # rib TO (Reissner-Mindlin plate-bending)
│ │ ├── plate_bending.py # plate-bending element + assembly
│ │ ├── simp.py # SIMP material interpolation + filter
│ │ ├── solver.py # main TO loop (OC update)
│ │ └── loads.py # multi-load-case (push, return, inertial, click)
│ ├── cfd/ # SU2 wrappers + post-processing
│ │ ├── mesh.py # Gmsh wrappers (2D slice + 3D corrugated)
│ │ ├── configs.py # SU2 .cfg generators from Jinja2 templates
│ │ ├── runner.py # subprocess + checkpointing for Colab session limits
│ │ ├── j_fan.py # canonical metric (§9.4 locked spec)
│ │ └── parsers.py # SU2 history.csv + VTU parsers
│ ├── bo/ # BoTorch optimization
│ │ ├── multi_fidelity.py # SingleTaskMultiFidelityGP + qMFKG
│ │ ├── architecture_bandit.py # outer-loop categorical search (combined Tier -1/0 promotion)
│ │ ├── turbo.py # trust-region BO inner loop
│ │ ├── saasbo.py # SAASBO fallback (≤500 inducing points)
│ │ ├── pareto.py # qNEHVI + Pareto front analysis (4 objectives)
│ │ └── orchestration.py # full BO loop (Phase 4 entry point)
│ ├── physical/ # post-processing of physical measurements
│ │ ├── imu.py # angular work per cycle
│ │ ├── acoustic.py # FFT + dominant frequency
│ │ └── anemometer.py # J_fan from physical measurement
│ └── utils/
│ ├── ledger.py # Drive/JSONL ledger
│ ├── drive_io.py # buffered JSONL append, markers, heartbeat, listing cache
│ ├── slicing.py # slice_assignments_v{N}.json round-robin assignment + rebalance
│ ├── colab.py # Colab session helpers (resume, checkpoint)
│ └── logging.py # structured logging
│
├── tests/ # mirrors src/fanopt/ structure
│ ├── test_geometry/
│ │ ├── test_envelope.py
│ │ ├── test_fields.py
│ │ ├── test_manufacturability.py
│ │ └── test_generator_integration.py
│ ├── test_topopt/
│ │ └── test_plate_bending_cantilever.py # benchmark vs published
│ ├── test_cfd/
│ │ ├── test_j_fan_synthetic.py # canonical metric on known velocity fields
│ │ └── test_su2_config_generation.py
│ ├── test_bo/
│ │ ├── test_synthetic_branin.py # BO on known function
│ │ └── test_architecture_bandit.py
│ └── test_physical/
│ ├── test_imu_known_waveform.py
│ └── test_acoustic_known_tones.py
│
├── scripts/ # entry-point CLI scripts (one per spike / phase)
│ ├── run_spike_0_4.py # click feature 1000-cycle test
│ ├── run_spike_0_7.py # generative-geometry + BO-infra sanity check
│ ├── run_phase1_smoketest.py
│ ├── run_phase2_to.py
│ ├── run_phase3_correlation.py
│ ├── run_phase4_bo.py
│ └── run_phase5_verify.py
│
├── notebooks/ # Colab + local Jupyter (thin orchestrators)
│ ├── colab_phase4_runner.ipynb # Phase 4 BO inner loop on Colab Pro
│ ├── pareto_analysis.ipynb # 4D Pareto front inspection
│ ├── geometry_inspection.ipynb # CadQuery generated-blade preview
│ └── physical_results.ipynb # IMU + acoustic + anemometer summary
│
├── configs/ # external tool configs (templated, not Python)
│ ├── su2/
│ │ ├── slice_steady.cfg.j2 # Tier -1 template
│ │ ├── slice_unsteady.cfg.j2 # Phase 3 inner-loop verification
│ │ ├── fan3d_steady.cfg.j2 # Tier 0 template
│ │ └── fan3d_unsteady.cfg.j2 # Tier 1 template
│ └── fusion/
│ └── fan_script.py # Fusion add-in (multi-blade assembly view)
│
├── data/ # .gitignore'd by default
│ ├── designs/ # JSON parameter files (one per design point)
│ ├── results/ # CFD outputs, TO outputs
│ ├── meshes/ # Gmsh outputs
│ └── physical/ # IMU CSVs, anemometer logs
│
└── docs/
 ├── plan_R11.md # this project plan (mirror of report-final.md)
 ├── phase_logs/ # phase-by-phase decision logs
 └── api/ # auto-generated from docstrings (sphinx)
```

### 12.2 Notebook Discipline

Notebooks **never contain real logic.** They are ~50-line orchestration files that:

- Import from the main package: `from fanopt.bo import architecture_bandit, turbo, pareto`
- Set up paths and parameters
- Call high-level functions: `pareto = run_phase4_bo(seed_data, config)`
- Visualize results: `plot_pareto_front(pareto)`

If a notebook starts growing logic, it's a signal to refactor that logic into `src/fanopt/`.

### 12.3 Tooling Choices (locked)

| Tool | Role | Notes |
|------|------|-------|
| **pytest** | Testing framework | Flexible, good fixture support; `pytest-cov` for coverage reporting. Target ~80% line coverage on `geometry/`, `topopt/`, `cfd/` modules. |
| **ruff** | Linting + formatting | Replaces black + isort + flake8 in one fast tool |
| **mypy** | Static type checking | Strict mode for new code; gradual for ported code |
| **pyproject.toml** | All project config | No `setup.py`, no `setup.cfg` |
| **pre-commit hooks** | Enforce style on every commit | `.pre-commit-config.yaml` configures ruff + mypy hooks |
| **conda** (via `environment.yml`) | Environment management | Required for CadQuery + FEniCSx + SU2; `uv` could be used for the pure-Python subset but the geometry/TO/CFD stack needs conda |
| **GitHub Actions** | Automated CI test runs on push | Runs the synthetic-data tests (geometry, BO Branin, j_fan known fields, IMU known waveforms); skips CFD-dependent tests (those run on Colab) |

### 12.4 Test Strategy

The test pyramid emphasizes fast, hermetic tests that run in CI:

- **`tests/test_geometry/`** — CadQuery generator unit tests + manufacturability filter checks + a small set of "golden" generator outputs (10 random seeds + 10 fixed JSONs producing known STLs).
- **`tests/test_topopt/`** — `test_plate_bending_cantilever.py` solves a standard published benchmark (Cantilever-beam plate-bending TO) and compares the converged compliance to published values within 5%.
- **`tests/test_cfd/`** — `test_j_fan_synthetic.py` verifies the `j_fan.py` post-processor against analytical velocity fields (uniform flow, oscillating sinusoid, vortex pair) where J_fan is known in closed form. `test_su2_config_generation.py` snapshots the rendered .cfg files for the four Jinja2 templates.
- **`tests/test_bo/`** — `test_synthetic_branin.py` runs the full multi-fidelity BO + architecture bandit + TuRBO on the synthetic Branin function (10D embedding); verifies the BO converges within 20% of the known optimum in ≤50 iterations. `test_architecture_bandit.py` exercises the combined Tier -1 + Tier 0 promotion logic on synthetic objective values.
- **`tests/test_physical/`** — `test_imu_known_waveform.py` feeds simulated IMU signals (pure sinusoid + amplitude-modulated sinusoid + recorded noise) and verifies `angular_work_per_cycle` matches analytical expectations. `test_acoustic_known_tones.py` verifies the FFT post-processor extracts known frequencies from synthetic 10-second audio.

### 12.5 Phase 0 Scaffolding Step

See §Phase 0 Step 0.0 — the canonical Step 0.0 description lives there. The earlier duplicate `Step 0.0 detail` block was deleted per M8 (it risked drift); cross-reference rather than re-describe.

The scaffolding step takes ~1-2 hours and produces a working CI pipeline on a brand-new repo. All subsequent spikes (**0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6c (H10), 0.6d (H10 supplement; 2026-05-14 addition), 0.7**) commit their scripts and outputs into the scaffolded structure, so by the end of Phase 0 the project is ready for Phase 1.

---

## 13. Future Work, V2 Backlog, and Baselines

This section consolidates the project's deferred items: the empirical and theoretical **baselines** that all V1 reporting compares against, the **Phase 5 deliverables** that are cheap-or-free additions to the existing campaign, and the **V2 backlog** of items deliberately out-of-scope for V1 but queued for the next revision. The detailed V2 entries also live in `docs/V2_backlog.md` (the canonical V2 plan); this section is the in-spec summary.

### 13.0 V1/V2 scope pivot (2026-05-13)

**Hardware-instrumented measurement is deferred to V2.** The operator (running this as a personal project) does not own and will not purchase anemometer, IMU, or torsional-pendulum measurement hardware. The decision record is `docs/phase_logs/phase_0_signoff.md`; per-spike sentinel files live at `data/spike_0_{2,3,5,7c}/deferral.json`; the full V2 specs are in `docs/V2_backlog.md` under "Deferred Phase-0 spikes".

**Deferred Phase-0 spikes (V1 substitute → V2 trigger):**

- **Spike 0.2 (torsional pendulum I_wrist measurement).** V1 uses the analytic `I_wrist_kgm2` from the §6.4 generator (`i_wrist_assembly`). Spike 0.4 force balance reads it via `scripts/run_spike_0_4.py --i-wrist-analytic <value> --f-friction-cumulative-n <value>`, with the safety factor bumped 2× → 3× to absorb unverified-inertia uncertainty. V2 revisit trigger: V1 ships a fan that subjectively feels better than baseline AND the operator wants quantitative confirmation.
- **Spike 0.3 (anemometer + IMU baseline).** V1 substitutes two co-baselines: (a) Phase 2a baseline CFD on the flat-panel 10-blade design as a sim-side baseline (every Phase-4 optimized design's simulated `J_fan` is compared against this number); (b) Phase 6 blinded A/B feel test of printed top-3 vs. printed baseline. Optional V2 upgrade paths in order of cheapness: kitchen scale + cardboard target (see `docs/spike_0_3_protocol.md` Appendix A; essentially free), Phyphox phone IMU (free; phone already owned), full anemometer rig.
- **Spike 0.5 (3-copy fab-noise CV).** V1 substitutes a same-design duplicate-print sanity check at Phase 6 — print one top candidate twice; compare by feel. If different, the print-noise floor is wider than the V1 gain target and the comparison is flagged.
- **Spike 0.7c (Sobol-vs-BO iso-compute baseline).** V1 substitutes a BO-stall fallback: if Phase 4 Tier-0 best-J_fan does not improve over 20 consecutive acquisitions within an architecture, switch to hand-picked diverse candidates spanning Layer 2 archetypes.

**Spikes still required for V1:** 0.0 (done), 0.1 (done), 0.4, 0.6, 0.6a/0.6b (gates), **0.6c.1 (non-negotiable — Phase 4 launch gate; 0.6c.2 deferred to Phase 5 per 2026-05-14 decision, see §Phase 0 Spike 0.6c notes and Phase 5 step 62.5)**, **0.6d.1 + 0.6d.2 (2026-05-14 addition — Tier-1 quantitative-sanity counter-checks; gate Phase 4 launch alongside 0.6c.1; 0.6d.3 advisory)**, 0.7a, 0.7b. The 15-30% gain target in §0 row 35 is **suspended for V1**; V1 reports sim-vs-sim relative gain plus the operator's qualitative feel comparison. The numerical gain target re-enters in V2 once a quantitative baseline exists.

**Cheap mitigations adopted at decision time** (no hardware cost):

1. **Diverse Phase 5 print candidates.** Top-3 must span Layer 2 archetypes (e.g., one louver-heavy, one TPMS-heavy, one near-baseline) — not 3 variations of one shape. Mitigates "BO exploits a sim artifact" failure mode.
2. **Print one top candidate twice.** Same-design sanity check at Phase 6 substitutes for the deferred Spike 0.5 fab-noise CV.
3. **Blinded A/B Phase 6 protocol.** Operator has someone else hand them fans without naming them; stopwatch-paced 20 strokes; 1-5 score on airflow / weight / sound / aesthetics; repeat on a different day. Catches confirmation bias for $0.
4. **BO-stall fallback.** Hand-picked diverse candidates if Phase 4 stalls; bounded by the 1300-h Phase-4 stop rule regardless.

### 13.1 Baselines (Phase 0 + Phase 5 reporting; no extra compute)

**Spike 0.3 empirical baseline (kept):** 10-blade flat-panel fan, IMU-normalized `J_fan_measured / W_cycle`. The 15-30% gain target is measured against this baseline. The protocol lives in §Phase 0 Spike 0.3.

**Theoretical momentum-piston upper bound (NEW in V1; add to Phase 5 reporting):**

```
J_max ≈ ρ_0 · V_tip · A_panel · duty_factor
```

With `ρ_0 = 1.225 kg/m³`, `V_tip = 2.20 m/s`, **A_panel ≈ 0.053 m² (M19 trapezoidal correction)** = trapezoidal per-blade area ∫_{HUB_RADIUS}^{L_blade} panel_width(r) dr ≈ ½·(b + B)·L_blade · 10 blades with b = 5 mm (pivot end), B = 45 mm (tip), L_blade = 0.200 m: per-blade area = ½ · (0.005 + 0.045) · 0.200 = 5.0e-3 m²; × 10 blades = 0.050 m². With the inner 20 mm HUB region (panel-only, narrow ~12 mm) added: A_panel ≈ **0.053 m²**. The earlier rectangular `0.045 × 0.20 × 10 = 0.09 m²` over-counted by ~70%. `duty_factor = 0.5` (productive half of cycle):

```
J_max ≈ 1.225 · 2.20 · 0.053 · 0.5 ≈ 0.071 N
```

This is the absolute physical ceiling under the trapezoidal panel geometry — a perfectly asymmetric fan that traps all the swept air on the productive stroke and feathers fully on the return. **Phase 5 reporting includes the ratio `J_fan_pareto / J_max` per Pareto point** so readers can see how far the optimizer reached toward the physical limit. No additional compute (post-hoc on Phase 5 outputs).

*Calculation history:* the earlier compaction summary cited `J_max ≈ 0.011 N` (order-of-magnitude low — missing the tip-widening). The round-2 spec corrected this to ~0.121 N using rectangular A_panel ≈ 0.09 m². The current M19 lock corrects A_panel to the actual trapezoidal area 0.053 m², giving J_max ≈ **0.071 N**. Any reported `J_fan_pareto > 0.071 N` indicates a sign error, integration-window bug, or CFD numerical artifact.

### 13.2 Pareto-by-architecture reporting (Phase 5 deliverable; no extra compute)

Phase 4 step 56's merged JSONL already tags every row with its architecture (blade count, Layer 2 activation profile, edge profile, print orientation, layer height). A post-campaign analysis script slices the 4D Pareto front by architecture and reports a "Pareto front per architecture" comparison. Useful for the user to see (a) whether the K = 3-5 promoted architectures cleanly separate or overlap, and (b) for V2 to know which architecture categories underperformed.

**Cost:** ~1-2 hours scripting, runs on the M3, no CFD. Stored at `results/phase5_<date>/pareto_by_architecture.ipynb`.

### 13.3 V2 backlog (queued, NOT in V1 scope)

The canonical V2 plan lives in `docs/V2_backlog.md`; this list is the in-spec summary. Each entry has a triggering condition that decides whether V2 work begins.

1. **Mid-Phase-4 rib re-tune.** Re-trigger Phase 2 rib SIMP TO every K Phase-4 architecture promotions, conditioned on the panel topology the architecture-bandit is actually selecting. Phase 2 currently runs once with smooth-baseline panel placeholder; V2 would re-tune once Phase 4 reveals which Layer 2 fields are winning. **Timing:** re-tune happens after the first K promotions complete (≈ Phase 4 month 1), not "mid-Phase-4" unspecified. **Cost:** ~3-5 additional Phase 2 SIMP solves × ~30 min each = 1.5-2.5 hours per re-tune; the cost is per-architecture-class, not per-design, so it scales with K not with total evaluations. **Trigger:** L7 diagnostic shows >15% Tier-0 ranking drift after first K promotions.

2. **V2 designed lock mechanism.** If Phase 6 testing shows the fan unlocks under sustained 2 Hz waving (the H6 V1 force balance passes in Spike 0.4 but fails in practice), V2 lands a designed lock. **Currently V1 has the rib-tab fallback armed conditionally** (`params.layer4.v1_lock_fallback_enabled`).

3. **Textured-PEI bed-surface portability.** §3.2.4 / M13 currently lock smooth-PEI (Bambu Cool Plate Super Tack AP05). Switching to textured PEI (Ra 10-30 µm) requires re-running the §3.2.4 wall-roughness calibration. V2 would document the calibration procedure so users on other bed surfaces (Prusa textured PEI, Anycubic frosted PEI) can re-derive the roughness-model parameters.

4. **Alternative MFBO architectures for TPMS/noise.** §6.2.2 documents but does not adopt two cleaner alternatives (currently handled by the 0.3/0.7 reweighting compromise): **(a)** disable multi-fidelity GP for TPMS/noise architectures (run Tier-0-only single-fidelity); **(b)** treat Tier -1 as a separate cheap-feature input rather than a fidelity column. **Trigger:** if the L7 empirical-bias diagnostic fires often in Phase 4 (mean `|Δ_TPMS / mean_J_fan_tier0| > 0.30`), V2 adopts (a) or (b) based on which pattern emerges.

5. **Centrifugal pivot stress as a real Filter (re-introduce Filter 4).** Filter 3 is a deprecated pass-through stub. If Phase 6 testing reveals fatigue failures driven by centrifugal pull at the pivot under aggressive waving (the kind the canonical Filter 2 misses), V2 introduces a proper Filter 4 with the correct kinematics (cyclic tangential reaction at click detent + centrifugal at boss, both computed with H8 wrist-to-tip lever arms).

6. **Asymmetric-stroke physics in J_fan.** The current J_fan metric is symmetric in time (integrates over full cycles). Real waving may have a deliberate productive-bias (user pushes harder than they return). V2 could explore an asymmetric weighting `J_fan_biased = w_p · J_productive_half + (1 − w_p) · J_return_half` with `w_p` measured from IMU. Would change the optimization target away from the parachute baseline more aggressively.

7. **`directional_asymmetry_score` functional-form refinement (closes the §Phase 3 step 33 C6 starter form).** V2 should converge on the form that best predicts the 3-radius J_fan spread — candidates: weighted sum of Layer 2 louver angles only, Fourier TE/LE phase difference only, integrated `|chord_z⁺(x) − chord_z⁻(x)|` over the planform (camber asymmetry), or the starter sum-of-three form from V1. V2 fits all candidates on the Phase 6 IMU-measured data (richer signal than the Phase 0 calibration sample) and picks the one with highest R².

8. **Elastic-Winkler rib-panel BC for Phase 2 SIMP TO (Architectural B; V1 uses rigid-panel Dirichlet).** Replace the V1 Dirichlet "fix all rib-panel interface DOFs" with an elastic Winkler-foundation BC where the rib's z-DOFs at the interface are coupled to a spring stiffness `k_panel` representing the local compliance of the Phase 2b-generated panel (with Layer 2/3 cutouts). Calibrate `k_panel` once from a Phase 0 FEA on a representative skeletal panel; embed in `phase2_rib_to_solver.py`. Removes the V1 over-stiffness assumption that under-builds the rib for the real panel. **Trigger:** V1's §59.5 gate rejecting >20% of Pareto top-3 candidates for rib-under-build (the rigid-panel BC's blind spot).

### 13.4 V2_backlog.md scaffold

The companion file `docs/V2_backlog.md` is the canonical, expanded V2 plan and has the following sections (created at Phase 0 alongside the rest of the doc scaffold):

```
# V2 Backlog

## Triggered items (V1 fail → V2 in-scope)
- V2 designed lock (trigger: Phase 6 unlocking under sustained 2 Hz)
- Centrifugal Filter 4 (trigger: Phase 6 centrifugal-driven fatigue failure)
- Alternative MFBO for TPMS/noise (trigger: L7 diagnostic > 30%)

## Optional (V1-complete, V2-improves)
- Mid-Phase-4 rib re-tune
- Textured-PEI portability
- Asymmetric-stroke J_fan
- directional_asymmetry_score functional-form refinement

## Out-of-scope (V3+ or research)
- Active electronic flow control
- Multi-DOF wrist motion (yaw + pitch + roll)
```

The trigger-tagged items are checked at Phase 6 wrap-up; if any trigger fires, the V2 effort begins with the corresponding entry as its first deliverable.

---

## Appendix A: Quick-Start Decision Flowchart

```
START: Which scope and tool stack?
|
|-- (V-unit blades + 4-layer hybrid generative design, single-material PETG, no FSI):
| |
| |-- Fusion (parametric per-blade STL + FEA verification)
| | + CadQuery (generative blade generator: outer envelope + Boolean
| | subtraction primitives + surface features + manufacturability filter; §9.7)
| | + FEniCSx plate-bending TO (rib only; panel topology emerges from
| | Boolean subtractions, not SIMP density)
| | + SU2 (compressible + low-Mach prec; rigid corrugated geometry;
| | locked dt=T/200, 5 cycles, J_fan in §9.4)
| | + Multi-fidelity stack: 2D steady (tier -1, ~5 min) +
| | 3D steady ranking-only (tier 0, ~30-90 min) +
| | 3D unsteady true J_fan (tier 1, ~3-6 hours).
| | QSST dropped (cannot represent generative topology).
| | + Architecture bandit (K=3-5 data-driven; default K=4 if Phase 3 R² in [0.5, 0.6)) + continuous TuRBO
| | inner loop (~37-46 dims total, ~20-30 continuous per architecture)
| | + PyFR (GPU verification on TOP-3 Pareto designs)
| | + IMU instrumentation + acoustic measurement in Phase 6
| | + 3-copy fab-noise study (Spike 0.5; CV < 5%)
| | + Spike 0.4 click feature 1000-cycle test
| | + Spike 0.7 generative-geometry + BO-infrastructure sanity check (NEW)
| | + Drive/JSONL ledger + JSON-driven design specs
| |
| | Materials: single-material PETG (no TPU, no FSI in baseline)
| | Claude writes: ~35-50 Python scripts + ~12-15 config files + ~3-4 Fusion add-ins
| | User runs: scripts (Mac + Colab), per-blade prints of top-3 Pareto designs,
| | IMU + acoustic tests on each
| | Timeline: ~10-13 weeks
| | Cost: $0 + Colab Pro subscription
| | Compute: Mac (dev) + Colab Pro (~300-600 hrs favorable / 500-800 expected over 2-4 wks
| | with 2-4 parallel sessions; G4 GPU 95 GB VRAM for PyFR p=3 top-3 — T4 16 GB insufficient per HIGH-11 Round-9 lock)
| |
|-- Prior draft (rich parametric generative design via Boolean subtraction primitives,
| flat ~45-55-var parameterization, Timeline ~10-13 weeks):
| | Superseded by the 4-layer hybrid that addresses the "Mr. Potato Head"
| | alignment problem, gives the optimizer direct access to asymmetric-drag
| | design families (louvers), and reduces CAD failure rate from ~5% to <2%.
| |
|-- Prior draft (aero-sensitivity-weighted plate-bending TO with 2.5D skin breakthroughs,
| Timeline ~11-13 weeks):
| | Rejected because adjoint linearization is invalid for large TO changes
| | (the whole point), and mixing TO with aero on the panel entangled structural +
| | aerodynamic problems.
| |
|-- Prior draft (multi-material with TPU membrane, FSI, Timeline ~15-17 weeks):
| | Rejected due to membrane folding paradox + TPU spring-back.
| |
|-- Prior draft (fabric membrane, coupled-rib TO): Timeline 15-16 weeks, superseded.
|-- Prior draft (SIMPLIFIED pure-Python, open questions deferred): 2-4 months, superseded.
|-- AGGRESSIVELY SIMPLIFIED (steady-only, single representative blade):
 | Timeline: ~6 weeks. Loses: unsteady ranking, verification, acoustic data.
```

## Appendix B: Estimated Computation Times

| Task | Hardware | Estimated Time | Notes |
|------|----------|---------------|-------|
| Fusion regeneration (10 V-unit blades from JSON) | Mac | 10-30 seconds | Headless add-in, full quality |
| **CadQuery generative blade generator** (Boolean subtraction + envelope + features) | Mac | **5-30 seconds per blade** | centerpiece; Boolean ops are the bottleneck. Manufacturability filter runs in <1 s. |
| FEniCSx 2D plate-bending TO single rib (Phase 2) | Mac | 5-30 minutes | Reissner-Mindlin; rib only (panel topology from generative pipeline) |
| Fusion Simulation FEA verification | Mac | 2-5 minutes | Orthotropic PETG, mesh refinement at pivot + click features |
| CalculiX FEA verification | Mac | 1-5 minutes | Backup path |
| Modal analysis (10 modes) | Mac | 2-10 minutes | First bending mode > 10 Hz check |
| Gmsh meshing (2D corrugated cross-section + Boolean subtractions) | Mac | 1-3 minutes | Phase 3 slice; includes Boolean subtractions |
| Gmsh meshing (3D deployed corrugated fan + subtractions, ~500K cells) | Colab Pro | 10-25 minutes | Phase 4 tier 0/1 |
| Gmsh meshing (3D resolved corrugated + subtractions, ~2-3M cells) | Colab Pro | 25-60 minutes | Phase 5 top-3 verification only |
| **SU2 CFD 2D STEADY slice** (Tier -1) | **Colab Pro CPU** (8 vCPU; standard runtime; 2-4 parallel sessions, 4-8 SU2 procs/session — **routing**) | **~5 minutes per eval** | Architecture-bandit screening at 30 evals/architecture; cheap by design; no GPU |
| SU2 CFD 2D unsteady slice (inner-loop verification only; not a screening tier) | **Colab Pro CPU** (8 vCPU; standard runtime) | 30-90 minutes | 5 cycles, dt=T/200; captures asymmetric drag; used in Phase 3 + ad-hoc inner-loop calls |
| SU2 CFD 3D steady (500K cells, low-Mach prec) | **Colab Pro CPU** (high-RAM if mesh > 1 M cells; 2-4 parallel sessions, 1-2 SU2 procs/session) | 30-90 minutes | Phase 4 tier 0 (ranking only) |
| SU2 CFD 3D unsteady (5 cycles, dual time) | **Colab Pro CPU** (standard or high-RAM; 1 SU2 proc/session; checkpointed) | 3-6 hours | Phase 4 tier 1; checkpointed |
| SU2 CFD 3D unsteady verification (resolved corrugated mesh + subtractions) | **Colab Pro CPU** (high-RAM; long-running with checkpoints) | 6-12 hours per design | Phase 5; top-3 |
| **Combined-blade structural FEA (canonical + stress-test) ** | **MacBook M3** (FEniCSx/CalculiX, orthotropic PETG, local) | **~5 min per design × 3 designs × 2 load cases = ~30 min total** (L15: was 15 min — missed the stress-test load case) | Runs locally on the M3 *before* any Colab SU2 verification; gates step 65+ |
| GP surrogate training (per-architecture, **~115 samples per architecture in ~20-30D continuous subspace**, where the 37-46D is the dimensionality of the *union* across architectures, not the per-architecture GP fit. ~500 samples is the SAASBO inducing-point cap if SAASBO is exercised on a larger pooled dataset.) | Mac | **30-60 seconds** | Validated in Spike 0.7b before Phase 4 commits |
| BO acquisition optimization (multi-fidelity KG, **~37-46D**) | Mac | 30-90 seconds | Per iteration |
| Architecture-bandit Tier -1 screening (~80-130 architectures × 30 evals each, H1 retuned) | Mac driver + **Colab Pro CPU** (2-4 parallel sessions; **NOT GPU** — routing) | **6-12 wall-clock hours (CPU only)** | Outer-loop screening; GPU credits stay untouched at this tier |
| **Spike 0.6c — Tier 1 cfg sanity (H10; V1 reduced scope, post-2026-05-14)** | Colab Pro CPU | **~1-2 hours** (Spike 0.6c.1 cfg sanity only; 0.6c.2 published-benchmark deferred to Phase 5 step 62.5) | Phase 0; gates Phase 4 launch (with Spike 0.6d). NOT booked against the 1000-h Phase 4 stop rule. |
| **Spike 0.6d — Tier-1 quantitative-sanity counter-checks (2026-05-14 addition; H10 supplement)** | Colab Pro CPU | **~5–9 hours total** (1–2 h 0.6d.1 + 2–4 h 0.6d.2 + 2–3 h 0.6d.3 advisory) | Phase 0; gates Phase 4 launch alongside 0.6c.1. NOT booked against the 1000-h Phase 4 stop rule. |
| **Phase 5 step 62.5 — body-in-still-air published-reference benchmark (2026-05-14 addition; replaces deferred 0.6c.2)** | Colab Pro CPU (SU2 + OpenFOAM) + Colab Pro G4 GPU (PyFR) | **~14-26 hours total** (SU2 6-12 h + PyFR 2-4 h + OpenFOAM 6-10 h) | Phase 5; bundles with existing top-3 PyFR verification. |
| **Spike 0.7c — Sobol vs BO comparison (H7)** | Colab Pro CPU | **430 h** (30 + 100 + 300 h serial budgets) | Phase 0; doubles as Phase 4 GP seed. NOT booked against the 1000-h Phase 4 stop rule. |
| **C6 radial-correction calibration** | Colab Pro CPU | **~50 h one-time** | Phase 0; alongside Spike 0.7b. NOT booked against Phase 4. |
| LHS init batch (per promoted architecture: 30 steady + 5 unsteady) | **Colab Pro CPU** (high-RAM for Tier 0/1 if needed) | 1-2 wall-clock days per architecture | × 4 promoted architectures |
| TuRBO inner loop (**~35 acquisitions per architecture, cap; early-stop fires first**) | Mac driver + **2-4 parallel Colab Pro CPU sessions** | 2-3 wall-clock weeks total | **~300-600 favorable / 500-800 expected hours** (see §6.2.3); CPU only |
| PyFR p=3 DG verification (top-3 Pareto designs) | **Colab Pro G4 GPU (95 GB VRAM) — the ONLY GPU step in the stack; T4 is insufficient per HIGH-11 Round-9 lock** (14-18 GB working set; T4 16 GB OOMs, G4 has 5-6× headroom) | 2-4 hours per design | 3 designs |
| IMU CSV → angular work per cycle | Mac | <1 second | Phase 6 post-processor |
| Acoustic FFT (10 s recording → dominant tone) | Mac | <1 second | Phase 6 post-processor |
| Per-blade PETG print | FDM printer | 1-2 hours per blade | 10 blades × 3 Pareto designs = 30-60 h total |
| Full-assembly PETG print (10 blades in one job) | ≥360 mm-bed FDM printer | 8-12 hours | Alternative to per-blade |

## Appendix C: Folding Fan vs. Paddle Fan -- Key Differences for Optimization

| Optimization Aspect | Paddle Fan (original report) | Folding Fan (V-unit blades, 4-layer hybrid generative panel, single-material PETG) |
|---------------------|------------------------------|----------------------------|
| TO design domain | Entire blade (single large 2D/3D domain) | One V-unit blade (2 ribs + 1 panel as one body) — exact symmetry applies the result to all 10 blades |
| TO problem size | Large (500K+ elements) | Small (~50-150K plate-bending elements per blade) |
| TO compute time | 2-4 hours per run | 5-30 minutes per blade (Phase 2 rib-only plate-bending) |
| TO formulation | 3D voxel SIMP | **2D plate-bending SIMP (Reissner-Mindlin)** |
| TO focus | Internal lattice structure | 2D planform cutouts + variable panel thickness + stress-constrained pivot + preserved click-feature footprint |
| ASO domain | Continuous solid surface | Corrugated rigid surface (10 cambered panels separated by 11 rib ridges); per-blade airfoil shape applied to all blades by exact symmetry |
| ASO + per-blade parameters | Planform, camber, thickness distribution | **~37-46 vars **: Layer 1 envelope + Fourier (~14) + Layer 2 macro-pattern + procedural math fields with 5-field library, 0-3 active (~15-20) + Layer 3 capped 0-1 primitive (~5-7) + Layer 4 manufacturing + click (~3-5) |
| CFD model | Standard bluff body | Compressible + low-Mach prec on rigid corrugated geometry; **no FSI in V1 baseline** |
| Structural failure mode | Distributed bending of plate | Concentrated stress at pivot hole + blade bending + click-feature fatigue |
| Critical stress location | Blade root | **Panel pivot hole (K_tt = 2.42 at d/w = 0.25 in the 12 mm boss; bearing mode binds first at 2.00 MPa Z)** — pivot is in the panel, not the rib — and rib-panel fillet + click-feature detent surfaces |
| Assembly | None (monolithic print) | 10 single-material PETG blades + 1 steel/brass pivot pin (≥45 mm; -Pin) + **3 mm outer-face reinforcement strip on each guard blade** |
| Print strategy | Single large flat print | Per-blade prints (default, ≤256 mm bed) OR full-assembly print (≥360 mm bed) |


---

## Appendix D: Revision History

Each line below is one locked decision in the current spec and a pointer to where it lives in the body. The latest value is the canonical spec; full chain history is not reproduced here.

### Needs reconciliation (un-superseded coexisting locks; future revision picks)

1. ~~Bearing vs tension allowable at the same hole.~~ **RESOLVED.** Both allowables coexist as per-mode checks: **5.58 MPa for tension (XY-plane, K_tt = 2.42 in 12 mm boss), 2.00 MPa for bearing (Z-direction, σ_y_Z = 30 MPa), 4.22 MPa for bending (XY, K_tb = 3.2)**. Filter 2 + §59.5 evaluate each mode independently; the **lowest (bearing 2.00 MPa) binds for the canonical baseline**. See §3.1.5 K_t table + §6.3.1 Filter 2.
2. ~~`panel_w ≥ 12 mm` tear-out vs 6-8 mm panel width.~~ **RESOLVED.** Locked V1 fix is local boss thickening: `make_panel_solid` in §9.7 adds a 12 mm-OD circular boss × panel_thickness centered on the pivot hole; the inter-rib panel stays 6-8 mm. The boss is part of the panel-domain mask (Layer 2/3 cannot carve into it). See §6.3.1 Filter 2.
3. ~~`physics_hash` contents split.~~ **RESOLVED.** Single canonical definition in §9.4.1: `physics_hash := blake2b(config_hash || materials_blob || motion_blob)` covering CROSS_TIER ∪ TIER_SPECIFIC ∪ MATERIAL_LOCKS ∪ MOTION_SPEC. `config_hash` is a strict subset (CFD-config keys only); `physics_hash` is the superset that also covers materials and motion. Every JSONL row carries both.
4. ~~K-decision N ≥ 30 vs 1000 h budget stop.~~ **RESOLVED.** If 1000 h is reached with N < 30 in the (Tier 0, Tier 1) overlap, K stays at its current value and the orchestrator force-grows the overlap by scheduling Tier-1 evaluations on existing Tier-0 points until N ≥ 30, then decides K. Budget extends by the force-grow cost; the K-drop rule resumes after K is decided. See §0 row "Architecture promotion K" + §6.2.2.

### Locked decisions (body destination = canonical spec; this row = one-line history)

| Topic | Current spec | Body section |
|-------|--------------|--------------|
| Fan architecture | Discrete V-unit blades + click-mating **panel outer tangential edges** (per item #3 panel-edge relocation; click does NOT live on the rib) | §1.1, §2.2, §0 rows 17 + 139 |
| Material | Single-material PETG except steel/brass pivot pin (~2.5 g) | §2.1, §6.4, §7 |
| Wrist-axis convention | +y (wrist-flexion hinge); forearm direction +x; pivot pin axis = +z (stacking) | §3.2.0, §6.4 |
| Coordinate convention | +z streamwise, FREESTREAM_DIRECTION=(0,0,±1), PITCHING_OMEGA=(0,ω,0), PITCHING_AMPL=(0, 0.6981, 0) | §3.2.0, §9.4.1 |
| Pivot architecture | Pin in +z through panel at y=0; 10 panels stack | §2.1, §2.3, §3.1.2 |
| Panel thickness | 2.2-3.8 mm at 3 control points (hard schema bound) | §2.1, §6.2.1 |
| Folded form factor | 22-42 mm at 10 blades | §1.1, §2.3, §6.4 |
| K_t at panel pivot hole | **2.42** (Peterson polynomial, d/w = 0.25 in 12 mm boss) | §3.1.5 |
| Cyclic allowables at panel pivot (3 modes, no scalar sum) | Tension **5.58 MPa** / bending **4.22 MPa** / bearing (Z) **2.00 MPa** (binds) | §3.1.7, §10.1 |
| K_t hotspot table (8 hotspots, 3 modes at panel pivot + click Z-floor) | Panel pivot tension K_tt = **2.42** / bending K_tb = 3.2 / bearing K_t_bearing = 1.5 (Z) / click detent (XY) 3.0 / click detent (Z-floor) 3.0 / TPMS hole 2.5 / fillet 1.5 / slot end 2.0 | §3.1.5 |
| r_CoM_wrist constraint | ≤ 0.160 m (= d_handle + 0.55·L_blade) | §6.4 |
| Pareto objective 2 | Minimize I_wrist about handle-grip +y axis | §6.4 |
| Lock mechanism | Deferred to V2; V1 has 3 mm outer-face reinforcement strip | §2.1, §2.2, §2.3 |
| Cycle count | 5 canonical, extend to 8 if cycle-2 vs cycle-3 > 5%; SE = std/√(n_cycles − 1) (= std/2 at 5 cycles, std/√6 at 8); diagnostic only | §9.4.1, §Phase 3 step 39 |
| Architecture K | K ∈ {3,4,5} from Spearman ρ² on pooled (Tier 0, Tier 1), N ≥ 30 | §6.2.2 |
| Acquisition cap | 35 hard cap; early-stop fires first | §6.2.2 |
| GP noise | **Fixed-floor epistemic** `train_Yvar = EPISTEMIC_NOISE_FLOOR` (per-tier scalar; NOT per-observation J_fan_se²); per-design SE stored as JSONL diagnostic only | §0, §6.2.3 |
| Design hashing | blake2b/12 with leaf-walking precision pre-walk | §6.2.5 |
| JSONL schema | v2 with `schema_version`, `physics_hash`, `material_hash`, `J_fan`/`J_fan_delta`, retriable + retry_count, `cfl_max` | §Phase 4 step 51 |
| Failure-code taxonomy | §9.4.2 retriable table including `transient_infra`, `cfl_excursion` | §9.4.2 |
| Filter chain (pre-CFD) | Filter 1 (mass + r_CoM + manufacturability) → Filter 2 (multi-mode struct: tension + bending + bearing + tear-out) → Filter 3 (deprecated, pass-through) → CFD dispatch | §6.3.1 |
| TPMS LE/TE protect | smoothstep mask `max(5, 0.05·chord)` baseline; widened to `max(10, 0.10·chord)` where local Re_x < 1e4 | §9.7.2 |
| Stress-test load | 2.5× p_aero + 2× α_max static; pass at < 12.4 MPa static and first bending mode > 10 Hz | §6.3.1, Phase 5 step 59.5, Spike 0.4, Phase 6 step 82 |
| Spike numbering | Spikes 0.1-0.7 + 0.6c + 0.6d (sequential; 0.6c added per the H10 Tier-1-cfg validation lock; 0.6d added 2026-05-14 as H10 supplement after the 0.6c.2 → Phase 5 deferral) | §Phase 0 |
| Plano-convex envelope | rib-flat → bottom face flat; deployed-V → both faces may curve | §6.2.1, §9.7.1 |
| 2D-cascade slip walls | `cascade_wall` marker + `MARKER_SYM` in slice configs | §9.6, §Phase 3 step 33 |
| Hypervolume early-stop | 50-round HV gain / HV_baseline < 1e-4; hard floor 500 rounds; HV_baseline fixed at round 200 | §6.4 |
| Dynamic-load assertion (Phase 2 one-shot) | α · m_rib · r_tip · N_blades < 0.1 · click_detent_allowable | §6.3.1 |
| Mid-iteration recovery | GP checkpoints + top-100 raw CFD + git tags + physics_hash + V2 backlog + V1 calibration | §Phase 4 step 56, §Phase 6 |
| Compute budget rebaseline | 600-1100 h expected / 1300 h pessimistic / 300-600 h favorable; 1000 h K-drop stop rule | §6.2.3 |
| Compute hardware routing | M3 + Colab CPU all SU2 tiers + Colab GPU PyFR only | Appendix B |
| Campaign tracker | Drive + per-session JSONL + content-hashed pointer + claim sentinel + heartbeat | §Phase 4 step 48, §12.1 |
