# Executive Summary: V-Unit Generative Design Folding Fan — Rib TO + 4-Layer Hybrid Panel Optimization Pipeline

**Date:** 2026-05-12 | **Revision:** R11 | **Source:** report-final.md

---

## Key Finding

R11 restructures R10's flat ~45-55-var Boolean-subtraction parameterization into a **4-layer hybrid (~37-46 vars)** while preserving every R10 architectural decision (V-unit blades, single-material PETG, click features, plate-bending rib TO, multi-fidelity BO, all 19 prior-review fixes). The change addresses three reviewed concerns: BO can't reliably converge correlated independent primitives ("Mr. Potato Head" alignment problem), rigid hand fans need direct access to angled-louver-style asymmetric-drag design families, and OpenCASCADE Boolean operations on arbitrarily-placed primitives fail too often (~5% in R10) for automated BO loops.

The four layers:
- **Layer 1 — Outer envelope + Fourier modulation (~14 vars):** smooth airfoil with Fourier-series leading/trailing-edge ripples for bat-wing / leaf-like / scalloped silhouettes.
- **Layer 2 — Macro-pattern + procedural math fields (~15-20 vars active, 0-3 per design):** library of 5 field types — louver / texture / edge feature / **noise threshold (2D Perlin or Simplex)** / **TPMS (gyroid or Schwarz-D lattice)**. Safe-by-construction; CAD failures <2% (vs R10's ~5%). The procedural math fields produce organic emergent topology (bone/coral/sponge-like, gyroid lattices) that no human pre-designed.
- **Layer 3 — Capped 0-1 independent primitive (~5-7 vars):** for asymmetric point features the pattern library can't produce. Wrapped in try/except (only place CAD failures occur).
- **Layer 4 — Manufacturing + click features (~3-5 vars):** unchanged from R10.

Panel thickness range expanded to **2-5 mm** (was 1-3 mm) for 3D-carved features. **Folded form factor added as 4th Pareto objective** so the optimizer exposes the bulk-vs-compactness trade-off rather than hiding it.

**Two earlier R10 drafts considered and rejected** (recorded so they aren't re-attempted):
1. **Density-based TO via SU2+Brinkman**: rejected because steady CFD cannot capture asymmetric drag (the unsteady mechanism that makes hand fans work); Brinkman + adjoint AD is research-level effort; grey-fluid artifacts vanish on binarization.
2. **Aero-sensitivity-weighted plate-bending TO with 2.5D skin breakthroughs (R9 default)**: rejected because adjoint linearization is invalid for large TO changes; mixing TO with aero on the panel entangled structural + aerodynamic problems; 2.5D breakthroughs produced too-thin walls.

R10's clean split: **rib TO via standard SIMP** (Reissner-Mindlin plate-bending; well-understood, no aero coupling) + **panel topology via parametric generative design** (BO finds the best subtraction pattern, BO sees real unsteady CFD physics).

Realistic gains: per-blade rib structural TO 15-30% mass reduction; generative parametric design 10-25% J_fan improvement (most upside from emergent topology — BO can produce slatted, lattice, scalloped, swiss-cheese, or solid designs); click-feature corrugation 3-8% J_fan. **Combined realistic IMU-normalized J_fan gain: 15-30% over Spike 0.3 baseline** (R11 renumber; was Phase 0.4).

---

## R10 Locked Decisions (Section 0 of report)

| Decision | Value |
|----------|-------|
| Fan architecture | Discrete V-unit blades with click-mating outer ribs |
| Per-blade structure | 2 side ribs + 1 generative panel, single rigid PETG piece |
| Folding mechanism | Pure rigid rotation about shared pivot |
| Inter-blade engagement | Mating chamfer + optional detent (Spike 0.4 validates) |
| Material | Single-material PETG |
| Optimization approach | **R11 4-layer hybrid**: Layer 1 envelope + Fourier; Layer 2 macro-pattern + procedural math fields (louver/texture/edge/noise-threshold/TPMS, 0-3 active); Layer 3 capped 0-1 primitive; Layer 4 manufacturing. ~37-46 vars. |
| Panel thickness range | **2-5 mm** (R11 expanded from R10's 1-3 mm) for 3D-carved features |
| Pareto objectives | 4D: J_fan, **I_pivot** (rotational inertia about wrist axis; R11 update — replaces total mass; aligns BO target with Phase 6 IMU angular-work metric), peak pivot stress, **folded form factor** (R11 NEW). Mass < 60 g and r_CoM ≤ 0.55·L_blade are hard constraints, not Pareto objectives. |
| TO scope | **Rib only** (2D plate-bending, Reissner-Mindlin); panel topology from generative design |
| Compliant panel | Out of R10 scope (R9 Phase 2d dropped) |
| Blade count (default) | 10 (BO range {8, 10, 12, 14}) |
| Spread angle (default) | 120° |
| Folded form factor target | ≤50 mm |
| Mass constraint | Total assembly < 60 g (hard) + **r_CoM ≤ 0.55·L_blade** (R11 NEW hard constraint; prevents tip-heavy solutions slipping past I_pivot Pareto by mass-balancing) |
| CFD fidelity | Multi-fidelity over (2D unsteady, 3D steady ranking, 3D unsteady true J_fan); **QSST DROPPED** |
| Compute budget | Colab Pro + G4 GPU; ~300-600 compute hours target |
| **Project structure (R11)** | Single Git repo `fan-optimization/` with `src/fanopt/{geometry, topopt, cfd, bo, physical, utils}/` + `tests/` + `scripts/` + `notebooks/` + `configs/`. Tooling: pytest + ruff + mypy + pyproject.toml + pre-commit + conda + GitHub Actions CI. Scaffolded in Phase 0 step 0.0 before any spike runs. See §12. |
| J_fan metric | Unchanged from R8 §9.4 locked spec |
| Realistic gain target | 15-30% IMU-normalized J_fan over **Spike 0.3 baseline** (R11 renumber; was Phase 0.4) |
| Emergent topology | The optimizer can produce slatted, lattice, swiss-cheese, scalloped, or solid panels; no visual-symmetry constraint |

---

## R10 Stack

| Layer | Tool | R10 specifics |
|-------|------|-------------|
| Geometry (primary) | Fusion 360 + Python add-in | Per-blade STL; multi-blade assembly view |
| Geometry (centerpiece) | **CadQuery generative blade generator (§9.7)** | `make_outer_envelope` + `apply_boolean_subtractions` + `add_surface_features` + `manufacturability_check` + `export_stl` |
| Rib TO (Phase 2) | FEniCSx 2D Reissner-Mindlin plate-bending SIMP | Rib only; multi-load-case (push + return + inertial + click); stress-constrained pivot; preserved click-feature footprint |
| Panel topology (Phase 2b/4) | **R11 4-layer hybrid**: envelope + Fourier + 5-field library (louver, texture, edge, noise-threshold, TPMS) + capped 0-1 primitive | No SIMP density on panel; BO optimizes generator parameters across layers |
| Meshing | Gmsh Python API | 2D corrugated slice + 3D corrugated fan with Boolean subtractions |
| CFD Tier -1 (R11.1 amendment) | SU2 **2D STEADY** (compressible + low-Mach prec, ~5 min/eval) | **Architecture screening only** at 30 evals/arch; 2D unsteady demoted to inner-loop verification |
| CFD Tier 0 | SU2 3D steady (compressible + low-Mach prec, ~30-90 min) | **Ranking-only**, NOT trusted as absolute J_fan; R11.2 uses combined Tier -1 + Tier 0 for bandit promotion |
| CFD Tier 1 | SU2 3D unsteady (compressible + low-Mach prec, dt=T/200, 5 cycles, ~3-6 h) | True J_fan via canonical j_fan.py |
| Verification | PyFR p=3 DG | GPU on Colab; top-3 Pareto designs |
| FSI | Not in R10 baseline | Phase 2d dropped |
| BO | Architecture bandit (K=4 promoted, fixed) + continuous TuRBO inner loop | **~37-46 dims total** (R11); ~20-30 continuous per architecture; SAASBO fallback with ≤500 inducing points |
| Physical validation | Anemometer + IMU + 3-copy fab-noise + acoustic | Print and validate top-3 Pareto designs |

---

## Phase Structure (R10)

| Phase | Weeks | Purpose |
|-------|-------|---------|
| 0 | 1-2 | Step 0.0 scaffolding + Spikes 0.1-0.7 (R11 renumber, sequential + dependency-respecting; see §8) |
| 1 | 1 | Generative parametric blade geometry pipeline + JSON schema (~45-55 vars) |
| 2 | 1 | Rib-only plate-bending TO (Reissner-Mindlin); closes OQ#2 |
| 2b | 3-4 | Generative parametric optimization seed: LHS across all 3 CFD tiers |
| 2c, 2d | REMOVED | (R8's membrane TO and R9's compliant-panel sub-study both removed in R10) |
| 3 | 1 | 2D unsteady CFD slice (Tier -1 of MF stack); steady-vs-unsteady correlation gate (R² ≥ 0.4) |
| 4 | 3-4 | Multi-fidelity BO at 45-55D on Colab Pro; 300-600 compute hours |
| 5 | 1-2 | High-fidelity verification + PyFR cross-solver on top-3 Pareto designs |
| 6 | 1-2 | Print top-3 designs + IMU + acoustic validation; user chooses preferred |

**Total:** ~10-13 weeks.

---

## R8/R9 Issue Disposition Table (R10)

**Issues still applying and fixed in R10:**

| R8 Issue | R10 Fix |
|----------|--------|
| #3 Peak angular acceleration (110 not 200 rad/s²) | Fixed in §2.4, §3.2.3 |
| #4 2D plane-stress for rib TO | Reissner-Mindlin plate-bending in §3.1, Phase 2 |
| #5 Print bed size | Per-blade fits 256 mm bed; full-assembly needs ≥360 mm |
| #6 Pure pitching kinematics | Validate via IMU-recorded real kinematics in Phase 6 |
| #7 QSST-CFD correlation | OBSOLETE in R10 (QSST removed entirely) |
| #8 GP at N≈2000 stalls | Sparse GP / ≤500 inducing points (defensive measure; R10 Tier -1 produces ~200 evals not 2000) |
| #9 PyFR top-1 only | Top-3 Pareto designs (Phase 5) |
| #10 Mass/ergonomics | Hard mass constraint m < 60 g + R11 r_CoM ≤ 0.55·L_blade constraint; **I_pivot** is the R11 Pareto objective (replaces R10 mass — aligns with Phase 6 IMU metric); N2 pivot-stack verification in Phase 6 |
| #12 Load case spec | Multi-load-case with peak-positive AND peak-negative pressures |
| #13 Optimization gain overclaim | Deflated to 15-30% J_fan |
| #15 Acoustic emission | Microphone measurement in Phase 6 |
| #16 Fab noise floor | Tightened CV to <5% (Spike 0.5) |
| #17 Architecture bandit math | K = 4 fixed |

**Issues obsolete in R10 (no action):**

| R8 Issue | Why obsolete |
|----------|--------------|
| #1 Folding geometry / mountain-valley | No membrane — rigid blade rotation |
| #2 Living-hinge spring-back | No living hinges |
| #11 Membrane elastic reaction | No flexible membrane |
| #14 Multi-material print safety | Single-material PETG |
| #18 TPU stringing | No TPU |
| #19 FSI non-convergence | No FSI |

**R10-specific new issues addressed:**

| R10 Issue | Addressed |
|-----------|-----------|
| N1 Click feature tolerance vs FDM ±0.1 mm | 0.15-0.20 mm design clearance per surface; Spike 0.4 validates |
| N2 Pivot pin mechanical design (20 ribs × 2 mm = 40 mm stack) | Pin ≥45 mm; Phase 6 verification |
| N3 Folded form factor verification | ≤50 mm target; Phase 6 verification |
| N4 Blade rigidity check | u_tip < 1 mm under peak load; Phase 2 sub-check |
| N5 Blade-blade aerodynamic interaction | Mesh resolves inter-blade gaps; Phase 3/4 CFD includes all blade forces |
| N6 Print orientation per blade | Categorical BO variable {rib-flat, deployed-V} |
| **N7** Generative geometry manufacturability | Aggressive CadQuery manufacturability filter (wall <0.8 mm, overhang >45°, disconnected pieces); Spike 0.7a |
| **N8** Parameter space dimensionality (37-46 vars, R11) | Spike 0.7b validates BO infrastructure at this scale |
| **N9** Pareto-design CFD verification cost | Top-3 selection rule: light corner / knee / heavy corner of Pareto front |

---

## What R10 Preserves from R8 Unchanged

- BoTorch + GPyTorch multi-fidelity BO infrastructure (§6, §9.5).
- SQLite experiment tracker with parallel-session atomic claim (Phase 0).
- Gmsh meshing scripts (adapted for blade-resolved geometry with Boolean subtractions).
- SU2 compressible + low-Mach preconditioning configs (§9.4).
- Canonical `j_fan.py` post-processor (§9.4 locked spec — unchanged).
- IMU-instrumented validation protocol (Phase 6).
- Spike 0.2 torsional-pendulum protocol (R11 renumber; was R9 Spike 0.8).
- 3-copy fab-noise study methodology.
- Hybrid Fusion + CadQuery dual-backend geometry pipeline (§4.6) — R10 leans more on CadQuery.

---

**Compute budget:** ~300-600 Colab Pro compute hours total, 2-4 wall-clock weeks with 2-4 parallel sessions. Mac handles all dev + BO driver. G4 GPU handles PyFR top-3 verification.
