# Challenger Review -- Round 2

**Report:** Topology Optimization and Aerodynamic Shape Optimization for a 3D-Printed Hand Fan
**Reviewer:** Senior Research Reviewer
**Date:** 2026-03-22
**Round:** 2 (following revisions addressing Round 1 feedback)

---

## Overall Assessment: HAS GAPS (improved from Round 1, but new issues identified)

The revised report has addressed the majority of Round 1 concerns substantively. The Fusion 360 pricing correction, the anisotropy section, the post-processing pipeline, and the revised aerodynamic objective are all genuine improvements, not hand-waving. However, this closer reading reveals new technical issues that were masked by the larger Round 1 problems, most critically around the SU2 configuration examples that a beginner would actually try to run.

---

## 1. Round 1 Fix Assessment

### 1.1 Fusion 360 pricing (Round 1 Issue 2.1) -- ADEQUATELY FIXED

The pricing note in Section 3.1 is now detailed, accurate, and prominently placed. The recommendation to use FreeCAD/BESO as the genuinely free path is clear. The report even mentions the pay-per-study cloud credits option. This concern is fully resolved.

### 1.2 FDM anisotropy (Round 1 Issue 2.2) -- ADEQUATELY FIXED

Section 2.1.5 is a thorough new addition. The mitigation strategies are well-ordered (print flat, directional safety factors, orthotropic FEA verification, advanced SOMP). The "key insight" summary at the end is exactly the kind of actionable guidance a beginner needs. The connection to print orientation (Section 7.4) is now explicit. Minor note: the Z-direction strength reduction values in Table 6.1 are reasonable and consistent with the literature. This concern is fully resolved.

### 1.3 TO/ASO coupling (Round 1 Issue 2.3) -- ADEQUATELY FIXED

Phase 4 Step 19 now explicitly frames the CFD verification as a coupling check with a clear 10% change threshold for triggering re-iteration. The aeroelastic criterion (2% chord deflection) is a reasonable engineering threshold. The mention of DAFoam + OpenMDAO for the advanced path is good. This concern is resolved.

### 1.4 Von Mises equation (Round 1 Issue 2.4) -- ADEQUATELY FIXED

Section 2.1.3 now presents the full 3D equation first, then derives the 2D simplification with a clear statement of when it applies and how to verify the assumption. Well done.

### 1.5 GP surrogate feasibility (Round 1 Issue 2.5) -- ADEQUATELY FIXED

Section 5.5 is a solid new addition covering dimensionality reduction, conservative sample counts, and the mandate for multi-fidelity. The distinction between beginner path (<10 variables, scikit-learn) and standard path (10-25 variables, BoTorch) is practical. The NRMSE > 15% validation criterion is a useful heuristic. This concern is resolved.

### 1.6 TO post-processing (Round 1 Issue 2.6) -- SUBSTANTIALLY FIXED

Section 8.3.1 is now a full multi-step pipeline with specific tool recommendations, which is a major improvement over the single bullet point. The inclusion of the verification FEA step after smoothing is important. The alternative suggestion of OpenLSTO is helpful. One minor gap: the section does not mention the specific Python workflow for marching cubes (it mentions `scikit-image.measure.marching_cubes` and `PyVista` but does not provide even a minimal code snippet, whereas Sections 9.1 and 9.4 provide detailed code). For consistency, a 10-line Python example of the marching cubes + Taubin smoothing pipeline would help beginners who are following the Python-heavy workflow. This is minor and does not warrant a "needs revision" flag.

### 1.7 Aerodynamic objective (Round 1 Issue 2.7) -- PARTIALLY FIXED (see new issue 2.1 below)

The conceptual fix is good: the report now clearly states that drag minimization is wrong, introduces the directed momentum flux metric, and provides the pressure coefficient integral proxy. However, the SU2 configuration example introduces a new problem -- see Issue 2.1 below.

### 1.8 Fatigue estimate (Round 1 Issue 2.8) -- ADEQUATELY FIXED

Section 2.1.4 now properly disclaims the metals-style endurance ratio, recommends S-N curves, and adjusts cycle count estimates to realistic values (600-1,800 per session). The conservative 30% of yield rule is practical. The raster angle dependence is correctly noted.

### 1.9 Missing elements from Round 1 -- MOSTLY ADDRESSED

- Modal analysis/resonance: Added as Phase 4 Step 18 with a specific f_1 > 5x f_waving criterion. Good.
- Multi-objective BO: Added as Section 5.2.4 with qNEHVI recommendation. Good.
- Beginner timeline: Revised to 3-6 months with a "minimum viable project" alternative. Good.
- Surface roughness: Added as Section 2.2.5. Good.
- Cost/benefit of optimization: Added as Section 1.4. Honest and well-balanced.

---

## 2. New Issues (not raised in Round 1)

### 2.1 MAJOR: The SU2 `SURFACE_PRESSURE_DROP` objective function may not work as described for this problem

**Challenge:** The revised SU2 configuration (Section 9.3) now uses `OBJECTIVE_FUNCTION= SURFACE_PRESSURE_DROP` instead of `DRAG`. While this is conceptually closer to the right objective, there are practical problems:

1. `SURFACE_PRESSURE_DROP` in SU2 computes the pressure difference between two marker boundaries (an inlet and an outlet). It is designed for internal flow applications (ducts, pipes, heat exchangers) where there is a well-defined upstream and downstream boundary. For an external flow problem like a fan blade in open air, there is no natural "inlet" and "outlet" pair bounding the fan. The configuration shown does not define the marker boundaries needed for this objective, and it is unclear how a beginner would set these up for an external aerodynamics problem.

2. The `OPT_OBJECTIVE= SURFACE_PRESSURE_DROP * 1.0` line references this objective but the configuration does not specify `MARKER_ANALYZE` or `MARKER_INLET`/`MARKER_OUTLET` pairs that would be needed to define where the pressure drop is measured.

**Evidence:** SU2's `SURFACE_PRESSURE_DROP` is documented in the [config template](https://github.com/su2code/SU2/blob/master/config_template.cfg) as a surface monitoring quantity requiring marker specifications. It is primarily used in internal flow optimization (e.g., duct shape optimization). For external aerodynamics, the standard approach is to use `DRAG`, `LIFT`, or custom objective functions.

**Alternative:** For a fan in external flow, a more appropriate SU2 configuration would be:

1. **Best practical approach:** Set up monitoring planes upstream and downstream of the fan as internal boundaries within the mesh, then compute the momentum flux difference. This requires custom mesh setup but gives the physically correct objective.
2. **Simpler proxy that actually works in SU2:** Use `INVERSE_DESIGN_PRESSURE` with a target pressure distribution on the fan surface (the target being a distribution that maximizes downstream momentum). This is a well-documented SU2 objective.
3. **Custom objective via SU2 Python wrapper:** The report mentions this as an option but provides no guidance. A 20-line Python example computing net downstream momentum flux from the SU2 solution would be far more useful than the current configuration, which a beginner will not be able to run as-is.

The report correctly identifies that drag minimization is wrong, but replaces it with an objective function that is designed for a different class of problems (internal flows) and will likely not work out-of-the-box for the fan's external flow configuration.

---

### 2.2 MAJOR: The SU2 unsteady configuration mixes incompressible solver with features that have documented compatibility issues

**Challenge:** The report's "Unsteady Approach" configuration (Section 9.3) specifies:

```
SOLVER= INC_NAVIER_STOKES
...
GRID_MOVEMENT= RIGID_MOTION
PITCHING_OMEGA= 0.0 0.0 6.2832
```

However, there is a [documented SU2 issue (#193)](https://github.com/su2code/SU2/issues/193) showing that the incompressible solver (`INC_NAVIER_STOKES`) had a bug with `PITCHING_OMEGA` where the pitching frequency was not properly non-dimensionalized, producing nonsensical values (8.19e+48 rad/s instead of the intended value). The SU2 developer response stated: "the incompressible formulation used in SU2 is only valid for steady problems and thus cannot be used for this kind of application."

While SU2's incompressible solver has evolved since that issue was filed, and more recent versions do support `TIME_DOMAIN= YES` with `INC_NAVIER_STOKES`, the combination of incompressible solver + rigid body pitching motion + unsteady adjoint is at the edge of what is tested and documented. The SU2 unsteady optimization tutorials all use the compressible solver (`NAVIER_STOKES` or `EULER`), not the incompressible variant.

**Evidence:** The SU2 [Unsteady Shape Optimization NACA0012 tutorial](https://su2code.github.io/tutorials/Unsteady_Shape_Opt_NACA0012/) -- the very tutorial the report references -- uses `SOLVER= EULER` (compressible), not `INC_NAVIER_STOKES`. Every unsteady pitching test case in the SU2 repository (`TestCases/unsteady/pitching_*`) uses the compressible formulation.

**Alternative:** The report should either:

1. Use the compressible solver with low-Mach preconditioning for the unsteady configuration: `SOLVER= NAVIER_STOKES` with `LOW_MACH_PREC= YES`. At Mach ~0.008 (2.64 m/s / 343 m/s), preconditioning is essential for convergence but the compressible formulation is better tested with rigid motion and unsteady adjoint.
2. Explicitly warn that the incompressible + pitching + unsteady adjoint combination is not well-tested in SU2 and may require troubleshooting.
3. Provide the compressible solver configuration as the primary unsteady example, since that matches the actual SU2 tutorials.

This matters because a beginner following the report's configuration will likely encounter convergence failures or incorrect results without understanding why.

---

### 2.3 MODERATE: The PETG material properties used for FEA are bulk datasheet values, not FDM-printed values

**Challenge:** Section 9.2 specifies PETG material properties for FEA as `E = 2100 MPa, nu = 0.38`. These are bulk/datasheet values for injection-molded PETG. FDM-printed PETG has significantly lower effective stiffness due to voids, inter-layer bonding, and infill patterns.

**Evidence:** Published experimental data on FDM-printed PETG specimens shows:

- Young's modulus in the XY (strong) direction: 1100-1500 MPa (not 2100 MPa)
- Young's modulus in the Z direction: 800-1200 MPa
- Poisson's ratio: 0.36-0.40 (this is approximately correct)

Sources: [Experimental and Numerical Analysis for FDM PETG (PMC, 2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7600181/) reports compression modulus values of 1117-1330 MPa depending on print direction. [FDM PETG material parameters (ResearchGate)](https://www.researchgate.net/figure/PETGs-material-parameters-Youngs-modulus-E-and-Poissons-ratio-n-are-determined-by_tbl4_351888845) shows similar ranges.

Using 2100 MPa (the datasheet value) in FEA will overestimate the part stiffness by 40-90%, leading to:
- Underestimated displacements (the fan blade will deflect more than predicted)
- A false sense of security about stress margins
- Incorrect natural frequency predictions (frequencies scale as sqrt(E), so a 50% overestimate in E gives ~22% overestimate in natural frequencies, which could cause the modal analysis to miss a resonance concern)

**Alternative:** The report should:

1. Use FDM-specific material properties: E_XY ~ 1300 MPa, E_Z ~ 1000 MPa (for 100% infill, 0.2mm layer height, standard print conditions)
2. Note that these values depend heavily on infill density -- at 30% gyroid infill (recommended in Section 6.3), the effective modulus is substantially lower still, roughly scaling with the density fraction
3. Recommend that users print and test tensile specimens in their specific printer/filament/settings combination to calibrate material properties before running FEA
4. At minimum, flag that the 2100 MPa value is for bulk PETG and that FDM-printed values are 30-50% lower

---

### 2.4 MODERATE: The steady-state "proxy" CFD objective lacks validation against the unsteady reality

**Challenge:** The report acknowledges that the fan operates in unsteady oscillatory motion but recommends a steady-state CFD proxy as the beginner approach (Section 9.3). This is pragmatic, but the report provides no evidence or citation that this proxy actually correlates with unsteady fan performance. It simply asserts that the "pressure integral on the downstream face at peak velocity...correlates with fan airflow performance" without justification.

**Evidence:** For oscillating flat plates at low Reynolds numbers, the flow physics at peak velocity is dominated by dynamic effects that a steady-state simulation cannot capture:

- **Added mass effects:** The acceleration of air in front of the plate contributes significantly to the instantaneous force but is absent in steady-state.
- **Vortex shedding phase:** At peak velocity, vortices shed during the previous stroke are still present in the wake and interact with the current stroke. Steady-state cannot capture this.
- **Reduced frequency:** For the report's parameters (f = 2 Hz, V_tip_max = 2.64 m/s, chord = 0.25 m), the reduced frequency k = pi * f * c / V = pi * 2 * 0.25 / 2.64 ~ 0.6. This is well into the unsteady regime (k > 0.05 is considered unsteady in aerodynamics). At k ~ 0.6, steady-state correlations are poor.

The reference cited by the report -- [Drag on Oscillating Flat Plates at Low Reynolds Numbers (Cambridge Core)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/drag-on-oscillating-flat-plates-in-liquids-at-low-reynolds-numbers/) -- specifically studies the unsteady effects and shows that they are significant at these Reynolds numbers and reduced frequencies.

**Alternative:** The report should either:

1. Provide a citation or argument for why the steady-state proxy is adequate for *shape optimization ranking* (i.e., even if the absolute values are wrong, does the ranking of shapes by the steady-state proxy correlate with the ranking by the unsteady metric?). If shapes A and B have the same relative performance ordering under steady and unsteady conditions, the proxy works for optimization even if the absolute values differ.
2. Recommend running at least 2-3 unsteady validation cases (baseline, best candidate, worst candidate) to verify that the steady-state proxy preserves the relative ranking.
3. Be more honest that this is a significant simplification that may lead to a suboptimal design, not just a minor approximation.

With a reduced frequency of ~0.6, the steady-state proxy is a substantial approximation, and the report should quantify this risk rather than glossing over it.

---

### 2.5 MINOR: The TopOpt Python library is alpha-stage software, and the code example may not run

**Challenge:** Section 9.1 recommends `pip install topopt` and provides a code example using `from topopt.boundary_conditions import MBBBeamBoundaryConditions`, etc. The TopOpt library is version 0.0.1-alpha.1, which is pre-release software. Its API may have changed, and the code example should be verified against the current version.

**Evidence:** The [TopOpt PyPI page](https://pypi.org/project/topopt/) shows version 0.0.1-alpha.1, and the [documentation](https://pytopopt.readthedocs.io/en/latest/) describes it as being "in early stages of development." The import paths and class names in the report's example may not match the current API.

**Alternative:** The report should:

1. Note the alpha status explicitly and warn that the API may change
2. Recommend the more mature [DTU TopOpt codes](https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python) as an alternative educational resource -- these are widely cited, well-documented 200-line Python scripts that have been stable for years
3. Verify the code example runs against the current TopOpt version before publishing

---

### 2.6 MINOR: The "minimum viable project" path still requires SU2, which is the hardest tool in the stack

**Challenge:** Section 8 proposes a "minimum viable project" that skips the ML surrogate and uses "SU2's built-in adjoint-based ASO directly." While this simplifies the ML component, SU2 is by far the hardest tool in the entire workflow for a beginner (command-line only, complex config files, external mesh generation, Linux/Mac preferred). The "minimum viable" path still requires the steepest learning curve component.

**Alternative:** A truly minimum viable project for a beginner should be:

1. **Phase 1:** Design baseline fan in FreeCAD, print it, measure airflow
2. **Phase 2:** Run CFD on 5-10 hand-designed shape variants using SimScale (browser-based, no installation), pick the best one based on the pressure distribution
3. **Phase 3:** Run TO on the best shape using FreeCAD/BESO
4. **Phase 4:** Print and validate

This eliminates SU2, BoTorch, and formal shape optimization entirely, replacing ASO with manual design-of-experiments using the beginner-friendly SimScale. It is achievable in 3-4 weeks and teaches the core concepts. The full SU2 + BO workflow can be presented as the "advanced project" for those who complete the MVP.

---

## 3. Agreements (what the revised report gets right)

- **Section 1.4 (cost-benefit analysis of optimization):** Honest, well-calibrated, and correctly identifies structural TO as the higher-value component for this application.
- **Section 2.1.5 (anisotropy section):** Thorough and actionable. The graduated mitigation strategies are well-designed.
- **Section 5.5 (high-dimensionality handling):** Practical and correctly identifies the key levers (PCA, ARD kernels, multi-fidelity).
- **Section 8.3.1 (post-processing pipeline):** Specific, tool-oriented, and includes the critical verification FEA step.
- **Phase 4 coupling check (Section 8, Step 19):** Clear threshold criteria for when to iterate.
- **Modal analysis addition (Section 8, Step 18 and 10.1):** Well-specified with a concrete frequency ratio criterion.
- **Revised timeline (3-6 months):** Honest and appropriate for a beginner.

---

## 4. Missing Elements (new, not raised in Round 1)

### 4.1 No discussion of mesh generation difficulty

The report mentions Gmsh multiple times but does not discuss the difficulty of generating a quality CFD mesh around a thin, curved fan blade. For a blade thickness of 2.5 mm and a chord of 250 mm (aspect ratio 100:1), meshing the boundary layer with sufficient resolution is non-trivial. The boundary layer thickness at Re = 30,000 is approximately 5/sqrt(Re) * c ~ 7 mm, so the first cell height needs to be on the order of 0.1-0.5 mm. This is a significant practical hurdle that gets zero discussion. A beginner spending 2-4 weeks learning SU2 will spend at least 1-2 of those weeks on mesh generation alone.

### 4.2 No guidance on when the optimization has "converged enough"

For both TO and BO, the report provides convergence criteria (0.1% compliance change for TO, NRMSE < 15% for GP) but does not discuss what to do when convergence is slow or oscillatory. In practice, TO convergence for complex 3D geometries with AM constraints often stalls or oscillates, and the user needs to know when to stop iterating, adjust filter parameters, or accept a non-converged result. Similarly, BO may exhaust the evaluation budget without finding a clearly optimal region. Practical guidance on diagnosing and addressing these situations would be valuable.

---

## 5. Verdict: NEEDS REVISION

### Specific changes needed (in priority order):

1. **Fix the SU2 objective function configuration** (Issue 2.1): Replace `SURFACE_PRESSURE_DROP` with an objective that works for external aerodynamics, or provide the custom Python objective approach with actual code. The current configuration will not run for the fan problem as described.

2. **Fix the SU2 unsteady configuration** (Issue 2.2): Either switch to the compressible solver with low-Mach preconditioning (which matches the actual SU2 tutorials), or explicitly warn about the incompressible + pitching compatibility issue.

3. **Use FDM-printed material properties, not datasheet values** (Issue 2.3): Change E = 2100 MPa to approximately 1300 MPa for FDM PETG, or at minimum add a prominent warning that the datasheet value overestimates FDM part stiffness by 40-90%.

4. **Add a validation argument for the steady-state proxy** (Issue 2.4): Either cite evidence that steady-state shape rankings correlate with unsteady rankings, or recommend unsteady validation runs. The reduced frequency of ~0.6 makes this a non-trivial simplification.

5. **Consider restructuring the "minimum viable project"** (Issue 2.6): Replacing SU2 with SimScale for the MVP path would make the project genuinely accessible to beginners.

---

*Review conducted 2026-03-22. Web searches used to verify SU2 objective function availability, SU2 incompressible solver + pitching motion compatibility ([GitHub Issue #193](https://github.com/su2code/SU2/issues/193)), FDM PETG material property literature ([PMC 7600181](https://pmc.ncbi.nlm.nih.gov/articles/PMC7600181/)), TopOpt library status ([PyPI](https://pypi.org/project/topopt/)), and reduced frequency effects on steady-state CFD validity. All cited sources were accessible at time of review.*
