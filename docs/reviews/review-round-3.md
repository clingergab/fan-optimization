# Challenger Review -- Round 3 (Final)

**Report:** Topology Optimization and Aerodynamic Shape Optimization for a 3D-Printed Hand Fan
**Reviewer:** Senior Research Reviewer
**Date:** 2026-03-22
**Round:** 3 (final adversarial review)

---

## Overall Assessment: STRONG

The R2 report has substantively addressed all major and moderate issues raised in Rounds 1 and 2. The remaining observations below are minor and do not warrant another revision cycle.

---

## 1. Round 2 Fix Assessment

### 1.1 SU2 objective function (Round 2 Issue 2.1) -- WELL FIXED

The three-tier objective approach (LIFT-based simplest, SURFACE_TOTAL_PRESSURE on downstream plane recommended, custom Python wrapper for highest fidelity) is a genuine improvement. Web search confirms that `SURFACE_TOTAL_PRESSURE` with `MARKER_ANALYZE` has adjoint support in SU2 and is used in the [Multi-Objective Shape Design tutorial](https://su2code.github.io/tutorials/Multi_Objective_Shape_Design/). The explicit explanation of why `SURFACE_PRESSURE_DROP` was wrong (internal flow inlet/outlet assumption) is helpful for readers who might encounter that option in SU2 documentation and wonder about it.

The Python wrapper example for the custom momentum flux objective (Approach 3) is illustrative and appropriately caveated as version-dependent. The note directing readers to the FSI Python wrapper tutorials for exact API calls is practical.

This concern is fully resolved.

### 1.2 SU2 unsteady solver (Round 2 Issue 2.2) -- WELL FIXED

Switching to `NAVIER_STOKES` + `LOW_MACH_PREC= YES` for the unsteady pitching configuration is correct. The documentation of GitHub Issue #193 and the explicit statement that all SU2 unsteady pitching tutorials use the compressible formulation provides strong justification. The retention of `INC_NAVIER_STOKES` for the steady-state proxy only is appropriate since no pitching motion is involved there.

Web search confirms that SU2's unsteady adjoint for pitching motion has been tested and published with the compressible solver, including drag reduction results on pitching airfoils ([AIAA 2023-3311](https://arc.aiaa.org/doi/10.2514/6.2023-3311)).

This concern is fully resolved.

### 1.3 FDM material properties (Round 2 Issue 2.3) -- WELL FIXED

The correction from E = 2100 MPa to E_XY ~ 1300 MPa and E_Z ~ 1000 MPa is consistent with the experimental literature. Web search confirms FDM PETG compression modulus values of 1117-1330 MPa ([PMC 7600181](https://pmc.ncbi.nlm.nih.gov/articles/PMC7600181/)). The addition of the FDM-specific rows in Table 6.1, the infill scaling note, and the recommendation to print tensile specimens for calibration are all practical and well-calibrated advice.

This concern is fully resolved.

### 1.4 Steady-state proxy limitations (Round 2 Issue 2.4) -- WELL FIXED

Section 9.3.1 is an excellent addition. The reduced frequency calculation (k ~ 0.6), the enumeration of missing physical effects, the discussion of whether shape rankings are preserved, and the mandatory 3-design validation protocol are all substantive. The guidance on when to skip the proxy entirely (deep camber, slotted geometries, flexible blades) is practical.

The report correctly maintains the steady-state proxy as a valid beginner starting point while being transparent about its limitations. This is the right pragmatic balance.

This concern is fully resolved.

### 1.5 Minimum viable project restructured (Round 2 Issue 2.6) -- WELL FIXED

The two-option MVP structure is a significant improvement. Option A (SimScale, no SU2) is genuinely beginner-accessible and achievable in 3-4 weeks. Option B (SU2 adjoint, no surrogate) gives an intermediate path. The explicit warning about SU2's learning curve is honest and helpful.

This concern is fully resolved.

### 1.6 Mesh generation (Round 2 Issue 4.1) -- WELL FIXED

The new subsection on mesh generation challenges is detailed and practical. The 100:1 aspect ratio discussion, boundary layer resolution requirements, trailing edge singularity handling, downstream analysis plane setup, and Gmsh tips are all items a beginner would otherwise discover painfully through trial and error. The 1-2 week time estimate for mesh generation learning is realistic.

This concern is fully resolved.

### 1.7 Convergence diagnosis (Round 2 Issue 4.2) -- WELL FIXED

Section 8.3.2 with the diagnostic tables for both TO and BO convergence problems is comprehensive. The symptom-cause-remedy format is actionable. The "when to stop iterating" guidance (1% stability over 20 iterations, less than 5% intermediate densities, connected load paths) is practical engineering advice that would otherwise require significant experience to develop.

This concern is fully resolved.

### 1.8 TopOpt alpha status (Round 2 Issue 2.5) -- WELL FIXED

The explicit alpha warning and the DTU TopOpt codes as a more mature fallback are appropriate.

This concern is fully resolved.

---

## 2. Remaining Observations (Minor -- none warrant a revision cycle)

### 2.1 MINOR: SURFACE_TOTAL_PRESSURE adjoint support should be explicitly confirmed

The report recommends `SURFACE_TOTAL_PRESSURE` as the primary objective (Approach 2) but does not explicitly state that this objective has adjoint support for gradient computation. Web search confirms it does -- the [SU2 Multi-Objective Shape Design tutorial](https://su2code.github.io/tutorials/Multi_Objective_Shape_Design/) uses `SURFACE_TOTAL_PRESSURE` as an objective in a discrete adjoint optimization. A one-sentence note confirming this adjoint compatibility would give beginners confidence that the optimization loop will actually compute gradients for this objective, rather than just monitoring it.

**Impact:** Low. A beginner who follows the tutorial links would discover this. But an explicit statement would save debugging time.

### 2.2 MINOR: The Python wrapper code example (Approach 3) may not work with the SU2 API as shown

The Python wrapper code uses methods like `driver.GetMarkerNormal()`, `driver.GetMarkerCoordinates()`, and `driver.GetMarkerVelocity()`. These method names are plausible but the exact API varies between SU2 versions, and some of these specific methods may not exist in the current release. The report does caveat this ("The Python wrapper API varies between SU2 versions. The above is illustrative"), which is appropriate. However, a beginner may still attempt to run this code verbatim and encounter errors.

**Impact:** Low, given the caveat. The code serves its purpose as an illustration of the approach, and the pointer to the FSI Python wrapper tutorials for exact API calls is adequate.

### 2.3 MINOR: No marching cubes code snippet in Section 8.3.1

As noted in Round 2 review (Section 1.6, minor gap), the post-processing pipeline mentions `scikit-image.measure.marching_cubes` and `PyVista` but does not provide the 10-line code snippet that would be consistent with the detail level of Sections 9.1 and 9.4. This is a stylistic inconsistency rather than a substantive gap.

**Impact:** Very low. ParaView GUI instructions are provided as the primary path, and the Python tools are mentioned as alternatives for scripting users.

### 2.4 NOTE: The reference numbering has a gap (55-60 appear after 71)

References 55-60 appear at the end of the references section under "General References" and "Fusion 360 Pricing," after references 61-71 which cover FDM material properties, SU2 solver compatibility, and mesh generation. This suggests references were added incrementally during revisions without renumbering. This is a cosmetic issue only and does not affect the report's technical quality.

---

## 3. Agreements (what the revised report gets right)

- **Three-tier SU2 objective function approach (Section 9.3):** This graduated approach -- from simple LIFT-based, to SURFACE_TOTAL_PRESSURE on a monitoring plane, to custom Python objective -- gives users at every skill level a workable starting point. The explicit note about why SURFACE_PRESSURE_DROP was removed demonstrates intellectual honesty.

- **Compressible solver with LOW_MACH_PREC for unsteady (Section 9.3):** This matches the SU2 tutorial ecosystem and has the strongest test coverage. The retention of the incompressible solver for steady-state only is the correct engineering choice.

- **FDM-specific material properties (Sections 6.1, 9.2):** Using actual FDM values rather than datasheet values, with the recommendation to print test specimens, is the kind of practical advice that separates a useful guide from an academic exercise.

- **Steady-state proxy validation protocol (Section 9.3.1):** The mandatory 3-design unsteady validation, with a clear decision criterion for whether the proxy preserves rankings, is methodologically sound. The reduced frequency calculation gives readers a quantitative basis for understanding the approximation quality.

- **Two-option MVP structure (Section 8):** Option A (SimScale, 3-4 weeks) genuinely makes the project accessible to beginners without CFD experience. Option B (SU2 adjoint, 4-6 weeks) provides a stepping stone to the full workflow. This tiered approach is excellent pedagogy.

- **Convergence diagnostics (Section 8.3.2):** The symptom-cause-remedy tables for both TO and BO are the kind of practical troubleshooting content that is rarely found in academic treatments and is extremely valuable for practitioners.

- **Post-processing pipeline (Section 8.3.1):** The five-step pipeline with specific tool recommendations, including the critical verification FEA after smoothing, addresses what is genuinely the hardest practical step in the workflow.

- **Mesh generation guidance (Section 9.3):** The discussion of aspect ratio challenges, boundary layer resolution, trailing edge handling, and time estimates fills a gap that would otherwise cost beginners weeks of frustration.

- **Revision log transparency:** The detailed revision log at the end, documenting what was changed and why in response to each review issue (including noting partial vs. full agreement), demonstrates intellectual rigor and helps readers understand the report's evolution.

---

## 4. Verdict: CONSENSUS

The report has reached a level of quality where it is accurate, practical, comprehensive, and actionable for the target user (a beginner with a good 3D printer and willingness to learn). All major and moderate issues from Rounds 1 and 2 have been substantively addressed with evidence, not hand-waving. The remaining observations (Section 2 above) are minor stylistic and completeness notes that do not affect the report's reliability or usefulness.

### Summary of quality:

- **Accuracy:** Technical equations, material properties, software configurations, and physical reasoning are correct. Known limitations and approximations are explicitly stated with quantitative context.
- **Practicality:** The tiered MVP approach, realistic timelines, mesh generation guidance, convergence diagnostics, and post-processing pipeline make this usable, not just theoretically sound.
- **Comprehensiveness:** The report covers the full workflow from physics background through software selection, execution, and validation, with appropriate depth at each stage.
- **Actionability:** Code examples, configuration files, tool recommendations with installation instructions, and decision flowcharts provide concrete next steps.

### Minor suggestions (not blocking):

1. Add a one-sentence confirmation that `SURFACE_TOTAL_PRESSURE` has adjoint support in SU2 (verifiable via the Multi-Objective tutorial).
2. Renumber references for consistency (cosmetic).
3. Consider adding a brief (~10-line) Python snippet for the marching cubes + Taubin smoothing step to match the detail level of Sections 9.1/9.4.

These are at the author's discretion and do not require another review cycle.

---

*Final review conducted 2026-03-22. Web searches used to verify: SU2 SURFACE_TOTAL_PRESSURE adjoint support ([SU2 Multi-Objective Tutorial](https://su2code.github.io/tutorials/Multi_Objective_Shape_Design/)), SU2 unsteady adjoint for pitching motion ([AIAA 2023-3311](https://arc.aiaa.org/doi/10.2514/6.2023-3311)), FDM PETG material properties ([PMC 7600181](https://pmc.ncbi.nlm.nih.gov/articles/PMC7600181/), [ScienceDirect 2025](https://www.sciencedirect.com/science/article/pii/S223878542501436X)). All cited sources were accessible at time of review.*
