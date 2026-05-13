# Challenger Review -- Round 1

**Report:** Topology Optimization and Aerodynamic Shape Optimization for a 3D-Printed Hand Fan
**Reviewer:** Senior Research Reviewer
**Date:** 2026-03-22

---

## Overall Assessment: HAS GAPS

The report is impressively comprehensive in scope and well-organized. The equations are largely correct, the software landscape is well-surveyed, and the general workflow is reasonable. However, there are several substantive issues ranging from a critical factual error about Fusion 360 pricing/availability, to a fundamental methodological concern about the decoupled TO/ASO workflow, to missing physics around FDM material anisotropy that undermines the validity of the topology optimization results. These issues need to be addressed before the report can serve as a reliable guide for a beginner.

---

## 1. Agreements (what the report gets right)

- **SIMP formulation and equations (Section 2.1.1-2.1.2):** The SIMP compliance minimization, sensitivity expression, and density filter are all correctly stated. The penalization parameter p=3 recommendation is standard and well-justified.

- **Flow regime analysis (Section 2.2.1):** The Reynolds number calculation is correct. Re = 10,000-50,000 for typical hand fan waving is a reasonable estimate, and the characterization as low-Re, laminar-to-transitional, unsteady flow is accurate.

- **Oscillating plate kinematics (Section 2.2.4):** The sinusoidal motion model and tip velocity calculation are correct. V_tip_max = 2.64 m/s for the given parameters checks out.

- **Material recommendation of PETG over PLA for the final fan (Section 6.2):** This is sound advice. PLA's brittleness and poor fatigue life make it unsuitable for repeated cyclic loading, and the report correctly identifies this.

- **Bayesian optimization workflow (Section 5.2.3, 9.4):** The BO formulation, acquisition functions (EI, UCB), and BoTorch code example are correct and practical. The recommendation to start with scikit-learn for beginners is sensible.

- **Validation protocol (Section 10):** The inclusion of mesh independence studies, physical anemometer testing, and smoke visualization is excellent and often missing from similar guides.

---

## 2. Disagreements (substantive challenges)

### 2.1 CRITICAL: Fusion 360 Generative Design is NOT free for personal use

**Challenge:** The report states Fusion 360 is "$545/yr (free for personal)" in the comparison table and later says "Fusion 360 with Generative Design is the most beginner-friendly path" as if it is accessible at no cost. This is misleading.

**Evidence:** Web search confirms that Fusion 360's free personal use tier does NOT include generative design. Generative design requires the Fusion Simulation Extension, which is a separate paid subscription, or purchase of Autodesk Cloud Credits on a pay-as-you-go basis. Only a 30-day free trial is available for generative design features. (Sources: [Autodesk Fusion Personal Use comparison](https://www.autodesk.com/products/fusion-360/personal), [Fusion 360 Generative Design free trial page](https://www.autodesk.com/ca-fr/products/fusion-360/generative-design-free-trial), [Autodesk support article on Generative Design access](https://www.autodesk.com/support/technical/article/caas/sfdcarticles/sfdcarticles/How-to-get-access-to-Generative-Design.html))

**Alternative:** The report should clearly state that generative design requires a paid subscription beyond the base Fusion 360 license. For a beginner wanting free tools, the FreeCAD + BESO path or the open-source Python route are the only genuinely free options. If the report recommends Fusion 360, it should give the real cost: base subscription ($545/yr) plus Simulation Extension (additional cost) or cloud credits.

**Impact:** A beginner following this report would discover after investing time learning Fusion 360 that the key feature they need is behind a paywall. This is a significant practical error.

---

### 2.2 MAJOR: FDM Material Anisotropy is completely ignored in the TO formulation

**Challenge:** The entire topology optimization formulation (Section 2.1) assumes isotropic material behavior -- SIMP uses "Solid Isotropic Material with Penalization." The report never acknowledges that FDM-printed parts are fundamentally anisotropic. This is not a minor oversight; it undermines the validity of the optimization results.

**Evidence:** FDM parts exhibit 20-40% reduction in mechanical properties in the build direction (Z-axis) compared to the in-plane (XY) directions due to layer bonding weakness. This is well-documented in peer-reviewed literature. (Sources: [Material Anisotropy in Additively Manufactured Polymers and Polymer Composites: A Review (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8512748/), [Topology optimization for 3D printing-driven anisotropic components (ScienceDirect, 2025)](https://www.sciencedirect.com/science/article/abs/pii/S014102962500046X), [Topology Optimization for Multipatch FDM 3D Printing (MDPI)](https://www.mdpi.com/2076-3417/10/3/943))

Specifically:
- Inter-layer bond strength in FDM PETG is substantially lower than in-plane strength
- The topology optimizer may produce features (e.g., thin ribs oriented in the Z-direction) that look optimal under isotropic assumptions but are structurally weak when actually printed
- Recent research (2025) explicitly addresses anisotropy-aware topology optimization for 3D printing and shows that ignoring anisotropy can lead to structures that are up to 25% heavier than necessary or, worse, that fail unexpectedly

**Alternative:** The report should at minimum:
1. Acknowledge this limitation explicitly
2. Recommend printing the fan flat (which it does in Section 7.4, but without connecting it to the isotropic TO assumption)
3. Apply a directional safety factor (e.g., 2x in the Z-direction) when interpreting TO results
4. For advanced users, mention anisotropy-aware TO formulations (e.g., orthotropic SIMP or the multipatch FDM approach from recent literature)

The report's Section 7.4 on print orientation partially addresses this, but it is disconnected from the TO formulation. The connection needs to be made explicit: "We use isotropic SIMP, which is valid IF and ONLY IF the primary load paths in the optimized structure align with the strong (XY) printing directions."

---

### 2.3 MAJOR: The TO and ASO are sequentially decoupled, but the report does not adequately justify this or warn about its limitations

**Challenge:** The workflow (Section 8) runs ASO first (Phase 2), then feeds the result to TO (Phase 3) in a one-way sequential coupling. The report acknowledges coupling in Section 1.3 ("These two problems are coupled") but then ignores this coupling in the actual workflow. There is no iteration between ASO and TO -- the TO-optimized internal structure could change the fan's mass distribution, stiffness, and deflection under load, which in turn changes the aerodynamic behavior (a flexible fan deforms under air loads, changing its effective shape).

**Evidence:** This is a well-known issue in multidisciplinary design optimization (MDO). The sequential approach can converge to suboptimal designs when the coupling is strong. For a thin, flexible fan blade, aeroelastic effects (blade bending under aerodynamic load) could meaningfully change the effective camber and thus the airflow characteristics.

**Alternative:** The report should either:
1. Add a coupling loop: After TO (Phase 3), re-run CFD on the deformed (loaded) shape to verify aerodynamic performance, and iterate if the performance has degraded significantly
2. Use a simple aeroelastic estimate to quantify whether the coupling matters: if the blade tip deflection under aerodynamic load is < 1-2% of the chord length, sequential decoupling is justified; if larger, iteration is needed
3. For advanced users, mention DAFoam + OpenMDAO as a tool that can handle coupled aero-structural optimization in a single framework (the report already mentions DAFoam but does not connect it to this coupling problem)

At minimum, Phase 4 Step 19 ("CFD verification on the TO-optimized geometry") should be emphasized as a coupling check, not just a verification step, with clear criteria for when re-iteration is needed.

---

### 2.4 MODERATE: The von Mises stress equation presented is the 2D plane stress version, but the project is described as 3D

**Challenge:** Section 2.1.3 presents the von Mises stress as:
```
sigma_vm = sqrt(sigma_x^2 + sigma_y^2 - sigma_x*sigma_y + 3*tau_xy^2)
```
This is the 2D plane stress simplification. The report describes a 3D optimization project with 3D FEA solvers (CalculiX), which will compute the full 3D stress tensor.

**Evidence:** The full 3D von Mises stress is:
```
sigma_vm = sqrt(0.5*[(sigma_x - sigma_y)^2 + (sigma_y - sigma_z)^2 + (sigma_z - sigma_x)^2 + 6*(tau_xy^2 + tau_yz^2 + tau_zx^2)])
```
(Source: [Von Mises yield criterion, Wikipedia](https://en.wikipedia.org/wiki/Von_Mises_yield_criterion))

**Alternative:** Present the full 3D equation, then note that for thin plate-like structures (which a fan blade is), plane stress assumptions may be reasonable and simplify to the 2D form. This gives the reader the complete picture and explains when each form is appropriate.

**Impact:** Low for practical purposes (the FEA software will use the correct 3D formulation regardless), but for an educational report targeting beginners, presenting the wrong dimensionality of a key equation without qualification is confusing.

---

### 2.5 MODERATE: GP surrogate feasibility with 15-25 design variables is oversold

**Challenge:** Section 5.2.1 states GP regression is suitable for "up to ~20 design variables," and Section 7.3 defines 15-25 ASO design variables. The report then recommends fitting a GP to 50-100 CFD samples (Phase 2, Step 9-10). This is at the edge of GP scalability, and the recommended sample count may be insufficient.

**Evidence:** The curse of dimensionality means that for 15-25 design variables, the number of training points needed for a reasonable GP fit grows dramatically. Rules of thumb suggest 5-10 samples per dimension for basic coverage, meaning 75-250 samples for 15-25 variables. However, at the upper end of this range, GP kernel hyperparameter optimization becomes unreliable with only 50-100 samples, and the uncertainty estimates (critical for BO) may be poorly calibrated. (Source: [A survey on high-dimensional Gaussian process modeling (arXiv)](https://arxiv.org/pdf/2111.05040))

Furthermore, the report recommends 30 initial LHS samples for 10 design variables in the BoTorch example (Section 9.4), which is 3x the dimensionality -- adequate for a smooth, low-dimensional function but potentially insufficient for the complex aerodynamic response surface of an oscillating fan.

**Alternative:**
1. Recommend dimensionality reduction before surrogate fitting: reduce the 15-25 FFD parameters to 5-10 via PCA or active subspace methods
2. Be more conservative about sample counts: recommend at least 5-10x the number of design variables for initial LHS
3. Emphasize that the multi-fidelity approach (Section 5.4) is not optional but essentially required to make this feasible within a reasonable compute budget
4. Mention that for 20+ variables, the report's "Advanced path" (BoTorch + GPyTorch) should be the default, not the "Beginner path" (scikit-learn), because scikit-learn's GP implementation does not scale well

---

### 2.6 MODERATE: Post-processing of TO results (the "interpretation gap") is dramatically underspecified

**Challenge:** Section 8, Phase 3, Step 16 says "Smooth the TO output, ensure printability, close small holes, verify minimum feature sizes." This single bullet point glosses over what is often the hardest and most time-consuming step in the entire workflow, especially for beginners.

**Evidence:** Topology optimization produces a density field (voxel-like output), not a clean CAD model. Converting this to a printable STL requires:
- Thresholding the density field (choosing a cutoff, typically 0.5)
- Surface extraction (e.g., marching cubes)
- Mesh smoothing (Laplacian or Taubin smoothing)
- Feature identification and manual cleanup
- Verification that the smoothed geometry still satisfies the structural constraints

Research confirms this is a major pain point: "The tools currently available for smoothing the geometry are not reliable enough and a manual reconstruction by a designer has to be considered." (Source: [Systematical redesign method for topology optimized results using 3D-printing (Springer, 2023)](https://link.springer.com/article/10.1007/s44245-023-00019-2), [Surface smoothing for topological optimized 3D models (Springer, 2021)](https://link.springer.com/article/10.1007/s00158-021-03027-6))

**Alternative:** The report should:
1. Dedicate a full subsection to post-processing, comparable in detail to the tool guides in Section 9
2. Recommend specific tools: MeshLab or Meshmixer for STL cleanup, or nTopology's implicit modeling if budget allows
3. Warn that post-processing can change structural performance by 10-20%, necessitating the FEA verification in Phase 4 Step 18
4. Suggest an alternative approach: use Fusion 360's generative design (if licensed) which produces editable CAD geometry rather than raw mesh, or use a level-set TO method (OpenLSTO) which produces smoother boundaries natively

---

### 2.7 MINOR: The aerodynamic objective function is poorly defined for an oscillating fan

**Challenge:** The report defines the ASO objective as "maximize directed airflow" or "minimize drag for given airflow" (Section 8, Step 8). However, for an oscillating fan (not a steady-state device), the objective function is more nuanced. Minimizing drag is actually counterproductive -- a fan that produces zero drag also produces zero airflow. The useful output is the net momentum flux directed toward the user, integrated over a full oscillation cycle.

**Evidence:** The report's SU2 configuration (Section 9.3) uses `OBJECTIVE_FUNCTION= DRAG` which optimizes for steady-state drag minimization. This is the wrong objective for a fan. A fan needs to maximize the pressure difference across its surface (and thus the induced airflow) while minimizing the portion of drag that is wasted (e.g., edge vortices, flow separation that does not contribute to directed airflow).

For an oscillating fan, the correct objective would be something like:
```
maximize: integral over one cycle of (net momentum flux toward user) dt
subject to: integral over one cycle of (waving power input) dt <= P_max
```
This is a fundamentally unsteady optimization problem. SU2 can handle unsteady adjoint optimization (as confirmed by the [SU2 Unsteady Shape Optimization tutorial](https://su2code.github.io/tutorials/Unsteady_Shape_Opt_NACA0012/)), but the setup is substantially more complex than the steady-state configuration shown in the report.

**Alternative:** The report should:
1. Define the fan efficiency metric clearly: eta = (directed momentum flux per cycle) / (energy input per cycle)
2. Acknowledge that the SU2 config shown is a simplified steady-state proxy and explain its limitations
3. Provide guidance on setting up the unsteady optimization (even if simplified to a few key snapshots in the oscillation cycle)
4. Alternatively, recommend a simpler proxy objective that correlates with fan performance: maximize the pressure coefficient integral on the downstream face at peak velocity, which is a steady-state surrogate for the unsteady problem

---

### 2.8 MINOR: Fatigue estimate for polymers is too simplistic

**Challenge:** Section 2.1.4 states `sigma_fatigue ~ 0.3 to 0.5 * sigma_ultimate` as a rough estimate for polymers. This single-number endurance ratio is borrowed from metals fatigue and is not well-supported for 3D-printed thermoplastics.

**Evidence:** 3D-printed polymer fatigue behavior is heavily dependent on print parameters (raster angle, infill density, layer height), environmental conditions (temperature, humidity), and loading frequency. For FDM PETG specifically, research shows that fatigue performance varies significantly with raster angle -- 45-degree raster outperforms 0-degree at low stress amplitudes, while 0-degree is better at high stress amplitudes. A single endurance ratio does not capture this complexity. (Source: [Tensile and Fatigue Analysis of 3D-Printed PETG (ResearchGate)](https://www.researchgate.net/publication/332001021_Tensile_and_Fatigue_Analysis_of_3D-Printed_Polyethylene_Terephthalate_Glycol), [Effect of 3D Printing Parameters on Fatigue Properties (MDPI, 2023)](https://www.mdpi.com/2076-3417/13/2/904))

Furthermore, the report's estimate of N = 10,000 cycles per session seems high. At 2 Hz waving frequency, that would be 83 minutes of continuous waving. A more realistic session might be 5-15 minutes (600-1800 cycles). However, the total lifetime cycle count (daily use over months/years) could be much higher, making the fatigue consideration more important, not less.

**Alternative:** Rather than attempting to apply a metals-style endurance ratio, the report should:
1. Recommend using published S-N curves for FDM PETG (available in the literature cited above)
2. Note that print parameters dramatically affect fatigue life
3. Recommend conservative design: keep peak stresses below 30% of yield as a pragmatic rule for FDM parts under cyclic loading
4. Adjust the cycle count estimate to reflect realistic usage patterns

---

## 3. Missing Elements

### 3.1 No discussion of aeroelastic flutter or resonance

A thin, topology-optimized fan blade with reduced material could have natural frequencies close to the waving frequency (1-3 Hz) or its harmonics. If the blade's first bending mode is excited during waving, resonance could cause excessive vibration, noise, or fatigue failure. The report should recommend a modal analysis (natural frequency calculation) as part of Phase 4 verification to ensure the fundamental frequency is well above the waving frequency and its first few harmonics.

### 3.2 No mention of multi-objective optimization

The report frames TO and ASO as single-objective problems. In reality, there are competing objectives: minimize weight vs. maximize stiffness vs. maximize airflow vs. minimize waving effort. A Pareto front showing the tradeoff between weight and airflow performance would be far more useful than a single "optimal" design. BoTorch supports multi-objective BO via `qExpectedHypervolumeImprovement` -- this should be mentioned.

### 3.3 Beginner feasibility is questionable for the full workflow

The report estimates 8-9 weeks for the full project. For a beginner to TO/ASO, this is highly optimistic. Learning SU2 alone (command-line CFD tool with complex configuration files) is a multi-week endeavor. Add FreeCAD + BESO setup, BoTorch/Python surrogate modeling, mesh generation with Gmsh, post-processing, and 3D printing iteration -- a realistic timeline for a beginner is 3-6 months.

The report should either:
1. Provide a "minimum viable project" that strips away the ML surrogate and uses direct optimization (SU2 adjoint-based ASO directly, without the surrogate layer) for a much simpler workflow
2. Or be honest about the time investment and skill prerequisites

### 3.4 No discussion of surface roughness effects

FDM-printed surfaces have significant roughness (layer lines) that can affect boundary layer transition at these low Reynolds numbers. The CFD simulation assumes smooth surfaces, but the printed fan will have staircase-like layer artifacts. At Re = 10,000-50,000, surface roughness can trigger early transition from laminar to turbulent flow, which changes the drag and lift characteristics. This could mean the optimized shape performs differently in reality than in simulation.

### 3.5 No cost/benefit analysis of optimization vs. simple design heuristics

The report does not address a fundamental question: is the optimization worth it? Traditional fan makers have centuries of empirical design knowledge. A simple cambered fan with a stiff rim and flexible center (achievable without any optimization) might capture 80% of the theoretical improvement. The report should justify the engineering effort by estimating the expected improvement (it mentions 15-30% in Section 10.2, but this is not tied to any analysis or literature).

---

## 4. Verdict: NEEDS REVISION

### Specific changes needed (in priority order):

1. **Fix the Fusion 360 pricing claim** -- generative design is not free for personal use. This is a factual error that will mislead beginners. (Section 3.1, 3.2, 3.3, Appendix A)

2. **Add a section on FDM anisotropy and its impact on TO validity** -- at minimum acknowledge the limitation, connect it to print orientation guidance, and recommend directional safety factors. (New subsection in Section 2.1 or Section 7)

3. **Address the TO/ASO coupling gap** -- add an iteration check between Phase 3 and Phase 2, or provide criteria for when sequential decoupling is acceptable. (Section 8, Phase 3-4)

4. **Expand the TO post-processing section** -- this is a major practical hurdle for beginners that is currently a single bullet point. (Section 8, Phase 3, Step 16 needs expansion into a full subsection)

5. **Fix the aerodynamic objective function** -- the current SU2 config optimizes the wrong thing for a fan. At minimum, acknowledge the limitation and suggest the correct unsteady formulation. (Sections 4.2, 8 Step 8, 9.3)

6. **Present the full 3D von Mises equation** with a note about when the 2D simplification applies. (Section 2.1.3)

7. **Add realistic timeline estimates for beginners** and consider providing a "minimum viable project" variant. (Section 8)

8. **Add modal analysis / resonance check** to the verification phase. (Section 10)

---

*Review conducted 2026-03-22. Web searches used to verify claims about Fusion 360 pricing, FDM material anisotropy, GP scalability, PETG fatigue data, SU2 unsteady optimization capabilities, and TO post-processing challenges. All cited sources were accessible at time of review.*
