# Topology Optimization and Aerodynamic Shape Optimization for a 3D-Printed Hand Fan

## Comprehensive Design, Optimization, and Fabrication Guide

**Date:** 2026-03-22 (Revised)
**Revision:** R4 -- Addressing Followup 1 Round 1 Review: ML Maturity Reclassification, Fusion 360 Credits Fix, Compute Budget Corrections
**Scope:** Combined structural topology optimization (TO) and aerodynamic shape optimization (ASO) with ML surrogate modeling for a 3D-printed Asian-style hand fan.

---

## Table of Contents

1. [Project Overview and Physics Background](#1-project-overview-and-physics-background)
2. [Key Equations and Physical Models](#2-key-equations-and-physical-models)
3. [Software Tools for Topology Optimization](#3-software-tools-for-topology-optimization)
4. [Software Tools for Aerodynamic Shape Optimization](#4-software-tools-for-aerodynamic-shape-optimization)
5. [ML Surrogate Modeling: Core Workflow](#5-ml-surrogate-modeling-core-workflow)
6. [3D Printing Materials Selection](#6-3d-printing-materials-selection)
7. [Design Constraints and Parameters](#7-design-constraints-and-parameters)
8. [Step-by-Step Project Execution Plan](#8-step-by-step-project-execution-plan)
9. [Step-by-Step Tool Guides](#9-step-by-step-tool-guides)
10. [Validation Approaches](#10-validation-approaches)
11. [References and Sources](#11-references-and-sources)

---

## 1. Project Overview and Physics Background

### 1.1 What We Are Designing

A hand fan is a deceptively complex engineering object. When waved back and forth, its blade must:

- **Maximize airflow** directed toward the user (aerodynamic performance).
- **Resist bending and fatigue** from repeated oscillatory loading (structural integrity).
- **Minimize weight** for comfortable prolonged use (material efficiency).
- **Be 3D-printable** without excessive supports or post-processing.

This project combines two distinct optimization disciplines --- structural topology optimization and aerodynamic shape optimization --- into a unified workflow, accelerated by machine learning surrogate models. ML is not an optional accelerator here; it is a core component of the design pipeline, enabling exploration of design spaces that would be computationally prohibitive with simulation alone.

### 1.2 Fan Types Considered

| Fan Type | Description | Optimization Complexity |
|----------|-------------|------------------------|
| **Rigid paddle fan** (uchiwa) | Fixed flat or curved blade on a handle | Simpler --- single-piece optimization |
| **Folding fan** (sensu/ogi) | Multiple ribs with fabric/membrane between them | Complex --- individual rib TO + overall blade ASO |

**Recommendation for a first project:** Start with a rigid paddle fan (uchiwa style). It presents a single continuous design domain for both TO and ASO. A folding fan can be attempted as a Phase 2 project once the workflow is established.

### 1.3 The Two Optimization Problems

**Topology Optimization (TO):** Given a fixed outer boundary (the fan blade envelope), determine the optimal distribution of material within that boundary to maximize stiffness while minimizing mass. The result is an organic, lattice-like internal structure.

**Aerodynamic Shape Optimization (ASO):** Determine the optimal outer shape of the fan blade --- its planform (outline), camber (curvature), and thickness distribution --- to maximize airflow generation when the fan is waved in an oscillatory motion.

These two problems are coupled: the outer shape affects the aerodynamic loads, and those loads determine the structural requirements that TO must satisfy. The internal structure produced by TO also affects mass distribution and stiffness, which in turn affects blade deflection under aerodynamic loads --- potentially changing the effective aerodynamic shape (aeroelastic coupling). Section 8 describes how we handle this coupling in practice through an iterative verification loop.

### 1.4 Is This Optimization Worth the Effort?

Before diving into the workflow, a pragmatic question: traditional fan makers have centuries of empirical design knowledge. A simple cambered fan with a stiff rim and flexible center --- achievable without any optimization --- likely captures a significant portion of the theoretical aerodynamic improvement.

**Justification for optimization:** The primary value of this project lies in (a) learning the TO/ASO workflow on a tangible, low-stakes project, (b) demonstrating quantifiable improvement through systematic engineering, and (c) the internal structure produced by TO is genuinely difficult to arrive at through intuition alone. Published TO studies on thin plate-like structures consistently show 15-30% stiffness-to-weight improvements over manually designed rib patterns. For aerodynamics, the gains are more modest (5-15% in airflow efficiency for a hand fan), making the structural optimization the higher-value component for this particular application.

### 1.5 User Priorities

This report is structured around the following explicit priority ordering:

1. **Minimize cost** --- student budget; free tools strongly preferred
2. **Minimize effort** --- established, well-documented software with existing tutorials; no bleeding-edge research prototypes
3. **Maximize results** --- best achievable fan performance within the above constraints

All tool recommendations are ranked against these priorities. Where a tool is technically superior but poorly documented or expensive, it is noted but not recommended as the primary path.

---

## 2. Key Equations and Physical Models

### 2.1 Structural Mechanics --- Topology Optimization

#### 2.1.1 The SIMP Formulation

The Solid Isotropic Material with Penalization (SIMP) method is the most widely used TO approach. Each finite element in the design domain is assigned a pseudo-density `rho_e` in [0, 1], where 0 = void and 1 = solid material.

**Material interpolation (power law):**

```
E(rho_e) = E_min + rho_e^p * (E_0 - E_min)
```

Where:
- `E_0` = Young's modulus of the solid material (e.g., PLA: ~3500 MPa)
- `E_min` = small value to prevent singularity (typically 1e-9 * E_0)
- `p` = penalization exponent (typically p = 3), which drives intermediate densities toward 0 or 1
- `rho_e` = element pseudo-density (design variable)

**Compliance minimization problem:**

```
minimize:    C(rho) = F^T * u = sum_e (rho_e^p * u_e^T * K_0_e * u_e)
subject to:  K(rho) * u = F                    (equilibrium)
             sum_e (rho_e * v_e) <= V*          (volume constraint)
             0 < rho_min <= rho_e <= 1          (bounds)
```

Where:
- `C` = compliance (strain energy; lower = stiffer)
- `F` = applied force vector (from aerodynamic loading and inertial forces)
- `u` = displacement vector
- `K(rho)` = global stiffness matrix, assembled from element stiffnesses
- `K_0_e` = element stiffness matrix for solid material
- `v_e` = element volume
- `V*` = maximum allowed material volume (volume fraction target)

**Sensitivity (gradient):**

```
dC/d(rho_e) = -p * rho_e^(p-1) * u_e^T * K_0_e * u_e
```

This analytical gradient makes SIMP highly efficient for gradient-based optimization.

#### 2.1.2 Density Filtering

To prevent checkerboard patterns and ensure mesh-independent results, a density filter is applied:

```
rho_tilde_e = (sum_i H_ei * v_i * rho_i) / (sum_i H_ei * v_i)
```

Where `H_ei = max(0, r_min - dist(e, i))` is a linear hat function with filter radius `r_min`. A typical `r_min` is 1.5 to 3 times the element size.

#### 2.1.3 Stress Constraints (Von Mises)

For the fan to survive repeated waving without failure, we use the von Mises yield criterion.

**Full 3D von Mises stress:**

```
sigma_vm = sqrt(0.5 * [(sigma_x - sigma_y)^2 + (sigma_y - sigma_z)^2 + (sigma_z - sigma_x)^2
           + 6 * (tau_xy^2 + tau_yz^2 + tau_zx^2)])
```

This is the general form that 3D FEA solvers (CalculiX, etc.) will compute internally.

**2D plane stress simplification (thin plate):**

For thin plate-like structures such as a fan blade, where through-thickness stresses are negligible (sigma_z ~ 0, tau_yz ~ 0, tau_zx ~ 0), this reduces to:

```
sigma_vm = sqrt(sigma_x^2 + sigma_y^2 - sigma_x * sigma_y + 3 * tau_xy^2)
```

This simplification is reasonable for a fan blade (thickness 1.5-4 mm, span 200-300 mm), but should be verified by checking that through-thickness stress components from the 3D FEA are indeed small (< 5% of in-plane stresses).

The von Mises stress at every point must satisfy `sigma_vm < sigma_yield / SF`, where `SF` is a safety factor (typically 1.5-2.0 for fatigue loading).

**Typical yield strengths for 3D-printed materials:**
- PLA: ~50-60 MPa (but brittle)
- PETG: ~40-50 MPa (more ductile)
- Nylon (PA): ~40-85 MPa (excellent fatigue life)
- PLA-CF: ~60-70 MPa (stiff but brittle)

#### 2.1.4 Fatigue Considerations

A hand fan undergoes oscillatory loading. For realistic usage estimates:
- **Per session:** At 2 Hz waving for 5-15 minutes, expect 600-1,800 cycles per session.
- **Lifetime:** Daily use over months could accumulate 50,000-500,000 total cycles, making fatigue a genuine design concern.

**Polymer fatigue is complex and differs fundamentally from metals fatigue.** The simple endurance ratio approach (sigma_fatigue ~ fraction of sigma_ultimate) commonly used for metals is poorly supported for FDM thermoplastics. For FDM-printed parts specifically:

- Fatigue life is heavily dependent on print parameters (raster angle, infill density, layer height), environmental conditions (temperature, humidity), and loading frequency.
- For FDM PETG, research shows that raster angle significantly affects fatigue performance: 45-degree raster outperforms 0-degree at low stress amplitudes, while 0-degree is better at high stress amplitudes.
- Inter-layer fatigue performance is substantially worse than in-plane fatigue.

**Practical recommendation:** Rather than applying a single endurance ratio:
1. Use published S-N curve data for FDM PETG when available (see References 43-44).
2. As a conservative design rule, keep peak cyclic stresses below 30% of the material's yield strength.
3. Recognize that print parameters (especially raster angle relative to load direction) dramatically affect fatigue life.
4. PLA is notably poor in fatigue (brittle fracture). PETG and nylon are significantly better choices for a repeatedly loaded structure.

#### 2.1.5 FDM Material Anisotropy and Its Impact on TO

**Critical limitation of the SIMP formulation:** SIMP stands for "Solid *Isotropic* Material with Penalization." The standard formulation assumes the material has identical mechanical properties in all directions. FDM-printed parts are fundamentally anisotropic --- they exhibit 20-40% reduction in mechanical properties in the build direction (Z-axis) compared to the in-plane (XY) directions, due to inter-layer bond weakness.

This means the topology optimizer may produce features (thin ribs, slender struts) that appear optimal under isotropic assumptions but are structurally weak when printed, particularly if those features carry loads across layer boundaries.

**Why this matters for our fan:** A fan blade printed flat on the build plate has its layer planes parallel to the blade surface. The primary bending loads during waving act in-plane (XY), which is the strong direction. This is favorable --- the isotropic SIMP assumption is most valid when the dominant load paths align with the strong printing directions. However, any out-of-plane ribs or features produced by TO would carry loads across layers and be weaker than SIMP predicts.

**Mitigation strategies (in order of increasing complexity):**

1. **Print flat (mandatory):** Always orient the fan blade flat on the build plate so primary bending loads are in the XY (strong) plane. Section 7.4 details print orientation.
2. **Apply directional safety factors:** When interpreting TO results, apply a 2x safety factor to any structural features that carry loads in the Z (inter-layer) direction.
3. **Verify with FEA post-optimization:** After post-processing the TO result, run a verification FEA (Phase 4, Step 18) using orthotropic material properties to check that stresses remain acceptable. In CalculiX, define the material with different E, G, and nu values for each direction.
4. **For advanced users --- anisotropy-aware TO:** Recent research (2024-2025) has developed Solid Orthotropic Material with Penalization (SOMP) formulations that account for directional material properties during optimization. These are available in research codes but not yet in standard tools like FreeCAD/BESO. See References 45-47 for relevant papers.

**Key insight:** The isotropic SIMP formulation is acceptable for this project IF AND ONLY IF (a) the fan is printed flat, and (b) the dominant load paths in the optimized structure align with the strong (XY) printing directions. The post-optimization FEA verification in Phase 4 serves as the safety net.

### 2.2 Aerodynamics --- Fan Blade in Oscillatory Motion

#### 2.2.1 Flow Regime

A hand fan waved at typical speeds operates in a specific flow regime:

- **Fan chord length (L):** 0.15-0.25 m
- **Waving velocity (V):** 1-3 m/s (tip speed depends on angular velocity and fan length)
- **Kinematic viscosity of air (nu):** 1.5e-5 m^2/s
- **Reynolds number:** Re = V * L / nu = 10,000 - 50,000

This is a **low Reynolds number, unsteady flow** regime. The flow is laminar to transitional, and unsteady effects (vortex shedding, wake interaction) are significant.

#### 2.2.2 Key Aerodynamic Quantities

**Drag force (resistance to waving motion):**

```
F_D = 0.5 * rho_air * V^2 * A * C_D
```

Where:
- `rho_air` = 1.225 kg/m^3 (air density at sea level)
- `V` = instantaneous fan velocity (m/s)
- `A` = frontal area of the fan blade (m^2)
- `C_D` = drag coefficient

For a flat plate normal to the flow (worst case, no curvature): `C_D ~ 1.28`. With aerodynamic shaping (camber, edge profiles), this can be modified to redirect airflow more efficiently.

**Fan performance metric --- directed momentum flux:**

The useful output of a hand fan is not simply "low drag" --- a fan that produces zero drag also produces zero airflow. The correct figure of merit is the **net directed momentum flux** toward the user per unit of energy input:

```
eta_fan = (integral over one cycle of net momentum flux toward user dt) /
          (integral over one cycle of waving power input dt)
```

In practice, we want to maximize the volume flow rate `Q` (m^3/s) of air directed toward the user per unit of waving effort. A useful proxy metric for steady-state CFD is the **pressure coefficient integral** on the downstream face at peak velocity:

```
J_fan = integral over downstream face of C_p dA
```

Where:
```
C_p = (p - p_inf) / (0.5 * rho_air * V^2)
```

Mapping `C_p` across the fan surface reveals regions that contribute most to airflow generation versus wasted drag (edge vortices, flow separation).

**Important:** Minimizing drag is NOT the correct objective for a fan. See Section 4.2 and 8 Step 8 for the proper ASO objective formulation.

#### 2.2.3 Unsteady Aerodynamics

Because a hand fan oscillates, the governing equations are the unsteady Navier-Stokes equations:

```
rho * (du/dt + (u . nabla)u) = -nabla(p) + mu * nabla^2(u) + f
nabla . u = 0    (incompressible continuity)
```

For CFD simulation, due to the low Reynolds number regime, a **laminar solver** or low-Re turbulence model (e.g., k-omega SST with low-Re corrections) is appropriate. Direct Numerical Simulation (DNS) may even be feasible for 2D simplified cases given the moderate Reynolds numbers.

#### 2.2.4 Simplified Oscillating Plate Model

The fan motion can be modeled as:

```
theta(t) = theta_max * sin(2*pi*f*t)
```

Where:
- `theta_max` = maximum waving angle (~30-45 degrees from center)
- `f` = waving frequency (~1-3 Hz)

The instantaneous velocity at the fan tip:

```
V_tip(t) = L_arm * d(theta)/dt = L_arm * theta_max * 2*pi*f * cos(2*pi*f*t)
V_tip_max = L_arm * theta_max * 2*pi*f
```

For `L_arm` = 0.3 m, `theta_max` = 0.7 rad (40 deg), `f` = 2 Hz:
`V_tip_max` = 0.3 * 0.7 * 2 * pi * 2 = 2.64 m/s

This confirms the Re ~ 10,000-50,000 regime.

#### 2.2.5 Surface Roughness Effects on Aerodynamic Performance

FDM-printed surfaces have significant roughness due to layer lines (typical Ra = 5-15 micrometers for 0.2mm layer height). At the low Reynolds numbers relevant to hand fans (Re = 10,000-50,000), surface roughness can meaningfully affect aerodynamic behavior:

- **Early boundary layer transition:** Layer-line roughness can trigger premature transition from laminar to turbulent flow, changing drag and lift characteristics compared to the smooth-surface CFD model.
- **Laminar separation bubble:** At these Reynolds numbers, laminar separation bubbles are common on smooth surfaces. Roughness can suppress the separation bubble, which may actually improve performance in some configurations (a fortuitous effect).
- **Simulation vs. reality gap:** CFD simulations typically assume smooth walls. The printed fan will have staircase-like layer artifacts that alter the boundary layer.

**Practical mitigation:**
1. In CFD, add wall roughness modeling if available (SU2 supports roughness via `WALL_ROUGHNESS` keyword).
2. Use 0.12-0.15mm layer heights for final prints to minimize roughness.
3. Post-process critical surfaces: light sanding with 400-800 grit, or vapor smoothing for ABS/ASA.
4. Accept a 5-10% performance gap between simulated and actual aerodynamic behavior, and use physical validation (Section 10.2) as the ground truth.

---

## 3. Software Tools for Topology Optimization

### 3.1 Comparison Table (Ranked by Cost-Effort-Results Priority)

Tools are ranked by the user's explicit priority: minimize cost, then minimize effort, then maximize results.

| Rank | Tool | Cost | Effort (Beginner) | Result Quality | AM Constraints | Notes |
|------|------|------|-------------------|----------------|----------------|-------|
| **1** | **Fusion 360 (Educational License)** | Free (students) | Low (5/5) | High | Yes (full AM) | Best overall IF you qualify. See Section 3.2 for caveats. |
| **2** | **FreeCAD + BESO + CalculiX** | Free | Medium (3/5) | Good | Partial | Best free option for non-students. Good tutorials. |
| **3** | **FreeCAD FEM + FEMbyGEN** | Free | Medium (3/5) | Good | Yes | GUI-based alternative to BESO |
| **4** | **TopOpt (Python)** | Free | Low for 2D (4/5) | 2D only | No | Learning tool only, not for production |
| **5** | **OpenLSTO** | Free | High (2/5) | High (smooth) | No | Smoother results but poor docs |
| **6** | **FEniCS + SIMP** | Free | High (2/5) | High | No | Research-grade, steep learning curve |
| **7** | **Fusion 360 (Commercial)** | $680+/yr | Low (5/5) | High | Yes | Only if student license unavailable |
| **8** | **Altair Inspire** | ~$7,000/yr | Medium (4/5) | Very High | Yes | Industry standard, out of budget |
| **9** | **ANSYS Topology Optimization** | ~$25,000+/yr | Medium (3/5) | Very High | Yes | Overkill |

### 3.2 Fusion 360 Educational License: Full Analysis

**Research finding: Fusion 360 Generative Design is included with educational licenses, with unlimited cloud credits when properly configured.**

Based on investigation of Autodesk's official documentation, community forums, and third-party analyses:

**What is included:**
- The educational license (free for students at accredited institutions with a .edu email) provides access to the Generative Design workspace in Fusion 360.
- Educational licenses support Generative Design studies, producing multiple topology-optimized design candidates with AM constraints (overhang angle, minimum thickness, build direction).
- Full CAD, CAM, simulation, and rendering capabilities are included.
- Export formats include STL, STEP, and native Fusion formats.
- **Unlimited cloud credits** for Generative Design and Simulation, drawn from the educational credit pool. This is confirmed by [Autodesk's official support documentation](https://www.autodesk.com/support/technical/article/caas/sfdcarticles/sfdcarticles/Cloud-credits-for-the-Education-Community.html) and multiple independent sources ([Product Design Online](https://productdesignonline.com/tips-and-tricks/how-to-get-fusion-360-for-free/)).

**Common "no credits" error --- this is a configuration issue, not a feature limitation:**
- Students who see a "no cloud credits" or "unlimited credits not available" error have typically not configured their account to draw from the educational credit pool. This is a known UI issue.
- **Fix:** Click your account icon (top right in Fusion 360) and ensure you are signed in under your educational license, not a personal or free-tier account. The educational pool is separate from any personal credit balance. ([Autodesk Support: Education Plan Unlimited Cloud Credits](https://www.autodesk.com/support/technical/article/caas/sfdcarticles/sfdcarticles/Education-Plan-Unlimited-Cloud-Credits-are-not-available-to-start-a-simulation-in-Fusion-360.html))
- If the issue persists after switching accounts, contact Autodesk Education support to verify your institution's enrollment status.

**Other limitations:**
- **No commercial use:** Designs created under the educational license cannot be sold or used commercially.
- **Cloud dependency:** Generative Design runs on Autodesk's cloud servers; no offline use is possible.

**Recommendation:** Verify your setup with a quick configuration check before committing to this path:
1. Register for the Fusion 360 educational license at [autodesk.com/education](https://www.autodesk.com/education/edu-software/fusion).
2. Verify your account is set to use educational credits (account icon > check license type).
3. Complete Autodesk's [Generative Design tutorial](https://www.autodesk.com/products/fusion-360/blog/topology-optimization-is-not-generative-design/).
4. Run a simple Generative Design study (e.g., a bracket). This should work with unlimited credits. If you encounter a "no credits" error, apply the account-switching fix above.
5. If you still cannot access credits after troubleshooting, fall back to FreeCAD/BESO (Rank 2).

### 3.3 Head-to-Head: Fusion 360 (Educational) vs. FreeCAD/BESO

| Criterion | Fusion 360 (Educational) | FreeCAD + BESO + CalculiX |
|-----------|--------------------------|---------------------------|
| **Cost** | Free (with .edu email) | Free (no restrictions) |
| **Ease of use** | Excellent. Modern GUI, built-in tutorials, one-click Generative Design setup. A complete beginner can run a study in 1-2 hours. | Moderate. FreeCAD GUI is functional but less polished. BESO plugin requires manual installation. Expect 1-2 days to run a first study. |
| **TO algorithm** | Cloud-based proprietary solver. Generates 5-50+ design candidates with different trade-offs. Handles AM constraints (overhang, min thickness) natively. | BESO (evolutionary method). Single candidate per run. AM constraints are partial (overhang angle via manual post-processing, not built into optimizer). |
| **Output quality** | Produces editable parametric CAD geometry (B-rep bodies). No post-processing mesh cleanup needed. This is a major advantage. | Produces a density field that requires extensive post-processing: thresholding, marching cubes surface extraction, smoothing, repair (see Section 8.3.1). This pipeline is the single biggest pain point. |
| **Export formats** | STL, STEP, 3MF, SAT, native Fusion (.f3d). STEP export preserves parametric features for further editing. | STL (from post-processed mesh). STEP export requires additional conversion. |
| **Integration with ASO pipeline** | Good for geometry export. Cannot integrate with SU2/BoTorch directly (closed ecosystem). Export STEP/STL from Fusion, then import into meshing tools (Gmsh). | Good. FreeCAD can export directly to formats consumed by Gmsh/SU2. Python scripting enables automation. |
| **Integration with ML pipeline** | Poor. Fusion 360 is a closed system --- you cannot script Generative Design studies programmatically or integrate with BoTorch. ML surrogate for TO must be done externally. | Good. CalculiX can be scripted via command-line. FEA results can be parsed and fed into Python ML pipeline. |
| **Limitations** | Cloud dependency (no offline use). Proprietary algorithm (cannot inspect or modify solver internals). Limited to Autodesk's parameter space for studies. Must ensure account is configured for educational credit pool (see Section 3.2). | BESO is less mathematically rigorous than SIMP. FreeCAD can crash on complex operations. CalculiX documentation is sparse. |
| **Tutorials available** | Extensive. Autodesk's own YouTube channel, Lars Christensen, Product Design Online, dozens of third-party courses. | Moderate. FreeCAD FEM tutorials exist (DigiKey, MangoJelly). BESO-specific tutorials are fewer. |
| **Community support** | Large (Autodesk forums, Reddit r/Fusion360). | Active but smaller (FreeCAD forum, GitHub issues). |

**Verdict:** If you have a .edu email, use Fusion 360 for TO --- educational licenses include unlimited cloud credits when properly configured (see Section 3.2). Fusion 360 eliminates the hardest part of the open-source pipeline (post-processing density fields into printable geometry). However, the ML-for-TO extension (DL4TO or custom, Section 5.3) must still be done externally in Python regardless of which TO tool you use. For the ASO pipeline, both paths converge on the same tools (SU2, BoTorch).

### 3.4 Tool Details

#### TopOpt (Python)
- **Pros:** Pure Python, pip-installable, minimal dependencies, educational, SIMP with MMA solver, runs in minutes for 2D problems
- **Cons:** Primarily 2D (extensions to 3D require FEniCS), no GUI, no AM constraint support, not production-ready
- **Best for:** Understanding the algorithm, prototyping optimization formulations
- **Tutorials:** DTU TopOpt codes are the gold standard for learning

#### FreeCAD + BESO + CalculiX
- **Pros:** Full GUI, integrated CAD-to-FEA-to-TO workflow, free and open source, active community, handles 3D tetrahedral meshes, export to STL/STEP
- **Cons:** BESO is not as mathematically rigorous as SIMP, FreeCAD GUI can be unintuitive, CalculiX documentation is sparse, setup requires multiple component installation, TO output requires substantial post-processing (see Section 8.3.1)
- **Best for:** Producing actual optimized fan geometry for 3D printing when Fusion 360 is unavailable
- **Tutorials:** [DigiKey FreeCAD FEM tutorial](https://www.digikey.com/en/maker/tutorials/2025/intro-to-freecad-part-10-finite-element-method-fem-workbench-tutorial), [BESO GitHub examples](https://github.com/calculix/beso)

#### Fusion 360 Generative Design
- **Pros:** Cloud-computed (no local hardware needed), generates multiple design candidates, handles AM constraints (overhang angle, minimum thickness), produces editable CAD geometry, intuitive workflow, extensive tutorials
- **Cons:** Requires .edu email for free educational access; cloud dependency (no offline use); limited control over optimization algorithm internals; proprietary; cannot integrate programmatically with ML pipeline. Students must ensure their account draws from the educational credit pool (see Section 3.2 for troubleshooting).
- **Best for:** Getting high-quality results quickly with minimal setup effort
- **Tutorials:** [Autodesk Generative Design getting started](https://www.autodesk.com/products/fusion-360/blog/topology-optimization-is-not-generative-design/), Product Design Online YouTube channel

---

## 4. Software Tools for Aerodynamic Shape Optimization

### 4.1 Comparison Table: Software Tools for Aerodynamic Analysis and Shape Optimization

Tools are ranked by the user's explicit priority: minimize cost, then minimize effort, then maximize results. Note that some tools perform CFD analysis only (no built-in optimization loop), while others support adjoint-based shape optimization.

| Rank | Tool | Cost | Effort (Beginner) | Can Perform ASO? | Adjoint Support | Unsteady | Notes |
|------|------|------|-------------------|------------------|-----------------|----------|-------|
| **1** | **SimScale** | Free tier | Low (5/5) | **No (analysis only)** | No | Yes | Browser-based CFD. Use for learning, validation, and ML training data generation. Cannot perform shape optimization --- it runs individual simulations only. |
| **2** | **SU2** | Free | High (3/5) | **Yes** | Yes (continuous + discrete) | Yes | Best for adjoint-based ASO. Steep learning curve. |
| **3** | **XFLR5** | Free | Low (4/5) | **No (analysis only)** | No | No | 2D panel method, good for quick screening |
| **4** | **OpenFOAM + adjointOptimisationFoam** | Free | Very High (2/5) | **Yes** | Yes (continuous) | Limited | Most flexible, hardest to learn |
| **5** | **DAFoam** | Free | Very High (2/5) | **Yes** | Yes (discrete) | Yes | For coupled aero-structural MDO |
| **6** | **AirShaper** | Free tier | Very Low (5/5) | **No (analysis only)** | No | Limited | Upload-and-go, no optimization loop |

**Note on ranking:** SimScale is ranked first for effort minimization because it requires zero installation, runs in a browser, and has excellent tutorials. However, **SimScale is a CFD analysis tool, not a shape optimizer** --- it cannot run an optimization loop or compute shape sensitivities. SU2 is essential for the adjoint-based ASO. In practice, use SimScale for learning, quick CFD validation, and generating training data for the ML surrogate, and SU2 for the actual optimization pipeline.

### 4.2 Recommended Tools: SU2 (ASO) + SimScale (Validation)

**SU2** is the recommended primary tool for aerodynamic shape optimization. It has built-in adjoint solvers for computing shape sensitivities, a shape optimization pipeline, and extensive tutorials. It handles incompressible and compressible flows and supports Free-Form Deformation (FFD) for parameterizing shape changes.

**Critical note on objective function:** For a fan, the objective is NOT drag minimization (see Section 2.2.2). SU2's default `OBJECTIVE_FUNCTION= DRAG` optimizes for steady-state drag reduction, which is wrong for a device whose purpose is to generate airflow. The correct approach is described in Section 8 Step 8 and Section 9.3.

**For unsteady optimization:** SU2 supports unsteady adjoint-based optimization (see the [Unsteady Shape Optimization NACA0012 tutorial](https://su2code.github.io/tutorials/Unsteady_Shape_Opt_NACA0012/)). This is the proper formulation for an oscillating fan but is substantially more complex to set up. Section 9.3 provides a steady-state proxy approach for beginners, with guidance on upgrading to the unsteady formulation.

**SimScale** is recommended as a complementary tool for quick CFD validation. Its browser-based interface requires no installation, has a free tier, and provides visual results that help build intuition. It is also useful for generating initial CFD training data for the ML surrogate (Section 5).

**For 2D preliminary studies:** XFLR5 provides fast panel-method analysis of airfoil-like cross-sections, useful for quickly screening fan blade cross-sectional profiles.

**For coupled aero-structural optimization:** DAFoam with OpenMDAO can handle coupled aerodynamic and structural optimization in a single MDO framework, avoiding the sequential decoupling limitations discussed in Section 8. This is the advanced path for users who find that aeroelastic coupling is significant for their design.

### 4.3 Tool Pros and Cons (Detailed)

#### SU2
- **Pros:** Purpose-built for shape optimization, adjoint-based sensitivities, Free-Form Deformation (FFD), well-documented tutorials, active community, supports RANS/Euler/incompressible flows, Python scripting interface, supports unsteady adjoint
- **Cons:** Command-line driven, requires mesh generation externally (Pointwise, Gmsh), configuration files are complex with hundreds of options, Linux/Mac preferred (Windows via WSL), unsteady optimization setup is substantially more complex than steady-state
- **Best for:** The actual shape optimization loop for the fan blade
- **Tutorials:** [SU2 Tutorial Collection](https://su2code.github.io/tutorials/home/), [SU2 Quick Start](https://su2code.github.io/docs/Quick-Start/)

#### SimScale
- **Pros:** Zero installation, browser-based, visual mesh and result inspection, free Community tier, handles transient simulations, built-in meshing
- **Cons:** No built-in shape optimization loop (manual iteration), free tier has limited compute time and public projects, no adjoint solver, internet-dependent
- **Best for:** Quick CFD validation, building aerodynamic intuition, generating training data for ML surrogates
- **Tutorials:** [SimScale Tutorial Library](https://www.simscale.com/tutorials/)

#### OpenFOAM + adjointOptimisationFoam
- **Pros:** Extremely flexible, handles any flow physics, adjoint shape optimization built into v1906+, massive solver library, well-validated
- **Cons:** Steep learning curve, Linux-only (or WSL/Docker), text-file-based setup, debugging is difficult, meshing requires separate tools (snappyHexMesh, blockMesh, or Gmsh)
- **Best for:** Users already familiar with OpenFOAM who need custom physics

#### DAFoam
- **Pros:** Discrete adjoint (more accurate than continuous for complex flows), Python interface, OpenMDAO integration for MDO, handles hundreds of design variables, well-documented, can handle coupled aero-structural optimization
- **Cons:** Complex installation (Docker recommended), requires OpenFOAM knowledge, primarily aimed at aerospace applications, steep learning curve
- **Best for:** Advanced users wanting to integrate aero and structural optimization in a single MDO framework

---

## 5. ML Surrogate Modeling: Core Workflow

**ML surrogate modeling is a core component of this project, not an optional accelerator.** This section details exactly which models to use, what training data to generate, and how the ML models integrate with both the TO and ASO pipelines.

### 5.1 Why ML Surrogates Are Mandatory

There are two distinct roles for ML in this project:

1. **ML for ASO (Aerodynamic Shape Optimization):** A single CFD simulation of the fan in unsteady oscillatory motion takes 1-4 hours. Exploring 15-25 shape parameters via direct CFD would require thousands of simulations (months of compute time). An ML surrogate replaces the CFD solver with a model that predicts aerodynamic performance in milliseconds, enabling Bayesian optimization over the full design space within a manageable compute budget.

2. **ML for TO (Topology Optimization):** ML can accelerate the inner loop of SIMP iterations by predicting compliance and sensitivities without solving the full FEA system at every iteration. This reduces TO wall-clock time by 5-50x. Additionally, ML can predict near-optimal topologies for new load cases, enabling rapid exploration of the TO design space when coupled with ASO. **Important maturity distinction:** Unlike ML-for-ASO (which uses production-quality tools like BoTorch), ML-for-TO methods are at an earlier maturity level. The most practical option is the DL4TO library (early-stage but functional); custom implementations require research-paper-level engineering effort. See Section 5.3.0 for a maturity assessment and Section 5.3 for tiered recommendations.

### 5.2 ML for Aerodynamic Shape Optimization (ASO)

#### 5.2.1 Recommended Model: Gaussian Process (GP) with BoTorch

**Why GP (not neural network) for ASO:** With a realistic compute budget of 100-300 CFD simulations, the training set is too small for neural networks to generalize reliably. GPs are the correct choice because:
- They provide calibrated uncertainty estimates (critical for Bayesian optimization)
- They perform well with 50-300 training samples
- They support Bayesian optimization natively via acquisition functions
- BoTorch provides a production-quality implementation with GPU acceleration

**Model specification:**

```python
# GP model for ASO surrogate
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean

# Kernel: Matern 5/2 with Automatic Relevance Determination (ARD)
# ARD learns a separate lengthscale per dimension, effectively performing
# implicit dimensionality reduction
kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=n_dims))

# The SingleTaskGP in BoTorch uses Matern 5/2 + ARD by default
gp_model = SingleTaskGP(train_X, train_Y)
```

**Why Matern 5/2:** It assumes the underlying function is twice-differentiable, which matches aerodynamic performance functions. The RBF (squared exponential) kernel assumes infinite differentiability, which can oversmooth and miss sharp features in the performance landscape.

**When to add a neural network:** If the initial GP surrogate has NRMSE > 15% after 150+ training samples, consider a hybrid approach: use a neural network to learn a feature representation, then fit a GP on the learned features (Deep Kernel Learning via GPyTorch). This is more data-efficient than a pure neural network while retaining GP uncertainty quantification.

#### 5.2.2 Training Data Pipeline for ASO

**Design parameters (inputs):**

| Parameter Group | Count | Range | Description |
|----------------|-------|-------|-------------|
| Planform control points | 8-12 | +/- 20mm | FFD control points defining fan outline |
| Camber parameters | 4-6 | 0-15mm | Blade curvature distribution |
| Thickness distribution | 3-5 | 1.5-4mm | Thickness from center to edge |
| Edge profile | 2 | 0.5-2mm | Leading/trailing edge radius |

**Total raw parameters:** 17-25. **After PCA dimensionality reduction:** 8-12 effective parameters.

**Outputs (targets):**

| Output | Source | Description |
|--------|--------|-------------|
| `J_fan` (airflow metric) | CFD | Pressure integral on downstream plane (Section 2.2.2) |
| `C_D` (drag coefficient) | CFD | Resistance to waving motion |
| `max_stress` (Pa) | FEA (optional) | Peak von Mises stress under aero loading |
| `mass` (g) | Geometry | Total fan blade mass |

**Data generation protocol:**

```
Step 1: Dimensionality reduction
  - Generate 300 random designs via LHS in the raw 17-25D space
  - Run PCA on the design matrix
  - Retain components explaining 95% of variance (typically 8-12)
  - All subsequent sampling is in PCA space

Step 2: Initial training set (low-fidelity)
  - Generate 200 designs via Latin Hypercube Sampling in PCA space
  - Run coarse-mesh steady CFD for each (~5 min/run on 4 cores)
  - Total compute: ~17 hours (parallelizable to ~4 hours on 4 machines)

Step 3: Initial training set (high-fidelity)
  - Select 40 designs from LHS (space-filling subset)
  - Run fine-mesh steady CFD for each (~1-2 hrs/run)
  - Total compute: ~40-80 hours (parallelizable)

Step 4: Multi-fidelity GP training
  - Fit multi-fidelity GP correlating low and high-fidelity data
  - Validate with leave-one-out cross-validation
  - Target: NRMSE < 10% on high-fidelity predictions

Step 5: Bayesian optimization loop (50-100 iterations)
  - Each iteration: GP predicts performance, acquisition function
    selects next design, one high-fidelity CFD confirms
  - Total additional high-fidelity CFD: 50-100 runs
```

**Total CFD budget:** ~200 low-fidelity + 90-140 high-fidelity runs = approximately 130-260 hours of single-core compute time.

#### 5.2.3 Bayesian Optimization Loop for ASO

**Acquisition function:** Expected Improvement (EI) is recommended as the default. For multi-objective optimization (airflow vs. weight), use `qNoisyExpectedHypervolumeImprovement` (qNEHVI) from BoTorch.

**Expected Improvement:**
```
EI(x) = E[max(f(x) - f_best, 0)]
       = (f_best - mu(x)) * Phi(Z) + sigma(x) * phi(Z)
where Z = (f_best - mu(x)) / sigma(x)
```

**Upper Confidence Bound (alternative):**
```
UCB(x) = mu(x) + beta * sigma(x)
```
where `beta` controls exploration vs. exploitation (typical: beta = 2.0).

**Complete BO loop pseudocode:**

```python
import torch
from botorch.models import SingleTaskGP, SingleTaskMultiFidelityGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc
import subprocess
import json

# ============================================================
# CONFIGURATION
# ============================================================
N_DIMS = 10          # PCA-reduced design dimensions
N_INITIAL_LF = 200   # Low-fidelity initial samples
N_INITIAL_HF = 40    # High-fidelity initial samples
N_BO_ITERS = 80      # Bayesian optimization iterations
NRMSE_THRESHOLD = 0.15

bounds = torch.tensor([[-0.02] * N_DIMS, [0.02] * N_DIMS], dtype=torch.double)

# ============================================================
# STEP 1: CFD SIMULATION WRAPPER
# ============================================================
def run_cfd(design_params, fidelity="high"):
    """
    Run SU2 CFD and return fan performance metric.

    Args:
        design_params: tensor of shape (N_DIMS,) in PCA space
        fidelity: "low" (coarse mesh, ~5 min) or "high" (fine mesh, ~1-2 hr)

    Returns:
        dict with keys: 'J_fan', 'C_D', 'mass'
    """
    # 1. Transform PCA params back to physical FFD control points
    physical_params = pca_inverse_transform(design_params.numpy())

    # 2. Write SU2 mesh deformation file
    write_ffd_file("fan_ffd.cfg", physical_params)

    # 3. Deform mesh
    subprocess.run(["SU2_DEF", "fan_ffd.cfg"], check=True)

    # 4. Run CFD (mesh density depends on fidelity)
    cfg_file = "fan_coarse.cfg" if fidelity == "low" else "fan_fine.cfg"
    subprocess.run(["SU2_CFD", cfg_file], check=True)

    # 5. Parse results
    results = parse_su2_history("history.csv")
    return results['surface_total_pressure']  # J_fan proxy

# ============================================================
# STEP 2: GENERATE INITIAL TRAINING DATA
# ============================================================
sampler = qmc.LatinHypercube(d=N_DIMS)

# Low-fidelity samples
X_lf = torch.tensor(sampler.random(n=N_INITIAL_LF), dtype=torch.double)
X_lf = bounds[0] + (bounds[1] - bounds[0]) * X_lf
Y_lf = torch.tensor(
    [run_cfd(x, fidelity="low") for x in X_lf], dtype=torch.double
).unsqueeze(-1)

# High-fidelity samples (space-filling subset)
hf_indices = select_space_filling_subset(X_lf, N_INITIAL_HF)
X_hf = X_lf[hf_indices]
Y_hf = torch.tensor(
    [run_cfd(x, fidelity="high") for x in X_hf], dtype=torch.double
).unsqueeze(-1)

# ============================================================
# STEP 3: MULTI-FIDELITY GP FITTING
# ============================================================
# Combine low and high fidelity data with fidelity indicator
# fidelity = 0 for low, 1 for high
X_all = torch.cat([
    torch.cat([X_lf, torch.zeros(len(X_lf), 1, dtype=torch.double)], dim=1),
    torch.cat([X_hf, torch.ones(len(X_hf), 1, dtype=torch.double)], dim=1),
])
Y_all = torch.cat([Y_lf, Y_hf])

mf_gp = SingleTaskMultiFidelityGP(
    X_all, Y_all,
    data_fidelities=[N_DIMS]  # last column is fidelity indicator
)
mf_mll = ExactMarginalLogLikelihood(mf_gp.likelihood, mf_gp)
fit_gpytorch_mll(mf_mll)

# ============================================================
# STEP 4: BAYESIAN OPTIMIZATION LOOP
# ============================================================
for iteration in range(N_BO_ITERS):
    # Fit GP on high-fidelity data only for acquisition
    gp = SingleTaskGP(X_hf, Y_hf)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # Validate surrogate every 10 iterations
    if iteration % 10 == 0:
        nrmse = compute_loo_nrmse(gp, X_hf, Y_hf)
        print(f"  Surrogate NRMSE: {nrmse:.3f}")
        if nrmse > NRMSE_THRESHOLD:
            print(f"  WARNING: NRMSE > {NRMSE_THRESHOLD}. "
                  f"Consider adding more training data.")

    # Compute acquisition function
    best_f = Y_hf.max()
    EI = ExpectedImprovement(model=gp, best_f=best_f)

    # Optimize acquisition function to find next candidate
    candidate, acq_value = optimize_acqf(
        EI, bounds=bounds, q=1, num_restarts=20, raw_samples=512
    )

    # Run HIGH-FIDELITY CFD on the candidate
    new_y = run_cfd(candidate.squeeze(), fidelity="high")

    # Update training data
    X_hf = torch.cat([X_hf, candidate])
    Y_hf = torch.cat([Y_hf, torch.tensor([[new_y]], dtype=torch.double)])

    print(f"Iter {iteration}: best J_fan = {Y_hf.max().item():.4f}, "
          f"acq = {acq_value.item():.6f}")

    # Early stopping: if acquisition value is negligible
    if acq_value.item() < 1e-6:
        print(f"Converged at iteration {iteration}")
        break

# ============================================================
# STEP 5: EXTRACT BEST DESIGN
# ============================================================
best_idx = Y_hf.argmax()
best_design_pca = X_hf[best_idx]
best_design_physical = pca_inverse_transform(best_design_pca.numpy())
print(f"Best design (physical FFD params): {best_design_physical}")
print(f"Best J_fan: {Y_hf.max().item():.4f}")
```

#### 5.2.4 Multi-Objective Bayesian Optimization

The fan design involves competing objectives: minimize weight vs. maximize stiffness vs. maximize airflow vs. minimize waving effort. Rather than collapsing these into a single scalar, multi-objective BO can discover the Pareto front --- the set of optimal trade-offs where improving one objective necessarily worsens another.

**BoTorch supports this natively** via `qNoisyExpectedHypervolumeImprovement` (qNEHVI) and `qExpectedHypervolumeImprovement` (qEHVI). These acquisition functions extend expected improvement to the multi-objective setting by computing improvement in hypervolume dominated by the Pareto front.

**Recommended formulation for this project:**

- **Objective 1:** Maximize airflow metric (J_fan from Section 2.2.2)
- **Objective 2:** Minimize total fan mass
- **Constraint:** Maximum stress < yield / SF
- **Reference point:** The baseline unoptimized fan's performance (used to compute hypervolume)

```python
from botorch.acquisition.multi_objective import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.utils.multi_objective.pareto import is_non_dominated

# Multi-objective GP: one GP per objective
from botorch.models import ModelListGP

gp_airflow = SingleTaskGP(X_hf, Y_airflow)
gp_mass = SingleTaskGP(X_hf, Y_mass)  # negative mass (to maximize)
model = ModelListGP(gp_airflow, gp_mass)

# Reference point: slightly worse than worst observed
ref_point = torch.tensor([Y_airflow.min() - 0.1, Y_mass.min() - 0.1])

qNEHVI = qNoisyExpectedHypervolumeImprovement(
    model=model,
    ref_point=ref_point,
    X_baseline=X_hf,
)

candidate, _ = optimize_acqf(
    qNEHVI, bounds=bounds, q=1, num_restarts=20, raw_samples=512
)
```

### 5.3 ML for Topology Optimization (TO)

**Can ML be used WITHIN the TO process itself?** Yes, but the maturity level is fundamentally different from ML-for-ASO. While the ASO surrogate uses production-quality tools (BoTorch, GPyTorch) with extensive tutorials and stable APIs, ML-for-TO methods remain in the research domain with no equivalent pip-installable, tutorial-documented solution. This section is structured to reflect that reality honestly.

#### 5.3.0 ML Maturity Assessment

Before selecting an approach, understand the maturity gap:

| ML Component | Maturity Level | Effort to Implement | Existing Libraries | Tutorials Available |
|---|---|---|---|---|
| GP + BoTorch (ASO surrogate) | **Production** | Low-Medium (1-2 weeks) | BoTorch, GPyTorch (pip-installable, stable API) | [Extensive](https://botorch.org/tutorials/) |
| Bayesian Optimization (ASO) | **Production** | Low (included in BoTorch) | BoTorch, Ax | [Extensive](https://ax.dev/tutorials/) |
| DL4TO (ML-accelerated TO) | **Early-stage library** | Medium (2-3 weeks) | [DL4TO](https://github.com/dl4to/dl4to) (pip-installable from GitHub, 124 stars, Apache 2.0) | [Documentation site](https://dl4to.github.io/dl4to/), conference paper |
| Custom MLP/U-Net for TO | **Research** | High (4-8 weeks) | None (implement from papers) | None (paper descriptions only) |

**Key distinction:** The ML-for-ASO pipeline (GP + BoTorch) uses production-quality tools that a beginner can set up by following tutorials. The ML-for-TO pipeline requires either (a) using the emerging DL4TO library, which is functional but early-stage with a small community, or (b) custom implementation from research papers, which is a substantial engineering undertaking. **ML-for-TO should be treated as an advanced extension of the project, not a core requirement for a first iteration.** The standard SIMP workflow (without ML acceleration) produces correct results --- ML-for-TO accelerates it, but a single TO run on 500K elements takes 1-4 hours (Section 5.3.3), which is tolerable without ML acceleration for a first project.

**Recommendation for balancing "ML is mandatory" with "minimize effort":**
- **Core path (mandatory):** ML-for-ASO via BoTorch (Section 5.2). This is where ML provides the most value per unit effort.
- **Recommended extension:** Use DL4TO for ML-accelerated TO (Section 5.3.1). This is a real library with documentation, not a from-scratch implementation.
- **Advanced extension:** Custom MLP/U-Net for TO (Sections 5.3.2-5.3.3). Only pursue this if you have ML engineering experience and want to deeply integrate ML into the TO loop.

#### 5.3.1 Recommended Approach: DL4TO Library (Practical ML-for-TO)

**[DL4TO](https://github.com/dl4to/dl4to)** is a PyTorch-based Python library specifically designed for integrating deep learning with 3D topology optimization. It is the closest thing to a "pip-installable ML-for-TO" tool that currently exists.

**Installation:**

```bash
pip install git+https://github.com/dl4to/dl4to
# Also requires: pip install torch pyvista==0.38.1
```

**What DL4TO provides:**
- Built-in SIMP solver using PyTorch's autodifferentiation (differentiable physics)
- Pre-built U-Net architecture for topology prediction
- Dataset generation and management for TO problems
- Supervised and unsupervised training pipelines
- Interactive 3D voxel mesh visualization
- Compatible with the [SELTO benchmark dataset](https://github.com/dl4to/dataset)

**What DL4TO does NOT provide:**
- Integration with external FEA solvers (CalculiX, ABAQUS) --- it uses its own finite difference PDE solver
- Unstructured mesh support (works on structured voxel grids only)
- Direct integration with the FreeCAD/BESO workflow described in this report

**Practical workflow with DL4TO:**

```
Step 1: Define the fan blade TO problem in DL4TO's structured grid format
  - Discretize the fan blade envelope into a voxel grid (e.g., 64x64x8)
  - Define loads from CFD results, boundary conditions, volume fraction

Step 2: Run SIMP baseline using DL4TO's differentiable solver
  - This gives you a baseline topology AND generates training data

Step 3: Train a neural network (DL4TO's built-in U-Net) to predict
  topologies for varied load cases
  - DL4TO handles the data pipeline, training loop, and evaluation

Step 4: Use the trained model to rapidly predict topologies for new
  ASO-proposed load cases (warm-starting SIMP)
```

**Limitations:** DL4TO operates on structured voxel grids, not unstructured tetrahedral meshes. For the fan blade geometry, this means either (a) approximating the blade shape on a voxel grid (acceptable for learning and rapid exploration), or (b) using DL4TO for coarse-resolution exploration and then refining the best candidates with a full SIMP run on the unstructured mesh in FreeCAD/BESO. Option (b) is recommended.

**Maturity caveat:** DL4TO has 124 GitHub stars, 15 commits on main, and is backed by University of Bremen research. It is functional and documented, but it is not as battle-tested as BoTorch. Expect to spend some time working through its API. Budget 2-3 weeks for integration.

#### 5.3.2 Advanced Extension A: ML-Accelerated SIMP (Compliance Prediction)

**Maturity: RESEARCH-STAGE. This approach requires custom implementation from papers. Budget 4-8 weeks for a first implementation. Only pursue this if you have prior PyTorch experience and want to deeply integrate ML into the CalculiX/BESO SIMP loop.**

**Concept:** During SIMP iterations, the most expensive step is solving the linear system `K(rho) * u = F` to compute displacements (and thus compliance and sensitivities). A neural network trained on previous iterations can predict compliance and approximate sensitivities, replacing 50-80% of the FEA solves.

**Architecture: 3D CNN (NOT a naive MLP).** The density field of a 3D TO problem has spatial structure that must be exploited. A naive MLP taking a flattened 500K-element density vector as input would have ~256 million parameters in the first layer alone and would massively overfit on the 200-500 training samples available from SIMP iterations. Research implementations use one of three approaches to handle this:

1. **3D Convolutional Network (recommended):** Treat the density field as a 3D volume and use 3D convolutions, which exploit spatial locality and have far fewer parameters.
2. **Dimensionality reduction + MLP:** Apply PCA to the density field (reducing 500K dimensions to 50-200 principal components), then use an MLP on the reduced representation.
3. **Coarse-grid proxy:** Run ML-accelerated SIMP on a coarser mesh (e.g., 50K elements), then refine the result on the full 500K mesh with standard SIMP for 10-20 iterations.

```python
import torch
import torch.nn as nn

class CompliancePredictor3DCNN(nn.Module):
    """
    Predicts compliance from 3D density field using convolutions.
    Input: (batch, 1, D, H, W) density volume
    Output: scalar compliance value

    For a 500K element mesh discretized as ~80x80x80 voxels:
    - Parameters: ~2M (vs ~256M for naive MLP)
    - Can train on 200-500 samples without severe overfitting
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool3d(2),  # 40x40x40
            nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool3d(2),  # 20x20x20
            nn.Conv3d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool3d(2),  # 10x10x10
            nn.Conv3d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),  # 1x1x1
        )
        self.regressor = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, rho):
        x = self.features(rho)
        x = x.view(x.size(0), -1)
        return self.regressor(x)
```

**Training data pipeline for ML-accelerated TO:**

```
Step 1: Run standard SIMP for 30-50 iterations (full FEA each iteration)
  - Record at each iteration:
    - Input: density field rho (3D volume)
    - Output: compliance C (scalar)
  - Also run SIMP on 5-10 different load cases (different pressure
    distributions from ASO candidates) to diversify training data

Step 2: Train compliance predictor (3D CNN)
  - Training set: ~200-500 (density_volume, compliance) pairs from Step 1
  - Validation: 20% held-out
  - Loss: MSE for compliance
  - Training time: ~15-30 minutes on a laptop GPU

Step 3: Hybrid SIMP loop
  - Iterations 1-30: Full FEA (collect training data)
  - Iterations 31+: Alternate between:
    - ML-predicted compliance (fast, ~10ms)
    - Full FEA verification every 5th iteration (expensive, confirms ML)
  - If ML prediction error > 5% vs FEA, retrain with new data

Step 4: Final verification
  - Always run full FEA on the final converged design
  - ML is used to accelerate exploration, not to replace final validation
```

**The real implementation challenge** is not the neural network architecture (shown above) but the glue code: parsing CalculiX FRD/VTK output files into PyTorch tensors, implementing the hybrid SIMP loop that switches between ML and FEA, handling the retraining trigger, and managing the data pipeline. None of this glue code exists in any library. You must implement it from scratch, which is where most of the 4-8 week effort goes.

**Expected speedup:** 5-10x reduction in TO wall-clock time. Published results report up to 50x speedup for large 3D problems (>500K elements), though speedup depends heavily on problem size.

**Is the speedup worth the implementation effort?** For this project: probably not for a first iteration. A single TO run on 500K elements takes 1-4 hours (see Section 5.3.3). The ASO-TO coupling loop might require 5-20 TO re-runs, totaling 5-80 hours of TO compute. ML acceleration could reduce this to 1-16 hours. The time saved (4-64 hours of compute) must be weighed against 4-8 weeks of implementation time. **ML-for-TO becomes valuable when running TO hundreds of times (e.g., tightly coupled ASO-TO co-optimization), which is the advanced use case.**

#### 5.3.3 Advanced Extension B: ML Topology Predictor (Design Space Exploration)

**Maturity: RESEARCH-STAGE. Same caveats as Section 5.3.2.**

**Concept:** Train a neural network to predict near-optimal topologies directly from load cases and volume fractions, without running SIMP at all. This is used for rapid exploration: when the ASO loop proposes a new outer shape (and thus new aerodynamic loads), the ML model instantly predicts what the optimal internal structure would look like for those loads.

**Architecture:** U-Net (encoder-decoder CNN) mapping (load case, boundary conditions, volume fraction) to density field.

```python
class TopologyUNet(nn.Module):
    """
    Predicts optimal density field from load/BC specification.
    Input: (batch, channels, H, W) where channels encode:
      - Channel 0: load magnitude at each node
      - Channel 1: load direction (x-component)
      - Channel 2: load direction (y-component)
      - Channel 3: boundary condition mask (1=fixed, 0=free)
      - Channel 4: volume fraction target (uniform fill)
    Output: (batch, 1, H, W) predicted density field
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = self._conv_block(5, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        # Decoder
        self.dec3 = self._conv_block(256 + 128, 128)
        self.dec2 = self._conv_block(128 + 64, 64)
        self.dec1 = nn.Conv2d(64, 1, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d3 = self.dec3(torch.cat([self.up(e3), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e1], dim=1))
        return torch.sigmoid(self.dec1(d2))
```

**Training data pipeline for topology predictor:**

```
Step 1: Generate diverse TO training problems
  - Vary: load magnitude (5-50 Pa), load direction, boundary conditions,
    volume fraction (0.2-0.5)
  - For each variant, run full SIMP TO to convergence
  - Collect 200-500 (input, converged_density_field) pairs

Step 2: Train U-Net
  - Loss: Binary Cross-Entropy on density field
    + compliance penalty (predicted topology should have low compliance)
  - Training: ~30-60 minutes on a laptop GPU
  - Validation: visual inspection + compliance comparison vs. true SIMP

Step 3: Use in ASO-TO coupling loop
  - When ASO proposes new shape -> new aero loads
  - U-Net predicts optimal topology in ~50ms
  - Use prediction as warm-start for 10-20 iterations of real SIMP
    (instead of 100-200 iterations from scratch)
```

**Expected benefit:** Reduces the number of SIMP iterations needed per new load case from 100-200 to 10-30, because the ML prediction is already close to the optimum and SIMP only needs to refine it.

**Realistic compute budget for training data generation:** The U-Net approach requires 200-500 full SIMP runs to convergence. For a 3D problem with 500K elements, each SIMP run involves 100-200 iterations, where each iteration requires a full FEA solve on the 500K element mesh. Realistic per-run times on an 8-core desktop are **2-4 hours** (not 30-60 minutes as originally estimated --- see Section 5.3.3 below for detailed scaling analysis). Total training data compute budget:

| Scenario | Runs | Time/Run | Total Core-Hours | Wall-Clock (8 cores) |
|---|---|---|---|---|
| Minimal | 200 | 2 hrs | 400 | ~4-5 days |
| Standard | 350 | 3 hrs | 1,050 | ~10-12 days |
| Thorough | 500 | 4 hrs | 2,000 | ~20-25 days |

This is a substantial upfront compute investment. It is only justified if you plan to run many ASO-TO coupling iterations (>20). For a first project with sequential ASO-then-TO, the standard SIMP workflow without ML acceleration is sufficient.

**Recommendation:** Generate training data in parallel with the ASO CFD data generation (they are independent tasks). If you have access to a university HPC cluster, the training data generation becomes feasible within 1-3 days.

#### 5.3.4 Realistic Compute Estimates for 3D TO at 500K Elements

The previous revision of this report estimated 30-60 minutes per full SIMP run at 500K elements. This was too optimistic. Here is a more detailed analysis:

**Per-iteration FEA cost:** Each SIMP iteration requires assembling and solving a sparse linear system with ~1.5M DOF (500K elements x 3 DOF/node for solid elements). Using a direct solver (e.g., PARDISO in CalculiX), this takes approximately 30-120 seconds per iteration depending on hardware. Using an iterative solver (e.g., PCG with AMG preconditioner), 10-60 seconds per iteration.

**Full SIMP run:** 100-200 iterations to convergence. At 30-120 seconds per iteration, that is **50 minutes to 6.7 hours** per complete run. The 2-4 hour range is typical for a well-configured run on an 8-core desktop with iterative solvers.

**Scaling reference:** The [PyTopo3D benchmark](https://arxiv.org/abs/2504.05604) reports 147 seconds for 200 iterations on 8,192 elements. Scaling to 500K elements (61x more elements) with O(n^1.2) to O(n^1.5) solver scaling gives roughly 30-120 seconds per iteration, consistent with the 2-4 hour total estimate.

**Bottom line:** Budget 2-4 hours per full 3D SIMP run at 500K elements on an 8-core desktop. For the ML-for-TO training data, budget 400-2000 core-hours total.

### 5.4 Multi-Fidelity Surrogate Strategy

For the ASO surrogate, a multi-fidelity approach is essential to make GP-based BO feasible within a reasonable compute budget:

- **Low fidelity:** Coarse mesh CFD (5 min/run), panel method (seconds/run), or simplified analytical models
- **High fidelity:** Fine mesh RANS CFD (1-4 hrs/run)

A multi-fidelity GP (e.g., AR1 model or MFDNN) correlates cheap low-fidelity data with expensive high-fidelity data:

```
f_high(x) = rho * f_low(x) + delta(x)
```

Where `rho` is a scaling factor and `delta(x)` is a GP capturing the discrepancy. This can reduce the number of expensive high-fidelity runs by 50-80%.

**Implementation in BoTorch:** Use `SingleTaskMultiFidelityGP` with a fidelity parameter (see code in Section 5.2.3).

### 5.5 Handling High Dimensionality (15-25 ASO Variables)

With 15-25 FFD design variables, standard GP regression with 50-100 samples is at the edge of feasibility. The following strategies are essential:

1. **Dimensionality reduction before surrogate fitting:** Reduce the 15-25 FFD parameters to 8-12 effective dimensions using PCA on the initial LHS sample set, or active subspace methods (which identify the directions in parameter space that most affect the output). This can dramatically improve GP fit quality.

2. **Conservative sample counts:** Use at least 8-10x the effective dimensionality for initial LHS sampling. For 10 effective dimensions, that means 80-100 initial samples.

3. **Multi-fidelity is mandatory:** Use 200-500 low-fidelity samples (coarse mesh, 5 min each) supplemented by 30-50 high-fidelity samples (fine mesh, 1-4 hrs each). The multi-fidelity GP leverages cheap data to fill the design space while using expensive data for accuracy.

4. **Use BoTorch, not scikit-learn:** For 15+ variables, BoTorch/GPyTorch with automatic relevance determination (ARD) kernels is strongly preferred. ARD learns per-dimension lengthscales, effectively performing implicit dimensionality reduction.

5. **Validate surrogate quality:** After fitting the GP, compute leave-one-out cross-validation error. If the normalized RMSE exceeds 15%, the surrogate is unreliable and more samples are needed.

### 5.6 Recommended ML Tools (Ranked by Cost-Effort-Results)

| Rank | Tool | Cost | Effort | Best For | Maturity | Tutorials |
|------|------|------|--------|----------|----------|-----------|
| **1** | **BoTorch + GPyTorch** | Free | Medium (3/5) | ASO Bayesian optimization (core tool) | Production | [BoTorch tutorials](https://botorch.org/tutorials/) |
| **2** | **Ax (Adaptive Experimentation)** | Free | Low (4/5) | High-level BO wrapper (simpler than raw BoTorch) | Production | [Ax tutorials](https://ax.dev/tutorials/) |
| **3** | **scikit-learn** | Free | Low (5/5) | Initial GP experimentation, quick prototyping | Production | Extensive sklearn docs |
| **4** | **DL4TO** | Free | Medium (3/5) | ML-accelerated TO (library-based, structured grids) | Early-stage | [DL4TO docs](https://dl4to.github.io/dl4to/) |
| **5** | **PyTorch (custom)** | Free | High (2/5) | Custom neural network surrogates for TO acceleration | N/A (framework) | PyTorch tutorials |

**Recommended stack:**
- **For ASO surrogate (core):** BoTorch + GPyTorch (Bayesian optimization with GP). Production-quality, extensive tutorials.
- **For TO acceleration (recommended extension):** DL4TO (structured-grid ML-accelerated TO). Early-stage but functional library with documentation.
- **For TO acceleration (advanced extension):** PyTorch custom implementation (3D CNN compliance predictor, U-Net topology predictor). Research-stage, implement from papers.
- **For quick prototyping:** scikit-learn GaussianProcessRegressor (limited to <10 variables, <500 samples)

**Note on Ax vs. BoTorch:** Ax is a higher-level wrapper around BoTorch that simplifies common BO workflows. If the BoTorch API feels overwhelming, start with Ax's `optimize` function for single-objective BO, then switch to BoTorch when you need multi-objective or multi-fidelity capabilities.

### 5.7 Summary: What ML Model Does What

| Task | ML Model | Maturity | Input | Output | Training Data Size | Training Data Compute | Training Time |
|------|----------|----------|-------|--------|-------------------|-----------------------|---------------|
| ASO surrogate | GP (BoTorch) | **Production** | 8-12 PCA shape params | J_fan, C_D | 200 LF + 40-140 HF CFD runs | 130-260 core-hrs | seconds |
| ASO optimization | Bayesian Opt (EI/qNEHVI) | **Production** | GP surrogate predictions | Next design to evaluate | N/A (uses GP) | N/A | seconds/iter |
| TO acceleration (DL4TO) | U-Net (DL4TO) | **Early-stage library** | structured grid + loads | density field | 50-200 SIMP runs (DL4TO solver) | 50-200 core-hrs | 30-60 min |
| TO acceleration (custom compliance) | 3D CNN (PyTorch) | **Research** | density volume (3D) | compliance (scalar) | 200-500 SIMP iterations | 20-100 core-hrs | 15-30 min |
| TO design space exploration | U-Net (PyTorch) | **Research** | load case + BCs + vol frac | density field | 200-500 full SIMP runs | 400-2000 core-hrs | 30-60 min |

---

## 6. 3D Printing Materials Selection

### 6.1 Material Properties Comparison

| Property | PLA | PETG | PA (Nylon) | PLA-CF | PA-CF |
|----------|-----|------|------------|--------|-------|
| **Tensile Strength (MPa)** | 50-60 | 40-50 | 40-85 | 60-70 | 80-110 |
| **Flexural Modulus, bulk (GPa)** | 3.5-4.0 | 2.0-2.1 | 1.2-2.0 | 5.0-7.0 | 4.0-7.0 |
| **Young's Modulus, FDM XY (GPa)** | 2.5-3.2 | 1.1-1.5 | 0.8-1.5 | 3.5-5.5 | 3.0-5.5 |
| **Young's Modulus, FDM Z (GPa)** | 1.8-2.5 | 0.8-1.2 | 0.6-1.0 | 2.0-3.5 | 2.0-4.0 |
| **Flexural Strength (MPa)** | ~97 | ~70 | 50-75 | 90-110 | 100-130 |
| **Elongation at Break (%)** | 3-6 | 15-25 | 30-100+ | 2-4 | 5-10 |
| **Impact Resistance** | Poor | Good | Excellent | Poor | Good |
| **Fatigue Resistance** | Poor | Good | Excellent | Poor | Good |
| **Density (g/cm^3)** | 1.24 | 1.27 | 1.10 | 1.20 | 1.15 |
| **Z-direction strength reduction** | 20-30% | 20-35% | 25-40% | 30-50% | 25-40% |
| **Print Difficulty** | Easy | Easy-Medium | Hard | Medium | Hard |
| **Print Temp (C)** | 190-220 | 230-250 | 250-270 | 200-230 | 260-280 |
| **Bed Temp (C)** | 50-60 | 70-80 | 70-90 | 50-60 | 80-100 |
| **Needs Enclosure** | No | No | Yes | No | Yes |
| **Needs Dry Box** | No | Recommended | Essential | Recommended | Essential |
| **Cost ($/kg)** | $15-25 | $20-30 | $30-50 | $30-50 | $50-80 |

**Note on Z-direction strength reduction:** This row quantifies the anisotropy discussed in Section 2.1.5. Values represent the typical reduction in tensile strength when specimens are loaded perpendicular to the layer planes vs. parallel. Carbon fiber composites (PLA-CF, PA-CF) show greater anisotropy because the fibers align in-plane during extrusion.

### 6.2 Material Recommendations by Priority

#### Best Overall: PETG

PETG offers the best balance of properties for a hand fan:
- Sufficient strength for the thin structures produced by TO
- Good flexibility prevents brittle fracture from repeated waving
- Good fatigue resistance for oscillatory loading
- Easy to print with no enclosure needed
- Reasonable cost ($20-30/kg)
- Slightly higher density than PLA but the flexibility advantage is worth it

#### Best Performance: PA-CF (Carbon Fiber Nylon)

If maximum performance is the goal and printing difficulty is acceptable:
- Highest stiffness-to-weight ratio
- Excellent fatigue life
- Superior dimensional stability
- Requires hardened nozzle (steel or ruby), dry filament storage, and an enclosed printer
- Significantly more expensive

#### Best for Prototyping: PLA

For initial test prints and design validation:
- Easiest to print with reliable results
- Cheapest
- Sufficient for evaluating geometry and fit
- Not recommended for the final fan due to brittleness and poor fatigue life

#### Not Recommended: PLA-CF for This Application

Despite its impressive stiffness numbers, PLA-CF is even more brittle than PLA. The carbon fibers increase stiffness but reduce elongation at break to 2-4%, making it prone to sudden fracture under the oscillatory loading of fan waving.

### 6.3 Print Settings for Fan Structures

For topology-optimized thin structures:

| Parameter | PLA (prototype) | PETG (final) | PA-CF (performance) |
|-----------|-----------------|--------------|---------------------|
| **Layer Height** | 0.15-0.2 mm | 0.15-0.2 mm | 0.12-0.15 mm |
| **Wall Count** | 3-4 | 3-4 | 3-4 |
| **Infill** | 20-40% gyroid | 20-40% gyroid | 15-30% gyroid |
| **Top/Bottom Layers** | 4-5 | 4-5 | 5-6 |
| **Print Speed** | 50-60 mm/s | 40-50 mm/s | 30-40 mm/s |
| **Cooling Fan** | 100% | 50-70% | 30-50% |
| **Nozzle** | Brass 0.4mm | Brass 0.4mm | Hardened steel 0.4mm |

**Infill pattern choice:** Gyroid infill is recommended for topology-optimized parts because its isotopic properties complement the TO results. The TO algorithm optimizes the outer shell and rib structure; gyroid infill fills remaining internal volumes efficiently.

---

## 7. Design Constraints and Parameters

### 7.1 Dimensional Constraints

| Parameter | Typical Range | Recommended Starting Point |
|-----------|--------------|---------------------------|
| **Fan blade width** | 200-300 mm | 250 mm |
| **Fan blade height** | 150-250 mm | 200 mm |
| **Handle length** | 80-120 mm | 100 mm |
| **Handle diameter** | 12-18 mm | 15 mm |
| **Blade thickness** | 1.5-4.0 mm | 2.5 mm (envelope for TO) |
| **Edge thickness** | 0.8-1.5 mm | 1.0 mm |
| **Minimum feature size** | 0.8-1.2 mm | 1.0 mm (2.5x nozzle diameter) |

### 7.2 Topology Optimization Constraints

- **Volume fraction target:** 0.3-0.5 (30-50% of the design envelope filled with material)
- **Minimum member size:** 1.0 mm (constrained by 0.4mm nozzle x 2.5 factor)
- **Maximum overhang angle:** 45 degrees from vertical (for support-free printing)
- **Symmetry:** Optional bilateral symmetry constraint (reduces design variables by 50%)
- **Fixed regions:** Handle attachment zone, outer rim of blade (preserved, not optimized)

### 7.3 Aerodynamic Design Variables

For Free-Form Deformation (FFD) in SU2:

- **Planform shape:** 8-12 control points defining the fan outline
- **Camber:** 4-6 control points defining the blade curvature profile
- **Thickness distribution:** 3-5 parameters controlling thickness from center to edge
- **Edge profile:** Leading/trailing edge radius (2 parameters)

Total design variables for ASO: approximately 15-25 parameters.

**Dimensionality reduction (strongly recommended):** Before fitting a GP surrogate, reduce these 15-25 parameters to 8-12 effective dimensions via PCA on an initial LHS sample set, or by fixing less-sensitive parameters based on a sensitivity screening (Morris one-at-a-time method). See Section 5.5.

### 7.4 Print Orientation Considerations

**Recommended orientation: Flat on the build plate (blade horizontal)**

Advantages:
- Maximum blade surface quality (top and bottom)
- No supports needed for flat or gently curved blades
- Strongest inter-layer bonding direction aligned with bending loads
- Fastest print time
- **Critical for TO validity:** Ensures that the isotropic SIMP assumption is most accurate, because primary bending loads act in the XY (strong) plane. See Section 2.1.5.

Disadvantages:
- Large footprint on build plate (may exceed bed size for large fans)
- Layer lines visible on surface (aerodynamic roughness --- see Section 2.2.5)

**Alternative: Blade at 45 degrees**

If the fan exceeds build plate dimensions, printing at 45 degrees reduces footprint but requires support material and increases print time. **Warning:** This orientation introduces significant inter-layer loads during fan waving, making the isotropic TO assumption less valid. Apply a 2x Z-direction safety factor if using this orientation.

**Critical consideration:** The weakest direction in FDM prints is between layers (Z-direction). Orient the fan so that the primary bending moment during waving acts along layer lines (XY plane), not between layers (Z). For a flat fan printed horizontally, this means the flexural loads during waving should be in-plane, which is naturally the case.

---

## 8. Step-by-Step Project Execution Plan

### Timeline Expectations

**Realistic timeline for a beginner:** 3-6 months. The estimate below assumes prior experience with at least one of: CAD, programming, or simulation. For a complete beginner to all three, add 2-4 additional weeks of learning time.

The main time sinks are:
- Learning SU2 (command-line CFD with complex config files): 2-4 weeks for a beginner
- FreeCAD + BESO setup and troubleshooting (or Fusion 360 setup if using student license): 1-2 weeks
- GP/BoTorch surrogate modeling (if no Python experience): 2-3 weeks
- ML-accelerated TO via DL4TO (recommended extension): 2-3 weeks
- ML-accelerated TO via custom implementation (advanced extension): 4-8 weeks
- TO post-processing (Section 8.3.1): 1-2 weeks (skipped if using Fusion 360)
- 3D printing iteration: 1-2 weeks

### Phase 1: Baseline Design and Preliminary Analysis (Week 1-2)

1. **Define requirements:** Target fan size, weight limit, desired airflow, material choice.
2. **Create baseline CAD geometry:** Use FreeCAD (or Fusion 360 if using student license) to model a simple flat paddle fan.
   - Dimensions: 250mm wide x 200mm tall x 2.5mm thick blade, 100mm handle.
3. **Print baseline fan:** Print in PLA for initial testing and tactile feedback.
4. **Measure baseline performance:** Use an anemometer or smoke visualization to characterize airflow from the unoptimized fan waved at a comfortable pace.
5. **Estimate loading conditions:** From waving motion kinematics, calculate approximate aerodynamic and inertial loads on the fan blade.

### Phase 2: Aerodynamic Shape Optimization with ML Surrogate (Week 3-7)

6. **Set up CFD model:** Install SU2 (or use SimScale for initial runs). Create a mesh around the baseline fan geometry.
7. **Run baseline CFD:** Simulate the fan at representative velocity (V = 2 m/s), compute pressure distribution and airflow pattern.
8. **Define ASO problem:**
   - **Objective:** Maximize directed momentum flux (NOT minimize drag --- see Section 2.2.2). In SU2, use `SURFACE_TOTAL_PRESSURE` on a downstream analysis plane as a proxy, or set up an unsteady optimization using the compressible solver with `LOW_MACH_PREC= YES` (see Section 9.3 for all three objective approaches and both solver configurations). **Note:** The steady-state proxy has significant limitations at this reduced frequency (k ~ 0.6) --- see Section 9.3.1 for mandatory validation steps.
   - **Design variables:** Planform shape, camber, thickness distribution. Reduce from 15-25 raw FFD parameters to 8-12 effective dimensions using PCA or sensitivity screening (Section 5.5).
   - **Constraints:** Fan fits within maximum envelope, minimum thickness for printability.
9. **Generate ML training data (mandatory):** Run CFD simulations using Latin Hypercube Sampling:
   - **Low-fidelity:** 200 coarse-mesh runs (~5 min each, ~17 hours total, parallelizable)
   - **High-fidelity:** 40 fine-mesh runs (~1-2 hrs each, ~40-80 hours total, parallelizable)
   - Record for each: design parameters (PCA-reduced), J_fan, C_D, mass
10. **Train ML surrogate (mandatory):**
    - Fit multi-fidelity GP using BoTorch `SingleTaskMultiFidelityGP`
    - Validate with leave-one-out cross-validation; NRMSE should be < 15%
    - If NRMSE > 15%: add more training data, try dimensionality reduction, or switch to Deep Kernel Learning
    - See Section 5.2 for complete code and details
11. **Run Bayesian optimization (mandatory):**
    - Use GP surrogate + Expected Improvement acquisition function (Section 5.2.3)
    - Run 50-100 BO iterations, each confirming with one high-fidelity CFD run
    - Consider multi-objective BO (Section 5.2.4) for trading off airflow vs. weight
    - See Section 5.2.3 for complete BO loop code
12. **Extract optimized shape:** The result is an optimized fan planform and curvature profile.

### Phase 3: Topology Optimization with ML Extension (Week 7-10, or Week 7-14 with advanced ML-for-TO)

13. **Apply aerodynamic loads:** Use the pressure distribution from the optimized CFD solution as the load case for TO.
14. **Set up TO problem:**
    - **If using Fusion 360 (student license):**
      - Import optimized outer shape from Phase 2
      - Define preserve regions (handle, blade rim), obstacle regions (void), load cases
      - Select manufacturing constraint: Additive, with build direction and overhang angle
      - Run Generative Design study (cloud-computed)
      - Select best candidate from generated designs
    - **If using FreeCAD/BESO:**
      - Design domain: The optimized fan blade envelope (from Phase 2)
      - Loads: Pressure distribution from CFD + inertial loads from waving acceleration
      - Constraints: Volume fraction = 0.35, minimum member size = 1.0mm, overhang angle <= 45 deg
      - Fixed regions: Handle, blade rim
      - Material model: Use isotropic properties for the optimization (SIMP)
15. **ML-accelerated TO (recommended extension for FreeCAD/BESO path):**
    - **Option A (recommended): DL4TO library** (Section 5.3.1)
      - Set up the fan blade problem in DL4TO's structured grid format
      - Run SIMP baseline, then train DL4TO's built-in U-Net for rapid TO exploration
      - Use DL4TO predictions as warm-starts for full-resolution SIMP runs
      - Budget: 2-3 weeks setup + 50-200 core-hours training data
    - **Option B (advanced): Custom 3D CNN compliance predictor** (Section 5.3.2)
      - Run standard SIMP for first 30-50 iterations, collecting training data
      - Train 3D CNN compliance predictor (NOT a naive MLP --- see Section 5.3.2)
      - Continue SIMP with ML acceleration for remaining iterations
      - Budget: 4-8 weeks setup + 20-100 core-hours training data
    - Verify final result with full FEA regardless of approach
16. **Train topology predictor for design space exploration (advanced extension):**
    - Run SIMP on 5-10 different load cases from ASO candidates
    - Train U-Net topology predictor (Section 5.3.3)
    - Use to rapidly evaluate TO implications of alternative ASO designs
    - **Note:** This requires 200-500 full SIMP runs at 2-4 hours each = 400-2000 core-hours. Only justified if planning many ASO-TO coupling iterations (>20).

#### 8.3.1 Post-Processing TO Results: From Density Field to Printable STL

**Note: This section applies only to the FreeCAD/BESO path. Fusion 360 Generative Design produces editable CAD geometry directly, skipping all of these steps.**

**This is often the hardest and most time-consuming step, especially for beginners.** Topology optimization produces a density field (voxel-like output), not a clean CAD model. Converting this to a printable STL requires a multi-step pipeline:

**Step 1: Density thresholding**
- Apply a cutoff (typically rho = 0.5) to convert the continuous density field to binary solid/void.
- Elements with rho > 0.5 become solid; others become void.
- Sensitivity check: try thresholds of 0.3, 0.5, and 0.7 to see how the result changes. If it changes dramatically, the optimization may not have converged well (too many intermediate densities).

**Step 2: Surface extraction**
- Use the marching cubes algorithm to extract an isosurface from the thresholded density field.
- In ParaView: open the VTK output from CalculiX/BESO, apply the "Contour" filter on the density field at value 0.5, then export as STL.
- Alternative: use Python with `scikit-image.measure.marching_cubes` or `PyVista`.

**Step 3: Mesh smoothing**
- The raw marching cubes output has staircase artifacts. Apply smoothing:
  - **Taubin smoothing** (preferred): Alternating shrink/expand to smooth without excessive volume loss. Available in MeshLab (Filters > Smoothing > Taubin Smooth, lambda=0.5, mu=-0.53, 10-30 iterations).
  - **Laplacian smoothing:** Simpler but causes volume shrinkage. Use sparingly.
- **Tools:** MeshLab (free, GUI), Meshmixer (free, Autodesk), PyMeshLab (Python scripting).

**Step 4: Mesh repair and cleanup**
- Close small holes, remove non-manifold edges, fix self-intersections.
- MeshLab: Filters > Cleaning and Repairing > Close Holes, Remove Duplicate Faces, etc.
- Meshmixer: Analysis > Inspector (auto-repair).
- Verify minimum feature sizes meet 3D printing requirements (Section 7.1).

**Step 5: Verification FEA on smoothed geometry**
- **Critical:** Post-processing changes the geometry, which changes the structural performance (typically by 10-20% compared to the raw TO result).
- Re-import the smoothed STL into FreeCAD FEM, apply the same loads and constraints, and run FEA to verify stresses are still acceptable.
- If stresses exceed limits, reduce the amount of smoothing or increase the volume fraction target and re-run TO.

**Recommended tools summary:**
| Step | Tool (free) | Tool (commercial) |
|------|-------------|-------------------|
| Surface extraction | ParaView, Python/scikit-image | - |
| Smoothing | MeshLab, PyMeshLab | Meshmixer |
| Repair | MeshLab | Meshmixer, nTopology |
| Verification FEA | FreeCAD FEM + CalculiX | SimScale, Fusion 360 |

**Alternative approach:** If this pipeline is too painful, consider using OpenLSTO (level-set method) instead of SIMP/BESO. Level-set TO produces smoother boundaries natively, largely avoiding Steps 2-4. The trade-off is a less mature toolchain and steeper learning curve.

#### 8.3.2 Diagnosing Slow or Oscillatory Convergence in TO and BO

**Topology optimization convergence issues:**

In practice, TO convergence for complex 3D geometries with AM constraints often stalls or oscillates rather than converging smoothly. Common symptoms and remedies:

| Symptom | Likely Cause | Remedy |
|---------|-------------|--------|
| Compliance oscillates between two values | Filter radius too small, or penalization too aggressive | Increase `r_min` to 3-4x element size; reduce penalization from p=3 to p=2 initially, then ramp to p=3 over iterations |
| Large gray regions (intermediate densities 0.3-0.7) persist | Penalization too low, or insufficient iterations | Increase p gradually (continuation: p=1 -> 2 -> 3 over iterations); run more iterations |
| Checkerboard patterns | No density filter or filter radius too small | Ensure density filter is active with r_min >= 1.5x element size |
| Result changes dramatically with mesh refinement | Mesh-dependent solution (no length-scale control) | Increase filter radius proportionally with element count |
| BESO deletes load-carrying members that never recover | BESO's hard add/remove heuristic | Reduce evolutionary rate (ER) to 1-2%; use SIMP instead if problem persists |
| Convergence stalls at a non-optimal volume fraction | Volume constraint too tight for the load case | Relax volume fraction by 5-10% and re-run |

**When to stop iterating:** If the compliance has stabilized to within 1% over the last 20 iterations but has not reached the 0.1% criterion, the result is likely acceptable. Check: (a) are intermediate densities less than 5% of the domain? (b) does the result look physically reasonable (connected load paths)? If both are true, accept the result and proceed to post-processing.

**Bayesian optimization convergence issues:**

| Symptom | Likely Cause | Remedy |
|---------|-------------|--------|
| BO keeps sampling at domain boundaries | GP lengthscale too large (oversmoothing) | Check GP hyperparameters; ensure ARD kernel is used; add interior training points |
| BO samples cluster in one region, ignoring others | Exploitation dominance (EI too greedy) | Increase exploration: use UCB with higher beta (3-4), or add random samples every 5th iteration |
| Surrogate NRMSE > 15% and not improving | Insufficient training data for the dimensionality | Add more LHS samples (target 10x dimensionality); reduce dimensionality via PCA |
| Best objective not improving after 20+ BO iterations | Near-optimal or surrogate is misleading | Run 5 random samples to check if BO region is truly optimal; verify surrogate on recent points |
| GP hyperparameter optimization fails (NaN, Inf) | Numerical issues from poorly scaled data | Normalize all inputs to [0,1] and outputs to zero mean, unit variance before fitting |

**Budget exhaustion without clear optimum:** If the BO evaluation budget is exhausted without a clearly optimal region emerging, report the Pareto front of evaluated designs and select the best among them. This is still a valid engineering outcome --- the BO has systematically explored the design space and the best evaluated design is likely near-optimal even if the surrogate has not fully converged.

17. **Export optimized geometry:** Save smoothed, verified STL or STEP file.

### Phase 4: Verification, Coupling Check, and Validation (Week 10-13)

18. **FEA verification with orthotropic properties:** Run a stress analysis on the final optimized geometry using direction-dependent material properties (not just isotropic) to verify that the structure is safe given FDM anisotropy:
    - In CalculiX, define orthotropic elastic properties: `*ELASTIC, TYPE=ENGINEERING CONSTANTS` with different E values for XY vs. Z directions.
    - Verify: `max(sigma_vm) < sigma_yield_Z / 1.5` for features loaded in the Z direction, where `sigma_yield_Z` is the reduced yield strength in the build direction.
    - Verify: `max(displacement) < 5mm` at blade tip (acceptable deflection).

19. **Modal analysis (resonance check):**
    - Compute the first 5-10 natural frequencies of the optimized fan blade using FreeCAD FEM or CalculiX (`*FREQUENCY` step).
    - The first bending mode natural frequency must be well above the waving frequency and its first few harmonics: `f_1 > 5 * f_waving` (i.e., f_1 > 10-15 Hz for 2-3 Hz waving).
    - If the natural frequency is too close to the waving frequency or its harmonics, the blade could exhibit resonant vibration, excessive noise, or accelerated fatigue failure.
    - A thin, topology-optimized blade with reduced material is at higher risk of low natural frequencies than a solid blade.

20. **CFD verification and coupling check:** Run a final high-fidelity CFD simulation on the TO-optimized geometry to verify aerodynamic performance is maintained.
    - This step serves as a **coupling check**, not just verification: if the TO-modified geometry (internal structure, edge profiles) significantly changes the aerodynamic performance compared to Phase 2 results, iteration is needed.
    - **Coupling criterion:** If the airflow metric changes by more than 10% compared to the Phase 2 optimized shape, return to Phase 2 Step 8 and re-optimize ASO with the TO-modified geometry as the new baseline.
    - **ML-assisted coupling:** If you have trained the topology predictor (Section 5.3.3, advanced extension), use it to rapidly check whether the new ASO shape would produce a significantly different TO result before committing to a full SIMP run.
    - **Aeroelastic estimate:** If maximum blade tip deflection under aerodynamic load exceeds 2% of chord length, the sequential decoupling assumption breaks down. Consider either (a) iterating between ASO and TO until convergence, or (b) switching to a coupled aero-structural framework (DAFoam + OpenMDAO).

21. **Print prototype:** Print in PETG with the recommended settings from Section 6.3.
22. **Physical testing:**
    - Subjective: Wave the fan and compare comfort/airflow to baseline.
    - Quantitative: Anemometer measurements at fixed distance (see Section 10).
23. **Iterate if necessary:** Adjust volume fraction, camber, or other parameters based on test results.

### Phase 5: Final Production (Week 13-15)

24. **Final print:** Print in PA-CF or PETG depending on performance requirements.
25. **Post-processing:** Light sanding if needed for surface finish (see Section 2.2.5 on roughness effects), handle wrapping for comfort.
26. **Documentation:** Record all parameters, settings, and results for reproducibility.

---

## 9. Step-by-Step Tool Guides

### 9.1 TopOpt (Python) --- Learning SIMP Topology Optimization

#### Installation

```bash
# Requires Python 3.7+
pip install topopt
# Or install from source for latest version:
git clone https://github.com/zfergus/topopt.git
cd topopt
pip install -e .
```

**Dependencies:** NumPy, SciPy, cvxopt (for MMA solver), matplotlib (for visualization).

**Alpha software warning:** The TopOpt library on PyPI is version 0.0.1-alpha.1, meaning it is pre-release software. The API may change, and the code examples below may require adjustment for future versions. The import paths and class names have been verified against the current alpha release but should be re-checked if installation fails.

**More mature alternative for learning:** The [DTU TopOpt Python codes](https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python) by Niels Aage and Villads Egede Johansen are widely cited, stable, self-contained ~200-line scripts that implement the same SIMP method. They have been used in engineering education for years and do not have external API dependencies. If the TopOpt pip package causes installation issues, the DTU codes are the recommended fallback.

#### Basic Usage: MBB Beam (Learning Example)

```python
import numpy as np
from topopt.boundary_conditions import MBBBeamBoundaryConditions
from topopt.problems import ComplianceProblem
from topopt.solvers import TopOptSolver
from topopt.filters import DensityBasedFilter
from topopt.guis import GUI

# Define problem
nelx, nely = 180, 60          # mesh resolution
volfrac = 0.4                  # volume fraction target
rmin = 2.4                     # filter radius
penal = 3.0                    # SIMP penalization

bc = MBBBeamBoundaryConditions(nelx, nely)
problem = ComplianceProblem(bc, volfrac, penal, rmin)
gui = GUI(problem, "Topology Optimization")
topopt_filter = DensityBasedFilter(nelx, nely, rmin)
solver = TopOptSolver(problem, volfrac, topopt_filter, gui)
x = solver.optimize(gui)
# Result: optimized density distribution in x
```

#### Adapting for a Fan Blade (2D Cross-Section)

To optimize a fan blade cross-section, you would:
1. Define a rectangular design domain representing the blade's width x thickness.
2. Apply distributed pressure loads on the top surface (from CFD results).
3. Fix the bottom-center nodes (handle attachment).
4. Set volume fraction to 0.35-0.40.
5. Run optimization and visualize the internal rib structure.

### 9.2 FreeCAD + BESO + CalculiX --- Production Topology Optimization

#### Installation

1. **Install FreeCAD** (version 0.21+): Download from https://www.freecadweb.org/downloads.php
2. **CalculiX** comes bundled with FreeCAD on most platforms.
3. **Install BESO plugin:**
   ```bash
   # Clone the BESO repository
   git clone https://github.com/calculix/beso.git
   # Copy to FreeCAD modules directory:
   # macOS: ~/Library/Application Support/FreeCAD/Mod/
   # Linux: ~/.local/share/FreeCAD/Mod/
   # Windows: %APPDATA%/FreeCAD/Mod/
   cp -r beso ~/Library/Application\ Support/FreeCAD/Mod/
   ```
4. Restart FreeCAD. BESO should appear in the workbench selector.

#### Workflow

1. **Create or import geometry** in Part workbench.
2. **Switch to FEM workbench.** Create a new analysis.
3. **Add material:** Select PETG with FDM-printed properties (see note below):
   - **FDM-printed PETG (100% infill, 0.2mm layer):** E_XY ~ 1300 MPa, E_Z ~ 1000 MPa, nu = 0.38, density = 1270 kg/m^3
   - **WARNING:** Datasheet values for PETG list E = 2100 MPa, but this is the bulk/injection-molded value. FDM-printed PETG has 30-50% lower stiffness due to inter-layer voids and bonding imperfections. Experimental studies (PMC 7600181, ScienceDirect 2025) consistently report FDM PETG Young's modulus in the range of 1100-1500 MPa in the XY (strong) direction and 800-1200 MPa in the Z (weak) direction. Using the bulk value overestimates stiffness by 40-90%, leading to underestimated deflections and potentially missed resonance concerns (natural frequency scales as sqrt(E), so a 50% E overestimate gives ~22% frequency overestimate).
   - **For reduced infill:** At 30% gyroid infill (recommended in Section 6.3), the effective modulus is substantially lower still, roughly scaling with the effective density fraction. For 30% infill, expect E_eff ~ 400-500 MPa.
   - **Best practice:** Print 3-5 tensile test specimens (ASTM D638 Type V, small size) with your specific printer, filament brand, and print settings. Test them to measure the actual E and yield strength. This one-time effort (2-3 hours of printing + testing) eliminates the largest source of FEA error.
4. **Add constraints:**
   - Fixed constraint on handle attachment faces.
   - Pressure load on blade faces (from CFD, or estimated uniform load of ~5-20 Pa).
5. **Mesh the part:** Use Gmsh mesher, element size 0.5-1.0 mm for fan blade.
6. **Run FEA** to verify the baseline stress and displacement.
7. **Switch to BESO toolbar:**
   - Define optimization domain (the fan blade body).
   - Set volume fraction target (0.35).
   - Set convergence criteria (0.1% change in compliance).
   - Run BESO optimization.
8. **View results** in FreeCAD or export VTK for ParaView.
9. **Post-process the density field** following Section 8.3.1.
10. **Export** smoothed, repaired geometry as STL.

### 9.3 SU2 --- Aerodynamic Shape Optimization

#### Installation

```bash
# Option 1: Pre-built binaries (easiest)
# Download from https://su2code.github.io/download.html
# Extract and add to PATH

# Option 2: Build from source (for adjoint support)
git clone https://github.com/su2code/SU2.git
cd SU2
# Configure with meson (requires Python 3, meson, ninja)
pip install meson ninja
./meson.py build -Denable-autodiff=true -Denable-pywrapper=true
./ninja -C build install

# Option 3: conda (simplest for Mac/Linux)
conda install -c conda-forge su2
```

#### Mesh Generation (Using Gmsh)

```bash
# Install Gmsh
pip install gmsh
# Or download from https://gmsh.info/

# Create a 2D mesh around the fan cross-section
# (Example Gmsh .geo file for a simple airfoil-like shape)
```

#### Mesh Generation Challenges for Thin Fan Blades

**This is a major practical hurdle that beginners should anticipate.** Generating a quality CFD mesh around a thin fan blade is significantly harder than meshing a typical airfoil or bluff body, for several reasons:

**Extreme aspect ratio:** A fan blade with 2.5 mm thickness and 250 mm chord has an aspect ratio of 100:1. This means:
- The boundary layer mesh near the blade surface needs cells on the order of 0.1-0.5 mm in height (first cell height), while the farfield is 10-20 chord lengths away (~2.5-5 m).
- The growth ratio between the smallest and largest cells spans ~4 orders of magnitude, requiring 30-50 layers of structured boundary layer cells.
- At Re ~ 30,000, the boundary layer thickness is approximately delta ~ 5c / sqrt(Re) ~ 7 mm, so 10-20 cells across this 7 mm layer are needed for adequate resolution.

**Trailing edge singularity:** The trailing edge of a thin blade (where thickness approaches zero) creates a near-singular point in the mesh. Gmsh handles this better than some alternatives, but you may need to:
- Specify a blunt trailing edge with finite thickness (0.5-1.0 mm) rather than a sharp edge.
- Use the `Field` feature in Gmsh to locally refine the mesh near the trailing edge.

**Downstream analysis plane:** If using Approach 2 (SURFACE_TOTAL_PRESSURE objective), the downstream monitoring plane must be defined as an internal boundary in the mesh. In Gmsh, create a surface at x = 1.5 * chord downstream and use `Physical Surface("downstream_plane")` to tag it.

**Practical tips for Gmsh:**
1. Use structured boundary layer meshing via `Field[1] = BoundaryLayer` with `Field[1].EdgesList` specifying the blade surface edges.
2. Set `Field[1].hwall_n = 0.0002` (first cell height ~0.2 mm) and `Field[1].ratio = 1.15` (15% growth rate).
3. Use 2D meshing first to validate the mesh quality before attempting 3D.
4. Check mesh quality with `Mesh.QualityType = 2` (minimum Jacobian) --- aim for all elements > 0.3.
5. Export with `Mesh.Format = 42` for SU2 native format.

**Time estimate:** Budget 1-2 weeks for mesh generation alone if you are new to Gmsh. The [Gmsh tutorial collection](https://gmsh.info/doc/texinfo/gmsh.html#Tutorial) and the [Lethe Gmsh introduction](https://chaos-polymtl.github.io/lethe/documentation/tools/gmsh/gmsh.html) are good starting resources.

Example Gmsh geometry file (`fan_blade.geo`):
```
// Fan blade cross-section
lc = 0.005;  // mesh size

// Define blade profile points (simplified)
Point(1) = {0, 0, 0, lc};
Point(2) = {0.25, 0.02, 0, lc};  // upper surface
Point(3) = {0.25, -0.01, 0, lc}; // lower surface
// ... more points defining the blade profile

// Create splines, surfaces, and farfield boundary
// ...

// Generate mesh
Mesh 2;
Save "fan_blade.su2";
```

#### SU2 Configuration: Steady-State Proxy (Beginner Approach)

**Important:** The configuration below uses a steady-state proxy objective. This is a simplification --- a real fan operates in unsteady oscillatory motion. See the "Steady-state proxy limitations" note below and the "Unsteady Approach" section for the proper formulation.

**Objective function strategy:** `DRAG` is the wrong objective for a fan (the fan's purpose is to push air, not minimize resistance). The correct objective is the net downstream momentum flux. For SU2 in external flow, we have three workable approaches, listed in order of increasing fidelity:

1. **Multi-objective with DRAG and LIFT (simplest, works out-of-the-box):** Maximize a weighted combination of LIFT (normal force on the fan face, which drives airflow) while penalizing DRAG. This uses SU2's standard external aerodynamics objectives and requires no custom code. The weighting must be tuned: a starting point is `OBJECTIVE_FUNCTION= LIFT` with a drag constraint.

2. **SURFACE_TOTAL_PRESSURE on a downstream analysis plane (recommended):** Place an internal monitoring plane (a mesh boundary) 1-2 chord lengths downstream of the fan blade. Use `OBJECTIVE_FUNCTION= SURFACE_TOTAL_PRESSURE` with `MARKER_ANALYZE= (downstream_plane)` to maximize the total pressure (a proxy for momentum flux) on that plane. This requires the downstream plane to be defined during mesh generation as an internal boundary. See the mesh generation section below for how to set this up in Gmsh.

3. **Custom objective via SU2 Python wrapper (most accurate):** Compute the net directed momentum flux integral directly from the SU2 solution. See the Python example below.

**Note on SURFACE_PRESSURE_DROP (why it was removed):** The previous revision used `SURFACE_PRESSURE_DROP` as the objective. This was incorrect for this problem. `SURFACE_PRESSURE_DROP` computes the pressure difference between two marker boundaries (inlet/outlet pairs) and is designed for internal flow applications (ducts, pipes, heat exchangers). For an external flow problem like a fan blade in open air, there is no natural inlet/outlet pair, and the required `MARKER_INLET`/`MARKER_OUTLET` specifications do not map naturally to the problem geometry.

```ini
% fan_optimization.cfg --- Approach 2: SURFACE_TOTAL_PRESSURE on downstream plane
%
% Physics (incompressible, appropriate for steady-state proxy)
SOLVER= INC_NAVIER_STOKES
KIND_TURB_MODEL= NONE
%
% Flow conditions (fan at peak velocity 2.64 m/s)
INC_VELOCITY_INIT= (2.64, 0.0, 0.0)
INC_DENSITY_INIT= 1.225
INC_TEMPERATURE_INIT= 300.0
VISCOSITY_MODEL= CONSTANT_VISCOSITY
MU_CONSTANT= 1.81e-5
%
% Mesh (must include downstream_plane as an internal marker)
MESH_FILENAME= fan_blade.su2
MESH_FORMAT= SU2
%
% Boundary conditions
MARKER_HEATFLUX= ( fan_blade, 0.0 )
MARKER_FAR= ( farfield )
MARKER_INTERNAL= ( downstream_plane )
%
% Objective: maximize total pressure at downstream analysis plane
OBJECTIVE_FUNCTION= SURFACE_TOTAL_PRESSURE
MARKER_ANALYZE= ( downstream_plane )
MARKER_MONITORING= ( fan_blade )
%
% Design variables (FFD)
DV_KIND= FFD_CONTROL_POINT
FFD_DEFINITION= (BOX, -0.01,-0.05,0.0, 0.26,-0.05,0.0, 0.26,0.05,0.0, -0.01,0.05,0.0, ...)
%
% Optimization
OPT_OBJECTIVE= SURFACE_TOTAL_PRESSURE * 1.0
OPT_CONSTRAINT= NONE
DEFINITION_DV= ...
```

**Custom objective via Python wrapper (Approach 3):**

```python
"""
Custom SU2 objective: net downstream momentum flux.
Requires SU2 built with -Denable-pywrapper=true.
"""
import pysu2
from mpi4py import MPI

comm = MPI.COMM_WORLD
driver = pysu2.CSinglezoneDriver("fan_optimization.cfg", 1, comm)

# Run flow solution
driver.StartSolver()

# Extract solution at downstream monitoring plane
# Get velocity and pressure at downstream plane nodes
marker_id = driver.GetMarkerIndex("downstream_plane")
n_vertices = driver.GetNumberMarkerNodes(marker_id)

momentum_flux = 0.0
rho = 1.225  # kg/m^3
for i_vertex in range(n_vertices):
    # Get vertex area and normal
    normal = driver.GetMarkerNormal(marker_id, i_vertex)
    # Get velocity components at this vertex
    coords = driver.GetMarkerCoordinates(marker_id, i_vertex)
    u = driver.GetMarkerVelocity(marker_id, i_vertex)
    # Momentum flux contribution: rho * u_x * (u dot n) * dA
    # where x is the fan-to-user direction
    u_dot_n = sum(u[j] * normal[j] for j in range(len(u)))
    momentum_flux += rho * u[0] * u_dot_n

print(f"Net downstream momentum flux: {momentum_flux:.4f} N")
# Use this value as the objective for external optimization
# (e.g., via scipy.optimize wrapping this script)
```

**Note:** The Python wrapper API varies between SU2 versions. The above is illustrative --- consult the [SU2 Python wrapper documentation](https://su2code.github.io/docs/Execution/) and the [FSI Python wrapper tutorials](https://su2code.github.io/tutorials/Static_FSI_Python/) for the exact API calls available in your version. The key point is that the Python wrapper provides access to all field variables at marker nodes, enabling arbitrary custom objectives.

**Steady-state proxy limitations:** This steady-state approach is a significant simplification. See Section 9.3.1 below for a quantitative discussion of when this proxy is and is not adequate.

#### SU2 Configuration: Unsteady Approach (Advanced)

For the proper unsteady formulation (optimizing over a full waving cycle), follow the [SU2 Unsteady Shape Optimization tutorial](https://su2code.github.io/tutorials/Unsteady_Shape_Opt_NACA0012/). Key additional configuration:

**Critical: Use the compressible solver for unsteady pitching motion.** The previous revision of this report used `SOLVER= INC_NAVIER_STOKES` (incompressible) for the unsteady configuration. This is problematic for two reasons:

1. **Known compatibility issues:** SU2 [GitHub Issue #193](https://github.com/su2code/SU2/issues/193) documented a bug where the incompressible solver produced nonsensical pitching frequency values (8.19e+48 rad/s) because the pitching frequency was not properly non-dimensionalized. While SU2 has evolved since that issue, the combination of incompressible solver + rigid body pitching + unsteady adjoint remains at the edge of what is well-tested and documented.

2. **Tutorial mismatch:** The SU2 [Unsteady Shape Optimization NACA0012 tutorial](https://su2code.github.io/tutorials/Unsteady_Shape_Opt_NACA0012/) --- the very tutorial this report references --- uses `SOLVER= EULER` (compressible), not `INC_NAVIER_STOKES`. Every pitching test case in the SU2 repository (`TestCases/unsteady/pitching_*`) uses the compressible formulation.

At the fan's operating speed (Mach ~ 0.008), the compressible solver requires low-Mach preconditioning for good convergence. SU2 provides this via the `LOW_MACH_PREC` option with Roe-Turkel preconditioning.

```ini
% Unsteady simulation --- COMPRESSIBLE solver with low-Mach preconditioning
% This matches the SU2 unsteady tutorials and is well-tested with pitching motion
SOLVER= NAVIER_STOKES
KIND_TURB_MODEL= NONE
%
% Flow conditions (compressible formulation)
MACH_NUMBER= 0.0077
AOA= 0.0
FREESTREAM_TEMPERATURE= 300.0
FREESTREAM_PRESSURE= 101325.0
REYNOLDS_NUMBER= 30000.0
REYNOLDS_LENGTH= 0.25
%
% Low-Mach preconditioning (essential at Mach ~ 0.008)
LOW_MACH_PREC= YES
MIN_ROE_TURKEL_PREC= 0.01
MAX_ROE_TURKEL_PREC= 0.2
%
% Time stepping
TIME_DOMAIN= YES
TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER
TIME_STEP= 0.001
MAX_TIME= 1.0      % One full waving cycle at 1 Hz
TIME_ITER= 1100
INNER_ITER= 100
%
% Unsteady adjoint
UNST_ADJOINT_ITER= 1000
ITER_AVERAGE_OBJ= 500
%
% Moving mesh for oscillating fan
GRID_MOVEMENT= RIGID_MOTION
MOTION_ORIGIN= 0.0 0.0 0.0
PITCHING_OMEGA= 0.0 0.0 6.2832    % 2*pi*f for 1 Hz waving
PITCHING_AMPL= 0.0 0.0 40.0       % 40 degree amplitude
%
% Numerical methods
NUM_METHOD_GRAD= GREEN_GAUSS
CONV_NUM_METHOD_FLOW= ROE
MUSCL_FLOW= YES
```

**Note on incompressible vs. compressible:** The incompressible solver (`INC_NAVIER_STOKES`) is still appropriate for the steady-state proxy configuration above, where no pitching motion is involved. For the unsteady pitching configuration, use the compressible solver with `LOW_MACH_PREC= YES` as shown above. At Mach ~ 0.008, the preconditioning is essential --- without it, the large acoustic-to-convective wave speed ratio causes extremely slow convergence or outright divergence.

The unsteady adjoint computes sensitivities of the cycle-averaged objective to design parameters, accounting for the full oscillatory motion. This is the correct formulation but requires 10-50x more compute time than the steady-state proxy.

#### 9.3.1 Steady-State Proxy: Limitations and Validation Strategy

**Quantifying the approximation:** The steady-state proxy assumes that the aerodynamic performance at peak waving velocity is representative of the cycle-averaged performance. For the fan's operating parameters:

- Waving frequency f = 2 Hz, chord c = 0.25 m, peak tip velocity V = 2.64 m/s
- Reduced frequency: k = pi * f * c / V = pi * 2 * 0.25 / 2.64 ~ 0.60

A reduced frequency of k ~ 0.6 places this problem firmly in the unsteady regime (k > 0.05 is considered unsteady in classical aerodynamics). At this reduced frequency, several physical effects are absent from the steady-state simulation:

- **Added mass:** The acceleration of the surrounding air contributes to instantaneous forces but does not exist in steady-state.
- **Wake interaction:** Vortices shed during the previous half-stroke interact with the current stroke. This is entirely absent in steady-state.
- **Dynamic stall effects:** The effective angle of attack changes rapidly during the waving cycle, potentially producing dynamic stall-like behavior.

**Does the proxy preserve shape rankings?** The critical question for optimization is not whether the steady-state values are accurate in absolute terms, but whether they preserve the *relative ranking* of designs. Published literature on oscillating flat plates at comparable reduced frequencies (see Ref. 24, Keulegan and Carpenter) shows that added mass and wake effects scale similarly across different plate geometries, suggesting that steady-state shape rankings may approximately correlate with unsteady rankings --- but this has not been rigorously demonstrated for the specific geometry variations in fan ASO.

**Validation protocol (mandatory):** To mitigate the risk of the steady-state proxy leading to a suboptimal design:

1. After completing steady-state ASO, select 3 designs: the baseline, the best candidate, and one intermediate candidate.
2. Run full unsteady simulations (2-3 waving cycles each) on all 3 designs using the compressible solver configuration above.
3. Verify that the ranking (best > intermediate > baseline) is preserved in the unsteady results.
4. If the ranking changes, the steady-state proxy is inadequate for this design space, and the full unsteady optimization must be used.

**When to skip the proxy entirely:** If the fan has features that strongly interact with unsteady effects (e.g., deep camber, slotted geometries, or very flexible blades), the steady-state proxy is likely unreliable. In these cases, begin directly with the unsteady formulation, accepting the higher computational cost.

#### Running Shape Optimization

```bash
# Step 1: Run baseline flow solution
SU2_CFD fan_optimization.cfg

# Step 2: Run adjoint solution
SU2_CFD_AD fan_adjoint.cfg

# Step 3: Project sensitivities onto design variables
SU2_DOT fan_optimization.cfg

# Step 4: Run optimization loop
# Using SU2's built-in shape_optimization.py script:
shape_optimization.py -f fan_optimization.cfg -n 8
```

The optimization loop will:
1. Evaluate the flow at the current design.
2. Compute adjoint-based sensitivities (gradient of objective w.r.t. each FFD control point).
3. Update design variables using a gradient-based optimizer (L-BFGS-B or SLSQP).
4. Deform the mesh according to new FFD control points.
5. Repeat until convergence.

### 9.4 BoTorch --- Bayesian Optimization with GP Surrogates

#### Installation

```bash
pip install botorch gpytorch torch
# For visualization:
pip install matplotlib plotly
# For Latin Hypercube Sampling:
pip install scipy
```

#### Complete Workflow

See Section 5.2.3 for the complete BO loop code, including:
- CFD wrapper function
- LHS initial sampling
- Multi-fidelity GP training
- Bayesian optimization with Expected Improvement
- Surrogate validation
- Best design extraction

**Key parameters to tune:**
- `num_restarts`: Number of random restarts for acquisition optimization (10-20)
- `raw_samples`: Number of raw samples for initial acquisition candidates (256-1024)
- Initial sample count: 8-10x the number of design variables (80-100 for 10 variables)
- Total budget: 100-200 evaluations for 10-15 effective design variables
- **Surrogate validation:** Check NRMSE periodically; if > 15%, add more training samples before continuing

### 9.5 ML-Accelerated Topology Optimization

#### Recommended Path: DL4TO Library

```bash
# Install DL4TO (the only library-based ML-for-TO option)
pip install git+https://github.com/dl4to/dl4to
pip install torch pyvista==0.38.1 numpy scipy meshio
```

**Maturity note:** DL4TO is an early-stage library (124 GitHub stars, Apache 2.0 license, University of Bremen). It works on structured voxel grids, not unstructured meshes. Use it for coarse-resolution ML exploration, then refine with full SIMP in FreeCAD/BESO.

See [DL4TO documentation](https://dl4to.github.io/dl4to/) for tutorials and API reference.

#### Advanced Path: Custom PyTorch Implementation

```bash
pip install torch numpy scipy
# For FEA integration:
pip install pyvista meshio
```

**Maturity note:** This path requires custom implementation from research papers. There is no turnkey library. Budget 4-8 weeks. See Section 5.3.2 for architecture details and Section 5.3.3 for the U-Net topology predictor.

#### Complete Workflow

See Section 5.3 for complete code and details on:
- DL4TO library-based approach (Section 5.3.1) --- recommended
- 3D CNN compliance predictor architecture and training (Section 5.3.2) --- advanced
- U-Net topology predictor for design space exploration (Section 5.3.3) --- advanced
- Realistic compute budgets for training data (Section 5.3.4)
- Integration with SIMP iteration loop
- Training data generation protocol

---

## 10. Validation Approaches

### 10.1 Computational Validation

#### FEA Stress Verification

After topology optimization, run a final FEA on the optimized geometry (not the pseudo-density field, but the actual solid model after interpretation/smoothing --- see Section 8.3.1):

1. Import the optimized STL/STEP into FreeCAD FEM or SimScale.
2. Apply the same loads used during TO (aerodynamic pressure + inertial).
3. **Use orthotropic material properties** to account for FDM anisotropy (Section 2.1.5).
4. Apply safety factor loading (1.5x the design loads).
5. Verify: `max(sigma_vm) < sigma_yield / 1.5` (using direction-appropriate yield strength)
6. Verify: `max(displacement) < 5mm` at blade tip (acceptable deflection).

#### Modal Analysis (Resonance Verification)

Compute natural frequencies to ensure the optimized fan blade does not resonate during waving:

1. In CalculiX, set up a `*FREQUENCY` analysis step.
2. Compute the first 5-10 modes.
3. Verify: first bending mode frequency > 5x the waving frequency (e.g., f_1 > 10 Hz for 2 Hz waving).
4. If any mode is close to a waving harmonic, modify the TO constraints (increase minimum member size or volume fraction) and re-run.

#### CFD Verification of Optimized Shape

After aerodynamic shape optimization, verify the final design with a refined CFD simulation:

1. Re-mesh the final geometry with a finer mesh (2x the optimization mesh density).
2. Run steady-state CFD at multiple velocities (1, 2, 3 m/s).
3. Run an unsteady simulation of 2-3 waving cycles.
4. Compare the fan performance metric (Section 2.2.2) against the surrogate model predictions.
5. Acceptable discrepancy: < 5% between surrogate prediction and fine-mesh CFD.

#### ML Surrogate Validation

After Bayesian optimization, verify that the GP surrogate was reliable:

1. Compute final NRMSE on all training data (should be < 10%).
2. Run 5 additional CFD simulations at random points in the design space. Compare GP predictions to actual CFD results.
3. Check that GP uncertainty estimates were well-calibrated: approximately 95% of actual values should fall within the GP's 95% confidence interval.
4. If calibration is poor, the BO results may be suboptimal. Consider re-running with additional training data.

#### Mesh Independence Study

For both FEA and CFD, verify results are mesh-independent:

1. Run at 3 mesh levels: coarse (1x), medium (2x), fine (4x element count).
2. If the quantity of interest changes by < 2% between medium and fine, the medium mesh is sufficient.
3. This is critical --- an optimization result on an under-resolved mesh is meaningless.

### 10.2 Physical Validation

#### Anemometer Testing Protocol

**Equipment:** A handheld anemometer (available for $15-30, e.g., BTMETER BT-100 or similar).

**Test setup:**
1. Mount fan on a pendulum or motorized arm for repeatable waving motion.
   - If manual: practice a consistent waving rhythm using a metronome app.
2. Place anemometer at a fixed distance: 300mm from the fan face center.
3. Record peak and average air velocity over 10 waving cycles.
4. Repeat 5 times and compute mean and standard deviation.

**Metrics to compare:**
- **Average airflow velocity** at 300mm distance (m/s)
- **Peak airflow velocity** (m/s)
- **Subjective comfort rating** (1-5 scale, blind test preferred)

**Expected improvement target:** 15-30% increase in average airflow velocity from the optimized fan vs. the baseline flat fan, at the same waving effort (frequency and amplitude). Note: this estimate is based on published TO results for thin plate structures (15-30% stiffness-to-weight improvement) and aerodynamic shaping literature (5-15% flow improvement for low-Re plates). The combined benefit may be less than the sum of parts due to coupling effects.

#### Smoke/Incense Visualization

For qualitative flow visualization:
1. Light an incense stick and position it 100-200mm from the fan face.
2. Wave the fan and observe smoke deflection patterns.
3. Record with a video camera for comparison between baseline and optimized designs.
4. Look for: directed airflow (vs. turbulent dispersion), vortex patterns, dead zones.

#### Structural Testing

1. **Static deflection test:** Clamp the handle, hang a known weight (50-200g) from the blade tip. Measure deflection. Compare to FEA prediction (using orthotropic properties).
2. **Fatigue test:** Wave the fan continuously for 30 minutes (~3600 cycles at 2 Hz). Inspect for cracks, delamination, or permanent deformation. Pay particular attention to layer boundaries.
3. **Impact test:** Drop the fan from 1m onto a hard surface. Check for fracture (relevant for PLA vs. PETG comparison).

---

## 11. References and Sources

### Topology Optimization

1. [TopOpt Python Library Documentation](https://topopt.readthedocs.io/en/documentation/TopOpt.html) --- Background on SIMP method and Python implementation.
2. [TopOpt on PyPI](https://pypi.org/project/topopt/) --- Installation and usage.
3. [DTU TopOpt: Topology Optimization Codes in Python](https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python) --- Educational codes from the Technical University of Denmark.
4. [FEniCS SIMP Tutorial](https://comet-fenics.readthedocs.io/en/latest/demo/topology_optimization/simp_topology_optimization.html) --- SIMP implementation using FEniCS.
5. [BESO for CalculiX (GitHub)](https://github.com/calculix/beso) --- Bi-directional Evolutionary Structural Optimization with CalculiX.
6. [FEMbyGEN: Generative Design for FreeCAD (GitHub)](https://github.com/Serince/FEMbyGEN) --- Topology optimization module for FreeCAD.
7. [OpenLSTO (GitHub)](https://github.com/M2DOLab/OpenLSTO) --- Open-source level set topology optimization.
8. [Open-Source Codes of Topology Optimization: A Summary for Beginners](https://www.sciencedirect.com/org/science/article/pii/S1526149223003569) --- Comprehensive survey paper.
9. [Altair Topology Optimization](https://altair.com/topology-optimization) --- Commercial software reference.
10. [Fusion 360 Topology Optimization Blog](https://www.autodesk.com/products/fusion-360/blog/topology-optimization-and-autodesk-fusion/) --- Autodesk's generative design approach.
11. [Topology Optimization for AM with Overhang Constraints (arXiv)](https://arxiv.org/abs/2204.07333) --- Additive manufacturing constraints in TO.

### Aerodynamic Shape Optimization

12. [SU2: Multiphysics Simulation and Design Software](https://su2code.github.io/) --- Official SU2 website.
13. [SU2 Tutorial Collection](https://su2code.github.io/tutorials/home/) --- Step-by-step tutorials.
14. [SU2 Quick Start Guide](https://su2code.github.io/docs/Quick-Start/) --- Installation and first simulation.
15. [SU2 Installation Guide](https://su2code.github.io/docs_v7/Installation/) --- Build instructions.
16. [SU2 Unsteady Shape Optimization Tutorial](https://su2code.github.io/tutorials/Unsteady_Shape_Opt_NACA0012/) --- Unsteady adjoint-based shape optimization.
17. [OpenFOAM](https://www.openfoam.com/) --- Open-source CFD toolbox.
18. [OpenFOAM adjointOptimisationFoam Manual (PDF)](https://www.openfoam.com/documentation/files/adjointOptimisationFoamManual.pdf) --- Adjoint-based shape optimization in OpenFOAM.
19. [DAFoam: Discrete Adjoint with OpenFOAM](https://dafoam.github.io/) --- High-fidelity MDO platform.
20. [DAFoam GitHub](https://github.com/mdolab/dafoam) --- Source code and documentation.
21. [SimScale: CFD Simulation Software](https://www.simscale.com/product/cfd/) --- Cloud-based CFD platform.
22. [AirShaper](https://airshaper.com/) --- Simplified aerodynamic analysis.

### Aerodynamic Physics

23. [Drag Coefficient - Wikipedia](https://en.wikipedia.org/wiki/Drag_coefficient) --- Reference for flat plate C_D values.
24. [Drag on Oscillating Flat Plates at Low Reynolds Numbers (Cambridge Core)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/drag-on-oscillating-flat-plates-in-liquids-at-low-reynolds-numbers/4A6222AC968750F25984BD2538E5DCDA) --- Directly relevant experimental data.
25. [NASA: Shape Effects on Drag](https://www.grc.nasa.gov/www/k-12/VirtualAero/BottleRocket/airplane/shaped.html) --- Educational resource on drag coefficients.

### ML Surrogate Modeling

26. [BoTorch: Bayesian Optimization in PyTorch](https://botorch.org/docs/overview/) --- Official documentation.
27. [GPyTorch Models in BoTorch](https://botorch.org/docs/models/) --- GP model reference.
28. [BoTorch Multi-Objective Bayesian Optimization Tutorial](https://botorch.org/docs/tutorials/multi_objective_bo/) --- qEHVI and qNEHVI examples.
29. [Multi-Fidelity Deep Neural Network Surrogate Model for ASO (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0045782520306708) --- MFDNN for aerodynamic optimization.
30. [Bayesian Optimization of Lightweight Neural Network for Aerodynamic Prediction (arXiv)](https://arxiv.org/html/2503.19479v1) --- Recent paper on BO for aero.
31. [Review of Deep Learning Surrogates for Aerodynamic Shape Optimization (AIP)](https://pubs.aip.org/aip/pof/article/37/4/041304/3345111/Review-of-deep-learning-based-aerodynamic-shape) --- Comprehensive 2025 review.
32. [Enhanced Data Efficiency with DNNs and GPs for Aero Design (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1270963821000341) --- Combining neural networks with Gaussian processes.
33. [Bayesian Optimization Introduction (Blog)](https://www.miguelgondu.com/blogposts/2023-07-31/intro-to-bo/) --- Beginner-friendly BO tutorial.
34. [A Survey on High-Dimensional Gaussian Process Modeling (HAL/INRIA)](https://inria.hal.science/hal-03419959/document) --- GP scalability and dimensionality.
35. [When Does Vanilla GPR Fail? (Blog)](https://www.miguelgondu.com/blogposts/2024-03-16/when-does-vanilla-gpr-fail/) --- GP limitations in high dimensions.

### ML-Accelerated Topology Optimization

72. [Topology optimization via machine learning and deep learning: a review (Oxford Academic, 2023)](https://academic.oup.com/jcde/article/10/4/1736/7223974) --- Comprehensive review of ML+TO methods. Confirms these methods remain in the research domain without production-ready tools.
73. [Deep learning accelerated efficient framework for topology optimization (ScienceDirect, 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0952197624007176) --- DNN-accelerated TO framework.
74. [Self-directed online machine learning for topology optimization (Nature Communications, 2022)](https://www.nature.com/articles/s41467-021-27713-7) --- Online learning during TO iterations.
75. [Accelerating gradient-based TO with dual-model ANN (Springer, 2021)](https://link.springer.com/article/10.1007/s00158-020-02770-6) --- Neural network for both forward and sensitivity predictions in SIMP.
76. [Compliance Prediction via Moment Invariants and GRNN (MDPI, 2023)](https://www.mdpi.com/1099-4300/25/10/1396) --- Generalized regression neural network for compliance prediction with R^2 > 0.97.
80. [DL4TO: A Deep Learning Library for Sample-Efficient Topology Optimization (Springer, 2023)](https://link.springer.com/chapter/10.1007/978-3-031-38271-0_54) --- Conference paper describing the DL4TO library.
81. [DL4TO GitHub Repository](https://github.com/dl4to/dl4to) --- pip-installable PyTorch library for ML-accelerated 3D TO on structured grids. 124 stars, Apache 2.0.
82. [DL4TO Documentation](https://dl4to.github.io/dl4to/) --- API reference and tutorials.
83. [PyTopo3D: A Python Framework for 3D SIMP-based Topology Optimization (arXiv, 2025)](https://arxiv.org/abs/2504.05604) --- Benchmark data for 3D SIMP scaling.

### 3D Printing Materials

36. [PETG vs PLA vs ABS: Strength Comparison (Ultimaker)](https://ultimaker.com/learn/petg-vs-pla-vs-abs-3d-printing-strength-comparison/) --- Material property comparison.
37. [PETG vs PLA: Strength and Printing Comparison (Xometry)](https://www.xometry.com/resources/3d-printing/petg-vs-pla-3d-printing/) --- Detailed mechanical properties.
38. [3D Printer Filament Comparison (MatterHackers)](https://www.matterhackers.com/3d-printer-filament-compare) --- Comprehensive filament database.
39. [Bambu Carbon Fiber Filaments (Blog)](https://blog.bambulab.com/bambu-carbon-fiber-filaments/) --- CF filament properties.
40. [3D Printing with PLA Carbon Fiber Filament (Nobufil)](https://www.nobufil.com/en/post/3d-printing-with-pla-cf-filament) --- PLA-CF material guide.

### 3D Printed Fan Designs

41. [3D Printed Hand Fan (GrabCAD)](https://grabcad.com/library/3d-printed-hand-fan) --- Example CAD model.
42. [Print-in-Place Hand Fan (Printables)](https://www.printables.com/model/132278-print-in-place-hand-fan) --- Practical 3D printed fan design.
43. [FreeCAD FEM Workbench Tutorial (DigiKey)](https://www.digikey.com/en/maker/tutorials/2025/intro-to-freecad-part-10-finite-element-method-fem-workbench-tutorial) --- FreeCAD FEM beginner guide.

### FDM Material Anisotropy and Fatigue

44. [Material Anisotropy in Additively Manufactured Polymers (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8512748/) --- Comprehensive review of AM polymer anisotropy.
45. [Topology Optimization for 3D Printing-Driven Anisotropic Components (ScienceDirect, 2025)](https://www.sciencedirect.com/science/article/abs/pii/S014102962500046X) --- Anisotropy-aware TO.
46. [Topology Optimization for Multipatch FDM 3D Printing (MDPI)](https://www.mdpi.com/2076-3417/10/3/943) --- FDM-specific TO.
47. [Dynamic Topology Optimization with Material Anisotropy for FDM (ScienceDirect, 2024)](https://www.sciencedirect.com/science/article/pii/S0168874X24001756) --- SOMP method for FDM.
48. [Tensile and Fatigue Analysis of 3D-Printed PETG (ResearchGate)](https://www.researchgate.net/publication/332001021_Tensile_and_Fatigue_Analysis_of_3D-Printed_Polyethylene_Terephthalate_Glycol) --- PETG fatigue data.
49. [Effect of 3D Printing Parameters on Fatigue Properties (MDPI, 2023)](https://www.mdpi.com/2076-3417/13/2/904) --- Print parameter effects on fatigue.

### TO Post-Processing

50. [Surface Smoothing for Topological Optimized 3D Models (Springer, 2021)](https://link.springer.com/article/10.1007/s00158-021-03027-6) --- Smoothing methods.
51. [Systematical Redesign Method for TO Results Using 3D-Printing (Springer, 2023)](https://link.springer.com/article/10.1007/s44245-023-00019-2) --- Practical post-processing.
52. [Smooth Geometry Extraction from SIMP via SDF (arXiv, 2025)](https://arxiv.org/html/2512.06976v1) --- Modern SDF-based approach.

### Surface Roughness and Aerodynamics

53. [Effects of Surface Roughness on Aerodynamic Performance at Low Re (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S1000936120303794) --- Roughness at low Re.
54. [Boundary Layer Transition Due to Distributed Roughness (Cambridge Core)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/boundary-layer-transition-due-to-distributed-roughness-effect-of-roughness-spacing/949313A4248D97BCD4EC0400A123F266) --- Transition mechanisms.

### FDM Material Properties (Experimental)

61. [Experimental and Numerical Analysis for FDM PETG (PMC 7600181)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7600181/) --- Compression modulus values of 1117-1330 MPa for FDM PETG.
62. [In-depth Study of PETG FDM Process Parameters (ScienceDirect, 2025)](https://www.sciencedirect.com/science/article/pii/S223878542501436X) --- Tensile and compressive strength optimization for FDM PETG.
63. [Mechanical Properties of AM Polymers PLA and PETG (PMC, 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11243948/) --- FDM-printed Young's modulus ~1200 MPa average.
64. [Printability and Tensile Performance of 3D Printed PETG (MDPI)](https://www.mdpi.com/2073-4360/11/7/1220) --- FDM PETG tensile characterization.

### SU2 Solver Compatibility

65. [SU2 GitHub Issue #193: Wrong pitching_omega for incompressible flow](https://github.com/su2code/SU2/issues/193) --- Documented incompressible solver + pitching motion bug.
66. [SU2 Config Template](https://github.com/su2code/SU2/blob/master/config_template.cfg) --- LOW_MACH_PREC and objective function documentation.
67. [SU2 Python Wrapper Build](https://su2code.github.io/docs/Python-Wrapper-Build/) --- Building SU2 with Python wrapper support.
68. [SU2 Static FSI Python Tutorial](https://su2code.github.io/tutorials/Static_FSI_Python/) --- Python wrapper API examples.

### Mesh Generation

69. [Gmsh Tutorial Collection](https://gmsh.info/doc/texinfo/gmsh.html#Tutorial) --- Official Gmsh tutorials.
70. [Lethe Gmsh Introduction](https://chaos-polymtl.github.io/lethe/documentation/tools/gmsh/gmsh.html) --- Practical Gmsh meshing guide.
71. [Best Meshing Practices for CFD (cfmesh)](https://cfmesh.com/best-meshing-practices-for-high-quality-cfd-simulations/) --- Boundary layer meshing best practices.

### General References

55. [Topology Optimization - Wikipedia](https://en.wikipedia.org/wiki/Topology_optimization) --- General reference.
56. [Von Mises Yield Criterion - Wikipedia](https://en.wikipedia.org/wiki/Von_Mises_yield_criterion) --- 3D von Mises equation reference.
57. [Optimization of Design Parameters and 3D-Printing Orientation (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2590048X25000470) --- Print orientation for TO parts.

### Fusion 360 Licensing

58. [Autodesk Fusion for Education](https://www.autodesk.com/education/edu-software/fusion) --- Educational license registration.
59. [How to Get Autodesk Fusion for Free (Product Design Online, 2024)](https://productdesignonline.com/tips-and-tricks/how-to-get-fusion-360-for-free/) --- License comparison.
60. [How to get access to Generative Design in Fusion (Autodesk Support)](https://www.autodesk.com/support/technical/article/caas/sfdcarticles/sfdcarticles/How-to-get-access-to-Generative-Design.html) --- Generative Design access requirements.
77. [Unlimited Cloud Credits are not available for Education license (Autodesk Support)](https://www.autodesk.com/support/technical/article/caas/sfdcarticles/sfdcarticles/Education-Plan-Unlimited-Cloud-Credits-are-not-available-to-start-a-simulation-in-Fusion-360.html) --- Configuration fix for educational cloud credits ("no credits" error is resolved by switching to educational account pool).
84. [Educational License shows no credits - Fusion 360 (Medium)](https://medium.com/@tirthengineer/educational-license-shows-no-credits-fusion-360-73b56639e737) --- Step-by-step fix for switching to educational credit pool.
78. [Fusion 360 Generative Design pricing (Develop3D)](https://develop3d.com/cad/fusion-350-generative-design-price/) --- Cloud credits pricing model.
79. [Topology Optimization is not Generative Design (Autodesk Blog)](https://www.autodesk.com/products/fusion-360/blog/topology-optimization-is-not-generative-design/) --- Distinction between TO and Generative Design.

---

## Appendix A: Quick-Start Decision Flowchart

```
START: Are you a student with a .edu email?
|
|-- YES:
|   |
|   |-- Register for Fusion 360 Educational License
|   |   Verify account is set to educational credit pool (Section 3.2)
|   |   Run a test Generative Design study
|   |   |
|   |   |-- STUDY RUNS SUCCESSFULLY (expected outcome):
|   |   |   |
|   |   |   |-- Fusion 360 (TO) + SU2 (ASO) + BoTorch (ML-ASO)
|   |   |   |   + DL4TO (ML-TO, recommended extension)
|   |   |   |   Timeline: 3-5 months
|   |   |   |   Cost: $0
|   |   |   |   Effort: Medium (SU2 is the hard part)
|   |   |   |
|   |   |-- "NO CREDITS" ERROR (configuration issue):
|   |       |
|   |       |-- Try fix: switch account to educational license (Section 3.2)
|   |       |-- If fix fails: fall through to NO path below
|   |
|-- NO:
    |
    |-- Do you have Python experience?
    |   |
    |   |-- YES --> FreeCAD/BESO (TO) + SU2 (ASO) + BoTorch (ML-ASO)
    |   |           + DL4TO (ML-TO, recommended extension)
    |   |           + TopOpt (2D learning)
    |   |           Timeline: 4-6 months
    |   |           Cost: $0
    |   |           Effort: High (BESO post-processing + SU2 learning curve)
    |   |
    |   |-- NO  --> FreeCAD/BESO (TO, GUI-based)
    |               + SimScale (CFD analysis, browser-based)
    |               + scikit-learn (simple GP, beginner-friendly)
    |               Timeline: 4-6 months (learning Python adds 2-4 weeks)
    |               Cost: $0
    |               Effort: High (need to learn Python for ML pipeline)
```

**Note:** All paths include ML surrogate modeling. The ML-for-ASO pipeline (BoTorch or scikit-learn) requires Python and is mandatory. The ML-for-TO pipeline (DL4TO or custom PyTorch) is a recommended extension that adds value but is not required for a first project --- standard SIMP produces correct results without ML acceleration. There is no viable path that avoids Python entirely while including ML.

## Appendix B: Estimated Computation Times

| Task | Hardware | Estimated Time | Notes |
|------|----------|---------------|-------|
| 2D TO (TopOpt Python, 200x60 mesh) | Any laptop | 5-30 seconds | |
| 3D TO single run (BESO/CalculiX, 500K elements) | Desktop, 8 cores | **2-4 hours** | Per full SIMP run to convergence (100-200 iterations). Previous estimate of 20-60 min was for coarser meshes or fewer iterations. |
| 3D TO (Fusion 360 cloud) | Cloud (Autodesk) | 30-120 minutes | Cloud-accelerated, faster than desktop |
| ML-accelerated 3D TO (after training) | Desktop, 8 cores | 5-15 minutes | Runtime only; excludes training data generation |
| **ML training data: TO (200-500 SIMP runs)** | **Desktop, 8 cores** | **4-25 days** | **400-2000 core-hours. Parallelizable. Only needed for advanced ML-for-TO extensions (Sections 5.3.2-5.3.3).** |
| **ML training data: DL4TO (50-200 runs)** | **Desktop, 8 cores** | **1-5 days** | **50-200 core-hours. Uses DL4TO's own solver on structured grids (faster than CalculiX on unstructured meshes).** |
| TO post-processing (smoothing, repair) | Any laptop | 1-4 hours (manual) | |
| 2D CFD steady (SU2, 50K cells) | Any laptop | 2-10 minutes | |
| 3D CFD steady (SU2, 500K cells) | Desktop, 8 cores | 30-120 minutes | |
| 3D CFD unsteady (SU2, 5 cycles) | Desktop, 8 cores | 2-8 hours | |
| Modal analysis (CalculiX, first 10 modes) | Desktop, 8 cores | 5-20 minutes | |
| GP surrogate training (100 samples, 10 dims) | Any laptop | 1-10 seconds | |
| GP surrogate training (200 samples, 12 dims) | Desktop | 10-60 seconds | |
| 3D CNN training (compliance predictor) | Laptop GPU | 15-30 minutes | |
| U-Net training (topology predictor) | Laptop GPU | 30-60 minutes | |
| BO iteration (acquisition opt) | Any laptop | 0.5-5 seconds | |
| Full BO loop (80 iterations with CFD) | Desktop, 8 cores | 3-7 days | |
| Full BO loop (80 iter with surrogate only) | Any laptop | 1-5 minutes | |
| Initial LHS CFD data generation (200 LF + 40 HF) | Desktop, 8 cores | 2-5 days | |

## Appendix C: Recommended Minimum Hardware

- **CPU:** 4+ cores (8+ recommended for CFD)
- **RAM:** 16 GB (32 GB for large 3D CFD)
- **Storage:** 50 GB free (CFD output files can be large)
- **GPU:** Not required for TO/CFD, but helpful for PyTorch surrogate training (any NVIDIA GPU with CUDA support)
- **OS:** Linux recommended for SU2/OpenFOAM (macOS works, Windows via WSL2)
- **3D Printer:** Any FDM printer with 220x220mm+ bed, heated bed, direct drive extruder preferred for PETG/PA

---

## Revision Log

### R4 (2026-03-22): Addressing Followup 1 Round 1 Review

Changes made in response to the challenger's Round 1 review of the Followup 1 revision:

1. **ML-for-TO reclassified with tiered maturity levels (Section 5.3):** The challenger correctly identified that ML-for-TO approaches (MLP compliance predictor, U-Net topology predictor) are research-stage methods requiring custom implementation from papers, conflicting with the "minimize effort" priority. ML-for-TO is now structured into three tiers:
   - **Core (mandatory):** ML-for-ASO via BoTorch (production-quality, unchanged)
   - **Recommended extension:** DL4TO library for ML-accelerated TO (early-stage but pip-installable, newly added)
   - **Advanced extension:** Custom 3D CNN/U-Net implementations (research-stage, retained but clearly labeled)
   Added ML Maturity Assessment table (Section 5.3.0) per reviewer request.

2. **MLP architecture replaced with 3D CNN (Section 5.3.2):** The reviewer correctly identified that the naive MLP with 500K-dimensional input would have ~256M parameters and massively overfit on 200-500 samples. Replaced with a 3D CNN (~2M parameters) that exploits spatial structure of the density field. Also documented dimensionality reduction and coarse-grid proxy alternatives.

3. **Fusion 360 cloud credits reframed as configuration issue (Section 3.2):** The reviewer was right that educational licenses include unlimited cloud credits when properly configured. Replaced the "uncertain availability" framing with specific troubleshooting steps (switch to educational account pool). Updated head-to-head table, tool details, and Appendix A flowchart accordingly.

4. **TO training data compute budget corrected (Sections 5.3.3, 5.3.4, Appendix B):** The reviewer correctly identified that 30-60 minutes per full 3D SIMP run at 500K elements was too optimistic. Updated to 2-4 hours per run based on solver scaling analysis and published benchmarks. Total training data budget now explicitly stated as 400-2000 core-hours. Added detailed scaling analysis in Section 5.3.4.

5. **SimScale clarified as analysis-only tool (Section 4.1):** Added "Can Perform ASO?" column to the ASO tools table. SimScale is now explicitly labeled as a CFD analysis tool that cannot perform shape optimization. Table renamed to "Software Tools for Aerodynamic Analysis and Shape Optimization."

6. **Appendix B updated:** Added rows for ML training data generation compute (both DL4TO and custom approaches). Fixed 3D TO single-run estimate from 20-60 minutes to 2-4 hours.

7. **DL4TO library added throughout:** Added as a practical middle ground between "no ML for TO" and "implement everything from papers." References 80-83 added.

### R3 (2026-03-22): Addressing Followup 1

Changes made in response to Followup 1:

1. **Fusion 360 educational license fully researched (Section 3.2):** Investigated whether Generative Design is included with educational licenses. Fusion 360 is now ranked #1 for TO with educational license, with FreeCAD/BESO as the fallback.

2. **All tools re-ranked by cost-effort-results priority (Sections 3.1, 4.1, 5.6):** Rebuilt comparison tables with explicit ranking. Cost is first priority (free tools preferred), effort is second (well-documented, established tools preferred), results quality is third. SimScale promoted for ASO initial work due to zero-effort setup. Research-grade tools (FEniCS, OpenLSTO) demoted due to poor documentation.

3. **ML surrogate modeling made mandatory and central (Section 5):** Complete rewrite of ML section. Now specifies:
   - Exactly which ML model for ASO: GP with Matern 5/2 + ARD kernel via BoTorch (Section 5.2.1)
   - Exactly which ML model for TO: Tiered approach with DL4TO (recommended) and custom implementations (advanced)
   - Complete training data pipeline with sample counts, parameter ranges, and compute budgets (Section 5.2.2)
   - Full BO loop code with multi-fidelity GP, EI acquisition, and surrogate validation (Section 5.2.3)
   - Summary table: which ML model does what (Section 5.7)
   - Removed all "skip ML" shortcuts and MVP paths that bypass ML

4. **Decision flowchart updated (Appendix A):** All paths now include ML. Added Fusion 360 student license branch with configuration check.

5. **User priorities stated explicitly (Section 1.5):** Added section documenting minimize cost > minimize effort > maximize results ordering.

### R2 (2026-03-22): Addressing Round 2 Review

Changes made in response to Round 2 challenger review:

1. **SU2 objective function replaced (Section 9.3):** Removed `SURFACE_PRESSURE_DROP`, which is designed for internal flow. Replaced with three-tier approach: LIFT-based, SURFACE_TOTAL_PRESSURE, and custom Python wrapper.

2. **SU2 unsteady solver switched to compressible (Section 9.3):** Replaced `INC_NAVIER_STOKES` with `NAVIER_STOKES` + `LOW_MACH_PREC= YES` for unsteady pitching.

3. **PETG material properties corrected to FDM values (Sections 6.1, 9.2):** Changed E from 2100 MPa (bulk) to 1300 MPa (FDM XY).

4. **Steady-state proxy limitations quantified (Section 9.3.1):** Added reduced frequency analysis (k ~ 0.6) and mandatory 3-design validation protocol.

5. **Mesh generation difficulty addressed (Section 9.3):** Added subsection on thin fan blade meshing challenges.

6. **Convergence diagnosis guidance added (Section 8.3.2):** Diagnostic tables for TO and BO convergence issues.

7. **TopOpt alpha status flagged (Section 9.1):** Added DTU TopOpt codes as stable alternative.
