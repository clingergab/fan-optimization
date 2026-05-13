# Topology Optimization and Aerodynamic Shape Optimization for a 3D-Printed Hand Fan

## Comprehensive Design, Optimization, and Fabrication Guide

**Date:** 2026-03-22 (Revised)
**Revision:** R1 -- Addressing Round 1 review feedback
**Scope:** Combined structural topology optimization (TO) and aerodynamic shape optimization (ASO) with ML surrogate modeling for a 3D-printed Asian-style hand fan.

---

## Table of Contents

1. [Project Overview and Physics Background](#1-project-overview-and-physics-background)
2. [Key Equations and Physical Models](#2-key-equations-and-physical-models)
3. [Software Tools for Topology Optimization](#3-software-tools-for-topology-optimization)
4. [Software Tools for Aerodynamic Shape Optimization](#4-software-tools-for-aerodynamic-shape-optimization)
5. [ML Surrogate Modeling Tools and Approaches](#5-ml-surrogate-modeling-tools-and-approaches)
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

This project combines two distinct optimization disciplines --- structural topology optimization and aerodynamic shape optimization --- into a unified workflow, accelerated by machine learning surrogate models.

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

### 3.1 Comparison Table

| Tool | Type | Cost | Beginner Rating | 3D Support | AM Constraints | Notes |
|------|------|------|----------------|------------|----------------|-------|
| **TopOpt (Python)** | Open source | Free | 4/5 | 2D mainly | No | Excellent for learning SIMP |
| **FEniCS + SIMP** | Open source | Free | 2/5 | Yes (3D) | No | Powerful but steep learning curve |
| **BESO + CalculiX** | Open source | Free | 3/5 | Yes (3D) | Partial | Evolutionary approach, good FreeCAD integration |
| **FreeCAD FEM + FEMbyGEN** | Open source | Free | 3/5 | Yes (3D) | Yes | GUI-based, good for beginners with CAD |
| **OpenLSTO** | Open source | Free | 2/5 | Yes (3D) | No | Level-set method, smoother boundaries natively |
| **Fusion 360 (Generative Design)** | Commercial | See note below | 5/5 | Yes (3D) | Yes (full AM) | Best beginner experience, cloud-based |
| **Altair Inspire** | Commercial | ~$7,000/yr | 4/5 | Yes (3D) | Yes | Industry standard |
| **ANSYS Topology Optimization** | Commercial | ~$25,000+/yr | 3/5 | Yes (3D) | Yes | Overkill for this project |
| **nTopology** | Commercial | Quote-based | 3/5 | Yes (3D) | Yes (lattice) | Excellent for lattice structures |

**Fusion 360 Generative Design pricing note:** Generative design is NOT included in the free personal-use tier of Fusion 360. It requires either: (a) a paid Fusion 360 subscription ($680/yr as of 2026) PLUS the Generative Design Extension ($1,600/yr or ~$33 in cloud credits per study), or (b) a 30-day free trial of the extension. The free personal-use license covers basic CAD, limited simulation, and basic CAM, but explicitly excludes generative design. Total cost for a commercial user with generative design: approximately $2,280/yr. For hobbyists, the pay-per-study cloud credits model ($33/study) on top of the base subscription is the most economical approach if only a few studies are needed.

### 3.2 Recommended Tool: TopOpt (Python) for Learning + FreeCAD/BESO for Production

**For learning TO concepts:** The Python TopOpt library provides a minimal, transparent implementation. You can see exactly how SIMP works, modify parameters, and understand the algorithm before scaling up.

**For the actual fan optimization:** FreeCAD with the BESO (Bi-directional Evolutionary Structural Optimization) plugin provides a GUI-based workflow with CalculiX as the FEA solver, supporting 3D structures and export to STL for printing. This is the recommended path for beginners who want a genuinely free, fully capable tool.

**If budget allows:** Fusion 360 with the Generative Design Extension is the most beginner-friendly path. It handles meshing, solving, and AM constraints automatically, and produces CAD-ready geometry rather than raw mesh results. However, be aware of the subscription cost (see pricing note above). The key advantage over the open-source path is that Fusion 360 produces editable parametric CAD geometry, whereas BESO/CalculiX produces a density field that requires significant post-processing (see Section 8.3.1).

**Alternative for smoother TO output:** OpenLSTO uses the level-set method, which produces smoother boundaries natively and avoids much of the post-processing pain of density-based methods. The trade-off is a steeper learning curve and less mature documentation.

### 3.3 Tool Pros and Cons (Detailed)

#### TopOpt (Python)
- **Pros:** Pure Python, pip-installable, minimal dependencies, educational, SIMP with MMA solver, runs in minutes for 2D problems
- **Cons:** Primarily 2D (extensions to 3D require FEniCS), no GUI, no AM constraint support, not production-ready
- **Best for:** Understanding the algorithm, prototyping optimization formulations

#### FreeCAD + BESO + CalculiX
- **Pros:** Full GUI, integrated CAD-to-FEA-to-TO workflow, free and open source, active community, handles 3D tetrahedral meshes, export to STL/STEP
- **Cons:** BESO is not as mathematically rigorous as SIMP, FreeCAD GUI can be unintuitive, CalculiX documentation is sparse, setup requires multiple component installation, TO output requires substantial post-processing (see Section 8.3.1)
- **Best for:** Producing actual optimized fan geometry for 3D printing

#### Fusion 360 Generative Design
- **Pros:** Cloud-computed (no local hardware needed), generates multiple design candidates, handles AM constraints (overhang angle, minimum thickness), produces editable CAD geometry, intuitive workflow
- **Cons:** Requires paid Autodesk subscription ($680/yr base + Generative Design Extension at $1,600/yr or pay-per-study); NOT free for personal use for generative design features; cloud dependency; limited control over optimization algorithm internals; proprietary
- **Best for:** Getting high-quality results quickly, especially if editable CAD output is important

---

## 4. Software Tools for Aerodynamic Shape Optimization

### 4.1 Comparison Table

| Tool | Type | Cost | Beginner Rating | Adjoint Support | Unsteady | Notes |
|------|------|------|----------------|-----------------|----------|-------|
| **SU2** | Open source | Free | 3/5 | Yes (continuous + discrete) | Yes | Best open-source ASO tool |
| **OpenFOAM + adjointOptimisationFoam** | Open source | Free | 2/5 | Yes (continuous) | Limited | Powerful but complex setup |
| **DAFoam** | Open source | Free | 2/5 | Yes (discrete) | Yes | Python/OpenFOAM, MDO capable |
| **SimScale** | Cloud (freemium) | Free tier available | 5/5 | No (manual) | Yes | Browser-based, great for CFD beginners |
| **AirShaper** | Cloud (freemium) | Free tier | 5/5 | No | Limited | Upload STL, get results, no optimization loop |
| **ANSYS Fluent** | Commercial | $25,000+/yr | 3/5 | Yes | Yes | Industry standard, overkill |
| **XFLR5** | Open source | Free | 4/5 | No | No | 2D/3D panel method, fast but limited |

### 4.2 Recommended Tool: SU2 for ASO + SimScale for Quick CFD Validation

**SU2** is the recommended primary tool for aerodynamic shape optimization. It has built-in adjoint solvers for computing shape sensitivities, a shape optimization pipeline, and extensive tutorials. It handles incompressible and compressible flows and supports Free-Form Deformation (FFD) for parameterizing shape changes.

**Critical note on objective function:** For a fan, the objective is NOT drag minimization (see Section 2.2.2). SU2's default `OBJECTIVE_FUNCTION= DRAG` optimizes for steady-state drag reduction, which is wrong for a device whose purpose is to generate airflow. The correct approach is described in Section 8 Step 8 and Section 9.3.

**For unsteady optimization:** SU2 supports unsteady adjoint-based optimization (see the [Unsteady Shape Optimization NACA0012 tutorial](https://su2code.github.io/tutorials/Unsteady_Shape_Opt_NACA0012/)). This is the proper formulation for an oscillating fan but is substantially more complex to set up. Section 9.3 provides a steady-state proxy approach for beginners, with guidance on upgrading to the unsteady formulation.

**SimScale** is recommended as a complementary tool for quick CFD validation. Its browser-based interface requires no installation, has a free tier, and provides visual results that help build intuition before committing to SU2's steeper learning curve.

**For 2D preliminary studies:** XFLR5 provides fast panel-method analysis of airfoil-like cross-sections, useful for quickly screening fan blade cross-sectional profiles.

**For coupled aero-structural optimization:** DAFoam with OpenMDAO can handle coupled aerodynamic and structural optimization in a single MDO framework, avoiding the sequential decoupling limitations discussed in Section 8. This is the advanced path for users who find that aeroelastic coupling is significant for their design.

### 4.3 Tool Pros and Cons (Detailed)

#### SU2
- **Pros:** Purpose-built for shape optimization, adjoint-based sensitivities, Free-Form Deformation (FFD), well-documented tutorials, active community, supports RANS/Euler/incompressible flows, Python scripting interface, supports unsteady adjoint
- **Cons:** Command-line driven, requires mesh generation externally (Pointwise, Gmsh), configuration files are complex with hundreds of options, Linux/Mac preferred (Windows via WSL), unsteady optimization setup is substantially more complex than steady-state
- **Best for:** The actual shape optimization loop for the fan blade

#### OpenFOAM + adjointOptimisationFoam
- **Pros:** Extremely flexible, handles any flow physics, adjoint shape optimization built into v1906+, massive solver library, well-validated
- **Cons:** Steep learning curve, Linux-only (or WSL/Docker), text-file-based setup, debugging is difficult, meshing requires separate tools (snappyHexMesh, blockMesh, or Gmsh)
- **Best for:** Users already familiar with OpenFOAM who need custom physics

#### DAFoam
- **Pros:** Discrete adjoint (more accurate than continuous for complex flows), Python interface, OpenMDAO integration for MDO, handles hundreds of design variables, well-documented, can handle coupled aero-structural optimization
- **Cons:** Complex installation (Docker recommended), requires OpenFOAM knowledge, primarily aimed at aerospace applications, steep learning curve
- **Best for:** Advanced users wanting to integrate aero and structural optimization in a single MDO framework, or when aeroelastic coupling is significant

#### SimScale
- **Pros:** Zero installation, browser-based, visual mesh and result inspection, free Community tier, handles transient simulations, built-in meshing
- **Cons:** No built-in shape optimization loop (manual iteration), free tier has limited compute time and public projects, no adjoint solver, internet-dependent
- **Best for:** Quick CFD validation, building aerodynamic intuition, verifying results from SU2

---

## 5. ML Surrogate Modeling Tools and Approaches

### 5.1 Why Surrogate Models?

A single CFD simulation of the fan in unsteady oscillatory motion might take 1-4 hours. Topology optimization might require 50-200 FEA evaluations. If we couple ASO with TO, the total simulation budget can easily reach thousands of evaluations. ML surrogate models replace expensive simulations with fast predictions (milliseconds per evaluation), enabling exploration of much larger design spaces.

### 5.2 Surrogate Modeling Approaches

#### 5.2.1 Gaussian Process Regression (Kriging)

**When to use:** Small datasets (10-500 training samples), low-to-moderate dimensionality (up to ~15 design variables without dimensionality reduction), need for uncertainty quantification.

**Scalability warning:** GP regression faces the curse of dimensionality. For d design variables, a rule of thumb is that you need at least 5-10x d training samples for basic coverage. At 15-25 design variables (typical for this project's ASO), this means 75-250 samples minimum. With only 50-100 samples, GP hyperparameter optimization can become unreliable, and the uncertainty estimates --- critical for Bayesian optimization --- may be poorly calibrated. See Section 5.5 for how to address this.

**Mathematical formulation:**

```
f(x) ~ GP(m(x), k(x, x'))
```

Where:
- `m(x)` = mean function (often zero or linear)
- `k(x, x')` = covariance/kernel function

Common kernels for engineering optimization:
- **Matern 5/2:** `k(r) = sigma^2 * (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)` --- good default
- **RBF (Squared Exponential):** `k(r) = sigma^2 * exp(-r^2/(2*l^2))` --- assumes very smooth functions

**Prediction with uncertainty:**

```
f*(x*) | X, y ~ N(mu*, sigma*^2)

mu* = K(x*, X) * [K(X, X) + sigma_n^2 * I]^(-1) * y
sigma*^2 = K(x*, x*) - K(x*, X) * [K(X, X) + sigma_n^2 * I]^(-1) * K(X, x*)
```

The uncertainty estimate `sigma*` is critical for Bayesian optimization --- it tells us where the model is uncertain and thus where to sample next.

#### 5.2.2 Neural Network Surrogates

**When to use:** Large datasets (500+ samples), high dimensionality, complex nonlinear relationships, field predictions (predicting entire pressure distributions rather than scalar outputs).

**Architecture recommendations for this project:**

- **For scalar outputs (C_D, C_L, max stress):** Fully connected network, 3-5 hidden layers, 64-256 neurons per layer, ReLU or GELU activation, trained with MSE loss.
- **For field outputs (pressure distribution, flow velocity field):** Convolutional neural network (CNN) or DeepONet architecture, mapping design parameters to 2D/3D fields.

**Training data generation strategy:**

1. Generate 100-500 fan designs using Latin Hypercube Sampling (LHS) over the design parameter space.
2. Run CFD simulation for each design (can be parallelized).
3. Train neural network on (design_parameters -> performance_metrics) mapping.
4. Validate on held-out test set (20% of data).
5. Use the trained surrogate in the optimization loop.

#### 5.2.3 Bayesian Optimization (BO)

**When to use:** As the optimization strategy wrapping the surrogate model. Particularly effective when simulation budget is limited (< 200 evaluations total).

**Acquisition functions:**

- **Expected Improvement (EI):**
  ```
  EI(x) = E[max(f(x) - f_best, 0)]
         = (f_best - mu(x)) * Phi(Z) + sigma(x) * phi(Z)
  where Z = (f_best - mu(x)) / sigma(x)
  ```

- **Upper Confidence Bound (UCB):**
  ```
  UCB(x) = mu(x) + beta * sigma(x)
  ```
  where `beta` controls exploration vs. exploitation (typical: beta = 2.0).

- **Knowledge Gradient:** More advanced, considers the value of information from each potential evaluation.

#### 5.2.4 Multi-Objective Bayesian Optimization

The fan design involves competing objectives: minimize weight vs. maximize stiffness vs. maximize airflow vs. minimize waving effort. Rather than collapsing these into a single scalar, multi-objective BO can discover the Pareto front --- the set of optimal trade-offs where improving one objective necessarily worsens another.

**BoTorch supports this natively** via `qNoisyExpectedHypervolumeImprovement` (qNEHVI) and `qExpectedHypervolumeImprovement` (qEHVI). These acquisition functions extend expected improvement to the multi-objective setting by computing improvement in hypervolume dominated by the Pareto front.

**Recommended formulation for this project:**

- **Objective 1:** Maximize airflow metric (J_fan from Section 2.2.2)
- **Objective 2:** Minimize total fan mass
- **Constraint:** Maximum stress < yield / SF
- **Reference point:** The baseline unoptimized fan's performance (used to compute hypervolume)

The output is a Pareto front, allowing the designer to choose the preferred trade-off between airflow and weight.

### 5.3 Recommended ML Tools

| Tool | Type | Best For | Beginner Rating |
|------|------|----------|----------------|
| **BoTorch** | Python library | Bayesian optimization with GPs | 3/5 |
| **GPyTorch** | Python library | Scalable Gaussian processes | 3/5 |
| **scikit-learn** | Python library | Basic GP regression, neural networks | 5/5 |
| **Ax (Adaptive Experimentation)** | Python platform | High-level BO interface | 4/5 |
| **PyTorch** | Python library | Custom neural network surrogates | 3/5 |
| **OpenTURNS** | Python library | Uncertainty quantification, surrogate modeling | 2/5 |

#### Recommended Stack for This Project

**Beginner path (up to ~10 design variables):**
1. **scikit-learn** for initial GP surrogate (`GaussianProcessRegressor`)
2. **scipy.optimize** for acquisition function optimization
3. Note: scikit-learn's GP implementation does not scale well beyond ~10 variables or ~500 training points

**Standard path (10-25 design variables, recommended):**
1. **BoTorch + GPyTorch** for Bayesian optimization with GPU-accelerated GPs
2. **PyTorch** for custom neural network surrogates if data volume justifies it
3. BoTorch handles multi-objective, multi-fidelity, and constrained BO natively

### 5.4 Multi-Fidelity Surrogate Strategy

For this project, a multi-fidelity approach is not optional but essentially required to make GP-based BO feasible within a reasonable compute budget, especially for 15-25 design variables:

- **Low fidelity:** Coarse mesh CFD (5 min/run), panel method (seconds/run), or simplified analytical models
- **High fidelity:** Fine mesh RANS CFD (1-4 hrs/run)

A multi-fidelity GP (e.g., AR1 model or MFDNN) correlates cheap low-fidelity data with expensive high-fidelity data:

```
f_high(x) = rho * f_low(x) + delta(x)
```

Where `rho` is a scaling factor and `delta(x)` is a GP capturing the discrepancy. This can reduce the number of expensive high-fidelity runs by 50-80%.

**Implementation in BoTorch:** Use `SingleTaskMultiFidelityGP` with a fidelity parameter.

### 5.5 Handling High Dimensionality (15-25 ASO Variables)

With 15-25 FFD design variables, standard GP regression with 50-100 samples is at the edge of feasibility. The following strategies are essential:

1. **Dimensionality reduction before surrogate fitting:** Reduce the 15-25 FFD parameters to 8-12 effective dimensions using PCA on the initial LHS sample set, or active subspace methods (which identify the directions in parameter space that most affect the output). This can dramatically improve GP fit quality.

2. **Conservative sample counts:** Use at least 8-10x the effective dimensionality for initial LHS sampling. For 10 effective dimensions, that means 80-100 initial samples.

3. **Multi-fidelity is mandatory:** Use 200-500 low-fidelity samples (coarse mesh, 5 min each) supplemented by 30-50 high-fidelity samples (fine mesh, 1-4 hrs each). The multi-fidelity GP leverages cheap data to fill the design space while using expensive data for accuracy.

4. **Use BoTorch, not scikit-learn:** For 15+ variables, BoTorch/GPyTorch with automatic relevance determination (ARD) kernels is strongly preferred. ARD learns per-dimension lengthscales, effectively performing implicit dimensionality reduction.

5. **Validate surrogate quality:** After fitting the GP, compute leave-one-out cross-validation error. If the normalized RMSE exceeds 15%, the surrogate is unreliable and more samples are needed.

---

## 6. 3D Printing Materials Selection

### 6.1 Material Properties Comparison

| Property | PLA | PETG | PA (Nylon) | PLA-CF | PA-CF |
|----------|-----|------|------------|--------|-------|
| **Tensile Strength (MPa)** | 50-60 | 40-50 | 40-85 | 60-70 | 80-110 |
| **Flexural Modulus (GPa)** | 3.5-4.0 | 2.0-2.1 | 1.2-2.0 | 5.0-7.0 | 4.0-7.0 |
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
- Reasonable cost
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

**Realistic timeline for a beginner:** 3-6 months (not 8-9 weeks). The estimate below assumes prior experience with at least one of: CAD, programming, or simulation. For a complete beginner to all three, add 2-4 additional weeks of learning time.

The main time sinks are:
- Learning SU2 (command-line CFD with complex config files): 2-4 weeks for a beginner
- FreeCAD + BESO setup and troubleshooting: 1-2 weeks
- GP/BoTorch surrogate modeling (if no Python experience): 2-3 weeks
- TO post-processing (Section 8.3.1): 1-2 weeks
- 3D printing iteration: 1-2 weeks

**Minimum viable project (for faster results):** Skip the ML surrogate layer entirely. Use SU2's built-in adjoint-based ASO directly (no surrogate needed --- SU2 computes exact gradients via the adjoint method). This eliminates BoTorch/GP setup and reduces the project to: baseline CFD -> adjoint ASO -> TO -> print. Estimated timeline: 4-6 weeks for someone with basic Python and CAD skills.

### Phase 1: Baseline Design and Preliminary Analysis (Week 1-2)

1. **Define requirements:** Target fan size, weight limit, desired airflow, material choice.
2. **Create baseline CAD geometry:** Use FreeCAD to model a simple flat paddle fan.
   - Dimensions: 250mm wide x 200mm tall x 2.5mm thick blade, 100mm handle.
3. **Print baseline fan:** Print in PLA for initial testing and tactile feedback.
4. **Measure baseline performance:** Use an anemometer or smoke visualization to characterize airflow from the unoptimized fan waved at a comfortable pace.
5. **Estimate loading conditions:** From waving motion kinematics, calculate approximate aerodynamic and inertial loads on the fan blade.

### Phase 2: Aerodynamic Shape Optimization (Week 3-6)

6. **Set up CFD model:** Install SU2 or use SimScale. Create a mesh around the baseline fan geometry.
7. **Run baseline CFD:** Simulate the fan at representative velocity (V = 2 m/s), compute pressure distribution and airflow pattern.
8. **Define ASO problem:**
   - **Objective:** Maximize directed momentum flux (NOT minimize drag --- see Section 2.2.2). In SU2, use the pressure integral on the downstream face as a proxy objective, or set up an unsteady optimization if you can afford the complexity (see Section 9.3 for both approaches).
   - **Design variables:** Planform shape, camber, thickness distribution. Reduce from 15-25 raw FFD parameters to 8-12 effective dimensions using PCA or sensitivity screening (Section 5.5).
   - **Constraints:** Fan fits within maximum envelope, minimum thickness for printability.
9. **Generate training data:** Run CFD simulations using Latin Hypercube Sampling:
   - For 8-12 effective dimensions: 80-120 initial samples (10x dimensionality)
   - Use multi-fidelity strategy: 150-200 coarse-mesh runs (5 min each) + 40-60 fine-mesh runs (1-4 hrs each)
   - **Alternative (minimum viable path):** Skip surrogate entirely, use SU2 adjoint-based ASO directly with 8-12 FFD parameters.
10. **Train ML surrogate:** Fit a GP using BoTorch (recommended) or scikit-learn (only if < 10 variables). Validate with leave-one-out cross-validation; normalized RMSE should be < 15%.
11. **Run Bayesian optimization:** Use the GP surrogate + Expected Improvement acquisition to find optimal shapes, running 20-50 additional high-fidelity CFD evaluations for validation and model refinement. Consider multi-objective BO (Section 5.2.4) if trading off airflow vs. weight.
12. **Extract optimized shape:** The result is an optimized fan planform and curvature profile.

### Phase 3: Topology Optimization (Week 6-9)

13. **Apply aerodynamic loads:** Use the pressure distribution from the optimized CFD solution as the load case for TO.
14. **Set up TO problem:** In FreeCAD/BESO:
    - Design domain: The optimized fan blade envelope (from Phase 2).
    - Loads: Pressure distribution from CFD + inertial loads from waving acceleration.
    - Constraints: Volume fraction = 0.35, minimum member size = 1.0mm, overhang angle <= 45 deg.
    - Fixed regions: Handle, blade rim.
    - **Material model:** Use isotropic properties for the optimization (SIMP), but record the assumption. The post-optimization verification (Phase 4) will use orthotropic properties to check validity.
15. **Run topology optimization:** Iterate until convergence (typically 50-200 iterations).

#### 8.3.1 Post-Processing TO Results: From Density Field to Printable STL

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

16. **Export optimized geometry:** Save smoothed, verified STL or STEP file.

### Phase 4: Verification, Coupling Check, and Validation (Week 9-12)

17. **FEA verification with orthotropic properties:** Run a stress analysis on the final optimized geometry using direction-dependent material properties (not just isotropic) to verify that the structure is safe given FDM anisotropy:
    - In CalculiX, define orthotropic elastic properties: `*ELASTIC, TYPE=ENGINEERING CONSTANTS` with different E values for XY vs. Z directions.
    - Verify: `max(sigma_vm) < sigma_yield_Z / 1.5` for features loaded in the Z direction, where `sigma_yield_Z` is the reduced yield strength in the build direction.
    - Verify: `max(displacement) < 5mm` at blade tip (acceptable deflection).

18. **Modal analysis (resonance check):**
    - Compute the first 5-10 natural frequencies of the optimized fan blade using FreeCAD FEM or CalculiX (`*FREQUENCY` step).
    - The first bending mode natural frequency must be well above the waving frequency and its first few harmonics: `f_1 > 5 * f_waving` (i.e., f_1 > 10-15 Hz for 2-3 Hz waving).
    - If the natural frequency is too close to the waving frequency or its harmonics, the blade could exhibit resonant vibration, excessive noise, or accelerated fatigue failure.
    - A thin, topology-optimized blade with reduced material is at higher risk of low natural frequencies than a solid blade.

19. **CFD verification and coupling check:** Run a final high-fidelity CFD simulation on the TO-optimized geometry to verify aerodynamic performance is maintained.
    - This step serves as a **coupling check**, not just verification: if the TO-modified geometry (internal structure, edge profiles) significantly changes the aerodynamic performance compared to Phase 2 results, iteration is needed.
    - **Coupling criterion:** If the airflow metric changes by more than 10% compared to the Phase 2 optimized shape, return to Phase 2 Step 8 and re-optimize ASO with the TO-modified geometry as the new baseline.
    - **Aeroelastic estimate:** If maximum blade tip deflection under aerodynamic load exceeds 2% of chord length, the sequential decoupling assumption breaks down. Consider either (a) iterating between ASO and TO until convergence, or (b) switching to a coupled aero-structural framework (DAFoam + OpenMDAO).

20. **Print prototype:** Print in PETG with the recommended settings from Section 6.3.
21. **Physical testing:**
    - Subjective: Wave the fan and compare comfort/airflow to baseline.
    - Quantitative: Anemometer measurements at fixed distance (see Section 10).
22. **Iterate if necessary:** Adjust volume fraction, camber, or other parameters based on test results.

### Phase 5: Final Production (Week 12-14)

23. **Final print:** Print in PA-CF or PETG depending on performance requirements.
24. **Post-processing:** Light sanding if needed for surface finish (see Section 2.2.5 on roughness effects), handle wrapping for comfort.
25. **Documentation:** Record all parameters, settings, and results for reproducibility.

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
3. **Add material:** Select PETG (or define custom: E = 2100 MPa, nu = 0.38, density = 1270 kg/m^3).
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

**Important:** The configuration below uses a steady-state proxy objective. This is a simplification --- a real fan operates in unsteady oscillatory motion. The proxy maximizes the pressure integral on the downstream face at peak waving velocity, which correlates with fan airflow performance. See the "Unsteady Approach" section below for the proper formulation.

```ini
% fan_optimization.cfg
%
% Physics
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
% Mesh
MESH_FILENAME= fan_blade.su2
MESH_FORMAT= SU2
%
% Objective function: pressure integral on downstream face
% This is a proxy for directed airflow, NOT drag minimization
OBJECTIVE_FUNCTION= SURFACE_PRESSURE_DROP
%
% Design variables (FFD)
DV_KIND= FFD_CONTROL_POINT
FFD_DEFINITION= (BOX, -0.01,-0.05,0.0, 0.26,-0.05,0.0, 0.26,0.05,0.0, -0.01,0.05,0.0, ...)
%
% Optimization
OPT_OBJECTIVE= SURFACE_PRESSURE_DROP * 1.0
OPT_CONSTRAINT= NONE
DEFINITION_DV= ...
```

**Note on objective function:** `DRAG` is the wrong objective for a fan. The fan's purpose is to push air, not to minimize resistance. `SURFACE_PRESSURE_DROP` between upstream and downstream monitoring planes is a better proxy. Alternatively, define a custom objective via SU2's Python wrapper that computes the net downstream momentum flux.

#### SU2 Configuration: Unsteady Approach (Advanced)

For the proper unsteady formulation (optimizing over a full waving cycle), follow the [SU2 Unsteady Shape Optimization tutorial](https://su2code.github.io/tutorials/Unsteady_Shape_Opt_NACA0012/). Key additional configuration:

```ini
% Unsteady simulation
TIME_DOMAIN= YES
TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER
TIME_STEP= 0.001
MAX_TIME= 1.0      % One full waving cycle at 1 Hz
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
```

The unsteady adjoint computes sensitivities of the cycle-averaged objective to design parameters, accounting for the full oscillatory motion. This is the correct formulation but requires 10-50x more compute time than the steady-state proxy.

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
```

#### Workflow: Surrogate-Assisted Optimization

```python
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np

# Step 1: Define design parameter bounds
# Example: 10 FFD control point displacements for fan shape
# (reduced from 15-25 raw parameters via PCA -- see Section 5.5)
n_dim = 10
bounds = torch.tensor([
    [-0.02] * n_dim,  # lower bounds (m)
    [0.02] * n_dim,   # upper bounds (m)
])

# Step 2: Generate initial training data via CFD
# Use at least 8-10x the number of design variables
def run_cfd_simulation(design_params):
    """
    Run SU2 CFD simulation and return performance metrics.
    Returns: fan performance metric (maximize).
    Note: This should be the pressure integral proxy,
    NOT negative drag coefficient.
    """
    # ... write SU2 config, run simulation, parse results ...
    pass

# Latin Hypercube Sampling for initial points
from scipy.stats import qmc
sampler = qmc.LatinHypercube(d=n_dim)
n_initial = 10 * n_dim  # 100 initial samples for 10 dimensions
X_init = torch.tensor(sampler.random(n=n_initial), dtype=torch.double)
X_init = bounds[0] + (bounds[1] - bounds[0]) * X_init

Y_init = torch.tensor(
    [run_cfd_simulation(x) for x in X_init],
    dtype=torch.double
).unsqueeze(-1)

# Step 3: Bayesian optimization loop
for iteration in range(50):
    # Fit GP model
    gp = SingleTaskGP(X_init, Y_init)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # Validate surrogate (check every 10 iterations)
    if iteration % 10 == 0:
        # Leave-one-out cross-validation
        from sklearn.model_selection import LeaveOneOut
        # ... compute NRMSE, warn if > 0.15 ...

    # Compute acquisition function
    best_f = Y_init.max()
    EI = ExpectedImprovement(model=gp, best_f=best_f)

    # Optimize acquisition function
    candidate, acq_value = optimize_acqf(
        EI, bounds=bounds, q=1, num_restarts=10, raw_samples=512
    )

    # Evaluate candidate with actual CFD
    new_y = run_cfd_simulation(candidate.squeeze())

    # Update training data
    X_init = torch.cat([X_init, candidate])
    Y_init = torch.cat([Y_init, torch.tensor([[new_y]], dtype=torch.double)])

    print(f"Iteration {iteration}: best performance = {Y_init.max().item():.4f}")

# Final best design
best_idx = Y_init.argmax()
best_design = X_init[best_idx]
print(f"Optimal design parameters: {best_design}")
```

**Key parameters to tune:**
- `num_restarts`: Number of random restarts for acquisition optimization (10-20)
- `raw_samples`: Number of raw samples for initial acquisition candidates (256-1024)
- Initial sample count: 8-10x the number of design variables (80-100 for 10 variables)
- Total budget: 100-200 evaluations for 10-15 effective design variables (not 50-100)
- **Surrogate validation:** Check NRMSE periodically; if > 15%, add more training samples before continuing

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

### General References

55. [Topology Optimization - Wikipedia](https://en.wikipedia.org/wiki/Topology_optimization) --- General reference.
56. [Von Mises Yield Criterion - Wikipedia](https://en.wikipedia.org/wiki/Von_Mises_yield_criterion) --- 3D von Mises equation reference.
57. [Optimization of Design Parameters and 3D-Printing Orientation (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2590048X25000470) --- Print orientation for TO parts.

### Fusion 360 Pricing

58. [Autodesk Fusion for Personal Use - Feature Comparison](https://www.autodesk.com/products/fusion-360/personal) --- What is and is not included in the free tier.
59. [Autodesk Fusion Plans and Pricing](https://www.autodesk.com/products/fusion-360/extensions) --- Extension pricing including Generative Design.
60. [Fusion 360 Generative Design Extension Pricing Change (Autodesk News)](https://adsknews.autodesk.com/en/news/generative-design-extension/) --- Cloud credits model.

---

## Appendix A: Quick-Start Decision Flowchart

```
START: What is your budget?
|
|-- $0 (Free tools only):
|   |
|   |-- Do you have Python experience?
|   |   |
|   |   |-- YES --> TopOpt (Python, 2D learning) + FreeCAD/BESO (3D production)
|   |   |           + SU2 (ASO) + BoTorch (surrogate)
|   |   |           Timeline: 3-6 months for full workflow
|   |   |
|   |   |-- NO  --> Start with FreeCAD/BESO (GUI-based TO)
|   |               + SimScale (browser-based CFD, free tier)
|   |               Skip ML surrogate; use SU2 adjoint directly
|   |               Timeline: 2-3 months for simplified workflow
|   |
|-- $680-2,280/yr (Autodesk subscription):
|   |
|   |-- Use Fusion 360 + Generative Design Extension (TO)
|   |   + SimScale (CFD validation)
|   |   Key advantage: produces editable CAD, not raw mesh
|   |   Timeline: 4-8 weeks
|   |
|-- Student/Educational:
|   |
|   |-- Check Autodesk educational license (Fusion 360 may include
|   |   generative design for educational users --- verify current terms)
|   |   Otherwise, follow the $0 path
```

## Appendix B: Estimated Computation Times

| Task | Hardware | Estimated Time |
|------|----------|---------------|
| 2D TO (TopOpt Python, 200x60 mesh) | Any laptop | 5-30 seconds |
| 3D TO (BESO/CalculiX, 500K elements) | Desktop, 8 cores | 20-60 minutes |
| 3D TO (Fusion 360 cloud) | Cloud (Autodesk) | 30-120 minutes |
| TO post-processing (smoothing, repair) | Any laptop | 1-4 hours (manual) |
| 2D CFD steady (SU2, 50K cells) | Any laptop | 2-10 minutes |
| 3D CFD steady (SU2, 500K cells) | Desktop, 8 cores | 30-120 minutes |
| 3D CFD unsteady (SU2, 5 cycles) | Desktop, 8 cores | 2-8 hours |
| Modal analysis (CalculiX, first 10 modes) | Desktop, 8 cores | 5-20 minutes |
| GP surrogate training (100 samples, 10 dims) | Any laptop | 1-10 seconds |
| GP surrogate training (100 samples, 20 dims) | Desktop | 10-60 seconds |
| BO iteration (acquisition opt) | Any laptop | 0.5-5 seconds |
| Full BO loop (50 iterations with CFD) | Desktop, 8 cores | 1-5 days |
| Full BO loop (50 iter with surrogate) | Any laptop | 1-5 minutes |

## Appendix C: Recommended Minimum Hardware

- **CPU:** 4+ cores (8+ recommended for CFD)
- **RAM:** 16 GB (32 GB for large 3D CFD)
- **Storage:** 50 GB free (CFD output files can be large)
- **GPU:** Not required for TO/CFD, but helpful for PyTorch surrogate training
- **OS:** Linux recommended for SU2/OpenFOAM (macOS works, Windows via WSL2)
- **3D Printer:** Any FDM printer with 220x220mm+ bed, heated bed, direct drive extruder preferred for PETG/PA

---

## Revision Log

### R1 (2026-03-22): Addressing Round 1 Review

Changes made in response to challenger review:

1. **Fusion 360 pricing corrected (Sections 3.1, 3.2, 3.3, Appendix A):** Generative design is NOT free for personal use. Updated pricing to reflect base subscription + extension cost. Revised decision flowchart to route free-tool users to open-source options.

2. **FDM anisotropy section added (Section 2.1.5, updated 6.1, 7.4, 10.1):** New subsection explaining isotropic SIMP limitation, connection to print orientation, directional safety factors, and references to SOMP formulations.

3. **TO/ASO coupling addressed (Sections 1.3, 8 Phase 4):** Added coupling criterion (10% performance change triggers re-iteration), aeroelastic deflection check (2% chord), and DAFoam + OpenMDAO as coupled alternative.

4. **Aerodynamic objective corrected (Sections 2.2.2, 4.2, 8 Step 8, 9.3):** Replaced drag minimization with directed momentum flux / pressure integral proxy. Added unsteady SU2 configuration. Explicit warning that DRAG is the wrong objective for a fan.

5. **GP surrogate feasibility improved (Sections 5.2.1, 5.5, 7.3, 9.4):** Added scalability warning, dimensionality reduction guidance, conservative sample counts (8-10x dimensionality), multi-fidelity as mandatory, ARD kernels, and LOO-CV validation.

6. **TO post-processing expanded (Section 8.3.1):** New subsection with 5-step pipeline: thresholding, marching cubes, Taubin smoothing, mesh repair, verification FEA. Tool recommendations included.

7. **Full 3D von Mises equation (Section 2.1.3):** Replaced 2D plane stress version with full 3D equation, with note on when 2D simplification applies.

8. **Modal analysis added (Phase 4 Step 18, Section 10.1):** Natural frequency check to prevent resonance during waving.

9. **Multi-objective optimization added (Section 5.2.4):** BoTorch qNEHVI for Pareto front discovery across weight, stiffness, and airflow objectives.

10. **Realistic timeline (Section 8):** Updated from 8-9 weeks to 3-6 months for beginners. Added minimum viable project option (skip surrogate layer).

11. **Surface roughness effects (Section 2.2.5):** New subsection on FDM layer-line roughness and its impact on boundary layer transition at low Re.

12. **Fatigue analysis improved (Section 2.1.4):** Corrected cycle count estimates, replaced single endurance ratio with recommendation for S-N curves and print-parameter-dependent behavior, referenced PETG-specific fatigue data.

13. **Cost/benefit discussion added (Section 1.4):** Justification for why optimization is worth the engineering effort.

---

*This report was generated through comprehensive web research on 2026-03-22 and revised the same day in response to expert review. Software versions and pricing are subject to change. Always verify current availability and compatibility before committing to a toolchain.*
