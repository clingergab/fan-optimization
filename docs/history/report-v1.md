# Topology Optimization and Aerodynamic Shape Optimization for a 3D-Printed Hand Fan

## Comprehensive Design, Optimization, and Fabrication Guide

**Date:** 2026-03-22
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

These two problems are coupled: the outer shape affects the aerodynamic loads, and those loads determine the structural requirements that TO must satisfy.

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

For the fan to survive repeated waving without failure:

```
sigma_vm = sqrt(sigma_x^2 + sigma_y^2 - sigma_x*sigma_y + 3*tau_xy^2)
```

The von Mises stress at every point must satisfy `sigma_vm < sigma_yield / SF`, where `SF` is a safety factor (typically 1.5-2.0 for fatigue loading).

**Typical yield strengths for 3D-printed materials:**
- PLA: ~50-60 MPa (but brittle)
- PETG: ~40-50 MPa (more ductile)
- Nylon (PA): ~40-85 MPa (excellent fatigue life)
- PLA-CF: ~60-70 MPa (stiff but brittle)

#### 2.1.4 Fatigue Considerations

A hand fan undergoes oscillatory loading. For N = 10,000 wave cycles per use session, the fatigue endurance limit matters:

```
sigma_fatigue ~ 0.3 to 0.5 * sigma_ultimate   (for polymers, rough estimate)
```

PLA is notably poor in fatigue (brittle fracture). PETG and nylon are significantly better choices for a repeatedly loaded structure.

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

**Thrust/airflow generation:**

The useful output of a hand fan is the momentum imparted to the air, creating a directed airflow. The figure of merit is:

```
eta_fan = (momentum of directed airflow) / (energy input by waving)
```

In practice, we want to maximize the volume flow rate `Q` (m^3/s) of air directed toward the user per unit of waving effort.

**Pressure coefficient:**

```
C_p = (p - p_inf) / (0.5 * rho_air * V^2)
```

Mapping `C_p` across the fan surface reveals regions that contribute most to airflow generation versus wasted drag.

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

---

## 3. Software Tools for Topology Optimization

### 3.1 Comparison Table

| Tool | Type | Cost | Beginner Rating | 3D Support | AM Constraints | Notes |
|------|------|------|----------------|------------|----------------|-------|
| **TopOpt (Python)** | Open source | Free | 4/5 | 2D mainly | No | Excellent for learning SIMP |
| **FEniCS + SIMP** | Open source | Free | 2/5 | Yes (3D) | No | Powerful but steep learning curve |
| **BESO + CalculiX** | Open source | Free | 3/5 | Yes (3D) | Partial | Evolutionary approach, good FreeCAD integration |
| **FreeCAD FEM + FEMbyGEN** | Open source | Free | 3/5 | Yes (3D) | Yes | GUI-based, good for beginners with CAD |
| **OpenLSTO** | Open source | Free | 2/5 | Yes (3D) | No | Level-set method, academic |
| **Fusion 360 (Generative Design)** | Commercial | $545/yr (free for personal) | 5/5 | Yes (3D) | Yes (full AM) | Best beginner experience, cloud-based |
| **Altair Inspire** | Commercial | ~$7,000/yr | 4/5 | Yes (3D) | Yes | Industry standard |
| **ANSYS Topology Optimization** | Commercial | ~$25,000+/yr | 3/5 | Yes (3D) | Yes | Overkill for this project |
| **nTopology** | Commercial | Quote-based | 3/5 | Yes (3D) | Yes (lattice) | Excellent for lattice structures |

### 3.2 Recommended Tool: TopOpt (Python) for Learning + FreeCAD/BESO for Production

**For learning TO concepts:** The Python TopOpt library provides a minimal, transparent implementation. You can see exactly how SIMP works, modify parameters, and understand the algorithm before scaling up.

**For the actual fan optimization:** FreeCAD with the BESO (Bi-directional Evolutionary Structural Optimization) plugin provides a GUI-based workflow with CalculiX as the FEA solver, supporting 3D structures and export to STL for printing.

**If budget allows:** Fusion 360 with Generative Design is the most beginner-friendly path. It handles meshing, solving, and AM constraints automatically, and produces CAD-ready geometry rather than raw mesh results.

### 3.3 Tool Pros and Cons (Detailed)

#### TopOpt (Python)
- **Pros:** Pure Python, pip-installable, minimal dependencies, educational, SIMP with MMA solver, runs in minutes for 2D problems
- **Cons:** Primarily 2D (extensions to 3D require FEniCS), no GUI, no AM constraint support, not production-ready
- **Best for:** Understanding the algorithm, prototyping optimization formulations

#### FreeCAD + BESO + CalculiX
- **Pros:** Full GUI, integrated CAD-to-FEA-to-TO workflow, free and open source, active community, handles 3D tetrahedral meshes, export to STL/STEP
- **Cons:** BESO is not as mathematically rigorous as SIMP, FreeCAD GUI can be unintuitive, CalculiX documentation is sparse, setup requires multiple component installation
- **Best for:** Producing actual optimized fan geometry for 3D printing

#### Fusion 360 Generative Design
- **Pros:** Cloud-computed (no local hardware needed), generates multiple design candidates, handles AM constraints (overhang angle, minimum thickness), produces editable CAD geometry, intuitive workflow
- **Cons:** Requires Autodesk subscription ($545/yr, but free for personal/educational use), cloud dependency, limited control over optimization algorithm internals, proprietary
- **Best for:** Getting high-quality results quickly without deep TO knowledge

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

**SimScale** is recommended as a complementary tool for quick CFD validation. Its browser-based interface requires no installation, has a free tier, and provides visual results that help build intuition before committing to SU2's steeper learning curve.

**For 2D preliminary studies:** XFLR5 provides fast panel-method analysis of airfoil-like cross-sections, useful for quickly screening fan blade cross-sectional profiles.

### 4.3 Tool Pros and Cons (Detailed)

#### SU2
- **Pros:** Purpose-built for shape optimization, adjoint-based sensitivities, Free-Form Deformation (FFD), well-documented tutorials, active community, supports RANS/Euler/incompressible flows, Python scripting interface
- **Cons:** Command-line driven, requires mesh generation externally (Pointwise, Gmsh), configuration files are complex with hundreds of options, Linux/Mac preferred (Windows via WSL), unsteady optimization is more complex
- **Best for:** The actual shape optimization loop for the fan blade

#### OpenFOAM + adjointOptimisationFoam
- **Pros:** Extremely flexible, handles any flow physics, adjoint shape optimization built into v1906+, massive solver library, well-validated
- **Cons:** Steep learning curve, Linux-only (or WSL/Docker), text-file-based setup, debugging is difficult, meshing requires separate tools (snappyHexMesh, blockMesh, or Gmsh)
- **Best for:** Users already familiar with OpenFOAM who need custom physics

#### DAFoam
- **Pros:** Discrete adjoint (more accurate than continuous for complex flows), Python interface, OpenMDAO integration for MDO, handles hundreds of design variables, well-documented
- **Cons:** Complex installation (Docker recommended), requires OpenFOAM knowledge, primarily aimed at aerospace applications, steep learning curve
- **Best for:** Advanced users wanting to integrate aero and structural optimization in a single MDO framework

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

**When to use:** Small datasets (10-500 training samples), low-to-moderate dimensionality (up to ~20 design variables), need for uncertainty quantification.

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

**Beginner path:**
1. **scikit-learn** for initial GP surrogate (`GaussianProcessRegressor`)
2. **scipy.optimize** for acquisition function optimization

**Advanced path:**
1. **BoTorch + GPyTorch** for Bayesian optimization with GPU-accelerated GPs
2. **PyTorch** for custom neural network surrogates if data volume justifies it

### 5.4 Multi-Fidelity Surrogate Strategy

For this project, a multi-fidelity approach is highly effective:

- **Low fidelity:** Coarse mesh CFD (5 min/run), panel method (seconds/run), or simplified analytical models
- **High fidelity:** Fine mesh RANS CFD (1-4 hrs/run)

A multi-fidelity GP (e.g., AR1 model or MFDNN) correlates cheap low-fidelity data with expensive high-fidelity data:

```
f_high(x) = rho * f_low(x) + delta(x)
```

Where `rho` is a scaling factor and `delta(x)` is a GP capturing the discrepancy. This can reduce the number of expensive high-fidelity runs by 50-80%.

**Implementation in BoTorch:** Use `SingleTaskMultiFidelityGP` with a fidelity parameter.

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
| **Print Difficulty** | Easy | Easy-Medium | Hard | Medium | Hard |
| **Print Temp (C)** | 190-220 | 230-250 | 250-270 | 200-230 | 260-280 |
| **Bed Temp (C)** | 50-60 | 70-80 | 70-90 | 50-60 | 80-100 |
| **Needs Enclosure** | No | No | Yes | No | Yes |
| **Needs Dry Box** | No | Recommended | Essential | Recommended | Essential |
| **Cost ($/kg)** | $15-25 | $20-30 | $30-50 | $30-50 | $50-80 |

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

### 7.4 Print Orientation Considerations

**Recommended orientation: Flat on the build plate (blade horizontal)**

Advantages:
- Maximum blade surface quality (top and bottom)
- No supports needed for flat or gently curved blades
- Strongest inter-layer bonding direction aligned with bending loads
- Fastest print time

Disadvantages:
- Large footprint on build plate (may exceed bed size for large fans)
- Layer lines perpendicular to primary bending axis (weaker direction)

**Alternative: Blade at 45 degrees**

If the fan exceeds build plate dimensions, printing at 45 degrees reduces footprint but requires support material and increases print time.

**Critical consideration:** The weakest direction in FDM prints is between layers (Z-direction). Orient the fan so that the primary bending moment during waving acts along layer lines (XY plane), not between layers (Z). For a flat fan printed horizontally, this means the flexural loads during waving should be in-plane, which is naturally the case.

---

## 8. Step-by-Step Project Execution Plan

### Phase 1: Baseline Design and Preliminary Analysis (Week 1-2)

1. **Define requirements:** Target fan size, weight limit, desired airflow, material choice.
2. **Create baseline CAD geometry:** Use FreeCAD or Fusion 360 to model a simple flat paddle fan.
   - Dimensions: 250mm wide x 200mm tall x 2.5mm thick blade, 100mm handle.
3. **Print baseline fan:** Print in PLA for initial testing and tactile feedback.
4. **Measure baseline performance:** Use an anemometer or smoke visualization to characterize airflow from the unoptimized fan waved at a comfortable pace.
5. **Estimate loading conditions:** From waving motion kinematics, calculate approximate aerodynamic and inertial loads on the fan blade.

### Phase 2: Aerodynamic Shape Optimization (Week 3-5)

6. **Set up CFD model:** Install SU2 or use SimScale. Create a mesh around the baseline fan geometry.
7. **Run baseline CFD:** Simulate the fan at representative velocity (V = 2 m/s), compute C_D, pressure distribution, and airflow pattern.
8. **Define ASO problem:**
   - Objective: Maximize directed airflow (or minimize drag for given airflow).
   - Design variables: Planform shape, camber, thickness distribution (15-25 FFD parameters).
   - Constraints: Fan fits within maximum envelope, minimum thickness for printability.
9. **Generate training data:** Run 50-100 CFD simulations using Latin Hypercube Sampling of the design parameter space.
10. **Train ML surrogate:** Fit a Gaussian Process to the CFD data using scikit-learn or BoTorch.
11. **Run Bayesian optimization:** Use the GP surrogate + Expected Improvement acquisition to find optimal shapes, running 20-50 additional high-fidelity CFD evaluations for validation and model refinement.
12. **Extract optimized shape:** The result is an optimized fan planform and curvature profile.

### Phase 3: Topology Optimization (Week 5-7)

13. **Apply aerodynamic loads:** Use the pressure distribution from the optimized CFD solution as the load case for TO.
14. **Set up TO problem:** In FreeCAD/BESO or Fusion 360 Generative Design:
    - Design domain: The optimized fan blade envelope (from Phase 2).
    - Loads: Pressure distribution from CFD + inertial loads from waving acceleration.
    - Constraints: Volume fraction = 0.35, minimum member size = 1.0mm, overhang angle <= 45 deg.
    - Fixed regions: Handle, blade rim.
15. **Run topology optimization:** Iterate until convergence (typically 50-200 iterations).
16. **Interpret and clean results:** Smooth the TO output, ensure printability, close small holes, verify minimum feature sizes.
17. **Export optimized geometry:** Save as STL or STEP file.

### Phase 4: Verification and Validation (Week 7-8)

18. **FEA verification:** Run a stress analysis on the final optimized geometry to confirm stresses are within material limits (von Mises stress < yield strength / safety factor).
19. **CFD verification:** Run a final high-fidelity CFD simulation on the TO-optimized geometry to verify aerodynamic performance is maintained.
20. **Print prototype:** Print in PETG with the recommended settings from Section 6.3.
21. **Physical testing:**
    - Subjective: Wave the fan and compare comfort/airflow to baseline.
    - Quantitative: Anemometer measurements at fixed distance (see Section 10).
22. **Iterate if necessary:** Adjust volume fraction, camber, or other parameters based on test results.

### Phase 5: Final Production (Week 8-9)

23. **Final print:** Print in PA-CF or PETG depending on performance requirements.
24. **Post-processing:** Light sanding if needed for surface finish, handle wrapping for comfort.
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
9. **Export** optimized geometry as STL.

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

#### SU2 Configuration File (Key Parameters)

```ini
% fan_optimization.cfg
%
% Physics
SOLVER= INC_NAVIER_STOKES
KIND_TURB_MODEL= NONE
%
% Flow conditions (fan at 2 m/s)
INC_VELOCITY_INIT= (2.0, 0.0, 0.0)
INC_DENSITY_INIT= 1.225
INC_TEMPERATURE_INIT= 300.0
VISCOSITY_MODEL= CONSTANT_VISCOSITY
MU_CONSTANT= 1.81e-5
%
% Mesh
MESH_FILENAME= fan_blade.su2
MESH_FORMAT= SU2
%
% Objective function
OBJECTIVE_FUNCTION= DRAG
%
% Design variables (FFD)
DV_KIND= FFD_CONTROL_POINT
FFD_DEFINITION= (BOX, -0.01,-0.05,0.0, 0.26,-0.05,0.0, 0.26,0.05,0.0, -0.01,0.05,0.0, ...)
%
% Optimization
OPT_OBJECTIVE= DRAG * 1.0
OPT_CONSTRAINT= (LIFT > 0.0)
DEFINITION_DV= ...
```

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
bounds = torch.tensor([
    [-0.02] * 10,  # lower bounds (m)
    [0.02] * 10,   # upper bounds (m)
])

# Step 2: Generate initial training data via CFD
# (Run 20-30 initial CFD simulations using Latin Hypercube Sampling)
def run_cfd_simulation(design_params):
    """
    Run SU2 CFD simulation and return performance metrics.
    Returns: negative drag coefficient (we minimize, so negate for maximization).
    """
    # ... write SU2 config, run simulation, parse results ...
    pass

# Latin Hypercube Sampling for initial points
from scipy.stats import qmc
sampler = qmc.LatinHypercube(d=10)
X_init = torch.tensor(sampler.random(n=30), dtype=torch.double)
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

    print(f"Iteration {iteration}: best C_D = {-Y_init.max().item():.4f}")

# Final best design
best_idx = Y_init.argmax()
best_design = X_init[best_idx]
print(f"Optimal design parameters: {best_design}")
```

**Key parameters to tune:**
- `num_restarts`: Number of random restarts for acquisition optimization (10-20)
- `raw_samples`: Number of raw samples for initial acquisition candidates (256-1024)
- Initial sample count: 2-3x the number of design variables (20-30 for 10 variables)
- Total budget: 50-100 evaluations typically sufficient for 10-20 design variables

---

## 10. Validation Approaches

### 10.1 Computational Validation

#### FEA Stress Verification

After topology optimization, run a final FEA on the optimized geometry (not the pseudo-density field, but the actual solid model after interpretation/smoothing):

1. Import the optimized STL/STEP into FreeCAD FEM or SimScale.
2. Apply the same loads used during TO (aerodynamic pressure + inertial).
3. Apply safety factor loading (1.5x the design loads).
4. Verify: `max(sigma_vm) < sigma_yield / 1.5`
5. Verify: `max(displacement) < 5mm` at blade tip (acceptable deflection).

#### CFD Verification of Optimized Shape

After aerodynamic shape optimization, verify the final design with a refined CFD simulation:

1. Re-mesh the final geometry with a finer mesh (2x the optimization mesh density).
2. Run steady-state CFD at multiple velocities (1, 2, 3 m/s).
3. Run an unsteady simulation of 2-3 waving cycles.
4. Compare C_D and airflow rate against the surrogate model predictions.
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

**Expected improvement target:** 15-30% increase in average airflow velocity from the optimized fan vs. the baseline flat fan, at the same waving effort (frequency and amplitude).

#### Smoke/Incense Visualization

For qualitative flow visualization:
1. Light an incense stick and position it 100-200mm from the fan face.
2. Wave the fan and observe smoke deflection patterns.
3. Record with a video camera for comparison between baseline and optimized designs.
4. Look for: directed airflow (vs. turbulent dispersion), vortex patterns, dead zones.

#### Structural Testing

1. **Static deflection test:** Clamp the handle, hang a known weight (50-200g) from the blade tip. Measure deflection. Compare to FEA prediction.
2. **Fatigue test:** Wave the fan continuously for 30 minutes (~3600 cycles at 2 Hz). Inspect for cracks, delamination, or permanent deformation.
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
16. [OpenFOAM](https://www.openfoam.com/) --- Open-source CFD toolbox.
17. [OpenFOAM adjointOptimisationFoam Manual (PDF)](https://www.openfoam.com/documentation/files/adjointOptimisationFoamManual.pdf) --- Adjoint-based shape optimization in OpenFOAM.
18. [DAFoam: Discrete Adjoint with OpenFOAM](https://dafoam.github.io/) --- High-fidelity MDO platform.
19. [DAFoam GitHub](https://github.com/mdolab/dafoam) --- Source code and documentation.
20. [SimScale: CFD Simulation Software](https://www.simscale.com/product/cfd/) --- Cloud-based CFD platform.
21. [AirShaper](https://airshaper.com/) --- Simplified aerodynamic analysis.

### Aerodynamic Physics

22. [Drag Coefficient - Wikipedia](https://en.wikipedia.org/wiki/Drag_coefficient) --- Reference for flat plate C_D values.
23. [Drag on Oscillating Flat Plates at Low Reynolds Numbers (Cambridge Core)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/drag-on-oscillating-flat-plates-in-liquids-at-low-reynolds-numbers/4A6222AC968750F25984BD2538E5DCDA) --- Directly relevant experimental data.
24. [NASA: Shape Effects on Drag](https://www.grc.nasa.gov/www/k-12/VirtualAero/BottleRocket/airplane/shaped.html) --- Educational resource on drag coefficients.

### ML Surrogate Modeling

25. [BoTorch: Bayesian Optimization in PyTorch](https://botorch.org/docs/overview/) --- Official documentation.
26. [GPyTorch Models in BoTorch](https://botorch.org/docs/models/) --- GP model reference.
27. [Multi-Fidelity Deep Neural Network Surrogate Model for ASO (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0045782520306708) --- MFDNN for aerodynamic optimization.
28. [Bayesian Optimization of Lightweight Neural Network for Aerodynamic Prediction (arXiv)](https://arxiv.org/html/2503.19479v1) --- Recent paper on BO for aero.
29. [Review of Deep Learning Surrogates for Aerodynamic Shape Optimization (AIP)](https://pubs.aip.org/aip/pof/article/37/4/041304/3345111/Review-of-deep-learning-based-aerodynamic-shape) --- Comprehensive 2025 review.
30. [Enhanced Data Efficiency with DNNs and GPs for Aero Design (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1270963821000341) --- Combining neural networks with Gaussian processes.
31. [Bayesian Optimization Introduction (Blog)](https://www.miguelgondu.com/blogposts/2023-07-31/intro-to-bo/) --- Beginner-friendly BO tutorial.

### 3D Printing Materials

32. [PETG vs PLA vs ABS: Strength Comparison (Ultimaker)](https://ultimaker.com/learn/petg-vs-pla-vs-abs-3d-printing-strength-comparison/) --- Material property comparison.
33. [PETG vs PLA: Strength and Printing Comparison (Xometry)](https://www.xometry.com/resources/3d-printing/petg-vs-pla-3d-printing/) --- Detailed mechanical properties.
34. [3D Printer Filament Comparison (MatterHackers)](https://www.matterhackers.com/3d-printer-filament-compare) --- Comprehensive filament database.
35. [Bambu Carbon Fiber Filaments (Blog)](https://blog.bambulab.com/bambu-carbon-fiber-filaments/) --- CF filament properties.
36. [3D Printing with PLA Carbon Fiber Filament (Nobufil)](https://www.nobufil.com/en/post/3d-printing-with-pla-cf-filament) --- PLA-CF material guide.

### 3D Printed Fan Designs

37. [3D Printed Hand Fan (GrabCAD)](https://grabcad.com/library/3d-printed-hand-fan) --- Example CAD model.
38. [Print-in-Place Hand Fan (Printables)](https://www.printables.com/model/132278-print-in-place-hand-fan) --- Practical 3D printed fan design.
39. [FreeCAD FEM Workbench Tutorial (DigiKey)](https://www.digikey.com/en/maker/tutorials/2025/intro-to-freecad-part-10-finite-element-method-fem-workbench-tutorial) --- FreeCAD FEM beginner guide.

### Topology Optimization Theory

40. [SIMP Method Numerical Analysis (arXiv)](https://arxiv.org/html/2211.04249) --- Mathematical analysis of SIMP convergence.
41. [Topology Optimization - Wikipedia](https://en.wikipedia.org/wiki/Topology_optimization) --- General reference.
42. [Optimization of Design Parameters and 3D-Printing Orientation (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2590048X25000470) --- Print orientation for TO parts.

---

## Appendix A: Quick-Start Decision Flowchart

```
START: Do you have CAD experience?
|
|-- YES --> Do you have budget for commercial software?
|           |
|           |-- YES --> Use Fusion 360 Generative Design (TO) + SimScale (CFD)
|           |           Easiest path, cloud-computed, minimal setup
|           |
|           |-- NO  --> Use FreeCAD + BESO (TO) + SU2 (ASO)
|                        More setup, but fully open-source and powerful
|
|-- NO  --> Start with:
            1. TopOpt (Python) to learn TO concepts (2D)
            2. SimScale (free tier) to learn CFD basics
            3. Then graduate to FreeCAD + SU2
```

## Appendix B: Estimated Computation Times

| Task | Hardware | Estimated Time |
|------|----------|---------------|
| 2D TO (TopOpt Python, 200x60 mesh) | Any laptop | 5-30 seconds |
| 3D TO (BESO/CalculiX, 500K elements) | Desktop, 8 cores | 20-60 minutes |
| 3D TO (Fusion 360 cloud) | Cloud (Autodesk) | 30-120 minutes |
| 2D CFD steady (SU2, 50K cells) | Any laptop | 2-10 minutes |
| 3D CFD steady (SU2, 500K cells) | Desktop, 8 cores | 30-120 minutes |
| 3D CFD unsteady (SU2, 5 cycles) | Desktop, 8 cores | 2-8 hours |
| GP surrogate training (100 samples) | Any laptop | 1-10 seconds |
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

*This report was generated through comprehensive web research on 2026-03-22. Software versions and pricing are subject to change. Always verify current availability and compatibility before committing to a toolchain.*
