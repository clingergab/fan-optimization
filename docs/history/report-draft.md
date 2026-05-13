# Topology Optimization and Aerodynamic Shape Optimization for a 3D-Printed Folding Fan

## Comprehensive Design, Optimization, and Fabrication Guide

**Date:** 2026-03-23 (Revised)
**Revision:** R6 -- Fixes fabricated PyTopo3D API (replaced with 2D SIMP via FEniCS or forked PyTopo3D), reformulates rib TO as 2D planform optimization, removes false SU2 porous media claim, fixes CadQuery installation, integrates multi-fidelity BO, adds membrane tension and discrete rib count handling
**Scope:** Combined structural topology optimization (TO) and aerodynamic shape optimization (ASO) with ML surrogate modeling for a 3D-printed folding fan (sensu/ogi style) with individually optimized ribs, fabric membrane, and pivot mechanism.

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

---

## 1. Project Overview: The Folding Fan

### 1.1 What We Are Designing

A folding fan (sensu/ogi style) consisting of multiple thin ribs radiating from a shared pivot point, connected by a fabric or paper membrane. When deployed, the ribs spread into a sector arc; when collapsed, they stack into a compact bundle. This is fundamentally different from a paddle fan (uchiwa) in every engineering dimension:

- **Structural domain:** Not a single continuous plate, but an array of individual thin beams (ribs), each of which is a separate TO domain.
- **Aerodynamic domain:** Not a solid surface, but a series of rigid ribs with a flexible membrane spanning the gaps between them. Air can leak through gaps, and the membrane billows under load.
- **Assembly:** Multiple printed parts (ribs) plus fabric plus a pivot pin, not a monolithic print.
- **Mechanism:** The pivot constrains the base geometry of every rib and introduces concentrated stress at the hinge point.

### 1.2 Why This Is Harder Than a Paddle Fan

| Aspect | Paddle Fan (uchiwa) | Folding Fan (sensu/ogi) |
|--------|---------------------|------------------------|
| TO domain | Single continuous plate | N individual rib beams (each optimized separately) |
| ASO domain | Solid surface | Ribs + membrane + gaps (porous surface) |
| CFD complexity | Standard bluff body | Rib-gap flow leakage, membrane deformation |
| Structural loads | Distributed bending | Bending per rib + concentrated stress at pivot |
| Assembly | Single print | N ribs + fabric + pivot pin |
| Print orientation | Flat on bed | Each rib printed flat; thin cross-section |

Despite this complexity, the folding fan IS the project. The engineering challenges are what make it interesting.

### 1.3 The Two Optimization Problems (Folding Fan Context)

**Topology Optimization (TO):** For each individual rib, determine the optimal **planform** -- where to place material (and where to create cutouts/voids) within the rib's 2D length-width envelope at constant thickness. The rib is a thin structural beam (typically 2 mm thick, 6-12 mm wide, 150-250 mm long). Because the rib is so thin (2 mm), through-thickness TO is not meaningful (see Section 9.2 for rationale); instead, TO operates in the plane of the rib, creating lightening holes and optimized web patterns that maximize stiffness-to-weight ratio under the combined loading of aerodynamic pressure (transmitted through the membrane), inertial forces from waving, and the reaction force at the pivot.

**Aerodynamic Shape Optimization (ASO):** Determine the optimal overall fan geometry -- the spread angle, rib count, rib spacing, membrane tension, and the curvature profile of the deployed fan surface -- to maximize directed airflow when waved. The ASO domain includes the gaps between ribs (where air leaks through) and the membrane behavior (which is not rigid).

### 1.4 User Priorities

1. **Minimize cost** -- student budget; free tools strongly preferred
2. **Minimize effort** -- scripted/programmatic workflows preferred over GUI tools; Claude Code handles most of the coding
3. **Maximize results** -- best achievable fan performance within the above constraints

### 1.5 Is This Optimization Worth the Effort?

The honest answer for a folding fan is: the aerodynamic gains from ASO will be modest (5-15%), because the membrane-and-rib geometry is already a well-evolved design. The structural TO of individual ribs is where the most tangible benefit lies -- published TO studies on thin beam-like structures consistently show 15-30% stiffness-to-weight improvements over uniform cross-sections. For a folding fan, lighter ribs mean less fatigue at the pivot, a more comfortable waving experience, and potentially more ribs (better membrane support) without added weight.

The primary value remains: (a) learning the TO/ASO/ML workflow on a tangible project, (b) producing quantifiably better ribs through systematic engineering, and (c) exploring a design space that intuition alone cannot navigate.

---

## 2. Folding Fan Geometry and Engineering Constraints

### 2.1 Typical Folding Fan Dimensions

Based on traditional Japanese sensu and ogi fans, and adapted for 3D printing:

| Parameter | Traditional Range | Recommended Starting Point | Notes |
|-----------|------------------|---------------------------|-------|
| **Rib count** | 10-30 | 15 | Edo-sensu typically use 15 thick ribs; more ribs = smoother surface but more parts |
| **Rib length** | 150-250 mm | 200 mm (approx. 7.5 inches) | Men's fans ~7 inches, women's ~6 inches |
| **Rib width (at widest)** | 8-15 mm | 12 mm | Tapers from base to tip |
| **Rib thickness** | 1.5-3.0 mm | 2.0 mm | Constrained by FDM minimum feature size |
| **Rib taper** | Linear or slight curve | Linear: 12 mm at base to 6 mm at tip | Traditional bamboo ribs taper |
| **Spread angle (deployed)** | 90-180 degrees | 120 degrees | ~1/3 of a circle; most common for practical use |
| **Guard sticks (outer ribs)** | 2 (one each side) | 2, wider and thicker | Protect the folded fan; 15-20 mm wide, 3 mm thick |
| **Pivot hole diameter** | 2-4 mm | 3 mm | For a metal rivet or pin |
| **Pivot location** | Base of rib, 5-10 mm from end | 8 mm from base end | All ribs share this pivot point |
| **Membrane material** | Paper, silk, fabric | Lightweight polyester or cotton | Glued to ribs; spans the gaps |
| **Membrane overhang** | 0-5 mm beyond rib tips | 3 mm | Stiffened edge optional |
| **Gap between ribs (deployed)** | Depends on spread angle and rib count | ~5 mm at tip for 15 ribs at 120 degrees | Membrane spans this gap |

### 2.2 Rib Geometry Details

Each rib is essentially a tapered cantilever beam fixed at the pivot. The cross-section at any point along the rib length is the TO design domain.

**Planform envelope (for TO):**
```
Width:  w(x) = w_base - (w_base - w_tip) * (x / L)
        w_base = 12 mm, w_tip = 6 mm, L = 200 mm

Thickness: h = 2.0 mm (uniform, constrained by FDM layer count)

The TO domain is the 2D tapered planform (length x width).
At constant 2 mm thickness, TO determines which portions of the planform
are material vs. void (creating cutout patterns and lightening holes).
See Section 9.2 for why 2D planform TO is preferred over 3D voxel TO for
this rib thickness.
```

**Pivot region constraints:**
- The bottom 8-10 mm of every rib must have a 3 mm hole for the pivot pin.
- This region cannot be topology-optimized -- it is a preserved zone.
- The pivot hole creates a stress concentration factor of approximately 2.5-3.0x (hole in a plate under tension/bending).
- The rib must transition from the full-width pivot region to the tapered body smoothly.

**Guard stick (outer rib) differences:**
- Wider (15-20 mm) and thicker (3 mm) than inner ribs.
- Not topology-optimized -- they serve as protective covers when folded.
- Include a locking tab or friction fit to hold the fan open.

### 2.3 Assembly and Mechanism

**Pivot assembly:**
1. All inner ribs (13 in a 15-rib fan) are stacked on a single pin/rivet.
2. The two guard sticks are the outermost pieces.
3. A washer or spacer between each rib prevents binding.
4. The rivet is peened or a nut is used to hold the assembly.
5. Ribs must rotate freely but with enough friction to hold position.

**Practical 3D printing approach:**
- Print each rib as a separate flat piece (printed flat on the build plate for maximum strength).
- The pivot hole is printed as a through-hole in the rib base.
- Use a 3 mm brass rod or machine screw as the pivot pin.
- Thin nylon washers (0.5 mm) between ribs for smooth rotation.
- Fabric is glued to the flat side of ribs after assembly.

**Print-in-place alternative:**
Several 3D-printed folding fan designs (e.g., [Printables #944878](https://www.printables.com/model/944878), [Printables #132278](https://www.printables.com/model/132278)) use print-in-place construction where the entire fan mechanism is printed as a single piece with integrated living hinges or captive pivots. This eliminates assembly entirely but imposes significant constraints:
- Material must be flexible enough for living hinges (typically TPU or thin PLA), which conflicts with the stiffness needed for effective fanning.
- Rib geometry is constrained by the print-in-place mechanism (uniform thickness, no cutouts near the pivot).
- TO of individual ribs is not possible since the ribs are integral with the mechanism.
- The fan cannot be disassembled for membrane attachment (the membrane must be part of the print or omitted).

For an engineering optimization project, the traditional assembly approach (separate ribs + pivot pin + membrane) is preferred because it allows independent optimization of each component. Print-in-place is better suited for quick functional prototypes where assembly effort is the primary concern.

**Membrane attachment:**
- The membrane (fabric/paper) is glued to one face of each rib, spanning the gap between adjacent ribs.
- The membrane does not wrap around ribs -- it sits on the front face only.
- When the fan is closed, the membrane folds in accordion pleats between the stacked ribs.
- Membrane tension affects both aerodynamic performance (taut = less billowing) and structural loads on ribs (tension pulls ribs together).

### 2.4 Structural Loading on Each Rib

Each rib experiences three types of loads:

**1. Aerodynamic pressure (distributed):**
- Transmitted through the membrane to the ribs.
- Each rib carries the pressure from half the gap on each side plus its own width.
- Tributary width per rib: w_rib + gap_between_ribs (approximately 13-15 mm per rib at mid-span).
- Pressure magnitude: 5-20 Pa at peak waving velocity (~2.5 m/s tip speed).
- This acts as a distributed transverse load along the rib length.

**2. Inertial forces (distributed):**
- From the rib's own mass undergoing oscillatory acceleration.
- At 2 Hz waving frequency with 40-degree amplitude: peak angular acceleration ~ 200 rad/s^2.
- A 2-gram rib at 100 mm centroid distance experiences ~ 0.04 N inertial force.
- Small compared to aerodynamic loads but contributes to fatigue.

**3. Pivot reaction (concentrated at base):**
- The pivot pin provides a fixed support (moment and shear reaction).
- All loads along the rib resolve to a reaction force and moment at the pivot.
- The pivot hole creates a stress concentration -- this is the most failure-prone location.
- Peak bending moment at pivot: M = integral of distributed load x distance.

**4. Membrane tension (distributed in-plane):**
- The membrane, when taut, pulls adjacent ribs toward each other. This creates an in-plane closing moment that acts to reduce the fan spread angle.
- For a membrane tension T (N/m) spanning a gap of width g at angle alpha between ribs, the in-plane force per unit length on each rib is approximately T * sin(alpha/2), directed toward the neighboring rib.
- At 120-degree spread with 15 ribs (inter-rib angle ~8.6 degrees), this force is small per rib but cumulative across all ribs. It primarily loads the pivot and the membrane attachment line along the rib.
- This in-plane component is NOT included in the basic transverse-pressure TO load case described above. For a first-pass TO, omitting it is acceptable because the transverse bending loads dominate. However, for a refined design, the membrane tension should be added as a secondary load case in the TO formulation (multi-load-case SIMP).

**5. Rib-rib interaction through membrane coupling:**
- Each rib is optimized independently in this formulation, but ribs interact through the shared membrane. Changing one rib's stiffness affects the membrane tension distribution and hence the loads on adjacent ribs.
- For a first project, this simplification is acceptable. The coupling is weak because membrane tension is small relative to aerodynamic pressure. However, this limitation should be noted: the optimal rib design found by independent TO is an approximation, not a true system optimum.

**Combined loading for TO:** The rib TO problem is a cantilever beam fixed at the pivot, with distributed transverse load (aerodynamic + inertial) as the primary load case and optional in-plane membrane tension as a secondary load case, seeking minimum compliance (maximum stiffness) subject to a volume fraction constraint. This is a classic TO problem well-suited to SIMP.

### 2.5 Deployed Fan Geometry for CFD

When deployed at 120 degrees, the 15 ribs form a sector with:
- Arc length at tip: 2 * pi * 200 mm * (120/360) = 419 mm
- Gap between rib tips: (419 - 15 * 6) / 14 = 23.5 mm (rib tip width = 6 mm)
- Gap between rib bases: much smaller (~2-3 mm)
- The membrane spans these gaps, creating a quasi-continuous surface with rib ridges.

**CFD modeling approaches for the rib-membrane geometry:**

1. **Simplified solid surface (recommended for initial ASO):** Model the deployed fan as a solid curved plate (sector shape) with the rib positions marked as thickened ridges. This ignores air leakage through gaps but captures the overall aerodynamic shape. Justification: with a taut membrane, leakage is minimal.

2. **Porous surface model (intermediate, OpenFOAM only):** Model the fan surface as a porous zone with direction-dependent permeability -- low permeability where membrane is present, high where gaps exist. OpenFOAM supports porous media natively via its `porousMedia` framework. **SU2 does NOT support porous media or porous jump boundary conditions** -- this has been confirmed via the [SU2 boundary condition documentation](https://su2code.github.io/docs_v7/Markers-and-BC/) and [CFD-Online forum discussions](https://www.cfd-online.com/Forums/su2/240454-su2-porous-media-porous-jump-model.html). Implementing custom source terms in SU2 would require modifying the C++ solver code, which is far beyond a scripted workflow. If you want the porous approach, use OpenFOAM (but note that OpenFOAM is significantly harder to script for automated ASO loops than SU2).

3. **Resolved rib-gap geometry (high fidelity):** Explicitly mesh each rib and gap, with the membrane as a thin wall and gaps as open passages. This is computationally expensive (requires fine mesh to resolve gap flows) but captures leakage effects accurately.

**Recommendation:** Start with approach 1 for initial ASO (shape optimization of the overall fan planform and camber). Use approach 3 for validation of the final design to quantify leakage effects. The leakage penalty can then be fed back as a correction factor.

---

## 3. Key Equations and Physical Models

### 3.1 Structural Mechanics -- Rib Topology Optimization

#### 3.1.1 The SIMP Formulation (Applied to a Single Rib)

Each finite element in the rib design domain is assigned a pseudo-density `rho_e` in [0, 1], where 0 = void and 1 = solid material.

**Material interpolation (power law):**

```
E(rho_e) = E_min + rho_e^p * (E_0 - E_min)
```

Where:
- `E_0` = Young's modulus of the solid material (e.g., PETG FDM-printed: ~1300 MPa in XY)
- `E_min` = small value to prevent singularity (typically 1e-9 * E_0)
- `p` = penalization exponent (typically p = 3)
- `rho_e` = element pseudo-density (design variable)

**Compliance minimization for a single rib:**

```
minimize:    C(rho) = F^T * u = sum_e (rho_e^p * u_e^T * K_0_e * u_e)
subject to:  K(rho) * u = F                    (equilibrium)
             sum_e (rho_e * v_e) <= V*          (volume constraint)
             0 < rho_min <= rho_e <= 1          (bounds)
             rho_e = 1 for pivot region          (preserved region)
```

Where:
- `C` = compliance (strain energy; lower = stiffer)
- `F` = applied force vector (aerodynamic + inertial loads on the rib)
- `u` = displacement vector
- `V*` = maximum allowed material volume (volume fraction target, typically 0.3-0.5)

**Sensitivity (gradient):**

```
dC/d(rho_e) = -p * rho_e^(p-1) * u_e^T * K_0_e * u_e
```

#### 3.1.2 Density Filtering

```
rho_tilde_e = (sum_i H_ei * v_i * rho_i) / (sum_i H_ei * v_i)
```

Where `H_ei = max(0, r_min - dist(e, i))` with filter radius `r_min` = 1.5 to 3 times the element size.

#### 3.1.3 Stress at the Pivot (Critical Failure Location)

The pivot hole creates a stress concentration. For a circular hole of diameter `d` in a plate of width `w` under bending:

```
K_t = 3.0 - 3.13*(d/w) + 3.66*(d/w)^2 - 1.53*(d/w)^3
```

For d = 3 mm, w = 12 mm (rib base width): d/w = 0.25, K_t approximately 2.65.

The peak stress at the pivot region:

```
sigma_max = K_t * sigma_nominal = K_t * (M * c) / I
```

Where M is the bending moment at the pivot, c is the distance from the neutral axis to the extreme fiber, and I is the second moment of area of the rib cross-section at the pivot (accounting for the hole).

**Fatigue at the pivot:** With K_t = 2.65 and cyclic loading at 2 Hz, the pivot region is the fatigue-critical location. Keep peak cyclic stress below 30% of yield / K_t to ensure adequate fatigue life (see Section 3.1.5).

#### 3.1.4 Von Mises Stress Criterion

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

#### 3.1.5 Fatigue Considerations for Rib Oscillatory Loading

A hand fan rib undergoes oscillatory loading:
- **Per session:** At 2 Hz for 5-15 minutes: 600-1,800 cycles per session.
- **Lifetime:** Daily use over months: 50,000-500,000 total cycles.

**Polymer fatigue for FDM parts:**
- Fatigue life depends on print parameters (raster angle, infill density, layer height).
- For FDM PETG, raster angle at 45 degrees outperforms 0 degrees at low stress amplitudes.
- Conservative design rule: keep peak cyclic stresses below 30% of yield strength.
- At the pivot with K_t = 2.65: effective allowable stress = 0.30 * 45 MPa / 2.65 = 5.1 MPa nominal. This is the critical constraint.

#### 3.1.6 FDM Material Anisotropy

FDM-printed ribs exhibit 20-40% reduction in mechanical properties in the build direction (Z-axis) compared to in-plane (XY). Since ribs are printed flat on the build plate:
- Primary bending loads act in-plane (XY) -- the strong direction.
- The isotropic SIMP assumption is most valid for this print orientation.
- Verify with FEA post-optimization using orthotropic properties.

**FDM-printed PETG properties (100% infill, 0.2mm layer):**
- E_XY ~ 1300 MPa (NOT the datasheet value of 2100 MPa which is for injection-molded)
- E_Z ~ 1000 MPa
- nu = 0.38, density = 1270 kg/m^3

### 3.2 Aerodynamics -- Folding Fan in Oscillatory Motion

#### 3.2.1 Flow Regime

- **Fan rib length (L):** ~0.20 m
- **Waving velocity (V):** 1-3 m/s (tip speed)
- **Reynolds number:** Re = V * L / nu = 10,000 - 40,000
- **Regime:** Low Reynolds number, unsteady, laminar to transitional

#### 3.2.2 Aerodynamic Differences: Folding Fan vs. Solid Plate

A folding fan differs from a solid plate in several ways that affect CFD:

1. **Gap leakage:** Air passes through gaps between ribs (especially near tips where gaps are widest). This reduces the effective pressure difference across the fan and decreases airflow efficiency.

2. **Rib ridges:** Ribs protrude from the membrane surface, creating small flow disturbances. These can trip the boundary layer and affect separation behavior.

3. **Membrane billowing:** Under aerodynamic load, the membrane between ribs deflects. This changes the effective camber of the fan surface locally.

4. **Non-rigid surface:** Unlike a solid plate, the deployed fan surface has some compliance, which could act as a flutter source at certain frequencies.

**Fan performance metric -- directed momentum flux:**

```
eta_fan = (integral over one cycle of net momentum flux toward user dt) /
          (integral over one cycle of waving power input dt)
```

#### 3.2.3 Oscillating Motion Model

```
theta(t) = theta_max * sin(2*pi*f*t)
V_tip(t) = L * theta_max * 2*pi*f * cos(2*pi*f*t)
V_tip_max = L * theta_max * 2*pi*f
```

For L = 0.3 m (arm + fan), theta_max = 0.7 rad (40 deg), f = 2 Hz:
V_tip_max = 0.3 * 0.7 * 2 * pi * 2 = 2.64 m/s

**Reduced frequency:** k = pi * f * c / V ~ 0.60 (firmly unsteady regime).

#### 3.2.4 Surface Roughness Effects

FDM layer lines (Ra = 5-15 micrometers) affect boundary layer transition at these Reynolds numbers. For ribs, the layer lines run along the rib length (parallel to flow for the primary waving direction), which minimizes their impact. The membrane surface is smooth (fabric/paper), which is favorable.

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
- **Claude can:** Write the complete problem definition with custom BCs (fixed pivot region) and loads (distributed pressure), run optimization, export results as coordinate arrays that feed into CadQuery for STL generation.
- **Limitation:** 2D formulation assumes constant rib thickness. Through-thickness features (if desired) require 3D TO.

#### PyTopo3D (Alternative: 3D TO with Source Modification)

- **What it does:** 3D SIMP topology optimization on structured voxel grids. `pip install pytopo3d` (Python 3.10+).
- **Actual API:** The real API is a single function `top3d(nelx, nely, nelz, volfrac, penal, rmin, disp_thres, obstacle_mask=None, use_gpu=False)` from `pytopo3d.core.optimizer`. There is NO class-based API, no `set_fixed_region()`, no `set_distributed_load()`, no `export_stl()` method. The boundary conditions and loads are **hardcoded** to the standard cantilever beam benchmark (one face fixed, point load on opposite face).
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
| **3D printing** | Physical task requiring a printer |
| **Rib assembly** | Physical: inserting pivot pin, adding washers, peening rivet |
| **Membrane attachment** | Physical: cutting fabric, gluing to ribs |
| **Physical testing** | Waving the fan, anemometer measurements, smoke visualization |
| **Material testing** | Printing and testing tensile specimens |
| **Visual inspection** | Checking print quality, layer adhesion, pivot fit |
| **Subjective evaluation** | How the fan feels to wave, comfort, aesthetics |

### 5.2 Recommended Workflow: Claude Writes, User Runs

The ideal workflow maximizes Claude's contribution:

```
Phase 1: Geometry and Baseline
  Claude writes: CadQuery rib generator script
  User runs:     python generate_ribs.py --> rib_01.stl ... rib_15.stl
  User does:     3D prints baseline ribs, assembles fan, tests

Phase 2: Structural TO of Ribs (2D Planform Optimization)
  Claude writes: 2D SIMP script with loads from Phase 1 estimates
  User runs:     python optimize_rib_2d.py --> optimized density field
  Claude writes: CadQuery script to convert density field to rib STL
  User runs:     python density_to_rib.py --> optimized_rib.stl
  Claude writes: CalculiX verification FEA input file
  User runs:     ccx verification_rib --> stress results
  Claude writes: Result parsing and visualization script
  User runs:     python plot_results.py

Phase 3: Aerodynamic Shape Optimization
  Claude writes: Gmsh meshing script for deployed fan shape
  User runs:     python mesh_fan.py --> fan.su2
  Claude writes: SU2 config file (steady-state, then unsteady)
  User runs:     SU2_CFD fan.cfg --> baseline CFD results
  Claude writes: LHS sampling + batch CFD runner
  User runs:     python run_cfd_batch.py --> training_data.csv (runs overnight)
  Claude writes: BoTorch GP fitting + Bayesian optimization
  User runs:     python optimize_shape.py --> best_shape_params.json
  Claude writes: Updated CadQuery script with optimized parameters
  User runs:     python generate_optimized_fan.py --> optimized ribs

Phase 4: Validation and Iteration
  Claude writes: Final FEA verification scripts
  User runs:     ccx final_verification
  Claude writes: Final CFD verification scripts
  User runs:     SU2_CFD final_verification.cfg
  User does:     Print optimized ribs, assemble, test
  User reports:  Test results to Claude
  Claude writes: Adjusted scripts based on test feedback
```

### 5.3 Estimated Scripting Effort for Claude

For the full project, Claude would write approximately:
- **15-25 Python scripts** (geometry, TO, CFD setup, ML, post-processing)
- **5-10 config files** (SU2, BESO, CalculiX)
- **3-5 shell wrapper scripts** (batch runners)
- **Total: ~2,000-4,000 lines of code**

This is well within Claude Code's capabilities. The user's primary effort is running the scripts, interpreting results, and performing physical tasks.

---

## 6. ML Surrogate Modeling: Core Workflow

### 6.1 Why ML Surrogates Are Important

**ML for ASO (Aerodynamic Shape Optimization):** A single CFD simulation of the deployed fan takes 30 minutes to 4 hours. Exploring the full design space (rib count, spread angle, camber profile, rib curvature) via direct CFD would require thousands of simulations. An ML surrogate replaces the CFD solver with a model that predicts aerodynamic performance in milliseconds.

**ML for TO (Topology Optimization):** For a single rib, TO runs in minutes to hours via PyTopo3D or BESO. ML acceleration is less critical here because the rib TO is a relatively small problem (compared to a full fan blade). However, if exploring many rib cross-section variants or coupling TO with ASO iteratively, ML can accelerate the inner loop.

**Recommendation:** ML-for-ASO via BoTorch is the core ML component. ML-for-TO is an optional extension for this project.

### 6.2 ML for ASO (Folding Fan Parameters)

#### 6.2.1 Design Parameters (Folding Fan Specific)

| Parameter Group | Count | Range | Description |
|----------------|-------|-------|-------------|
| Spread angle | 1 | 90-150 deg | Total deployment angle |
| Rib count | 1 | 10-25 | Number of ribs (**discrete**; see note below on handling) |
| Rib curvature profile | 3-4 | 0-10 mm | Camber of individual ribs (defines fan surface curvature) |
| Rib length | 1 | 150-250 mm | Overall fan size |
| Rib taper ratio | 1 | 0.3-0.8 | Tip width / base width |
| Membrane tension | 1 | 0-5 N/m | Affects billowing and effective camber |
| Edge profile | 1-2 | 0.5-2 mm | Rib cross-section shape at leading/trailing edge |

**Total parameters:** 9-11. Manageable for GP-based Bayesian optimization without dimensionality reduction.

**Discrete rib count handling:** Rib count is fundamentally discrete and topological -- changing it from 15 to 16 ribs requires a completely different CFD mesh (one more rib to resolve, different gap widths, different membrane panels). Simple rounding of a continuous variable is inadequate because:
- Each rib count requires a separate Gmsh meshing run and SU2 mesh file.
- The `run_cfd()` wrapper must handle variable-topology meshes.
- The gap geometry changes non-smoothly with rib count.

**Recommended approach:** Use BoTorch's mixed continuous-discrete optimization. Fix rib count at a few candidate values (e.g., 12, 15, 18, 21) and run separate sub-campaigns or use `MixedSingleTaskGP` which handles categorical/discrete variables natively. Alternatively, fix rib count at 15 (the traditional sensu standard) for the initial ASO and only vary it in a follow-up sensitivity study.

#### 6.2.2 Recommended Model: GP with BoTorch

```python
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, ScaleKernel

# Matern 5/2 with ARD (learns per-dimension importance)
# SingleTaskGP uses this by default
gp_model = SingleTaskGP(train_X, train_Y)
```

**Why GP (not neural network):** With a realistic budget of 100-200 CFD simulations, GPs are the correct choice. They provide calibrated uncertainty estimates, perform well with small datasets, and support Bayesian optimization natively.

#### 6.2.3 Multi-Fidelity Bayesian Optimization Loop

**Why multi-fidelity:** At reduced frequency k ~ 0.6, the flow is firmly unsteady. Published studies on oscillating flat plates at similar reduced frequencies show that quasi-steady models can overpredict peak forces by 30-50% and, critically, can **change the relative ranking** of different geometries (because unsteady effects like dynamic stall and leading-edge vortex formation interact differently with different planforms and camber profiles). Running the entire BO budget on steady-state CFD alone risks optimizing against a proxy that does not preserve design rankings.

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

N_DIMS = 10          # Fan design parameters (excluding fidelity column)
N_STEADY_INIT = 60   # Initial steady-state LHS samples
N_UNSTEADY_INIT = 10 # Initial unsteady samples (subset of above)
N_BO_ITERS = 50      # Multi-fidelity BO iterations

# Bounds: design params + fidelity column (last dim, 0=steady, 1=unsteady)
bounds = torch.tensor(
    [[...] + [0.0],   # lower bounds + fidelity lower
     [...] + [1.0]],  # upper bounds + fidelity upper
    dtype=torch.double
)
target_fidelities = {N_DIMS: 1.0}  # optimize at highest fidelity

def run_cfd(params, fidelity):
    """Run SU2 CFD at specified fidelity. Claude writes this wrapper."""
    # fidelity=0: steady-state (~30-60 min)
    # fidelity=1: unsteady, 5 cycles (~2-8 hours)
    # 1. Generate fan geometry from params via CadQuery
    # 2. Mesh with Gmsh
    # 3. Run SU2_CFD with appropriate config (steady or unsteady)
    # 4. Parse results: compute momentum flux metric (see Section 9.4)
    # 5. Return J_fan metric
    ...

def cost_model(X):
    """Cost model: unsteady is ~10x more expensive than steady."""
    fidelity = X[..., -1]
    return 1.0 + 9.0 * fidelity  # steady=1, unsteady=10

# Generate initial training data (mostly steady, some unsteady)
sampler = qmc.LatinHypercube(d=N_DIMS)
X_steady = torch.tensor(sampler.random(n=N_STEADY_INIT), dtype=torch.double)
X_steady = bounds[0, :-1] + (bounds[1, :-1] - bounds[0, :-1]) * X_steady
X_steady = torch.cat([X_steady, torch.zeros(N_STEADY_INIT, 1)], dim=-1)  # fidelity=0

# Run unsteady on a subset
unsteady_indices = torch.randperm(N_STEADY_INIT)[:N_UNSTEADY_INIT]
X_unsteady = X_steady[unsteady_indices].clone()
X_unsteady[:, -1] = 1.0  # fidelity=1

X_init = torch.cat([X_steady, X_unsteady])
Y_init = torch.tensor(
    [run_cfd(x[:-1], x[-1].item()) for x in X_init],
    dtype=torch.double
).unsqueeze(-1)

# Multi-fidelity BO loop
X_train, Y_train = X_init, Y_init
for iteration in range(N_BO_ITERS):
    gp = SingleTaskMultiFidelityGP(
        X_train, Y_train,
        outcome_transform=Standardize(m=1),
        data_fidelities=[N_DIMS],  # last column is fidelity
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    cost_utility = InverseCostWeightedUtility(cost_model=cost_model)
    qMFKG = qMultiFidelityKnowledgeGradient(
        model=gp,
        current_value=gp.posterior(
            X_train[X_train[:, -1] == 1.0]  # value at target fidelity
        ).mean.max() if (X_train[:, -1] == 1.0).any() else torch.tensor(0.0),
        cost_aware_utility=cost_utility,
        project=lambda X: project_to_target_fidelity(X, target_fidelities),
    )

    # Optimize over design params + discrete fidelity choice
    candidate, acq_value = optimize_acqf_mixed(
        qMFKG, bounds=bounds, q=1,
        fixed_features_list=[{N_DIMS: 0.0}, {N_DIMS: 1.0}],
        num_restarts=20, raw_samples=512,
    )

    fidelity = candidate[0, -1].item()
    new_y = run_cfd(candidate[0, :-1], fidelity)
    X_train = torch.cat([X_train, candidate])
    Y_train = torch.cat([Y_train, torch.tensor([[new_y]], dtype=torch.double)])

    n_steady = (X_train[:, -1] == 0.0).sum().item()
    n_unsteady = (X_train[:, -1] == 1.0).sum().item()
    print(f"Iter {iteration}: best = {Y_train.max():.4f}, "
          f"fidelity={fidelity:.0f}, total: {n_steady} steady + {n_unsteady} unsteady")

# Extract best at target fidelity
mask_hf = X_train[:, -1] == 1.0
best_idx = Y_train[mask_hf].argmax()
best_params = X_train[mask_hf][best_idx, :-1]
```

**Budget allocation:** The multi-fidelity GP will automatically balance steady vs. unsteady evaluations based on the cost-information tradeoff. Expect roughly 70-80% steady evaluations and 20-30% unsteady, with the unsteady evaluations concentrated in the promising regions of the design space. Total effective budget: ~120-160 evaluations (equivalent to ~40-60 unsteady runs in compute time).

### 6.3 ML for TO (Optional Extension)

For rib TO, the problem is small enough (a single thin beam) that ML acceleration is unnecessary for a first project. A single 2D SIMP run on a rib discretized at 0.5 mm elements (400 x 24 = ~9,600 elements in the planform) converges in seconds to minutes.

If exploring many rib variants (different cross-section profiles, taper ratios, load cases), DL4TO can train a neural network to predict near-optimal rib topologies instantly. See Section 9.5 for details.

### 6.4 Multi-Objective Optimization

The fan design involves competing objectives:
- **Maximize airflow** (J_fan)
- **Minimize total weight** (sum of rib masses)
- **Constraint:** Peak stress at pivot < allowable fatigue stress

BoTorch supports multi-objective BO via `qNoisyExpectedHypervolumeImprovement`:

```python
from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement
from botorch.models import ModelListGP

gp_airflow = SingleTaskGP(X, Y_airflow)
gp_weight = SingleTaskGP(X, Y_weight)
model = ModelListGP(gp_airflow, gp_weight)

ref_point = torch.tensor([Y_airflow.min() - 0.1, Y_weight.min() - 0.1])
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

---

## 8. Step-by-Step Project Execution Plan

### Timeline Expectations

**Realistic timeline:** 2-4 months (shorter than the previous paddle-fan estimate because the Claude-scripted workflow eliminates much of the tool-learning overhead).

**Time breakdown:**
- Phase 1 (Geometry + baseline): 1-2 weeks
- Phase 2 (Rib TO): 1-2 weeks
- Phase 3 (ASO with ML): 3-5 weeks (dominated by CFD compute time)
- Phase 4 (Validation + iteration): 2-3 weeks
- Phase 5 (Final production): 1 week

### Phase 1: Geometry Generation and Baseline (Week 1-2)

**Claude writes:**
1. CadQuery parametric rib generator (`generate_ribs.py`)
2. Full fan assembly script for visualization (`assemble_fan.py`)
3. Guard stick geometry script

**User runs:**
4. `python generate_ribs.py --rib-count 15 --length 200 --spread 120 --output ribs/`
5. Print baseline ribs in PLA (13 inner ribs + 2 guard sticks).
6. Assemble with 3 mm brass pin, nylon washers, fabric membrane.
7. Test: wave the fan, note subjective feel. Measure airflow with anemometer at 300 mm.

**Outcome:** Baseline fan with measured performance metrics.

### Phase 2: Rib Topology Optimization -- 2D Planform (Week 3-4)

**Claude writes:**
8. 2D SIMP optimization script (`optimize_rib_2d.py`) using estimated loads from Phase 1.
   - Design domain: rib planform envelope (tapered shape, 200 mm long x 12 mm wide at base tapering to 6 mm at tip)
   - Formulation: 2D plane-stress with constant 2 mm thickness
   - Preserved region: pivot base (10 x 12 mm rectangle around 3 mm hole) -- density fixed to 1.0
   - Loads: distributed transverse pressure (from estimated aero loads) applied as body force proportional to tributary width; optional membrane tension as secondary load case
   - Volume fraction: 0.4 (40% material)
   - Filter radius: 1.5 mm (3 elements at 0.5 mm element size)
9. CadQuery post-processing script (`density_to_rib.py`) that converts the 2D density field to a 3D rib STL: elements with density > 0.5 become solid 2mm-thick material, creating a rib with optimized cutout patterns.
10. FEA verification script (CalculiX .inp file or SfePy Python script) with orthotropic material properties.
11. Stress visualization and analysis script.

**User runs:**
12. `python optimize_rib_2d.py` --> `optimized_density.npy` (runs in seconds for 2D rib domain)
13. `python density_to_rib.py` --> `optimized_rib.stl`
14. Review the optimized rib planform. The 2D TO should produce a rib with:
    - Full-width material near the pivot (where bending moment is highest)
    - Lightening holes or cutouts in the mid-span and tip regions (where moment is lower)
    - Tapered web structure that follows the stress flow paths
    - Full material around the pivot hole (preserved region)
    - Note: the rib maintains constant 2 mm thickness everywhere; the optimization varies which PORTIONS of the planform are material vs. void
15. Run FEA verification: `ccx rib_verification` or `python run_fea.py`
16. Print optimized ribs in PETG. Assemble and compare to baseline.

**Outcome:** TO-optimized rib planforms with cutout patterns, verified stress distribution, and weight savings.

### Phase 3: Aerodynamic Shape Optimization with ML (Week 5-9)

**Claude writes:**
15. Gmsh meshing script for deployed fan geometry (`mesh_fan.py`).
    - For initial ASO: model as solid curved plate (simplified surface, approach 1 from Section 2.5).
    - Define downstream analysis plane as internal boundary.
16. SU2 config files:
    - `fan_steady.cfg` (steady-state at peak velocity, incompressible solver)
    - `fan_unsteady.cfg` (unsteady pitching, compressible solver with low-Mach preconditioning)
17. Batch CFD runner with LHS sampling and multi-fidelity support (`run_cfd_batch.py`).
18. Multi-fidelity BoTorch optimization script (`optimize_shape_mfbo.py`).
19. Updated CadQuery generator with optimized parameters.

**User runs:**
20. `python mesh_fan.py` --> `fan.su2`
21. `SU2_CFD fan_steady.cfg` (baseline steady CFD, ~30-60 min)
22. `SU2_CFD fan_unsteady.cfg` (baseline unsteady CFD, ~2-8 hours, for validation)
23. `python run_cfd_batch.py` (60 steady + 10 unsteady initial LHS samples; runs over a weekend)
24. `python optimize_shape_mfbo.py` (multi-fidelity BO loop with ~50 additional evaluations mixing steady and unsteady; runs over 3-5 days)
25. Review optimized shape parameters and update geometry.

**Outcome:** Optimized fan planform (spread angle, camber, rib spacing) validated at both steady and unsteady fidelity levels.

#### 8.3.1 Post-Processing TO Results (2D SIMP Path)

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

### Phase 4: Verification, Coupling, and Validation (Week 10-12)

**Claude writes:**
25. Final FEA verification with orthotropic properties on optimized ribs under optimized aero loads.
26. Modal analysis setup (CalculiX `*FREQUENCY` step) to check resonance.
27. High-fidelity CFD verification script (resolved rib-gap geometry, approach 3 from Section 2.5).
28. Comparison plots: optimized vs. baseline.

**User runs:**
29. FEA verification: check `sigma_max < sigma_allowable` at pivot, tip deflection < 5 mm.
30. Modal analysis: first bending mode > 10 Hz (5x waving frequency).
31. High-fidelity CFD: quantify gap leakage effect on airflow performance.
32. Print optimized fan in PETG. Full physical testing protocol (Section 10.2).

**Coupling check:** If the aerodynamic loads on the TO-optimized ribs differ by >10% from the loads used during TO (Phase 2), re-run Phase 2 TO with updated loads. Claude writes the updated scripts; user re-runs.

### Phase 5: Final Production (Week 12-13)

33. Print final ribs in PETG (or PA-CF for maximum performance).
34. Assemble with high-quality pivot hardware.
35. Attach membrane with fabric glue.
36. Light sanding of rib edges for smooth membrane attachment.
37. Final testing and documentation.

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

def generate_rib(
    length=200.0,        # mm, rib length from pivot to tip
    base_width=12.0,     # mm, width at pivot end
    tip_width=6.0,       # mm, width at tip
    thickness=2.0,       # mm, rib thickness
    pivot_hole_dia=3.0,  # mm, pivot pin diameter
    pivot_offset=8.0,    # mm, distance from base end to pivot center
    pivot_clearance=0.2, # mm, clearance for pivot pin
    camber=0.0,          # mm, maximum camber (curvature) at mid-span
):
    """Generate a single fan rib as a CadQuery Workplane object."""

    # Define rib outline points (tapered trapezoid)
    half_base = base_width / 2
    half_tip = tip_width / 2

    # Create 2D profile with taper
    points = [
        (0, -half_base),          # base, bottom edge
        (length, -half_tip),      # tip, bottom edge
        (length, half_tip),       # tip, top edge
        (0, half_base),           # base, top edge
    ]

    # Create the rib body
    rib = (
        cq.Workplane("XY")
        .polyline(points)
        .close()
        .extrude(thickness)
    )

    # Add pivot hole
    hole_radius = (pivot_hole_dia + pivot_clearance) / 2
    rib = (
        rib
        .faces(">Z")
        .workplane()
        .center(pivot_offset, 0)
        .hole(pivot_hole_dia + pivot_clearance * 2)
    )

    # Add camber (optional curvature along the length)
    # For camber > 0, we would use a lofted or swept profile
    # This simple version uses a flat rib; camber is added in a more
    # advanced version using spline-based sweep

    return rib

def generate_guard_stick(
    length=200.0,
    width=18.0,
    thickness=3.0,
    pivot_hole_dia=3.0,
    pivot_offset=8.0,
):
    """Generate a guard stick (outer rib, wider and thicker)."""
    guard = (
        cq.Workplane("XY")
        .rect(length, width)
        .extrude(thickness)
        .faces(">Z")
        .workplane()
        .center(-length/2 + pivot_offset, 0)
        .hole(pivot_hole_dia + 0.4)
    )
    return guard

def main():
    parser = argparse.ArgumentParser(description="Generate folding fan ribs")
    parser.add_argument("--rib-count", type=int, default=15)
    parser.add_argument("--length", type=float, default=200.0)
    parser.add_argument("--base-width", type=float, default=12.0)
    parser.add_argument("--tip-width", type=float, default=6.0)
    parser.add_argument("--thickness", type=float, default=2.0)
    parser.add_argument("--output", type=str, default="ribs/")
    args = parser.parse_args()

    import os
    os.makedirs(args.output, exist_ok=True)

    # Generate inner ribs
    n_inner = args.rib_count - 2  # subtract 2 guard sticks
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
    main()
```

### 9.2 Rib Topology Optimization -- 2D Planform Formulation

#### Why 2D, Not 3D

The rib is 200 mm long, 6-12 mm wide, and 2 mm thick. A 3D TO formulation with 0.5 mm voxels yields only **4 voxels through the 2 mm thickness**. With 4 discrete levels, the optimizer can only choose 25%, 50%, 75%, or 100% thickness at each planform location -- this is thickness optimization with 4 levels, not topology optimization. Features like lightening holes require at least 3-4 voxels to form (one void surrounded by material on each side), which consumes the entire thickness.

By contrast, a 2D plane-stress formulation treats the rib as having constant 2 mm thickness and optimizes material distribution in the length-width plane. At 0.5 mm element size, the rib has 400 x 24 = 9,600 elements with full design freedom in both directions. The optimizer can create genuine topological features: lightening holes, truss-like structures, tapered webs that follow stress flow paths.

This is the correct formulation for thin-walled structures where through-thickness variation is not meaningful.

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
# Or: pip install fenics-dolfinx  (on supported platforms)

# FEniCS SIMP tutorial:
# https://comet-fenics.readthedocs.io/en/latest/demo/topology_optimization/simp_topology_optimization.html
```

#### Example: 2D Rib Planform TO (Claude Would Write This)

```python
"""
2D topology optimization of a fan rib planform using SIMP.
Based on DTU TopOpt Python codes (adapted for non-rectangular domain).
Claude Code writes this; user runs it.

The rib is treated as a 2D plane-stress problem with constant 2mm thickness.
TO determines where material is placed within the tapered planform envelope.
"""
import numpy as np
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# --- Problem parameters ---
L = 200.0       # mm, rib length
w_base = 12.0   # mm, width at pivot end
w_tip = 6.0     # mm, width at tip
thickness = 2.0  # mm, constant rib thickness (used for plane-stress stiffness)
volfrac = 0.40   # volume fraction target
penal = 3.0      # SIMP penalization exponent
rmin = 1.5       # filter radius in mm
E0 = 1300.0      # MPa, FDM PETG in-plane modulus
nu = 0.38        # Poisson's ratio

# --- Discretization ---
elem_size = 0.5  # mm
nelx = int(L / elem_size)         # 400 elements along length
nely_max = int(w_base / elem_size) # 24 elements max width

# Create tapered domain mask: which elements are inside the rib envelope
active = np.zeros((nely_max, nelx), dtype=bool)
for ix in range(nelx):
    x_frac = ix / nelx
    w_local = w_base - (w_base - w_tip) * x_frac
    n_active = int(w_local / elem_size)
    y_start = (nely_max - n_active) // 2
    active[y_start:y_start + n_active, ix] = True

n_active_elems = active.sum()
print(f"Active elements: {n_active_elems} (of {nelx * nely_max} total)")

# --- Preserved region: pivot base (first 16 elements, full width) ---
# Elements in the pivot region have density fixed to 1.0
pivot_length_elems = 16  # 8 mm
preserved = np.zeros_like(active, dtype=bool)
preserved[:, :pivot_length_elems] = active[:, :pivot_length_elems]

# --- Boundary conditions: fix left edge (pivot) ---
# All DOFs on the left edge of active elements are fixed.
# (Specific node numbering depends on the FE assembly; Claude handles this.)

# --- Loads: distributed pressure on one face ---
# Aerodynamic pressure: 10 Pa over tributary width ~15 mm
# Force per unit length on rib: 10 Pa * 15 mm = 0.15 N/m = 0.00015 N/mm
# Applied as nodal forces on the top edge of the 2D domain.

# --- Core SIMP optimization loop (OC update) ---
# [Claude writes the full FE assembly, sensitivity computation, filtering,
#  and optimality criteria update here -- approximately 100-150 lines.
#  The structure follows the DTU 88-line code adapted for:
#  (1) non-rectangular (tapered) domain via the 'active' mask,
#  (2) preserved regions via the 'preserved' mask,
#  (3) distributed load instead of point load.]

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
    nelx=400, nely=24, nelz=10,  # 10 voxels through 5mm at 0.5mm resolution
    volfrac=0.4,
    penal=3.0,
    rmin=3.0,
    disp_thres=0.5,
    obstacle_mask=None,  # optional: binary mask for forbidden regions
    use_gpu=False,
)
```

To use this for fan ribs, Claude would need to **fork the PyTopo3D repository** and modify `pytopo3d/core/optimizer.py` to:
1. Accept custom fixed-DOF arrays instead of hardcoded left-face fixity.
2. Accept custom force vectors instead of the hardcoded point load.
3. Add a preserved-region mask (density fixed to 1.0 in pivot area).

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
1300.0, 1300.0, 1000.0,  0.38, 0.38, 0.38,  500.0, 500.0,
385.0
*DENSITY
1.27e-9
**
*SOLID SECTION, ELSET=ALL_ELEMENTS, MATERIAL=PETG_FDM
**
** Boundary conditions: fix pivot region
*BOUNDARY
PIVOT_NODES, 1, 3, 0.0
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

#### Steady-State Configuration (Claude Writes This)

```ini
% SU2 config for deployed folding fan (steady-state proxy)
% Fan modeled as solid curved plate (simplified surface)
%
SOLVER= INC_NAVIER_STOKES
KIND_TURB_MODEL= NONE
%
% Flow conditions (fan at peak waving velocity)
INC_VELOCITY_INIT= (2.64, 0.0, 0.0)
INC_DENSITY_INIT= 1.225
INC_TEMPERATURE_INIT= 300.0
VISCOSITY_MODEL= CONSTANT_VISCOSITY
MU_CONSTANT= 1.81e-5
%
MESH_FILENAME= fan_deployed.su2
MESH_FORMAT= SU2
%
MARKER_HEATFLUX= ( fan_surface, 0.0 )
MARKER_FAR= ( farfield )
%
% Objective: total pressure at downstream plane (ROUGH PROXY ONLY -- see note below)
% For production optimization, replace with custom momentum flux post-processing.
OBJECTIVE_FUNCTION= SURFACE_TOTAL_PRESSURE
MARKER_ANALYZE= ( downstream_plane )
MARKER_MONITORING= ( fan_surface )
%
% Numerics
NUM_METHOD_GRAD= GREEN_GAUSS
CFL_NUMBER= 10.0
ITER= 5000
CONV_RESIDUAL_MINVAL= -8
%
% Output
OUTPUT_FILES= RESTART, PARAVIEW
TABULAR_OUTPUT= CSV
HISTORY_OUTPUT= ITER, RMS_RES, AERO_COEFF, SURFACE_TOTAL_PRESSURE
```

#### Objective Function for a Fan

**Critical note:** `DRAG` is the wrong objective. A fan's purpose is to push air. The correct objectives for a folding fan:

1. **Custom momentum flux via Python post-processing (recommended):** Directly compute the directed momentum flux integral from SU2 field output. This is the most physically meaningful metric and should be the primary approach, not a fallback:
   ```python
   # Post-process SU2 volume/surface output to compute momentum flux
   # through a downstream plane at distance d behind the fan.
   # J_fan = integral over plane of (rho * u_n * u_target_dir) dA
   # where u_n = velocity normal to the plane, u_target_dir = velocity
   # component in the desired airflow direction.
   # Claude writes this as a Python script using SU2's CSV or VTK output.
   ```
   Note: `SURFACE_TOTAL_PRESSURE` on a downstream plane is NOT the same as directed momentum flux. Total pressure includes both static and dynamic pressure contributions, and its surface integral does not directly give you the net momentum flux in the desired direction. Total pressure is conserved along streamlines in inviscid flow, so a total pressure integral can be misleading when comparing designs that differ in how they redirect flow.

2. **Multi-objective LIFT + DRAG** (simpler, works out-of-the-box) -- maximize the normal force component (which drives airflow) while penalizing waving effort. This is an approximation but uses SU2 built-in objectives.

3. **SURFACE_TOTAL_PRESSURE** (rough proxy only) -- acceptable for early screening but should be replaced by the momentum flux integral for final optimization.

#### Unsteady Configuration (Advanced)

For unsteady simulation with pitching motion, use the compressible solver with low-Mach preconditioning (the incompressible solver has known issues with pitching motion):

```ini
SOLVER= NAVIER_STOKES
KIND_TURB_MODEL= NONE
%
MACH_NUMBER= 0.0077
AOA= 0.0
FREESTREAM_TEMPERATURE= 300.0
FREESTREAM_PRESSURE= 101325.0
REYNOLDS_NUMBER= 30000.0
REYNOLDS_LENGTH= 0.25
%
LOW_MACH_PREC= YES
MIN_ROE_TURKEL_PREC= 0.01
MAX_ROE_TURKEL_PREC= 0.2
%
TIME_DOMAIN= YES
TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER
TIME_STEP= 0.001
MAX_TIME= 1.0
TIME_ITER= 1100
INNER_ITER= 100
%
GRID_MOVEMENT= RIGID_MOTION
MOTION_ORIGIN= 0.0 0.0 0.0
PITCHING_OMEGA= 0.0 0.0 6.2832
PITCHING_AMPL= 0.0 0.0 40.0
```

**Steady-state proxy limitations:** The reduced frequency k ~ 0.6 places this firmly in the unsteady regime. At this k, added mass effects, dynamic stall, and leading-edge vortex formation significantly alter force histories compared to quasi-steady predictions. Published studies on oscillating flat plates show quasi-steady models can overpredict peak forces by 30-50% and, critically, change the **relative ranking** of different geometries. **Do not run the entire BO budget on steady-state alone.** Use the multi-fidelity BO approach described in Section 6.2.3, which mixes cheap steady-state evaluations with expensive unsteady ones via `SingleTaskMultiFidelityGP`. This lets the GP learn the steady-to-unsteady correlation and allocate budget intelligently. As a minimum, validate the final top-3 designs through unsteady CFD before committing to printing.

### 9.5 BoTorch -- Bayesian Optimization

#### Installation

```bash
pip install botorch gpytorch torch scipy matplotlib
```

Claude writes the complete multi-fidelity BO loop as described in Section 6.2.3. Key parameters:

- `num_restarts`: 20 (for acquisition function optimization)
- `raw_samples`: 512 (initial candidates for acquisition optimization)
- Initial samples: 60 steady-state + 10 unsteady (seed the multi-fidelity GP with both fidelity levels)
- Total budget: 120-160 evaluations (mixed steady and unsteady, allocated by the multi-fidelity knowledge gradient)
- Surrogate: `SingleTaskMultiFidelityGP` with fidelity column indicating steady (0) vs. unsteady (1)
- Acquisition: `qMultiFidelityKnowledgeGradient` with cost-aware utility (unsteady ~10x cost of steady)
- Surrogate validation: NRMSE < 15% on leave-one-out cross-validation at target fidelity

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

Budget 1-2 weeks for mesh generation even with Claude writing the scripts, because mesh quality tuning requires iteration.

### 9.7 DL4TO -- ML-Accelerated TO (Optional Extension)

```bash
pip install git+https://github.com/dl4to/dl4to
pip install torch pyvista==0.38.1
```

For rapid exploration of many rib variants (different cross-sections, load cases), DL4TO can train a U-Net to predict near-optimal rib topologies from load specifications. This is only worth the setup effort if you plan to iterate on rib design extensively (>20 variants).

---

## 10. Validation Approaches

### 10.1 Computational Validation

#### FEA Stress Verification (Per Rib)

Claude writes the verification script. Key checks:

1. Peak von Mises stress at pivot hole < yield / (SF * K_t) where K_t ~ 2.65 and SF = 1.5.
   - For PETG (yield ~45 MPa): allowable = 45 / (1.5 * 2.65) = 11.3 MPa nominal stress at pivot.
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
- **Assembly test:** Open/close the fan 100 times. Check for rib binding, washer wear, fabric delamination.

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

## Appendix A: Quick-Start Decision Flowchart

```
START: Do you want a fully scripted workflow (Claude Code writes the scripts)?
|
|-- YES (recommended):
|   |
|   |-- CadQuery (geometry) + 2D SIMP (TO) + SU2 (CFD/ASO) + BoTorch (multi-fidelity ML)
|   |   + Gmsh (meshing) + CalculiX (FEA verification)
|   |
|   |   Claude writes: ~20 Python scripts + ~10 config files
|   |   User runs:     scripts from terminal, prints and tests physically
|   |   Timeline:      2-4 months
|   |   Cost:          $0 (all free/open-source)
|   |
|-- NO (prefer GUI tools):
    |
    |-- FreeCAD (geometry + FEA) + BESO (TO) + SU2 (ASO) + BoTorch (ML)
    |   Timeline: 3-5 months (more manual interaction)
    |   Cost: $0
    |
    |-- Note: Fusion 360 is not recommended due to steep GUI learning curve
    |   and inability for Claude to assist with GUI operations.
```

## Appendix B: Estimated Computation Times

| Task | Hardware | Estimated Time | Notes |
|------|----------|---------------|-------|
| CadQuery rib generation (15 ribs) | Any laptop | <5 seconds | Parametric, instant |
| 2D SIMP single rib (9.6K elements) | Any laptop | 5-60 seconds | 2D planform; very fast |
| 2D SIMP single rib (38K elements, finer) | Any laptop | 1-5 minutes | Higher resolution planform |
| BESO/CalculiX single rib (50K elements) | Desktop, 8 cores | 30-120 minutes | Unstructured mesh |
| CalculiX FEA verification | Any laptop | 1-5 minutes | Single static analysis |
| CalculiX modal analysis (10 modes) | Any laptop | 2-10 minutes | |
| Gmsh meshing (2D fan profile) | Any laptop | <1 minute | |
| Gmsh meshing (3D deployed fan) | Desktop | 5-30 minutes | Complex boundary layer |
| SU2 CFD steady (50K cells, 2D) | Any laptop | 2-10 minutes | |
| SU2 CFD steady (500K cells, 3D) | Desktop, 8 cores | 30-120 minutes | |
| SU2 CFD unsteady (5 cycles, 3D) | Desktop, 8 cores | 2-8 hours | |
| GP surrogate training (100 samples) | Any laptop | 1-10 seconds | |
| BO iteration (acquisition opt) | Any laptop | 0.5-5 seconds | |
| Multi-fidelity BO loop (~50 iter, mixed fidelity) | Desktop, 8 cores | 3-7 days | ~70% steady + ~30% unsteady evaluations |
| LHS batch (60 steady + 10 unsteady initial) | Desktop, 8 cores | 2-5 days | Parallelizable; unsteady runs dominate |

## Appendix C: Folding Fan vs. Paddle Fan -- Key Differences for Optimization

This appendix summarizes why the original paddle fan approach does not apply to the folding fan and what changes in the optimization formulation.

| Optimization Aspect | Paddle Fan (original report) | Folding Fan (this revision) |
|---------------------|------------------------------|----------------------------|
| TO design domain | Entire blade (single large 2D/3D domain) | Individual ribs (small 3D beam domains, one per rib) |
| TO problem size | Large (500K+ elements for full blade) | Small (30K-150K voxels per rib) |
| TO compute time | 2-4 hours per run | 5-120 minutes per rib |
| TO focus | Internal lattice structure | 2D planform cutout patterns, lightening holes, web distribution (constant thickness) |
| ASO geometry | Continuous solid surface | Sector of ribs + membrane + gaps |
| ASO parameters | Planform, camber, thickness distribution | Spread angle, rib count, rib camber, rib spacing |
| CFD model | Standard bluff body | Simplified surface OR resolved rib-gap geometry |
| Structural failure mode | Distributed bending of plate | Concentrated stress at pivot hole + rib bending |
| Critical stress location | Blade root | Pivot hole (K_t ~ 2.65) |
| Assembly | None (monolithic print) | Ribs + pivot pin + washers + membrane |
| Print strategy | Single large flat print | Multiple small flat prints (one per rib) |
