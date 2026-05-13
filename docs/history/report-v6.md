# Topology Optimization and Aerodynamic Shape Optimization for a 3D-Printed Folding Fan

## Comprehensive Design, Optimization, and Fabrication Guide

**Date:** 2026-03-22 (Revised)
**Revision:** R5 -- Fundamental restructuring: folding fan (sensu/ogi) as sole design target, easiest-path tooling with scripted Python workflows, Claude Code delegation analysis
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

**Topology Optimization (TO):** For each individual rib, determine the optimal cross-sectional shape and material distribution along its length. The rib is a thin structural beam (typically 1.5-3 mm thick, 5-15 mm wide, 150-250 mm long). TO optimizes where to place material within the rib's envelope to maximize stiffness-to-weight ratio under the combined loading of aerodynamic pressure (transmitted through the membrane), inertial forces from waving, and the reaction force at the pivot.

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

**Cross-section envelope (for TO):**
```
Width:  w(x) = w_base - (w_base - w_tip) * (x / L)
        w_base = 12 mm, w_tip = 6 mm, L = 200 mm

Height: h = 2.0 mm (uniform, constrained by FDM layer count)

The TO domain at each cross-section is a rectangle w(x) x h.
Along the length, TO determines material distribution within this envelope.
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

**Combined loading for TO:** The rib TO problem is a cantilever beam fixed at the pivot, with distributed transverse load (aerodynamic + inertial), seeking minimum compliance (maximum stiffness) subject to a volume fraction constraint. This is a classic TO problem well-suited to SIMP.

### 2.5 Deployed Fan Geometry for CFD

When deployed at 120 degrees, the 15 ribs form a sector with:
- Arc length at tip: 2 * pi * 200 mm * (120/360) = 419 mm
- Gap between rib tips: (419 - 15 * 6) / 14 = 23.5 mm (rib tip width = 6 mm)
- Gap between rib bases: much smaller (~2-3 mm)
- The membrane spans these gaps, creating a quasi-continuous surface with rib ridges.

**CFD modeling approaches for the rib-membrane geometry:**

1. **Simplified solid surface (recommended for initial ASO):** Model the deployed fan as a solid curved plate (sector shape) with the rib positions marked as thickened ridges. This ignores air leakage through gaps but captures the overall aerodynamic shape. Justification: with a taut membrane, leakage is minimal.

2. **Porous surface model (intermediate):** Model the fan surface as a porous zone with direction-dependent permeability -- low permeability where membrane is present, high where gaps exist. OpenFOAM supports porous media natively; SU2 can approximate this with source terms.

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
| **1** | **CadQuery + PyTopo3D + SU2 + BoTorch** | Free | Medium | **Low (scripted)** | **Full** | **Yes (all of it)** | Entire pipeline is Python scripts + config files. No GUI needed. |
| **2** | **CadQuery + BESO/CalculiX + SU2 + BoTorch** | Free | Medium | Medium | High | Yes (most) | BESO can be run headless via config files. CalculiX is CLI. |
| **3** | **FreeCAD (scripted) + BESO + SU2 + BoTorch** | Free | Medium-High | Medium | Medium | Partially | FreeCAD has Python API but some GUI is hard to avoid. |
| **4** | **Fusion 360 + SU2 + BoTorch** | Free (edu) | **High (steep GUI learning curve)** | High (manual) | Low | No (Fusion is GUI-only) | User rejected this path. |

### 4.3 Recommended Stack: The Fully Scripted Pipeline

**Primary recommendation: CadQuery (geometry) + PyTopo3D (TO) + SU2 (CFD/ASO) + BoTorch (ML optimization)**

This stack is chosen because:

1. **CadQuery** -- Pure Python parametric CAD. Claude writes CadQuery scripts to generate rib geometry with any parameterization. Exports directly to STL/STEP. No GUI required. `pip install cadquery`.

2. **PyTopo3D** -- Pure Python 3D SIMP topology optimization. `pip install pytopo3d`. Claude writes the problem setup (design domain, loads, BCs, volume fraction). Reads STL for design domain import. Exports optimized geometry as STL. Command-line and Python API. Released 2025, actively maintained.

3. **SU2** -- Command-line CFD solver with built-in adjoint ASO. Config files are text-based -- Claude writes them. `conda install -c conda-forge su2` or build from source.

4. **BoTorch/GPyTorch** -- Python Bayesian optimization. Claude writes the entire optimization loop. `pip install botorch gpytorch`.

5. **Gmsh** -- Scriptable mesh generation. Python API available. Claude writes meshing scripts. `pip install gmsh`.

6. **CalculiX** (optional, for FEA verification) -- CLI-based FEA solver. Input files are text-based -- Claude writes them. Python wrappers available (pyccx, pycalculix).

### 4.4 Tool Details

#### CadQuery (Parametric Geometry Generation)

- **What it does:** Python library for creating parametric 3D CAD models. Based on OpenCASCADE (same kernel as FreeCAD).
- **Why it fits:** Fan ribs are simple extruded shapes with holes -- ideal for CadQuery. The entire rib parameterization (length, taper, thickness, pivot hole, cross-section profile) can be expressed in ~50 lines of Python.
- **Installation:** `pip install cadquery` (or `conda install -c conda-forge cadquery`)
- **Export:** STL, STEP, AMF, SVG
- **Claude can:** Write the complete parametric rib generator, modify parameters, generate variants for optimization studies.
- **Limitation:** No built-in visualization in headless mode. Use `cadquery-ocp` or export and view in a separate tool.

#### PyTopo3D (Topology Optimization)

- **What it does:** 3D SIMP topology optimization on structured grids with STL domain import, direct STL export.
- **Why it fits:** Can import rib geometry as STL, define loads and BCs, run TO, and export the optimized rib directly. Pure Python with CLI and API.
- **Installation:** `pip install pytopo3d` (requires Python 3.10+)
- **Key features:** AM constraints (overhang angle), STL import for design domain, accelerated KD-Tree filtering, 3D visualization, direct STL export of optimized geometry.
- **Claude can:** Write the problem definition, set up loads from CFD results, configure volume fraction and constraints, run the optimization, post-process results.
- **Limitation:** Structured voxel grid (not unstructured mesh). For thin ribs, the voxel resolution must be fine enough to resolve the 2 mm thickness -- may need 0.25-0.5 mm voxel size.

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

### 4.5 Head-to-Head: PyTopo3D vs. FreeCAD/BESO vs. Fusion 360

| Criterion | PyTopo3D | FreeCAD/BESO/CalculiX | Fusion 360 (Educational) |
|-----------|----------|----------------------|--------------------------|
| **Cost** | Free | Free | Free (with .edu) |
| **GUI required?** | No (CLI + Python API) | Partially (FreeCAD GUI for setup) | Yes (entirely GUI-based) |
| **Claude can write it?** | Yes -- 100% scriptable | Yes -- mostly (BESO config + CalculiX .inp) | No |
| **Install effort** | `pip install pytopo3d` | Multiple components | Register + download |
| **TO algorithm** | SIMP with OC update | BESO (evolutionary) | Proprietary cloud solver |
| **Mesh type** | Structured voxel grid | Unstructured tetrahedral | Proprietary |
| **STL import** | Yes (design domain) | Yes (via FreeCAD) | Yes |
| **STL export** | Yes (direct) | Requires post-processing pipeline | Yes (direct, clean geometry) |
| **AM constraints** | Yes (overhang angle) | Partial (manual) | Yes (full AM) |
| **Thin rib handling** | Needs fine voxel grid | Good (tets conform to thin geometry) | Good |
| **Post-processing** | Minimal (direct STL export) | Extensive (Section 8.3.1) | None needed |
| **Best for** | Scripted workflow with Claude | Higher mesh fidelity on thin parts | Users comfortable with GUI |

**Verdict:** PyTopo3D is the easiest path for a Claude Code workflow. It eliminates the two biggest pain points: (1) no GUI interaction needed, and (2) direct STL export without the density-to-mesh post-processing pipeline that makes BESO so painful. The voxel resolution limitation is manageable for fan ribs if the grid is fine enough.

For users who need higher fidelity on the thin rib cross-section, the BESO/CalculiX path is still viable and also mostly scriptable. Both paths converge on SU2 + BoTorch for ASO and ML.

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
| **TO problem setup** | PyTopo3D | Python script defining domain, loads, BCs, constraints | ~50-80 | Load values from CFD results |
| **TO batch runner** | PyTopo3D + shell | Script running TO for multiple parameter sets | ~40 | For design space exploration |
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

Phase 2: Structural TO of Ribs
  Claude writes: PyTopo3D setup script with loads from Phase 1 estimates
  User runs:     python optimize_rib.py --> optimized_rib.stl
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
| Rib count | 1 | 10-25 | Number of ribs (discrete, handled via rounding) |
| Rib curvature profile | 3-4 | 0-10 mm | Camber of individual ribs (defines fan surface curvature) |
| Rib length | 1 | 150-250 mm | Overall fan size |
| Rib taper ratio | 1 | 0.3-0.8 | Tip width / base width |
| Membrane tension | 1 | 0-5 N/m | Affects billowing and effective camber |
| Edge profile | 1-2 | 0.5-2 mm | Rib cross-section shape at leading/trailing edge |

**Total parameters:** 9-11. Manageable for GP-based Bayesian optimization without dimensionality reduction.

#### 6.2.2 Recommended Model: GP with BoTorch

```python
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, ScaleKernel

# Matern 5/2 with ARD (learns per-dimension importance)
# SingleTaskGP uses this by default
gp_model = SingleTaskGP(train_X, train_Y)
```

**Why GP (not neural network):** With a realistic budget of 100-200 CFD simulations, GPs are the correct choice. They provide calibrated uncertainty estimates, perform well with small datasets, and support Bayesian optimization natively.

#### 6.2.3 Bayesian Optimization Loop

Claude writes this entire loop as a Python script:

```python
# Claude writes this script; user runs it
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc
import subprocess, json

N_DIMS = 10          # Fan design parameters
N_INITIAL = 80       # Initial LHS samples (8x dimensionality)
N_BO_ITERS = 60      # Bayesian optimization iterations

bounds = torch.tensor([[...], [...]], dtype=torch.double)  # Parameter bounds

def run_cfd(params):
    """Run SU2 CFD and return fan performance. Claude writes this wrapper."""
    # 1. Generate fan geometry from params via CadQuery
    # 2. Mesh with Gmsh
    # 3. Run SU2_CFD
    # 4. Parse results
    # 5. Return J_fan metric
    ...

# Generate initial training data
sampler = qmc.LatinHypercube(d=N_DIMS)
X_init = torch.tensor(sampler.random(n=N_INITIAL), dtype=torch.double)
X_init = bounds[0] + (bounds[1] - bounds[0]) * X_init
Y_init = torch.tensor([run_cfd(x) for x in X_init], dtype=torch.double).unsqueeze(-1)

# Bayesian optimization loop
X_train, Y_train = X_init, Y_init
for iteration in range(N_BO_ITERS):
    gp = SingleTaskGP(X_train, Y_train)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    EI = ExpectedImprovement(model=gp, best_f=Y_train.max())
    candidate, acq_value = optimize_acqf(
        EI, bounds=bounds, q=1, num_restarts=20, raw_samples=512
    )

    new_y = run_cfd(candidate.squeeze())
    X_train = torch.cat([X_train, candidate])
    Y_train = torch.cat([Y_train, torch.tensor([[new_y]], dtype=torch.double)])

    print(f"Iter {iteration}: best = {Y_train.max():.4f}")
    if acq_value.item() < 1e-6:
        break

best_idx = Y_train.argmax()
best_params = X_train[best_idx]
```

### 6.3 ML for TO (Optional Extension)

For rib TO, the problem is small enough (a single thin beam) that ML acceleration is unnecessary for a first project. A single PyTopo3D run on a rib discretized at 0.5 mm voxels (400 x 24 x 4 = ~38K voxels) converges in minutes.

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

### Phase 2: Rib Topology Optimization (Week 3-4)

**Claude writes:**
8. PyTopo3D optimization script (`optimize_rib.py`) using estimated loads from Phase 1.
   - Design domain: rib envelope (tapered beam, 200 x 12 x 2 mm)
   - Preserved region: pivot base (10 x 12 x 2 mm with 3 mm hole)
   - Loads: distributed pressure (from estimated aero loads) + inertial
   - Volume fraction: 0.4 (40% material)
   - AM constraints: overhang angle 45 degrees
9. FEA verification script (CalculiX .inp file or SfePy Python script) with orthotropic material properties.
10. Stress visualization and analysis script.

**User runs:**
11. `python optimize_rib.py` --> `optimized_rib.stl` (runs in minutes for rib-sized domain)
12. Review the optimized rib cross-section. The TO should produce a rib with:
    - Thicker web near the pivot (where bending moment is highest)
    - Potential lightening holes or thinned sections toward the tip
    - Full material around the pivot hole (preserved region)
13. Run FEA verification: `ccx rib_verification` or `python run_fea.py`
14. Print optimized ribs in PETG. Assemble and compare to baseline.

**Outcome:** TO-optimized ribs with verified stress distribution and weight savings.

### Phase 3: Aerodynamic Shape Optimization with ML (Week 5-9)

**Claude writes:**
15. Gmsh meshing script for deployed fan geometry (`mesh_fan.py`).
    - For initial ASO: model as solid curved plate (simplified surface, approach 1 from Section 2.5).
    - Define downstream analysis plane as internal boundary.
16. SU2 config files:
    - `fan_steady.cfg` (steady-state at peak velocity, incompressible solver)
    - `fan_unsteady.cfg` (unsteady pitching, compressible solver with low-Mach preconditioning)
17. Batch CFD runner with LHS sampling (`run_cfd_batch.py`).
18. BoTorch optimization script (`optimize_shape.py`).
19. Updated CadQuery generator with optimized parameters.

**User runs:**
20. `python mesh_fan.py` --> `fan.su2`
21. `SU2_CFD fan_steady.cfg` (baseline CFD, ~30-60 min)
22. `python run_cfd_batch.py` (80 initial LHS samples, runs overnight or over a weekend)
23. `python optimize_shape.py` (BO loop with 60 CFD evaluations, runs over 2-4 days)
24. Review optimized shape parameters and update geometry.

**Outcome:** Optimized fan planform (spread angle, camber, rib count).

#### 8.3.1 Post-Processing TO Results (PyTopo3D Path)

**PyTopo3D produces direct STL output**, largely eliminating the painful post-processing pipeline that plagues the BESO path. However, the voxelized output may have staircase artifacts that require light smoothing:

1. **Export STL** from PyTopo3D (built-in).
2. **Light smoothing** (optional): Claude writes a PyMeshLab script for Taubin smoothing (10-20 iterations, lambda=0.5, mu=-0.53).
3. **Verify** minimum feature sizes meet printability requirements.

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
pip install cadquery
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

### 9.2 PyTopo3D -- Rib Topology Optimization

#### Installation

```bash
pip install pytopo3d
# Requires Python 3.10+
```

#### Example: Rib TO Setup (Claude Would Write This)

```python
"""
Topology optimization of a single fan rib using PyTopo3D.
Claude Code writes this; user runs it.
"""
from pytopo3d import TopOpt3D

# Rib dimensions discretized as voxel grid
# Rib: 200 mm x 12 mm x 2 mm
# Voxel size: 0.5 mm -> 400 x 24 x 4 voxels
nelx, nely, nelz = 400, 24, 4

# Initialize optimizer
opt = TopOpt3D(
    nelx=nelx, nely=nely, nelz=nelz,
    volfrac=0.40,          # 40% material
    penal=3.0,             # SIMP penalization
    rmin=3.0,              # filter radius (1.5 mm = 3 voxels)
)

# Define boundary conditions: fix the pivot region (x = 0 to 16 voxels)
# In PyTopo3D, fixed nodes are specified as coordinate ranges
# The pivot region is preserved (not optimized)
opt.set_fixed_region(x_range=(0, 16), y_range=(0, 24), z_range=(0, 4))

# Define loads: distributed pressure on top face
# Approximate aerodynamic + inertial load: 10 Pa over tributary width 15 mm
# Force per voxel on top face: 10 Pa * 0.5mm * 0.5mm = 0.0025 N
opt.set_distributed_load(
    face="top",
    magnitude=0.0025,  # N per surface voxel
    direction=(0, 0, -1),  # downward (normal to rib face)
)

# Run optimization
result = opt.optimize(max_iter=200, tol=0.001)

# Export optimized rib as STL
opt.export_stl("optimized_rib.stl")
print(f"Final compliance: {result.compliance:.4f}")
print(f"Final volume fraction: {result.volume_fraction:.4f}")
```

**Note:** The exact PyTopo3D API may differ from the above -- Claude would consult the current documentation and adjust. The key point is that the entire setup is a Python script that Claude writes and the user runs.

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
% Objective: maximize total pressure at downstream plane
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

1. **SURFACE_TOTAL_PRESSURE on a downstream plane** (recommended for steady-state proxy) -- measures momentum flux through a plane behind the fan.

2. **Multi-objective LIFT + DRAG** (simpler, works out-of-the-box) -- maximize the normal force component (which drives airflow) while penalizing waving effort.

3. **Custom momentum flux via Python wrapper** (most accurate) -- directly compute the integral of rho * u_x * (u dot n) over a downstream surface.

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

**Steady-state proxy limitations:** The reduced frequency k ~ 0.6 places this firmly in the unsteady regime. The steady-state proxy may not preserve design rankings. Mandatory validation: after steady-state ASO, run 3 designs (baseline, best, intermediate) through unsteady CFD to verify the ranking holds.

### 9.5 BoTorch -- Bayesian Optimization

#### Installation

```bash
pip install botorch gpytorch torch scipy matplotlib
```

Claude writes the complete BO loop as described in Section 6.2.3. Key parameters:

- `num_restarts`: 20 (for acquisition function optimization)
- `raw_samples`: 512 (initial candidates for acquisition optimization)
- Initial samples: 80-100 (8-10x the number of design variables)
- Total budget: 140-160 evaluations (80 initial + 60 BO iterations)
- Surrogate validation: NRMSE < 15% on leave-one-out cross-validation

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

---

## Appendix A: Quick-Start Decision Flowchart

```
START: Do you want a fully scripted workflow (Claude Code writes the scripts)?
|
|-- YES (recommended):
|   |
|   |-- CadQuery (geometry) + PyTopo3D (TO) + SU2 (CFD/ASO) + BoTorch (ML)
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
| PyTopo3D single rib (38K voxels) | Any laptop | 5-30 minutes | Small domain; fast |
| PyTopo3D single rib (150K voxels, finer) | Desktop | 30-120 minutes | For higher resolution |
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
| Full BO loop (60 iter with steady CFD) | Desktop, 8 cores | 1-4 days | Dominated by CFD |
| LHS batch (80 steady CFD runs) | Desktop, 8 cores | 1-3 days | Parallelizable |

## Appendix C: Folding Fan vs. Paddle Fan -- Key Differences for Optimization

This appendix summarizes why the original paddle fan approach does not apply to the folding fan and what changes in the optimization formulation.

| Optimization Aspect | Paddle Fan (original report) | Folding Fan (this revision) |
|---------------------|------------------------------|----------------------------|
| TO design domain | Entire blade (single large 2D/3D domain) | Individual ribs (small 3D beam domains, one per rib) |
| TO problem size | Large (500K+ elements for full blade) | Small (30K-150K voxels per rib) |
| TO compute time | 2-4 hours per run | 5-120 minutes per rib |
| TO focus | Internal lattice structure | Cross-section profile, lightening features, web distribution |
| ASO geometry | Continuous solid surface | Sector of ribs + membrane + gaps |
| ASO parameters | Planform, camber, thickness distribution | Spread angle, rib count, rib camber, rib spacing |
| CFD model | Standard bluff body | Simplified surface OR resolved rib-gap geometry |
| Structural failure mode | Distributed bending of plate | Concentrated stress at pivot hole + rib bending |
| Critical stress location | Blade root | Pivot hole (K_t ~ 2.65) |
| Assembly | None (monolithic print) | Ribs + pivot pin + washers + membrane |
| Print strategy | Single large flat print | Multiple small flat prints (one per rib) |
