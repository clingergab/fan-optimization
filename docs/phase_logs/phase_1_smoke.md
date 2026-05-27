# Phase 1 Smoke — CadQuery generator pipeline shakedown

**Date:** 2026-05-21
**Branch:** main (commits 02f16db → 63b613a)
**Scope:** Exercise the Phase 1 CadQuery generator end-to-end on two
representative designs and document anything important that surfaces.

## What ran

| Script | Input | Wall-time (M3) |
|---|---|---|
| `scripts/smoke_test.py` | `data/examples/baseline.json` | ~8 s |
| `scripts/smoke_test.py` | `data/examples/features_light.json` | ~7 s |
| `scripts/smoke_test.py` | `data/examples/features.json` (TPMS + 4 fields + L3) | **killed at 15 min** |
| `scripts/print_strategy.py` | both designs | ~6 s each |
| `scripts/fan_addin.py` | `data/examples/baseline.json` | ~12 s (10 STLs + 1 STEP) |
| `scripts/fan_addin.py` | `data/examples/features_light.json` | ~6 s (10 STLs + 1 STEP) |
| Full pytest | 881 tests, 3 pre-existing skips, 0 failures | 5 min 10 s |

All commands run on MacBook Pro M3 with CadQuery 2.7.0 + OpenCascade
backing OCP. No Colab needed. No Fusion 360 needed.

## Baseline design (`data/examples/baseline.json`)

The thinnest schema-legal panel (2.2 mm uniform thickness), no camber,
no twist, no Fourier modulation, no Layer 2/3 features. The closest
design the locked architecture allows to the C9 100 g mass cap.

| Field | Value |
|---|---|
| Status | `ok` |
| Per-blade mass | **15.30 g** |
| Total mass (10 blades) | **152.97 g** |
| `total_mass_under_cap` | **False** (cap = 100 g) |
| Centre of mass (m) | `(0.128, ~0, 0.0011)` |
| `I_wrist_kgm2` | **5.23 × 10⁻⁴** |
| Manufacturability score | 0.90 (passed) |
| Pending CadQuery checks | `#1, #8` (per-feature tracking; Phase-2) |

### Print-strategy decision

| Dim | Fan | Bed (default) | Fits? |
|---|---|---|---|
| x | 203 mm | 256 mm | ✅ |
| y | 369 mm | 256 mm | ❌ |

→ **`decision = "per-blade"`** for the default 256 × 256 mm bed (e.g.,
Bambu Lab X1 / Prusa MK4). The deployed fan spans ~370 mm tangentially
at the tip (10 blades × 13.3° = 133° arc × 200 mm radius × 2 sin(66.5°)),
which exceeds the standard print bed. **Per-blade printing is the V1
default;** the slicer accepts the per-blade STLs separately.

### STLs + STEP

`data/exports/baseline/` contains:

* `blade_0.stl` … `blade_9.stl` — 10 per-blade meshes, ~4.3 KB each.
  The small size reflects the simple flat-panel geometry (no Layer 2/3
  cuts to mesh).
* `deployed_fan.step` — 1.2 MB STEP of the full deployed assembly.

These STLs ARE printable by any FDM slicer that ingests STL +
manufacturability check #14 (support-scar) passed under flat
orientation, so no support material is required.

## Light-features design (`data/examples/features_light.json`)

To exercise the full Phase-1 stack while staying inside an interactive
wall-time budget: thicker panel (3.0 / 2.8 / 2.6 mm thickness profile)
+ camber + Fourier LE modulation + Layer 2 louver (6 slots, 1 mm wide)
+ Layer 3 ellipsoid (5 × 3 × 0.8 mm subtract). No TPMS, no edge, no
texture.

| Field | Value |
|---|---|
| Wall-time (smoke) | **~7 s** |
| Wall-time (fan_addin export) | **~6 s** |
| Per-blade mass | **21.68 g** (vs baseline 15.30 g — thicker panel + camber outweigh the louver/L3 cuts) |
| Total mass (10 blades) | **216.8 g** (over C9) |
| `I_wrist_kgm2` | **7.46 × 10⁻⁴** |
| Manufacturability | 0.90 (passed; same #4 SOFT failure as baseline) |
| STL size per blade | **~68 KB** (vs baseline 4.3 KB — Layer 2/3 cuts mesh more detail) |
| STEP size | **21 MB** (vs baseline 1.2 MB) |
| Print decision | per-blade (same 369 mm y-extent) |

The exported STLs land in `data/exports/features_light/`. Bumped STL
size (~15×) is the visible signature of the Layer 2 louver cuts +
Layer 3 ellipsoid in the mesh.

## Important findings

### 1. The locked architecture is tight against C9

The schema-thinnest design comes in at **~153 g total**, **over the
100 g C9 mass cap by ~53 %**. Implications:

* The C9 cap is achievable only by aggressive Layer 2/3 carving (TPMS
  porosity + louver cuts + Layer 3 ellipsoid). The feature-rich design
  in `data/examples/features.json` exercises this; results below.
* If the baseline thinnest-panel design can't pass C9 even with full
  Layer-2 carving, the Phase 4 BO will need to flag C9 as a constraint
  that may not be jointly satisfiable with J_fan maximisation. Worth
  surfacing to the operator before Phase 4 launches.
* The `total_mass_under_cap` flag in `smoke_test.py` output is the
  per-design indicator; downstream consumers (BO ledger, manufacturability
  score) should treat C9 as a soft penalty rather than a hard reject if
  the design space is genuinely empty.

### 2. The Phase-1 V-unit double-counts shared ribs

`make_vunit_blade` builds a panel + 2 ribs (one each at the LE and TE
tangential edges). In the deployed fan, **adjacent V-units share a
rib**: the TE rib of blade `i` is the LE rib of blade `i+1`. The V1
mass aggregation (`compute_mass_kg(blade) × blade_count`) therefore
**double-counts every interior rib**.

* For 10 blades there are 11 ribs in the assembled fan (or 9 if both
  end edges are open). My current accounting sums `10 × (panel +
  2 ribs)` = 10 panels + 20 ribs.
* Each rib ≈ 1.65 cm³ × 1.27 g/cm³ ≈ **2.1 g**. Double-counting 9 ribs
  adds ~19 g of phantom mass. Subtract that from baseline's 153 g →
  real mass ~134 g, still over C9 but closer.
* **Phase-2 follow-up:** rework `deploy_fan` to deduplicate shared ribs,
  OR have `make_vunit_blade` emit a single-rib variant that the
  assembly stitches together properly. The single-blade STLs are
  printable as-is (each blade has its own 2 ribs); only the deployed
  fan mass accounting is affected.

### 3. The fan doesn't fit a 256 × 256 mm bed

The 370 mm tangential extent of a 10-blade fan means **per-blade
printing is the V1 default** for standard hobbyist beds (256 × 256 mm
Bambu Lab X1, 220 × 220 mm Prusa MK4, etc.). Implications for Phase 6:

* The blinded A/B feel-test fan is assembled from 10 separately-printed
  blades via the pivot pin. No "print the whole fan flat" alternative.
* Print-time per fan ≈ 10 × (single-blade print time). Bambu X1 prints
  one PETG blade in ~30 min → ~5 hours per fan.
* If Phase 6 operator wants a full-bed-assembly print, the deployed
  fan would need a smaller `blade_count` (8 blades at 13.3° = 106° →
  y-extent ~320 mm, still over) OR a non-default bed (e.g., Voron 2.4
  with a 350 mm² bed fits).

### 4. Per-feature CadQuery checks (#1, #8) remain PENDING in V1

Manufacturability checks #1 (min feature size) and #8 (aspect ratio)
need per-feature tracking that's deferred to Phase 2. The current
shape-level proxies fire false positives on loft tessellation edges
(sub-mm sampling, not real features) or on the whole-blade aspect
ratio (~70:1 by design). **Both are documented in the per-check
docstring and surface in the smoke summary's `pending_cadquery` list.**

The other 8 geometry-level checks (2, 3, 4, 5, 6, 12, 13, 14) all run
real shape inspection. The baseline passes all of them; manufacturability
score 0.90 (start at 1.0, the #4 "bridging" SOFT check fires on the
panel bottom edges and subtracts 0.1 — Phase-2 refinement needed).

### 5. Layer 2 field application is slow when 3+ fields stack

The full features design (`data/examples/features.json`: TPMS + louver +
edge serration + Layer 3 ellipsoid) **did not complete in 15 minutes**
of 100 % CPU on M3 and was killed. The light features design (louver +
L3 ellipsoid only) ran in **7 s**. Diagnosis:

* Each Phase-1 field function (TPMS, noise, texture, louver, edge)
  builds a list of N cutters and iteratively `.fuse()`s them together
  in a Python loop. That's O(n²) in Boolean operations.
* For TPMS at `cell_size_m = 5 mm` on the ~165 mm × 0–22 mm carve band,
  N ≈ 200–500 cutters. Each fuse takes ~20 ms after the union grows
  past ~50 cutters; total per-field cost grows quadratically.
* Stacking 3+ active fields multiplies the cost. The light design (1
  active field + L3) is comfortably interactive; the 3-field design
  isn't.

This is a **pure Phase-1 implementation choice** — not a CadQuery limit
or a hardware limit. **Colab CPU would run this at the same speed as
local M3** (similar single-core Boolean throughput). The fix is to
refactor each field function to:

* Build cutters as a single `cq.Compound.makeCompound([s1, s2, ...])`
  in O(n), then `.cut()` once.
* Or use CadQuery's `Sketch` with multiple primitives + single extrude.
* Or pass the list to `Solid.fuse([s1, s2, ...])` directly — the
  underlying OCP method uses balanced-binary-tree merging.

**Phase-2 optimization.** Until then, V1 Phase 4 BO should bias toward
designs with ≤ 2 simultaneously-active fields (the H1 lock already
forbids noise + TPMS co-activation; combined with the ≤ 3 cardinality,
the practical worst case is TPMS + louver + edge).

### 6. Phase-1 placeholders working as advertised

The TPMS placeholder produces a regular grid of cylindrical through-
holes at the cell pitch (NOT a true gyroid). The noise placeholder
uses a sum-of-sines instead of Perlin. Both **honor their schema
parameters** (cell_size_m, thickness_gradient, threshold_retention,
etc.) and produce observable volume reductions in tests. Phase-2
refinement replaces them with marching-cubes / Perlin.

The Layer 3 primitive's slot construction uses `slot2D` + extrude
instead of box-with-fillet — the latter fails OpenCascade's BRep API
when the fillet radius equals half the side length. Pinned in the test
suite.

## What is NOT in this smoke run

* SU2 CFD on the meshed blade — needs the Phase-2 `mesh_2d_slice.py` +
  `j_fan.py` modules (currently Phase-1 stubs per the checklist) and
  SU2 install. Tier -1 / 0 / 1 cfg renderers DO exist and are unit-tested.
* FEniCSx FEA on the rib for the Phase-2 SIMP TO — needs FEniCSx install
  on Mac (Spike 0.6b) + the rib mesh.
* Mass-cap optimisation under Layer 2 carving — Phase 4 BO objective.

## Reproduce

```sh
# Install local cadquery if not present
conda install -c conda-forge cadquery

# Generate baseline design
python3 -c "
from fanopt.geometry.envelope import Layer1Params
from fanopt.geometry.fields import Layer2Params
from fanopt.geometry.primitives import Layer3Primitive
from fanopt.geometry.manufacturability import Layer4Params
from fanopt.geometry.generator import BladeDesignParams
import json
design = BladeDesignParams(
    layer1=Layer1Params(blade_count=10, camber_knots_m=(0,0,0),
        twist_knots_rad=(0,0), thickness_knots_m=(0.0022,)*3,
        edge_profile='rounded',
        fourier_le_amplitudes=(0,0,0), fourier_te_amplitudes=(0,0,0)),
    layer2=Layer2Params.all_inactive(),
    layer3=Layer3Primitive.absent(),
    layer4=Layer4Params(print_orientation='flat', layer_height_m=0.0002,
        click_chamfer_angle_deg=45.0, click_detent_size_m=0.0004,
        click_design_clearance_m=0.00018),
)
print(json.dumps(design.to_dict(), indent=2))
" > design.json

# Run the smoke trio
python3 scripts/smoke_test.py --params design.json
python3 scripts/print_strategy.py --params design.json
python3 scripts/fan_addin.py --params design.json --out-dir exports/
```
