# Locks-to-Location Index — V1 Spec

For each lock, lists every section in the spec (and every file in the
production codebase) that consumes the lock's value. When a lock changes,
walk its entry and verify every consuming section is updated.

Maintain incrementally: every review round that touches a lock must update
the entry's consumer list. The retired-phrase gate
(`tests/test_audit/test_no_stale_architecture_refs.py`) catches the BACKWARD
direction (stale phrases left behind); this index catches the FORWARD
direction (a lock updated but not yet propagated to all consumers).

Naming: locks are referenced by their absorbed global-namespace ID (C12,
H16, etc.) where one exists, or by their Architectural-letter ID
(Architectural A/D/E) where the lock is a higher-level architecture
decision.

Sectioning below: **Architectural locks** (Section A) → **Material locks**
(Section B; §10.1) → **CFD constants** (Section C; §9.4.1 CROSS_TIER /
TIER_SPECIFIC) → **BO + campaign locks** (Section D). Sections B-D are
scaffolding only; Phase 0 Step 0.0 extends them to full coverage.

---

## A. Architectural locks

### Architectural A — RIB_TIP_TAPER = 0.015 m outer rib boundary

- **Definition:** §0 row 45 (Click-feature footprint); §3.1.2a
- **Value:** RIB_TIP_TAPER_M = 0.015 m (rib terminates 15 mm short of L_blade)
- **Consumed by:**
  - §0 row 24 (rib preserved zones)
  - §0 row 45 (click-feature footprint X coordinate)
  - §0 row 138 (panel-width formula; rib-present band)
  - §2.3 lines 200-202 (rib structure: 165 mm radial length)
  - §3.1.2a (rib radial band)
  - §9.1 `generate_rib` code block (RIB_TIP_TAPER_M = 0.015)
  - §9.7 (generator rib step)
  - §12 CI gate `test_rib_radial_extent.py`
  - GEOMETRY_LOCKS (geometry_hash input)

### Architectural D / C7 — HUB_RADIUS = 0.020 m inner rib boundary

- **Definition:** §0 row 24; §0 row 45
- **Value:** HUB_RADIUS_M = 0.020 m
- **Consumed by:**
  - §0 row 24 (rib preserved zones)
  - §0 row 45 (rib-present band, click footprint)
  - §0 row 138 (panel-width formula)
  - §2.3 lines 199-202 (rib structure)
  - §3.1.2a (rib radial band)
  - §6.3.1 Filter 1 (geometric feasibility)
  - §9.1 `generate_rib` code block (HUB_RADIUS_M = 0.020)
  - §9.2 SIMP code block (mm and m versions)
  - §9.7 (generator rib step)
  - §12 CI gate `test_rib_radial_extent.py`
  - GEOMETRY_LOCKS

### Architectural E / C10 — print-orientation-conditional midplane symmetry

- **Definition:** §0 row 47; §3.2.0 row 574-580
- **Value:** rib-flat → coplanar bottom face at z = 0 (one-sided +z corrugation);
  deployed-V / edge → midplane symmetry (no plano-convex constraint)
- **Consumed by:**
  - §0 row 47 (plano-convex envelope rule)
  - §0 row 561 (corrugation sign convention)
  - §3.2.0 lines 574-580 (full conditional spec)
  - §3.2.2 line 517 (corrugation amplitude in CFD-differences section)
  - §3.2.4 wall-roughness calibration anchor
  - §9.4 J_fan modeling
  - §2.1 click-engagement Z math (panel top-bottom face geometry)
  - Phase 3 step 33/35 (2D slice geometry)
  - §12 CI gate `test_rib_bottom_coplanar_panel_bottom.py`

### C8 — 13.3° blade angular pitch; 133.3° fan extent

- **Definition:** §0 row 132-133
- **Value:** INTER_BLADE_ANGLE_RAD = 0.232; deployed extent = blade_count × 13.3°
- **Consumed by:**
  - §0 row 132 (pitch lock)
  - §0 row 133 (deployed extent)
  - §0 row 138 (panel-width formula uses INTER_BLADE_ANGLE_RAD = 0.232)
  - §1.2 line 108 (fan macro)
  - §2.5 line 278 (deployed extent description)
  - §6.2.1 line 934 (parameter table — spread_angle row deleted per HIGH-5 Round-8)
  - §6.2.2 line 1009 (architecture-bandit enumeration: 3 blade counts)
  - §Phase 3 step 33 (cascade arc-length spacing)
  - GEOMETRY_LOCKS (INTER_BLADE_ANGLE_RAD)

### C9 — m_total < 100 g

- **Definition:** §0 row 28
- **Value:** 0.100 kg (was 0.060 kg pre-Round-5; relaxed for trapezoidal panel
  mass budget)
- **Consumed by:**
  - §0 row 28 (mass constraint lock)
  - §0 row 32 (Pareto objective: hard constraints)
  - §6.3.1 Filter 1
  - §6.4 line 1551 (mass constraint statement)
  - §6.4 line 1552 (r_CoM sanity check uses 100 g distribution)
  - Phase 4 step 51 / step 59 (JSONL schema, handoff gate)
  - Phase 5 verification gate
  - §Phase 4 line 2206 (Hard mass constraint per HIGH-7 Round-8 fix)
  - §Phase 5 line 2319 (Mass constraint per HIGH-8 Round-8 fix)

### C11 — PITCHING_OMEGA = (0, -12.5664, 0) negative sign

- **Definition:** §0 row 26
- **Value:** PITCHING_OMEGA_SIGNED_Y_RADS = -12.5664 (NEGATIVE — right-hand-rule
  on productive stroke)
- **Consumed by:**
  - §0 row 26 (coordinate convention)
  - §9.4 line 501 (rotation axis description)
  - §9.4.1 cross-tier hash (PITCHING_OMEGA_SIGNED_Y_RADS = -12.5664)
  - §9.4.1 cfg template (PITCHING_OMEGA = 0.0 -12.5664 0.0)
  - §12 CI gate `test_pitching_physical_motion.py`
  - §12 CI gate (config_hash content)

### C13 — rib code SI/correct-dimensions

- **Definition:** §9.1 `generate_rib` code block; §9.2 SIMP code block
- **Value:** rib_radial_length = 0.165 m, base_width = 0.004 m, tip_width = 0.006 m,
  thickness = 0.002 m, ALL in SI meters
- **Consumed by:**
  - §9.1 lines 2411-2419 (production code SI/meters)
  - §9.2 lines 2598-2601 (pedagogical mm code with production callout)
  - §12 CI gate `test_rib_code_matches_locks.py`
- **Notes:** §9.2 mm block is INTENTIONALLY mm-units (pedagogical DTU TopOpt
  reproduction); production code in src/fanopt/topopt/ is SI per C9 lock.

### C14 — Phase 2.5 uses 3D static FEA (not 2D plate-bending)

- **Definition:** §Phase 2.5
- **Consumed by:**
  - §Phase 2.5 (toolchain spec — FEniCSx 3D or CalculiX)
  - §3.1.2a CRIT-3 note (rib SIMP density-clamp is geometric, not stress-correct)
  - Spike 0.6b (validates FEniCSx/CalculiX on M3)
  - §Phase 5 step 64.5 fallback path reference

### C15 — diagonal rib-panel BC mask

- **Definition:** §9.2 SIMP code block
- **Consumed by:**
  - §9.2 `get_rib_panel_interface_mask` function
  - §12 CI gate `test_rib_bc_diagonal.py`

### H9 (Round-9 MED-4 supersedes) — FEA loading model: stagnation-peak + 2.5× stress-test multiplier; 5× total conservatism

- **Definition:** §59.5 line 2288 (with Round-9 MED-4 disclosure)
- **Value:** uniform p = p_stagnation_peak ≈ 3.0 Pa (canonical) /
  2.5 × 3.0 ≈ 7.5 Pa (stress-test). Total bending-moment conservatism vs
  nominal physical loading: 5× (2× from uniform-vs-r² + 2.5× stress-test).
- **Consumed by:**
  - §59.5 line 2288 (load specification + MED-4 Round-9 disclosure)
  - §10.1 cyclic-fatigue allowables (stress comparison basis)
  - Phase 5 verification (top-3 designs against this loading)
- **Notes:** V2 backlog: switch to actual `p(r) ∝ r²` distribution if margin
  pressure emerges in Phase 5.

### H12 — RIB_BASE_WIDTH_M = 0.004, RIB_TIP_WIDTH_M = 0.006 (locked constants, NOT BO variables)

- **Definition:** GEOMETRY_LOCKS
- **Value:** RIB_BASE_WIDTH_M = 0.004 m (4 mm narrow root); RIB_TIP_WIDTH_M = 0.006 m (6 mm wider tip)
- **Consumed by:**
  - GEOMETRY_LOCKS (the lock itself)
  - §6.2.1 lines 937-938 (DELETED per HIGH-10 Round-9; rib widths not BO variables)
  - §9.1 `generate_rib` (RIB_BASE_WIDTH_M, RIB_TIP_WIDTH_M)
  - §9.2 SIMP rib code blocks
  - §0 row 138 (panel-width formula uses 2·rib_width(r) term)
  - §N7 check 15 (1 mm rib-panel fillet — depends on rib widths)

### H14 (Round-9 MED-3 supersedes) — r_CoM trapezoidal centroid sanity check, corrected kinematics

- **Definition:** §6.4 line 1552 (Round-9 MED-3 lock)
- **Value:** r_CoM ≈ 0.133 m (was 0.122 m pre-MED-3); margin against 0.160 m
  bound = 17% (was 19% under earlier incorrect kinematics)
- **Consumed by:**
  - §6.4 line 1552 (sanity check derivation)
  - Phase 4 step 51 (BO mass-distribution checking)

### MED-10 (Round 7) — 14-blade trim

- **Definition:** §0 row 125, row 133
- **Value:** blade_counts ∈ {8, 10, 12}; 14-blade removed (186.2° unergonomic)
- **Consumed by:**
  - §0 row 125 (Blade count)
  - §0 row 133 (Deployed fan extent BO range)
  - §0 row 134 (architecture-bandit enumeration)
  - §1.2 line 108 (fan macro) — HIGH-6 Round-8 fix
  - §6.2.1 line 934 (parameter table)
  - §6.2.2 line 1009 (3 blade counts × 3 × 3 × 3 × 20 × 2 = 3,240; HIGH-1 Round-8 fix)
  - §Phase 2b line 1879 (Layer 1 envelope) — HIGH-6 Round-8 fix
  - §6.2.2 line 2026 (fan-macro continuous) — HIGH-6 Round-8 fix
  - §Phase 4 line 3121 (discrete handling) — HIGH-3 Round-9 fix
  - Footnote at line 77

### item #3 + HIGH-8 Round-9 Option A — panel-edge click; chamfered butt joint with detent

- **Definition:** §0 row 139 (Option A engagement); §0 row 140 (mating chamfer)
- **Value:** click chamfer is a 0.5-1 mm × 0.5-1 mm corner bevel at the panel's
  outer tangential edge; engagement is friction + detent; NO Z-axis overlap.
- **Consumed by:**
  - §0 row 17 (Fan architecture) — HIGH-1 Round-9 fix
  - §0 row 33 (panel_thickness_tip drives chamfer-depth budget per HIGH-8)
  - §0 row 45 (Click-feature footprint)
  - §0 row 139 (engagement architecture — Option A friction + detent)
  - §0 row 140 (mating chamfer — 0.5-1 mm bevel, NOT full-z)
  - §1.2 line 75 (Executive Summary) — MED-10 Round-8 fix
  - §1.2 line 84 (Mechanism) — MED-10 Round-8 fix
  - §2.1 line 139 (click feature row)
  - §2.2 click-mating mechanism subsection (lines 217-236; HIGH-8 Round-9 rewrite)
  - §2.3 line 268 (click-engagement force) — MED-1 Round-9 fix
  - §3.1 (K_t hotspot table — click detent)
  - §3.1.2 (rib SIMP TO scope)
  - §7.4.1 line 1669 (Click Feature Print Considerations) — MED-2 Round-9 fix
  - §Phase 0 Spike 0.4 line 1785-1787 (force gauge protocol)
  - §Phase 0 Spike 0.4 line 1794 (test article print) — HIGH-4 Round-9 fix
  - §Phase 5 step 80 (visual inspection) — MED-10 Round-8 fix
  - §N7 check 7 (click clearance manufacturability) — MED-10 Round-8 fix
  - Appendix B/D line 3928 (locked decisions index) — HIGH-2 Round-9 fix
  - §12 CI gates: `test_click_z_lap.py` (Round-9 HIGH-8 update), `test_click_chamfer_face.py`

### item #35 — trapezoidal panel widening

- **Definition:** §0 row 138; H13
- **Value:** panel_width(r) = r · 0.232 − 2·rib_width(r) − 0.5 mm; tip width ≈ 45 mm
- **Consumed by:**
  - §0 row 28 (mass constraint — trapezoidal mass scaling)
  - §0 row 45 (CLICK_FOOTPRINT_Y_RANGE — panel_tangential_outer)
  - §0 row 138 (panel-width formula)
  - §3.2.0 line 494 (blade planform description; MED-11 Round-8 fix)
  - §6.4 line 1552 (r_CoM trapezoidal centroid sanity check; MED-3 Round-9 fix)
  - §9.7 panel domain mask
  - GEOMETRY_LOCKS (CLICK_FOOTPRINT_Y_RANGE_PANEL_EDGE_M_AT_TIP)
  - M19 J_max trapezoidal correction (A_panel = 0.053 m²)

### Phase 5 verification configuration (G4 GPU per HIGH-11 Round-9)

- **Definition:** §0 row 31; Appendix B line 3887
- **Value:** Colab Pro G4 GPU (95 GB VRAM) for PyFR p=3 top-3 verification.
  T4 (16 GB) is insufficient (14-18 GB working set OOMs on T4).
- **Consumed by:**
  - §0 row 31 (compute hardware table)
  - Appendix B line 3887 (Phase 5 verification row)
  - Appendix B compute-time table
  - §Phase 4 line 2199 (hardware routing — HIGH-11 Round-9 fix)
- **Notes:** G4 replaces T4 with 5-6× memory headroom. PyFR p=3 retained as
  the original verification intent.

### §9.4.1 unsteady cfg freestream (HIGH-12 Round-9; Round-10 follow-up)

- **Definition:** §9.4.1 unsteady cfg block (primary syntax); §9.4.1
  CROSS_TIER / TIER_SPECIFIC dicts (config-hash assertion source-of-truth).
- **Value:** Primary: FREESTREAM_OPTION = FREESTREAM_VELOCITY + MACH = 1e-9 +
  FREESTREAM_VELOCITY = (0, 0, 0.001); FREESTREAM_DIRECTION NOT set. Fallback:
  MACH = 1e-9 + REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE.
- **MACH is tier-specific (Round-10 follow-up):**
  - Tier -1 (2D steady): MACH = 0.0064 (V_tip as freestream, body stationary)
  - Tier 0 (3D steady): MACH = 0.0064 (V_tip as freestream, body stationary)
  - Tier 1 (3D unsteady): MACH = 1e-9 (ambient near-zero; body via GRID_MOVEMENT)
  MACH lives in `TIER_SPECIFIC`, NOT `CROSS_TIER`. A single cross-tier MACH
  would fire `config_mismatch` on every Tier-1 run because the rendered .cfg
  has `MACH = 1e-9` while the lock would assert `0.0064`.
- **Consumed by:**
  - §9.4.1 CROSS_TIER / TIER_SPECIFIC dicts (MACH tier-specific; config-hash
    assertion)
  - §9.4.1 unsteady cfg block (primary syntax + body-vs-ambient explanation)
  - §9.4.1 Spike 0.6c.1 fallback syntax
  - Spike 0.6c.1 (syntax verification)
  - §12 CI gate `test_unsteady_freestream_consistency.py` (NEW per HIGH-12)
  - `docs/retired_phrases.yaml` entry `KEEPS\s+MACH_NUMBER\s*=\s*0\.0064`
    (Round-10 catalog entry catches reintroduction of the retired phrasing)
- **Notes:** Working syntax determined by Spike 0.6c.1 (one of the two paths
  above). Without one of them, SU2's default TEMPERATURE_FS would compute
  freestream as MACH × c_ref = V_tip — zero relative velocity, physically
  nonsensical CFD.

---

## B. Material locks (§10.1)

Scaffolding placeholder. Phase 0 Step 0.0 extends to full coverage.

Known locks: σ_y_XY = 45 MPa; σ_y_Z = 30 MPa; K_tt = 2.42 (12 mm boss);
K_tb = 3.2; K_t_bearing = 1.5; cyclic allowables 5.58 / 4.22 / 2.00 MPa.

---

## C. CFD constants (§9.4.1 CROSS_TIER / TIER_SPECIFIC)

Scaffolding placeholder. Phase 0 Step 0.0 extends to full coverage.

Known locks: MACH_NUMBER (steady/unsteady tier-specific per HIGH-12 Round-9);
REYNOLDS_LENGTH = 0.20 m (Tier -1) / 0.25 m (Tier 0/1); cycle count = 5
canonical (extend to 8 if cycle-2 vs cycle-3 > 5%); cost tuple (2, 10, 50).

---

## D. BO + campaign locks

Scaffolding placeholder. Phase 0 Step 0.0 extends to full coverage.

Known locks: HV early-stop floor = 500 rounds; K_promoted ∈ {3, 4, 5} per
Spearman ρ²; cost tuple (2, 10, 50); fixed-floor GP noise (NOT
heteroscedastic); slice_size = 30-60 designs.

---

## Maintenance policy

When a lock changes:

1. Update the lock's `Definition` and `Value` fields in this index.
2. Walk the `Consumed by` list and verify each consuming section reflects
   the new value.
3. If new consumers are added during the change, append them to the list.
4. Update `docs/retired_phrases.yaml` if any pattern is now retired
   (per `docs/review_process.md §4`).
5. Run all `tests/test_audit/*` gates to verify no drift was missed.
