# Effective Spec (human-readable mirror of `effective_spec.yaml`)

This document is a flattened, current-state-only view of every locked decision in the project. It contains **no history, no superseded options, no amendment cross-references** — those live in Appendix D of `../report-final.md` for traceability.

**Single source of truth:** all production code, CI tests, SU2 templates, and Phase-N launch scripts read from `effective_spec.yaml`. This `.md` file is a human-readable rendering of the same data; the YAML is canonical when the two disagree (CI test `tests/test_audit/test_effective_spec_consistency.py` enforces this).

When a spec changes: update `effective_spec.yaml` first, then this `.md`, then `../report-final.md`. CI fails if test fixtures or code constants reference a value that differs from `effective_spec.yaml`.

**Schema version: 3** (bumped 2026-05-12; v2→v3 covers the C-series sign + projection corrections, H-series operational policies, M-series wording and locking, L-series schema fixes, plus the Z-crank rib-containment + heteroscedastic-purge + JSONL-nullable cleanup batch).

---

## Architecture

- **Blade type:** V-unit (2 ribs + 1 panel as one rigid body)
- **Material:** single-material PETG except **steel or brass** pivot pin (~2.5 g; PETG pin not permitted)
- **Default n_blades:** 10 (BO bandit explores {8, 10, 12, 14})
- **L_blade:** 200 mm (pinned, NOT a BO variable)
- **L_wrist_to_tip:** 250 mm (d_handle + L_blade; **the canonical lever arm for τ→F conversions outside the pivot region**)
- **Pivot architecture:** pin through panel at y = 0; ribs carry no pivot holes.
- **Pivot center (base-relative):** `pivot_center_x = 8 mm`
- **Boss:** 12 mm-OD circular boss × `panel_thickness` centered at `(pivot_center_x, 0)`
- **Keep-out:** `pivot_keep_out_radius = 7 mm` → PANEL_PIVOT_REGION (circular mask)
- **Pin:** 3 mm steel or brass rod, ≥ 45 mm long
- **n_panels_stacked:** 10
- **Wrist axis:** +y
- **Wrist origin:** world (0, 0, 0); pivot at (+0.05, 0, 0). Locked single convention.
- **d_handle:** 0.05 m

## Dimensions and ranges

| Quantity | Value |
|----------|-------|
| Rib width (base / tip) | 12 / 6 mm |
| Rib thickness | 2.0 mm |
| Panel width between ribs (at pivot end) | 6-8 mm |
| Blade tangential width at radius r | `r · 0.232 − 2·rib_width(r) − 0.5 mm` (panel widens with r) |
| **Panel thickness control points** | **2.2-3.8 mm** (cleanup lock — lower bumped from 2.0 to give the click chamfer 0.1 mm Z-clearance on each side) |
| Panel thickness named derivations | `panel_thickness_pivot`, `panel_thickness_tip`, `panel_thickness_max`, `panel_thickness_mid` |
| Deployed fan extent (C-2 lock) | 10 × 13.3° = **133.3°** at default 10 blades; derived from blade pitch × blade count (not an independent BO axis) |
| Inter-blade angle | 13.3° |
| Folded stack (10 blades) | 22-42 mm |

## Click features

- **Engagement:** Z-direction lap **with crank z-contained in the rib's z-extent** — `|z_click − z_rib_midplane| ≤ rib_thickness/2 = 0.001 m` (cleanup lock; prevents folded-state collision with adjacent blade's panel)
- **Chamfer angle:** 45° on the rib's top/bottom z-face
- **Chamfer overlap:** 0.5-1 mm
- **Chamfer clearance per side:** 0.1 mm (sets `panel_thickness_min = rib_thickness + 0.2 = 2.2 mm`)
- **Detent radius:** 0.3-0.5 mm
- **Design clearance:** 0.15-0.20 mm per mating surface
- **Naming:** `click_detent` (bare word `detent` reserved for V2-deferred locking)

## Kinematics (H8 symbol-table lock)

| Symbol | Value | Used at |
|--------|-------|---------|
| `f_wave` | 2 Hz | §3.2.3, §9.4 |
| `T_cycle` | 0.5 s | §9.4 |
| `θ_max` | 0.6981 rad (40°) | SU2 PITCHING_AMPL |
| `ω_SHM` | 12.566 rad/s | SU2 PITCHING_OMEGA |
| `ω_blade_max` | 8.8 rad/s | V_tip, V_local |
| `α_max` | 110 rad/s² | §2.4, Filter 2 |
| `V_tip` | 2.20 m/s | Re_global, Mach |
| `V_local(r)` | `ω_blade_max · r_wrist` | §3.2.4 BL, C6 multi-radius |
| `Re_global` | 37000 (Tier 0/1) | SU2 cfg |
| `k_reduced` | 0.57 | §3.2.3 |
| `L_blade` | 0.20 m | TO domain |
| `L_wrist_to_tip` | **0.25 m** | **all torque→force conversions** |

**Lever-arm audit:** torque-to-force conversions outside the pivot region MUST use `L_wrist_to_tip = 0.25 m`, NOT `L_blade = 0.20 m`. CI gate `tests/test_audit/test_lever_arm_uses_wrist_to_tip.py`.

## SU2 / CFD lock (C2 sign convention)

| Key | Value |
|-----|-------|
| MACH_NUMBER | 0.0064 |
| REYNOLDS_LENGTH (Tier -1) | 0.20 m |
| REYNOLDS_LENGTH (Tier 0, Tier 1) | 0.25 m |
| REYNOLDS_NUMBER (Tier -1 / 0/1) | 18000 / 37000 |
| MOTION_ORIGIN | (0, 0, 0) (wrist-grip at world origin) |
| **PITCHING_OMEGA_AXIS** | (0, 1, 0) — **TIER_SPECIFIC[1] only** (L5 lock; Tier -1/0 have no pitching motion) |
| **PITCHING_AMPL_AXIS** | (0, 1, 0) — TIER_SPECIFIC[1] |
| PITCHING_OMEGA magnitude / AMPL | 12.5664 rad/s / 0.6981 rad |
| AMPL unit | radians (pinned via SU2 build commit; verified by amplitude assertion) |
| **FREESTREAM_PRODUCTIVE** (C2) | **(0, 0, -1)** — air flows -z past stationary fan that's actually being swept in +z toward the user |
| **FREESTREAM_RETURN** (C2) | **(0, 0, +1)** |
| Unsteady ambient velocity (Tier 1) | ≈ 0.001 m/s (near-zero; H10 fallback if FREESTREAM_VELOCITY unsupported: `MACH_NUMBER=1e-9 + REF_DIMENSIONALIZATION=FREESTREAM_PRESS_EQ_ONE`) |
| TIME_STEP / MAX_TIME / TIME_ITER | 0.0025 s / 2.5 s / 1000 |
| n_cycles | 5 (extend to 8 if cycle-2-vs-3 > 5%) |
| J_fan SE | `std / √(n_cycles − 1)` — **diagnostic only, NOT fed to GP** |
| CFL_NUMBER / retries | 10.0 / 1 (halve on retry) |

## J_fan integration (C2 sign-correct two-eval delta)

- Plane Σ: 600 × 600 mm at 300 mm forward of the pivot along +z
- ρ_freestream: 1.225 kg/m³
- t̂ = +ẑ (explicit; **NOT** anatomical "from pivot toward user" phrasing)
- Cycles discarded: 1; integrate cycles 2 through N
- Cycle-2-vs-3 trigger: 5%, floored as `rel = |J2-J3| / max(|J2|, |J3|, 0.10·J_baseline)`
- **Steady-proxy** (C2): `J_fan_steady_proxy = Drag_productive − Drag_return`. A productive louver scores positive. The earlier "Drag_forward − Drag_backward" with `forward = (0,0,+1)` was inverted.

## Material constants

| Symbol | Value |
|--------|-------|
| ρ_PETG / σ_y_XY / σ_y_Z | 1270 kg/m³ / 45 MPa / **30 MPa** |
| E_PETG_XY / E_PETG_Z | 1300 MPa / 1000 MPa |
| ρ_pin (steel / brass) | 7850 / 8500 kg/m³ |
| **BED_SURFACE** | **smooth_pei** (M13 — Bambu AP05 or equivalent Ra ≤ 5 µm; textured PEI NOT permitted) |
| **Filter 2 p_aero_reference** | **10 Pa** (M3 — fixed canonical baseline; NOT per-design CFD) |

## K_t hotspot table (8 hotspots; bearing 2.00 MPa binds canonical)

| Hotspot | K_t | Mode | Cyclic allowable |
|---------|-----|------|------------------|
| Panel pivot — tension (XY) | 2.42 | tension | 5.58 MPa |
| Panel pivot — bending (XY) | 3.2 | bending | 4.22 MPa |
| Panel pivot — bearing (Z) | 1.5 | bearing | **2.00 MPa** (binds) |
| Click detent (XY) | 3.0 | mixed | 4.50 MPa |
| Click detent (Z-floor) | 3.0 | Z | 2.00 MPa |
| TPMS through-hole | 2.5 | tension | 5.40 MPa |
| Rib-panel fillet | 1.5 | bending | 9.00 MPa |
| Slot end (louver) | 2.0 | mixed | 6.75 MPa |

**§59.5 stress-test load (H9 stagnation pressure):** `p_uniform = p_stagnation_peak = ½·ρ_0·V_local_max² ≈ 3.0 Pa` (canonical); `p_stress_test = 2.5 · p_stagnation_peak ≈ 7.5 Pa`. `V_local_max = ω_blade_max · L_wrist_to_tip = 2.20 m/s`. The 2.5× multiplier stacks on top of the spatial-distribution conservatism of using stagnation pressure (NOT spatial-average F_peak / A_panel).

## Print-frame ↔ deployed-frame mapping (M15 single canonical table)

```
Print frame (build plate at z=0, build direction +z):
  - Bed-contact face: outward normal = -z (faces bed)
  - Top of print: outward normal = +z

Deployed frame (user at +z relative to fan):
  - Aero-functional face (productive +z toward user): outward normal = +z
  - Return-stroke face: outward normal = -z

Rib-flat print orientation:
  - Print bed-contact face → Deployed +z face (calibrated for wall roughness)
  - Print top → Deployed -z face (return; symmetric roughness model)
```

§N7 Check 14 (M1 reworded): "the panel's bottom face (bed-contact in print frame) has its outward normal pointing in −z (into the bed)." The calibrated face is the bed-contact face — no support scars; Ra ≈ 5-10 µm from layer lines + smooth-PEI contact.

## Constraints and Pareto

- m_total < 60 g; r_CoM_wrist ≤ 0.160 m (computed via **XZ-projection** to the +y wrist axis per C1 lock); manufacturability ≥ 0.5
- **4 Pareto objectives:** maximize J_fan (multi-fidelity CFD); minimize I_wrist about +y (deterministic; C4 rationale = peak felt torque, NOT cycle work); minimize peak panel-pivot stress (own fidelity chain); minimize folded form factor (deterministic)

## Filter 2 (three independent modes; H4 hotspot list)

| Mode | Location | Allowable |
|------|----------|-----------|
| Bending | panel skin | 4.22 MPa |
| Tension | hole equator | 5.58 MPa |
| Bearing | pin bore (Z) | 2.00 MPa (binds) |

Hotspot tightening at 0.40× headroom for: **panel pivot all 3 modes + click detent + rib-panel fillet + slot ends + TPMS through-holes + Layer 3 primitive boundaries**. The earlier "any K_t feature within 3 mm of the rib axis" criterion missed the panel-pivot bearing (at y = 0) and is removed.

Tip-deflection prefilter: < 0.6 mm. Reference pressure: `p_aero_reference = 10 Pa` (M3 fixed constant; Filter 2 has no per-design CFD).

## Acquisition pipeline (two-stage MFMO)

1. **qMFKG on J_fan only** (the one objective with CFD fidelity tiers); chooses Tier -1/0/1.
2. **qNEHVI over all 4 objectives at target fidelity** using per-objective surrogates stitched into a ModelListGP.

Per-objective models: `J_fan` MF GP; `i_wrist` / `folded_form_factor` deterministic; `peak_pivot_stress` single-fidelity GP on Filter 2 closed-form.

## Bayesian optimization

| Key | Value |
|-----|-------|
| N_DIMS | 37-46 |
| COST_TUPLE | (2.0, 10.0, 50.0) |
| HV_baseline | HV at round 200 (fixed) |
| HV early-stop floor | 500 rounds |
| HV 50-round gain threshold | 1e-4 · HV_baseline |
| Acquisition cap | 35 |
| **K seed at launch** (H3) | from **Phase 3 R²** (2D steady ↔ 2D unsteady): K=3 if ≥0.6; K=4 if 0.5≤R²<0.6; K=5 if <0.5 |
| **K running update during Phase 4** (H3) | from **Spearman ρ² on (Tier 0, Tier 1) overlap**, recomputed every 20 Tier-0 completions, once N ≥ 30 |
| K locked range | {3, 4, 5} (no K=2 path) |
| 1000-h fallback | hold K, force-grow Tier-1 overlap (NOT terminate) |
| ρ freeze threshold | 0.4 (general) / 0.2 (TPMS or noise) |
| GP noise | fixed-floor scalar per tier (NOT per-observation; **CI gate `test_no_heteroscedastic.py`** asserts no stale refs) |
| L7 Tier-1 bias diagnostic | within first 100 Tier-0 evals, if |Δ_TPMS / mean_J_fan_tier0| > 0.30 → switch TPMS/noise architectures to Tier-0-only promotion |

## H5 stop-criteria priority ladder

1. **HV plateau** (50-round gain / HV_baseline < 1e-4 after round 500) → converge + reallocate.
2. **1000-h cumulative compute** → K stays in {3, 4, 5}, force-grow (Tier 0, Tier 1) overlap. **CONTINUE Phase 4, NOT terminate.**
3. **1300-h pessimistic ceiling** → hard stop, hand off current Pareto.

If #1 and #2 fire simultaneously, **#1 wins**. If #2 fires and force-grow completes but BO has not converged by #3, terminate at #3.

## Campaign infrastructure

| Key | Value |
|-----|-------|
| Parallel Colab sessions (locked) | 2-4 |
| SU2 procs per session | 5 |
| Retrain interval | 30 min |
| Slice size | 30-60 designs |
| Compute budget (expected/pess/fav) | 600-1100 / 1300 / 300-600 h |
| **Phase 0 sub-spike budget allocations** (H7) | NOT booked against the 1000-h Phase 4 stop rule: Spike 0.6c (H10) ≈ 12 h; Spike 0.7c (H7) 430 h; C6 radial-correction 50 h |
| Stop rule | 1000 h (counted from `git tag phase4-launch`, NOT Phase 0 start) |
| Slice file pattern | `slice_assignments_v{N}.json` (versioned everywhere) |
| Drive pointer format | two-version |
| **Cross-session claim post-open sleep** | **30 s** (M6 lock; bumped from 5 s; Drive consistency window) |
| Drive latency monitoring | P99 > 25 s over 5 consecutive steps → bump sleep to 60 s |
| Schema version | 3 |
| Dedup composite key | `(design_hash, physics_hash, config_hash, material_hash, fidelity, run_direction)` |
| Hash dependency | `physics_hash = blake2b(config_hash ‖ material_hash ‖ motion_blob)` |

## JSONL schema nullability (cleanup CLEANUP-2)

Required all tiers: `design_hash`, `session_id`, `timestamp_utc`, `tier`, `status`, `J_fan`.
Tier -1/0 populate (and Tier 1 NULLs): `J_fan_productive`, `J_fan_return`, `J_fan_delta` (C2-renamed from forward/backward).
Tier -1/0 NULLs, Tier 1 populates: `J_fan_se`, `J_fan_peak`.

## C6 radial-correction calibration (Phase 0; alongside Spike 0.7b)

- 15-20 designs × 3 radii `{0.10, 0.15, 0.20} m`
- Fit `J_fan_corrected = J_fan_at_mid + β · directional_asymmetry_score`
- Artifact: `gdrive/fan-optimization/phase0/radial_correction_v1.json`
- Compute: ~50 h one-time
- Escalation if R² < 0.70: 18 LHS × 3 radii per architecture (360-1080 h fallback)

## H1 Layer 2 enumeration (locked at 20 combinations)

From 32 = 2⁵: drop 6 at {4, 5}-active; drop 4 with both {noise, TPMS}; drop 2 weak singletons. **Production list = 20 profiles.** Pre-commit gate asserts count = 20.

## H6 V1 lock (Spike 0.4 force balance + fallback)

- Pass criterion: `F_friction_cumulative ≥ 2 × F_inertial_at_click`
- `F_inertial_at_click = τ_inertial_peak / L_wrist_to_tip = I_wrist · α_max / 0.25` (NOT `/0.20`)
- Fallback geometry: 3 mm × 5 mm × 1.5 mm printed rib-tab on each guard blade
- Flag: `params.layer4.v1_lock_fallback_enabled` (default `False`; auto-armed if Spike 0.4 force balance fails)

## H10 Tier 1 unsteady-config validation (Spike 0.6c)

- **0.6c.1 — cfg sanity:** verify the rendered Tier 1 cfg parses + runs for 1 inner iteration. If `FREESTREAM_VELOCITY = (0, 0, 0.001)` is unsupported, fall back to `MACH_NUMBER = 1e-9 + REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE`. The working syntax becomes the locked Tier 1 cfg.
- **0.6c.2 — benchmark:** NACA 0012 pitching at `k = 0.5-0.6`, Re 30-50k; lift/drag within ±15% of published over 5 cycles (discard 1). Cost ~12 h.
- Phase 4 launch is **gated** on Spike 0.6c passing.

## CI gate additions (§12.0 — newly added by this revision pass)

- `test_units_meters.py` — bounding box matches SI extents; off-by-1000 fails
- `test_r_com_xz_projection.py` — point at (0.1, 0, 0.05) returns ≈ 0.112 m (not 0.1)
- `test_steady_proxy_sign.py` — 45° productive louver → `J_fan_steady_proxy > 0`
- `test_jsonl_tier1_nullable.py` — Tier 1 rows validate with the productive/return/delta nulls
- `test_no_heteroscedastic.py` — purge stale heteroscedastic noise refs
- `test_bed_surface_locked.py` — calibration metadata `BED_SURFACE` matches lock
- `test_click_z_contained.py` — `|z_click − z_rib_midplane| ≤ 0.001 m` for every blade
- `test_lever_arm_uses_wrist_to_tip.py` — torque→force conversions use 0.25 m, not 0.20 m

## Phase ordering (Phase 0 sub-spikes expanded)

| Phase | Name | Weeks |
|-------|------|-------|
| 0 | Scaffold + Spikes 0.1-0.7 + **0.6c (H10)** + **0.7c (H7)** + **C6 radial-correction** | 1-2 |
| 1 | 4-layer generative geometry | 3 |
| 1.9 | SIMP strip sanity check (**N=30 per branch**, H2) | 3.7 |
| 2a | Baseline 2D CFD | 3.5 |
| 2 | Rib SIMP TO (anchored at rib-panel y-edge) | 4 |
| 2.5 | Rib-panel fillet FEA revalidation | 4.5 |
| 2b | Generative panel optimization | 5-8 |
| 3 | 2D CFD slice | 6 |
| 4 | Multi-fidelity BO (two-stage MFMO; **launch gated on Spike 0.6c + 0.7c + C6**) | 8-10 |
| 5 | Verification + PyFR top-3 | 11-12 |
| 5.5 (step 59.5) | Combined-blade FEA gate (within Phase 5) | — |
| 6 | Print + IMU + acoustic (top 3) | 12-13 |

## Migration policy

- Schema version bumps on any field add/rename.
- `schema_migrations.py` provides `vN → v(N+1)` upgrades for persisted JSONL rows.
- **v2 → v3 migration (2026-05-12):** add r_com XZ-projection; rename JSONL fields `J_fan_forward → J_fan_productive`, `J_fan_backward → J_fan_return`; mark Tier-specific fields Optional[float]; bump panel_thickness_min 2.0→2.2 mm; move PITCHING_*_AXIS to TIER_SPECIFIC[1]; replace ω with ω_SHM / ω_blade_max subscripted symbols; replace §59.5 spatial-average pressure with stagnation-peak; bump Phase 1.9 N=10→30; rewrite W_cycle justification to peak-felt-effort framing; rewrite §N7 Check 14 to "−z outward normal"; lock BED_SURFACE=smooth_pei; lock `p_aero_reference = 10 Pa`; bump Drive post-claim sleep 5→30 s + latency monitoring; add Layer 2 20-combo enumeration; add C6 radial-correction; add Spike 0.6c (H10) + 0.7c (H7) compute booked under Phase 0.
- CI fixtures at `tests/fixtures/effective_spec_v{N}.yaml` preserve prior versions.
- Narrative plan in `../report-final.md` Appendix D records per-decision history; this `.md` and `.yaml` stay current-state.
