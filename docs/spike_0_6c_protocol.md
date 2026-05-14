# Spike 0.6c — Tier-1 unsteady-config benchmark validation (H10 lock)

**Spec reference:** `docs/plan_R11.md §Phase 0 Spike 0.6c` (lines 1839-1844).

**Lock callouts:** H10 (Tier-1 cfg benchmark validation), Round-9 HIGH-12
(= C12, the unsteady `MACH = 1e-9` lock).

**Why this exists.** The compressible-with-low-Mach-prec + RIGID_MOTION +
near-zero-ambient + 5-cycle-dual-time-stepping numerics combination is
locked on engineering judgment. Without benchmark validation, the entire
Tier 1 dataset (the only "true J_fan" tier) rests on unvalidated
numerics — a silent error in any of the locked numerics would propagate
through every Phase 4/5 Tier-1 result.

**Phase 4 launch is gated on this spike passing.**
`scripts/launch_phase4.py` refuses to create the `phase4-launch` git tag
if `data/spike_0_6c/PASS` is absent.

| Sub-spike | What it gates | Pass criteria |
|---|---|---|
| **0.6c.1 — cfg sanity** | sub-spike 0.6c.2 (the benchmark itself) | rendered cfg parses; `MACH == 1e-9` (Round-9 HIGH-12 lock); EITHER `FREESTREAM_OPTION = FREESTREAM_VELOCITY` OR `REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE`; SU2 completes ≥ 1 outer time step on a probe mesh |
| **0.6c.2 — benchmark** | Phase 4 launch (`scripts/launch_phase4.py`) | every reported metric within ±15% of its published reference, integrated over the last 4 of 5 cycles |

---

## The Round-9 HIGH-12 (= C12) lock — what 0.6c.1 actually checks

Per `CLAUDE.md` and `docs/retired_phrases.yaml`:

> Round-9 HIGH-12 unsteady cfg: unsteady SU2 cfg uses `MACH = 1e-9` with
> `FREESTREAM_OPTION = FREESTREAM_VELOCITY` override (or fallback
> `REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE`). MACH is
> tier-specific: steady tiers (-1 / 0) use 0.0064, unsteady tier (1) uses
> 1e-9. `CROSS_TIER` dict does NOT carry MACH.

The spec text in 0.6c.1 anticipated a fallback path: "If
`FREESTREAM_VELOCITY = (0, 0, 0.001)` is NOT a valid compressible-solver
directive in the deployed SU2 build, replace with the alternative
compressible-zero-flow trick: `MACH_NUMBER = 1e-9` plus explicit
`REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE`". Both paths are
accepted by `check_tier1_cfg_sanity` — the cfg gate is satisfied by
EITHER syntax under the locked MACH value.

The cfg under test is the canonical one rendered from
`configs/su2/fan3d_unsteady.cfg.j2` (Phase 1 lights up the Jinja2
substitution; until then the runner falls back to the §9.4.1 spec block
verbatim, which is the source-of-truth for the production cfg).

---

## Apparatus

You need:

- A working CFD environment with the project Python installed.
  Optional but recommended:
  - `SU2_CFD` on `PATH` — required to clear the outer-step gate in
    sub-spike 0.6c.1. Without it, the runner falls back to a cfg-only
    parser check and reports `passed = False` until SU2 is wired in
    (by design — the cfg-only path cannot satisfy the spec's
    "1 outer time step on a probe mesh" requirement).
  - Gmsh — needed to mesh the NACA 0012 case for sub-spike 0.6c.2.
- A published reference dataset for the case you choose to reproduce
  (see §Reference data below). The runner ships a default
  `NACA0012_REFERENCE` dict representative of the McAlister/Carr UH110A
  studies; for a publication-grade run, override it with values taken
  directly from one cited paper.

---

## Step 1 — Sub-spike 0.6c.1 (Tier-1 cfg sanity check)

This MUST run before the benchmark; if the cfg is broken, the benchmark
numbers mean nothing.

```
python scripts/run_spike_0_6c_1.py
```

The runner:

1. Renders the canonical Tier-1 cfg via `src/fanopt/cfd/configs.py`.
   (Phase-0 fallback: extracts the §9.4.1 spec block from
   `docs/report-final.md` verbatim — the spec block IS the production
   cfg until the renderer goes live.)
2. Validates that the rendered cfg carries `MACH = 1e-9` and EITHER
   `FREESTREAM_OPTION = FREESTREAM_VELOCITY` (primary syntax) OR
   `REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE` (fallback syntax).
3. If `SU2_CFD` is on PATH, invokes SU2 on a probe mesh for one outer
   time step and counts completed outer-step markers in the stdout.
4. Writes `data/spike_0_6c/sub_1_result.json` + a
   `data/spike_0_6c/sub_1.{PASS,FAIL}` marker.

**Pass criterion:** cfg parses + `mach_value == 1e-9` + lock-compatible
freestream syntax + `outer_time_steps_completed >= 1`.

**Fail action:** investigate the parse error / MACH mismatch / SU2
launch failure before running the benchmark. If `FREESTREAM_VELOCITY` is
the failing directive, switch the template to the
`REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE` fallback and re-run
this sub-spike; record the working syntax in the phase log so §9.4.1's
reference cfg can be tightened.

---

## Step 2 — Sub-spike 0.6c.2 (NACA 0012 oscillating-airfoil benchmark)

### 2.1 Mesh the NACA 0012

Generate an O-grid or C-grid mesh of the NACA 0012 airfoil with Gmsh.
Industry-standard sizing for low-Re oscillating-airfoil cases:

- Boundary-layer first-cell height: ~1e-4 · chord
- Growth ratio in the boundary layer: 1.15
- Aspect ratio at the wall: ≤ 50:1
- Far-field radius: ≥ 50 · chord (Riemann far-field)

Write the mesh to `data/spike_0_6c/naca0012.su2` (or wherever the cfg's
`MESH_FILENAME` points).

### 2.2 Render the benchmark cfg

Use `configs/su2/oscillating_airfoil_benchmark.cfg.j2` (locked Tier-1
numerics: `MACH = 1e-9`, `FREESTREAM_OPTION = FREESTREAM_VELOCITY`,
`LOW_MACH_PREC = YES`, `TIME_MARCHING = DUAL_TIME_STEPPING-2ND_ORDER`,
`GRID_MOVEMENT = RIGID_MOTION`).

Operator-supplied template variables (see the template's header for the
canonical list):

- `mesh_filename` — path to the .su2 mesh
- `marker_airfoil` — physical-group name on the airfoil surface
- `marker_farfield` — physical-group name on the far-field
- `reynolds_number` — target Re in [30000, 50000]
- `reynolds_length` — reference chord (e.g., 1.0 m, or whatever matches
  the mesh)
- `pitching_omega_y` — angular frequency (rad/s); compute from
  `k_reduced = ω · c / (2 · U_∞)` with target `k_reduced ≈ 0.55`
- `pitching_ampl_y` — pitch amplitude (rad); 10° = 0.1745 rad
- `motion_origin_x` — quarter-chord position
- `time_step` — T / 200 with T = 2π / `pitching_omega_y`
- `max_time` — 5 · T
- `time_iter` — 5 × 200 = 1000

### 2.3 Run the benchmark

```
SU2_CFD configs/su2/oscillating_airfoil_benchmark.cfg
```

Expected wall-time: ~6-12 hours on Colab Pro CPU (per spec line 1843;
booked under Phase 0).

### 2.4 Extract per-cycle aerodynamic coefficients

For each of the 5 cycles, extract from SU2's `surface_flow.csv` /
`history.csv`:

- `c_l_max` — peak lift coefficient
- `c_l_min` — trough lift coefficient
- `c_d_mean` — cycle-mean drag coefficient
- `c_l_hysteresis_area` — signed area inside the `C_l(α)` loop

Write the rows to `data/spike_0_6c/measured.csv` with header
`cycle_index, c_l_max, c_l_min, c_d_mean, c_l_hysteresis_area`. The
shipped template is `data/spike_0_6c/measured.template.csv`.

### 2.5 Run the analyzer

```
python scripts/run_spike_0_6c_2.py \
  --measured data/spike_0_6c/measured.csv \
  --reference data/spike_0_6c/reference.json \
  --k-reduced 0.55 \
  --reynolds 40000 \
  --reference-source "McAlister/Carr UH110A 1978 — fig 7c"
```

The analyzer discards cycle 0 (initial transient), integrates cycles
1-4, and compares each integrated metric to the reference. Writes
`data/spike_0_6c/sub_2_result.json` + a `sub_2.{PASS,FAIL}` marker.

**Pass criterion:** every metric within ±15% of its reference value.

---

## Reference data

The shipped `NACA0012_REFERENCE` dict in
`src/fanopt/cfd/spike_0_6c.py` is a *template default* representative of
the McAlister/Carr UH110A studies + Anderson oscillating-airfoil DB
low-Re symmetric-foil subset. For a publication-grade run, override the
reference values with numbers taken directly from one of:

- **McAlister, K.W. & Carr, L.W. (1978)** — Water-tunnel visualizations
  of dynamic stall on the UH110A airfoil. NASA Tech Memo. Provides
  lift/drag hysteresis loops for `k_reduced ≈ 0.5-0.6` at low Re.
- **Anderson, J.M. et al. (1998)** — Oscillating foils of high
  propulsive efficiency. J. Fluid Mech. The Anderson oscillating-airfoil
  database is widely cited for NACA 0012 benchmark cases.

Pass the chosen paper's values via `--reference data/spike_0_6c/reference.json`
and cite the paper in `--reference-source`. The shipped template is
`data/spike_0_6c/reference.template.json`.

---

## Step 3 — Aggregate and write the Phase 4 launch marker

```
python scripts/run_spike_0_6c.py
```

This consumes the per-sub-spike result JSONs and writes:

- `data/spike_0_6c/results.json` — aggregate result.
- `data/spike_0_6c/PASS` (or `FAIL`) — the Phase 4 launch gate marker.

`scripts/launch_phase4.py` checks for the `PASS` marker before creating
the `phase4-launch` git tag.

---

## Fail-action decision tree

If sub-spike 0.6c.1 fails:

| Symptom | Likely cause | Fix |
|---|---|---|
| `parsed_ok = False` | MACH directive missing / malformed | Re-render the cfg; verify Jinja2 substitution is wired |
| `mach_value != 1e-9` | TIER_SPECIFIC[1] lock drift | Verify §9.4.1 TIER_SPECIFIC[1] dict; ensure CROSS_TIER does NOT carry MACH (per Round-9 HIGH-12 / C12) |
| `freestream_option == ""` and `ref_dimensionalization is None` | Neither H10 syntax present | Add `FREESTREAM_OPTION = FREESTREAM_VELOCITY` (primary) OR `REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE` (fallback) |
| `outer_time_steps_completed == 0` with SU2 installed | SU2 launch error | Inspect SU2 stdout in `data/spike_0_6c/probe/`; common causes: mesh missing, marker names mismatch |
| `outer_time_steps_completed == 0` without SU2 | SU2 not on PATH | Install SU2 or run this sub-spike on a Colab Pro CPU session |

If sub-spike 0.6c.2 fails (any metric > ±15% deviation):

| Symptom | Likely cause | Fix |
|---|---|---|
| Drag metric high; lift roughly OK | Mesh quality / first-cell height | Halve first-cell height; verify boundary-layer y+ < 1 |
| Lift peaks late in the cycle | dt convergence | Halve `TIME_STEP`; re-run; verify cycle 2 vs cycle 3 < 5% |
| Forces oscillate within a cycle | Inner-iter convergence | Bump `INNER_ITER` from 100 to 200; verify residuals reach `CONV_RESIDUAL_MINVAL` |
| Hysteresis area very low | Low-Mach prec coefficients | Tighten `MIN_ROE_TURKEL_PREC` / `MAX_ROE_TURKEL_PREC` band |
| All metrics scaled wrong by ~constant | AMPL unit (rad vs deg) drift | Verify SU2 build commit pinned in `material_locks.SU2_COMMIT` |

**Do NOT silently proceed.** Phase 4 launch is hard-gated on this spike
passing.

---

## What this rig is reused for

- **Phase 4 launch gate** — `scripts/launch_phase4.py` consults the
  `phase0/spike_0_6c/PASS` marker before tagging `phase4-launch`.
- **Phase 4/5 Tier-1 trust** — once 0.6c.2 passes, the Tier-1 numerics
  are validated against published data, so every subsequent Tier-1
  evaluation can be trusted as a "true J_fan" sample (modulo the per-run
  config-hash assertion in §9.4.1).
- **Regression baseline** — if SU2 / Gmsh / mesh-gen pipelines change
  later, re-running this spike confirms the change preserved the
  numerics.
