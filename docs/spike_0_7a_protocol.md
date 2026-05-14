# Spike 0.7a — Generative-geometry sanity check protocol

**Spec reference:** `docs/plan_R11.md §Phase 0 Spike 0.7` (sub-spike 0.7a).
Pass criterion from the spec:

> (i) click-feature footprint is bit-for-bit intact on every blade,
> (ii) the rib material from the Phase 2 TO output is bit-for-bit preserved
>      (no Boolean subtraction reaches into the rib region — enforced by the
>      §9.7.1 panel-domain invariant).

**Why this exists.** Phase 2b runs the CadQuery 4-layer generator across
~37-46 design variables. Two classes of failure are pre-emptively gated here:

1. **Click footprint invasion.** Under the old (guard-only) Check 7, Layer 2
   carving on inner blades could push cuts into the click chamfer footprint
   at the panel's outer tangential edge. The new Check 7 is widened to all
   blades, but the gate needs a regression test.
2. **Rib material punch-through.** Under a no-panel-mask generator, Layer 2/3
   subtractive features could reach into the rib material from the Phase 2
   SIMP TO output. The §9.7.1 panel-domain invariant says Layer 2/3 carving
   operates ONLY on the panel domain.

The spike runs 10 random parameter sets + 3 hand-picked adversarial sets
through a 4-stage pipeline (generator → manufacturability filter →
click-footprint check → rib-material check) and gates the result.

---

## Bounding-box shim (read this first)

The §9.7 generator + §N7 manufacturability filter are Phase 0 / Phase 1
scaffolds (their `src/fanopt/geometry/{generator,manufacturability}.py`
files contain only docstrings). Spike 0.7a runs against a **bounding-box
shim** instead, defined in `scripts/run_spike_0_7a.py`:

- **Shim generator** produces a `dict` of Layer 2/3 feature bboxes from a
  parameter set; no actual STL is emitted (`stl_path.txt` is a placeholder).
- **Shim manufacturability filter** runs the subset of §N7 checks expressible
  on bboxes (TPMS cell ≥ 0.006 m, louver spacing ≥ 0.005 m, primitive ≥5 mm
  from the click region, panel thickness ∈ [2.2, 3.8] mm).
- **Shim click-footprint check** flags any subtractive feature bbox that
  overlaps a 1-mm-radius bbox around any of the four click-footprint corners
  `(x = L_blade ± rib_tip_taper, y = ±panel_tangential_outer)`.
- **Shim rib-material check** flags any subtractive feature bbox that
  overlaps either rib strip at
  `x ∈ [HUB_RADIUS, L_blade − RIB_TIP_TAPER], y ∈ ±[RIB_CENTER ± rib_width/2]`.

The shim's bbox check is **strictly weaker** than the bit-for-bit voxel
comparison the spec calls for. BUT it catches the failure mode each
adversarial set is designed to expose. Once Phase 2 SIMP TO output exists
and a real STL voxelizer is wired in, replace each shim with its production
counterpart; the rest of the pipeline (records, analyze gate, CLI) is
unchanged. The two regression tests
(`tests/test_geometry/test_click_feature_preservation.py`,
`tests/test_geometry/test_panel_mask.py`) carry a `BBOX_SHIM = True`
constant — flip it to `False` when the voxel comparator lands.

---

## Step 1 — Generate the 10 random parameter sets

```
python scripts/run_spike_0_7a.py --n-random 10 --seed 42
```

This:

1. Draws 10 random parameter dicts from `SCHEMA_BOUNDS` (defined in
   `src/fanopt/geometry/spike_0_7a.py`; replace with the real
   `fanopt.geometry.schema.SCHEMA_BOUNDS` import once Phase 1 lands).
2. Appends the 3 adversarial sets from `ADVERSARIAL_PARAM_SETS`.
3. Runs each through the shim pipeline.
4. Writes:
   - `data/spike_0_7a/results.json` — the aggregate result + per-design
     records.
   - `data/spike_0_7a/<params_hash>/params.json` — the input dict.
   - `data/spike_0_7a/<params_hash>/record.json` — the four-check outcome.
   - `data/spike_0_7a/<params_hash>/stl_path.txt` — placeholder
     (`(shim: no STL emitted)` until Phase 1 lands).

**Pass criterion (auto-checked by the CLI):**

- ALL adversarial sets blocked (rejected by at least one of: manufacturability
  filter, click check, rib check), AND
- (≥ 7 of 10 random sets pass) OR (≥ 3 random sets fail with non-empty
  `rejection_reasons`).

Exit code 0 = pass; 1 = fail; 2 = bad input.

---

## Step 2 — Visual inspection (manual, post-Phase-1)

**Pre-Phase 1 status:** the shim emits no STL. This step is documented for
the post-Phase-1 follow-up run.

Once the real `fanopt.geometry.generator.generate_blade` writes an STL into
each per-design subdir:

1. Open each `data/spike_0_7a/<params_hash>/blade.stl` in MeshLab / Fusion /
   PrusaSlicer.
2. Eyeball check:
   - Overall topology looks like a fan blade (no degenerate self-
     intersections, no inverted normals).
   - The click chamfer corners are intact and visible at the panel tip.
   - No through-cuts visible in the rib regions.
   - Layer 2 features (louvers, TPMS) sit only on the panel.
3. Tag each design in `data/spike_0_7a/<params_hash>/visual_inspection.md`
   with `pass` / `fail` / `borderline` and a one-line rationale.

---

## Step 3 — Manual manufacturability filter run

Same status: pre-Phase 1 the shim runs the bbox subset of §N7 automatically;
this step covers the manual cross-check once the real filter lands.

1. For each random design, run the production `fanopt.geometry.
   manufacturability.evaluate` against its STL.
2. Confirm:
   - The 6 designs the eyeball check tagged `pass` are accepted.
   - Any `fail` designs are rejected with a specific rule citation.
   - Any `borderline` designs are flagged for follow-up (the filter's
     pass/fail is the truth; the eyeball disagreement is the audit trail).
3. Note any disagreements in the phase log.

---

## Step 4 — Print 2 of the passing designs

Pass criterion (from the spec): the printed parts must print without
failures AND the click features must engage correctly when the fan is
assembled.

1. Pick 2 designs from the `pass` set, preferring those with active Layer 2
   features (so the print exercises the most generator code).
2. Slice each in PrusaSlicer with the locked profile (`profiles/petg_v1.ini`
   if it exists, otherwise the standard PETG 0.2 mm profile).
3. Print 10 blades per design + 2 guard sticks (the V1 configuration).
4. Confirm:
   - No print failures (no stringing into the click chamfer, no warping
     on the panel).
   - Assembled fan deploys and retracts.
   - Click features engage with the expected tactile detent at the locked
     positions (do NOT skip this — the click engagement is the whole point
     of the printed-part check).
5. Record:
   - Print time, filament consumed.
   - Any visible defects (with photos in `data/spike_0_7a/<params_hash>/
     print_defects/`).
   - Click engagement: `clean` / `weak` / `none`.

If either print fails or either fan's click features fail to engage, the
spike fails — re-tighten the JSON schema bounds and re-run (see fallback
below).

---

## Step 5 — Phase log

Fill out `docs/phase_logs/spike_0_7a.md` with the date, operator, results
table, and any findings. Sign off on the checklist at the bottom.

---

## Adversarial sets (the heart of the spike)

The three adversarial sets in `ADVERSARIAL_PARAM_SETS` exercise the three
failure modes the spec sub-clause names:

| ID | Target | Pre-fix failure mode |
|---|---|---|
| `a_louver_clustered_tip` | click footprint | Layer 2 louver with full cluster-at-tip + minimum spacing pushes cuts into the click region under the old guard-only Check 7. |
| `b_tpms_min_cell_click_rotation` | rib material + click | TPMS at minimum cell (0.006 m) rotated 45° puts through-cuts across BOTH the ribs and the click region under a no-panel-mask generator. |
| `c_primitive_bounds_edge` | click footprint | Layer 3 primitive parked exactly at the new "≥5 mm from outer-rib click region" boundary, sized to the max, kisses the click footprint. |

Pass criterion: every adversarial set is blocked at one of the four pipeline
stages (generator returns None / manufacturability filter rejects / click
check rejects / rib check rejects). If ANY adversarial set returns
`passed=True`, the spike fails — the §9.7.1 panel-domain invariant or the
widened Check 7 has a hole that needs fixing before Phase 2b launches.

---

## Inner-blade scope

Per the spec sub-clause: ``Run on inner blades (not just guards).`` The bbox
shim's click-footprint corners cover both guard-stick tips (at
`x = L_blade − rib_tip_taper`) and inner-blade tips (at `x = L_blade`). The
adversarial sets use `blade_count = 10` so all 10 of the V1 fan's blades are
exercised; 2 are guards and 8 are inner.

---

## Fallback if the spike fails

Per the spec: ``tighten the JSON schema bounds (disallow primitives very
close to each other; cap primitive sizes); add more aggressive
manufacturability filter rules.``

Specific actions, in order:

1. **Identify which adversarial set slipped through.** The CLI table prints
   each record's `manuf`/`click`/`rib` columns; the failing row's reasons
   tell you which check needs strengthening.
2. **Tighten the schema in `src/fanopt/geometry/spike_0_7a.py::SCHEMA_BOUNDS`
   (and the production schema once Phase 1 lands).** Likely candidates:
   - Raise the TPMS cell-size floor from 0.006 m to 0.008 m.
   - Lower the Layer 3 primitive maximum size from 0.015 m to 0.010 m.
   - Forbid `louver_cluster_tip > 0.7` (the most damaging cluster setting).
3. **Add manufacturability rules.** If the slip was a Layer 3 primitive
   adjacent to another primitive (a configuration the bbox shim doesn't
   know about yet), add a primitive-to-primitive spacing rule.
4. **Re-run the spike.** Iterate until all 3 adversarial sets are blocked.

If the random-set pass rate is too low (< 7 / 10 with < 3 documented
failures), the schema is over-constrained — relax the bounds the
manufacturability filter is over-flagging until the pass rate rises.

---

## Companion regression tests

These run on every CI build, not just at spike time:

- `tests/test_geometry/test_click_feature_preservation.py` — gates the
  click-footprint invariant against the louver and primitive adversarial
  sets.
- `tests/test_geometry/test_panel_mask.py` — gates the §9.7.1 panel-domain
  invariant against the TPMS adversarial set.

Both tests carry a `BBOX_SHIM = True` flag; flip to `False` when Phase 2
output + the bit-for-bit voxel comparator are wired in. Until then the
tests run the bbox version.
