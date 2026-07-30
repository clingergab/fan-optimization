# ADR-0005 — Trapezoid blade geometry redesign

**Status:** Accepted (operator, 2026-07-29). Supersedes the 200 mm pie-slice / panel-pivot
blade geometry recorded in `docs/report-final.md` §0/§2/§3 and the geometry locks in
`docs/locks_index.md` / `docs/effective_spec.*`. Motivated by ADR-0004 (the 3D-objective redo)
and its campaign finding that the panel grid was starved by containment.

## Context

The Stage-3 campaign converged on rib-meridian shapes and left the panel a maxed-but-tiny
ripple: containment caps panel authority at ~3 % of the rib bow. Rendering the winners exposed
two structural problems in the pie-slice geometry:

- **A ~1.4 mm root neck.** The pie-slice blade fills the full angular pitch, so its width is
  `∝ r` and collapses to ~1.4 mm where it meets the boss — a weak stress concentration at the
  highest-moment location.
- **Tangent ≠ connected.** The panel met the boss circle at a single tangent point (two parts),
  not a real merge.

The operator iterated the geometry via interactive 3D renders and locked the decisions below.

## Decision — the new blade geometry

- **12 blades, FIXED** (no longer a BO `{8,10,12}` categorical). Sets the deployed span.
- **`INTER_BLADE_ANGLE` unchanged at 13.3°** (deploy pitch). 12 × 13.3° ≈ 159.6° → **deployed
  span ≈ 43 cm** at the new length.
- **Blade length 220 mm** (was 185 mm rib tip / `L_RIB` 165 mm / `L_BLADE` 200 mm). The live
  length lever is `RIB_TIP_RADIUS_M` / `L_RIB_M`, not `L_BLADE_M` (which is unused by the live
  geometry).
- **Trapezoid planform** replacing the pie-slice sector (`width = r·INTER_BLADE_ANGLE`):
  straight tangential edges, **root width = 12 mm (= boss diameter) starting at the pivot
  centre r = 0**, tip width ≈ **51 mm** (flush tiling at 13.3° over 220 mm). Root moved 6 mm
  inward so the blade overlaps the boss disk.
- **Blade + boss = ONE solid.** The trapezoid overlaps the boss at r = 0; the union is a single
  body (the pin bore stays open). Fixes tangent-≠-connected.
- **Panel base thickness 3 mm** (was ~1.3 mm) for more BO carving room.
- **Two BO starting points, both injected into the campaign** (which previously seeded from pure
  Sobol only): **A = panel + ribs** (3 mm panel, 4 mm ribs, boss 4.4 mm); **B = single-thickness
  no-rib** (uniform 3.5 mm sheet, boss 3.9 mm). **B-proper:** a `BladeParams` rib-mode/uniform
  flag + relaxed no-rib fold/containment + unpinned panel edges, so the optimizer has the SAME
  shaping freedom on B as on A, under the SAME hard fold constraint (blades must z-stack into the
  folded position).
- **Boss height = max blade thickness + fold clearance** — `layer_spacing_m` and the fold model
  become panel-aware (was rib-only), so mode B (panel is the thickest member) folds correctly.

## Consequences

- **Stage 3 reruns cold** with the two seeds (cold start was already ADR-0004 policy; the codec
  is feasible-by-construction *for the geometry*, so the old ledger is void).
- **Stage 1 = targeted CFD-config re-derivation** (not a redo): `REYNOLDS_NUMBER_GLOBAL` and
  `MACH_STEADY` in `cfd/configs.py` are keyed to the old 185 mm tip speed → re-derive for 220 mm.
  The mesh auto-scales (bbox + absolute padding); force/kinematics/period are size-independent.
- **Stage 2 = cheap re-validation gate** before the multi-day run: re-confirm the coarse tier
  `VerifyConfig(3, 30)` still ranks like fine on the new geometry (τ was locked on the old shape).
- Codec rebuilt for the new fold/containment/thickness model; `blade_count` categorical dropped.
- Retired the pie-slice `width = r·INTER_BLADE_ANGLE` formula everywhere it appears in LIVE
  modules. The 2D `blade_slice` / `blade_objective` path is already retired (ADR-0004) and is not
  touched.

## Resolved

- **Mass cap = 300 g** (operator, 2026-07-29). `schema.py MAX_TOTAL_MASS_KG = 0.300` is
  authoritative for the rerun (a pre-TO fold/mass feasibility bound; TO trims mass later). The
  `< 100 g` references in `report-final.md` / `CLAUDE.md` / `locks_index.md` are superseded.

## Open items (resolved by the implementation)

- **`RIB_TIP_TAPER`** (rib ends 15 mm short of the tip): does it still apply to a flush trapezoid
  tip, or do mode-A ribs run to the tip?
- **Inner `HUB_RADIUS` band** (rib-absent for r < 20 mm): void now that the blade starts at r = 0?
- **Pivot keep-out region** `PANEL_PIVOT_REGION` / `PIVOT_CENTER_X`: reconcile with boss = root
  at r = 0.
- **Mode-B rib-coupled locks:** which of H12 rib-width, containment, `folded_stack_height`
  (rib-only) are relaxed vs retained for the no-rib mode.
- **Panel-vs-rib thickness ordering (mode A):** 4 mm rib > 3 mm panel inverts the old
  `panel ≥ rib + 0.2` click-chamfer clearance assumption — confirm the click clearance still holds.

These are being settled by the `blade-redesign-trapezoid` implementation and its adversarial
verification; this ADR will be finalized to match the landed code.
