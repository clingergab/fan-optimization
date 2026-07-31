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

## Fold-feasibility fix — surface of revolution (2026-07-29)

The first trapezoid CAD build was **not feasible-by-construction on the authoritative swept-volume
fold gate**: ~90 % of decoded designs collided when folded (only ~10 % clear over a 371-design
Sobol + extremes probe). **Root cause:** the blade height was computed as `z = f(x-station)` — a
flat Cartesian strip, NOT a surface of revolution. Stacked neighbours are placed by *rotation*
about the pin, and a strip's profile goes out of phase when rotated, so multi-hump / zigzag
meridians crossed. The retired pie-slice sector used `z = f(true radius)` and folded any meridian.

**Fix (implemented, verified):**

1. **Height = surface of revolution.** The mean surface and material thickness are functions of the
   **true radius** `rho = √(x² + y²)` (clamped to the tip radius at the corners), with the trapezoid
   **planform outline unchanged** (`x = r_station`, `y = s·half_width(r_station)`). Rotated
   neighbours are then congruent, so a meridian of **any** shape/amplitude (multi-hump, zigzag,
   base→tip wave) nests. No meridian reparametrisation / monotonic restriction — full design
   freedom is retained. Holds for both ribbed (A) and uniform no-rib (B) modes.
2. **Boss-flat meridian.** The blade root overlaps the boss (a straight pin bearing that can't rise
   with the meridian). The meridian is pinned flat (`z = 0`) inside `MERIDIAN_ROOT_FLAT_RADIUS_M`
   (9 mm ≈ 1.5× boss radius) so a steep root rise can't climb into the next layer's boss. All five
   meridian knots sit at `r ≥ 44 mm`, so no aero shape freedom is lost.
3. **Panel root taper.** The panel displacement grid is smoothly tapered to zero over the same
   near-boss region (`_root_taper`) so Way-2 face waves also can't lift the buried root into the
   neighbour's boss.
4. **Panel-aware layer spacing** (already in the codec/geometry) means a panel that pokes past the
   rib rail just makes a uniformly thicker blade that still nests — so the analytic
   `containment_margin` is now a *conservative* proxy; the CAD swept-volume boolean is authoritative.
5. **Rib-rail displacement window** (`_panel_window`, the 2026-07-30 completion). A harder
   adversarial probe (subprocess-isolated, mesh-refinement classifier) showed the height fix alone
   was **incomplete**: max-panel-offset *ribbed* designs still collided ~7 mm³ and the collision
   GREW under mesh refinement (a real interference, not faceting), slipping the old 10 mm³ gate.
   **Root cause:** `displacement_at` pins the panel mean displacement to 0 only at the exact
   trapezoid edge (`v = ±1`), but a rib rail is a whole *band* (`edge_dist ≤ rib_width`); inside it
   the displacement was still ~0.8 mm nonzero, so the thick rail rode up on the panel wave and poked
   above the `±t_rib/2` containment envelope. Containment (the codec) proved only that the thin
   *panel* fits the rib envelope — never that the rails stay on the mean. `_panel_window` zeroes the
   panel displacement across the rail band (smoothstep ramp 0→1 over `[rib_width, 2·rib_width]` in
   `edge_dist`), so the rails sit on the pure meridian (surface of revolution) and only the contained
   interior panel waves — exactly the intended ribbed Way-2 model (rails = frame on the meridian;
   panel undulates between them). Aero freedom is untouched (interior mean + the independent-faces
   thickness grid are unchanged).

**Two orthogonal design axes preserved:** rib structure (ribbed A / no-rib B) stays independent of
panel shaping freedom (Way-1 whole-panel meridian wave ↔ Way-2 independent top/bottom face waves ↔
any mix); the optimizer roams both on every design.

**Result (2026-07-30, completed fix):** over a **571-design adversarial probe** (Sobol + box-corners
+ all 2⁵ meridian-knot corners × interp × mode + adversarial panel-offset extremes, 290 ribbed / 281
uniform, subprocess-isolated) **100 % fold clear**; the mesh-refinement classifier re-measured every
suspect + a random-clear sample at 2× mesh and **all shrank** (coarse max 3.4 mm³ → fine max 0.19 mm³),
**zero grew, zero false-negatives** — confirming the residual is a polyhedral **faceting artifact**
(sharp `linear` zigzag meridians only; `smooth` meridians facet to ~0). To keep the gate reliable on
the relaxed meridian ranges, the default fold mesh was raised 60×18 → **90×27** (faceting worst-case
3.4 → 0.8 mm³) and the `fold_collision_clear` threshold tightened **10 mm³ → 5 mm³** (~3× above the
90×27 faceting floor, catches any real ≥5 mm³ interference; a gross surface-of-revolution break is
~10² mm³). The CAD fold gate is wired into the 3D objective as a cheap pre-CFD backstop, so no
un-foldable design is ever evaluated. Fold is the HARD by-construction constraint; mass stays SOFT.

## Analytic fold gate — replaces the CAD boolean in-loop (2026-07-30)

The bipolar relaxation (`RIB_BOW_RANGE_M → ±20 mm`) made the CAD swept-volume boolean untenable as
the in-loop gate on two counts: (1) it **hangs indefinitely** on a valid-but-pathological decoded
solid — a steep bipolar-zigzag meridian + **checkerboard (adjacent opposite-sign) max panel
offsets** — whose `isValid()`-true single solid builds fine but whose boolean never terminates (at
any mesh); (2) at the fine mesh the tight gate needed, a *normal* bipolar design's boolean runs
~35–45 s, so a timeout can't separate slow-valid from hang → ~20 % of random Sobol designs were
false-rejected. Both are decode-reachable, so the gate would stall or starve the campaign.

**Fix — `fold_penetration_m` (analytic surface-gap gate).** `fold_collision_clear` now evaluates the
deepest interpenetration between two rotated neighbours directly from the **smooth surface-of-
revolution height field** (the same mean/thickness formulas the CAD solid is lofted from), sampling
the planform and the fold swing. It is:

- **Faceting-free / exact** — no polyhedral artifact, so the clear tolerance is a true 0.05 mm
  (`_FOLD_PENETRATION_EPS_M`), not a mm³ faceting floor;
- **Hang-proof** — no boolean; the checkerboard design that hangs OCP is resolved in ~40 ms and, in
  fact, **folds** (its hang was never a real collision);
- **~1000× faster** — ~40–50 ms vs ~35–45 s, negligible before the minutes-long CFD.

Validated: over **512 Sobol-init designs the analytic gate is 100 % fold-clear** (max interpenetration
−0.40 mm = one full fold clearance; every decoded design nests by construction), with **no hangs**;
it agrees with the CAD boolean on non-pathological designs and correctly **catches** a real collision
(recreated by disabling the rib-rail window: +0.5 mm interpenetration → rejected). This **restores
full feasible-by-construction** on the relaxed ranges. The CAD boolean `fold_collision_volume_m3` is
retained as an offline deep-verify cross-check; the lofted-solid density (`N_RADIAL_SECTIONS` ×
`N_TANGENTIAL_SAMPLES` = 60×18) reverts to serving only the CFD/FEA mesh export, not the fold gate.

## Resolved

- **Mass cap = 300 g** (operator, 2026-07-29). `schema.py MAX_TOTAL_MASS_KG = 0.300` is
  authoritative for the rerun (a pre-TO fold/mass feasibility bound; TO trims mass later). The
  `< 100 g` references in `report-final.md` / `CLAUDE.md` / `locks_index.md` are superseded.
- **Wrist-to-tip lever = 0.27 m** (operator, 2026-07-30). The 220 mm blade moves the H8 τ→F lever
  to `D_HANDLE_M` + live tip `blade.RIB_TIP_RADIUS_M` (0.22) = 0.27 m (was 0.25 m at the 0.20 m
  blade). `schema.L_WRIST_TO_TIP_M` is decoupled from the legacy `L_BLADE_M` (0.20, retained for
  the retired 2D/plano-convex/rib-TO stack) and a cross-check test pins it to the live tip. CFD
  Stage-1 re-derived and made self-consistent: `V_TIP` 2.20 → 2.37 m/s, `REYNOLDS_NUMBER_GLOBAL`
  40000 → **43000** (both tip speed AND `REYNOLDS_LENGTH` now use 0.27 m), `MACH_STEADY` = 0.0070.
- **Blade count = EXACTLY 12, always** (operator, 2026-07-30, reaffirmed). Not a BO variable, never
  {8,10,12,14}. `blade.BLADE_COUNT = 12`; the codec carries no blade-count dimension and every
  vector decodes to 12 (`test_decode_blade_count_is_fixed_at_12` pins it over 20 random vectors).

## Open items — resolved by the landed implementation

- **`RIB_TIP_TAPER`** — the live trapezoid has NO tip taper: `blade.half_width_at(r)` grows linearly
  root→tip and the rib rails (`blade_cad._is_rib_rail`) run within `rib_width_at(r)` of the edge for
  the full radial extent to the flush 51 mm tip. `schema.RIB_TIP_TAPER_M` (0.015) is legacy (retired
  rib-TO stack only).
- **Inner `HUB_RADIUS` band** — void for the live blade: it starts at `BLADE_ROOT_RADIUS_M = 0` and
  overlaps the boss. The near-hub region is governed by the boss-flat meridian
  (`MERIDIAN_ROOT_FLAT_RADIUS_M = 9 mm`) + panel `_root_taper`, not the legacy 20 mm rib-absent band.
- **Pivot keep-out** — the live blade root **is** the boss (union at r = 0, one solid, pin bore left
  open by `_boss_solid`). `PANEL_PIVOT_REGION` / `PIVOT_CENTER_X` are legacy panel-pivot concepts,
  not used by the live trapezoid.
- **Mode-B (uniform) rib-coupled locks** — resolved: the no-rib sheet has no rails, so H12 rib-width
  does not apply; nesting is bounded by the fold stack-height instead of rib containment
  (`containment_margin_m` returns `fold_margin_m` for `uniform`), and `folded_stack_height_m` /
  `layer_spacing_m` are panel-aware so the sheet folds correctly.
- **Panel-vs-rib thickness ordering (mode A)** — the ribbed rib (fold-capped, up to 12 mm) is ≥ the
  panel (3–10 mm) by codec containment (`panel_thickness ≤ min(t_rib, P_HI)`); the analytic fold gate
  confirms the assembly folds. The legacy `panel ≥ rib + 0.2` click-chamfer ordering does not apply
  to the live trapezoid; the click-feature geometry for the trapezoid is a separately-deferred item
  (it is off the live aero path — see the legacy click footprint in `schema`).
