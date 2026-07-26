# Blade Architecture — Design Review & Redesign Proposal

> ✅ **This is ADR-0003, ACCEPTED — the current blade geometry.** Unaffected by the 2026-07-26
> optimization pivot (ADR-0004): the audit confirmed the CAD faithfully honors the wave/camber/
> thickness; only the *scoring* layer around it was broken. One change since: `N_RADIAL_SECTIONS`
> 12→40 (finer meridian). ADR index: **`docs/adr/README.md`**.

**Date:** 2026-07-19 · **Status:** decision document (operator-driven). Triggered by a
design review of the Phase-5-verified blades: the geometry that scored highest on
airflow is aggressive/self-intersecting, the panel is *thicker* than the ribs, and
the fan's foldability is asserted but never verified. This doc consolidates a
three-part investigation (current-architecture audit, data-validity audit, and
folding-fan/two-sided-design research) and proposes a new blade architecture.

---

## 1. Verdict

The current V1 blade architecture has **confirmed, real problems** that make it worth
redesigning **before** spending more compute:

1. **The panel is always thicker than the ribs** (2.2–3.8 mm panel vs 2.0 mm rib) —
   backwards from "ribs are the support beams," and confirmed by construction.
2. **Folding is never verified.** The *only* guarantee the fan folds is a single
   scalar cap (`panel ≤ 3.8 mm`) whose derivation lives in prose. There is **no
   collision / nesting / stack-height check anywhere in the code or tests**, and
   `deploy_fan` rotates blades with no overlap test at all.
3. **Only one face is optimized** (plano-convex: flat bottom, cambered top), a
   printability choice — not the both-faces free-form the design intent wanted.
4. **The panel gets no topology optimization** — it's generative-only; only the rib
   is TO'd, for structure (not aero).
5. **Boss↔rib load path is unverified** — the pivot boss connects to the ribs only
   through a thin panel sliver, and the FEA that would check it (Phase 2.5 / §59.5)
   was never run.

**Good news — it's a front-end swap, not a rebuild-from-zero.** The expensive
machinery (the BO engine, the FE/SIMP core, the whole CFD run→parse→correlate→verify
backbone, the SU2 templates, the NACA benchmark, the pitching-units fix, the Phase-5
parallelism/checkpoint fixes, and nearly all Phase-6 tooling) is **geometry-agnostic
and survives.** The damage is concentrated in the one parameterization module + its
CadQuery generators + the objective adapters that decode it — plus the Phase-4/5
*data*, which is tied to the old codec and the old blade shape.

---

## 2. Confirmed problems with the current architecture

| # | Problem | Evidence |
|---|---|---|
| P1 | Panel (2.2–3.8 mm) is always thicker than the rib (2.0 mm) | `schema.py:138,185-189`; `envelope.py` clamps every grid point into [2.2, 3.8] |
| P2 | Folding never verified — only a scalar cap `panel ≤ 3.8 mm`; no collision/nesting/stack check in `src/` or `tests/`; `deploy_fan` has no overlap test | whole-repo search for `fold\|collision\|nest\|stack` |
| P3 | Plano-convex: flat bottom, only the top face optimized; camber forced ≥ 0 | `envelope.py:79-81`, `generator.py:69-80` |
| P4 | Panel is generative-only, no TO; only the rib is TO'd (for structure) | `report-final.md:131,202`; `topopt/*` is rib-only |
| P5 | Boss↔rib cohesion unverified — Phase 2.5 fillet FEA "not run", §59.5 blocked on Phase 2b | `docs/phase_checklist.md` |
| P6 | Twist is currently inert (fixed flat print orientation zeros it) | `codec.py:14-21` |

---

## 3. What survives, what dies, what adapts (reuse map)

The single causal rule: **anything that operates on generic STEPs / polygons / CSVs /
arrays / unit-cube vectors survives; anything bound to the 35-var codec or the
flat-bottom/panel-thinner-than-rib geometry dies.**

### KEEP as-is (geometry-agnostic — the crown jewels)
- **BO engine** — `bo/backbone.py` (GP + qLogNEHVI + TuRBO + hypervolume + SAASBO;
  infers dimensionality, just needs a new search space).
- **TO core** — `topopt/plate_bending.py` + `topopt/simp.py` (plate FE + SIMP/OC;
  ready to drive the *new* panel TO).
- **CFD numerics/analysis** — `parsers.py`, `correlation.py`, `j_fan.py`,
  `config_hash.py`, the mesh cores (`mesh.py`, `airfoil_mesh.py`,
  `build_cascade_slice_mesh`), and **all 6 SU2 `.cfg.j2` templates**.
- **Recent wins** — the NACA solver-validation benchmark, the pitching-units fix
  (rad→deg + z-axis), and the Phase-5 parallelism/checkpoint/fault-isolation fixes.
- Nearly all **Phase-6 bench tooling** + the utils/ledger/audit layer.

### THROW AWAY (bound to the old parameterization/geometry)
- **Phase-4 campaign data** (`checkpoint.npz` + `evaluations.jsonl`) and **Phase-5
  `verification.json`** — points and measurements in a coordinate system and on a
  blade shape that cease to exist. The GP cannot be warm-started across a codec change.
- **The codec** (`bo/codec.py`) and **Layer-1 parameterization** (`geometry/envelope.py`)
  — they *are* what's being replaced.
- The plano-convex encoders (`cfd/panel_slice.py`, `geometry/assembly_cad.py`,
  `geometry/envelope_cad.py`), the flat-slice objective (`bo/objective.py`), the
  thickness-grid stiffness/inertia objectives (`bo/structural.py`, `bo/inertia.py`),
  the rib-specific TO problem (`topopt/loads.py`), and the old-fold click rig
  (`physical/click_rig.py`).

### ADAPT (engine kept, wiring/content rewritten)
- `bo/orchestration.py`, `bo/results.py`, `bo/cfd_objective.py` — keep the
  loop/ledger/pool, rewire to the new codec, drop the plano-convex fallbacks.
- `topopt/solver.py` — same TO loop, new **panel** problem object (not rib).
- `cfd/phase3.py`, `cfd/phase5.py` — keep the correlate/verify spine + the
  parallelism fixes; replace the codec-bound design build.
- `geometry/schema.py`, `generator.py`, `fan_assembly.py`, `manufacturability.py` —
  constants + scaffolds mostly hold; the thickness-relationship constants, the
  plano-convex enforcement, and the manufacturability orientation checks change; and
  **`deploy_fan` needs a net-new fold-collision check.**

---

## 4. Proposed new architecture

Synthesizing the operator's intent with the folding-fan/two-sided research.

### 4.1 Core idea — the fan z-stacks like a deck; the panel flies free

**How it folds (corrected 2026-07-19).** A folding fan does *not* pack its blades
side-by-side into the hub's arc length. It **z-stacks like a deck of cards**: every
blade shares the pin and sits at a **fixed z-layer** on it, and folding is each blade
rotating about the pin into a tight stack (the operator's "slide into a deck" — a deck
*is* a z-stack). So thickness stacks along the pin, and the fan folds iff:

1. **Rib = surface of revolution** — every blade's rib height depends only on radius
   (the `)` meridian, revolved about the pin). Because rotation about the pin then
   leaves the rib height field unchanged, adjacent blades keep a **constant vertical
   gap = the layer spacing at *every* swing angle** — so the ribs never collide,
   folded, deployed, or mid-swing.
2. **Panel contained** — `|panel offset(r,θ)| ≤ (t_rib(r) − t_panel)/2` (the operator's
   "panel never thicker than the rib"). During the swing, adjacent blades' sectors
   overlap; this bound keeps the *panels* from colliding given the layer spacing.
3. **Layer spacing ≥ thickest rib** — `s = maxᵣ t_rib(r) + c`, set by the boss stack.
   Rigid blades share one `s`, so it must clear the thickest section. The folded
   bundle (and the deployed z-stagger) is then `N·s`.

If 1+2 hold, **nothing on blade *i* touches blade *i+1* at any swing angle, regardless
of the panel's aero shape** — camber, zigzag, louvers are all free. The only *cost* of
folding is stack height `N·s`: a fat rib → a fat folded bundle. So "does it fold?"
becomes "is the folded stack acceptably thin?", an ergonomic bound, not a hard gate.

Key consequences:
- **Only the rib must be a surface of revolution** (any one nests, not just a cone), so
  it can be a curved `)` meridian. The panel underneath is free.
- **The panel shape never threatens folding** — the fold guarantee is z-stacking of
  congruent blades, and stacked copies keep a constant gap however the panel undulates.
- **Thin ribs everywhere** (not just the hub) keep the folded bundle and the deployed
  z-stagger small — the thickest section sets the whole stack pitch.

*(Confidence note: the z-stacking model is standard folding-fan kinematics; the
swept-volume gate in §4.5 stays as the authoritative numeric no-collision check through
the swing.)*

### 4.2 The panel — both faces free, camber surface ± thickness field

Inside the rib envelope the panel is parameterized as a **base (camber) surface** plus
a **thickness field** `t_panel(r, θ)`; the two printed faces are `base ± t_panel/2`
along the surface normal. Two free faces, no forced flat bottom, faces can't cross.
- **AO** shapes the base surface (camber/curvature) → wind generation.
- **TO** (SIMP over `t_panel`) places material where bending demands → the **panel
  finally gets topology optimization** — the one hard bound being containment (§4.1
  rule 2): `t_panel ≤ t_rib` everywhere.

### 4.3 Ribs — the structural surface-of-revolution frame (resolves P1, P4, P5)

The ribs are the thick `)`-meridian edges revolved about the pivot axis, running hub→
tip. Because they're the thickest members and the panel is bounded under them:
- **panel ≤ rib everywhere** (by construction — the rib envelope is the ceiling),
- **ribs are the structural load path**, TO-placed / TO-shaped by SIMP,
- **boss↔rib cohesion automatic** — the rib meridian runs continuously out of the
  pivot boss (no thin-panel sliver in the load path).

### 4.4 Thin ribs (the fold cost drives this)

The folded bundle and the deployed z-stagger are both `N·s` with `s = maxᵣ t_rib + c`,
so the fold cost is set by the **thickest rib section** — thin ribs *everywhere* (not
just the hub) keep the fan compact folded and near-planar deployed. The pivot boss sets
the layer spacing `s` and stacks to `N·s` tall. (The hub is still the tightest spot for
*containment* — the thin rib there most constrains panel relief — but it no longer
uniquely drives the fold; the thickest section does.)

### 4.5 A real fold gate (fixes P2)

- **In the BO loop:** a cheap **stack-height check** — `N·(maxᵣ t_rib + c) ≤` an
  ergonomic bound (`MAX_FOLDED_STACK_HEIGHT_M`) — plus containment `|offset| ≤
  (t_rib−t_panel)/2`. Both are fast analytic proxies (`geometry/blade.py`).
- **Final CAD gate:** a full **swept-volume boolean** in CadQuery — build adjacent
  blades stacked one layer apart and rotated through the fold, assert
  `solid_i ∩ solid_j = ∅` (with margin `c`) across the swing. Run per candidate, not
  in the inner loop.
- **Clearance:** `c ≥ 0.3–0.4 mm` per interface for PETG FDM.

### 4.6 Printing (fixes P3)

Print each blade **vertical / on-edge** so both faces get equal sidewall finish with
minimal supports (0.12–0.16 mm layers; brim for stability). Reserve soluble supports
(dual-extruder) only if both-face finish becomes critical.

---

## 5. Open decisions for the operator

Before the rebuild, these choices shape the new codec:

1. **Rib meridian freedom:** how much shape freedom does the `)` rib profile get —
   fixed curve, a few control points, or a full spline? (It must stay a surface of
   revolution to guarantee nesting; within that it can be as rich as we like. More
   freedom = more fold-envelope tuning, more variables.)
2. **How much AO freedom on the panel base surface** (camber only, or camber + spanwise
   curvature + edge shaping)? The panel is unconstrained except containment under the
   ribs — more freedom = richer wind shapes, harder to keep the search small.
3. **Panel TO fidelity:** 2D plate-bending SIMP (reuse the current core, fast) vs a
   thickness-field density interpretation. (Plate-bending is the cheap reuse.)
4. **Keep the click-mate deployed surface**, or let the deployed shape be free too?
5. **Blade count / pivot / mass caps** — keep the current locks ({8,10,12}, 100 g,
   12 mm boss)? (Recommend keep — they're geometry-agnostic and validated.)

---

## 6. Rebuild sequencing — AERO-FIRST (true north, operator 2026-07-19)

**Binding direction: aero drives; structure serves it.** The fan exists to move air, so
the aero shape is the *primary* objective and the structure exists only to support the
winning shape while staying light, foldable, and printable. **TO-before-AO is rejected**
— a structural pass that runs first can silently remove material that shapes the wind
(the surface and the material are the same knob in a thin solid blade). This reverses
the earlier draft ordering (which had a standalone panel TO as step 4).

**Validated (2026-07-19):** a real SU2 sweep of flat / cambered / zigzag mid-chord
sections confirmed the premise — flat ≈ zero net wind (parachute), camber produces net
directional wind, zigzag moves 3× the air. The displacement grid is a *dominant* aero
lever, so aero-first has real signal to optimize.

1. **New codec + parameterization** (`geometry/blade.py`, landed) + **CadQuery generator**
   (`geometry/blade_cad.py`, landed) + **fold gate** (stack-height in `blade.py` +
   swept-volume in `blade_cad.py`, landed).
2. **Aero objective** — new blade → 2D slice → SU2 → `J_fan`. This is the optimizer's
   **primary** objective. (Reuse the Phase-3 slice/CFD machinery; replace the plano-convex
   slice encoder.)
3. **Multi-objective BO** — maximize `J_fan` (wind), with mass, fold stack-height, and
   deflection as **constraints that protect the wind**, not a separate up-front TO.
   Structure is a coarse thickness the BO sets + a fast deflection check; fine SIMP-TO is
   deferred to V1.5.
4. **Re-run Phase 3 → 4 → 5** on the new blade (pipeline + engines unchanged).

Everything in §3 "KEEP" is reused verbatim; the work is §3 "THROW AWAY" + "ADAPT".

*(Parameterization-freedom note: the V1 grid can mix/match aero shapes spatially — camber
near the hub, corrugation outboard, any blend — because each grid node is independent. It
CANNOT create through-slots / vented topologies (it is a solid continuous surface); those
require the V2 free-form / neural-implicit route in `V2_backlog.md`.)*

---

## 7. Lean codec scope (V1) — the new parameterization

Decision (operator, 2026-07-19): **go lean, but keep the panel's shape free.** Get one
clean, fold-verified fan through the whole pipeline first; widen the search space in
V1.5 where the ML/GPU push lives. The panel aero surface is a **free displacement grid**
(not a fixed camber hump) so the optimizer discovers the panel *shape type* — camber,
base→tip zigzag, louvers, multi-hump — rather than only tuning a prescribed camber
(operator question 2026-07-19: "what if AO wants a zigzag, not camber?"). This replaces
the old 35-variable codec with **17 continuous variables + 1 categorical**.

### 7.1 Fixed locks (NOT optimized — imported from `geometry/schema.py`)

`HUB_RADIUS_M = 0.020`, `L_RIB_M = 0.165` (tip at 0.185), `PIVOT_BOSS_OD_M = 0.012`,
`RIB_BASE_WIDTH_M = 0.004` / `RIB_TIP_WIDTH_M = 0.006` (rib tangential footprint),
mass cap `< 100 g`, fold clearance `c ≈ 0.4 mm` (PETG). Blade count is the outer
architecture bandit.

### 7.2 The 17 continuous variables

| Group | Var(s) | Meaning | Rationale |
|---|---|---|---|
| **Rib meridian** `z_rib(r)` — the `)` generatrix, revolved about the pivot axis (anchored `z=0` at the boss) | `rib_bow_mid_m`, `rib_bow_tip_m` | out-of-plane rise at mid-span / tip | 2 control points → a smooth curved `)` (thin-hub, belly-out) while staying a surface of revolution → nesting guaranteed |
| **Rib thickness** `t_rib(r)` | `t_rib_hub_m`, `t_rib_tip_m` | rib thickness at hub / tip | thin ribs → thin folded stack (`s = maxᵣ t_rib + c`); TO/§59.5 refines structure within |
| **Panel aero surface** — a free displacement grid inside the rib envelope | `panel_z_{i}_{j}`, a **4×3 grid** (12 vars) of surface-normal offsets; the two rib edges are pinned to 0 | the panel mean surface = `z_rib(r) + interp(grid)` | the optimizer discovers the *shape type* (camber / zigzag / louvers / multi-hump), not just a magnitude. Each node bounded by containment `|offset| ≤ (t_rib−panel)/2` |
| **Panel thickness** | `panel_thickness_nom_m` | nominal membrane thickness; the SIMP TO field fills *within* this, bounded `≤ t_rib` | the TO density field itself is solved per-eval, not a BO var |

Plus the `blade_count` ∈ {8, 10, 12} categorical (outer bandit) → **18 dims total**,
comfortably inside GP-BO's ~40-dim tractable range.

**Grid resolution (4 radial × 3 interior tangential):** 4 radial rows give enough
base→tip control for stepped/zigzag shapes; 3 interior tangential points (edges pinned
to the ribs) give chordwise camber/asymmetry. Bump either in V1.5 if too coarse.

**Deferred to V1.5 (kept out of the lean codec on purpose):** finer grid, edge/tip
profile shaping, twist, per-stroke face asymmetry (the effort-minimizing `)(` from
`V2_backlog.md`), and a richer rib meridian (spline vs 2-point). All reachable without
breaking the lean structure.

**V2 — unbound the panel entirely:** the displacement grid is still a *bounded basis*
(it can only express what a 4×3 lattice interpolates). The `V2_backlog.md` ML track
removes the grid — neural-implicit / free-form surface — so the panel can take *any*
shape the physics prefers, not just grid-expressible ones. The grid is the V1 stepping
stone toward that.

### 7.3 Constraints (enforced/penalized, not variables)

1. **Fold (stack height):** `N·(maxᵣ t_rib + c) ≤ MAX_FOLDED_STACK_HEIGHT_M` — the
   folded bundle stays ergonomically thin (z-stacking model, §4.1).
2. **Containment:** `|panel offset(r,θ)| ≤ (t_rib(r) − t_panel)/2` — keeps the panels
   from colliding mid-swing (the operator's "panel never thicker than the rib").
3. **Mass:** `m_total < 100 g`.
4. **Geometry validity:** the 3D solid must pass `isValid()` / no self-intersection —
   the Phase-5 lesson (`V2_backlog.md` geometry-validity filter), as a pre-filter so
   the BO never chases un-buildable shapes.

Objectives are unchanged: **J_fan ↑, I_wrist ↓, panel deflection ↓** — so the Pareto /
`recommend` / verification machinery in §3 "KEEP"/"ADAPT" carries over as-is.

### 7.4 Module map for the build (maps onto §6 sequencing)

- **Parameterization** (`geometry/blade.py`, landed): the `BladeParams` dataclass +
  `rib_z_at`, `rib_thickness_at`, `displacement_at`, and the constraint-margin
  functions (`fold_margin_m`, `containment_margin_m`, `estimate_mass_kg`).
- **Codec** (`bo/blade_codec.py`, landed): encode/decode the 18-vector ↔ `BladeParams`
  + `bounds()`.
- **CAD generator** (`geometry/blade_cad.py`, new): revolve the rib meridian, loft the
  displacement-grid panel inside the envelope, boolean onto the boss, both faces,
  on-edge print.
- **Fold gate**: stack-height check (in-loop, landed in `blade.py`) + a swept-volume
  boolean over stacked/rotated blades (CAD, `blade_cad.py`).
- **Panel TO** (`topopt/loads.py` successor): SIMP over the panel domain via the kept
  `plate_bending` + `simp` cores.
- **Objective rewire**: point `bo/objective.py` / `cfd_objective.py` at the new geometry.
