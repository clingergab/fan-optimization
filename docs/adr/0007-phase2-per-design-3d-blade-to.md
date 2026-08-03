# ADR-0007 — Phase-2 structural TO: per-design 3D SIMP on the solid blade

**Status:** ✅ Accepted (2026-08-02).
**Supersedes** the Phase-2 TO *approach* of ADR-0001 / `report-final.md` (§Phase 2, §3.1): "one
representative rib, 2D plate-bending SIMP, applied to all ribs by exact symmetry." Geometry
(ADR-0003), the optimization redo (ADR-0004), locks, and the V1↔V2 split are unaffected.

## Context — the superseded decision
The R11 plan (`report-final.md`) specified Phase-2 TO as a **single, design-independent** 2D
plate-bending SIMP run on one representative rib (the tapered 4→6 mm rail), clamped against a
smooth-baseline panel placeholder, then applied to all ribs by exact symmetry. That was written
for the **discrete V-unit blade** architecture. The implementation of that plan is
`scripts/run_phase2_to.py` (+ `topopt/solver.py`, `topopt/loads.build_rib_problem`,
`topopt/plate_bending.py`).

ADR-0003 replaced the discrete V-unit blade with the **aero-first solid surface-of-revolution
blade** (thick rib rails + thin panel membrane, meshed as one solid). On that geometry a single
"representative rib" is no longer meaningful: every top design has a **distinct** meridian, rib
thickness, and panel — so a design-independent rib TO cannot express what the winners actually
need. The 2D rib script was never retired, which caused confusion about what "TO on the top 10"
meant.

## Decision
Phase-2 structural TO for V1 is **per-design 3D SIMP on each design's own solid blade**:

- **Per design** (top-10 by fine 3D `J_fan`): mesh the solid (`topopt/blade_fea_mesh`), build the
  problem (`topopt/blade_topopt.build_blade_to_problem`), run a multi-load SIMP OC loop
  (`run_blade_topology_optimization`), emit a carved density field + a structural screen.
- **Frozen aero skin (holes forbidden):** a solid shell of `skin_thickness_m` (~0.8× the 1.5 mm
  mesh) on every air-facing top/bottom surface is held at ρ≡1, so the air-pushing faces are never
  punctured; the thick rib cores and any thick-panel interior are carvable. A per-element
  rib-vs-panel split is **not** used — the rib (2–12 mm) and panel (3–10 mm) thickness ranges
  overlap, so it isn't well-defined; the uniform-shell coating is aero-safe and printable.
- **Loads:** the four locked cases (productive / return / inertial / click). The inertial body
  force is held at the **full-solid mass** (design-independent) — this keeps the compliance
  sensitivity the exact self-adjoint SIMP form and is conservative (heaviest load); a ρ-dependent
  inertial load would need the omitted `2 uᵀ ∂f/∂ρ` adjoint term (its omission was
  finite-difference-proven to flip the sign of ~⅓ of the design gradients).
- **Screen (not binding):** worst tip deflection (< 1 mm) and worst von Mises over all four load
  cases, over all elements. This is **screening**; the binding structural certification remains
  the **§59.5 combined-blade FEA gate** (`report-final.md` §Phase 2 — not yet built).
- **Screened default (max removal that keeps integrity):** rather than a fixed volume fraction, the
  default per design searches a volfrac ladder (most-aggressive first) and accepts the lowest that
  still passes the screen with a safety factor (`topology_optimize_blade_screened`) — "remove as
  much as possible while keeping integrity", decided by the measured screen, not a guess.
- **Fidelity is a knob:** finer meshes resolve internal beam structure in the thick ribs and allow
  a thinner frozen skin (more carvable interior), at steeply rising RAM/time. The four load cases
  share each iteration's stiffness, so it is factorized once and back-substituted per load
  (`solve_displacements_multi`, which also frees the factorization + `malloc_trim`s each iteration so
  RSS does not climb), keeping a finer mesh tractable. Direct-solve RAM is **superlinear** in mesh:
  measured 1.0 mm ≈ 63 s/iter / ~6 GB, 0.8 mm ~18 GB, **0.6 mm ~57 GB** (per worker). Default 0.6 mm
  mesh / 0.5 mm skin.

Entry points: `scripts/run_phase2_blade_to.py`, `notebooks/colab_stage4_blade_to.ipynb` (Stage 4 in
the campaign numbering — TO follows the Stage-3 BO campaign); design selection via
`cfd/blade_verify.top_verified_designs` (ranks by fine 3D `J_fan`). Designs are independent, so the
batch parallelizes across them (`run_blade_to_batch(n_workers=...)`, a process pool) — **RAM-bound**
(each worker holds a full factorization: ~57 GB at 0.6 mm, ~18 GB at 0.8 mm — measured, superlinear), so
`n_workers ≈ session_RAM / per_design_RAM`. The CPU sparse direct solve is not GPU-accelerated;
parallel CPU is the throughput lever.

## Consequences
- `scripts/run_phase2_to.py` and its test are **retired** (removed). The 2D machinery it used
  (`topopt/solver.py`, `topopt/loads.py`, `topopt/plate_bending.py`) is **kept** — it retains
  independent tests and other consumers (`bo/structural.py`; `blade_topopt` imports SIMP constants
  from `loads`).
- `report-final.md`'s Phase-2 prose (2D representative rib) is now stale; this ADR is the current
  decision per the "code + newest ADR wins on drift" rule in `CLAUDE.md §3`.
- **Known limitation:** for thin winning designs the uniform-shell freeze leaves little to carve (a
  ≤ ~2.4 mm rib is entirely shell), so mass removal is modest on thin blades. `skin_thickness_m` is
  the aggressiveness knob (thinner → more carving, at printability/puncture risk). Documented, not
  a defect.
- **Screening caveats (deferred to §59.5):** von Mises on intermediate-density elements overstates
  true stress (conservative), and an isotropic von Mises vs the PETG weak-axis yield is a proxy for
  a full Tsai-Wu / interlaminar criterion.
