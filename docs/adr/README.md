# Architecture Decision Records (ADRs)

**This is the single entry point for "what is the current design decision?"** The docs in this
repo accumulated across several major pivots (R11 → V1-slim → aero-first redesign → the 3D-
objective redo). To avoid conflicting or confusing guidance, **read the CURRENT decision for
each topic below**, then follow the supersession chain only if you need the history.

**Convention (lightweight, adopted 2026-07-26):** each major design decision is an ADR with a
Status — *Proposed / Accepted / Superseded*. A new decision that overrides an old one names it
under **Supersedes**; the superseded ADR is marked here and forward-points in its own banner.
Long-form legacy docs are retrofitted as ADRs **by reference** in the ledger below rather than
rewritten. We are not back-filling full ADR files for historical decisions — the ledger + the
current-decision table are enough to avoid confusion.

---

## CURRENT decision, by topic — read these

| Topic | Current decision | Where |
|---|---|---|
| **Optimization approach** (objective, fidelity, BO) | **ADR-0004** — redo on a correct 3D objective (CFz thrust, whole-fan periodic cascade, cold-start coarse→fine BO) | [`0004-optimization-3d-objective-redo.md`](0004-optimization-3d-objective-redo.md) |
| **Blade geometry / architecture** | **ADR-0003** — aero-first solid blade (surface-of-revolution, thick rib + thin panel, N_RADIAL_SECTIONS=40) | [`../blade_architecture_redesign.md`](../blade_architecture_redesign.md) |
| **Phase-2 structural TO** | **ADR-0007** — per-design 3D SIMP on each solid blade (freeze aero skin, carve rib/interior); supersedes the 2D representative-rib TO | [`0007-phase2-per-design-3d-blade-to.md`](0007-phase2-per-design-3d-blade-to.md) |
| **Locked constants** (geometry / kinematics / CFD) | unchanged by the pivot | [`../locks_index.md`](../locks_index.md) |
| **Unsteady CFD regime** | **MACH=1e-9 compressible** (V1, `../locks_index.md §9.4.1`); switch to incompressible **proposed for V2** | [`0006-unsteady-cfd-regime-incompressible.md`](0006-unsteady-cfd-regime-incompressible.md) |
| **Product scope / V1↔V2 split** | unchanged | [`../phase_logs/phase_0_signoff.md`](../phase_logs/phase_0_signoff.md) |
| **V2 / deferred ideas** | backlog | [`../V2_backlog.md`](../V2_backlog.md) |

If two docs seem to conflict, the table above wins.

---

## ADR ledger (supersession chain, newest first)

- **ADR-0008 — Aero BO ranked on rigid geometry (flex-blind ranking)** (2026-08-04). Doc:
  [`0008-flex-blind-aero-ranking.md`](0008-flex-blind-aero-ranking.md).
  **Status: ✅ ACCEPTED — known-limitation record.** Annotates ADR-0004 (does NOT supersede it): the V1
  aero BO ranked designs by `J_fan` on the **rigid** blade (the §2.3 "No FSI" lock), but the panel flexes
  5–15 mm under aero (§3.1) and flex is likely **differential** across designs — so the V1 selection
  *order* is unverified against flex. Fidelity gap, not a wrong-objective error. V1 proceeds (the feel
  test judges the real flexing blade); V1.5/V2 action = one-way FSI across a stiffness spread, re-rank
  finalists before any BO redo. Tracked in `../V2_backlog.md`.
- **ADR-0007 — Phase-2 per-design 3D blade TO** (2026-08-02). Doc:
  [`0007-phase2-per-design-3d-blade-to.md`](0007-phase2-per-design-3d-blade-to.md).
  **Status: ✅ ACCEPTED — CURRENT.** Supersedes the Phase-2 TO *approach* of ADR-0001 /
  `report-final.md` (single 2D representative-rib SIMP by symmetry) — the aero-first solid blade
  (ADR-0003) has no single representative rib, so TO runs **per-design 3D SIMP** on each of the
  top-10 winners (freeze the aero skin, carve rib cores + interior). Retires
  `scripts/run_phase2_to.py`. Geometry, locks, and the optimization redo (ADR-0004) are unaffected.
- **ADR-0006 — Unsteady CFD regime → incompressible** (2026-08). Doc:
  [`0006-unsteady-cfd-regime-incompressible.md`](0006-unsteady-cfd-regime-incompressible.md).
  **Status: 🕐 PROPOSED (V2).** Does not change V1. Would supersede the unsteady-tier MACH=1e-9 regime in
  `../locks_index.md §9.4.1` (Round-9 HIGH-12) if accepted — switching the pitching-fan tier from
  compressible-forced-to-MACH=1e-9 to a truly incompressible solver to remove the low-Mach stiffness that
  drives the ~3h eval cost + divergences. From the `docs/audit_2026-08_toolchain_and_approach.md` retrospective.
- **ADR-0004 — Optimization redo on a 3D objective** (2026-07-26). Doc:
  [`0004-optimization-3d-objective-redo.md`](0004-optimization-3d-objective-redo.md).
  **Status: ✅ ACCEPTED — CURRENT.** Supersedes the *optimization approach* of ADR-0001 and
  ADR-0002. Trigger: an audit found the 2D-slice objective was blind to the rib wave and the 3D
  verification measured the wrong force axis (CFx not CFz) — no design was ever optimized for
  wind. Geometry (ADR-0003), locks, and product goal are unaffected.
- **ADR-0003 — Aero-first blade redesign** (2026-07-19). Doc:
  [`../blade_architecture_redesign.md`](../blade_architecture_redesign.md).
  **Status: ✅ ACCEPTED.** The solid surface-of-revolution blade. Geometry is trustworthy and
  unaffected by ADR-0004; only the scoring layer around it was broken.
- **ADR-0002 — V1-slim** (2026-07-02). Doc: [`../plan_v1_slim_latest.md`](../plan_v1_slim_latest.md).
  **Status: ⚠️ SUPERSEDED IN PART** by ADR-0004. Its single-2D-slice optimization was found
  invalid; its V1↔V2 split and non-optimization scope still hold.
- **ADR-0001 — R11 comprehensive plan** (2026-05-12). Doc: [`../report-final.md`](../report-final.md).
  **Status: ⚠️ SUPERSEDED** by ADR-0002 (for V1 scope) — historical reference only. (Ironically
  its multi-fidelity BO *with 3D tiers* is closer to ADR-0004 than V1-slim's 2D-only cut was;
  the 2D-only shortcut is what caused the waste.)

---

## Adding a new decision
Create `adr/NNNN-slug.md` (next number), set **Status**, list what it **Supersedes**, mark the
superseded ADR here and add a forward-pointing banner to its doc, and update the
**current-decision-by-topic** table above. Keep decisions immutable once Accepted — a change is
a *new* ADR that supersedes, not an edit.
