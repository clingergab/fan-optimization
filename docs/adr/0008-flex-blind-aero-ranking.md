# ADR-0008 — The aero BO ranked designs on rigid geometry (flex-blind ranking)

**Status:** ✅ Accepted (2026-08-04) — **known-limitation record**, not a new design decision.
**Relates to** ADR-0004 (the optimization approach it annotates — does **not** supersede it), the V1
**"No FSI" lock** (`report-final.md` §2.3), and the panel-flex note (`report-final.md` §3.1). Geometry
(ADR-0003), locks, and the V1↔V2 split are unaffected.

## Context
V1's aero BO (ADR-0004) scored and **ranked** every design by `J_fan` computed on the **rigid**
CadQuery geometry. That aero runs on rigid geometry is a *documented V1 simplification* — the "No FSI"
lock (§2.3) — chosen for tractability, and `report-final.md` §3.1 already notes the panel **flexes
5–15 mm under aero load**. Stage-4 TO independently exposed the flex structurally: ~10 mm under the
inertial wrist-snap (peaks at the stroke turning point, wind-irrelevant, elastic — σ_VM 3–6 MPa ≪ 30 MPa
yield, so it springs back) and ~1.3 mm under the 10 Pa screen-placeholder aero load (the real
productive-stroke pressure is larger — hence the §3.1 5–15 mm).

The sharper point, not previously written down: **flex was never in the objective, and it almost
certainly does not scale uniformly across designs.** A design that pushes more air sees higher reaction
pressure → flexes more → loses more of that wind; a thinner-rib design flexes more than a thick-rib one.
So the rigid-`J_fan` **ranking** — *which* design wins — may not survive on the real flexed blade.

## Decision (what we record)
- The V1 aero **selection order is unverified against flex.** The risk is not that absolute `J_fan` is
  off by a constant (harmless to a ranking) but that flex is **differential**, so the rigid winner may
  not be the flexed winner. There is a plausible **systematic bias**: the BO may have quietly favored
  thinner / higher-`J_fan` designs that flex the most and lose the most wind.
- This is a **fidelity gap** (correct objective on idealized rigid geometry — standard first-pass aero
  practice), **not** a wrong-objective error like the wave-blind finding in ADR-0004. Magnitude is
  **unmeasured**.
- **V1 proceeds as-is** (past the point of no return). Mitigating fact: the V1 blinded A/B **feel test
  judges the real, flexing printed blade**, so the final V1 *pick* is not flex-blind even though the BO
  *selection* was. The gap is in the ranking that fed selection, not in the ultimate V1 verdict.

## Consequences / action (V1.5–V2, do NOT redo V1)
- **One-way FSI check** to quantify it: deform each candidate under the **real** productive-stroke aero
  pressure (not the 10 Pa `DEFAULT_AERO_PRESSURE_PA` placeholder), re-run CFD on the **deformed** shape,
  compare flexed-`J_fan` to rigid-`J_fan`. It must run across a **spread of designs** on the
  stiffness / rib-thickness axis — differential flex and ranking reshuffle are only visible by
  *comparing* designs, never from one. Cost is ~a handful of designs × ≥2 CFD each (~8–16 runs), **not
  one run**.
- **If the ranking reshuffles:** re-rank the finalists on flexed-`J_fan` (cheap) before considering a BO
  redo. Only redo the *search* if flex changes which *regions* of design space are good (a stronger
  condition the spread-of-designs check would reveal).
- This **sharpens why the V1.5 staggered AO↔TO loop's static-deflection step matters** (`V2_backlog.md`
  → V1.5): it is not only a co-optimization opportunity but a **ranking-validity** check on the V1 result.
- **Deflection limit from physics, not a guess:** the Stage-4 TO screen's 1 mm tip-deflection limit is
  arbitrary and conflates wind / survival / feel. The right aero-flex limit is derived from the
  Δ`J_fan`-vs-flex curve this FSI check produces (see ADR-0007 screening caveats).

Tracked in `V2_backlog.md` (V1.5 section, "Rigid-aero ranking validity"). Not a V1 corrective action.
