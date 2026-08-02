# ADR-0006 — Unsteady CFD regime: reconsider compressible@MACH=1e-9 → incompressible (V2)

**Status:** Proposed (2026-08, from the `docs/audit_2026-08_toolchain_and_approach.md` retrospective).
**This ADR does NOT change V1.** The current locked regime (`MACH = 1e-9`, compressible, per
`docs/locks_index.md §9.4.1` / Round-9 HIGH-12) remains in force for all V1 work, including the running
fine-verify. This ADR records the reconsideration so the decision is made deliberately at V2 start, not
silently — and gives the CFD-regime decision a home in `docs/adr/` instead of restated in `CLAUDE.md`.

## Context — the current (V1) decision
A hand fan moves air at ~2 m/s, so the flow is effectively **incompressible** (density ≈ constant). V1
models this with SU2's **compressible** Navier-Stokes solver forced to a near-zero Mach number
(`MACH = 1e-9`, with `FREESTREAM_OPTION = FREESTREAM_VELOCITY` / fallback
`REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE`) — a "body in still air" approximation of incompressible
flow. MACH is tier-specific: the steady tiers use a small real Mach; only the **unsteady** (pitching-fan)
tier uses 1e-9. Authoritative values: `docs/locks_index.md §9.4.1`.

## Problem (2026-08 audit)
Running the *compressible* equations at MACH=1e-9 is the textbook **low-Mach stiffness trap**: the solver
still tracks acoustic (sound) waves travelling ~5×10⁴× faster than the actual air, so the implicit
time-stepping is maximally stiff. This — not the geometry — is the root cause of:
- **~3 h per unsteady evaluation** (Colab CPU), which bottlenecked the whole BO campaign; and
- the **occasional divergences** (thin sharply-curved meridians blowing up around cycle 4-5; residual
  baseline ~1e15 with only ~5 orders of headroom to the 1e20 guard).
The audit (and its adversarial pass, which recomputed the objective from raw data) confirmed the *metric*
and *result* are sound — the issue is purely the numerical formulation of the unsteady tier.

## Proposed decision (V2)
Switch the unsteady tier from *compressible-forced-to-MACH=1e-9* to a **truly incompressible solver** that
solves the incompressible equations directly (no Mach number, no acoustic waves), e.g.
**SU2 `INC_NAVIER_STOKES`** (pressure-based; smallest migration, keeps the gmsh mesh + rigid-motion infra)
or **OpenFOAM `pimpleFoam`** (most battle-tested moving-mesh incompressible path, likely overset for the
±40° sweep). Expected: **~2-5× faster and the divergence class eliminated**, same physics done correctly.

Because an incompressible solver has **no Mach number**, accepting this **retires the MACH=1e-9 unsteady
lock** (HIGH-12) and supersedes the unsteady-tier portion of `locks_index.md §9.4.1`.

## Consequences / what acceptance requires (why it is V2, not a quick edit)
- **Re-derive** the unsteady cfg for the incompressible solver (BCs, mesh motion, time-stepping).
- **Re-validate ranking correlation:** re-run a handful of designs on both solvers and confirm the
  incompressible tier **preserves the design ranking** (Kendall-τ against the compressible campaign), so
  results stay comparable and the switch doesn't quietly change which blade wins.
- Optional pairing: a GPU lattice-Boltzmann tier (XLB / FluidX3D) as the fast high-fidelity confirmation,
  finally unblocking the PyFR-p3 gap (HIGH-11).
- **No V1 impact** — V1 ships on the existing compressible@1e-9 path.

## Supersedes / relates to
- Would supersede: the unsteady-tier MACH=1e-9 regime in `docs/locks_index.md §9.4.1` (Round-9 HIGH-12) —
  **only if accepted.** Until then, that lock stands.
- Relates to: ADR-0004 (optimization objective/fidelity), `docs/V2_backlog.md` (Audit-2026-08 items),
  `docs/audit_2026-08_toolchain_and_approach.md` (source).

## Decision owner
Operator, at V2 kickoff. Do not implement while Status = Proposed.
