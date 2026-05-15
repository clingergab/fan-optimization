# Spike 0.6c — Tier-1 unsteady-config sanity (V1 scope)

**Spec reference:** `docs/plan_R11.md §Phase 0 Spike 0.6c` (lines 1839-1844).

**Lock callouts:** H10 (Tier-1 cfg sanity), Round-9 HIGH-12
(= C12, the unsteady `MACH = 1e-9` lock).

**V1 scope (2026-05-14 revision):** sub-spike **0.6c.1 only** gates
Phase 4 launch. Sub-spike 0.6c.2 (NACA 0012 numerical-consistency
benchmark) was **deferred to Phase 5** after the 2026-05-14 regime
diagnostic confirmed the production-faithful MACH=1e-9 cfg produces
body-in-still-air added-mass/drag forces that can't be validated
against any published wind-tunnel NACA 0012 benchmark in the same
frame. Full decision record:
`docs/phase_logs/spike_0_6c.md` → "2026-05-14 diagnostic addendum".

**Why this exists.** The compressible-with-low-Mach-prec + RIGID_MOTION +
near-zero-ambient + 5-cycle-dual-time-stepping numerics combination is
locked on engineering judgment. Sub-spike 0.6c.1 confirms the cfg
parses cleanly under the deployed SU2 build AND that SU2 completes at
least one outer time-step on a probe mesh — that the cfg is
SYNTACTICALLY VALID and the solver-cfg combination LAUNCHES. The
remaining numerical-validation work (does the solver produce *correct*
numbers?) is the Phase 5 cross-solver gate (SU2 vs PyFR; PyFR already
provisioned in the Phase 5 verification budget).

**Phase 4 launch is gated on sub-spike 0.6c.1 passing.**
`scripts/launch_phase4.py` refuses to create the `phase4-launch` git tag
if `data/spike_0_6c/PASS` is absent. The aggregator
(`scripts/run_spike_0_6c.py`) writes the PASS marker iff
`sub_06c_1.passed` is True (V1 scope).

| Sub-spike | Phase 0 gate? | Pass criteria |
|---|---|---|
| **0.6c.1 — cfg sanity** | ✅ V1 Phase 4 launch gate | rendered cfg parses; `MACH == 1e-9` (Round-9 HIGH-12 lock); EITHER `FREESTREAM_OPTION = FREESTREAM_VELOCITY` OR `REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE`; SU2 completes ≥ 1 outer time step on a probe mesh |
| **0.6c.2 — benchmark** | ❌ Deferred to Phase 5 | SU2 vs PyFR cross-solver agreement on a published wind-tunnel NACA 0012 oscillating-airfoil case (acceptance bounds: C_L_max within ±20%, C_d_mean within ±25%, hysteresis loop sign matches + area within 2×) |

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
- No external reference dataset is required (see §Reference data
  below). Sub-spike 0.6c.2 runs two **internal-consistency** gates on
  the SU2 output alone; the literature-comparison gate that earlier
  draft protocols enumerated has been retired (see the explanation in
  §Reference data).

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

## Step 2 — Sub-spike 0.6c.2 (NACA 0012 benchmark) — **DEFERRED TO PHASE 5**

The previously-shipped `scripts/run_spike_0_6c_2.py` analyzer + the
shipped wind-tunnel-style benchmark cfg pipeline have been **removed
from V1**. The 2026-05-14 regime diagnostic on Cell 8's history.csv
(see `docs/phase_logs/spike_0_6c.md` → "2026-05-14 diagnostic
addendum") confirmed:

- The production-faithful MACH=1e-9 cfg produces body-in-still-air
  added-mass/quadratic-drag forces, NOT wind-tunnel-like aerodynamic
  lift.
- CL oscillates at **2× the prescribed pitching frequency** (clear
  quadratic-in-velocity signature), with cycle-mean **2.234× the
  amplitude** — both characteristic of body-in-still-air physics.
- The 2026-05-13 internal-consistency revision (convergence + symmetry
  gates) implicitly assumed wind-tunnel physics and is therefore
  unsound for the production cfg's regime.

**Where 0.6c.2 lives now (Phase 5):** quantitative cross-solver
validation (SU2 vs PyFR on a published wind-tunnel NACA 0012
oscillating-airfoil case in the conventional frame) with the
researcher-recommended acceptance bound:

- C_L_max within ±20%
- C_d_mean within ±25%
- Hysteresis loop sign matches + area within a factor of 2

PyFR is already provisioned in the Phase 5 verification budget; the
cross-solver script + cfg land as Phase 5 deliverables.

---

## Step 3 — Aggregate and write the Phase 4 launch marker

```
python scripts/run_spike_0_6c.py
```

V1 scope: the aggregator reads only the sub-spike 0.6c.1 result and
writes:

- `data/spike_0_6c/results.json` — aggregate result (with a
  `v1_scope_note` field documenting the 0.6c.2 → Phase 5 deferral).
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
| `outer_time_steps_completed == 0` with SU2 installed | SU2 launch error OR log not captured | Inspect SU2 stdout in `data/spike_0_6c/probe/`; common causes: mesh missing, marker names mismatch, sub_1 runner not capturing SU2's stdout to the log file the parser scans |
| `outer_time_steps_completed == 0` without SU2 | SU2 not on PATH | Install SU2 or run this sub-spike on a Colab Pro CPU session |

**Do NOT silently proceed.** Phase 4 launch is hard-gated on this spike
passing.

---

## What this rig is reused for

- **Phase 4 launch gate** — `scripts/launch_phase4.py` consults the
  `data/spike_0_6c/PASS` marker before tagging `phase4-launch`.
- **Phase 4/5 Tier-1 trust** — sub-spike 0.6c.1 confirms the production
  Tier-1 cfg parses + launches under the deployed SU2 build. Numerical
  correctness validation (the "are the numbers *right*?" question) lives
  in Phase 5 via the SU2-vs-PyFR cross-solver gate; until that
  cross-check completes, Tier-1 evaluations are trusted only as
  "syntactically valid + solver-launchable".
- **Regression baseline** — if SU2 / Gmsh / mesh-gen pipelines change
  later, re-running 0.6c.1 confirms the change preserved the cfg-load +
  solver-launch path.
