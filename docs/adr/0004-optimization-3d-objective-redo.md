# ADR-0004 — Optimization Redo on a Correct 3D Objective

| | |
|---|---|
| **Status** | ✅ Accepted — **CURRENT** (governs the optimization approach) |
| **Date** | 2026-07-26 |
| **Supersedes** | the optimization approach of ADR-0001 (R11 multi-fidelity plan) and ADR-0002 (V1-slim 2D-slice BO) |
| **Unaffected** | ADR-0003 (aero-first blade geometry — the CAD is trustworthy), the locked constants (`locks_index.md`), and the product goal / V1↔V2 split (`phase_logs/phase_0_signoff.md`) |
| **Index** | [`docs/adr/README.md`](README.md) |

This ADR changes **how** we optimize and records **why**. It does **not** change the product
goal (an aero-first 3D-printed folding fan judged by a blinded A/B feel test vs. a flat-panel
baseline) or the locked geometry/kinematics constants.

---

## 0. TL;DR

A pipeline-correctness audit found that **the entire optimization was measuring the wrong
thing — twice.** The 2D-slice objective the BO maximized is analytically **blind to the rib
meridian wave** (the dominant 3D wind lever), and the 3D verification we treated as ground
truth extracted **CFx (spanwise force), not CFz (thrust)**. Consequence: **no design was ever
optimized for wind, and the 3D rankings we picked V1 candidates from are invalid.** All prior
2D and 3D result data is discarded.

**New direction:** redo the optimization on a *correct 3D objective*, cold-start, unified
parametrization, whole-fan via periodic cascade — after fixing the pipeline and empirically
confirming the objective is a live signal (the "N1" question: *does a rigid scoop under
symmetric pitch even produce measurable net wind?*).

---

## 1. What was invalidated, and why (verified in code)

### 1a. The 2D optimization objective is wave-blind
`bo/blade_objective.py` → `cfd/blade_aero.py` → `cfd/blade_slice.py` at `radial_u=0.5`: a
single mid-radius 2D cascade slice. The airfoil is `displacement_at(r,v) ± panel_thickness_at(r,v)/2`;
the rib meridian `rib_z_at(r)` is a **rigid z-offset per slice (inert)**, so it never affects
`j_fan`. The 6 meridian DOFs (`rib_bow_knots_m`×5 + `rib_bow_interp`) had **zero optimization
signal**. The 2D-slice `j_fan` also predicts real 3D wind at only Kendall τ≈0.33.
→ **The BO optimized panel camber (a partial lever), blind to the wave.** Design 44 had the
*worst* 2D slice (rank 45/45) yet the best (wrongly-measured) 3D value — the metrics were
disconnected. Root cause: the R11 plan's multi-fidelity BO (with 3D tiers) was cut to a single
2D slice for the "V1-slim" scope; that cut silently dropped the dominant lever out of the loop.

### 1b. The 3D verification measured the wrong force axis (audit B1, CRITICAL — now FIXED)
`cfd/phase5.py:extract_j_fan_3d` used the parser default `_UNSTEADY_FORCE_CANDIDATES =
("CFx","CD","CFz")`, first-match → **always CFx**. In the 3D frame the blade spans +x, pitches
about +y, and pushes air in ±z, so thrust is **CFz**. CFx is the radial/spanwise force —
~orthogonal to thrust and ~0 by symmetry. The default tuple was correct for the 2D slice (x is
user-ward there) and copied verbatim into 3D. → **Every bakeoff / tiebreak / fishing J_fan was
cycle-mean spanwise force.** Design 44 "winning", "thin wins", the 3 V1 candidates — all invalid.

### 1c. Second geometry bug (separate, now FIXED): 12-facet CAD
`N_RADIAL_SECTIONS` was 12, so a smooth Catmull-Rom rib was chopped into a 12-facet zigzag
(indistinguishable from `linear`, and feeding the CFD spurious sharp edges → likely the
smooth-design divergence). Fixed to **40** (`blade_cad.py`; mass shifts <0.5%). The 2D slice is
analytic (does not use N_RADIAL), so this hit only 3D verification.

**Net: discard all 2D-campaign and 3D-verification data.** The V1-candidate pick, the
"thin/linear wins" and "meridian diverges" conclusions were all downstream of these bugs.

---

## 2. Full audit results (adversarial multi-agent, 9 confirmed of 24)

### Blockers (fix before any re-optimization)
- **B1 — 3D J_fan reads CFx not CFz.** CRITICAL. **FIXED** (`extract_j_fan_3d` forces
  `_THRUST_Z_CANDIDATES`; new test guards it). Also fixed **N2** (period from the `dt=T/200`
  lock, not inferred from length — a diverged run now raises instead of laundering garbage).
- **B2 — the meridian is invisible to ALL in-loop scores** (j_fan, mass, deflection), not just
  the 2D slice. Fix: the **3D objective must be the actual scorer**; drop the 2D slice from the
  loop entirely (τ=0.33 makes it a bad low-fidelity signal too).
- **B3 — `blade_count` is aero-inert** (the CFD meshes one isolated blade; 8/10/12 give
  byte-identical aero, only mass scales → bandit always picks 8 on a fake gradient). **Decision
  (user, 2026-07-26): whole-fan.** Fix via a **3D periodic-cascade** single-blade sim (captures
  the packed-blade interaction the isolated blade ignored) with **whole-fan J_fan = blade_count
  × per-blade cascade thrust**, making blade_count a real total-wind-vs-mass trade.
- **B4 — fold-feasibility ignores the rib bow.** `folded_stack_height_m` counts only rib
  thickness, never the up-to-30 mm meridian rise on the same fold axis, so "feasible by
  construction" is false for bowed ribs (the designs B2 makes the optimizer chase). Fix:
  `(N-1)·spacing + bow_max + t_rib`, and rebuild the codec cap on the corrected formula.

### Existential — "N1" — ✅ RESOLVED 2026-07-26: concept works, metric = cycle-mean CFz
The open question was whether a *rigid* blade under symmetric ±40° pitch produces measurable
**net** wind (the cycle-mean can cancel to ≈0), and which reduction (cycle-mean vs rectified
peak) is the live signal. The N1 discriminator (flat vs rib-dish scoop vs panel-camber, CFz-
corrected pipeline at N_RADIAL=40) answered all of it:

| design | cycle-mean CFz | peak | per-cycle |
|---|---|---|---|
| flat | −3.9e10 (≈0) | 7.8e11 | [−4.1e10, −3.7e10] |
| panel_camber | −2.7e10 (≈0) | 8.5e11 | [−3.1e10, −2.3e10] |
| **rib_dish** | **+1.38e11** | 9.5e11 | [+1.24e11, +1.51e11] |

- **The rigid scoop makes net wind:** rib_dish's cycle-mean CFz is clearly positive and
  per-cycle-consistent (converged), vs flat ≈0. The aero-first concept is validated.
- **Metric = cycle-mean CFz** (not peak). It cleanly discriminates — positive for the scoop, ≈0
  for symmetric/flat, which is *physically* the net directed wind. Peak is a poor discriminator
  (all ~8–9e11, dominated by instantaneous stroke force, not net wind). `Blade3DObjective`
  defaults to `metric="mean"`.
- **The rib wave is the lever:** rib_dish (+1.38e11) ≫ panel_camber (−2.7e10), confirming the
  meridian is the dominant wind mechanism — exactly what the 2D slice was blind to.

Consequence: the plane-momentum-flux fallback is not needed; cycle-mean CFz is the objective.

### Non-blocking (real, tolerable for a relative ranking; fix opportunistically)
- **N3** — `deflection_m` is degenerate (one of ~30 DOFs, `const/t³` of the thinnest node;
  gates nothing). Consider dropping/replacing as a Pareto objective.
- **N4** — pitch is about `MOTION_ORIGIN=(0,0,0)` (the pin), not the wrist; V_tip/Re use a
  0.25 m lever the motion never has (~20% operating-point offset; cancels in relative ranking,
  fix before any V2 absolute quantification).
- **N5** — 2D/3D operating-point mismatch (retired once the 2D slice leaves the loop).
- **N6 — mass proxy ~5% high** (off-by-one Riemann sum) + omits bow arc-length. Trivial fix;
  CAD mass is authoritative at verify.
- **N7** — a full steady SU2 solve is run and discarded (`blade_aero.py`); cost, not correctness.
- **N8** — lock drift: `schema.py MAX_TOTAL_MASS_KG=0.300` vs CLAUDE.md C9 `<100 g`; reconcile.

### Trust map
- **Trustworthy:** steady thrust parser (`_THRUST_Z_CANDIDATES`, CFz — the reference the
  unsteady path should have copied), `reduce_cycles` mechanics (incl. the unused `j_fan_peak`),
  CAD generation (honors the wave/camber/thickness), containment margin, codec structure.
- **Fixed-and-trustworthy:** B1 (CFz), N2 (period guard), N_RADIAL=40.
- **Still uncertain (needs the N1 result + a 2nd audit pass):** whether cycle-mean CFz is a
  usable BO signal; blade_count semantics downstream; **geometry-space consistency** (the BO
  must score in the *same* geometry space it verifies in — after B1/B2 re-run this check); the
  never-wired plane-flux path if N1 forces adopting it.

---

## 3. The new optimization direction

- **Objective = a correct 3D signal**, not the 2D slice. Fidelity ladder is **coarse-3D →
  fine-3D** (both see the wave; 2D is dropped — τ=0.33 makes it a bad cheap tier).
- **Whole-fan wind via 3D periodic cascade** (one blade in its fan context) × `blade_count`.
- **Unified single run** over the full current `BladeParams` (5-knot rib bow + interp + per-node
  panel offset/thickness grids + widened ranges) — supersedes the separate meridian/thickrib
  runs. **Shape-agnostic:** zigzags/stairs/waves/flat all compete with no prior.
- **Cold start.** All prior data is garbage; nothing valid to seed with. (Even *consistent*
  warm-starting would only be safe from same-objective data, which we do not have.)
- **Sample-efficient + cheap, not brute parallel.** Cheapening the eval (cheapest coarse-3D that
  still ranks correctly — validate first) is the orthogonal free win; TuRBO/SAASBO cut eval
  count; parallelism helps but trades against sequential learning, so use *moderate* batch
  (q≈32–48) with extra cores feeding the multi-fidelity hierarchy, not inflating q.
- **Compute:** 2× Colab G4 runtimes for their **48 CPU cores each** (SU2 is CPU-bound; GPU only
  useful later for a PyFR fine tier), running an **asynchronous shared-ledger BO** — sessions
  can't message but share the Drive ledger, so each refits its GP on the *combined* data and
  claims non-overlapping batches. That's real distributed optimization, not blind parallelism.

### Sequence ("do it right — no more cut corners")
1. **Fix the pipeline.** B1 ✓, N2 ✓, N_RADIAL=40 ✓. Remaining: **B3** (periodic cascade ×
   blade_count), **B4** (bow-aware fold), plus opportunistic N3/N6.
2. **N1 existential test** (in progress) — does the scoop make net wind; which metric.
3. **Second audit pass** on the fixes + the seams flagged uncertain (geometry-space consistency,
   blade_count downstream, the objective path).
4. **3D shape-space probe** — confirm 3D wind varies enough with design to be worth optimizing
   (the old "~0.8–1.1T plateau" was measured on the wrong axis — re-establish it).
5. **Unified coarse→fine 3D BO** on the corrected, trusted objective.

**Estimate:** ~2–3 weeks wall-clock to a genuinely optimized, verified blade, compressible to
~3–5 active days with the async-distributed + cheap-eval + TuRBO setup. This is the cost of the
shortcut that skipped 3D in the loop.

---

## 4. Downstream implications
- **V1 pick is un-made.** The 3 "thickrib" candidates, "design 44", "thin wins", "smooth
  diverges" are all invalid. There is currently **no V1 blade**.
- **TO tooling** (`src/fanopt/topopt/` Phases A–C.1) **remains valid and reusable** — it carves
  whatever blade the corrected optimization selects. No rework there.
- **Heterogeneous blades** (different shapes per fan position to optimize the whole fan) — logged
  to the V2 backlog as *measure-then-decide*: it breaks periodicity (needs full-fan sim, N× cost)
  for a likely second-order gain; quantify end-effects from a full-sector verification first.
