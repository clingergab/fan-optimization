# Audit 2026-08 — toolchain, pipeline & approach (V1 retrospective → V2 input)

**Method.** A clean-slate evaluation run near the end of V1 (only TO + print left). Five parallel domain
deep-dives (geometry/rendering, CFD, BO, distributed infra, overall approach), each combining codebase
assessment with external tool/library research, then synthesized and attacked by two adversarial
challengers (one of which **recomputed the objective from raw SU2 data**). Verdicts below are
*post-adversarial* and confidence-rated; the final subsection records what the adversarial pass changed.

**One-line takeaway.** The toolchain and the *result* are largely sound — the campaign found a real,
non-obvious optimum on a *validated* metric. The weaknesses are **formulation** (over-parameterized
search), **CFD regime** (compressible-at-zero-Mach), **substrate** (Colab+Drive), and **one validity gap
in V1's planned validation instrument**. **None of these would improve the V1 designs you already have**
(they're locked and the metric is validated) — they are almost entirely a **V2 roadmap**, plus one cheap
V1 course-correction on *how* to validate.

---

## The linchpin (resolved): the campaign optimized a REAL signal, not noise
Challenger recomputed `J_fan` from the raw `CFz` histories (`data/n1_discriminator/`, `data/stage2_probe/`):
a flat blade nets **−3.9e10** (air pushed *away*), a cupped blade nets **+1.4e11** (air pushed *toward*
you) — real aerodynamic **rectification**, sign-separating, dish-vs-flat gap ~16× the cycle-to-cycle noise.
- The cycle-**mean** is therefore the *correct* metric (it rewards rectifying cupped shapes; the *peak* is
  ~equal across designs and does **not** discriminate). → The raw research rec to "switch to RMS/peak" was
  **overturned**; keep the cycle-mean.
- **Qualifier (real):** the signal is a ~2-7% residual of a huge ±2e12 added-mass oscillation, so it's
  expensive to resolve and the code's "added-mass bias cancels in ranking" defense (`objective.py:74-78`)
  has a shape-dependent hole (a deep cup has more added mass than a flat plate). Worth a one-off
  `dt`/cycle-count convergence check; confirm production runs ≥5 cycles (the probes ran 3).

## Domain verdicts (post-adversarial)

1. **Geometry / rendering — KEEP AS-IS.** *(high)* Really a numpy/mesh pipeline using CadQuery/OCC only as a
   thin STEP/volume container; the fold-nesting fix (surface-of-revolution `z=f(rho)`) is a modeling insight,
   already done right; the analytic fold gate is the crown jewel. Don't migrate to build123d (identical OCC
   kernel) or SDF/implicit (wrong for thin printed shells). *Footnote: revisit SDF only if V2 goes
   3D-cellular/lattice structural.*

2. **CFD — the *regime* is a mismatch; the *metric* is right.** *(high)* Compressible equations at MACH=1e-9
   is the textbook low-Mach stiffness trap — the true cause of the ~3h cost **and** the divergences (not the
   geometry). A **truly incompressible** solver (SU2 INC / OpenFOAM pimpleFoam) would kill the divergence
   class and speed it ~2-5× — but this **touches the C12 / Round-9 HIGH-12 lock** (MACH=1e-9), so it needs an
   ADR + operator authorization, not a silent change. The cheap analytic strip surrogate is a **FILTER, not a
   ranker**: the project's own data (`gentle_dish > deep_dish`) shows rectification is *non-monotonic* in cup
   depth, which a quasi-steady strip model would rank backwards — so use it only to reject the flat/symmetric
   region, keep real CFD to rank the cupped finalists. GPU-LBM (XLB/FluidX3D) is the natural fast high-fidelity
   tier that finally unblocks the PyFR gap.

3. **BO — sound optimizer, over-parameterized problem.** *(high)* 24 of 33 dims are the panel grid; the 12
   panel-*offset* dims are near-inert — the *radial* offsets are redundant with the ±20mm meridian, the
   *tangential* ones are fold-locked. → **collapse them to 1-2 tangential-shape scalars** at the containment
   max (not a blunt zero-cut; the optimizer kept pinning them to their bound). Does **not** foreclose the V1.5
   panel lever (that was always rib-thickness/containment). Also cheap: a √D-scaled GP lengthscale prior
   (retire the dead SAASBO stub). Real 2-fidelity MF-BO = ~30-50% more savings *conditional on* measuring
   coarse-vs-fine R²>0.75 first. Ax migration deletes ~980 lines of DIY async but is an *engineering* win, not
   an eval-count win. `bo/multi_fidelity.py` and `bo/saasbo.py` are empty stubs; `fit_saas_gp` is never wired.

4. **Infra — the substrate is the pain source, but shrinking beats migrating.** *(high)* ~70-80% of the
   reliability hell (stale reads, under-pooling, claim deadlocks) is inherent to Colab+Drive, not the code
   (which is near the ceiling Drive allows). **Cross-cutting insight both challengers reached:** the surrogate
   + dim-reduction shrink the next campaign to *tens* of real CFD runs — which **fits in one Colab session
   with a local ledger**, deleting the entire distributed failure class for free. So **don't migrate to Modal
   yet** (its cost is ~2× the naive estimate, ~$135-510; ROI weak on hardened code for a possibly-final
   campaign). If a campaign *stays* big, the cheap stopgap is **gcsfuse with `ttl-secs=0`** (keeps the Path
   code, strong consistency) — *not* the client-API rewrite. Migration is **contingent on the surrogate
   failing to shrink the campaign.**

5. **Approach / proportionality — over-sized for the aero lever, right-sized for the learning goal.** *(high)*
   An 800-eval 33-D BO over 3h CFD is ~10× more machinery than the *aero answer* (tip-loaded gentle cupping)
   needed, and `plan §8.2` predicted the physics was low-dimensional. **But** "just print a scoop" measures
   against the wrong objective: this is partly a learning project, and — decisively — a naive physical-first
   builder would have printed the *deepest* cup and landed on a *worse* fan (the true optimum is *gentle*, the
   non-obvious interior optimum the search actually found). The search earned its keep. The honest path is the
   middle one: keep the machinery, fix the real flaws, run it *small*, add physical validation.

## ⚠️ The most important NEW finding (all 5 domain agents missed it)
**V1's planned validation instrument measures the wrong quantity.** `physical/anemometer.py` computes
airflow from *unsigned speed magnitude* (~RMS/peak), which the data shows is ~equal across designs — while
the CFD optimizes the *signed cycle-mean* (net directed rectification). A plain/thermal anemometer would show
flat-vs-winner as *similar even though their net directed flows have opposite sign*. The **human A/B feel
test is the *aligned* validator** (you feel the directed puff). So validate V1 with the feel test + a
**directional** instrument — a **vane** anemometer read in AVG mode, or (cleanest, ~$0) a **kitchen scale +
light target** measuring net directed force — **not** an unsigned/thermal anemometer.

---

## V1 vs V2 mapping (the decision frame)
| Finding | Improves current V1 designs? | Where it belongs |
|---|---|---|
| Metric is validated (signed rectification) | — (confirms V1 is sound) | V1 reassurance; note in backlog |
| Validation instrument must be *directional* | **YES — the one V1 hook** | Do before the V1 feel test; V2_backlog Spike 0.3 |
| ≥5-cycle / `dt` convergence of the 2-7% residual | partial (fine-verify covers it) | confirm in V1 fine-verify |
| CFD regime → incompressible | no | **ADR (locked decision)** + V2 |
| Cheap surrogate as a *filter* | no | V2 (MFBO section) |
| Collapse panel-offset dims / √D prior / real MF-BO | no | V2 (design-space + MFBO sections) |
| Substrate (shrink first; Modal/one-box/gcsfuse) | no | V2, contingent on campaign staying big |

**None of the fixes yield a better V1 blade** — the panel dims were ≤1.4% authority (a re-run finds the same
optimum ~1-2% away), the metric was valid, and even the under-pooling only cost *efficiency*, not the final
result. → **Finish V1 as-is; the audit is V2's roadmap + one V1 validation correction.**

## Recommended sequence (each shrinks the next)
- **P0 (V1):** finish fine-verify → print flat baseline + winner(s) → blinded feel test + a **directional**
  airflow check (vane-AVG or scale+target). Fix the instrument *before* the feel test.
- **P1 (V2 prep, cheap):** collapse the 12 panel-offset dims → 1-2 tangential scalars; add the √D GP prior;
  one-off convergence check on the residual.
- **P2 (V2):** *if* another campaign is needed, run it *small* on one node (surrogate pre-filter → real CFD on
  finalists); re-scope infra only if it stays big; **(ADR-gated)** switch CFD regime compressible@1e-9 →
  incompressible.

## Don't churn (confirmed fine)
Geometry/mesh/render toolchain · qLogNEHVI + objective normalization · feasible-by-construction codec ·
coarse→fine + Kendall-τ *concept* · the **cycle-mean metric** (validated) · the now-hardened claim/heartbeat
code (near the substrate ceiling).

## What the adversarial pass changed vs. the raw research
- **Overturned:** "the campaign optimized noise" (it's real signed rectification); "switch to RMS/peak"
  (mean is correct).
- **Qualified:** the surrogate (filter-only, not ranker); the panel cut (collapse, don't zero); the infra
  migration (likely moot if you shrink first — sequence, don't parallelize).
- **New:** the validation-instrument mismatch none of the five saw; the added-mass-bias convergence hole;
  the ≥5-cycle margin caution.
