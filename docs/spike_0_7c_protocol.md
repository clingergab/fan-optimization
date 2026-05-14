# Spike 0.7c — Sobol-vs-BO iso-compute comparison protocol

**Spec reference:** `docs/plan_R11.md §Phase 0 Spike 0.7c` (lines ~1859-1867).

**Depends on:** Spike 0.7b (BO infrastructure scaling sanity check) must
have passed — Spike 0.7c reuses that BO stack against a Sobol baseline.

**Why this exists.** Before Phase 4 commits 600-1100 h of CFD compute to
a BO inner loop, this spike confirms BO actually beats a uniform-random
Sobol baseline on the same parameter box under matched compute budgets.
If BO cannot beat Sobol by ≥ 5% on at least 2 of 3 fixed budgets, the
upstream GP / acquisition is the wrong tool for this design space — fall
back to SAASBO or to a smaller (collapsed) architecture set per the
spec's fallback rules below.

---

## Budget-allocation lock (H7)

The 30 + 100 + 300 = **430 h** of cumulative compute consumed by this
spike is **booked under Phase 0**, NOT against the 1000-h Phase 4
stop-rule budget (§6.2.3). The Phase 4 counter starts at the
`phase4-launch` git tag created by `scripts/launch_phase4.py`. Without
this split, the 430 h of Sobol-seed + BO-iteration evaluations would
consume 43 % of the Phase 4 1000-h budget before Phase 4 launches —
leaving only ~570 h for the BO inner loop, incompatible with the
600-1100 h expected range.

**The Sobol seed runs double as Phase 4 GP initialisation.** The 50
records at `gdrive/fan-optimization/phase0/sobol_seed/results.jsonl`
are not re-run inside Phase 4: the Phase 4 launch script reads them as
the initial training set. So even if BO does win this spike cleanly,
the day's compute is not wasted.

---

## Budget accounting — locked rules

1. **"Hours" = cumulative tier-(-1) + tier-0 + tier-1 compute including
   Sobol seed runs.** No "free" Sobol allowance. The §6.2.3 1000-h stop
   rule uses the same accounting; Spike 0.7c's 430 h is fully booked
   against the campaign budget under Phase 0.
2. **Budgets run serially with results gating next.** B=30 must
   complete before B=100 starts (so the 100-h GP uses the 30-h seed
   data); B=100 must complete before B=300 starts. This makes the
   GP-fit time monotonically realistic — at B=30 the GP has only seed
   data; at B=300 it has full multi-fidelity training.
3. **Wall-clock vs cumulative-compute.** Colab Pro CPU runs 2-4
   parallel sessions, so 100 h of cumulative compute is ~25-50 h of
   wall-clock. The comparison axis is **cumulative compute** (matches
   the §6.2.3 stop rule), not wall-clock.

---

## Pass criterion

Given equal CFD budgets `B ∈ {30, 100, 300} h`, BO's best `J_fan`
exceeds Sobol's best `J_fan` by **≥ 5 % on at least 2 of the 3
budgets** (`BO_MINUS_SOBOL_PCT_GATE = 5.0`,
`BUDGETS_PASS_THRESHOLD = 2`). The constants are locked in
`src/fanopt/bo/spike_0_7c.py`.

---

## Three-step serial procedure

### Step 1 — Sobol seed generation (B = 30 h tranche)

Generates 50 Sobol samples at tier -1 and evaluates them. These records
also serve as the Phase 4 GP initialisation set (see the H7 reuse-note
above).

```bash
python scripts/run_spike_0_7c_seed.py \
    --n 50 \
    --tier -1 \
    --out gdrive/fan-optimization/phase0/sobol_seed/results.jsonl
```

Add `--seed 42` for reproducibility; add `--d 40` to override the
default dimensionality. The stub script generates synthetic samples
when no CFD runner is wired in — replace the `_evaluate` hook with a
call to the production CFD bridge before running this for real.

### Step 2 — BO production run (B = 100 h tranche, then B = 300 h tranche)

Runs 100 BO iterations on the same architecture set as Phase 4 using
the production GP + qMFKG configuration. The BO loop must be seeded
with the 50 Sobol records from Step 1 (matches the §H7 reuse and the
serial-budget locked rule).

The BO outer loop (architecture bandit × TuRBO × multi-fidelity GP)
lives in `src/fanopt/bo/orchestration.py`. The Spike-0.7c protocol does
not re-implement that loop — it consumes the JSONL ledger that the
production runner writes.

The BO records ledger conventionally lives at
`gdrive/fan-optimization/phase0/spike_0_7c/bo_results.jsonl`.

### Step 3 — Comparison

```bash
python scripts/run_spike_0_7c.py \
    --sobol-results gdrive/fan-optimization/phase0/sobol_seed/results.jsonl \
    --bo-results   gdrive/fan-optimization/phase0/spike_0_7c/bo_results.jsonl \
    --budgets 30,100,300 \
    --out data/spike_0_7c/results.json
```

Output:

* `data/spike_0_7c/results.json` — per-budget table + overall pass /
  fail + (if failed) a fallback recommendation.
* stdout — a one-row-per-budget comparison table.
* Exit code — 0 on PASS, 1 on FAIL, 2 on input error.

If the BO run encountered ≥ 1 iteration where the GP fit exceeded the
60-s gate from Spike 0.7b, pass the sub-axis label via
`--gp-fit-time-above-60s-on`:

```bash
python scripts/run_spike_0_7c.py ... --gp-fit-time-above-60s-on high_d
# or
python scripts/run_spike_0_7c.py ... --gp-fit-time-above-60s-on wide_architecture_set
```

This routes the fallback recommendation correctly when the spike fails.

---

## Smoke test (no CFD, no Colab)

The 430 h experiment is not runnable in sandboxes. To exercise the
harness end-to-end on a synthetic objective in seconds:

```bash
python scripts/run_spike_0_7c_smoke.py
```

Generates 50 synthetic Sobol records + 100 synthetic BO records on the
same synthetic objective the BO infrastructure uses
(`fanopt.bo.spike_0_7b.synthetic_objective`), writes both ledgers to a
tempdir, and runs the comparison. Asserts PASS; returns 0 on PASS.

---

## Fallback decision tree if the spike fails

Per spec (line 1865-1866): "if BO doesn't beat Sobol on 2 of 3 fixed
budgets, fall back to SAASBO inner-loop (with the ≤ 500-inducing-point
cap from §6.2.2) OR fix the architecture set (reduce dimensionality by
collapsing Layer 2 categoricals)". The choice depends on which sub-axis
the GP fit time exceeded 60 s on.

| Diagnosis | Recommendation | Mechanism |
|---|---|---|
| GP fit time > 60 s due to high D (n_train large, exact-GP factorisation slow) | **SAASBO ≤ 500 inducing points** | `src/fanopt/bo/saasbo.py`; switch the inner loop from TuRBO to SAASBO and cap inducing points per §6.2.2. |
| GP fit time > 60 s due to wide architecture set (categorical blowup) | **Collapse Layer 2 categoricals** | Pin Layer 2 activation profile to a single combination (e.g., `{louver, TPMS}`); re-run with the reduced set. |
| BO under-exploited without hitting the 60-s GP gate | **Re-tune TuRBO / qMFKG hyperparameters** | Diagnose with `--gp-fit-time-above-60s-on` empty; re-tune `trust_region_init_length`, the qMFKG cost-aware multiplier, and the multi-fidelity-bridge calibration before reaching for the bigger fallbacks. |

The CLI emits its recommendation as
`results.json::fallback_recommendation`, one of:

* `"saasbo"`
* `"fix_architecture_set"`
* `"retune_acquisition"`

---

## Reuse note

Regardless of pass / fail, the 50 Sobol seed records at
`gdrive/fan-optimization/phase0/sobol_seed/results.jsonl` are reused as
the Phase 4 GP initialisation set. The `phase4-launch` script reads
that ledger and does not re-run the Sobol seed evaluations.

---

## Reproducibility

* Default seed for the synthetic smoke runner = 42 (`--seed` flag).
* The synthetic objective and the synthetic BO descent are deterministic
  given a seed — two smoke runs with identical CLI args produce
  byte-identical JSONL ledgers and `results.json` (modulo
  filesystem-tempdir noise).
* The production Sobol-seed runner derives its sample set from
  `scipy.stats.qmc.Sobol(d=d, scramble=True, seed=seed)`. With seed
  fixed, the sample set is reproducible — the per-evaluation J_fan is
  bounded only by CFD-run determinism (mesh / solver convergence
  tolerances; see Phase 3's repeatability bound).
