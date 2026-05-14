# Spike 0.7b — BO infrastructure scaling sanity check protocol

**Spec reference:** `docs/plan_R11.md §Phase 0 Spike 0.7b` (lines ~1855-1858).

**Depends on:** none. Pure-synthetic; no CFD, no physical bench, no Colab.

**Why this exists.** Phase 4 launches the full BO loop (architecture
bandit × TuRBO × multi-fidelity GP) over a 37-46-dimensional design
space. Before we commit Colab Pro hours to Phase 4, this spike confirms
the BO infrastructure runs end-to-end at production dimensionality under
a wall-clock budget compatible with the per-iteration acquisition step.
It runs on **synthetic** objective values so the gate is purely about
the BO stack, not the geometry/CFD bridge.

---

## Pass criteria (three gates, all must pass)

1. **GP fit-time gate.** Per-iteration GP fit wall-clock
   `≤ GP_FIT_TIME_GATE_S = 60 s` for every iteration. The lock is in
   `src/fanopt/bo/spike_0_7b.py`. If even one iteration exceeds the gate,
   the spike fails.
2. **Architecture-bandit promotion gate.** The bandit must promote
   exactly `K_PROMOTED_SANITY = 4` architectures from the synthetic
   candidate pool. K = 4 is **hard-coded for this synthetic sanity check
   only**; the production K is determined later from Phase 3's measured
   R² on the multi-fidelity bridge.
3. **TuRBO trust-region update gate.** Across the run's trust-region
   state log, there must be **at least one valid shrink** (failure-count
   increment paired with a length decrease) and **at least one valid
   grow** (success-count increment paired with a length increase).
   Verifies the TR state machine wiring is correct, not its full
   optimality.

The overall spike passes iff all three gates pass.

---

## Setup

The reference environment is `environment.yml`, which installs the
optional `[bo]` extras (`torch`, `botorch`, `gpytorch`). Steps:

```bash
conda env create -f environment.yml
conda activate fanopt
pip install -e ".[bo]"
```

The script supports a **numpy-only fallback** GP (`--gp-backend numpy`)
for sandboxes without BoTorch. The fallback is a faithful exact GP (RBF
kernel + Cholesky solve, length-scale grid search). It is in the same
complexity class as BoTorch's `SingleTaskGP` on CPU and so produces a
fair wall-clock comparison for the gate, **but production verification
must use the real BoTorch backend** — that's the stack Phase 4 will run.

---

## Run

Canonical sanity run (mid-range dimensionality, n_lhs in spec range):

```bash
python scripts/run_spike_0_7b.py --d 40 --n-lhs 8 --seed 42
```

Tighter / faster (still inside spec ranges where it matters):

```bash
python scripts/run_spike_0_7b.py --d 37 --n-lhs 5 --seed 42 --n-iters 5
```

Forced BoTorch / numpy backend selection:

```bash
python scripts/run_spike_0_7b.py --d 40 --n-lhs 8 --gp-backend botorch
python scripts/run_spike_0_7b.py --d 40 --n-lhs 8 --gp-backend numpy
```

Output:

* `data/spike_0_7b/results.json` — per-iteration GP timings, TR log,
  bandit records, and the three gates.
* stdout — a 4-line pass/fail table summarising the gates.
* Exit code — 0 iff all three gates pass; 1 otherwise.

---

## Interpreting `results.json`

| Key | What it locks |
|---|---|
| `inputs.d` | Design-space dimensionality (37-46 lock). |
| `inputs.n_lhs` | Initial LHS sample count (5-10 lock). |
| `inputs.gp_fit_time_gate_s` | 60-s GP gate (locked in `spike_0_7b.py`). |
| `gp_fit_timings[*].wall_time_s` | Per-iteration GP fit wall-clock. |
| `gp_fit_timings[*].passed` | True iff wall-clock ≤ 60 s. |
| `gates.all_gp_fits_under_60s` | Gate 1. |
| `gates.k_promoted` / `k_promoted_passes` | Gate 2 (= 4 for synthetic). |
| `turbo_trs[*]` | TR length / success / failure counts per iter. |
| `gates.turbo_trs_update_correctly` | Gate 3 (shrink + grow both observed). |
| `passed` | AND of all three gates. |

---

## Fallback decisions if the spike fails

These map back to the spec's stated fallbacks (line 1858):

* **GP gate fails (fit time > 60 s consistently).**
  * Tighten the architecture-bandit screening budget: more time per
    architecture, fewer architectures promoted. The K_promoted gate is
    deliberately hard-coded for the synthetic sanity check, so this
    knob is the production-K decision in Phase 3.
  * Switch from TuRBO to **SAASBO with ≤ 500 inducing points** —
    slower per iteration but more reliable at high D. Scaffold is in
    `src/fanopt/bo/saasbo.py`.
  * Or shrink the design space by fixing categoricals upfront. Two
    concrete options:
    * Freeze Layer 2 activation profile to a 2-field combination like
      `{louver, TPMS}`, freeing 3 binary activation dims.
    * Pin Layer 3 primitive presence to 1, freeing 1 binary.
* **K-promoted gate fails (synthetic).** Implementation bug in the
  bandit's selection wiring. Investigate `architecture_bandit.py` before
  blaming the GP.
* **TR-update gate fails.** Implementation bug in TuRBO's success /
  failure accounting in `turbo.py`. Verify the success / failure
  thresholds and the length-scaling factors against the standard TuRBO
  paper.

---

## Reproducibility

* Default seed = 42 (CLI flag `--seed`).
* The synthetic objective is deterministic at `noise_std = 0` (its
  default). All randomness in the run comes from LHS / bandit / TR
  jitter, all of which key off `--seed`.
* Two runs with the same `--seed`, `--d`, `--n-lhs`, `--n-iters`, and
  `--gp-backend` produce byte-identical `results.json` modulo
  `wall_time_s` (which is wall-clock-noise-bounded but should fall in a
  narrow band on the same machine).
