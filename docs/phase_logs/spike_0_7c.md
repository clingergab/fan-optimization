# Spike 0.7c — Sobol-vs-BO iso-compute comparison

**Spec reference:** `docs/plan_R11.md §Phase 0 Spike 0.7c` (lines ~1859-1867).

**Question:** Does the production GP + qMFKG BO inner loop beat a
uniform-random Sobol baseline by at least 5 % on at least 2 of the 3
fixed CFD-hour budgets {30, 100, 300} h?

**Procedure:** `docs/spike_0_7c_protocol.md`.

**Artifacts shipped with this spike:**

* `docs/spike_0_7c_protocol.md` — operator procedure (three-step serial
  run, fallback decision tree, reuse-note).
* `src/fanopt/bo/spike_0_7c.py` — iso-compute pass-criteria library
  (`compute_iso_compute_point`, `analyze_spike_07c`, `record_to_jsonl`).
* `scripts/run_spike_0_7c.py` — comparison CLI (consumes Sobol + BO
  JSONL ledgers, writes per-budget results.json).
* `scripts/run_spike_0_7c_seed.py` — Sobol-seed generator stub (50
  samples at tier -1; doubles as Phase 4 GP init per H7).
* `scripts/run_spike_0_7c_smoke.py` — synthetic-objective end-to-end
  smoke runner.
* `tests/test_bo/test_spike_0_7c.py` — unit tests for the library.
* `tests/test_scripts/test_run_spike_0_7c.py` — CLI smoke + explicit
  FAIL tests.
* `data/spike_0_7c/results.json` — populated by the run.

**Pass criterion.** `BO_MINUS_SOBOL_PCT_GATE = 5.0` % on best-J_fan,
on at least `BUDGETS_PASS_THRESHOLD = 2` of the
`BUDGETS_HOURS = (30, 100, 300)` budgets summing to
`BUDGETS_TOTAL_HOURS = 430` h, booked under Phase 0 per the H7
budget-allocation lock.

---

## Run log

| Field | Value |
|---|---|
| Date | _to be filled_ |
| Operator | _to be filled_ |
| Host / hardware | _e.g., Colab Pro G4 GPU, 95 GB RAM_ |
| Conda env | `environment.yml @ commit ____` |
| Sobol seed runner version | `scripts/run_spike_0_7c_seed.py @ commit ____` |
| BO runner version | `src/fanopt/bo/orchestration.py @ commit ____` |
| `--n` (Sobol samples) | _50 (spec lock)_ |
| `--d` | _37-46_ |
| `--tier` | _-1 (spec lock)_ |
| BO iterations | _100 (spec lock)_ |
| Cumulative compute B=30 budget | _hours_ |
| Cumulative compute B=100 budget | _hours_ |
| Cumulative compute B=300 budget | _hours_ |
| `sobol_best_j_fan` @ B=30 | _value_ |
| `bo_best_j_fan` @ B=30 | _value_ |
| `bo_minus_sobol_pct` @ B=30 | _percent_ |
| `bo_beats` @ B=30 | _yes / no_ |
| `sobol_best_j_fan` @ B=100 | _value_ |
| `bo_best_j_fan` @ B=100 | _value_ |
| `bo_minus_sobol_pct` @ B=100 | _percent_ |
| `bo_beats` @ B=100 | _yes / no_ |
| `sobol_best_j_fan` @ B=300 | _value_ |
| `bo_best_j_fan` @ B=300 | _value_ |
| `bo_minus_sobol_pct` @ B=300 | _percent_ |
| `bo_beats` @ B=300 | _yes / no_ |
| `n_budgets_bo_beats` | _0-3_ |
| Overall `passed` | _true / false_ |
| Fallback recommendation | _none / saasbo / fix_architecture_set / retune_acquisition_ |
| `results.json` path | _e.g., `data/spike_0_7c/results.json`_ |
| GP fit time > 60 s on (sub-axes) | _empty / high_d / wide_architecture_set_ |
| Sobol ledger reused for Phase 4 GP init | _yes (per H7 lock)_ |
| Notes | _any timing anomalies, fallback events, etc._ |

## Decision

* If `passed = true`: BO infrastructure is cleared on this design space.
  Move on to Phase 4 launch (`scripts/launch_phase4.py`). The 50 Sobol
  seed records are reused as the Phase 4 GP initialisation set.
* If `passed = false`: apply the fallback in
  `results.json::fallback_recommendation`:
  * `saasbo` — switch to SAASBO with ≤ 500 inducing points
    (`src/fanopt/bo/saasbo.py`) and re-run Spike 0.7c.
  * `fix_architecture_set` — collapse Layer 2 categoricals (e.g., pin
    activation profile to one combination) and re-run Spike 0.7c.
  * `retune_acquisition` — re-tune TuRBO / qMFKG hyperparameters and
    re-run Spike 0.7c.
* In all branches, the Sobol seed ledger is preserved as the Phase 4
  GP init set.

---

## Round-table review

_Empty until adversarial review opens against this spike's results._
