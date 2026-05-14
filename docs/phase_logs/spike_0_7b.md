# Spike 0.7b — BO infrastructure scaling sanity check

**Spec reference:** `docs/plan_R11.md §Phase 0 Spike 0.7b` (lines ~1855-1858).

**Question:** Does the BO infrastructure (architecture bandit × TuRBO ×
multi-fidelity GP) run end-to-end at the spec's 37-46 D on a synthetic
objective inside the 60-s-per-iteration wall-clock budget, with the
synthetic-objective K = 4 promotion gate and TuRBO TR shrink/grow
behaviour?

**Procedure:** `docs/spike_0_7b_protocol.md`.

**Artifacts shipped with this spike:**

* `docs/spike_0_7b_protocol.md` — operator procedure
* `src/fanopt/bo/spike_0_7b.py` — pass-criteria library
* `scripts/run_spike_0_7b.py` — CLI runner (BoTorch + numpy-fallback GP)
* `tests/test_bo/test_spike_0_7b.py` — unit gates for library
* `tests/test_scripts/test_run_spike_0_7b.py` — CLI smoke test
* `data/spike_0_7b/results.json` — populated by the run

**Pass criteria (all three must hold):**

1. Every per-iteration GP fit wall-clock ≤ 60 s
   (`gates.all_gp_fits_under_60s`).
2. Architecture bandit promotes K = 4 architectures on the synthetic
   objective (`gates.k_promoted_passes`; K = 4 is hard-coded **only**
   for the synthetic sanity check — production K comes from Phase 3's
   measured R²).
3. TuRBO trust regions both shrink (after a failure-count increment)
   and grow (after a success-threshold crossing)
   (`gates.turbo_trs_update_correctly`).

---

## Run log

| Field | Value |
|---|---|
| Date | _to be filled_ |
| Operator | _to be filled_ |
| Host / hardware | _e.g., M2 Max, 32 GB RAM_ |
| Conda env | `environment.yml @ commit ____` |
| BoTorch installed | _yes / no_ |
| `--d` | _37-46_ |
| `--n-lhs` | _5-10_ |
| `--n-iters` | _≥ 3 (default 10)_ |
| `--seed` | _42 (default) or override_ |
| `--gp-backend` | _auto / botorch / numpy_ |
| Resolved backend | _botorch_singletask / numpy_rbf_ |
| Max GP fit time | _seconds_ |
| Mean GP fit time | _seconds_ |
| `K_promoted` | _integer; expect 4_ |
| TR shrink observed | _yes / no_ |
| TR grow observed | _yes / no_ |
| Overall `passed` | _true / false_ |
| `results.json` path | _e.g., `data/spike_0_7b/results.json`_ |
| Notes | _any backend fallback, warning lines, etc._ |

## Decision

* If `passed = true` and the backend resolved to `botorch_singletask`:
  Phase 4 is cleared for launch on this hardware. Move on to Phase 1.
* If `passed = true` but the backend resolved to `numpy_rbf`: re-run on
  a host with the `[bo]` extras installed before clearing Phase 4. The
  numpy fallback is a fair *lower bound* but is not the production
  stack.
* If `passed = false`: apply the fallback decisions in
  `docs/spike_0_7b_protocol.md §Fallback decisions if the spike fails`
  and re-run.

---

## Round-table review

_Empty until adversarial review opens against this spike's results._
