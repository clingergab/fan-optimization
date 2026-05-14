# Spike 0.7a — Generative-geometry sanity check (phase log)

**Spec reference:** `docs/plan_R11.md §Phase 0 Spike 0.7` (sub-spike 0.7a).
**Protocol:** `docs/spike_0_7a_protocol.md`.

**Question:** Does the CadQuery 4-layer generator + §N7 manufacturability
filter combination block every adversarial parameter set targeting (a) the
click-feature footprint and (b) the rib material from the Phase 2 SIMP TO
output, while accepting a healthy fraction of random parameter draws from
the JSON-schema bounds?

**Artifacts shipped with this spike:**
- `src/fanopt/geometry/spike_0_7a.py` — library (param sampling, records,
  analyze gate, adversarial sets).
- `scripts/run_spike_0_7a.py` — CLI runner with bbox-shim pipeline.
- `tests/test_geometry/test_spike_0_7a.py` — library tests.
- `tests/test_geometry/test_click_feature_preservation.py` — click-footprint
  regression.
- `tests/test_geometry/test_panel_mask.py` — §9.7.1 panel-domain regression.
- `tests/test_scripts/test_run_spike_0_7a.py` — CLI smoke tests.
- `docs/spike_0_7a_protocol.md` — operator procedure.

**Pass criterion:**
1. ALL 3 adversarial sets blocked (by manufacturability filter, click check,
   or rib check — any of the four pipeline stages can be the blocker).
2. Random-set acceptance: (≥ 7 / 10 random sets pass) OR (≥ 3 random sets
   fail with non-empty `rejection_reasons`).

**Implementation status (pre-Phase-1):** the §9.7 generator + §N7 filter are
Phase 0 / Phase 1 scaffolds. Spike 0.7a runs against a **bounding-box shim**
in `scripts/run_spike_0_7a.py` (documented in detail in the protocol). The
shim catches the failure modes the adversarial sets target, but the spike's
Step 2 (visual STL inspection) and Step 4 (printed-part click engagement
check) are documented for the post-Phase-1 follow-up run — they cannot be
exercised with the current shim because no STL is emitted.

---

## Run log

| Field | Value |
|---|---|
| Date | _to be filled_ |
| Operator | _to be filled_ |
| Seed | `42` (CLI default) |
| `n_random` | `10` |
| `n_adversarial` | `3` |
| `n_passing` (random subset) | _to be filled_ |
| `adversarial_blocked_count` | _to be filled_ |
| `passed` | _true / false_ |
| `results.json` path | `data/spike_0_7a/results.json` |
| Printed designs (Step 4) | _hash1, hash2_ |
| Print 1 result | _clean / weak / none — click engagement_ |
| Print 2 result | _clean / weak / none — click engagement_ |

CLI invocation:

```
python scripts/run_spike_0_7a.py --n-random 10 --seed 42
```

---

## Diagnostics if either gate fails

| Symptom | Likely cause | Fix |
|---|---|---|
| Adversarial `a_louver` slipped click check | Louver bbox heuristic too lax | Lower `louver_cluster_tip` upper bound; widen click-footprint bbox in shim |
| Adversarial `b_tpms` slipped rib check | TPMS bbox shim ignores rotation | Replace bbox with rotated-bbox or signed-distance check |
| Adversarial `c_primitive` slipped manuf | 5 mm clearance shim has off-by-1 | Verify `CLICK_CLEARANCE_M` in `run_spike_0_7a.py` matches the lock |
| Random pass rate < 7 / 10 AND < 3 documented failures | Schema bounds too loose / filter too strict | Tighten schema OR relax filter rule; re-run |
| Random pass rate << 7 / 10 (e.g., 2 / 10) | Schema bound interaction (joint constraint) | Add joint constraint to schema, e.g., `prim_active ∧ louver_active → prim_y_m * louver_angle_rad < 0` |

---

## Findings (post-run)

> _What worked, what surprised you, anything that should propagate to the
> protocol doc, the analyzer, or the §9.7 generator / §N7 manufacturability
> filter design._

---

## Sign-off

- [ ] CLI ran end-to-end at the locked seed (42) and produced
      `data/spike_0_7a/results.json`.
- [ ] All 3 adversarial sets blocked.
- [ ] Random-set gate met (≥ 7 / 10 pass OR ≥ 3 documented fails).
- [ ] `tests/test_geometry/test_spike_0_7a.py` green.
- [ ] `tests/test_geometry/test_click_feature_preservation.py` green.
- [ ] `tests/test_geometry/test_panel_mask.py` green.
- [ ] `tests/test_scripts/test_run_spike_0_7a.py` green.
- [ ] (Post-Phase 1) Visual STL inspection of 10 random designs complete.
- [ ] (Post-Phase 1) Manual manufacturability filter cross-check complete.
- [ ] (Post-Phase 1) 2 designs printed; both print clean and click engages.
- [ ] (Post-Phase 1) `BBOX_SHIM = False` in both companion regression tests;
      bit-for-bit voxel comparator wired in.
- [ ] This log committed to `docs/phase_logs/`.
- [ ] Spike 0.7a closed in `docs/phase_checklist.md`.
