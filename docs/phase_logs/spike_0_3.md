# Spike 0.3 — Baseline physical measurement

**Spec reference:** `docs/plan_R11.md §Phase 0 Spike 0.3` + L8 lock.

**Depends on:** Spike 0.2 closed (I_wrist measured, `data/spike_0_2/results.json`
written with `passed: true`).

**Procedure:** `docs/spike_0_3_protocol.md`.

**Artifacts shipped with this spike:**
- `docs/spike_0_3_protocol.md` — operator procedure
- `src/fanopt/physical/imu.py` — W_cycle, IMU kinematics extraction
- `src/fanopt/physical/anemometer.py` — 9-point grid J_fan_proxy
- `scripts/run_spike_0_3_baseline.py` — CLI runner
- `tests/test_physical/test_imu_known_waveform.py` — analytic SHM check
- `tests/test_physical/test_anemometer.py` — uniform-flow integration check

**Pass criterion:** the spike produces a defensible `J_fan_proxy / W_cycle`
number for the 10-blade flat-panel baseline that all Phase-6 optimized
designs target ≥15% improvement against.

---

## Run log

| Field | Value |
|---|---|
| Date | _to be filled_ |
| Operator | _to be filled_ |
| Baseline JSON | _path / commit hash_ |
| I_wrist (kg·m², from Spike 0.2) | _to be filled_ |
| Generator `I_wrist_kgm2` | _to be filled_ |
| IMU device | _e.g., phone Sensor Logger / strap-down gyro_ |
| Anemometer device | _make/model_ |
| Metronome rate (Hz) | 2.0 |
| Anemometer points | 9 (3×3 grid at 600×600 mm @ 300 mm; 200 mm pitch) |
| IMU trials | 5 × 10 cycles each |

### Anemometer 9-point grid (mean velocity per point)

| Point | x (m) | y (m) | v_mean (m/s) | v_peak (m/s) | notes |
|---|---|---|---|---|---|
| p1 | −0.2 | −0.2 | _ | _ | |
| p2 |  0.0 | −0.2 | _ | _ | |
| p3 |  0.2 | −0.2 | _ | _ | |
| p4 | −0.2 |  0.0 | _ | _ | |
| p5 |  0.0 |  0.0 | _ | _ | |
| p6 |  0.2 |  0.0 | _ | _ | |
| p7 | −0.2 |  0.2 | _ | _ | |
| p8 |  0.0 |  0.2 | _ | _ | |
| p9 |  0.2 |  0.2 | _ | _ | |

### IMU kinematics sanity (from runner output)

| Quantity | Target | Measured | OK? |
|---|---|---|---|
| f_wave (Hz) | 2.00 ±10% | _ | _ |
| ω_max (rad/s) | 8.8 ±15% | _ | _ |
| θ_max (rad) | 0.70 ±20% | _ | _ |

### Published baseline

| Field | Value |
|---|---|
| `J_fan_proxy_N` (mean) | _ |
| `J_fan_proxy_peak_N` | _ |
| `W_cycle_J` (mean of 5 trials) | _ |
| `W_cycle_J` std | _ |
| **`J_per_W`** (canonical) | _ |
| trial-to-trial W_cycle std/mean | _ (sanity: < 20%) |

Runner invocation:

```
python scripts/run_spike_0_3_baseline.py \
  --imu data/spike_0_3/imu_trial{1..5}.csv \
  --anemometer data/spike_0_3/anemometer_grid.csv \
  --inertia data/spike_0_2/results.json \
  --out data/spike_0_3/baseline.json
```

---

## Findings (post-run)

> _Surprises, anything that should propagate to the protocol or the runner._

---

## Sign-off

- [ ] Baseline 10-blade fan printed; build-quality eyeball OK.
- [ ] Spike 0.2 I_wrist applies to this exact assembled fan.
- [ ] 5 IMU trials recorded; kinematics within sanity bands (or re-shot).
- [ ] 9-point anemometer grid recorded.
- [ ] `data/spike_0_3/baseline.json` committed (or pinned to Drive).
- [ ] Published `J_per_W` number is the canonical Phase-6 baseline.
- [ ] This log committed to `docs/phase_logs/`.
- [ ] Spike 0.3 closed in `docs/phase_checklist.md`.
