# Spike 0.3 — Baseline physical measurement protocol

> **🚧 V1 STATUS: DEFERRED TO V2.** This protocol's anemometer + IMU bench
> measurements are deferred. V1 uses sim-side baselines (Phase 2a CFD) plus
> a Phase 6 qualitative blinded A/B feel test. See
> `docs/phase_logs/phase_0_signoff.md` for the decision record and
> `data/spike_0_3/deferral.json` for the sentinel. **Appendix A below is
> the recommended V2 cheap upgrade if you want a quantitative baseline
> later without buying specialized hardware.** The body of this protocol
> documents the originally-specified anemometer rig as the eventual full-
> rigor V2 path.

**Spec reference:** `docs/plan_R11.md §Phase 0 Spike 0.3` (lines ~1798–1801),
**L8 lock** at line ~2392 (9-point anemometer grid), and §6.4 (I_wrist
denominator).

**Depends on:** Spike 0.2 (`docs/spike_0_2_protocol.md`) must be closed —
the IMU-normalized comparison needs a measured I_wrist for the same fan.

**Why this exists.** Every Pareto-optimized design that emerges from Phase
4–6 must beat *something*. The canonical comparison metric is
`J_fan_measured / W_cycle`, evaluated on the simplest defensible reference
geometry — a 10-blade fan with **flat panels, no TO, no airfoil camber**.
Whatever number this spike publishes becomes the divisor that Phase 6 step
79 must clear by ≥15%.

**What gets measured.**

| Quantity | How | Reported as |
|---|---|---|
| `J_fan_proxy` (N) | Anemometer 9-point grid × plane area | Bench-anemometer plane integral at 300 mm |
| `W_cycle` (J) | `∫\|I_wrist · ω · (dω/dt)\| dt` over one cycle | Rectified inertial power per cycle |
| **`J_fan_proxy / W_cycle`** | the ratio | **The canonical baseline — published number** |

The rectified-power integration (`∫\|P(t)\| dt`) is intentional — for ideal
periodic SHM the signed integral is identically zero (energy oscillates
between speeds). The operator's muscle, however, dissipates both
positive-power and negative-power phases, so the absolute integral is what
"work the wrist actually does per cycle" means physically.

---

## Apparatus

1. **Baseline fan** — 10 blades, flat panels, no TO, no airfoil camber.
   The geometry must come out of the §9.7 generator at its default JSON
   (`generator_baseline.json`) so the same `I_wrist_kgm2` cross-check from
   Spike 0.2 applies. Print all 10 in PETG; assemble on the steel/brass
   pivot pin.
2. **Spike 0.2 inertia rig** (in calibrated state — re-measure I_wrist on
   *this* assembled baseline if there's any ambiguity about which copy).
3. **IMU on the handle** — strapdown gyro or phone IMU streaming
   `(t, theta_rad, omega_rad_per_s)` to CSV. Sampling ≥ 100 Hz. The IMU
   axis must align with the **+y wrist axis** (same axis as I_wrist) —
   otherwise the rotation it records is not the rotation that lives in the
   W_cycle integral.
4. **Anemometer** — handheld vane or hotwire anemometer. Must read
   m/s on a 1 Hz or faster cadence; record min, max, and time-average
   over each measurement window.
5. **Anemometer 9-point jig** — a board or cardboard reticle that lets you
   place the anemometer probe at 9 specific points on a 600 × 600 mm plane,
   at 300 mm in +z from the pivot center. The 9 points are a 3×3 grid with
   200 mm pitch, centered on the +z axis through the pivot (§9.4
   integration plane, L8 lock).
6. **Metronome** at 2 Hz (any phone metronome). The waving cadence must be
   tight to 2 Hz or the IMU-derived ω_max won't match the locked spec
   numbers (V_tip = 2.20 m/s, α_max = 110 rad/s²) and the optimized
   designs being compared against this baseline will be apples-to-oranges.

---

## Step 1 — Pre-measurement sanity

Before the operator picks up the fan:

1. Confirm Spike 0.2 closed: `data/spike_0_2/results.json` exists with
   `passed: true`. Note the published `I_wrist_kgm2`.
2. Run `smoke_test.py` on the baseline geometry; record
   `I_wrist_kgm2_generator` from its output. (Once Phase 1 lands.)
3. The 9-point grid is at `(±0.2, 0, 0.3), (0, ±0.2, 0.3), (±0.2, ±0.2, 0.3)`
   relative to the **pivot center** (NOT the wrist origin — the J_fan plane
   in §9.4 is centered on the fan's pivot, where the broad face's wake is).
   Mark the 9 points on the reticle and label them p1–p9 in a consistent
   raster order (e.g., row-major from −x,−y corner). Same labels go into
   the CSV.

---

## Step 2 — IMU recording

For each of 5 trials:

1. Start the metronome at 2 Hz. Wait 2–3 beats to settle.
2. Start IMU recording (`*.csv` per trial).
3. Wave the fan for **10 full cycles** at metronome cadence with the
   intended productive-stroke direction (+z toward the user; see §3.2.0).
4. Stop IMU recording.
5. Spot-check the CSV: ω_max should be in the 8–10 rad/s ballpark (locked
   spec: 8.8 rad/s). If you're hitting 5 rad/s your amplitude is too small;
   if you're hitting 14 rad/s your cadence drifted above 2 Hz.

Recorded CSV format (`data/spike_0_3/imu_trial<N>.csv`):

```
t_s,theta_rad,omega_rad_per_s
0.000,0.0000,0.000
0.010,0.0008,0.158
...
```

The analyzer can also synthesize `theta` from `omega` (or vice versa) via
numerical integration / differentiation — only one of the two columns is
strictly required, but recording both lets the analyzer cross-check.

---

## Step 3 — Anemometer 9-point grid

For each of the 9 grid points `p1..p9`:

1. Position the anemometer probe at the grid point.
2. Start the metronome at 2 Hz. Have the operator wave for **10 full
   cycles** at the same cadence as Step 2.
3. Record the anemometer's reading. For the published baseline use the
   **time-average velocity over the 10 cycles** at that point. (If the
   anemometer also reports peak: record that too — peak feeds the
   "asymmetric drag" diagnostic, mean feeds the canonical J_fan_proxy.)

Recorded CSV format (`data/spike_0_3/anemometer_grid.csv`):

```
point,x_m,y_m,z_m,v_mean_m_per_s,v_peak_m_per_s,notes
p1,-0.2,-0.2,0.3,0.18,0.41,
p2, 0.0,-0.2,0.3,0.27,0.62,
p3, 0.2,-0.2,0.3,0.21,0.48,
p4,-0.2, 0.0,0.3,0.25,0.55,
p5, 0.0, 0.0,0.3,0.42,0.93,
p6, 0.2, 0.0,0.3,0.26,0.57,
p7,-0.2, 0.2,0.3,0.19,0.43,
p8, 0.0, 0.2,0.3,0.28,0.65,
p9, 0.2, 0.2,0.3,0.20,0.47,
```

(45 min of bench time per L8: 9 points × 10 cycles × 30 s ≈ 45 min.)

---

## Step 4 — Compute J_fan_proxy and W_cycle

Run the baseline runner:

```
python scripts/run_spike_0_3_baseline.py \
  --imu data/spike_0_3/imu_trial1.csv \
       data/spike_0_3/imu_trial2.csv \
       data/spike_0_3/imu_trial3.csv \
       data/spike_0_3/imu_trial4.csv \
       data/spike_0_3/imu_trial5.csv \
  --anemometer data/spike_0_3/anemometer_grid.csv \
  --inertia data/spike_0_2/results.json \
  --out data/spike_0_3/baseline.json
```

The runner reports:

| Field | Meaning |
|---|---|
| `J_fan_proxy_N` | ρ · ⟨v⟩ · A_plane (mean) — coarse plane integral |
| `J_fan_proxy_peak_N` | ρ · ⟨v_peak⟩ · A_plane — peak-velocity-based proxy |
| `W_cycle_J` | mean rectified-power integral across 5 trials |
| `W_cycle_per_trial_J` | array of per-trial W_cycle |
| `J_per_W` | `J_fan_proxy / W_cycle` — the canonical baseline |
| `f_wave_Hz` | IMU-derived dominant frequency (sanity: ≈ 2 Hz) |
| `omega_max_rad_per_s` | IMU-derived peak ω (sanity: ≈ 8.8 rad/s) |
| `theta_max_rad` | IMU-derived amplitude (sanity: ≈ 0.7 rad = 40°) |

**Sanity gates (warnings, NOT hard fails — they flag operator-cadence drift):**

| Quantity | Target | Tolerance |
|---|---|---|
| `f_wave_Hz` | 2.00 | ±10% |
| `omega_max_rad_per_s` | 8.8 | ±15% |
| `theta_max_rad` | 0.70 (40°) | ±20% |

If the IMU-derived kinematics are wildly off the locked spec, the W_cycle
denominator is anchored to a different operating point than every other
number in the campaign — re-shoot the trials.

---

## What this baseline is used for

1. **Phase 6 step 79** — the published `J_fan_proxy / W_cycle` is the
   denominator in the ≥15% improvement target.
2. **Phase 2a** — the printed baseline's geometry is what runs through the
   2D CFD slice once to seed `phase3_baseline.csv` for Phase 2 rib TO loads.
3. **Spike 0.5** — the 3-copy CV measurement uses the same fan + 3 fresh
   blades in slot N. Don't recycle the same blades across Spike 0.3 and
   Spike 0.5 measurements — Spike 0.5 needs the original 9 plus 3 fresh.

---

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `f_wave_Hz` off > 10% | Operator drifted from metronome | Re-record with louder metronome / count out loud |
| `omega_max` way below 8.8 | Amplitude too small | Wave through full ~40° each way |
| `J_per_W` weirdly high (> 0.1) | Anemometer in stroke wake, not plane integral | Check probe placement against the reticle |
| `W_cycle` ≈ 0 | IMU axis isn't the +y wrist axis (rotation cancels) | Re-mount the IMU; verify with a known-good hand wave |
| Different `W_cycle` per trial | Operator inconsistency | Average ≥ 5 trials; if std/mean > 20%, re-shoot |

---

## Appendix A — Kitchen-scale + cardboard-target measurement (optional V2 upgrade)

**Use this if** you've shipped V1 (a printable fan that feels better than
the baseline) and now want a quantitative number to confirm the gain —
without buying an anemometer.

**Cost: essentially $0.** Most people already own a kitchen scale; a
piece of stiff cardboard is free.

### Apparatus

- A **kitchen scale** that reads to 1 g resolution and can tare. Almost
  any modern digital kitchen scale works; the cheap ones are fine.
- A **cardboard target**: 150 × 150 mm square cut from stiff cardboard
  (cereal box weight or sturdier). Tape it to a wooden dowel or skewer.
- **The Spike 0.3 baseline fan** (10-blade flat-panel PETG) for the
  baseline measurement, plus each optimized design for the comparison
  measurement.
- A **metronome** at 2 Hz (any phone metronome app).

### Setup

1. Stand the dowel vertically on the scale platform so the cardboard
   square is upright and facing the operator. The dowel's weight will
   register on the scale.
2. **Tare** the scale so the displayed reading is 0 with the dowel +
   cardboard in place but no airflow.
3. Position the fan at **300 mm** in front of the cardboard target,
   centered horizontally and vertically.

### Measurement protocol

For each fan under test (baseline first; each printed top-3 candidate
afterwards):

1. Start the metronome at 2 Hz.
2. Wait 2-3 beats to settle into the cadence.
3. Wave the fan for **10 full strokes** at the metronome cadence,
   aiming each productive stroke at the cardboard square.
4. **Record the peak reading** the scale displays during the 10
   strokes — this is the peak momentum flux of the air column hitting
   the target. Most kitchen scales hold the peak for a fraction of a
   second; if yours doesn't, use a phone video camera in slow-motion
   pointed at the display.
5. Repeat **5 trials** of 10 strokes each. Take the mean of the 5 peak
   readings as that fan's number.

The scale measures grams; multiply by `g = 9.81 m/s²` and divide by
1000 to convert to Newtons of momentum flux.

### Recording

`data/spike_0_3/kitchen_scale.csv`:

```
design_label,trial,peak_grams,notes
baseline,1,12,clean
baseline,2,11,clean
baseline,3,13,slight cadence drift
baseline,4,12,clean
baseline,5,12,clean
fan_v1,1,16,clean
fan_v1,2,15,clean
...
```

### Comparison

The gain is `(mean_design − mean_baseline) / mean_baseline`. Repeatable
to ~5%; sensitive enough to detect 10-30% design deltas (which is the
V1 target range).

**Sanity check:** if `fan_v3` measures 25 g and `baseline` measures 12 g,
that's a 108% gain. If it does NOT also feel meaningfully stronger in
hand, suspect operator-cadence drift between the two measurements
(measure them back-to-back in one session to mitigate).

### What this method does NOT give you

- No `W_cycle` denominator. You're measuring J_fan-proxy only. A
  heavier fan that moves more air "wins" by this metric even though it
  takes more wrist effort. For V2 paired with the Phyphox-phone IMU
  (free), you can recover `W_cycle` and re-normalize.
- No 9-point grid. The kitchen scale measures momentum flux at one
  point (the cardboard target). Spatial uniformity of the airflow is
  not captured.

Both limitations are acceptable for a personal-project V2 quantification
step. The full anemometer + 9-point grid + Phyphox IMU + Spike 0.2
torsional-pendulum path remains the rigorous V2++ option for anyone who
wants research-grade numbers.

---

## Appendix B — Phyphox phone-IMU protocol (optional V2 upgrade)

**Use this if** you have a smartphone (you do) and want `W_cycle`
measurements without buying a dedicated IMU. The phone's accelerometer +
gyroscope at 100-500 Hz is more than enough for the 2 Hz wrist-flexion
signal.

### Setup

1. Install **Phyphox** (free, open-source, iOS + Android).
2. Open the "Acceleration with g" + "Gyroscope" experiment.
3. Tape the phone to the fan handle with the **phone's y-axis aligned
   with the wrist-flexion axis**. Most phones: y-axis runs along the
   length of the phone. Tape so the phone sits flush against the handle
   with its length along the handle's length.
4. Verify alignment: cock your wrist up and down (no fan motion); the
   Phyphox `gyro_y` reading should spike. If `gyro_x` or `gyro_z`
   spikes instead, re-tape.

### Measurement

1. Start Phyphox recording.
2. Wave the fan at 2 Hz for 10 full cycles.
3. Stop recording.
4. Export the CSV (Phyphox "Share" → "Export data" → CSV).

### Processing

The exported CSV has columns `t, gyro_x, gyro_y, gyro_z` (and
acceleration). `src/fanopt/physical/imu.py` already reads CSVs in this
shape — point it at the Phyphox export and it computes `W_cycle` using
the analytic `I_wrist_kgm2` from the §6.4 generator (no Spike 0.2
needed). The full Spike 0.3 runner accepts these IMU CSVs:

```
python scripts/run_spike_0_3_baseline.py \
  --imu phyphox_trial1.csv phyphox_trial2.csv phyphox_trial3.csv \
  --anemometer data/spike_0_3/kitchen_scale.csv \
  --inertia data/spike_0_2/results.json \
  --out data/spike_0_3/baseline.json
```

(The `--anemometer` flag accepts the kitchen-scale CSV from Appendix A
if you've also done that step — the runner doesn't care whether the
plane-integrated `J_fan` came from an anemometer or a cardboard
target.)

