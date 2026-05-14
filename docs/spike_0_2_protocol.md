# Spike 0.2 — Torsional-pendulum rotational-inertia measurement protocol

**Spec reference:** `docs/plan_R11.md §Phase 0 Spike 0.2` (lines ~1791–1796) and
`§6.4` (I_wrist Pareto objective).

**Why this exists.** Phase 6 reports the fan's performance as
`J_fan_measured / W_cycle`, where `W_cycle` is the rotational kinetic-energy
budget the operator's wrist has to deliver per stroke:
`W_cycle ∝ I_wrist · ω_max²`. That denominator needs a measured `I_wrist`. The
CadQuery generator (§6.4 `i_wrist_assembly`) also emits an `I_wrist_kgm2`
prediction; this spike's job is to **measure the same quantity physically and
cross-check it** before any Phase 6 number is published.

**Axis lock (do not skip).** All measurements are about the **+y wrist axis
through the handle-grip point** — NOT the pivot pin. The pin is offset by
`d_handle = 0.05 m` in +x from the wrist axis (§0 row 27, §6.4). A pendulum
suspended from the pivot pin measures the wrong quantity by ~`m_total·d_handle²`
(roughly 20–35% lower than the wrist-axis value). The whole spike is wasted if
this gets reversed.

---

## Apparatus

You need:

- A **torsion wire** (music wire or hobby spring steel, ~0.5–0.8 mm dia,
  100–300 mm length) clamped vertically. **Or** a **bifilar setup**: two
  parallel strings of equal length suspended from a fixed bar.
- A **wrist-grip mount block** that grips the fan handle at the wrist-grip
  point (world origin in §6.4's frame). Print this; it is reused for every
  pendulum measurement in V1. The mount must:
  - Clamp the handle rigidly (no slop — slop adds noise to T_osc).
  - Suspend the handle such that **the torsion-wire axis passes through the
    wrist-grip point AND is parallel to +y (the wrist-flexion axis)**. In the
    pendulum frame: the wire is *horizontal*, and the fan is laid flat-ish
    with the planform in the (x, z)-plane of gravity. The mount block
    converts your bench's vertical-wire setup into the §6.4 +y horizontal
    axis. (Equivalent option: a horizontal torsion rod clamped at both ends
    with the fan handle attached mid-span, giving you a true horizontal
    rotation axis.)
- A **timer** capable of ≥10 ms resolution. Phone stopwatch is fine; better is
  a phototransistor + Arduino counting full periods.
- A **reference mass set** for κ-calibration: ≥2 known masses at known radii.
  Suggested: a uniform aluminium or steel rod, dia 6–10 mm, length 80–150 mm,
  weighed on a kitchen scale and measured with calipers. Compute its analytic
  `I_ref` as `(1/12)·m·L²` about its midpoint (rod about transverse axis
  through center).

**Coordinate sanity check before you start.** Lay the fan flat on the bench
deployed. The handle points in **+x**. The pivot pin is vertical (+z if you
were standing the fan up, but lying flat the pin is horizontal — whatever, the
key fact is the pin axis is perpendicular to the handle). Now mentally rotate
the fan about the line passing through the wrist-grip point and parallel to
the handle direction... NO. The +y wrist axis is **perpendicular to the
handle** in the planform, NOT along the handle. It's the wrist-flexion axis
(the joint that lets you cock your hand up and down at the wrist). When the
handle is at +x, the wrist axis is along +y. If your pendulum's rotation axis
is along +x (i.e., the fan twirls about the handle like a barbecue skewer),
you are measuring `I_xx`, not `I_yy`. Re-mount.

---

## Step 1 — Calibrate the torsion constant κ

For a torsion wire: `κ = I_ref · (2π / T_ref)²` where `T_ref` is the period
of the reference rod oscillating on the rig.

1. Mount the **reference rod** on the rig in the same way you will mount the
   fan handle. The rod's symmetry axis must coincide with the wire axis (or
   pass through it for bifilar) and its mass distribution must be predictable.
2. Displace ~5–10° from rest. Release. Let 1–2 cycles damp out, then time **10
   full periods**. Period `T_ref = (time for 10 periods) / 10`.
3. Repeat 5 times. Take the mean. Repeatability < 1% on the reference rod is a
   prerequisite for measuring the fan within 3%.
4. Compute `I_ref = (1/12) · m_rod · L_rod²` (uniform rod about transverse
   midpoint). **Or** use a calibration mass at a known radius, then
   `I_ref = m_cal · r_cal²`.
5. Compute `κ = I_ref · (2π / T_ref)²`. Units: **N·m / rad** (≡ kg·m²/s²).
6. **Sanity check:** repeat with a *second* reference geometry (different m
   and/or r). κ should agree within 2%. If not, your mount has slop or the
   torsion wire is non-linear at your amplitude — reduce the displacement
   angle.

For a **bifilar pendulum** (alternative): `κ_eff = m·g·b²/L`, where `b` is the
half-spacing between the two strings, `L` is the string length, `m` is the
suspended mass. Calibration is geometric — no reference rod needed — but the
pendulum is sensitive to string-length asymmetry. Pick one of the two methods
and stick with it.

Record in `data/spike_0_2/calibration.csv`:

```
kappa_Nm_per_rad,T_ref_s,m_ref_kg,L_ref_m,I_ref_kgm2,method,notes
0.001234,2.840,0.0521,0.120,6.252e-5,torsion_wire,music_wire_0p6mm_L150mm
```

---

## Step 2 — Measure T_osc on the Spike 0.3 baseline fan

The baseline fan: 10 blades, flat PETG panels, no TO, no airfoil camber
(§Phase 0 Spike 0.3 — geometry must be the one the §9.7 generator emits at
the locked default JSON so the generator's `I_wrist_kgm2` is comparable).

1. Mount the assembled baseline fan with **the handle clamped at the
   wrist-grip point** and the +y wrist axis coincident with the pendulum
   rotation axis. Confirm by hand: rotate the fan a few degrees — its motion
   should look like cocking your wrist up and down with the fan deployed, NOT
   like spinning a propeller about the pivot pin.
2. Displace ~5–10° from rest. Release. Damp out 1–2 cycles. Time **10 full
   periods**.
3. Repeat **5 times** at the same amplitude.
4. Record each `T_osc_i` and any qualitative notes (does it look like
   single-mode oscillation? any wobble?) in
   `data/spike_0_2/measurements.csv`:

```
trial,T_osc_s,amplitude_deg,notes
1,3.124,8,clean
2,3.118,8,clean
3,3.131,8,slight wobble first 2 cycles
4,3.120,8,clean
5,3.125,8,clean
```

(`T_osc_s` is **one period**, not the 10-period total. Divide before
recording.)

---

## Step 3 — Compute I_wrist and pass-criteria check

Run the analyzer:

```
python scripts/spike_0_2_analyze.py \
  --calibration data/spike_0_2/calibration.csv \
  --measurements data/spike_0_2/measurements.csv \
  --generator-i-wrist <I_wrist_kgm2 from smoke_test.py on the baseline> \
  --out data/spike_0_2/results.json
```

The analyzer reports:

| Field | Meaning | Pass criterion |
|---|---|---|
| `I_wrist_kgm2` | Mean of `κ · (T̄_osc / 2π)²` across trials | (the measured value — published) |
| `repeatability_pct` | `std(I_wrist_i) / mean(I_wrist_i)` × 100 | **< 3%** |
| `cross_check_pct` | `\|I_meas − I_gen\| / I_gen` × 100 | **< 10%** |
| `passed` | Both gates above passed | `true` for the spike to close |

**Pass criterion (§Phase 0 Spike 0.2):**

1. Repeatability < 3% across 5 measurements of the same fan.
2. Cross-check vs `I_wrist_kgm2` from the generator: agree within ±10%.

**If repeatability fails (> 3%):**
- Mount slop is the usual culprit. Re-print the wrist-grip block with
  tighter clamping.
- Or: amplitude too high → torsion wire is non-linear. Drop to 3–5°.
- Or: pendulum is being damped (air drag, wire friction). Lighter
  reference rod, longer wire.

**If cross-check fails (> 10%):**
- Axis convention bug: did you measure about the +y wrist axis or
  accidentally about a different axis? Re-verify Step 2.1.
- Density mismatch: the generator uses `RHO_PETG = 1270 kg/m³` and
  `RHO_PIN ∈ {7850 steel, 8500 brass} kg/m³` (§2.3, §6.4). If your printed
  PETG comes in at 1180 kg/m³ (under-filled walls / sparse infill), the
  generator over-predicts I_wrist by ~7%. Weigh a single blade and back-out
  the actual density.
- Mount block mass: did you subtract the mount-block contribution from
  `I_wrist`? The pendulum measures `I_total = I_fan + I_mount`. The analyzer
  has an optional `--mount-i-wrist` arg to subtract; measure it once on the
  empty rig.

---

## What this rig is reused for

- **Spike 0.3** — the baseline measurement publishes
  `J_fan_measured / W_cycle` using this I_wrist as the denominator.
- **Phase 6 step 77** — one I_wrist measurement per printed top-3 Pareto
  design (and per copy in the 3-copy noise recheck) so the canonical
  comparison vs Spike 0.3 baseline is apples-to-apples.

Calibration κ persists across runs as long as nothing on the rig changes
(wire is the same, mount block is the same). Re-calibrate if you swap any
component.
