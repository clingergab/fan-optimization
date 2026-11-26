# V2 W1 — inertial effort of the V1 designs on one print recipe (2026-11-25)

`docs/V2_plan.md` W1: give every V1 campaign design the same, real print recipe, compute how hard the
fan is to swing (its inertia about the wrist), and see whether ranking by wind ÷ effort differs from
ranking by wind alone. No CFD.

**Result:** a fitted slicer mass model (error 0.11 g per blade; the slicer itself is within 1.2 % of the
weighed fan) puts every V1 fan at 224–267 g on the
recipe, and the inertial effort varies by only ±4 % between designs. Ranking by wind ÷ inertial effort
is almost the same as ranking by wind: Spearman 0.999, the best design (00) unchanged, 9 of the top 10
kept. The first half of hypothesis H1 (V2 plan §3a) holds on V1 data.

## 1. The print recipe

- **Wall:** each design's thickness field scaled about its mid-surface to the printed blade 00's mean
  true wall, **2.47 mm** (00's design wall × 0.8, the operator's 20 % thinning). Same mean for every
  design, so differences come only from how the thickness is spread and how much surface the shape has.
- **Inside:** slicer infill (Bambu PLA Matte, 15 % grid, the operator's printed `.gcode.3mf` settings);
  no TO hollow.
- **Fan:** 12 blades spread evenly about the handle axis at 13.29°; wrist axis 50 mm behind the pin
  (`D_HANDLE_M`); stroke ±40° at 2 Hz (sinusoidal).

## 2. Mass model, calibrated against the slicer

`scripts/calibrate_print_mass.py` built 23 real print blades (6 designs spread over their wall, incl.
00; each at 1.6 / 2.0 / 2.5 / 3.0 mm mean wall where thinner than the design) through the same
`build_print_blades.py` pipeline as the printed fans (solid inside, no detent), sliced each in Bambu
Studio with the printed settings and the brim off, and fitted
`fanopt.physical.print_effort.PrintMassModel` (`data/v2_w1/print_mass_calibration.json`):

| Fitted | Value | Meaning |
|---|---|---|
| Solid shell | **2.01 mm** | up to this wall thickness the slicer prints solid (2 wall loops per face + gap fill) |
| Core fill | **10 %** | effective fill of the wall beyond 2.01 mm |
| Base | ≈ 0 g / blade | the sampled sheet already covers the hub and boss |
| Error | **RMSE 0.11 g**, leave-one-out 0.13 g (max 0.30 g) | on ~15–21 g blades (~0.6 %) |
| Density | 1.32 g/cm³ | the printed profile's (Bambu PLA Matte), read from the template — the slicer weighs with it |

The wall material is priced per unit of **surface** area (the curve's extra length counts); a first fit
per unit of planform area missed it and was off by up to ±1.5 g per design.

What it says about thickness — blade 00, sliced, per blade:

| Mean wall | 1.6 mm | 2.0 mm | 2.5 mm | 3.0 mm |
|---|---|---|---|---|
| Mass | 15.6 g | 19.1 g | 20.0 g | 20.7 g |

Below ~2 mm every 0.1 mm of wall costs ~0.9 g per blade (~11 g per fan); above it only ~0.15 g. **Going
thinner than ~2 mm is where the mass (and the inertial effort) drops**, which is why the W0 thickness
series and its stiffness test matter for V2.

For comparison, the printed 00 (TO hollow, flat overlap strips, detent, 15 % infill) slices to 17.9 g per
blade (Bambu's auto brim adds none to these blades) and **weighs 17.7 g** (212 g for 12, W0); the same
design solid-inside on this recipe is 20.1 g. The TO
hollow and the overlap flattening save at least ~2.2 g per blade (≥ 11 %).

## 3. Effort of the 750 valid designs

`scripts/rerank_v1_effort.py` → `data/v2_w1/v1_inertial_effort.csv` (one row per design) and
`v1_inertial_effort_summary.json`.

| Per fan | Min | Median | Max | Spread (CV) |
|---|---|---|---|---|
| Mass | 224 g | 237 g | 267 g | 3.0 % |
| Inertia about the wrist | 0.0054 kg·m² | 0.0057 kg·m² | 0.0065 kg·m² | 3.8 % |
| Inertial work per cycle, push-only | 0.41 J | 0.44 J | 0.50 J | 3.8 % |
| Centre of mass from the wrist | 141 mm | 144 mm | 148 mm | 0.7 % |

All are under the 300 g cap and the 160 mm centre-of-mass limit. Push + brake is exactly twice push-only
(V2 plan D1), so it ranks identically. At 2 Hz, the median fan's push-only inertial work is ~52 J per
minute of waving.

## 4. Re-ranking

| Ranking by | vs `J_fan` alone: Spearman | Kendall | top-10 kept | top-50 kept | best design |
|---|---|---|---|---|---|
| `J_fan` ÷ inertial work (this recipe) | **0.999** | 0.976 | 9 / 10 | 45 / 50 | 00 (unchanged) |
| `J_fan` ÷ ledger solid mass (for contrast) | 0.995 | 0.946 | 8 / 10 | 45 / 50 | 00 (unchanged) |

Median rank move 4 places, largest 50. Within the top 10, designs swap a few places (e.g. 03 moves from
4th to 2nd, 02 from 3rd to 5th): the top 10 span only 10 % in `J_fan`, inside the ~15 % the coarse CFD
resolves, so these swaps are within the CFD's own error.

**Why so little:** every design has the same planform and, on a common recipe, nearly the same mass in
nearly the same place, so the inertia is almost the same for every shape, while `J_fan` ranges from
negative to 9.7e12 across the campaign. Inertia cannot tell shapes apart; what can is the **aero** part of
effort (the air pushing back on the blade), which needs the CFD moment history (W2).

## 5. Limits

- **Slicer weights, checked once.** The model is fitted to Bambu Studio's estimates; the first W0 weighing
  (fan 00: 212 g on the scale vs 214.6 g sliced, −1.2 %) says the slicer is close. The flat control and
  the thickness series will check it across thickness.
- **Inertia only.** No aero torque yet (W2); the hand and arm are not included (same for every design).
- **No TO hollow** in the recipe (slicer infill instead), so absolute masses are ~11 % above the printed
  00; the ranking is unaffected because every design gets the same inside.
- Assumes the fan opens evenly about the handle axis at the design pitch (13.29°; the prints open at
  13.09°) and the base mass sits at the pin.

## 6. Reproduce

```
python3 scripts/calibrate_print_mass.py --shards data/campaign_trapezoid \
    --template "print_ready/H2C_blade_00_2026-11-23_thin-20pct_open-13.09_flat-overlap_PLA-15%infill.gcode.3mf" \
    --include ffe19f9e3e7fa81ba847bbe8 --work print_ready/mass_calibration \
    --out data/v2_w1/print_mass_calibration.json        # ~45 min (builds + slices); resumable
python3 scripts/rerank_v1_effort.py --shards data/campaign_trapezoid \
    --calibration data/v2_w1/print_mass_calibration.json \
    --out-csv data/v2_w1/v1_inertial_effort.csv --out-json data/v2_w1/v1_inertial_effort_summary.json
```
