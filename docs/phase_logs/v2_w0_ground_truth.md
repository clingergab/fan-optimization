# V2 W0 — V1 ground truth (physical measurements)

`docs/V2_plan.md` W0: the printed fans measured, so the simulations can be checked against them. Results
are added here as they come in; nothing below is estimated unless it says so.

## 1. Fan weights

| Fan | Blades | Weighed | Slicer estimate | Difference | Date |
|---|---|---|---|---|---|
| 00 (PLA Matte, 15 % grid, 20 % thinned, flat overlap, detent) | 12, no pin | **212 g** (17.7 g / blade) | 214.6 g (17.9 g / blade; auto brim adds none) | −1.2 % | 2026-11-25 |
| 02 | | not yet | | | |
| Flat control | | not yet (not printed) | 212.3 g (12 × 17.69 g) | | |
| Store-bought fan (bamboo sticks + fabric; listing: 38 cm wide × 21 cm tall open) | — | **~24 g** | — | | 2026-11-25 |

The slicer's weight is within ~1 % of the scale for this printer and filament, so the slicer-fitted mass
model of W1 (`v2_w1_inertial_effort.md`) holds for blade 00. The flat control and the thickness series
will check it across shapes and walls.

## 2. First wind runs (2026-11-25)

Vane anemometer (HoldPeak HP-866B) on a small tripod; each fan waved by hand a few cm in front of it;
10 s average. **One run per fan** (as reported), so differences of ~0.2 m/s are within run-to-run noise
until repeated.

| Fan | Mass | 10 s average |
|---|---|---|
| Store-bought | ~24 g | 3.4 m/s |
| 00, as assembled | 212 g | 3.2 m/s |
| 00, blades taped at the clicked-open position (§3) | 212 g + tape | 3.4 m/s |

At about equal wind, fan 00 weighs ~9× the store fan.

## 3. Assembly checks on fan 00 (2026-11-25)

- **Click test** (each pair opened only until its detent clicks): neighbouring tips barely overlap, and a
  little light comes through the open seams. The seam gap is the blades not lying against each other
  along the overlap (pressing them by hand closes it), not the detent bumps propping them.
- **Ribbon test** (tape across the back near the tips, fan at the clicked-open position): slightly
  better, not a big difference; still a little awkward to hold. Wind 3.4 m/s (§2).
- **Operator's notes:** the folded fan is ~30 mm thick at the hub (12 × 2.5 mm) and the thickness and
  opening/closing feel awkward; the detent does not stop the fan opening past the design angle. Unpinned
  vs pinned, the folded stack leans slightly differently (blades nest by sliding when free; the pin holds
  the hubs in line).

## 4. Still to measure

- Feel test, blinded: flat control vs 00 (and 02); store-bought fan as a benchmark.
- Wind, repeated: 3–5 runs per fan, alternating fans, at a fixed marked distance (flat control, 00, store fan).
- Store fan: stick count and opening angle (measured, not from the listing).
- Tip-deflection test: clamp at the hub, known weight at the tip; 15 % and 100 % infill.
- Thickness series (e.g. 80 / 70 / 60 %): weight and tip deflection.
