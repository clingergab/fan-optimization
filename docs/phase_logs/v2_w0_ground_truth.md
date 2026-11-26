# V2 W0 — V1 ground truth (physical measurements)

`docs/V2_plan.md` W0: the printed fans measured, so the simulations can be checked against them. Results
are added here as they come in; nothing below is estimated unless it says so.

## 1. Fan weights

| Fan | Blades | Weighed | Slicer estimate | Difference | Date |
|---|---|---|---|---|---|
| 00 (PLA Matte, 15 % grid, 20 % thinned, flat overlap, detent) | 12, no pin | **212 g** (17.7 g / blade) | 214.6 g (17.9 g / blade; auto brim adds none) | −1.2 % | 2026-11-25 |
| 02 | | not yet | | | |
| Flat control | | not yet (not printed) | 212.3 g (12 × 17.69 g) | | |
| Store-bought fan | | not yet | — | | |

The slicer's weight is within ~1 % of the scale for this printer and filament, so the slicer-fitted mass
model of W1 (`v2_w1_inertial_effort.md`) holds for blade 00. The flat control and the thickness series
will check it across shapes and walls.

## 2. Still to measure

- Feel test, blinded: flat control vs 00 (and 02); store-bought fan as a benchmark.
- Directional wind: kitchen scale + light card (or a vane anemometer in AVG mode), each fan, same stroke.
- Tip-deflection test: clamp at the hub, known weight at the tip; 15 % and 100 % infill.
- Thickness series (e.g. 80 / 70 / 60 %): weight and tip deflection.
