# V1 print guide — Bambu Lab P2S, PETG

Three optimized blade designs; each fan is 12 identical blades threaded on a Ø3 mm pin. Each design is
printed as its own fan and judged in the blinded A/B feel test against the printed flat baseline.

![V1 blades — 00 (green), 02 (blue), 03 (orange)](images/v1_blades.png)

## 1. Files

The full-resolution print files are **not in the repo** (70 MB each). Download them from the
[`v1.0-print` release](https://github.com/clingergab/fan-optimization/releases/tag/v1.0-print).
Lightweight previews (150k faces, < 0.4 mm deviation, viewable in GitHub's 3D viewer) are in
[`docs/models/`](models/).

| Print file | Role | Rib restore width | Folded stack | Pin |
|---|---|---|---|---|
| `P2C_blade_00_ribadd.stl` | **best wind** — marginal on stress (FOS ≈ 1.9) and fold (0.5 mm under the 90 mm cap) | 2.0 mm | 89.5 mm | Ø3 × 60 mm |
| `P2C_blade_02_ribadd.stl` | **best structure** | 1.5 mm | 88.1 mm | Ø3 × 59 mm |
| `P2C_blade_03_ribadd.stl` | **clean** all-rounder | 2.0 mm | 86.1 mm | Ø3 × 57 mm |

Also in the release: `P2C_blade_00_ribadd.3mf` (Bambu Studio project) and
`P2C_blade_00_ribadd.gcode.3mf` (sliced plate for the P2S), plus the raw topology-optimization output
`P2C_blade_{00,02,03}.stl` before rib restore (reference only — do not print).

All print files are in millimetres, pre-oriented **on-edge**, bbox ≈ 228 × 46 × 53 mm (fits the P2S
256³ mm volume). "P2C" in the filenames is a historical typo for P2S — the files are correct.

**What the blade is:** the hollow topology-optimized (TO) shell — two thin aero faces joined by solid
leading/trailing-edge ribs, with a hollow core. The TO had carved the edge ribs away toward the tip
(the faces came apart there), so the `_ribadd` files restore the rib only where it was fragmented
(≈ 34–40 % of each edge, mostly the last ~55 mm) and add a 0.6 mm cosmetic tip-end cap. The core is
intentionally left hollow.

## 2. Slicer settings (as used in the sliced 00 project)

| Setting | Value |
|---|---|
| Printer / nozzle | Bambu Lab P2S, 0.4 mm stock nozzle |
| Filament | Generic PETG @BBL P2S |
| Nozzle / bed | 255 °C / 70 °C |
| Plate | Textured PEI |
| Layer height | 0.20 mm |
| Walls | 2 loops |
| Infill | 15 % grid |
| Supports | on, normal (auto), top Z-gap 0.2 mm |
| Brim | auto, 5 mm |

## 3. Plate plan and estimates

- **3 blades per plate** → 4 plates per fan (12 blades), 12 plates for all three designs.
- Slicer estimate for design 00: **~7.6 h and ~60 g PETG per plate** (incl. supports) →
  roughly **30 h and ~240 g per fan**, ~0.75 kg for all three fans.

## 4. Assembly (per fan)

1. Peel supports and brim; deburr the bore ends and the brim edge.
2. Ream the bore to Ø3 mm if it printed under, or use a Ø2.9 mm rod for a light sliding fit.
3. Thread 12 blades on a Ø3 mm steel/brass pin (length per §1), **all the same way up** (camber in the
   same direction) so they nest in a clean z-stack.
4. Cap both ends (cap / washer / push-nut) with light axial preload — friction holds the fan open at
   any angle. There is no click detent in V1 (a hub detent was found geometrically infeasible).

## 5. Before committing 36 blades

1. Print **one** blade: check that supports peel cleanly and the thin edges are acceptable.
2. Print a **pair** + pin: check that the blades nest when folded and deploy freely.
3. Design 00 in particular is 0.5 mm under the fold cap — confirm 12 of them actually stack before
   printing the full run.

## 6. Known V1 limitations

- **Flex:** the TO minimized compliance only, so it hollowed the core with no internal webs; the
  blades will flex. V1.5 fix: re-run TO with a stiffness objective + buckling constraint
  (`V2_backlog.md`).
- **Ranking was flex-blind:** designs were ranked on rigid-blade CFD (ADR-0008); the feel test judges
  the real, flexing blade.
- **Hub root is un-optimized CAD** (the TO clamped a retired 20 mm hub band) — sound for V1, fixed in
  V1.5 (`V2_backlog.md`).
