# V1 print guide — Bambu Lab H2C, PLA (PETG optional)

Three optimized blade designs; each fan is 12 identical blades threaded on a Ø3 mm pin. Each design is
printed as its own fan. The current round prints full fans of **00** and **02** in PLA (§1).

![V1 blades — 00 (green), 02 (blue), 03 (orange)](images/v1_blades.png)

> **Do not print the `_ribadd` files** (or `_ribadd_clean`, `_supported`, `_raft`, or any project made
> from them). They are a ~0.25 mm skin, not the designed blade: an Aug-7 post-processing step replaced
> every solid region with a hollow shell thinner than the nozzle, and they print as spaghetti. Use the
> rebuilt files below. The image above and the previews in [`docs/models/`](models/) were made from
> the old files and predate the rebuild.

## 1. Files

The print files are too large for the repo (≈ 60 MB STL, ≈ 12 MB 3MF each). They are generated into
`print_ready/` (git-ignored) by `scripts/build_print_blades.py` (§7). They are **not yet** in the
[`v1.0-print` release](https://github.com/clingergab/fan-optimization/releases/tag/v1.0-print), which still
holds the old `_ribadd` files.

Current print files (2026-11-23, PLA):

| Print file | Role | Edge rib | Folded stack | Pin |
|---|---|---|---|---|
| `P2C_blade_00_2026-11-23_thin-20pct_open-13.09_flat-overlap_PLA.3mf` | **best wind** — lowest stress margin (§4c) | 2.0 mm | 71.8 mm | Ø3 × ~42 mm |
| `P2C_blade_02_2026-11-23_thin-20pct_open-13.09_flat-overlap_PLA.3mf` | **best structure** | 1.5 mm | 71.0 mm | Ø3 × ~41 mm |

Both are 20% thinner than the design, open 0.2° less per blade, and have their overlap strips flattened
(§4c). Design 03 has not been rebuilt with these changes; its newest file is
`P2C_blade_03_2026-11-23_thin-20pct_bump-1.0.3mf` (thinner only, PETG profile). Rebuild it with the §7
command before printing a 03 fan.

Pin length = the 12 stacked bosses (30.0 / 29.0 mm) + the same ~12 mm for end caps the earlier pins
allowed. Longer pins still work if you already have them.

Earlier builds stay in `print_ready/` under their dated names (nothing is overwritten, §7): the
full-thickness `*_2026-11-22_bump-1.0_recess-1.05`, the thinner `*_2026-11-23_thin-20pct_bump-1.0`
(its `.json` `folded_stack_mm` is wrong — it used the un-thinned thickness — and the geometry is fine),
and the PETG-profile copies of the current files (same name without `_PLA`, plus
`P2C_blade_00_2026-11-23_thin-20pct_open-13.09`, the angle change alone).

Print the `.3mf` (Bambu Studio project, H2C settings below). Each has a matching `.stl` (same blade, for
other slicers) and a `.json` report (volumes, mass, checks). "P2C" in the filenames is a historical typo;
the files are correct.

Files are in millimetres, pre-oriented **on-edge** with the leading-edge wall flat on the bed; bbox
≈ 227 × 45 × 51 mm.

**What the blade is:** the topology-optimized (TO) solid, rebuilt directly from the optimizer's own
density on its own tetrahedral mesh — surface within 0.06 mm (median) of the design. The TO left the root
⅔ of the blade solid and carved the outer ~80 mm into a hollow with 0.5 mm-or-thicker aero faces. The
rebuild adds only what the design specified:

- the 0.5 mm frozen aero skin (re-imposed; the TO's per-element density left pinholes along mesh edges);
- the solid hub (r ≤ 20 mm) with a Ø3.0 mm through-bore for the pin;
- the leading/trailing-edge ribs (width in the table), where the TO had carved them away near the tip;
- a 0.6 mm tip-end cap;
- the Ø12 mm pivot boss made **flush with the blade where it joins the boss** — the same thickness
  (2.501 / 2.413 mm for 00 / 02 on the current files) at the same height — so neighbouring blades
  touch with no gap (§4b). The original design spacing (4.038 / 3.904 mm) left a 0.65–0.9 mm gap next to
  the hub;
- the **open/closed hold detent** (§4a) — added after the TO, not part of it.

The current files also make three print-only changes after the TO, each from a printed-pair test (§4c):
the whole blade 20% thinner, the overlap strips no thicker than the boss, and 0.2° less opening.

The hollow is intentional. Under 0.01 % of the area is thinner than the 0.4 mm nozzle.

## 1a. Flat control fan (2026-11-25)

`P2C_blade_00-FLAT-CONTROL_2026-11-25_thin-20pct_open-13.09_flat-overlap_PLA-15pct.3mf` is blade 00 with
its curve taken out, for the W0 test in `V2_plan.md`: a printed fan identical to 00 except for the shape,
so a feel test and a directional wind test (scale + card, or a vane anemometer on AVG) check whether the
curve is what the simulation says makes the wind.

**Same as the printed 00:** planform and the 220 mm reach (pin to tip, straight), the TO inner hollow
(the same tets with the same density), the true wall thickness (measured normal to the surface), the 20 %
thinning, the flat overlap strips, the 13.09° opening, the hold detent (same seats and sizes), the boss
(2.50 mm), and **every slicer setting** — the project was made from the printed
`H2C_blade_00_…_PLA-15%infill.gcode.3mf` (Bambu PLA Matte, 15 % grid, textured PEI, auto brim).

**Different:** the blade is flat. The curved 00 is 232.5 mm long along its surface (5.7 % longer than its
220 mm reach) and its thickness was laid out vertically, so on a slope its real wall is thinner by the
cosine of the slope (6 % on average, up to 16 % at the steepest 32°). Unbending keeps that real wall, so
the flat blade has less material: 13.21 vs 13.93 cm³ solid (−5.2 %). After slicing at 15 % infill the
difference is small because walls dominate: **17.69 g and 48 min per blade** (with brim) vs 17.88 g for
00, so about 2 g lighter per fan. The folded fan is only ~30 mm tall (no curve to nest).

Weigh both fans before the test and record the weights; the effort side of the V2 objective uses them.

## 2. Slicer settings (in the `.3mf`)

| Setting | Value |
|---|---|
| Printer / nozzle | Bambu Lab H2C, 0.4 mm |
| Process | 0.20mm Standard @BBL H2C |
| Filament | Generic PLA @BBL H2C 0.4 nozzle (switch for PETG — §2a) |
| Nozzle / bed (PLA) | 220 °C / 55 °C |
| Plate | Textured PEI |
| Walls | 2 loops |
| Infill | **100 % zig-zag** in the file — the TO sized and stress-checked the blade as solid plastic. The V1 fans are printed at **15 % grid** instead (§2b) |
| Supports | **off** — on-edge, only ~15–25 mm² per blade faces down |
| Brim | outer only, 8 mm (steadies the tall thin wall) |
| Raft | none |
| Placement | centred in the bed region both H2C nozzles reach (x 25–325 mm) |

Bambu Studio may warn that the object "has floating regions" (00 and 02; it comes and goes between
slices). It flags sub-nozzle slivers on the inside wall of the sealed hollow, each within 0.35 mm of
the wall below, which merge into the wall or are too small to print. Ignore it and keep supports
**off** — supports would grow inside the sealed hollow, where they can never be removed.

## 2a. Material: PLA (chosen) or PETG

The current `.3mf` files carry the **Generic PLA @BBL H2C 0.4 nozzle** profile (built from
`print_ready/rebuild/H2C_pla_template.3mf`). To print PETG instead, use the copy without `_PLA` or switch
the filament to **Generic PETG @BBL H2C 0.4 nozzle** in Bambu Studio before slicing. The profile sets the
nozzle/bed temperatures and cooling; the other settings in §2 stay the same.

The topology optimization and its stress checks used **PETG** properties (`schema.py`). PLA is not
re-optimized; these are typical-datasheet trade-offs, not measured on our prints:

| | PETG (as designed) | PLA |
|---|---|---|
| Density | 1.27 g/cm³ | 1.24 g/cm³ (≈ 2 % lighter) |
| Stiffness | baseline (FDM 1.3 GPa in-plane) | roughly 1.5–2× stiffer → less flex |
| Failure | bends before breaking | stronger but brittle — snaps on a drop or a hard fold |
| Heat | softens ≈ 70–80 °C | softens ≈ 55–60 °C (a hot car can warp it) |
| Flexing parts (the blades press and flex, §4a–4b) | suited | likely to crack or fatigue |

The printed 00 pair (20% thinner, PLA) felt sturdy. Watch the detent bumps and the hub for cracks
during the feel test.

Current-file masses (`.json` reports, solid volume × filament density):

| Design | Volume | PLA blade / 12-blade fan | PETG blade / 12-blade fan |
|---|---|---|---|
| 00 | 13.93 cm³ | 17.3 g / 207 g | 17.7 g / 212 g |
| 02 | 13.52 cm³ | 16.8 g / 201 g | 17.2 g / 206 g |

All are under the 300 g fan cap. `--material petg|pla` on `scripts/build_print_blades.py` sets the
filament for the reported mass only; the `.3mf` takes its filament profile from `--template`.

## 2b. Infill as printed: 15 % grid (operator choice, 2026-11-24)

Every print so far, and the V1 fans, used Bambu Studio's default **15 % grid** sparse infill with 2 walls,
not the 100 % the file sets (the setting reset when the blades were placed on a new plate). The operator
kept it on purpose: the blades are lighter, and strength and flex have been fine, including at 20 %
thinner.

- **Slicer infill is not the TO hollow.** The TO void toward the tip is part of the blade's shape and stays
  empty at any infill. 15 % only turns the solid core (between the walls, toward the hub) into a lattice.
- **It saves little, because the blade is mostly walls.** The blade is ~2.5 mm thick and prints on edge,
  so its two 2-loop walls (~0.8 mm each) leave a core under 1 mm wide over most of it. Bending stiffness
  comes mostly from those outer walls.
- **Reference print:** `print_ready/H2C_blade_00_2026-11-23_thin-20pct_open-13.09_flat-overlap_PLA-15%infill.gcode.3mf`
  — 12 × blade 00 on one plate, Bambu PLA Matte, auto brim, **214.6 g, 12 h 18 m** (≈ 17.9 g per blade
  with brim, against ≈ 20.0 g at 100 %, §3).
- **Masses in §2a assume solid blades,** so a 15 % fan is ~10 % lighter; use the scale. Print 00 and 02 at
  the same infill and record it with every measurement.
- **What 15 % weakens:** the solid spots that take concentrated loads — the hub around the pin (bolt
  clamp, pivot twist), the edge ribs and the area under the detent bumps. Watch them for cracks; 4 wall
  loops make the ring around the pin solid for almost no weight.
- **For 100 %:** Bambu Studio refuses 100 % with grid or zig-zag; use **rectilinear**.

## 3. Estimates

Bambu Studio CLI slice of each current `.3mf` as shipped (H2C, Generic PLA, one blade per plate):

| Design | Time per blade | Filament per blade | 12 blades (one at a time) |
|---|---|---|---|
| 00 | 1 h 17 m | 20.0 g | ~15.5 h, ~240 g |
| 02 | 1 h 18 m | 19.5 g | ~15.5 h, ~234 g |

Slicer filament is ~2.7 g above the blade's own mass, mostly the 8 mm brim. Several blades per plate fit
on the H2C bed and save heat-up/homing time per blade, but have not been sliced.

## 4. Assembly (per fan)

1. Peel the brim; deburr the bore ends and the brim edge. The detent bumps print just above the brim
   (§4a): if one fused to the brim, trim it free and keep its dome intact.
2. The Ø3.0 mm bore is exactly the pin diameter (no clearance), and printed holes usually come out a
   little small: ream to Ø3 mm, or use a Ø2.9 mm rod for a light sliding fit.
3. Thread 12 blades on a Ø3 mm steel/brass pin (length per §1), **all the same way up** — bumps all
   facing the same way — so they nest in a clean z-stack and each bump meets the recesses above it.
4. Cap both ends (cap / washer / push-nut) with light axial preload. Do not over-tighten: the blades
   already press on each other (§4b) and must still flex 1.0 mm apart for the detent to click.

## 4a. Open/closed hold detent

Each blade carries 5 small domes (bumps) on its top face along the overlapping edge, 80–120 mm from the
pin, and 10 shallow dimples (recesses) in its underside. Each bump rides in the underside of the blade
above it:

- **Folded:** each bump sits in the blade above's *closed* recess, right over it.
- **Open:** turning the blade above 13.09° slides the bump 18–27 mm into its *open* recess.
- **The hold:** the blades touch at every seat (§4b), and each bump stands 1.0 mm proud of the blade
  face, so the blade above must flex up 1.0 mm to slide out of a recess. The blade's own flex is the
  spring.

Every seat on all three designs: bump **1.0 mm** tall, 3.0 mm across, cut from a 1.6 mm-radius sphere
(the same size as the first printed bumps). Recess **1.05 mm** deep, ~3.2 mm across, so the whole bump
drops in and the blades sit flat against each other, folded and open, with 0.05 mm of room under the bump
and ~0.1 mm around it. At least 1.44 mm (00) / 1.35 mm (02) of blade is left under a recess. Sizes are
measured on each blade's own surface at each seat (`detent.seats` in the `.json` report).

History: the first printed pair had ~1 mm bumps but only 0.40–0.49 mm recesses (the rest of the bump
crossed the old air gap at the hub), and the 0.3 mm bump / 0.35 mm recess build of 2026-11-22
(`*_boss-flush_recess-0.35`) shrank the bump instead of deepening the recess.

**The detent works in one opening direction only.** With the pin at the bottom, the fan opens **to the
right**: the front blade (the one you push) swings right. This holds whichever face you look at. Check
it on the first pair (§5) before printing the rest; the other direction needs a rebuild with
`--deploy-direction 1` (§7).

The 1.0 mm bump height is the first printed size, not a tested hold strength for touching blades. If the pair's hold is too
weak or too stiff, rebuild with a different `DetentSpec.interference_m` (`src/fanopt/geometry/blade_detent.py`).
A detent on the pivot boss is not possible (13.3° × 6 mm boss ≈ 0.8 mm of travel, smaller than any bump),
which is why it sits out on the blade.

## 4b. Blades touch, and press where they are thicker

The boss sets how far apart neighbouring blades sit on the pin. It is as thick as the blade where the
blade joins it, so there is no gap next to the hub. Further out the optimized blade is thicker in
places, so neighbours press there and flex. Measured on the current files (rigid check, the two blades
as printed, 0.25 mm grid):

| Design | Folded: most pressing | Open: most pressing in the overlap | Open: overlap touching flat |
|---|---|---|---|
| 00 | 0.44 mm | 0.18 mm | 89 % (the rest ≤ 0.2 mm, nearly all) |
| 02 | 0.43 mm | 0.09 mm | 96 % (the rest ≤ 0.2 mm, nearly all) |

Folded, the pressing is in the middle of the blade (accepted: the blades flex, and it costs only some
friction). Open, the overlap strips are flattened (§4c), so neighbours lie flat instead of propping
each other apart.

## 4c. Print-only changes from the printed-pair tests (2026-11-23)

Three changes made after printed pairs, none of them part of the optimization:

1. **20% thinner** (`--thickness-scale 0.8`, "the design felt bulky"). Every point moves toward the
   blade's mid-surface: the outline and the curved surface that pushes the air (dish, camber, waves)
   are unchanged. The TO density is mapped first and then thinned with the mesh, so the hollow layout is
   the same; the 0.5 mm skins, edge ribs and tip cap are rebuilt at full thickness. Hub 3.13 → 2.50 mm
   (00) and 3.02 → 2.41 mm (02); folded fan 79.4 → 71.8 mm (00) and 78.3 → 71.0 mm (02); about 4 g
   lighter per blade.
2. **Flat overlap strips** (`--flatten-overlap`). Light came through the seam of an open pair, and
   pinching the blades closed most of it: the blades thicken toward the tip, so they were propping each
   other apart. Where open neighbours overlap (about 11 mm wide at the hub, narrowing toward the tip),
   the blade is now no thicker than the boss, blending back over 3 mm. Each face moves at most ~0.2 mm;
   the middle of the blade keeps its optimized thickness.
3. **Opens 0.2° less** (`--open-pitch-deg 13.09`, the design pitch is 13.29°). The open pair barely
   overlapped over the last centimetre. The open recesses moved 0.2° closer, so the overlap now reaches
   the tip: 1.5 mm at r = 200 mm and 0.6 mm at r = 218 mm (was 0.8 mm and none past r = 214 mm). The fan's
   spread drops from 146° to 144°. The blade shape is unchanged.

Not checked by simulation: the stress of the thinner blade (hand estimate: about a third less stiff,
skin stress about 25% higher, blade 00's FOS ≈ 1.9 → ≈ 1.5) and the wind, which was simulated at the
full thickness and the 13.29° pitch. The previews of the open pair before/after the last two changes
are in `print_ready/previews/00_open_overlap_fixB_vs_fixBA.png`.

## 5. Before committing 36 blades

1. Print **one** blade: check the first layer runs the full length and the tip skins close.
2. Print a **pair** + pin: check that the blades nest when folded, that the fan opens to the right
   (§4a), that the bumps seat fully in the recesses, and that it clicks into the open and closed
   positions and holds there without forcing. Hold the open pair up to a lamp: little or no light should
   come through the seam (§4c).
3. Print a few more and stack them: the blades press on each other (§4b) — check that opening still
   feels smooth with several on the pin.

## 6. Known V1 limitations

- **Flex:** the TO minimized compliance only, so it left the outer third hollow with no internal webs
  (and its gray, not-fully-converged elements there were printed as void). The blades will flex. V1.5 fix:
  re-run TO with a stiffness objective + buckling constraint (`V2_backlog.md`).
- **Ranking was flex-blind:** designs were ranked on rigid-blade CFD (ADR-0008); the feel test judges
  the real, flexing blade.
- **Hub root is un-optimized** (the TO clamped a retired 20 mm hub band; the rebuild fills it solid) —
  sound for V1, fixed in V1.5 (`V2_backlog.md`).

## 7. Rebuilding the print files

The TO densities and campaign shards are recovered from Drive into `print_ready/drive_recovery/`.

1. Export a Bambu template for the printer/filament (Bambu Studio writes every printer setting):

   ```
   python3 scripts/make_bambu_template.py --model <any .stl/.3mf> \
       --machine "Bambu Lab H2C 0.4 nozzle" --process "0.20mm Standard @BBL H2C" \
       --filament "Generic PLA @BBL H2C 0.4 nozzle" --out print_ready/rebuild/H2C_pla_template.3mf
   ```

   (`H2C_petg_template.3mf` is the same with `--filament "Generic PETG @BBL H2C 0.4 nozzle"`.)

2. Build each blade (`--rib-width-mm` 2.0 for 00 and 03, 1.5 for 02). The current files:

   ```
   python3 scripts/build_print_blades.py --name 00_ffe19f9e3e7fa81ba847bbe8 --rib-width-mm 2.0 \
       --shards print_ready/drive_recovery/campaign_trapezoid --to-dir print_ready/drive_recovery \
       --thickness-scale 0.8 --open-pitch-deg 13.09 --flatten-overlap --material pla \
       --out print_ready/P2C_blade_00_<YYYY-MM-DD>_<what-changed>.stl \
       --template print_ready/rebuild/H2C_pla_template.3mf
   ```

   Every build gets a new, dated name: the script refuses to overwrite an existing `.stl`, `.json` or
   `.3mf`, so earlier versions (including ones already printed) are always kept.

   Design names: `00_ffe19f9e3e7fa81ba847bbe8`, `02_248e298991dfd83a3f7bdfe0`,
   `03_1be42b2b86422ad222fb548a`. Each build takes ~5 min (the three run in parallel fine).

   `--thickness-scale 0.8` builds the blade 20% thinner about its mid-surface; `--flatten-overlap` caps
   the open-fan overlap strips at the boss thickness; `--open-pitch-deg` sets the open angle per blade
   (default the design's 13.29°). All three are off by default (§4c).

   The flat control (§1a) is the 00 command plus `--flat-meridian --template-settings-only`, with
   `--template` set to the printed `.gcode.3mf`: `--flat-meridian` unbends the blade at the same reach and
   true wall thickness, and `--template-settings-only` keeps every template setting (no 100 % infill or
   brim override).

   The detent is on by default: `--detent-seats 5` (`0` builds the blade without it) and
   `--deploy-direction -1` (fan opens to the right, §4a; `1` reverses it).
