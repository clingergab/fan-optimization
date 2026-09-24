# Notebooks

Colab + local Jupyter notebooks. They are **thin orchestration** over the `fanopt` package (all logic
lives in `src/fanopt/` with tests — see `CLAUDE.md` §6). Most are Colab-only: SU2 is a single-threaded
CPU solver, so the CFD and TO runs fan out across Colab CPU sessions that share a Google Drive folder.

## V1 pipeline (current, in run order)

| Notebook | Stage | What it does |
|---|---|---|
| `aero_objective_walkthrough.ipynb` | explainer | Walks through the 3D aero objective (`J_fan`) on one blade — what the CFD measures and why |
| `colab_stage2_probe.ipynb` | Stage 2 | De-risk probe: coarse↔fine CFD fidelity check + shape-headroom check before the campaign (ADR-0004) |
| `colab_stage3_campaign.ipynb` | Stage 3 | Distributed, cold-start BO campaign on the 220 mm trapezoid blade (run in several sessions at once) |
| `colab_stage3c_verify.ipynb` | Stage 3.C | Fine-tier 3D CFD re-verification of the campaign's top designs; picks the top-10 for TO |
| `colab_stage4_blade_to.ipynb` | Stage 4 | Per-design 3D SIMP topology optimization of the top-10 blades (ADR-0007) |
| `colab_stage4_rescreen.ipynb` | Stage 4 | Re-screens saved TO density fields (deflection, solid-only stress) without re-optimizing |
| `colab_fan_review.ipynb` | review | Final visual review: carved blade → watertight STL → 12-blade fan, folded + deployed |
| `physical_results.ipynb` | Phase 6 / V2 | Reduces bench measurements (IMU, anemometer, microphone) against predictions — awaits printed fans |

## Phase 0 validation spikes

| Notebook | What it does |
|---|---|
| `colab_spike_0_6c.ipynb` | Spike 0.6c — unsteady SU2 config + NACA 0012 benchmark (gate PASSED; run instructions below) |
| `colab_spike_0_6d.ipynb` | Spike 0.6d — unsteady CFD quantitative sanity checks (gate PASSED) |
| `colab_naca_benchmark.ipynb` | Spike 0.6c.2 — oscillating NACA 0012 SU2 validation (cross-solver gate, deferred to Phase 5) |

## Superseded (kept for the record)

These belong to the earlier 2D-slice optimization, which ADR-0004 found blind to the rib wave. Their
results are **not** a basis for the V1 designs.

| Notebook | What it did |
|---|---|
| `colab_phase2a_baseline_cfd.ipynb` | Baseline 2D-slice CFD for the retired 2D rib TO |
| `colab_phase3_correlation.ipynb` | Steady↔unsteady 2D-slice correlation gate |
| `colab_phase4_runner.ipynb` | V1-slim 2D-slice multi-objective BO |
| `colab_phase4_aero_campaign.ipynb` | First aero-first BO campaign (2D-slice objective) |
| `colab_phase5_verify.ipynb`, `colab_phase5_bakeoff.ipynb` | 3D verification / tiebreak of the 2D-campaign winners |
| `render_top_blades.ipynb`, `render_failed_blades.ipynb` | Renders of the Phase-5 verified / failed blades |
| `learn_self_intersection.ipynb` | Explainer for the self-intersecting-geometry failures seen in Phase 5 |

## Scaffolds (never filled in)

`geometry_inspection.ipynb`, `pareto_analysis.ipynb` — one-cell stubs from the original plan layout.

## Running `colab_spike_0_6c.ipynb`

### 1. Upload to Colab

Three options, easiest first:

- **From GitHub directly:** in Colab, `File → Open notebook → GitHub` tab,
  paste the repo URL, pick `notebooks/colab_spike_0_6c.ipynb`. The notebook
  opens read-only at first; Colab auto-creates an editable copy when you
  start running cells.
- **From Drive:** push the notebook to a Drive folder that Colab can see,
  then open it from `File → Open notebook → Google Drive`.
- **Direct file upload:** in Colab, `File → Upload notebook` and pick the
  `.ipynb` file from your local repo.

### 2. Edit the cell-1 constants

The first cell defines:

```python
GIT_REPO   = 'https://github.com/YOUR-GH-USER/fan-optimization.git'  # <-- EDIT
GIT_BRANCH = 'main'
```

Point it at your fork. If you only ever run this from the canonical repo,
hard-code the URL once.

If you want the notebook to push the PASS marker back to the repo
(cell 9), also edit:

```python
GIT_USER  = 'YOUR-GH-USER'    # cell 9
GIT_EMAIL = 'YOUR-GH-EMAIL'   # cell 9
```

and add a GitHub Personal Access Token as the Colab secret `GITHUB_PAT`
(left sidebar → 🔑 → Add new secret).

### 3. Run cells in order

| Cell | What | Wall-time |
|---|---|---|
| 1 | Clone repo, mount Drive | ~30 s |
| 2 | `pip install` Python deps | ~1 min |
| 3 | Install SU2 (Drive-cached on re-run) | 5–45 min first run / ~10 s re-run |
| 4 | Generate meshes (Drive-cached) | ~1 min first run / ~10 s re-run |
| 5 | Sub-spike 0.6c.1 cfg sanity | ~2 min |
| 6 | Sub-spike 0.6c.2 SU2 benchmark | **6–12 h** |
| 7 | Parse SU2 history → measured.csv | ~10 s |
| 8 | Run analyzer + write PASS marker | ~10 s |
| 9 | (Optional) commit + push marker to GitHub | ~20 s |

Total first run: ~7–13 hours of wall-clock. Most of that is cell 6.

### 4. If Colab disconnects mid-cell-6

Colab Pro keeps sessions alive 24 hours but disconnects after long
idleness. The notebook handles this gracefully:

- The SU2 run writes its `history.csv` continuously to **Drive**
  (`/content/drive/MyDrive/fan-optimization/spike_0_6c/sub_2_run/`), so
  the partial data survives a disconnect.
- The mesh + SU2 binary are cached on Drive (cell 3, cell 4) — re-runs
  skip the install.
- To resume: re-open the notebook, re-run cells 1–4 (fast under Drive
  cache), then re-run cell 6. SU2 picks up from the last checkpoint via
  its restart-file mechanism.

If you want to keep the session alive without re-running cells, open
the notebook tab and click into the notebook once every ~30 min, OR
use a "keep-alive" Chrome extension.

### 5. SU2 install fallback paths

Cell 3 tries paths in order:

1. **Drive cache.** Skips install entirely if a prior run cached the
   SU2 binary on Drive.
2. **Pre-built binary** from `https://github.com/su2code/SU2/releases`.
   The asset name changes per version — check the [releases page](https://github.com/su2code/SU2/releases)
   and adjust `SU2_VERSION` in cell 3 if the latest version has a
   different asset.
3. **Source build** with `meson` + `ninja`. Slow (~30–45 min) but
   reliable. Use this if the binary download fails or you need a
   custom build.

After install, the binary is cached to
`/content/drive/MyDrive/fan-optimization/su2_cache/<version>/` so the
next notebook run skips the install.

### 6. What "PASS" actually means

- **0.6c.1 PASS** = the locked Tier-1 unsteady cfg (HIGH-12 lock —
  `MACH=1e-9`, `FREESTREAM_OPTION=FREESTREAM_VELOCITY`, C11
  `PITCHING_OMEGA` negative-y, etc.) parses cleanly and SU2 launches at
  least one outer time-step against it.
- **0.6c.2 PASS** = the NACA 0012 oscillating-airfoil benchmark
  reproduces published lift/drag coefficients within ±15% on cycles 2–5
  (discarding cycle 1 transient).
- **Aggregate PASS** = both. Writes `data/spike_0_6c/PASS` empty
  marker file. `scripts/launch_phase4.py --check` then returns 0 and
  `phase4-launch` git tag becomes creatable.

### 7. Caveats — please read before declaring V1 closed

- **Reference data is hand-typed.** `src/fanopt/cfd/spike_0_6c.py`
  declares `NACA0012_REFERENCE` as four scalars labeled "representative
  ranges". Before V1 ship, replace with a real published citation
  (McAlister/Carr UH-110A study or NASA Anderson oscillating-airfoil
  database) and pass it via `--reference data/spike_0_6c/references/naca0012_<source>.json`.
  The notebook's cell 8 already supports `--reference` — just point it at
  the citation JSON.
- **Mesh quality is "good enough" not "research-grade".** The
  Gmsh-generated O-grid (cell 4) has 30–40k cells and y+ that's
  acceptable for the ±15% gate but coarser than NASA TMR validation
  grids. If the benchmark misses by 10–14%, a finer mesh may close the
  gap. If it misses by >20%, the issue is probably numerics (dt, inner
  iters) not the mesh.
- **k_reduced and Re are operator-chosen.** Cell 6 derives them from
  the locked Re=40000 and c=1.0 m; the picked k=0.55 sits in the middle
  of the spec band [0.5, 0.6]. If you want to validate against a
  reference at a different (k, Re), update both cell 6's pitching
  parameters AND the reference JSON.

## Notebook discipline

If a notebook grows past ~50 lines of logic, refactor that logic into `src/fanopt/` (with tests) and
re-thin the notebook.
