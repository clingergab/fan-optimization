# Notebooks

Colab + local Jupyter notebooks. Per `docs/report-final.md §12.2`, notebooks
should stay thin — ~50 lines of orchestration over the `fanopt` package.
The exception is `colab_spike_0_6c.ipynb`, which has more setup boilerplate
because it bootstraps SU2 on a fresh Colab VM.

| Notebook | Purpose | Status |
|---|---|---|
| `colab_spike_0_6c.ipynb` | **Run Spike 0.6c on Colab Pro CPU** — H10 gate that unblocks `scripts/launch_phase4.py`. | implemented |
| `colab_phase4_runner.ipynb` | Phase 4 BO inner loop, multi-session orchestration | stub — Phase 4 work |
| `pareto_analysis.ipynb` | 4D Pareto front inspection (post Phase 4) | stub — Phase 5 work |
| `geometry_inspection.ipynb` | CadQuery-generated blade STL preview | stub — Phase 1 work |
| `physical_results.ipynb` | IMU + acoustic + anemometer summary (V2) | stub — Phase 6 work |

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

## Other notebooks

The four other `.ipynb` files in this directory are Phase 1+ work — they
ship as 1-cell scaffolds and get filled out as those phases land. The
canonical orchestration pattern (per §12.2) is:

```python
from fanopt.bo import architecture_bandit, turbo, pareto
pareto = run_phase4_bo(seed_data, config)
plot_pareto_front(pareto)
```

If a notebook grows past ~50 lines of logic, refactor that logic into
`src/fanopt/` and re-thin the notebook.
