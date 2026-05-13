# fan-optimization

3D-printed folding hand fan — topology optimization of the rib structure plus
aerodynamic shape optimization of the panel envelope. Multi-fidelity Bayesian
optimization over the joint design space; SU2 CFD on three tiers (2D steady,
3D steady, 3D unsteady); CadQuery generative geometry with an
§N7 manufacturability filter; physical validation on a 1000-cycle click-feature
fatigue rig and an end-of-Phase-5 anemometer/IMU/acoustic battery.

The canonical project plan is **`docs/plan_R11.md`** (mirror of the upstream
`report-final.md`). Every numbered lock in this codebase — material constants,
geometry locks, CFD config keys, BO hyperparameters — traces to a single
section in that plan. The locks-to-location index is `docs/locks_index.md`.

## Status

Phase 0 (scaffolding). Empty module stubs in `src/fanopt/` mirror the §12.1
repository layout. Real implementations land per the Phase 0 / 1 / 2 sequencing
in `docs/plan_R11.md §0` and the phase tables in §Phase N.

## Setup

```bash
# 1. Conda env (CadQuery + FEniCSx + SU2 + PyTorch — the heavy deps)
conda env create -f environment.yml
conda activate fanopt

# 2. Editable install of the project package
pip install -e ".[dev]"

# 3. Pre-commit hooks
pre-commit install
```

## Quick reference

| Want to … | Look at |
|---|---|
| Read the canonical plan | `docs/plan_R11.md` |
| See which sections consume a given lock | `docs/locks_index.md` |
| Understand the review process | `docs/review_process.md` |
| Find retired architectural phrases (for grep-based audits) | `docs/retired_phrases.yaml` |
| Trace prior adversarial-review rounds | `docs/reviews/` |
| See what was deferred to V2 | `docs/V2_backlog.md` |

## Repository layout

See `docs/plan_R11.md §12.1` for the authoritative tree. Top-level:

```
src/fanopt/          # main package (geometry, topopt, cfd, bo, physical, utils)
tests/               # mirrors src/fanopt/ structure, plus test_audit/
scripts/             # one entry-point per spike / phase
notebooks/           # thin Colab + local Jupyter orchestrators (~50 lines each)
configs/             # SU2 + Fusion templates (Jinja2 .cfg.j2 files)
data/                # .gitignore'd; designs/, results/, meshes/, physical/
docs/                # plan_R11.md, locks_index.md, reviews/, followups/, history/
```

## Tests

```bash
pytest                    # full suite
pytest -m "not slow"      # skip slow tests
pytest tests/test_audit   # prose-vs-locks gates only
```

The Round-9 audit gates live in `tests/test_audit/` (retired-phrase scanner)
and `tests/test_cfd/test_unsteady_freestream_consistency.py` /
`tests/test_geometry/test_click_z_lap.py` /
`tests/test_geometry/test_no_rib_pivot_hole.py`. These run against the spec
text in `docs/report-final.md` (a copy of `docs/plan_R11.md` kept under that
filename so the gates resolve the spec path without modification).
