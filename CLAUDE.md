# CLAUDE.md — project context for Claude Code

## What this project is

A 3D-printed folding hand fan optimized in two stages:

1. **Rib topology optimization** (SIMP / Reissner-Mindlin plate-bending) under
   four locked load cases (productive stroke, return stroke, inertial, click
   engagement).
2. **Panel aerodynamic shape optimization** (CadQuery generative geometry +
   SU2 CFD on three tiers) inside a multi-fidelity BoTorch Bayesian
   optimization loop with an outer architecture bandit over categorical
   choices (blade count, panel topology family, click variant).

The canonical specification is `docs/plan_R11.md` (mirror of the upstream
`report-final.md`). All numerical locks live there; every section in the
codebase that consumes a lock is enumerated in `docs/locks_index.md`.

## Key locks the implementer must respect

- **Panel-pivot architecture (Architectural A / C7 / D / E):** the 3 mm pivot
  pin runs through the panel at y = 0, NOT through the rib. Rib radial extent
  is `[HUB_RADIUS = 0.020 m, L_blade − RIB_TIP_TAPER = 0.185 m]`.
- **C11 sign lock:** `PITCHING_OMEGA = (0, -12.5664, 0)` — negative y. Right-
  hand-rule on productive stroke.
- **HIGH-12 unsteady cfg (Round 9):** unsteady SU2 cfg uses `MACH = 1e-9`
  with `FREESTREAM_OPTION = FREESTREAM_VELOCITY` override (or fallback
  `REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE`). MACH is tier-specific:
  steady tiers (-1 / 0) use 0.0064, unsteady tier (1) uses 1e-9. CROSS_TIER
  dict does NOT carry MACH.
- **HIGH-8 Round-9 Option A click chamfer:** 0.5-1 mm corner bevel at the
  panel's outer tangential edge — NOT a full-panel-thickness face. Adjacent
  panels meet at a 45° butt-joint LINE; no Z-axis overlap.
- **Mass constraint (C9):** `m_total < 100 g`.
- **Hardware (HIGH-11 Round 9):** Phase 5 verification uses Colab Pro G4 GPU
  (95 GB), not T4 (16 GB OOMs at PyFR p=3).

## Where things go

| Want to add | Goes in |
|---|---|
| New BO acquisition strategy | `src/fanopt/bo/` |
| New CadQuery primitive | `src/fanopt/geometry/primitives.py` (Layer 3) or `fields.py` (Layer 2) |
| New SU2 config template | `configs/su2/*.cfg.j2` + `src/fanopt/cfd/configs.py` Jinja2 renderer |
| Phase entry-point script | `scripts/run_phase{N}_*.py` |
| New retired phrase to gate | `docs/retired_phrases.yaml` + run `tests/test_audit/test_no_stale_architecture_refs.py` |

## Process locks

- Adversarial review process is in `docs/review_process.md`. Stopping rule:
  0 CRITICAL + < 3 HIGH after fixes land.
- Round-internal labels (CRIT-N, HIGH-N, MED-N) MUST NOT appear bare in
  production prose. Use either an explicit round prefix ("Round-9 HIGH-12")
  or the absorbed global label (C12, H16). Retired-phrase gate enforces this.

## What NOT to do

- Don't drill a pivot hole through the rib (panel-pivot architecture).
- Don't set MACH = 0.0064 in the unsteady cfg (HIGH-12 retired this).
- Don't add `rib_base_width` / `rib_tip_width` as BO variables (locked to
  4 mm / 6 mm per H12).
- Don't reintroduce 14-blade or rib-crank architectures (MED-10 Round 7 / CRIT-1).
- Don't bypass `tests/test_audit/test_no_stale_architecture_refs.py` if a
  catalog regex fires — either fix the prose or extend the allow-list with
  a justified disclaimer; do not delete the entry.
