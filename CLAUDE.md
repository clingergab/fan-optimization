# CLAUDE.md — Project rules manual

This file is binding instruction for Claude Code working in this repo. Read it
before any session. The plan (`docs/report-final.md`) is the source of truth
for **what** the project builds; this file is the source of truth for **how**
code is allowed to be written, tested, and shipped here.

---

## 1. Project

A 3D-printed folding hand fan optimized in two stages:

1. **Rib topology optimization** — SIMP / Reissner-Mindlin plate-bending under
   four locked load cases (productive stroke, return stroke, inertial, click
   engagement).
2. **Panel aerodynamic shape optimization** — CadQuery generative geometry +
   SU2 CFD on three fidelity tiers inside a multi-fidelity BoTorch Bayesian
   optimization loop with an outer architecture bandit over categorical
   choices (blade count, panel topology family, click variant).

V1 scope: ship a printable optimized fan judged by qualitative blinded A/B
feel test against the printed flat-panel baseline. V2 scope: quantify the
gain. See `docs/phase_logs/phase_0_signoff.md`.

## 2. Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+; SU2 cfg .j2 templates; SU2 .su2 meshes |
| Package layout | `src/fanopt/{bo,cfd,geometry,physical,topopt,utils}/`, importable as `fanopt.*` |
| Scripts | `scripts/*.py` — flat directory of CLI entry points; **not** a Python package |
| Tests | `tests/test_<area>/*.py` mirroring `src/fanopt/` + `scripts/` paths |
| Geometry | CadQuery (Layer 1–3 generator); Gmsh (CFD meshes); Fusion 360 (visual / cleanup, headless add-in) |
| CFD | SU2 (CPU; G4 GPU for PyFR p=3 verification only — HIGH-11 lock) |
| Optimization | BoTorch + GPyTorch (multi-fidelity GP + qMFKG + qNEHVI); TuRBO; SAASBO fallback |
| Structural | FEniCSx / CalculiX (Phase 2 SIMP TO + Phase 5 verification FEA) |
| Build / quality | pyproject.toml; pytest + pytest-cov; ruff + black; mypy; pre-commit |
| Compute | MacBook Pro M3 for local dev / geometry / Phase 5 step 59.5 FEA / Phase 6 post-processing; Colab Pro CPU for SU2 Tier -1/0/1; Colab Pro G4 GPU (95 GB) for PyFR p=3 only |
| Persistence | Drive/JSONL ledger with composite-key markers (H16 7-tuple lock); pre-sliced round-robin assignment |

## 3. Architectural locks (non-negotiable)

These flow from the plan's §0. Code that contradicts any of them is wrong.

- **Panel-pivot architecture** (Architectural A / C7 / D / E): the 3 mm pivot
  pin runs through the panel at `y = 0`, NOT through the rib. Rib radial
  extent is `[HUB_RADIUS = 0.020 m, L_blade − RIB_TIP_TAPER = 0.185 m]`.
- **C11 PITCHING_OMEGA sign**: `(0, -12.5664, 0)` — **negative y** is part of
  the lock, not just the magnitude. Right-hand-rule on productive stroke.
- **Round-9 HIGH-12 unsteady cfg**: SU2 unsteady cfg uses `MACH = 1e-9` with
  `FREESTREAM_OPTION = FREESTREAM_VELOCITY` override (or fallback
  `REF_DIMENSIONALIZATION = FREESTREAM_PRESS_EQ_ONE`). MACH is **tier-
  specific**: steady tiers (-1 / 0) use 0.0064, unsteady tier (1) uses 1e-9.
  `CROSS_TIER` dict does NOT carry MACH.
- **Round-9 HIGH-8 Option A click chamfer**: 0.5–1 mm corner bevel at the
  panel's outer tangential edge — NOT a full-panel-thickness face. Adjacent
  panels meet at a 45° butt-joint LINE; no Z-axis overlap.
- **H8 lever-arm lock**: τ → F conversions at the click region use
  `L_WRIST_TO_TIP = 0.25 m`, NOT `L_blade = 0.20 m`.
- **C9 mass cap**: `m_total < 100 g`.
- **Round-9 HIGH-11 hardware**: PyFR p=3 (Phase 5 verification only) needs
  Colab Pro G4 GPU (95 GB VRAM). T4 (16 GB) OOMs.
- **Round-9 H10 Phase 4 gate**: Phase 4 launch is gated on
  `data/spike_0_6c/PASS` — `scripts/launch_phase4.py` enforces this.

## 4. Code rules

### 4.1 Imports

- **NEVER use inline imports. No exceptions.** Imports go at the **top of the
  file**, before any business logic, ahead of all `try` blocks and any other
  control flow. Never inside functions, classes, methods, conditionals,
  context managers, `try`/`except` blocks, or test bodies. Never. If the
  import might fail because a dependency is optional, deal with that at the
  call site (gate the function with a runtime check after a top-of-file
  unconditional import; or skip the test module via
  `importlib.util.find_spec(...)` + `pytest.skip(..., allow_module_level=True)`
  before doing any import that would fail; or simply require the dependency
  and let the module fail to import in environments without it).
- **In notebooks:** all imports go in **one cell near the top of the
  notebook**, before any business logic. Within a cell, imports go at the
  top of the cell. Same rule — no `try` blocks around imports, no imports
  buried below code.
- If a notebook is platform-specific (e.g., Colab-only), import the
  platform-specific module unconditionally at the top. If someone tries to
  run it on the wrong platform, they get a clean `ImportError` immediately;
  that's the correct behavior.

### 4.2 File layout

- Code that's part of the project package: `src/fanopt/<area>/<module>.py`.
  Importable as `fanopt.<area>.<module>`.
- CLI scripts: `scripts/<script>.py`. Flat layout. Not importable as a
  package. Each script has a `main(argv=None) -> int` entry point and is
  callable as `python3 scripts/<script>.py` from the repo root.
- Heavy logic does NOT live in `scripts/`. Scripts are thin wrappers around
  importable functions in `src/fanopt/`. If a script grows past ~150 lines of
  non-trivial logic, refactor into a module.
- Templates: `configs/<tool>/<template>.cfg.j2` paired with a renderer in
  `src/fanopt/<tool>/configs.py`. Templates are Jinja2; renderers validate
  inputs and emit text. Never inline a cfg in Python code.
- Notebooks: `notebooks/<purpose>.ipynb`. **Logic in notebooks is forbidden**
  beyond ~50 lines of orchestration. Everything substantive lands in
  `src/fanopt/` or `scripts/` first, with tests, and the notebook calls into
  it.

### 4.3 Naming

- Constants in `SCREAMING_SNAKE_CASE`. Floats with SI units carry the unit in
  the name (`PIVOT_BOSS_RADIUS_M`, not `PIVOT_BOSS_RADIUS`). Locked constants
  live in `src/fanopt/geometry/schema.py` (geometry, kinematics) and
  `src/fanopt/cfd/configs.py` (CFD tier dicts). Don't re-declare locked
  constants in other modules — import them.
- Round-internal labels (CRIT-N, HIGH-N, MED-N, LOW-N) MUST NOT appear bare
  in code. Use either an explicit round prefix (`Round-9 HIGH-12`) or the
  absorbed global label (`C12`, `H16`). The retired-phrase audit gate
  enforces this on `src/fanopt/**/*.py`.

### 4.4 Comments + docstrings

- Default to writing no comments. Add one when the WHY is non-obvious: a
  hidden constraint, a subtle invariant, a workaround for a specific bug.
- Docstrings: one short line for simple functions; expanded docstrings only
  for non-obvious public APIs.
- Never write multi-paragraph comment blocks. If something needs that much
  explanation, it belongs in a doc, not a `#` block.

### 4.5 Dependency direction + loose coupling

Keep the dependency graph **acyclic and one-directional** so modules stay
testable in isolation and refactors stay local. These are hygiene rules,
not architecture ceremony — **you do NOT need interfaces / ABCs / Protocols
/ ports / adapters for a project this size**. Concrete callables passed by
reference are the right level.

- **Dependencies flow in one direction within `src/fanopt/`.** Lower-level
  modules (locked constants, pure-data dataclasses, math helpers) don't
  import higher-level ones. Concretely: `geometry/schema.py` doesn't import
  from `bo/` or `cfd/`; `cfd/configs.py` may import from
  `geometry/schema.py` but not the reverse; `utils/` is the lowest layer
  and depends on nothing in the package.
- **Scripts depend on `src/fanopt/`, never the reverse.** A module in
  `src/fanopt/` must never `import` from `scripts/`. Scripts are consumers,
  modules are providers. Tests follow the same direction — tests import the
  code under test; code never imports its tests.
- **No circular imports — ever.** If two modules need each other, one of
  them is doing the wrong job. Push the shared piece down into a third
  lower-level module that both can import. Don't paper over a cycle with
  string-typed forward references unless there is no other option.
- **Pass dependencies as function arguments where practical.** A function
  that takes its inputs as arguments is easy to test with synthetic inputs
  and easy to reuse. A function that reaches into module-level state, hits
  the network directly, or hard-codes filesystem paths is none of those.
  This is dependency injection — the cheap kind, without DI containers.
- **Prefer pure functions.** Same input → same output, no side effects.
  When side effects are necessary (file I/O, subprocess, RNG), keep them at
  the edges: a CLI/script does I/O at its boundary, then calls pure helpers
  for the actual work, then writes results at the boundary. Helpers should
  take a `Path` or a value, not reach into the global state.
- **No hidden globals.** A function that reads or mutates module-level
  state, env vars, or config files inside its body is hard to test. Either
  take the state as an argument, or extract it once at the top of the
  caller and pass it in.
- **YAGNI on abstractions.** Don't add an abstract base class until you
  have a second concrete implementation. Don't add a Protocol until two
  callers need different concrete types. Concrete code with clean
  dependency direction outperforms premature abstraction every time at
  this project's size.

## 5. Testing rules

### 5.1 Coverage

- **All new code has tests.** No exceptions. If you wrote a function, the
  same PR / change adds a test for it.
- Tests mirror the source tree: `tests/test_<area>/test_<module>.py` for
  `src/fanopt/<area>/<module>.py`; `tests/test_scripts/test_<script>.py` for
  `scripts/<script>.py`.
- Target **>80% line coverage** on every new module. Hard rule: every
  **critical path** + every **main business-logic branch** must be covered.
  "Critical" = code that consumes a project lock, gates a decision, or
  produces a Phase-N artifact.
- Trivial getters / `__repr__` / pragmas don't need tests. Defensive branches
  that are unreachable under upstream validation get `# pragma: no cover`
  with a one-line comment explaining why.

### 5.2 Test patterns

- One assertion per test where practical; clear test names that describe the
  invariant being checked.
- Optional-dependency tests (gmsh, SU2, FEniCSx, BoTorch) use
  `importlib.util.find_spec` at module top + `pytest.skip(...,
  allow_module_level=True)`. Do NOT lazy-import inside a fixture or test
  function to gate availability.
- No mocking the project's own internals to make a test pass. Mock external
  systems (subprocess, network, filesystem-with-cost) only at the boundary.
- Float comparisons via `pytest.approx`.

### 5.3 Pre-existing tests

- Don't bypass `tests/test_audit/test_no_stale_architecture_refs.py` if it
  fires. Either fix the code-comment phrasing or extend the catalog's
  allow-list disclaimer with a justified note. Do not delete the catalog
  entry.

## 6. Notebook rules

- All imports at the top of the notebook (one consolidated cell). Within a
  cell, imports at top. Top-of-cell `try:` for optional deps is allowed.
- **Notebooks contain orchestration, not logic.** If you find yourself
  defining a function in a notebook beyond a tiny helper (~10 lines), move
  it to `src/fanopt/` or `scripts/`, write tests for it, then import it back.
- Cells have a single responsibility. A long cell that does install + render
  + analyze is three cells.
- Cells should be **re-runnable independently** wherever practical. If
  cell N must always be preceded by cell N-1, document that in the markdown
  above N. Idempotent operations (mkdir, file-cached downloads) are strongly
  preferred over destructive ones.

## 7. Process rules

### 7.1 Verification (binding)

**No implementation is "done" until the verification pass completes.** The
verification pass is:

1. **Review all code generated/changed in the session** — read each modified
   file end-to-end. Spot inline imports, dead code, untested branches,
   incorrect lock references, plan-vs-code drift.
2. **Run the full test suite** — `pytest`. Zero failures, no new skips
   without a documented reason. New code paths exercised by new tests.
3. **Run the audit gate** — `pytest tests/test_audit/`. Zero violations.
4. **Coverage check** — at least the new modules report >80% line coverage.

If any of 1–4 fails, the work is not done. Fix and re-verify.

### 7.2 Plan and audit-catalog boundaries

- `docs/report-final.md` is the plan. **Claude does not edit the plan**
  except when the user explicitly authorizes a specific additive edit (e.g.,
  "add a deferral note in §13.0"). Never edit a locked decision. Never edit
  to make tests pass.
- `docs/retired_phrases.yaml` is the audit-catalog source of truth. **Claude
  does not edit the catalog to silence the gate.** If a catalog entry fires
  on Claude's code, the right move is fix the code's phrasing — not widen
  the allow-list. The user authors catalog edits when prose conventions
  evolve.

### 7.3 Tasks and trust

- Don't add features beyond what the user asked for. A bug fix is a bug
  fix, not also a refactor.
- Three similar lines of code are better than one premature abstraction.
- When unsure about scope, ask before implementing.
- When something is blocked by missing user input, say so clearly — don't
  invent values or paper over gaps with placeholders that look real.

## 8. Where things go

| Want to add | Goes in |
|---|---|
| New BO acquisition strategy | `src/fanopt/bo/` |
| New CadQuery primitive | `src/fanopt/geometry/primitives.py` (Layer 3) or `fields.py` (Layer 2) |
| New SU2 config template | `configs/su2/*.cfg.j2` + renderer in `src/fanopt/cfd/configs.py` |
| New locked geometry constant | `src/fanopt/geometry/schema.py` + a test in `tests/test_geometry/test_schema.py` |
| New CFD tier constant | `src/fanopt/cfd/configs.py` `TIER_SPECIFIC` or `CROSS_TIER` (NOT MACH in CROSS_TIER) |
| Phase entry-point script | `scripts/run_phase{N}_*.py` |
| Spike runner | `scripts/run_spike_<N>.py` or `scripts/<spike-name>_<verb>.py` |
| New retired phrase to gate | `docs/retired_phrases.yaml` + verify `tests/test_audit/test_no_stale_architecture_refs.py` |
| New Pareto objective | `src/fanopt/bo/pareto.py` + tests |
| New JSONL ledger field | `src/fanopt/utils/ledger.py` `LedgerRow` dataclass + bump `SCHEMA_VERSION` |

## 9. What NOT to do

- Don't drill a pivot hole through the rib (panel-pivot architecture).
- Don't set MACH = 0.0064 in the unsteady cfg (Round-9 HIGH-12 retired this).
- Don't add `rib_base_width` / `rib_tip_width` as BO variables (locked to
  4 mm / 6 mm per H12).
- Don't reintroduce 14-blade or rib-crank architectures (MED-10 Round 7 /
  CRIT-1).
- Don't put MACH in `CROSS_TIER` — it's tier-specific per HIGH-12.
- Don't use inline imports, anywhere, ever — including inside `try`/`except`,
  inside notebooks, inside test functions. Top of file / top of cell only.
- Don't add abstract base classes, Protocols, or interfaces speculatively —
  wait until a second concrete implementation forces the abstraction.
- Don't introduce circular imports. Push the shared piece down into a
  lower-level module.
- Don't put logic in notebooks. Refactor into `src/fanopt/` + tests first.
- Don't edit `docs/report-final.md` without explicit user authorization for
  a specific additive edit.
- Don't edit `docs/retired_phrases.yaml` to make a test pass.
- Don't declare work "done" until the §7.1 verification pass completes.
- Don't bypass `tests/test_audit/test_no_stale_architecture_refs.py` if a
  catalog regex fires — either fix the code's phrasing or extend the
  allow-list with a justified disclaimer; do not delete the entry.
- Don't invent values to fill in missing measurements, citations, or
  reference data. Mark the gap and surface it.
- Don't sign or add yourself to commit messages, for example never add 'Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>' 
