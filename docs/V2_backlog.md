# V2 Backlog

Canonical, expanded V2 plan. The in-spec summary lives at `../report-final.md` §13; this file is the authoritative location for V2 work descriptions, triggers, and acceptance criteria.

## Triggered items (V1 failure → V2 in-scope)

Each item has a triggering condition that fires at Phase 6 wrap-up (or earlier if the trigger is a Phase 4 diagnostic). If the trigger fires, the V2 effort begins with the corresponding entry as its first deliverable.

### V2 designed lock mechanism

**Trigger:** Phase 6 testing shows the fan unlocks under sustained 2 Hz waving (the H6 V1 force balance passes in Spike 0.4 but fails in practice).

**V1 fallback:** the rib-tab fallback (`params.layer4.v1_lock_fallback_enabled`) is armed conditionally if Spike 0.4 force balance fails. V2 supersedes the rib-tab with a designed lock.

**V2 scope:**
- Locked-cantilever snap engaging the guard rib outer face at the deployed angle (120° default).
- Magnetic-catch alternative (embedded N52 neodymium discs, 2 mm × 1 mm; ~1 g per pair, ~20 g across 10 blades; within the 60 g mass budget).
- Cycle-life test: ≥10,000 deploy/fold cycles without functional degradation.

**Acceptance:** 30-min sustained 2 Hz waving session at 40° amplitude with no inter-pair friction-driven unlock event.

### Centrifugal Filter 4 (re-introduce a real Filter 4)

**Trigger:** Phase 6 testing reveals fatigue failures at the pivot driven by centrifugal pull under aggressive waving (the kind the canonical Filter 2 cyclic check misses).

**V1 status:** Filter 3 is a deprecated pass-through stub. Centrifugal load is covered by the one-shot Phase-2 dynamic-load assertion (`α_peak · m_rib · r_tip · N_blades < 0.1 · click_detent_allowable`), not per-design.

**V2 scope:**
- Proper per-design Filter 4 with the correct kinematics: cyclic tangential reaction at click detent + centrifugal stress at the boss, both computed with **H8 wrist-to-tip lever arms** (`L_wrist_to_tip = 0.25 m`, NOT `L_blade = 0.20 m`).
- σ_centrifugal at boss = `m_blade · ω_blade_max² · r_boss / (2 · A_boss)` evaluated at the boss radius (7 mm); compared against the §10.1 bearing allowable (2.00 MPa Z-direction binds).

**Acceptance:** Filter 4 rejects designs that fail the Phase 6 centrifugal fatigue criterion, on the cohort of Phase 4 designs evaluated by the time the trigger fires.

### Alternative MFBO architectures for TPMS/noise

**Trigger:** L7 empirical-bias diagnostic fires often in Phase 4 (mean `|Δ_TPMS / mean_J_fan_tier0| > 0.30` across the first 100 Tier-0 evals).

**V1 status:** TPMS/noise architectures use the 0.3/0.7 reweighting compromise (Tier 0 weight 0.7, Tier -1 weight 0.3) for promotion decisions.

**V2 scope (pick (a) or (b) based on Phase 4 diagnostics):**

(a) **Disable multi-fidelity GP for TPMS/noise architectures** — run Tier-0-only single-fidelity GP per affected architecture. Removes the Tier -1 ↔ Tier 0 correlation kernel where it's known to be mis-specified.

(b) **Treat Tier -1 as a separate cheap-feature input** — concatenate `J_fan_tier_minus_1_proxy` as an extra GP input dimension rather than a fidelity tag. Preserves the cheap-screening signal without forcing the multi-fidelity kernel.

**Acceptance:** affected architectures' Tier 0 → Tier 1 Spearman ρ² improves by ≥ 0.1 over the V1 0.3/0.7 reweighting baseline.

## Optional (V1-complete, V2-improves)

Items where V1 ships a working solution but V2 has a clear path to a better one. No triggering condition required; V2 picks these up as time permits.

### Mid-Phase-4 rib re-tune

**Current V1 spec:** Phase 2 rib SIMP TO runs once with a smooth-baseline panel placeholder.

**V2 scope:** re-trigger Phase 2 rib SIMP TO every K Phase-4 architecture promotions, conditioned on the panel topology the architecture-bandit is actually selecting. Timing: re-tune happens after the first K promotions complete (≈ Phase 4 month 1).

**Cost:** ~3-5 additional Phase 2 SIMP solves × ~30 min each = 1.5-2.5 hours per re-tune; per-architecture-class, not per-design.

### Textured-PEI bed-surface portability

**Current V1 spec:** §3.2.4 / M13 lock smooth-PEI (Bambu Cool Plate Super Tack AP05, Ra ≤ 5 µm).

**V2 scope:** document the §3.2.4 wall-roughness calibration procedure so users on other bed surfaces (Prusa textured PEI Ra 10-30 µm, Anycubic frosted PEI Ra 15-25 µm) can re-derive the roughness-model parameters. Includes a portable Phase 0 sub-spike that measures the bed-contact face Ra in-situ and refits the calibration coefficients.

### Asymmetric-stroke physics in J_fan

**Current V1 spec:** J_fan is symmetric in time (integrates over full cycles).

**V2 scope:** explore an asymmetric weighting `J_fan_biased = w_p · J_productive_half + (1 − w_p) · J_return_half` with `w_p` measured from IMU during Phase 6. Would change the optimization target away from the parachute baseline more aggressively (the symmetric metric rewards equal forward/backward drag; the biased metric rewards productive-stroke drag specifically).

### `directional_asymmetry_score` functional-form refinement

**Current V1 spec (C6 lock):** starter form from §Phase 3 step 33:

```
directional_asymmetry_score(design) :=
    sum over Layer 2 louver fields:
        (louver_count) × |sin(louver_angle)| × (active flag)
    + |Fourier_LE_phase_offset − Fourier_TE_phase_offset| · 0.1
    + sum over Layer 3 primitives:
        (polarity_sign) × (primitive_size_relative_to_chord)
```

**V2 scope:** converge on the functional form that best predicts the Phase 6 IMU-measured J_fan spread across designs (richer signal than the Phase 0 3-radius calibration sample). Candidates:
- weighted sum of Layer 2 louver angles only,
- Fourier TE/LE phase difference only,
- integrated `|chord_z⁺(x) − chord_z⁻(x)|` over the planform (camber asymmetry),
- the starter sum-of-three form from V1.

Pick the candidate with highest R² on the Phase 6 dataset. The score is dimensionless; β carries dimensional scaling to J_fan units.

## Out-of-scope (V3+ or research)

Items that are not in the V1 or V2 roadmap, queued for either a V3 effort or a research follow-on.

- **Active electronic flow control** — embedded micro-blowers or piezo actuators in the panel cutouts. Adds power + control complexity; out of V2 scope.
- **Multi-DOF wrist motion** — current model assumes pure +y wrist rotation (flexion). Real waving has yaw + pitch + roll components. V3 could extend the SU2 unsteady cfg to support compound rotations.
- **Multi-material printing** — TPU membranes, dual-extruder panels. V1 explicitly rejects this (single-material PETG except the steel/brass pin); V2 stays single-material. V3 could revisit if multi-material AMS toolchains improve.
- **Adjoint-based aero shape optimization on the panel envelope** — V1 uses generative parametric design (4-layer hybrid); V3 could couple SU2 continuous adjoint to the Layer 1 envelope spline directly for a finer-grained gradient-based refinement of the top-1 Pareto design.
