# Adversarial Review Process — V1 Spec

This document codifies the process used during R11.x adversarial review rounds
(Rounds 1-9) and locks the rules for any future review activity (V2, Phase 0
audit pass, etc.).

Owner: V1 spec lead. Updated whenever a review round modifies the process.

---

## 1. Stopping rule (Round-9 lock)

A review round may terminate the adversarial-review campaign IFF:

- **0 CRITICAL findings** per the severity bar in §2, AND
- **< 3 HIGH findings** per the severity bar in §2.

MEDIUM and LOW findings do NOT gate stopping; they accumulate for Phase 0
implementation cleanup or a V2 cleanup pass. The retired-phrase CI gate (§4)
provides automated checking for one large class of findings; its output IS
valid input to the stopping decision.

If the stopping criterion is not met, the next round is required.

The stopping rule applies to the END of a round, after all locked fixes have
landed. A round that *generates* 5 CRITICAL findings but lands all 5 fixes
in-round (such that the post-round audit finds 0 CRITICAL + < 3 HIGH) satisfies
the rule.

---

## 2. Severity bar — blocker test (Round-9 user lock)

**Operational test:** anything that is a blocker for other items is HIGH+.

Specifically:

### CRITICAL

A contradiction that would produce **wrong numerical results** or an
**unbuildable artifact silently** — an implementer who follows the spec
faithfully produces broken output without an obvious failure signal.

Examples from rounds 1-9: σ_y_Z = 55 MPa (Round 1 — non-conservative bearing
allowable); panel_x_start = 0.020 m vs panel-pivot architecture (Round 2 —
geometry inconsistency); PITCHING_OMEGA sign (Round 5 C11 — silent
wrong-direction CFD); MACH × c_ref = V_tip tailwind under default SU2
TEMPERATURE_FS (Round 9 HIGH-12 — zero net relative velocity in unsteady).

### HIGH

A contradiction that **blocks an implementer** or downstream work. The
implementer hits a fork and must choose; the work after the fork depends on
which choice they make. Includes:

- Spec internal contradictions that an engineer would notice
- Code drift between spec prose and code blocks (e.g., comment says X, code
  body does Y)
- Locks that aren't propagated to all consuming sections
- Hardware/toolchain specs that don't actually run (Round 9 HIGH-11 T4 OOM)
- Math errors that cascade into Pareto evaluation (Round 9 MED-3 r_CoM kinematics)

### MEDIUM

Cosmetic drift, documentation inconsistency, or future fragility. Doesn't
block current implementers but adds friction for future readers or makes future
changes harder.

### LOW

Preventive process cleanup; nothing currently broken but worth fixing before
it compounds. Examples: namespace collisions (Round 7 HIGH-11; promoted from
LOW to HIGH because future review rounds would compound it).

### When in doubt

If the item blocks another item, it's HIGH+.

---

## 3. Audit-by-layer structure (Round-7 meta lock + Round-9 ordering update)

Every review round runs these passes in order:

### Pass B (FIRST per Round-9 workflow lock)

Run the retired-phrase CI gate at
`tests/test_audit/test_no_stale_architecture_refs.py`. Treat its output as the
primary finding source for prose-vs-locks drift. Any violations become Round-N
findings to fix.

Why Pass B first: the gate catches the largest class of recurring drift
(stale architectural phrases) automatically, before human-driven passes spend
time on the same drift. Anything the gate misses goes to Pass A-F.

### Pass A — Locks taxonomy

Grep every named lock (C-N, H-N, M-N, Architectural X, item #N) and verify
each has a single canonical definition. Catch reuse of the same label for
different concepts (the Round-7 HIGH-11 namespace collision class).

### Pass C — Code vs locks

For each code block in the spec (§9.x, §6.x), verify every hardcoded
numerical literal traces to a locked constant or is documented as a
pedagogical/test-only exception. Round 7 CRIT-2 and Round 9 HIGH-9 both came
from this pass.

### Pass D — Schema vs locks

Verify JSONL schema fields, cross-tier hash inputs, and CI gate assertions
reference current lock values. Round 7 HIGH-9 (GEOMETRY_LOCKS) came from this
pass.

### Pass E — Cross-section consistency

Walk inter-section dependencies (e.g., §59.5 stress-test pass criteria vs
§3.1.5 K_t allowables; §0 row 17 Fan architecture vs §0 row 139 click lock);
verify cited numbers and conventions match. Round 8 CRIT-1 and Round 9 HIGH-1
both came from this pass.

### Pass F — CI gate implementation status

Verify each gate described in §12 of the spec exists as a runnable Python file
(not spec-only). Run the gates; their output is part of the review's findings.
Round 9 HIGH-5 came from this pass — Round 8 added gate *descriptions* without
the runnable code.

---

## 4. Retired-phrase catalog maintenance (Round-8 v2 + Round-9 lock)

After every review round that retires an architectural pattern:

1. Open `docs/retired_phrases.yaml`.
2. Add a new entry with:
   - `pattern`: regex matching the retired phrase
   - `retired_by`: round + lock that retired it (e.g., "HIGH-8 Round-9 Option A lock")
   - `allow_list_sections`: markdown section headings where matches are allowed
     (Appendix D revision history, etc.)
   - `allow_list_disclaimers`: phrases that, if present on the matching line,
     mark the match as a legitimate historical reference ("earlier draft",
     "retired", etc.)
3. Run `tests/test_audit/test_no_stale_architecture_refs.py` to verify the
   catalog entry doesn't false-positive on current spec (false positives = wrong
   regex; tune the pattern OR widen the allow_list).
4. Commit spec edits AND catalog entry in the SAME PR (Round-8 sequencing lock:
   land the gate's new entry FIRST so the gate has the pattern in its catalog
   before the prose edits remove the phrase from the spec; otherwise the gate
   has nothing to assert against after the fact).

---

## 5. Naming conventions for review-round findings (Round-8 LOW-12 lock)

### Within-round labels

Within-round labels (CRIT-N, HIGH-N, MED-N, LOW-N) are scoped to that round
only. They appear in the round's edit list and in commit messages, NOT in
production prose.

### When fixes land

- **New architectural locks**: absorb into the global namespace (C12, C13, ...,
  H15, H16, ..., M21, M22, ...). The global namespace is monotonic — never
  reuse a number, never re-issue a retired label.
- **Propagation cleanups**: no global namespace entry; reference by file/line
  in the commit message only.

### Forbidden in production prose

Bare round-internal labels (CRIT-2, HIGH-9) MUST NOT appear in production
prose. The retired-phrase gate (§4) catches violations. Allowed forms inside
prose: "Round-7 CRIT-2", "Round-9 HIGH-9" (explicit round prefix), or
absorbed global labels (C14, H16).

---

## 6. Locks-to-location index

See `docs/locks_index.md` for the mapping from each lock to every section that
consumes it. Update incrementally when any lock changes.

The index is the answer to "what else needs to change when lock X changes?"
A spec edit that doesn't update the index leaves a propagation trap for the
next reviewer.

The retired-phrase gate (§4) catches the *backward* direction (a stale phrase
left in the spec); the locks index catches the *forward* direction (a lock
that's been updated but not yet propagated to all consumers).

---

## 7. Round closure checklist

Before declaring a round closed:

- [ ] Edit list addressed: every CRITICAL, HIGH, MEDIUM, LOW item has a
      committed fix or an explicit deferral with rationale
- [ ] Retired-phrase catalog updated (§4): new entries for any retired
      architectural pattern, allow-lists tuned to zero false positives
- [ ] Locks-to-location index updated (§6): every changed lock has its entry
      walked to confirm all consuming sections updated
- [ ] CI gates green: `tests/test_audit/test_no_stale_architecture_refs.py`,
      `test_no_lock_namespace_collision.py`, and all other §12 gates run clean
- [ ] Stopping rule check (§1): post-round audit yields 0 CRITICAL + < 3 HIGH,
      OR the next round is scheduled with explicit owner and target date

When all checked, the round is closed.
