---
description: Draft the implementation plan for a roadmap step (no code)
argument-hint: <step number 1-6>
---

You are planning **roadmap step $ARGUMENTS** of GN_Next_Gen. Do not write or modify any code, data or legacy files in
this command; the only file you create is the plan.

1. Read, in order: `CLAUDE.md`; `docs/05_roadmap/README.md` (check the status table: earlier steps this one depends on
   should be Verified — if not, say so and ask the PI whether to proceed); the spec
   `docs/05_roadmap/step-$ARGUMENTS-*.md`; every item in the spec's **Required reading**; `docs/06_decisions/README.md`
   and any ADR the spec cites; all files in `docs/results/` from earlier steps; any existing
   `docs/plans/step-$ARGUMENTS-plan.md` (if it exists, you are revising it).
2. Inspect the current code (`src/`, `tests/`, `configs/`, `scripts/`) so the plan reuses what exists. Check
   `docs/03_legacy/reuse_map.md` for legacy pieces to port.
3. Ask the PI the spec's **Open questions** plus any new ones that change the design (use the question tool; offer a
   recommended option). Do not guess on decisions that belong to the PI.
4. Write `docs/plans/step-$ARGUMENTS-plan.md` from `docs/templates/step-plan.md`, with Status **Draft**. Make every
   acceptance criterion in the spec concrete and measurable. Cite report sections and ADRs for design choices. Respect
   the hard rules in `CLAUDE.md` (firewall, estimand, no SILSO in calibration, immutable data/raw and legacy).
5. If the plan needs a decision that contradicts an ADR, propose a new ADR in the plan's Open questions instead of
   silently deviating.
6. Update the step's row in the status table only if the PI approves the plan in this session ("Planned").
7. End with a short summary: the plan path, the main design choices, and the open questions awaiting the PI.
