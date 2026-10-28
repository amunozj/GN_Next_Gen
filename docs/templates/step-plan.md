# Step N plan: <title>

- Spec: `docs/05_roadmap/step-N-<slug>.md`
- Author: Claude Code session, <date>
- Status: **Draft** <!-- Draft → Approved (by PI, with date) → Superseded -->

## Context
Why this step, what earlier steps delivered (link `docs/results/`), and what this plan must decide.

## Scope
In scope / out of scope (copy the spec's out-of-scope list and add anything new).

## Design
Data flow, model/algorithm choices with justification (cite research report sections and ADRs), parameters and
defaults, how the firewall (ADR-002) and estimand (ADR-003) are respected.

## Files
New/modified files under `src/`, `tests/`, `scripts/` or `notebooks/`, with one line each. Legacy code to port (cite
`legacy/...` paths and `docs/03_legacy/reuse_map.md`).

## Tests
Unit tests, property tests, synthetic-data recovery tests; how long the fast suite takes.

## Acceptance criteria
Map every criterion in the spec to a concrete, checkable measurement (number, threshold, figure). Add any new ones.

| Spec criterion | Measurement | Pass condition |
|---|---|---|

## Compute and data
Expected runtime/memory (CPU vs GPU), new data needed (and `MANIFEST.csv` rows).

## Risks
What could fail, how it would show, fallback.

## Open questions for the PI
Numbered; the plan is not approved until these are answered.

## Approval
<!-- PI writes: "Approved <date>" or requested changes. /implement-step refuses to run without "Approved". -->
