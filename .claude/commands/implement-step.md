---
description: Implement an approved roadmap step plan
argument-hint: <step number 1-6>
---

You are implementing **roadmap step $ARGUMENTS** of GN_Next_Gen.

1. Read `CLAUDE.md`, `docs/05_roadmap/step-$ARGUMENTS-*.md`, and `docs/plans/step-$ARGUMENTS-plan.md`.
   **Stop** if the plan does not exist or its Approval section does not say "Approved"; tell the PI to run
   `/plan-step $ARGUMENTS` and approve it.
2. Create or switch to branch `step-$ARGUMENTS-<short-slug>` (never commit to `main` directly).
3. Implement the plan's **Files** and **Tests** sections in small, reviewable commits. Each commit leaves the fast test
   suite passing (`pytest`). Write tests alongside code, including synthetic-recovery tests for estimators.
4. Follow the hard rules in `CLAUDE.md`: never modify `data/raw/` or `legacy/`; new raw data need a `MANIFEST.csv` row;
   keep the firewall (ADR-002); no SILSO in calibration; reproducible configs with seeds in `configs/`; large outputs in
   `outputs/` only.
5. If you must deviate from the plan, stop and ask the PI, then record the deviation in the plan (append a
   "Deviations" section) before continuing.
6. Update the step's row in `docs/05_roadmap/README.md` to "In progress" when you start and "Implemented" when done;
   tick any paper items produced.
7. Do not push or open a PR unless the PI asks.
8. Finish with: what was built (files), test results (pasted summary), anything left undone, and the suggestion to run
   `/verify-step $ARGUMENTS`.
