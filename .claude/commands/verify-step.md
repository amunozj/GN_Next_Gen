---
description: Verify an implemented roadmap step against its acceptance criteria
argument-hint: <step number 1-6>
---

You are verifying **roadmap step $ARGUMENTS** of GN_Next_Gen. Report outcomes faithfully: a failed criterion is a
failure, with the evidence pasted.

1. Read `CLAUDE.md`, `docs/05_roadmap/step-$ARGUMENTS-*.md`, and the approved `docs/plans/step-$ARGUMENTS-plan.md`
   (including any Deviations section).
2. Record the environment (env file, CPU/GPU, `python -c "import jax; print(jax.__version__, jax.devices())"`), branch
   and commit.
3. Run the full test suite (`pytest`, then `pytest -m slow` if the plan defines slow tests) and paste the summaries.
4. For every acceptance criterion in the plan's table, run the measurement it specifies (scripts/configs from the plan)
   and record value, threshold and pass/fail. Do not loosen thresholds; if a threshold looks wrong, say so separately.
5. Check the hard rules: `git diff main -- data/raw legacy` is empty (except new raw folders with manifest rows);
   firewall respected for sunspot-only runs; no SILSO in calibration code paths.
6. Write `docs/results/step-$ARGUMENTS-verification.md` from `docs/templates/step-verification.md`, including findings
   that affect later steps (and propose roadmap/ADR changes there rather than editing other specs silently).
7. Update the status table in `docs/05_roadmap/README.md` (Verified / Pass with caveats / Fail).
8. Summarize the verdict and the evidence for the PI.
