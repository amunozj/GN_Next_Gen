# Step N verification: <title>

- Plan: `docs/plans/step-N-plan.md` (approved <date>)
- Branch / commit verified: <branch> @ <sha>
- Environment: <environment file, CPU/GPU, JAX version>
- Date: <date>

## Acceptance criteria

| Criterion | Measurement | Result | Pass? |
|---|---|---|---|

## Test suite
Command(s) run and summary (passed / failed / skipped). Paste failures verbatim.

## Figures
**Embed every figure referred to anywhere in this document**, each with a caption (CLAUDE.md, "Reports and
figures"). One block per figure:

```markdown
![what the panels show](figures/step-N/fNN_slug_light.png)

**Figure fNN** — what to look at, then the claim it supports.
```

Embed the **light** variant as plain Markdown; see CLAUDE.md for why a theme-switching `<picture>`
block does not work on GitHub.

Committed figures live under `docs/results/figures/step-N/` in both modes and stay small; the CSVs behind them
stay in git-ignored `outputs/` and are listed here by path.

## Deviations from the plan
What changed and why.

## Findings that affect later steps
Including any change to the roadmap, a new known issue, or a needed ADR.

## Verdict
Pass / Pass with caveats / Fail. Update the status table in `docs/05_roadmap/README.md`.
