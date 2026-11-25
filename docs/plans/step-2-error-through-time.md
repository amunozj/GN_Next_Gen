# Plan: Step 2 — error through time, per method (out-of-band)

- Raised: 2026-11-24, in session, by the PI: "how do the errors evolve with time
  for each method combining all worlds … in which eras do different methods
  perform better".
- Status: **done 2026-11-24.** Implemented, re-scored, figures rebuilt, reports
  updated. No baseline re-fit and no world regeneration was needed.
- Affected: `src/gn_next_gen/benchmark/scoring.py`, `scripts/score_benchmark.py`,
  `src/gn_next_gen/viz/benchmark.py`, `scripts/make_step2_figures.py`,
  `tests/test_scoring.py`, `tests/test_viz.py`; new §5 in
  `docs/results/step-2-blind-benchmark.md` with §5–§13 renumbered to §6–§14; F5,
  A5 and A11 in `docs/results/step-2-verification.md`; three bullets in
  `docs/05_roadmap/README.md`.

## Why

Every Step 2 number was pooled over the whole record. `scores.csv` collapses a
world × method to one row, so the report could say HoSc98 wins and could not say
*when*. `scoring.align` already built the per-day truth-beside-estimate frame for
every cell and threw it away after reducing it to scalars, so the year axis was
one function call from existing — the data loss was in what got written, not in
what was kept.

Two open items fell out of the same machinery:

- The Step 2 plan's **A5** asks for coverage "stratified by era"
  (`step-2-plan.md:246`, `:374`). `_coverage_table` grouped by scenario only; the
  clause was never implemented.
- **F5** in the verification doc tabulated coverage by era, by hand, on
  2026-11-22, and was never recomputed after the 2026-11-24 `_yearly_sd` fix. Its
  numbers were the pre-fix bug's.

## What was built

- `scoring.ERAS` — nine windows tiling 1610–2026, every boundary traceable to
  something in the repo (`MAUNDER_YEARS`, the two zero-edge cuts in f28, ADR-009's
  `rgo_15_19`, the v1.3 database's last year). Named for the network, not for
  solar history: the truth is a replay of RGO cycles, so only the calendars are
  real.
- `scoring.yearly_frame` — extracted from `uncertainty_scores`, which already
  built it internally. **Both the pooled coverage and the era coverage now read
  this one frame**, which is the fix for how F5 drifted from `coverage.csv` in the
  first place; a test asserts the per-year interval flags average to exactly what
  `uncertainty_scores` reports.
- `scoring.yearly_errors`, `pool_yearly`, `era_errors` — the per-year residual
  with its era and interval flags, and the scenario-balanced pooling over
  S2/S3/S4. Equal weight per scenario, not per world: S1 holds 20 of the 51 and
  would otherwise decide the answer.
- `score_benchmark.py` writes `yearly_errors.csv`, `yearly_pooled.csv` and
  `era_scores.csv`; `make_step2_figures.py` writes the committed
  `docs/results/data/step2_era_errors.csv` and builds **f41** (error through time,
  with a winner ribbon) and **f42** (bias through time, coverage by era).
  f29–f40 are Step 3's, which is why the codes start at f41.

`scores.csv` came out **byte-identical** after the refactor (md5
`333f2dc8…`), which is the check that the extraction moved nothing.

## What changed in the conclusions

1. **HoSc98 wins two eras of eight.** CEA17 wins `anchor-epoch` (1912–1969) at
   0.125 against HoSc98's 0.277, and is 16× worse than its own best in 1749–1798.
   VHS24 wins 1874–1911, Pathfinding 1970–2010, UEA21 the two eras only it and GRP
   reach. Distance from the anchor, not the method, is what sets the error.
2. **Exactly one era has any published method beating the floor.** The best
   method's RMSE divided by `naive-median`'s is above 1 in seven eras of eight;
   only `anchor-epoch` is below, at 0.55. The published methods carry information
   the raw counts do not only where the reference observer already looks.
3. **CEA17's calibration is not uniform.** It over-covers at 0.888 inside the
   anchor epoch and under-covers at 0.224 after it; the pooled 0.591 is the
   average. §6's sentence "none of this is a quirk of one era" was wrong and is
   corrected.
4. **VHS24 is centred only by cancellation** (+1.10 before 1800 against −0.54
   after 1970); **CEA17 is centred in every era separately**. §4 called both
   "centred and useless" on one pooled number.
5. **F5's direction holds, its numbers were low by about 2×.** HoSc98 runs
   0.534 → 0.257 from 1749 to 2010, not 0.218 → 0.082.
6. **`pre-network` (1610–1644) is empty, and that is a measurement.** An S2 world
   has *zero* observation rows before 1660 against 2,855 in the perfect-observer
   world S5 on identical calendars: real acuity plus MNAR omission of spotless
   days erases the first half-century before any method sees it.

## Checks

- S1 is excluded from the pooled curve and kept as its own rows in
  `era_scores.csv`. **It picks the same winner as the pooled curve in all four
  eras it covers** — VHS24, CEA17, Pathfinding, GRP — so the ranking is not an
  artifact of which scenarios were pooled.
- Re-aggregating the per-year interval flags reproduces `coverage.csv` exactly
  (max |diff| = 0.0 at nominal 0.50 over every scenario × method).

## A defect found while drawing f41

The first version of the floor-ratio panel divided the yearly RMSE by the floor's
yearly RMSE. The floor's pooled RMSE passes through 0.0008 in 1878 and 0.0002 in
1968 — years where the median of the raw counts happens to sit on the truth in
every world at once — so the ratio spiked past 1000 and buried the comparison.
The panel now pools the squared error over the cycle window first and then roots
it, which removes the singularity without clipping anything. A test pins it.

## Reproducing

```bash
conda activate gn-next-gen
python scripts/score_benchmark.py      # ~2 min, reads the baseline parquets on disk
python scripts/make_step2_figures.py   # f41, f42, and step2_era_errors.csv
pytest && pytest -m slow && ruff check . && ruff format --check .
```
