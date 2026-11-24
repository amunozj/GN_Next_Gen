# Plan: Step 2 scorer follow-ups (out-of-band)

- Raised: 2026-11-24, in session, from PI questions about the Step 2 results doc
- Status: **implemented 2026-11-24** on branch `step-2-scorer-followups`, except the
  grid run itself. The PI chose, in session: the full two-component decomposition for
  item 1, and reselect-and-regrid for item 4 with **the PI launching the run** (plan
  decision A13 — no run over ~15 minutes from a Claude Code session).
- **Outstanding, and it is the only outstanding thing:**

  ```bash
  ./scripts/run_step2_grid.sh --detach
  ```

  It regenerates the 51 worlds, re-fits 11 methods over them, re-scores, and redraws
  every figure; about 20 minutes on 64 cores. Until it runs, §§4–9 of the results doc
  carry the 2026-11-23 numbers and say so in a status box. Staleness is enforced, not
  trusted: `is_complete` refuses a world whose `generator_version` is older than the
  code's, and `cell_done` refuses a fit whose parquet lacks `gn_sd_factor`, so no
  stage can silently skip stale work and no `--force` is needed.
- Scope, per item. **Items 1-3 are scorer and nulls only**: no baseline is re-fitted
  and no world regenerated, so the whole re-run is `score_benchmark.py` plus
  `make_step2_figures.py` (~16 min on 64 cores, less if only scoring is redone) —
  the 51 worlds and 459 fits stay on disk and both runners skip existing cells.
  **Item 4 changes a sealed observer parameter**, so it regenerates the worlds and
  re-fits. **Item 5 is documentation and one figure**, and touches no run at all.
- Affects: `docs/results/step-2-blind-benchmark.md` §4, §5, §7, §11a, the section
  numbering from §10 on, f18, f20, f23, f25 and f27, and the summary table's
  coverage row. §6, §8 and §9 keep their claims; §8's can only strengthen.

## 1. Yearly coverage treats a year's daily errors as independent

**What.** `uncertainty_scores` aggregates daily `gn_sd` to the year as
`sqrt(mean(sd^2) / n_days)` (`src/gn_next_gen/benchmark/scoring.py`). The dominant
term in a chained method's `sd` is the observer's calibration-factor error, which is
identical on every day of the year and does not average down. The rule shrinks each
interval by roughly a factor of 19.

**Why it matters.** Measured on 3 S2 worlds, carrying the factor error as fully
correlated instead moves nominal-50% coverage from 0.177 to 0.810 (HoSc98), 0.161 to
0.824 (Pathfinding), 0.145 to 0.730 (SvSc16), 0.064 to 0.781 (VHS24) and 0.064 to
0.613 (CEA17) — a sign change in §5's finding. UEA21 (0.025 to 0.116) and GRP (0.014
to 0.246) undercover under either rule.

**Fix.** Neither rule is right: the propagated-factor component is fully correlated
within a year, the between-observer-spread component is not, and `combine_observers`
collapses both into one `sd` column. So:

1. `combine_observers` (`src/gn_next_gen/baselines/base.py`) emits `gn_sd_factor` and
   `gn_sd_spread` beside `gn_sd`, which keeps every current consumer working.
2. `CalibrationResult.series` documents the two new optional columns; a method that
   supplies only `gn_sd` (CEA17's PMF, both nulls) is aggregated as today, stated as
   such.
3. `uncertainty_scores` aggregates as
   `sqrt(mean(sd_factor)^2 + mean(sd_spread^2) / n_days)`.
4. Add a scorer test that exercises the aggregation with **many days per year** —
   the current nominal-coverage test sets one day per year, so the division is a
   no-op and cannot catch this (`tests/test_scoring.py`).
5. Re-score, redraw f20, rewrite §5 and the summary table's coverage row.

**Recorded meanwhile** as §11 item 8 of the results doc, with §5 flagged as a lower
bound.

## 2. The random-segmentation floor is the weak one

**What.** §4 reports that four of seven methods lose to `null-segmentation`. A plain
daily **median of the raw, uncalibrated reported counts** is a much harder floor,
because the segmentation's random `exp(N(0, 0.3))` factors average out over ~450
observers and a year of days — it is the naive mean plus noise.

**Measured** on the same 3 S2 worlds (yearly cRMSE): HoSc98 0.343, **naive-median
0.391**, **naive-mean 0.401**, Pathfinding 0.448, SvSc16 0.479, VHS24 0.670,
naive-max 0.671, null-segmentation 0.711, UEA21 0.864, CEA17 0.966, GRP 1.011,
null-prior 2.394. **Six of the seven lose to the median of the raw counts**; only
HoSc98 beats it.

**Fix.** Add `naive-mean` and `naive-median` to `NULLS`
(`src/gn_next_gen/baselines/nulls.py`), ~15 lines each, no graph and no reference
observer. Keep `null-segmentation`: it instantiates Lindau & Venema's claim about
*segmented* adjustment and isolates "did your breakpoints carry information", which a
naive median cannot. Report the new floors on the **cRMSE column only** — a naive
median has no anchoring, so its bias is meaningless and it reports no uncertainty, so
no coverage, CRPS or PIT. Then rewrite §4's verdict around the harder floor.

## 3. "Beats the null?" should be a paired test

§4's column is editorial: a hand comparison of two pooled means. Make it a per-world
paired comparison — in how many of the 51 worlds does each method beat the floor —
computed in `score_benchmark.py` rather than written by hand.

## 4. `clean-modern` measures era activity, not acuity

**What.** §7 reports that the two acuity methods return a median **300 MSH against a
true 10** for `clean-modern`. Diagnosed on all ten S2 worlds: the recovered acuity
tracks how active that observer's window was in each world, not the observer.
Spearman rho between the window's true mean count and UEA21's recovered acuity is
**-0.93** (p = 1e-4); against the spotless fraction, **+0.93**. In the one world
whose window is more active than the anchor epoch (5.27 groups/day against 3.23),
UEA21 returns **12 MSH against a true 10** — essentially exact.

**Why.** Two stacked causes, one ours and one the method's.

- *Ours.* `clean-modern` is selected as the densest post-1959 calendar
  (`src/gn_next_gen/benchmark/case_studies.py`), which is station **722**, whose real
  career is **517 days, 1992-01 to 1993-05** — about 17 fixed months against
  `MIN_MONTHS = 12`. All fifty replicates reuse that one window, and the replay drops
  a different synthetic Sun into it each time (window mean 0.02 to 5.27 groups/day
  across the ten worlds). `acuity.py` itself notes that recovery correlation rises
  from 0.27 to 0.61 for observers with 3,000+ observing days.
- *The method's, and published.* The ADF fingerprint cannot separate "few active days
  because the observer missed groups" from "few active days because the Sun was
  quiet". The library is built once over the anchor epoch (mean 3.23 groups/day, 26%
  spotless) with no conditioning on activity or cycle phase — deliberately, since
  phase conditioning belongs to the 2017 scanning-window variant that was not ported.
  So a quiet-era observer has an ADF below every library entry and the fit runs to the
  grid ceiling: in world 1 the observer's ADF median is 0.000 while the lowest the
  library reaches is 0.600 at 300 MSH, and the distance falls monotonically to that
  top. That is the cap A157 documents for UEA21. At the good end the library saturates
  — ADF mean 1.000 at acuity 0 against 0.975 at 10 MSH — so acuities below ~30 MSH are
  barely distinguishable even with matched activity.

**Fix.** Reselect `clean-modern` as the densest calendar with **>= 3,000 observing
days** so the case study tests a clean record rather than a short one; add the
window's true mean activity as a covariate to f23 and f25 so the +89/+87 MSH bias is
diagnosed rather than merely reported; and say in §7 that the bias is largely
era-activity mismatch, quoting the one matched world's 12-against-10. Case-study
selection changes the sealed observer parameters, so this one **does** require
regenerating the worlds and re-fitting — unlike items 1-3, it is not scorer-only.

## 5. The addendum becomes a comparison of all eight series

**What.** §11a is framed around *acquisition* — which two series arrived late, that
CDS was reachable, who supplied the supplement — and f27 draws the three newly added
in colour over the five already held. The framing has no reader value now: what
matters is how the eight series compare, not when each was obtained.

**Fix.**

1. **Promote it to a numbered results section** placed **before** the limitations:
   new **§10. The eight published series, compared**, with `Reproducing this` becoming
   §11, `What this step does not establish` §12 and `What Steps 3-4 inherit` §13.
   Cross-references to fix: §5's pointer to `§11.8` becomes `§12.7`; `§§1-10` in the
   old addendum text; `§11a` at the end of the struck-through limitation; and in
   `docs/results/step-2-verification.md`, the references to §11 and §11a.
2. **Redraw f27 with all eight on equal footing** — no "newly added" styling — in
   `viz.published_series_figure`. Keep the lower panel: the 12.08 check is what
   licenses one axis (**D24**), but it is a caption, not a narrative.
3. **Make the section a comparison, with the numbers that matter.** They are already
   computable from the frozen table: over the common modern window **1749-1995** the
   series agree to about 25% (mean daily group count 3.83 to 4.80 — VH-Hoyt-update
   3.83, HoSc98 3.84, Usoskin 4.19, SvSc16 4.46, CEA17 4.52, VH2024 4.80), while over
   the Maunder Minimum **1645-1715** they span **0.05 to 3.29**, a factor of **66**
   (HoSc98 0.05, VH-Hoyt-update 0.05, VH2024 0.10, Vaquero 0.20, SvSc16 0.98,
   Zolotova 3.29). Note that Vaquero and Zolotova are partial and irregularly sampled,
   so those two are windows, not full series.
4. **Tie it to §6, which is the point.** Five of the eight series were produced by
   methods scored in this benchmark (HoSc98, SvSc16, CEA17, UEA21, VHS24), and the
   published disagreement is concentrated exactly where the benchmark finds no skill:
   §6 measures a 23-25x overestimate of the Maunder level by the only two methods that
   reach the era, and the published spread there is 66x. The section's claim should be
   that the benchmark *explains the size of* the published disagreement, not merely
   that the disagreement exists.
5. **Drop the acquisition narrative**, including the struck-through item 6 of the
   limitations list. Provenance stays where it belongs: `MANIFEST.csv`,
   `provenance.json` and the folder README.
6. **Keep, relocated**, the two substantive items the addendum carries: the D25
   sentinel correction (one line, with the detail left in `known_issues.md`) and the
   rule that `validate_run_config` refuses a `kind: published_series` dataset in role
   `calibration` in **both** variants, which is about these series specifically and
   belongs in the new section's closing note. The paragraph on the supplement's
   corrected 1610-2022 observation database stays too — it is the only thing in hand
   that runs past v1.3's 2010 end.

## What was done, per item

| item | code | docs | needs the re-run |
|---|---|---|---|
| 1 coverage aggregation | `base.py` emits `gn_sd_factor`/`gn_sd_spread` from `combine_observers` and a new shared `inverse_variance_daily` used by CEA17, UEA21, GRP and the Pathfinding re-pool; `scoring._yearly_sd` pools them; 4 new tests, two with 200-day years | §12.7 rewritten; §5 flagged | yes |
| 2 naive floors | `naive-mean`, `naive-median` in `NULLS` and in `configs/step2_baselines.yaml`; registered in the viz order, markers and linestyles | §12.8 rewritten | yes |
| 3 paired floor test | `score_benchmark._floor_comparison` writes `floors.csv`: per-world wins per method per floor | §12.8 | yes |
| 4 `clean-modern` | `MIN_CLEAN_MODERN_DAYS = 3_000`; station 722 (517 days) → 459 (6,922 days, 1977–1995); f23 gained an era-activity panel; the scorer writes `era_mean_truth` and `era_spotless_fraction` per case-study row | §12.9 | yes |
| 5 published series | f27 redrawn: all eight alike, smoothed over one cycle on a log axis, plus a window-comparison panel and the 12.08 check; `series_window_means` returns the numbers the prose quotes | §11a → §10, renumbering §§10–13; acquisition narrative dropped | **no** — done and committed |
| — | marker fallback no longer collides with a named method's shape once there are more than nine methods | — | no |

## Approval

- [ ] PI approved: _date_
