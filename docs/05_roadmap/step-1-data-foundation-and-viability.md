# Step 1 — Data foundation and viability checks

## Goal
Trustworthy, tested loaders for all raw data; resolution of the known data issues; current data acquired side by side;
and two empirical checks that decide whether (and in what form) the Step 4 latent-population engine is viable.

## Why
The joint model rests on two untested assumptions (assessment §4, research report "Two empirical checks"): that an
immigration-death chain plus observer thinning reproduces the reference's daily statistics, and that the area
distribution of groups, conditional on activity, is stable across the reference era and (by assumption) across
centuries. Both can be checked cheaply with data already in the repo.

## Inputs
`data/raw/*` (see `docs/04_data/`), `docs/04_data/known_issues.md` (D1–D9), `docs/04_data/refresh_todo.md`.

## Deliverables
1. **Loaders** (`gn_next_gen.io`): observations (long format: date, ordinal, station, groups), observers, reference group
   areas (long: date, group slot, area; plus a per-day status observed/spotless/no-observation), SILSO (comparison only).
   Cleaned outputs cached to `data/interim/` (parquet), regenerated on demand.
2. **Data-issue resolutions** for D1–D5, D8, D9 with tests; a characterization of the DRGO vs SC_SP_RG differences (D3)
   and an ADR choosing the reference file and the **anchored reference epoch** (ADR-003).
3. **Data refresh** per `refresh_todo.md`: new raw folders + `MANIFEST.csv` rows (URL, date, version, SHA-256,
   license). At minimum the latest SILSO GN database and the Debrecen catalogue to present; record what was unavailable.
4. **Check A — dynamics:** fit immigration-death models (survival φ, births λ(t) smooth) with 1 and 2 area classes to the
   reference daily group counts, at several acuity thresholds (e.g. 0, 10, 30, 60, 100 MSH). Compare simulated vs
   observed: daily autocorrelation (lags 1–30 days), zero fraction and count distribution as functions of activity,
   run lengths of spotless and active days. If the per-group RGO catalogue is acquired, also compare lifetime
   distributions directly.
5. **Check B — stationarity:** compare the area distribution of groups conditional on activity level across epochs
   (RGO early/late, Debrecen, Kislovodsk) and across cycles; test the two-population (small/large group) structure.
6. **Results doc** `docs/results/step-1-*.md` with a go/no-go and concrete settings for Step 4: number of area classes,
   state-space truncation K (note observers report up to 58 groups/day; reference max 26), lifetime parametrization,
   and any change of plan.

## Acceptance criteria (quantify in the plan)
- Loader tests: row counts and coverage match `MANIFEST.csv`; sentinels handled; area columns always all 26; leap-day
  handling verified; `pytest` fast suite < ~1 min.
- Every known issue D1–D9 has a status (resolved / documented / deferred with reason).
- Check A: pre-registered discrepancy measures (e.g. ACF RMSE, zero-fraction error per activity bin) with pass
  thresholds; decision on 1 vs 2 classes justified by them.
- Check B: a stated test and result; if non-stationary, the consequence for the detection layer is written down.

## Decisions already made
Estimand = reference-equivalent count (ADR-003); no SILSO in calibration (diagnostic comparison allowed, but activity
conditioning in the checks should use the reference's own smoothed counts); data scope (ADR-004); raw data immutable (ADR-006).

## Open questions for the planning session
1. Which reference file (DRGO vs SC_SP_RG) and which epoch to anchor? (Needs the D3 characterization first.)
2. Acquire the per-group RGO catalogue (with group identities) now, for lifetime checks?
3. Fitting method for Check A (maximum likelihood with the forward filter is natural and also prototypes Step 4's engine).
4. Which activity variable to condition on in Check B (smoothed reference counts vs total area)?

## Required reading
`CLAUDE.md`; `docs/04_data/*`; `docs/06_decisions/ADR-003*`, `ADR-004*`, `ADR-006*`; `docs/03_legacy/reuse_map.md`;
`docs/02_research/adjacent-methods-report.md` sections "Integer autoregressions give lifetimes…", "M/G/∞ inference…",
"Mark-dependent thinning…", "Size-biased detection…", "A 41-state forward filter…"; notes
`counting_processes_underreporting.md` and `sbi_and_scalable_inference.md` (state-space sizing).

## Out of scope
Observer calibration, benchmarks, the full joint model, proxies.
