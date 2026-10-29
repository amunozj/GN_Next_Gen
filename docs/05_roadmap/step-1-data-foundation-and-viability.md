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
`data/raw/*` (see `docs/04_data/`), `docs/04_data/known_issues.md` (D1–D21), `docs/04_data/refresh_todo.md`.

**The 2026-10 raw-data renovation has already landed** (`docs/plans/raw-data-renovation-plan.md`, `docs/results/raw-data-renovation.md`, ADR-009):
the reference is now `reference_groups_2026-10/` (five per-survey per-group catalogues, 1825–2026, with group
identities and lifetimes), the observation database is `gn_database_v1.3/` (1610–2010), and
`gn_observer_extension_2026-10/` carries daily group counts from KMAS and SWPC to 2026. Deliverable 3 below is
therefore largely done; what remains of it is the SILSO SN refresh and the published GN comparison series.

## Deliverables
1. **Loaders** (`gn_next_gen.io`): observations (long format: date, ordinal, station, groups), observers, the reference
   group catalogue (already long: survey, group, date, areas, plus `day_status` per survey-day) and the observer-count
   extension, SILSO (comparison only). Cleaned outputs cached to `data/interim/` (parquet), regenerated on demand.
   The reference no longer needs un-pivoting from `AREA1..AREA26`; what it does need is a **documented resolution rule
   for D16** (217 duplicate `(date, station)` keys in v1.3, 95 with conflicting counts) before any loader keys on
   `(date, station)`, and a **config-level validator for the KMAS single-use rule** (ADR-009).
2. **Data-issue resolutions.** D1–D7 and D9 already have a status in `docs/04_data/known_issues.md`, and ADR-009 has
   chosen the reference and the anchored epochs. What remains for Step 1: **D16** (duplicate-key rule), **D17**
   (robust K, below), **D22** (exclude the colliding KMAS groups from lifetime work), **D11** (Spörer's area
   unit, if it can be settled), a config-level validator for the KMAS single-use rule, and a status for
   D10–D22 at the end.
3. **Data refresh** per `refresh_todo.md`: new raw folders + `MANIFEST.csv` rows (URL, date, version, SHA-256,
   license). At minimum the latest SILSO GN database and the Debrecen catalogue to present; record what was unavailable.
4. **Check A — dynamics:** fit immigration-death models (survival φ, births λ(t) smooth) with 1 and 2 area classes to the
   reference daily group counts, at several acuity thresholds (e.g. 0, 10, 30, 60, 100 MSH). Compare simulated vs
   observed: daily autocorrelation (lags 1–30 days), zero fraction and count distribution as functions of activity,
   run lengths of spotless and active days. **Lifetimes are now available directly** — `group_single_entry.parquet`
   carries `obs_duration` for every group in all five surveys — so compare lifetime distributions directly rather
   than inferring them. **But exclude the 289 KMAS groups flagged by D22 first**: KMAS reuses a `group_id` for
   physically distinct groups on 414 group-days, which inflates their apparent lifetime (median 11 days against
   5 for clean KMAS groups). RGO and Debrecen are clean and are the safer basis for the lifetime comparison.
5. **Check B — stationarity:** compare the area distribution of groups conditional on activity level across epochs
   (RGO early/late, Debrecen, Kislovodsk) and across cycles; test the two-population (small/large group) structure.
6. **Results doc** `docs/results/step-1-*.md` with a go/no-go and concrete settings for Step 4: number of area classes,
   state-space truncation K, lifetime parametrization, and any change of plan. **K must come from a robust upper
   quantile of reported counts, not from the maximum** — see D17: v1.3's two largest values (93 and 58) are
   implausible outliers above a tail that stops at 53, the 93 is new in v1.3, and the reference survey's all-time
   daily maximum is 26. The 99.99th percentile of `GROUPS` is below 40.

## Acceptance criteria (quantify in the plan)
- Loader tests: row counts and coverage match `MANIFEST.csv`; sentinels handled; leap-day handling verified; `pytest`
  fast suite < ~1 min. (The "area columns always all 26" criterion is retired: the reference is long-format now, and
  the snapshot's own invariants are covered by `tests/test_reference_groups_snapshot.py`.)
- Every known issue D1–D22 has a status (resolved / documented / deferred with reason).
- Check A: pre-registered discrepancy measures (e.g. ACF RMSE, zero-fraction error per activity bin) with pass
  thresholds; decision on 1 vs 2 classes justified by them.
- Check B: a stated test and result; if non-stationary, the consequence for the detection layer is written down.

## Decisions already made
Estimand = reference-equivalent count (ADR-003); reference and anchored epochs (ADR-009); no SILSO in calibration
(diagnostic comparison allowed, but activity conditioning in the checks should use the reference's own smoothed
activity — the `amp_*` columns are exactly that); data scope (ADR-004); raw data immutable (ADR-006).

## Open questions for the planning session
1. ~~Which reference file and which epoch to anchor?~~ **Answered by ADR-009**: the reference is
   `reference_groups_2026-10/`, with two co-equal anchors — Debrecen cycles 21–23 (primary) and RGO cycles 15–19
   (secondary). Steps 3–4 report the series under both.
2. ~~Acquire the per-group RGO catalogue for lifetime checks?~~ **Done** — the reference is per-group throughout, with
   `obs_duration` for every group in all five surveys.
3. Fitting method for Check A (maximum likelihood with the forward filter is natural and also prototypes Step 4's engine).
4. Which activity variable to condition on in Check B (smoothed reference counts vs total area)?

## Required reading
`CLAUDE.md`; `docs/04_data/*`; `docs/06_decisions/ADR-003*`, `ADR-004*`, `ADR-006*`, `ADR-009*`;
`docs/results/raw-data-renovation.md`; `docs/03_legacy/reuse_map.md`;
`docs/02_research/adjacent-methods-report.md` sections "Integer autoregressions give lifetimes…", "M/G/∞ inference…",
"Mark-dependent thinning…", "Size-biased detection…", "A 41-state forward filter…"; notes
`counting_processes_underreporting.md` and `sbi_and_scalable_inference.md` (state-space sizing).

## Out of scope
Observer calibration, benchmarks, the full joint model, proxies.
