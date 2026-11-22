# Step 1 plan: Data foundation and viability checks

- Spec: `docs/05_roadmap/step-1-data-foundation-and-viability.md`
- Author: Claude Code session, 2026-10-29; **revised 2026-11-21** (v1.3_4 workbook; visual-first verification)
- Status: **Approved 2026-11-21** (PI, in session) <!-- Draft → Approved (by PI, with date) → Superseded -->

## Context

Step 1 is the last step before modelling. It owes three things downstream: tested loaders that every later
step reads data through, a resolved status for every known data issue, and the two empirical checks that
decide whether the Step 4 latent-population engine is viable at all (assessment §4; report section
"A 41-state forward filter…").

The 2026-10 raw-data renovation (`docs/results/raw-data-renovation.md`, ADR-009) landed the reference
(`data/raw/reference_groups_2026-10/`, five per-survey per-group catalogues 1825–2026), the observation
database v1.3 (1610–2010), and the post-2010 count extension. It left four follow-ups that land here:
**D16** (duplicate `(date, station)` keys), **D17** (robust `K`), **D22** (colliding KMAS group ids), and a
config-level validator for the **KMAS single-use rule**.

Since that plan was drafted the PI has supplied **`Revised collection_v1.3_4.xlsx`**, which fixes most of what
the renovation reported upstream. Step 1 uses it as the observation database (clarification 1 below), and all
verification in this step is **figure-first** (clarification 2).

## Clarification 1 — the observation database is `Revised collection_v1.3_4.xlsx`

The new workbook is in the working tree at `data/raw/gn_database_v1.3/Revised collection_v1.3_4.xlsx`
(28,239,022 bytes, untracked). I audited it read-only during this planning session; every number below is
measured, and all of them move the plan.

| | v1.3_3 (frozen) | **v1.3_4 (new)** |
|---|---|---|
| Data rows | 1,156,673 | **1,156,368** |
| Rows `STATION > 0` | 1,100,144 | **1,099,986** |
| Stations with data | 740 | **740** (unchanged) |
| Observers listed | 775 | **776** (station 776 is new) |
| Observers with `TOT_OBS = 0` | 33 | **34** |
| Coverage | 1610-01-01 → 2010-12-31 | **unchanged** |
| **D16** duplicate `(date, station)` keys | 217 (95 conflicting, 444 rows) | **3** (1 conflicting, 6 rows; stations 773 ×2, 768 ×1) |
| **D17** `max(GROUPS)` | 93 | **58** — the implausible 93 is **gone**; the inherited 58 (1992-03-18, station 712) remains |
| `GROUPS` 99.99 / 99.999 pct | 26 / 38 | **25 / 36** |
| **D18** string `'NA'` in `INITIAL`/`FINAL` | 33 observers | **none** — the sentinel is gone |
| **D19** flag column name | `OBSER` | **`OBSERVER`** (`ORDINAL`/`FRACYEAR` still absent, still derived) |
| **D20** "changed in v1.3" | green cell fill only | **explicit column** `CHANGED IN v1.3 (1=YES, 0=NOT)`; 99 cells still carry fill against 94 in _3 |
| Changelog layout | one sheet, four side-by-side blocks | **four sheets** (`Differences-v1_12-HS98`, `-v1_2-v1_12`, `-v1_21-v1_2`, `-v1_3-v1_21`), ~601 entries against 586 |

Observer-day diff against the frozen `_3` parquet: **144 observer-days are new, 78 are gone, and 351 shared
observer-days changed `GROUPS`**. Net rows fall by 305 because removing the ~428 duplicate rows outweighs the
144 additions — so `_4` reads as a correction-plus-deduplication release with a small net reduction, not a net
extension. Coverage still stops at 2010-12-31, so `gn_observer_extension_2026-10/` is still what carries the
record to 2026.

**Four consequences for the work:**

1. **`_4` supersedes `_3` in place (PI, 2026-11-21), which needs ADR-010.** `Revised
   collection_v1.3_3.xlsx` is committed (`6632bdb`) and checksummed in `MANIFEST.csv`, but its working-tree
   copy is **deleted** and `_4` was dropped into the same folder, so `tests/test_data_manifest.py` is red.
   ADR-006 says refreshed data go into a *new* subfolder and never overwrite, so this step carries
   **ADR-010** amending it: a correction release of the same database may supersede its predecessor **inside**
   `gn_database_v1.3/`, provided (a) every release's workbook is retained byte-exact with its own
   `MANIFEST.csv` row, (b) the derived parquet tracks the newest release and its manifest rows move with it,
   and (c) the converter can regenerate any earlier release's tables on demand into `data/interim/`, so
   superseded numbers stay reproducible from retained bytes plus code. First implementation commit:
   `git restore` the `_3` workbook; then land `_4` beside it and regenerate the three derived tables from `_4`.
2. **The converter needs a `_4` reader.** `scripts/convert_gn_database_v13.py` reads a single changelog sheet
   named `"Differences with HS98"`, which does not exist in `_4`; it would crash. It also hard-codes
   `OBSER` and reads `changed_in_v13` from cell fill. Plan: add a release-aware reader
   (`--release v1.3_3|v1.3_4`) that keeps `_3` byte-reproducible (`--check` must still pass) while reading
   `_4`'s four changelog sheets, its `OBSERVER` header and its explicit changed-in-v1.3 column.
   Where the explicit column and the green fill disagree (99 fills vs the column's count), the **column**
   wins and the disagreement is reported as a finding, since `_3`'s fill-only encoding is what D20 was about.
3. **D16–D20 statuses change, and the D16 rule gets cheap.** Only 3 duplicate keys survive and only one
   conflicts — Takuma (station 768) on 1982-10-13 with `GROUPS` 8 and 7, whose neighbouring days run 9
   (10-12), 7 (10-14) and 8 (10-16). The other two are Wolf (station 773) on 1874-04-22 and 1876-05-06, both
   `GROUPS = 0` twice. **PI decision: the default `duplicate_rule` is `max`** — identical duplicates collapse
   and the Takuma day is kept at 8 — with `drop_conflicting`, `first` and `keep_both` retained as tested
   sensitivities. All three keys were already in `_3`; `_4` cleared stations 755, 329 and 245 entirely
   (211 of 217 keys, 94 of the 95 conflicts). Step 2's legacy baselines still read `_3` and v1.22.
4. **`K` is re-derived from `_4`.** Under the pre-registered rule (below) `_4` gives a minimum of **39** and
   `_3` gave **41**; `K = 41` satisfies the rule under both releases and matches the report's figure, so
   **`K` = 41** (states 0…40, tail mass at `K − 1`) is adopted and recorded with both numbers.

Step 2's legacy baselines keep reading v1.22 and `_3`; the lineage record names the release per run.

## Clarification 2 — verification is visual first

Every check, audit and loader test in this step reports **through figures**, not through printed text. Concretely:

- **A figure per acceptance criterion.** The acceptance table below names the figure that demonstrates each
  one. `/verify-step 1` is a review of that figure set plus a green test suite, not a read of console output.
- **`gn_next_gen.viz` is built in this step, not later.** One house style module (`viz/style.py`), one
  function per diagnostic, every function returning a Matplotlib figure and taking an `ax=None`. Scripts
  compose panels; nothing plots as a side effect of importing or loading.
- **Scripts print nothing but progress.** Each writes (a) a multi-panel PDF/PNG to
  `outputs/step1/figures/` (git-ignored), (b) the small summary figures that the results doc embeds
  (committed, ≤ ~300 KB each), and (c) a machine-readable `scores.csv`/`.json` that the results doc's numbers
  and the tests both read. The CSV is the audit trail; the figure is the deliverable.
- **Data-issue audits become plots.** The `_3`-vs-`_4` diff, the duplicate-key timeline, the `GROUPS` tail and
  the `K` justification, the sentinel/coverage maps and the leap-day audit are panels, replacing the
  print-based style of `scripts/audit_gn_database_v13.py` (which stays as is — it backs a published report).
- **Tests stay assertions** — figures are not a substitute for them — but every test that pins a measured
  quantity names the figure that shows it, and the check scripts are smoke-tested (`--quick`) so a broken
  figure fails the suite.
- Exploratory figure work goes to `notebooks/` as py:percent text (jupytext), never as the source of truth.

Figure set (all in `outputs/step1/figures/`, the starred ones committed under `docs/results/figures/step-1/`):

| Figure | Shows |
|---|---|
| `f01_coverage.pdf`* | Observer-days per year 1610–2026, v1.3_4 + extension; stations active per year; sentinel and no-observation map |
| `f02_v13_release_diff.pdf`* | `_3` vs `_4`: new/removed observer-days and changed counts over time, per station; duplicate-key timeline before and after |
| `f03_groups_tail_and_K.pdf`* | `GROUPS` survival function on log axes with the 99.99/99.999 percentiles, the 58, the removed 93, the reference daily maxima, and the chosen `K` |
| `f04_leapday_calendar.pdf` | Feb-29 presence by year against the Gregorian rule; `ORDINAL` ↔ date residuals |
| `f05_reference_daystatus.pdf`* | `day_status` per survey through time — the panel that makes D23 (no spotless days before 1874) visible |
| `f06_lifetimes.pdf`* | Visible-duration histograms per survey, D22-clean, with the 13.5-day disk-passage cliff and the fitted survival curve |
| `f07_checkA_acf.pdf`* | Observed vs simulated ACF, lags 1–30, raw and activity-detrended, per threshold; lag-13 trough and lag-27 bump marked |
| `f08_checkA_zero_counts.pdf`* | Zero fraction and count PMF per activity bin, observed vs simulated; run-length distributions |
| `f09_checkA_scores.pdf`* | The five discrepancy measures against their pass thresholds, 1-class vs 2-class, per survey × threshold — the go/no-go panel |
| `f10_checkB_areas.pdf`* | `log10 area_corr` distributions by epoch within activity bins; percentile ladders; above-threshold fractions |
| `f11_checkB_mixtures.pdf` | 1- vs 2-component lognormal fits per epoch × bin, BIC and component drift |
| `f12_filter_benchmark.pdf` | Forward-filter wall time and peak memory vs `T` and `K`, CPU vs A100, against the report's estimates |

## Scope

In scope:
1. Loaders (`gn_next_gen.io`) with interim parquet caching.
2. Run-config layer with the KMAS single-use validator and per-run data-lineage records (ADR-002).
3. v1.3_4 landed in place under **ADR-010**, release-aware converter, `_3` workbook restored.
4. Status for every issue D1–D23.
5. SILSO SN refresh (comparison only) into a new dated raw folder with `MANIFEST.csv` rows.
6. Check A (dynamics) and Check B (stationarity), with pre-registered scores.
7. `gn_next_gen.viz` and the figure set above.
8. Results doc with go/no-go and the Step 4 settings: area classes, `K`, lifetime parametrization.
9. The Step 4 "early task 1" filter benchmark, since Check A builds that filter anyway.

Out of scope (spec's list): observer calibration, benchmarks, the full joint model, proxies. Added: the
published GN comparison series (PI: SILSO only — it moves to Step 2, where the benchmark consumes it);
recovering Schwabe's spotless days (PI: defer to Step 4); any other change to `data/raw/` or `legacy/`.

## New findings from the planning-session measurements

**N1 — Spörer does not track groups, and neither pre-1874 survey has spotless days.** Measured from
`day_status.parquet` and `group_single_entry.parquet`:

| survey | day_status 0 (no obs) | 1 (spotless) | 2 (active) | groups | obs_duration |
|---|---|---|---|---|---|
| schwabe | 7,377 | **0** | 8,328 | 6,903 | median 6, max 17 |
| sporer | 8,107 | **0** | 4,311 | 5,421 | **≡ 1 for all 5,421** |
| rgo | 561 | 6,709 | 30,350 | 25,611 | median 5, max 22 |
| debrecen | 343 | 2,231 | 13,862 | 11,412 | median 6, max 41 |
| kmas | 1,319 | 3,320 | 22,024 | 22,295 | median 5 (clean), max 375 |

Spörer's `group_obs_kind` is 1 (single-observation group) for every row: there is no multi-day tracking to
recover. **PI decision (2026-11-21): Step 1 excludes both Spörer and Schwabe** — Spörer because it tracks no
groups, Schwabe because it records no spotless days; Schwabe's handling is deferred to Step 4. Both are
registered as **D23** and shown in `f05`. No ADR amendment is needed: ADR-009 fixes what the reference
snapshot *contains*, and this is a scope choice about what one step *uses*.

**N2 — `obs_duration` is visible residence, not intrinsic lifetime.** It equals the observed span
(`max − min + 1`) exactly (RGO: 1.000 agreement; 2,597 RGO groups have interior gaps, so it is *not* the
number of observing days). Its distribution is hard-capped by disk passage — RGO counts at 12/13/14/15 days
are 1,791 / 1,976 / 343 / 25, max 22 — and the empirical survival ratio `S(t+1)/S(t)` sits at 0.87–0.89 for
t = 1…9 before collapsing (0.567 at t = 11, 0.158 at t = 12). A geometric chain alone cannot produce that
cliff. **PI decision: compare visible durations under an explicit rotation kernel** (`f06`), so Step 4's φ is
documented as a visible-residence parameter.

**N3 — v1.3_4 fixes most of what was reported upstream**, as tabulated in clarification 1.

Supporting measurements: RGO daily counts at T = 30 have zero fraction 0.228 and ACF 0.941 (lag 1),
0.640 (lag 13), **0.736 (lag 27)** — the predicted recurrence bump is present. `log10 area_corr` is
consistent across the three modern surveys (RGO 1.94 ± 0.61, Debrecen 1.96 ± 0.63, KMAS 1.93 ± 0.65) and
differs for the early two (Schwabe 1.83 ± 0.51, Spörer 2.32 ± 0.45 — D11 visible). `amp_whole_sm91` is
non-null on ≥ 99.3% of group-days in every survey.

## Design

### Decisions taken by the PI

| Decision | Choice | Why |
|---|---|---|
| Observation database | **v1.3_4**, landed in its own folder; `_3` retained | Clarification 1 |
| Verification style | **Figure-first** | Clarification 2 |
| D16 rule | Collapse identical-value duplicate keys to one row; resolve a conflict by **`max`** (default). Switch: `duplicate_rule: max \| drop_conflicting \| first \| keep_both` | Only one conflict survives in `_4` (Takuma 1982-10-13, 8 vs 7), and 8 matches its neighbours. `drop_conflicting` stays as the sensitivity |
| Surveys used in Step 1 | **RGO, Debrecen and KMAS only** — Spörer *and* Schwabe excluded | PI, 2026-11-21. Spörer has no multi-day group tracking (N1); Schwabe has no spotless days (D23) |
| Pass thresholds | Measures pre-registered; **thresholds advisory on this first pass**, fixed in the results doc once the discrepancy magnitudes are known | PI, 2026-11-21. Recorded as a deviation from the spec's "pre-registered … with pass thresholds" |
| v1.3 releases | `_4` **supersedes `_3` in place**; both workbooks retained byte-exact; **ADR-010** amends ADR-006 | PI, 2026-11-21 |
| KMAS default role | **Reference areas**; modern observer counts come from SWPC alone | KMAS is the only survey bridging the Debrecen 21–23 and RGO 15–19 anchors through cycle 20 (ADR-009) |
| Check A fit | Exact forward filter in JAX, MLE, moment/ACF initialization | Spec open question 3; prototypes the Step 4 engine and settles its unverified cost estimates |
| Check B conditioner | 91-day smoothed **group count** at T = 0, same survey; `amp_whole_sm91` as a sensitivity | Conditioning the area distribution on smoothed *area* conditions on a function of the quantity under test |
| Lifetimes | Visible durations vs a rotation kernel | N2 |
| Data refresh | SILSO SN V2.0 daily + monthly only | Comparison-only (ADR-002/003) |
| Environment | Create `gn-next-gen` from `environment.yml` | First step with production code and JAX; tests the env file before Steps 2–4 depend on it |
| Spörer, Schwabe | Both excluded from Step 1's analyses; whether the exclusion is permanent is a Steps 3–4 decision | N1, D23 |

### `K` — the state-space truncation (D17)

Pre-registered rule, fixed before any fit: **`K` is the smallest integer with `P(GROUPS ≥ K) < 10⁻⁵` over
rows with `STATION > 0`, and `K` greater than the largest reference daily count.** Under `_4` the 99.99th
percentile is 25 and the 99.999th is 36; 9 of 1,099,986 rows reach 39, so the rule's minimum is **39**
(it was 41 under `_3`). The largest reference daily `n_groups` is KMAS **31** (RGO 26, Debrecen 21) — the
"all-time reference maximum is 26" in the current docs is RGO's, and the results doc corrects it.
**Adopted: `K` = 41** (states 0…40, tail mass at `K − 1`) — the smallest value satisfying the rule under both
releases, so a future correction release does not move the state space. `max(GROUPS)` is never used. `f03` is
the figure that justifies it.

### Loaders (`gn_next_gen.io`)

Long format throughout; no dense day × observer matrix (ADR-001; report "A 41-state forward filter…").

- `load_observations(release="v1.3_4", ...)` → `date_ord, station, groups`: drop `STATION == 0` /
  `GROUPS == −1` sentinel rows, apply the D16 rule, optionally concatenate the extension via
  `groups_v13_convention` (stations 1006 KMAS / 1008 SWPC), and **refuse KMAS counts when
  `kmas_role == "reference"`**. `release` selects `v1.3_4` (default), `v1.3_3` or `v1.22`.
- `load_observers()` → the `_4` observers table; missing `INITIAL`/`FINAL` as nulls (D18 now numeric),
  `changed_in_v13` from the explicit column (D20).
- `load_reference_groups(surveys=...)` → `group_day_observations`, Spörer and Schwabe excluded by default (PI, 2026-11-21),
  the 289 D22-colliding KMAS ids excluded from anything keyed on `group_id`; the **row** is the unit.
- `load_day_status()`, `daily_counts(survey, threshold)` — counts are rows with `area_corr` **strictly
  greater** than `T` (legacy convention; `reuse_map.md` row 1 — port the idea, not the shape, of
  `legacy/ADFSSN/scripts/SSN_ADF.py` l. 837–841).
- `load_silso(...)` — tagged comparison-only; the firewall validator refuses it as a calibration input.
- Caching: `data/interim/<name>__<hash>.parquet`, the hash covering the raw files' `MANIFEST.csv` SHA-256s
  and the loader options, so a stale cache cannot be read by accident. `--force` regenerates.

### Run config, firewall and the KMAS single-use rule (`gn_next_gen.config`)

`RunConfig` from `configs/*.yaml`: `variant` (`sunspot_only` | `proxy_informed`), `datasets`, `release`,
`kmas_role`, `duplicate_rule`, `thresholds`, `seed`. `validate(cfg)` raises on: KMAS as both reference areas
and observer counts (ADR-009); any proxy/isotope/aurora or SILSO dataset as a calibration input under
`sunspot_only` (ADR-002); an unknown dataset path or one whose checksum does not match `MANIFEST.csv`. Every
run writes `lineage.json` naming each raw input, its SHA-256 and its role — the per-run lineage ADR-002 wants.

### Check A — dynamics

Per (survey ∈ {rgo, debrecen, kmas}, threshold T ∈ {0, 10, 30, 60, 100}, classes ∈ {1, 2}):

1. **Latent chain** (`models/dynamics.py`): survivors ~ Bin(N_{t−1}, φ), births ~ Pois(λ_t), transition
   factorized `A_t = S · B_t` — two matrix-vector products per day, never a T×K×K tensor. Two classes = a
   Kronecker-factorized pair of chains with classes fixed by an **observable** area split (default 100 MSH,
   near the surveys' median), so class membership is not latent and the fit stays well posed.
2. **Fit** (`inference/forward_filter.py`): exact forward filter in `lax.scan` with `jax_enable_x64` for
   log-likelihood accumulation; `log λ(t)` on a 27-day knot grid with a second-difference penalty (strength
   chosen on held-out days); L-BFGS from moment/ACF initialization (φ̂ from the lag-1 ACF drop, λ̂ from the
   mean). Unobserved days (`day_status == 0`) get a flat emission, so the real calendar is respected.
3. **Predict**: simulate the fitted chain on the real calendar; separately simulate a marked rotation model
   (`models/visibility.py`: emergence at uniform Carrington longitude, ~13.5-day transit, intrinsic lifetime
   matched to the fitted φ) for visible-duration predictions (N2).
4. **Score** on the pre-registered measures against advisory targets. Lifetimes use **RGO and Debrecen**
   (D22-clean) and KMAS with the 289 colliding ids dropped. **Schwabe and Spörer are excluded from Step 1
   entirely** (PI, 2026-11-21): Spörer has no multi-day groups, Schwabe no spotless days (D23).

ACF is computed raw **and** on activity-detrended residuals (count minus its 91-day smooth), because the raw
long-lag ACF rides on the solar cycle: the lag-13 trough and lag-27 bump must survive detrending to count as
the rotation signature (`f07`).

### Check B — stationarity

Epochs: RGO-early (1874–1911, D5-excluded from the anchor), RGO-anchor (cycles 15–19), RGO-late (1970–1976),
Debrecen (21–23), KMAS (20–24). Schwabe and Spörer are excluded (PI, 2026-11-21), so Check B spans
1874–2026 and the pre-1874 area distribution stays an open question for Step 4. Within each epoch, bin
group-days by quintiles of the 91-day smoothed group count (T = 0, same survey) and compare
`log10 area_corr` across epochs within each bin:

- Two-sample Anderson–Darling and KS with a **27-day block bootstrap** null, because groups within a day are
  dependent and a naive i.i.d. test rejects on everything.
- Effect sizes the detection layer actually consumes: the 10th/50th/90th log-area percentiles and the
  fraction above each threshold `T` — the `q_o = E_F[π_o(a)]` of the report's "Mark-dependent thinning…".
- Two-population structure: 1- vs 2-component lognormal mixtures per epoch × bin by EM, compared by BIC;
  report weights, component means and their drift (`f11`).
- Primary test restricted to `area_corr ≥ 30 MSH` (upstream's recommended floor, D14), unrestricted version
  alongside, so completeness floors are not mistaken for solar change.

`area_corr` only; `area_corr_dbcal*` appears as a cross-check panel, never as a pre-correction (ADR-003, D5).
`cycle`/`amp_*` stay diagnostic; cycle phase stays latent (ADR-003).

## Files

New under `src/gn_next_gen/`:

| Path | What |
|---|---|
| `io/observations.py`, `io/observers.py` | release-aware v1.3_4 / _3 / v1.22 + extension loaders, D16 rule, sentinels, KMAS role gate, D18/D20 |
| `io/reference.py` | group-day / single-entry / day-status loaders, D22 and Spörer exclusions |
| `io/silso.py` | comparison-only SN loaders, tagged non-calibration |
| `io/cache.py` | checksum-keyed interim parquet cache |
| `config/run_config.py`, `config/validate.py`, `config/lineage.py` | run config, KMAS single-use + firewall validators, `lineage.json` |
| `reference/counts.py` | daily counts above threshold, run lengths, ADF-style fingerprints |
| `reference/lifetimes.py` | visible durations, gap handling, D22 filter |
| `reference/areas.py` | area distributions, activity binning, mixture fits |
| `models/dynamics.py` | immigration–death chain, `S·B_t`, 1- and 2-class |
| `models/visibility.py` | rotation kernel + marked simulator |
| `inference/forward_filter.py` | exact filter in `lax.scan`, float64 accumulation, MLE driver |
| `stats/discrepancy.py` | ACF, zero fraction per bin, count-PMF distance, run lengths, KS/AD, block bootstrap |
| `viz/style.py`, `viz/data.py`, `viz/checks.py` | house style; the coverage/diff/tail/day-status panels; the Check A/B panels |

Modified: `scripts/convert_gn_database_v13.py` (release-aware; `_3` stays byte-reproducible),
`src/gn_next_gen/paths.py` (new folder constants), `tests/test_gn_database_v13*.py` (parametrized by release).

New scripts and configs: `scripts/land_gn_database_v134.py` (convert + MANIFEST rows + README),
`scripts/fetch_silso.py`, `scripts/build_interim_cache.py`, `scripts/check_a_dynamics.py`,
`scripts/check_b_stationarity.py`, `scripts/benchmark_forward_filter.py`, `scripts/make_step1_figures.py`;
`configs/step1_check_a.yaml`, `configs/step1_check_b.yaml` (with seeds).

Docs: `docs/results/step-1-data-foundation.md` + `docs/results/figures/step-1/`, D23 and revised D16–D20
statuses in `docs/04_data/known_issues.md`, the new folder in `docs/04_data/data_dictionary.md`,
`docs/06_decisions/ADR-010-in-place-release-supersession.md`, `refresh_todo.md`, `docs/05_roadmap/README.md` status row and
paper-item checkboxes.

`data/raw/` is touched only by **additions**: the restored `_3` workbook, the new `gn_database_v1.3_4/`
folder and the new SILSO folder (ADR-006). `legacy/` is untouched.

## Tests

Fast suite (whole suite stays under ~60 s; today it is 5.1 s / 84 tests):

- Loader row counts and coverage against `MANIFEST.csv`, per release: `_4` 1,156,368 rows / 1,099,986 with
  `STATION > 0` / 740 stations / 776 observers; `_3` unchanged at 1,156,673 / 1,100,144 / 740 / 775.
- Sentinel rows absent after load; `ORDINAL` ↔ (Y, M, D) consistent on every row; 708 Feb-29 rows across 97
  years and **no** Feb 29 in a non-leap year (measured on `_4`).
- D16: 3 duplicate keys and 1 conflicting pinned for `_4`, 217/95 for `_3`; each `duplicate_rule` yields its
  documented row count.
- D17: `max(GROUPS)` is 58 for `_4` and 93 for `_3`; the chosen `K` satisfies the rule under both.
- D18 nulls; D20 explicit column vs 99 filled cells (a disagreement fails loudly); D22 exclusion counts
  (839 / 414 / 289); Spörer absent by default and present when asked.
- KMAS single-use validator raises in both directions; firewall validator rejects SILSO or a proxy dataset as
  a calibration input under `sunspot_only`; `lineage.json` lists every input with a matching checksum.
- Forward filter vs a brute-force dense-transition reference at K = 8, T = 50 (< 1e-10); stationary
  distribution matches Pois(λ/(1−φ)) for constant λ.
- Visibility simulator: mean visible duration matches its analytic expectation within Monte-Carlo error.
- Count/run-length/ACF statistics on toy series with known answers.
- Every figure function returns a figure for a minimal input without touching global state; each check script
  runs under `--quick` and writes its figures and `scores.csv`.

`@pytest.mark.slow`: synthetic recovery of the Check A fitter (T = 20,000 days, known φ = 0.88, sinusoidal
λ(t); φ̂ within ±0.02 and λ-knot RMSE within 5%); the `_3` and `_4` converter `--check` reproductions; the
full-length filter benchmark; end-to-end Check A and Check B on one survey.

## Acceptance criteria

| Spec criterion | Measurement | Pass condition | Figure |
|---|---|---|---|
| Loader row counts and coverage match `MANIFEST.csv` | Per-release row counts and coverage dates against the manifest | Exact match for all raw inputs, both releases | `f01`, `f02` |
| Sentinels handled | `STATION == 0` / `GROUPS == −1` absent after load; missing observer years null | No sentinel survives; 740 of 776 stations carry data | `f01` |
| Leap-day handling verified | Feb-29 rows by year vs the Gregorian rule; `ORDINAL` ↔ date | 708 Feb-29 rows, 97 years, zero non-leap violations in 1,156,368 rows | `f04` |
| Fast suite < ~1 min | `pytest` wall time | < 60 s | — |
| Every issue D1–D22 has a status | `docs/04_data/known_issues.md` | All of D1–D23 resolved / documented / deferred-with-reason; D16–D20 restated against `_4`; D16, D17, D22, D23 and the KMAS rule each name the code that implements them | `f02`, `f03`, `f05` |
| Check A: pre-registered discrepancy measures, **advisory** thresholds (PI) | (1) ACF RMSE over lags 1–30 on detrended residuals; (2) max and mean \|Δ zero-fraction\| over 5 activity bins; (3) count-PMF total variation and \|Δ dispersion index\| per bin; (4) spotless/active run-length mean error and KS; (5) visible-duration KS and the ≥ 13-day fraction | (1) RMSE ≤ 0.05, \|Δρ(1)\| ≤ 0.03, lag-13 trough and lag-27 bump each within 0.05; (2) max ≤ 0.05, mean ≤ 0.02; (3) TV ≤ 0.05, \|ΔD\| ≤ 0.15; (4) means within 15%, KS ≤ 0.05; (5) KS ≤ 0.05, ≥ 13-day fraction within 20% relative. All five are **reported against these targets**; the binding go/no-go rule and final thresholds are set in the results doc from the measured magnitudes | `f06`–`f09` |
| 1 vs 2 classes justified by them | Aggregate normalized discrepancy score | Adopt 2 classes only if the score improves by ≥ 25%, or if 1 class fails ≥ 2 of the 5 measures; otherwise 1 class (2 classes cost ~900 states) | `f09` |
| Check B: a stated test and result | AD/KS with 27-day block bootstrap per epoch pair per activity bin, plus percentile and above-threshold effect sizes | Stationary if, for `area_corr ≥ 30 MSH`, the 10th/50th/90th log-area percentiles agree across epochs within 0.1 dex and above-threshold fractions within 10% relative | `f10`, `f11` |
| If non-stationary, consequence written down | Section in the results doc | States whether `q_o` needs an epoch effect or an epoch-local threshold calibration, with the measured size | `f10` |
| **New** v1.3_4 landed under ADR-010 | `_3` workbook restored; `_4` workbook beside it; derived parquet regenerated from `_4`; `MANIFEST.csv` rows updated; ADR-010 written | `tests/test_data_manifest.py` green; `--check` matches for `_3` (into `data/interim/`) and `_4`; `git diff main -- data/raw legacy` touches only `gn_database_v1.3/` and the new SILSO folder | `f02` |
| **New** `K` from a robust quantile (D17) | `P(GROUPS ≥ K) < 10⁻⁵` and `K >` max reference daily count | `K` = 41 recorded with the rule's minimum under both releases (39 for `_4`, 41 for `_3`) and the reference maximum 31; `max(GROUPS)` never used | `f03` |
| **New** KMAS single-use validator | Test | Raises on both role conflicts; default role is reference areas | — |
| **New** SILSO refresh | New dated folder + `MANIFEST.csv` rows + checksum test | Daily and monthly SN V2.0 at the current release; `refresh_todo.md` records anything unavailable | `f01` |
| **New** Step 4 early task 1 | `benchmark_forward_filter.py` at T ≈ 152,000, K = 41 and 900 states, CPU and A100 | Wall time and peak memory recorded, replacing the report's unverified estimates | `f12` |
| **New** environment | `conda env create -f environment.yml` | `gn-next-gen` created and the full suite passes in it, or the failure and fallback are recorded as a deviation | — |
| **New** figure-first verification | The 12 figures exist, are regenerated by `make_step1_figures.py`, and the starred ones are committed | Every criterion above with a figure column has its figure, each ≤ ~300 KB when committed | all |

## Compute and data

Check A: T ≈ 37,620 days (RGO) × K = 41 is trivial — one likelihood evaluation is ~2K²T ≈ 1.3×10⁸ flops.
With ~1,400 λ knots and L-BFGS, a 1-class fit is seconds; a 2-class (≈ 900 states) fit is minutes. The grid is
3 surveys × 5 thresholds × 2 class counts = 30 fits: **budget < 2 h on one A100**, and it runs on CPU too.
Check B is pandas/NumPy over ~339,000 group-days plus block bootstraps: minutes. Converting `_4` is one
openpyxl pass over 1.16M rows (~10 min, the dominant single cost in the data work). Interim caches
≈ 50–100 MB in git-ignored `data/interim/`; figures in git-ignored `outputs/step1/figures/` with ~9 small
ones committed. New raw data: `gn_database_v1.3_4/` (~28 MB workbook + ~4 MB parquet) and one SILSO folder
(~2 MB), taking `data/raw/` from 140 MB to roughly 175 MB — still inside ADR-006's revisit triggers.

## Risks

| Risk | How it shows | Fallback |
|---|---|---|
| `_4` differs from `_3` in a way beyond the audited columns | Converter `--check` passes but downstream counts shift unexpectedly | `f02` makes every changed observer-day visible before any fit consumes it; both releases stay loadable and the lineage names which one ran |
| The `_3` workbook is not restored and the manifest test stays red | `pytest tests/test_data_manifest.py` fails | First commit of the implementation restores it; nothing else proceeds until that test is green |
| `gn-next-gen` env fails on CUDA wheel resolution | `conda env create` error | Run in `ssn` (jax 0.10 + CUDA on 4× A100, as the renovation did) and record the deviation |
| λ(t) with dense knots overfits daily noise, flattering Check A | Held-out day log-likelihood worsens as the penalty falls | Penalty chosen on held-out days; `f09` reports the score at three penalty strengths |
| 2-class fit ill-posed | L-BFGS stalls or classes swap between restarts | Classes fixed by an observable area split, not latent; if still unstable, report 1-class only and say so |
| Check B rejects stationarity everywhere because of completeness floors (D14) | Every epoch pair rejects, RGO-anchor vs RGO-late included | Primary test restricted to ≥ 30 MSH; effect sizes, not p-values, drive the conclusion |
| Check A fails and the Step 4 engine needs rethinking | ≥ 2 of 5 measures fail for both anchors at every threshold | That *is* the step's deliverable: the results doc says so and names the next option (age-structured latent state, or the pairwise bivariate-Poisson composite likelihood of research note Q2(e)1) |
| `obs_duration` semantics misread | Visible-duration comparison passes for the wrong reason | A test pins `obs_duration == span` and counts the 2,597 RGO gap groups separately |
| Figures become the only record and numbers drift from them | Results doc disagrees with a figure | Every figure is generated from the same `scores.csv` the doc and tests read |

## Verification (how `/verify-step 1` will check this)

1. `conda activate gn-next-gen && pytest` — fast suite green, < 60 s; `pytest -m slow` green.
2. `ruff check . && ruff format --check .` — clean.
3. `pytest tests/test_data_manifest.py` — every row including `_3`, `gn_database_v1.3_4/` and the new SILSO folder.
4. `python scripts/convert_gn_database_v13.py --release v1.3_3 --check` and `--release v1.3_4 --check` — both MATCH.
5. `python scripts/build_interim_cache.py` then `--check` — caches regenerate identically.
6. `python scripts/check_a_dynamics.py --config configs/step1_check_a.yaml` and
   `check_b_stationarity.py` — write `scores.csv` and their figures; every number in the results doc traces
   to those files.
7. `python scripts/benchmark_forward_filter.py` → `f12`.
8. `python scripts/make_step1_figures.py` — regenerates all 12 figures from the committed configs and seeds.
9. `git diff main -- data/raw legacy` — additions only (`_3` restored, the two new folders).
10. Review `docs/results/step-1-data-foundation.md`: go/no-go, `K`, class decision, lifetime parametrization,
    D1–D23 statuses, lineage statement, and the embedded figures.

## Decisions resolved before approval (2026-11-21)

1. **Surveys for Step 1 — RGO, Debrecen and KMAS only.** Spörer *and* Schwabe are excluded from this step's
   analyses. Spörer has no multi-day group tracking at all (N1) and Schwabe carries no spotless days (D23).
   No ADR is needed: ADR-009 fixes what the reference *snapshot* contains, and this is a scope choice for one
   step's analyses, recorded here, in D23 and in the results doc. If the exclusion should become permanent for
   Steps 3–4, that is an ADR then, on evidence from this step.
2. **Pass thresholds are advisory on this first pass.** The five discrepancy *measures* are fixed before any
   fit runs; their *thresholds* are reported as targets and the binding go/no-go rule is set in the results
   doc once the magnitudes are known. This is a deliberate, recorded deviation from the spec's
   "pre-registered discrepancy measures … with pass thresholds", and the results doc states it plainly so no
   reader mistakes the thresholds for pre-registered ones.
3. **v1.3_4 supersedes v1.3_3 in place**, under **ADR-010** amending ADR-006. Both workbooks are retained
   byte-exact; the derived parquet tracks `_4`; earlier releases regenerate on demand into `data/interim/`.
4. **`duplicate_rule` defaults to `max`**, so the one surviving conflict (Takuma 1982-10-13) is kept at 8.
5. **`K` = 41**, as the plan derived it: the rule's minimum is 39 under `_4` and 41 under `_3`, so 41 is the
   value stable across both releases.

## Approval

**Approved 2026-11-21** (PI, in session), together with the five decisions above.

## Deviations

Recorded as they arise, per `CLAUDE.md`'s workflow.

1. **`data/raw/gn_database_v1.3/` is modified in place** (PI decision, 2026-11-21), which ADR-006 forbade.
   Authorized by **ADR-010**, written in the first implementation commit: both workbooks are retained
   byte-exact, the derived parquet tracks `_4`, and `--release v1.3_3` regenerates `_3`'s tables into
   `data/interim/`. Three `MANIFEST.csv` rows change value rather than being appended — the one
   append-only property this step gives up, and the reason ADR-010 states its three conditions.
2. **Check A's pass thresholds are advisory, not pre-registered** (PI decision, 2026-11-21). The spec asks for
   "pre-registered discrepancy measures … with pass thresholds"; the five *measures* are fixed before any fit
   runs, but the thresholds are reported as targets and the binding go/no-go rule is set in the results doc
   from the measured magnitudes. The results doc says so explicitly.
3. **Schwabe is excluded from Step 1 as well as Spörer** (PI decision, 2026-11-21), so Check A and Check B
   cover 1874–2026 and no pre-1874 reference statistics are produced in this step. D23 carries the reason.
4. **Six planned files were never created, and the work was re-homed** — found at verification
   (2026-11-22, finding F5), not recorded at the time. `io/observers.py` → `io/observations.py::load_observers`;
   `reference/counts.py` → `io/reference.py::daily_counts`; `reference/lifetimes.py` → `load_group_single_entry`'s
   `obs_duration` plus the check script's own helper; `reference/areas.py` → `reference/area_law.py` and
   `area_fit.py` under Revision 1; `scripts/land_gn_database_v134.py` → `convert_gn_database_v13.py --release`
   under ADR-010. **`scripts/build_interim_cache.py` was never written at all**: caching is lazy inside the
   loaders, keyed by manifest checksums, with three tests. The Files table above is therefore the plan's
   intent, not the delivered layout, and the plan's verification step 5 names a script that does not exist.
5. **`class_split_msh` survived Revision 1** (finding F6). Revision 1 specified replacing it with the computed
   crossover and its outcome section reports it as found, but the key stayed in `configs/step1_check_a.yaml`
   at its arbitrary 100.0 until verification removed it. Nothing read it.


---

# Revision 1 (2026-11-21): the area law is Weibull + log-normal

Approved by the PI on 2026-11-21, after Step 1 had shipped. The approved plan above and
its Deviations section stay as the record of what was built first; this revision
corrects Check B and is recorded in [ADR-011](../06_decisions/ADR-011-group-area-law.md).
Results in [`step-1-data-foundation.md`](../results/step-1-data-foundation.md) §2, whose
superseded two-log-normal section carries a retraction in place.

## Context

Step 1 shipped, but **Check B's two-population analysis is wrong**. The PI has supplied two
authoritative corrections.

1. **Fitting two log-normals folds observational effects into the small-object component.** The
   correct form is **Weibull (small objects, fragmentation regime) + log-normal (large objects,
   bipolar active regions)**, as implemented and calibrated in
   `/d0/amunozj/git_repos/sunspot-homogenization`. That calibration is mature; **nothing is refit**.
   Upstream's own paper says this outright (`paper/ms.tex` l. 199, 207) and names the two-log-normal
   form as the alternative it rejected.
2. **The log-normal component is strongly determined by activity level; the Weibull component is
   not** (Muñoz-Jaramillo et al. 2015, ApJ 804:68). Its §7 fixes `(k, λ)` across activity bins and
   finds `μ`, `σ` and the mixing weight follow power laws in activity level, while the Weibull
   parameters show no correlation — the "magnetic baseline".

What Step 1 shipped instead: `scripts/check_b_stationarity.py::lognormal_mixture`, an EM fit of
1- against 2-component **log-normal** mixtures on **group-days**, conditioned on smoothed **counts**,
reporting ΔBIC +965…+4,268 for "two populations". The functional form, the grain and the
conditioning variable are each wrong against the authoritative sources. That result is **retracted**.

This is not only a correction. It hands Step 4 what the original plan could not: an
**activity-dependent area law**, so the detection layer's `q_o(t) = E_{F(·|AL(t))}[π_o(a)]` varies
with the cycle in a published, calibrated way — the standing explanation for why ADF-style k-factors
depend on activity (research report, "Mark-dependent thinning…").

### What already checks out

Upstream's fit **reproduces exactly** from data already frozen here, which fixes the grain, the area
column and the threshold beyond doubt:

| survey | `threshold_MSH` (frozen) | upstream `n_kept` | ours: `area_corr >` threshold on **single-entry** |
|---|---|---|---|
| rgo | 28.084 | 16,880 | **16,880** |
| debrecen | 20.424 | 8,306 | **8,306** |
| kmas | 23.415 | 15,662 | **15,662** |

So: **one area per group** (`group_single_entry.parquet`, *not* group-days), column `area_corr`,
**strictly greater** than the per-survey threshold already in
`data/raw/reference_groups_2026-10/survey_calibration.csv` — which is upstream's
`lifetime_divergence` threshold under the 10× rule of the 2015 paper §2.4.

Mature per-survey posterior medians, from `data/processed/calibration/solo_wln/*.pkl`:

| survey | `k_W` | `λ_W` (MSH) | `μ_LN` | `σ_LN` | `π_W` | `σ_floor` |
|---|---|---|---|---|---|---|
| rgo | 0.451 | 5.68 | 5.470 | 0.894 | 0.791 | 0.171 |
| debrecen | 0.445 | 7.30 | 5.474 | 0.950 | 0.725 | 0.157 |
| kmas | 0.439 | 14.46 | 5.712 | 0.847 | 0.772 | 0.167 |

`k_W ≈ 0.44–0.45` against the 2015 paper's **0.46** — independent agreement on the baseline
component, which is what licenses holding it fixed.

## Decisions taken by the PI (2026-11-21)

| Decision | Choice |
|---|---|
| Activity level | **Whole-Sun** 182-day Gaussian on `day_status.area_total`, carrying no positional information so it may feed Step 4 (ADR-004). The paper-exact **hemispheric** version is computed from `area_north`/`area_south` as a **diagnostic only** |
| Activity dependence | **Hold `(k, λ)` fixed** at upstream's per-survey values; refit only `μ(AL)`, `σ(AL)`, `c(AL)` — the paper's §7 procedure. Published coefficients are a cross-check, not an input |
| Grain | **Both.** The per-group law (upstream's and the paper's object) *and* the per-group-day law Step 4 integrates over, with the relationship between them reported |
| Knock-ons | Re-ground Check A's two classes on the Weibull/log-normal split; flag that T = 0 and T = 10 sit below the surveys' own completeness thresholds; write an ADR |

## Design

### 1. Freeze upstream's area-law calibration into `data/raw` (ADR-006)

The posteriors live in a git-ignored tree in another repository, so nothing may read them at run
time. Land a small snapshot in a **new folder, overwriting nothing**:

`data/raw/area_law_2026-11/`
- `solo_wln_posterior.csv` — one row per survey: `k_W, log_lambda_W, mu_LN, sigma_LN, pi_W,
  sigma_floor` as median plus 16/84 percentiles, with `y_trunc`, `n_kept`, `n_below`, `log_Z`,
  `area_column`, `threshold_source`, `trunc_factor` and the upstream commit.
- `README.md` — the model, the **`π_W` ↔ `c` mapping** (upstream weights the *Weibull*, the paper
  weights the *log-normal*, so `c = 1 − π_W`), the truncation recipe, the provenance.
- `provenance.json` and `MANIFEST.csv` rows, following the renovation's pattern.

Exported by `scripts/export_area_law.py`: deterministic, `--check` verifies byte reproducibility,
and it refuses to run if the upstream commit does not match the one recorded.

### 2. `gn_next_gen.reference.area_law` — the law itself

Ported from `sunspot_pipeline/processing/calibration_models.py` (`_log_pdf_latent_wln`,
`_emit_hard_truncated_quadrature`) and citing those paths, as the hard rules require.

- `log_pdf_latent(z, k, log_lambda, mu, sigma, pi_w)` — the mixture on `z = log A`. The Weibull's
  density in log-area is `log k − k·log λ + k·z − exp(k·(z − log λ))`, a left-skewed Gumbel, mixed
  with `N(z; μ, σ)`. **Not** two Gaussians in `z`.
- `log_pdf_truncated(..., y_trunc, sigma_row, sigma_floor)` — per-row truncated likelihood by
  quadrature on a fixed `z` grid (upstream's `-10, 10, 256`), carrying per-row measurement error from
  `area_corr_unc` plus upstream's `sigma_floor`.
- `load_upstream_parameters(survey)` — reads the frozen snapshot.
- `fraction_above(threshold, params)` — the `q`-style functional the detection layer wants.
- `activity_level(day_status, sigma_days=182, scope="whole")` — 182-day Gaussian on `area_total`,
  or `area_north`/`area_south` for the diagnostic. **Units: `area_total` is μHem and the paper's AL
  is mHem, so `AL = smoothed / 1000`** — the power-law coefficients are meaningless in the wrong
  unit, so this is asserted in a test, not left to a comment.

### 3. Check B, rewritten (`scripts/check_b_stationarity.py`)

Three tests replace the percentile/KS/ΔBIC battery. Data: `group_single_entry`, D22-clean,
`area_corr > threshold_MSH`, Step 1's three surveys.

1. **Is the Weibull component activity-independent?** All five parameters free, per AL bin, per
   survey; report Spearman ρ of `k_W` and `λ_W` against AL with confidence levels, as the paper
   does. *This test licenses step 2.* If `(k, λ)` move with AL, holding them fixed is unjustified
   and the run says so instead of proceeding.
2. **Refit the log-normal activity dependence.** Hold `(k, λ)` at upstream's per-survey values; fit
   `μ_j, σ_j, c_j` per AL bin by truncated MLE; fit power laws `μ = a·AL^b` and so on. Report
   coefficients, ρ and reduced χ² **against the published** `μ = 5.97·AL^0.021`,
   `σ = 1.08·AL^0.117`, `c = 0.66·AL^0.230`.
3. **Stationarity, in the right parameterization.** At **matched AL**, compare `(μ, σ, c)` across
   epochs and surveys. This is what Step 1 owes Step 4 — is `F(·|AL)` the same law in 1874 as in
   2014? — now asked about three parameters rather than about percentiles.

Then the **grain bridge**: the per-group-day law, fitted the same way on `group_day_observations`,
with the mapping between the two reported as the size-biasing that lifetime × visibility imposes
(large groups live longer, so the visible-on-a-day law is biased large). Check A already showed the
model must carry that weighting; this measures it.

Model comparison by AIC across four nested forms — no AL dependence / free per bin / `(k,λ)` fixed
per bin / analytic power-law dependence — mirroring the paper's Table 2, which is also the clean way
to retire the two-log-normal form on the record rather than by assertion.

### 4. Check A's two classes, re-grounded

`configs/step1_check_a.yaml`'s `class_split_msh: 100.0` was arbitrary. Replace it with the
**Weibull/log-normal crossover** computed from the fitted law per survey — the area where the two
components' densities cross. Re-run only the two 2-class cells (RGO and Debrecen at T = 30); the
other 15 are untouched. The verdict is expected to stand — rotation is what is missing, not lifetime
heterogeneity — but the 1-vs-2 class decision then means something physical.

### 5. The threshold finding

The frozen truncation thresholds are **20.4–28.1 MSH**, so the sweep's `T = 0` and `T = 10` sample
the regime upstream deliberately truncates away, where the Weibull component and observational
incompleteness dominate. Record in the results doc and in **D14**: counts below a survey's own
threshold are not comparable across surveys, and Steps 2–4 should treat `T ≥ 30` as the trustworthy
band. This also explains why Check A's `T = 0` fits sit apart from the rest.

### 6. ADR-011

*The group-area law is Weibull + log-normal, calibrated upstream and not refitted; its log-normal
component depends on activity level.* Records the functional form and why two log-normals are
rejected; that upstream's posterior is adopted as-is; that `(k, λ)` are held fixed and only the
log-normal's activity dependence is fitted here; that activity level is whole-Sun by ADR-004 with
hemispheric as diagnostic; and the consequence for Step 4 — emission tables become
activity-dependent.

## Files

| Path | Change |
|---|---|
| `data/raw/area_law_2026-11/` | **new** frozen snapshot + `MANIFEST.csv` rows |
| `scripts/export_area_law.py` | **new** exporter; `--check` reproducible, upstream commit pinned |
| `src/gn_next_gen/reference/area_law.py` | **new** — the law, the truncated likelihood, AL, the fitters |
| `src/gn_next_gen/paths.py` | constants for the new folder |
| `scripts/check_b_stationarity.py` | rewritten around the three tests; `lognormal_mixture` **deleted** |
| `src/gn_next_gen/viz/checks.py` | `f10` becomes the law and its AL dependence; `f11` becomes parameter-vs-AL panels with the published power laws overlaid |
| `configs/step1_check_b.yaml` | AL definition, bins, fixed-`(k,λ)` switch, published coefficients as cross-check |
| `configs/step1_check_a.yaml` | `class_split_msh` → crossover, computed not hard-coded |
| `docs/06_decisions/ADR-011-*.md` | **new**, indexed in `docs/06_decisions/README.md` |
| `docs/results/step-1-data-foundation.md` | §2 rewritten; the ΔBIC two-log-normal claim **retracted in place**, not quietly deleted |
| `docs/04_data/known_issues.md` | D14 sharpened with the threshold finding |
| `docs/04_data/data_dictionary.md` | the new raw folder |
| `tests/test_area_law.py` | **new** |

## Tests

- **The reproduction test that anchors everything**: `area_corr > threshold_MSH` on single-entry
  returns exactly 16,880 / 8,306 / 15,662 for RGO / Debrecen / KMAS. If this fails, the grain or the
  column is wrong and nothing downstream is trustworthy.
- `log_pdf_latent` integrates to 1 over the `z` grid; reduces to a pure log-normal as `π_W → 0` and a
  pure Weibull as `π_W → 1`; matches a `scipy` Weibull transformed to log-area.
- Truncated likelihood normalizes over the retained support and agrees with a brute-force integral.
- **Synthetic recovery**: simulate from the law at known `(k, λ, μ, σ, π)`, truncate, refit with
  `(k, λ)` fixed, recover `μ, σ, c` within tolerance (slow).
- Activity level: a 182-day Gaussian on a synthetic sinusoid returns the right amplitude and phase;
  whole-Sun equals the area-weighted combination of the hemispheric series; **a unit test asserts AL
  is in mHem**.
- Parameter snapshot pinned against `MANIFEST.csv`; `--check` reproducibility (slow).
- Regression: `lognormal_mixture` is gone and nothing imports it.

## Acceptance criteria

| Criterion | Measurement | Pass |
|---|---|---|
| Upstream's fit reproduces | kept-counts above `threshold_MSH` | exactly 16,880 / 8,306 / 15,662 |
| Weibull is activity-independent | Spearman ρ of `k_W`, `λ_W` vs AL, per survey | small \|ρ\| at low confidence, as the paper finds; **if not, stop and report** |
| Log-normal dependence recovered | ρ and power-law fits for `μ, σ, c` vs AL | ρ positive and significant for `μ` and `σ`; reduced χ² < 1, as in the paper |
| Published coefficients cross-checked | fitted exponents vs 5.97/0.021, 1.08/0.117, 0.66/0.230 | agreement, or a **stated and quantified** offset with its cause (`π_W` vs `c` convention, calibration scale) |
| Stationarity at matched AL | `(μ, σ, c)` across epochs within AL bins | spread stated with uncertainties; consequence for `q_o` written down |
| Grain bridge | per-group vs per-group-day law | both fitted; difference reported as size-biasing |
| Model comparison | AIC over the four nested forms | the AL-dependent form wins, as in the paper's Table 2 |
| Two classes re-grounded | crossover split; 2 cells re-run | split computed from the law; decision restated |
| Thresholds flagged | D14 + results doc | the 20.4–28.1 MSH band recorded with the T = 0/10 caveat |
| Retraction visible | results doc §2 | the superseded two-log-normal result shown as retracted, with its reason |

## Verification

```bash
conda activate gn-next-gen
pytest && pytest -m slow
python scripts/export_area_law.py --check          # byte-reproducible, commit pinned
pytest tests/test_data_manifest.py                 # new folder checksummed
python scripts/check_b_stationarity.py --config configs/step1_check_b.yaml
python scripts/check_a_dynamics.py --config configs/step1_check_a.yaml \
       --surveys rgo debrecen --thresholds 30      # the two re-grounded 2-class cells
python scripts/make_step1_figures.py
ruff check . && ruff format --check .
git diff --name-status data/raw legacy             # additions only, bar the new folder
```

Then read `docs/results/step-1-data-foundation.md` §2 with figures `f10` and `f11`: the
parameter-vs-AL panels must show a flat Weibull and a rising log-normal, with the published power
laws overlaid.

## Risks

| Risk | How it shows | Fallback |
|---|---|---|
| `π_W` (0.72–0.79) and the paper's `1 − c` (≈ 0.45) disagree | the refit `c(AL)` sits well off the published curve | Expected, not a bug: different truncation and calibration scale. Report both and attribute the offset rather than tuning to match |
| `(k, λ)` turn out to move with AL | test 1 shows significant ρ | Do **not** proceed to the fixed-`(k,λ)` refit. Report that the baseline claim fails on this data and escalate — that would contradict the paper and matters more than the rest of this revision |
| Few groups per AL bin at high activity | wide intervals in the top bins, as in the paper's Fig. 5 | Equal-count rather than equal-width bins; report counts per bin |
| Per-group-day fit dominated by long-lived groups | per-day `μ` far above per-group `μ` | That *is* the size-biasing; report it as the measured bridge, not as a discrepancy |
| Schwabe/Spörer kept-counts do not reproduce (6,813 vs 5,588; 5,189 vs 5,202) | — | Both are excluded from Step 1 (D23), and Schwabe's calibration is umbral so its threshold applies to a different column. Note and move on |

## Open questions

None blocking — the four design decisions were taken by the PI on 2026-11-21 and are recorded above.


## Revision 1 outcome

Implemented as planned, with these results and one reversal:

- The anchor check passed exactly (RGO 16,880 / Debrecen 8,306 / KMAS 15,662), fixing
  the grain, the column and the threshold.
- The Weibull gate passed: no correlation with activity reaches 0.95 confidence and the
  signs disagree across surveys, so holding `(k, λ)` fixed is licensed.
- The log-normal's activity dependence reproduces the published power laws, with KMAS
  matching the published `μ` exponent to three decimals. Amplitudes sit below the
  published ones for the two reasons the plan predicted.
- **The grain bridge came out with the opposite sign to the plan's expectation.** The
  plan anticipated the per-group-day law would be *larger* in `μ` from size-biasing; it
  is smaller, by 0.19–0.26, because the per-group law uses each group's peak area while
  the per-day law sees instantaneous ones. The plan's risk table called the opposite
  outcome "the measured bridge, not a discrepancy" — that framing was wrong about the
  direction, and the measurement stands on its own.
- Model comparison diverges from the paper for RGO and Debrecen, which prefer the
  flexible per-bin fit over the analytic power laws; reported as a difference.
- Two defects found while re-grounding Check A's classes: `class_split_msh` was dead
  config, and the two-class filter's docstring claimed observable classes while the code
  made them latent — which is why the earlier symmetric-saddle bug was possible.
