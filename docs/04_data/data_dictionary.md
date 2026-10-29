# Data dictionary (`data/raw/`)

All files are byte-exact copies (see `data/raw/MANIFEST.csv` for sizes, SHA-256, coverage, origin). Never edit them;
Step 1 loaders write cleaned versions to `data/interim/`. Facts below were checked against the files on 2026-10-28.

**Current inputs** (2026-10 renovation, `docs/plans/raw-data-renovation-plan.md`):
`reference_groups_2026-10/` is the reference survey (ADR-009), `gn_database_v1.3/` the observation
database, and `gn_observer_extension_2026-10/` carries the record past 2010.
`reference_areas/` and `gn_database_v1.22/` are **superseded but retained**: Step 2's legacy baselines
must reproduce against the data the legacy code actually used.

## `gn_database_v1.22/GNobservations_JV_V1.22.csv` — SUPERSEDED by `gn_database_v1.3/`

> Retained for the Step 2 legacy baselines and for the v1.22-vs-v1.3 diff. Not the current input.

Revised collection of sunspot group numbers ([Vaquero et al. 2016, Sol. Phys.](https://arxiv.org/pdf/1609.04882), based
on Hoyt & Schatten 1998), as distributed by SILSO and converted to CSV with ORDINAL and FRACYEAR added.

- **15 comment lines** starting with `%` (description and column meanings), then the header row. Read with
  `pd.read_csv(path, header=15)` or `comment`-aware parsing. Line endings CRLF.
- Columns:

| Column | Meaning |
|---|---|
| `YEAR`, `MONTH`, `DAY` | Calendar date (proleptic Gregorian) |
| `ORDINAL` | Python `datetime.date.toordinal()` (0001-01-01 = 1); 1610-01-01 = 587676 |
| `STATION` | Observer/station code (join to `GNobservers`). **0 = no data / to be ignored** |
| `OBSERVER` | 1 if data, else 0 (one observer per station in this version) |
| `GROUPS` | Number of groups reported. **−1 = missing**; appears only with `STATION = 0` |
| `FRACYEAR` | `YEAR + (ORDINAL − ordinal(Jan 1)) / days_in_year` |

- One row per (day, station) observation; days with no observation at all appear as a `STATION = 0`, `GROUPS = −1` row.
  1,047,761 rows have `STATION > 0`; 715 distinct stations have data. Coverage 1610-01-01 to 2010-12-31.
- Maximum reported value is 58 groups in a day (relevant to truncating the latent count state space).
- February 29 was removed by hand for 1700, 1800 and 1900 (not leap years in the Gregorian calendar; verified absent).

## `gn_database_v1.22/GNobservers_JV_V1.22.csv` — SUPERSEDED by `gn_database_v1.3/`

| Column | Meaning |
|---|---|
| `STATION` | Code used in the observations file (743 stations) |
| `INITIAL`, `FINAL` | First and last year of observations |
| `TOT.OBS` | Number of observation days |
| `OBSERVER` | `"SURNAME, INITIALS, PLACE"` (quoted; legacy code takes the text before the first comma as the name) |

## `reference_areas/*.csv` — SUPERSEDED by `reference_groups_2026-10/`

> The old day × `AREA1..AREA26` matrices. Retained byte-exact for the Step 2 legacy baselines and
> because the D3 characterization is about these two files. **Not the reference** — see ADR-009.

One row per calendar day (no gaps from 1874-04-17), one column per group observed that day.

| Column | Meaning |
|---|---|
| `YEAR`, `MONTH`, `DAY` | Date |
| `AREA1` … `AREA26` | Areas of the individual groups present that day, integers in millionths of the solar hemisphere (MSH), 1 to ~11,280. Unused trailing columns are **NaN in all three files** (verified 2026-10-28: 8,139 rows have `AREA1 > 0` with `AREA2 = NaN`, and none have `AREA2 = 0`; the zeros occur only as fully-zero spotless rows). Whether areas are whole-spot and foreshortening-corrected should be confirmed against the Debrecen/RGO documentation in Step 1 |

Encoding of the day (both conventions hold in all three files):
- `AREA1 = NaN` → **no observation** that day (477 days in the SC_SP_RG version, mostly the 1870s).
- `AREA1 = 0` → **observed, spotless** (8,758 all-zero days in the SC_SP_RG version; 5,712 in DRGO).
- Number of groups above an acuity threshold `T` (MSH) = number of `AREA*` values `> T` (legacy convention).
- Maximum 26 groups on one day (1937-07-12).

| File | Coverage | Header | Notes |
|---|---|---|---|
| `SC_SP_RG_DB_KM_group_areas_by_day.csv` | 1825-11-05 → 2018-06-19 | **missing** | RGO (1874–1976) + Debrecen DPD + Kislovodsk (KMAS). First 17,695 rows (to 1874-04-16) are all NaN; the rest is identical to the `_pre5898ab8` file |
| `SC_SP_RG_DB_KM_group_areas_by_day_pre5898ab8.csv` | 1874-04-17 → 2018-06-19 | yes | Same data, recovered from ADFSSN history (`5898ab8~1`) |
| `DRGO_group_areas_by_day.csv` | 1874-04-17 → 1976-12-31 | yes | Earlier RGO-era version used by the first notebooks; differs from SC_SP_RG on 12,469 days (see known issues) |

Sources: Debrecen Heliophysical Observatory sunspot databases (GPR/DPD,
http://fenyi.solarobs.csfk.mta.hu/en/databases/), Kislovodsk Mountain Astronomical Station
(https://doi.org/10.7910/DVN/M7NDXN). The prefix `SC_SP` suggests Schwabe and Spörer were planned, but no pre-1874 areas
are present.

## `gn_database_v1.3/` — group-count observation database (current)

The v1.3 revised collection, **1610-01-01 → 2010-12-31**. Supersedes `gn_database_v1.22/`. The
authoritative artifact is `Revised collection_v1.3_3.xlsx` (28.6 MB, 5 sheets); the three derived
tables beside it are produced deterministically by `scripts/convert_gn_database_v13.py` and frozen
here, following the same precedent as v1.22 (itself a conversion). Full detail in that folder's
`README.md`.

| File | Grain | Rows |
|---|---|---|
| `gn_v13_observations.parquet` | one row per (day, station) observation record | 1,156,673 |
| `gn_v13_observers.parquet` | one row per listed observer | 775 |
| `gn_v13_changelog.csv` | `version, name, year, note` — the four version blocks unstacked | 586 |

Observations: `YEAR, MONTH, DAY, ORDINAL, STATION, OBSER, GROUPS, FRACYEAR, SHEET`. Sentinels are
v1.22's — **`STATION = 0` = no data**, **`GROUPS = −1` = missing** (only ever together). `ORDINAL`
and `FRACYEAR` are **derived** by the converter using v1.22's definitions (1610-01-01 = 587676) so
the schema stays compatible; the workbook has neither, and names the flag column `OBSER` rather than
`OBSERVER` (**D19**). `SHEET` records which of `1610-1946`, `1947-1988`, `1989-2010` a row came from.

Observers: `STATION` (1–775), `INITIAL`, `FINAL` (**null** where the sheet carries the string `'NA'`),
`TOT_OBS`, `OBSERVER`, `changed_in_v13`. 740 of the 775 codes carry data; the other 33 are listed
with no observations (**D18**). `changed_in_v13` is recovered from green cell fill, the only place
that information exists (**D20**), and is true for 94 observers.

What changed from v1.22: 1,100,144 rows with `STATION > 0` (from 1,047,761), 740 stations with data
(from 715), 775 observers listed (from 743), and 109 documented corrections with DOIs where v1.22
shipped none — which resolves **D7** for 1610–2010. Coverage is unchanged: **v1.3 does not extend
past 2010**, and `gn_observer_extension_2026-10/` closes that gap from a different source.

Caveats before writing a loader: **D16** — 217 duplicate `(date, station)` keys, 95 with conflicting
`GROUPS`, so do **not** key on `(date, station)` until a resolution rule is chosen; **D17** — the two
implausible extreme counts, so size state spaces from a robust quantile and not from `max(GROUPS)`.

## `reference_groups_2026-10/` — the reference survey (current)

Five per-survey, per-group, per-day catalogues — Schwabe, Spörer, RGO, Debrecen, KMAS — spanning
**1825-11 → 2026** with no gaps, frozen from `sunspot-homogenization` at a pinned commit. Fixed by
**ADR-009**. Full schema, the two renames and the anchored epochs are in that folder's `README.md`;
this is the short form.

| File | Grain | Rows |
|---|---|---|
| `group_day_observations.parquet` | one row per (survey, group, day) — **the reference** | ~459,000 |
| `group_single_entry.parquet` | one row per (survey, group); carries `obs_duration` (lifetimes) | ~106,000 |
| `day_status.parquet` | one row per (survey, day) | ~109,000 |
| `survey_calibration.csv` | per survey: `map_kind`, `CF_med/lo/hi`, `threshold_MSH`, `n_kept` | 5 |
| `survey_cycle_coverage.csv` | per (survey, cycle): extent and the `complete` flag | ~30 |
| `provenance.json` | commit, posterior checksum, coverage, row counts, lineage | — |

Key points, all of which bite if missed:

- **`area_corr` is the model-facing area** (native, whole-spot, foreshortening-corrected, MSH).
  `area_corr_dbcal*` is the same on the common Debrecen scale and is a **prior/cross-check only**,
  never a hard pre-correction — otherwise the upstream calibration is applied twice (ADR-003, D5).
  `_unc_corr` is correlated across a survey (adds linearly in a daily sum); `_unc_indep` is
  per-object (adds in quadrature).
- **`day_status`**: 0 = no observation, 1 = observed spotless, 2 = observed active. This replaces
  the old `AREA1 = NaN` / `AREA1 = 0` convention explicitly.
- **`group_obs_kind`** (in the single-entry table only): 0 = placeholder row, 1 = single-observation
  group, 2 = multi-observation group. It and `day_status` are both called `obs_type` upstream, where
  they mean different things — see **D10**.
- Groups above acuity threshold `T` on a day = rows for that (survey, day) with area **strictly
  greater** than `T` (the legacy convention). The snapshot is **unthresholded** so Step 1 can sweep
  T ∈ {0, 10, 30, 60, 100} MSH; **D14** bounds what the lowest thresholds mean per survey.
- `cycle`, `cycle_parity`, `amp_*` are **diagnostic only, never calibration inputs** (ADR-003 keeps
  cycle phase latent); they come from the surveys' own positions and areas, not from SILSO.
- `latitude`, `longitude`, `hemisphere`, `dist_center` are **QA metadata only** — ADR-004 bars
  positions as model inputs. Retained so the D12 disk-angle audit stays reproducible.
- **KMAS is dual-use**: it is also an observer-count source below. See the single-use rule.

Area semantics (resolves **D4**): all areas are millionths of the solar hemisphere. `area_corr` is
whole-spot and foreshortening-corrected; `area_umbra_*` are the umbral equivalents; upstream halves
millionths-of-disk at parse time and recomputes foreshortening from the raw projected areas.

## `gn_observer_extension_2026-10/` — daily group counts, KMAS and SWPC (1954–2026)

Carries the observation record past v1.3's 2010-12-31 end. Counts only, no areas. Both surveys over
their full span, with `extends_v13` marking the rows after 2010-12-31; the pre-2010 overlap is what
calibrates them against the existing network. Full schema in that folder's `README.md`.

| Column | Meaning |
|---|---|
| `survey`, `station` | `kmas`/1006 or `swpc`/1008 (upstream's survey tags; cannot collide with v1.3's 1–775) |
| `threshold_msh` | acuity threshold; a group counts when its area is **strictly greater** |
| `groups_native`, `groups_dbcal` | counts from `area_corr` and `area_corr_dbcal`; **null** where not observed |
| `groups_v13_convention` | `groups_native` with v1.3's sentinel (**−1 = no observation**), so it concatenates onto v1.3 |
| `day_status`, `n_groups` | as in the reference snapshot |
| `extends_v13`, `role` | true after 2010-12-31; the single-use rule, per row |

Thresholds are 0, 10, 30, 60, 100 MSH plus each survey's own anchor threshold (KMAS 23.41,
SWPC 20.68). See **D21**: these are survey-derived counts, not visual observer counts.

### ⚠ The KMAS single-use rule

KMAS supplies reference areas in `reference_groups_2026-10/` **and** observer counts here, and its
counts are derivable from its areas.

> **Any run must declare KMAS's role as either reference areas or observer counts, never both.**

SWPC carries no such constraint — its areas are not in the reference. See ADR-009 and
`docs/04_data/known_issues.md`.

## `silso/SN_d_tot_V2.0.csv` and `silso/SN_m_tot_V2.0.csv`

International Sunspot Number V2.0 from WDC-SILSO, Royal Observatory of Belgium (https://www.sidc.be/SILSO/), converted to CSV.

| File | Columns | Coverage | Missing values |
|---|---|---|---|
| Daily | `YEAR, MONTH, DAY, FRACYEAR, DAILYSN, DAILYSTD, NOBS, MARKER` | 1818-01-01 → 2018-07-31 | `DAILYSN = −1` |
| Monthly | `YEAR, MONTH, FRACYEAR, MMEAN, MSTD, NOBS, MARKER` | 1749-01 → 2017-10 | `MSTD`, `NOBS = −1` |

`MARKER`: 1 = definitive, 0 = provisional (122 provisional days at the end of the daily file). **The new model must not
use SN for calibration** (see ADR-002 and ADR-003); it is kept for comparison and for reproducing legacy baselines.

## Licensing and citation

SILSO data are distributed under CC BY-NC 4.0; cite WDC-SILSO and the relevant papers.

For the **observation database**, cite the v1.3 revised collection together with Vaquero et al. (2016) and Hoyt &
Schatten (1998); `gn_database_v1.3/gn_v13_changelog.csv` carries the per-observer attributions with DOIs for the
109 corrections v1.3 applied.

For the **reference survey**, cite the source surveys — Schwabe and Spörer via Arlt et al.; the Royal Greenwich
Observatory and Debrecen Heliophysical Observatory (GPR/DPD), http://fenyi.solarobs.csfk.mta.hu/; Kislovodsk
Mountain Astronomical Station (Pulkovo), https://doi.org/10.7910/DVN/M7NDXN; NOAA/SWPC via NOAA NCEI — and the
`sunspot-homogenization` processing and cross-calibration (MIT).
