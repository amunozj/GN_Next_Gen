# Data dictionary (`data/raw/`)

All files are byte-exact copies (see `data/raw/MANIFEST.csv` for sizes, SHA-256, coverage, origin). Never edit them;
Step 1 loaders write cleaned versions to `data/interim/`. Facts below were checked against the files on 2026-10-28.

## `gn_database_v1.22/GNobservations_JV_V1.22.csv`

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

## `gn_database_v1.22/GNobservers_JV_V1.22.csv`

| Column | Meaning |
|---|---|
| `STATION` | Code used in the observations file (743 stations) |
| `INITIAL`, `FINAL` | First and last year of observations |
| `TOT.OBS` | Number of observation days |
| `OBSERVER` | `"SURNAME, INITIALS, PLACE"` (quoted; legacy code takes the text before the first comma as the name) |

## `reference_areas/*.csv` — daily group areas of the reference survey

One row per calendar day (no gaps from 1874-04-17), one column per group observed that day.

| Column | Meaning |
|---|---|
| `YEAR`, `MONTH`, `DAY` | Date |
| `AREA1` … `AREA26` | Areas of the individual groups present that day, integers in millionths of the solar hemisphere (MSH), 1 to ~11,280. Unused columns are NaN (DRGO) or 0 (SC_SP_RG versions). Whether areas are whole-spot and foreshortening-corrected should be confirmed against the Debrecen/RGO documentation in Step 1 |

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

## `silso/SN_d_tot_V2.0.csv` and `silso/SN_m_tot_V2.0.csv`

International Sunspot Number V2.0 from WDC-SILSO, Royal Observatory of Belgium (https://www.sidc.be/SILSO/), converted to CSV.

| File | Columns | Coverage | Missing values |
|---|---|---|---|
| Daily | `YEAR, MONTH, DAY, FRACYEAR, DAILYSN, DAILYSTD, NOBS, MARKER` | 1818-01-01 → 2018-07-31 | `DAILYSN = −1` |
| Monthly | `YEAR, MONTH, FRACYEAR, MMEAN, MSTD, NOBS, MARKER` | 1749-01 → 2017-10 | `MSTD`, `NOBS = −1` |

`MARKER`: 1 = definitive, 0 = provisional (122 provisional days at the end of the daily file). **The new model must not
use SN for calibration** (see ADR-002 and ADR-003); it is kept for comparison and for reproducing legacy baselines.

## Licensing and citation

SILSO data are distributed under CC BY-NC 4.0; cite WDC-SILSO and the relevant papers. Cite Vaquero et al. (2016) for the
group-number database, and the Debrecen and Kislovodsk sources for areas.
