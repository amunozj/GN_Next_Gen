# `gn_database_v1.3/` — revised collection of group counts, v1.3 (1610–2010)

Current observation database for this project; supersedes `../gn_database_v1.22/`, which is kept so
Step 2 can reproduce the legacy baselines against the database the legacy code actually used.

## Two releases live here (ADR-010)

| Release | File | Status |
|---|---|---|
| **v1.3_4** | `Revised collection_v1.3_4.xlsx` (28.2 MB, 8 sheets) | **current** — the derived tables are built from it |
| v1.3_3 | `Revised collection_v1.3_3.xlsx` (28.6 MB, 5 sheets) | superseded, retained byte-exact |

`_4` arrived on 2026-11-21 and **fixes most of what this project reported upstream** in
`docs/results/gn-v13-feedback-report.md`. Per [ADR-010](../../../docs/06_decisions/ADR-010-in-place-release-supersession.md)
the derived tables track the newest release in place, every release's workbook is retained with its
own `MANIFEST.csv` row, and an earlier release is regenerated on demand:

```
python scripts/convert_gn_database_v13.py --release v1.3_3   # -> data/interim/gn_database_v1.3_3/
```

The script **refuses** to write a superseded release into `data/raw/`. That is what keeps the
numbers in `docs/results/raw-data-renovation.md` and in the feedback report reproducible even though
`data/raw` no longer carries `_3`'s tables.

## Files

| File | What |
|---|---|
| `Revised collection_v1.3_4.xlsx` | the current upstream artifact, byte-exact |
| `Revised collection_v1.3_3.xlsx` | the superseded artifact, byte-exact |
| `gn_v13_observations.parquet` | the three observation sheets of `_4` concatenated, with `ORDINAL` and `FRACYEAR` added |
| `gn_v13_observers.parquet` | the 776 listed observers, with `changed_in_v13` from `_4`'s explicit column |
| `gn_v13_changelog.csv` | `_4`'s four `Differences-*` sheets, unstacked into one table |

The three derived files are produced by `scripts/convert_gn_database_v13.py`, which is deterministic:
re-running it on the same workbook reproduces them byte-for-byte (`--check` verifies this). They are
frozen here rather than regenerated into `data/interim/` because parsing 1.16 M rows out of the xlsx
takes minutes, and because `gn_database_v1.22/` set the same precedent (its manifest row records it
as "converted to CSV with ORDINAL and FRACYEAR added").

The **output schema is identical for both releases** — including the flag column, which stays
`OBSER` although `_4`'s sheets head it `OBSERVER` — so loaders never branch on release.

## Schema — `gn_v13_observations.parquet`

One row per (day, station) observation record. 1,156,368 rows, **in date order**
(`_3`'s rows were not sorted, which is why this file is ~40% smaller than the one it
replaced: sorted `ORDINAL` and `FRACYEAR` compress far better).

| Column | Meaning |
|---|---|
| `YEAR`, `MONTH`, `DAY` | calendar date (proleptic Gregorian) |
| `ORDINAL` | `datetime.date.toordinal()`; 1610-01-01 = 587676, the same convention as v1.22 (**derived**) |
| `STATION` | observer/station code, join to the observers table. **0 = no data / ignore** |
| `OBSER` | 1 if data, else 0. Named `OBSERVER` in v1.22 and again in `_4`'s sheets; the column keeps this name across releases (D19) |
| `GROUPS` | number of groups reported. **−1 = missing**, only ever with `STATION = 0` |
| `FRACYEAR` | `YEAR + (ORDINAL − ordinal(Jan 1)) / days_in_year` (**derived**) |
| `SHEET` | source sheet, one of `1610-1946`, `1947-1988`, `1989-2010` |

`ORDINAL`, `FRACYEAR` and `SHEET` do not exist in the workbook; the first two reproduce v1.22's
definitions so downstream code is schema-compatible, and `SHEET` preserves which sheet a row came
from.

## Schema — `gn_v13_observers.parquet`

| Column | Meaning |
|---|---|
| `STATION` | 1–776 |
| `INITIAL`, `FINAL` | first/last year of observations, **null** for placeholders — blank in `_4`, the string `'NA'` in `_3` (D18) |
| `TOT_OBS` | number of observation days (0 for the 34 placeholder observers) |
| `OBSERVER` | `"SURNAME, INITIALS, PLACE"` |
| `changed_in_v13` | true for the 100 observers `_4`'s explicit column flags; the green fill `_3` relied on marks 99 and disagrees only on station 36 (D20) |

## Schema — `gn_v13_changelog.csv`

`version, name, year, note` — 597 rows: 403 for v1.12, 72 for v1.2, 2 for v1.21, **120 for v1.3**
(`_3` had 109). The
v1.3 block carries the Maunder-Minimum and early-record corrections with DOIs (Hayakawa et al.
2021/2023/2024, Carrasco et al. 2019/2020/2021/2022/2025, Neuhäuser & Neuhäuser 2018, Vokhmyanin
et al. 2020/2021, and others), which is what resolves known issue D7 for 1610–2010.

## What changed from v1.22

| | v1.22 | v1.3 |
|---|---|---|
| Rows with `STATION > 0` | 1,047,761 | **1,100,144** |
| Distinct stations with data | 715 | **740** |
| Observers listed | 743 | **775** |
| Max `GROUPS` | 58 | **93** in `_3` (implausible, see D17); **58** again in `_4`, which removed it |
| Coverage | 1610-01-01 → 2010-12-31 | **unchanged** |
| Correction provenance | none | 109 documented entries with DOIs in `_3`, 120 in `_4` |

## What changed from v1.3_3 to v1.3_4

| | `_3` | `_4` |
|---|---|---|
| Data rows | 1,156,673 | **1,156,368** |
| Rows with `STATION > 0` | 1,100,144 | **1,099,986** |
| Stations with data | 740 | **740** |
| Observers listed | 775 | **776** |
| **D16** duplicate `(date, station)` keys | 217 (95 conflicting) | **3 (1 conflicting)** |
| **D17** max `GROUPS` | 93 | **58** |
| **D18** string `'NA'` sentinel | 33 observers | **none** |
| **D19** sheet flag column | `OBSER` | **`OBSERVER`** |
| **D20** changed-in-v1.3 | green fill only | **explicit column** |
| Row order | unsorted | **date-sorted** |

Observer-day diff: **144 new, 78 removed, 351 with changed `GROUPS`**. Net rows fall because removing
the ~428 duplicate rows outweighs the additions, so `_4` is a correction-and-deduplication release
rather than an extension. **Coverage still stops at 2010-12-31.**

**v1.3 does not extend past 2010.** It is a correction release. `../gn_observer_extension_2026-10/`
closes 2011–2026 from KMAS and SWPC daily group counts.

## Caveats (see `docs/04_data/known_issues.md`)

- **D16** — mostly fixed. `_4` leaves **3** duplicate `(date, station)` keys, of which **one
  conflicts**: Takuma (station 768) on 1982-10-13 reports 8 and 7. The other two are Wolf
  (station 773) on 1874-04-22 and 1876-05-06, `GROUPS = 0` both times. The converter still preserves
  every row; the loader resolves them, and its default rule is **`max`**, so the Takuma day is kept
  at 8 (its neighbours run 9, 7, 8). `_3`'s 217 keys and 95 conflicts in stations 755, 329 and 245
  are gone.
- **D17** — partly fixed. The implausible `GROUPS = 93` (1982-09-26, station 768) is **gone from
  `_4`**; `58` (1992-03-18, station 712), inherited unchanged from v1.22 at an unflagged station,
  remains and is now the maximum. Still size state-space truncations from a robust quantile: the
  99.99th percentile is 25 and the 99.999th is 36.
- **D18** — fixed. `_4` leaves placeholder years blank instead of writing the string `'NA'`; 34
  observers have `TOT_OBS = 0`.
- **D19** — partly fixed. `_4`'s sheets name the flag column `OBSERVER`; `ORDINAL` and `FRACYEAR` are
  still absent and still derived.
- **D20** — fixed. `_4` carries an explicit `CHANGED IN v1.3 (1=YES, 0=NOT)` column, which the
  converter prefers. It marks 100 observers against the green fill's 99, differing on station 36
  (MOUTON, G., LYON — a placeholder with no observations), which the column flags and the fill does
  not. The converter reports any such disagreement rather than hiding it.

Verified clean: no invalid calendar dates; February 29 of 1700, 1800 and 1900 correctly absent
(done upstream in v1.21, where v1.22 needed it by hand); `GROUPS = −1` occurs only with
`STATION = 0`; every station carrying observations appears in the observers sheet.

## Citation

Cite the v1.3 revised collection and the corrections its changelog attributes, alongside Vaquero
et al. 2016 (Sol. Phys.) and Hoyt & Schatten 1998. `gn_v13_changelog.csv` carries the per-observer
attributions with DOIs.
