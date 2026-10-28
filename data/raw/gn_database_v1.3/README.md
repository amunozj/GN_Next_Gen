# `gn_database_v1.3/` — revised collection of group counts, v1.3 (1610–2010)

Current observation database for this project; supersedes `../gn_database_v1.22/`, which is kept so
Step 2 can reproduce the legacy baselines against the database the legacy code actually used.

## Files

| File | What |
|---|---|
| `Revised collection_v1.3_3.xlsx` | the authoritative upstream artifact, byte-exact (28.6 MB, 5 sheets) |
| `gn_v13_observations.parquet` | the three observation sheets concatenated, with `ORDINAL` and `FRACYEAR` added |
| `gn_v13_observers.parquet` | the 775 listed observers, with `changed_in_v13` recovered from cell fill |
| `gn_v13_changelog.csv` | the four side-by-side version blocks of `Differences with HS98`, unstacked |

The three derived files are produced by `scripts/convert_gn_database_v13.py`, which is deterministic:
re-running it on the same workbook reproduces them byte-for-byte (`--check` verifies this). They are
frozen here rather than regenerated into `data/interim/` because parsing 1.16 M rows out of the xlsx
takes minutes, and because `gn_database_v1.22/` set the same precedent (its manifest row records it
as "converted to CSV with ORDINAL and FRACYEAR added").

## Schema — `gn_v13_observations.parquet`

One row per (day, station) observation record. 1,156,673 rows.

| Column | Meaning |
|---|---|
| `YEAR`, `MONTH`, `DAY` | calendar date (proleptic Gregorian) |
| `ORDINAL` | `datetime.date.toordinal()`; 1610-01-01 = 587676, the same convention as v1.22 (**derived**) |
| `STATION` | observer/station code, join to the observers table. **0 = no data / ignore** |
| `OBSER` | 1 if data, else 0. Named `OBSERVER` in v1.22 |
| `GROUPS` | number of groups reported. **−1 = missing**, only ever with `STATION = 0` |
| `FRACYEAR` | `YEAR + (ORDINAL − ordinal(Jan 1)) / days_in_year` (**derived**) |
| `SHEET` | source sheet, one of `1610-1946`, `1947-1988`, `1989-2010` |

`ORDINAL`, `FRACYEAR` and `SHEET` do not exist in the workbook; the first two reproduce v1.22's
definitions so downstream code is schema-compatible, and `SHEET` preserves which sheet a row came
from.

## Schema — `gn_v13_observers.parquet`

| Column | Meaning |
|---|---|
| `STATION` | 1–775 |
| `INITIAL`, `FINAL` | first/last year of observations, **null** where the sheet carries the string `'NA'` |
| `TOT_OBS` | number of observation days (0 for the 33 placeholder observers) |
| `OBSERVER` | `"SURNAME, INITIALS, PLACE"` |
| `changed_in_v13` | true for the 94 observers whose row is highlighted in the sheet |

## Schema — `gn_v13_changelog.csv`

`version, name, year, note` — 586 rows: 403 for v1.12, 72 for v1.2, 2 for v1.21, 109 for v1.3. The
v1.3 block carries the Maunder-Minimum and early-record corrections with DOIs (Hayakawa et al.
2021/2023/2024, Carrasco et al. 2019/2020/2021/2022/2025, Neuhäuser & Neuhäuser 2018, Vokhmyanin
et al. 2020/2021, and others), which is what resolves known issue D7 for 1610–2010.

## What changed from v1.22

| | v1.22 | v1.3 |
|---|---|---|
| Rows with `STATION > 0` | 1,047,761 | **1,100,144** |
| Distinct stations with data | 715 | **740** |
| Observers listed | 743 | **775** |
| Max `GROUPS` | 58 | **93** (see D17 — implausible, do not size state spaces with it) |
| Coverage | 1610-01-01 → 2010-12-31 | **unchanged** |
| Correction provenance | none | 109 documented entries with DOIs |

**v1.3 does not extend past 2010.** It is a correction release. `../gn_observer_extension_2026-10/`
closes 2011–2026 from KMAS and SWPC daily group counts.

## Caveats (see `docs/04_data/known_issues.md`)

- **D16** — 217 duplicate `(date, station)` keys, 122 with identical `GROUPS` and **95 with
  conflicting values**, concentrated in stations 755, 329, 245, 773 and 768. v1.22 had none. The
  converter preserves every row rather than collapsing them; **the resolution rule is undecided**,
  so do not key on `(date, station)` without choosing one.
- **D17** — `GROUPS = 93` (1982-09-26, station 768) and `58` (1992-03-18, station 712) are the only
  values above 53. 93 groups in a day is physically implausible when the reference survey's all-time
  maximum is 26. Size state-space truncations from a robust quantile, not the maximum.
- **D18** — 33 observers are listed with `'NA'` years and `TOT_OBS = 0`.
- **D20** — "changed in v1.3" exists only as cell fill in the workbook; `changed_in_v13` is the only
  machine-readable form, and it would be lost by any re-export of the sheet.

Verified clean: no invalid calendar dates; February 29 of 1700, 1800 and 1900 correctly absent
(done upstream in v1.21, where v1.22 needed it by hand); `GROUPS = −1` occurs only with
`STATION = 0`; every station carrying observations appears in the observers sheet.

## Citation

Cite the v1.3 revised collection and the corrections its changelog attributes, alongside Vaquero
et al. 2016 (Sol. Phys.) and Hoyt & Schatten 1998. `gn_v13_changelog.csv` carries the per-observer
attributions with DOIs.
