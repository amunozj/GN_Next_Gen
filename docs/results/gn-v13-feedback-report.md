# Notes on the v1.3 revised collection of sunspot group counts

**For:** the authors and maintainers of the v1.3 revised collection
**From:** Andrés Muñoz-Jaramillo (GN_Next_Gen project)
**Date:** 2026-10-28
**Workbook audited:** `Revised collection_v1.3_3.xlsx`, 28,573,960 bytes, 5 sheets

We are building a new group sunspot number series on a joint generative model, and v1.3 is its
primary observation input. Before writing loaders against it we ran a full audit of the workbook.
This note records what we found, so anything that is an actual defect can be fixed at source
rather than worked around indefinitely downstream.

**First, the headline:** v1.3's `Differences with HS98` sheet carries **109 documented corrections
with DOIs** for v1.3 alone — Hayakawa et al. (2021, 2023, 2024), Carrasco et al. (2019, 2020, 2021,
2022, 2025), Neuhäuser & Neuhäuser (2018), Vokhmyanin et al. (2020, 2021), Aparicio et al. (2014,
2022), Nogales et al. (2020), Fujiyama et al. (2019) and others. v1.22 as it reached us shipped no
provenance at all. Having the attribution per observer and per year, in the file itself, is a large
practical improvement: it is what let us close a long-standing open issue in our own repository
about whether the Maunder-Minimum corrections had been applied.

Every number below is printed by a single script, `scripts/audit_gn_database_v13.py` in our
repository, which reads the `.xlsx` directly and takes no other input, so each claim can be
re-derived against your own copy. Where we could not tell intent from the file, we have asked a
question rather than called it an error.

## Scale, for reference

| | v1.22 (as we received it) | v1.3 |
|---|---|---|
| Data rows | 1,100,251 | 1,156,673 |
| Rows with `STATION > 0` | 1,047,761 | 1,100,144 |
| Distinct stations with data | 715 | 740 |
| Observers listed | 743 | 775 |
| Coverage | 1610-01-01 → 2010-12-31 | 1610-01-01 → 2010-12-31 |

Sheet row counts: `1610-1946` 405,140; `1947-1988` 347,248; `1989-2010` 404,285.

## 1. 217 duplicated `(date, station)` keys, 95 of them with conflicting counts

This is the only finding that blocks us, so we have put it first. We also think we can point at
where it came from.

v1.3 contains **217 `(date, station)` pairs that occur more than once**. Of these, **122 repeat an
identical `GROUPS` value** and **95 carry two or more different values** for the same observer on
the same day.

Two pieces of context make this look like something that arrived with v1.3 rather than a
long-standing feature of the collection:

- **v1.22 contains exactly one such duplicated key. v1.3 contains 217.**
- **All five affected stations are flagged as changed in v1.3** (the green highlighting in
  `Observers`), and **no unflagged station has a duplicated key at all** — 5 of the 94 flagged
  stations, 0 of the other 681.

The five stations:

| Station | Duplicated keys | Observer | Changed in v1.3 |
|---|---|---|---|
| 755 | 94 | SACRAMENTO PEAK OBSERVATORY, NEW MEXICO | yes |
| 329 | 76 | SECCHI, ROME | yes |
| 245 | 41 | LINDENER, B.A., GLATZ | yes |
| 773 | 4 | WOLF, R., ZURICH (Friedli recount) | yes |
| 768 | 2 | TAKUMA, JAPAN | yes |

By year, the largest clusters are 1950 (35 keys), 1871 (26), 1815 (15), 1949 (11), 1873 (10) and
1966 (8).

Examples with conflicting values:

```
station 245  1801-06-04  GROUPS = [3, 3, 4]
station 245  1815-05-05  GROUPS = [2, 3]
station 245  1816-06-07  GROUPS = [3, 1]
station 329  1859-02-09  GROUPS = [11, 8]
station 329  1860-07-18  GROUPS = [7, 7, 7, 7, 6]
station 329  1860-12-14  GROUPS = [8, 7]
station 329  1861-04-27  GROUPS = [11, 13]
station 329  1864-07-23  GROUPS = [3, 2]
```

**Our question is what these are meant to represent.** Given that the affected stations are exactly
the ones v1.3 revised, our leading guess is a **merge artifact**: old and recounted records landing
side by side rather than the new one replacing the old. Station 773 is explicitly labelled
"Friedli recount" in the observers sheet, which fits. If that is right, the 122 identical-value
cases are duplicates to drop and the 95 conflicting ones need adjudication between the old and
revised value.

Two other readings are possible and would mean something quite different:

- **Several observations on one day.** Station 755 is an observatory rather than an individual, so
  multiple daily reports are plausible there, and `[7, 7, 7, 7, 6]` could be five reports. Against
  this: 329 and 245 are single named observers, and five same-day entries for Secchi is harder to
  justify. If this reading is right anywhere, an explicit intra-day sequence or time column would
  resolve it, or a line in the documentation stating that repeated keys are intentional and how to
  combine them.
- **One station code covering more than one observer or site**, which a sub-identifier would fix.

We have deliberately **not** guessed. Our conversion preserves every row and our loader refuses to
key on `(date, station)` until we know which reading applies: silently taking the first, the maximum
or the mean of `[11, 13]` would change the group count for that day in a way nobody could later
audit. If the intra-day reading turns out to be correct, this is a documentation matter rather than
a data defect, and we would simply note it and move on.

## 2. Two implausible `GROUPS` values — one of them new in v1.3

Only two records in the whole database exceed 53 groups:

```
1982-09-26  station 768 (TAKUMA, JAPAN)                   GROUPS = 93   [sheet 1947-1988]
1992-03-18  station 712 (WATUKOSEK OBSERVATORY, INDONESIA) GROUPS = 58   [sheet 1989-2010]
```

For context, every `GROUPS` value above 40 in v1.3 is: **42, 44, 48, 50, 53, 58, 93**. So 93 sits
far above the next value, and 58 is itself detached from 53.

The two have different histories, which we think matters:

**The 93 is new in v1.3.** There is no row for station 768 on 1982-09-26 in v1.22 at all, and
station 768 is one of the flagged changed stations. The same observer's counts through the rest of
that month are:

```
Sep 1982, station 768:  8 10 10 14 15  8 10  6  9  9  6  6  5  8  [93]  9  7
                                                                   ^ 26 Sep
```

So 93 is roughly ten times what this observer reported on the days either side, and it entered with
the v1.3 revision. **A transcription slip looks far more likely than a real observation** — 9 would
sit naturally in that series, and 1982-09-26 was not an exceptional day by any other measure.

**The 58 is inherited.** It is present in v1.22 with the same value, and station 712 is *not*
flagged as changed, so this one is pre-existing rather than a v1.3 regression. It may still be
worth a look, but it is not something v1.3 introduced.

For scale: the modern photographic reference surveys we cross-check against — RGO, Debrecen,
Kislovodsk — record an all-time daily maximum of **26** groups. A visual count of 93 would be three
and a half times the best-ever reference day.

These two matter to us out of proportion to their number, because the maximum reported count is the
natural thing to size a latent state space with: taken at face value, 93 would inflate that state
space by about 60% for the sake of two rows. We are using a robust upper quantile instead — the
99.99th percentile of `GROUPS` is below 40 — but it would be better if the values themselves were
right or flagged.

## 3. 33 observers listed with no observations

33 of the 775 rows in `Observers` carry the **string** `'NA'` in `INITIAL` and `FINAL` with
`TOT.OBS = 0`:

```
station  11  MARIUS, S., NUREMBERG
station  20  ZAHN, J., NUREMBERG
station  24  CRABTREE, W., ENGLAND
station  27  LINEMANNS, A., REGIOMONTUS
station  31  PICARD/KEILL, PARIS
station  35  MARALDI, F., BONONIA
station  36  MOUTON, G., LYON
station  40  WEIGEL, E., JENA
   ... 25 more
```

This is self-consistent and matches the v1.12 changelog's "Removed" entries, so we read it as
deliberate: the station code is retained so numbering stays stable while the records are withdrawn.
Two small points. The `'NA'` is a text value in otherwise numeric columns, so a naive numeric read
of `INITIAL`/`FINAL` fails on these rows — an empty cell would be friendlier. And it is worth saying
in the documentation that 740 of the 775 codes carry data, since "775 observers" otherwise overstates
the network by 35.

## 4. The v1.3 change flags exist only as cell formatting

The `Observers` sheet says, in cell `I1`, "All new changes are indicated in green", and **94 of the
775 station cells are filled**. We could read them, and we now carry the result as a boolean column.

The difficulty is that this information exists **nowhere in the cell values**. It does not survive
export to CSV, it does not survive most programmatic re-saves, and it is invisible to anyone reading
the file with a library that ignores styling. It is genuinely useful information — knowing which 94
observers moved in v1.3 is exactly what a downstream user wants — so we would suggest promoting it
to a real column, for instance `CHANGED_IN_V1.3` with a 1/0 or the version string. We have pinned
the count at 94 in a test so that if our fill-reading ever silently breaks, we find out.

## 5. Layout and schema notes

These are conveniences rather than defects, but all three cost us code:

- **`Differences with HS98` is four tables side by side in one sheet**, at columns A, I, Q and Y for
  v1.12, v1.2, v1.21 and v1.3 (403, 72, 2 and 109 entries). A `VERSION` column in one long table
  would be directly machine-readable, and would also make it obvious that the four blocks are
  cumulative history rather than alternatives.
- **The `1610-1946` sheet has 96,662 trailing blank rows** (declared extent 501,803 against 405,140
  rows of data). Harmless once known, but a reader that trusts the declared sheet extent gets 24%
  phantom rows. The other two sheets are clean.
- **Column naming and derived columns.** The observation sheets use `OBSER`, where v1.22 as
  distributed to us used `OBSERVER`; and v1.3 has no `ORDINAL` or `FRACYEAR`. We now understand
  those two were added downstream rather than being yours, so this is only a note for anyone
  migrating from a v1.22-derived CSV: the schema is not quite drop-in.

## What we verified as clean

Stated for completeness, because these are the things we checked that came back right:

| Check | Result |
|---|---|
| Invalid calendar dates | 0 |
| February 29 of 1700, 1800, 1900 present | 0 — correctly absent |
| `GROUPS < 0` while `STATION > 0` | 0 |
| `STATION = 0` rows whose `GROUPS` is not `−1` | 0 |
| Stations carrying data but missing from `Observers` | 0 |
| Max station code, data vs listed | 775 vs 775 |
| Coverage | continuous 1610-01-01 → 2010-12-31 |
| Sheet seams | tile exactly at 1946/1947 and 1988/1989 |

The sentinel conventions are internally consistent throughout, and the removal of the three
non-leap February 29ths is handled in the file itself — v1.22 required us to verify that by hand.

## One request, if it is ever on the table

v1.3 ends 2010-12-31. We extend past it for now with daily group counts derived from Kislovodsk and
NOAA/SWPC, which works but is not the same kind of record as the observer network. If an extension
of the observer collection past 2010 is ever planned, we would be glad to hear of it, and glad to
help.

## Summary

| # | Finding | Our read |
|---|---|---|
| 1 | 217 duplicate `(date, station)` keys, 95 with conflicting counts; v1.22 had 1, and all 5 affected stations are v1.3-revised | **Blocking for us** — looks like a merge artifact, but needs a statement of intent |
| 2 | `GROUPS` = 93 (new in v1.3, ~10× the same observer's neighbouring days) and 58 (inherited from v1.22) | 93 looks like a transcription slip; would value your judgement |
| 3 | 33 observers with `'NA'` years and `TOT.OBS = 0` | Deliberate; documentation and sentinel-type suggestion |
| 4 | 94 change flags carried only as cell fill | Works, but fragile — suggest a real column |
| 5 | Side-by-side changelog blocks, 96,662 blank rows, `OBSER`/derived columns | Convenience issues |

Only item 1 actually blocks us, and it may well turn out to be a documentation question rather than
a defect. Items 2 to 5 we can and do handle on our side; we raise them because fixing them at source
would save every other user the same work.

Happy to share the audit script, the full duplicate-key list, or anything else that would be useful.

---

*Reproduce with `python scripts/audit_gn_database_v13.py` against
`Revised collection_v1.3_3.xlsx` (sha256 `31988ff4cb9d...`, full value in `data/raw/MANIFEST.csv`).*
