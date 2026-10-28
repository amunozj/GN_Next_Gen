# `gn_observer_extension_2026-10/` — daily group counts from KMAS and SWPC (1954–2026)

Extends the observation record past `../gn_database_v1.3/`, which ends **2010-12-31**. v1.3 is a
correction release, not an extension, so the missing 15 years come from a different source: the
Kislovodsk Mountain Astronomical Station (KMAS) and the NOAA/SWPC Solar Region Summary, both
processed by `sunspot-homogenization` and reduced here to daily group counts.

Both surveys are carried over their **full span**, not only the post-2010 part, with `extends_v13`
marking the genuinely new rows. The pre-2010 overlap — KMAS from 1954, SWPC from 1996 — is what lets
each be calibrated against the existing observer network rather than grafted on unanchored at the
end of the series.

## ⚠ The KMAS single-use rule

KMAS appears **twice** in `data/raw`: as a reference survey in `../reference_groups_2026-10/`, and
as an observer-count source here. Its counts are derivable from its areas, so using both puts the
same observations into the likelihood twice and makes the posterior overconfident.

> **Any run must declare KMAS's role as either reference areas or observer counts, never both.**

**SWPC carries no such constraint** — its areas are not in the reference, so its counts are free.
That is why both surveys are here: SWPC gives a modern observer with no role conflict, so KMAS's
constraint only ever costs one of its two roles.

Enforcing this in the run configuration is a Step 1 follow-up. Until then the rule lives here, in
`provenance.json`, in the `role` column of every row, and in ADR-009.

## Files

| File | What |
|---|---|
| `observer_counts.parquet` | long over (survey, day, threshold): the counts |
| `station_assignment.csv` | the two new station codes, spans, roles and anchor thresholds |
| `provenance.json` | upstream commit, posterior checksum, thresholds, the single-use rule, lineage |

Produced by `scripts/export_observer_extension.py`, which is deterministic (`--check` verifies byte
reproducibility).

## Schema — `observer_counts.parquet`

| Column | Meaning |
|---|---|
| `survey` | `kmas` or `swpc` |
| `station` | **1006** (KMAS) or **1008** (SWPC) — upstream's own survey tags, so they cannot collide with v1.3's 1–775 |
| `date_ord`, `date_frac`, `year`, `month`, `day` | the day |
| `threshold_msh` | acuity threshold; a group counts when its area is **strictly greater** than this (the legacy convention) |
| `groups_native` | count from `area_corr` (native, foreshortening-corrected). **Null** where the survey did not observe |
| `groups_dbcal` | count from `area_corr_dbcal` (Debrecen-calibrated common scale). Null where not observed |
| `groups_v13_convention` | `groups_native` with v1.3's own sentinel: **−1 = no observation**, so the table concatenates onto v1.3 without a join-time rule |
| `day_status` | 0 no observation, 1 observed spotless, 2 observed active |
| `n_groups` | the survey's own unthresholded group count for that day |
| `extends_v13` | true after 2010-12-31 — the rows that are genuinely new |
| `role` | the single-use rule above, carried per row |

Counts appear on **both** area scales because the thresholds are defined on the common Debrecen
scale while native `area_corr` is the model-facing input (ADR-009). Thresholds are the Step 1 sweep
grid 0, 10, 30, 60, 100 MSH plus each survey's own anchor threshold from the joint posterior
(KMAS 23.41, SWPC 20.68 MSH).

**Counts only — no areas.** SWPC is excluded from the reference because its areas sit ~43% below
Debrecen (known issue D13). The export reads its group catalogue to derive counts and writes only
counts, and both the script and a test assert that no area column survives, so the exclusion cannot
be reached around by accident.

## Caveats

- **D21** — these are **survey-derived counts, not visual observer counts**. They come from a
  catalogue after upstream's validity filters, and a "group" is upstream's grouping rather than an
  observer's judgement at an eyepiece. They enter as observers with their own detection parameters,
  which absorbs a systematic difference, but they are not interchangeable in kind with the
  historical records. SWPC areas are quantized in ~5 MSH steps, so its counts at the lowest
  thresholds are coarser than the threshold grid suggests.
- Neither ADR-009 anchor covers 2018–2026 (Debrecen ends 2018-06, RGO 1976-12), so the modern end
  of the series rests on these two as calibrated observers under both anchorings.
- Continuity with the v1.3 network over the overlap is checked in
  `docs/results/raw-data-renovation.md`. A visible step at 2010/2011 would mean the extension is
  grafted rather than calibrated.

## Citation

Kislovodsk Mountain Astronomical Station (Pulkovo Observatory), https://doi.org/10.7910/DVN/M7NDXN;
NOAA/SWPC Solar Region Summary reports via NOAA NCEI. Processing: `sunspot-homogenization` (MIT).
