# `reference_groups_2026-10/` — the reference survey: five per-survey group catalogues (1825–2026)

Current reference survey for this project, fixed by **ADR-009**. Supersedes
`../reference_areas/`, whose day × `AREA1..AREA26` matrices are kept byte-exact so Step 2 can
reproduce the legacy baselines and so the D3 characterization still has something to characterize.

Frozen snapshot of the per-survey output of
[`sunspot-homogenization`](https://github.com/amunozj) (MIT, same PI) at a pinned commit; see
`provenance.json` for the commit, the calibration posterior and its checksum.

## What this is, and what it replaces

| | `../reference_areas/` | this snapshot |
|---|---|---|
| Coverage | 1874-04-17 → 2018-06-19 | **1825-11 → 2026**, continuous |
| Granularity | day × 26 anonymous area slots | one row per (survey, group, day) |
| Group identities, lifetimes | none | `group_id`, `obs_duration` |
| Day status | inferred from `AREA1` being NaN or 0 | explicit `day_status` + `n_groups` |
| Uncertainties | none | per-area, split correlated / independent |
| Area semantics | undocumented (issue D4) | whole-spot and umbral, foreshortening-corrected, MSH |
| Cross-survey scale | none | joint posterior, Debrecen anchor, with credible intervals |

## Surveys

The **core five**. SOON is excluded entirely; SWPC is excluded from the reference and contributes
observer *counts* only, in `../gn_observer_extension_2026-10/` (known issue D13).

| `survey` | Coverage | Role |
|---|---|---|
| `schwabe` | 1825–1867 | observer |
| `sporer` | 1861–1894 | observer |
| `rgo` | 1874–1976 | **secondary anchor**, cycles 15–19 |
| `debrecen` | 1974–2018 | **primary anchor**, cycles 21–23 |
| `kmas` | 1954–2026 | observer; bridges the two anchors via cycle 20. **Dual-use, see below** |

## Files

| File | What |
|---|---|
| `group_day_observations.parquet` | **the reference**: one row per group observation (survey, group, day) |
| `group_single_entry.parquet` | one row per (survey, group): canonical peak day, `obs_duration` |
| `day_status.parquet` | one row per (survey, day): observation status and daily totals |
| `survey_calibration.csv` | per-survey scale factor, credible interval and anchor threshold |
| `survey_cycle_coverage.csv` | per (survey, cycle): extent and the `complete` flag ADR-009's anchors rest on |
| `provenance.json` | commit, posterior checksum, coverage, row counts, lineage statement |

Produced by `scripts/export_reference_groups.py` from `configs/reference_groups_export.yaml`;
deterministic, and `--check` verifies byte reproducibility.

## Which area column to use

**`area_corr` is the model-facing area** — native, whole-spot, foreshortening-corrected, MSH.

`area_corr_dbcal` and its uncertainty components are the same areas mapped onto the common Debrecen
scale by the upstream joint posterior. They are a **prior and cross-check only, never a hard
pre-correction**: this project fits the inter-survey detection differences itself, as ADR-003 and
issue D5 require, so the upstream calibration must not be applied twice.

`area_corr_dbcal_unc_corr` is correlated across a whole survey (it is the calibration factor, so it
adds *linearly* when groups are summed into a daily total) and `area_corr_dbcal_unc_indep` is
per-object (it adds in quadrature). That split, not the per-object scatter, dominates the error
budget of any long-term integrated index.

## Columns

`group_day_observations` and `group_single_entry`:

- identity — `survey`, `group_id`, `group_id_orig`
- date — `date_ord`, `date_frac`, `year`, `month`, `day`
- areas (MSH) — `area_obs(_unc)`, `area_corr(_unc)`, `area_umbra_obs(_unc)`, `area_umbra_corr(_unc)`
- calibrated areas — `area_corr_dbcal`, `_unc`, `_unc_corr`, `_unc_indep`, `_above_thr`
- diagnostics — `cycle`, `cycle_parity`, `amp_whole_sm91`, `amp_hemi_sm91`, `amp_whole_dbcal_sm91`, `amp_hemi_dbcal_sm91`
- QA metadata — `hemisphere`, `dist_center`, `latitude`, `longitude`
- `group_single_entry` adds `group_obs_kind` and `obs_duration`

`day_status`: `survey`, `date_ord`, `date_frac`, `year`, `month`, `day`, `day_status`, `n_groups`,
`area_total`, `area_north`, `area_south`.

### Two renames you must know about (issue D10)

`obs_type` means **two different things** upstream, so neither name survives the export:

| Upstream | Here | Values |
|---|---|---|
| `obs_type` in the single-entry stage | **`group_obs_kind`** | 0 placeholder row, 1 single-observation group, 2 multi-observation group |
| `obs_type` in the `flags` stage | **`day_status`** | 0 no observation, 1 observed spotless, 2 observed active |

`day_status` is the explicit replacement for the old `AREA1 = NaN` / `AREA1 = 0` convention.

### Counting groups above a threshold

The number of groups a survey saw above acuity threshold `T` on a day is the number of rows for
that (survey, day) with area **strictly greater** than `T` — the legacy convention. The snapshot is
deliberately **unthresholded** so Step 1 can sweep T ∈ {0, 10, 30, 60, 100} MSH; issue D14 bounds
what the lowest thresholds can mean, since per-survey completeness floors differ.

## Not model inputs

- `cycle`, `cycle_parity` and the `amp_*` columns come from upstream's butterfly-diagram fit on the
  surveys' own group positions and their own smoothed areas — **not** from SILSO. They are
  **diagnostic only and never calibration inputs**, because ADR-003 requires cycle phase to stay
  latent. The `amp_*` columns are the "reference's own smoothed activity" the Step 1 spec permits
  for conditioning in Check B.
- `latitude`, `longitude`, `hemisphere` and `dist_center` are **QA and provenance metadata only**.
  ADR-004 bars positions as model inputs. They are retained so the foreshortening and disk-angle
  audits (issue D12) are reproducible without re-running upstream, and because `cycle` was derived
  from them.

## The anchored epochs (ADR-009)

A cycle counts as **complete** in a survey only if that survey also carries its neighbours — the
neighbour test is what catches a cycle clipped at the edge of a survey's coverage.
`survey_cycle_coverage.csv` carries the flag.

- **Primary anchor — Debrecen, cycles 21–23.** Debrecen is the identity map in the joint posterior
  (CF ≡ 1), so native and calibrated scales coincide at the anchor and no external correction enters
  the estimand there.
- **Secondary anchor — RGO, cycles 15–19.** Complete, and entirely post-1915, matching the legacy
  code's decision to drop earlier RGO (issue D5).

The two share no cycle. **Cycle 20 is in neither** — it is precisely the cycle RGO and Debrecen
would have split — and is complete in **KMAS alone**, so KMAS bridges the two anchorings.

## ⚠ KMAS is dual-use

KMAS is a reference survey here **and** supplies observer counts in
`../gn_observer_extension_2026-10/`. Its counts are derivable from its areas, so:

> **Any run must declare KMAS's role as either reference areas or observer counts, never both.**

SWPC has no such constraint. See that folder's README and ADR-009.

## ⚠ KMAS group identities are not unique (issue D22)

KMAS reuses a `group_id` for physically distinct groups on **414 (group, day) keys** — 839 rows,
289 distinct ids, 1.30% of its 22,295 groups. The rows within a key sit at different heliographic
positions, so they are separate groups rather than one group measured twice: `KMAS_1989_389` on
1989-10-05 carries six rows at latitudes −29, 29, 24, 18, 22 and 11. The other four surveys are
clean.

Consequences:

- **The row is the unit of observation, not `(group_id, date_ord)`.** Daily counts, which count
  rows, are correct — including `n_groups` here and the counts in the extension folder.
- **Anything keyed on `group_id` is not.** The 289 affected groups have a median `obs_duration` of
  11 days against 5 for clean KMAS groups (max 314), because two apparitions are read as one. Step 1
  Check A should exclude them from lifetime distributions, or use RGO and Debrecen, which are clean.
  It is under 2% of KMAS groups, so dropping them is cheap.

## Other caveats

See `docs/04_data/known_issues.md` D10–D15: the `obs_type` rename (D10), Spörer's unverified area
unit (D11), upstream's unapplied `max_disk_angle` (D12), the SOON/SWPC area offsets (D13),
per-survey completeness floors (D14) and the absence of upstream tags or checksums (D15).

## Citation

Cite the source surveys — Schwabe and Spörer via Arlt et al.; Royal Greenwich Observatory and the
Debrecen Heliophysical Observatory (GPR/DPD), http://fenyi.solarobs.csfk.mta.hu/; Kislovodsk
Mountain Astronomical Station (Pulkovo), https://doi.org/10.7910/DVN/M7NDXN — and the
`sunspot-homogenization` processing and cross-calibration (MIT).
