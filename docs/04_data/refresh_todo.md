# Data to acquire or refresh

The repository started with a frozen 2017–2020-era snapshot. Current versions are acquired **side by side** (new
folders under `data/raw/`, never overwriting), each with a `MANIFEST.csv` row (URL or origin, retrieval date, version,
SHA-256, license).

**Most of this is done.** The 2026-10 raw-data renovation (`docs/plans/raw-data-renovation-plan.md`,
`docs/results/raw-data-renovation.md`, ADR-009) landed the reference survey, the observation database and the
post-2010 extension. What is left is below.

## Done (2026-10)

| Item | Delivered as | Notes |
|---|---|---|
| Revised group-number database (Vaquero et al. 2016 successors, incl. the MM corrections by Hayakawa et al. 2021/2024 and Carrasco et al. 2021) | `gn_database_v1.3/` | v1.3, 1610–2010. +52,383 observations, +25 stations with data over v1.22. 109 documented corrections with DOIs → resolves **D7** for 1610–2010. **Does not extend past 2010** |
| Debrecen (DPD) to present | `reference_groups_2026-10/` (`debrecen`) | 1974 → 2018-06. Upstream does not read the `HMI_DPD2014–2016` files, so this is as far as Debrecen goes; the modern end comes from KMAS and SWPC instead |
| Kislovodsk (KMAS) group areas update | `reference_groups_2026-10/` (`kmas`) | 1954 → **2026**. Cycle 24–25 continuity achieved |
| Original per-group catalogue with identifiers, for lifetimes | `reference_groups_2026-10/` | All five surveys are per-group throughout; `group_single_entry.parquet` carries `obs_duration`. This was wanted for RGO specifically and is now available for RGO, Debrecen, KMAS, Schwabe and Spörer |
| *(not previously listed)* pre-1874 reference coverage | `reference_groups_2026-10/` (`schwabe`, `sporer`) | 1825–1894. The old `SC_SP_RG…` filename promised Schwabe/Spörer areas but contained none (**D2**); they genuinely exist now |
| *(not previously listed)* observation record past 2010 | `gn_observer_extension_2026-10/` | KMAS and SWPC daily group counts, 1954/1996 → 2026. Counts only. **KMAS single-use rule applies** (ADR-009) |

## Done (2026-11, Step 2)

| Item | Delivered as | Notes |
|---|---|---|
| Published GN series for comparison | `gn_published_series_2026-11/` | **Five**, exported from the read-only 2017 MATLAB snapshot rather than re-downloaded -- those are the exact bytes the legacy analyses used, so a comparison against the legacy baselines is like for like: Hoyt & Schatten 1998, the Svalgaard & Schatten backbone, a Usoskin ADF-family series, and Vaquero's and Zolotova's reconstructions. **Comparison only**, now enforced: `validate_run_config` refuses a `kind: published_series` dataset in role `calibration` in either variant (2026-11-23), and each run's lineage records it. The snapshot records no version string for the Usoskin file, so it is *not* asserted to be Usoskin et al. 2021. The two that were outstanding landed on 2026-11-23 — see the next row |
| Published GN series: **Chatzistergos et al. 2017** and **Velasco Herrera et al. 2024** | `gn_published_series_2026-11/chatzistergos_2017/` and `.../velasco_herrera_2024/velasco_herrera_2024.zip` | **Acquired 2026-11-23**, closing the Step 2 gap; the table is now eight series. *Chatzistergos*: CDS catalogue J/A+A/602/A69 (yearly, monthly, daily, 1739–2010, with **asymmetric** errors), downloaded from the NAOJ VizieR mirror. CDS was not down — its own `/ftp/` path serves a placeholder page, while `cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/602/A69` redirects to the mirror, which works; the earlier "unreachable" note was wrong about the cause. *Velasco Herrera*: the Adv. Space Res. supplement ("Group Number series VERSION 3.14"), supplied by the PI from his ScienceDirect access — the article is closed and the series is in no open repository, so the zip is the archival copy and is read in place. It also carries a **Hoyt & Schatten update to 2022** (12.08-scaled, see **D24**) and a corrected raw observation database to 2022, which is *not* the database of record (ADR-010) but is the only thing in hand that runs past v1.3's 2010 |

## Done (2026-11, Step 1)

| Item | Delivered as | Notes |
|---|---|---|
| SILSO SN V2.0 daily and monthly to present | `silso_2026-11/` | Daily 1818-01-01 → **2026-10-31**, monthly 1749-01 → **2026-10**, from SILSO's own downloads (semicolon-separated, headerless). Resolves **D6** for SILSO. The 2018-era `silso/` is retained; `gn_next_gen.io.silso` reads both. **Comparison only** — the config validator refuses it as a calibration input |
| Observation database correction release | `gn_database_v1.3/Revised collection_v1.3_4.xlsx` | Supplied by the PI 2026-11-21 and landed in place under **ADR-010**; resolves D16–D20 largely on its own. Still stops at 2010-12-31 |

## Still outstanding

| Item | Why | Where to look |
|---|---|---|
| Observer metadata (aperture, projection vs direct, career dates, age) where available | Optional covariates under ADR-004; v1.3's `OBSERVER` field carries some instrument notes in free text (e.g. station 773's "83/1320 mm Fraunhofer refractor") but it is not a structured table | v1.3 observers sheet; literature |
| An observer-collection extension past 2010 | The extension in hand is survey-derived, not visual observer counts (**D21**). Raised with the v1.3 authors in `docs/results/gn-v13-feedback-report.md` | v1.3 maintainers |
| Annual ¹⁴C (Brehm et al. 2021), ¹⁰Be, aurora catalogues | Step 6 only (proxy-informed variant) | See `docs/05_roadmap/step-6-*.md` |

## Open questions referred upstream

`docs/results/gn-v13-feedback-report.md` is a report to the v1.3 authors on five findings. Two bear on work here:
**D16** (duplicate `(date, station)` keys) and **D17** (`GROUPS = 93`) were both raised there and both are
**largely fixed in release v1.3_4**: the duplicates fall from 217 keys to 3, and the 93 is gone. The remaining
conflict is one observer-day (Takuma, 1982-10-13), and the loader resolves it by a documented rule.

Not in scope: sunspot drawings / group positions and butterfly data (decided by the PI; see ADR-004). The reference
snapshot does carry positional columns, but as QA metadata only — ADR-004 bars them as model inputs. A positional
collection exists locally on the original machine (`Codes/Bfly Diagram Update/`) if this decision is revisited.
