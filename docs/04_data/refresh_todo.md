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

## Still outstanding

| Item | Why | Where to look |
|---|---|---|
| SILSO SN V2.0 daily and monthly to present | Comparison only, never calibration (ADR-002, ADR-003). Current snapshots end 2018-07 and 2017-10 (**D6**) | https://www.sidc.be/SILSO/datafiles |
| Published GN series for comparison (Svalgaard & Schatten 2016, Usoskin et al. 2016/2021, Chatzistergos et al. 2017, Velasco Herrera et al. 2022, Hoyt & Schatten 1998) | Benchmarks for the paper; `legacy/GN_Series_matlab/data/` has three older yearly series | Papers' supplementary data |
| Observer metadata (aperture, projection vs direct, career dates, age) where available | Optional covariates under ADR-004; v1.3's `OBSERVER` field carries some instrument notes in free text (e.g. station 773's "83/1320 mm Fraunhofer refractor") but it is not a structured table | v1.3 observers sheet; literature |
| An observer-collection extension past 2010 | The extension in hand is survey-derived, not visual observer counts (**D21**). Raised with the v1.3 authors in `docs/results/gn-v13-feedback-report.md` | v1.3 maintainers |
| Annual ¹⁴C (Brehm et al. 2021), ¹⁰Be, aurora catalogues | Step 6 only (proxy-informed variant) | See `docs/05_roadmap/step-6-*.md` |

## Open questions referred upstream

`docs/results/gn-v13-feedback-report.md` is a report to the v1.3 authors on five findings. Two bear on work here:
**D16** (217 duplicate `(date, station)` keys, 95 with conflicting counts) needs a resolution rule before any loader
keys on `(date, station)`, and **D17** (`GROUPS = 93`, new in v1.3 and ~10× the same observer's neighbouring days)
affects how the Step 4 state space is sized.

Not in scope: sunspot drawings / group positions and butterfly data (decided by the PI; see ADR-004). The reference
snapshot does carry positional columns, but as QA metadata only — ADR-004 bars them as model inputs. A positional
collection exists locally on the original machine (`Codes/Bfly Diagram Update/`) if this decision is revisited.
