# Data to acquire or refresh (Step 1 task)

The repository starts with a frozen 2017–2020-era snapshot (`data/raw/`). Before producing a series that competes with
current ones, acquire current versions **side by side** (new folders under `data/raw/`, never overwriting), each with a
`MANIFEST.csv` row (URL, retrieval date, version, SHA-256, license). URLs below are starting points; confirm the current
locations when retrieving.

| Item | Why | Where to look |
|---|---|---|
| Latest SILSO group-number database (Vaquero et al. 2016 successors, including MM corrections by Hayakawa et al. 2021/2024 and Carrasco et al. 2021) | Current observer records, corrected spotless days, observers added since V1.22, coverage past 2010 | https://www.sidc.be/SILSO/groupnumberv3 and SILSO data portal; papers in assessment §2 |
| SILSO SN V2.0 daily and monthly to present | Comparison only (not used in calibration) | https://www.sidc.be/SILSO/datafiles |
| Debrecen sunspot data (DPD, HMI-based catalogues) to present | Extend the reference past 2018-06 | http://fenyi.solarobs.csfk.mta.hu/en/databases/ |
| Kislovodsk (KMAS) group areas update | Cycle 24–25 reference continuity | https://doi.org/10.7910/DVN/M7NDXN |
| Original RGO catalogue (per-group records with identifiers) | Group identities and lifetimes for Step 1 checks (not available from day-by-area tables) | Debrecen GPR / NOAA NCEI / UK Solar System Data Centre |
| Published GN series for comparison (Svalgaard & Schatten 2016, Usoskin et al. 2016/2021, Chatzistergos et al. 2017, Velasco Herrera et al. 2022, Hoyt & Schatten 1998) | Benchmarks for the paper; `legacy/GN_Series_matlab/data/` has three older yearly series | Papers' supplementary data |
| Annual ¹⁴C (Brehm et al. 2021), ¹⁰Be, aurora catalogues | Step 6 only (proxy-informed variant) | See `docs/05_roadmap/step-6-*.md` |

Not in scope: sunspot drawings / group positions and butterfly data (decided by the PI; see ADR-004). A positional
collection exists locally on the original machine (`Codes/Bfly Diagram Update/`) if this decision is revisited.
