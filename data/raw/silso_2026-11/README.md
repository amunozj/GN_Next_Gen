# `silso_2026-11/` — SILSO sunspot number V2.0, refreshed 2026-11

International Sunspot Number V2.0 from WDC-SILSO, Royal Observatory of Belgium
(https://www.sidc.be/SILSO/). Supersedes the 2018-era snapshot in `../silso/`, which is retained.

## ⚠ Comparison only

**SN V2.0 may not enter calibration of the new model** (ADR-002, ADR-003). It is a composite, so
calibrating against it would make this series partly a re-derivation of the one it is meant to be
compared with. Cycle phase is latent in the model. `gn_next_gen.io.silso` tags what it loads
`comparison_only`, and `gn_next_gen.config.validate_run_config` refuses any sunspot-only run that
lists SILSO as a calibration input.

## Files

| File | Grain | Coverage |
|---|---|---|
| `SN_d_tot_V2.0.csv` | one row per day | 1818-01-01 → 2026-10-31 |
| `SN_m_tot_V2.0.csv` | one row per month | 1749-01 → 2026-10 |

**Format differs from `../silso/`.** These are SILSO's own downloads: **semicolon-separated, no
header row**. The 2018 snapshot was converted to comma-separated with a header during the ADFSSN
era. Both are byte-exact as received, and the loader accepts either.

Daily columns: year; month; day; fractional year; daily total SN (**-1 = missing**); standard
deviation; number of observations; definitive/provisional marker (1/0).

Monthly columns: year; month; fractional year; monthly mean SN (**-1 = missing**); standard
deviation; number of observations; definitive/provisional marker.

## Citation

WDC-SILSO, Royal Observatory of Belgium, Brussels. Distributed under CC BY-NC 4.0.
