# Published group-number series (2026-11)

**Comparison only, and enforced in code.** The Step 2 scorer never reads this
folder, and ADR-003 forbids calibrating on a published series -- doing so would
close a loop through the same observers the model is meant to recalibrate. Since
2026-11-23 `gn_next_gen.config.validate_run_config` **refuses** a run that gives
a dataset of `kind: published_series` the role `calibration`, in *either*
variant, and every run's `lineage.json` carries a
`published_series_calibration_inputs` list and a `circularity_statement` as the
evidence. That rule is deliberately separate from the ADR-002 proxy firewall:
these are sunspot data, so listing them as proxy-derived would make the run's own
firewall statement false. Declare this folder as:

```yaml
datasets:
  - path: gn_published_series_2026-11
    role: comparison
    kind: published_series
```

`published_gn_series.csv` is yearly and long: one row per
`(series, year)`, with `gn` the **mean daily group count** for every series.
`provenance.json` carries each source file's SHA-256.

## From the read-only 2017 MATLAB snapshot

`legacy/GN_Series_matlab/data/`, **not** re-downloaded: those are the exact bytes
the legacy analyses used, which makes a comparison against the Step 2 legacy
baselines like for like.

| series | years | rows | reference |
|---|---|---|---|
| `hoyt_schatten_1998` | 1610-1995 | 386 | Hoyt & Schatten 1998, Sol. Phys. 179, 189 |
| `svalgaard_schatten_2016_backbone` | 1610-2015 | 406 | Svalgaard & Schatten 2016, Sol. Phys. 291, 2653 |
| `usoskin_series` | 1749-1995 | 247 | Usoskin et al., active-day-fraction family (exact version not recorded upstream) |
| `vaquero` | 1639-1714 | 26 | Vaquero et al., as carried by the 2017 snapshot |
| `zolotova` | 1635-1720 | 127 | Zolotova & Ponyavin, as carried by the 2017 snapshot |

## From upstream supplements, frozen here (2026-11-23)

The two the Step 2 plan named and Step 2 could not get. Both are archived in
this folder and read in place; nothing is extracted into `data/raw/`.

| series | years | rows | reference |
|---|---|---|---|
| `chatzistergos_2017` | 1739-2010 | 272 | Chatzistergos, Usoskin, Kovaltsov, Krivova & Solanki 2017, A&A 602, A69 (CDS J/A+A/602/A69) |
| `velasco_herrera_2024` | 1610-2022 | 413 | Velasco Herrera, Soon, Babynets, Murakozy, Tlatov, Nagovitsyn, Qiu, Svanda & Velasco Herrera 2024, Adv. Space Res. 73, 2788 (doi:10.1016/j.asr.2023.12.011) |
| `velasco_herrera_2024_hoyt_update` | 1610-2022 | 413 | Velasco Herrera et al. 2024 (as above), updating Hoyt & Schatten 1998 |

- **`chatzistergos_2017/`** -- CDS catalogue J/A+A/602/A69 (yearly, monthly,
  daily, plus the byte-by-byte `ReadMe`), from the NAOJ VizieR mirror
  `http://vizier.nao.ac.jp/ftp/J/A+A/602/A69/`. CDS's own `/ftp/` path serves a
  placeholder page, which is what Step 2 recorded as unreachable; the
  `viz-bin/cat` endpoint redirects to the mirror. Errors are **asymmetric**, so
  they land in `gn_sd_hi`/`gn_sd_lo` and `gn_sd` stays null.
- **`velasco_herrera_2024/velasco_herrera_2024.zip`** -- the Adv. Space Res.
  supplement, "Group Number series VERSION 3.14", supplied by the PI from his
  ScienceDirect access. The article is closed and the series is in no open
  repository, so this zip is the archival copy. It also contains monthly and
  daily files, a corrected raw observation database to 2022, the v1.12 and v1.21
  Vaquero databases and a list of excluded values -- see the next section.

## The 12.08 scale trap (D24)

The Velasco Herrera supplement ships two families of file whose data column is
labelled `GSN` in both:

| folder in the zip | units | at 1610 |
|---|---|---|
| `New_Reconstruction_GSN/` | mean daily group **count** | 3.27 |
| `Database_GSN_HOYT/` | Hoyt & Schatten's **12.08 x count** | 72.0 |

The exporter divides the second by 12.08, which reproduces this folder's
`hoyt_schatten_1998` on all **386** overlapping years to within **9e-4** and
extends it from 1995 to 2022. `tests/test_published_series.py` pins that
agreement, so a future change of scale convention fails loudly rather than
silently rescaling a series by a factor of twelve.

## Other things in the Velasco Herrera zip, deliberately not used

`New_Raw_Database_GSN/corrected_raw_GSN_database_1610_2022_Elsy.txt` is a
competing revision of the raw observation database, 1610-2022, in the same shape
as `gn_database_v1.22/` (year, month, day, fractional year, station, observer
flag, count). It is **not** the project's database of record -- ADR-010 makes
that `gn_database_v1.3/` release `_4` -- and it is a *corrected* database, so its
corrections are an author's judgement to be compared against v1.3's changelog,
not adopted. It is interesting for one reason worth a later look: it runs to
**2022**, where v1.3 stops at 2010.

## Year convention

`year` is **as published**: mid-year (`1610.5`) for Hoyt & Schatten, the
backbone, Usoskin and Chatzistergos; integer for Vaquero and both Velasco
Herrera series; exact fractional dates for Zolotova. Floor it before joining
series.

## A caveat on the Usoskin series

The snapshot's `GNiu_y2.txt` carries no version string, so it is recorded as
"Usoskin et al., active-day-fraction family" and **not** asserted to be Usoskin
et al. 2021. Do not cite it as a specific publication without checking.
