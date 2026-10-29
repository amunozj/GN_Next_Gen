# Results: raw-data renovation (2026-10)

- Plan: [`docs/plans/raw-data-renovation-plan.md`](../plans/raw-data-renovation-plan.md) (Approved 2026-10-28)
- Decision: [ADR-009](../06_decisions/ADR-009-reference-survey-and-anchored-epochs.md)
- Branch: `data/raw-renovation-2026-10`
- Environment: conda env `ssn` (Python 3.11) with `PYTHONNOUSERSITE=1`; see *Deviations*
- Upstream pinned at `sunspot-homogenization` commit `9d6abd16d590eb67eb7fae215bb7f3afdfa0bffc`
  (branch `joint-generative-calibration`, no tags exist), calibration posterior
  `trusted_core_debrecen_umbral.pkl` sha256 `09026bda…8851`, model version `joint_umbral_v5`

## What landed

| Workstream | Deliverable | Status |
|---|---|---|
| A | Reference survey → `data/raw/reference_groups_2026-10/` | Done |
| B | Observation database v1.22 → v1.3 (`data/raw/gn_database_v1.3/`) | Done |
| C1 | Post-2010 extension → `data/raw/gn_observer_extension_2026-10/` | Done |
| C2 | Report to the v1.3 authors | Done |

`data/raw` grew from 66 MB to 140 MB across 16 new `MANIFEST.csv` rows — inside ADR-006's revisit
triggers (~100 MB per file, ~1 GB total); the largest file is still v1.22's 46.8 MB CSV.

## A — the reference survey

Five per-survey, per-group, per-day catalogues, **1825-11-05 → 2026-03-20**, with no gap at any
handover. Measured row counts:

| Survey | Group-days | Groups | Calendar days | Span |
|---|---|---|---|---|
| schwabe | 31,530 | 6,903 | 15,705 | 1825–1867 |
| sporer | 5,421 | 5,421 | 12,418 | 1861–1894 |
| rgo | 143,772 | 25,611 | 37,620 | 1874–1976 |
| debrecen | 68,678 | 11,412 | 16,436 | 1974–2018 |
| kmas | 126,399 | 22,295 | 26,663 | 1954–2026 |
| **total** | **375,800** | **71,642** | **108,842** | **1825–2026** |

(The plan estimated ~459,000 group-days from the upstream CSV line counts. Those counts included
the placeholder calendar rows that carry no group; those rows belong in `day_status`, so the
group tables are correspondingly smaller. No data is missing.)

### Cycle completeness and the anchors

`survey_cycle_coverage.csv` reproduces ADR-009's table exactly. A cycle counts as complete in a
survey only if that survey also carries its neighbours — the test that catches a cycle clipped at
the edge of a survey's coverage:

| Survey | Cycles present | **Complete** | Clipped (and why) |
|---|---|---|---|
| schwabe | 7–10 | **8, 9** | 7 (no cycle 6), 10 (no cycle 11) |
| sporer | 8–11 | **9, 10** | 8 (no cycle 7), 11 (no cycle 12) |
| rgo | 11–20 | **12–19** | 11 (no cycle 10), **20 (no cycle 21)** |
| debrecen | 20–24 | **21, 22, 23** | **20 (no cycle 19)**, 24 (no cycle 25) |
| kmas | 19–25 | **20–24** | 19 (no cycle 18), 25 (no cycle 26) |

So the two anchors hold as specified and **share no cycle**: Debrecen 21–23 (primary, 1974-05-15 →
2012-07-27) and RGO 15–19 (secondary, 1911-12-18 → 1969-03-20). Cycle 20 — the one cycle RGO
(1963-10-08 → 1976-12-31) and Debrecen (1974-01-02 → 1980-03-08) would have split — is in neither,
and is complete in **KMAS alone** (1960-07-21 → 1979-11-19), so KMAS bridges the two anchorings.

### Old versus new

The superseded `SC_SP_RG_DB_KM_group_areas_by_day_pre5898ab8.csv` holds 52,659 days, 52,182 of them
observed and 8,758 spotless. Comparing daily counts above a threshold, **per survey** rather than
pooled (pooling would double-count the 1954–2018 overlaps, which the old stitched file did not
have), at T = 30 MSH:

| Decade | RGO agrees | Debrecen agrees | KMAS agrees | old median count |
|---|---|---|---|---|
| 1870s | 97% (1,771 d) | — | — | 0 |
| 1880s | 87% | — | — | 2 |
| 1890s | 78% | — | — | 2 |
| 1900s | 83% | — | — | 2 |
| 1910s | 78% | — | — | 2 |
| 1920s | 75% | — | — | 2 |
| 1930s | 72% | — | — | 2 |
| 1940s | 68% | — | — | 4 |
| 1950s | 67% | — | 43% | 4 |
| 1960s | 70% | — | 55% | 3 |
| 1970s | 72% | 74% | 58% | 3 |
| 1980s | — | 72% | 49% | 4 |
| 1990s | — | 73% | 57% | 3 |
| 2000s | — | 77% | 64% | 2 |
| 2010s | — | 75% | 64% | 3 |

Two things follow. First, **the old file is RGO up to 1976 and Debrecen thereafter** — RGO matches
it on 76% of 37,038 days in 1874–1976 and Debrecen on 74% of 15,140 days in 1977–2018, while KMAS
matches far worse in both eras (55% and 57%). That settles the composition the filename only hinted
at. Second, the **median difference is 0 at every threshold** (T = 0, 10, 30, 60, 100), so the new
reference is on the same level as the old one; it differs on about a quarter of individual days,
which is the reprocessing and filtering difference that issue D3 was about. Day status barely
moved: 0 days went from observed to unobserved, and 34 gained an observation.

## B — the observation database

v1.3 landed and converted; every figure below is independently reproduced by
`scripts/audit_gn_database_v13.py` and pinned in `tests/test_gn_database_v13.py`.

| | v1.22 | v1.3 |
|---|---|---|
| Data rows | 1,100,251 | **1,156,673** |
| Rows with `STATION > 0` | 1,047,761 | **1,100,144** |
| Stations with data | 715 | **740** |
| Observers listed | 743 | **775** |
| Coverage | 1610-01-01 → 2010-12-31 | **unchanged** |
| Documented corrections | none | **109, with DOIs** |

**D7 is resolved for 1610–2010.** The v1.3 changelog is the correction literature the issue asked
for, per observer and with DOIs: Hayakawa et al. 2021 (18 entries), Carrasco et al. 2019 (10), 2022
(8), Neuhäuser & Neuhäuser 2018 (8), Carrasco et al. 2021 (6), 2020 (4), Vokhmyanin et al. 2021
(3), Hayakawa et al. 2024 (3), 2023 (3), and more.

**v1.3 does not extend past 2010**, so D6 stays open for v1.3 itself; workstream C closes the gap
from a different source.

Five new issues came with it (D16–D20). Two matter beyond documentation and both are reported
upstream:

- **D16** — 217 duplicate `(date, station)` keys, 95 with conflicting `GROUPS`. v1.22 has exactly
  one. All five affected stations are flagged as changed in v1.3 and no unflagged station has one
  (5 of 94 flagged, 0 of the other 681), so they arrived with the corrections — most likely recounted
  records landing beside the old ones. **No loader may key on `(date, station)` until a rule is
  chosen**; this plan characterizes it and leaves the rule to Step 1.
- **D17** — `GROUPS = 93` (1982-09-26, station 768 Takuma) and `58` (1992-03-18, station 712) are
  the only values above 53. The 93 is **new in v1.3**: no such row exists in v1.22, station 768 is
  flagged as changed, and the same observer reported 5–15 groups on every other day that month. It
  reads as a transcription slip. The 58 is inherited unchanged from v1.22 at an unflagged station.
  **Consequence:** the Step 4 state-space truncation K must come from a robust upper quantile, not
  the maximum — the 99.99th percentile of `GROUPS` is below 40. The Step 1 spec has been corrected.

## C1 — the post-2010 extension

227,916 rows: KMAS 159,978 (6 thresholds × 25,344 observed days) and SWPC 67,938 (6 × 11,062).
**70,128 rows carry `extends_v13 = true`**, covering 2011-01-01 → 2026-12-31. Counts only; a test
asserts no area column survives.

### Is it calibrated or grafted?

The point of carrying both surveys over their full span was the pre-2010 overlap. Yearly mean daily
count at each survey's anchor threshold, against the v1.3 network (median across stations with ≥50
observing days that year):

| Year | v1.3 network | KMAS | SWPC | KMAS/net | SWPC/net |
|---|---|---|---|---|---|
| 1996 | 0.58 | 0.50 | 0.37 | 0.87 | 0.64 |
| 1998 | 4.67 | 4.09 | 3.20 | 0.87 | 0.68 |
| 2000 | 8.47 | 7.49 | 6.63 | 0.89 | 0.78 |
| 2002 | 8.47 | 6.92 | 7.07 | 0.82 | 0.83 |
| 2004 | 3.40 | 2.75 | 2.68 | 0.81 | 0.79 |
| 2006 | 1.47 | 1.14 | 1.12 | 0.78 | 0.76 |
| 2008 | 0.27 | 0.19 | 0.17 | 0.69 | 0.63 |
| 2010 | 1.56 | 1.39 | 1.09 | 0.89 | 0.70 |

Over the full 15-year overlap: **KMAS/network = 0.825 ± 0.062** (range 0.691–0.894) and
**SWPC/network = 0.734 ± 0.100** (range 0.490–0.847). The relationship is a near-constant
multiplicative offset, stable through a complete solar cycle, which is exactly what a per-observer
detection parameter absorbs. The extension is calibrated against the network, not grafted onto it.
The mild dip at the 2008–2009 deep minimum (0.69 and 0.63) is the expected acuity effect at very low
activity and is itself information the detection model can use.

**The apparent jump across the boundary is solar, not instrumental.** KMAS goes from a 2006–2010
mean of 0.71 to 4.31 in 2011–2015, and SWPC from 0.63 to 3.72 — but the v1.3 network averages 0.86
over 2006–2010 because that window *is* the cycle 23/24 minimum, and cycle 24 rose after 2010. The
network shows the same rise in its own last years (0.27 in 2008 → 1.56 in 2010).

### The KMAS single-use rule

KMAS supplies reference areas in workstream A **and** observer counts here, and its counts are
derivable from its areas, so a run that used both would put the same observations into the
likelihood twice. The rule — **reference areas or observer counts, never both** — is recorded in
ADR-009, both folder READMEs, `provenance.json`, the `role` column of every extension row, and
`docs/04_data/known_issues.md`. A config-level validator is a named Step 1 follow-up; **until it
exists this is enforced only by documentation.** SWPC carries no such constraint.

## C2 — report to the v1.3 authors

[`docs/results/gn-v13-feedback-report.md`](gn-v13-feedback-report.md), also published as a shareable
page. Covers D16–D20 with reproducible evidence, named examples, and what verified clean; framed as
questions where intent could not be read from the file. `scripts/audit_gn_database_v13.py` prints
every number in it, and `tests/test_gn_v13_audit.py` keeps the two from drifting apart.

## New issue found during the work

**D22 — KMAS reuses a `group_id` for physically distinct groups.** 414 (group, day) keys carry 2–6
rows each: 839 rows, 289 distinct ids, 1.30% of KMAS's 22,295 groups. The rows within a key sit at
**different heliographic positions**, which rules out the benign reading of one group measured twice:
`KMAS_1989_389` on 1989-10-05 has six rows at latitudes −29, 29, 24, 18, 22 and 11. The other four
surveys are clean.

- **Daily counts are unaffected**, because they count rows, which is the correct unit — this
  includes upstream's own `n_groups` and the extension's counts.
- **Lifetimes are affected.** The 289 groups have a median `obs_duration` of 11 days against 5 for
  clean KMAS groups (max 314), because two apparitions are read as one. **Step 1 Check A must
  exclude them from lifetime distributions**, or use RGO and Debrecen, which are clean. At under 2%
  of KMAS groups, dropping them is cheap.

Affected keys cluster at the recent end (91 in 2023, 40 in 1990, 31 in 1989), which suggests a
reader or merge issue rather than a historical one. Worth raising upstream.

## Verification

| Check | Result |
|---|---|
| `pytest` (fast suite) | **84 passed**, 1 deselected, 5.1 s |
| `pytest -m slow` | **1 passed** (the audit-script reproduction) |
| `data/raw/MANIFEST.csv` checksums | all 23 rows verified by `tests/test_data_manifest.py` |
| Old raw files untouched | `git diff main -- data/raw legacy` shows only additions; the three `reference_areas/*.csv` and both `gn_database_v1.22/*` checksums are unchanged |
| Reference export reproducible | `--check`: all 5 files **MATCH** |
| Extension export reproducible | `--check`: both files **MATCH** |
| v1.3 converter reproducible | `--check`: all 3 files **MATCH** |
| Cycle completeness and anchors | reproduce ADR-009 exactly (table above) |
| Coverage, no handover gap | 1825-11-05 → 2026-03-20, verified by test |
| Threshold monotonicity | counts non-increasing in `threshold_msh` for every (survey, day) |
| Extension ties to workstream A | at T = 0, KMAS counts equal `day_status.n_groups` exactly |
| No SWPC areas under `data/raw` | asserted by the exporter and by test |
| Firewall (ADR-002) | both `provenance.json` files carry an explicit lineage statement; no isotope, aurora, proxy or SILSO input |
| `ruff check .` | **all checks passed** (14 findings fixed 2026-10-29; see below) |
| `ruff format --check .` | **65 files already formatted**; `legacy/` untouched |

## Deviations from the plan

1. **`ruff` was initially unavailable; now resolved (2026-10-29).** It was not installed in any
   conda env on this machine, and the project's own `gn-next-gen` env does not exist here, so
   `pytest` ran in `ssn` with `PYTHONPATH=src` and lint was replaced by hand checks. The PI has
   since installed `ruff` into `ssn`, and it was run properly: **14 findings, all in code added by
   this branch** — one unused import (`F401`), two `datetime.timezone.utc` → `datetime.UTC`
   (`UP017`), one unsorted import block (`I001`), and nine `zip()` calls without `strict=`
   (`B905`). The `B905` cases were fixed with `strict=True` rather than ruff's suggested
   `strict=False`: each zips columns of a single DataFrame, so lengths are equal by construction
   and a mismatch should raise rather than truncate silently. `ruff format` then reformatted the
   nine new files; no pre-existing file changed and `legacy/` was untouched. Both now pass clean.
   `pytest` still runs in `ssn`, because the project env still does not exist here.
2. **The upstream pipeline could not run as the plan assumed.** `sunspot_pipeline/processing/cycles.py`
   imports `cuml` (GPU) unconditionally, and `cuml` cannot import in the `ssn` env: it pulls in
   `cudf` → `numba`, and the only `numba` visible is a user-site build requiring NumPy ≤ 2.3 where
   the env has 2.4.3. Since the upstream repository must not be modified, the run installs a shim
   that maps `cuml.cluster.hdbscan` to the CPU `hdbscan` package (same `HDBSCAN` signature).
   **This was verified rather than assumed:** re-running Debrecen, RGO and KMAS at the pinned commit
   reproduced their existing cycle assignments **bit-identically** (`cycle`, `cycle_parity`, and
   `area_corr`/`area_corr_dbcal` to zero difference), which also confirms the pinned commit
   reproduces the 2026-06-13 outputs for the anchor-critical surveys.
3. **An upstream cache was overwritten and is not recoverable.** The first (aborted) run refitted
   and rewrote `data/processed/calibration/schwabe_to_rgo_umbra.pkl` (64,854 → 64,834 bytes) before
   backups covered that directory, so the original Schwabe→RGO umbral posterior cache is gone. It is
   git-ignored upstream, so it is not in version control. It is a regenerable cache, not primary
   data, and every snapshot here is internally consistent with the regenerated one — but the
   2026-06-13 outputs can no longer be byte-reproduced. Contrary to the plan's assumption, the run
   is **not** refit-free: the Schwabe stage refits because HEAD's cache spec differs from the code
   that produced the June outputs. It is fast (~100 s for the whole Schwabe survey) and CPU-only.
   `data/processed_backup_2026-10-28/` in the upstream repo holds everything else as it was.
4. **SWPC was added to the upstream run**, which the plan foresaw, but `pipelines/run_all.py`
   restricts `--surveys` to the core five, so a small driver calling `run_survey` directly was used
   instead. Group CSVs were written outside the upstream tree; only
   `data/processed/<survey>/stage4_cycles.parquet` was overwritten there, and those were backed up.
5. **Row-count estimates in the plan were high** for the group tables (459,000 → 375,800 group-days;
   106,000 → 71,642 groups) because the plan counted upstream CSV lines, which include placeholder
   calendar rows. Those rows are in `day_status`, where they belong.
6. **D22 was not foreseen** and is new work: characterized, tested, and documented above.

## Follow-ups for Step 1

1. A documented resolution rule for **D16** before any loader keys on `(date, station)`.
2. Size the state-space truncation K from a robust quantile, not `max(GROUPS)` (**D17**).
3. A config-level validator for the **KMAS single-use rule**; it is documentation-only today.
4. Exclude the 289 colliding KMAS groups from lifetime work, or use RGO and Debrecen (**D22**).
5. ~~Run `ruff check .` and `ruff format --check .`~~ **done 2026-10-29**, both clean.
6. Still outstanding in `refresh_todo.md`: the SILSO SN refresh and the published GN comparison
   series. Both are comparison-only, so neither blocks Step 1's modelling work.

All four remaining follow-ups are named in `CLAUDE.md`'s data-gotchas section and in the Step 1
spec, so a session that reads either will meet them before it can trip on them.

## Reported upstream

`sunspot-homogenization` now carries `docs/findings-2026-10-gn-next-gen.md` and
`validation/repro_gn_next_gen_findings.py`, with a pointer from its `CLAUDE.md`. Five findings, all
still reproducing: the unconditional `cuml` import (which blocks every survey, not just GPU work),
**D22**, the unapplied `max_disk_angle`, the `obs_type` ambiguity, and an invalid build backend.
The repro script runs against that repo alone and exits with the number still reproducing, so it
doubles as a regression gate. Its `CLAUDE.md` implementation-status and repository-layout sections
were also corrected: both predated SOON, SWPC and the `experiments/` tree and claimed the readers
were still stubs.
