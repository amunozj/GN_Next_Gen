# ADR-010: A correction release of the same database may supersede its predecessor in place

- Status: Accepted (2026-11-21, PI)
- Context: [ADR-006](ADR-006-data-immutability-and-storage.md) says new or refreshed data go into **new
  subfolders, never overwriting**. That rule was written for *different* datasets arriving side by side
  (v1.22 beside v1.3, the old `reference_areas/` beside `reference_groups_2026-10/`), where both versions
  stay in use: Step 2's legacy baselines must reproduce against the data the legacy code actually used.

  `Revised collection_v1.3_4.xlsx` is a different case. It is a **correction release of the same database**,
  landed by the PI on 2026-11-21, and it fixes most of what `docs/results/gn-v13-feedback-report.md` reported
  upstream: duplicate `(date, station)` keys fall from 217 to 3 (95 conflicts to 1, **D16**), the implausible
  `GROUPS = 93` is gone (**D17**), the string `'NA'` sentinel is gone (**D18**), the flag column is named
  `OBSERVER` (**D19**), and "changed in v1.3" is now an explicit column rather than green cell fill
  (**D20**). Nothing downstream wants `_3`'s numbers *as inputs*; what must stay reproducible is the
  renovation results doc that cites them. Giving `_4` its own folder would duplicate 28 MB and leave two
  nearly identical folders where only one is ever a model input — an invitation to read the wrong one.

- Decision: A **correction release of a dataset already present** may supersede its predecessor **inside the
  existing folder**, provided all three hold:

  1. **Every release's source artifact is retained byte-exact**, each with its own `MANIFEST.csv` row. Here
     both `Revised collection_v1.3_3.xlsx` and `Revised collection_v1.3_4.xlsx` live in
     `data/raw/gn_database_v1.3/`.
  2. **Derived tables track the newest release**, and their `MANIFEST.csv` rows (size, SHA-256, row counts)
     are updated in the same commit. Here `gn_v13_observations.parquet`, `gn_v13_observers.parquet` and
     `gn_v13_changelog.csv` are regenerated from `_4`.
  3. **Any earlier release's derived tables can be regenerated on demand** by a release-aware converter,
     writing to `data/interim/` (git-ignored) and never into `data/raw/`. Here
     `scripts/convert_gn_database_v13.py --release v1.3_3` reproduces `_3`'s tables from the retained
     workbook, so every number in `docs/results/raw-data-renovation.md` and
     `docs/results/gn-v13-feedback-report.md` stays reproducible from retained bytes plus code.

  A release that **changes a dataset's scope** (new surveys, new coverage, a different source) is not a
  correction release and still takes a new folder under ADR-006. The two existing side-by-side pairs
  (v1.22 vs v1.3, `reference_areas/` vs `reference_groups_2026-10/`) are unaffected.

- Alternatives considered:
  - *`_4` in a new folder `gn_database_v1.3_4/`, `_3` retained* — strict ADR-006, and what the Step 1 plan
    first proposed; rejected by the PI as 28 MB of duplication plus two folders a later reader could confuse.
  - *`_4` in a new folder, `_3`'s workbook retired* — saves the duplication but discards the bytes the
    published feedback report rests on.
  - *Keep `_3` as the input and treat `_4` as a future upgrade* — leaves known-bad rows (the `93`, the 95
    conflicting keys) in the model's input for no benefit.

- Consequences:
  - `data/raw/gn_database_v1.3/` now holds two workbooks and one set of derived tables, which the folder
    README states explicitly; `data/raw/MANIFEST.csv` gains a row for `_4` and has three rows updated.
  - The **derived-table checksums in `MANIFEST.csv` are no longer append-only**. `tests/test_data_manifest.py`
    still verifies every row, so a drift fails loudly, but a reviewer reading history will see three rows
    change value in this commit. That is the cost of this ADR and is why condition 1 is non-negotiable.
  - `docs/results/raw-data-renovation.md` keeps `_3`'s numbers and gains a pointer saying they are
    reproducible with `--release v1.3_3`; it is a dated record of what landed in 2026-10 and is not rewritten.
  - Tests that pin v1.3 figures become **release-parametrized**: `_4` is 1,156,368 rows / 1,099,986 with
    `STATION > 0` / 740 stations / 776 observers, `_3` is 1,156,673 / 1,100,144 / 740 / 775.
  - ADR-006's storage triggers are unchanged: `data/raw` grows to ~168 MB, still inside the ~1 GB revisit
    threshold, and the largest file is still v1.22's 46.8 MB CSV.
