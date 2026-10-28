# Out-of-band plan: renovate both raw inputs (reference survey and GN observation database)

- Spec: none — out-of-band data refresh under the `CLAUDE.md` rule ("Out-of-band work (bug fixes,
  data refresh) still gets a short plan in `docs/plans/` if it changes behavior"). Covers Step 1
  deliverable 3 and the reference half of deliverable 2.
- Author: Claude Code session, 2026-10-28
- Status: **Approved 2026-10-28** (PI, in session)

## Context
`data/raw/reference_areas/` is a 2017–2020-era snapshot inherited from ADFSSN (1874-04-17 →
2018-06-19, day × `AREA1..AREA26` matrices, no group identities) carrying issues D1–D4, D6 and D9.
`data/raw/gn_database_v1.22/` ends 2010-12-31 and carries D7. Both are superseded by material now
available: `/d0/amunozj/git_repos/sunspot-homogenization` (MIT, same PI) for the reference, and the
v1.3 revised collection the PI placed in `data/raw/gn_database_v1.3/` for the observations.

Three workstreams, one branch (`data/raw-renovation-2026-10`):

- **A** — reference survey replaced by a frozen, per-group snapshot of the upstream pipeline's
  core-five output (Schwabe, Spörer, RGO, Debrecen, KMAS), covering 1825–2026 with group identities,
  lifetimes, umbral and whole-spot areas, and correlated/independent uncertainties.
- **B** — observation database moved v1.22 → v1.3: +52,383 observations, +25 observers with data,
  and a 109-entry changelog with DOIs that resolves D7 for 1610–2010.
- **C** — the post-2010 gap closed with KMAS and SWPC daily group counts, plus a feedback report
  for the v1.3 authors on the five issues the audit found.

The full approved plan, with all measured numbers and the reasoning behind each decision, is the
session plan file; this document is the repository's record of it.

## Scope
In scope: the three workstreams above; `MANIFEST.csv` rows; ADR-009; updates to the data dictionary,
known issues, refresh list, the Step 1 spec, `CLAUDE.md` and `README.md`; schema and invariant tests.

Out of scope: `gn_next_gen.io` loaders and `data/interim/` caching (Step 1 deliverable 1); Checks A
and B (deliverables 4–5); emission tables and thresholding; SILSO SN refresh; resolving (as opposed
to characterizing) D16 and D17; enforcing the KMAS single-use rule in code; any change to the
upstream repository.

Both old datasets are kept, not deleted — Step 2's legacy baselines must reproduce against them.

## Design
See ADR-009 for the reference choice and the two anchored epochs, `docs/04_data/data_dictionary.md`
for the schemas, and `docs/04_data/known_issues.md` D10–D21 for the caveats carried over.

Key decisions: native `area_corr` is the model-facing area and `area_corr_dbcal` is a
prior/cross-check only, so the upstream calibration is never applied twice (ADR-003/D5); anchors are
whole solar cycles exclusive to one survey; SOON is excluded entirely and SWPC contributes counts
only, never areas; `obs_type` is renamed at the boundary because it means two different things
upstream (D10).

Firewall (ADR-002): no isotope, aurora, proxy or SILSO input anywhere; `provenance.json` carries an
explicit lineage statement per snapshot. Estimand (ADR-003): `cycle`, `cycle_parity` and `amp_*`
are diagnostic only and never calibration inputs, so cycle phase stays latent.

## Files
New: `scripts/export_reference_groups.py`, `scripts/convert_gn_database_v13.py`,
`scripts/export_observer_extension.py`, `scripts/audit_gn_database_v13.py`;
`configs/reference_groups_export.yaml`; `data/raw/reference_groups_2026-10/`,
`data/raw/gn_observer_extension_2026-10/` and four files in `data/raw/gn_database_v1.3/`;
`docs/06_decisions/ADR-009-reference-survey-and-anchored-epochs.md`;
`docs/results/raw-data-renovation.md`, `docs/results/gn-v13-feedback-report.md`; tests for each
snapshot and each converter.

Modified: `data/raw/MANIFEST.csv` (16 new rows), `src/gn_next_gen/paths.py`,
`docs/04_data/{data_dictionary,known_issues,refresh_todo}.md`,
`docs/05_roadmap/step-1-data-foundation-and-viability.md`, `CLAUDE.md`, `README.md`,
`environment.yml`, `environment-cpu.yml` (add `openpyxl`).

No legacy code is ported; `legacy/` and the existing `data/raw` files are untouched.

## Tests
Manifest checksums for all 16 new files; schema, sentinel, seam and cross-table invariant tests for
each snapshot; converter unit tests on synthetic workbooks and frames; and characterization tests
that pin D16's 217 duplicate keys, D17's two extreme counts, D18's 33 placeholder observers and
D20's 94 change flags so none can drift unnoticed. Fast suite target < ~1 min; whole-file invariants
marked `slow` if they exceed it.

## Acceptance criteria

| Criterion | Measurement | Pass condition |
|---|---|---|
| Raw data immutable | `git diff main -- data/raw legacy` | only new folders and appended `MANIFEST.csv` rows; old checksums unchanged |
| Snapshot reproducible | re-run each export/converter | every derived file byte-identical |
| Reference granularity | uniqueness check | `group_day_observations` unique on (survey, group_id, date_ord) |
| Day-status consistency | cross-table check | `day_status` 1 ⟺ no group rows; 2 ⟺ ≥1; `n_groups` equals the group-row count |
| Anchors well-formed | `survey_cycle_coverage.csv` | Debrecen 21–23 and RGO 15–19 complete and disjoint; cycle 20 in neither, complete in KMAS |
| Coverage | per-survey first/last day | union 1825-11 → 2026 with no gap at any handover |
| v1.3 landed correctly | row/station/observer counts | 1,156,673 rows; 1,100,144 with STATION>0; 740 stations; 775 observers; Feb 29 of 1700/1800/1900 absent |
| Extension sane | threshold monotonicity | counts non-increasing in `threshold_msh` for every (survey, day); no step at 2010/2011 vs the network |
| No SWPC areas in raw | column-name check | no area column for SWPC anywhere under `data/raw` |
| Firewall | lineage statement | no isotope/aurora/proxy/SILSO source in any `provenance.json` |
| Tests | `pytest`, `pytest -m slow`, `ruff` | green |

## Compute and data
Upstream re-run is CPU-only (cached posterior, no refit) over six surveys; SWPC's 11,065 daily files
dominate the runtime. `data/raw` grows 66 MB → ~165 MB, well inside ADR-006's revisit triggers
(~100 MB/file, ~1 GB total). 16 new `MANIFEST.csv` rows.

## Risks
Chiefly: the re-run overwrites upstream's `stage4_cycles.parquet` (backed up first); KMAS being used
in both roles at once (documented rule plus a `role` column, validator deferred to Step 1); the
extension proving discontinuous with the v1.3 network at its overlap (checked explicitly before the
data is relied on); and `openpyxl` fill-reading brittleness (pinned by test to exactly 94). Full
table in the session plan file.

## Open questions for the PI
None outstanding — all seven were answered in session. Four items are recorded for awareness in the
session plan file: upstream's cycle assignment is a dependency, RGO cycles 12–14 are a named anchor
extension, the v1.3 xlsx keeps its upstream filename, and the frozen conversion sits in `data/raw/`
beside the xlsx rather than in `data/interim/`.

## Approval
Approved 2026-10-28 by the PI in session (plan-mode approval of the session plan file).
