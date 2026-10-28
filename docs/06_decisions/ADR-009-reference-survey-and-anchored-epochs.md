# ADR-009: The reference is five per-survey catalogues; two co-equal anchored epochs (Debrecen 21–23, RGO 15–19)

- Status: Accepted (2026-10-28, PI)
- Context: ADR-003 fixed the estimand as the reference-equivalent group count anchored to **one
  reference epoch**, and deferred the choice of reference file and epoch to Step 1. The material
  available then was `data/raw/reference_areas/` — day × `AREA1..AREA26` matrices covering
  1874-04-17 → 2018-06-19 with no group identities, carrying issues D1–D4, D6 and D9, and in
  particular D3, where the two RGO-era files disagree on 12,469 of 37,514 days. That disagreement
  was named as the thing blocking both choices.

  `/d0/amunozj/git_repos/sunspot-homogenization` (MIT, same PI) supersedes those files. It reads
  each survey from its original source, recomputes foreshortening from the raw projected areas,
  converts millionths-of-disk to millionths-of-hemisphere at parse time, and cross-calibrates the
  surveys with a shared-latent hierarchical Bayesian model whose anchor is Debrecen. Its per-survey
  output is per-group and per-day, with group identities, lifetimes, umbral and whole-spot areas,
  and an uncertainty split into a component correlated across a survey (the calibration factor) and
  a per-object independent component. Upstream's stitched composite does not exist
  (`build_composite` is a stub), which suits this project: ADR-003 already wants the reference
  segments kept apart as observers with their own detection parameters.

- Decision:

  1. **The reference** is the frozen snapshot `data/raw/reference_groups_2026-10/`, holding the
     **core five** surveys — Schwabe, Spörer, RGO, Debrecen, KMAS — as per-survey, per-group,
     per-day catalogues spanning 1825-11 → 2026 with no gaps. SOON is excluded entirely. SWPC is
     excluded from the reference and contributes observer *counts* only (see D13 and
     `gn_observer_extension_2026-10/`).
  2. **`area_corr` (native, foreshortening-corrected, MSH) is the model-facing area.** The
     Debrecen-calibrated columns `area_corr_dbcal*` are carried alongside and used as a **prior and
     cross-check only**, never applied as a hard pre-correction. The model fits the inter-survey
     detection differences itself, as ADR-003 and D5 require, so the upstream calibration is never
     applied twice.
  3. **A cycle counts as complete in a survey only if that survey also carries its neighbours.**
     The neighbour test is what catches a cycle clipped at the edge of a survey's coverage, which
     a look at contiguity alone would miss. Measured from the snapshot
     (`survey_cycle_coverage.csv`): Schwabe 8–9, Spörer 9–10, RGO 12–19, Debrecen 21–23, KMAS 20–24.
  4. **Two co-equal anchored epochs**, each a set of whole cycles entirely covered by one survey,
     and **disjoint in cycle number** between the two:
     - **Primary — Debrecen, cycles 21–23** (1974.4 → 2012.6). Debrecen is the identity map in the
       joint posterior (CF ≡ 1), so native and calibrated scales coincide at the anchor and no
       external correction enters the estimand there; it is the deepest survey (29.4% of its
       objects below 10 MSH against 6.2% for KMAS over 1974–2014); and it is the only core survey
       with per-spot data. Cycle 24 is excluded because Debrecen stops 2018-06, mid-cycle.
     - **Secondary — RGO, cycles 15–19** (1912.0 → 1969.2). Complete, and incidentally entirely
       post-1915, matching the legacy MATLAB code's decision to drop earlier RGO (D5). Cycle 20 is
       excluded because RGO stops 1976-12, mid-cycle.
     Steps 3–4 report the series under **both** anchorings as first-class output; the difference
     between them is the anchor-sensitivity result ADR-003 asks for, rather than a one-off check.
  5. `cycle`, `cycle_parity` and the `amp_*` columns are **diagnostic only and never calibration
     inputs**, so cycle phase stays latent as ADR-003 requires.

- Alternatives considered:
  - *Keep the old `reference_areas/` files* — no group identities (so no lifetimes for Step 1
     Check A), no pre-1874 coverage, stops 2018-06, and D3 unresolved.
  - *Ingest only the Debrecen-calibrated areas as a single homogeneous reference* — simpler, but it
     imports an external calibration into the estimand and removes this project's ability to check it.
  - *Ingest only native areas and drop `area_corr_dbcal`* — strictest firewall, but throws away the
     joint posterior's scale factors and their uncertainties, which would have to be re-derived.
  - *All seven upstream surveys in the reference* — SOON and SWPC sit 26% and 43% below Debrecen in
     area, and upstream's own paper carries an unresolved author flag on exactly that offset.
  - *A single anchor (Debrecen only)* — what ADR-003 literally asked for, but Debrecen ends 2018-06
     and covers only three complete cycles; a second anchor in a different century is a stronger
     test of the identifiability claim the paper has to make.
  - *Anchor on partial cycles to gain coverage* — rejected by the PI: an anchor split across two
     surveys mid-cycle confounds the anchor's detection parameters with a survey handover.

- Consequences:
  - D1, D2, D4 and D9 cease to apply to the reference; D3 becomes moot for it and is retained only
     to explain the legacy baselines; D6 is resolved for the reference (now to 2026). D5 stands and
     is now explicit: five surveys, each its own observer. New caveats D10–D15 arrive with the
     upstream pipeline and are recorded in `docs/04_data/known_issues.md`.
  - **Cycle 20 is in neither anchor** — it is precisely the cycle that would have been split, with
     RGO holding 1963.8–1977.0 and Debrecen 1974.0–1980.2. It is complete in **KMAS alone**
     (1960.6–1979.9), so KMAS bridges the two anchors, spanning cycles 20–24 and overlapping
     Debrecen's 21–23. The two anchorings are therefore connected through one well-observed survey
     rather than across a survey handover, which is what makes the Step 3 effective-resistance
     diagnostics between them meaningful instead of degenerate.
  - **KMAS is dual-use and must be used once.** It is a reference survey here and also supplies
     observer counts in `gn_observer_extension_2026-10/`. Its counts are derivable from its areas,
     so any run must declare KMAS's role as reference areas **or** observer counts, never both.
     SWPC has no such constraint. Enforcing this in config is a Step 1 follow-up; until then the
     rule is documented in the extension's README and carried in its `role` column.
  - Neither anchor covers 2018–2026 (Debrecen ends 2018-06, RGO 1976-12), so the modern end of the
     series rests on KMAS and SWPC as calibrated observers under both anchorings.
  - RGO cycles 12–14 are complete by the rule and would extend the secondary anchor back to 1877.
     They are excluded by default on D5 grounds and recorded as a named extension to test in Step 3.
  - The anchors are defined by **cycle number**, not by dates, so they survive small boundary
     movements in a future upstream refit. The snapshot tests fail loudly if a cycle changes
     completeness status. Upstream's cycle assignment is a dependency: it comes from a
     butterfly-diagram fit on the surveys' own positions, with one hand-tuned correction
     (`cycle_boundary_S_offsets: {20: -2.0}` for RGO, because the 19/20 southern minimum is poorly
     resolved). That correction touches cycle 20, which this ADR excludes from both anchors, so it
     does not move either anchor boundary.
  - The old `reference_areas/*.csv` are retained but superseded; they are what Step 2's legacy
     baselines must reproduce against.
