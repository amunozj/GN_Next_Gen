# Step 3 — Global network calibration (Approach 2)

## Goal
Calibrate all observers **concurrently** over the whole overlap graph, replacing backbone / shortest-path linking;
produce a baseline series with an ensemble, graph diagnostics, and initial observer parameters for Step 4.

## Why
Every field that moved from chained to concurrent calibration reduced errors (SDSS ubercal, multi-link equating,
whole-history ratings). Chained variance grows like the effective resistance between an observer and the reference;
parallel paths reduce it; systematic per-link bias adds coherently. This step is cheap, explainable, and gives the paper
a direct comparison with backbone methods.

## Design (decided at assessment level; detail in the plan)
- Observer parameters θ_o(t) = soft acuity threshold, splitting rate, (optionally) detection noise, with a random-walk or
  spline prior over each career; anchored reference epoch fixed (ADR-003, **as amended by ADR-009:
  two co-equal anchors**, Debrecen cycles 21–23 and RGO cycles 15–19, each fitted and reported).
- Pairwise terms for overlapping observers: P(y_b | y_a, Δt, θ_a, θ_b), derived from thresholded reference areas, with a
  **lifetime-based decorrelation in Δt** (from Step 1) instead of a hard ±2-day window.
- Observers with no overlap connect through a fingerprint likelihood (occupancy form P(active | N) = 1 − (1 − r)^N, or
  the daily count distribution) evaluated against reference windows — no EMD/NBEST grid search.
- Composite-likelihood correction for double counting (e.g. sandwich/curvature adjustment), checked on the benchmark.
- Graph diagnostics: effective resistance from each observer and each year to the reference epoch —
  **computed to each of the two ADR-009 anchors separately**, plus the resistance **between the two
  anchors**, which is the number that says whether the two anchorings are tied together or floating
  independently. That path runs through KMAS: cycle 20 is the only cycle RGO and Debrecen would have
  split, and it is complete in KMAS alone, so KMAS is the bridge and its own resistance matters most.
  Also bridge observers, path-consistency tests, connected components.
- Series: combine calibrated observers with a simple continuous smoother, propagating parameter draws into an ensemble.

## Deliverables
`gn_next_gen.graph`, `gn_next_gen.models.network` (names refined in the plan); benchmark scores vs Step 2 baselines;
real-data run with observer parameter table and diagnostics figures; results doc.

## Acceptance criteria (quantify in the plan)
- Matches or beats legacy baselines on the benchmark at the stated time scales.
- Interval coverage after the composite-likelihood correction within a stated tolerance of nominal.
- Diagnostics computed for every observer; observers in disconnected components flagged.
- Runtime on the full network stated (target: minutes on CPU).

## Open questions for the planning session
1. Parametric form of the pairwise term (exact via thresholded-reference emission tables vs approximations).
2. Drift prior and its sensitivity.
3. Correction method for the composite likelihood.
4. How fingerprint and pairwise terms are weighted when both exist.

## Required reading
`CLAUDE.md`; assessment §4 (Approach 2) and §6; `docs/03_legacy/gn_series_matlab.md`, `reuse_map.md`; report sections
"Ubercal: simultaneous solution…", "Equating quantifies chain drift…", "Whole-history ratings…", "Dawid-Skene
generalizes…", ranked ideas 1 and 7; notes `raters_linking_ratings.md`, `intercalibration_homogenization.md`;
Step 1 and Step 2 results docs.

## Out of scope
Latent daily dynamics of the full population (Step 4).
