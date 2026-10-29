# Reuse map: legacy pieces → new code

What to port (re-implement in `src/gn_next_gen/`, vectorized, tested), what to keep only as a baseline, and what to drop.
Line numbers refer to the files in `legacy/`.

## Port (core building blocks)

| Legacy piece | Where | Becomes | Step |
|---|---|---|---|
| Counting reference groups above an area threshold: `np.nansum(np.greater(areas, Thr), axis=1)` with AREA1=NaN → no observation | `ADFSSN/scripts/SSN_ADF.py` l. 837–841 (and l. 116 for the unthresholded count) | Detection layer: emission tables P(y \| N, θ) built by soft-thresholding real reference areas; synthetic observers for the benchmark. **Port the idea, not the shape:** the current reference is long-format, so the count is the number of rows for that (survey, day) with area strictly greater than the threshold, and no-observation comes from `day_status`, not from a NaN in `AREA1`. The row is the unit — do not deduplicate on `group_id` (D22). See ADR-009 and `data/raw/reference_groups_2026-10/README.md` | 1, 2, 4 |
| Missing-day imprinting (observer NaNs onto reference windows and vice versa) | `SSN_ADF.py` l. 867–871; `GRPSSN/scripts/SSN_GRP.py` l. 844–848 | Benchmark generator: transplant real observing calendars onto synthetic truth | 2 |
| Filling an observer's record to a daily grid with NaNs for missing days | `SSN_ADF.py` l. 414–448 | Data loader (long format; do not densify the whole day × observer matrix) | 1 |
| Overlap detection with a day tolerance | `GN_Series_matlab/A1_Matrix_Calculation_weighted_fit.m` l. 71–117; `ADFSSN/notebooks_text/AMJ_JMV_MM_Perspective.py` (cells after "Day adjacency tolerance") | Overlap graph + diagnostics (effective resistance, bridges); pairwise terms of the network calibration | 2, 3 |
| Robust pairwise fits between overlapping observers | `A1_Matrix_Calculation_weighted_fit.m` l. 129–168 | Legacy-style baseline and diagnostic only (the new model replaces linear/proportional links) | 2, 3 |
| ADF / daily-count-distribution fingerprints | `SSN_ADF.py` l. 884–900, 1040–1113; `SSN_GRP.py` l. 851–866 | Summary statistics for posterior-predictive checks; the occupancy form P(active \| N) = 1 − (1 − r)^N in Step 3 | 3, 4 |

## Keep only as baselines (Step 2)

ADF calibration (ADFSSN driver), GRP calibration (GRPSSN driver), shortest-path / multipath composite (MATLAB). Either
run them in `environment-legacy.yml` on benchmark data or port them faithfully and check against
`legacy/ADFSSN/runcsvs/`.

## Drop

| Legacy piece | Why |
|---|---|
| SILSO-based cycle-phase masks and DTh activity limits (`SSN_ADF.py` l. 129–232, 238–328) | Makes calibration depend on SN v2; in the new model cycle phase is latent |
| EMD + `NBEST` grid search with distance-weighted threshold spread | Replaced by likelihood-based posteriors |
| Fixed-length "months" reshaping | Discards day order; the new model is daily |
| Forward-in-time-only directed edges; multiplicative chaining; 1/σ² path combination | Replaced by concurrent global fitting |
| ±2-day "simultaneous" assumption | Replaced by explicit birth-death dynamics (or a lifetime-based decorrelation kernel in Step 3) |
