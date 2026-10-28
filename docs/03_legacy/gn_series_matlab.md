# GN-series MATLAB — observer network and shortest-path backbones (legacy)

Code: `legacy/GN_Series_matlab/` (2017). Aim: replace human-chosen backbones with an objective path through the observer
overlap graph, with error propagation. Uses the GN database **V1-12** station numbering.

## Pipeline

1. **Link matrices** (`A1_Matrix_Calculation_weighted_fit.m`): drop no-data days and RGO before 1915 (station 332).
   For each observer, expand its observing days by ±`dtol` (= 2) days and find overlapping observers. For each pair:
   Spearman ρ on matched days; collapse matched (count_1, count_2) pairs to unique pairs weighted by frequency; keep the
   edge only if it moves **forward in time** (observer 1 ends earlier and has an earlier mean date) and has ≥ `MinCon`
   unique pairs and ρ ≥ `MinRho` (= 0.6). Fit a proportional model `a·x` and a linear model `a·x + b` (robust bisquare,
   weighted), store values, 95%-CI half-widths and adjusted R². Saves `GN_Links_Mat_Fits_*.mat`.
2. **Edge cost** (`A3_*`, `A4_*`, `AA_*`): DM = |σ_a / a| of the proportional fit (squared in some scripts), replaced by
   the linear fit's relative uncertainty where the linear model is clearly better (R² ratio `Rtol` and significant
   intercept), and in most scripts divided by the number of overlap days.
3. **Paths**: `shortestpath` (Dijkstra) or Yen's `kShortestPath` (up to 1000/10000 paths), e.g. Staudach (445) →
   Locarno (590), Schwabe → present, Staudach → RGO; Maunder-Minimum variants in `A4_Shortpath_Plot_MM.m`.
4. **Composite** (`GN_Reconstruction_MultiPath.m` and siblings): per connected component, find the path between the
   earliest and latest observers, inspect (plot) links that could be bridged directly (their removal is commented
   out), chain calibration factors
   multiplicatively with relative uncertainties added in quadrature
   (σ_total = factor · sqrt(Σ (σ_i/a_i)²)), calibrate each path observer's daily counts, then remove used nodes and repeat
   (10 iterations); combine composites with weights 1/σ².
5. `B1_Minimum_GN.m`: minimum count seen by at least 2/3/4 observers on the same day (a lower-envelope diagnostic).

## Information lost / weaknesses

- ±2 days are treated as simultaneous; groups evolve over that time, and no model of evolution is used.
- Proportional vs linear link chosen per edge; non-linear (threshold) relations are not modeled.
- Error grows with path length (multiplicative chaining); different paths share edges, but the 1/σ² combination treats
  them as independent.
- Only forward-in-time directed edges; information from later observers does not flow back.
- Path choice remains a modeling decision (which start/end nodes, which cost); uncertainty from that choice is not
  quantified.
- Requires `.mat` intermediates that are not included (regenerable with `A1`) and the V1-12 database.

## Related

- `SN_Daisy` (2017) and the `AMJ_JMV_MM_Perspective` notebook implement the same overlap network in Python
  (`DayTol=2`, minimum overlap days `MinOvr=60`, minimum distinct values `minObVal=4`).
- In the cross-domain research, chained calibration error is equivalent to effective resistance on the overlap graph
  (report: "Equating quantifies chain drift…" and "Ubercal…").
