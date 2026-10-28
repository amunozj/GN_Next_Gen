# Provenance: legacy/GN_Series_matlab

| Field | Value |
|---|---|
| Original path | `C:\Users\amunoz\OneDrive - SWRI\Documents\AAResearch\Codes\Dynamo\Code in Exp4\Sunspot Data\GN Series` |
| Version control | **None** (not a git repository); this folder is the only versioned copy |
| Latest `.m` modification | 2018-03-19 (`AA_Network_Plot_GN.m`); most scripts date from Feb–Mar 2017 |
| Snapshot date | 2026-10-28 |

MATLAB code (2017) that treats observers as a graph: nodes are observers, edges join observers with overlapping
observations (within `dtol = 2` days), each edge carries robust (bisquare, frequency-weighted) proportional `a·x` and
linear `a·x + b` fits between their counts, and the edge cost is the relative coefficient uncertainty divided by the
number of overlap days. Calibration paths from early observers to the present are found with Dijkstra / Yen's
k-shortest paths, and calibration factors and uncertainties are propagated multiplicatively along each path.

## What is here

| Path | Role |
|---|---|
| `A1_Matrix_Calculation_weighted_fit.m` | Builds the link matrices (overlap days, unique pairs, Spearman ρ, fits and their uncertainties) → `GN_Links_Mat_Fits_*.mat` |
| `A2_Network_Plot_Random_Plots.m`, `AA_Network_Plot_GN.m` | Network plots and random edge scatter plots |
| `A3_Schwabe_to_present.m`, `A3_Staudach_to_Present*.m`, `A3_Staudach_to_RGO*.m` | Shortest / k-shortest paths between specific start and end observers (e.g. Staudach → Locarno, 1000 paths) |
| `A4_Shortpath_Plot.m`, `A4_Shortpath_Plot_MM.m` | Path plots, including a Maunder-Minimum variant |
| `A5_Composite_Merge.m` | Stub |
| `B1_Minimum_GN.m` | Minimum group count seen by ≥2, 3, 4 observers per day |
| `GN_Fit_Test.m` | Visual check of random edge fits |
| `GN_Reconstruction*.m`, `GN_Reconstruction_MultiPath.m`, `GN_Reconstruction_RobustFit.m` | Composite reconstruction by connected component; MultiPath iteratively removes used nodes, finds 10 paths, and combines them with 1/σ² weights |
| `SSN.m` | Plots of published group-number series (e.g. Hoyt & Schatten 1998) from `SSN_Vers.mat` |
| `data/GNbb2_y.txt` | Yearly GN comparison series, 1610–2015 (inferred from name/coverage: Svalgaard & Schatten 2016 backbone) |
| `data/GNhs_y.txt` | Yearly GN comparison series, 1610–1995 (inferred: Hoyt & Schatten 1998) |
| `data/GNiu_y2.txt` | Yearly GN comparison series, 1749–1995 (inferred: Usoskin et al. 2016 ADF) |
| `data/Vaquero.csv`, `data/Zolotova.csv` | Maunder Minimum yearly GN estimates (Vaquero et al. 2015; Zolotova & Ponyavin 2015) |
| `data/SSN_Vers.mat`, `data/Figure_NatAstron.xlsx` | Published series versions plotted by `SSN.m`; data table behind a Nature Astronomy figure |

The three yearly comparison series' identities are inferred from file names and coverage; confirm before citing them.

## Deliberately excluded (regenerable or superseded)

| Excluded | Size | Produced by / reason |
|---|---|---|
| `GNobservations_JV_V1-12.txt/.mat`, `GNobservers_JV_V1-12.txt`, `GN_Dat.csv`, `GN_Obs.xls` | ~41 MB | GN database **V1-12** (older than the V1.22 snapshot in `data/raw/`; station numbering may differ) |
| `GN_Links_Mat_2d.mat`, `GN_Links_Mat_Fits*.mat` (14 files) | ~45 MB | `A1_Matrix_Calculation_weighted_fit.m` (different robust options / `MinCon`) |
| `GN_181_composites.mat` | 6.6 MB | `GN_Reconstruction*.m` |
| `Vaquero.mat`, `Zolotova.mat` | small | Duplicates of the CSVs |
| `2way/`, `sctrr/`, `Opti/`, `Scw_iter/` | ~82 MB | PNG figure folders |

## External dependencies (not included)

`kShortestPath` (Yen's algorithm, MATLAB File Exchange), `export_fig` and `jbfill` (File Exchange), Curve Fitting Toolbox (`fit`,
`fittype`, `confint`), Statistics Toolbox (`corr`). Scripts also hard-code station numbers from the **V1-12** database
(e.g. `stnd = 445`, `endnd = 590` (Locarno), RGO = 332 with RGO counts before 1915 removed).

See `docs/03_legacy/gn_series_matlab.md` for the method details and known limitations. **Do not edit files in this folder.**
