# ADFSSN — active-day-fraction calibration (legacy)

Code: `legacy/ADFSSN/scripts/` (engine `SSN_ADF.py`, driver `Main_SSN_ADF_all_Observers.py`, config `SSN_Config.py`).
Authors of the notebook series: A. Muñoz-Jaramillo, L. Lefèvre, I. Usoskin. Line numbers below refer to
`legacy/ADFSSN/scripts/SSN_ADF.py`.

## Idea

Estimate each observer's acuity threshold (minimum group area in MSH that the observer would see) by finding the area
threshold at which the reference survey, **masked with the observer's own missing days** and taken over a **window of
the same length in the same cycle phase**, produces the same distribution of monthly active-day fractions (ADF) as the
observer. The reference is then thresholded to calibrate the observer.

## Pipeline (`SSN_ADF.py`)

1. `__init__` (l. 28–379): reads reference daily group areas (`REF_Dat`, one column per group, `AREA1..AREA26`),
   SILSO monthly/daily SN; counts groups per day (`REF_Grp`); smooths monthly SN (Gaussian window, 8 months) and finds
   maxima/minima with `detect_peaks`; builds rising/declining "search window" masks of ±`phTol` years around phase
   centers, subsampled every `dt` days (l. 204–232). In `DTh` mode it also fits activity-level limits (from daily SN)
   that switch between ADF, quiet-day fraction and denominators (l. 238–328).
2. `processObserver` (l. 382–629): fills missing days with NaN (l. 414–448), attaches daily SN, splits the record into
   rising/declining intervals bracketed by SILSO extrema, reshapes each interval into fixed-length "months" of `MoLngt`
   days, marks an interval valid if more than `minADFmnth` months have 0 < ADF < 1 and observed fraction ≥ `minObD`,
   and keeps at most `maxValInt` intervals.
3. `ADFscanningWindowEMD` (l. 699–1432): for every valid interval × threshold `thS..thE` × window shift (phase-matched,
   excluding the true overlap when `noOvrlpSw`): threshold reference areas (l. 837–841), **imprint missing days both
   ways** (l. 867–871), count active/observed/quiet days per month, build ADF histograms (and/or daily group-count
   histograms when `disDis` is 'Grp'/'Both'), and compute the distance (EMD with |i−j|^`Dis_Pow` cost via `pyemd`, or L2
   when `emdSw=False`). Keeps the `NBEST` best (threshold, shift) points; threshold = distance-weighted mean, "uncertainty"
   = weighted SD (l. 1160–1198). If the observer overlaps the reference, computes R² and residuals of y = x scatter.
4. `ADFsimultaneousEMD` (l. 1515–): one common threshold across intervals, each interval free to pick among its best
   `NTshifts` shifts (combinatorial, capped by `maxIter`), keeps the `NBEST` combinations.
5. `smoothedComparison` (l. 2248–2390): Gaussian-smoothed (`gssnKrnl` days) observer vs thresholded reference; error
   metrics; "real threshold" minimizing mean difference on the overlap.

## Driver defaults (`Main_SSN_ADF_all_Observers.py`, working tree)

`minYrRef=1900, dt=10, phTol=1, thS=1, thE=90, thI=1, thNPc=20, thIPc=5, MoLngt=20, minObD=0.33, minADFmnth=5,
maxValInt=3`; `ADFscanningWindowEMD(noOvrlpSw=True, fulActSw=True, emdSw=False, Dis_Pow=1)`;
`ADFsimultaneousEMD(NTshifts=10, maxIter=5000)`; `smoothedComparison(gssnKrnl=75)`. Config (`SSN_Config.py`):
`NBEST=50, PCTLO=85, PCTHI=90, QTADF=0.25, ACADF=0.75`, `NUM_TYPE='ADF'`, `DEN_TYPE='OBS'`. Observers 318–418,
skipping 574 and 579. Reference file: `SC_SP_RG_DB_KM_group_areas_by_day.csv`. Output: one CSV row per observer (the
format of `legacy/ADFSSN/runcsvs/`).

## Notebook series (`legacy/ADFSSN/notebooks_text/`)

| Notebook | Finding / purpose |
|---|---|
| 01 Uniform vs Masked Removal | Masking the reference with the observer's missing days beats uniform random removal for comparable ADF distributions; introduces EMD |
| 02 Optimizing ADF time threshold | Joint optimization of window shift and threshold; promising but can lock onto wrong phases; strong yearly observing variation hurts (e.g. Woinoff) |
| 03 Min_Max / Max_Max / True_ADF | Break observers into rising/declining intervals that float independently; simultaneous common-threshold fit |
| 04 Filling missing data / Interval Distribution | Activity-dependent thresholds; quiet-interval-duration distribution as an alternative fingerprint |
| 05 Metrics Comparison / Using Group Distribution | Compare metrics; add daily group-count distributions to ADF |
| 06 Using Only Group Distribution | Group-count distribution only (precursor of GRPSSN) |
| RGO_Analysis | RGO-against-itself validation of the interval method |
| SSN4 program VICTOR, SSN4_AL_Optimal_Limits2 | Low/middle/high activity split for the "DTh" ADF definition |
| AMJ_JMV_MM_Perspective | Figures for Muñoz-Jaramillo & Vaquero 2019 (Nat. Astron.): MM observer network (`DayTol=2`, `MinOvr=60`, `minObVal=4`), activity mask, observational coverage, butterfly data, comparison of published GN series |

Interesting observers named in the notebooks: Spörer 318 (many missing zeros), Leppig 324 (short, many missing days),
Wolfer 338 (long, dense), Moye 412 and Broger 370 (time optimization works), Woinoff 376 (strong yearly variation,
fails), Sykora 366, Guillaume 386, Stempell 389, Koyama 445, National Astron. Obs. Japan 447 (station numbers of V1.22).

## Information lost / weaknesses

- Fixed-length "months" discard day order and the temporal emergence pattern.
- Mask imprinting assumes the observer's missingness is independent of activity: no model of under-reported zeros.
- Uncertainty is the spread of the `NBEST` grid points, not a posterior; thresholds are discrete grid values.
- Overlaps between historical observers are ignored; every observer is linked only to the reference.
- Cycle phases and DTh activity levels come from SILSO SN v2, so the GN calibration is not independent of SN.
- Activity-level dependence of ADF calibration is documented in the literature (Willamo et al. 2018).

## Known bugs (do not fix in legacy/; avoid in ports)

- `SSN_ADF.py:97` reads the reference with `header=0`, but the committed reference file has no header (see
  `docs/04_data/known_issues.md`) → `REF_Dat['YEAR']` KeyError with the current file.
- Inconsistent area-column slicing: `REF_Dat.values[:, 3:shape[1]-2]` (all 26 AREA columns, because ORDINAL and
  FRACYEAR are appended) at l. 116, 838, 2280, 2361 vs `shape[1]-3` (drops AREA26) at l. 252, 1208, 1397, 1809, 1840,
  2161, 2192, 2285, 2300, 2377. Matters only on 26-group days, but scan and scoring use different group sets.
  `legacy/ADFSSN/uncommitted.diff` shows a partial fix.
- The driver's `SKIP_OBSERVERS_WITH_PLOTS` branch (`Main_SSN_ADF_all_Observers.py` l. 371–384) passes
  `SSN_ADF_Config.ADF_TYPE`/`MONTH_TYPE`, which do not exist (the config uses `NUM_TYPE`/`DEN_TYPE`).
- Library-era APIs: `np.int`, `DataFrame.append`, `sp.in1d`, `signal.gaussian` (see `legacy/README.md`).
