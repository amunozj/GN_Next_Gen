# GRPSSN — group-count-distribution calibration (legacy)

Code: `legacy/GRPSSN/scripts/` (engine `SSN_GRP.py`, driver `Main_SSN_GRP_all_Observers.py`). Fork of ADFSSN; same
reference thresholding, phase-matched sliding windows and missing-day imprinting (see [adfssn.md](adfssn.md)), but the
fingerprint is the **distribution of daily group counts** instead of monthly ADF.

## Differences from ADFSSN (`SSN_GRP.py`)

- `GRPscanningWindowDis` (l. 700–): per interval × threshold × shift, imprints missing days both ways (l. 844–848), keeps
  only days with ≥ `LowLim` groups (l. 851–856), histograms counts over `NGrpsBins` bins (PDF if `density=True`), and
  computes EMD (`emdSw=True`) or mean squared difference.
- Smooths the (threshold × shift) distance surface with `gaussian_filter(sigmaTh/thI, sigmaT/dt)` (l. 941).
- Two-stage selection: `nBest1` best by distance, then `nBest2` closest in number of active days (relative difference);
  best threshold = first of the second set (l. 943–988).
- `GRPsimultaneousEMD(nBest1m, nBest2m)` for a common threshold across intervals.
- Interval validity uses a minimum number of non-zero days (`minObD=30`) instead of ADF months.

## Driver defaults (working tree)

Observers 387–709, skipping 574, 579, 635. `minYrRef=1900, dt=20, phTol=1.0, thS=1, thE=120, thI=5, minObD=30,
maxValInt=2, GssKerPlt=75`; `processObserver(realThTol=0.05, medianSw=True)`; `GRPscanningWindowDis(noOvrlpSw=True,
emdSw=True, LowLim=2, density=True, Dis_Pow=1, NGrpsBins=26, nBest1=100, nBest2=10, sigmaTh=2.0, sigmaT=15)`;
`GRPsimultaneousEMD(nBest1m=100, nBest2m=10)`; `smoothedComparison(gssnKrnl=75)`.

## Information lost / weaknesses

- `LowLim=2` discards 0- and 1-group days, which carry the most information about the acuity threshold at low activity.
- No temporal structure (order of days, persistence of groups).
- Same independence-from-SILSO, missing-at-random and grid-spread-as-uncertainty issues as ADFSSN.
- Reads the same header-less reference file with the same `header=0` pattern.
