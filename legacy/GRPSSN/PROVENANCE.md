# Provenance: legacy/GRPSSN

| Field | Value |
|---|---|
| Original path | `C:\Users\amunoz\OneDrive - SWRI\Documents\AAResearch\Codes\GRPSSN` |
| Remote | https://github.com/amunozj/GRPSSN |
| Branch at snapshot | `master` (branch `GRPRefactor` merged via PR #1) |
| HEAD at snapshot | `5044a3187f4c52a98f0bd0871f75882edbf981d0` (2020-10-02, "Correct group by day data") |
| Snapshot date | 2026-10-28 |
| Working tree state | Dirty: `scripts/Main_SSN_GRP_all_Observers.py`, `scripts/SSN_GRP.py`, `ADF_SSN_06_Using_Only_Group_Distribution.ipynb` modified |

GRPSSN is a fork of ADFSSN that replaces the monthly active-day-fraction (ADF) fingerprint with the **distribution of
daily group counts** (only days with at least `LowLim` groups), compared by EMD between the observer and thresholded,
missing-day-masked reference windows. It smooths the (threshold × shift) distance matrix with a Gaussian filter and keeps
`nBest1` matches by distance, then `nBest2` by similarity in the number of active days.

## What is here

| Path | Content |
|---|---|
| `scripts/` | `SSN_GRP.py` (engine), `SSN_GRP_Plotter.py`, `SSN_Config.py`, `SSN_Input_Data.py` (identical to ADFSSN's), `Main_SSN_GRP_all_Observers.py` (driver). Working-tree versions, including uncommitted edits |
| `functions/detect_peaks.py` | Identical to `legacy/ADFSSN/functions/detect_peaks.py` |
| `notebooks_text/` | The 6 notebooks as py:percent text (mostly the ADFSSN notebook series; `ADF_SSN_06` is the GRP-specific one). `_index.csv` has hashes and status |
| `uncommitted.diff` | `git diff` of tracked non-notebook files at snapshot time |
| `README_original.md` | Original README |

## Deliberately excluded

| Excluded | Size | Reason |
|---|---|---|
| `input_data/` | 60 MB | Byte-identical to ADFSSN's; lives in `data/raw/` |
| `First Test/` (untracked), `testBoth/`, `Braak_Dis_int.png` | ~2.2 GB | Run outputs and figures |
| `.ipynb_checkpoints/`, `.idea/` | small | Clutter |

See `docs/03_legacy/grpssn.md` for parameters and known issues. **Do not edit files in this folder.**
