# Provenance: legacy/ADFSSN

| Field | Value |
|---|---|
| Original path | `C:\Users\amunoz\OneDrive - SWRI\Documents\AAResearch\Codes\ADFSSN` |
| Remote | https://github.com/amunozj/ADFSSN |
| Branch at snapshot | `Development` |
| HEAD at snapshot | `5898ab8f55f96dee4f6af916cc01b6d85ba6de47` (2020-10-02, "Update Group by date file") |
| Snapshot date | 2026-10-28 |
| Working tree state | Dirty: `scripts/SSN_ADF.py` and `ADF_SSN_06_Using_Only_Group_Distribution.ipynb` modified; many untracked notebooks/scripts |

## What is here

| Path | Content |
|---|---|
| `scripts/` | The refactored ADF pipeline as it exists in the working tree (includes the uncommitted `SSN_ADF.py` edits): `SSN_ADF.py` (engine), `SSN_ADF_Plotter.py`, `SSN_Config.py`, `SSN_Input_Data.py`, `Main_SSN_ADF_all_Observers.py` (driver), `Observer_Category.py`, `FlagFitting.py` (old TF1 classifier experiment) |
| `functions/detect_peaks.py` | Peak finder used for cycle maxima/minima (identical in GRPSSN) |
| `root_scripts/` | Older, pre-refactor class-based versions (`SSN_ADF.py`, `SSN_ADF_Int.py`, `SSN_ADF_Whl.py`) and their driver `ADF_SSN_03_ADF_all_Observers*.py` (untracked in the original repo) |
| `runcsvs/` | Per-observer outputs of past ADF runs (one row per observer; thresholds, R², residuals, observing statistics). File names encode the config: `A`=ADF numerator, `O`/`D`=OBS/DTh denominator, `NB`=NBEST, `PL`/`PH`=PCTLO/PCTHI, `QD`/`AD`=QTADF/ACADF. `Observer_ADF_Int.csv` comes from the older `SSN_ADF_Int` pipeline. Useful as regression targets when porting ADF as a Step 2 baseline |
| `notebooks_text/` | All notebooks exported to py:percent text (code + markdown, outputs stripped). `_index.csv` lists source, status (tracked/untracked/modified), SHA-256, cell counts, and duplicates |
| `uncommitted.diff` | `git diff` of tracked non-notebook files at snapshot time (the `SSN_ADF.py` edits) |
| `README_original.md` | The original repository README (setup instructions for the old conda/pyemd environment) |

## Untracked notebooks (exist only in the original working copy)

`ADF_SSN_03_Optimizing_ADF_time_threshold-Max_Max`, `ADF_SSN_03_Optimizing_ADF_time_threshold-Min_Max-True_ADF`,
`ADF_SSN_04_Interval_Distribution(-Copy1)` (quiet-interval-duration fingerprint), `ADF_SSN_05_Metrics_Comparison-Copy1`,
`ADF_SSN_05_Using_Group_Distribution-Copy1`, `AMJ_JMV_MM_Perspective` (code behind the Muñoz-Jaramillo & Vaquero 2019
Nature Astronomy perspective figures: Maunder Minimum observer network with `DayTol`/`MinOvr`, coverage, butterfly data),
`RGO_Analysis`, `SSN4 program VICTOR`, `SSN4_AL_Optimal_Limits2` (activity-level-dependent "DTh" limits).
Their figures/outputs are **not** preserved here; the original `.ipynb` files remain on the original machine (OneDrive).

## Deliberately excluded

| Excluded | Size | Reason |
|---|---|---|
| `input_data/` | 60 MB | Moved to `data/raw/` (byte-identical, checksummed) |
| `output/`, `outputEMD/`, `outputTestFrag/`, `output_Int/`, `output_int15/`, `output_whol/`, `test/`, `testBoth/` | ~4 GB | Run figures/CSVs; regenerable |
| Root-level `*.png` | ~10 MB | Figures |
| `.ipynb_checkpoints/` | 80 MB | Editor clutter |
| `ADFSSN/`, `PyCharmEnv/`, `legend.py1/` | ~54 MB | Virtual environments (the `ADFSSN/` venv is broken: "No module named encodings") |
| `.idea/`, `.claude/` | small | IDE / local tool settings |
| `research_notes/`, `reports/` | 0.4 MB | Moved to `docs/02_research/` |
| `modernization.md` | 5 KB | Moved to `docs/00_brief/modernization.md` |

## Running it

Legacy code reads `../input_data/...` relative to `scripts/` and needs old library versions (`np.int`, `DataFrame.append`,
`scipy.signal.gaussian`, `pyemd`). See `legacy/README.md` and `environment-legacy.yml`. Known bugs are listed in
`docs/03_legacy/adfssn.md`. **Do not edit files in this folder**; port what you need into `src/`.
