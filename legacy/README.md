# Legacy code (read-only reference)

Snapshots of the earlier group-number (GN) calibration codebases, taken 2026-10-28 so this repository is
self-contained. They are **reference material**: never edit them. Port what you need into `src/gn_next_gen/` and cite
the legacy file in a comment or docstring.

| Folder | What it is | Origin | Method doc |
|---|---|---|---|
| [`ADFSSN/`](ADFSSN/PROVENANCE.md) | Active-day-fraction (ADF) calibration of each observer against thresholded, missing-day-masked reference windows (sliding windows by cycle phase, EMD/L2 distance, NBEST matches) | github.com/amunozj/ADFSSN @ `5898ab8` (dirty) | [docs/03_legacy/adfssn.md](../docs/03_legacy/adfssn.md) |
| [`GRPSSN/`](GRPSSN/PROVENANCE.md) | Fork of ADFSSN that matches daily group-count distributions instead of ADF | github.com/amunozj/GRPSSN @ `5044a31` (dirty) | [docs/03_legacy/grpssn.md](../docs/03_legacy/grpssn.md) |
| [`GN_Series_matlab/`](GN_Series_matlab/PROVENANCE.md) | 2017 MATLAB observer-network / backbone code: robust pairwise fits, edge cost from fit uncertainty, (k-)shortest paths, multipath composite | Not under git (OneDrive folder) | [docs/03_legacy/gn_series_matlab.md](../docs/03_legacy/gn_series_matlab.md) |
| [`SN_Daisy/`](SN_Daisy/PROVENANCE.md) | 2017 notebook building the observer overlap network | github.com/amunozj/SN_Daisy @ `cf4a193` | [docs/03_legacy/sn_daisy.md](../docs/03_legacy/sn_daisy.md) |
| `tools/export_notebook_text.ps1` | The PowerShell exporter that produced every `notebooks_text/*.py` | Written for this snapshot | — |

What is worth reusing and what to drop is summarized in [docs/03_legacy/reuse_map.md](../docs/03_legacy/reuse_map.md).

## Notebooks are text, not .ipynb

Every notebook was exported to **py:percent** text (`# %%` code cells, `# %% [markdown]` markdown cells) with outputs and
figures stripped. Each file starts with a header giving the original path, git status, SHA-256 of the original `.ipynb`
and cell counts; `notebooks_text/_index.csv` lists all of them, including duplicates that were not re-exported.
To get a runnable notebook back: `jupytext --to ipynb <file>.py` (outputs are not recoverable). Several notebooks were
**untracked** in their original repos and exist elsewhere only on the original machine's OneDrive.

## Path mapping

Legacy scripts read data relative to their own folders (e.g. `scripts/` reads `../input_data/<file>`). In this repo the
same bytes live in `data/raw/`:

| Legacy path | This repo |
|---|---|
| `input_data/GNobservations_JV_V1.22.csv` | `data/raw/gn_database_v1.22/GNobservations_JV_V1.22.csv` |
| `input_data/GNobservers_JV_V1.22.csv` | `data/raw/gn_database_v1.22/GNobservers_JV_V1.22.csv` |
| `input_data/SC_SP_RG_DB_KM_group_areas_by_day.csv` | `data/raw/reference_areas/SC_SP_RG_DB_KM_group_areas_by_day.csv` (header row missing; see `docs/04_data/known_issues.md`) |
| (git history `5898ab8~1`) same file with header | `data/raw/reference_areas/SC_SP_RG_DB_KM_group_areas_by_day_pre5898ab8.csv` |
| `input_data/DRGO_group_areas_by_day.csv` | `data/raw/reference_areas/DRGO_group_areas_by_day.csv` |
| `input_data/SN_d_tot_V2.0.csv`, `SN_m_tot_V2.0.csv` | `data/raw/silso/` |
| GN database V1-12 (`GNobservations_JV_V1-12.*`) used by MATLAB and SN_Daisy | **Not included** (superseded by V1.22; station numbers may differ) |

## Running legacy code (only to produce baselines)

The legacy Python needs old libraries: `np.int` (removed in NumPy 1.24), `DataFrame.append` (removed in pandas 2.0),
`scipy.signal.gaussian` (now `scipy.signal.windows.gaussian`), SciPy's re-exported NumPy aliases such as `sp.in1d`,
`pyemd`, `astropy`. Use the separate, older environment:

```bash
conda env create -f environment-legacy.yml
conda activate gn-legacy
```

Then copy (do not move) a legacy folder to a scratch location outside the repo, place the needed `data/raw` files under
`input_data/` next to it, and run from there. Note `legacy/ADFSSN/scripts/SSN_ADF.py:97` reads the reference file with `header=0`, which
fails on the headerless current file; supply the `_pre5898ab8` file (renamed) or add the header in the scratch copy. The
MATLAB code needs MATLAB with the Curve Fitting and Statistics toolboxes plus File Exchange `kShortestPath`,
`export_fig` and `jbfill`, and the V1-12 database (not included).

Step 2 of the roadmap decides whether baselines are produced by running legacy code this way or by validated ports;
`ADFSSN/runcsvs/` holds past ADF outputs that a port can be checked against.
