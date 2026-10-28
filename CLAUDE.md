# GN_Next_Gen — guide for Claude Code sessions

## Mission
Build a new **group sunspot number (GN) series, 1610–present, released as a posterior ensemble**, plus a methods paper
(ADR-008). The core is a joint generative model: a latent daily population of sunspot groups (birth–death dynamics) seen
by ~700 historical observers through observer-specific, drifting detection and reporting models, anchored to a modern
reference survey (RGO/Debrecen/Kislovodsk group areas). Two variants: **sunspot-only** and **proxy-informed**
(¹⁴C/¹⁰Be/aurorae), separated by a data-lineage firewall (ADR-002). PI: Andrés Muñoz-Jaramillo. Stack: Python +
JAX/NumPyro (ADR-001).

## Read first (in this order)
1. `docs/05_roadmap/README.md` — the 6 steps, dependency graph, **status table** (what is done / next).
2. `docs/01_assessment/assessment.md` — why this approach; state of the field; the three approaches; identifiability traps.
3. The spec of the step you are working on: `docs/05_roadmap/step-N-*.md` and its **Required reading** list.
4. `docs/06_decisions/README.md` — accepted decisions (do not relitigate; propose a new ADR instead).
5. As needed: `docs/04_data/` (data dictionary, known issues D1–D9), `docs/03_legacy/` (what old code did, known bugs,
   reuse map), `docs/02_research/` (cross-domain methods; mind the evidence labels), `docs/glossary.md`.

## Repository map
| Path | What |
|---|---|
| `src/gn_next_gen/` | The package (starts as a stub; `paths.py` has repo-relative paths) |
| `tests/` | pytest suite; `test_data_manifest.py` verifies raw-data checksums |
| `data/raw/` | Frozen raw data + `MANIFEST.csv` (**immutable**) |
| `data/interim/` | Regenerated derived data (git-ignored) |
| `legacy/` | Read-only snapshots of ADFSSN, GRPSSN, MATLAB GN-series code, SN_Daisy (notebooks as py:percent text) |
| `docs/` | Brief, assessment, research, legacy docs, data docs, roadmap specs, ADRs, templates, plans, results |
| `.claude/commands/` | `/plan-step`, `/implement-step`, `/verify-step` |
| `environment*.yml`, `pyproject.toml` | Conda environments (GPU default, CPU, legacy) and package config |

## Workflow for each roadmap step
1. `/plan-step N` → read spec + required reading, ask the PI clarifying questions, write `docs/plans/step-N-plan.md`
   from `docs/templates/step-plan.md`. **No code in this phase.**
2. PI reviews and marks the plan "Approved <date>" in its Approval section.
3. `/implement-step N` → branch `step-N-<slug>`, implement with tests, small commits, update the status table.
4. `/verify-step N` → run acceptance criteria and tests, write `docs/results/step-N-verification.md`, update status.
Out-of-band work (bug fixes, data refresh) still gets a short plan in `docs/plans/` if it changes behavior.

## Hard rules
- **Never modify `data/raw/` or `legacy/`.** New raw data → new subfolder + `MANIFEST.csv` row. Port legacy code into
  `src/` and cite the legacy file.
- **Firewall (ADR-002):** the sunspot-only variant may not use isotope/aurora/proxy-derived information anywhere
  (priors, hyperparameters, grand-minimum priors, threshold populations). Record data lineage per run.
- **No SILSO SN in calibration** of the new model (ADR-003); SN is for comparison only. Cycle phase is latent.
- **Estimand:** reference-equivalent group count anchored to one reference epoch (ADR-003).
- **Benchmark before claims** (ADR-007): no method is "better" without Step 2 benchmark scores including coverage.
- References tagged **[nv]** in `docs/02_research/` are unverified; check DOI and claim before citing. Sentences marked
  *(derived)* are not published results. All cost estimates there are unbenchmarked.
- Do not push to GitHub or open PRs unless the PI asks. Commit messages end with the attribution line the harness provides.

## Environment
- `conda env create -f environment.yml` (GPU, CUDA 12 JAX wheels) or `-f environment-cpu.yml`; then
  `conda activate gn-next-gen` (or `gn-next-gen-cpu`). The package is installed editable by the env file.
- **JAX GPU needs Linux or WSL2**; on native Windows use the CPU env.
- `environment-legacy.yml` (`gn-legacy`) is only for running legacy baselines (old NumPy/pandas, pyemd).
- Tests: `pytest` (fast suite); mark slow tests `@pytest.mark.slow` and run them with `pytest -m slow`.
- Lint/format: `ruff check .` and `ruff format .`.

## Data gotchas (details: `docs/04_data/`)
- `GNobservations_JV_V1.22.csv`: 15 `%` comment lines before the header; `STATION=0` = no data; `GROUPS=-1` = missing;
  up to 58 groups/day reported.
- `SC_SP_RG_DB_KM_group_areas_by_day.csv` has **no header** and 17,695 leading all-NaN rows; the `_pre5898ab8` file is
  the same data with a header. Area columns `AREA1..AREA26` (MSH); `AREA1=NaN` = no observation; `AREA1=0` = spotless.
- `DRGO` and `SC_SP_RG` disagree on 12,469 days 1874–1976 (issue D3) — resolve before choosing the reference.
- Raw files mix CRLF and LF line endings; keep bytes unchanged (`.gitattributes`).
- All snapshots are old (GN to 2010, SILSO to 2018, areas to 2018-06); refreshing is a Step 1 task.

## Code conventions
- Python ≥ 3.11, `src/` layout, type hints, NumPy-style docstrings, small pure functions.
- JAX: explicit PRNG keys; `jit` at API boundaries; float64 for log-likelihood accumulation
  (`jax.config.update("jax_enable_x64", True)` in inference entry points); `lax.scan` for long time loops; long-format
  (sparse) observation tables, never a dense day × observer matrix.
- Run configurations in `configs/` (YAML) with seeds; outputs to `outputs/` (git-ignored); commit only small summaries.
- Exploratory notebooks go in `notebooks/` as py:percent text (jupytext), not as the source of truth.
- Every new module gets tests; synthetic-recovery tests for every estimator.
