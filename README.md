# GN_Next_Gen

A next-generation reconstruction of the **group sunspot number (GN), 1610–present**, released as a **posterior
ensemble**, with an accompanying methods paper.

Earlier calibrations link observers either by chains of pairwise regressions (backbone / daisy chain) or by matching each
observer's active-day-fraction distribution to a reference survey. This project replaces both with a joint generative
model: a latent daily population of sunspot groups with birth–death dynamics, observed by every historical observer
through an observer-specific, drifting detection and reporting model, anchored to the RGO/Debrecen/Kislovodsk group-area
catalogue. It uses group lifetimes, overlaps between observers, under-reported spotless days and blanket historical
statements, and gives uncertainties as posterior draws. A sunspot-only series and a proxy-informed series (¹⁴C, ¹⁰Be,
aurorae) are produced separately.

**Status:** clean slate. Knowledge base, data snapshot, legacy code and roadmap are in place; no model code yet. See
[docs/05_roadmap/README.md](docs/05_roadmap/README.md) for step status.

## Start here

| If you want to… | Read |
|---|---|
| Understand the problem and the chosen approach | [docs/01_assessment/assessment.md](docs/01_assessment/assessment.md) (brief: [docs/00_brief/modernization.md](docs/00_brief/modernization.md)) |
| See the plan of work | [docs/05_roadmap/README.md](docs/05_roadmap/README.md) |
| Know what has been decided | [docs/06_decisions/README.md](docs/06_decisions/README.md) |
| Use the data | [docs/04_data/README.md](docs/04_data/README.md) and [data/raw/MANIFEST.csv](data/raw/MANIFEST.csv) |
| Understand the old code | [legacy/README.md](legacy/README.md), [docs/03_legacy/README.md](docs/03_legacy/README.md) |
| Borrow methods from other fields | [docs/02_research/README.md](docs/02_research/README.md) |
| Work with Claude Code | [CLAUDE.md](CLAUDE.md) |

## Quick start

```bash
git clone https://github.com/amunozj/GN_Next_Gen.git
cd GN_Next_Gen
conda env create -f environment.yml        # GPU (Linux/WSL2); or: -f environment-cpu.yml
conda activate gn-next-gen                 # or: gn-next-gen-cpu
python -c "import jax; print(jax.devices())"
pytest                                     # import smoke tests + raw-data checksum checks
```

## Working with Claude Code

Each roadmap step follows the same cycle:

1. `/plan-step N` — Claude reads the step spec and its required reading, asks questions, and writes
   `docs/plans/step-N-plan.md`. No code is written.
2. You review the plan and write "Approved <date>" in its Approval section.
3. `/implement-step N` — Claude implements the plan on branch `step-N-<slug>` with tests.
4. `/verify-step N` — Claude runs the acceptance criteria and writes `docs/results/step-N-verification.md`.

## Layout

```
CLAUDE.md            Guide for Claude Code sessions (rules, workflow, gotchas)
docs/                Brief, assessment, research, legacy docs, data docs, roadmap, decisions, templates, plans, results
data/raw/            Frozen raw data + MANIFEST.csv (immutable)
data/interim/        Regenerated derived data (git-ignored)
legacy/              Read-only snapshots of ADFSSN, GRPSSN, MATLAB GN-series code, SN_Daisy
src/gn_next_gen/     Python package (stub)
tests/               pytest suite
configs/ scripts/ notebooks/   Run configs, result-producing scripts, py:percent notebooks
environment*.yml     Conda environments: GPU (default), CPU, legacy
```

## Data and licensing

Code: MIT (see `LICENSE`). Data in `data/raw/` come from third parties and keep their own terms: SILSO sunspot numbers
(WDC-SILSO, Royal Observatory of Belgium, CC BY-NC 4.0), the revised group-number database (Vaquero et al. 2016, via
SILSO), and reference group areas from the Royal Greenwich Observatory, Debrecen Heliophysical Observatory and Kislovodsk
Mountain Astronomical Station. Cite the original sources (see [docs/04_data/data_dictionary.md](docs/04_data/data_dictionary.md)).
