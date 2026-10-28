# Step 2 — Blind benchmark of synthetic observers

## Goal
A reusable, seeded benchmark that generates synthetic observer networks with known truth, plus a scorer, and the
scores of the legacy methods on it. Every later method is judged on this benchmark (ADR-007).

## Why
There is no credible head-to-head comparison of backbone, ADF and shortest-path methods, and no evidence that any
method's uncertainties are calibrated. Homogenization fields (HOME benchmark), pseudo-proxy experiments in
paleoclimate, and sea-level reconstructions show that blind synthetic benchmarks, sampling-only experiments and
generator ≠ model tests expose biases that real-data comparisons cannot. The closest heliophysics precedent is
Chatzistergos et al. (2025, A&A 699, A157); position against it.

## Inputs
Step 1 loaders and reference tables; real observing calendars and the real overlap graph from the GN database.

## Deliverables
1. **Generator** (`gn_next_gen.benchmark`): truth = reference-equivalent daily counts with real group areas; synthetic
   observers with soft area thresholds, splitting/merging, career drift, real observing calendars transplanted from
   historical observers (including sparse Maunder-Minimum-like calendars), MNAR spotless-day omission, and synthetic
   blanket statements. Scenarios as config files; every dataset reproducible from config + seed.
2. **Truth handling:** truth stored separately from what fitting code may read (API discipline; tests enforce it).
3. **Long-time truth:** the reference covers ~145 years but the network spans ~400. Provide at least two generators:
   (a) reference replay (re-sequencing real reference cycles into a 1610–present truth), (b) a model-based generator
   whose dynamics differ from the Step 4 model, plus (c) grand-minimum scenarios built by randomly deleting whole groups.
4. **Scorer:** errors at daily, monthly, yearly and cycle scales; cycle amplitudes; long-term trend; Maunder-Minimum
   level; interval coverage (e.g. 50% and 90%); recovery of observer parameters; a Hay-style sampling-only experiment
   (apply real sampling to known truth, measure the induced bias).
5. **Baselines:** ADF (ADFSSN), GRP (GRPSSN) and shortest-path / multipath backbone (MATLAB logic), either as validated
   ports in `gn_next_gen.baselines` (check against `legacy/ADFSSN/runcsvs/`) or run in `environment-legacy.yml`.
6. **Results doc** with baseline scores and the benchmark design figure.

## Acceptance criteria (quantify in the plan)
- Determinism: same config + seed → identical datasets (tested).
- Scorer unit-tested on hand-made cases.
- At least the ADF baseline and one backbone/shortest-path baseline scored on every scenario.
- A port, if used, reproduces legacy thresholds on a stated set of observers within a stated tolerance.

## Decisions already made
Benchmark before claims; existing methods as limiting cases (ADR-007); estimand (ADR-003).

## Open questions for the planning session
1. How to place synthetic observers on the real overlap graph across 400 years with a 145-year reference (replay design)?
2. Port vs run legacy code for each baseline?
3. Scenario list and sizes (full network vs sub-networks per era) given compute.
4. Which drift and splitting models to use in the generator so the Step 4 model is misspecified on purpose.

## Required reading
`CLAUDE.md`; `docs/01_assessment/assessment.md`; `docs/06_decisions/ADR-007*`; `docs/03_legacy/*` (especially
`adfssn.md`, `gn_series_matlab.md`, `reuse_map.md`); report sections "Station homogenization supplies benchmarks…",
"Circularity firewalls and pseudo-proxy benchmarks", "Sea level shows the sampling pattern itself biases trends", and
ranked idea 4; note `intercalibration_homogenization.md` (7-step benchmark design).

## Out of scope
New calibration methods (Steps 3–4).
