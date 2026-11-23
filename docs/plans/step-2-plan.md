# Step 2 plan: Blind benchmark of synthetic observers

- Spec: `docs/05_roadmap/step-2-blind-benchmark.md`
- Author: Claude Code session, 2026-11-22
- Status: **Approved 2026-11-22** (PI, in session) <!-- Draft → Approved (by PI, with date) → Superseded -->

## Context

Step 1 is verified (*Pass with caveats*, 2026-11-22) and `main` now carries the renovated raw data,
which closes the Step-1 finding asking for the renovation branch to be merged. Step 2 is the gate
every later method passes through: **ADR-007 forbids calling any method better than another without
these scores, including coverage.**

What Step 1 hands this step:
- Loaders for every raw input, long-format, checksum-keyed caching, a firewall validator and
  per-run `lineage.json` ([results](../results/step-1-data-foundation.md) §1.2–1.3).
- The group-area law as a frozen Weibull + log-normal with an activity-dependent log-normal
  component (ADR-011), and `fraction_above()` — the detection functional a synthetic observer needs.
- A rotation/visibility kernel (`models.visibility`) that reproduces the 13.6-day disk-passage cliff,
  and the finding that a plain birth–death chain misses the lag-13 trough, the lag-27 recurrence and
  the lifetime distribution **by 3–5×**. The generator must carry rotation explicitly.
- `T ≥ 30 MSH` as the trustworthy threshold band (D14), `K = 41`, and the per-group vs per-group-day
  area-law shift of −0.19 to −0.26 in `μ` (§2.4) — the per-group-day law is what a daily detection
  layer integrates over.

Two items arrive from outside the spec. The PI moved **acquisition of the published GN comparison
series** into this step on 2026-11-21 (`docs/04_data/refresh_todo.md`). And the PI supplied the
Chatzistergos et al. (2025) PDF during this planning session, which converts the spec's
"position against it" from guesswork into a measured comparison — see the next section.

## What Chatzistergos et al. 2025 (A&A 699, A157) actually did

The assessment recorded this paper as "full text not read; which method is unconfirmed"
([assessment](../01_assessment/assessment.md) §2). Read, it settles both the open claim and this
step's positioning.

| | Chatzistergos et al. 2025 | This benchmark |
|---|---|---|
| Truth | RGO, **1900–1976 only** (cycles 14–20) | 1610–2026; RGO, Debrecen, KMAS and Schwabe group-level data |
| Grand minimum | none | S3, by whole-group deletion |
| Observers | 121, each a **hard acuity threshold** 0–300 msd | soft thresholds, drift, breaks, split/merge, seeing noise |
| Drift, breaks, conventions | **explicitly excluded**: "which we do not consider here" (§2.1) | all present, some documented as metadata, some not |
| Spotless days | reported as-is | MNAR under-reporting; blanket statements |
| Group lifetimes | none — a per-day threshold | real `obs_duration` distribution, through a rotation kernel |
| Sampling | days **deleted at random** to hit 50 %…1 % overlap | real historical calendars transplanted from the v1.3_4 record |
| Overlap structure | each observer compared **pairwise to RGO** | the real overlap graph, with its bottlenecks |
| Daisy-chaining | **not evaluated** (their §4) | the backbone composite is a scored baseline |
| Metrics | RMS, mean, relative, max-absolute difference | those **plus** interval coverage, CRPS, PIT, trend, MM level |
| Uncertainty calibration | **none** | the load-bearing metric (ADR-007) |
| Blind / generator ≠ model | no; the generator is the hard threshold the methods assume | generator deliberately misspecified (below) |

Their closing sentence names our gap for us:

> "The effect of the daisy-chaining process was not evaluated in this study, but it would be
> important to address it in forthcoming studies."

**The assessment's unverified claim resolves to UEA21**: it "systematically overestimates the acuity
of the observer, thus in the end also amplifies the group numbers" (their §3.2). Their headline is
that the linear methods (HoSc98, SvSc16, VHS24) **amplify strong cycles and suppress weak ones**, a
secular bias, while CEA17 is most reliable. Two of their results become validation targets for our
scorer (acceptance criteria below).

One flagged gap closes incidentally. `docs/02_research/notes/raters_linking_ratings.md:41,72` records
the conditioning direction of the CEA17 probability-distribution matrix as unresolved. Their §2.2.3
states it: a column is indexed by the **secondary** observer's count, and each column is a normalised
histogram (PMF) over the **reference** observer's count on those days. Record this in the note.

## PI decisions taken in session (2026-11-22)

| # | Decision | Consequence |
|---|---|---|
| 1 | **Replay is hybrid** — 1825–2026 uses the real reference as-is; pre-1825 is built from whole resequenced reference cycles with amplitude set by whole-group deletion | Spec OQ1. Amplitudes come from a synthetic quasi-periodic model, never SILSO or a proxy |
| 2 | **Port every baseline; run no legacy code** | Spec OQ2. No `gn-legacy` environment, no `pyemd`; 1-D EMD is implemented exactly as the L1 distance between CDFs |
| 3 | **ADF derives cycle phase from the observations** on the benchmark | New. Legacy ADF takes phase from SILSO v2, which a synthetic world does not have. Documented deviation from legacy ADF |
| 4 | **Six scenarios × 10 realizations** (S1 at 20) | Spec OQ3 |
| 5 | **Seven baselines**: HoSc98, SvSc16, VHS24, CEA17, UEA21 (ADF), GRP, backbone | New. Makes this benchmark a strict superset of A157's and gives two validation targets |
| 6 | **Generator differs from the Step 4 model in four ways**: lifetime law, emergence clustering, drift form, correlated observer errors | Spec OQ4 |
| 7 | **Drop the `runcsvs/` reproduction check** | Spec acceptance criterion 4 is deliberately not met — recorded as a deviation, with a replacement criterion (below) |
| 8 | **Acquire all five published GN series** into a new raw folder, `role: comparison` | PI, 2026-11-21, carried out here |
| 9 | **Ten named case-study observers get their own panels**, chosen to expose strengths, weaknesses and failure modes | New figures `f25`–`f26` and a committed per-observer CSV; criterion A12 |
| 10 | **The session never launches a run over ~15 minutes** — Claude Code dies on a disconnect | The session delivers code, `--quick` tests and command lines; the PI launches the grid detached. Every long script is resumable and reports progress |

### What decision 7 gives up, and what replaces it

`legacy/ADFSSN/runcsvs/A_O_NB1_Observer_ADF.csv` holds 208 observers' integer `AvThreshold`, but it
was **not produced by the committed driver settings**. Recoverable from the files: threshold grid
0–99, `MoLngt = 15`, `maxValInt ≥ 5`, `NBEST = 1`, observers to 632. Unknown: `dt`, `phTol`,
`minObD`, `minADFmnth`, `emdSw`, `fulActSw`, `Dis_Pow`, `minYrRef`. Worse, `legacy/ADFSSN/uncommitted.diff`
is **already applied in the snapshot**, so the snapshot is not the code that made those numbers.
Chasing them is a timeboxed search over eight undocumented parameters against a moving target.

The replacement is stronger than it sounds, because it is external and quantitative: **the ports must
reproduce A157's published behaviour on A157's own setup** (RGO ≥ 1900, hard acuity thresholds, full
overlap). That is criterion A7 below. The legacy files stay in the repo and the recovered parameters
are written into `docs/03_legacy/adfssn.md` as a finding, so a later session can pick the check up.

## Scope

In scope:
1. `gn_next_gen.benchmark` — truth generators, synthetic observers, calendar/graph transplant,
   scenarios, sealed truth store.
2. `gn_next_gen.baselines` — seven ported calibration methods.
3. `gn_next_gen.graph` — overlap graph construction and the diagnostics Step 2 needs
   (components, bridges, effective resistance). Step 3 extends it; Step 2 builds the minimum.
4. Loaders for the two retained superseded folders (`reference_areas/`, `gn_database_v1.22/`).
5. The scorer, including coverage, CRPS, PIT, trend, cycle amplitude, MM level, parameter recovery,
   break-detection skill and two null floors.
6. The Hay-style sampling-only experiment.
7. Published GN series acquisition into `data/raw/gn_published_series_2026-11/`.
8. Results doc with the benchmark design figure and all baseline scores.

Out of scope (spec's list): new calibration methods — Steps 3–4. Added: any change to Step 4's model;
running legacy code (decision 2); the `runcsvs/` reproduction (decision 7); proxies (Step 6).

## Design

### Module layout

| Module | What |
|---|---|
| `benchmark/truth.py` | `Truth` dataclass (daily counts, per-group areas and lifetimes, activity series, anchoring); the **only** module that reads a sealed truth file |
| `benchmark/generators/replay.py` | Hybrid replay (decision 1) |
| `benchmark/generators/synthetic.py` | Model-based generator, deliberately not Step 4's model |
| `benchmark/generators/grand_minimum.py` | Whole-group deletion, 10–100× thinning |
| `benchmark/observers.py` | Synthetic observer population: soft threshold, drift, breaks, split/merge, seeing, MNAR, blanket statements, copying |
| `benchmark/calendars.py` | Real observing calendars transplanted from v1.3_4 + extension; promotes `transplant_calendar` out of `scripts/check_a_dynamics.py` |
| `benchmark/scenarios.py` | Config-driven assembly; one scenario = (generator, observer population, calendar set, anchoring, seed) |
| `benchmark/store.py` | Writes `observations.parquet` (open) and `truth.parquet` (sealed) to separate roots; own content-hash keying, because `io.cache.manifest_checksum` refuses unmanifested inputs by design |
| `benchmark/scoring.py` | The scorer |
| `baselines/{hosc98,svsc16,vhs24,cea17,uea21_adf,grp,backbone}.py` | The seven ports |
| `graph/overlap.py` | Day-tolerant overlap graph, components, bridges, effective resistance |
| `io/legacy_inputs.py` | `reference_areas/` (day × `AREA1..AREA26`, **no header** — D1) and `gn_database_v1.22/` readers |
| `viz/benchmark.py` | Figures f13–f24 |

### Truth generators

**Hybrid replay (`replay.py`).** 1825–2026 is the real reference, used as-is: real `area_corr`, real
`obs_duration`, real daily structure. Pre-1825 is assembled from **whole donor cycles** drawn from the
ADR-009 cycle inventory, each rescaled in amplitude by deleting whole groups at random.

Donor pool: **RGO (12–19), Debrecen (21–23), KMAS (20–24, D22-collisions dropped) and Schwabe (8–9)**.
Schwabe is included — it has real multi-day groups (median `obs_duration` 6, max 17), and the truth
generator needs lifetimes and areas, not spotless days, so **D23 does not bite here as it did in Step 1**.
**Spörer is excluded**: every one of its 5,421 groups has `obs_duration ≡ 1`, so it supplies no
lifetimes (D23).

Schwabe's area calibration is umbral and its absolute scale is suspect, so **each realization draws
all donor cycles from a single survey by default**, which makes an absolute scale offset harmless —
the truth defines its own reference-equivalent scale (ADR-003). A mixed-donor variant is available as
a sensitivity, applying the per-survey ADR-011 log-normal `μ` offset to put donors on the anchor
survey's scale and recording the offset applied.

Amplitudes come from a **synthetic quasi-periodic Gaussian process with jittered period**, seeded from
the run config. **No SILSO, no proxy, no published series enters the amplitude sequence** — the
firewall holds and the truth stays genuinely unknown to every method under test.

**Model-based (`synthetic.py`).** A marked point process: emergences clustered in active longitudes,
log-normal intrinsic lifetimes, areas drawn from the ADR-011 law at the scenario's activity level,
pushed through `models.visibility`'s rotation kernel. Extends `simulate_marked_process` with an
empirical-lifetime branch and a clustered-emergence branch.

**Grand minimum (`grand_minimum.py`).** Whole-group deletion at a rate rising to 10–100× thinning over
a stated interval, plus amplitude suppression. **Stated limitation:** thinned modern groups are not a
real regime change (Smerdon's caution; `paleoclimate_historical_evidence.md` records that *no source
was found* for generating grand-minimum synthetic truths). The MM scores are conditional on that
assumption and the results doc must say so.

### Synthetic observers, and the deliberate misspecification

| Layer | Generator (Step 2) | Step 4's model | Misspecified? |
|---|---|---|---|
| Intrinsic lifetime | **log-normal**, through the rotation kernel | geometric survival × rotation kernel | **yes** |
| Emergence | **clustered in active longitudes / nests** | Poisson births, `λ_t` | **yes** |
| Detection | logistic in log-area (soft threshold, two parameters) + per-day multiplicative seeing jitter | soft threshold, smooth `q_o` | **yes** (jitter) |
| Career drift | **abrupt breaks + random walk**; some breaks exposed as metadata, some not | smooth random walk / spline prior | **yes** |
| Observer errors | **a stated fraction copy a neighbour** (shared compiler) | independent observers | **yes** |
| Splitting / merging | per-group Bernoulli split and merge | splitting rate | partly |
| Spotless reporting | MNAR — omission probability depends on activity | MNAR reporting layer | matched by design |
| Blanket statements | literal/rhetorical mixture with unknown effort | same form | matched by design |

**Stated limitation of decision 6:** without a matched-model control arm, the measured overconfidence
is a single number against the real generator, not a decomposition into model error and data error.
It is a lower bound on nothing and an upper bound on nothing — it is the overconfidence under *this*
generator. If the paper needs the decomposition, a matched-model arm is a cheap later addition.

### Calendars, the overlap graph, and the estimand

Calendars are transplanted per observer from the real record: the set of `date_ord` for a station in
`io.load_observations(release="v1.3_4", include_extension=True)`, including the sparse
Maunder-era calendars, which is where the graph bottlenecks live. The overlap graph is built with a
day tolerance (legacy `DayTol = 2`; `reuse_map.md` row 4), and Step 2 reports components, bridges and
effective resistance so the scorer can relate each baseline's error to graph distance — identifiability
trap 3 from assessment §6.

Every scenario **declares its anchoring** (ADR-009: Debrecen cycles 21–23 primary, RGO 15–19 secondary).
The scorer compares reference-equivalent counts in the declared anchor's units; it never scores an
absolute count, because counts alone cannot identify absolute scale (trap 1).

### Truth handling, and how "blind" is meant here

`truth.parquet` goes to `outputs/benchmark/<scenario>/<seed>/sealed/`; `observations.parquet` to
`.../open/`. Only `benchmark.truth` and `benchmark.scoring` may read the sealed root, enforced by an
**import-audit test** modelled on the Step-1 firewall audit: no module under `baselines/`, `models/`
or `inference/` may import `benchmark.truth` or name the sealed path.

**Stated honestly:** this is not blind in the HOME sense — one developer writes both sides. Blindness
is enforced by API and audited by test, and the baselines are frozen by commit before truth is
generated for the scoring run. The results doc says this plainly rather than claiming a blind trial.

### The seven baselines

| Baseline | Form | Cost |
|---|---|---|
| HoSc98 | `k = ΣG*/ΣG` over common spotted days; used only if `0.6 < k < 1.4` | trivial |
| SvSc16 | `k = Y*/Y` on annual means, forced through the origin, **not** restricted to common days | trivial |
| VHS24 | `k = σ*/σ` with an additive intercept `b` | trivial |
| CEA17 | PMF conversion matrix; columns indexed by the secondary's count, each a PMF over the reference's; sparse columns filled by a 1000-iteration MC bootstrap fitting `⟨G*|G⟩ − G = R₀ − B·e^{−aG}`; the quality criteria of their §2.2.3 (≥ 20 overlapping days, > 4 years for long records, χ²/dof ≤ 6) | moderate |
| UEA21 (ADF) | acuity from the cumulative active-day distribution against a precomputed reference library, then that acuity's PMF matrix. **Cycle phase from a smoothed composite of raw observer counts, not SILSO** (decision 3) | moderate |
| GRP | daily group-count distribution fingerprint; `LowLim = 2`, two-stage `nBest1/nBest2` selection, smoothed distance surface (`gn_series_matlab.md`, `grpssn.md`) | moderate |
| Backbone | day-tolerant overlap graph, robust weighted pairwise proportional and linear fits, edge cost `|σ_a/a| / overlap_days`, k-shortest paths, multiplicative chaining with relative uncertainties in quadrature, `1/σ²` composite | moderate |

Ports cite their legacy source file and line range, per the hard rules. Three legacy bugs are **not**
reproduced (no reproduction target any more, decision 7) and each is recorded in the port's docstring:
the inconsistent `AREA26` slicing; `mDOO` computed from the last interval only rather than the
accumulated arrays; and the `1e16` sentinel that distorts the `NBEST` weights. The MATLAB composite's
double scaling by `LnkVlP` (`GN_Reconstruction_MultiPath.m` l. 668–669, which inflates the propagated
uncertainty by one factor of `a`) is also **not** reproduced, and is reported upstream as a finding.

Station-number caution for the backbone port: the MATLAB scripts use **V1-12** numbering and V1-12 is
not in the repo. `stnd = 445`/`endnd = 590` are Koyama and Janssens in V1.22; the intended nodes are
**466 (Staudach)** and **614 (Locarno)**. The port resolves endpoints **by observer name**, never by
a hard-coded number.

### The scorer

Computed per (scenario, realization, baseline):

- **Centred RMSE** at daily, monthly, yearly and cycle-mean scales.
- **Cycle amplitude error** and **cycle-to-cycle ratio error**.
- **Secular trend error** over 1610–present, 1700–present and 1850–present.
- **Maunder-Minimum mean level error** (S3).
- **CRPS** on yearly GN; **PIT histogram** and its KS distance from uniform.
- **Interval coverage at 50 %, 68 %, 90 % and 95 %**, stratified by era. The spec's **50 % and 90 %**
  are the headline; the other two are free and reconcile the three different conventions in the
  research notes.
- **Observer-parameter recovery**: bias and RMSE of recovered acuity threshold against truth.
- **Break-detection skill** (S4): POD, FAR and Heidke skill score against known break times.
- **Two null floors**: a random-segmentation null (Lindau & Venema 2018 — at realistic SNR a random
  segmentation explains about half the break variance) and a return-the-prior null, which TARP-style
  reasoning says a coverage test must be able to reject.

Reuses `stats.discrepancy` (`autocorrelation`, `count_pmf`, `ks_distance`, `block_bootstrap`,
`run_lengths`); monthly/yearly/cycle aggregation, coverage, CRPS and PIT are new.

### Scenarios

| | Scenario | Realizations | Span |
|---|---|---|---|
| S1 | Modern sub-network (development + repeated calibration checks) | 20 | 1874–2026 |
| S2 | Full network | 10 | 1610–2026 |
| S3 | Grand minimum inserted by whole-group deletion | 10 | 1610–2026 |
| S4 | Synchronous-break stress: era-wide acuity improvement at a stated date | 10 | 1610–2026 |
| S5 | Hay sampling-only: real calendars on known truth, **zero detection error** | 1 + bootstrap over 25–600 observers | 1610–2026 |
| S6 | Structural ensemble: perturbed free choices of each baseline, applied to S2 | 5 perturbations per knob | 1610–2026 |

### Firewall and estimand

`variant: sunspot_only`. SILSO is not needed anywhere in this step — decision 3 removed ADF's only
use of it — so it appears solely as `role: comparison`, as do the five published GN series. The
replay's amplitude model is synthetic and seeded. Every run writes `lineage.json` via
`config.write_lineage`, as `tests/test_step1_scripts.py::test_every_run_records_its_lineage_and_the_firewall`
already requires.

## Files

New under `src/gn_next_gen/`: `benchmark/` (`truth.py`, `store.py`, `observers.py`, `calendars.py`,
`scenarios.py`, `scoring.py`, `generators/{replay,synthetic,grand_minimum}.py`); `baselines/`
(seven modules + `__init__.py`); `graph/overlap.py`; `io/legacy_inputs.py`; `viz/benchmark.py`.

Modified: `models/visibility.py` (empirical-lifetime and clustered-emergence branches);
`paths.py` (the new raw folder); `viz/style.py` only if a new role slot is needed;
**`tests/test_viz.py::test_every_committed_figure_has_both_modes`**, which is hard-coded to `step-1`
and needs generalizing to `step-*` (one line); `docs/03_legacy/adfssn.md` (the recovered `runcsvs/`
parameters and the four undocumented bugs); `docs/02_research/notes/raters_linking_ratings.md` (the
CEA17 conditioning direction, now resolved).

New scripts: `fetch_published_gn_series.py`, `generate_benchmark.py`, `run_baselines.py`,
`score_benchmark.py`, `sampling_only_experiment.py`, `validate_against_a157.py`,
`make_step2_figures.py`. New configs: `configs/step2_benchmark.yaml` (scenario grid, seeds),
`configs/step2_baselines.yaml` (each baseline's free choices and the S6 perturbation grid).

`data/raw/` is touched by **one addition**: `gn_published_series_2026-11/` with `MANIFEST.csv` rows.
`legacy/` is untouched.

### Figures (f13–f26, continuing Step 1's numbering)

`f13` benchmark design schematic · `f14` replay construction and donor-cycle provenance ·
`f15` overlap graph through time, with effective resistance to each anchor · `f16` synthetic observer
population: thresholds, drift realizations, break times · `f17` calendar transplant, real vs synthetic
coverage · `f18` **baseline scores at four time scales** (headline) · `f19` cycle-amplitude and
secular-trend error per baseline · `f20` **nominal vs empirical interval coverage** at 50/68/90/95 ·
`f21` Hay sampling-only: induced trend bias vs observer count · `f22` structural-ensemble spread beside
parametric uncertainty · `f23` observer-parameter recovery, recovered vs true acuity · `f24` the A157
validation: our ports against their published Fig. 3/4 behaviour · **`f25`–`f26` the ten case-study
observers** (below).

### The ten case-study observers (`f25`, `f26`)

Aggregate scores say which method wins; they do not say *why*, and they hide the observers that break
a method. Ten synthetic observers are therefore fixed by name and seed in
`configs/step2_observers_case_study.yaml`, drawn from S2, and each gets a panel: truth, the observer's
raw counts, and every baseline's calibrated series, with the **true** acuity marked and each method's
**recovered** acuity annotated. They are chosen to span the failure modes, not sampled at random.

| | Observer | What it is | What it should expose |
|---|---|---|---|
| 1 | `clean-modern` | dense calendar, low threshold (~10 MSH), no drift, no breaks | The best case — every method should succeed; a method that fails here is broken |
| 2 | `sparse-maunder` | an MM-era calendar transplanted verbatim: few days a year, mostly spotless | Where overlap-based methods have nothing to work with; the graph-bottleneck case |
| 3 | `break-undocumented` | abrupt acuity improvement mid-career, **no** metadata | PHA-style attribution with nothing to key on; the drift-vs-break confusion |
| 4 | `break-documented` | the same break, exposed as metadata | Contrast with 3 — how much the documentation is worth |
| 5 | `slow-drift` | random-walk acuity over a long career | Identifiability trap 2: era-wide drift against real solar change |
| 6 | `poor-instrument` | high threshold (~100 MSH), otherwise clean | A157 finds errors grow with acuity for *every* method; this is where we should see it |
| 7 | `leaf-node` | exactly one overlap partner | "Leaf nodes in the overlap graph cannot be attributed by counting" — where chained error is unbounded |
| 8 | `copier` | reports a near-copy of a neighbour (shared compiler) | Correlated observer error, which no method under test models; should look falsely confident |
| 9 | `zero-omitter` | omits spotless days at a rate rising as activity falls (MNAR) | Breaks ADF's missing-at-random assumption directly, at the low-activity end that decides the MM |
| 10 | `blanket-statements` | most of the record is yearly rhetorical statements, not daily counts | The Hoyt & Schatten reading problem, as a generator rather than an argument |

`f25` carries 1–5 (well-behaved through drift), `f26` carries 6–10 (the hard cases), five panels each
so both stay legible and under the 400 kB limit. The per-observer numbers — true acuity, each method's
recovered acuity, calibrated-series RMSE and interval coverage — go to a committed
`docs/results/data/step2_case_study_observers.csv`, so every number in the captions traces to a file.

## Tests

Fast suite (target: the whole suite stays under ~90 s; it is 41 s today):

- **Determinism**: same config + seed → identical `observations.parquet` and `truth.parquet` by
  content hash, for every scenario at `--quick` size.
- **Truth isolation**: import audit — no module under `baselines/`, `models/` or `inference/`
  imports `benchmark.truth` or names the sealed path.
- **Scorer on hand-made cases**: a perfect reconstruction scores 0 centred RMSE and nominal coverage;
  a constant-offset reconstruction recovers the offset exactly; a Gaussian ensemble with known
  variance gives 50/68/90/95 coverage within Monte-Carlo error; CRPS of a point forecast equals MAE;
  PIT of a correctly specified ensemble is uniform.
- **Null floors**: the return-the-prior null fails the coverage test; the random-segmentation null
  scores between the prior null and the real baselines.
- **Baselines on analytically known cases**: HoSc98/SvSc16/VHS24 recover their `k` exactly on
  constructed data; CEA17's matrix is column-stochastic and its bootstrap fill is reproducible;
  the backbone recovers a known chain of proportional factors on a hand-built graph.
- **Graph**: effective resistance on a path graph equals the sum of edge resistances; bridges on a
  hand-built graph; components match `networkx`.
- **Legacy loaders**: `reference_areas/` reads with supplied column names (D1), `AREA1 = NaN` means
  no observation, and row counts match `MANIFEST.csv`; v1.22 row counts and station count match.
- Every figure function returns a figure in both modes; each script runs under `--quick`.

`@pytest.mark.slow`: synthetic recovery of each baseline's observer parameters on S1; the A157
validation run; one full S2 realization end to end.

## Acceptance criteria

| Spec criterion | Measurement | Pass condition | Figure |
|---|---|---|---|
| Determinism: same config + seed → identical datasets (tested) | SHA-256 of `observations.parquet` and `truth.parquet` across two runs, every scenario | Byte-identical for all 6 scenarios | — |
| Scorer unit-tested on hand-made cases | The scorer test list above | All pass; coverage of a known Gaussian ensemble within ±2 % of nominal at 50/68/90/95 | `f20` |
| At least ADF and one backbone baseline scored on every scenario | Score table rows | **All 7 baselines × 6 scenarios = 42 cells populated**, none missing | `f18` |
| A port reproduces legacy thresholds within a stated tolerance | **Not met — PI decision 7.** Replaced by A7 below | Deviation recorded; recovered `runcsvs/` parameters written into `docs/03_legacy/adfssn.md` | — |
| **A1** Truth is sealed from fitting code | Import audit over `src/` | Zero imports of `benchmark.truth` outside `benchmark/`; test green | — |
| **A2** Truth carries real areas *and* real lifetimes | Generated truth vs reference | Per-group area and `obs_duration` distributions match the donor survey within KS 0.05; the 289 D22-colliding KMAS ids absent | `f14` |
| **A3** Real calendars and the real graph are used | Synthetic vs real observing-day counts per station per year | Exact match of observing-day sets for every transplanted station; graph component count and bridge list reported | `f15`, `f17` |
| **A4** Generator ≠ Step 4's model in four stated ways | The misspecification table, each with a test | All four present and each individually switchable for sensitivity | `f16` |
| **A5** Interval coverage reported for every baseline | Empirical coverage at 50/68/90/95, stratified by era | Reported for all 42 cells; **this is a measurement, not a bar the baselines must clear** | `f20` |
| **A6** Hay sampling-only bias measured | Induced trend bias from calendars alone, bootstrapped over 25–600 observers | Bias and its bootstrap interval reported; the curve's level-off stated | `f21` |
| **A7** Ports reproduce A157's published behaviour on A157's setup (replaces the dropped check) | RGO ≥ 1900, hard thresholds, full overlap, annual means | **Sign agreement on all five methods** in their Fig. 3b mean-difference panel: HoSc98 and SvSc16 mean difference decreasing monotonically with acuity; CEA17, UEA21, VHS24 near zero. Their exclusion thresholds reproduce: HoSc98 at `A ≥ 31` msd, CEA17 at `A ≥ 87` msd | `f24` |
| **A8** A157's headline secular bias reproduces on the real graph | Cycle-amplitude error vs cycle strength, S2 | Linear methods amplify strong and suppress weak cycles; non-linear methods do not. Sign and ordering match their §4 | `f19` |
| **A9** Published GN series acquired | New raw folder + `MANIFEST.csv` rows + checksum test | All five present, `role: comparison`, `tests/test_data_manifest.py` green; the three `legacy/GN_Series_matlab/data/` series cross-check within stated tolerance | — |
| **A10** Structural ensemble reported beside parametric uncertainty | S6 spread vs posterior/parametric width | Both reported per baseline; stated whether the ensemble is wider | `f22` |
| **A12** Ten case-study observers, shown individually | Each of the ten named observers gets a panel with truth, raw counts and all seven calibrated series; true and recovered acuity annotated | All ten present and traced to `step2_case_study_observers.csv`. At least `sparse-maunder`, `leaf-node`, `copier` and `zero-omitter` must each show **a visible failure in at least one baseline** — if every method handles all ten, the generator is too tidy and the results doc says so | `f25`, `f26` |
| **A13** No long job run in session | Session log and `outputs/step2/logs/` | Every run exceeding ~15 min was launched by the PI from the documented commands; the session ran only `--quick`/S1 work (PI, 2026-11-22) | — |
| **A11** Figure-first results doc | f13–f26 committed in both modes, embedded and captioned | Every figure ≤ 400 kB, both modes present, each embedded as plain Markdown with a `**Figure fNN**` caption; `tests/test_viz.py` green after the `step-*` generalization | all |

## Compute and data

The generator is cheap; the baselines are the cost. One world is ~1.1 M observation rows (~35 MB
parquet). 71 worlds across the grid ≈ 2.5 GB in git-ignored `outputs/benchmark/`.

Per world: the three linear baselines are O(n) — milliseconds. CEA17's MC bootstrap is the heaviest;
vectorized least-squares over 1000 draws gives ~0.1 s/observer → ~75 s. UEA21's reference library is
built once per scenario, then ~0.05 s/observer. Backbone: ~5,000 edges × a vectorized robust fit ≈ 5 s,
plus k-shortest paths in `networkx`. **Estimate ~3–5 min per world**, so the 71-world grid is
**roughly 4–6 h on CPU**, parallelized over realizations; S6's perturbation grid is capped at 5 per
knob and applied only to S2. Budget < 8 h. No GPU — Step 1 measured the CPU as 4.8× faster for this
shape, and nothing here is a batched filter.

New raw data: `gn_published_series_2026-11/` (~1 MB), taking `data/raw/` from ~176 MB to ~177 MB —
well inside ADR-006's revisit triggers.

### Long runs are handed to the PI, not executed in session

**Rule (PI, 2026-11-22): the Claude Code session never launches a job expected to exceed ~15 minutes**,
because a disconnect kills it. The session writes the code, runs `--quick` smoke tests, and hands over
command lines. The PI launches the grid; the session then builds figures and the results doc from the
outputs on disk.

Every long script is therefore **resumable per `(scenario, realization, baseline)` cell** — it skips a
cell whose output already exists unless `--force` — and takes `--jobs` for process-level parallelism,
so a partial grid still scores and an interrupted run resumes rather than restarting.

| Stage | Expected | Who runs it |
|---|---|---|
| `pytest`, `ruff`, `--quick` smoke of every script | < 2 min | session |
| `fetch_published_gn_series.py` | < 2 min | session |
| S1 only (`--scenarios S1`), 2 realizations | ~10 min | session, as the end-to-end check |
| **Full generate + baselines + score** | **4–8 h** | **PI** |
| A157 validation (121 synthetic observers × 5 methods) | **16 s** (the 30–60 min estimate was wrong) | session |
| `make_step2_figures.py` from existing outputs | < 3 min | session |

The grid is **one command**, `scripts/run_step2_grid.sh`, which the PI launches:

```bash
./scripts/run_step2_grid.sh --detach     # returns at once, survives logout
./scripts/run_step2_grid.sh --status     # how far has it got
```

It sequences the stages, enforces the one hard barrier (nothing that reads a
world may start before generation finishes, because the runners glob for
*complete* realizations), runs the three independent middle stages and the two
scoring passes in parallel, and aborts if generation fails. Re-running it
resumes. A failing stage does not abort the stages that do not depend on it, and
the run ends with a summary and a non-zero exit code.

Each stage writes its own `progress.json` and `lineage_*.json` beside its log, so progress and
provenance are readable without tailing anything, and two concurrent stages cannot overwrite each
other's record.

## Risks

| Risk | How it shows | Fallback |
|---|---|---|
| CEA17/UEA21 reimplementation diverges from the published methods | A7 fails — wrong signs against their Fig. 3b | A7 is the detector, and it runs on their setup where their answer is published. Iterate on the port until signs agree, or report the residual difference as a finding and mark those two baselines "our reading of" |
| The replay's donor-cycle mixing introduces an artificial scale jump | Cycle-amplitude error correlates with donor-survey boundaries | Single-donor-survey per realization is the default; mixed-donor is a flagged sensitivity with the ADR-011 offset recorded |
| Thinned-RGO grand minima are not a real regime change | S3 scores look good but transfer poorly | Stated as a limitation in the results doc, not smoothed over; no source exists for grand-minimum synthetic truths and the plan says so |
| No matched-model control, so overconfidence is not decomposed | The paper wants method error separated from data error | Stated limitation; a matched-model arm is a cheap later addition |
| The benchmark flatters every method because the generator is still too tidy | All baselines show near-nominal coverage | The four deliberate misspecifications plus correlated observer errors are the guard; if coverage is still nominal everywhere, that is itself a reportable result and the stress scenarios S3/S4 are where it should break |
| Backbone port's V1-12 station numbers | Paths terminate at the wrong observers | Endpoints resolved by observer name, never by number; a test pins Staudach → 466 and Locarno → 614 in V1.22 |
| Published series are behind paywalls or have unclear licences | `fetch_published_gn_series.py` cannot retrieve one | Record what is unavailable in `refresh_todo.md`, use the `legacy/GN_Series_matlab/data/` copy where one exists, and say which series is missing |
| Runtime overruns the estimate | The grid does not finish in a working day | S1 is the development scenario; every long script is resumable per (scenario, realization, baseline) cell and writes `progress.json`, so a partial grid still scores and a re-launch resumes rather than restarting |
| A long run dies with the session | Hours of compute lost on a disconnect | The session never launches one (PI, 2026-11-22). The grid runs detached under `nohup`/`tmux` from the documented commands, and resumability makes a lost run cheap |

## Verification (how `/verify-step 2` will check this)

Split the same way the runs are: short checks in session, the grid launched by the PI.

**In session (< 15 min total):**

```bash
conda activate gn-next-gen
pytest && pytest -m slow
ruff check . && ruff format --check .
pytest tests/test_data_manifest.py                   # the new published-series folder
python scripts/fetch_published_gn_series.py --check  # byte-reproducible
python scripts/generate_benchmark.py --config configs/step2_benchmark.yaml \
       --scenarios S1 --realizations 2              # end-to-end on the cheap scenario
python scripts/generate_benchmark.py --config configs/step2_benchmark.yaml \
       --scenarios S1 --realizations 2 --check      # determinism: hashes match
python scripts/make_step2_figures.py                 # rebuilds f13-f26 from outputs on disk
git diff --name-status main -- data/raw legacy       # one addition, legacy untouched
```

**Launched by the PI** (the four `nohup` commands under *Compute and data*), then back in session:

```bash
python scripts/make_step2_figures.py                 # now with the full grid behind it
```

Verification reads the committed CSVs the grid produced — `scores.csv` (42 cells), `coverage.csv`,
`step2_case_study_observers.csv`, `a157_validation.csv`, `sampling_only.csv` — and confirms every
number in the results doc traces to one of them.

Then read `docs/results/step-2-blind-benchmark.md`: the 42-cell score table, the coverage figure, the
ten case-study observers, the A157 positioning, the sampling-only result, and the stated limitations.

## Open questions for the PI

None blocking — the ten decisions above were taken in this session. Two are recorded for visibility
rather than as questions:

1. **Spec acceptance criterion 4 is deliberately not met** (decision 7). Criterion A7 replaces it with
   an external, published target. If you later want the `runcsvs/` check, the recovered parameters are
   preserved in `docs/03_legacy/adfssn.md`.
2. **No new ADR is proposed.** Nothing here contradicts an accepted decision: ADR-003's no-SILSO rule
   binds the new model's calibration and decision 3 removes SILSO from the baselines entirely; ADR-009's
   anchors are declared per scenario; ADR-007 is what this step implements.

## Approval

**Approved 2026-11-22** (PI, in session), together with the ten decisions above.

## Deviations

Recorded as they arise, per `CLAUDE.md`'s workflow.

1. **Spec acceptance criterion 4 is not met** (PI decision 7, 2026-11-22). The spec asks
   that "a port, if used, reproduces legacy thresholds on a stated set of observers
   within a stated tolerance". `legacy/ADFSSN/runcsvs/` was not produced by the
   committed driver settings and the snapshot already carries `uncommitted.diff`, so the
   target is not well defined. Criterion **A7** replaces it with an external, published
   target. The recovered `runcsvs/` parameters are preserved in
   `docs/03_legacy/adfssn.md` so a later session can pick the check up.

2. **A7 passes in part, and the part that does not is explained.** All five ported
   methods agree in sign with A157's Fig. 3b. Their *exclusion limits* do not: HoSc98
   gives up at 73 msd here against their 31. The cause is measured, not guessed --
   HoSc98's factor is a ratio of sums, so losing a fraction `f` of group-days implies
   `k ~ 1/(1-f)` and its 0.6-1.4 gate trips at `f = 0.29`, which our renovated RGO
   reaches between 60 and 87 msd. A157 used the 2017 catalogue, whose small-group
   completeness differs (D14 puts RGO's floor at 28.1 MSH). `a157_catalogue.csv` carries
   the evidence. This is a data difference, not a port defect.

3. **The replay's verbatim window is the donor survey's own span**, not 1825-2026.
   The plan says "1825-2026 is the real reference, used as-is". With one donor survey
   per realization -- which the plan also requires, because Schwabe's areas are umbral
   and mixing donors would plant a scale jump -- the verbatim window is that survey's
   own span: RGO 1877-1969 by default. Everything outside it, early *and* late, is
   resequenced from the same survey's complete cycles. Tested: inside that window the
   truth reproduces RGO's daily reference-equivalent counts exactly over 24,219 days.

4. **Break-detection skill is NaN for all seven baselines.** None of them attempts to
   detect a career break, so POD, FAR and the Heidke score have nothing to score. The
   metric is implemented and unit-tested so Steps 3-4 can be scored on it, and the
   absence is reported as a result -- break attribution is one of the generalizations a
   joint model adds -- rather than filled with a proxy.

5. **Three of the five published GN series were acquired, not five.** Chatzistergos
   et al. (2017) and Velasco Herrera et al. (2024) live at CDS/VizieR, which is
   unreachable from this machine (connection timeout), and no alternative open source
   was found. The three in hand were exported from the read-only 2017 MATLAB snapshot
   rather than re-downloaded, which is better for the purpose: they are the exact bytes
   the legacy analyses used. Recorded in `docs/04_data/refresh_todo.md`. The benchmark
   does not consume any of them.

6. **The A157 validation takes 16 seconds, not the 30-60 minutes the plan budgeted.**
   The estimate was wrong by two orders of magnitude, so this runs in session rather
   than being handed to the PI. The other three long stages stand as planned.

7. **The module layout differs from the plan's Files table**, recorded here rather than
   discovered at verification (Step 1's finding F5). Added: `benchmark/case_studies.py`
   (the ten named observers, which the plan described but did not place),
   `benchmark/progress.py` (resumable-run bookkeeping, which decision 10 requires to be
   workable), `baselines/base.py` (the shared pairwise/chaining machinery) and
   `baselines/nulls.py` (the two floors). The plan's `benchmark/generators/synthetic.py`
   -- a second, model-based generator -- **was not written**: the replay generator
   already supplies the four structural misspecifications of decision 6 through real
   data, and a parametric second generator would have added a knob without adding a
   departure from the Step 4 model. Stated as a reduction in scope, not an omission.

8. **Only a two-world S1 slice was run in session** (decision 10). The full grid, the
   S6 structural ensemble and the scenarios S2-S4 are the PI's to launch from the
   documented commands; the results doc marks every number that comes from the slice.
