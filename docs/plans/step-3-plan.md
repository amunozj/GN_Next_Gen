# Step 3 plan: Global network calibration (Approach 2)

- Spec: `docs/05_roadmap/step-3-global-network-calibration.md`
- Author: Claude Code session, 2026-11-24
- Status: **Approved 2026-11-24** (PI, in session) <!-- Draft → Approved (by PI, with date) → Superseded -->

## Context

Steps 1 and 2 are both **Pass with caveats**, verified 2026-11-22. Step 3 is the first step that
*builds a calibration method*, and Step 2 set the bar it has to clear — measured, not asserted:

| What Step 2 measured | Number | Why Step 3 exists |
|---|---|---|
| Best point error among seven published methods | yearly cRMSE **0.301** (HoSc98) | the bar |
| The hard floor | daily median of raw counts, **0.360**, beats **six of seven** in most of 51 worlds | a method that loses to this has demonstrated nothing |
| Only near-calibrated intervals | CEA17 **0.591 / 0.862** at nominal 0.50 / 0.90, 90% width **1.92** | the three chains are ~1.4× too narrow, UEA21 6.6× |
| Reach | **five of seven stop at 1774**; 39 of 422 observers never reach the anchor; 1727–1745 and 1777–1798 have **zero** edges crossing | the open goal |
| Maunder level | the two methods that reach it are **23–25×** high; the raw median is right to 3% | the headline gap |
| Acuity recovery | recovered acuity tracks the *era's* activity at ρ = **−0.62 / −0.73** | "not a per-observer error, it is a missing covariate" |
| Hardest observers | `poor-instrument` (3 of 7 typically >2× off, SvSc16 silent in all 50); `slow-drift` breaks all seven occasionally | a drift prior is not optional |
| Structural spread | narrower than every method's own reported sd | the error is not in the graph knobs, it is in the emission model |

Step 1 hands over: long-format loaders with checksum caching, the firewall and KMAS single-use rule
**enforced in code**, the ADR-011 area law with `fraction_above()`, the activity-level functional,
`K = 41`, `T ≥ 30 MSH` as the trustworthy band, the per-group vs per-group-day `μ` shift of −0.19 to
−0.26, and the finding that **`q_o` is not a constant per observer — it moves with the cycle**, which
is "the standing explanation for ADF's activity-dependent k-factors"
([step-1 results](../results/step-1-data-foundation.md) §5).

This plan must decide the four spec open questions plus four more that change the design. **All eight
were answered by the PI in this planning session** (table below).

## PI decisions taken in session (2026-11-24)

| # | Question | Decision |
|---|---|---|
| 1 | Pairwise term form (spec OQ1) | **Both kernels**: the conditional-binomial reduction *and* the exact marginalization over N |
| 2 | Composite-likelihood correction (spec OQ3) | **Godambe sandwich**, score variance `G` by **day-block bootstrap** |
| 3 | Drift prior (spec OQ2) | **Hierarchical annual random walk** on log-threshold + free jumps at documented breaks |
| 4 | Fingerprint vs pairwise weighting (spec OQ4) | **One composite likelihood at weight 1**; Godambe absorbs the overlap; ablation at w ∈ {0, 0.5, 1, 2} |
| 5 | Activity-conditioned detection | **Yes** — `k_o(t)` from the activity-dependent area law, solved by outer iteration |
| 6 | KMAS and the cycle-20 bridge | **`kmas_role=observer_counts`, full extension span** (`extension_post_v13_only=False`) |
| 7 | Anchorings | **Both ADR-009 anchors released**; **RGO 12–19 run once as a named sensitivity** |
| 8 | Ensemble | **Laplace at the mode with the sandwich covariance**; NUTS cross-check on one dense sub-network |

Carried forward from Step 2 decision 10, unchanged: **no run over ~15 minutes is launched from a
Claude Code session.** Every long script is resumable, writes `progress.json` via
`benchmark.progress.Progress`, and is launched detached by the PI.

### Why "both kernels" is one design and not a hedge

The research corpus documents the conditional-binomial reduction and its limit in the same breath:
if `Y_{o,t} ~ Poisson(k_o(t)·λ_t)` then conditioning on the pair's same-day total gives
`Y_a ~ Binomial(n, k_a/(k_a+k_b))`, "a Bradley-Terry likelihood with λ_t eliminated, so overlap days
are literally match results" — but it "breaks down for nonlinear threshold effects … and **at low
activity, where zeros dominate**. A threshold-thinning model is then needed and the exact BT
reduction is lost" (`notes/raters_linking_ratings.md` Q4, flagged *derived*). Low activity with
zeros dominating **is** the Maunder Minimum, which is the era this step has to reach.

So the two kernels are a default and a low-count path, and their disagreement is a deliverable:
the equating section's standing advice is "Build the concurrent Bayesian model as the primary
product. Keep a Haberman-style linking of separately estimated pairwise calibrations as a
diagnostic. **Treat disagreement between the two as a misspecification signal.**"

## Scope

In scope:

1. `gn_next_gen.models.emissions` — per-observer emission tables `P(y | N, θ_o, t)` by soft-thresholding
   real reference areas. Shared with Step 4 (`inference/forward_filter.count_emissions`'s docstring
   already promises this module).
2. `gn_next_gen.models.network` — the concurrent composite likelihood: two pairwise kernels, the
   fingerprint term, the drift priors, the anchor constraint.
3. `gn_next_gen.models.activity_map` — the GN → activity-level map and the outer fixed-point loop.
4. `gn_next_gen.inference.composite` — mode finding, Hessian, day-block bootstrap, Godambe sandwich,
   draws, eigenmode diagnostics.
5. `gn_next_gen.graph.diagnostics` — resistance between two anchor sets, per-year resistance, loop
   closure over a cycle basis, per-observer diagnostic table, per-pair Δt extraction.
6. Benchmark scoring of the new methods against all eleven Step 2 methods, including coverage.
7. The real-data run, 1610–2026, under both anchorings plus the three named sensitivities, with the
   observer parameter table and the posterior ensemble.
8. Results doc, figure-first, figures `f29`–`f40`.

Out of scope (the spec's list, plus additions):

- **Latent daily dynamics of the full population — Step 4.** Specifically: no birth–death transition,
  no FFBS, no rotation kernel inside the likelihood, no λ(t) state-space GP. λ(t) here is a
  penalized spline used only to supply `AL_t` and the N prior, and it is refit by profiling, not
  sampled. This boundary is the main scope risk and is restated in Risks.
- Proxies (Step 6); neural emission calibration (Step 5); any change to Step 4's model.
- Running legacy code (Step 2 decision 2 stands).

## Design

### Data flow

```
io.load_observations(release="v1.3_4", include_extension=True,
                     kmas_role="observer_counts",
                     extension_post_v13_only=False)      ──┐
io.reference.daily_counts(threshold, surveys=[anchor])   ──┤
reference.area_law.load_upstream_parameters(anchor)      ──┤
                                                            ▼
                                        graph.build_overlap_graph(...)
                                                            │
        ┌───────────────────── outer fixed point (3–5 passes) ─────────────────────┐
        │  AL_t   ← activity_map(GN̂_t)        map fitted on 1874–2026 reference    │
        │  k_o(t) ← fraction_above(T_o(t); AL_t) / fraction_above(T_ref; AL_t)     │
        │  θ̂     ← inference.composite.fit(...)   L-BFGS on the composite log-lik  │
        │  GN̂_t   ← models.network.combine(θ̂)     calibrated counts + smoother      │
        └──────────────────────────────────────────────────────────────────────────┘
                                                            ▼
                     V = H⁻¹ G H⁻¹  (G by day-block bootstrap)  →  θ draws  →  ensemble
                                                            ▼
                  CalibrationResult(series, observers, name, meta)  +  lineage.json
```

### The detection model: `k_o(t)` and why it is activity-conditioned

The estimand is the anchor survey's own reference-equivalent count (ADR-003, ADR-009): the number of
groups the anchor would have counted above its own completeness threshold `T_ref`
(`benchmark.truth.anchor_threshold`; Debrecen 20.42, RGO 28.08 MSH). An observer with soft threshold
`T_o(t)` is modelled as seeing a *relative* fraction

```
k_o(t) = fraction_above(T_o(t); params(AL_t)) / fraction_above(T_ref; params(AL_t))   ·  (1 + s_o)
```

with `fraction_above` the Step 1 functional, `params(AL_t)` the ADR-011 log-normal parameters at
activity level `AL_t`, and `s_o ≥ 0` the splitting rate (which lets `y > N`). `k_o` can exceed 1 when
`T_o < T_ref`. The anchor itself has `k_ref ≡ 1` by construction, which pins the absolute scale —
the only scale unmarked counts can identify (ADR-003; the N-mixture degeneracy "multiplying all
`p_o` by c and dividing `λ_t` by c leaves the marginal unchanged").

**Why the activity term is in, and why it is fitted rather than fixed.** Step 2 §7 f23 measured the
published methods' acuity error as a missing covariate, not a per-observer mistake
(ρ = −0.62 UEA21, −0.73 GRP; both pin at the 300 MSH grid ceiling for quiet-era observers). The
ubercal derivation names the fix: "A scalar k-factor is FGCM's gray approximation; **a detection
function applied to an area distribution that changes with activity is the chromatic correction**."
SDSS also supplies the warning about *fixing* such a term: a time derivative of extinction that was
"held fixed rather than fitted accumulated into a ~10 mmag survey-wide tilt, which the authors call
the most serious systematic." The GN analogue is the activity exponents of ADR-011. They are held
at upstream's values by ADR-011, so the tilt they could induce is measured as a **named sensitivity**
(below) rather than assumed away.

`AL_t` is unavailable before 1874 (no areas). It comes from a monotone map `AL = g(GN)` fitted on
1874–2026, where both the reference-equivalent count and the measured activity level exist, then
applied to the current `GN̂_t` and re-evaluated each outer pass. **Firewall:** the map's inputs are
reference areas and the observers' own counts. No SILSO, no proxy, no published series — the last of
these refused by `config.validate_run_config` as `kind: published_series` in role `calibration`.

### The composite likelihood

```
log CL(θ) =   Σ_{edges (a,b)}  Σ_{matched day pairs}   ℓ_pair(y_a, y_b; Δt, θ_a, θ_b)
            + Σ_{observers o}                           ℓ_fing(y_{o,·}; θ_o)          [w = 1]
            + log π_drift(θ) + log π_pop(θ)
```

**Kernel A — `pair_conditional` (default).** Beta-binomial on the pair's same-day total:

```
y_a | n_ab ~ BetaBinom(n_ab, p_ab, κ(Δt)),      p_ab = k_a(t) / (k_a(t) + k_b(t))
```

The latent count drops out exactly. `κ(Δt)` is the **lifetime-based decorrelation** the spec asks
for, replacing the ±2-day "simultaneous" assumption that `reuse_map.md` puts on the Drop list:
`κ` is set from the reference's own detrended autocorrelation, already measured in Step 1 (`f07`,
with the 13-day transit trough and the 27-day recurrence bump), so Δt = 0 is a plain binomial and
larger lags are honestly overdispersed instead of silently equated. Cost: one evaluation per matched
day-pair, vectorized; no sum over N.

**Kernel B — `pair_marginal` (low counts and the cross-check).**

```
P(y_b | y_a, Δt, θ) = Σ_N  P(y_b | N, θ_b) · P(N | y_a, Δt, θ_a),    N = 0..K−1,  K = 41
```

with the emission tables from `models.emissions` and `P(N | y_a) ∝ P(y_a | N, θ_a) · π_N(t)`. The
prior `π_N(t)` is **not** a free choice: it is Poisson with rate `λ̂(t)` from the same outer loop
that supplies `AL_t`, which is what makes this kernel admissible in Step 3 at all. Used wherever the
pair's mean count falls below a stated threshold (default: either observer's career-mean < 2 groups),
and run over the whole network once as the misspecification cross-check (`f38`).

**Fingerprint term (`ℓ_fing`).** The occupancy form the spec names, written as a likelihood with the
activity covariate the published methods lack. For an observer-day with latent `N_t`:

```
P(active | N_t) = 1 − Π_{g ∈ day t} (1 − F_o(A_g))      →      1 − (1 − r_o(t))^{N_t}
```

the Royle-Nichols form `ecology_imperfect_detection.md` identifies as "the formal version of the ADF
fingerprint", identified "once the distribution of `N_t` (and of areas) at that activity level is
known from the reference". Step 3 uses the **full daily count distribution**, not only the active-day
fraction, because the pair (mean, zero fraction) is what separates `T_o` from the local activity
level — the symmetry break the corpus points at ("features of the count distribution that a pure
rescaling cannot change … could break the symmetry"). This is the term that lets an observer with no
usable edge be calibrated at all, and therefore the term that decides whether Step 3 reaches past
1774. Weight 1, in the same sum (PI decision 4), because running it as a separate pre-calibration
stage is the circularity `ecology_imperfect_detection.md` warns against.

**Drift prior.** Uto (2023)'s Markov severity prior, taken over directly with its measured
hyperparameters:

```
log T_{o,y} = log T_{o,y−1} + σ_o ζ_{o,y},   ζ ~ N(0,1)        (non-centred, annual knots)
log σ_o ~ N(μ_σ, 1),        μ_σ = −2        (Uto's recommended location)
+ a free jump at each metadata-documented break   (benchmark `metadata.parquet`; ADR-004)
```

Uto measured RMSE gains of 0.079 / 0.089 / 0.124 from the Markov prior, "**the benefit grew as data
per parameter shrank**" — which is the 17th century — and measured the prior sensitivity this plan
must report: "μ_σ = −5 over-shrinks change, and μ_σ = 0 is unstable". `σ_o` is pooled
hierarchically so isolated early observers borrow strength. Undocumented breaks are *not* given free
jumps: the random walk absorbs them, and whether it does so well enough is what the paired
`break-documented` / `break-undocumented` case studies measure (Step 2 priced documentation at a
factor 2–3).

Deliberately **not** done: cutting careers at detected breaks. "Every cut … raises effective
resistance and lets real change leak away", and at realistic signal-to-noise "even a random
segmentation explains about half the break variance" — which is exactly why
`null-segmentation` is a registered floor.

### Identifiability, stated as first-class output

| Trap (assessment §6) | How this step handles it | Where it is reported |
|---|---|---|
| 1. Absolute scale unidentifiable from counts | `k_ref ≡ 1` at the declared anchor; never compare two anchorings' absolute levels | `f36`, both anchors side by side |
| 2. Era-wide common drift ≡ solar change | not identified. Reported as **prior sensitivity** over μ_σ ∈ {−5, −2, 0} | `f35` |
| 3. Chained error ∝ effective resistance | concurrent fit; resistance reported per observer, per year and **between the two anchors** | `f29`, `f30` |

Two more that the solver forces into the open:

- **One exact Laplacian null mode per connected component.** The anchor pins only its own component.
  Every other component's level comes from the fingerprint term alone. `f39` names which components
  are anchored and which float; the observer table carries the flag per station. Step 2 measured 14
  components, largest 383, and 39 of 422 stations with no path to the anchor.
- **Weakly-constrained eigenmodes.** SDSS "projected out and then restored with Gaussian priors"
  the eigenmodes "that could cause errors above about 1%". The same is done here: the Hessian's
  eigenvalue spectrum is reported, and modes whose posterior is prior- rather than data-dominated
  are labelled as such. This is the mechanism behind the per-year prior→posterior contraction map
  ADR-008 wants, delivered in Step 3 for the network parameters and in Step 4 for the series.

### The correction, and why it is derived here

The research corpus contains **no** sandwich/Godambe derivation, **no** reference and **no** magnitude
estimate for the over-counting. It gives the mechanism three times (the assessment's "pairwise
likelihoods double-count days seen by many observers (overconfident unless corrected)";
`counting_processes_underreporting.md` Q2(e)'s "composite-likelihood posteriors need curvature
adjustment for calibrated uncertainty"; Patz et al.'s facets-model warning that ignoring dependence
among ratings of the same response "appear[s] to give infinitely precise proficiency as raters are
added"). The correction below is therefore **derived**, and its only validation is empirical, on the
Step 2 benchmark — which is what the spec's acceptance criterion already asks for.

```
H = −∇² log CL(θ̂)                        curvature at the mode
G = Var(∇ log CL)                        resampling WHOLE DAYS, B = 200 (100 on benchmark cells)
V = H⁻¹ G H⁻¹                            Godambe / sandwich
θ_s ~ N(θ̂, V),  s = 1..1000              → series ensemble
```

Days are the resampling unit because the day is the thing double-counted: one day seen by twelve
observers enters 66 pairwise terms. The bootstrap resamples the pairwise *and* fingerprint scores
jointly, which is why decision 4 needs no weight — the information each term really contributes is
measured. `f33` reports coverage under the uncorrected Laplace covariance and under `V`, side by
side, so the correction's effect is a number rather than a claim.

### From θ to the series

`GN̂_t = Σ_o w_{o,t} · y_{o,t} / k_o(t)`, inverse-variance weights, then the "simple continuous
smoother" the spec names (a penalized spline, the same basis as `inference.forward_filter.knot_basis`
with `second_difference_penalty`). Each θ draw gives one series; the ensemble's spread is the
uncertainty. The result carries Step 2's mandatory decomposition:

- `gn_sd_factor` — the part from `k_o`, correlated along a career (does **not** average down);
- `gn_sd_spread` — observers disagreeing on the day itself (does average down).

Without the split a yearly interval is too narrow by up to √365 ≈ 19 — the defect that made Step 2's
first coverage table say the opposite of what it now says (§12.7). `baselines.base.combine_observers`
and `inverse_variance_daily` already emit both and are reused.

### Registration and the comparison

Two methods, registered in `BASELINES` in `src/gn_next_gen/baselines/__init__.py` as thin adapters
over `models.network` (which lives in `models/`, already inside `BLIND_PACKAGES`):

| Name | Kernel |
|---|---|
| `NetCal` | Kernel A everywhere, Kernel B below the low-count threshold |
| `NetCal-M` | Kernel B everywhere — the cross-check arm |

`scripts/score_benchmark.py` discovers cells by globbing `*_series.parquet` and derives the method
name from the filename, so **no scorer change is needed**. The new cells are written into the
existing `outputs/step2/baselines/<scenario>/seed-%04d/` tree and scored in the same pass as the
eleven. A regression check asserts the existing eleven methods' score rows are byte-identical before
and after, so adding Step 3 cannot silently move Step 2's published numbers.

`viz/benchmark.py`'s `METHOD_ORDER` and `METHOD_MARKERS` gain the two names (there is a
`MARKER_FALLBACK` tuple for exactly this).

Also delivered: a **finite break-detection score**. `scoring.break_detection_skill` returns NaN for
all seven baselines — "the metric is implemented and tested so Steps 3–4 can be scored on it". The
fitted random walk supplies a detector (a year whose increment exceeds `c·σ_o`), so Step 3 reports
POD / FAR / Heidke as measured numbers, whatever they turn out to be.

### The real-data run

| Run | Anchoring | Status |
|---|---|---|
| R1 | `debrecen_21_23` | **released** (primary) |
| R2 | `rgo_15_19` | **released** (secondary) |
| R3 | `rgo_12_19` | sensitivity — discharges ADR-009's named Step 3 task; D5 caveat stated, reported as a delta against R2, not as a third release |
| S-drift | R1 at μ_σ ∈ {−5, 0} | drift-prior sensitivity (`f35`) |
| S-tilt | R1 with the ADR-011 activity exponents perturbed ±1σ | the SDSS drift-tilt analogue |
| S-ablate | R1 at fingerprint w ∈ {0, 0.5, 2} | what the fingerprint buys, and what it drags in |
| S-tol | R1 at `day_tolerance` ∈ {2, 5, 10} | the kernel now down-weights Δt honestly, so the legacy 2 is no longer forced. **Default stays 2** for like-for-like comparison with Step 2 |

**KMAS (decision 6).** `kmas_role="observer_counts"` with `extension_post_v13_only=False`, so KMAS's
1954–2026 counts are present and cycle 20 is bridged — ADR-009: cycle 20 "is complete in KMAS alone
… so KMAS bridges the two anchors … which is what makes the Step 3 effective-resistance diagnostics
between them meaningful instead of degenerate". Its areas go unused in these runs, satisfying the
single-use rule, which `io.load_observations` and `config.validate_run_config` each enforce
independently. **Open loose end to measure, not assume:** whether Kislovodsk already appears as a
v1.3 station, which would make the bridge two copies of one observer. The run reports the overlap
between station 1006's days and every v1.3 station's days; if a near-duplicate exists the decision
taken is recorded in `lineage.json`.

## Files

### New

| Path | One line |
|---|---|
| `src/gn_next_gen/models/emissions.py` | `P(y \| N, θ_o, t)` tables by soft-thresholding real reference areas; shared with Step 4 |
| `src/gn_next_gen/models/activity_map.py` | monotone `AL = g(GN)` fitted on 1874–2026; the outer fixed-point driver |
| `src/gn_next_gen/models/network.py` | the two pairwise kernels, the fingerprint term, drift priors, parameter packing, the composite log-lik in JAX (float64) |
| `src/gn_next_gen/inference/composite.py` | L-BFGS mode, Hessian, day-block bootstrap of the score, Godambe `V`, draws, eigenmode spectrum |
| `src/gn_next_gen/graph/diagnostics.py` | resistance between two anchor sets, per-year resistance, loop closure over a cycle basis, the per-observer diagnostic table |
| `src/gn_next_gen/viz/network.py` | `f29`–`f40` |
| `scripts/run_network_calibration.py` | the real-data runs R1–R3 and the five sensitivities; resumable, `Progress` |
| `scripts/run_step3_benchmark.py` | the two methods over the 51 existing worlds; resumable per cell |
| `scripts/make_step3_figures.py` | rebuild `f29`–`f40` from committed inputs, both modes |
| `scripts/run_step3_grid.sh` | one-command detached orchestration, `--detach` / `--status`, mirroring `run_step2_grid.sh` |
| `configs/step3_network.yaml` | `RunConfig`: anchorings, kernels, priors, bootstrap size, graph knobs, sensitivity grid |
| `tests/test_emissions.py`, `test_network_model.py`, `test_composite_inference.py`, `test_graph_diagnostics.py` | see Tests |

### Modified

| Path | Change |
|---|---|
| `src/gn_next_gen/graph/overlap.py` | `build_overlap_graph(..., return_pairs=False)` optionally returns per-edge matched-day Δt. `OverlapGraph` today carries only aggregate `overlap_days`, and both kernels need the Δt distribution. Backward compatible; default unchanged |
| `src/gn_next_gen/baselines/__init__.py` | register `NetCal`, `NetCal-M` |
| `src/gn_next_gen/viz/benchmark.py` | add the two names to `METHOD_ORDER` / `METHOD_MARKERS` |
| `docs/05_roadmap/README.md` | status row; paper-item checkboxes for the Step 3 items |
| `docs/04_data/known_issues.md` | D5 status note if the RGO 12–14 sensitivity speaks to it |

### Legacy to port (`docs/03_legacy/reuse_map.md`)

- Row 4 — overlap detection with a day tolerance (`GN_Series_matlab/A1_Matrix_Calculation_weighted_fit.m`
  l. 71–117): already in `graph.overlap`; extended here with per-pair Δt.
- Row 5 — robust pairwise fits (l. 129–168): **baseline and diagnostic only**, already ported as
  `baselines.pathfinding.robust_proportional_fit`; the loop-closure diagnostic (`f30`) uses it.
- Row 6 — ADF / count-distribution fingerprints (`SSN_ADF.py` l. 884–900, 1040–1113;
  `SSN_GRP.py` l. 851–866): the reuse map assigns the occupancy form
  `P(active \| N) = 1 − (1 − r)^N` to **Step 3**. Ported as `ℓ_fing`, as a likelihood, with activity
  conditioning and no EMD/`NBEST` grid search.
- Row 1 — strictly-greater thresholding of reference areas: reused through
  `io.reference.daily_counts` and `reference.area_law.fraction_above`; not re-implemented.

## Tests

Fast suite (target: the four new files add **< 25 s**, keeping `pytest` near its current ~61 s):

- `test_emissions.py` — tables are row-stochastic to 1e-12; a hard threshold reproduces
  `io.reference.daily_counts` exactly on a fixture day; splitting rate 0 gives mass only at `y ≤ N`;
  `k_o` monotone decreasing in `T_o`; `k_o(T_ref) == 1` to 1e-12.
- `test_network_model.py` — **synthetic recovery**, one per kernel: simulate a 40-observer network
  with known `θ`, recover base thresholds with |bias| ≤ 10% and drift σ within a factor 1.5; the two
  kernels agree to 2% at moderate counts and diverge in the stated direction below 2 groups/day;
  the anchor's `k ≡ 1` is held exactly; a sum-shift of all `log k` is rejected by the anchor
  constraint (the degeneracy test).
- `test_composite_inference.py` — on a hand-made case with a known over-counting factor, `V` exceeds
  the naive `H⁻¹` by the right factor; a one-observer-per-day network (no double counting) leaves
  `V ≈ H⁻¹` within 10%; coverage of the draws on a matched-model simulation is within ±0.05 of
  nominal at 50/68/90/95; the eigenmode spectrum flags a deliberately unanchored component.
- `test_graph_diagnostics.py` — resistance between two sets on a hand-computable graph; removing the
  bridge sends anchor-to-anchor resistance to ∞; loop closure is exactly 0 on a consistent synthetic
  network and recovers a planted inconsistency; every station gets a diagnostic row.
- `test_viz.py` — unchanged mechanism, new figures picked up automatically (both modes, < 400 kB,
  embedded **and** captioned).
- `test_config.py` — `configs/step3_network.yaml` passes the validator; a KMAS double-use variant and
  a `kind: published_series` calibration variant are both refused.
- `test_benchmark_blindness.py` — unchanged; `models` is already in `BLIND_PACKAGES`, and the audit
  must still show 0 imports of `benchmark.truth` from it.

Slow (`-m slow`): the noise-free recovery test below; the full-network mode fit timing; one
200-resample bootstrap on a reduced network.

**The noise-free test, mirroring SDSS.** Padmanabhan et al.'s noise-free simulations reached ~0.1%,
"so the residual error came from model misspecification rather than from statistics", and the report
concludes "Calibration quality is therefore limited by model misspecification, not by solver or
statistics." The analogue here: on a synthetic network generated by *this* emission model with no
seeing noise, no MNAR and no copying, the fit must recover `θ` to within 1%. Anything left on the
blind benchmark is then attributable to misspecification rather than to the solver — which is the
honest way to read `f32`, and the reason the deliverable is the emission model, not the optimizer.

## Acceptance criteria

| Spec criterion | Measurement | Pass condition | Figure |
|---|---|---|---|
| Matches or beats legacy baselines at the stated time scales | `crmse_yearly` from `scores.csv`, pooled over the 51 worlds, and the paired per-world win fraction against `naive-median` in `floors.csv` | yearly cRMSE **≤ 0.301** (HoSc98's); beats `naive-median` in **≥ 35 of 51** worlds (HoSc98's own margin); daily and monthly reported | `f32` |
| Interval coverage after the CL correction within a stated tolerance | `coverage_50/68/90/95` and `interval_width_90` over the 450 calibration cells, under `H⁻¹` and under `V` | **\|empirical − nominal\| ≤ 0.10 at all four levels**, and mean 90% width **≤ 1.92** groups (CEA17's). Both halves required — the `null-prior` floor passes coverage at a width of 9.99 | `f33` |
| Diagnostics computed for every observer; disconnected components flagged | `observers.parquet`: `station, component, anchored, resistance_debrecen, resistance_rgo, degree, on_bridge, reached, n_edges, fingerprint_only` | **0** stations without a row; every component labelled anchored or floating; the 39 Step 2 unreachables all appear with `fingerprint_only = True` | `f39` |
| Runtime on the full network stated (target: minutes on CPU) | wall clock per stage, measured on this machine, CPU and one A100 | the mode fit **≤ 10 min** single-process CPU over 1610–2026; `B = 200` bootstrap **≤ 60 min** at `--jobs 16`; both reported as measured numbers, and the CPU-vs-GPU comparison **measured, not assumed** (Step 1 found GPU 4.8× slower for a sequential scan; this likelihood is a parallel sum, so the sign may flip) | `f40` |

Added criteria, each traceable to a Step 2 measurement:

| # | Criterion | Measurement | Pass condition | Figure |
|---|---|---|---|---|
| A14 | **Reach** — the open goal | earliest year with an estimate; share of 1610–2026 covered | an estimate in **≥ 95%** of years holding ≥ 1 observation; earliest year **≤ 1660** (against 1774 for five of seven) | `f34` |
| A15 | **Maunder level** | `maunder_estimate / maunder_truth` on S3 | ratio **≤ 3×** (against 24.6× UEA21, 22.7× GRP), reported with the standing caveat that the thinned truth is this project's choice | `f34` |
| A16 | **The activity confound is removed** | Spearman ρ between recovered threshold and the era's true mean count, over all case-study fits | **\|ρ\| ≤ 0.25** (against −0.62 and −0.73) | `f31` |
| A17 | **Anchor sensitivity delivered** | both anchored series released; per-era difference; the RGO 12–19 delta | both runs complete; the difference reported per era with a figure; ADR-009's named Step 3 task discharged | `f36` |
| A18 | **Concurrent beats chained on the same information** | `NetCal` vs `Pathfinding` on `crmse_yearly`, paired per world, and against each observer's resistance | `NetCal` wins in **≥ 2/3** of worlds, and its advantage **grows with resistance** (positive slope) — the direct test of the step's thesis | `f30` |
| A19 | **The two kernels' disagreement is reported** | per-observer, per-era difference in implied `log k_o` between `NetCal` and `NetCal-M` | reported, not thresholded. A disagreement > 0.3 in log k for > 10% of observers in an era is declared a misspecification finding for that era | `f38` |
| A20 | **Drift is reported as prior sensitivity, not as a result** | the series under μ_σ ∈ {−5, −2, 0} | the era-wide drift band published; no claim of identified secular drift anywhere in the results doc | `f35` |
| A21 | **Break-detection skill finite** | `break_pod`, `break_far`, `break_heidke` | finite for both methods (NaN for all seven baselines). No threshold — the number is the result | `f31` |
| A22 | **Noise-free recovery** | θ recovery on a matched, noiseless synthetic network | within **1%**, so blind-benchmark residual is attributable to misspecification | — (table) |
| A23 | **Firewall and provenance** | `lineage.json` per run; AST import audit | every run carries `variant: sunspot_only`, the explicit firewall statement and the circularity statement; **0** SILSO/proxy/published-series imports under `models/`, `inference/`, `baselines/`, `graph/` | — (table) |

### Figures (figure-first; `f29`–`f40`)

Step 2 ended at `f28`. Each is rendered in both modes by `viz.style.save_both_modes`, committed under
`docs/results/figures/step-3/` if under 400 kB, and **embedded with a caption that states the claim**
— `tests/test_viz.py` enforces the mechanical half.

| Code | What it shows |
|---|---|
| `f29` | Resistance to each anchor per observer and per year; the anchor-to-anchor resistance with and without the KMAS bridge |
| `f30` | Loop closure over a cycle basis; and `NetCal` − `Pathfinding` error against resistance (criterion A18) |
| `f31` | Parameter recovery on the benchmark: recovered vs true acuity, and the activity-confound panel that `f23` reported (criterion A16); break-detection skill beside it |
| `f32` | Point error at daily / monthly / yearly against the seven baselines and four floors |
| `f33` | Coverage at 50/68/90/95, uncorrected vs Godambe-corrected, with the 90% width beside it |
| `f34` | Reach and the Maunder level against the seven |
| `f35` | Drift-prior sensitivity: the series under μ_σ ∈ {−5, −2, 0} |
| `f36` | The two released series 1610–2026 with ensemble bands, their difference, and the RGO 12–19 delta |
| `f37` | The observer parameter table as a picture: thresholds through time, drift magnitudes, splitting rates |
| `f38` | The two kernels' disagreement per observer and per era — the misspecification signal |
| `f39` | Components, bridges, floating components and fingerprint-only observers through time; the w-ablation's effect on reach |
| `f40` | Measured runtime by stage, CPU against one A100 |

## Compute and data

**No new raw data, so no `MANIFEST.csv` rows.** Every input is already frozen:
`gn_database_v1.3` (v1.3_4), `gn_observer_extension_2026-10`, `reference_groups_2026-10`,
`area_law_2026-11`. `silso_2026-11` appears in the config as `role: comparison` only.

| Stage | Needs | ~Time | Device |
|---|---|---|---|
| Emission tables + activity map | raw data only | < 1 min | CPU |
| One mode fit, full network, one anchoring | tables, graph | target ≤ 10 min | CPU (GPU measured) |
| Bootstrap `B = 200`, one anchoring | the mode | ≤ 60 min at `--jobs 16` | CPU |
| Benchmark cells: 2 methods × 51 worlds | Step 2's worlds on disk | ~1–3 h at `--jobs 16`, `B = 100` | CPU |
| Re-score (11 + 2 methods) | the cells | ~10 min | CPU |
| Real runs R1–R3 + 5 sensitivities | tables, graph | ~4–8 h at `--jobs 16` | CPU |
| `make_step3_figures.py` | everything | < 3 min | CPU |

Memory: the composite likelihood is a sum over matched day-pairs in long format — nothing densifies a
day × observer matrix (CLAUDE.md). Expect well under 4 GB per worker. The Hessian is the one dense
object: ~15–20k parameters would be 2–3 GB dense, so it is held **block-diagonal plus the pairwise
coupling as a sparse matrix**, with `V` applied as a solve rather than formed; only the anchor-relevant
sub-block is ever materialized. This is the one place the design could hit a wall, and it is in Risks.

Outputs to git-ignored `outputs/step3/` (series ensembles, bootstrap draws, CSVs); committed:
`f29`–`f40` in both modes, the per-observer summary CSV behind `f37`, and the score table behind
`f32`–`f34` under `docs/results/data/`.

## Risks

| Risk | How it shows | Fallback |
|---|---|---|
| **Scope creep into Step 4.** The outer loop already estimates λ̂(t); adding dynamics is one small step away | the likelihood grows a transition matrix; the step stops being Approach 2 | the boundary is written into the plan and the tests: no birth–death transition, no FFBS, no rotation kernel inside the likelihood. If dynamics look necessary, that is a Step 4 finding, reported as such |
| **The outer fixed point does not converge**, or oscillates between `AL̂` and `GN̂` | `AL̂` changing by more than a stated tolerance after pass 5 | damped updates; failing that, freeze `AL_t` at the reference-era map's extrapolation and report the activity term as a fixed covariate — which reinstates the SDSS tilt risk, and must then be reported as such |
| **The Hessian is too large to factor** | memory blow-up at ~20k parameters | sparse/block structure as designed; then coarser drift knots (5-year instead of annual); then drift only for observers with ≥ N observing days |
| **The Godambe correction does not reach nominal coverage** | `f33` still below the diagonal | report it honestly as Step 2 did, then fall back to bootstrap-refit intervals for the released series while keeping `V` for the parameter table. **A failed criterion is a result, not a reason to retune** |
| **The fingerprint term drags the Maunder Minimum**, inheriting ADF's activity dependence | A15's ratio stays large while A14's reach passes | the w-ablation (`f39`) separates the two; if w > 0 buys reach at the cost of level, both are reported and the choice is the PI's |
| **Effective resistance oversold.** Ranked idea 7's own cost column: "Captures random error only; **systematic bias adds coherently**" | a reader takes `f29` as the scale uncertainty of the 17th century | every caption and the results doc say resistance is the *random* part only; the systematic part is what `f30`'s loop closure and `f38`'s kernel disagreement probe |
| **Separate-then-link may beat concurrent where overlaps are short.** Cited: "Separate calibration was more accurate when there were few common items", mapped as "GN observers with short overlaps are common" | A18 fails, or fails in the early era | that is a publishable finding against the step's own thesis, reported per era rather than pooled. `Pathfinding` stays in the comparison for exactly this reason |
| **Kislovodsk double-counted** through the KMAS extension | the bridge is two copies of one observer; anchor-to-anchor resistance falsely low | measured before the fit (day-set overlap of station 1006 against every v1.3 station); the decision recorded in `lineage.json` |
| **Adding methods moves Step 2's published numbers** | `scores.csv` rows change for the eleven | the regression check asserts byte-identical rows for the eleven before and after |

## Open questions for the PI

All four spec open questions and four new ones were **answered in this session** (decision table
above). Nothing is blocking. Three items are flagged for awareness rather than decision:

1. **The composite-likelihood correction is derived, not cited.** No sandwich/Godambe reference, and
   no magnitude estimate, exists anywhere in `docs/02_research/`. The derivation and the day-block
   bootstrap are this project's own, validated only empirically on the Step 2 benchmark. The results
   doc will label it as derived. No ADR is needed — ADR-007 already makes the benchmark the arbiter.
2. **No ADR is contradicted by this plan.** ADR-009's default exclusion of RGO cycles 12–14 is
   respected: R3 is a sensitivity reported as a delta, not a third release, which is what ADR-009
   itself asks Step 3 to do. Had the PI chosen three co-equal releases, an amendment would have been
   required; it was not chosen.
3. **If A18 fails in the early era**, the step's own thesis — concurrent beats chained — is
   contradicted where it matters most. The plan reports that per era rather than pooling it away. The
   PI may want to say now whether such a result changes Step 4's priority on the fingerprint/emission
   side; the plan assumes it does not.

## Approval

**Approved 2026-11-24** by the PI, in session, together with the eight decisions in the table above.

<!-- PI writes: "Approved <date>" or requested changes. /implement-step refuses to run without "Approved". -->
