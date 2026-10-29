# Assessment: why a new GN calibration approach, and which one

Written 2026-10-28 at the end of a Claude Code session in the ADFSSN repository. Inputs were a reading of the ADFSSN,
GRPSSN and MATLAB GN-series codebases (now in `legacy/`), a heliophysics literature search, Q&A with the PI, and a
six-track cross-domain search ([../02_research/](../02_research/README.md)). The original problem statement is
[../00_brief/modernization.md](../00_brief/modernization.md).

> **Dated document — the reasoning stands, the data facts have moved on.** This was written against the
> clean-slate data snapshot. The raw inputs were renovated in 2026-10: the reference survey is now five
> per-survey, per-group catalogues covering **1825–2026** with group lifetimes (not RGO/Debrecen/Kislovodsk
> day-by-area matrices from 1874), the observation database is **v1.3** (740 stations with data, not V1.22's
> 715), and the anchored epoch question this document leaves open is settled by
> [ADR-009](../06_decisions/ADR-009-reference-survey-and-anchored-epochs.md) with two co-equal anchors.
> For current data facts use `CLAUDE.md`, [../04_data/](../04_data/README.md) and
> [../results/raw-data-renovation.md](../results/raw-data-renovation.md); read this for *why* the approach
> was chosen.

## 1. Problem in one paragraph

About 700 historical observers (1610–present; 743 stations in the Vaquero et al. 2016 database V1.22) report daily
integer counts of sunspot groups. The true daily number of groups is latent. Observers differ in acuity (small groups
missed), group-splitting conventions, instruments and methods (projection vs. filter), and these can drift over a career.
Observing days are irregular and weather-driven; spotless days may be under-reported. Some observers overlap in time,
many do not. Groups live days to weeks, so nearby-day observations carry information. The modern reference
(RGO 1874–1976, Debrecen, Kislovodsk) records individual group areas, so acuity thresholds can be simulated. The Maunder
Minimum (~1645–1715) has sparse data, suspect spotless days, blanket "no spots this year" statements, almost no overlap,
and activity levels absent from the reference. Two method families exist (below); neither uses group lifetimes,
cross-observer redundancy or prior-plus-evidence inference, and neither gives a proper posterior.

## 2. State of the field (heliophysics)

- **No agreed GN method yet.** The ISSI-team status report ([Clette et al. 2023](https://arxiv.org/abs/2301.02084)) says
  GN reconstruction methods need more work before they can be compared with benchmarks and independent proxies; it
  anticipated new SN/GN versions in 2024 (release not confirmed in this search).
- **ADF bias is documented.** [Willamo, Usoskin & Kovaltsov 2018](https://arxiv.org/abs/1803.10501): ADF overestimates
  GN by up to ~30% at weak-to-moderate activity, underestimates by ≤10% at high activity, is not applicable to grand
  minima, and flattens long-term trends. Original method: [Usoskin et al. 2016](https://arxiv.org/pdf/1512.06421);
  update [Willamo et al. 2017](https://arxiv.org/pdf/1705.05109).
- **Regression is fragile.** [Lockwood et al. 2016](https://arxiv.org/abs/1510.07809): even at r > 0.98, calibrated
  maxima can be off by more than 30%; fits forced through the origin (as in the backbone) are unreliable.
- **Backbone** ([Svalgaard & Schatten 2016](https://arxiv.org/pdf/1506.00755)) and the non-parametric
  probability-distribution-matrix daisy chain ([Chatzistergos et al. 2017](https://arxiv.org/pdf/1702.06183)) remain
  disputed.
- **Synthetic-observer benchmark already exists.** [Chatzistergos, Krivova, Sundermann & Usoskin 2025 (A&A 699, A157)](https://www.aanda.org/articles/aa/full_html/2025/07/aa54896-25/aa54896-25.html)
  test cross-calibration techniques on synthetic observers; at least one common method systematically overestimates
  observer acuity and so inflates GN (full text not read; which method is unconfirmed). **The methods paper must be
  positioned against this.**
- **Closest probabilistic work:** [Mathieu et al. 2019](https://arxiv.org/abs/2009.09810) (multiplicative error model
  with excess zeros, modern network only) and [nonparametric monitoring](https://arxiv.org/pdf/2106.13535);
  [Velasco Herrera et al. 2022](https://link.springer.com/doi/10.1007/s11207-021-01926-x) (machine-learning "Bayesian"
  GSN built on Hoyt & Schatten data). No search found a joint latent-count model with daily resolution and lifetimes.
- **Maunder Minimum:** database cleaning of suspect spotless days ([Carrasco et al. 2021](https://academic.oup.com/mnras/article/504/4/5199/6249448);
  [Hayakawa et al. 2024](https://academic.oup.com/mnras/article/528/4/6280/7485915)); SN–ADF relation for the MM
  ([Carrasco et al. 2022](https://iopscience.iop.org/article/10.3847/1538-4357/ac7045)); Hoyt & Schatten read blanket
  statements as daily observations ([Neuhäuser et al. 2018](https://arxiv.org/pdf/1807.10241)).
- **Proxies:** annual ¹⁴C resolves cycles through the MM ([Brehm et al. 2021](https://doi.org/10.1038/s41561-020-00674-0) [nv];
  [Usoskin et al. 2021](https://cc.oulu.fi/~usoskin/personal/aa40711-21.pdf)); a fully Bayesian solar-modulation model
  already runs in NumPyro ([Schanner et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12592250/)).
- **Group lifetimes:** Gnevyshev–Waldmeier rule and parabolic decay ([Petrovay & van Driel-Gesztelyi 1997](https://arxiv.org/pdf/astro-ph/9706029));
  two group populations and large scatter ([2021 A&A study](https://arxiv.org/pdf/2106.04917);
  [Several populations of GN](https://arxiv.org/pdf/2011.05356)).

## 3. What the existing codebases do and what they throw away

| Codebase | Does well | Information lost / weaknesses |
|---|---|---|
| ADFSSN (`legacy/ADFSSN`) | Thresholds reference group areas (MSH); imprints the observer's missing-day mask on the reference; compares in matching cycle phases; EMD/L2 distances | Collapses days into fixed 20-day "months" (order lost); mask assumes missingness independent of activity (no MNAR zeros); "uncertainty" = weighted SD of the 50 best grid points, not a posterior; ignores overlaps; uses SILSO SN v2 for cycle phases and for the DTh activity levels, so calibration is not independent of SN |
| GRPSSN (`legacy/GRPSSN`) | Daily group-count distribution as fingerprint; smooths the distance surface | `LowLim=2` drops 0- and 1-group days (the most threshold-informative); still no temporal structure |
| GN-series MATLAB (`legacy/GN_Series_matlab`) | Overlap graph; ±2-day tolerance; robust weighted fits; k-shortest paths; multipath composite | ±2 days treated as simultaneous (no evolution model); per-edge linear vs proportional choice by R²; multiplicative error chaining grows with path length; 1/σ² path combination ignores shared edges; edges only forward in time |

Details and known bugs: [../03_legacy/](../03_legacy/README.md).

## 4. Three approaches considered

### Approach 1 — Joint latent-population state-space model (recommended core)

One generative model of the Sun and every observer:

- **Activity:** log emergence rate λ(t) as a quasi-periodic state-space Gaussian process with jittered period and a
  prior allowing grand minima. Cycle phase is latent, so SILSO is not needed.
- **Population:** N_t = number of groups the reference observer would count; survivors ~ Bin(N_{t−1}, φ),
  births ~ Pois(λ_t). If one class cannot reproduce RGO autocorrelation, use two area classes (small/short-lived,
  large/long-lived).
- **Detection:** P(y | N_t, θ_o(t)) built by thresholding real RGO/Debrecen areas (the legacy `np.greater(areas, Thr)`
  machinery), with a soft threshold, splitting rate, career drift (random walk) and a population prior with metadata
  covariates where available.
- **Reporting:** recording probability lower when y = 0 (MNAR zeros). A blanket statement "no spots in year X" has
  likelihood Π_t [1 − q_o(1 − P(y=0 | N_t))], which factorizes over days given N_t and is therefore one more emission
  term in the forward filter, plus a literal-vs-rhetorical mixture weight.
- **Anchor:** one reference epoch has fixed detection; GN is the **reference-equivalent count**. Early RGO and other
  reference segments are observers with their own parameters.
- **Proxy variant:** λ → open flux → ¹⁴C/¹⁰Be (differentiable carbon box model, e.g. `ticktack`) + aurorae, behind a
  data-lineage firewall.
- **Inference:** exact forward filter over N (K ≈ 41 states) with `jax.lax.scan`; estimated ~5×10⁸ flops and ~25 MB
  per likelihood evaluation (unbenchmarked); forward-filter backward-sample (FFBS) draws of N_{1:T} are the posterior GN
  ensemble; blocked Gibbs (FFBS ↔ vectorized NUTS per observer) or tempered SMC.

Pros: uses lifetimes, redundancy, all overlaps without path choice, prior+evidence and blanket statements in one
likelihood; uncertainty widens automatically where the overlap graph thins; ADF and backbone are limiting cases (ADF =
hard threshold, binary emissions, missing completely at random; backbone = single-path chained linking), which gives the
methods paper a unifying story.
Cons: most engineering; Gibbs mixing risk before 1874; misspecification of one bridge observer spreads downstream;
rests on two untested assumptions (birth-death + thinning reproduces RGO daily statistics; activity-conditional area
distribution is stationary across centuries); era-wide common drift is unidentifiable from sunspot data and must be
shown as prior sensitivity.

### Approach 2 — Global network calibration with a fingerprint likelihood ("ubercal" for observers)

Replace k-shortest paths with one concurrent fit of all observer parameters over the whole overlap graph (as in SDSS
ubercalibration, multi-link test equating, TrueSkill Through Time). Each overlapping pair contributes P(y_b | y_a, Δt)
derived from thresholded RGO, with lifetime-based decorrelation in Δt instead of a ±2-day window. Observers without
overlap connect through the ADF/GRP fingerprint written as a proper likelihood (P(active | N) = 1 − (1 − r)^N). Then
combine calibrated observers with a simple continuous smoother, propagating parameter draws into an ensemble.

Pros: incremental (reuses overlap matrices and ADF thresholding); seconds to run; easy to explain; free graph
diagnostics (effective resistance to the reference, bridge observers, path consistency); good first deliverable and
initializer for Approach 1.
Cons: pairwise likelihoods double-count days seen by many observers (overconfident unless corrected); redundancy only
pairwise; lifetimes only via the kernel; MNAR zeros and blanket statements bolted on; fingerprint term inherits ADF's
activity dependence, so the MM stays weak.

### Approach 3 — Amortized simulation-based inference (SBI) of observers

Turn the sliding-window search into neural posterior estimation: simulate RGO windows plus a stochastic emergence model
(including grand-minimum-like regimes via random deletion of whole groups or dynamo runs), apply the observer operator
(threshold, splitting, drift, real calendar, MNAR zeros, blanket statements), and train a sequence encoder on raw daily
records to output posteriors over observer parameters and local activity; amortized over all observers.

Pros: simulator can include hard-to-write effects; learned summaries keep temporal emergence patterns (GRPSSN's
direction); direct successor of the legacy code; standard calibration checks (SBC, TARP, coverage).
Cons: the MM is out of distribution; SBI misspecification fixes are not reliable and neural posteriors tend to be
overconfident; cannot carry the shared Sun without Approach 1 or 2; opaque to reviewers; JAX tooling still moving.

| | 1. Joint model | 2. Global network | 3. Amortized SBI |
|---|---|---|---|
| Group lifetimes | Full | Pairwise kernel only | Learned per observer |
| Shared Sun / redundancy | Full | Pairwise only | Weak alone |
| Uncertainty | Joint posterior ensemble | Needs correction | Per observer, if coverage passes |
| Maunder Minimum | Built in | Weak | Out-of-distribution risk |
| Effort | High | Low–medium | Medium |
| Main risk | Misspecification, mixing | Overconfidence | Extrapolation |

## 5. Recommendation → the 6-step roadmap

Approach 1 is the core of the series and the paper, reached through Approach 2, with Approach 3 as an optional module:

1. **Data foundation and viability checks**: does birth-death + thinning reproduce reference daily autocorrelation and
   zero fraction vs activity (1 vs 2 area classes)? Is the activity-conditional area distribution stationary?
2. **Blind benchmark** of synthetic observers on the real overlap graph, fitted with a model different from the
   generator; run legacy ADF/GRP/shortest-path on it.
3. **Approach 2** as baseline, diagnostic and initializer.
4. **Approach 1** with the FFBS posterior ensemble.
5. **Neural emission calibrator** (Approach 3) only for observers whose parametric emission fails posterior-predictive checks.
6. **Proxy-informed variant** last, behind the data-lineage firewall.

Specs: [../05_roadmap/](../05_roadmap/README.md).

## 6. Identifiability traps to frame the methods paper

From the cross-domain report: (1) absolute scale cannot be recovered from counts alone → define GN as the
reference-equivalent count; (2) drift shared by a whole era is indistinguishable from solar change → explicit drift
priors, sensitivity, proxies in the proxy variant; (3) chained links accumulate random error (variance ∝ graph
effective resistance) and add systematic error coherently → concurrent fitting plus graph diagnostics.

## 7. Side findings in the legacy code (not fixed in legacy/)

- ADFSSN commit `5898ab8` replaced `SC_SP_RG_DB_KM_group_areas_by_day.csv` with a header-less version;
  `scripts/SSN_ADF.py:97` reads it with `header=0`, so `REF_Dat['YEAR']` should raise a KeyError. The new version only
  prepends 17,695 all-NaN rows (1825-11-05 to 1874-04-16) to otherwise identical data.
- Reference area columns are sliced inconsistently in `scripts/SSN_ADF.py`: `shape[1]-2` (all 26 AREA columns) at lines
  116, 838, 2280, 2361 vs `shape[1]-3` (drops AREA26) at 252, 1208, 1397, 1809, 1840, 2161, 2192, 2285, 2300, 2377. It
  only matters on 26-group days (e.g. 1937-07-12), but scan and final scoring threshold slightly different group sets.
