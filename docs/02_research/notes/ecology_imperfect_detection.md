# Ecological models for counts with imperfect detection and observer effects, read as analogs for sunspot group number (GN) homogenization

Citation note: entries tagged **[nv]** are canonical references cited from domain knowledge. Their DOIs and claims were not re-checked in this session and should be confirmed before they go into the paper. Every other citation was checked this session against a search result or a fetched page, usually the abstract or index record rather than the full text.

Terms used throughout. "Reference" means the RGO/Debrecen catalogue, which records group identities and areas. "Observer o" means a historical observer. "Day t" is the time unit. In most ecological setups the Sun corresponds to a single "site", so days (or short windows) take over the role that spatial sites usually play.

---

## 1. Static and dynamic (open-population) N-mixture models: mapping, identifiability, and the role of replicate (overlapping) observers

### Takeaway
N-mixture models are the most direct analog. Latent group counts N_t are observed through observer-specific binomial thinning, and Dail-Madsen dynamics map onto group birth and decay. The 2018 identifiability literature is clear on one point: with unmarked counts alone, abundance and detection are reliable only in a relative sense, and absolute inference needs auxiliary data. For GN, the RGO/Debrecen area catalogue can be that auxiliary data. Defining the target as "the count the reference observer would have made" avoids the absolute-N identifiability problem altogether.

### Cited Findings
**(a) Core models**
- **Static N-mixture (Royle 2004).** For site i and visit j: N_i ~ Poisson(λ_i) (or negative binomial or zero-inflated Poisson), and y_ij | N_i ~ Binomial(N_i, p_ij). The population is assumed closed across the replicate visits. The likelihood is Π_i Σ_{N=max_j y_ij}^{K} [Π_j Bin(y_ij; N, p_ij)] Pois(N; λ_i), with a finite upper bound K on the sum. — [Royle 2004, Biometrics (USGS record)](https://www.usgs.gov/publications/n-mixture-models-estimating-population-size-spatially-replicated-counts); [Kéry & Royle, AHM vol. 1, ch. 6 (PDF)](https://www.mbr-pwrc.usgs.gov/workshops/ahm2023/06_N-mixture/Chap.%206%20AHM1%202016.pdf)
- **Dynamic N-mixture (Dail & Madsen 2011).** N_{i,1} ~ Pois(λ). Survivors are S_{i,t} ~ Bin(N_{i,t-1}, ω) and recruits are G_{i,t} ~ Pois(γ), or Pois(γ·N_{i,t-1}) in the autoregressive form. Then N_{i,t} = S_{i,t} + G_{i,t}, and y_{i,j,t} ~ Bin(N_{i,t}, p). Closure is needed only within a primary period. — [Dail & Madsen 2011, Biometrics, doi:10.1111/j.1541-0420.2010.01465.x](https://doi.org/10.1111/j.1541-0420.2010.01465.x) [nv]
- **Hostetler & Chandler 2015.** They note three problems with conventional Gaussian state-space models: they allow negative abundance, they treat false-positive and false-negative errors as equally likely, and some of their parameters are not estimable. They adopt the Dail-Madsen class, which attributes observation error to imperfect detection, and extend it to classical growth models (trend, Ricker, Gompertz), zero-inflation, random effects and forecasting. R and JAGS code is in the Ecological Archives supplement. — [Hostetler & Chandler 2015, Ecology 96:1713-1723 (abstract)](https://repository.si.edu/items/ce4eaaee-084c-4548-8beb-51d53304441c/full); [supplement and code](https://esapubs.org/archive/ecol/E096/153/)
- **Development history.** N-mixture models appeared in 2004 for point counts where capture-recapture was impractical. Population dynamics were added in 2011, and multiple states (age class, sex) in 2014. — [Madsen & Royle 2023, "A review of N-mixture models", WIREs Comp Stat, doi:10.1002/wics.1625](https://pubs.usgs.gov/publication/70274657)
- **Stage-structured open N-mixture for unmarked individuals.** This is the template for an area-class or age-structured group population. — [Zipkin et al. 2014, Ecology 95:22-29, doi:10.1890/13-1131.1](https://doi.org/10.1890/13-1131.1) [nv]

**(d) Identifiability and robustness**
- **Barker et al. 2018** argue from a capture-recapture perspective that not marking animals loses information that is critical for modelling N and p. A model in which p varies across repeat visits is overspecified. Even with constant p, alternative models in which abundance is non-identifiable, or not a parameter at all, produce data that are practically indistinguishable from N-mixture data, especially when data are sparse. Under constant p, reliable inference is mainly available for relative abundance. Absolute abundance needs auxiliary data that inform p. — [Barker, Schofield, Link & Sauer 2018, Biometrics 74:369-377](https://pubs.usgs.gov/publication/70198028)
- **Link et al. 2018** studied three violations: double counting, unmodelled temporal variation in N (closure violation), and unmodelled temporal variation in p. Small violations can produce large biases that goodness-of-fit tests often miss. They recommend auxiliary data such as marked recaptures. — [Link, Schofield, Barker & Sauer 2018, Ecology 99:1547-1551](https://pubs.usgs.gov/publication/70198662)
- **Kéry 2018** screened 137 bird data sets from 2,037 sites. Poisson and ZIP binomial N-mixture models showed almost no identifiability problems. Negative-binomial mixtures had problems in about a quarter of the data sets. Problems became more frequent with fewer sites, and covariates made no difference. He recommends always checking identifiability and using simpler models, integrated models or informative priors when problems appear. — [Kéry 2018, Ecology, doi:10.1002/ecy.2093 (summary via search)](https://doi.org/10.1002/ecy.2093)
- **Apparent conflict.** Kéry (2018) reports that estimability is usually fine, while Link et al. and Barker et al. (2018) report that the estimates are fragile. The two claims are about different things: numerical identifiability in one case, bias under misspecification in the other. A field test argues that N-mixture models can estimate abundance reliably (title claim; full text not read). — [Ornithological Applications 124(1): duab054](https://bioone.org/journals/ornithological-applications/volume-124/issue-1/duab054/N-mixture-models-estimate-abundance-reliably--A-field-test/10.1093/ornithapp/duab054.full)
- **Overdispersion.** Binomial N-mixture estimates are sensitive to overdispersion, so model fit must be checked. The `nmixgof` R package provides Pearson-residual c-hat statistics at the marginal, site-sum and observation levels for `unmarked::pcount` fits. — [Knape et al. 2018, MEE 9:2102-2114, doi:10.1111/2041-210X.13062](https://doi.org/10.1111/2041-210X.13062); [nmixgof manual](https://jknape.r-universe.dev/nmixgof/doc/manual.html)
- **Bias from unmodelled heterogeneity, with diagnostics and alternatives.** — [Duarte et al. 2018, Ecological Modelling 374:51-59, doi:10.1016/j.ecolmodel.2018.02.007](https://doi.org/10.1016/j.ecolmodel.2018.02.007) [nv]
- **Infinite or boundary abundance estimates and computational issues** in the N-mixture likelihood. — [Dennis, Morgan & Ridout 2015, Biometrics 71:237-246, doi:10.1111/biom.12246](https://doi.org/10.1111/biom.12246) [nv]
- **Non-replicated (single-visit) counts.** Separating N from p then relies on critical parametric assumptions. — [Knape & Korner-Nievergelt 2015, MEE 6:298-306, doi:10.1111/2041-210X.12329](https://doi.org/10.1111/2041-210X.12329) [nv]
- **Madsen & Royle 2023** list two main drawbacks: (i) abundance and detection are estimated jointly from unmarked counts, so they may not be identifiable; (ii) the likelihood becomes infeasible for large counts. They propose a normal-approximation likelihood that also works as a diagnostic for estimability. — [Madsen & Royle 2023](https://pubs.usgs.gov/publication/70274657)
- **Correlated detection across replicates.** In 396 eBird species-subregion data sets, N-mixture models with beta-binomial detection were often among the best-supported models. GLMMs were more stable but do not separate detection from abundance. — [Goldstein & de Valpine 2022, Sci. Rep. 12:12276](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9296480/)

**How replicates fix identifiability: the double-observer analog**
- **Double-observer design (Nichols et al. 2000).** A primary observer calls out every detection and a secondary observer records those plus any the primary missed; the two swap roles during the survey. Observer identity and species both affected detection. Combined detection exceeded 0.95. The authors recommend fixed-radius counts because observers differ in their effective detection radius. — [Nichols et al. 2000, Auk 117:393-408](https://pubs.usgs.gov/publication/5223914)

### Inferences
**(c) Mapping to GN**
- **Site:** the Sun on day t, or a short window of days. There is only one Sun, so the replication that N-mixture models normally get from many i.i.d. sites has to come from days. Days are autocorrelated, not i.i.d., which pushes the design toward dynamic (Dail-Madsen) models rather than static ones.
- **Visit or replicate:** the different observers who report on the same calendar day. Closure is nearly exact within a day. Nearby days act as "secondary occasions", linked through group survival.
- **Abundance N_t:** the latent number of groups on day t under the reference definition of a group.
- **Detection p_{o,t}:** observer o's probability of reporting a reference group. It depends on group area (threshold), seeing conditions and instrument, and may drift with career year.
- **Survival ω:** the daily probability that a group persists.
- **Recruitment γ_t:** the daily rate of group emergence, which follows the solar cycle.
- **Species:** optionally, area classes of groups. A multi-"species" N-mixture with class-specific detection is one tractable way to encode size-dependent detection (see section 6).

**Why the binomial assumption breaks for GN.** All observers miss the same small groups. That is the "heterogeneity in p across individuals" that produces large N-mixture biases (Link et al. 2018; Knape et al. 2018). A plain binomial N-mixture would therefore misattribute part of the shared small-group deficit to abundance.

**The identification algebra.** With Poisson N and binomial detection, one observer's counts are marginally Poisson(λp), so only the product λp is identified. For two observers on the same day, E[y1]·E[y2]/Cov(y1, y2) = λ. Abundance is identified from the covariance of simultaneous counts, which is how replicate surveys work, but only if the binomial model is exactly right. Shared misses of small groups inflate that covariance and so bias λ. This is the mechanism behind the Link (2018) fragility result, applied to GN.

**What the robustness literature means for GN.** It recommends auxiliary data that inform p (Barker; Link). In GN, the RGO/Debrecen catalogue gives each group's identity and area. That plays the role of marked-individual or double-observer data: one can simulate exactly what an observer with threshold A_o would have counted. The defensible strategy is to anchor detection to the reference catalogue. Abundance should then be defined as the reference-equivalent count, which is a relative-abundance target and the robust case in Barker et al. 2018, rather than an absolute "true N".

**Dail-Madsen dynamics as group birth and decay.** Constant ω implies geometric (memoryless) lifetimes. Sunspot group lifetimes scale with maximum area, so a single ω is misspecified. Two better options are a stage- or area-structured open model in the style of Zipkin 2014, or a mixture of survival classes. In both, the survival and emergence parameters can be estimated from the reference catalogue and held fixed or given tight priors for historical eras.

**Splitting and merging conventions.** These correspond to the "double counting" violation in Link et al. 2018, which has no place in pure binomial thinning. Observers who count more groups than the reference need an explicit over-count term. For example, each detected group could be reported as 1 + Poisson(s_o), with merging probabilities that rise with activity.

**Linking non-overlapping observers through dynamics.** Observers on nearby days can be linked through the dynamics. The information decays on the timescale of group lifetimes, so it is weak for gaps longer than a few weeks.

### Gaps
- I found no paper that applies N-mixture or Dail-Madsen models to solar or sunspot data. I did not search the solar-physics literature exhaustively, because other researchers cover it.
- I found no quantitative result on how many simultaneous replicate observers per closure period are needed for identifiability when detection is heterogeneous. Simulation in the GN setting will be needed.
- I did not verify whether Hostetler & Chandler's simulations quantified bias from misspecified survival (non-geometric lifetimes). Only the abstract was read.

---

## 2. Occupancy models with imperfect detection and false positives: spotless vs active days and under-reported zeros

### Takeaway
"Active vs spotless day" is an occupancy state. The Royle-Nichols model, in which detection of ≥1 item depends on abundance, is mathematically the same idea as the active-day-fraction (ADF) approach. False-positive occupancy models (Royle & Link 2006; Miller et al. 2011) handle observers who report groups on days the reference marks as spotless. Under-reported zeros are a reporting-process (preferential-sampling, missing-not-at-random) problem. Occupancy models do not fix that problem by themselves, but group persistence makes it partly identifiable.

### Cited Findings
**(a) Core models**
- **MacKenzie et al. 2002 site occupancy.** z_i ~ Bern(ψ) and y_ij | z_i ~ Bern(z_i·p_ij), with closure across repeat visits. Repeat visits separate non-detection from absence. — [MacKenzie et al. 2002, Ecology 83:2248-2255, doi:10.1890/0012-9658(2002)083[2248:ESORWD]2.0.CO;2](https://doi.org/10.1890/0012-9658(2002)083[2248:ESORWD]2.0.CO;2) [nv]
- **Royle-Nichols 2003.** Detection heterogeneity arises from variation in abundance: site-level detection is p_i = 1 − (1 − r)^{N_i}, with N_i ~ Pois(λ). This lets one estimate the abundance distribution from repeated presence-absence data, and implies ψ = 1 − e^{−λ}. It is implemented as `unmarked::occuRN`. — [Royle & Nichols 2003, Ecology 84:777-790](https://pubs.usgs.gov/publication/5224229); [occuRN docs](https://rdrr.io/pkg/unmarked/man/occuRN.html)
- **Royle & Link 2006 false-positive occupancy.** y | z=1 ~ Bern(p11) and y | z=0 ~ Bern(p10). Identifiability requires a constraint such as p11 > p10, because the likelihood is multimodal under label switching. — [Royle & Link 2006, Ecology 87:835-841, doi:10.1890/0012-9658(2006)87[835:GSOMAF]2.0.CO;2](https://doi.org/10.1890/0012-9658(2006)87[835:GSOMAF]2.0.CO;2) [nv]
- **Miller et al. 2011.** Models that handle false negatives and false positives together. In the "multiple detection methods" design, some detections come from a method assumed to have no false positives. In simulation these estimates were more accurate and precise than both Royle-Link and MacKenzie 2002. Implemented as `unmarked::occuFP`. — [Miller et al. 2011, Ecology 92:1422-1428 (PDF)](https://www.originalwisdom.com/wp-content/uploads/bsk-pdf-manager/2019/03/Miller-et-al_2011_improving-occupancy-estimation-when-two-types-of-observational-error-occur.pdf); [occuFP docs](https://rdrr.io/cran/unmarked/man/occuFP.html)
- **False-positive designs.** Site-confirmation versus observation-confirmation designs for false-positive occupancy. — [Chambert, Miller & Nichols 2015, Ecology 96:332-339, doi:10.1890/14-1507.1](https://doi.org/10.1890/14-1507.1) [nv]
- **Dynamic occupancy** (colonization and extinction between seasons), the template for transitions between active and spotless runs. — [MacKenzie et al. 2003, Ecology 84:2200-2207, doi:10.1890/02-3090](https://doi.org/10.1890/02-3090) [nv]

**Missing-not-at-random and preferential sampling**
- **Preferential sampling.** When survey locations or times depend on the latent state, model-based estimates are biased. Conn, Thorson & Johnson give diagnostics and model-based fixes that model the sampling process jointly with the state. Code is on Zenodo. — [Conn, Thorson & Johnson 2017, MEE 8:1535-1546 (NOAA repository)](https://repository.library.noaa.gov/view/noaa/52909); [code](https://zenodo.org/record/569035)
- **Complete checklists in eBird.** Almost 90% of eBird checklists are "complete", meaning all detected species are reported, so unreported species can be treated as zeros. Data with no completeness flag do not support that inference. — [Goldstein & de Valpine 2022](https://link.springer.com/10.1038/s41598-022-16368-z)
- **Detection and survey bias in presence-only data** (the analog of observers who record only active days). — [Dorazio 2014, Global Ecol. Biogeogr. 23:1472-1484, doi:10.1111/geb.12216](https://doi.org/10.1111/geb.12216) [nv]

### Inferences
**(c) Mapping to GN**
- **Site:** day t (or a window).
- **z_t:** 1 if the reference has ≥1 group.
- **Visit:** each observer's report on that day.
- **Detection:** P(observer reports ≥1 group | groups present). This is the active-day concept.
- **Royle-Nichols as ADF.** With N_t groups each detected independently with probability r_o, P(active report) = 1 − (1 − r_o)^{N_t}. With area-dependent detection, the expression becomes 1 − Π_{g∈day t}(1 − F_o(A_g)). That is the formal version of the ADF fingerprint: an observer's active-day fraction as a function of activity level is a Royle-Nichols detection curve. It is identified from the observer's data once the distribution of N_t (and of areas) at that activity level is known from the reference.
- **The binary channel is the robust one.** Royle-Nichols, and occupancy in general, uses only the presence/absence channel. That channel is less sensitive to splitting and merging conventions than the counts are, which is why the ADF approach is robust to counting conventions. The cost is lost information at high activity, where nearly every day is active (ψ → 1).
- **False positives:** the observer reports ≥1 group on a reference-spotless day. Causes include pores, misidentification, or the reference missing a short-lived group. The Miller 2011 "multiple methods" structure fits well if RGO/Debrecen is treated as the near-certain method on overlap days. In Royle & Link 2006 the p11 > p10 constraint is natural for GN.
- **Missing-not-at-random zeros.** Model the reporting process jointly: R_{o,t} ~ Bern(logit⁻¹(a_o + b_o·1[N_t > 0] + weather terms)). An observer with b_o > 0 under-reports spotless days. b_o is identified only where N_t is known independently, from overlapping observers or the reference. For isolated observers it is identified only through (i) a hierarchical prior over observers and (ii) group persistence: a gap inside a run of days with long-lived groups cannot be spotless, so the length distribution of reporting gaps compared with expected spotless runs informs b_o. This is the preferential-sampling fix of Conn et al. 2017 applied to time instead of space.
- **Completeness flag.** Historical records could be classified as "complete" (zeros are explicit) or "incomplete" (only active days recorded), analogous to eBird complete checklists. Only complete records would contribute zeros directly.

### Gaps
- I found no occupancy literature that treats an informative non-report whose probability depends on the occupancy state, with time-series persistence, as a single named model. The construction above is an inference.
- I did not verify whether false-positive occupancy models had been extended to false-positive N-mixture models with count over-reporting (splitting). The search found an arXiv "Nc-mixture occupancy model" ([arXiv:2304.02851](https://arxiv.org/pdf/2304.02851)), but I did not read its content.

---

## 3. Observer effects in North American Breeding Bird Survey (BBS) hierarchical models: observer random effects, novice effects, career drift

### Takeaway
The BBS model is a log-linear Poisson model with observer-route random effects, a first-year (novice) indicator and overdispersion. In other words it is a hierarchical, Bayesian version of the classic multiplicative k-factor. Its main lessons for GN are that leaving out observer effects biases trends, that first-year undercounting produces spurious increases, and that observer aging (hearing loss) degrades detection. All three map directly onto observer thresholds and career drift. Its limitation for GN is the multiplicative form: threshold detection is not multiplicative.

### Cited Findings
**(a) Core model (Link & Sauer 2002; Sauer & Link 2011)**
- **Model structure.** Counts Y_{i,j,t} on route i, observer-route combination j, year t follow Y ~ Poisson(λ_{i,j,t}), with log λ_{i,j,t} = S_i + β_i(t − t*) + γ_{i,t} + ω_j + η·I(j,t) + ε_{i,j,t}. S_i is the stratum intercept, β_i the trend, and γ_{i,t} the year effects. ω_j ~ N(0, σ²_ω) is the observer-route effect, η is the first-year effect (I = 1 in the observer's first year on the route), and ε is overdispersion. Fitted by MCMC. The equation is as given in the papers; I could not re-fetch the full-text PDF this session. — [Sauer & Link 2011, Auk 128:87-98 (USGS record)](https://pubs.usgs.gov/publication/70036906); [Link & Sauer 2002, Ecology 83:2832-2840, doi:10.1890/0012-9658(2002)083[2832:AHAOPC]2.0.CO;2](https://doi.org/10.1890/0012-9658(2002)083[2832:AHAOPC]2.0.CO;2) [nv for equation details]
- **Results.** Hierarchical trend estimates were generally more precise than route-regression estimates. The two methods disagreed substantially for 15 species. — [Sauer & Link 2011 (USGS record)](https://pubs.usgs.gov/publication/70036906)
- **Current implementation (bbsBayes2).** All model variants share "parameters and priors designed to adjust estimates for variations among observers, routes, and first-year observer start-up effects". The variants are slope, first-difference (a random walk on year effects), GAM, and GAM plus year effects (GAMYE). Each comes in hierarchical, spatial (iCAR) and non-hierarchical forms. Even the non-hierarchical form stays hierarchical over routes, observers and years. — [bbsBayes2 model documentation](https://bbsbayes.github.io/bbsBayes2/articles/models.html); [Smith & Edwards 2021, Ornithological Applications, doi:10.1093/ornithapp/duaa065](https://doi.org/10.1093/ornithapp/duaa065) [nv]
- **Model selection and cross-validation for BBS models.** — [Link et al. 2020, "Model selection for the North American Breeding Bird Survey"](https://pubs.usgs.gov/publication/70211831)

**Observer and novice effects**
- **Sauer, Peterjohn & Link 1994.** Observer differences affected counts for about half of 369 species, and later observers often counted more birds, consistent with observer quality improving over time. Leaving observers out as covariables gives an overly optimistic view of population trends. — [Sauer, Peterjohn & Link 1994, Auk 111:50-62](https://pubs.usgs.gov/publication/5222733); [Kendall et al. 1996 record (summarizing the 1994 result)](https://pubs.usgs.gov/publication/5223223)
- **Kendall, Peterjohn & Sauer 1996.** Removing each observer's first year on each route (1966-1991) lowered average unweighted trend estimates for 415 of 459 species (90%), significantly for 213 (46%). Once routes were weighted by precision, the differences were generally small. — [Kendall, Peterjohn & Sauer 1996, Auk 113:823-829](https://pubs.usgs.gov/publication/5223223); [PDF](https://sora.unm.edu/sites/default/files/journals/auk/v113n04/p0823-p0829.pdf)
- **Farmer et al. 2014 (career-end drift).** Older observers recorded fewer detections for 13 of 43 species in the Ontario Breeding Bird Atlas. In the BBS, detections declined with observer age for 4 of 6 vocalization-frequency groups, covering 59 of 64 species. The effect was strongest for high-pitched calls, consistent with hearing loss, though declines at other frequencies suggest other aging effects as well. Observer aging can bias published trends, and the authors recommend modelling observer age. — [Farmer et al. 2014, Ecology and Evolution (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4203298/)

### Inferences
**(c) Mapping to GN**
- **Route:** the Sun, a single route, so ω_j reduces to an observer effect ω_o.
- **Year effects γ_t:** solar activity, smoothed with a GAM or random walk.
- **η:** the novice effect in the first year or years of a career.
- **ε:** day-level overdispersion from seeing conditions.
- **Career drift:** ω_o could be generalized to ω_o(τ), a random walk or spline in career year τ. That covers both learning (Kelling 2015; Sauer 1994) and aging (Farmer 2014, the analog of eyesight decline). Instrument changes would be breakpoints in ω_o(τ).

**The multiplicative-form problem.** The BBS observer effect is a constant multiplicative factor. In GN terms it is a Bayesian hierarchical k-factor. A detection threshold produces a ratio between observer and reference that depends on activity: at low activity a larger share of groups are small. So ω_o should be replaced by a detection-function parameterization (threshold A_o and slope), with the BBS random-effect machinery kept for the observer-level parameters. That means a hierarchical prior over (A_o, slope), novice offsets, and drift.

**How BBS identifies observer effects, and the GN risk.** In BBS, observer effects are identified because many routes share year effects within a stratum. Successive observers on one route are separated through those shared year effects, even though they never overlap on that route. The GN analog of "other routes" is other contemporaneous observers. Where only one observer is active (for example, parts of the Maunder Minimum), ω_o is confounded with γ_t. Only two things then constrain it: the hierarchical prior, which is effectively shrinkage toward the population distribution of thresholds from well-calibrated observers, and smoothness priors on γ_t.

**Novice-effect lesson.** Kendall's result (90% of trends lowered by dropping first years) shows that a systematic within-career change, if unmodelled, turns directly into a spurious trend. Uncorrected early-career undercounting in long single-observer stretches would bias the secular GN trend upward in the same way.

### Gaps
- I could not retrieve the Sauer & Link 2011 full text (the server returned HTTP 500), so the exact priors (for example, whether ω uses a t distribution) and how observer-route combinations are indexed are unverified.
- I found no BBS paper that models observer effects as non-multiplicative or dependent on abundance, or that models within-observer drift as a continuous process. Farmer 2014 uses observer age as a covariate.
- Jiguet (2009, Bird Study) reportedly replicated a first-time observer effect in the French BBS. I did not verify this.

---

## 4. eBird and citizen-science observer-expertise models: estimating skill without overlap from an observer's own data distribution

### Takeaway
eBird work estimates observer skill from each observer's own data. Kelling 2015 uses species accumulation curves against effort. Johnston 2018 uses an expertise score, an effort-adjusted mean species count, as a detection covariate. Yu et al. 2010 treat expertise as a latent class. This is close to the ADF "fingerprint" idea. In both cases skill is identified only by conditioning on a shared estimate of the true state, which raises the circularity and extrapolation risks discussed below.

### Cited Findings
- **Kelling et al. 2015.** Species accumulation curves (species reported against time spent) differ among observers. Steeper curves mean higher detection rates, especially for hard-to-identify species. Accumulation rates rose with continued participation, i.e. observers learned. The authors propose the curves as an index of observer skill built from routinely submitted data. — [Kelling et al. 2015, PLOS ONE 10:e0139600 (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4599805/); [data on Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.2k27f)
- **Johnston et al. 2018.** An expertise score based on each observer's effort-adjusted average species count was used as a detection covariate in single-species occupancy models. Reporting rates increased with the score for 197 of 200 species. Including expertise improved fit and validation-set prediction. Its effect on detectability was about as large as the effect of observation duration. Differences in estimated occupancy were small in that example. — [Johnston et al. 2018, MEE, doi:10.1111/2041-210X.12838 (St Andrews record)](https://research-portal.st-andrews.ac.uk/en/publications/estimates-of-observer-expertise-improve-species-distributions-fro/); [data on Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.1f5r2)
- **Yu, Wong & Hutchinson 2010, Occupancy-Detection-Expertise (ODE) model.** A probabilistic model that adds birder expertise (expert vs novice) to occupancy-detection. It improves prediction of whether a species is observed at a site, can predict a birder's expertise from their checklist history, and identifies species that are hard for novices to detect. — [Yu, Wong & Hutchinson 2010, ICDM pp. 1157-1162](https://ieeexplore.ieee.org/document/5694101/authors); [DBLP](https://dblp1.uni-trier.de/rec/conf/icdm/YuWH10.html)
- **Johnston-style expertise correction** is described as usable in any regression model of species occurrence, occupancy, abundance or density. — [summary via search, St Andrews record](https://research-portal.st-andrews.ac.uk/en/publications/estimates-of-observer-expertise-improve-species-distributions-fro/)
- **Occupancy models designed for citizen-science data.** — [USGS: "Occupancy models for citizen-science data"](https://pubs.usgs.gov/publication/70202265) (abstract not read)

### Inferences
**(c) Mapping to GN**
- **eBird "checklist" → observer-day report.**
- **"Effort" (duration, distance) → observing conditions.** These are mostly unrecorded historically, except for instrument and aperture.
- **Species accumulation curve → fingerprint.** The analog is the observer's count distribution, or ADF, as a function of activity level. Possible fingerprints include P(y = 0 | activity), the minimum non-zero counts, or the slope of the observer's count against the reference count.
- **Expertise score → threshold A_o.**
- **ODE latent expertise class → mixture of observer types,** for example "large-aperture careful observer" versus "naked-eye/projection observer".

**Why the circularity matters.** In eBird, expertise is identified without direct pairwise overlap because many observers sample the same spatial cells and seasons, and the true occupancy of a cell is estimated from everyone. In GN, the analog of "cell" is "epoch", and the true activity level of an epoch is common to all observers on one Sun. A fingerprint therefore has to condition on an activity estimate, which itself comes from (other) observers. A joint model avoids the two-stage circularity of computing a fingerprint and then calibrating against it. This is the main argument for embedding the ADF fingerprint in a single hierarchical likelihood rather than using it as a separate pre-calibration step.

**Learning and drift.** Kelling 2015 found accumulation rates rose with participation. Fingerprints should therefore be allowed to vary within a career, with a time-varying A_o.

**Extrapolation risk.** eBird models extrapolate badly to habitats sampled only by a few observers. The GN analog is the Maunder Minimum, where activity levels lie outside the range of the reference catalogue. A fingerprint calibrated on modern activity levels must be extrapolated, and the area distribution at very low activity is uncertain.

### Gaps
- I did not read the full text of Kelling 2015 or Johnston 2018. The exact covariate adjustment used to compute the expertise score (GAM terms, random effects) is not verified here.
- I did not verify whether the ODE model allows novices to produce false positives as well as lower detection.
- A scalable multi-species occupancy paper exists ([arXiv:2206.08894, "Scaling multi-species occupancy models to large citizen science datasets"](https://arxiv.org/pdf/2206.08894)). I verified the title only.

---

## 5. Integrated population models (IPMs) and data integration: one joint likelihood for daily counts, blanket statements and proxies

### Takeaway
IPMs join independent data likelihoods through shared latent states and demographic parameters. For GN, daily counts, the RGO/Debrecen area catalogue (the analog of "demographic" or marked data), qualitative blanket statements, and cosmogenic or auroral proxies can each be a component tied to the same latent daily group population. Ecological work suggests that modest dependence between components has little effect, but data sets that literally share observations must not be entered twice.

### Cited Findings
- **Besbeas et al. 2002.** The prototype IPM: a joint likelihood formed as the product of a state-space census likelihood and a mark-recapture-recovery likelihood that share survival and productivity parameters. — [Besbeas, Freeman, Morgan & Catchpole 2002, Biometrics 58:540-547, doi:10.1111/j.0006-341X.2002.00540.x](https://doi.org/10.1111/j.0006-341X.2002.00540.x) [nv]
- **Schaub & Abadi 2011.** A review of IPMs as a framework for deeper insight into population dynamics. — [Schaub & Abadi 2011, J. Ornithol. 152:227-237, doi:10.1007/s10336-010-0632-8](https://doi.org/10.1007/s10336-010-0632-8) [nv for content]
- **Abadi et al. 2010.** In simulations, violating the independence assumption had minimal effect on parameter estimates. Analyzing all three data sets jointly gave a gain in precision and accuracy, especially with small samples. — [Abadi, Gimenez, Arlettaz & Schaub 2010, Ecology 91:7-14](https://boris.unibe.ch/5197)
- **Barraquand 2024 on independence.** The question is often framed wrongly. What matters is probabilistic independence, which does not follow automatically from collecting data on the same physical individuals. Some IPMs assume conditional rather than full independence. — [Barraquand 2024, "Independence in Integrated Population Models", arXiv:2411.01877](https://arxiv.org/abs/2411.01877)
- **Textbook.** Schaub & Kéry 2022, *Integrated Population Models: Theory and Ecological Applications with R and JAGS* (Academic Press). — [Bern IPM group page](https://www.cb.iee.unibe.ch/research/integrated_population_model/index_eng.html)
- **Kéry 2018** recommends integrated models and external information (informative priors, penalized likelihood) as remedies when N-mixture identifiability fails. — [Kéry 2018](https://doi.org/10.1002/ecy.2093)
- **Data-integration reviews for species distributions** (combining structured counts, presence-only and citizen-science data): [Isaac et al. 2020, TREE 35:56-67, doi:10.1016/j.tree.2019.08.006](https://doi.org/10.1016/j.tree.2019.08.006) [nv]; [Miller et al. 2019, MEE 10:22-37, doi:10.1111/2041-210X.13110](https://doi.org/10.1111/2041-210X.13110) [nv]; [Fletcher et al. 2019, Ecology 100:e02710, doi:10.1002/ecy.2710](https://doi.org/10.1002/ecy.2710) [nv]
- **Speed.** Marginalizing discrete latent states is the standard way to make Bayesian IPMs fast; see section 7. — [Yackulic et al. 2020](https://pubs.usgs.gov/publication/70209184)

### Inferences
**(c) Mapping GN data types onto IPM components, all sharing the latent N_t (or group emergence rate γ_t and area distribution)**
1. **Daily observer counts:** an N-mixture or area-thinned Poisson component, with observer detection functions.
2. **RGO/Debrecen catalogue (1874 onward):** an individual-level "demographic" component. It informs the group emergence rate, the distribution of maximum area given activity, and lifetime given area. Like mark-recapture data in an IPM, it fixes the parameters (detection by area, survival) that counts alone cannot identify.
3. **Blanket statements** (for example, "no spots seen for N years", "spots seldom seen"): censored or interval observations of the occupancy or count process over an interval. The likelihood contribution is P(all reported days spotless, or count ≤ c, over the interval | latent path), with an unknown "coverage" parameter for how thoroughly the writer actually observed.
4. **Proxies** (cosmogenic ¹⁴C/¹⁰Be, aurorae): index components with their own observation models, linked to smoothed activity (for example, a heliospheric-modulation transfer function). They work like the population index in an IPM. They constrain low-frequency activity in sparse eras without informing daily counts.

**Independence pitfalls specific to GN.**
- Do not include both an observer's daily counts and that same observer's published monthly or yearly means. That uses the same information twice.
- Compilations such as Hoyt & Schatten 1998 may contain duplicated or transcribed records. These should be deduplicated rather than entered as independent observers.
- Weather is shared across nearby observers, which produces correlated detection. That breaks conditional independence between "replicate" observers in the same region on the same day, so a day-by-region random effect is advisable.

**Dominance risk.** The RGO component is very large and will dominate shared parameters. If historical-era group area distributions differ (an activity dependence outside the reference range), the model will impose modern demographics. Component-wise posterior predictive checks are needed, along with era-specific parameters that have hierarchical priors.

### Gaps
- I did not read the full texts of Schaub & Abadi 2011 or the 2022 book. Specific guidance on weighting conflicting components (for example, power priors or cut-likelihood "modular" inference) was not retrieved and is a gap.
- I found no ecological IPM that integrates qualitative or "blanket" textual evidence. The censored-likelihood construction above is an inference.

---

## 6. Detection that depends on individual size or abundance (size-biased detection, distance-sampling-like detection functions): the analog of area-threshold detection

### Takeaway
Distance sampling gives the conceptual template. Detection probability is a function g(·) of a covariate, here group size. Ignoring that dependence biases the mean size of detected objects upward, and the standard remedy is to model detection as a function of size. For GN, the "distance" covariate is group area (log area). The decisive advantage is that the reference catalogue supplies the full area distribution of undetected groups, which distance sampling has to infer.

### Cited Findings
- **Size bias.** In line-transect sampling, small clusters are harder to see at large distances, so the observed mean cluster size is biased upward. That inflates abundance estimates of individuals obtained as group density × mean group size. Drummer & McDonald (1987, Biometrics) is the standard source and treats size bias as weighted sampling. — [Distance sampling workshop: group-size demo](https://workshops.distancesampling.org/standrews-2019/intro/practicals/group-size-demo.html); [NCSU repository report on the Drummer-McDonald bivariate sighting function](https://repository.lib.ncsu.edu/server/api/core/bitstreams/dfa3d8be-9243-4eee-ad74-8e6889a5d3a3/content)
- **Standard remedy and its limits.** The standard fix is to put cluster size into the detection function (for example, `~size` in Distance/`ds`) and derive abundance with Horvitz-Thompson-type estimators. Size-as-covariate models can fail to converge when cluster sizes span several orders of magnitude. Regressing log cluster size on detection probability is an alternative. — [Distance online course, practical 9](https://distancesampling.org/online-course/09-clusters/R-prac/Pr9-instructions.html); [Distance: remedying size bias for dolphin surveys](https://distancesampling.org/Distance/articles/web-only/groupsize/Remedy-size-bias-for-dolphin-surveys.html); [practical 9 solution](https://distancesampling.org/online-course/09-clusters/R-prac/Prac9_solution.html)
- **Abundance-dependent detection.** Royle-Nichols: P(detect ≥1) = 1 − (1 − r)^N. — [Royle & Nichols 2003](https://pubs.usgs.gov/publication/5224229)
- **Hierarchical distance sampling** (an abundance model plus a distance-based detection function, fitted to binned counts): [Royle, Dawson & Bates 2004, Ecology 85:1591-1597, doi:10.1890/03-3127](https://doi.org/10.1890/03-3127) [nv]. It is implemented in `unmarked` (distsamp, gdistsamp) and in spAbundance. — [spAbundance paper (arXiv)](https://arxiv.org/html/2310.19446v2)
- **Observer-specific effective detection radius.** Observers differ in effective detection radius, which is why Nichols et al. recommend fixed-radius counts. This is the direct analog of observer-specific area thresholds. — [Nichols et al. 2000](https://pubs.usgs.gov/publication/5223914)

### Inferences
**(c) Mapping to GN**
- **Distance x → log group area.** Detection is monotone increasing in area rather than decreasing in distance.
- **g(0) = 1 → very large groups always detected.** This is plausible for any telescopic observer.
- **Detection function → F_o(A).** For example, logit F_o(A) = κ_o(log A − log A_o), possibly with seeing-day effects and drift in A_o(τ).
- **Cluster size → number of spots or area within a group.**

**Tractable likelihood.** Discretize area into bins b. Let N_{t,b} ~ Pois(λ_t·π_b(activity)), with π_b estimated from RGO/Debrecen, and y_{o,t} = Σ_b Binomial(N_{t,b}, F_o(A_b)) (+ splitting). By Poisson thinning, a single observer's marginal count is Poisson(λ_t·Σ_b π_b F_o(A_b)) or negative binomial, so no latent summation is needed. Simultaneous observers share the N_{t,b}. Because bins are independent Poisson, the joint likelihood for overlapping observers can be written in closed form, or computed with a small per-bin enumeration. This "multi-species N-mixture with known relative bin abundances" removes the binomial-heterogeneity bias described in section 1, because heterogeneity is now modelled explicitly.

**Days with reference data.** On overlap days with reference data the group list and areas are known. y_{o,t} then follows an exact Poisson-binomial distribution with probabilities {F_o(A_g)}. The model becomes a logistic-regression-like "distance sampling with known distances", which is strongly identified. This is the empirical core of threshold calibration.

**Size-range convergence issue.** The convergence warning about sizes spanning orders of magnitude applies directly: sunspot group areas range from a few to thousands of millionths of a hemisphere. Use log area, and put weakly informative priors on the slope κ_o.

**Two open issues.**
- The area distribution π_b may depend on activity level. Smaller groups dominate at low activity. Area distributions in sparse eras must therefore be extrapolated, which is the main source of uncertainty for the Maunder Minimum.
- The bias goes in the same direction as distance-sampling size bias. Groups that historical observers detected were larger on average than the reference population. Any per-group quantity inferred from historical data, such as hemispheric asymmetry or latitude, inherits that selection unless it is reweighted.

### Gaps
- I found no ecology paper that estimates a detection function where the full covariate distribution of undetected individuals is known from an external census. The closest analogs are fisheries gear-selectivity curves fitted against known length compositions, which I did not research.
- I did not read Drummer & McDonald 1987 directly; the findings above come from secondary course material.

---

## 7. Cross-cutting: scalability and software (unmarked, spOccupancy/spAbundance, NIMBLE, Stan, NumPyro)

### Takeaway
Every ecological package marginalizes the discrete latent abundance with a truncated sum, or a forward algorithm for dynamic models. Because GN counts are small (tens), exact marginalization is cheap. A daily Dail-Madsen-type forward recursion over roughly 150,000 days is feasible in JAX with `lax.scan`. The main computational risk is not size but mixing and identifiability, so prototyping in `unmarked` or NIMBLE for cross-checks is valuable.

### Cited Findings
- **unmarked.** Fits hierarchical abundance and occurrence models for point counts, site occupancy, distance sampling, removal and double-observer sampling, with covariates on state and observation processes. Functions include `pcount` (Royle 2004), `pcountOpen` (Dail-Madsen and Hostetler-Chandler dynamics), `occuRN`, `occuFP`, `distsamp`. — [Kellner et al. 2023, "The unmarked R package: Twelve years of advances…", MEE 14:1408-1415, doi:10.1111/2041-210X.14123](https://cran.ms.unimelb.edu.au/web/packages/unmarked/citation.html); [occuFP](https://rdrr.io/cran/unmarked/man/occuFP.html); [occuRN](https://rdrr.io/pkg/unmarked/man/occuRN.html)
- **nmixgof.** Goodness-of-fit and overdispersion checks for `unmarked` N-mixture fits. — [nmixgof](https://jknape.r-universe.dev/nmixgof/doc/manual.html)
- **spAbundance (Doser et al. 2024, MEE 15:1024-1033).** Bayesian single- and multi-species spatial N-mixture, hierarchical distance sampling and GLMMs, using nearest-neighbour Gaussian processes and latent factors. — [arXiv version](https://arxiv.org/html/2310.19446v2); [CRAN](https://cloud.r-project.org/web/packages/spAbundance/index.html)
- **spOccupancy** (the companion package for occupancy). — [Doser et al. 2022, MEE 13:1670-1678, doi:10.1111/2041-210X.13897](https://doi.org/10.1111/2041-210X.13897) [nv]
- **NIMBLE and nimbleEcology.** nimbleEcology provides marginalized distributions for N-mixture, occupancy and hidden-Markov capture-recapture models. Goldstein & de Valpine 2022 supply faster N-mixture likelihood code and a reparameterization for interpreting unstable estimates. — [nimbleEcology citation](https://cran.r-project.org/web/packages/nimbleEcology/citation.html); [Goldstein & de Valpine 2022 (DOAJ)](https://doaj.org/article/e4fcfb9f6af344b19158af8e7031e246); [de Valpine et al. 2017, JCGS 26:403-413, doi:10.1080/10618600.2016.1172487](https://doi.org/10.1080/10618600.2016.1172487) [nv]
- **Marginalization speedups (Yackulic et al. 2020).** Marginalizing discrete latent states made Bayesian population models 5 to more than 1000 times faster than code that samples the discrete states, with parameter inferences close to the discrete versions. The latent states (for example, abundance) can be recovered afterwards. Stan implementations of marginalized models are reported as especially efficient. — [Yackulic et al. 2020, Ecological Applications 30:e02112](https://pubs.usgs.gov/publication/70209184)
- **Large counts.** The N-mixture likelihood becomes infeasible for large counts, and Madsen & Royle propose a normal-approximation likelihood. — [Madsen & Royle 2023](https://pubs.usgs.gov/publication/70274657)
- **Recent computational preprints (titles verified only):**
  - "Investigating the efficiency of marginalising over discrete parameters in Bayesian computations" — [arXiv:2204.06313](https://arxiv.org/pdf/2204.06313)
  - "Sequential reduction for discrete latent variables in ecological and evolutionary models using RTMB" (2026) — [arXiv:2609.13333](https://arxiv.org/pdf/2609.13333)
  - "Fast and flexible Bayesian Jolly-Seber models" — [bioRxiv 2023.05.25.542303](https://www.biorxiv.org/content/10.1101/2023.05.25.542303.full.pdf)
- **ubms** (a Stan back-end with `unmarked`-style syntax). — [Kellner et al. 2022, MEE, doi:10.1111/2041-210X.13777](https://doi.org/10.1111/2041-210X.13777) [nv]

### Inferences
**JAX/NumPyro design**
- NUTS cannot sample discrete N_t. Marginalize explicitly:
  - static or within-day: `logsumexp` over N = max(y)…K, with K ≈ 50-80 groups;
  - dynamic: an HMM-style forward filter over N_t with a K×K transition matrix (binomial survival convolved with Poisson recruitment), implemented with `jax.lax.scan`.
- **Cost.** About 1.5×10⁵ days × K² (~4×10³) ≈ 6×10⁸ operations per likelihood evaluation, which is feasible on a GPU. Building day-specific transition matrices from γ_t is vectorizable.
- **Recovering N_t.** Use forward-filtering backward-sampling after MCMC, as Yackulic et al. describe for recovering discrete states, to produce the posterior GN ensemble.
- **Area-binned thinned-Poisson formulation (section 6).** This removes most latent summation, because single-observer days have closed-form marginals. Only multi-observer days need joint enumeration, so it is the most scalable option.
- **Validation path.**
  - Simulate observers by applying thresholds to the RGO catalogue, a "known-truth" design akin to the ecological field tests.
  - Fit with `unmarked::pcountOpen` and nimbleEcology on subsets to cross-check the NumPyro likelihood.
  - Run `nmixgof`-style overdispersion diagnostics and component-wise posterior predictive checks.

### Gaps
- I did not verify NumPyro-specific documentation for enumerating discrete latents (`funsor` / `config_enumerate`). Enumeration requires finite support, so Poisson-distributed N_t must be explicitly truncated in any case.
- I found no published benchmark of N-mixture or Dail-Madsen models in JAX or NumPyro.
