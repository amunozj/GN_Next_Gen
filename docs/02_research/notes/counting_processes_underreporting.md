# Counting processes with lifetimes and imperfect, under-reported observation: statistical analogs for homogenizing the sunspot group number (GN)

Notation used throughout the Inferences sections (this is my own notation, not taken from a source):
- `N_t` = latent number of groups on the visible disk on day t (the "occupancy").
- `lambda(t)` = emergence rate (groups/day, whole Sun or visible hemisphere), which varies with the ~11-yr cycle.
- `L_g`, `a_g(u)` = lifetime and area trajectory (at age u) of group g. Marks come from the RGO/Debrecen catalogue.
- `Y_{o,t}` = integer count reported by observer o on day t. `R_{o,t}` = indicator that o recorded anything on day t.
- `pi_o(a)` = probability that observer o detects a group of apparent area a (acuity / completeness function). ADF uses a hard threshold `pi_o(a) = 1{a > S_o}`.

Equations in the "Inferences" sub-sections are my derivations or mappings, built on the cited results. They are not quotes.

---

## Q1. INAR(1)/INARMA with binomial thinning, and hidden or under-reported INAR: how do survival, innovation and observer detection map to GN?

### Takeaway
A Poisson INAR(1), `N_t = alpha o N_{t-1} + eps_t` with `eps_t ~ Poisson(lambda_t)`, is a discrete-time immigration-death process with geometric "visible lifetimes". Adding a second binomial thinning `Y_t | N_t ~ Bin(N_t, p_o)` gives the Fernández-Fontelo hidden-INAR under-reporting model. Its likelihood is an HMM forward recursion on a truncated integer state space, which is cheap for GN because daily counts are small. Bracher (2018) showed the critical pitfall: in this model the reporting probability is identified only through the assumed lag structure of the latent process. For sunspots that structure is wrong, because lifetimes are non-geometric and detection of a given group persists from day to day. So p_o must be anchored by the RGO/Debrecen reference and by overlaps, not by the observer's own autocorrelation.

### Cited Findings
- **Binomial thinning.** It is defined as `alpha o X := sum_{i=1}^X Y_i` with i.i.d. Bernoulli(alpha) `Y_i` independent of X (Steutel & van Harn 1979). Its moments are `E[alpha o X] = alpha mu_X` and `Var[alpha o X] = alpha^2 sigma_X^2 + alpha(1-alpha) mu_X`. INAR(1) is the recursion "thinned previous value plus innovation". — [Weiß 2008, "Thinning operations for modeling time series of counts—a survey", AStA](https://dx.doi.org/10.1007/s10182-008-0072-3) (definition and moments as summarized in search results; the exact page was not verified)
- **Survey of thinning operators.** Weiß's 2008 survey reviews thinning operators beyond binomial thinning. It notes that binomial thinning suits Poisson counts but is awkward for other marginal distributions. — [Weiß 2008](https://dx.doi.org/10.1007/s10182-008-0072-3)
- **Review of thinning-based models.** Scotto, Weiß & Gouveia (2015), *Statistical Modelling* 15(6):590–618, review thinning-based models as discrete counterparts of ARMA and bilinear models. — [Statistical Modelling abstract](https://statmod.org/smij/Vol15/Iss6/Scotto/Abstract.html)
- **Canonical INAR(1) reference.** Al-Osh & Alzaid, "First-order integer-valued autoregressive (INAR(1)) process", *J. Time Series Anal.* 8(3):261–275 (1987). — [Citation as given in Springer article](https://link.springer.com/article/10.1007/BF02677403); [JTSA bibliography](https://ftp.math.utah.edu/pub/tex/bib/jtimeseranal.pdf)
- **Recent extensions (2026).**
  - A "modulating operator" generalizes binomial and negative-binomial thinning and is applied to INAR(1), INMA(1) and minification processes (Kella, Löpker & Weiß). — [revision PDF](https://offerkella.huji.ac.il/sites/default/files/ofer.kella/files/revision.pdf)
  - A survey covers INARMA models for count random fields. — [arXiv 2605.26888](https://arxiv.org/pdf/2605.26888)
- **Hidden INAR for under-reporting.** Fernández-Fontelo, Cabaña, Puig & Moriña (2016), "Under-reported data analysis with INAR-hidden Markov chains", *Statistics in Medicine* 35:4875–4890, use hidden INAR processes to estimate reporting probabilities for traditionally under-reported occupational and environmental diseases. — [UBC seminar abstract](https://www.stat.ubc.ca/sites/default/files/abstract-22jun2017.pdf); [Bracher comment, which cites it](https://arxiv.org/pdf/1812.06688)
- **Bracher's critique (2018).** A Poisson INAR(1) with time-homogeneous under-reporting "can be expressed equivalently as a completely observed INAR(∞) model with a geometric lag structure". Hence "estimated reporting probabilities depend on the assumed lag structure of the latent process". The authors published a reply, which I did not read. — [Bracher 2018, arXiv 1812.06688](https://arxiv.org/abs/1812.06688)
- **Follow-up applications by the same group.**
  - Fernández-Fontelo et al. (2019), "Untangling serially dependent underreported count data for gender-based violence", *Stat. Med.* 38(22):4404–4422. — [UAB listing](https://webs.uab.cat/advancedstochasticmodelling/biblio/untangling-serially-dependent-underreported-count-data-gender%e2%80%90based-violence/)
  - Fernández-Fontelo et al. (2020), *PLoS ONE* 15(12):e0242956. The hidden layer is a Po-INAR(1) "regeneration process with Poisson immigration". Innovations have a time-dependent expectation fed by an SIR system. The under-reporting parameter is allowed to vary in time (seasonality or trend). Estimation is by maximum likelihood. — [arXiv 2008.00262](https://arxiv.org/pdf/2008.00262)
- **Standard broad reference.** Weiß's book is cited as the standard broad review of INAR models. — [arXiv 2310.10798](https://arxiv.org/pdf/2310.10798)

### Inferences
**(a) Model and mapping to GN**
- **Latent layer.** `N_t = alpha o N_{t-1} + eps_t`, with `eps_t ~ Poisson(lambda_t)`.
  - alpha is the daily probability that a group visible on day t−1 is still visible on day t. It combines intrinsic decay with rotation off the west limb.
  - `eps_t` is the inflow: groups emerging on the visible disk plus groups rotating onto the east limb. The east-limb inflow is not a true birth, but in a stationary state it is Poisson with rate proportional to the far-side population.
  - The implied visible residence time is geometric with mean `1/(1-alpha)`. The stationary marginal is Poisson with mean `lambda/(1-alpha)`.
- **Observation layer (the second thinning).**
  - Homogeneous version: `Y_{o,t} | N_t ~ Bin(N_t, p_o)`.
  - Area-dependent version: `Y_{o,t} | groups ~ Poisson-binomial({pi_o(a_g(t))})`.
  - Splitting and merging can be added as a further compound step (see Q3).

**(b) Identifiability: the key result for GN (my derivation)**
- Binomial thinning of a Poisson variable is again Poisson. So marginally `Y_{o,t} ~ Poisson(p_o lambda/(1-alpha))`, and from one observer's marginals only the product `p_o * lambda` is identified.
- Under day-to-day independent detection, `Cov(Y_t, Y_{t+k}) = p^2 alpha^k mu` and `Var(Y_t) = p mu`. Hence the observed ACF is `rho_Y(k) = p alpha^k` for k ≥ 1.
- p is therefore identified only by the "drop" between lag 0 and lag 1 relative to the geometric decay. This is exactly the channel that Bracher showed an INAR(∞) with geometric lags can mimic.
- For sunspots the channel is broken twice:
  1. Detection of a given group is persistent rather than independent day to day: a group large enough to be seen today is very likely seen tomorrow. The observer's series then behaves like a fully observed immigration-death process with a lower arrival rate, and no lag-0/lag-1 drop appears.
  2. Lifetimes are heavy-tailed (Gnevyshev-Waldmeier rule plus a lognormal-like area distribution), not geometric.
- Conclusion: p_o (or S_o) must come from overlap with the reference, or with observers already chained to it, never from a single observer's dynamics.

**(c) Overdispersion**
- Poisson INAR(1) is equidispersed. Real daily GN is overdispersed because lambda(t) varies with the cycle and because emergence clusters (activity nests).
- Remedies: NB or compound-Poisson innovations, or a random-effect lambda(t). Note the interplay with thinning: `Var(Y) = p^2 Var(N) + p(1-p) E N`, so the dispersion index is `D_Y = 1 + p (D_N - 1)`.
- The dispersion index shrinks toward 1 as acuity falls. This is a weak, model-dependent extra handle on p when D_N is known from RGO.

**(d) Non-geometric lifetimes**
- Use INAR(p)/INARMA, or better an age-structured multi-compartment INAR: `N_t^{(a+1)} = alpha_a o N_{t-1}^{(a)}`, `N_t^{(0)} ~ Poisson(lambda_t)`, `N_t = sum_a N_t^{(a)}` (plus a visibility geometry).
- This is the discrete-time M/G/∞ (see Q2). It also lets detection depend on age, through the area trajectory.

**(e) Likelihood and scalability**
- The scalar INAR(1) plus binomial observers is an HMM on `{0..K}`, with K ≈ 40–60 since daily group counts rarely exceed ~30.
- The transition is `P(j|i) = sum_k Bin(k; i, alpha) Pois(j-k; lambda_t)`. Emissions are products of binomials over all observers present that day, assuming conditional independence. Days with no observers have no emission.
- Cost is `O(T K^2)` with T ≈ 1.5×10^5 days (1610–2025), about 4×10^8 flops per likelihood evaluation. This is easy in JAX `lax.scan`, can be differentiated for NUTS, and allows forward-filter backward-sample draws of N_t for the posterior GN ensemble.
- The age-structured version has a vector state and no longer fits a scalar HMM. Use PAL (Q5), particle filters, or the pairwise-Poisson composite likelihood (Q2).

### Gaps
- McKenzie (1985) and the Weiß 2018 book: I did not retrieve primary URLs, so their bibliographic details are unverified.
- Fernández-Fontelo et al.'s reply to Bracher: not read, so their counter-arguments are unknown.
- I found no application of INAR or hidden-INAR models to sunspot data.
- I found no published treatment of hidden INAR where detection persists per individual (correlated thinning across days). This is the structurally correct case for GN.

---

## Q2. M/G/∞ queue and immigration-death inference from occupancy counts alone

### Takeaway
Estimating λ and the service-time distribution G of an M/G/∞ queue from "number of busy servers" sampled on a grid is a mature nonparametric literature: Bingham & Pitts 1999; Hall & Park 2004; Goldenshluger 2016 and 2018; Goldenshluger & Koops 2019 for nonhomogeneous arrivals; and the discrete-time GI/G/∞ case. It is almost exactly "infer emergence rate and lifetime distribution from daily group counts". The central identity is `1 − G(t) = −R'(t)/λ`, where R is the autocovariance of occupancy. The ADF method corresponds to this literature's "busy-period indicator" observation scheme. For GN the service kernel must include rotation (visibility windows and ~27-day recurrence), and nonparametric rates are slow. So G should be learned from RGO and then held fixed for historical observers.

### Cited Findings
- **Bingham & Pitts (1999).** "Non-parametric estimation for the M/G/∞ queue", *Ann. Inst. Statist. Math.* 51:71–97. They estimate λ and G from queue-length data by exploiting the link between queue-length correlation and the stationary-excess distribution, and give standard time-series results for the estimators. The motivation was testing departures of G from exponentiality. — [AISM page](https://www.ism.ac.jp/editsec/aism/51/5.html); [Springer](https://link.springer.com/article/10.1023/A:1003831118254)
- **Goldenshluger (2016), *Adv. Appl. Probab.* 48(4).** — [ar5iv full text](https://ar5iv.arxiv.org/html/1508.00076); [Project Euclid](https://projecteuclid.org/journals/advances-in-applied-probability/volume-48/issue-4/Nonparametric-estimation-of-the-service-time-distribution-in-the-M/aap/1482548431.full)
  - **Observation scheme.** X(t) (busy servers) is observed at `t_i = i delta`, i = 1..n.
  - **Key identities.** X(t) is marginally Poisson(ρ), with ρ = λ/μ. The correlation function is `H(t) = mu * int_t^inf [1 − G(x)] dx`, the covariance is `R(t) = rho H(t)`, and therefore `1 − G(t) = −(1/λ) R'(t)`.
  - **Estimator.** Local-polynomial weights applied to the sample autocovariances.
  - **Rates.** Maximal risk is of order `T^{−β/(2β+2)}`, and a matching lower bound shows rate optimality. The bound is non-uniform in x₀, and for large x₀ the trivial estimator 1 can be better. With continuous observation, λ can be estimated from jump counts with MSE λ/T.
- **Three observation schemes.** The literature distinguishes (i) arrival and departure epochs without matching, (ii) queue length on a grid, and (iii) the busy-period indicator `1(X(t) > 0)`. Schemes (i) and (ii) are essentially equivalent. — [ar5iv 1508.00076](https://ar5iv.arxiv.org/html/1508.00076)
- **Hall & Park (2004).** *JRSS-B* 66(4):861–875. Only busy-period durations, i.e. concatenations of service times, are available. They recast recovery of the service-time distribution as deconvolution, using an empirical-characteristic-function, kernel-type estimator with conventional bandwidth choice. Unimodal versus multimodal service densities can indicate one population versus several. — [IDEAS/RePEc](https://ideas.repec.org/a/bla/jorssb/v66y2004i4p861-875.html); [ANU repository](https://openresearch-repository.anu.edu.au/items/4d6e0c24-b172-42a3-a442-557fe126b919)
- **Goldenshluger (2018), *Bernoulli*, "The M/G/∞ estimation problem revisited".** Estimates G from arrival and departure data without identifying customers, with exact non-asymptotic MSE expressions. — [Project Euclid](https://projecteuclid.org/euclid.bj/1522051217)
- **Goldenshluger & Koops (2019), *Stochastic Systems*.** Nonhomogeneous Poisson arrivals, with only the number of busy servers observed. The problem is reduced to deconvolution, solved with Laplace transforms and regularizing kernels. — [ar5iv 1805.10353](https://ar5iv.arxiv.org/html/1805.10353); [INFORMS PDF](https://pubsonline.informs.org/doi/pdf/10.1287/stsy.2018.0026)
- **Further references** (as listed in Goldenshluger's bibliography). — [ar5iv 1508.00076](https://ar5iv.arxiv.org/html/1508.00076)
  - Brown (1970), *Ann. Math. Statist.* 41:651–654 (pairs each departure with the nearest preceding arrival).
  - Pickands & Stine (1997), *Biometrika* 84:295–308, "Estimation for an M/G/∞ queue with incomplete information".
  - Grübel & Wegener (2011), *J. Appl. Probab.* 48:131–144, "Matchmaking and testing for exponentiality in the M/G/∞ queue".
  - Schweer & Wichelhaus (2014), arXiv:1409.1386, "Nonparametric estimation of the service time distribution in the discrete-time GI/G/∞ queue with partial information".
  - Blanghaps, Nov & Weiss (2013), *J. Appl. Probab.* 50:1044–1056, "Sojourn time estimation in an M/G/∞ queue with partial information".
- **Discrete-time networks.** Nonparametric inference also exists for networks of Geom^X/G/∞ queues in discrete time, i.e. batch (clustered) arrivals. I saw the title only. — [Adv. Appl. Probab. (Cambridge)](https://www.cambridge.org/core/journals/advances-in-applied-probability/article/nonparametric-inference-for-queueing-networks-of-geomx-g-queues-in-discrete-time/662259D2C455B6A46B67FD4151F644BE)

### Inferences
**(a) Model for GN**
- Treat groups as a Poisson (or Cox) process of emergences in time and Carrington longitude with i.i.d. marks `(L_g, a_g(·), phi_g)`.
- The visibility kernel is `h_g(u) = 1{u < L_g} · v(phi_g + Omega u)`, where v is the visible-hemisphere indicator. The occupancy is `N(t) = sum_g h_g(t − tau_g)`, a 0/1-kernel Poisson shot noise.
- Campbell's theorem then gives `E N = lambda E int h` and `Cov(N(t), N(t+tau)) = lambda int E[h(u) h(u+tau)] du`. This generalizes Goldenshluger's identity, which is recovered when `h = 1{u < L}`.
- The rotational geometry produces a predicted signature in the daily GN autocovariance: a fast decay over ~13.5 days, a near-zero trough, and a recurrence bump near ~27 days for long-lived groups.

**(b) Observer thinning stays within the class**
- If observer o detects group g at age u with probability `pi_o(a_g(u))`, the detected occupancy is again a Poisson shot noise with kernel `h_g(u) pi_o(a_g(u))`.
- For a hard threshold, it is an M/G/∞ with thinned arrival rate `lambda q_o` and a service distribution equal to the law of the above-threshold portion of each group's life.
- Two effects compete: (i) selection toward large, hence long-lived, groups (the Gnevyshev-Waldmeier rule), and (ii) truncation of each group's visible time.
- The observer's own ACF shape therefore carries information about S_o, separately from the count level. This is a testable, overlap-free identification channel, but it depends on the latent model.

**(c) ADF is scheme (iii)**
- ADF uses only `1{Y_{o,t} > 0}`, which is the busy-period indicator scheme. Under Poisson occupancy, `P(Y_{o,t} = 0) = exp(−Lambda_o(t))`, so the active-day fraction is a monotone transform of the detected mean.
- Overdispersion, from cycle variation and clustering, breaks this simple inversion. Monthly ADF partly absorbs it.
- Hall & Park's busy- and idle-period deconvolution suggests that the run-length distributions of spotless and active days near solar minimum carry lifetime and acuity information. These are under-used statistics for GN.

**(d) Separation of time scales**
- Lifetimes are days to weeks, while lambda(t) varies over years. Local stationarity within a month is therefore a good approximation, and Goldenshluger & Koops' nonhomogeneous deconvolution is available if needed.

**(e) Tractable likelihoods**
1. **Pairwise composite likelihood (closed form; my derivation from the Poisson marking/colouring theorem).** For any two days t and t+τ, classify each group as seen on both days, only the first, or only the second. These classes are independent Poisson counts. So `(N_t, N_{t+τ}) = (A + C, B + C)` is bivariate Poisson with `E[C] = R(tau)`. The same holds for any pair of observers, or observer-days, with the detection kernels inserted. This gives an exact closed-form pairwise likelihood that scales as O(T × number of lags) and is easy in JAX.
2. **Whittle or Gaussian likelihood on the ACF** (Bingham-Pitts style). Cheap, but crude for small counts.
3. **Age-structured HMM or compartmental model.** Exact likelihood is intractable, so use PAL (Q5) or SMC.
4. **Simulation-based inference**, using the RGO group catalogue as the mark sampler.

**(f) Pitfall: slow rates**
- Nonparametric G recovery is slow (`T^{−β/(2β+2)}`) and noisy at long lags. Estimate G and the kernel from RGO/Debrecen, which are long and record individual groups. Treat it as fixed or tightly constrained for historical observers, whose data are sparse.

### Gaps
- I did not read the full texts of Bingham & Pitts, Grübel & Wegener, Pickands & Stine, or Schweer & Wichelhaus, so their specific estimators are not summarized.
- I found no application of M/G/∞ inference to sunspot or solar data.
- I found no published result for M/G/∞ with periodic visibility kernels. The shot-noise generalization above is my derivation.

---

## Q3. Thinned / marked point processes and Cox processes where detection depends on a mark (size)

### Takeaway
GN is a mark-dependent thinning problem: true groups form a Cox process with heavy-tailed size marks, and each observer retains a group with probability `pi_o(area)`. Independent thinning of a Poisson or Cox process keeps it Poisson or Cox with intensity multiplied by the detection fraction. So each observer's counts are closed-form Poisson given the latent intensity. The unthinned intensity is recoverable only if the mark (area) distribution is known from the reference and assumed stable across eras. Astronomy's completeness-function literature (e.g., MATHPOP) is a direct methodological analog. Mark-dependent thinning explains why observer-to-reference relationships are non-linear and depend on activity level, as Willamo et al. (2018) found for ADF.

### Cited Findings
- **MATHPOP (astronomy analog).** Observed globular clusters are modelled as an independent `t(s, m)`-thinning of the true marked point process, with detection depending on magnitude (the mark) and location. The observed intensity is the true intensity times the thinning probability. — [arXiv 2409.06040](https://arxiv.org/pdf/2409.06040)
- **Data augmentation for thinned points (sigmoidal Gaussian Cox processes).** The thinned, unobserved points are treated as latent. Their joint density with the observed points is the marked point-process density at the thinning labels times a combinatorial factor, and marginalizing recovers the observed-data likelihood. — [arXiv 2203.06743](https://arxiv.org/pdf/2203.06743)
- **Myllymäki (2009) dissertation.** Covers models and inference for spatial point patterns with intensity-dependent marks, including Bayesian modelling and mark-dependent thinning. I saw the catalogue record only. — [JYU thesis record](https://thesis.jyu.fi/Record/jykdok.1113940)
- **ADF treatment of acuity.** Observer quality is an observational threshold area S_S (msd), and the observer "would miss all of the groups with an area smaller than S_S". — [Usoskin et al. 2016, arXiv 1512.06421](https://arxiv.org/pdf/1512.06421)
  - Calibration curves come from RGO, with groups below threshold treated as unobserved and days removed randomly to match the observer's filling factor.
  - The 1749–1996 update assumes the threshold is constant over a career and leaves time-varying acuity for future work. — [arXiv 1705.05109](https://arxiv.org/pdf/1705.05109)
- **Activity-level dependence of ADF.** The ADF method estimates GN precisely for moderate activity, with less precision at other activity levels. — [Willamo, Usoskin & Kovaltsov 2018, ar5iv 1803.10501](https://ar5iv.arxiv.org/html/1803.10501)

### Inferences
**(a) Model**
- Groups form a marked Cox process with intensity `Lambda(t) dt × F(dm)`, where the mark m holds lifetime, area trajectory, latitude and longitude.
- Detection on day t is Bernoulli with probability `pi_o(a_g(t) · c(mu_g(t)))`. Here c(·) is a foreshortening or limb-darkening factor on apparent area at disk position mu.
- **Soft acuity (recommended).** Use a soft logistic acuity, `pi_o(a) = logistic((log a − log S_o)/w_o)`. It nests the ADF hard threshold as `w_o → 0`, gives HMC smooth gradients, and represents seeing- and day-dependent variability.
- **Drift.** Let `log S_o(t)` follow a hierarchical random walk on a yearly grid within each career, to capture drift.

**(b) Closed-form daily likelihood**
- Conditional on Λ, `Y_{o,t} ~ Poisson(Lambda_o(t))` with `Lambda_o(t) = int Lambda(t − u) E_m[h_m(u) pi_o(a_m(u))] du`.
- The expectation over marks can be precomputed by Monte Carlo over the empirical RGO/Debrecen group tracks, on a grid of `(S_o, w_o)` values. Store it as a lookup table and interpolate differentiably in JAX. This avoids latent-point augmentation entirely.

**(c) Why k-factors fail**
- The effective detection fraction `q_o = E_F[pi_o(a)]` depends on the area distribution F. If F shifts with activity (relatively more large groups near maximum), q_o varies with cycle phase.
- The observer-to-reference ratio is then non-linear in activity, consistent with the activity dependence reported for ADF calibration.

**(d) Identifiability pitfalls**
- From counts alone only `Lambda × q_o` is identified per observer.
- Recovering Λ (the reference-equivalent GN) requires two assumptions:
  1. F, or `F | activity`, is known from the reference.
  2. F is stationary across centuries. This is untestable from counts and is the key assumption to state in the methods paper.
- The shape of π_o (S_o versus w_o) is only weakly identified. It is informed by the activity dependence of the observer/reference ratio and by the ACF shape (Q2). Hierarchical priors across observers of the same era or instrument are needed.

**(e) Splitting and merging**
- Model these as a mark-dependent compound operator. A detected group of area a contributes `1 + Bernoulli(s_o(a))` counts (over-splitting of large complex groups), or pairs of nearby groups merge with probability `m_o`.
- Under Poisson latent groups, splitting produces a compound-Poisson count. Its pmf is still computable by convolution, and it adds overdispersion that can be confounded with acuity.

### Gaps
- I found no statistical literature on observer-dependent splitting or merging of clusters ("lumping/splitting" detection), even though it is analogous to species-delimitation issues.
- I did not read the full texts of Myllymäki (2009) or the MATHPOP inference section, so their estimation details are unknown.
- I found no paper testing stationarity of the sunspot-group area distribution across centuries. This is critical and should be covered by the solar-specific research stream.

---

## Q4. State-space count models (latent log-intensity random walk) and Gaussian-process Cox models; combining smooth activity with day-level birth-death structure

### Takeaway
Standard Poisson/NB state-space models fit naturally as the slow layer, with `log lambda(t)` a random walk or GP. Software exists for exponential-family state-space models (KFAS, importance sampling), scalable GPs (Hilbert-space approximate GPs), and JAX state-space inference (Dynamax; NumPyro HMM enumeration with parallel scan). The right architecture for GN is a three-layer hierarchy:
1. Smooth activity, a GP or random walk on `log lambda`.
2. Daily occupancy, an INAR or M/G/∞ birth-death with RGO-calibrated lifetimes.
3. Observers: thinning, splitting, drift and missingness.

The layers are linked by the separation of time scales.

### Cited Findings
- **KFAS.** Implements state-space models with Gaussian, Poisson, binomial, negative binomial and gamma observations, including Poisson forecasting examples (Helske 2017, *J. Stat. Softw.* 78(10)). Its `importanceSSM` function simulates states conditional on observations with importance weights, using a linearization (Durbin–Koopman-style). — [JSS article](https://www.jstatsoft.org/article/view/v078i10); [arXiv 1612.01907](https://arxiv.org/pdf/1612.01907); [importanceSSM docs](https://rdrr.io/github/helske/KFAS/man/importanceSSM.html)
- **Hilbert-space approximate GPs.** A low-rank basis-function approximation built on Laplace eigenfunctions, for stationary kernels, with practical rules for choosing the number of basis functions and the boundary factor, plus diagnostics. The linear structure gives favorable scaling in probabilistic programming (Riutort-Mayol, Bürkner, Andersen, Solin & Vehtari, *Statistics and Computing* 33:17, 2023). PyMC ships HSGP and HSGPPeriodic. — [arXiv 2004.11408](https://arxiv.org/pdf/2004.11408); [Aalto record](https://research.aalto.fi/en/publications/practical-hilbert-space-approximate-bayesian-gaussian-processes-f/); [PyMC HSGPPeriodic](https://docs.pymc.io/en/stable/api/gp/generated/pymc.gp.HSGPPeriodic.html)
- **Dynamax.** A JAX library for HMMs, linear-Gaussian SSMs, and nonlinear or generalized-Gaussian SSMs, built on functionally pure inference kernels and compatible with optax and Blackjax (Linderman et al., *JOSS* 10(108):7069, 2025). A secondary source states that it implements Särkkä & García-Fernández (2021) parallel filtering and smoothing. — [JOSS](https://joss.theoj.org/papers/10.21105/joss.07069); [arXiv 2509.21716 (secondary)](https://arxiv.org/pdf/2509.21716)
- **NumPyro HMM enumeration.**
  - Discrete latents are marginalized by variable elimination inside MCMC, and `scan` reduces compile time.
  - Stacking along time "allows the joint density to be computed in parallel", and a parallel-scan algorithm reduces parallel complexity from O(length) to O(log(length)).
  - Enumerated variables need a low-treewidth dependency graph. — [NumPyro hmm_enum example](https://num.pyro.ai/en/0.19.0/examples/hmm_enum.html)
  - Users report broadcasting and shape errors when enumerating inside `scan`. — [Pyro forum](https://forum.pyro.ai/t/scan-enumerate-and-how-to-handle-broadcasting/4791)

### Inferences
**(a) Composite model ("smooth activity + birth-death + observers")**

| Layer | Content |
|---|---|
| A, activity | `log lambda(t) = m + f(t)`. f is an HSGP with Matérn kernel on a monthly time grid, optionally plus a quasi-periodic ~11-yr kernel, or a second-order random walk on months. A hemispheric split `lambda_N`, `lambda_S` is optional. |
| B, occupancy | Daily `N_t`: either the scalar INAR(1)/HMM (Q1), or the age-structured or shot-noise structure (Q2) with RGO-calibrated lifetime and area-trajectory marks. |
| C, observers | `Y_{o,t}`: soft-threshold thinning `pi_o` with career drift (Q3), optional splitting, NB observation overdispersion, and the MNAR recording model (Q6). Hierarchical priors pool observer parameters by era and instrument class. |
| Output | Posterior draws of `lambda(t)` and `N_t^{ref}`, the count a reference observer with acuity `S_ref` would report. This is the GN ensemble. |

**(b) Computation**
- **Scalar-HMM variant.**
  - The forward algorithm costs `O(T K^2)` and is differentiable.
  - With NUTS, the cost is roughly (#leapfrog steps) × (#iterations) × 4×10^8 flops, i.e. ~10^14 flops for a few thousand iterations. That is feasible on a GPU in hours.
  - SVI or Laplace approximations on the GP layer are faster alternatives.
  - Use FFBS for N_t draws.
- **Approximate variant.** Monthly-stationary pairwise-Poisson composite likelihood (Q2), with daily structure inside each month, plus the GP across months. Much cheaper, but composite-likelihood posteriors need curvature adjustment for calibrated uncertainty.
- **Durbin–Koopman-style approach.** A Laplace or importance-sampling approximation for `log lambda` (KFAS), but KFAS has no birth-death layer and is R-only. In JAX, Dynamax or hand-written `lax.scan` / `associative_scan` code is the practical route.

**(c) Overdispersion versus thinning**
- In the GP-Cox layer, cycle variation creates Poisson-lognormal overdispersion at the monthly scale. Do not also add free NB overdispersion at the observer level without priors, or the dispersion handle on p_o (`D_Y = 1 + p(D_N − 1)`) is lost.
- Calibrate `D_N | lambda` on RGO.

**(d) Time-scale separation as a modelling principle**
- GP length scales (months to years) should be constrained well above lifetimes (days to weeks) through priors. Otherwise the GP can absorb birth-death autocorrelation and bias lifetime and acuity estimates.

### Gaps
- I did not retrieve primary URLs for the Durbin & Koopman book or Fahrmeir's work on dynamic count models, so they are not cited.
- I found no published model combining a GP-Cox activity layer with an explicit occupancy birth-death layer and multiple imperfect observers. The architecture above is a synthesis.
- I did not verify run times for NumPyro or Dynamax at T ≈ 1.5×10^5.

---

## Q5. Epidemiological under-reporting and nowcasting: what transfers to observer-specific reporting probabilities and under-reported zeros?

### Takeaway
The epidemiological under-reporting literature transfers almost directly. The template is the Poisson-logistic ("Pogit") hierarchy: latent Poisson truth, binomial reporting, and covariate models for both layers (Stoner et al. 2019; Dvorzak & Wagner 2016). Its key lessons are that under-reporting is identified only via informative priors, covariates, or validation data, and that prior sensitivity must be reported. Nowcasting tools (Höhle & an der Heiden 2014; NobBS; epinowcast) contribute random-walk time-varying reporting processes and hierarchical pooling more than the delay machinery itself. Poisson Approximate Likelihoods (Whitehouse et al. 2023) are the most promising route to fast, tuning-free likelihoods for an age-structured birth-death process with binomial detection over 150k days, but must be checked at the small counts typical of GN.

### Cited Findings
- **Stoner, Economou & Drummond (2019).** "A hierarchical framework for correcting under-reporting in count data", *JASA* 114(528):1481–1492. — [arXiv 1809.00544](https://arxiv.org/pdf/1809.00544); [Glasgow open-access record](https://eprints.gla.ac.uk/249540)
  - Model: a binomial model for recorded counts on top of a latent Poisson for true counts.
  - It handles fully under-reported data through an informative prior on the mean reporting rate, with covariates informing both the true-count and the reporting processes and spatio-temporal structure.
  - It is validated with simulation-based prior and covariate sensitivity analyses plus prior and posterior predictive checks. The application is tuberculosis in Brazil.
- **The Pogit model.** Introduced by Winkelmann & Zimmermann (1993). Dvorzak & Wagner (2016), "Sparse Bayesian modelling of underreported count data", *Statistical Modelling* 16(1):24–46, add spike-and-slab variable selection for both the count and the reporting processes, using MCMC with data augmentation and auxiliary mixture sampling. It is implemented in the R package `pogit`. — [pogit CRAN docs](https://search.r-project.org/CRAN/refmans/pogit/html/pogit-package.html)
- **Höhle & an der Heiden (2014).** *Biometrics*, doi 10.1111/biom.12194. Frames nowcasting as an occurred-but-not-yet-reported problem (Lawless 1994). It uses negative-binomial sampling and handles right truncation of the delay distribution with a conjugate generalized-Dirichlet framework, motivated by the 2011 STEC O104:H4 outbreak. — [preprint](https://staff.math.su.se/hoehle/pubs/hoehle_anderheiden2014-preprint.pdf)
  - In R `surveillance`, the `bayes.trunc` option implements it, and `bayes.trunc.ddcp` allows change points in the delay distribution. — [surveillance::nowcast docs](https://search.r-project.org/CRAN/refmans/surveillance/html/nowcast.html)
- **McGough et al. (2020), NobBS.** *PLoS Comput. Biol.* 16(4):e1007735. Jointly learns the delay distribution and the epidemic curve. Adding a temporal relationship between cases improved performance when delays change. The CRAN package (v1.1.0, 2025) uses JAGS. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7162546); [CRAN](https://search.r-project.org/CRAN/packages/NobBS/index.html)
- **epinowcast.** A semi-mechanistic Bayesian package for data with reporting delays, right truncation, missing data and "incomplete ascertainment". It has a formula interface with random effects, random walks and time-varying parameters, and is marked experimental. It "could be applied to any set of right-truncated time-series count data". — [epinowcast manual](https://epinowcast.r-universe.dev/epinowcast/doc/manual.html); [r-universe page](https://epinowcast.r-universe.dev/epinowcast)
- **Poisson Approximate Likelihoods (PAL).** Whitehouse, Whiteley & Rimella (2023), *JRSS-B* 85:1173–1203, build PALs from approximate filtering equations for finite-population stochastic compartmental models. Maximum-PAL estimators are consistent in the large-population limit. PALs "require only elementary arithmetic operations and no tuning parameters" and need no simulation. — [arXiv 2205.13602](https://arxiv.org/abs/2205.13602)
  - A 2025 extension (LawPAL) treats time-varying reporting probabilities as latent and integrates them out with Laplace approximations, for over-dispersed partial observations. — [arXiv 2505.06935](https://arxiv.org/pdf/2505.06935v2)
- **Time-varying under-reporting in hidden INAR.** Fernández-Fontelo et al. (2020) let the under-reporting parameter vary over time (seasonality or trend) inside a Po-INAR(1) HMM. — [arXiv 2008.00262](https://arxiv.org/pdf/2008.00262)

### Inferences
**(a) Mapping**

| Epidemiology | GN analog |
|---|---|
| Reporting probability with covariates | Observer acuity (S_o, w_o), with covariates: telescope aperture and magnification, era, site or seeing climate, years since career start (learning and ageing), and projection versus direct viewing. |
| Spatio-temporal random effects | Hierarchical pooling across ~600 observers, plus career random walks. |
| Fully under-reported regime with an informative prior on the mean reporting rate (Stoner) | Eras with no overlap with RGO or the backbone chain, such as pre-1700 or the Maunder Minimum. Instrument-based or chain-based informative priors are unavoidable there. Copy Stoner's simulation-based prior-sensitivity design for the methods paper. |
| Validation data (Pogit literature) | RGO/Debrecen overlap years, where individual group areas give the true F and N_t. |

**(b) Nowcasting**
- Nowcasting proper (right-truncated reporting delays) has no direct GN analog. The transferable pieces are:
  - random-walk, time-varying reporting parameters with change points (`bayes.trunc.ddcp`, NobBS), which map to observer drift and instrument changes;
  - partial pooling across strata (epinowcast), which maps to observers of a given era;
  - jointly modelling the "epidemic curve" (lambda(t)) and the reporting process, rather than estimating them in separate stages.

**(c) PAL for GN**
- An age-structured immigration-death process with binomial survival and binomial detection is a linear compartmental model, the most favorable case for PAL.
- PAL turns the vector-state problem into `O(T × #compartments)` arithmetic, is differentiable in JAX, and allows gradient-based fitting of hundreds of observer parameters.
- Caveat: PAL's consistency is a large-population result, and GN counts are 0–30. Validate PAL against the exact scalar-HMM forward algorithm (Q1) and against SMC on simulated RGO-like data before adopting it.

**(d) Identifiability (shared lesson)**
- Every one of these frameworks reports that the reporting rate and the true intensity are confounded without either informative priors or validation data. GN is in a better position than most epidemiology because the reference provides validation data. The pitfall is extrapolating calibration to eras without overlap.

### Gaps
- I did not locate the Magal & Webb under-reporting/HMM work in this session, so it is not cited.
- I did not examine EpiNow2.
- I found no evidence on PAL accuracy at very small counts (0–30) with near-zero detection probabilities. This needs a simulation check.
- I did not verify whether Dvorzak & Wagner (2016) identify reporting through a validation sample; this detail is unconfirmed.

---

## Q6. Missing-not-at-random zeros, selection models (Heckman-type, pattern-mixture), and zero-inflated/hurdle models: modelling "spotless days less likely to be recorded"

### Takeaway
The GN problem is the reverse of zero inflation. Spotless days are selectively missing from the record, so the record is zero-deflated, an outcome-dependent selection problem. A selection model is the natural fit: `P(R = 1 | Y)` depends on whether Y = 0, plus weather and season. It slots into the forward recursion in closed form. However, the selection parameter is identified only parametrically, and the MNAR literature insists on sensitivity or tipping-point analyses. Hurdle structure matches ADF-type binary active/inactive data. Blanket statements such as "no spots this year" are censored aggregate observations that can enter the likelihood as `P(sum = 0)`.

### Cited Findings
- **Zero-inflated Poisson.** Lambert (1992), *Technometrics* 34(1):1–14, defines the ZIP model: with probability p the observation is 0 (a "perfect" state), and otherwise it is Poisson(λ). Both p and λ can depend on covariates, and p can be tied to λ via `p = 1/(1 + λ^γ)`. — [Lambert 1992 PDF](https://www.stat.cmu.edu/technometrics/90-00/vol-34-01/v3401001.pdf)
- **Two MNAR factorizations.** Selection and pattern-mixture models factor the joint distribution of outcome and missingness in opposite orders. Selection models model the missingness mechanism; pattern-mixture sensitivity parameters are differences in outcome expectations between missing and observed groups. — [arXiv 2108.13286](https://arxiv.org/pdf/2108.13286)
- **Kaciroti & Raghunathan (2014), *Stat. Med.*.** Give parameterizations translating between pattern-mixture and selection forms for exponential-family outcomes, which include Poisson. They place informative priors on sensitivity parameters and fit with Gibbs sampling. — [Deep Blue PDF](https://deepblue.lib.umich.edu/bitstream/2027.42/109600/1/sim6302.pdf)
- **Non-identifiability.** Both MNAR families rely on non-identifiable sensitivity parameters that the analyst must set. — [VicBiostat seminar abstract](https://www.vicbiostat.org.au/node/2107)
- **Practical rule.** If results under MNAR differ from those under MAR, the MAR conclusion is in question. — [SAS PROC MI documentation](https://support.sas.com/documentation/cdl/en/statug/67523/HTML/default/statug_mi_details56.htm)
- **ADF filling factor.** The ADF filling factor counts days with reported observations, including no-spot observations. ADF simulates observers by randomly removing days, which is a MAR (in fact MCAR) assumption. — [Usoskin et al. 2016, arXiv 1512.06421](https://arxiv.org/pdf/1512.06421)

### Inferences
**(a) Selection model for GN**
- Recording model: `logit P(R_{o,t} = 1 | Y_{o,t}) = gamma_o + s(doy) + delta_o · 1{Y_{o,t} = 0}`, plus an optional year- or career-level random effect. Here `s(doy)` is a seasonal or weather term, and `delta_o < 0` means spotless days are under-recorded.
- On a recorded day: `P(Y = y, R = 1) = P(Y = y) rho_o(y)`.
- On an unrecorded day within a career: `sum_y P(Y = y)(1 − rho_o(y))`.
- Both factors enter the forward recursion as emissions with closed-form cost. Heckman-type latent-Gaussian selection is a probit alternative, with no obvious advantage for integer counts.

**(b) Identification of delta_o**
- Parametric identification comes from the assumed count model, which is weak. Three better sources:
  1. Same-day overlap with other observers or the reference. An observer's missing days that coincide with reference-recorded zeros at a higher rate than weather alone predicts.
  2. Activity dependence of the recording rate. Weather-driven gaps should not depend on solar activity once season is controlled, so a recording rate that drops at solar minimum signals MNAR.
  3. Clustering of record gaps inside spotless runs.
- Report GN under a grid of delta values (a tipping-point analysis) and state which conclusions change, for example Maunder Minimum levels or the cycle amplitudes of the 1700s.

**(c) Pattern-mixture alternative**
- For unrecorded days, set `P(Y = 0 | R = 0) = P(Y = 0 | R = 1) × e^Delta` (a delta adjustment on the zero probability), with Delta given an informative prior. This is easier to explain to solar physicists, and Kaciroti & Raghunathan-type mappings relate it to delta_o.

**(d) Zero-inflated versus hurdle**
- Zero inflation (extra structural zeros) fits observers who wrote "no spots" by default, or who were insensitive for long stretches. That is an acuity-collapse state, best modelled as a Markov-switching "bad period" in the observer layer.
- The hurdle form (Bernoulli zero/non-zero, then a zero-truncated count) matches data that record only spotted/unspotted, i.e. ADF-type data. The Bernoulli part is exactly `1 − exp(−Lambda_o(t))` under the Poisson model (Q2/Q3).

**(e) Blanket statements**
- "No spots this year" by observer o is an aggregate censored observation. Its likelihood is `P(sum_{t in year} Y_{o,t} = 0)`, which under Cox-thinning is `E[exp(−sum_t Lambda_o(t))]` given lambda.
- These statements are themselves MNAR: they were written because absence was notable. Add a reporting model for statements, or weight them as soft constraints with discounting.
- With unknown acuity they mainly constrain the product of lambda and acuity, so instrument priors matter most here.

### Gaps
- I found no literature specifically on outcome-dependent missingness of zero counts in count time series, or on Heckman models for integer outcomes. This search was not done in depth.
- I found no statistical treatment of qualitative "blanket" historical statements as censored observations. The approach above is a proposal.

---

## Q7. Capture-recapture / multiple-systems estimation with heterogeneous detection: can multiple unlabeled same-day counts bound the true number of groups?

### Takeaway
Without group identities, multiple-systems estimation reduces to N-mixture models (Royle 2004). The open-population Dail–Madsen version (survival plus recruitment plus binomial detection) is structurally the same as Q1's hidden INAR with many observers. Replicated counts identify N only under strong assumptions: conditional independence and homogeneous, constant p. Barker et al. (2018) show that alternative models can produce practically indistinguishable data, that goodness-of-fit tests rarely detect the violations, and that only relative abundance is reliable without auxiliary data. Area-dependent acuity is shared heterogeneity: all observers miss small groups. Same-day counts therefore estimate a Chao-style lower bound, the population detectable by the best observers, not the true N. GN should be defined relative to a reference acuity, which is identifiable, rather than as "all groups".

### Cited Findings
- **Royle (2004) N-mixture model.** *Biometrics* 60(1):108–115. Site abundances are treated as random draws from a mixing distribution (e.g., Poisson), and parameters are estimated from the marginal likelihood of replicated counts. Spatial replication informs the mixing distribution. In simulations it outperformed Carroll & Lombard (1985) in bias and coverage. — [USGS record](https://pubs.usgs.gov/publication/5224798); [PDF](https://cals.arizona.edu/classes/wfsc578/Royle%202004.%20N-mixture%20models%20for%20estimating%20population%20size.pdf)
  - The `unmarked::pcount` function allows Poisson, NB or ZIP mixing. — [pcount docs](https://rdrr.io/cran/unmarked/man/pcount.html)
- **Infinite abundance estimates.** Dennis, Morgan & Ridout (2014, *Biometrics*) show these can arise, especially when detection probability and the number of sampling occasions are small. — [PMC4406156](https://pmc.ncbi.nlm.nih.gov/articles/PMC4406156)
- **Barker, Schofield, Link & Sauer (2018), *Biometrics*.** — [USGS record](https://pubs.usgs.gov/publication/70198028)
  - Without marking, reliable modelling of N and p from counts is problematic.
  - Visit-specific detection makes the model overspecified.
  - Even with constant p, plausible alternative models produce practically indistinguishable data in which N is non-identifiable, especially for sparse data.
  - Reliable inference is mainly for relative abundance, and auxiliary data on p is recommended.
- **Link et al. (2018), *Ecology*.** Violations of N-mixture assumptions are unlikely to be detected by goodness-of-fit tests. — [USGS record](https://www.usgs.gov/publications/robustness-n-mixture-models)
- **Dail & Madsen (2011) open N-mixture.** *Biometrics* 67:577–587. It generalizes Royle (2004) to open populations, with an initial-abundance model, a recruitment rate, survival and detection. `unmarked::pcountOpen` supports constant, autoregressive, trend, Ricker and Gompertz dynamics, plus immigration. `multmixOpen` handles removal and double-observer designs. — [pcountOpen docs](https://search.r-project.org/CRAN/refmans/unmarked/html/pcountOpen.html); [multmixOpen docs](https://rdrr.io/github/rbchan/unmarked/man/multmixOpen.html)
- **Chao (1987).** *Biometrics* 43:783–791. Gives an estimator for closed-population size under heterogeneous capture probabilities, commonly written `n + f1^2/(2 f2)` and interpreted as a lower bound. — [Chao 1987 PDF](https://gwern.net/doc/statistics/order/capture/1987-chao.pdf)
  - A covariate generalization (Böhning et al. 2013) is asymptotically unbiased if the covariates fully explain the heterogeneity. — [Mahidol repository](https://repository.li.mahidol.ac.th/handle/123456789/30943)

### Inferences
**(a) Structural identity**
- Dail–Madsen `N_t = Bin(N_{t−1}, omega) + Pois(gamma)`, with `y_{it} ~ Bin(N_t, p_i)`, is the Q1 model: omega maps to alpha (persistence) and gamma to lambda (emergence plus rotation inflow).
- GN is a single "site" (the Sun) with a variable number of replicate observers per day. The `pcountOpen`-style forward likelihood is the same truncated-HMM computation, so ecological software is a useful cross-check.

**(b) Moment identification from replicates (my derivation)**
- Under conditional independence, homogeneous p, and Poisson N with mean μ:
  - `E Y_a = p_a mu`
  - `Cov(Y_a, Y_b) = p_a p_b mu`
  - hence `mu = E[Y_a] E[Y_b] / Cov(Y_a, Y_b)`.
- This is a label-free Lincoln–Petersen analog.
- With overdispersed N (e.g., Cox-varying lambda), the covariance also contains `p_a p_b Var(lambda)`, so the estimator must be applied within short windows where lambda is roughly constant.

**(c) Heterogeneity bias**
- With area-dependent detection, both observers preferentially see large groups. Detection is positively dependent across observers, which inflates the covariance relative to the product of means, so μ is underestimated.
- The estimand becomes "groups detectable by these observers", the same logic by which heterogeneity makes Chao-type estimators lower bounds.
- Counts alone therefore cannot recover groups below the best observer's acuity. Only the reference with measured areas, plus an assumed area distribution, can extrapolate.
- Define the GN target as `N_t^{ref}`, the expected count of a reference observer with acuity S_ref. This makes the target identifiable and matches what ADF and backbone methods effectively do.

**(d) Bivariate-Poisson structure (closed form)**
- Under Poisson latent groups with mark thinning, two observers on the same day satisfy `(Y_a, Y_b) = (Z_a + Z_ab, Z_b + Z_ab)` with independent Poisson components, where `E Z_ab = int Lambda E[h pi_a pi_b]`.
- The joint pmf is closed form. With RGO-known marks and parametric π, it constrains differences in acuity between observers, not only ratios. It is a better use of overlap days than ratio-based k-factors.

**(e) Robust per-day bound**
- If observers do not over-split groups or report spurious ones, `N_t ≥ max_o Y_{o,t}`. This can be enforced as a support constraint in the forward recursion.
- With over-splitting by some observers it fails, so it is useful mainly as a diagnostic: observers frequently exceeding the reference-model posterior are candidates for splitting.

**(f) True capture-recapture is possible for drawings**
- When sunspot drawings with positions exist (e.g., Staudacher, Schwabe, and many 19th-century observers), groups can be matched across observers and days. Log-linear and latent-class multiple-systems estimation with heterogeneity then becomes available.
- Such a matched subsample is the strongest possible auxiliary data for estimating π_o, splitting, and the effect of shared heterogeneity. Barker et al. recommend exactly this kind of auxiliary data.

**(g) Pitfall summary**
- N-mixture identifiability relies on the binomial-given-N form and conditional independence. Shared heterogeneity, observer splitting, and dependence between observers (copying, or the same site and weather) all bias the result and are hard to detect by goodness-of-fit tests (Link et al.).
- Therefore use replicate-count information only alongside reference calibration, never as the sole source of absolute scale.

### Gaps
- I did not search for Bayesian latent-class multiple-systems estimation or log-linear list models, so specific references (e.g., Manrique-Vallier's work) are not cited.
- I found no literature that bounds population size from multiple unlabeled counts beyond the N-mixture family. The bivariate-Poisson construction above is my derivation.
- Results of the flying-squirrel comparison of Dail–Madsen and CJS survival estimates appeared in search snippets, but I could not attribute them to a specific URL, so they are omitted.
