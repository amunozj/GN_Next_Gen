# Latent-truth rater models, test equating/linking, and dynamic rating systems as analogs for homogenizing the sunspot group number (GN)

Scope note for the report writer: this file covers three adjacent method families (multi-annotator latent-truth models, psychometric rater/equating models, dynamic paired-comparison ratings) plus physical-science uses of these ideas. Every method is mapped onto the GN problem: about 600 observers from 1610 to the present, a latent true daily group count, observer-specific detection thresholds and split/merge conventions that may drift over a career, irregular observing days, under-reported zero days, and a sparse, chain-like overlap graph. Current practice is daisy-chain/backbone pairwise calibration (Chatzistergos et al. 2017 and others) or active-day-fraction (ADF) matching to a modern reference.

Notation used in the inferences below (my notation, not from any one source): t = day; G_t = true number of groups on day t (latent integer); lambda_t = smoothly varying solar activity rate (latent, continuous); o = observer; Y_{o,t} = reported count; theta_o(t) = observer parameters (acuity/threshold, split/merge tendency, reporting propensity), possibly time-varying.

Provenance caveat: several findings below come from search-engine summaries of abstracts and secondary citing papers rather than full texts, because PDFs could not be parsed in this session. These are flagged "(secondary)" and the corresponding precise numbers are listed under Gaps.

## Q1. Dawid-Skene latent-class annotator models and their extensions to ordinal/count labels; generalizing Chatzistergos et al.'s probability-distribution-matrix calibration

### Takeaway
A Dawid-Skene (DS) model with one confusion matrix per observer, P(Y_{o,t}=y | G_t=g), is the latent-truth generalization of Chatzistergos et al.'s (2017) empirical probability-distribution-matrix (PDM) calibration. It replaces "the backbone observer is truth" with "truth is latent and estimated jointly from all observers who saw that day." For GN it needs three changes to the classic model: (i) a dynamic prior on G_t (a time-varying solar rate instead of DS's single global class prevalence), (ii) a low-dimensional parameterization of each observer's confusion matrix (binomial or threshold thinning plus split/merge), with hierarchical pooling across observers, and (iii) an explicit reporting model for missing and under-reported zero days. All of these are straightforward in NumPyro, which already ships a Bayesian Dawid-Skene example.

### Cited Findings
**(a) Core model and (b) key references**
- Dawid & Skene (1979), "Maximum likelihood estimation of observer error-rates using the EM algorithm" (JRSS C 28(1):20-28), is the founding model. Each item has a latent true class, each annotator has a confusion matrix of P(reported label | true class), and EM estimates the confusion matrices and the latent classes together. NumPyro's annotation example implements this with discrete latent classes indexed by annotator and item, citing Dawid & Skene as the original source. — [NumPyro: Bayesian Models of Annotation](https://num.pyro.ai/en/0.9.2/examples/annotation.html)
- Paun et al. (2018), "Comparing Bayesian Models of Annotation" (TACL), compares six model-based annotation models, generalizing some from binary to multiclass labels. Its priors cover annotator accuracies, annotator biases and class prevalence. The NumPyro example is built on this paper. — [Paun et al. 2018 (ACL Anthology PDF)](https://preview.aclanthology.org/moar-dois/Q18-1040.pdf); [NumPyro example](https://num.pyro.ai/en/0.9.2/examples/annotation.html)
- Raykar et al. (2010), "Learning From Crowds" (JMLR 11:1297-1322), is a probabilistic approach for multiple noisy annotators with no gold standard. It estimates both annotator performance and the hidden true labels. — [JMLR page](https://jmlr.org/papers/v11/raykar10a.html)
- In the binary case Raykar et al. parameterize each annotator by sensitivity and specificity. A review notes that this parameterization becomes ambiguous for multi-class labels and that performance then deteriorates. — [Knowledge Learning with Crowdsourcing review, arXiv 2206.09315](https://arxiv.org/pdf/2206.09315)
- Raykar et al. replace DS's simple multinomial class-prevalence model with a logistic regression on item features (Stan forum description; secondary). — [Stan discourse thread](https://discourse.mc-stan.org/t/raykar-et-al-learning-from-crowds-model/5364)
- Li, Rubinstein & Cohn's work on Bayesian truth inference for highly redundant crowd annotations reports that existing methods beat simple baselines (such as majority vote) mainly when there are few annotations per item (3 or fewer) or when workers are unreliable (secondary summary). — [arXiv 1902.08918](https://arxiv.org/pdf/1902.08918)

**Count-label analogs (from ecology; this closes the "count label" gap)**
- In the Royle (2004) N-mixture model (Biometrics 60:108-115), site abundance N_i is latent with a mixing distribution such as Poisson, and each repeated count is y_ij | N_i ~ Binomial(N_i, p). — [Royle 2004 PDF](https://cals.arizona.edu/classes/wfsc578/Royle%202004.%20N-mixture%20models%20for%20estimating%20population%20size.pdf); [USGS AHM workshop notes](https://www.mbr-pwrc.usgs.gov/workshops/ahm2023/06_N-mixture/n-mixture.html)
- In N-mixture models, abundance is identified by variation across sites and detection by variation between visits to the same site. With a single visit, lambda and p cannot be separated. Negative-binomial and zero-inflated Poisson mixing distributions are common. — [Metricgate N-mixture summary (secondary)](https://metricgate.com/calculator/n-mixture-royle-abundance); [unmarked::gpcount docs](https://rdrr.io/cran/unmarked/man/gpcount.html)
- The `unmarked` R package fits generalized binomial N-mixture models, including an availability process (Chandler et al. 2011). — [unmarked gpcount](https://search.r-project.org/CRAN/refmans/unmarked/html/gpcount.html)

**The sunspot-side baseline being generalized**
- Chatzistergos, Usoskin, Kovaltsov, Krivova & Solanki (2017, A&A 602, A69) use a non-parametric calibration in place of linear rescaling. Each observer is calibrated with a probability-distribution matrix built from all daily values in the overlap period with a backbone (BB) observer, in a daisy-chain/backbone scheme. Errors are propagated by Monte Carlo. The series contains 314 observers. The method assumes that differences between observers reflect different acuity thresholds, constant over each observer's observing period. — [arXiv 1702.06183](https://arxiv.org/abs/1702.06183)
- A related paper tests whether ADF calibration depends on the level of solar activity, which in equating terms is a population-invariance question. — [arXiv 1803.10501](https://arxiv.org/pdf/1803.10501)
- Mathieu, von Sachs, Ritter, Delouille & Lefèvre model sunspot-network observations as Y_i(t) = (eps1 + eps2)·s(t) outside minima, where s(t) is the true count, eps1 a short-term error and eps2 a long-term error, and Y_i(t) = eps3 at minima. Each station is rescaled by a factor kappa_i estimated against the network median. — [arXiv 2009.09810 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2009.09810)
- In the same paper, a log-log regression of conditional variance on conditional mean gives a slope of about 0.96 for group counts N_g (close to Poisson), against about 1.5 for spot counts N_s. Zeros are handled with a zero-altered (hurdle) model, preferred to zero-inflation because true and false zeros do not arise in the same term. — [arXiv 2009.09810 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2009.09810)

**(e) Software and scalability**
- NumPyro has a ready Bayesian annotation example (Dawid-Skene and related models) that handles discrete latent classes by enumeration. — [NumPyro annotation example](https://num.pyro.ai/en/0.9.2/examples/annotation.html)
- For dynamic, time-varying annotator confusion matrices, Simpson, Roberts, Psorakis & Smith's DynIBCC uses variational Bayes. They report VB as faster and more accurate than Gibbs sampling (13 s vs 349 s convergence) on Galaxy Zoo Supernovae (reported as 963 tasks and 1,705 classifiers; secondary). — [arXiv 1206.1831](https://arxiv.org/abs/1206.1831v1)

### Inferences
**(c) Mapping to GN**
- Raters are observers, items are days, and the latent class is the true daily group count G_t, an integer from 0 to about 40. The DS confusion matrix pi_o[g, y] = P(Y_{o,t}=y | G_t=g) is a forward observation model for observer o.
- Chatzistergos et al.'s PDM is, as I read the method (verify direction in their Sect. 2), an empirical conditional distribution linking observer counts to backbone counts on common days. It is therefore a DS confusion matrix estimated against a proxy for truth (the BB observer) instead of latent truth, one observer pair at a time.
- The DS generalization does three things the PDM cannot:
  - it uses every observer who reported on a given day, not just the BB;
  - it propagates the BB's own noise instead of treating the BB as exact;
  - it yields a posterior over G_t on days observed only by non-BB observers.

**Required GN-specific modifications**
1. **Dynamic prevalence.** DS assumes a single global class-prevalence vector, which suits exchangeable items. GN days are strongly autocorrelated and nonstationary (solar cycles, Maunder Minimum). Replace the prevalence with G_t ~ Poisson or NegBin(lambda_t), where log lambda_t follows a Gaussian process or random walk (state-space). Mathieu et al.'s near-Poisson dispersion for N_g supports a Poisson or mildly overdispersed choice.
2. **Structured confusion matrices.** A free K-by-K matrix per observer (K of about 30 gives about 900 parameters) is unidentifiable for the many observers with few overlap days. Parameterize it physically instead:
   - (i) binomial thinning Y ~ Binomial(G, p_o), the N-mixture analog;
   - (ii) acuity-threshold thinning: a group is detected if its area exceeds A_o, with detection probability depending on the group-size distribution and possibly on activity level. This is the physical assumption behind ADF and the PDM;
   - (iii) a split/merge kernel, for example Y = thinned(G) + Poisson(s_o·thinned(G)) for splitters, which lets mass sit above the diagonal;
   - (iv) hierarchical priors across observers, by era, instrument class or institution. These provide the partial pooling that Paun et al.-style models implement, so isolated early observers borrow strength.
3. **Reporting and missingness model.** Under-reported zero days are a missing-not-at-random problem: the probability that observer o reports on day t depends on G_t (lower when G_t = 0). Add P(report | G_t, o) as a factor. This is the DS "abstain" mechanism made class-dependent. The N-mixture literature's separation of availability from detection (gpcount) is the closest template.
4. **Drift.** The Chatzistergos assumption of a constant threshold per observer becomes theta_o(t) following a random walk (see Q2 and Q4). DynIBCC shows that time-varying confusion matrices are tractable at crowd scale.

**(d) Pitfalls and identifiability**
- **Absolute scale.** As in N-mixture models (one visit cannot separate lambda from p), the absolute detection level is not identified from counts alone. Multiplying all p_o by c and dividing lambda_t by c leaves the binomial-Poisson marginal unchanged, because Poisson thinning gives Poisson(p·lambda). The scale must come from an anchor (a modern reference defined as the GN scale, the convention used in practice) or from a physical prior on the group-size distribution.
- **Label switching** is not an issue when classes are ordered integers with a monotone thinning model, but it is for a free confusion matrix.
- **Days seen by a single observer** inform only that observer's prior-weighted reconstruction. Linking information comes solely from multi-observer days, so the sparse overlap graph directly controls the posterior width of theta_o.
- **Overconfidence** follows if the per-observer independence assumption fails, for example when observers copy each other or report through the same compiler. Li et al.'s finding that sophisticated aggregation helps most at low redundancy suggests most GN gains will appear in the sparse early centuries, where they are also hardest to validate.

**(e) Scalability**
- About 415 years is about 151,000 days. With G_max of about 40, exact marginalization of G_t is about 6 million log-sum-exp terms per likelihood evaluation, which is trivial in JAX on a GPU.
- lambda_t can be a sparse or Markov GP (for example a state-space GP or a random-walk prior on log lambda) sampled by HMC/NUTS, or fitted with SVI for speed.

### Gaps
- I did not verify whether Raykar et al. (2010) contains explicit ordinal and continuous-regression annotator extensions with annotator-specific noise precision. The text was not retrieved; check the JMLR PDF.
- I found no peer-reviewed annotator model built for integer count labels with annotator-specific under- or over-counting in the ML crowdsourcing literature. The closest analogs are ecological N-mixture models and Mathieu et al.'s multiplicative error model. A targeted search such as "crowd counting annotator bias Poisson" was not run.
- Paun et al.'s specific ranking of pooled, unpooled and partially pooled models was not retrieved.
- Known reliability concerns about N-mixture identifiability (sensitivity to unmodeled heterogeneity in p) were not verified in this session.
- The exact construction of Chatzistergos et al.'s PDM (conditioning direction, normalization, handling of sparse rows) was not extracted from the full text.
- Mathieu et al. is listed as ApJ 2019 in a search summary, but the arXiv version (2009.09810, 2020) showed no journal reference. Confirm the venue.

## Q2. Many-facet Rasch / rater-effect models and rater drift (Linacre, Wolfe, Patz-Junker HRM, DeCarlo's latent-class SDT): how severity and its drift are modeled and separated from true change

### Takeaway
Psychometrics offers two structures: severity as an additive facet (many-facet Rasch), and ratings as noisy signals of an "ideal" latent category (HRM, HRM-SDT). The second is the right template for GN: solar activity, then true daily groups, then each observer's report. Rater drift is now modeled with random-walk (Markov) priors on time-indexed severity (Uto 2023). In rating studies, drift is separated from true change because the rated objects (essays) have fixed quality and are randomly assigned over time. GN has no such re-rateable objects, so era-wide common-mode drift is not identified from data and must come from priors, physical constraints or external proxies.

### Cited Findings
**(a) Core models and (b) references**
- Many-facet Rasch model (MFRM) form, as used by Uto: P(score k | examinee j, rater r, time t) is a softmax over cumulative step terms D(theta_j − beta_rt − d_rm). Here theta_j is ability, beta_rt is time-specific rater severity, d_rm are rater-specific step parameters, and D = 1.7. — [Uto 2023, Behavior Research Methods 55(7):3910-3928 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10615980/)
- In the Hierarchical Rater Model (Patz, Junker, Johnson & Mariano 2002, JEBS 27(4):341-384), rater scores are not direct indicators of ability. They indicate the "ideal" category of the response, which in turn indicates examinee proficiency. The model captures rater severity and consistency and models consensus among raters. — [Patz et al. 2002 PDF](https://www.stat.cmu.edu/~brian/trs/patz-et-al-hrm-jebs-2002.pdf); [IDEAS](https://ideas.repec.org/a/sae/jedbes/v27y2002i4p341-384.html)
- The HRM critique of facets models: as the number of raters per item grows, IRT facets models appear to give infinitely precise measurement of proficiency, because they ignore the dependence among ratings of the same response. The HRM avoids this. — [Patz et al. 2002 (summary)](https://www.stat.cmu.edu/~brian/trs/patz-et-al-hrm-jebs-2002.pdf)
- Mariano & Junker (2007) extend the HRM with covariates of the rating process, such as rater training. — [Mariano & Junker 2007 PDF](https://www.stat.cmu.edu/~brian/trs/mariano-junker-jebs-2007.pdf)
- DeCarlo, Kim & Johnson (2011, JEM 48(3):333-356) use a latent-class signal-detection (SDT) model for the first (rater) level of the HRM and an IRT model (typically the generalized partial credit model) for the second level, and apply it to large-scale assessment data. — [DeCarlo et al. 2011 PDF](https://www.tc.columbia.edu/faculty/ld208/faculty-profile/files/JEM_11.pdf); [sirt::rm.sdt docs](https://rdrr.io/cran/sirt/man/rm.sdt.html)
- At the first level of the HRM-SDT, ratings follow an SDT model with a logistic distribution function, estimated by an EM algorithm adapted from multilevel latent-class analysis (sirt documentation). — [sirt rm.sdt](https://search.r-project.org/CRAN/refmans/sirt/html/rm.sdt.html)

**Rater drift**
- Wolfe, Moulder & Myford (2001) define "differential rater functioning over time" (DRIFT) as rater-by-time interactions: primacy/recency, centrality/extremism and practice/fatigue. Rasch-based detection worked well for primacy, recency and differential centrality/extremism at moderate or larger effect sizes; practice/fatigue was harder to classify (secondary). — [ETS listing](https://ets.org/research/policy_research_reports/publications/article/2001/hhqf.html); [Iowa listing](https://iro.uiowa.edu/esploro/outputs/journalArticle/Detecting-differential-rater-functioning-over-time/9985123694402771)
- Myford & Wolfe (2009, JEM) monitored rater performance over time. Some raters showed statistically significant changes in accuracy as scoring progressed, and others showed differential scale-category use over time (secondary). — [Iowa listing](https://iro.uiowa.edu/esploro/outputs/journalArticle/Monitoring-Rater-Performance-Over-Time-A/9985123700002771)
- Uto (2023), Bayesian MFRM with Markov drift: beta_rt ~ N(beta_{r,t−1}, sigma_r) for t > 1, with beta_r1 ~ N(0,1). sigma_r is a per-rater drift scale with prior LN(mu_sigma, 1), and mu_sigma = −2 is recommended. Identification uses the constraints d_r1 = 0 and sum of d_rm = 0, plus the N(0,1) prior on beta_r1. Estimation is NUTS in RStan. — [Uto 2023 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10615980/)
- Uto's results:
  - Markov dependence improved RMSE of beta_rt in all simulated conditions: average improvements of 0.079 (fully crossed), 0.089 (3 raters per examinee) and 0.124 (2 raters per examinee). The benefit grew as data per parameter shrank.
  - WAIC and WBIC favored the Markov model on real data (134 essays, 10 raters, 4 days, plus 5 intentionally biased control raters). Estimated sigma_r were large for raters instructed to drift.
  - Prior sensitivity: mu_sigma = −5 over-shrinks change, and mu_sigma = 0 is unstable.
  - The paper does not explicitly discuss separating rater drift from true change in examinee ability.

  — [Uto 2023 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10615980/)
- Neittaanmäki & Lamprianou (Language Testing, 2024) found that raters as a group became somewhat more lenient over time in a speaking assessment (secondary). This is an example of common-mode group drift. — [JYX repository](https://jyx.jyu.fi/bitstreams/6a3cd46c-1776-4f26-9beb-9268f7b78535/download)

**(e) Software**
- `immer::immer_hrm` (HRM), `sirt::rm.sdt` (HRM-SDT), and RStan code for Uto's Markov MFRM. — [immer_hrm](https://rdrr.io/github/alexanderrobitzsch/immer/man/immer_HRM.html); [sirt rm.sdt](https://rdrr.io/cran/sirt/man/rm.sdt.html); [Uto 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10615980/)

### Inferences
**(c) Mapping to GN**
- The HRM's three levels map directly:
  - examinee proficiency theta maps to the solar activity rate lambda_t, which is smooth;
  - the ideal rating xi maps to the true daily group count G_t, a discrete draw around lambda_t;
  - rater scores map to observer reports Y_{o,t}.
- The HRM's warning about "infinitely precise measurement" carries over: if observers are modeled as independent noisy measures of a smooth lambda_t, days with many observers will overstate precision on lambda_t, because the intrinsic Poisson-like scatter of G_t around lambda_t is shared by all observers that day. The GN paper must say whether the product is G_t (a daily count, which many observers can pin down) or lambda_t (an activity rate with irreducible scatter). Their uncertainty bands differ.
- DeCarlo's SDT rater level maps cleanly onto detection-threshold language. A rater's "criterion" location corresponds to an observer's severity or detection threshold, and "discrimination" to an observer's acuity or consistency. For counts, the natural analog is the threshold-thinning model of Q1 rather than ordinal criteria. Interpreting SDT criteria and discrimination this way is my reading, not taken from the retrieved text.
- Uto's Markov severity prior beta_rt ~ N(beta_{r,t−1}, sigma_r) is directly usable for an observer's log-acuity or threshold over a career, with an observer-specific drift scale sigma_o under a hierarchical lognormal prior. The prior-sensitivity result matters: the drift prior's location decides how much apparent change is attributed to the observer versus the Sun. Report results under several mu_sigma values.

**(d) How drift is separated from true change, and why GN is harder**
- In rater studies, the rated objects (essays) have fixed quality and are rated at different times by different raters. If quality is fixed and the assignment of objects to time is random or balanced, a time trend in scores belongs to raters.
- In GN, each "object" (day) exists only once and the Sun's true state changes. An observer's drift is identified only relative to contemporaneous observers, assuming their drifts are independent.
- Common-mode drift (all observers of an era changing together, for example through a new group-splitting convention or better telescopes) is formally confounded with true solar change. The likelihood is invariant to theta_o(t) to theta_o(t)·c(t) for all active o together with lambda_t to lambda_t / c(t), under multiplicative thinning.
- This confounding is resolved only by:
  - (i) the random-walk prior's assumption that individual observers change slowly while careers overlap, so the scale is carried through career handoffs;
  - (ii) physical constraints (group-size or area distributions; ratios of spots to groups);
  - (iii) external proxies (geomagnetic indices, cosmogenic isotopes), which other research streams may cover.
- Uto's silence on this point shows the issue is underdeveloped even in psychometrics.

**Pitfalls**
- Time-specific severities without a Markov prior are poorly estimated in sparse designs, which is exactly the GN situation in the 17th and 18th centuries. Uto found the largest gains in the sparsest design.
- A free per-observer random walk combined with a free lambda_t walk is weakly identified wherever an observer has no contemporaries.

### Gaps
- Linacre's original MFRM text and the Facets software documentation were not retrieved in this session. The MFRM equation above is cited via Uto (2023). Facets is a well-known commercial program (winsteps.com) but was not checked here.
- The HRM-SDT parameter details (rater-specific criteria and discrimination) and whether DeCarlo et al. list "severity/leniency-only rater effects" as a limitation of the original HRM or of their own model are ambiguous in the secondary summary. Check the JEM PDF.
- I found no rater-drift paper that explicitly handles a changing true signal together with drifting raters and no re-rated anchor objects, which is the GN-specific configuration. Educational "vertical scaling" or "longitudinal growth plus rater drift" literature may exist but was not searched.

## Q3. Test equating and linking under the NEAT design: chained vs post-stratification vs concurrent calibration; error accumulation along long linking chains ("scale drift"); kernel equating

### Takeaway
Equating is the closest formal analog to GN homogenization, and its central results carry over:
- Chained (daisy-chain) linking accumulates error like a random walk. Linking each new form to a single old form gives the largest and growing scale drift, while linking to two or three old forms stabilizes conversions.
- Simultaneous "multiple linking" estimators (Haberman 2009; Battauz 2017) solve all linking constants on the link graph at once and return standard errors.
- Concurrent calibration (one joint model) is more efficient when the model is right, but less robust to misspecification.
- Chained and post-stratification methods rest on different, untestable missing-data assumptions, which map onto the ADF/PDM "population invariance" assumption in GN.

### Cited Findings
**(a) Core designs and (b) references**
- Kolen & Brennan, *Test Equating, Scaling, and Linking* (3rd ed., Springer 2014), is the standard textbook. — [Springer](https://link.springer.com/book/10.1007/978-1-4939-0317-7) (not opened in this session)
- von Davier, Holland & Thayer, *The Kernel Method of Test Equating* (Springer 2004), defines kernel equating. The `kequate` R package implements it for the EG, SG, CB, NEAT chain (CE), NEAT post-stratification (PSE) and NEC designs. It uses Gaussian, logistic or uniform kernels and provides standard errors of equating and of the difference between two equating functions for all designs. — [kequate manual](https://cran.r-universe.dev/kequate/doc/manual.html); [kequate CRAN mirror](https://www.stats.bris.ac.uk/R/web/packages/kequate/index.html)
- `kequate` reference paper: Andersson, Bränberg & Wiberg (2013), JSS 55(6):1-25, doi:10.18637/jss.v055.i06. — [kequate manual](https://cran.r-universe.dev/kequate/doc/manual.html)
- PSE (frequency estimation) reweights both groups to a common anchor-score distribution and then equates. CE equates form X to the anchor and then the anchor to form Y. — [Cambridge Assessment, "Can AI learn to equate?"](https://www.cambridgeassessment.org.uk/Images/424229-can-ai-learn-to-equate-.pdf); [ERIC EJ1110992](https://files.eric.ed.gov/fulltext/EJ1110992.pdf) (secondary summary of these results)
- When the two groups differ greatly in ability, CE showed smaller equating bias than PSE, because the anchor score fails to remove unintended group differences. PSE is regarded as more theoretically sound, yet CE is more widely used in practice (secondary, from the same result set). — [ERIC EJ1110992](https://files.eric.ed.gov/fulltext/EJ1110992.pdf); [PARE 2020](https://openpublishing.library.umass.edu/pare/article/2020/galley/1985/view/)
- Sinharay & Holland (2009), in a comparison of the methods' missing-data assumptions, conclude that chain equating is "somewhat more satisfactory overall" under NEAT (secondary). The related ETS work: Holland, Sinharay, von Davier & Han (2008) on evaluating the missing-data assumptions of CE and PSE. — [ETS 2009 report page](https://www.ets.org/research/policy_research_reports/publications/report/2009/hvss.html); [ETS 2008 article page](https://www.ets.org/research/policy_research_reports/publications/article/2008/gpll.html)
- Sinharay & Holland (2006) found that anchors with a narrower spread of difficulty than the total test perform about as well as a "minitest" anchor in equating bias and SE. — [ETS "Choice of Anchor Test in Equating"](https://www.ets.org/research/policy_research_reports/publications/report/2006/hslb.html)

**Long chains and scale drift (secondary summaries; full texts not retrieved)**
- Haberman & Dorans: only systematic error induces scale drift, while random error does not. Haberman (2010) analyzes systematic error in equating chains. — [ETS "Sources of Score Scale Inconsistency"](https://www.ets.org/research/policy_research_reports/publications/report/2011/imyj.html); summarized in [CASMA Research Report 54](https://education.uiowa.edu/sites/education.uiowa.edu/files/2022-10/casma-research-report-54.pdf)
- Guo, Liu, Dorans & Feigenbaum (2011): single-link conversions drift further from operational ones as more equatings are carried out, and the paper proposes a random-walk model for drift caused by random equating error. — [ETS "Multiple Linking in Equating and Random Scale Drift"](https://www.ets.org/research/policy_research_reports/publications/report/2011/jdyg.html)
- The two summaries disagree on random error: Haberman & Dorans say it does not induce scale drift, while Guo et al. model drift as a random walk driven by it. The likely reconciliation is that random error produces non-systematic but accumulating divergence (variance growing with the number of links) and systematic error produces directional drift. — same sources as the previous two items
- Puhan (2007) compared parallel equating chains with a single long chain. Drift between conversions was present, but its effect on pass/fail status was small. — [ETS "Scale Drift in Equating on a Test That Employs Cut Scores"](https://www.ets.org/research/policy_research_reports/publications/report/2007/hsnt.html)
- Liu, Guo & Dorans (2014): equating to a single old form gave the greatest scale drift, which tended to increase over time. Equating to two or three old forms gave more stable conversions with less variation (secondary). — summarized in [CASMA Research Report 54](https://education.uiowa.edu/sites/education.uiowa.edu/files/2022-10/casma-research-report-54.pdf)
- Kim, Kim & Moses: equating error accumulates over a series of equatings, so cumulative error must be monitored to avoid scale drift (secondary). — summarized in [CASMA Research Report 54](https://education.uiowa.edu/sites/education.uiowa.edu/files/2022-10/casma-research-report-54.pdf)

**Simultaneous multiple-form linking (the graph analog)**
- `equateMultiple` (Battauz) equates many separately calibrated forms at once. It implements multiple mean-geometric mean (Haberman 2009), multiple mean-mean, multiple item-response-function and multiple test-response-function methods (Battauz 2017, Psychometrika 82:610-636), and likelihood-based linking (Battauz & Leoncio 2023). It returns equating coefficients, item parameters on a common scale, and standard errors for both. The package is described in JSS 2025, doi:10.18637/jss.v115.i11. — [equateMultiple CRAN refman](https://cran.r-project.org/web/packages/equateMultiple/refman/equateMultiple.html)
- Battauz (Statistica Neerlandica 2022), "Testing for differences in chain equating", develops tests for whether equating through different chains gives different results (inferred from title and repository listing; full text not read). — [Univ. Udine repository PDF](https://air.uniud.it/retrieve/f290da09-07d4-4033-8235-48704184e713/Statistica%20Neerlandica%20-%202022%20-%20Battauz%20-%20Testing%20for%20differences%20in%20chain%20equating.pdf)
- Robitzsch (2025, AppliedMath 5(1):7) proposes a pairwise Haberman-linking variant that avoids joint item-parameter estimation and eases linking-error computation under differential item functioning. A companion 2025 paper compares robust Haberman linking with invariance alignment. — [Leibniz-IPN listing](https://www.leibniz-ipn.de/en/research/publications/linking-error-estimation-in-haberman-linking); [robust Haberman vs alignment](https://www.leibniz-ipn.de/en/research/publications/comparing-robust-haberman-linking-and-invariance-alignment)

**Concurrent vs separate calibration**
- Hanson & Béguin (2002, APM 26(1):3-24) compared separate and concurrent IRT estimation in the common-item nonequivalent-groups design by simulation. The design had 60 items and varied the program (MULTILOG vs BILOG-MG), sample size (3,000 vs 1,000), number of common items (20 vs 10) and group equivalence. — [Univ. Twente record](https://research.utwente.nl/en/publications/obtaining-a-common-scale-for-item-response-theory-item-parameters/)
- Hanson & Béguin (2002) is commonly cited, with Kim & Cohen (1998), for the view that concurrent calibration is more accurate when the data fit the IRT model (secondary). — [NWEA Robust IRT Scaling paper](https://www.nwea.org/uploads/2021/06/Robust-IRT-Scaling_NWEA_conferencePaper.pdf); [Asia Pacific Educ. Rev. article](https://mijn.bsl.nl/article/10.1007/s12564-011-9197-2)
- Kim & Cohen (1998) found separate and concurrent estimation similar, except that separate estimation was more accurate when the number of common items was small (as reviewed in Hanson & Béguin's 1999 ACT report). — [ACT RR 99-08](https://act.org/content/dam/act/unsecured/documents/ACT_RR99-08.pdf)
- Béguin & Hanson (2001), with noncompensatory multidimensional data: unidimensional concurrent estimation generally gave lower or equal total error than separate estimation, with a few exceptions. — [Cito 2001 report](https://cito.nl/media/vonblb4u/2001-effect-of-noncompensatory-multidimentionality-on-separate-and-concurrent-estimation-in-irt-observed-score-equating.pdf)

### Inferences
**(c) Mapping to GN**
- An equating "form" is an observer's counting scale, including its detection threshold and split/merge convention.
- "Examinee groups" are the populations of days each observer covered. They are non-equivalent because solar activity differs between careers.
- The NEAT "anchor" is the set of common observing days. Strictly this is common persons, not common items, so the overlap days form a single-group sub-design embedded in a non-equivalent-groups design.
- The method correspondences:
  - daisy-chain/backbone calibration is chained equating across many forms;
  - PDM or regression calibration on overlap days, applied to the observer's whole career, is observed-score equating that assumes population invariance (the linking function estimated on overlap days holds at all activity levels). This is PSE-like: it assumes P(observer scale | anchor value) is invariant across subpopulations;
  - the ADF approach, which matches an observer's active-day distribution to a reference's, resembles equipercentile equating under an "equivalent groups conditional on activity level" assumption;
  - a single joint latent model of all observers is concurrent calibration;
  - Haberman/Battauz multiple linking is a graph-based global solve: estimate pairwise links with their SEs, then least-squares-reconcile all observer-to-reference transforms. This is a principled version of the existing "graph shortest-path" variant that uses all paths instead of one.
- The test of whether ADF calibration depends on activity level (arXiv 1803.10501) is the GN version of a population-invariance check (Dorans-Holland style). It should be standard for any calibration method.

**Error growth on the GN link graph (my derivation from the random-walk drift model)**
- If each pairwise link adds independent error variance v_e, an observer reached through a single chain of L links has scale variance relative to the anchor of sum v_e, about L·v. This is Guo et al.'s random walk.
- With multiple linking, the variance equals the effective resistance between the observer node and the anchor in a network whose edge conductances are 1/v_e. Parallel paths reduce it (two independent equal paths halve it), which matches Liu-Guo-Dorans's finding that linking to two or three old forms stabilizes conversions.
- For GN this gives a quantitative, design-level way to report how uncertain the 18th-century scale is relative to the modern one, and which observers are critical bridges (high "current" edges). It also turns "path choice is arbitrary" into a well-posed weighted least-squares or Bayesian problem.
- Systematic link errors (for example a PDM whose population-invariance assumption fails at low activity) do not average down over paths that share the same bias. Haberman & Dorans's systematic-error point implies that bias, not variance, is the main long-chain danger. In GN the obvious systematic sources are activity-level dependence of calibration and common-mode era conventions.

**Concurrent vs chained for GN**
- The evidence (Hanson & Béguin; Kim & Cohen; Béguin & Hanson) suggests that a correctly specified joint model beats separate-then-link, while separate linking is safer when there are few common items, the analog of short overlaps. GN observers with short overlaps are common.
- Practical recommendation:
  1. Build the concurrent Bayesian model as the primary product.
  2. Run a Haberman-style graph linking of separately estimated pairwise calibrations as a robustness and diagnostic layer.
  3. Use Battauz-style tests of path consistency to flag links whose implied scales disagree, which reveals misfit, drift or data errors.
- Kernel equating's standard errors of the difference between two equating functions are the template for testing whether two calibration routes (for example ADF vs backbone) differ significantly for an observer.

**(d) Pitfalls**
- NEAT methods rely on untestable missing-data assumptions (Holland et al. 2008). GN calibration has the same character: each observer's counts on days they did not observe are missing, and missing-at-random fails for zero days (under-reporting) and for weather-driven gaps correlated with season.
- Scale indeterminacy is handled in equating by declaring a base form; in GN by declaring a reference observer or era. The choice affects only the location of uncertainty growth, not the relative series, if the model is right.

**(e) Software**
- R: `kequate`, `equateIRT`, `equateMultiple`, `sirt` (Haberman linking). No JAX equivalents are known. The Haberman multiple-linking estimator is a small weighted least-squares problem that is easy to reimplement in JAX, or to embed as a likelihood in NumPyro.

### Gaps
- I could not locate or verify Haberman's "Limits on the accuracy of linking" (2010) or extract quantitative error-growth formulas from Haberman's chain papers; ETS pages returned header-overflow errors. The claims above rest on secondary summaries.
- The exact quantitative conclusions of Hanson & Béguin (2002) were not retrieved; the PDFs could not be parsed. They are reported only through citing papers.
- No retrieved source directly compares error growth of concurrent calibration vs chained linking over many (more than 10) linked forms in a single study. Vertical-scaling and longitudinal-linking literature (for example Kolen & Brennan's chapters on vertical scaling) may contain this but was not accessed.
- Fixed-parameter calibration drift (holding the reference fixed and calibrating sequentially, the literal backbone analog) was not searched.

## Q4. Dynamic paired-comparison and skill rating across eras without direct contact (TrueSkill Through Time, Glicko/Glicko-2, dynamic Bradley-Terry, Whole-History Rating, historical chess ratings): information propagation, uncertainty far from anchors, and inflation/deflation

### Takeaway
Under a Poisson observation model, GN calibration between two observers on a common day reduces exactly to a dynamic Bradley-Terry comparison: conditional on the pair's total, Y_A ~ Binomial(Y_A+Y_B, k_A/(k_A+k_B)). So the historical-chess rating literature is a direct analog for propagating an observer scale through a sparse, time-evolving overlap graph. Smoothing methods (TTT, WHR, Edo) that use the whole history beat forward-only filters (Elo, Glicko, TrueSkill) for early-era estimates and cross-era comparability. In all of them the global level at each epoch ("inflation/deflation") is not identified by the data and is fixed by priors: random-walk smoothness of individual players plus entrant priors. This is exactly GN's common-mode scale-drift problem.

### Cited Findings
**(a) Core models and (b) references**
- Elo's system is a special case of the Bradley-Terry model. Zermelo (late 1920s) estimated chess strengths in incomplete round-robins. — [Wikipedia: Elo rating system](https://en.wikipedia.org/wiki/Elo_rating_system)
- Glickman (1999), "Parameter Estimation in Large Dynamic Paired Comparison Experiments" (JRSS C 48(3):377-394), models time-varying paired-comparison abilities as a nonlinear state-space model. It fits them with a computationally simple non-iterative algorithm that improves on Elo by carrying parameter variability (this is Glicko), and applies it to ranking chess players of all time and top tennis players. Glicko-2 (Glickman 2001) adds stochastic volatility. — [IDEAS](https://ideas.repec.org/a/bla/jorssc/v48y1999i3p377-394.html); [glicko.net research](https://glicko.net/research.html)
- Coulom (2008), Whole-History Rating (Computers and Games, LNCS 5131:113-124): a dynamic Bradley-Terry model whose solution is the most probable rating history of all players jointly, rather than incremental updates. A player's later rating is Gaussian around the earlier one (a Wiener process with hyperparameter w controlling drift speed). It reports better prediction than Elo, Glicko, TrueSkill and decayed-history methods, at higher but still real-time-feasible cost. — [WHR page](https://www.remi-coulom.fr/WHR/); [Renju News on WHR](https://renjunews.com/?p=3109)
- WHR implementations and uses: a Python `whr` package (C++ core); BoxRec boxing ratings; Coulom's Go ratings. — [PyPI whr](https://pypi.org/project/whr/); [BoxRec ratings description](https://boxrec.com/wiki/index.php/BoxRec_Ratings_Description)
- Dangauthier, Herbrich, Minka & Graepel (NIPS 2007), "TrueSkill Through Time: Revisiting the History of Chess":
  - each player's skill in a given year is latent and linked to the previous and following years;
  - the model includes player-specific draw margins;
  - inference is approximate message passing (expectation propagation) that smooths through time rather than filtering;
  - they report that overall playing strength increased over 150 years and that modeling draw ability significantly improves prediction.

  — [NeurIPS proceedings](https://papers.nips.cc/paper/3331-trueskill-through-time-revisiting-the-history-of-chess)
- Landfried & Mocskos (2025), JSS 112(6):1-41, doi:10.18637/jss.v112.i06:
  - Elo and TrueSkill pass information only forward in time, whereas TTT "propagates all historical information throughout a single causal network", giving low-uncertainty estimates at each time and comparability across distant times and places;
  - analytical approximations plus message passing make inference feasible on low-end computers for networks with millions of nodes;
  - software for Julia, Python and R; demonstrated on historical ATP tennis.

  — [JSS article](https://www.jstatsoft.org/article/view/v112i06)
- TTT Python defaults are mu = 0, sigma = 6, beta = 1, gamma = 0.03 (per-step skill drift; gamma = 0 means static skills) and p_draw = 0. The History object takes timestamps; `convergence()` iterates message passing (for example epsilon = 0.01, 10 iterations). After convergence, two identical players receive equal estimates with reduced uncertainty. The README states that TrueSkill, Glicko and IRT do not correctly model the temporal aspect in the way TTT does. — [TrueSkillThroughTime.py GitHub](https://github.com/glandfried/TrueSkillThroughTime.py)
- Cattelan, Varin & Firth (2013), "Dynamic Bradley-Terry modelling of sports tournaments" (JRSS C 62(1):135-150): team abilities evolve over time, driven by past results through exponentially weighted moving-average processes, applied to the NBA and Serie A. It remains a common benchmark; a 2026 esports preprint uses it as the main baseline. — [IDEAS](https://ideas.repec.org/a/bla/jorssc/v62y2013i1p135-150.html); [Univ. Venice repository](https://iris.unive.it/handle/10278/30886); [arXiv 2609.08060](https://arxiv.org/pdf/2609.08060)
- Further dynamic-BT work: Gaussian-process priors for dynamic paired comparisons and nonparametric (kernel-smoothed) estimation of dynamic BT. — [arXiv 1902.07378](https://arxiv.org/pdf/1902.07378); [arXiv 2003.00083](https://ar5iv.labs.arxiv.org/html/2003.00083)

**Historical chess ratings and inflation**
- Edo (Rod Edwards) is a retroactive system that computes ratings over the whole period at once with an iterative Bradley-Terry method on the network of player interactions over time. Edwards's pages list spans of 1814-1941 and 1814-1947 (inconsistent). — [ChessBase on Edo](https://en.chessbase.com/newsroom/post/historical-chess-ratings-dynamically-presented); [Edwards board-games page](https://web.uvic.ca/~edwards/boardgames.html)
- Chessmetrics (Jeff Sonas) rates players retrospectively and claims to correct for the rating inflation the Elo system apparently suffered. Sonas holds that ratings measure dominance over contemporaries rather than absolute skill. A Chessmetrics rating drops after a month without play. One cross-era ranking uses each player's largest gap over the world No. 100 as an inflation adjustment. — [Wikipedia: Comparison of top chess players throughout history](https://en.wikipedia.org/wiki/Comparison_of_top_chess_players_throughout_history)
- Sonas has reportedly moved from worrying about rating inflation to worrying about deflation, and FIDE proposed rating changes partly on his advice (2023 podcast description). — [Perpetual Chess Podcast ep. 343](https://www.perpetualchesspod.com/new-blog/2023/8/15/ep-343-rating-deflation-roundtable-with-statistician-jeff-sonas-and-data-scientist-fm-nate-solon-discuss-fides-proposed-changes-to-their-rating-system)

### Inferences
**(c) Mapping to GN: observers as players**
- Suppose Y_{o,t} ~ Poisson(k_o(t)·lambda_t) independently given the Sun. Then conditional on n = Y_A + Y_B on a common day, Y_A ~ Binomial(n, k_A/(k_A+k_B)), a Bradley-Terry likelihood with strengths k. This is a standard Poisson identity, not taken from the sources.
- lambda_t drops out (a conditional-likelihood, Rasch-style elimination of the nuisance), so overlap days directly give "match results" between observers.
- log k_o(t) plays the role of skill, with a random-walk prior (TTT gamma, WHR w) for career drift.
- Days with three or more observers are multi-player matches, which TTT handles natively.
- Mathieu et al.'s near-Poisson dispersion for group counts (slope about 0.96; Q1) makes this approximation reasonable at moderate activity. It breaks down for nonlinear threshold effects (the Chatzistergos argument against linear scaling) and at low activity, where zeros dominate. A threshold-thinning model is then needed and the exact BT reduction is lost, but the state-space structure remains.

**Information propagation through a sparse, time-evolving graph**
- Forward filters (Elo, Glicko, TrueSkill, and effectively the backbone method run from the present backward) propagate information in one direction only. Early observers' estimates then ignore later evidence, and the result depends on processing order and on path choice.
- Smoothers (TTT forward-backward EP, WHR joint MAP, Edo global BT) use all links in both directions. They are the rating-system equivalent of concurrent calibration (Q3).
- The JSS paper's claim of comparability across distant times is the property GN needs.

**Uncertainty far from the anchor**
- In a random-walk state-space model, a player's posterior variance grows linearly with time since the last informative comparison (gamma^2·Delta t per step in TTT; w^2·Delta t in WHR). It accumulates along chains of players who connect two eras, and shrinks wherever many overlapping players connect.
- For GN, the 17th- and 18th-century scale uncertainty relative to the modern anchor equals the accumulated random-walk variance along career handoffs: a temporal version of Q3's effective-resistance result.
- Observer gaps, such as a long hiatus within a career, should inflate that observer's acuity uncertainty in proportion to the gap length.

**Inflation/deflation equals GN scale drift**
- The BT likelihood depends only on differences of log-strengths, so adding c(t) to every active player's skill at time t leaves the likelihood unchanged.
- TTT's 2007 finding that "overall playing strength increased over 150 years" is therefore a prior-driven inference: it depends on individual skills changing slowly and on the prior mean for newcomers.
- Chessmetrics and Sonas sidestep the problem by defining ratings as dominance over contemporaries, which is the relative-scale choice. GN cannot do that, because an absolute count scale is the scientific target.
- The GN analog is exact: era-wide observer drift is confounded with solar change, and only overlapping careers combined with a slow-drift prior transfer the absolute scale.
- Consequences for a GN methods paper:
  - (i) show sensitivity of the reconstructed Maunder-to-modern scale ratio to the observer drift prior (gamma or sigma_o), as Uto did (Q2);
  - (ii) report the posterior of the scale-transfer factor as a function of epoch;
  - (iii) where possible, use physically anchored constraints that break the c(t) symmetry (for example, structural features of the count distribution such as the fraction of single-spot groups, which a pure scaling cannot change).

**Rating-system design choices relevant to GN**
- Player-specific draw margins (TTT) correspond to an observer-specific zero-reporting or rounding behavior.
- A separate "home/away" effect (Cattelan) corresponds to instrument or site effects shared by observers at an observatory.
- Glicko-2's stochastic volatility corresponds to observers whose drift rate itself changes, for example a telescope change mid-career.

**(e) Scalability and software**
- TTT handles millions of nodes on low-end hardware, which exceeds GN scale (about 600 observers, at most about 100k observer-days of overlap).
- For a JAX/NumPyro build, the dynamic-BT likelihood is a few lines. A random walk on log k_o can be non-centered for NUTS, or a Kalman/EP smoother can be used for speed. TTT (Python/Julia/R) and `whr` are useful baselines and cross-checks.

### Gaps
- I could not extract WHR's specific anchoring device (Coulom's prior or virtual-game construction for each player's first rating) or its covariance/uncertainty computation, because the PDF could not be parsed.
- The size of the TTT 2007 chess dataset, its specific uncertainty plots for early players, and its comparison with Chessmetrics and Edo were not on the abstract page.
- Chessmetrics' exact inflation-correction formula and Edo's handling of sparse early networks (for example any hypothetical or prior games) were not found.
- No source was found that explicitly analyzes rating identifiability of a global time-varying offset in TTT or WHR. The inference above is mathematical.

## Q5. Applications of these ideas to physical-science instrument or observer intercalibration

### Takeaway
Physical-science analogs exist at two levels of sophistication:
- **Astronomy citizen science.** Space Warps and Galaxy Zoo Supernovae used per-volunteer confusion matrices (Dawid-Skene-style), including time-varying ones (DynIBCC).
- **Solar instrument merging.** The TSI composite of Dudok de Wit et al. (2017) merges all instrument records with a probabilistic multiscale maximum-likelihood method, normalizes to the most accurate instruments, and produces time-dependent uncertainties. Sunspot-network uncertainty quantification (Mathieu et al.) estimates a common latent solar signal with observer-specific multiplicative errors and zero-altered models.

None of these does a full concurrent latent-truth calibration with drifting observers over centuries, which is the GN gap.

### Cited Findings
- **Space Warps** (Marshall et al. 2016, MNRAS 455:1171; arXiv 1504.06148): each volunteer gets a software "agent" with a 2-by-2 confusion matrix, learned from performance on training images with simulated lenses and known non-lenses. The agent updates after each classification, classifications are weighted by estimated volunteer skill, and the prior lens probability is 2×10^-4 per subject. — [arXiv 1504.06148](https://arxiv.org/pdf/1504.06148); [RERO record](https://doc.rero.ch/record/301978)
- In a reuse of the SWAP pipeline, each volunteer's initial confusion matrix is set to 0.5 (random) until data update it. — [arXiv 1707.05223](https://arxiv.org/pdf/1707.05223)
- **Galaxy Zoo Supernovae / DynIBCC** (Simpson, Roberts, Psorakis, Smith): Bayesian combination of imperfect human classifiers with priors for sparse training data. The model is extended to track changing classifier performance over time, and decision makers form natural groupings by inferred strategy. — [arXiv 1206.1831](https://arxiv.org/abs/1206.1831v1); [Springer chapter](https://link.springer.com/doi/10.1007/978-3-642-36406-8_1)
- **TSI composite** (Dudok de Wit, Kopp, Fröhlich & Schöll 2017, GRL 44:1196-1203): a probabilistic method for merging irradiance records that uses a multiscale maximum-likelihood estimator. It handles gaps, uses all instruments, weights them without bias, normalizes to the most accurate instruments, and gives time-dependent uncertainties with 1/f scaling. — [arXiv 1702.02341](https://arxiv.org/abs/1702.02341v1); [CU Experts listing](https://experts.colorado.edu/display/pubid_153164)
- **Sunspot networks** (Mathieu et al.): per-station scaling factors kappa_i fitted against the network median over blocks of 8, 14 and 10 months for N_s, N_g and N_c. The common signal is a transformed network median (Anscombe transform, Wiener filtering). The model separates short-term, long-term and minima errors, uses zero-altered (hurdle) distributions and negative-binomial mixtures, and proposes near-real-time drift monitoring (for example CUSUM), noting CUSUM's poor fit to skewed counts. — [arXiv 2009.09810 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2009.09810); [Mathieu thesis](https://sidc.be/valusun/web/pdf/Thesis_Sophie_Mathieu.pdf)
- **GN-specific baselines**: the Chatzistergos et al. (2017) PDM backbone calibration of 314 observers with Monte Carlo errors, and the activity-level dependence test of ADF calibration. — [arXiv 1702.06183](https://arxiv.org/abs/1702.06183); [arXiv 1803.10501](https://arxiv.org/pdf/1803.10501)

### Inferences
- Space Warps and DynIBCC show that per-observer confusion matrices, gold-standard anchoring (simulated lenses play the role of modern reference days), and time-varying observer skill all work at the scale of thousands of observers. GN is smaller in observers but longer in time and sparser in overlap.
- The TSI composite has the same structure as GN: multiple instruments, gaps, degradation (drift), and a latent true series. Its key design choices are:
  - use all records simultaneously (concurrent) rather than daisy-chaining;
  - normalize the absolute scale to designated reference instruments;
  - report time-dependent uncertainty.

  These match the recommendations from Q3 and Q4. The GN version adds discrete counts and threshold nonlinearity.
- Mathieu et al. estimate the common signal by a robust network median and per-station scalings, a "separate estimation" approach in equating terms. A concurrent Bayesian model would subsume it: kappa_i becomes an observer parameter, the median becomes the posterior of lambda_t, the hurdle becomes the reporting/zero model, and the long-term error eps2 becomes the observer random walk theta_o(t). Their empirical dispersion and zero-handling results are directly reusable as likelihood choices.

### Gaps
- Satellite radiometer intercalibration (for example GSICS, simultaneous-nadir-overpass chains) and other daisy-chained climate-record homogenizations were not researched here. Likely relevant chained-vs-global evidence exists in that literature.
- Bayesian or state-space TSI composites published after 2017 were not searched.
- No published GN reconstruction using a full Dawid-Skene, HRM or dynamic-BT latent-truth model was found in this session. A targeted search ("Bayesian hierarchical group sunspot number observers latent") was not run and may be covered by other researchers.

## Cross-cutting synthesis: global/concurrent vs chained calibration, and a combined GN model

### Takeaway
All three method families agree:
- chained, path-dependent calibration accumulates error like a random walk and is order- and path-dependent;
- global methods (concurrent calibration, multiple linking, whole-history smoothing) use all links, reduce variance through parallel paths and provide coherent uncertainty;
- no method identifies a common-mode (era-wide) scale shift from relative data alone.

The recommended GN architecture is a concurrent hierarchical state-space model (HRM structure, Dawid-Skene-style observation model, TTT/WHR- or Uto-style random-walk observer drift), with a Haberman/Battauz-style graph-linking layer as a diagnostic.

### Cited Findings
- Linking to multiple old forms reduces scale drift compared with a single chain. — summarized in [CASMA RR 54](https://education.uiowa.edu/sites/education.uiowa.edu/files/2022-10/casma-research-report-54.pdf); [ETS Guo et al. 2011](https://www.ets.org/research/policy_research_reports/publications/report/2011/jdyg.html)
- Simultaneous linking of many forms with standard errors is available (Haberman 2009; Battauz 2017). — [equateMultiple](https://cran.r-project.org/web/packages/equateMultiple/refman/equateMultiple.html)
- Concurrent calibration is more accurate when the model fits, while separate estimation can be more accurate with few common items. — [NWEA](https://www.nwea.org/uploads/2021/06/Robust-IRT-Scaling_NWEA_conferencePaper.pdf); [ACT RR 99-08](https://act.org/content/dam/act/unsecured/documents/ACT_RR99-08.pdf)
- Whole-history methods outperform forward-only filters and allow cross-era comparison. — [WHR](https://www.remi-coulom.fr/WHR/); [Landfried & Mocskos 2025](https://www.jstatsoft.org/article/view/v112i06)
- Markov (random-walk) priors on rater severity improve estimates most in sparse designs but are prior-sensitive. — [Uto 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10615980/)
- The HRM structure avoids the spurious infinite precision that arises when ratings of the same object are treated as independent measures of ability. — [Patz et al. 2002](https://www.stat.cmu.edu/~brian/trs/patz-et-al-hrm-jebs-2002.pdf)

### Inferences
**Proposed GN model (NumPyro)**
- **Solar level:** log lambda_t follows a GP or random walk on a daily or Carrington-rotation grid.
- **Truth level:** G_t ~ NegBin(lambda_t, phi) or Poisson, marginalized by enumeration over 0 to G_max.
- **Observer level:**
  - P(Y_{o,t} | G_t, theta_o(t)) is a structured confusion matrix combining threshold or binomial thinning with a split/merge kernel;
  - theta_o(t) follows a random walk with drift scale sigma_o ~ hierarchical lognormal (Uto);
  - observers are pooled hierarchically by era and instrument.
- **Reporting level:** P(observer reports on day t | G_t, o), so that under-reported zeros are missing-not-at-random.
- **Anchor:** modern reference observers' theta fixed, or tightly informed by modern data. The scale is propagated back through overlapping careers.

**Outputs and diagnostics**
- (i) A posterior ensemble of G_t and lambda_t.
- (ii) The posterior of the epoch-dependent scale-transfer factor, the GN "inflation/deflation" curve.
- (iii) An effective-resistance map of the overlap graph showing critical bridge observers.
- (iv) Path-consistency tests between alternative linkings (Battauz), and activity-level invariance checks (the ADF-dependence question).
- (v) Sensitivity to the drift-prior hyperparameters.

**Expected trade-off (from equating evidence)**
- The concurrent model is most efficient if the observation model is right. If it is misspecified (for example linear thinning where thresholds are nonlinear), errors can spread globally.
- Therefore run separate pairwise calibrations plus graph linking as an independent check, and treat large disagreement as a model-criticism signal.

### Gaps
- No single source quantitatively benchmarks concurrent vs chained calibration on a long, sparse, chain-like linking graph resembling GN (more than 100 nodes, about 400 years, sparse early era). A simulation study in the methods paper would be novel: synthetic observers with known thresholds and drift on the real GN overlap graph, comparing backbone, ADF, graph least-squares and concurrent Bayesian methods.
- The relative size of systematic (bias) and random link errors in existing GN calibrations is not quantified in any source retrieved here.
