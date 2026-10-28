# Simulation-based inference (A) and scalable latent count state-space inference in JAX/NumPyro (B) for homogenizing the sunspot group number

Scope note: every topic is mapped onto the GN problem. Roughly 600 observers report daily group counts from 1610 to the present. The latent true daily group count follows birth-death dynamics. Each observer has an area threshold, a splitting convention, career drift and possibly missing zeros. RGO/Debrecen (1874 to present) is the area-resolved reference. The record spans about 151,900 daily steps (1610-01-01 to 2025-12-31 is 416 yr x 365.2425 d). References marked † are canonical works cited from their published arXiv or DOI identifiers. I did not re-fetch them in this session. Unmarked references were opened or found by search in this session. Cost figures in the Inferences sections are my own back-of-envelope arithmetic, not measured benchmarks.

---

## A1. NPE, NLE and NRE, and ABC with learned summaries: can an amortized posterior over observer parameters replace the EMD sliding-window search?

### Takeaway
Yes, within the reference era. A neural posterior estimator q(θ_o | x_o) can be trained on synthetic observers made by thresholding and masking RGO/Debrecen windows. Here θ_o = threshold, splitting factor, drift, missing-zero rate and observing probability, and x_o = the observer's raw daily record plus its observing mask. It returns a calibrated-able posterior for each of the ~600 observers in one forward pass. That replaces "k-nearest-neighbour ABC with EMD on ADF histograms". However, per-observer SBI against reference windows keeps two structural weaknesses of the ADF method:
- it ignores the cross-observer overlap information;
- it assumes the observer's epoch looks like some reference window, which fails for the Maunder Minimum.

So it is best used as an emission-model calibrator, prior or proposal generator inside a joint likelihood-based model (Part B), not as the final GN engine.

### Cited Findings
- **Core idea (NPE).** Papamakarios & Murray (2016) fit a conditional density estimator q(θ|x) (a mixture density network) to simulated (θ, x) pairs. This gives "ε-free" posteriors with no tolerance kernel, unlike ABC. — [Papamakarios & Murray 2016 †](https://arxiv.org/abs/1605.06376)
- **Sequential NPE.** Automatic Posterior Transformation (APT/SNPE-C) lets NPE be trained sequentially on proposals focused near the observed data while still recovering the correct posterior. — [Greenberg, Nonnenmacher & Macke 2019 †](https://arxiv.org/abs/1905.07488)
- **NLE.** Sequential neural likelihood learns q(x|θ) and samples the posterior with MCMC. This makes it easy to multiply likelihoods of many observations. — [Papamakarios, Sterratt & Murray 2019 †](https://arxiv.org/abs/1805.07226)
- **NRE.** A classifier is trained to tell joint (θ, x) pairs from marginal pairs. Its logit estimates log p(x|θ)/p(x), which is then used inside MCMC. — [Hermans, Begy & Louppe 2020 †](https://arxiv.org/abs/1903.04057)
- **Review.** The canonical review frames SBI as amortized neural surrogates of the posterior, likelihood or ratio for simulators with intractable likelihoods. — [Cranmer, Brehmer & Louppe 2020, PNAS †](https://www.pnas.org/doi/10.1073/pnas.1912789117)
- **Practical workflow (2025).** "Simulation-Based Inference: A Practical Guide" (Deistler et al., 14 authors, Tübingen) covers simulator and prior setup, method and architecture choice, training, validation and interpretation, with astrophysics, psychophysics and neuroscience examples. The companion code is at sbi-dev/sbi-practical-guide. — [Deistler et al. 2025](https://arxiv.org/abs/2508.12939)
- **Review of neural amortized inference.** Covers neural Bayes estimators, NPE and summary learning from a statistics perspective. — [Zammit-Mangion, Sainsbury-Dale & Huser 2024 †](https://arxiv.org/abs/2404.12484)
- **Learned ABC summaries (semi-automatic ABC).** If the summary is the posterior mean E[θ|x], the ABC posterior mean matches the true posterior mean as the tolerance goes to 0 under quadratic loss. In practice E[θ|x] is estimated by regression on pilot simulations. — [Fearnhead & Prangle 2012, JRSS-B †](https://arxiv.org/abs/1004.1112)
- **Neural approximate sufficient statistics.** Summaries are learned by maximizing the mutual information between summary and parameter (infomax), then used in ABC or sequential neural likelihood. — [Chen et al. 2021, ICLR †](https://arxiv.org/abs/2010.10079)
- **Summaries for Markovian time series.** Partially exchangeable networks (PENs) build Markov-order invariance into summary networks for time-series ABC. This matches daily count sequences. — [Wiqvist et al. 2019, ICML †](https://arxiv.org/abs/1901.10230)
- **ABC regression adjustment** corrects accepted samples toward the observed summaries. It is the classical fix for coarse acceptance rules such as "keep the N best". — [Beaumont, Zhang & Balding 2002 †](https://doi.org/10.1093/genetics/162.4.2025)
- **BayesFlow.** A summary network and an invertible (normalizing-flow) posterior network are trained jointly, so the summaries are learned end to end for amortized inference. — [Radev et al. 2020 †](https://arxiv.org/abs/2003.06281)
- **Missing or unstructured data.** Simformer (transformer plus score-based diffusion) can sample arbitrary conditionals of the joint over parameters and data, including posterior and likelihood. It also handles missing or unstructured data, which is relevant to irregular observing masks. — [Gloeckler et al. 2024, ICML †](https://arxiv.org/abs/2404.09636)
- **Truncated sequential NPE** focuses simulations on the prior region supported by the data without APT's leakage issues. — [Deistler, Goncalves & Macke 2022 †](https://arxiv.org/abs/2210.04815)
- **Flow-matching and consistency-model NPE** are already implemented in sbijax, alongside NLE, NPE and contrastive NRE. — [sbijax README](https://github.com/dirmeier/sbijax)

### Inferences
**(c) Mapping to GN.**
- **The current ADF/EMD method is ABC.** Its summary is the histogram of monthly active-day fractions, or the daily-count distribution. Its distance is the EMD. Its acceptance rule keeps the N best matches, which is kNN-ABC. The window start date is an implicit nuisance parameter for "activity level during the career". Its uncertainty is "spread of the N best", which is not a posterior. The tolerance is implicit, and there is no regression adjustment or calibration.
- **Simulator.** For a sample of θ_o and a random reference start date τ:
  1. take the RGO/Debrecen group list in [τ, τ + career length];
  2. apply the observer's actual observing-day mask, or a mask drawn from a mask model;
  3. drop groups with area < A_thr·exp(noise);
  4. apply split/merge, for example y = d + Poisson(s·d) or a merge kernel;
  5. apply drift as a slow multiplicative trend on A_thr;
  6. delete zero-count reports with probability r_o.

  This is an array-thresholding operation that can be vmapped in JAX. About 10^6 synthetic careers of ~10^4 days each is about 10^10 cheap element operations, which takes minutes on a GPU. Training the network dominates the cost (hours).
- **Network input.** Use the raw daily sequence as (day index, count, observed flag) tokens, embedded with:
  - a 1-D CNN or temporal-convolution stack over the dense daily array with the mask as a channel; or
  - a set/transformer embedding over observed days only. This copes with career lengths from 5 to 40 yr and with irregular gaps.

  Output: q(θ_o, τ-or-activity-level | x_o).
- **Amortization payoff.** One trained network serves all ~600 observers and any re-runs. The ADF sliding-window search is redone per observer and per threshold grid.
- **NLE variant, often more useful here.** Learn an emission likelihood q(y_{o,t} | latent state, θ_o) from reference simulations. It plugs into the Part-B forward algorithm, so the latent activity series stays likelihood-based and cross-observer information is pooled exactly.
- **Learned summaries inside the existing ADF code.** This is the cheapest upgrade. Replace ADF histograms with Fearnhead-Prangle or infomax summaries (a small network trained to regress θ_o), keep rejection or SMC-ABC, and add regression adjustment. The existing pipeline then gets proper ABC posteriors with little re-engineering.

**(d) Pitfalls.**
- **Missing pooling across observers.** Per-observer NPE ignores that overlapping observers see the same Sun. That shared information is what backbone and daisy-chain methods exploit, and per-observer SBI throws it away unless the model is made hierarchical or joint (see A2 and B).
- **Prior dependence.** The prior over τ (which reference epoch) sets the implicit activity prior. The reference covers about 13 cycles of "modern" activity. NPE then extrapolates poorly for weak-cycle or Maunder observers (see A3).
- **Masks.** Observing masks are informative: an observer may report only spotted days. If the network sees the mask as a channel, it can learn the missing-zero rate, but only if the simulator's mask model includes that mechanism.
- **Overconfidence.** NPE trained on finite simulations can be overconfident (see A4). Use ensembles and coverage checks.

**(e) JAX tooling.**
- sbijax has NPE, NLE, CNRE, flow matching, SMC-ABC and automatic summary learning in pure JAX.
- BayesFlow 2 runs on the Keras 3 JAX backend, has flexible summary networks, and is recommended as the fastest backend.
- sbi (PyTorch) has the most methods and diagnostics. Simulations can be generated in JAX and passed as NumPy arrays.

See A5 for details.

### Gaps
- I found no published application of NPE, NLE or learned-summary ABC to sunspot-observer calibration. This appears to be novel for the GN problem.
- I did not find benchmark numbers for training cost of NPE on sequences of length 10^4 with irregular masks. Hours on one GPU is an estimate.

---

## A2. Amortized inference across many units: hierarchical SBI with observers as units sharing a population prior

### Takeaway
Hierarchical SBI is mature for "many exchangeable units with local parameters plus a low-dimensional global parameter" (HNPE, Heinrich et al. 2024, Habermann et al. 2025 in BayesFlow). The GN hierarchy is: population hyperparameters η (era-dependent threshold distribution, drift scale, split-factor distribution), then observer θ_o, then data. That part fits these methods well.

What does not fit is that observers are also coupled through a shared 150k-day latent activity series, which is far from low-dimensional. The pragmatic design is two-stage:
1. amortized local posteriors q(θ_o | x_o) under a broad working prior;
2. importance re-weighting to learn η, or embedding the local posteriors as priors or proposals in the joint likelihood model of Part B.

### Cited Findings
- **HNPE.** Leverages global parameters shared across a set of auxiliary observations so that local parameters become identifiable. This is neural posterior estimation for hierarchical structures. — [Rodrigues, Moreau, Louppe & Gramfort 2021, NeurIPS †](https://arxiv.org/abs/2102.06477)
- **Hierarchical neural SBI over event ensembles.** Builds neural likelihood-ratio or posterior estimators for datasets made of many events with both per-event and dataset-wide parameters. It reports that modelling the hierarchy explicitly gives tighter constraints, with particle physics and cosmology case studies (TMLR 2024). — [Heinrich, Mishra-Sharma, Pollard & Windischhofer 2024](https://arxiv.org/abs/2306.12584); [TMLR listing](https://mlanthology.org/tmlr/2024/heinrich2024tmlr-hierarchical)
- **Amortized Bayesian multilevel models.**
  - Neural networks whose structure follows the probabilistic factorization of multilevel models give efficient training and near-instant posterior inference on unseen datasets.
  - They are benchmarked against Stan where possible, with an open-source implementation.
  - Published in Bayesian Analysis (2025), DOI 10.1214/25-BA1570.

  — [Habermann, Schmitt, Kühmichel, Bulling, Radev & Bürkner 2025](https://arxiv.org/abs/2408.13230); [Bayesian Analysis PDF](https://paulbuerkner.com/publications/pdf/2025__Habermann_et_al__Bayesian_Analysis.pdf)
- **BayesFlow 2.0 hierarchical support.** The BayesFlow 2.0 paper lists hierarchical modelling as a new capability, together with posterior, likelihood and ratio estimation, hyperparameter optimization and design optimization. — [BayesFlow 2.0 preprint, arXiv:2602.07098 (Feb 2026)](https://arxiv.org/abs/2602.07098)
- **Composing many observations.** Compositional score modelling learns scores of the diffused posteriors induced by single observations, then combines them at inference time to sample the posterior given all observations. It targets the case where NPE struggles to condition on many observations. — [Geffner, Papamakarios & Mnih 2023, ICML](https://proceedings.mlr.press/v202/geffner23a.html)
- **Two-stage hierarchical Bayes.** Unit-level posteriors are fitted under a working prior and then combined into the full hierarchical posterior by re-weighting or MCMC on the stored unit posteriors. — [Lunn, Barrett, Sweeting & Thompson 2013, JRSS-C †](https://doi.org/10.1111/rssc.12007)

### Inferences
**(c) Mapping to GN.**
- **Two-stage recipe (cheap and principled).**
  1. Train NPE once under a broad working prior π₀(θ).
  2. Draw S samples from each observer's q(θ_o | x_o). For example, S = 1,000 samples x 600 observers is 6x10^5 draws.
  3. The marginal likelihood of η is approximately the product over o of the mean over s of p(θ_o^s | η) / π₀(θ_o^s).
  4. Sample η with NUTS. This costs microseconds per evaluation.
  5. Re-weight each observer's samples by p(θ|η)/π₀(θ) to get shrunken local posteriors.

  This gives partial pooling for poorly constrained observers (short careers, sparse records), which ADF currently cannot do.
- **Era-dependent population prior.** Let η depend on epoch, for example log A_thr ~ N(μ(t_o), σ²), where μ(t) is a smooth function of the career midpoint t_o that captures telescope and technique improvements. This is a natural hyperprior, and it is cheap in the two-stage scheme.
- **The shared latent series.**
  - Hierarchical SBI (HNPE, Heinrich) treats "global" parameters as a fixed-size vector. In GN the coupling is the latent daily series S_{1:T}, and each observer sees only its window.
  - In the reference era, condition on the reference (S known), and the units become conditionally independent. That is exactly where NPE works.
  - Before 1874, observers are coupled only through unknown S, so the coupling should be handled by the Part-B likelihood model. NLE emission models or local NPE posteriors enter that model as building blocks.
- **Geffner et al. (2023)** is relevant if one tries a fully SBI treatment of the activity level during overlapping careers. Scores from each overlapping observer can be composed for the shared activity parameter, for example the mean activity in a 1-yr bin. This scales poorly to a 150k-dimensional latent series and is not recommended as the main engine.

**(d) Pitfalls.**
- Importance re-weighting degenerates if the population prior is much narrower than π₀, or if local posteriors are near the edge of π₀. Monitor the effective sample size per observer and widen π₀.
- Exchangeability across observers is violated by:
  - shared instruments and schools, for example Wolf and his assistants;
  - copied or compiled records, for example later compilers.

  Group-level random effects (observatory or compiler) are needed.
- Learned amortized posteriors under a working prior can be miscalibrated in the tails that matter after re-weighting. Check calibration (A4) after re-weighting, not only before.

**(e) JAX tooling.**
- BayesFlow 2 on the JAX backend supports the hierarchical or multilevel architectures of Habermann et al.
- Two-stage pooling is about 50 lines of NumPyro: `numpyro.factor` with logsumexp over stored samples.

### Gaps
- I did not verify the exact BayesFlow 2 API names for the multilevel networks. Check the repo examples.
- I found no hierarchical-SBI method designed for units coupled through a long shared latent time series. That is a methods-paper opportunity.

---

## A3. Model misspecification and out-of-distribution robustness (the Maunder Minimum problem)

### Takeaway
Naive NPE, NLE and NRE can fail badly under misspecification, and no single mitigation works in all cases. Robust options are:
- explicit error models on summaries (RNPE, RSNL);
- learning summaries that penalize simulation-data mismatch (Huang et al. 2023);
- generalized Bayes;
- data-driven calibration with a small labelled real set (RoPE);
- detection via summary-space MMD (Schmitt et al.).

For GN, the most effective defences are structural:
1. make the simulator generate low-activity regimes, for example by thinning whole reference groups;
2. keep the latent activity likelihood-based (Part B), so that low activity is a parametric extrapolation of the birth rate rather than an out-of-distribution input to a neural net;
3. flag observers whose learned summaries fall outside the training typical set.

### Cited Findings
- **First comprehensive study** of neural SBI under misspecification. Misspecification "can have a profoundly deleterious effect on performance". Some mitigations were explored, but none prevents failure in all cases. The authors are Cannon, Ward & Schmon (arXiv 2209.01845). The prompt's "Cannon, Ward & Sisson" appears to be a mis-attribution. — [Cannon, Ward & Schmon 2022](https://arxiv.org/abs/2209.01845)
- **Robust NPE (RNPE)** explicitly models the discrepancy between simulated and observed summaries. That gives both model criticism (which summaries are misspecified) and robust inference. Naive NPE gave "misleading and erratic posteriors" on misspecified tasks (NeurIPS 2022). — [Ward, Cannon, Beaumont, Fasiolo & Schmon 2022](https://arxiv.org/abs/2210.06564); [NeurIPS page](https://papers.nips.cc/paper_files/paper/2022/hash/db0eac6747e3631eb91095cd76065611-Abstract-Conference.html)
- **Misspecification-robust sequential neural likelihood (RSNL)** adds adjustment parameters on summaries to the neural likelihood approach. It cites RNPE as the NPE counterpart. — [Kelly, Nott, Frazier, Warne & Drovandi 2023/2024](https://arxiv.org/abs/2301.13368)
- **ABC under misspecification.** ABC posteriors concentrate on pseudo-true values defined by the summary discrepancy, and there are diagnostics for detecting misspecification. — [Frazier, Robert & Rousseau 2020, JRSS-B †](https://arxiv.org/abs/1708.01974)
- **Robust synthetic likelihood** uses mean or variance adjustment parameters with a shrinkage prior. — [Frazier & Drovandi 2021, JCGS †](https://arxiv.org/abs/1904.04551)
- **Detection in amortized inference (Schmitt et al.).**
  - Training is augmented so that the learned summary space has a known inlier distribution, for example a unit Gaussian.
  - Misspecification is detected by a high MMD between observed and simulated summaries.
  - Posterior errors grow with the distance between the true data-generating distribution and the simulation typical set in summary space.

  — [Schmitt, Bürkner, Köthe & Radev (arXiv 2112.08866)](https://arxiv.org/abs/2112.08866); [extended version 2024](https://arxiv.org/abs/2406.03154)
- **Learning robust statistics.** Summaries are trained with a penalty on the discrepancy between simulated and observed summary distributions, which makes NPE or ABC robust (NeurIPS 2023). — [Huang, Bharti, Souza, Acerbi & Kaski 2023 †](https://arxiv.org/abs/2305.15871)
- **Generalized Bayesian inference for simulators** via amortized cost estimation (ACE). It replaces the likelihood with a loss-based Gibbs posterior that is less brittle under misspecification. — [Gao, Deistler & Macke 2023 †](https://arxiv.org/abs/2305.15208)
- **RoPE (ICML 2025).** Models the gap between simulated and real observations as an optimal-transport problem over learned representations, without assuming the form of the misspecification. It needs a correctly specified prior and a small real-world calibration set of paired parameters and observations. It gave calibrated, informative credible intervals on synthetic and real benchmarks. — [Wehenkel et al. 2025, PMLR v267](https://proceedings.mlr.press/v267/wehenkel25a.html); [arXiv 2405.08719](https://arxiv.org/abs/2405.08719)
- **Review (ICLR 2026).** A blog-track piece, "Model Misspecification in Simulation-Based Inference: Recent Advances and Open Challenges", reviews formal definitions, detection and mitigation methods, and their assumptions. — [ICLR 2026 listing](https://iclr.cc/virtual/2026/poster/10012116)
- **Overconfidence.** Even when well specified, all benchmarked SBI algorithms ((S)NPE, (S)NRE, SNL, ABC variants) can produce overconfident posteriors. Ensembles of posterior surrogates mitigate this. — [Hermans et al. 2022, TMLR](https://arxiv.org/abs/2110.06581)

### Inferences
**(c) Mapping to GN. Sources of out-of-distribution behaviour:**
1. Activity regimes absent from 1874-2025, such as the Maunder Minimum (1645-1715) and possibly the Dalton Minimum.
2. Observer behaviours absent from the simulator: recording conventions, reporting only spotted days, rounding, copying.
3. Area-distribution shifts. The distribution of group areas in a weak regime may differ from the reference, so thresholding the reference may not reproduce weak-regime detection statistics. This is a physical assumption that needs checking against the literature, which other researchers are covering.

**Concrete defences, in order of value:**
- **(i) Activity-thinning augmentation.** Randomly delete whole groups (by group ID across their full lifetime) from reference windows with retention probability p ~ U(0.01, 1). If emergence is close to Poisson, independent group thinning gives a lower-activity realization with the same lifetime and area statistics. The simulator then covers Maunder-like activity levels, so in-distribution coverage extends downward. It does not address area-distribution shift. That needs a parametric area model with an activity-dependent slope, and a sensitivity analysis.
- **(ii) Likelihood-based latent activity.** If the emission model is learned from the reference and the latent activity is modelled parametrically (Part B: birth rate λ_t → 0 allowed), low activity is an extrapolation in one scalar parameter. It is not an unseen input to a neural network. This is a much safer extrapolation than NPE on raw records.
- **(iii) Detection.** Train with the Schmitt et al. summary-space regularizer, or compute MMD or density-ratio scores of each observer's summary against the simulation typical set. Flag observers above the 99th percentile. Report flagged observers separately, and do not trust their amortized posteriors.
- **(iv) RNPE/RSNL-style slack on summaries.** For example, summary-specific Laplace or spike-and-slab error terms. These tell you which summary is misspecified, for example the zero-day fraction for a spotted-days-only observer.
- **(v) RoPE.** RoPE needs a small set of real observers with known θ. A candidate set is observers overlapping RGO/Debrecen for whom θ (threshold, split factor) can be estimated directly by daily matching against the area-resolved catalogue. That is an empirical calibration set unique to this problem.

**(d) Pitfalls.**
- Augmentation only works if the augmentation mechanism is right. Thinning assumes that groups in weak regimes are statistically like modern groups.
- MMD detection flags out-of-distribution inputs but does not fix them.
- RoPE's calibration observers are themselves in the reference era, so they do not test Maunder extrapolation.
- Generalized Bayes posteriors need a learning-rate or temperature choice, and their calibration is not automatic.

**(e) JAX tooling.**
- MMD is trivial in JAX.
- RNPE's reference code was written in JAX/NumPyro, according to my recollection. This is not verified this session, so it is listed as a gap.
- sbijax and BayesFlow cover the base estimators, and robust variants must be custom-built.

### Gaps
- I did not confirm whether reference implementations of RNPE, RSNL or RoPE are in JAX.
- I found no literature evidence in this search scope on whether sunspot group area or lifetime distributions differ between grand-minimum and normal regimes. This is critical for the thinning assumption and is deferred to the solar-physics researchers.

---

## A4. Calibration diagnostics: SBC, TARP, expected coverage

### Takeaway
Use three layers.
1. **SBC rank histograms** (per-parameter, plus data-dependent test quantities) for computational correctness of each estimator.
2. **TARP** for joint-posterior coverage of the multivariate θ_o.
3. **Expected-coverage curves**, with ensembles or balanced training to bias toward conservativeness.

None of these detects misspecification. For GN, add an end-to-end mock historical network: simulate 1610-2025 with known truth, run the full pipeline, and check GN coverage at monthly and yearly resolution. Also add real-data posterior-predictive checks on reference-overlap observers.

### Cited Findings
- **SBC.** If data are simulated from the prior predictive and the algorithm is correct, the rank of the true parameter among posterior draws is uniform. Non-uniform shapes diagnose bias or over/under-dispersion. — [Talts, Betancourt, Simpson, Vehtari & Gelman 2018 †](https://arxiv.org/abs/1804.06788)
- **Choice of test quantities.** The test quantity determines SBC sensitivity. Data-dependent quantities, such as the log-likelihood, catch failures that parameter-only ranks miss. — [Modrák et al. 2023, Bayesian Analysis †](https://arxiv.org/abs/2211.02383)
- **TARP.** Tests of Accuracy with Random Points estimate the expected coverage probability of a generative posterior estimator using only posterior samples. It tests the joint posterior, and the test is necessary and sufficient for posterior accuracy. — [Lemos, Coogan, Hezaveh & Perreault-Levasseur 2023, ICML †](https://arxiv.org/abs/2302.03026)
- **TARP in astrophysics practice.**
  - It is used as a joint-posterior coverage test. Per-parameter SBC can be more variance-prone and flag deviations that do not matter in practice. — [VROOM-SBI, arXiv 2605.27538](https://arxiv.org/pdf/2605.27538)
  - TARP catches estimators that simply return the prior, which some highest-density-region coverage tests miss. — [arXiv 2512.16869](https://arxiv.org/pdf/2512.16869)
  - It is commonly run via the sbi package's implementation. — [arXiv 2510.04735](https://arxiv.org/pdf/2510.04735); [arXiv 2508.05744](https://arxiv.org/pdf/2508.05744)
- **"A trust crisis in SBI?"** Expected-coverage diagnostics show that all benchmarked algorithms can be overconfident. Ensembling posterior surrogates gives more reliable, more conservative approximations (TMLR 2022). — [Hermans, Delaunoy, Rozet, Wehenkel, Begy & Louppe 2022](https://arxiv.org/abs/2110.06581)
- **Balanced NRE** adds a balancing regularizer that biases ratio estimators toward conservative (not overconfident) posteriors. — [Delaunoy et al. 2022, NeurIPS †](https://arxiv.org/abs/2208.13624)
- **sbi diagnostics.** The sbi toolkit (v0.23+) bundles a broad set of diagnostics alongside its inference methods (JOSS 2025). — [Boelts et al. 2025, "sbi reloaded"](https://joss.theoj.org/papers/10.21105/joss.07754)

### Inferences
**(c) GN calibration protocol.**
1. **Per-estimator calibration (in-distribution).** Use about 1,000 synthetic observers drawn from the simulator: random reference window plus thinning augmentation plus real observing masks resampled from the 600 historical masks.
   - Run SBC on each component of θ_o, with test quantities A_thr, split factor, drift slope, zero-loss rate, and the log-likelihood of the synthetic record.
   - Run TARP on the joint θ_o.
   - Cost: 1,000 x 1,000 posterior draws from an amortized NPE takes seconds.
2. **Stratified calibration.** Stratify by career length, observing density and activity level (thinning p). Poorly constrained observers are where overconfidence hurts the GN.
3. **Likelihood-based Part-B model.** Run SBC on the hyperparameters and on derived GN quantities, such as yearly means of N_t, using small mock networks: about 50 observers and 50 years, so that each run takes minutes. The same code then scales up to the full problem.
4. **End-to-end mock (most important for the methods paper).**
   - Simulate a synthetic 1610-2025 Sun. Use the reference era as-is, and use a parametric birth-death process with a Maunder-like low-λ interval before that.
   - Simulate the actual 600-observer network (real masks, drawn θ_o) and run the full pipeline.
   - Report coverage of 50%, 68% and 95% intervals of yearly GN versus truth, and stratify by era.

   Even 10-20 such mocks are informative, and they are the only test of the whole chain, including cross-observer coupling.
5. **Real-data checks.**
   - For observers overlapping RGO/Debrecen, compare posterior-predictive daily counts and active-day fractions against the record.
   - Do leave-one-observer-out prediction of overlapping observers' records.

**(d) Pitfalls.**
- SBC and TARP are evaluated under the simulator's own prior predictive, so they validate computation, not the model.
- TARP's power depends on the reference-point distribution.
- Coverage evaluated against the working prior π₀ is not coverage after the hierarchical re-weighting of A2. Re-run after pooling.
- A 2026 preprint titled "Coverage is not enough" argues for stronger frequentist tests (see Gaps).

**(e) JAX tooling.**
- TARP, SBC and expected coverage are implemented in sbi (PyTorch).
- They are simple to re-implement in JAX: rank statistics and coverage curves are about 30 lines.
- ArviZ provides rank plots for MCMC output.

### Gaps
- I did not read "Coverage is not enough: Frequentist tests of simulation-based inference for primordial non-Gaussianity" (arXiv 2605.00980). Its specific recommendations are unknown. — [arXiv 2605.00980](https://arxiv.org/pdf/2605.00980)
- I did not verify whether sbijax or BayesFlow 2 ship TARP or expected-coverage utilities.

---

## A5. SBI tooling and JAX compatibility (sbi, BayesFlow, sbijax, LAMPE)

### Takeaway
For a JAX/NumPyro stack, BayesFlow 2 on the JAX backend is the most capable amortized-inference option. It has summary networks, hierarchical support and active development, but its API is still occasionally breaking. sbijax is the most "native" pure-JAX option, with NPE, NLE, CNRE, flow matching, SMC-ABC and summary learning, but it is a small project built on Haiku, which is in maintenance mode. sbi (PyTorch) is the most complete and has the best diagnostics. It can still be used by generating simulations in JAX and passing arrays.

### Cited Findings
- **sbijax.**
  - Implements SBI algorithms in JAX using Haiku, Distrax and BlackJAX.
  - The current README lists SMC-ABC, NLE and NPE variants, contrastive NRE, and flow-matching and consistency-model posterior estimation.
  - The API is inspired by the PyTorch sbi package, which the authors describe as more feature-complete and better documented.
  - It runs training in parallel on CPU or GPU and includes conventional ABC, model diagnostics and automatic summary-statistic estimation.

  — [sbijax GitHub](https://github.com/dirmeier/sbijax); [Dirmeier, Ulzega, Mira & Albert 2024, arXiv 2409.19435](https://arxiv.org/abs/2409.19435)
- **Haiku status.** sbijax's neural-network layer, Haiku, has been declared in maintenance mode by DeepMind, which recommends Flax for new projects. — [dm-haiku README †](https://github.com/google-deepmind/dm-haiku)
- **BayesFlow 2.**
  - A complete rewrite on Keras 3, backend-agnostic across JAX, PyTorch and TensorFlow, and not API-compatible with 1.x.
  - The repository recommends JAX as currently the fastest backend.
  - The maintainers warn that breaking changes still occur, especially around saving and loading, and aim to stabilize from 2.1.
  - The 2026 preprint describes posterior, likelihood and ratio estimation, generative networks, hyperparameter optimization, design optimization and hierarchical modelling.

  — [BayesFlow GitHub](https://github.com/bayesflow-org/bayesflow); [BayesFlow 2.0 preprint](https://arxiv.org/abs/2602.07098)
- **sbi.** A PyTorch toolkit with a broad set of inference methods, neural network architectures, samplers and diagnostics. "sbi reloaded" was published in JOSS 10(108):7754 (April 2025). — [JOSS](https://joss.theoj.org/papers/10.21105/joss.07754); [GitHub sbi-dev/sbi †](https://github.com/sbi-dev/sbi)
- **LAMPE** ("likelihood-free AMortized Posterior Estimation") is a PyTorch library from the Liège group (NPE, NRE, flow-matching and diagnostics). — [LAMPE GitHub †](https://github.com/probabilists/lampe)

### Inferences
**JAX compatibility.**

| Library | Backend | JAX-native | Strengths for GN | Concerns |
|---|---|---|---|---|
| BayesFlow 2 | Keras 3 (JAX/PyTorch/TF) | Yes via JAX backend | Summary nets for time series and sets, hierarchical/multilevel, amortized | Breaking changes before 2.1 |
| sbijax | JAX (Haiku/Distrax/BlackJAX) | Yes | Pure JAX, SMC-ABC + NPE/NLE/CNRE/FMPE in one place | Small team, Haiku maintenance mode |
| sbi | PyTorch | No | Most methods and diagnostics (TARP/SBC/coverage) | Cross-framework data passing; cannot jit the simulator together with the network |
| LAMPE | PyTorch | No | Clean NPE/NRE building blocks | Smaller community; maintenance status unverified |

**Recommendation.**
- Write the forward simulator in pure JAX (vmap over synthetic observers), so it can be reused by any library and by the Part-B posterior-predictive checks.
- Use BayesFlow 2 with the JAX backend for NPE and NLE emission models.
- Use sbi offline for diagnostics if needed. TARP is easy to port.

### Gaps
- I did not verify LAMPE's current maintenance status, or whether it has been superseded.
- I did not verify sbijax's current release version or whether it has migrated off Haiku.

---

## B1. Marginalizing discrete latent counts: enumeration, forward algorithm, parallel-in-time HMM filtering, dynamax, and costs for about 151,900 steps

### Takeaway
Model the latent visible-group count N_t ∈ {0, …, K−1}, with K ≈ 41, as an immigration-death (Poisson-INAR(1)) chain:
- survival by binomial thinning with daily survival φ;
- Poisson(λ_t) births;
- observers as conditionally independent thinned emissions.

The exact forward algorithm then costs about 5x10^8 flops per likelihood evaluation, latency-bound at about 0.2-3 s. It needs only a T x K filter array (25 MB in float32). The fully parallel associative-scan filter (Hassan, Särkkä & García-Fernández 2021, implemented in dynamax) reduces the depth to about 18 levels but needs a T x K x K array (about 1 GB in float32). A chunked hybrid gets most of the speed at negligible memory.

Avoid NumPyro's automatic enumeration for the 150k-step chain. Write the forward pass yourself (or call dynamax) and add it with `numpyro.factor`.

### Cited Findings
- **NumPyro enumeration.**
  - Discrete latent states are marginalized with `config_enumerate` and `scan`. `scan` cuts compile time compared with Python loops.
  - The model must have a narrow-treewidth dependency graph and must broadcast over enumerated values.
  - `scan` stacks each step's parameters and values along a time dimension, so the joint density can be computed in parallel.

  — [NumPyro HMM enumeration example](https://num.pyro.ai/en/0.19.0/examples/hmm_enum.html)
- **Known NumPyro enumeration-in-scan pitfalls:**
  - indexing errors for factorial HMMs — [forum](https://forum.pyro.ai/t/enumerating-factorial-hmm-with-scan-errors-in-mcmc-and-svi/5523);
  - shape and broadcast errors when continuous latents depend on enumerated discretes — [forum](https://forum.pyro.ai/t/an-issue-with-the-shape-of-a-continous-latent-that-depends-on-a-enumerated-discrete-inside-scan/6356);
  - `DiscreteHMCGibbs` failing to detect discrete latents inside scan — [forum](https://forum.pyro.ai/t/discretehmcgibbs-enumeration-cannot-detect-any-discrete-latent/5314).
- **How Pyro and NumPyro contract the chain.** Enumeration contracts Markov chains by tensor variable elimination over plated factor graphs. For a Markov time dimension this is a log-depth pairwise contraction of K x K factors. — [Obermeyer et al. 2019, ICML †](https://arxiv.org/abs/1902.03210)
- **Parallel HMM inference.** HMM filtering, smoothing and MAP (Viterbi) are written as associative operators, so they become all-prefix-sums problems solvable by parallel scan in O(log T) span. — [Hassan, Särkkä & García-Fernández 2021, IEEE TSP](https://arxiv.org/abs/2102.05743)
- **Parallel Gaussian smoothers** are the Kalman/RTS analogue for linear-Gaussian models. — [Särkkä & García-Fernández 2021, IEEE TAC †](https://arxiv.org/abs/1905.13002)
- **dynamax parallel HMM module.** `dynamax/hidden_markov_model/parallel_inference.py` defines:
  - `hmm_filter(initial_probs, transition_matrix, log_likelihoods)`, using `jax.lax.associative_scan`, which returns the marginal log-likelihood and the filtered and predicted probabilities;
  - `hmm_smoother`, which computes smoothed marginals by differentiating the log-normalizer (`value_and_grad`) rather than running an explicit backward pass;
  - `hmm_posterior_sample(key, initial_distribution, transition_matrix, log_likelihoods)`, a parallel forward-filter backward-sample.

  Filter scan elements have shape T x K x K (one K x K matrix per step) plus T x K log-normalizers. Sampling elements are T x K integer vectors. — [dynamax source](https://github.com/probml/dynamax/blob/main/dynamax/hidden_markov_model/parallel_inference.py)
- **Parallel sampler crossover.** The parallel HMM posterior sampler (PR #342) "starts performing better than serial sampling when the sequence length exceeds a few thousand". — [dynamax PR #342](https://github.com/probml/dynamax/pull/342)
- **dynamax** is a JAX library for probabilistic state-space models (HMMs, LGSSMs, nonlinear SSMs). — [JOSS paper](https://www.theoj.org/joss-papers/joss.07069/10.21105.joss.07069.pdf)
- **Parallel filtering in dynamax.** A 2025 unifying-framework paper states that dynamax implements the Särkkä & García-Fernández parallel filtering and smoothing. — [arXiv 2509.21716](https://arxiv.org/pdf/2509.21716)
- **INAR(1).** N_t = φ∘N_{t−1} + ε_t, with binomial thinning "∘" and Poisson innovations. Its stationary marginal is Poisson(λ/(1−φ)). It is the discrete-time count analogue of the immigration-death (M/M/∞) process. — [Al-Osh & Alzaid 1987, J. Time Series Analysis †](https://doi.org/10.1111/j.1467-9892.1987.tb00438.x)

### Inferences
**(a)+(c) Model on the GN problem.**
- **Latent chain.** N_t = number of groups on the visible disk that a reference observer (RGO-equivalent, q = 1) would count.
  - Transition: P(N_t = j | N_{t−1} = i) = Σ_s Bin(s; i, φ) · Pois(j − s; λ_t).
  - φ ≈ exp(−1/τ_eff), where τ_eff combines intrinsic lifetime with disk-passage loss of about 13.5 days for a random disk position.
  - λ_t is the daily birth rate, including groups rotating onto the disk, with log λ_t given a smooth cycle model (see B2).
  - Truncation: put the tail mass at K−1. K = 41 easily covers daily group counts. Check against the maximum daily reference count, and use K ≈ 60 for modern split-heavy observers.
- **Factorized transition, the key cost trick.** A_t = S · B_t, where:
  - S is the fixed K x K binomial-survival matrix (lower-triangular, depends only on φ);
  - B_t is a K x K upper-triangular Toeplitz Poisson-birth matrix that depends only on λ_t.

  So α_t = (α_{t−1} S) B_t takes two matrix-vector products (about 2K² ≈ 3.4x10³ flops), and the T x K x K transition tensor never has to be stored. With heavier-tailed lifetimes, use a 2-component mixture (short- and long-lived groups). The joint state is then K₁ x K₂, and the transition stays Kronecker-factorized.
- **Emission per observer-day, with ADF's ingredients made explicit:**
  - Detection d ~ Bin(N_t, q_o(t)), where q_o(t) = P(area > A_o(t)) under the reference area distribution. q_o(t) can be activity-dependent, and A_o(t) carries drift.
  - Splitting or merging: y = d + Pois(s_o·d), or a beta-binomial or NB overdispersion layer.
  - Missing zeros: an observed y = 0 report has weight r_o (the probability a zero is reported). A day in the active window with no report has likelihood (1 − π_o) + π_o(1 − r_o)P(y = 0 | N_t), where π_o is the observing probability.
  - This turns the "under-reported zeros" problem into an explicit missing-not-at-random likelihood term instead of an ADF heuristic.
  - An ADF-consistent robustness variant: emit only the binary active/inactive indicator, P(y > 0 | N) = 1 − (1 − q_o)^N. This mirrors ADF's insensitivity to splitting conventions.
- **Cheap emission evaluation.**
  - Per likelihood evaluation, build for each observer (or observer-epoch, for drift) a K x Y_max table of log P(y | N = k, θ_o), with Y_max ≈ 60. The cost is about K·Y_max·K ≈ 10^5 flops per table; 600 observers x 20 epochs is about 1.2x10^9 flops.
  - Then gather one row per record and `jax.ops.segment_sum` into a T x K log-emission array (151,941 x 41 ≈ 6.2x10^6 entries, 25 MB in float32).
  - Use long-format (observer, day, count) records. Never build a dense T x O array: that would be 9.1x10^7 entries per field, and almost all would be empty.

**Cost and memory estimates for one likelihood evaluation** (T ≈ 1.52x10^5, K = 41). These are my estimates and need benchmarking.

| Algorithm | Work | Sequential depth | Peak memory (forward) | Est. wall time |
|---|---|---|---|---|
| Sequential `lax.scan` forward, factorized S·B_t | ≈ 2K²T ≈ 5x10^8 flops | 1.52x10^5 steps | T x K ≈ 25 MB (f32) | ~0.2-1 s CPU; ~1-3 s single-chain GPU (per-step launch overhead dominates) |
| Same, batched over 256-1,024 parameter particles or chains (vmap) | x batch | same | x batch (≈ 6-25 GB at 1,024 in f32; use remat) | ≈ same wall time as one chain on GPU, giving roughly linear throughput gain |
| Full associative scan (dynamax `parallel_inference.hmm_filter`) | ≈ 2K³T ≈ 2x10^10 flops | ⌈log₂T⌉ = 18 levels | T x K x K ≈ 2.55x10^8 entries ≈ 1.0 GB f32 (2 GB f64), x 2-3 for intermediates; gradients more | tens of ms on a modern GPU, memory-bandwidth bound |
| Chunked hybrid: C = 256 chunks of ~600 days; within-chunk sequential K x K transfer matrices (vmapped over chunks), then associative scan over chunks | ≈ K³T ≈ 1x10^10 | ~600 + 8 | C x K x K ≈ 4.3x10^5 entries (~2 MB) + T x K | ~0.05-0.3 s GPU |
| 2 area classes (K = 30 each → 900 states), Kronecker transitions | ≈ 2·900·30·T ≈ 8x10^9 | 1.52x10^5 | T x 900 ≈ 0.55 GB f32 | several s CPU |
| 3 area classes (K = 20 → 8,000 states) | ≈ 3·8000·20·T ≈ 7x10^10 | 1.52x10^5 | T x 8000 ≈ 4.9 GB f32 | tens of s; parallel scan infeasible (8000² x T) |

- **Gradients.** Reverse-mode through the sequential scan stores per-step residuals. That is fine for T x K. If λ_t-dependent B_t is built inside the scan body, wrap the body in `jax.checkpoint` so that K x K per-step intermediates are not stored (about 1 GB otherwise). [jax.checkpoint docs †](https://jax.readthedocs.io/en/latest/gradient-checkpointing.html)
- **Sampling N_{1:T}.**
  - Forward-filter backward-sample (serial: dynamax `hmm_posterior_sample`; parallel: `parallel_inference.hmm_posterior_sample`) gives exact joint draws of the latent daily series.
  - These draws are the GN posterior ensemble: aggregate each draw to monthly or yearly means.
  - The parallel sampler wins beyond a few thousand steps (PR #342).
- **Identifiability anchor.** Fix q = 1 and s = 0 (or the known calibration) for the reference observer, as ADF does. Otherwise the scale of N and all q_o trade off.
- **Numerical precision.** The total log-likelihood is a sum of about 10^6-10^7 terms. In float32, rounding in that sum (about 0.1-1 nat) can disturb MH acceptance and NUTS energy errors. Enable `jax_enable_x64`, at least for the accumulation, or use a two-level (per-chunk then global) sum.

**(d) Pitfalls.**
- **Binomial thinning ignores area persistence.** Per-day independent thinning ignores that the same large group persists across days, so an observer's detections are positively autocorrelated given N_t. Without area classes or overdispersion, posterior uncertainty on q_o will be too narrow. The 2-class factorial chain is the cheapest fix that keeps exactness.
- **Time-varying transitions in dynamax.** I could not confirm that dynamax's parallel `hmm_filter` accepts a time-varying (T−1) x K x K transition array. If it does not, writing the associative operator directly with `jax.lax.associative_scan` is about 30 lines.
- **Automatic enumeration.** NumPyro's automatic enumeration over a 151k-step `scan` creates K x K x T factors and is fragile (forum threads above). A hand-written forward pass plus `numpyro.factor` is more controllable.

**(e) JAX tooling.**
- `jax.lax.scan` / `jax.lax.associative_scan` — [associative_scan docs †](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.associative_scan.html)
- dynamax: `hidden_markov_model.inference` (serial) and `hidden_markov_model.parallel_inference` (parallel filter, smoother, sampler)
- `numpyro.factor` for the marginal log-likelihood
- `jax.ops.segment_sum` for sparse emissions

### Gaps
- No published benchmark gives per-step overhead of `jax.lax.scan` on CPU versus GPU for K ≈ 40 at T ≈ 1.5x10^5. The wall times above are estimates and should be measured early (a one-day task).
- I did not verify whether NumPyro's enumerated scan uses log-depth contraction (with its T x K x K memory) or a sequential contraction in current releases.
- Whether a Poisson-INAR immigration-death chain fits observed daily RGO group-count autocorrelations needs an empirical check, for example matching the RGO autocorrelation function and the distribution of the number of days a group stays on disk.

---

## B2. Continuous relaxations: latent log-intensity random walk or GP with Poisson/NB observations, Laplace/INLA, and state-space GPs

### Takeaway
Use a continuous latent for the slow component only: the 11-yr-modulated log birth rate log λ_t, as a state-space GP or random walk on a coarse grid (27-day Carrington rotations ≈ 5,600 knots for 1610-2025). Keep the fast, discrete group-count dynamics in the exact HMM (B1). A fully continuous daily model (log-Gaussian Cox-like, Poisson or NB observations) is O(T) with Kalman-structured Laplace or INLA-type methods. However, it misrepresents the many-zeros, low-count regime (the Maunder Minimum) and the day-to-day persistence of groups. It is best as a fast baseline and initializer.

### Cited Findings
- **State-space GPs.** Stationary GPs with rational spectra (Matérn family) can be written as linear SDEs, so GP regression becomes Kalman filtering and RTS smoothing at O(n) cost. — [Hartikainen & Särkkä 2010, IEEE MLSP †](https://doi.org/10.1109/MLSP.2010.5589113)
- **Periodic covariances in state space.** Periodic and quasi-periodic covariance functions have explicit state-space representations. These suit a quasi-periodic ~11-yr cycle prior. — [Solin & Särkkä 2014, AISTATS †](http://proceedings.mlr.press/v33/solin14.html)
- **Parallel Kalman filtering and smoothing.** O(log T) span via associative scan for linear-Gaussian SSMs. — [Särkkä & García-Fernández 2021 †](https://arxiv.org/abs/1905.13002) Implemented in dynamax. — [arXiv 2509.21716](https://arxiv.org/pdf/2509.21716)
- **INLA.** Nested Laplace approximations for latent Gaussian models with non-Gaussian likelihoods. It gives fast, deterministic posterior marginals for hyperparameters and latent fields. — [Rue, Martino & Chopin 2009, JRSS-B †](https://doi.org/10.1111/j.1467-9868.2008.00700.x)
- **HMC with an embedded Laplace approximation.** Marginalizing the latent Gaussian field with a Laplace approximation inside HMC, differentiated with an adjoint method, makes HMC over hyperparameters scale. — [Margossian et al. 2020, NeurIPS †](https://arxiv.org/abs/2004.12550)
- **Bayes-Newton.** Unifies Laplace, EP, variational and posterior-linearization methods for (state-space) GPs with PSD guarantees, implemented in JAX (BayesNewton library). — [Wilkinson, Särkkä & Solin 2023, JMLR †](https://jmlr.org/papers/v24/21-1298.html)

### Inferences
**(c) Mapping to GN, two recommended uses.**
- **(i) Hybrid, recommended as the main model.**
  - log λ_t = μ + f_cycle(t) + f_slow(t), placed on a 27-day grid with linear interpolation to days.
  - f_cycle is a quasi-periodic state-space GP (≈ 11-yr period with period jitter) or a Matérn-3/2 GP with 3-6 yr length scale. f_slow captures secular and grand-minimum modulation.
  - The latent dimension is about 5,600 x (state dim 2-4). NUTS handles this well with non-centered innovations, i.e. sample white-noise innovations and run the Kalman-form recursion forward with `lax.scan`.
  - The birth-death count chain of B1 sits on top.
- **(ii) Fast continuous baseline.**
  - y_{o,t} ~ NB(mean = c_o · exp(g_t), dispersion κ_o), with g_t a daily state-space GP (slow cycle plus OU with τ ≈ 10-20 d for group persistence).
  - A Laplace approximation by Gauss-Newton iterated Kalman smoothing is O(T·d³) per iteration, about 1.5x10^5 x 64 ≈ 10^7 flops, which takes milliseconds.
  - Useful for initial observer scale factors c_o (the classical k-factor) and for initialization of the full model.
- **Observer thresholds.** In the continuous model the threshold enters only as a multiplicative scale c_o (and perhaps zero inflation). It cannot represent threshold-dependent nonlinearity: at low activity a high-threshold observer loses proportionally more groups. The HMM thinning with activity-dependent q_o can.

**(d) Pitfalls.**
- Laplace and Gaussian approximations of a log-intensity posterior are least accurate when counts are mostly 0 or 1. This is exactly the Maunder regime, and an inference from general latent-Gaussian-model experience rather than a sourced statement.
- Daily-resolution GP latents with NUTS (≈ 1.5x10^5 x d parameters) give long trajectories and slow mixing. Use coarse grids.
- A quasi-periodic GP prior with fixed ~11-yr period can impose cycles during the Maunder Minimum. Use a period-jittered or non-stationary prior, and test sensitivity with a plain Matérn prior.

**(e) JAX tooling.**
- dynamax LGSSM (serial and parallel Kalman);
- BayesNewton (JAX; Laplace, EP and VI for state-space GPs);
- NumPyro `scan` with a non-centered GP;
- tinygp also offers quasiseparable/celerite-style O(n) GP kernels in JAX (from memory; not verified this session).

### Gaps
- I did not verify BayesNewton's maintenance status, or whether it uses objax versus plain JAX in current versions.
- I found no JAX/NumPyro implementation of INLA proper. An INLA-style workflow would have to be assembled: Laplace inner loop plus HMC or quadrature over hyperparameters.

---

## B3. Particle MCMC, SMC and differentiable particle filters, for when the latent is a population of groups with ages and areas

### Takeaway
Particle methods are needed only if the latent state must carry individual groups (age, area, longitude). In JAX this is feasible with padded fixed-size group arrays, at about 10^10 element operations per filter pass with 1,000 particles.

PMMH over the full 151k-day record is infeasible. The likelihood estimator's variance grows with T, so particle numbers must grow roughly linearly in T. Particle Gibbs with ancestor sampling, applied in time blocks, is workable.

Recommended role: a validation tool. Use it to test whether the B1 HMM approximation (count-only or 2-class) biases q_o or GN, on selected decades. Do not use it as the production engine.

### Cited Findings
- **Particle MCMC.** PMMH uses an unbiased particle-filter likelihood estimate inside MH. Particle Gibbs uses conditional SMC to sample latent paths. — [Andrieu, Doucet & Holenstein 2010, JRSS-B †](https://doi.org/10.1111/j.1467-9868.2009.00736.x)
- **PMMH tuning.** Efficiency is best when the standard deviation of the log-likelihood estimator is about 1-1.7, which implies particle numbers that must grow with data length. — [Doucet, Pitt, Deligiannidis & Kohn 2015, Biometrika †](https://arxiv.org/abs/1210.1871)
- **Particle Gibbs with ancestor sampling (PGAS)** mixes well with few particles on long series. — [Lindsten, Jordan & Schön 2014, JMLR †](https://jmlr.org/papers/v15/lindsten14a.html)
- **SMC².** Nested SMC over parameters and states for state-space models. — [Chopin, Jacob & Papaspiliopoulos 2013, JRSS-B †](https://arxiv.org/abs/1101.1528)
- **Differentiable particle filtering.** Entropy-regularized optimal-transport resampling makes the particle-filter likelihood differentiable for gradient-based learning. — [Corenflos, Thornton, Deligiannidis & Doucet 2021, ICML †](https://arxiv.org/abs/2102.07850)
- **BlackJAX** is a composable JAX library with HMC, NUTS, MALA, VI (mean-field, full-rank, Pathfinder), SG-MCMC, and SMC with resampling and tempering building blocks. — [BlackJAX paper, arXiv 2402.10797](https://arxiv.org/abs/2402.10797)
- **smcjax** extends dynamax and BlackJAX with bootstrap, auxiliary and Liu-West particle filters and Bayesian-workflow diagnostics. It is a small, individually maintained (thesis-linked) project. — [smcjax on PyPI](https://pypi.org/project/smcjax/)
- **particles** (Chopin) is a comprehensive SMC/PMCMC library in NumPy, not JAX. — [particles GitHub †](https://github.com/nchopin/particles)

### Inferences
**(c) Mapping to GN.**
- **Particle state.** Fixed arrays of G_max ≈ 64 slots per particle: (alive flag, birth day, peak area, current area, longitude or disk-passage phase).
- **Transition per day:**
  - Poisson(λ_t) births into empty slots, drawing area from the heavy-tailed reference area distribution;
  - lifetime ∝ peak area (Gnevyshev-Waldmeier-type rule);
  - deterministic rotation on and off the disk.
- **Emission.** Each observer thresholds the visible-group areas (exactly the ADF forward simulator) plus split and noise.
- **Cost.** Per step about P x G_max = 6.4x10^4 operations at P = 1,000, so about 10^10 over the full record. That is seconds on a GPU, but sequential depth is still T (latency-bound, ~1-3 s per pass).
- **PMMH infeasibility.** With thousands of informative observation-days, the log-likelihood variance at fixed P will be far above 1. Particle numbers of order 10^5-10^6 would be needed, so PMMH is ruled out for the full series.
- **Workable variants:**
  - PGAS inside a blocked Gibbs sampler (latent paths | θ; θ | paths), run over decade blocks;
  - a differentiable PF for approximate MLE or VI of a few global physical parameters on selected decades (1850-1900, with RGO overlap, as a check);
  - the PF as a posterior-predictive simulator to check the HMM model.
- **Data-tempering SMC** (Chopin 2002 †, [Biometrika](https://doi.org/10.1093/biomet/89.3.539)) over static parameters with the exact HMM likelihood from B1 is a strong alternative to NUTS. See B4/B5.

**(d) Pitfalls.**
- Resampling breaks differentiability. OT resampling costs O(P²) per step and gives biased gradients.
- Weight degeneracy on days with many observers. Use the auxiliary PF or tempered emissions.
- Fixed G_max must exceed the maximum number of visible plus far-side tracked groups.
- The JAX PF ecosystem is thin. BlackJAX has SMC building blocks (resampling, tempering) but no turnkey state-space PF. smcjax is small.

**(e) JAX tooling.**
- BlackJAX resampling functions plus a hand-written `lax.scan` filter;
- smcjax for reference implementations;
- dynamax for the HMM pieces.

### Gaps
- I found no maintained JAX library for particle Gibbs with ancestor sampling. It would need custom implementation, roughly 200 lines.
- I did not verify current BlackJAX API names for SMC adaptation or partial-posterior SMC.

---

## B4. Stochastic VI with minibatching over time blocks and observers, data-tempering SMC, divide-and-conquer posteriors, and block structure

### Takeaway
The model is block-structured: the latent activity is shared, observer parameters are local, and observers are conditionally independent given N_{1:T}. That favours:
1. a blocked Gibbs or HMC-within-Gibbs sampler: parallel FFBS for N | θ, then vmapped per-observer updates θ_o | N, then a λ-GP update;
2. SVI with minibatches of time blocks with buffers, for fast development.

Minibatching over observers alone is not valid with the latent marginalized, because the observers are coupled through N. It is valid conditional on sampled N. Divide-and-conquer by era is possible because observer overlaps link eras, but boundary handling makes Gibbs-style blocking preferable to consensus averaging.

### Cited Findings
- **SVI for HMMs.** Uses minibatches of subchains with buffer regions to correct boundary effects for long chains. — [Foti, Xu, Laird & Fox 2014, NeurIPS †](https://arxiv.org/abs/1411.1670)
- **SG-MCMC for state-space models.** Buffered stochastic gradients over subsequences, with gradient bias decaying geometrically in buffer length for mixing latent chains. — [Aicher, Ma, Foti & Fox 2019, SIAM J. Math. Data Sci. †](https://arxiv.org/abs/1810.09098)
- **Stochastic variational inference** (global/local structure, natural-gradient minibatch updates). — [Hoffman, Blei, Wang & Paisley 2013, JMLR †](https://arxiv.org/abs/1206.7051)
- **Consensus Monte Carlo.** Run MCMC on data shards and combine the subposterior draws by weighted averaging. It is exact for Gaussian subposteriors and approximate otherwise. — [Scott et al. 2016 †](https://doi.org/10.1080/17509653.2016.1142191)
- **Data-tempering (sequential) SMC for static models.** Add data batches progressively, re-weighting and moving particles with MCMC kernels. — [Chopin 2002, Biometrika †](https://doi.org/10.1093/biomet/89.3.539)
- **BlackJAX tempered and SMC building blocks** (resampling, tempering) — [BlackJAX paper](https://arxiv.org/abs/2402.10797).
- **Two-stage or recursive hierarchical combination** of unit-level posteriors — [Lunn et al. 2013 †](https://doi.org/10.1111/rssc.12007).

### Inferences
**(c) Mapping to GN, with cost estimates.**
- **(i) Blocked Gibbs (recommended production sampler).**
  1. N_{1:T} | θ, λ: parallel FFBS (dynamax), about 0.05-1 s per draw.
  2. θ_o | N_{1:T}: 600 independent low-dimensional posteriors (4-6 parameters each). Use vmapped NUTS or MALA steps in BlackJAX, or NumPyro `HMCGibbs` with a custom Gibbs function. Each observer's data are small, so this takes milliseconds per sweep on GPU.
  3. Survivors s_t and births b_t = N_t − s_t | N_{1:T} (per-day discrete draws), then log λ | b, which is a Poisson-log-Gaussian problem on the 27-day grid (HMC, or elliptical slice sampling with a state-space GP).
  4. Population hyperparameters η | θ_{1:O} (conjugate or NUTS).

  One sweep takes about 1-3 s, so 10^4 sweeps take about 3-8 h. Running 64-256 chains vmapped on one GPU costs about the same wall time.
- **Gibbs mixing risk.** Strong posterior coupling between N and θ_o (thresholds versus latent level, especially pre-1874 where few observers overlap) can make Gibbs mix slowly. Mitigations:
  - a collapsed step updating θ_o with N marginalized for the worst observers;
  - parameter expansion: a joint scale move on all q_o and λ;
  - checking split-R̂ / ESS on yearly GN.
- **(ii) SVI with time-block minibatches.**
  - Variational q over θ_o (mean-field Normal), η and the λ grid. The HMM latent is collapsed exactly inside each block.
  - Blocks of about 4 yr (1,461 days) plus a 100-200-day buffer each side. The latent mixing time is about the group lifetime of weeks; 1/(1−φ) ≈ 10-30 d. So the buffer bias is small, following Aicher et al.'s geometric-decay result.
  - Each step costs a scan of about 1,800 days, which takes milliseconds. Scale the ELBO by T/block length. About 10^4-10^5 steps run in minutes to an hour on CPU.
  - Best for exploration and model development. VI posteriors will be under-dispersed, so do not use them for final UQ.
- **(iii) Likelihood-tempered SMC (final UQ alternative to Gibbs).**
  - Use the exact B1 marginal likelihood with 1,000-4,000 particles over static parameters (θ_{1:O}, η, λ-GP), with HMC moves.
  - Batched over particles, each tempering stage of about 10-20 HMC moves with about 10 leapfrog steps needs about 200-400 batched forward-and-backward passes. At about 1-3 s each (batched on GPU), that is about 4-20 min per stage, and 20-40 stages take about 2-12 h.
  - It is embarrassingly parallel across particles and gives a marginal-likelihood estimate for model comparison: count-only versus 2-class, with or without drift.
  - **Data tempering in time.** Start from the reference era and add observers backward in time era by era. This is natural for GN and shows how each era's observers move the posterior, which is useful in the methods paper.
- **(iv) Divide-and-conquer by era.** For example 1610-1750, 1749-1880 and 1874-2025, with overlapping observers shared.
  - Consensus averaging assumes near-Gaussian subposteriors on shared parameters. The shared quantities here (observer parameters of bridging observers, η) are low-dimensional, so it is plausible.
  - Boundary days must be handled. Prefer blocked Gibbs over eras, conditioning on boundary latent states.

**(d) Pitfalls.**
- Minibatching observers while marginalizing N gives a biased ELBO, because observer likelihood terms do not factorize given only global parameters.
- Time-block minibatches give biased gradients near block edges without buffers.
- SVI under-dispersion.
- Consensus combination fails for multimodal or skewed subposteriors, for example a poorly identified pre-1700 observer threshold.

**(e) JAX tooling.**
- NumPyro SVI with custom `numpyro.factor` block likelihoods, and `numpyro.plate(subsample_size=...)` for subsampling over blocks;
- NumPyro `HMCGibbs`;
- BlackJAX NUTS/MALA plus SMC with tempering;
- dynamax FFBS.

### Gaps
- No source found for a published blocked-Gibbs design for "shared latent count chain plus many local observer emission models" at this scale. The scheme is standard in spirit (data augmentation for HMMs), but the GN-specific mixing behaviour is unknown and must be measured on mocks.

---

## B5. Practical JAX/NumPyro limits: scan memory for 150k steps x hundreds of observers, sparse observations, GPU versus CPU, and NUTS mixing for hierarchical observer parameters

### Takeaway
At this problem size, the binding constraints are sequential latency (151k scan steps) and NUTS gradient cost. Memory and flops are not binding.
- Keep the latent HMM state small (K ≈ 41-60, at most 2 area classes).
- Store observations in long format with segment sums.
- Use `jax.checkpoint` on scan bodies and float64 accumulation.
- Batch many chains or particles on a GPU to amortize per-step latency.
- Use non-centered parameterizations for observer parameters and the λ-GP.
- Expect full joint NUTS over all parameters with the marginalized HMM to be too slow (days per thousand draws). Blocked Gibbs or tempered SMC is the practical route.

### Cited Findings
- **Gradient checkpointing.** `jax.checkpoint` (remat) trades recomputation for memory in reverse-mode autodiff. It is applicable to `lax.scan` bodies. — [JAX gradient-checkpointing docs †](https://jax.readthedocs.io/en/latest/gradient-checkpointing.html)
- **Hierarchical geometry.** Hierarchical models create funnel geometries that HMC struggles with. Non-centered parameterizations fix this when data per unit are weak. — [Betancourt & Girolami 2015 †](https://arxiv.org/abs/1312.0906)
- **Automatic reparameterization.** Variationally inferred and automatic choices between centered and non-centered parameterizations for probabilistic programs. — [Gorinova, Moore & Hoffman 2020, ICML †](https://arxiv.org/abs/1906.03028) NumPyro exposes `LocScaleReparam(centered=...)`. — [NumPyro reparam docs †](https://num.pyro.ai/en/stable/reparam.html)
- **NumPyro discrete-in-scan fragility.** Enumeration plus scan plus `DiscreteHMCGibbs` combinations have known failure modes (forum threads cited in B1). — [forum](https://forum.pyro.ai/t/discretehmcgibbs-enumeration-cannot-detect-any-discrete-latent/5314)
- **dynamax parallel HMM** elements are T x K x K. — [dynamax source](https://github.com/probml/dynamax/blob/main/dynamax/hidden_markov_model/parallel_inference.py)
- **BlackJAX** supports composable kernels (HMC, NUTS, MALA, SG-MCMC, SMC) that can be vmapped across chains. — [BlackJAX paper](https://arxiv.org/abs/2402.10797)
- **Many-short-chains HMC adaptation.** ChEES-HMC (trajectory-length adaptation across many parallel chains) suits GPU-batched sampling. — [Hoffman, Radul & Sountsov 2021, AISTATS †](http://proceedings.mlr.press/v130/hoffman21a.html)

### Inferences
**Memory budget (float32; double for float64).**
- **Log-emission array.** T x K = 6.2x10^6 entries, about 25 MB.
- **Forward filter.** T x K, about 25 MB.
- **Reverse-mode residuals.** About 1-3 x that, or up to 1 GB if K x K per-step intermediates are stored, which is why remat is needed.
- **Parallel filter.** About 1 GB per T x K x K copy, and a few copies, so 3-6 GB with gradients. That fits on 24-80 GB GPUs, but not if batched across chains. Use the chunked hybrid for batched settings.
- **Observations.** Long-format records (observer, day, count, flags): about 2-5x10^6 observer-active-days x 16 bytes is under 100 MB. Never use a dense T x O array.
- **Per-observer emission tables.** O x epochs x K x Y_max ≈ 600 x 20 x 41 x 60 ≈ 3x10^7 entries, about 120 MB.

**Sparse observation handling.**
- Precompute, for each record, (day index, observer index, epoch index, count).
- Per likelihood evaluation:
  1. build emission tables;
  2. `table[obs, epoch, :, y]` gather gives an M x K array;
  3. `segment_sum` over day gives T x K.
- Days with no observers contribute a zero row and cost nothing extra.
- For the missing-not-at-random no-report term, either generate implicit "no-report" records for active-window days, or compute it per observer-epoch in closed form if π_o is constant.

**GPU versus CPU.**
- A single-chain 151k-step sequential scan with tiny (41-vector) bodies is latency-bound. CPU is often as fast or faster.
- GPUs win when:
  - many chains or SMC particles are vmapped (256-4,096), so per-step cost is amortized;
  - the parallel or chunked associative scan is used;
  - NPE networks are trained.
- Practical split: CPU for single-chain development and debugging, and GPU for batched production (tempered SMC or many-chain Gibbs) and SBI training. These are estimates; benchmark early.

**NUTS feasibility (full joint).**
- Parameters: about 600 x 4-6 observer parameters, plus about 5,600 x 2-4 λ-GP innovations, plus η. That is about 1.5-3x10^4 dimensions.
- With tree depth 8-10 (255-1,023 gradients per iteration) and about 1-3 s per gradient, one iteration takes 4-50 min. 2,000 iterations would take days to weeks per chain.
- Hence the blocked-Gibbs, tempered-SMC or SVI recommendations in B4.
- NUTS is fine for:
  - per-observer conditionals θ_o | N (5 dimensions, cheap);
  - the λ-GP conditional given births, about 10^4 dimensions, but with a cheap Poisson likelihood, so milliseconds per gradient;
  - small mock networks.

**Non-centered parameterizations.**
- θ_o = μ_η(era) + σ_η · z_o with z_o ~ N(0, 1), for observers with sparse or short records.
- Centered may be better for long-career, data-rich observers. NumPyro `LocScaleReparam(centered=...)` lets this be learned or set per observer group.
- The λ-GP should use white-noise innovations driving the state-space recursion.
- Drift: a per-observer random walk over 5-yr epochs, non-centered, with a tight prior on step size.

**Compilation.**
- `lax.scan` over 151k steps compiles once, in seconds.
- Avoid Python loops over observers. Use index arrays with vmap or segment ops.
- Keep shapes static by padding observer epochs.

### Gaps
- No authoritative published numbers were found for JAX `lax.scan` per-iteration overhead on GPU versus CPU at this size. The timing figures here are estimates.
- I did not verify that BlackJAX's current release exposes ChEES/MEADS adaptation under the same names.

---

## Cross-cutting: what to implement for the GN series and the methods paper

### Takeaway
A hybrid is the defensible design.
- **SBI (Part A)** calibrates the observer emission model against the area-resolved reference and quantifies its uncertainty. It replaces ADF's EMD/kNN-ABC with amortized NPE or NLE plus SBC/TARP.
- **Exact likelihood-based state-space inference (Part B)** carries the shared latent daily group count through 1610-2025: an immigration-death HMM with a state-space-GP birth rate, inferred by blocked Gibbs (parallel FFBS) and/or tempered SMC in JAX.
- **Out-of-distribution eras** are handled by parametric extrapolation of λ_t, reference group-thinning augmentation, summary-space out-of-distribution flags, and an end-to-end mock-network coverage test.

### Cited Findings
- Hierarchical and amortized SBI methods target exchangeable units with low-dimensional shared parameters. — [Habermann et al. 2025](https://arxiv.org/abs/2408.13230); [Heinrich et al. 2024](https://arxiv.org/abs/2306.12584)
- Naive neural SBI is fragile under misspecification. — [Cannon, Ward & Schmon 2022](https://arxiv.org/abs/2209.01845)
- Naive neural SBI is often overconfident. — [Hermans et al. 2022](https://arxiv.org/abs/2110.06581)
- Exact parallel-in-time HMM filtering, smoothing and sampling exist in JAX (dynamax), with T x K x K elements. — [dynamax source](https://github.com/probml/dynamax/blob/main/dynamax/hidden_markov_model/parallel_inference.py); [Hassan et al. 2021](https://arxiv.org/abs/2102.05743)
- BayesFlow 2 (Keras 3, JAX backend) and sbijax provide JAX-compatible amortized SBI. — [BayesFlow](https://github.com/bayesflow-org/bayesflow); [sbijax](https://github.com/dirmeier/sbijax)

### Inferences
**Implementation roadmap, roughly in order. Effort estimates are my own.**
1. **JAX forward simulator** from the RGO/Debrecen group list: threshold, mask, split, drift, zero-drop, and whole-group thinning. 1-2 weeks.
2. **Emission calibration via SBI.**
   - NPE (BayesFlow-JAX) for θ_o of every observer, as a direct ADF replacement and a baseline for the paper.
   - SBC and TARP on synthetic observers.
   - Out-of-distribution flags via summary-space MMD.
   - Two-stage hierarchical re-weighting for era-dependent population priors.

   2-4 weeks, plus about 1 GPU-day of training.
3. **Exact HMM likelihood model.**
   - INAR immigration-death chain with K ≈ 41-60.
   - Optional 2-class area extension.
   - Activity-dependent thinning q(A_thr, λ) learned from the reference.
   - Missing-not-at-random zero model.
   - λ on a 27-day state-space-GP grid.
   - Unit-test the forward algorithm against brute-force enumeration on short series, and the parallel filter against the serial one.

   2-3 weeks.
4. **Inference.** SVI with buffered time-blocks for development, then blocked Gibbs (parallel FFBS plus vmapped per-observer NUTS plus λ-HMC) with 64-256 GPU-batched chains. Optionally tempered SMC with era-wise data tempering for model evidence and robustness. Hours per production run.
5. **Validation for the methods paper.**
   - End-to-end mock networks (10-20 mocks) with injected Maunder-like low activity, giving GN coverage by era.
   - Leave-one-observer-out predictive checks in the reference era.
   - Sensitivity analyses: binary-only (ADF-like) emissions versus counts; 1 versus 2 area classes; GP prior choice; thinning versus parametric area model.
   - Particle-filter group-population model on 1-2 decades as a structural check of the HMM approximation.
6. **Outputs.** Posterior draws of N_{1:T} from FFBS, aggregated to daily, monthly and yearly GN with credible bands. Per-observer posteriors (threshold, split, drift, zero-loss) with out-of-distribution flags.

**Why not pure SBI end to end.** The shared latent series is about 1.5x10^5-dimensional. Observers before 1874 are coupled only through it, and the activity regimes most in need of homogenization are out of distribution for any reference-trained network. Likelihood-based inference uses all cross-observer overlap information exactly and extrapolates through an explicit, inspectable model.

**Why not pure likelihood.** The observer detection process (threshold on heavy-tailed areas, splitting conventions, persistence of large groups) is most faithfully represented by simulation from the reference. NLE or NPE trained on those simulations is the principled way to import it, together with its uncertainty, and SBC/TARP give the calibration evidence that ADF lacks.

### Gaps
- Two empirical facts underpin the whole design and are untested here:
  - whether the immigration-death count chain plus thinning reproduces RGO daily-count statistics (autocorrelation, zero fraction versus activity);
  - whether grand-minimum group statistics resemble thinned modern groups.

  Both should be checked early, and the second requires solar-physics literature outside this scope.
- No published end-to-end Bayesian or SBI reconstruction of GN with this architecture was found in this search. Overlap with existing Bayesian GN work by other groups should be checked by the researchers covering sunspot-specific literature.
