# Instrument/Survey Intercalibration and Long-Record Homogenization Methods as Analogs for the Sunspot Group Number (GN) Record

Scope note: these notes cover methods outside sunspot work. Each section maps them onto the GN problem: about 600 observers from 1610 to the present, daily integer group counts, a latent true count, observer-specific thresholds and drifts, irregular observing days, sparse chain-like overlaps, and current backbone/daisy-chain or distribution-matching (ADF) calibration against RGO/Debrecen. Facts carry inline sources. Items marked "Inference" are my own reasoning or derivations. A few canonical references I did not open this session are flagged as such.

---

## Q1. Astronomical self-calibration (SDSS ubercal, Holmes-Hogg-Rix survey design, Pan-STARRS ubercal, DES FGCM): global simultaneous solution vs. chaining, unconstrained modes, and error growth with graph distance

### Takeaway
Every large photometric survey dropped chained calibration (standard star → secondary patch → field) in favour of one global fit of all zero-points and instrument parameters to every overlap at once. Two things limit how well that fit works. One is the topology of the overlap graph: regular, repeated tilings leave whole modes unconstrained. The other is exactly degenerate modes, namely the overall zero-point, disconnected sub-networks, and slow "tilt" modes driven by systematic drifts. These modes have to be fixed by a small number of absolute anchors or by priors. The SDSS experience maps closely onto GN. A chained calibration's errors were dominated by the chain's weakest link. The global fit roughly halved the scatter. The worst residual systematic was a survey-wide tilt that built up from a slowly varying, unmodelled term, which is the photometric counterpart of an unmodelled observer drift.

### Cited Findings

**(a) Core idea and math**
- **SDSS ubercalibration** (Padmanabhan et al. 2008). It "simultaneously solves for the calibration parameters and relative stellar fluxes using overlapping observations," which reduces absolute calibration to "determining a few numbers for the entire survey." The result is about 1% relative error over 8500 deg² in griz and about 2% in u — [Padmanabhan et al. 2008, ApJ 674, 1217 (arXiv:astro-ph/0703454)](https://arxiv.org/abs/astro-ph/0703454)
- The ubercal model is m = m_ADU + a − [k + (dk/dt)(t − t_ref)]·x + f(j). Here a is a zero-point per night and camera column, k an extinction term per night, x the airmass, and f a flat field per season. Each star's true magnitude is an inverse-variance weighted mean that is marginalized analytically out of χ². The χ² has the form (Ap − b)ᵀC⁻¹(Ap − b), with A about 50,000,000 × 2000 and "extremely sparse, and amenable to sparse matrix techniques." It is solved through the normal equations, with flats on a quadratic B-spline of 17 knots — [Padmanabhan et al. 2008 (ar5iv full text)](https://ar5iv.labs.arxiv.org/html/astro-ph/0703454)
- **Holmes, Hogg & Rix (2012) self-calibration** uses the model c_i = f(x_i | q)·s_k + e_i, with flat-field parameters q and source fluxes s_k. The objective is χ² = Σ_k Σ_{i∈O(k)} (c_i − f_i s_k)²/σ_i². It is solved bilinearly: a closed-form flux step, then a linear solve G q' = b for the flat field, iterated to convergence. The test deliberately used a wrong noise model and a fitted basis (8th-order polynomial, 45 coefficients) that does not contain the true 260-parameter response — [Holmes, Hogg & Rix 2012, PASP (arXiv:1203.6255; ar5iv)](https://ar5iv.labs.arxiv.org/html/1203.6255); DOI 10.1086/668656 per [arXiv listing](https://arxiv.org/abs/1203.6255)
- **Pan-STARRS1 ubercal** (Schlafly et al. 2012) uses repeat observations of stars, "solving simultaneously for the system throughput, the atmospheric transparency, and the large-scale detector flat field." It reaches <10 mmag relative precision in g, r, i and about 10 mmag in z, y. A comparison with SDSS shows that errors in both surveys contribute similarly to their differences — [Schlafly et al. 2012, ApJ 756, 158 (arXiv:1201.2208)](https://arxiv.org/abs/1201.2208)
- **DES Forward Global Calibration Method (FGCM)** (Burke et al. 2018) combines auxiliary instrument data, survey images, and instrument and atmosphere models into a forward model of each exposure's passband. A passband-plus-spectrum ("chromatic") correction to a standard system is needed for sub-percent calibration. Residual random error is 5–6 mmag per exposure, uniformity is 7 mmag over 5000 deg², and spectral systematics are <5 mmag for 0.5 < g−i < 3.0 — [Burke et al. 2018, AJ 155, 41 (arXiv:1706.01542)](https://arxiv.org/abs/1706.01542)

**Global vs. chained: direct evidence**
- The pre-ubercal SDSS calibration was a multi-stage chain: BD+17°4708 → USNO primary network → 20-inch Photometric Telescope (PT) → secondary patches. It mixed absolute and relative errors, and "the Data Release errors are therefore dominated by the Photometric Telescope (PT) based calibration method." Against DR4, ubercal differs by about 2% rms in griz and 3% in u. The differences show striping and correlations across columns that trace back to the 40′ chain-calibration patches — [Padmanabhan et al. 2008 (ar5iv)](https://ar5iv.labs.arxiv.org/html/astro-ph/0703454)
- SDSS's normal scanning strategy had few repeats: "the only significant repeat observations occur at the poles of the survey." The team added the "Apache Wheel," a set of long cross-scans, to connect weakly linked regions. In poorly connected regions it cut errors from 10.4 to 7.8 mmag. Calibration errors are smaller near the poles, where overlaps are denser — [Padmanabhan et al. 2008 (ar5iv)](https://ar5iv.labs.arxiv.org/html/astro-ph/0703454)

**Overlap topologies that leave unconstrained modes**
- Holmes et al. compared four surveys of equal size, each with about 1,290 pointings over 64 deg²:
  - **A**, a regular grid aligned in all 9 passes, did not converge (5,000 iterations) and gave 1.17% source rms and 1.68% badness.
  - **B**, the same grid rotated 40° per pass, gave 0.20% rms and 0.51% badness.
  - **C**, alternating 12×12, 13×11 and 11×13 grids, gave 0.16% rms and 0.36% badness.
  - **D**, quasi-random offsets and orientations, gave 0.13% rms and 0.32% badness, with best-in-basis badness of 0.012%.

  In A the flat-field is "effectively unconstrained in the interior," because repeats link only the edges of the focal plane. In B the centre stays poorly connected. — [Holmes et al. 2012 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1203.6255)
- The paper's central recommendation is: "Do not use a regular, repeated tiling of the sky; instead return the same sources to very different focal-plane positions." It also says connectivity matters more than uniform exposure time, and that small changes to the pointing pattern (survey C) can drastically improve performance — [Holmes et al. 2012 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1203.6255); [arXiv abstract](https://arxiv.org/abs/1203.6255)

**(d) Degeneracies and how they were handled**
- Flux and flat-field are determined only up to a common scale. You cannot tell whether every source is fainter or the response is uniformly lower. A few absolute standards break this. The method also assumes the response is stable in time; with time variation, you need a time-evolving response model and sources observed at different times — [Holmes et al. 2012 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1203.6255)
- SDSS: "any overall additive shift of all the a-terms does not change" χ². Regions with no spatial or temporal overlap can shift independently. Flats are degenerate with a-terms, which was fixed by forcing zero-mean flats. With little airmass range, a − kx is nearly degenerate. Eigenmodes with small eigenvalues (those that could cause errors above about 1%) were projected out and then restored with Gaussian priors. "The worst constrained mode is, as expected, the zero point of the calibration which is exactly degenerate." — [Padmanabhan et al. 2008 (ar5iv)](https://ar5iv.labs.arxiv.org/html/astro-ph/0703454)
- **Error accumulation from an unmodelled slow drift.** Errors in dk/dt (a time derivative of extinction that was fixed, not fitted) built up into a survey-wide tilt of about 10 mmag, because the sky was always scanned west to east. The authors call this "the most serious systematic error in the calibration." Natural overlaps mostly fell on the same CCD column, so they were partly degenerate with the flat fields — [Padmanabhan et al. 2008 (ar5iv)](https://ar5iv.labs.arxiv.org/html/astro-ph/0703454)
- Errors were dominated by "the unmodeled random fluctuations in the k-terms" (atmosphere). Noise-free simulations reach about 0.1%. Calibration quality is therefore limited by model misspecification, not by solver or statistics — [Padmanabhan et al. 2008 (ar5iv)](https://ar5iv.labs.arxiv.org/html/astro-ph/0703454)

### Inferences

**(c) Mapping to GN**
- Each observer is a "camera column" or "exposure." Each observer-day is a "detection." The latent true daily group count G(t) is the "star flux." The target is one joint likelihood over all observers' daily reports, with observer transfer parameters θ_o and latent G(t) fitted together. This is the ubercal analog of replacing backbone chaining. The current backbone method is the SDSS PT chain: errors are dominated by the bridge links, and they get imprinted as steps where the chain changes. The ADF-to-RGO approach is closer to "calibrate everyone directly to a standard," but its accuracy is bounded by how representative the reference's activity distribution is for each observer's era.

**Uncertainty growth with graph distance (derivation, not from a fetched source)**
- In log space let a_o = log k_o. Each overlap gives a noisy difference y_ij = a_i − a_j + ε_ij with weight w_ij = 1/Var(ε_ij). The weighted least-squares solution has covariance L⁺, the pseudo-inverse of the weighted graph Laplacian. So Var(a_i − a_j) equals the effective resistance R_ij between nodes i and j, with each edge a resistor of value 1/w_ij.
  - For a pure chain, R_ij = Σ 1/w along the path. Variance grows linearly with the number of links (a random walk), and the standard deviation grows as √N.
  - Parallel paths combine like parallel resistors, so the global solution is never worse than the best single chain and is usually much better.
  - This explains why the choice of path in daisy-chaining is "arbitrary." A shortest-path or backbone method throws away the parallel conductances.
- That result holds only for random link errors. A systematic bias per link (for example, from regression through the origin or a non-linear transfer) adds coherently along a chain and grows linearly in N, not as √N. The SDSS dk/dt tilt is the photometric example of this.
- The Laplacian has one exact null mode per connected component: a global scale for each component. Its smallest non-zero eigenvectors, the Fiedler-like modes, are the poorly constrained large-scale modes. They concentrate at graph bottlenecks.
  - For GN the obvious bottlenecks are the 17th to early 18th century (including the Maunder Minimum) and the gaps around the late 18th century. Those eras will be dominated by such modes.
  - A useful diagnostic for a methods paper: compute the overlap-graph Laplacian with edge weights equal to effective overlap-day counts divided by per-pair residual variance. Report the effective resistance from every observer to the reference, and the number of connected components by era.
- The Holmes result transfers directly. Overlap topology matters more than overlap volume. Observers whose careers span many others (long, central nodes) act like the "Apache Wheel" cross-scans. Sparse "regular-tiling-like" overlaps, where every observer overlaps only their immediate predecessor, give a chain with no redundancy.

**Forward-model (FGCM) analog**
- A per-observer "gray" k-factor resembles a gray zero-point. FGCM shows that sub-percent work needs a passband (response function) convolved with the source spectrum. For GN, the "passband" is the observer's detection probability as a function of group area or complexity, and the "spectrum" is the group-area distribution, which changes with activity level.
- So a k-factor that depends on activity is the GN counterpart of a chromatic term. A forward model that thins a reference area distribution with an observer-specific detection function (an ADF generalization) is the FGCM analog. The scalar ratio is the gray approximation that fails.

**Pitfalls for GN**
- The overall scale is exactly degenerate and must be anchored. Choosing the anchor (RGO vs. Debrecen vs. a modern observer, and which years) behaves like choosing the standard-star system.
- Pan-STARRS showed that the "reference" survey (SDSS) contributes half the disagreement. RGO/Debrecen should likewise be modelled as an observer with its own uncertainty, not as truth.
- Disconnected eras can each float independently unless a smoothness or physical prior on G(t), such as cycle-shape constraints, ties them. That prior then becomes the dominant source of information. It must be stated explicitly and tested in sensitivity runs. SDSS likewise restored projected-out null modes with priors.

**(e) Scalability and software**
- SDSS solved an approximately 50M × 2000 sparse system, so GN is tiny by comparison: about 600 observers times a handful of parameters, plus latent G(t) over about 151,600 days from 1610 to 2025. The latent field dominates the cost.
- In JAX/NumPyro the options are:
  - a Markov (state-space or AR) prior on log G(t), evaluated with lax.scan or a Kalman filter, which is linear in the number of days;
  - a coarser latent grid (for example, monthly) with daily observation noise;
  - SVI for exploration, then NUTS on reduced problems.
- The bilinear alternating scheme in Holmes et al. (fluxes in closed form, then response) is a cheap way to initialize HMC.

### Gaps
- I did not fetch a source that explicitly gives calibration error as a function of overlap-graph path length in these surveys. The effective-resistance result above is a standard least-squares derivation (my inference), not a quoted claim.
- I did not open the full Schlafly et al. (2012) or Burke et al. (2018) texts. Details of their degeneracy handling (for example, how FGCM fixes gray offsets with standard stars) and their public code repositories are unverified here. I believe FGCM has public Python code, but I did not confirm a URL.
- Gaia DR3 photometric calibration and Rubin/LSST calibration (2015–2026 developments) were not researched within the tool budget.

---

## Q2. Satellite record merging (MSU/AMSU: RSS, UAH, NOAA STAR SNO; SBUV merged ozone; GSICS SNO/double-differencing): overlaps, drift, and structural-uncertainty ensembles

### Takeaway
Satellite climate records are the closest operational analog to GN. They are a chain of instruments with short overlaps, instrument-specific offsets, non-linear responses, and drifts, such as diurnal sampling drift and calibration-target (warm-load) contamination. Two lessons matter most for GN:
- **Uncertainty is shown by ensembles.** RSS generates 400 error realizations and pushes them through the same processing code. SBUV and GSICS use Monte Carlo too. Independent teams (RSS, UAH, STAR) give different merged series, and that spread is the structural uncertainty.
- **Merging and drift uncertainty dominate on long time scales.** They can turn apparently significant trends into non-significant ones. Overlap matchups alone cannot fix absolute coefficients when the instruments are collinear (all similar).

### Cited Findings

**(a) Core idea and math: SNO intercalibration (NOAA STAR)**
- Zou et al. (2006) recalibrated MSU from level 0 using simultaneous nadir overpass (SNO) matchups: nadir pixels within <100 s and 111 km, for every overlapping NOAA satellite pair. The calibration equation is a dominant linear response of raw counts to Earth-view radiance plus a smaller quadratic non-linearity. Offsets and non-linear coefficients are fit by regression on the SNO pairs — [Zou et al., ITSC-15 (2006), companion to JGR 2006 paper](https://pubs.ssec.wisc.edu/research_Resources/publications/pdfs/ITSC15/zou01_ITSC15_2006.pdf)
- "SNO matchups alone cannot determine the absolute values of the coefficients because there is a high degree of colinearity between satellite SNO observations." Absolute values come from sensitivity experiments instead — [Zou et al., ITSC-15](https://pubs.ssec.wisc.edu/research_Resources/publications/pdfs/ITSC15/zou01_ITSC15_2006.pdf) (as summarized by search; full JGR text not fetched)
- Warm-target contamination is a drift-like instrument artifact, where the calibration target temperature varies with orbit. It was corrected in two ways: by using SNOs to find a best non-linearity, or by an empirical regression on warm-target temperature — [Zou & Wang 2010, J. Atmos. Oceanic Technol.](https://www.star.nesdis.noaa.gov/smcd/emb/mscat/documents/Zou.2010.JTech.Stability.pdf); operational description at [NOAA STAR MSCAT algorithm page](https://www.star.nesdis.noaa.gov/smcd/emb/mscat/algorithm.php) (criteria given there as <2 min and 111 km)

**(a) Structural uncertainty ensemble: RSS Monte Carlo**
- Mears, Wentz, Thorne & Bernie (2011, JGR) build total uncertainty from three parts: sampling error, the pre-merge adjustments to each satellite, and the merging procedure — [Mears et al. 2011 (Maynooth repository)](https://mural.maynoothuniversity.ie/6529); [full text PDF](https://mural.maynoothuniversity.ie/6529/1/PT_assessing%20uncertainty.pdf)
- RSS produced 400 realizations of the possible error for each MSU/AMSU dataset. Each realization is gridded like the data and can be added to the measurements to make one plausible version of the observations, so uncertainty is available at every space and time scale — [Mears, ITSC-17 slides](https://itwg.ssec.wisc.edu/wordpress/wp-content/uploads/2023/05/6.13_mears_itsc17.pdf); [Mears AGU 2011 poster](https://images.remss.com/papers/rssconf/mears_AGU_Fall_2011_Poster_discover_uncertainty.pdf)
- Sampling dominates at small scales. On long time scales, intersatellite merging matters at all levels, and the diurnal adjustment is a critical uncertainty for layers with a surface component, especially over land — [Mears et al. 2011 (Maynooth)](https://mural.maynoothuniversity.ie/6529)
- Earlier MSU uncertainty statements were "goodness of fit to a line" criteria, and the Monte Carlo approach is presented as different and complementary — [Mears, ITSC-17 slides](https://itwg.ssec.wisc.edu/wordpress/wp-content/uploads/2023/05/6.13_mears_itsc17.pdf)

**Drift handling and the size of structural differences**
- Most satellites' local observing time drifts, which aliases the diurnal cycle into the long-term record. No highly accurate independent source of the diurnal cycle exists. RSS V4.0 TLT (2017) optimized the diurnal adjustment in the same way as TMT V4.0 — [RSS FAQ on V4.0 TLT](https://www.remss.com/blog/faq-about-v40-tlt-update/)
- RSS V4 raised the TLT trend from 0.136 to 0.184 K/decade, widening the gap with UAH. Mears et al. (2005) found an error in the UAH diurnal correction behind a 40% jump in the UAH trend between versions 5.1 and 5.2 — [Wikipedia: UAH satellite temperature dataset](https://en.wikipedia.org/wiki/UAH_satellite_temperature_dataset) (secondary source; numbers not checked against primary papers)

**SBUV merged ozone**
- The SBUV v8.6 merged profile ozone uncertainty uses Monte Carlo simulation of each instrument's calibration and drift uncertainty. Merging uncertainty dominates the total for trends since 2001. Trends that look significant on statistical error alone become non-significant at 2σ once instrument uncertainty is included. Fitting the 1979–2015 record to EESC (effective stratospheric chlorine) remains significant in the upper stratosphere — [Frith et al. 2017, ACP 17, 14695](https://acp.copernicus.org/articles/17/14695/2017/)
- An earlier Monte Carlo approach for the merged total ozone data set included systematic and random inter-instrument differences and potential drift — [ACPD 2006](https://acp.copernicus.org/preprints/6/3883/2006/acpd-6-3883-2006.pdf). The total-ozone Monte Carlo was updated through June 2014, and it found a 1.5% offset in OMI's ozone cross-sections relative to SBUV/OMPS — [Frith et al. AGU 2014 poster](https://acd-ext.gsfc.nasa.gov/People/Frith/webdir/modprof/pubpage/AGU_Fall_2014_Frith_et_al.pdf)

**GSICS SNO and double-differencing**
- In double-differencing (DD), a third instrument or a radiative-transfer model is the transfer standard. This allows comparing two satellites that never observe simultaneously: each is compared to the transfer, and the two differences are differenced. NOAA ICVS-LTM uses both third-sensor DD (tied to SNO/SCO) and RTM-based DD — [NOAA repository report](https://repository.library.noaa.gov/view/noaa/65872)
- GSICS fits weighted linear regressions to collocated pairs with similar viewing geometry. It reports the bias at a standard scene radiance, with an uncertainty, and estimates per-collocation uncertainty from the radiance variance within the footprint — [Hewison, GSICS overview](https://ipwg12.imgw.pl/wp-content/uploads/2026/07/S10.9_Hewison_GSICS_updated.pdf); [Hewison, ITSC-21](https://itwg.ssec.wisc.edu/wordpress/wp-content/uploads/2023/05/3.01_hewison_itsc21.pdf)
- SEVIRI IR13.4 vs. IASI-A/B gave double differences <0.1 K, with no significant drift over 2013–2017 and a trend standard uncertainty of about 3 mK/yr — [NOAA repository report](https://repository.library.noaa.gov/view/noaa/65872/noaa_65872_DS1.pdf) (as summarized by search)
- SNO works well with large, globally representative matchup samples but struggles with sparse sampling or orbital drift. DD gave higher accuracy in a FY-3D MERSI study — [Remote Sensing 17, 03353 (2025)](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-17-03353/article_deploy/remotesensing-17-03353.pdf)

### Inferences

**(c) Mapping to GN**
- SNO matchups correspond to same-day observations by two observers, the only true "simultaneous overpasses." Double-differencing corresponds to comparing two non-overlapping observers through a third observer or a model, which is what the backbone does implicitly.
  - DD through a model is the more relevant variant here. The latent G(t) in a joint Bayesian model plays the role of the RTM: each observer is compared to the model, not to each other. This removes the need for exact same-day overlaps, provided the latent-process prior is credible across gaps.
- GSICS reports bias "at a standard scene radiance" because the bias depends on the scene. This matches the GN finding that k-factors depend on activity level. The fix is the same in both fields: report or model the transfer function (bias as a function of level), not one ratio. Zou's linear-plus-quadratic calibration shows that the merging community treats non-linearity as the default.
- Zou's colinearity result matters for GN drift models. If adjacent observers share similar thresholds and drifts, overlaps constrain only their differences. A common-mode drift shared by all observers in an era is unidentifiable from overlaps and is confounded with real secular change in G.
  - In GN terms: if many 19th-century observers had similar telescope upgrades over the same decades, a slow common drift is aliased into the long-term trend. This is the "drift vs. real trend" confound.
  - It can be broken only by (i) observers known on independent grounds to be stable, (ii) physical or statistical priors on G(t), or (iii) external proxies (cosmogenic isotopes, geomagnetic indices), each of which brings its own assumptions.
- **RSS-style ensemble for GN.** Generate N posterior draws of all observer transfer parameters and latent G(t). Publish them as realizations, not just a mean ± σ, so users can propagate uncertainty to cycle amplitudes, Gleissberg-scale trends, or cross-correlations at any scale.
- **Structural uncertainty for GN.** Add ensemble members that differ in modelling choices, as RSS/UAH/STAR do:
  - drift parameterization (none, linear, random walk, changepoints);
  - choice of reference (RGO vs. Debrecen vs. a modern observer);
  - observation model (Poisson vs. negative binomial vs. thinning-of-areas);
  - inclusion rules for observers.

  The SBUV result suggests merging uncertainty will dominate long-term GN trend uncertainty, so it must be included before any statement about secular trends.
- The RSS V4 vs. UAH history is a cautionary analog. A single adjustment component (diurnal drift) moved the trend by tens of percent and caused years of dispute. GN has had the same pattern: competing reconstructions differ mainly in how a few bridge links are calibrated. A methods paper should decompose GN trend uncertainty by component (sampling, per-observer calibration, merging/bridging, drift model), following Mears et al. (2011).

**(d) Pitfalls**
- Collinearity and common-mode drift (above).
- Overlaps that sample only part of the activity range bias a linear calibration, much as SNOs at high latitudes sample only cold scenes. An overlap during solar minimum constrains only the low-count end of the transfer function.
- Ensemble spread is only as good as the error model. RSS's realizations reflect assumed error structures, so "unknown unknowns" are not captured.

**(e) Scalability**
- Monte Carlo ensembles are trivially parallel. In JAX, posterior predictive realizations come directly from NUTS samples. For structural ensembles, vmap over configurations or run separate jobs.

### Gaps
- I did not retrieve the UAH methodology papers (Christy/Spencer, e.g., Spencer et al. 2017) or the NOAA STAR merged-series uncertainty documentation, so the UAH "backbone"-style approach and STAR's structural uncertainty estimates are undocumented here.
- The full Zou et al. (2006) JGR text was not fetched. The colinearity statement comes from a summary of the ITSC companion paper.
- The 0.136 → 0.184 K/decade figure is from Wikipedia and was not verified against Mears & Wentz (2017, J. Climate).
- No single authoritative GSICS document deriving the full DD uncertainty budget was found.

---

## Q3. Total solar irradiance composites (Dudok de Wit et al. 2017 multi-scale ML; Montillet et al. 2022 Bayesian/GP fusion): time-scale-dependent weighting and degradation

### Takeaway
The TSI community moved from PI-selected, switch-at-transition composites to statistical fusion. Every instrument contributes at every time, weighted by its noise at each time scale. Noise is estimated from the data and found to be roughly 1/f (flicker), so long-term stability is far worse than short-term precision. The key GN lessons:
- Uncertainty must be scale-dependent. Cycle-to-cycle and secular uncertainties are not the daily noise divided by √N.
- Trend significance depends heavily on the assumed noise model. Under white noise, spurious trends look significant.
- The absolute level is set by an explicitly chosen reference.
- Correcting degradation, which here plays the role of observer drift, was deliberately kept outside the statistical merge. That is a limitation GN should avoid by modelling drift inside the likelihood.

### Cited Findings

**(a) Dudok de Wit, Kopp, Fröhlich & Schöll (2017, GRL)**
- The method is a "fully-traceable approach" that "relies on a multi-scale maximum likelihood estimator," uses "data-driven estimates of the uncertainties," and "overcomes the problem of data gaps in a natural way." The composite is closest to PMOD and NRLTSI2. Long-term uncertainties "reveal a 1/f scaling" — [Dudok de Wit et al. 2017, GRL 44, 1196 (arXiv:1702.02341)](https://arxiv.org/abs/1702.02341)
- **Data model.** Each record is true TSI plus an offset plus errors within stated uncertainty, on a regular daily grid. Each record is decomposed with a "pyramidal wavelet transform with Gaussian kernels," which is redundant in time, translation-invariant, and handles scales independently — [Dudok de Wit et al. 2017 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1702.02341)
- **Noise estimation.** Per-instrument precision is the RMS innovation of an AR(5) fit, ranging from about 0.07 W/m² (TIM) to about 0.8 W/m² (ERBE). Residuals against the mean of ACRIM3/VIRGO/TIM have a power spectrum near 1/f. Each instrument's scale-dependent uncertainty σ_k(a,t) extrapolates its precision with a 1/f model — [Dudok de Wit et al. 2017 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1702.02341)
- **Weighting.** I_comp(a,t) = Σ_k I_k(a,t) w_k(a,t) / Σ_k w_k(a,t), with w_k = σ_k⁻²(a,t) when observing. When not observing, w_k = σ_k⁻²(a,t)·exp(−2|T_k(t)|/a), where T_k is the distance to the instrument's nearest observation. This is how the 1986–1991 ACRIM gap is bridged. The inverse wavelet transform returns the composite. Composite uncertainty comes from Monte Carlo over more than 1000 composites — [Dudok de Wit et al. 2017 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1702.02341)
- **Gaps.** Gaps are pre-filled by expectation-maximization and flagged. Wavelet coefficients with no data are dropped within a cone of influence |t − t_gap| < a. Bootstrap uncertainties for gaps up to three months are comparable to the observational uncertainty — [Dudok de Wit et al. 2017 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1702.02341)
- **Degradation and absolute level.** The main composite uses the original records "without any correction or rescaling." A variant applies Fröhlich's (2006) early-mission corrections. The authors argue the statistical method and the scientific choice of corrections should be kept separate. The largest-scale level is set to the TIM/PREMOS average. The composite is "not definitive," and the 1/f scaling could not be checked beyond decadal scales — [Dudok de Wit et al. 2017 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1702.02341)
- **Trend significance.** Only the 1996–2009 decrease is significant at α = 5%. Under a white-noise model all three trends would look significant, which the authors take as evidence that white noise is unrealistic — [Dudok de Wit et al. 2017 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1702.02341)
- Earlier composites were built by instrument PIs and reflected their choices. They switched instruments at transitions, which could cause jumps, and they usually lacked uncertainties. The new method uses a statistically determined weighting of all instruments rather than binary choices. It became the basis for the "community-consensus TSI composite" — [CU Boulder experts record](https://experts.colorado.edu/display/pubid_153164); [HAL: community-consensus TSI composite](https://hal-insu.archives-ouvertes.fr/insu-02089437)

**(a) Montillet et al. (2022, JGR Atmospheres)**
- This is a three-step pipeline: (1) Bayesian data fusion of multiple instruments with Gaussian processes, (2) linking the resulting sub-series into one 41-year record, and (3) wavelet filtering to remove unwanted correlated noise. It includes a stochastic noise model for short- and long-term correlations. An earlier abstract describes a dual kernel of white plus coloured noise — [Montillet et al. 2022, JGR Atmos. 127, e2021JD036146 (ESS Open Archive)](https://essopenarchive.org/users/560943/articles/608699-data-fusion-of-total-solar-irradiance-composite-time-series-using-41-years-of-satellite-measurements); [EGU21 abstract](https://meetingorganizer.copernicus.org/EGU21/EGU21-4382.html); [PMOD/WRC project page](https://www.pmodwrc.ch/?p=23322)
- It is a successor to Fröhlich's PMOD composite. It differs from earlier products by <0.2 W/m² in mean level. The authors conclude that the trend in these composites is mostly an artifact of the coloured noise — [PMOD/WRC](https://www.pmodwrc.ch/?p=23322); [ESS Open Archive](https://essopenarchive.org/doi/full/10.1002/essoar.10508721.2) (as summarized by search; full text not fetched)
- A follow-on machine-learning plus data-fusion methodology for processing TSI observations exists — [ESS Open Archive preprint](https://essopenarchive.org/users/529759/articles/601177-a-new-methodology-to-process-the-total-solar-irradiance-observations-using-machine-learning-and-data-fusion) (not read)

### Inferences

**(c) Mapping to GN**
- The exp(−2|T_k|/a) weight is a clean heuristic for irregular observing days. An observer who has not observed recently still informs long scales but not short ones. In a fully Bayesian GN model this emerges automatically from a latent process with a proper temporal covariance (GP or state-space), which is effectively what Montillet's GP fusion does. NumPyro can represent the GN version directly. The latent log G(t) is a GP or Markov process. Each observer's report is a likelihood conditioned on G(t). Observer error has a white part (daily seeing, counting errors) plus a coloured or 1/f part (slow drift, career changes).
- Per-scale weighting suggests that an observer can be good at one scale and bad at another. A visual observer with a high threshold may track cycle phase well but give a biased amplitude. A model with separate per-observer short-scale noise and long-scale drift terms captures this.
- 1/f noise in observer residuals would mean GN secular uncertainty grows much faster with time scale than a white-noise analysis predicts. A methods paper should estimate the residual spectra of observer-vs-reference residuals in overlaps, as Dudok de Wit did with ACRIM3/VIRGO/TIM, before choosing a drift prior.

**(d) Pitfalls**
- Keeping degradation (drift) correction outside the merge, as Dudok de Wit did, leaves the result dependent on upstream choices. TSI composites still disagree on long-term trends largely for this reason, for example in the ACRIM-gap controversy. GN should model drift jointly, but then it faces the drift-vs-trend degeneracy from Q2.
- The absolute-level reference (TIM/PREMOS) is a choice. For GN, the RGO vs. Debrecen vs. modern reference choice should be propagated as a structural ensemble member.
- Montillet's finding that "trends are mostly an artifact of coloured noise" warns that apparent secular GN changes (for example, a Modern Maximum) need a coloured-noise error model before they are called significant.

**(e) Software**
- Wavelet multiscale fusion is easily implemented in NumPy (e.g., PyWavelets à-trous / stationary wavelet transform).
- GP fusion with white plus coloured kernels is natural in tinygp (JAX) or NumPyro GP. State-space (celerite-like) kernels keep cost linear in the number of days.

### Gaps
- The full Montillet et al. (2022) text was not fetched. Its kernel forms, how sub-series are linked, and how instrument degradation enters the GP model are unverified.
- Post-2022 community-consensus TSI updates (e.g., Kopp et al. or the ISSI team, 2023–2026) were not researched.

---

## Q4. Probabilistic sea-level reconstruction from gappy tide gauges (Hay et al. 2015 Kalman smoother/GP; Dangendorf et al. 2017/2019): why probabilistic reconstruction changed the 20th-century trend

### Takeaway
A sparse, unevenly distributed, gappy network biased the global trend in a specific direction, upward by about 0.4 mm/yr, or about 30%. Hay et al. showed that a probabilistic state-space reconstruction with physically structured priors, which treats each gauge as a noisy local sample of global processes, removes the bias. Dangendorf et al. independently got about 1.1 mm/yr using careful datum unification, vertical-land-motion correction, and area weighting. The GN lesson is that the sampling pattern itself (who observed, when, and where in the activity range) can bias long-term estimates. It must be modelled, not averaged over. Diagnostic experiments that resample the reconstructed truth with the real sampling pattern are the way to show this.

### Cited Findings

**(a) Core idea and math**
- Hay, Morrow, Kopp & Mitrovica (2015) use two methods: Kalman smoothing (KS) and Gaussian-process regression (GPR).
  - **KS state.** Sea level at 622 tide-gauge sites, plus scalar weights for 3 ice sheets, 18 glacier regions, and a uniform thermal-expansion term.
  - **KS dynamics.** Melt weights follow a random walk and map to sea level through precomputed "fingerprints." GIA rates and CMIP5 ocean dynamics are included.
  - **KS observations.** Each gauge is sea level plus white noise.
  - **KS smoothing.** A forward pass, a backward pass, and a smoothing step. Results are likelihood-weighted across 161 Earth-rheology models and 6 climate models.
  - **GPR.** Decadal-mean sea level is a sum of GP-prior components (GIA, land ice, local ocean/tectonics).

  — [Hay et al. 2015 summary (sealeveldocs)](https://sealeveldocs.readthedocs.io/en/latest/hay15.html); citation: Nature 517, 481–484, doi:10.1038/nature14093, erratum Nature 552, 278 (2017), per the same page
- **Result.** 1901–1990 GMSL rate of 1.2 ± 0.2 mm/yr (KS, 90% CI) and 1.1 ± 0.4 (GPR), vs. 1.6–1.9 mm/yr in earlier tide-gauge analyses (Church & White 1.5 ± 0.2; Jevrejeva 1.9). For 1993–2010, 3.0 ± 0.7 mm/yr. Acceleration is 0.017 ± 0.003 mm/yr² vs. Church & White's 0.009 ± 0.002. KS closes the sea-level "budget" gap (sum of contributions about 0.5 mm/yr below Church & White) — [sealeveldocs](https://sealeveldocs.readthedocs.io/en/latest/hay15.html)
- **Diagnostic experiments.**
  - KS with regional binning and the real gauge sampling, missing data included, gives 1.6 ± 0.4 mm/yr. That reproduces the earlier high values from the sampling pattern alone.
  - Binning without missing data gives 1.0 ± 0.4.
  - Sampling the full global KS field gives 1.2 ± 0.1, so binning itself is unbiased.
  - Bootstrapping 25–600 sites levels off about 0.4 mm/yr above the KS value.
  - Applying the Church & White EOF method to the KS field gives 1.4 ± 0.4, an EOF-method bias.

  "Combined spatial sparsity and missing data generate an upward bias in estimates of GMSL rates." KS "extracts global information from the sparse field." — [sealeveldocs](https://sealeveldocs.readthedocs.io/en/latest/hay15.html)
- **Dangendorf et al. (2017, PNAS)** got 1.1 ± 0.3 mm/yr (1σ) before 1990, vs. 1.3–2 in earlier reconstructions, and 3.1 ± 1.4 mm/yr for 1993–2012, consistent with altimetry. They attribute the lower rate "roughly equally" to vertical-land-motion corrections, area weighting, and a common reference datum for tide gauges — [Dangendorf et al. 2017, PNAS 114, 5946 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5468634); [TU Delft record](https://research.tudelft.nl/en/publications/reassessment-of-20th-century-global-mean-sea-level-rise)
- Later work describes Hay et al.'s approach as a Kalman smoother that "assimilates prior information on individual processes and their spatial covariances" — [as quoted in search summary of Dangendorf-related work](https://www.beg.utexas.edu/node/6516)

### Inferences

**(c) Mapping to GN**
- The GN latent state is log G(t), and possibly hemispheric or latitudinal components if group-latitude data exist. The "fingerprints" are each observer's transfer function: the known physics of how a true group population appears to an observer with a given aperture or threshold. The observation model is each observer's count given the state. A Kalman/RTS smoother or GP over log G(t) handles gaps natively and uses every observer's data in proportion to its information. This is the opposite of selecting a backbone.
- **Sampling-bias analog.** Historical observers are not a random sample of days or activity levels. Observers may be more likely to record, or to have surviving records, on active days. Early observers often reported only spotted days. Observer density is also not uniform across eras. Hay et al. show that such non-uniform sampling with gaps biases the trend in a definite direction, and that averaging or binning methods inherit the bias.
  - The GN equivalent of their diagnostic: take the posterior or reference G(t) (or a synthetic truth), impose the real historical observer-day sampling and selection mechanism, run the competing calibration methods, and measure the induced trend and amplitude biases.
- **Dangendorf's three corrections map onto GN:**
  - vertical land motion corresponds to observer drift;
  - area weighting corresponds to weighting observers by how representative they are, not by how many records they have;
  - a common reference datum corresponds to a consistent absolute anchor across all chains.
- Probabilistic and deterministic reanalyses differed by about 30% in the 20th-century rate. A full-Bayesian GN reconstruction could plausibly differ from backbone/ADF series by a comparable fraction in poorly sampled centuries. The methods paper should be prepared to attribute any difference to specific sampling or bridging effects.

**(d) Pitfalls**
- The KS result depends on physically informed priors (fingerprints, GIA models, CMIP ocean dynamics) and on averaging over model ensembles. For GN, the analogous priors are cycle-shape or smoothness priors and the area-distribution model. They inject information into sparse eras, so the posterior there may be prior-dominated and must be flagged. Report prior-vs-posterior contraction by era.
- Assuming white gauge noise can understate uncertainty if real errors are correlated. The same applies to observer errors (see the TSI 1/f result).

**(e) Scalability and software**
- Kalman/RTS smoothing is linear in the number of time steps. In JAX: dynamax (state-space models), or a hand-written lax.scan Kalman filter. With count observations (Poisson/NegBin) the Kalman step is no longer exact, so use NUTS on a non-centred state-space model, an extended or unscented approximation, or particle methods.
- GP options: tinygp/celerite kernels.

### Gaps
- Dangendorf et al. (2019, Nature Climate Change, "Persistent acceleration...") and Frederikse et al. (2020, Nature, sea-level budget since 1900) are canonical 2019–2020 follow-ups. I did not fetch them this session, so their methods (hybrid reconstruction, ensemble budget) are not documented here.
- The rate numbers come from a documentation page that summarizes the Nature paper (sealeveldocs), not the Nature PDF itself. The summary is detailed and internally consistent, but the primary source was not opened.

---

## Q5. Climate station homogenization (NOAA PHA, Berkeley Earth scalpel + kriging, HOME/HOMER/ACMANT benchmarks, Bayesian changepoints): neighbor-based breakpoint detection and benchmarking with synthetic truth

### Takeaway
Station homogenization is the most mature field for "unknown, undocumented changes in many overlapping noisy instruments," and it has a benchmarking culture that a GN methods paper should copy directly:
- blind tests on synthetic networks with known inserted breaks, drifts, outliers, and gaps;
- evaluation at several averaging scales and on trend error;
- perturbed-parameter ensembles of the algorithm itself to express structural uncertainty.

On detection, breaks are found in difference series against many neighbours and attributed to the station common to the flagged pairs (PHA). Alternatively, records are cut at breaks into independent fragments and recombined in a statistical field model (Berkeley Earth). Methods designed to work with an inhomogeneous reference perform best (HOME). At low signal-to-noise, break detection is unreliable (Lindau & Venema), and synchronous breaks across stations defeat even the best methods (MULTITEST).

### Cited Findings

**(a) NOAA Pairwise Homogenization Algorithm (PHA)**
- PHA builds pairwise difference series between a target and neighbouring stations' monthly data. It tests each difference series for undocumented shifts and automatically identifies the station responsible. It uses station-history metadata when available and adjusts only when the shift size can be estimated reliably. It was adopted for USHCN version 2 in 2009 — [Menne & Williams 2009, J. Climate (NCEI PDF)](https://ncei.noaa.gov/pub/data/ushcn/papers/menne-williams2009.pdf) (abstract-level summary via search; full text not parsed)
- Pairwise methods have three steps: break detection, attribution to a station, and sizing the break by comparison with neighbours — [ASCMO 2018 (Lindau & Venema)](https://ascmo.copernicus.org/articles/4/1/2018/)
- **Benchmarking PHA.** Williams, Menne & Thorne (2012, JGR 117, D05116) ran a randomized, perturbed ensemble of the USHCN PHA against a suite of synthetic benchmark analogs of real monthly data. Homogenized data are generally closer to truth than raw data. Tmax trends may still be underestimated. For Tmin, the ensemble error bounds include the unhomogenized trends — [Williams et al. 2012 (Maynooth)](https://mural.maynoothuniversity.ie/6495); [PDF](https://mural.maynoothuniversity.ie/6495/1/PT_benchmarking.pdf)
- Hausfather et al. (2016, GRL) found that USHCN adjustments brought trends closer to the nearby reference network USCRN for 2004–2015 — [as cited in Williams et al. summary (Maynooth record / search)](https://mural.maynoothuniversity.ie/6495) (secondary)

**(a) Berkeley Earth "scalpel" + kriging**
- The Berkeley Earth averaging process is a kriging-based framework that accepts short and discontinuous records, so nearly all archived thermometer data can be used — [Rohde et al. 2013, Geoinfor. Geostat. 1(2) (DOI 10.4172/2327-4581.1000103)](https://0-doi-org.brum.beds.ac.uk/10.4172/2327-4581.1000103)
- The workflow detects breakpoints through neighbour difference-series comparisons, cuts stations into fragments at breakpoints (the "scalpel"), and combines the fragments into a temperature field. The weighting reflects correlation with other stations and station density — [Rohde & Hausfather 2020, ESSD 12, 3469](https://essd.copernicus.org/articles/12/3469/2020/); [Physics Today](https://physicstoday.aip.org/news/earths-land-surface-temperature-trends-a-new-approach-confirms-previous-results). Outliers are reportedly down-weighted rather than discarded (secondary forum source; not verified).

**(a) Bayesian changepoint methods**
- **Hannart & Naveau.** Bayesian decision theory with renewal-process priors on jump frequency and size. Segments contain at most one jump, and calculations are explicit, without MCMC. It was applied to Marseille temperature homogenization — [Hannart & Naveau, Technometrics](https://www.lsce.ipsl.fr/wp-content/uploads/2020/12/HannartNaveauTechnometrics.pdf); [EGU 2009](https://meetingorganizer.copernicus.org/EGU2009/EGU2009-10930.pdf)
- **Hannart, Mestre & Naveau.** Extends this to pairwise comparisons with neighbours. Posterior distributions quantify uncertainty in shift position and amplitude, and performance is similar to manually reviewed pairwise methods — [EGU 2009](https://meetingorganizer.copernicus.org/EGU2009/EGU2009-10948.pdf)
- **Li, Lund & Hewaarachchi.** A minimum-description-length method with an empirical-Bayes prior that makes documented times (metadata) more likely to be breaks. It substantially improves detection power — [arXiv:1511.07238](https://ar5iv.labs.arxiv.org/html/1511.07238)
- A Journal of Climate review covers good practices and common pitfalls in climate changepoint techniques — [NSF PAR record](https://par.nsf.gov/biblio/10473093-good-practices-common-pitfalls-climate-time-series-changepoint-techniques-review) (full text not read)

**(d) Limits of break detection**
- **Lindau & Venema (2018).** At low but realistic signal-to-noise ratios, even a random segmentation explains about half the break variance, so reliable break detection needs new approaches — [Lindau & Venema 2018, ASCMO 4, 1](https://ascmo.copernicus.org/articles/4/1/2018/)

**Benchmarking with synthetic truth**
- **HOME (COST Action ES0601; Venema et al. 2012).** A blind intercomparison on monthly temperature and precipitation. It received 25 blind contributions plus 22 after the inserted inhomogeneities were revealed. The benchmark combined real inhomogeneous data with surrogate and synthetic networks containing breaks (including breaks shared by several stations), outliers, missing data, and local trends.
  - **Metrics:** (i) centred RMSE against truth at several averaging scales, (ii) linear-trend error, (iii) contingency skill scores for break detection.
  - **Findings:** Relative methods improved temperature, but only the best improved precipitation. The best performers were designed for an inhomogeneous reference. Automatic methods matched manual ones.

  — [Venema et al. 2012, Clim. Past 8, 89 (PDF)](https://opac.geologie.ac.at/ais312/dokumente/copernicus.org/cp-8-89-2012.pdf); [CROSBI record](https://www.bib.irb.hr/580690)
- **Killick (2016) daily benchmark.** The first intercomparison at the daily level. Synthetic daily temperature is built from a Generalised Additive Model fitted to real data, for 4 US regions with up to 4 inhomogeneity scenarios each, over 1970–2011. It is distributed by the Met Office — [Killick 2016, Exeter thesis](https://ore.exeter.ac.uk/repository/handle/10871/23095); [Met Office benchmarks](https://www.metoffice.gov.uk/hadobs/benchmarks/)
- **MULTITEST (Spain).** Repeatedly resampled synthetic homogeneous networks of 100 monthly series, inserted inhomogeneities, and ran many packages. ACMANT was likely the most accurate, but ACMANTv4 performed poorly with synchronous breaks. ACMANTv5 added combined comparisons to handle them, yet on the Killick daily data its residual errors could exceed v4's. Daily benchmarking has lower signal-to-noise, so its results are not directly comparable with monthly rankings — [Domonkos, MDPI Climate 11, 224](https://www.mdpi.com/2225-1154/11/11/224); [preprint](https://www.preprints.org/manuscript/202309.1895)

### Inferences

**(c) Mapping to GN: detection and attribution**
- An observer's career change (new telescope, new counting convention, eyesight decline, assistant taking over) is a station break. PHA's attribution logic transfers directly. Form the ratio or log-difference series of observer o against every overlapping observer (or against the latent G(t)). Flag changepoints, and attribute a break to o when it appears in many of o's pairs but not in the partners' other pairs. Sparse historical overlaps make this weak before about 1850, which is exactly where Lindau & Venema's low-SNR warning applies.
- The Berkeley "scalpel" is attractive for GN. Instead of fitting a drift model, split each observer at detected or documented breaks into independent "pseudo-observers," each with its own transfer parameters, and let the joint model recombine them. This replaces an assumed drift form with more nuisance parameters.
  - The cost is that each cut weakens the overlap graph: it adds a node and splits edges, raising effective resistance (Q1).
  - Over-cutting therefore lets real long-term change leak into nuisance parameters. A long career cut into pieces can no longer anchor the secular trend. This is the drift-vs-trend trade-off in another form.
- **Bayesian changepoints in NumPyro.** Discrete changepoint locations are not NUTS-friendly. Options:
  - a random-walk or horseshoe-prior piecewise-constant drift for each observer (continuous relaxation, with sparse increments);
  - marginalizing discrete change locations on a coarse grid, for example yearly (enumeration in NumPyro);
  - the Li-Lund-Hewaarachchi idea of raising prior break probability at documented events such as instrument changes and known career milestones.
- HOME's finding that methods "designed to work with an inhomogeneous reference" do best is the station-network equivalent of global self-calibration: never assume the reference (neighbour or RGO) is clean.

**Benchmark design for the GN methods paper (synthesis)**
1. **Truth.** Generate daily true group populations, with group areas and lifetimes, from RGO/Debrecen group-level data, or from a stochastic emergence model with realistic cycles, including grand-minimum-like episodes.
2. **Observers.** Assign each synthetic observer an area or visibility detection function (threshold, steepness), a per-day "seeing" noise, and a convention (for example, splitting or merging groups). Include drift (linear, random walk) and abrupt breaks at known times, some documented as "metadata" and some not.
3. **Sampling.** Impose the real historical observer-day calendar and overlap topology, and optionally selective reporting (for example, spotless days under-reported), so the benchmark has the real graph bottlenecks. This is the Hay et al. resampling diagnostic.
4. **Blind protocol.** Have the competing methods (backbone, ADF, shortest-path, full Bayesian) run blind, as in HOME. Release the truth afterwards.
5. **Metrics (HOME-style, extended).**
   - centred RMSE at daily, monthly, yearly, and cycle-averaged scales;
   - errors in cycle amplitude and cycle-to-cycle ratio;
   - secular trend error over 1700–present and 1850–present;
   - break-detection skill;
   - calibration of posterior credible intervals: empirical coverage of the 68%/95% intervals of the truth across many synthetic worlds;
   - sensitivity to reference choice.
6. **Structural ensemble (Williams et al. 2012 style).** Run perturbed-parameter ensembles of each method's free choices, such as overlap thresholds, minimum days, drift prior scale, and reference years. Report the spread alongside the parametric posterior.
7. **Stress tests (MULTITEST lesson).** Synchronous breaks across many observers, for example era-wide adoption of better telescopes, should be included explicitly. They are the case where pairwise methods fail and where common-mode drift is confounded with real trend.

**(d) Pitfalls**
- Low SNR in daily integer counts: Poisson noise at low activity plus irregular days.
- Synchronous or common-mode breaks.
- Over-segmentation removing real signal.
- Attribution ambiguity when an observer has few partners. Leaf nodes in the overlap graph cannot be attributed by counting.
- Benchmarks that reuse the same generative model as the inference method will overstate performance. The truth generator should differ from the fitted observation model, as Holmes et al. deliberately did with a wrong noise model and basis (Q1).

**(e) Scalability and software**
- PHA and Berkeley Earth operate on tens of thousands of stations, so GN's 600 observers are small.
- Known packages: NOAA PHA (Fortran), ACMANT, HOMER (R), Climatol (R), and Berkeley Earth's Matlab code. Locations and licences were not verified this session.
- For GN in JAX, the pairwise-detection stage can be vectorized over observer pairs with vmap. The joint model's cost is dominated by the latent G(t) (Q1 and Q4).

### Gaps
- I could not parse the Menne & Williams (2009) full text: the PDF extraction failed because the local Python environment is broken. Specifics are therefore not verified here: neighbour count and correlation criteria, the SNHT/two-phase-regression and BIC model-selection details, the "unconfounding" attribution rule, and the median-of-pairs adjustment with confidence intervals. These details are well known from the paper but need checking in the primary source.
- The HOMER (Mestre et al. 2013) and ACMANT original papers, and ISTI's global benchmarking framework (Willett et al. 2014), were not fetched.
- I found no fully Bayesian MCMC (e.g., reversible-jump) multi-station homogenization paper within budget. The Bayesian methods found are decision-theoretic or empirical-Bayes rather than full posterior samplers.
- The Berkeley Earth "scalpel" terminology and outlier down-weighting are supported here only by secondary sources. The original Rohde et al. (2013) methods text was not read.
