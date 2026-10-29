# Step 4 — Joint latent-population model (Approach 1)

## Goal
The core model and the **sunspot-only GN series v0**: a posterior ensemble of daily reference-equivalent group counts,
1610–present, from one generative model of the Sun and every observer.

## Design (from the assessment; settings from Step 1)
- **Activity layer:** log λ(t) as a non-centred, quasi-periodic state-space Gaussian process with jittered period and a
  prior that allows grand minima (from sunspot-era data/physics only — firewall, ADR-002).
- **Population layer:** N_t (reference-equivalent count): survivors ~ Bin(N_{t−1}, φ), births ~ Pois(λ_t); one or two
  area classes per Step 1; truncation K per Step 1 — **K comes from a robust upper quantile of reported
  counts, never from `max(GROUPS)`** (issue D17: v1.3's two largest values, 93 and 58, are implausible
  outliers above a tail that stops at 53, and taking 93 at face value would inflate the state space by
  ~60% for two rows; the 99.99th percentile is below 40). Transition factorizes as A_t = S·B_t (two
  matrix–vector products per day; never store T×K×K).
- **Detection layer:** emission tables P(y | N, θ_o) built from real reference areas with a soft threshold, splitting
  (allows y > N), career drift; hierarchical observer population with optional metadata covariates (ADR-004).
- **Reporting layer:** MNAR spotless-day recording; blanket statements as Π_t [1 − q_o(1 − P(y=0 | N_t))] per statement
  (an extra per-day emission factor) with a literal/rhetorical mixture weight.
- **Anchor:** reference epoch with fixed detection (ADR-003, **as amended by ADR-009: two co-equal
  anchors**). Fit and release the posterior ensemble under **both** anchorings — Debrecen cycles
  21–23 and RGO cycles 15–19 — since the difference between them is the anchor-sensitivity result
  ADR-003 requires, not an optional robustness check. Record which anchoring each run used.
- **Inference:** exact forward filter in `jax.lax.scan` with float64 log-likelihood accumulation; emissions from long-format
  records via `segment_sum`; FFBS draws = ensemble; production sampler = blocked Gibbs (FFBS ↔ vectorized NUTS/MALA per
  observer ↔ λ update ↔ hyperparameters); tempered SMC for marginal likelihoods / model comparison; SVI only for
  development. Initialize from Step 3.

## Early tasks (before the full build)
1. Benchmark the forward filter on CPU and GPU at full length (the research cost figures — ~5×10⁸ flops, ~25 MB, ~1 s
   per evaluation — are unverified estimates).
2. Check whether `dynamax` parallel filtering supports time-varying transitions; otherwise a ~30-line associative operator.
3. Reproduce a Step 1 Check A fit with the production filter.

## Deliverables
`gn_next_gen.models.joint`, `gn_next_gen.inference` (filter, FFBS, samplers), PPC suite (ADF, daily count distributions,
run lengths per observer), benchmark scores vs Steps 2–3, real-data run with convergence diagnostics, the ensemble
(written to `outputs/`, summary tables committed), drift-prior and MNAR/blanket-statement sensitivity analyses,
results doc and paper figures.

## Acceptance criteria (quantify in the plan)
- Benchmark: recovers truth with nominal coverage at daily→cycle scales; beats Step 3 and legacy baselines on stated metrics.
- Per-observer PPCs: list of observers failing (input to Step 5).
- Convergence: split-R̂ and ESS thresholds on yearly GN and key hyperparameters.
- Runtime and memory within a stated budget on the target GPU.
- Era-wide drift sensitivity and per-year prior→posterior contraction reported.

## Open questions for the planning session
1. Lifetime model (geometric per class vs age-structured) given Step 1.
2. Splitting model form.
3. Grand-minimum prior form without proxy information.
4. Output format for the ensemble (netCDF/zarr, number of draws, aggregation levels).

## Required reading
`CLAUDE.md`; assessment §4 (Approach 1), §6; ADR-001, -002, -003, -004; report sections "Ecology already wrote the GN
likelihood…", "Epidemiology's under-reporting models…", "Spotless days are a selection problem…", "Absence of evidence
is data…", "A 41-state forward filter…", "Samplers…", ranked ideas 2, 3, 5, 6, 8; notes
`sbi_and_scalable_inference.md`, `ecology_imperfect_detection.md`, `counting_processes_underreporting.md`,
`paleoclimate_historical_evidence.md` (blanket statements); Step 1–3 results docs.

## Out of scope
Proxies (Step 6); neural emission models (Step 5).
