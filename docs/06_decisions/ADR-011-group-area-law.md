# ADR-011: The group-area law is Weibull + log-normal, calibrated upstream and not refitted; its log-normal component depends on activity level

- Status: Accepted (2026-11-21, PI)
- Context: Step 1's Check B characterized the reference's area distribution by fitting
  **two log-normal components** by EM and comparing them with BIC, reporting ΔBIC
  +965…+4,268 in favour of two populations
  ([step-1 results](../results/step-1-data-foundation.md)). The two-population finding
  is right; the parameterization is not. **Fitting two log-normals folds observational
  effects into the small-object component** — resolution, detection bias, the
  convolution of instrumental cadence with feature lifetime, and underestimation in
  poor seeing all distort the distribution near the detection limit, and a second
  log-normal absorbs them as if they were solar.

  `/d0/amunozj/git_repos/sunspot-homogenization` (MIT, same PI) already models and
  calibrates this correctly, following Muñoz-Jaramillo et al. (2015, ApJ 804:68): a
  **Weibull** component for the small objects (pores and ephemeral regions, a
  fragmentation regime) plus a **log-normal** for the large ones (bipolar active
  regions, multiplicative growth). Its paper states the choice and names the
  two-log-normal form as the alternative it rejected (`paper/ms.tex` l. 199, 207).
  That calibration is mature.

  The 2015 paper adds the structure that matters for this project: binned by activity
  level, the **Weibull component does not move** — the magnetic baseline — while the
  **log-normal's parameters and the mixing weight follow power laws in activity level**.

- Decision:

  1. **The group-area law is a Weibull + log-normal mixture on log-area** `z = ln A`:

         p(z) = pi_W · Weibull_{k,λ}(z) + (1 − pi_W) · Normal(z; μ, σ)

     with `log p_W(z) = log k − k·log λ + k·z − exp(k·(z − log λ))`. Two log-normals
     are not used anywhere in this project.
  2. **Upstream's per-survey calibration is adopted as-is and never refitted here.**
     It is frozen in `data/raw/area_law_2026-11/` with its provenance. The fit is **one
     area per group** (`group_single_entry`, column `area_corr`), hard-truncated at each
     survey's own `threshold_MSH` — upstream's `lifetime_divergence` threshold under the
     10× rule of the 2015 paper §2.4. Those kept counts reproduce exactly here
     (RGO 16,880, Debrecen 8,306, KMAS 15,662), which is the check that the grain, the
     column and the threshold all match.
  3. **`(k, λ) are held fixed** at upstream's per-survey values when fitting activity
     dependence, and only `μ`, `σ` and the weight are free — the 2015 paper's §7
     procedure. Holding them fixed is licensed by a test, not assumed: Step 1 first
     fits all five parameters per activity bin and checks that the Weibull ones do not
     correlate with activity. **If they do, the fixed-Weibull stage does not run.**
  4. **Activity level is whole-Sun**, a six-month (182-day) Gaussian smoothing of daily
     sunspot area, in mHem. The paper's hemispheric version is computed as a
     **diagnostic only**: hemisphere assignment is positional, and ADR-004 bars
     positions as model inputs, so a hemispheric covariate could not feed Step 4.
  5. **Two weight conventions are kept distinct.** `pi_W` weights the *Weibull*, as
     upstream does; the 2015 paper's `c` weights the *log-normal*, so `c = 1 − pi_W`.
     Its published `c(AL) = 0.66·AL^0.230` is on the log-normal weight.
  6. **Two activity weightings are kept distinct.** The paper's is *emergence-weighted*
     — one peak area per group on its canonical day — and is the default, because the
     published coefficients attach to it. Summing every group present on each day gives
     the *instantaneous* disk area, about six times larger, which is the natural
     covariate for the per-group-day law. Mixing them silently rescales every power law.

- Alternatives considered:
  - *Keep the two-log-normal mixture* — rejected by the PI as authoritative error: it
    attributes observational effects to a solar population.
  - *Refit the Weibull + log-normal law here from scratch* — unnecessary and worse:
    upstream's fit is mature, uses nested sampling with per-object measurement error and
    a principled truncation, and is the same calibration the reference snapshot already
    rests on.
  - *Adopt the 2015 published coefficients verbatim with no fitting* — they were
    calibrated on a different composite (RGO×1.06 matched to KMAS) and a different area
    scale, so an offset would propagate silently. They are a cross-check instead.
  - *Use hemispheric activity level everywhere, as the paper does* — more faithful, but
    it puts positional information into the model and would need an ADR-004 amendment.

- Consequences:
  - Step 1's two-log-normal result is **retracted** in the results doc, with its reason,
    rather than quietly replaced.
  - **Step 4's detection layer gains an activity-dependent area law.** The effective
    detection fraction `q_o(t) = E_{F(·|AL(t))}[π_o(a)]` now varies with the cycle in a
    calibrated way, which is the standing explanation for why ADF-style k-factors depend
    on activity (research report, "Mark-dependent thinning explains why k-factors depend
    on activity"). It also means an observer's k-factor is **not** a constant, and the
    model must not be built as though it were.
  - The per-survey truncation thresholds (20.4–28.1 MSH for Debrecen, KMAS and RGO) are
    the surveys' own completeness limits, so **counts at T = 0 and T = 10 sample a
    regime upstream deliberately truncates away**. Recorded against **D14**.
  - Check A's two area classes are re-grounded on the Weibull/log-normal crossover
    rather than a round number.
  - `data/raw` gains one small folder; `data/raw/reference_groups_2026-10/` is unchanged.
