# `area_law_2026-11/` — the group-area law: Weibull + log-normal, per survey

Frozen summaries of `sunspot-homogenization`'s per-survey fits of the **group-area
distribution**. Fixed by **ADR-011**. This project **does not refit them**.

## The law

The area of a sunspot group is modelled on **log-area** `z = ln A` as a two-component mixture:

    p(z) = pi_W * Weibull_k,lambda(z)  +  (1 - pi_W) * Normal(z; mu, sigma)

where the Weibull's density in log-area is

    log p_W(z) = log k - k*log(lambda) + k*z - exp(k*(z - log lambda))

a left-skewed Gumbel. The Weibull captures **small objects** (pores and ephemeral
regions, a fragmentation regime); the log-normal captures **large objects** (bipolar
active regions, multiplicative growth).

**This is not a two-log-normal mixture.** Fitting two log-normals folds observational
effects — resolution, detection bias, cadence-vs-lifetime convolution — into the
small-object component. Upstream's paper adopts Weibull+log-normal because it is the
form Muñoz-Jaramillo et al. (2015, ApJ 804:68) found to best reconcile eleven
databases, and names the two-log-normal form as the alternative it rejected.

## ⚠ Two mixing-weight conventions

| source | weights | relation |
|---|---|---|
| upstream, and this file | `pi_W` weights the **Weibull** | — |
| Muñoz-Jaramillo et al. (2015) | `c` weights the **log-normal** | **`c = 1 - pi_W`** |

Getting this backwards silently swaps the two populations. The 2015 paper's published
activity-level power law `c(AL) = 0.66*AL^0.230` is on the **log-normal** weight.

## Truncation

Each survey is hard-truncated at its own completeness threshold before fitting — the
recipe of the 2015 paper §2.4 (an order of magnitude above the smallest detected
structure), implemented upstream as `threshold_method = "lifetime_divergence"` with
`trunc_factor = 10`. Those thresholds are the `threshold_MSH` column of
`../reference_groups_2026-10/survey_calibration.csv`, and the fit uses **one area per
group** (`group_single_entry.parquet`, column `area_corr`, strictly greater than the
threshold). The kept counts reproduce exactly:

| survey | threshold (MSH) | `n_kept` |
|---|---|---|
| rgo | 28.084 | 16,880 |
| debrecen | 20.424 | 8,306 |
| kmas | 23.415 | 15,662 |

Schwabe is calibrated on the **umbral** scale, so its threshold applies to a different
area column and does not reproduce against `area_corr`; Spörer differs by 13 groups.
Neither is used in Step 1 (issue **D23**).

## Activity-level dependence

The 2015 paper finds the **Weibull component independent of activity level** — the
"magnetic baseline" — while the **log-normal component varies strongly** with it. The
fits frozen here are pooled over all activity levels; Step 1 holds `(k, lambda)` fixed
at these values and fits the log-normal's activity dependence on top. See ADR-011 and
`docs/results/step-1-data-foundation.md`.

## Files

| File | What |
|---|---|
| `solo_wln_posterior.csv` | one row per survey: parameter medians and 16/84
  percentiles, truncation, kept counts, evidence |
| `provenance.json` | upstream commit, source paths, checksums, lineage |

Produced by `scripts/export_area_law.py`, which pins the upstream commit and verifies
byte reproducibility with `--check`.

## Citation

`sunspot-homogenization` (MIT, same PI) for the calibration; Muñoz-Jaramillo et al.
(2015, ApJ 804:68) for the Weibull+log-normal form and its activity-level dependence.
