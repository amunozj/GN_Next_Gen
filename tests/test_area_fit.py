"""Fitting the area law's activity dependence, in the 2015 paper's two stages."""

from __future__ import annotations

import numpy as np
import pytest

from gn_next_gen.reference.area_fit import (
    activity_bins,
    fit_bin,
    fit_power_law,
)
from gn_next_gen.reference.area_law import (
    AreaLawParams,
    TruncatedLikelihood,
    log_pdf_latent,
    log_pdf_truncated,
    z_grid,
)

TRUTH = AreaLawParams(
    survey="synthetic",
    k_W=0.45,
    log_lambda_W=float(np.log(5.7)),
    mu_LN=5.47,
    sigma_LN=0.89,
    pi_W=0.70,
    sigma_floor=0.05,
    threshold_msh=28.08,
)


def sample_law(params: AreaLawParams, n: int, seed: int = 0) -> np.ndarray:
    """Draw log-areas from the latent law by inverting its CDF on the grid."""
    rng = np.random.default_rng(seed)
    grid = z_grid(4096)
    density = np.exp(
        log_pdf_latent(
            grid, params.k_W, params.log_lambda_W, params.mu_LN, params.sigma_LN, params.pi_W
        )
    )
    cdf = np.cumsum(density)
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, grid)


# --- the precompiled likelihood -------------------------------------------


def test_precompiled_likelihood_matches_the_reference_implementation():
    """The optimisation that made these fits affordable must change no arithmetic."""
    values = sample_law(TRUTH, 800, seed=3)
    kept = values[values > np.log(TRUTH.threshold_msh)]
    sigma = np.full(kept.shape, 0.03)
    y_trunc = float(np.log(TRUTH.threshold_msh))

    reference = float(log_pdf_truncated(kept, TRUTH, sigma_row=sigma, y_trunc=y_trunc).sum())
    compiled = TruncatedLikelihood(kept, sigma, y_trunc, sigma_floor=TRUTH.sigma_floor)(TRUTH)
    assert compiled == pytest.approx(reference, abs=1e-9)


def test_likelihood_peaks_near_the_truth():
    values = sample_law(TRUTH, 4000, seed=5)
    kept = values[values > np.log(TRUTH.threshold_msh)]
    likelihood = TruncatedLikelihood(
        kept,
        np.zeros_like(kept),
        float(np.log(TRUTH.threshold_msh)),
        sigma_floor=TRUTH.sigma_floor,
    )
    best = likelihood(TRUTH)
    for wrong in (
        TRUTH.replace(mu_LN=TRUTH.mu_LN + 0.5),
        TRUTH.replace(sigma_LN=TRUTH.sigma_LN * 1.5),
    ):
        assert likelihood(wrong) < best


# --- binning and power laws -----------------------------------------------


def test_equal_count_bins_are_balanced():
    """Activity is right-skewed; equal-width bins would starve the top ones."""
    rng = np.random.default_rng(0)
    activity = rng.lognormal(-2.0, 0.8, 5000)
    index, edges = activity_bins(activity, 10, equal_count=True)
    counts = np.bincount(index[index >= 0], minlength=10)
    assert counts.min() > 0.8 * counts.max()
    assert len(edges) == 11
    assert np.isneginf(edges[0]) and np.isposinf(edges[-1])

    wide_index, _ = activity_bins(activity, 10, equal_count=False)
    wide_counts = np.bincount(wide_index[wide_index >= 0], minlength=10)
    assert wide_counts.min() < 0.2 * wide_counts.max(), (
        "equal-width binning should be visibly unbalanced on this distribution"
    )


def test_power_law_recovers_known_coefficients():
    activity = np.linspace(0.05, 0.5, 20)
    values = 5.97 * activity**0.021
    law = fit_power_law(activity, values, "mu_LN", published=(5.97, 0.021))
    assert law.a == pytest.approx(5.97, rel=1e-6)
    assert law.b == pytest.approx(0.021, abs=1e-6)
    assert law.spearman == pytest.approx(1.0)
    row = law.as_row()
    assert row["published_a"] == 5.97
    assert row["a_ratio"] == pytest.approx(1.0, rel=1e-6)


def test_power_law_reports_no_correlation_when_there_is_none():
    rng = np.random.default_rng(1)
    activity = np.linspace(0.05, 0.5, 30)
    law = fit_power_law(activity, rng.uniform(0.9, 1.1, 30), "k_W")
    assert abs(law.spearman) < 0.6
    assert abs(law.b) < 0.5


def test_power_law_survives_too_few_points():
    law = fit_power_law([0.1], [5.0], "mu_LN")
    assert np.isnan(law.a) and np.isnan(law.b)


# --- the fitter -----------------------------------------------------------


def test_fixed_weibull_fit_leaves_the_weibull_alone():
    """Stage 2 must not move (k, lambda) -- that is the whole point of fixing them."""
    values = sample_law(TRUTH, 3000, seed=7)
    kept = values[values > np.log(TRUTH.threshold_msh)]
    fitted, _, _ = fit_bin(kept, np.zeros_like(kept), TRUTH, free_weibull=False)
    assert fitted.k_W == TRUTH.k_W
    assert fitted.log_lambda_W == TRUTH.log_lambda_W


def test_free_fit_moves_all_five():
    values = sample_law(TRUTH, 3000, seed=9)
    kept = values[values > np.log(TRUTH.threshold_msh)]
    start = TRUTH.replace(k_W=0.8, mu_LN=5.0, sigma_LN=1.2)
    fitted, loglik, _ = fit_bin(kept, np.zeros_like(kept), start, free_weibull=True, n_restarts=2)
    assert fitted.k_W != start.k_W
    assert np.isfinite(loglik)


@pytest.mark.slow
def test_fit_recovers_the_lognormal_from_synthetic_data():
    """Synthetic-recovery test for the Check B estimator, as CLAUDE.md requires.

    Only the log-normal parameters are recoverable above the truncation: most of the
    Weibull's mass is below the cut, which is itself a finding and the reason stage 2
    holds it fixed.
    """
    values = sample_law(TRUTH, 40_000, seed=11)
    kept = values[values > np.log(TRUTH.threshold_msh)]
    assert len(kept) > 5000

    start = TRUTH.replace(mu_LN=5.1, sigma_LN=1.1, pi_W=0.5)
    fitted, _, _ = fit_bin(kept, np.zeros_like(kept), start, free_weibull=False)
    assert fitted.mu_LN == pytest.approx(TRUTH.mu_LN, abs=0.12)
    assert fitted.sigma_LN == pytest.approx(TRUTH.sigma_LN, abs=0.12)
    assert fitted.c_LN == pytest.approx(TRUTH.c_LN, abs=0.15)
