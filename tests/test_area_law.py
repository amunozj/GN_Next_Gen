"""The group-area law: Weibull + log-normal, and its activity-level machinery.

The anchoring test here is :func:`test_upstream_kept_counts_reproduce`. Upstream
fitted this law on one area per group, hard-truncated at each survey's own
threshold; if our grain, column or threshold differed, that count would differ and
nothing built on the law would be trustworthy. It reproduces exactly, which is what
licenses using upstream's posterior without refitting it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, weibull_min

from gn_next_gen.io import load_day_status, load_group_single_entry
from gn_next_gen.reference.area_law import (
    ACTIVITY_SIGMA_DAYS,
    MUHEM_PER_MHEM,
    PUBLISHED_2015,
    AreaLawParams,
    activity_level,
    component_crossover,
    fraction_above,
    gaussian_smooth,
    load_upstream_parameters,
    log_pdf_latent,
    log_pdf_truncated,
    log_pdf_weibull,
    lognormal_weight,
    published_power_law,
    z_grid,
)

STEP1 = ("rgo", "debrecen", "kmas")

#: Upstream's kept counts, from its own fit metadata.
UPSTREAM_KEPT = {"rgo": 16_880, "debrecen": 8_306, "kmas": 15_662}


@pytest.fixture(scope="module")
def params():
    return {survey: load_upstream_parameters(survey) for survey in STEP1}


# --- the anchor -----------------------------------------------------------


def test_upstream_kept_counts_reproduce(params):
    """One area per group, `area_corr`, strictly above the survey's threshold.

    Exact agreement fixes all three choices at once. D22's colliding KMAS ids are
    kept here because upstream's fit did not drop them.
    """
    groups = load_group_single_entry(STEP1, drop_kmas_collisions=False)
    for survey, expected in UPSTREAM_KEPT.items():
        subset = groups[groups["survey"] == survey]
        kept = int((subset["area_corr"] > params[survey].threshold_msh).sum())
        assert kept == expected, f"{survey}: {kept} kept, upstream fitted {expected}"
        assert params[survey].n_kept == expected


def test_thresholds_come_from_the_frozen_calibration(params):
    calibration = pd.read_csv("data/raw/reference_groups_2026-10/survey_calibration.csv").set_index(
        "survey"
    )
    for survey in STEP1:
        assert params[survey].threshold_msh == pytest.approx(
            float(calibration.loc[survey, "threshold_MSH"])
        )


# --- the law --------------------------------------------------------------


def test_latent_density_integrates_to_one(params):
    grid = z_grid(4096)
    for survey in STEP1:
        p = params[survey]
        density = np.exp(log_pdf_latent(grid, p.k_W, p.log_lambda_W, p.mu_LN, p.sigma_LN, p.pi_W))
        # The grid truncates a little Weibull mass below exp(-10) MSH; upstream uses
        # the same grid, so matching it matters more than chasing the last 0.4%.
        assert np.trapezoid(density, grid) == pytest.approx(1.0, abs=5e-3)


def test_weibull_in_log_area_matches_scipy():
    """The pushforward of a Weibull on linear area to z = log A."""
    k, lam = 0.45, 5.7
    z = np.linspace(-4, 6, 400)
    ours = np.exp(log_pdf_weibull(z, k, np.log(lam)))
    # p_Z(z) = p_A(e^z) * e^z
    theirs = weibull_min.pdf(np.exp(z), k, scale=lam) * np.exp(z)
    assert np.allclose(ours, theirs, rtol=1e-8, atol=1e-12)


def test_mixture_reduces_to_each_component():
    z = np.linspace(-2, 8, 200)
    k, log_lam, mu, sigma = 0.45, np.log(5.7), 5.5, 0.9
    pure_ln = np.exp(log_pdf_latent(z, k, log_lam, mu, sigma, 1e-12))
    assert np.allclose(pure_ln, norm.pdf(z, mu, sigma), rtol=1e-6)
    pure_w = np.exp(log_pdf_latent(z, k, log_lam, mu, sigma, 1 - 1e-12))
    assert np.allclose(pure_w, np.exp(log_pdf_weibull(z, k, log_lam)), rtol=1e-6)


def test_it_is_not_two_lognormals(params):
    """The small-object component must be Weibull-shaped, not Gaussian in log-area.

    A regression guard on the correction this revision exists for: the Weibull's
    density in z is a left-skewed Gumbel, so it is measurably asymmetric where a
    log-normal would be symmetric.
    """
    p = params["rgo"]
    grid = z_grid(4096)
    density = np.exp(log_pdf_weibull(grid, p.k_W, p.log_lambda_W))
    mass = np.trapezoid(density, grid)
    mean = np.trapezoid(grid * density, grid) / mass
    var = np.trapezoid((grid - mean) ** 2 * density, grid) / mass
    skew = np.trapezoid((grid - mean) ** 3 * density, grid) / mass / var**1.5
    assert skew < -0.5, f"log-area Weibull must be left-skewed, got {skew:.3f}"


def test_weight_conventions_are_complementary(params):
    """pi_W weights the Weibull; the paper's c weights the log-normal."""
    for survey in STEP1:
        p = params[survey]
        assert p.c_LN == pytest.approx(1.0 - p.pi_W)
        assert lognormal_weight(p.pi_W) == pytest.approx(p.c_LN)
    # Upstream is Weibull-heavy, the paper log-normal-heavier; that gap is real and
    # is reported rather than reconciled.
    assert all(params[s].pi_W > 0.5 for s in STEP1)


def test_truncated_likelihood_normalizes(params):
    """Integrating the truncated density over the retained support gives one."""
    # Upstream's own sigma_floor: a near-zero width would be a delta the 256-point
    # quadrature grid cannot resolve, which would test the grid, not the density.
    p = params["rgo"]
    y_trunc = float(np.log(p.threshold_msh))
    grid = np.linspace(y_trunc, y_trunc + 10, 4000)
    density = np.exp(log_pdf_truncated(grid, p, sigma_row=0.0, y_trunc=y_trunc))
    assert np.trapezoid(density, grid) == pytest.approx(1.0, abs=2e-2)


def test_truncated_likelihood_is_finite_on_real_data(params):
    groups = load_group_single_entry(["rgo"])
    p = params["rgo"]
    kept = groups[groups["area_corr"] > p.threshold_msh].head(500)
    values = log_pdf_truncated(
        np.log(kept["area_corr"].to_numpy()),
        p,
        sigma_row=np.nan_to_num(kept["area_corr_unc"].to_numpy() / kept["area_corr"].to_numpy()),
    )
    assert np.isfinite(values).all()


def test_fraction_above_is_monotone(params):
    p = params["rgo"]
    values = [fraction_above(t, p) for t in (1.0, 10.0, 30.0, 100.0, 300.0)]
    assert all(a > b for a, b in zip(values[:-1], values[1:], strict=True))
    assert 0.0 < values[-1] < values[0] < 1.0


def test_crossover_separates_the_populations(params):
    """Below the crossover the Weibull dominates, above it the log-normal does."""
    for survey in STEP1:
        p = params[survey]
        cross = component_crossover(p)
        assert np.isfinite(cross) and 10.0 < cross < 1000.0, (survey, cross)
        for factor, weibull_should_win in ((0.3, True), (3.0, False)):
            z = np.log(cross * factor)
            weibull = np.log(p.pi_W) + log_pdf_weibull(z, p.k_W, p.log_lambda_W)
            lognormal = np.log(1 - p.pi_W) + norm.logpdf(z, p.mu_LN, p.sigma_LN)
            assert (weibull > lognormal) == weibull_should_win, (survey, factor)


def test_unknown_survey_is_rejected():
    with pytest.raises(KeyError, match="no area law"):
        load_upstream_parameters("soon", "umbra")


# --- activity level -------------------------------------------------------


def test_gaussian_smooth_recovers_a_known_sinusoid():
    days = np.arange(4000)
    period = 4000.0
    signal = 1.0 + 0.5 * np.sin(2 * np.pi * days / period)
    smoothed = gaussian_smooth(signal, 182)
    # A Gaussian filter attenuates a sinusoid by exp(-(2 pi sigma / period)^2 / 2).
    attenuation = np.exp(-0.5 * (2 * np.pi * 182 / period) ** 2)
    interior = slice(1000, 3000)
    amplitude = (smoothed[interior].max() - smoothed[interior].min()) / 2
    assert amplitude == pytest.approx(0.5 * attenuation, rel=0.1)


def test_gaussian_smooth_ignores_gaps():
    values = np.array([1.0, np.nan, 1.0, 1.0, np.nan, 1.0])
    assert np.allclose(gaussian_smooth(values, 1.0), 1.0)


def test_activity_level_is_in_mhem():
    """The published power laws are meaningless in the wrong unit.

    The paper's activity level runs to about 0.4 mHem per hemisphere. Ours must be on
    that scale, not in muHem, which would be a thousand times larger.
    """
    status = load_day_status(["rgo"])
    groups = load_group_single_entry(["rgo"])
    hemispheric = activity_level(status, groups, scope="north")
    assert 0.1 < np.nanmax(hemispheric) < 0.6, np.nanmax(hemispheric)
    whole = activity_level(status, groups)
    assert 0.2 < np.nanmax(whole) < 1.0


def test_the_two_weightings_differ_by_about_six(params):
    """Emergence weighting is the paper's; instantaneous is the disk-area one.

    Long-lived groups are counted on every visible day in the instantaneous version
    and once in the emergence version, so the gap is the mean visible residence. It is
    pinned because mixing the two would silently rescale every fitted power law.
    """
    del params
    status = load_day_status(["rgo"])
    groups = load_group_single_entry(["rgo"])
    emergence = activity_level(status, groups, weighting="emergence")
    instantaneous = activity_level(status, weighting="instantaneous")
    ratio = np.nanmax(instantaneous) / np.nanmax(emergence)
    assert 4.0 < ratio < 9.0, ratio


def test_emergence_weighting_requires_the_group_table():
    with pytest.raises(ValueError, match="single-entry group table"):
        activity_level(load_day_status(["rgo"]), weighting="emergence")


def test_unknown_weighting_is_rejected():
    with pytest.raises(ValueError, match="unknown weighting"):
        activity_level(load_day_status(["rgo"]), weighting="rolling")


def test_hemispheres_sum_to_the_whole():
    status = load_day_status(["rgo"])
    groups = load_group_single_entry(["rgo"])
    whole = activity_level(status, groups)
    north = activity_level(status, groups, scope="north")
    south = activity_level(status, groups, scope="south")
    assert np.allclose(np.nan_to_num(north + south), np.nan_to_num(whole), rtol=1e-8)


def test_published_power_laws_evaluate_in_the_paper_s_range():
    activity = np.array([0.05, 0.1, 0.2, 0.4])
    mu = published_power_law("mu_LN", activity)
    sigma = published_power_law("sigma_LN", activity)
    c = published_power_law("c_LN", activity)
    # Figures 6(g)-(i): mu rises from ~5.5 to ~5.9, sigma ~0.6 to ~0.95, c ~0.2 to ~0.55.
    assert (mu > 5.3).all() and (mu < 6.1).all()
    assert (sigma > 0.5).all() and (sigma < 1.1).all()
    assert (c > 0.1).all() and (c < 0.7).all()
    assert (np.diff(mu) > 0).all() and (np.diff(c) > 0).all()
    assert PUBLISHED_2015["k_W"] == 0.46
    assert ACTIVITY_SIGMA_DAYS == 182
    assert MUHEM_PER_MHEM == 1000.0


def test_upstream_weibull_shape_agrees_with_the_published_value(params):
    """Independent agreement on the baseline component, which licenses fixing it."""
    for survey in STEP1:
        assert params[survey].k_W == pytest.approx(PUBLISHED_2015["k_W"], abs=0.03)


def test_area_law_params_are_immutable(params):
    with pytest.raises(Exception):
        params["rgo"].k_W = 0.9
    changed = params["rgo"].replace(pi_W=0.5)
    assert changed.pi_W == 0.5 and params["rgo"].pi_W != 0.5
    assert isinstance(changed, AreaLawParams)
