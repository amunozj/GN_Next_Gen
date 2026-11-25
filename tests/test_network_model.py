"""The concurrent network model: kernels, priors, and parameter recovery.

The recovery tests here use a **matched** simulator, which tests the solver and the
parameterization and nothing else.  That is deliberate and is the SDSS test: their
noise-free simulations reached about 0.1%, "so the residual error came from model
misspecification rather than from statistics".  Whatever the blind Step 2 benchmark
leaves over is therefore attributable to the emission model rather than to the fit.
A matched simulation is never a benchmark -- "benchmarks that reuse the same
generative model as the inference method will overstate performance".
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from gn_next_gen.models import emissions as em
from gn_next_gen.models import network as nw
from gn_next_gen.models.activity_map import ActivityMap


@pytest.fixture(scope="module")
def detection() -> em.DetectionGrid:
    return em.detection_grid("rgo")


@pytest.fixture(scope="module")
def activity_map(detection) -> ActivityMap:
    return ActivityMap(
        survey="rgo",
        amplitude=0.2155,
        exponent=1.1414,
        r_log=0.995,
        residual_sd=0.11,
        gn_lo=0.17,
        gn_hi=10.5,
        n_days=37_000,
        threshold_msh=detection.reference_threshold_msh,
    )


@pytest.fixture(scope="module")
def small_network(detection, activity_map):
    return nw.simulate_matched_network(
        detection,
        n_observers=14,
        n_days=1400,
        activity_map=activity_map,
        seed=7,
    )


@pytest.fixture(scope="module")
def base_fit(small_network, detection, activity_map):
    """The default fit, computed once -- most tests only inspect it."""
    observations, _truth = small_network
    return _fit(observations, detection, activity_map)


def _fit(observations, detection, activity_map, **kwargs):
    defaults = dict(
        detection=detection,
        activity_map=activity_map,
        anchor_station=nw.ANCHOR_STATION,
        kernel="conditional",
        fingerprint_weight=0.0,
        lambda_penalty=0.0,
        n_outer=2,
        correct=False,
        maxiter=250,
        min_overlap_days=30,
    )
    defaults.update(kwargs)
    return nw.calibrate_network(observations, **defaults)


def _true_log_r(fit, truth, detection):
    tmap = truth.set_index("station")["true_threshold_msh"]
    index = {int(y): i for i, y in enumerate(fit.inputs.years)}
    slot_activity = fit.inputs.activity[[index[int(y)] for y in fit.inputs.slot_year]]
    return np.asarray(detection(tmap.reindex(fit.inputs.slot_station).to_numpy(), slot_activity))


# --------------------------------------------------------------------------
# The anchor, and the degeneracy it removes
# --------------------------------------------------------------------------


def test_the_anchor_is_pinned_exactly(base_fit, detection):
    """``k_ref = 1`` is what makes the absolute scale identified at all (ADR-003).

    SDSS calls the unpinned version "the zero point of the calibration which is
    exactly degenerate"; here the pin is a hard constraint in ``_unpack``, so no
    amount of optimization can move it.
    """
    fit = base_fit
    anchor = fit.observers[fit.observers["station"] == nw.ANCHOR_STATION]
    assert len(anchor) == 1
    assert float(anchor["factor"].iloc[0]) == pytest.approx(1.0, abs=1e-9)
    assert float(anchor["recovered_threshold_msh"].iloc[0]) == pytest.approx(
        detection.reference_threshold_msh, rel=1e-9
    )


def test_the_anchor_cannot_be_moved_by_its_own_parameters(base_fit):
    fit = base_fit
    moved = fit.theta.copy()
    slices = fit.layout.slices()
    moved[slices["log_threshold"].start + fit.layout.anchor_row] += 3.0
    moved[slices["log_split"].start + fit.layout.anchor_row] += 3.0
    before = nw.composite_loglik(jnp.asarray(fit.theta), fit.inputs, fit.layout, fit.static)
    after = nw.composite_loglik(jnp.asarray(moved), fit.inputs, fit.layout, fit.static)
    assert float(before) == pytest.approx(float(after), rel=1e-10)


def test_a_common_shift_of_every_free_scale_is_rejected(base_fit):
    """The degeneracy the anchor removes: shifting everyone together must cost.

    Without the pin, multiplying every observer's detection by ``c`` and dividing
    the latent level by ``c`` leaves a binomial-Poisson marginal unchanged.  The
    anchor breaks it, so the objective has to fall when the free observers move
    together and the anchor does not.
    """
    fit = base_fit
    best = float(nw.composite_loglik(jnp.asarray(fit.theta), fit.inputs, fit.layout, fit.static))
    slices = fit.layout.slices()
    for shift in (-0.4, +0.4):
        moved = fit.theta.copy()
        block = np.arange(slices["log_threshold"].start, slices["log_threshold"].stop)
        free = np.delete(block, fit.layout.anchor_row)
        moved[free] += shift
        worse = float(nw.composite_loglik(jnp.asarray(moved), fit.inputs, fit.layout, fit.static))
        assert worse < best


# --------------------------------------------------------------------------
# Recovery
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_the_rate_ratio_recovers_on_a_matched_network(detection, activity_map):
    """``log r`` is the quantity the series divides by, so it is the one to check.

    Measured on a 30-observer matched network: bias **-0.020** and scatter
    **0.033** in ``log r``, i.e. about 2% on each observer's calibration factor.
    """
    observations, truth = nw.simulate_matched_network(
        detection, n_observers=30, n_days=3000, activity_map=activity_map, seed=1
    )
    fit = _fit(observations, detection, activity_map, maxiter=400, n_outer=3)
    got, _ = nw.log_rate_ratio(jnp.asarray(fit.theta), fit.inputs, fit.layout, fit.static)
    free = fit.inputs.slot_station != nw.ANCHOR_STATION
    error = np.asarray(got)[free] - _true_log_r(fit, truth, detection)[free]
    assert abs(float(error.mean())) < 0.10
    assert float(error.std()) < 0.15


@pytest.mark.slow
def test_the_threshold_is_weakly_identified_and_shrinks_to_its_prior(detection, activity_map):
    """Why an acuity-recovery figure must not be read as a goodness test.

    ``d log k / d log T`` is about **-0.24** near 35 MSH, so the data constrains the
    *threshold* roughly four times less well than the rate it implies.  The fitted
    threshold therefore shrinks toward its prior -- measured here as a bias of about
    +0.15 in log with a strongly negative correlation against the true threshold --
    while ``log r`` stays accurate to a few percent.  Step 2's f23 plots the
    threshold, so this is the caveat that belongs beside it, for the published
    methods as much as for this one.
    """
    observations, truth = nw.simulate_matched_network(
        detection, n_observers=30, n_days=3000, activity_map=activity_map, seed=1
    )
    fit = _fit(observations, detection, activity_map, maxiter=400, n_outer=3)
    _, log_threshold = nw.log_rate_ratio(jnp.asarray(fit.theta), fit.inputs, fit.layout, fit.static)
    tmap = truth.set_index("station")["true_threshold_msh"]
    free = fit.inputs.slot_station != nw.ANCHOR_STATION
    true_threshold = tmap.reindex(fit.inputs.slot_station).to_numpy()[free]
    error = np.log(np.exp(np.asarray(log_threshold))[free] / true_threshold)

    sensitivity = float(
        (detection(40.0, detection.activity_ref) - detection(30.0, detection.activity_ref))
        / (np.log(40.0) - np.log(30.0))
    )
    assert -0.4 < sensitivity < -0.1

    per_station = (
        pd.DataFrame(
            {
                "station": fit.inputs.slot_station[free],
                "error": error,
                "true": true_threshold,
            }
        )
        .groupby("station")
        .mean()
    )
    correlation = float(np.corrcoef(np.log(per_station["true"]), per_station["error"])[0, 1])
    assert correlation < -0.4, "threshold error should shrink toward the prior"


@pytest.mark.slow
def test_the_activity_fixed_point_converges(detection, activity_map):
    observations, _truth = nw.simulate_matched_network(
        detection, n_observers=20, n_days=2200, activity_map=activity_map, seed=3
    )
    fit = _fit(observations, detection, activity_map, n_outer=5, maxiter=300)
    assert fit.passes[-1]["max_log_activity_shift"] < 0.05
    assert len(fit.passes) <= 5


@pytest.mark.slow
def test_the_two_kernels_agree_at_moderate_counts(detection, activity_map):
    """Kernel B integrates the latent level where Kernel A eliminates it.

    Where counts are moderate both are valid and they should land in the same
    place; their disagreement elsewhere is the misspecification signal the
    equating section asks to be kept and reported.
    """
    observations, _truth = nw.simulate_matched_network(
        detection, n_observers=18, n_days=2000, activity_map=activity_map, seed=5
    )
    conditional = _fit(observations, detection, activity_map, kernel="conditional")
    marginal = _fit(observations, detection, activity_map, kernel="marginal")

    merged = conditional.observers.merge(marginal.observers, on="station", suffixes=("_a", "_b"))
    merged = merged[merged["station"] != nw.ANCHOR_STATION]
    disagreement = np.log(merged["factor_a"] / merged["factor_b"]).abs()
    assert float(disagreement.median()) < 0.5


# --------------------------------------------------------------------------
# The terms
# --------------------------------------------------------------------------


def test_zero_lag_pairs_use_a_plain_binomial(base_fit):
    """At ``dt = 0`` the kernel must be exactly binomial, not nearly so.

    Both branches are evaluated and selected rather than pushing the concentration
    to infinity, because differencing two very large ``gammaln`` values loses the
    precision the float64 accumulation exists to protect.
    """
    fit = base_fit
    static = dict(fit.static)
    static["kappa0"] = jnp.asarray(1.0)
    log_r, _ = nw.log_rate_ratio(jnp.asarray(fit.theta), fit.inputs, fit.layout, fit.static)

    exact = np.asarray(fit.static["rho"]) >= 1.0 - 1e-12
    assert exact.any(), "the fixture should contain same-day pairs"

    only_zero = dict(static)
    only_zero["weight"] = jnp.where(jnp.asarray(exact), static["weight"], 0.0)
    value = float(nw.pairwise_loglik_conditional(log_r, only_zero))

    # Recomputed by hand from the binomial pmf.
    from scipy.stats import binom

    r = np.exp(np.asarray(log_r))
    y_a = np.asarray(static["count_a"])[exact]
    y_b = np.asarray(static["count_b"])[exact]
    p = r[np.asarray(static["slot_a"])[exact]] / (
        r[np.asarray(static["slot_a"])[exact]] + r[np.asarray(static["slot_b"])[exact]]
    )
    weight = np.asarray(static["weight"])[exact]
    resample = np.asarray(static["year_weight"])[np.asarray(static["pair_year"])][exact]
    n = y_a + y_b
    manual = np.where(n > 0, binom.logpmf(y_a, n, p), 0.0)
    assert value == pytest.approx(float((weight * resample * manual).sum()), rel=1e-9)


def test_a_lag_is_overdispersed_relative_to_the_same_day(base_fit):
    """The lifetime-based decorrelation, replacing "+/-2 days is simultaneous"."""
    fit = base_fit
    log_r, _ = nw.log_rate_ratio(jnp.asarray(fit.theta), fit.inputs, fit.layout, fit.static)
    rho = np.asarray(fit.static["rho"])
    assert rho.max() >= 1.0 - 1e-12
    if float(rho.min()) >= 1.0 - 1e-12:
        pytest.skip("fixture has no lagged pairs")

    static = dict(fit.static)
    static["kappa0"] = jnp.asarray(5.0)
    lagged = rho < 1.0 - 1e-12
    base = dict(static)
    base["weight"] = jnp.where(jnp.asarray(lagged), static["weight"], 0.0)
    with_lag = float(nw.pairwise_loglik_conditional(log_r, base))

    pretend_simultaneous = dict(base)
    pretend_simultaneous["rho"] = jnp.ones_like(static["rho"])
    as_if = float(nw.pairwise_loglik_conditional(log_r, pretend_simultaneous))
    # Treating a lagged pair as simultaneous claims more information than there is,
    # so the honest likelihood is the lower one.
    assert with_lag < as_if


def test_the_reference_autocorrelation_decays_from_one():
    counts = np.sin(np.arange(900) / 9.0) + 3.0
    acf = nw.reference_autocorrelation(counts, 5)
    assert acf[0] == pytest.approx(1.0)
    assert len(acf) == 6
    assert np.all(acf <= 1.0)


def test_documented_breaks_get_a_free_jump_and_others_do_not(
    small_network, detection, activity_map
):
    observations, _truth = small_network
    station = int(observations["station"].iloc[0])
    year = int(np.floor(nw.decimal_year(observations["date_ord"].iloc[0]))[0])
    metadata = pd.DataFrame(
        {"station": [station], "date_ord": [observations["date_ord"].iloc[0]], "note": ["break"]}
    )
    fit = _fit(observations, detection, activity_map, metadata=metadata)
    documented = np.asarray(fit.static["documented"])
    assert documented.sum() == 1
    slot = fit.inputs.slot_of[(station, year)]
    assert documented[slot]

    # The innovation at a documented break carries no prior penalty.
    theta = fit.theta.copy()
    slices = fit.layout.slices()
    base = float(nw.log_prior(jnp.asarray(theta), fit.layout, fit.static))
    theta[slices["zeta"].start + slot] += 2.0
    assert float(nw.log_prior(jnp.asarray(theta), fit.layout, fit.static)) == pytest.approx(base)


def test_an_undocumented_innovation_is_penalised(base_fit):
    fit = base_fit
    theta = fit.theta.copy()
    slices = fit.layout.slices()
    base = float(nw.log_prior(jnp.asarray(theta), fit.layout, fit.static))
    theta[slices["zeta"].start] += 2.0
    assert float(nw.log_prior(jnp.asarray(theta), fit.layout, fit.static)) < base


def test_an_unknown_kernel_is_refused(small_network, detection, activity_map):
    observations, _truth = small_network
    with pytest.raises(ValueError, match="unknown kernel"):
        _fit(observations, detection, activity_map, kernel="vibes")


def test_a_network_without_the_anchor_is_refused(small_network, detection, activity_map):
    observations, _truth = small_network
    without = observations[observations["station"] != nw.ANCHOR_STATION]
    with pytest.raises(ValueError, match="anchor station"):
        _fit(without, detection, activity_map)


# --------------------------------------------------------------------------
# Anchors and the series contract
# --------------------------------------------------------------------------


def test_the_series_carries_the_uncertainty_split_step_two_requires(base_fit):
    """Without the split a yearly interval is too narrow by up to sqrt(365)."""
    fit = base_fit
    for column in ("date_ord", "gn", "gn_sd", "gn_sd_factor", "gn_sd_spread", "n_observers"):
        assert column in fit.series.columns
    assert (fit.series["gn"] >= 0).all()
    assert np.isfinite(fit.series["gn"]).all()


def test_the_observer_table_has_the_columns_the_scorer_reads(base_fit):
    fit = base_fit
    for column in (
        "station",
        "factor",
        "factor_sd",
        "recovered_threshold_msh",
        "n_hops",
        "reached",
    ):
        assert column in fit.observers.columns
    assert len(fit.observers) == fit.inputs.n_stations


@pytest.mark.parametrize(
    "anchoring,first_year,last_year",
    [("debrecen_21_23", 1974, 2012), ("rgo_15_19", 1911, 1969), ("rgo_12_19", 1877, 1969)],
)
def test_anchor_spans_match_adr_009(anchoring, first_year, last_year):
    """ADR-009 defines the anchors by cycle number, not by date."""
    first, last = nw.anchor_span(anchoring)
    assert int(nw.decimal_year(first)[0]) == first_year
    assert int(nw.decimal_year(last)[0]) == last_year


def test_an_unknown_anchoring_is_refused():
    with pytest.raises(ValueError, match="unknown anchoring"):
        nw.anchor_span("rgo_20_24")


@pytest.mark.slow
def test_anchor_observations_are_the_surveys_own_counts():
    frame, threshold = nw.anchor_observations("rgo_15_19")
    assert set(frame.columns) >= {"date_ord", "station", "groups"}
    assert (frame["station"] == nw.ANCHOR_STATION).all()
    assert threshold == pytest.approx(28.084, abs=0.01)
    first, last = nw.anchor_span("rgo_15_19")
    assert frame["date_ord"].min() >= first
    assert frame["date_ord"].max() <= last
