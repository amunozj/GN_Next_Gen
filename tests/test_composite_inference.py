"""Mode finding, and the Godambe correction for a composite likelihood.

The correction is derived in this project rather than taken from a source --
``docs/02_research/`` has the mechanism three times over and no derivation, no
reference and no magnitude estimate.  So it is pinned here against a case whose
answer is known in closed form: if every observation is counted ``m`` times, the
curvature is ``m`` times too sharp, the naive interval is ``sqrt(m)`` times too
narrow, and the sandwich has to put exactly that factor back.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gn_next_gen.inference.composite import (
    draw_parameters,
    eigen_report,
    expand_covariance,
    fit_mode,
    godambe_correction,
)

jax.config.update("jax_enable_x64", True)


def gaussian_problem(n_blocks: int = 400, multiplicity: int = 1, seed: int = 0):
    """``log L(theta) = sum_b m * N(x_b | theta, 1)`` -- each block counted m times.

    The mode is ``mean(x)`` whatever ``m`` is, and its true sd is
    ``1 / sqrt(n_blocks)``.  The naive Laplace sd is ``1 / sqrt(m * n_blocks)``, so
    the correction has to inflate by ``sqrt(m)``.
    """
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal(n_blocks))

    def objective(theta, weights=None):
        if weights is None:
            weights = jnp.ones(n_blocks)
        return -0.5 * multiplicity * jnp.sum(weights * (x - theta[0]) ** 2)

    mode = jnp.asarray([float(jnp.mean(x))])

    # Weights traced, gradient compiled once: a fresh closure per resample would
    # recompile every time and measure the compiler rather than the correction.
    @jax.jit
    def gradient(weights):
        return jax.grad(lambda sub: objective(sub, weights))(mode)

    def block_score(weights):
        return np.asarray(gradient(jnp.asarray(weights, dtype=float)))

    return objective, block_score, np.asarray(x)


# --------------------------------------------------------------------------
# Mode finding
# --------------------------------------------------------------------------


def test_fit_mode_finds_a_known_optimum():
    objective, _score, x = gaussian_problem()
    fit = fit_mode(objective, np.array([3.0]))
    assert fit.success
    assert float(fit.theta[0]) == pytest.approx(float(x.mean()), abs=1e-6)
    assert fit.seconds >= 0.0
    assert fit.meta["n_parameters"] == 1


def test_fit_mode_reports_failure_rather_than_pretending():
    def diverging(theta):
        return theta[0]

    fit = fit_mode(diverging, np.array([0.0]), maxiter=5)
    assert isinstance(fit.message, str) and fit.message


# --------------------------------------------------------------------------
# The correction, against a closed-form answer
# --------------------------------------------------------------------------


@pytest.mark.parametrize("multiplicity", [1, 4, 9, 25])
def test_the_sandwich_puts_back_exactly_the_over_counting_factor(multiplicity):
    """Counting each block ``m`` times must inflate the interval by ``sqrt(m)``.

    This is the whole correction in one number.  At ``m = 1`` the terms really are
    independent and the sandwich must be a no-op -- which is the property that
    makes it safe to apply unconditionally.
    """
    n_blocks = 400
    objective, block_score, x = gaussian_problem(
        n_blocks=n_blocks, multiplicity=multiplicity, seed=1
    )
    theta = np.array([float(x.mean())])
    correction = godambe_correction(
        objective,
        theta,
        indices=np.array([0]),
        block_score=block_score,
        n_bootstrap=400,
        n_blocks=n_blocks,
        seed=3,
    )
    assert correction.median_inflation == pytest.approx(np.sqrt(multiplicity), rel=0.15)

    corrected_sd = np.sqrt(correction.covariance[0, 0])
    assert corrected_sd == pytest.approx(1.0 / np.sqrt(n_blocks), rel=0.15)

    naive_sd = np.sqrt(correction.naive_covariance[0, 0])
    assert naive_sd == pytest.approx(1.0 / np.sqrt(multiplicity * n_blocks), rel=1e-6)


def test_with_independent_terms_the_correction_is_a_no_op():
    n_blocks = 500
    objective, block_score, x = gaussian_problem(n_blocks=n_blocks, multiplicity=1, seed=2)
    correction = godambe_correction(
        objective,
        np.array([float(x.mean())]),
        indices=np.array([0]),
        block_score=block_score,
        n_bootstrap=400,
        n_blocks=n_blocks,
        seed=5,
    )
    assert correction.median_inflation == pytest.approx(1.0, rel=0.12)


def test_the_bootstrap_needs_to_know_how_many_blocks_there_are():
    objective, block_score, x = gaussian_problem()
    with pytest.raises(ValueError, match="n_blocks is required"):
        godambe_correction(
            objective,
            np.array([float(x.mean())]),
            indices=np.array([0]),
            block_score=block_score,
        )


def test_correction_records_what_it_measured():
    n_blocks = 200
    objective, block_score, x = gaussian_problem(n_blocks=n_blocks, multiplicity=4)
    correction = godambe_correction(
        objective,
        np.array([float(x.mean())]),
        indices=np.array([0]),
        block_score=block_score,
        n_bootstrap=200,
        n_blocks=n_blocks,
    )
    assert correction.n_blocks == n_blocks
    assert correction.n_bootstrap == 200
    assert "median_inflation" in correction.meta
    assert "max_inflation" in correction.meta
    assert np.isfinite(correction.meta["condition_number"])


# --------------------------------------------------------------------------
# Degeneracy, which the graph guarantees will happen
# --------------------------------------------------------------------------


def test_a_flat_direction_is_reported_as_prior_dominated():
    """One exact null mode per floating connected component, by construction.

    The Laplacian of a graph with two components has two null modes; the anchor
    pins one, and the other is a direction the data says nothing about.  The
    spectrum has to say so rather than inverting it into a huge variance.
    """

    def objective(theta, weights=None):
        # theta[1] does not appear: an exactly flat direction.
        return -0.5 * 100.0 * theta[0] ** 2

    def block_score(weights):
        return np.array([0.0, 0.0])

    correction = godambe_correction(
        objective,
        np.array([0.0, 0.0]),
        indices=np.array([0, 1]),
        block_score=block_score,
        n_bootstrap=20,
        n_blocks=10,
    )
    assert correction.n_weak_modes >= 1
    report = eigen_report(correction)
    assert bool(report["prior_dominated"].any())
    assert len(report) == 2


def test_draws_put_no_variation_along_a_flat_direction():
    """A Cholesky would simply fail here; the eigendecomposition answers honestly."""
    covariance = np.array([[0.25, 0.0], [0.0, 0.0]])
    draws = draw_parameters(np.array([1.0, 2.0]), covariance, n_draws=500, seed=0)
    assert draws.shape == (500, 2)
    assert draws[:, 1].std() == pytest.approx(0.0, abs=1e-12)
    assert draws[:, 0].std() == pytest.approx(0.5, rel=0.15)
    assert draws[:, 0].mean() == pytest.approx(1.0, abs=0.1)


def test_draws_are_reproducible_from_a_seed():
    covariance = np.eye(2) * 0.1
    first = draw_parameters(np.zeros(2), covariance, n_draws=50, seed=11)
    second = draw_parameters(np.zeros(2), covariance, n_draws=50, seed=11)
    assert np.array_equal(first, second)
    other = draw_parameters(np.zeros(2), covariance, n_draws=50, seed=12)
    assert not np.array_equal(first, other)


def test_draws_recover_nominal_coverage_on_a_known_gaussian():
    covariance = np.array([[4.0]])
    draws = draw_parameters(np.array([0.0]), covariance, n_draws=20_000, seed=2)
    for nominal, z in [(0.50, 0.6745), (0.90, 1.6449)]:
        inside = np.mean(np.abs(draws[:, 0]) <= z * 2.0)
        assert inside == pytest.approx(nominal, abs=0.02)


def test_expand_covariance_places_the_block_and_zeroes_the_rest():
    correction_like = godambe_correction(
        *gaussian_problem(n_blocks=50)[:1],
        np.array([0.0]),
        indices=np.array([0]),
        block_score=gaussian_problem(n_blocks=50)[1],
        n_bootstrap=20,
        n_blocks=50,
    )
    full = expand_covariance(correction_like, 5)
    assert full.shape == (5, 5)
    assert full[0, 0] > 0
    assert np.allclose(full[1:, :], 0.0)
    assert np.allclose(full[:, 1:], 0.0)


@pytest.mark.slow
def test_inflation_grows_with_how_much_the_network_double_counts():
    """The property the correction exists for, on an actual network.

    A day seen by ``n`` observers enters ``n(n-1)/2`` pairwise terms, so the
    over-counting rises steeply with network density.  Measured on matched networks
    of identical length, the 90th-percentile inflation tracks it:

    =========  ===========  ==============
    observers  pairs/day    p90 inflation
    =========  ===========  ==============
    6          4.6          1.19
    12         16.2         1.32
    24         64.3         1.38
    48         259.5        2.01
    =========  ===========  ==============
    """
    from gn_next_gen.models import emissions as em
    from gn_next_gen.models import network as nw
    from gn_next_gen.models.activity_map import ActivityMap

    detection = em.detection_grid("rgo")
    activity_map = ActivityMap(
        "rgo", 0.2155, 1.1414, 0.995, 0.11, 0.17, 10.5, 37_000, detection.reference_threshold_msh
    )

    inflations = []
    for n_observers in (6, 48):
        observations, _truth = nw.simulate_matched_network(
            detection,
            n_observers=n_observers,
            n_days=6000,
            activity_map=activity_map,
            observing_fraction=0.9,
            career_fraction=1.0,
            seed=4,
        )
        fit = nw.calibrate_network(
            observations,
            detection=detection,
            activity_map=activity_map,
            anchor_station=nw.ANCHOR_STATION,
            kernel="conditional",
            fingerprint_weight=0.0,
            lambda_penalty=0.0,
            n_outer=1,
            correct=True,
            n_bootstrap=150,
            n_draws=0,
            maxiter=200,
            min_overlap_days=30,
        )
        inflations.append(float(np.nanpercentile(fit.correction.inflation, 90)))

    sparse, dense = inflations
    assert dense > sparse, "a denser network double-counts more and must inflate more"
    assert dense > 1.2, "a network with hundreds of pairs per day is not independent"
