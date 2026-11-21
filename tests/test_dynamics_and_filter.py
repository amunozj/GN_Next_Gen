"""The Step 4 engine, built here because Check A needs it.

Three things have to be right or every later step inherits the error: the chain's
algebra, the filter's exactness, and the fitter's ability to recover parameters it
generated. Each is checked against something independent -- closed forms, a dense
brute-force filter, and simulated data with known truth.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gn_next_gen.inference.forward_filter import (
    brute_force_loglik,
    count_emissions,
    fit_immigration_death,
    forward_loglik,
    forward_loglik_two_class,
    knot_basis,
    moment_initialization,
    second_difference_penalty,
)
from gn_next_gen.models.dynamics import (
    birth_matrix,
    expected_lifetime_days,
    phi_from_lifetime,
    simulate_counts,
    stationary_distribution,
    stationary_mean,
    survival_matrix,
    transition_matrix,
)
from gn_next_gen.models.visibility import (
    MAX_LIFETIME_DAYS,
    SYNODIC_ROTATION_DAYS,
    expected_visible_span,
    max_transit_days,
    simulate_marked_process,
)

jax.config.update("jax_enable_x64", True)

K = 41


# --- the chain's algebra ---------------------------------------------------


def test_survival_matrix_is_a_lower_triangular_stochastic_matrix():
    s = survival_matrix(0.85, K)
    assert jnp.allclose(s.sum(axis=1), 1.0)
    assert jnp.allclose(jnp.triu(s, 1), 0.0), "survivors cannot exceed the previous count"
    # Row i is Binomial(i, phi), so its mean is i * phi.
    counts = jnp.arange(K, dtype=jnp.float64)
    assert jnp.allclose(s @ counts, counts * 0.85)


def test_birth_matrix_is_upper_triangular_toeplitz_with_the_tail_held():
    b = birth_matrix(1.4, K)
    assert jnp.allclose(b.sum(axis=1), 1.0), "the truncated tail must be held, not lost"
    assert jnp.allclose(jnp.tril(b, -1), 0.0), "births cannot reduce the count"
    # Toeplitz away from the tail column.
    interior = b[:, : K - 1]
    for offset in range(1, 5):
        diag = jnp.diagonal(interior, offset=offset)
        assert jnp.allclose(diag, diag[0])


def test_transition_is_stochastic_and_factorizes():
    direct = transition_matrix(0.9, 1.1, K)
    assert jnp.allclose(direct.sum(axis=1), 1.0)
    assert jnp.allclose(direct, survival_matrix(0.9, K) @ birth_matrix(1.1, K))


def test_stationary_distribution_is_a_fixed_point():
    phi, lam = 0.88, 1.0
    pi = stationary_distribution(phi, lam, K)
    assert jnp.allclose(pi.sum(), 1.0)
    assert float(jnp.abs(pi @ transition_matrix(phi, lam, K) - pi).max()) < 1e-12
    # And its mean is the closed form, while truncation is negligible at K = 41.
    assert float(pi @ jnp.arange(K)) == pytest.approx(float(stationary_mean(phi, lam)), rel=1e-6)


def test_lifetime_helpers_invert_each_other():
    assert float(expected_lifetime_days(0.875)) == pytest.approx(8.0)
    assert float(phi_from_lifetime(8.0)) == pytest.approx(0.875)


def test_simulation_matches_its_own_stationary_moments():
    phi, lam = 0.88, 1.0
    path = np.asarray(simulate_counts(jax.random.key(0), phi, jnp.full(15_000, lam), K))
    assert path.mean() == pytest.approx(float(stationary_mean(phi, lam)), rel=0.08)
    # rho(1) = phi for this chain.
    assert np.corrcoef(path[:-1], path[1:])[0, 1] == pytest.approx(phi, abs=0.03)


# --- the filter's exactness ----------------------------------------------


def test_forward_filter_matches_a_dense_brute_force_filter():
    """The factorized scan and an explicit dense filter must agree to machine precision."""
    rng = np.random.default_rng(0)
    k, n_days = 8, 50
    phi, lam = 0.8, 0.9
    lam_t = np.full(n_days, lam)
    path = np.asarray(simulate_counts(jax.random.key(1), phi, jnp.asarray(lam_t), k))
    observed = rng.random(n_days) > 0.3

    emissions = count_emissions(jnp.asarray(path), jnp.asarray(observed), k)
    got = float(forward_loglik(emissions, phi, jnp.asarray(lam_t), k))
    want = brute_force_loglik(path, observed, phi, lam_t, k)
    assert got == pytest.approx(want, abs=1e-9)


def test_unobserved_days_contribute_no_likelihood_but_do_propagate():
    """A gap must not be free *and* must not be interpolated: it propagates the chain."""
    k, n_days = 10, 12
    lam_t = jnp.full(n_days, 0.8)
    counts = jnp.zeros(n_days, dtype=jnp.int32)
    none_seen = count_emissions(counts, jnp.zeros(n_days, bool), k)
    assert float(forward_loglik(none_seen, 0.8, lam_t, k)) == pytest.approx(0.0, abs=1e-9)

    # Two observations far apart are less constrained than two adjacent ones.
    seen = np.zeros(n_days, bool)
    seen[[0, 1]] = True
    adjacent = count_emissions(jnp.asarray([5] + [0] * (n_days - 1)), jnp.asarray(seen), k)
    seen_far = np.zeros(n_days, bool)
    seen_far[[0, n_days - 1]] = True
    far = count_emissions(jnp.asarray([5] + [0] * (n_days - 2) + [0]), jnp.asarray(seen_far), k)
    assert float(forward_loglik(far, 0.8, lam_t, k)) != float(
        forward_loglik(adjacent, 0.8, lam_t, k)
    )


def test_loglikelihood_is_accumulated_in_float64():
    emissions = count_emissions(jnp.zeros(100, dtype=jnp.int32), jnp.ones(100, bool), 10)
    assert forward_loglik(emissions, 0.8, jnp.full(100, 0.5), 10).dtype == jnp.float64


def test_two_class_filter_reduces_to_one_class_when_a_class_is_empty():
    """With no births in class 2 the joint filter must equal the single-class filter."""
    k, n_days = 12, 40
    phi = jnp.asarray([0.85, 0.85])
    lam_t = jnp.stack([jnp.full(n_days, 0.7), jnp.full(n_days, 1e-12)], axis=1)
    path = np.asarray(simulate_counts(jax.random.key(2), 0.85, lam_t[:, 0], k))
    observed = jnp.ones(n_days, bool)

    joint = float(forward_loglik_two_class(jnp.asarray(path), observed, phi, lam_t, k))
    single = float(
        forward_loglik(count_emissions(jnp.asarray(path), observed, k), 0.85, lam_t[:, 0], k)
    )
    assert joint == pytest.approx(single, abs=1e-6)


def test_two_class_total_is_what_is_observed():
    """The split is latent: only N1 + N2 is pinned by an observation."""
    k, n_days = 10, 5
    phi = jnp.asarray([0.8, 0.9])
    lam_t = jnp.stack([jnp.full(n_days, 0.6), jnp.full(n_days, 0.4)], axis=1)
    counts = jnp.asarray([2, 2, 2, 2, 2], dtype=jnp.int32)
    seen = jnp.ones(n_days, bool)
    value = float(forward_loglik_two_class(counts, seen, phi, lam_t, k))
    assert np.isfinite(value) and value < 0


# --- the lambda(t) basis --------------------------------------------------


def test_knot_basis_interpolates_and_partitions_unity():
    basis = knot_basis(100, 27)
    assert basis.shape[0] == 100
    assert np.allclose(basis.sum(axis=1), 1.0)
    assert (basis >= 0).all()
    # A constant on the knots gives a constant on the days.
    knots = np.full(basis.shape[1], 2.0)
    assert np.allclose(basis @ knots, 2.0)


def test_second_difference_penalty_is_zero_on_a_straight_line():
    assert float(second_difference_penalty(jnp.arange(10.0))) == pytest.approx(0.0)
    assert float(second_difference_penalty(jnp.asarray([0.0, 1.0, 0.0, 1.0]))) > 0


def test_moment_initialization_recovers_the_right_ballpark():
    phi, lam = 0.85, 1.0
    path = np.asarray(simulate_counts(jax.random.key(4), phi, jnp.full(6_000, lam), K))
    phi_hat, lam_hat = moment_initialization(path, np.ones(len(path), bool))
    assert phi_hat == pytest.approx(phi, abs=0.05)
    assert lam_hat == pytest.approx(lam, rel=0.3)


# --- the rotation kernel --------------------------------------------------


def test_disk_transit_ceiling_is_half_a_rotation():
    assert max_transit_days() == pytest.approx(SYNODIC_ROTATION_DAYS / 2, rel=1e-9)
    assert 13.0 < max_transit_days() < 14.0


def test_first_apparition_spans_cannot_exceed_one_transit():
    """The reference gives a returning group a new id, so this is the comparable quantity."""
    sim = simulate_marked_process(np.full(5_000, 1.2), 8.0, seed=1)
    assert sim.first_apparition_spans.max() <= np.ceil(max_transit_days())
    assert sim.visible_spans.max() > sim.first_apparition_spans.max(), (
        "unrestricted spans must show recurrence, which is the mismatch being avoided"
    )
    assert sim.n_recurrent > 0


def test_rotation_censors_lifetimes_producing_a_cliff():
    """A geometric lifetime plus rotation gives the sharp 13-14 day edge the data show."""
    sim = simulate_marked_process(np.full(8_000, 1.2), 8.0, seed=2)
    spans = sim.first_apparition_spans
    counts = np.bincount(spans.astype(int), minlength=20)
    assert counts[13] > 5 * counts[15:].sum() + 1, "mass must pile up just below the ceiling"
    assert spans.mean() < 8.0, "visible residence is shorter than intrinsic lifetime"


def test_some_groups_are_never_visible():
    """Far-side emergences are invisible to the reference too; that is the selection."""
    sim = simulate_marked_process(np.full(5_000, 1.0), 4.0, seed=3)
    assert sim.n_never_visible > 0
    assert sim.n_never_visible < len(sim.intrinsic_lifetimes)


def test_expected_visible_span_grows_with_intrinsic_lifetime_but_saturates():
    short = expected_visible_span(2.0, n=8_000, seed=0)
    medium = expected_visible_span(8.0, n=8_000, seed=0)
    long_lived = expected_visible_span(40.0, n=8_000, seed=0)
    assert short < medium < long_lived
    assert long_lived < max_transit_days() + 1, "one apparition cannot outlast a transit"


def test_lifetime_cap_is_documented_and_harmless():
    assert MAX_LIFETIME_DAYS > 10 * max_transit_days()


def test_weibull_lifetimes_keep_the_requested_mean():
    sim = simulate_marked_process(np.full(4_000, 2.0), 8.0, seed=5, lifetime="weibull")
    assert sim.intrinsic_lifetimes.mean() == pytest.approx(8.0, rel=0.1)


def test_unknown_lifetime_model_is_rejected():
    with pytest.raises(ValueError, match="unknown lifetime model"):
        simulate_marked_process(np.full(10, 1.0), 5.0, lifetime="lognormal")


# --- recovery (slow) ------------------------------------------------------


@pytest.mark.slow
def test_fit_recovers_phi_and_lambda_from_simulated_data():
    """Synthetic-recovery test for the Check A estimator, as CLAUDE.md requires.

    A cycle-like lambda(t) and a 25% missing-day calendar, which is roughly what the
    reference era looks like. The moment initialization starts well off (the cycle
    inflates the lag-1 autocorrelation); the MLE has to pull it back.
    """
    rng = np.random.default_rng(0)
    n_days = 6_000
    phi_true = 0.88
    days = np.arange(n_days)
    lam_true = np.exp(np.log(1.2) + 0.9 * np.sin(2 * np.pi * days / 2_000))
    path = np.asarray(simulate_counts(jax.random.key(3), phi_true, jnp.asarray(lam_true), K))
    observed = rng.random(n_days) > 0.25

    fit = fit_immigration_death(path, observed, k=K, knot_days=27, penalty=10.0)

    assert fit.success
    assert fit.phi[0] == pytest.approx(phi_true, abs=0.02)
    relative = np.abs(fit.lam_t - lam_true) / lam_true
    assert np.sqrt((relative**2).mean()) < 0.25
    assert fit.mean_visible_days[0] == pytest.approx(1 / (1 - phi_true), rel=0.2)


# --- the symmetric saddle in the two-class fit ----------------------------


def test_two_class_symmetric_point_is_a_saddle():
    """Identical classes are exchangeable, so antisymmetric gradients vanish there.

    This is the trap the two-class fitter has to avoid, and it is measured rather
    than argued: with phi_1 = phi_2 and weight = 1/2 the configuration maps to itself
    under swapping the labels, so no gradient can separate the classes.
    """
    k, n_days = 15, 1500
    shape = jnp.full(n_days, 1.1)
    left = np.asarray(simulate_counts(jax.random.key(0), 0.70, jnp.full(n_days, 0.9), k))
    right = np.asarray(simulate_counts(jax.random.key(1), 0.95, jnp.full(n_days, 0.25), k))
    counts = jnp.asarray(left + right)
    seen = jnp.ones(n_days, bool)

    def loglik(params):
        phi, weight = params[:2], params[2]
        lam = jnp.stack([shape * weight, shape * (1.0 - weight)], axis=1)
        return forward_loglik_two_class(counts, seen, phi, lam, k)

    grad = jax.grad(loglik)(jnp.asarray([0.84, 0.84, 0.5]))
    assert float(grad[0] - grad[1]) == pytest.approx(0.0, abs=1e-9)
    assert float(grad[2]) == pytest.approx(0.0, abs=1e-6)

    # Away from it the classes separate, so the saddle is the only obstacle.
    asymmetric = jax.grad(loglik)(jnp.asarray([0.75, 0.93, 0.5]))
    assert abs(float(asymmetric[0] - asymmetric[1])) > 1.0


def test_two_class_fit_does_not_start_on_the_saddle():
    """The guard: a two-class fit must begin with the classes already distinct."""
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 8, 400)
    observed = np.ones(400, dtype=bool)
    fit = fit_immigration_death(
        counts, observed, k=11, n_classes=2, k_per_class=8, knot_days=200, maxiter=1
    )
    assert fit.phi[0] != fit.phi[1], "identical starting phis cannot ever separate"


@pytest.mark.slow
def test_two_class_fit_recovers_two_distinct_lifetimes():
    """With the saddle avoided, the fitter finds classes that are really there."""
    k, n_days = 21, 4000
    short = np.asarray(simulate_counts(jax.random.key(0), 0.70, jnp.full(n_days, 1.2), k))
    long_lived = np.asarray(simulate_counts(jax.random.key(1), 0.95, jnp.full(n_days, 0.2), k))
    counts = short + long_lived
    observed = np.ones(n_days, dtype=bool)

    fit = fit_immigration_death(
        counts, observed, k=k, n_classes=2, k_per_class=k, knot_days=200, penalty=1.0, maxiter=200
    )
    assert fit.phi[0] != fit.phi[1]
    assert abs(fit.phi[1] - fit.phi[0]) > 0.05, f"the classes must stay distinct, got {fit.phi}"
