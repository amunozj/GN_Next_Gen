"""The immigration-death chain that Step 4's population layer rests on.

``N_t`` is the number of groups a reference observer would count:

    survivors ~ Binomial(N_{t-1}, phi),    births ~ Poisson(lambda_t)

The transition factorizes as ``A_t = S . B_t`` (assessment section 4; research report
"A 41-state forward filter makes the 1610-present posterior tractable"):

* ``S`` is the fixed lower-triangular binomial-survival matrix, depending only on phi;
* ``B_t`` is an upper-triangular Toeplitz Poisson-birth matrix, depending only on
  lambda_t.

One step is therefore two matrix-vector products and the ``T x K x K`` transition
tensor is never built.  Truncation holds the tail mass at ``K - 1`` so every row is a
proper distribution; ``K = 41`` comes from a robust quantile of reported counts and
never from ``max(GROUPS)`` (issue **D17**).

``phi`` here is a **visible-residence** parameter, not an intrinsic decay rate: the
reference's ``obs_duration`` is censored by disk passage, so a group leaves the state
either by decaying or by rotating off the west limb.  :mod:`gn_next_gen.models.visibility`
carries the rotation kernel that makes that explicit.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln


def survival_matrix(phi: jax.Array | float, k: int) -> jax.Array:
    """``S[i, j] = Binomial(j; i, phi)`` -- lower triangular, rows sum to 1."""
    phi = jnp.clip(jnp.asarray(phi, dtype=jnp.float64), 1e-12, 1 - 1e-12)
    i = jnp.arange(k)[:, None]
    j = jnp.arange(k)[None, :]
    log_binom = gammaln(i + 1) - gammaln(j + 1) - gammaln(i - j + 1)
    log_p = log_binom + j * jnp.log(phi) + (i - j) * jnp.log1p(-phi)
    return jnp.where(j <= i, jnp.exp(log_p), 0.0)


def birth_matrix(lam: jax.Array | float, k: int) -> jax.Array:
    """``B[i, j] = Poisson(j - i; lam)`` with the tail mass held at ``j = k - 1``.

    Upper triangular and Toeplitz.  Putting the truncated tail in the last column
    keeps every row a distribution, so the filter's normalizer stays exact for the
    truncated model rather than leaking probability.
    """
    lam = jnp.clip(jnp.asarray(lam, dtype=jnp.float64), 1e-12, None)
    i = jnp.arange(k)[:, None]
    j = jnp.arange(k)[None, :]
    step = j - i
    log_pmf = step * jnp.log(lam) - lam - gammaln(step + 1)
    pmf = jnp.where(step >= 0, jnp.exp(log_pmf), 0.0)
    interior = jnp.where(j < k - 1, pmf, 0.0)
    tail = 1.0 - interior.sum(axis=1)
    return interior.at[:, k - 1].set(jnp.clip(tail, 0.0, 1.0))


def transition_matrix(phi: jax.Array | float, lam: jax.Array | float, k: int) -> jax.Array:
    """``A = S . B`` -- built explicitly, for tests and small-scale checks only."""
    return survival_matrix(phi, k) @ birth_matrix(lam, k)


def push_forward(
    alpha: jax.Array, phi: jax.Array | float, lam: jax.Array | float, k: int
) -> jax.Array:
    """One step of ``alpha . S . B`` without forming ``A``."""
    return (alpha @ survival_matrix(phi, k)) @ birth_matrix(lam, k)


def stationary_mean(phi: jax.Array | float, lam: jax.Array | float) -> jax.Array:
    """``lam / (1 - phi)`` -- the untruncated stationary mean of the chain.

    Exact for the untruncated process, whose stationary law is Poisson with this
    mean.  :func:`stationary_distribution` gives the truncated version.
    """
    return jnp.asarray(lam) / (1.0 - jnp.asarray(phi))


def stationary_distribution(phi: jax.Array | float, lam: jax.Array | float, k: int) -> jax.Array:
    """Poisson(``lam / (1 - phi)``) truncated to ``0 .. k - 1`` with tail mass at the top."""
    mean = stationary_mean(phi, lam)
    n = jnp.arange(k, dtype=jnp.float64)
    log_pmf = n * jnp.log(mean) - mean - gammaln(n + 1)
    pmf = jnp.exp(log_pmf)
    interior = pmf.at[k - 1].set(0.0)
    return interior.at[k - 1].set(jnp.clip(1.0 - interior.sum(), 0.0, 1.0))


def expected_lifetime_days(phi: jax.Array | float) -> jax.Array:
    """Mean residence of one group, ``1 / (1 - phi)`` days.

    Geometric, so it cannot by itself produce the sharp cliff the reference's
    visible durations show near disk transit -- which is why Check A compares
    against the rotation kernel rather than against this number.
    """
    return 1.0 / (1.0 - jnp.asarray(phi))


def phi_from_lifetime(days: jax.Array | float) -> jax.Array:
    """Inverse of :func:`expected_lifetime_days`."""
    return 1.0 - 1.0 / jnp.asarray(days)


def simulate_counts(
    key: jax.Array,
    phi: float,
    lam_t: jax.Array,
    k: int,
    n0: int | None = None,
) -> jax.Array:
    """Sample one path ``N_1..N_T`` of the chain on a daily grid.

    Starts from the stationary distribution unless ``n0`` is given, so a simulated
    series is comparable with an observed one from its first day.
    """
    lam_t = jnp.asarray(lam_t, dtype=jnp.float64)
    key_init, key_path = jax.random.split(key)
    if n0 is None:
        probs = stationary_distribution(phi, lam_t[0], k)
        start = jax.random.choice(key_init, k, p=probs)
    else:
        start = jnp.asarray(n0)

    def step(state, inputs):
        lam, subkey = inputs
        k_surv, k_birth = jax.random.split(subkey)
        survivors = jax.random.binomial(k_surv, state, phi).astype(jnp.int32)
        births = jax.random.poisson(k_birth, lam).astype(jnp.int32)
        new = jnp.clip(survivors + births, 0, k - 1)
        return new, new

    keys = jax.random.split(key_path, lam_t.shape[0])
    _, path = jax.lax.scan(step, start.astype(jnp.int32), (lam_t, keys))
    return path
