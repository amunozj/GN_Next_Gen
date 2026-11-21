"""Exact forward filter over the latent daily count, and the Check A fitter.

This is the engine Step 4 needs, built here because Check A needs it anyway (the
plan's "fitting method: forward-filter MLE with moment initialization").  Three
properties matter and are tested:

* **exact**, not approximate -- checked against a dense brute-force transition at
  small ``K``;
* **float64 accumulation** of the log-likelihood.  The total is a sum of 1e4-1e7
  terms and float32 rounding there runs to 0.1-1 nat, enough to disturb an optimizer
  or a Metropolis acceptance (ADR-001; research report);
* **the observing calendar is respected**: an unobserved day contributes a flat
  emission, so the chain propagates across gaps instead of being interpolated. That
  is the whole reason a filter is needed to fit a near-perfectly-observed reference.

Two latent structures are supported.  One area class is a ``K``-state chain; two
classes are a pair of chains whose **total** is observed, so the split is latent and
the transition stays Kronecker-factorized -- the cheapest exact fix for the fact that
day-by-day independent thinning ignores that large groups persist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from gn_next_gen.models.dynamics import (
    birth_matrix,
    stationary_distribution,
    survival_matrix,
)

jax.config.update("jax_enable_x64", True)

LOG_ZERO = -1e30


def count_emissions(counts: jax.Array, observed: jax.Array, k: int) -> jax.Array:
    """``log P(y_t | N_t = n)`` for a directly observed count.

    The reference survey *is* the observer in Check A, so an observed day pins the
    state exactly and an unobserved day is flat.  Step 4 replaces this with
    per-observer soft-threshold emission tables; the filter does not change.
    """
    counts = jnp.asarray(counts)
    observed = jnp.asarray(observed)
    states = jnp.arange(k)
    match = counts[:, None] == states[None, :]
    log_emit = jnp.where(match, 0.0, LOG_ZERO)
    return jnp.where(observed[:, None], log_emit, 0.0)


def forward_loglik(
    log_emissions: jax.Array,
    phi: jax.Array | float,
    lam_t: jax.Array,
    k: int,
    initial: jax.Array | None = None,
) -> jax.Array:
    """Total log-likelihood by the scaled forward recursion, accumulated in float64.

    ``alpha_t = (alpha_{t-1} S) B_t`` -- two matrix-vector products per day, with the
    ``T x K x K`` tensor never formed.
    """
    lam_t = jnp.asarray(lam_t, dtype=jnp.float64)
    log_emissions = jnp.asarray(log_emissions, dtype=jnp.float64)
    surv = survival_matrix(phi, k)
    alpha0 = stationary_distribution(phi, lam_t[0], k) if initial is None else initial

    def step(carry, inputs):
        alpha, loglik = carry
        lam, log_emit = inputs
        alpha = (alpha @ surv) @ birth_matrix(lam, k)
        alpha = alpha * jnp.exp(log_emit)
        total = alpha.sum()
        safe = jnp.maximum(total, 1e-300)
        return (alpha / safe, loglik + jnp.log(safe)), None

    (_, loglik), _ = jax.lax.scan(
        jax.checkpoint(step), (alpha0, jnp.float64(0.0)), (lam_t, log_emissions)
    )
    return loglik


def forward_loglik_two_class(
    counts: jax.Array,
    observed: jax.Array,
    phi: jax.Array,
    lam_t: jax.Array,
    k: int,
) -> jax.Array:
    """Two latent classes whose **total** is observed; the split stays latent.

    The joint state is ``k x k`` and the transition is ``kron(A_small, A_large)``,
    applied as ``S_1' M B_1' ...`` on the reshaped array, so the cost is
    ``O(k^3)`` per day rather than ``O(k^4)``.
    """
    lam_t = jnp.asarray(lam_t, dtype=jnp.float64)  # (T, 2)
    counts = jnp.asarray(counts)
    observed = jnp.asarray(observed)
    surv = [survival_matrix(phi[c], k) for c in range(2)]

    grid = jnp.arange(k)[:, None] + jnp.arange(k)[None, :]

    init = jnp.outer(
        stationary_distribution(phi[0], lam_t[0, 0], k),
        stationary_distribution(phi[1], lam_t[0, 1], k),
    )

    def step(carry, inputs):
        alpha, loglik = carry
        lam, count, seen = inputs
        alpha = surv[0].T @ alpha @ surv[1]
        alpha = birth_matrix(lam[0], k).T @ alpha @ birth_matrix(lam[1], k)
        mask = jnp.where(grid == count, 1.0, 0.0)
        alpha = jnp.where(seen, alpha * mask, alpha)
        total = alpha.sum()
        safe = jnp.maximum(total, 1e-300)
        return (alpha / safe, loglik + jnp.log(safe)), None

    (_, loglik), _ = jax.lax.scan(
        jax.checkpoint(step), (init, jnp.float64(0.0)), (lam_t, counts, observed)
    )
    return loglik


# --- lambda(t) as a penalized spline on a knot grid ------------------------


def knot_basis(n_days: int, knot_days: int) -> np.ndarray:
    """Linear interpolation matrix from knot values to days.

    A 27-day knot spacing (one solar rotation) lets lambda follow the cycle without
    chasing daily noise; the second-difference penalty below does the rest.
    """
    n_knots = max(int(np.ceil(n_days / knot_days)) + 1, 2)
    positions = np.linspace(0, n_days - 1, n_knots)
    days = np.arange(n_days)
    basis = np.zeros((n_days, n_knots))
    idx = np.clip(np.searchsorted(positions, days, side="right") - 1, 0, n_knots - 2)
    left, right = positions[idx], positions[idx + 1]
    weight = (days - left) / np.maximum(right - left, 1e-9)
    basis[days, idx] = 1.0 - weight
    basis[days, idx + 1] = weight
    return basis


def second_difference_penalty(values: jax.Array) -> jax.Array:
    """Sum of squared second differences -- smoothness of ``log lambda`` on the knots."""
    if values.shape[0] < 3:
        return jnp.float64(0.0)
    return jnp.sum(jnp.diff(values, n=2) ** 2)


@dataclass
class FitResult:
    """What a Check A fit produced, and enough to reproduce it."""

    phi: np.ndarray
    log_lambda_knots: np.ndarray
    lam_t: np.ndarray
    loglik: float
    penalty: float
    objective: float
    n_classes: int
    k: int
    class_weight: float | None = None
    holdout_loglik: float | None = None
    success: bool = True
    message: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def mean_visible_days(self) -> np.ndarray:
        """``1 / (1 - phi)`` per class -- visible residence, not intrinsic lifetime."""
        return 1.0 / (1.0 - self.phi)


def moment_initialization(counts: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    """``(phi, mean lambda)`` from the lag-1 autocorrelation and the mean.

    For this chain ``rho(1) = phi``, so the lag-1 correlation of consecutive observed
    days is a direct estimate, and ``lambda = mean * (1 - phi)`` follows from the
    stationary mean.  Both are crude -- the cycle inflates ``rho(1)`` -- but they put
    L-BFGS in the right basin.
    """
    seen = np.asarray(observed, dtype=bool)
    values = np.where(seen, np.asarray(counts, dtype=float), np.nan)
    pairs = np.column_stack([values[:-1], values[1:]])
    pairs = pairs[~np.isnan(pairs).any(axis=1)]
    if len(pairs) < 10:
        return 0.85, max(float(np.nanmean(values)) * 0.15, 0.05)
    rho = float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1])
    phi = float(np.clip(rho, 0.2, 0.97))
    mean = float(np.nanmean(values))
    return phi, max(mean * (1.0 - phi), 1e-3)


def fit_immigration_death(
    counts: np.ndarray,
    observed: np.ndarray,
    *,
    k: int = 41,
    n_classes: int = 1,
    k_per_class: int | None = None,
    knot_days: int = 27,
    penalty: float = 10.0,
    holdout_mask: np.ndarray | None = None,
    maxiter: int = 300,
) -> FitResult:
    """Maximum penalized likelihood for ``(phi, log lambda(t))``, and the class split.

    Parameters
    ----------
    k_per_class
        State-space size for *each* class of a two-class fit.  The joint space is
        ``k_per_class ** 2``, so the full ``k`` would cost 1,681 states where two
        chains each reaching ~20 suffice; the default halves ``k`` and rounds up.
        The sum still reaches ``2 * (k_per_class - 1)``, well past the counts seen.

    Returns
    -------
    FitResult
        With ``holdout_loglik`` filled in when ``holdout_mask`` is given, which is how
        the penalty strength is chosen rather than guessed.
    """
    from scipy.optimize import minimize

    counts = np.asarray(counts, dtype=np.int32)
    observed = np.asarray(observed, dtype=bool)
    k_class = k_per_class or (k + 1) // 2
    fit_mask = observed if holdout_mask is None else observed & ~np.asarray(holdout_mask, bool)

    basis_np = knot_basis(len(counts), knot_days)
    basis = jnp.asarray(basis_np)
    n_knots = basis_np.shape[1]

    phi0, lam0 = moment_initialization(counts, fit_mask)
    x0 = np.concatenate(
        [
            _logit(np.full(n_classes, phi0)),
            np.full(n_knots, np.log(lam0)),
            [0.0] if n_classes == 2 else [],
        ]
    )

    counts_j = jnp.asarray(counts)
    fit_j = jnp.asarray(fit_mask)

    def unpack(x):
        phi = _sigmoid(x[:n_classes])
        log_knots = x[n_classes : n_classes + n_knots]
        weight = _sigmoid(x[-1]) if n_classes == 2 else None
        return phi, log_knots, weight

    def objective(x, mask):
        phi, log_knots, weight = unpack(x)
        lam = jnp.exp(basis @ log_knots)
        if n_classes == 1:
            emissions = count_emissions(counts_j, mask, k)
            loglik = forward_loglik(emissions, phi[0], lam, k)
        else:
            lam_two = jnp.stack([lam * weight, lam * (1.0 - weight)], axis=1)
            loglik = forward_loglik_two_class(counts_j, mask, phi, lam_two, k_class)
        return -(loglik - penalty * second_difference_penalty(log_knots))

    value_and_grad = jax.jit(jax.value_and_grad(lambda x: objective(x, fit_j)))

    def scipy_fun(x):
        value, grad = value_and_grad(jnp.asarray(x))
        return float(value), np.asarray(grad, dtype=float)

    result = minimize(scipy_fun, x0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})

    phi, log_knots, weight = unpack(jnp.asarray(result.x))
    lam_t = np.asarray(jnp.exp(basis @ log_knots))
    pen = float(second_difference_penalty(log_knots))
    loglik = float(-result.fun + penalty * pen)

    holdout = None
    if holdout_mask is not None:
        held = jnp.asarray(observed & np.asarray(holdout_mask, bool))
        holdout = float(-objective(jnp.asarray(result.x), held) + penalty * pen)

    return FitResult(
        phi=np.asarray(phi, dtype=float),
        log_lambda_knots=np.asarray(log_knots, dtype=float),
        lam_t=lam_t,
        loglik=loglik,
        penalty=pen,
        objective=float(result.fun),
        n_classes=n_classes,
        k=k,
        class_weight=None if weight is None else float(weight),
        holdout_loglik=holdout,
        success=bool(result.success),
        message=str(result.message),
        meta={
            "knot_days": knot_days,
            "penalty": penalty,
            "n_iter": int(result.nit),
            "phi_init": phi0,
            "lam_init": lam0,
            "k_per_class": k_class if n_classes == 2 else None,
        },
    )


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-jnp.asarray(x)))


def brute_force_loglik(
    counts: np.ndarray, observed: np.ndarray, phi: float, lam_t: np.ndarray, k: int
) -> float:
    """Dense reference implementation for the exactness test.

    Builds every ``K x K`` transition and multiplies them out in NumPy, which is what
    the factorized scan must agree with.  Only usable at tiny ``K`` and ``T``.
    """
    from gn_next_gen.models.dynamics import transition_matrix

    alpha = np.asarray(stationary_distribution(phi, lam_t[0], k), dtype=float)
    loglik = 0.0
    for t in range(len(counts)):
        alpha = alpha @ np.asarray(transition_matrix(phi, lam_t[t], k), dtype=float)
        if observed[t]:
            emit = np.zeros(k)
            if counts[t] < k:
                emit[counts[t]] = 1.0
            alpha = alpha * emit
        total = alpha.sum()
        loglik += np.log(max(total, 1e-300))
        alpha = alpha / max(total, 1e-300)
    return loglik
