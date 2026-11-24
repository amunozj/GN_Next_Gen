"""Fitting a composite likelihood, and correcting the uncertainty it gets wrong.

A composite likelihood is a sum of terms that are each a valid likelihood but are
**not independent of one another**.  Here one calendar day seen by twelve observers
enters sixty-six pairwise terms, so the sum counts that day's information
sixty-six times and the curvature at the mode is correspondingly too sharp.  Left
alone, the intervals come out too narrow -- which is exactly the failure Step 2
measured in every chained baseline (about 1.4x too narrow for the three scalings,
6.6x for UEA21).

**This correction is derived here, not cited.**  ``docs/02_research/`` contains no
sandwich or Godambe derivation, no reference for one, and no magnitude estimate for
the over-counting.  What it contains is the mechanism, stated three times: the
assessment's "pairwise likelihoods double-count days seen by many observers
(overconfident unless corrected)"; ``counting_processes_underreporting.md``'s
"composite-likelihood posteriors need curvature adjustment for calibrated
uncertainty"; and Patz et al.'s warning that facets models "appear to give
infinitely precise proficiency as raters are added, since they ignore the
dependence among ratings of the same response".  So the construction below is this
project's own and its only validation is empirical, on the Step 2 benchmark, which
is what the Step 3 spec's acceptance criterion already asks for.

The correction
--------------
::

    H = -grad^2 log CL(theta_hat)        curvature at the mode
    G = Var(grad log CL)                 resampled over BLOCKS
    V = H^-1 G H^-1                      Godambe / sandwich

With independent terms ``G = H`` and ``V`` collapses to ``H^-1``, the ordinary
Laplace covariance -- so the correction is self-cancelling exactly when it is
unnecessary, which is the property that makes it safe to apply always.

**The block is a calendar year, not a day.**  The pair table is collapsed over
identical ``(edge, year, counts, lag)`` rows, which is what makes the real network
affordable, and that collapse deliberately discards day identity.  A year block
contains every one of its days, so resampling years captures the within-day
double counting *and* within-year correlation as well -- it is at least as
conservative as a day bootstrap, never less.

What the interval does and does not cover
-----------------------------------------
``reduced=True`` (the default) computes ``V`` over the per-observer level
parameters and the global overdispersion, holding the drift trajectory at its
posterior mode.  Carrying the drift innovations would mean a Hessian over ~15,000
parameters on the real network, and the drift's effect on the series is reported
the way the plan requires instead: as **prior sensitivity** over
``mu_sigma`` in {-5, -2, 0} (criterion A20), rather than folded into one interval.
Era-wide drift is not identified from sunspot data at all (identifiability trap 2),
so an interval that claimed to cover it would be claiming too much.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

#: Bootstrap resamples for the score variance.  200 on a real run, 100 on a
#: benchmark cell where the grid multiplies the cost by 51 worlds.
N_BOOTSTRAP = 200

#: An eigenmode of the curvature below this fraction of the largest is reported as
#: prior-dominated rather than data-constrained.  SDSS did the same thing and said
#: why: the weakly constrained modes were "projected out and then restored with
#: Gaussian priors", because they are where a least-squares solution invents
#: structure the data does not support.
WEAK_MODE_RATIO = 1e-8


@dataclass
class ModeFit:
    """The penalized-composite mode and what finding it cost."""

    theta: np.ndarray
    objective: float
    n_iterations: int
    success: bool
    message: str
    seconds: float
    meta: dict = field(default_factory=dict)


def fit_mode(
    objective: Callable,
    theta0: np.ndarray,
    *,
    maxiter: int = 600,
    tol: float = 1e-7,
) -> ModeFit:
    """L-BFGS-B on ``-objective``, with JAX supplying the gradient.

    ``objective`` returns the composite log-likelihood plus log prior, so this
    maximizes it.  float64 throughout: the objective is a sum of millions of terms
    and float32 rounding there runs to a nat or more (ADR-001), which is the same
    reason Step 1's forward filter pins float64.
    """
    import time

    value_and_grad = jax.jit(jax.value_and_grad(objective))

    def fun(vector):
        value, gradient = value_and_grad(jnp.asarray(vector))
        return -float(value), -np.asarray(gradient, dtype=float)

    started = time.time()
    result = minimize(
        fun,
        np.asarray(theta0, dtype=float),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": tol, "gtol": tol},
    )
    return ModeFit(
        theta=np.asarray(result.x, dtype=float),
        objective=-float(result.fun),
        n_iterations=int(result.nit),
        success=bool(result.success),
        message=str(result.message),
        seconds=time.time() - started,
        meta={"n_parameters": int(len(theta0)), "maxiter": maxiter},
    )


@dataclass
class Correction:
    """The Godambe sandwich and the diagnostics that say whether it was needed."""

    covariance: np.ndarray
    naive_covariance: np.ndarray
    indices: np.ndarray
    inflation: np.ndarray
    n_blocks: int
    n_bootstrap: int
    eigenvalues: np.ndarray
    n_weak_modes: int
    meta: dict = field(default_factory=dict)

    @property
    def median_inflation(self) -> float:
        """Median ratio of corrected to naive standard deviation.

        The number that says how overconfident the uncompensated composite
        likelihood was.  1.0 means the terms were effectively independent.
        """
        return float(np.nanmedian(self.inflation))


def godambe_correction(
    objective: Callable,
    theta: np.ndarray,
    *,
    indices: np.ndarray,
    block_score: Callable[[np.ndarray], np.ndarray],
    deterministic_objective: Callable | None = None,
    n_bootstrap: int = N_BOOTSTRAP,
    n_blocks: int | None = None,
    seed: int = 0,
    ridge: float = 1e-8,
) -> Correction:
    """``V = H^-1 G H^-1`` on the selected parameter block.

    Parameters
    ----------
    objective
        The full penalized composite log-likelihood, as a function of the full
        parameter vector.
    indices
        Which entries of the vector the covariance is computed over.  The others
        are held at ``theta``.
    block_score
        ``block_score(multiplicity) -> gradient`` over the reduced indices, where
        ``multiplicity`` is a per-block resampling weight.  This is what makes the
        bootstrap cheap: one gradient evaluation per resample, no refitting.
    deterministic_objective
        The part of ``objective`` that the bootstrap does **not** resample -- the
        priors and the smoothness penalty.  Required whenever the objective is
        penalized, and getting it wrong is not a detail: the curvature ``H``
        contains the penalty's own information while the bootstrapped score
        variance cannot, so a plain ``H^-1 G H^-1`` understates the variance by
        however much the penalty contributes.  Measured on a small network it came
        out **12x too narrow** -- the correction running backwards.  The meat is
        therefore ``G_likelihood + H_penalty``, which reduces to ``H`` exactly when
        the likelihood terms are independent, so the correction remains a no-op
        when it should be one.
    """
    indices = np.asarray(indices, dtype=np.int64)
    full = jnp.asarray(theta)

    def reduced_objective(sub):
        return objective(full.at[indices].set(sub))

    sub_theta = jnp.asarray(theta[indices])
    hessian = np.asarray(jax.hessian(reduced_objective)(sub_theta), dtype=float)
    curvature = -0.5 * (hessian + hessian.T)
    curvature = curvature + ridge * np.eye(len(indices)) * np.trace(curvature) / max(
        len(indices), 1
    )

    eigenvalues = np.linalg.eigvalsh(curvature)
    n_weak = int(np.count_nonzero(eigenvalues < WEAK_MODE_RATIO * eigenvalues.max()))

    naive = np.linalg.pinv(curvature, hermitian=True)

    if n_blocks is None:
        raise ValueError("n_blocks is required: the bootstrap resamples blocks")
    rng = np.random.default_rng(seed)
    scores = np.empty((n_bootstrap, len(indices)))
    for b in range(n_bootstrap):
        draw = rng.integers(0, n_blocks, n_blocks)
        multiplicity = np.bincount(draw, minlength=n_blocks).astype(float)
        scores[b] = np.asarray(block_score(multiplicity), dtype=float)
    score_variance = np.atleast_2d(np.cov(scores, rowvar=False))

    prior_curvature = np.zeros_like(curvature)
    if deterministic_objective is not None:

        def reduced_prior(sub):
            return deterministic_objective(full.at[indices].set(sub))

        prior_hessian = np.asarray(jax.hessian(reduced_prior)(sub_theta), dtype=float)
        prior_curvature = -0.5 * (prior_hessian + prior_hessian.T)

    meat = score_variance + prior_curvature
    covariance = naive @ meat @ naive
    covariance = 0.5 * (covariance + covariance.T)

    naive_sd = np.sqrt(np.maximum(np.diag(naive), 0.0))
    corrected_sd = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        inflation = np.where(naive_sd > 0, corrected_sd / naive_sd, np.nan)

    return Correction(
        covariance=covariance,
        naive_covariance=naive,
        indices=indices,
        inflation=inflation,
        n_blocks=int(n_blocks),
        n_bootstrap=int(n_bootstrap),
        eigenvalues=eigenvalues,
        n_weak_modes=n_weak,
        meta={
            "prior_information_share": float(
                np.trace(prior_curvature) / max(np.trace(curvature), 1e-300)
            ),
            "condition_number": float(
                eigenvalues.max() / max(eigenvalues[eigenvalues > 0].min(), 1e-300)
            )
            if np.any(eigenvalues > 0)
            else float("inf"),
            "median_inflation": float(np.nanmedian(inflation)),
            "max_inflation": float(np.nanmax(inflation)),
        },
    )


def expand_covariance(correction: Correction, n_parameters: int) -> np.ndarray:
    """Place a reduced covariance back into the full parameter space, zero elsewhere.

    The zeros are a statement, not a convenience: the drift trajectory is held at
    its mode and its effect on the series is reported as prior sensitivity instead
    (criterion A20).
    """
    full = np.zeros((n_parameters, n_parameters))
    index = correction.indices
    full[np.ix_(index, index)] = correction.covariance
    return full


def draw_parameters(
    theta: np.ndarray,
    covariance: np.ndarray,
    *,
    n_draws: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Gaussian draws at the mode, from a (possibly singular) covariance.

    An eigendecomposition rather than a Cholesky, because a composite-likelihood
    sandwich is routinely rank-deficient -- a floating connected component gives an
    exact null mode, one per component, which is the Laplacian degeneracy the graph
    diagnostics report. A Cholesky would simply fail there; this draws zero
    variation along those directions, which is the honest answer for a direction
    the data does not constrain.
    """
    rng = np.random.default_rng(seed)
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    values = np.maximum(values, 0.0)
    scale = vectors * np.sqrt(values)
    noise = rng.standard_normal((n_draws, len(theta)))
    return np.asarray(theta)[None, :] + noise @ scale.T


def eigen_report(correction: Correction) -> object:
    """The curvature spectrum, and which modes the prior rather than the data holds."""
    import pandas as pd

    values = np.sort(correction.eigenvalues)[::-1]
    largest = values.max() if len(values) else np.nan
    return pd.DataFrame(
        {
            "rank": np.arange(len(values)),
            "eigenvalue": values,
            "ratio_to_largest": values / largest,
            "prior_dominated": values < WEAK_MODE_RATIO * largest,
        }
    )
