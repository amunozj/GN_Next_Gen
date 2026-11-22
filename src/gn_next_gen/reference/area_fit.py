"""Fitting the Weibull + log-normal area law, in the 2015 paper's two stages.

Stage 1 lets every parameter vary per activity-level bin and asks whether the
Weibull component moves with activity.  The 2015 paper finds it does not -- the
"magnetic baseline" -- and that is the finding that licenses stage 2, which holds
``(k, lambda)`` fixed and fits only the log-normal's parameters and the mixing
weight, then describes their activity dependence with power laws.

Nothing here refits what upstream already calibrated: ``(k, lambda)`` in stage 2
are upstream's per-survey values, and the free parameters are the ones the paper
found to vary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from gn_next_gen.reference.area_law import AreaLawParams, TruncatedLikelihood


@dataclass
class BinFit:
    """One activity-level bin's fit."""

    survey: str
    bin_index: int
    activity: float
    n_groups: int
    params: AreaLawParams
    loglik: float
    n_free: int
    converged: bool
    stage: str
    activity_lo: float = np.nan
    activity_hi: float = np.nan

    @property
    def aic(self) -> float:
        return 2 * self.n_free - 2 * self.loglik

    def as_row(self) -> dict:
        return {
            "survey": self.survey,
            "stage": self.stage,
            "bin_index": self.bin_index,
            "activity": self.activity,
            "activity_lo": self.activity_lo,
            "activity_hi": self.activity_hi,
            "n_groups": self.n_groups,
            "k_W": self.params.k_W,
            "lambda_W": self.params.lambda_W,
            "mu_LN": self.params.mu_LN,
            "sigma_LN": self.params.sigma_LN,
            "pi_W": self.params.pi_W,
            "c_LN": self.params.c_LN,
            "loglik": self.loglik,
            "n_free": self.n_free,
            "aic": self.aic,
            "converged": self.converged,
        }


def fit_bin(
    log_area: np.ndarray,
    sigma_row: np.ndarray,
    base: AreaLawParams,
    *,
    free_weibull: bool,
    maxiter: int = 2000,
    n_restarts: int = 6,
) -> tuple[AreaLawParams, float, bool]:
    """Maximum truncated likelihood for one bin.

    With ``free_weibull`` all five shape parameters vary (stage 1); without it,
    ``(k, lambda)`` stay at ``base``'s upstream values and only ``mu``, ``sigma`` and
    the weight move (stage 2, the paper's Sec. 7 procedure).
    """
    likelihood = TruncatedLikelihood(
        log_area,
        sigma_row,
        float(np.log(base.threshold_msh)),
        sigma_floor=base.sigma_floor,
    )

    def unpack(x: np.ndarray) -> AreaLawParams:
        if free_weibull:
            k, log_lambda, mu, log_sigma, logit_pi = x
        else:
            mu, log_sigma, logit_pi = x
            k, log_lambda = base.k_W, base.log_lambda_W
        return base.replace(
            k_W=float(np.clip(k, 0.05, 5.0)),
            log_lambda_W=float(np.clip(log_lambda, -4.0, 8.0)),
            mu_LN=float(mu),
            sigma_LN=float(np.exp(np.clip(log_sigma, -4.0, 2.0))),
            pi_W=float(1.0 / (1.0 + np.exp(-np.clip(logit_pi, -12.0, 12.0)))),
        )

    def objective(x: np.ndarray) -> float:
        value = likelihood(unpack(x))
        return 1e12 if not np.isfinite(value) else -value

    start_common = [base.mu_LN, np.log(base.sigma_LN), float(np.log(base.pi_W / (1 - base.pi_W)))]
    x0 = np.array(([base.k_W, base.log_lambda_W] if free_weibull else []) + start_common)

    # Restarts, because above the truncation the Weibull component is only weakly
    # identified and a single Nelder-Mead run can settle in a local optimum. Each
    # evaluation costs ~0.3 ms, so being thorough here is free; without it, "the
    # Weibull parameters wander with activity" could be a statement about the
    # optimizer rather than about the Sun.
    rng = np.random.default_rng(0)
    starts = [x0]
    if free_weibull:
        for _ in range(n_restarts):
            jitter = rng.normal(0.0, [0.15, 0.8, 0.2, 0.2, 0.8][: len(x0)])
            starts.append(x0 + jitter)

    best, best_value, converged = None, np.inf, False
    for start in starts:
        result = minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "fatol": 1e-6, "xatol": 1e-6},
        )
        if result.fun < best_value:
            best, best_value, converged = result.x, float(result.fun), bool(result.success)
    return unpack(best), float(-best_value), converged


def activity_bins(activity: np.ndarray, n_bins: int, *, equal_count: bool = True):
    """Bin indices and edges.

    Equal-count bins by default: activity is strongly right-skewed, so equal-width
    bins would leave the top bins with too few groups to fit, which is visible as the
    wide error bars in the 2015 paper's Figure 5.
    """
    activity = np.asarray(activity, dtype=float)
    finite = np.isfinite(activity)
    if equal_count:
        edges = np.quantile(activity[finite], np.linspace(0, 1, n_bins + 1))
    else:
        edges = np.linspace(activity[finite].min(), activity[finite].max(), n_bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    index = np.full(activity.shape, -1, dtype=int)
    index[finite] = np.clip(np.digitize(activity[finite], edges[1:-1]), 0, n_bins - 1)
    return index, edges


def fit_by_activity(
    groups: pd.DataFrame,
    base: AreaLawParams,
    *,
    n_bins: int = 10,
    free_weibull: bool,
    stage: str,
    min_groups: int = 200,
) -> list[BinFit]:
    """Fit every activity bin of one survey."""
    index, edges = activity_bins(groups["activity"].to_numpy(), n_bins)
    fits: list[BinFit] = []
    for b in range(n_bins):
        subset = groups[index == b]
        if len(subset) < min_groups:
            continue
        log_area = np.log(subset["area_corr"].to_numpy())
        sigma = np.nan_to_num(
            subset["area_corr_unc"].to_numpy() / subset["area_corr"].to_numpy(),
            nan=0.0,
            posinf=0.0,
        )
        params, loglik, ok = fit_bin(log_area, sigma, base, free_weibull=free_weibull)
        fits.append(
            BinFit(
                survey=base.survey,
                bin_index=b,
                activity=float(subset["activity"].median()),
                activity_lo=float(edges[b])
                if np.isfinite(edges[b])
                else float(subset["activity"].min()),
                activity_hi=float(edges[b + 1])
                if np.isfinite(edges[b + 1])
                else float(subset["activity"].max()),
                n_groups=len(subset),
                params=params,
                loglik=loglik,
                n_free=5 if free_weibull else 3,
                converged=ok,
                stage=stage,
            )
        )
    return fits


@dataclass
class PowerLaw:
    """``y = a * AL**b`` fitted in log-log space."""

    name: str
    a: float
    b: float
    spearman: float
    p_value: float
    reduced_chi2: float
    published: tuple[float, float] | None = None
    meta: dict = field(default_factory=dict)

    def predict(self, activity):
        return self.a * np.power(np.asarray(activity, dtype=float), self.b)

    def as_row(self) -> dict:
        row = {
            "parameter": self.name,
            "a": self.a,
            "b": self.b,
            "spearman_rho": self.spearman,
            "p_value": self.p_value,
            "reduced_chi2": self.reduced_chi2,
        }
        if self.published:
            row["published_a"], row["published_b"] = self.published
            row["a_ratio"] = self.a / self.published[0]
        return row | self.meta


def fit_power_law(activity, values, name: str, published=None, weights=None) -> PowerLaw:
    """Least squares in log-log space, with Spearman's rho as the paper reports it."""
    from scipy.stats import spearmanr

    activity = np.asarray(activity, dtype=float)
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(activity) & np.isfinite(values) & (activity > 0) & (values > 0)
    activity, values = activity[ok], values[ok]
    if len(activity) < 3:
        return PowerLaw(name, np.nan, np.nan, np.nan, np.nan, np.nan, published)

    weight = np.ones_like(values) if weights is None else np.asarray(weights)[ok]
    design = np.column_stack([np.ones_like(activity), np.log(activity)])
    sqrt_w = np.sqrt(weight)
    coefficients, *_ = np.linalg.lstsq(
        design * sqrt_w[:, None], np.log(values) * sqrt_w, rcond=None
    )
    a, b = float(np.exp(coefficients[0])), float(coefficients[1])

    rho, p_value = spearmanr(activity, values)
    predicted = a * activity**b
    residual = values - predicted
    dof = max(len(values) - 2, 1)
    scatter = np.std(residual, ddof=1) if len(residual) > 1 else 1.0
    chi2 = float(np.sum((residual / max(scatter, 1e-12)) ** 2) / dof)
    return PowerLaw(name, a, b, float(rho), float(p_value), chi2, published)
