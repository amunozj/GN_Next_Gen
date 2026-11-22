"""The group-area law: a Weibull + log-normal mixture on log-area.

The size distribution of sunspot groups has **two populations**, and they are not
two log-normals.  Fitting two log-normals folds observational effects --
resolution, detection bias, the convolution of cadence with feature lifetime --
into the small-object component.  The form adopted here, fixed by **ADR-011**, is

    p(z) = pi_W * Weibull_{k,lambda}(z)  +  (1 - pi_W) * Normal(z; mu, sigma),
    z = ln A,

with the Weibull's density in log-area

    log p_W(z) = log k - k log(lambda) + k z - exp(k (z - log lambda)),

a left-skewed Gumbel.  The Weibull describes small objects (pores and ephemeral
regions, a fragmentation regime); the log-normal describes large ones (bipolar
active regions, multiplicative growth).  Ported from
``sunspot-homogenization/sunspot_pipeline/processing/calibration_models.py``
(``_log_pdf_latent_wln``, ``_emit_hard_truncated_quadrature``), whose per-survey
calibration this project uses as-is and never refits.

**Two weight conventions.**  ``pi_W`` here weights the *Weibull*, as upstream does.
Munoz-Jaramillo et al. (2015, ApJ 804:68) write ``c`` for the weight on the
*log-normal*, so ``c = 1 - pi_W``.  Their published power law
``c(AL) = 0.66 AL^0.230`` is on the log-normal weight.  Swapping the two silently
exchanges the populations, so :func:`lognormal_weight` makes the conversion
explicit at the call site.

**Activity level.**  The same paper finds the Weibull component independent of
activity level -- the magnetic baseline -- while the log-normal's parameters follow
power laws in it.  :func:`activity_level` computes it the paper's way: a six-month
Gaussian smoothing of daily total sunspot area.  Its **units are mHem** while the
snapshot's areas are muHem, so the conversion happens here rather than at the call
site; the published coefficients are meaningless in the wrong unit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, logsumexp
from scipy.stats import norm

from gn_next_gen.paths import AREA_LAW_POSTERIOR, REFERENCE_SURVEY_CALIBRATION

#: Quadrature grid on log-area, as upstream uses it.
Z_GRID_LO, Z_GRID_HI, Z_GRID_N = -10.0, 10.0, 256

#: Six-month Gaussian, the 2015 paper's activity-level filter.
ACTIVITY_SIGMA_DAYS = 182

#: Areas are millionths of the solar hemisphere; the paper's activity level is
#: thousandths.
MUHEM_PER_MHEM = 1000.0

#: Published activity-level dependence (2015, Sec. 7), for cross-checking only.
#: ``parameter = a * AL**b`` with AL in mHem; ``c_LN`` is the LOG-NORMAL weight.
PUBLISHED_2015 = {
    "k_W": 0.46,
    "lambda_W": 11.49,
    "mu_LN": (5.97, 0.021),
    "sigma_LN": (1.08, 0.117),
    "c_LN": (0.66, 0.230),
}


@dataclass(frozen=True)
class AreaLawParams:
    """One survey's Weibull + log-normal parameters."""

    survey: str
    k_W: float
    log_lambda_W: float
    mu_LN: float
    sigma_LN: float
    pi_W: float
    sigma_floor: float
    threshold_msh: float
    n_kept: int | None = None

    @property
    def lambda_W(self) -> float:
        return float(np.exp(self.log_lambda_W))

    @property
    def c_LN(self) -> float:
        """The 2015 paper's weight on the LOG-NORMAL component."""
        return 1.0 - self.pi_W

    def replace(self, **changes) -> AreaLawParams:
        return replace(self, **changes)


def lognormal_weight(pi_w):
    """Convert upstream's Weibull weight to the 2015 paper's log-normal weight ``c``."""
    return 1.0 - np.asarray(pi_w)


def z_grid(n: int = Z_GRID_N) -> np.ndarray:
    """The fixed quadrature grid on ``z = ln A``."""
    return np.linspace(Z_GRID_LO, Z_GRID_HI, n)


def log_pdf_weibull(z, k: float, log_lambda: float) -> np.ndarray:
    """``log p(z)`` for ``A ~ Weibull(k, lambda)`` pushed to ``z = ln A``."""
    z = np.asarray(z, dtype=float)
    return np.log(k) - k * log_lambda + k * z - np.exp(k * (z - log_lambda))


def log_pdf_latent(
    z, k: float, log_lambda: float, mu: float, sigma: float, pi_w: float
) -> np.ndarray:
    """``log p(z)`` for the Weibull + log-normal mixture on log-area."""
    z = np.asarray(z, dtype=float)
    log_w = np.log(np.clip(pi_w, 1e-300, 1.0)) + log_pdf_weibull(z, k, log_lambda)
    log_n = np.log(np.clip(1.0 - pi_w, 1e-300, 1.0)) + norm.logpdf(z, mu, sigma)
    return logsumexp(np.stack([log_w, log_n], axis=0), axis=0)


def log_pdf_truncated(
    log_area,
    params: AreaLawParams,
    *,
    sigma_row=0.0,
    y_trunc: float | None = None,
    n_grid: int = Z_GRID_N,
) -> np.ndarray:
    """Per-row hard-truncated log-likelihood, by quadrature over the latent ``z``.

    ``log p(y | y > y_trunc) = log p(y) - log P(y > y_trunc)``, both on the shared
    grid, with the normalisation computed **per row** because each row carries its own
    measurement error.  Follows upstream's ``_emit_hard_truncated_quadrature``; the
    observation model is ``y | z ~ Normal(z, sqrt(sigma_row^2 + sigma_floor^2))``.
    """
    observed = np.asarray(log_area, dtype=float)
    if y_trunc is None:
        y_trunc = float(np.log(params.threshold_msh))

    grid = z_grid(n_grid)
    log_dz = np.log(grid[1] - grid[0])
    log_p_z = log_pdf_latent(
        grid, params.k_W, params.log_lambda_W, params.mu_LN, params.sigma_LN, params.pi_W
    )
    sigma_y = np.sqrt(np.asarray(sigma_row, dtype=float) ** 2 + params.sigma_floor**2)
    sigma_y = np.broadcast_to(np.atleast_1d(sigma_y), observed.shape)
    sigma_y = np.maximum(sigma_y, 1e-6)

    log_p_y_given_z = norm.logpdf(observed[:, None], grid[None, :], sigma_y[:, None])
    log_p_y = logsumexp(log_p_z[None, :] + log_p_y_given_z + log_dz, axis=1)

    log_above_given_z = norm.logsf((y_trunc - grid[None, :]) / sigma_y[:, None])
    log_p_above = logsumexp(log_p_z[None, :] + log_above_given_z + log_dz, axis=1)
    return log_p_y - log_p_above


def fraction_above(threshold_msh: float, params: AreaLawParams, n_grid: int = 2048) -> float:
    """``P(A > threshold)`` under the latent law -- the functional ``q_o`` averages.

    This is what the detection layer consumes: an observer's effective detection
    fraction is an expectation of its acuity curve over this law, so if the law moves
    with activity, so does the observer's apparent k-factor.
    """
    grid = z_grid(n_grid)
    density = np.exp(
        log_pdf_latent(
            grid, params.k_W, params.log_lambda_W, params.mu_LN, params.sigma_LN, params.pi_W
        )
    )
    mass = np.trapezoid(density, grid)
    above = grid > np.log(threshold_msh)
    return float(np.trapezoid(density[above], grid[above]) / mass)


def component_crossover(params: AreaLawParams, n_grid: int = 4096) -> float:
    """Area (MSH) where the weighted Weibull and log-normal densities cross.

    The physically motivated boundary between the two populations, and so the split a
    two-class latent chain should use instead of a round number.  ``nan`` when the
    components do not cross inside the grid, which happens when a weight is degenerate.
    """
    grid = z_grid(n_grid)
    log_w = np.log(np.clip(params.pi_W, 1e-300, 1.0)) + log_pdf_weibull(
        grid, params.k_W, params.log_lambda_W
    )
    log_n = np.log(np.clip(1.0 - params.pi_W, 1e-300, 1.0)) + norm.logpdf(
        grid, params.mu_LN, params.sigma_LN
    )
    diff = log_w - log_n
    crossings = np.nonzero(np.diff(np.sign(diff)))[0]
    if len(crossings) == 0:
        return float("nan")
    # The last crossing is the one of interest: below it the Weibull dominates.
    i = crossings[-1]
    span = diff[i + 1] - diff[i]
    frac = 0.0 if span == 0 else -diff[i] / span
    return float(np.exp(grid[i] + frac * (grid[i + 1] - grid[i])))


def load_upstream_parameters(
    survey: str, scale: str = "total", *, path=AREA_LAW_POSTERIOR
) -> AreaLawParams:
    """One survey's frozen upstream parameters, with its truncation threshold."""
    table = pd.read_csv(path)
    row = table[(table["survey"] == survey) & (table["scale"] == scale)]
    if row.empty:
        available = sorted(set(zip(table["survey"], table["scale"], strict=True)))
        raise KeyError(f"no area law for ({survey}, {scale}); have {available}")
    row = row.iloc[0]

    calibration = pd.read_csv(REFERENCE_SURVEY_CALIBRATION)
    threshold = calibration.loc[calibration["survey"] == survey, "threshold_MSH"]
    kept = row.get("n_kept_R")
    return AreaLawParams(
        survey=survey,
        k_W=float(row["k_W_med"]),
        log_lambda_W=float(row["log_lambda_W_med"]),
        mu_LN=float(row["mu_LN_med"]),
        sigma_LN=float(row["sigma_LN_med"]),
        pi_W=float(row["pi_W_med"]),
        sigma_floor=float(row["sigma_floor_med"]),
        threshold_msh=float(threshold.iloc[0]) if len(threshold) else float("nan"),
        n_kept=int(kept) if kept is not None and pd.notna(kept) else None,
    )


# --- activity level --------------------------------------------------------


def gaussian_smooth(values, sigma_days: float) -> np.ndarray:
    """NaN-aware Gaussian smoothing on a contiguous daily grid."""
    values = np.asarray(values, dtype=float)
    half = int(np.ceil(4 * sigma_days))
    offsets = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (offsets / sigma_days) ** 2)
    kernel /= kernel.sum()
    finite = np.where(np.isnan(values), 0.0, values)
    weight = (~np.isnan(values)).astype(float)
    num = np.convolve(finite, kernel, mode="same")
    den = np.convolve(weight, kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def activity_level(
    day_status: pd.DataFrame,
    groups: pd.DataFrame | None = None,
    *,
    scope: str = "whole",
    weighting: str = "emergence",
    sigma_days: float = ACTIVITY_SIGMA_DAYS,
) -> pd.Series:
    """Activity level in **mHem**, indexed by ``date_ord``.

    Two weightings, and the difference is a factor of about six, so the choice is not
    cosmetic -- the published power-law coefficients attach to one of them only:

    ``"emergence"`` (default, **the 2015 paper's definition**)
        One area per group -- its largest, on its canonical day -- summed per day and
        smoothed.  This is what the paper's data selection produces ("we assign to the
        group the single largest reported area in all its days of observation"), and it
        measures an emergence *rate*, not an instantaneous disk area.  Requires
        ``groups`` (``group_single_entry``).  RGO peaks at 0.50 mHem whole-Sun and
        about 0.25 per hemisphere, matching the paper's 0-0.4 mHem axis.
    ``"instantaneous"``
        Every group present on the day, from ``day_status.area_total`` -- the actual
        area on the disk.  RGO peaks at 3.24 mHem, roughly six times the emergence
        weighting, because long-lived groups are counted on every day they are visible.
        This is the natural covariate for the per-group-*day* area law.

    ``scope="whole"`` carries no positional information and may feed the model
    (ADR-004); ``"north"``/``"south"`` reproduce the paper exactly but are **diagnostic
    only**, since hemisphere assignment comes from positions.
    """
    frame = day_status.sort_values("date_ord")
    observed = frame[frame["day_status"] > 0]
    first, last = int(observed["date_ord"].min()), int(observed["date_ord"].max())
    grid = np.arange(first, last + 1)
    seen = np.zeros(grid.shape, dtype=bool)
    seen[observed["date_ord"].to_numpy() - first] = True

    if weighting == "instantaneous":
        column = {"whole": "area_total", "north": "area_north", "south": "area_south"}[scope]
        series = np.full(grid.shape, np.nan)
        series[observed["date_ord"].to_numpy() - first] = observed[column].to_numpy()
    elif weighting == "emergence":
        if groups is None:
            raise ValueError(
                "weighting='emergence' needs the single-entry group table; pass "
                "load_group_single_entry(...) as `groups`"
            )
        selected = groups
        if scope != "whole":
            if "hemisphere" not in selected.columns:
                raise ValueError("hemispheric activity needs the `hemisphere` column")
            sign = 1.0 if scope == "north" else -1.0
            selected = selected[np.sign(selected["hemisphere"]) == sign]
        series = np.zeros(grid.shape)
        index = selected["date_ord"].to_numpy().astype(int) - first
        inside = (index >= 0) & (index < len(grid))
        np.add.at(series, index[inside], selected["area_corr"].to_numpy()[inside])
        series[~seen] = np.nan
    else:
        raise ValueError(
            f"unknown weighting {weighting!r}; expected 'emergence' or 'instantaneous'"
        )

    smoothed = gaussian_smooth(series, sigma_days) / MUHEM_PER_MHEM
    return pd.Series(smoothed, index=grid, name="activity_level")


def published_power_law(name: str, activity) -> np.ndarray:
    """Evaluate a 2015 published power law, for overlaying on the refit."""
    a, b = PUBLISHED_2015[name]
    return a * np.power(np.asarray(activity, dtype=float), b)


class TruncatedLikelihood:
    """Pre-compiled truncated likelihood for one fixed set of observations.

    The expensive part of :func:`log_pdf_truncated` is the ``(N, G)`` matrix of
    ``p(y_i | z_g)`` and its tail counterpart -- and **neither depends on the fitted
    parameters**, only on the data, the measurement errors and the grid.  Building them
    once and reducing each evaluation to two matrix-vector products turns a fit that
    took minutes into one that takes a second, with no approximation: the arithmetic is
    the same, in the same order, in float64.

    The reduction is done in linear space.  Densities here are O(1) and the truncated
    likelihood is a ratio, so the common scale cancels; the guard against a vanishing
    denominator is explicit.
    """

    def __init__(
        self, log_area, sigma_row, y_trunc: float, *, sigma_floor: float, n_grid: int = Z_GRID_N
    ):
        observed = np.asarray(log_area, dtype=float)
        self.grid = z_grid(n_grid)
        self.dz = float(self.grid[1] - self.grid[0])
        self.n_obs = observed.size

        sigma = np.sqrt(np.asarray(sigma_row, dtype=float) ** 2 + sigma_floor**2)
        sigma = np.maximum(np.broadcast_to(np.atleast_1d(sigma), observed.shape), 1e-6)

        standardized = (observed[:, None] - self.grid[None, :]) / sigma[:, None]
        self._pdf = np.exp(-0.5 * standardized**2) / (sigma[:, None] * np.sqrt(2 * np.pi))
        # P(y > y_trunc | z) for each row's own measurement error.
        self._tail = np.exp(log_ndtr((self.grid[None, :] - y_trunc) / sigma[:, None]))

    def __call__(self, params: AreaLawParams) -> float:
        """Total truncated log-likelihood for these parameters."""
        density = np.exp(
            log_pdf_latent(
                self.grid,
                params.k_W,
                params.log_lambda_W,
                params.mu_LN,
                params.sigma_LN,
                params.pi_W,
            )
        )
        weighted = density * self.dz
        p_y = self._pdf @ weighted
        p_above = self._tail @ weighted
        if not (np.all(p_y > 0) and np.all(p_above > 0)):
            return -np.inf
        return float(np.sum(np.log(p_y) - np.log(p_above)))
