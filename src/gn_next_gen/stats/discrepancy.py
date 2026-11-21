"""Discrepancy measures for Check A, and the summary statistics they rest on.

The five measures are fixed before any fit runs (the plan's pre-registration); their
*thresholds* are advisory on this first pass, by the PI's decision of 2026-11-21, so
the binding go/no-go rule is set in the results doc from the measured magnitudes.
Each one is reported with its target so the two can never be confused.

Series are handled on a **contiguous daily grid with NaN for unobserved days**.  That
matters: the record's gaps are informative (a spotless day is less likely to be
recorded than an active one), and a measure computed on the compacted series would
quietly treat two days separated by a month as adjacent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

SPOTLESS, ACTIVE = 0, 1


def smooth_centered(values: np.ndarray, window_days: int) -> np.ndarray:
    """NaN-aware centred moving average; the activity variable Check A conditions on."""
    values = np.asarray(values, dtype=float)
    finite = np.where(np.isnan(values), 0.0, values)
    weight = (~np.isnan(values)).astype(float)
    kernel = np.ones(window_days)
    num = np.convolve(finite, kernel, mode="same")
    den = np.convolve(weight, kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return out


def autocorrelation(
    values: np.ndarray, max_lag: int = 30, detrend_days: int | None = None
) -> np.ndarray:
    """ACF at lags ``1..max_lag`` over pairs where both days were observed.

    With ``detrend_days`` the series first has its own centred moving mean over that
    window subtracted, which removes the solar cycle.  Without it the
    long-lag ACF mostly measures the cycle envelope, so the ~13-day trough and the
    ~27-day rotation bump cannot be read off it.
    """
    values = np.asarray(values, dtype=float)
    if detrend_days:
        values = values - smooth_centered(values, detrend_days)
    out = np.full(max_lag, np.nan)
    for lag in range(1, max_lag + 1):
        a, b = values[:-lag], values[lag:]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 30:
            a, b = a[ok], b[ok]
            if a.std() > 0 and b.std() > 0:
                out[lag - 1] = float(np.corrcoef(a, b)[0, 1])
    return out


def activity_bins(activity: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Quantile bin index per day, ``-1`` where activity is unknown."""
    activity = np.asarray(activity, dtype=float)
    finite = np.isfinite(activity)
    edges = np.quantile(activity[finite], np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    out = np.full(activity.shape, -1, dtype=int)
    out[finite] = np.clip(np.digitize(activity[finite], edges[1:-1]), 0, n_bins - 1)
    return out


def zero_fraction_by_bin(counts: np.ndarray, bins: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Fraction of observed days with zero groups, per activity bin."""
    counts = np.asarray(counts, dtype=float)
    out = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = (bins == b) & np.isfinite(counts)
        if sel.sum() > 10:
            out[b] = float((counts[sel] == 0).mean())
    return out


def count_pmf(counts: np.ndarray, k: int) -> np.ndarray:
    """Empirical PMF of the counts on ``0..k-1``, with the tail folded into ``k-1``."""
    counts = np.asarray(counts, dtype=float)
    counts = counts[np.isfinite(counts)]
    clipped = np.clip(counts.astype(int), 0, k - 1)
    pmf = np.bincount(clipped, minlength=k).astype(float)
    return pmf / max(pmf.sum(), 1.0)


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """Total-variation distance, ``0.5 * sum |p - q|``."""
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def dispersion_index(counts: np.ndarray) -> float:
    """Variance-to-mean ratio; 1 for Poisson.

    The research report's ``D_Y = 1 + p(D_N - 1)`` makes this the handle on how far
    the thinned observation is from Poisson, so it belongs in the comparison.
    """
    counts = np.asarray(counts, dtype=float)
    counts = counts[np.isfinite(counts)]
    mean = counts.mean()
    return float(counts.var() / mean) if mean > 0 else np.nan


def run_lengths(counts: np.ndarray, kind: int = SPOTLESS) -> np.ndarray:
    """Lengths of maximal runs of spotless (or active) **observed** days.

    Unobserved days break a run rather than extending it: a gap is not evidence of
    either state.  Hall & Park's busy/idle-period deconvolution is the reason these
    carry lifetime and acuity information at all (research note Q2(c)).
    """
    counts = np.asarray(counts, dtype=float)
    state = np.where(np.isnan(counts), -1, (counts > 0).astype(int))
    runs, current = [], 0
    for value in state:
        if value == kind:
            current += 1
        else:
            if current:
                runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return np.asarray(runs, dtype=float)


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov distance; NaNs dropped."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    grid = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    cdf_b = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.abs(cdf_a - cdf_b).max())


def block_bootstrap(values: np.ndarray, block_days: int, rng: np.random.Generator) -> np.ndarray:
    """One block-bootstrap resample, preserving dependence within a block.

    Group-days within a day, and days within a rotation, are dependent, so an i.i.d.
    resample would understate every uncertainty Check B reports.
    """
    values = np.asarray(values)
    n = len(values)
    n_blocks = int(np.ceil(n / block_days))
    starts = rng.integers(0, max(n - block_days, 1), n_blocks)
    pieces = [values[s : s + block_days] for s in starts]
    return np.concatenate(pieces)[:n]


# --- the five Check A measures --------------------------------------------

#: Advisory targets, from the Step 1 plan.  Reported alongside every measure; the
#: binding rule is set in the results doc (PI, 2026-11-21).
TARGETS = {
    "acf_rmse": 0.05,
    "acf_lag1_error": 0.03,
    "acf_trough_error": 0.05,
    "acf_bump_error": 0.05,
    "zero_fraction_max_error": 0.05,
    "zero_fraction_mean_error": 0.02,
    "count_pmf_total_variation": 0.05,
    "dispersion_error": 0.15,
    "run_length_mean_rel_error": 0.15,
    "run_length_ks": 0.05,
    "lifetime_ks": 0.05,
    "lifetime_long_fraction_rel_error": 0.20,
}

TROUGH_LAG, BUMP_LAG = 13, 27


@dataclass
class CheckAScores:
    """The five pre-registered measures, with every component that feeds them."""

    acf_rmse: float
    acf_lag1_error: float
    acf_trough_error: float
    acf_bump_error: float
    zero_fraction_max_error: float
    zero_fraction_mean_error: float
    count_pmf_total_variation: float
    dispersion_error: float
    run_length_mean_rel_error: float
    run_length_ks: float
    lifetime_ks: float
    lifetime_long_fraction_rel_error: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def passes(self) -> dict[str, bool]:
        """Per-measure comparison against the advisory targets."""
        return {
            "acf": (
                self.acf_rmse <= TARGETS["acf_rmse"]
                and abs(self.acf_lag1_error) <= TARGETS["acf_lag1_error"]
                and self.acf_trough_error <= TARGETS["acf_trough_error"]
                and self.acf_bump_error <= TARGETS["acf_bump_error"]
            ),
            "zero_fraction": (
                self.zero_fraction_max_error <= TARGETS["zero_fraction_max_error"]
                and self.zero_fraction_mean_error <= TARGETS["zero_fraction_mean_error"]
            ),
            "count_distribution": (
                self.count_pmf_total_variation <= TARGETS["count_pmf_total_variation"]
                and self.dispersion_error <= TARGETS["dispersion_error"]
            ),
            "run_lengths": (
                self.run_length_mean_rel_error <= TARGETS["run_length_mean_rel_error"]
                and self.run_length_ks <= TARGETS["run_length_ks"]
            ),
            "lifetimes": (
                self.lifetime_ks <= TARGETS["lifetime_ks"]
                and self.lifetime_long_fraction_rel_error
                <= TARGETS["lifetime_long_fraction_rel_error"]
            ),
        }

    @property
    def n_passed(self) -> int:
        return sum(self.passes().values())

    @property
    def aggregate(self) -> float:
        """Mean of each measure normalized by its target; lower is better.

        Used only to compare 1-class against 2-class on a single number, which is
        what the plan's "25% improvement" rule needs.
        """
        values = self.as_dict()
        ratios = [
            abs(values[key]) / target for key, target in TARGETS.items() if np.isfinite(values[key])
        ]
        return float(np.mean(ratios)) if ratios else np.nan


def _long_fraction(spans: np.ndarray, threshold: int = 13) -> float:
    spans = np.asarray(spans, dtype=float)
    spans = spans[np.isfinite(spans)]
    return float((spans >= threshold).mean()) if len(spans) else np.nan


def score_check_a(
    observed_counts: np.ndarray,
    simulated_counts: np.ndarray,
    observed_activity: np.ndarray,
    simulated_activity: np.ndarray,
    observed_spans: np.ndarray,
    simulated_spans: np.ndarray,
    *,
    k: int = 41,
    max_lag: int = 30,
    n_bins: int = 5,
    detrend_days: int = 91,
) -> CheckAScores:
    """Compare one simulated realization against the observed series.

    ``simulated_counts`` must carry the **same NaN pattern** as the observed series:
    the real observing calendar is transplanted onto the simulation, after
    ``legacy/ADFSSN/scripts/SSN_ADF.py`` l. 867-871 (missing-day imprinting).
    """
    acf_obs = autocorrelation(observed_counts, max_lag, detrend_days)
    acf_sim = autocorrelation(simulated_counts, max_lag, detrend_days)
    both = np.isfinite(acf_obs) & np.isfinite(acf_sim)
    acf_rmse = (
        float(np.sqrt(np.mean((acf_obs[both] - acf_sim[both]) ** 2))) if both.any() else np.nan
    )

    obs_bins = activity_bins(observed_activity, n_bins)
    sim_bins = activity_bins(simulated_activity, n_bins)
    zf_obs = zero_fraction_by_bin(observed_counts, obs_bins, n_bins)
    zf_sim = zero_fraction_by_bin(simulated_counts, sim_bins, n_bins)
    zf_err = np.abs(zf_obs - zf_sim)

    pmf_obs = count_pmf(observed_counts, k)
    pmf_sim = count_pmf(simulated_counts, k)

    obs_runs = run_lengths(observed_counts, SPOTLESS)
    sim_runs = run_lengths(simulated_counts, SPOTLESS)
    run_mean_err = (
        abs(sim_runs.mean() - obs_runs.mean()) / obs_runs.mean()
        if len(obs_runs) and len(sim_runs) and obs_runs.mean() > 0
        else np.nan
    )

    long_obs = _long_fraction(observed_spans)
    long_sim = _long_fraction(simulated_spans)

    return CheckAScores(
        acf_rmse=acf_rmse,
        acf_lag1_error=float(acf_sim[0] - acf_obs[0]),
        acf_trough_error=float(abs(acf_sim[TROUGH_LAG - 1] - acf_obs[TROUGH_LAG - 1])),
        acf_bump_error=float(abs(acf_sim[BUMP_LAG - 1] - acf_obs[BUMP_LAG - 1])),
        zero_fraction_max_error=float(np.nanmax(zf_err)) if np.isfinite(zf_err).any() else np.nan,
        zero_fraction_mean_error=float(np.nanmean(zf_err)),
        count_pmf_total_variation=total_variation(pmf_obs, pmf_sim),
        dispersion_error=float(
            abs(dispersion_index(simulated_counts) - dispersion_index(observed_counts))
        ),
        run_length_mean_rel_error=float(run_mean_err),
        run_length_ks=ks_distance(obs_runs, sim_runs),
        lifetime_ks=ks_distance(observed_spans, simulated_spans),
        lifetime_long_fraction_rel_error=(
            float(abs(long_sim - long_obs) / long_obs) if long_obs and long_obs > 0 else np.nan
        ),
    )
