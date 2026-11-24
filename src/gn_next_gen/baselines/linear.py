"""The three linear scalings: Hoyt & Schatten 1998, Svalgaard & Schatten 2016, VHS24.

Chatzistergos et al. (2025) found these "systematically amplify strong cycles and
underestimate weak cycles", biasing the secular variability -- the single result
this benchmark most wants to reproduce on a *real overlap graph* rather than on
pairwise comparisons to RGO.  All three are cheap, so there is no reason to leave
them out and every reason to include them: they are the methods behind the
published series that diverge before 1880.

Each differs from the others in one deliberate way, and each difference is a
documented source of bias rather than an implementation choice of ours:

* **HoSc98** uses only days on which *both* observers counted at least one group,
  and refuses an observer whose factor falls outside ``0.6 < k < 1.4``.
* **SvSc16** regresses annual means through the origin, and computes each
  observer's annual mean on **its own observing days** rather than on common days
  -- "annual means underlying the comparison might eventually be derived from
  observations on different days, which can lead to biases" (their §2.2.2).
  Forcing through the origin also asserts that a zero for one observer is a zero
  for the other, which is exactly what a threshold difference denies.
* **VHS24** takes the ratio of standard deviations over the whole records and adds
  an intercept.  Its own weakness is structural: the standard deviation of group
  counts varies over the cycle, so comparing records that cover different activity
  levels compares incomparable spreads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gn_next_gen.baselines.base import PairEstimate, PairwiseEstimator, matched_counts

#: 1970-01-01 as a proleptic-Gregorian ordinal, for vectorised year extraction.
UNIX_EPOCH_ORDINAL = 719_163

#: HoSc98's usability gate on the scaling factor.
HOSC98_K_RANGE = (0.6, 1.4)

#: Below this many usable points a pairwise estimate is not trusted.
MIN_POINTS = 10


def _ratio_relative_sd(target: np.ndarray, source: np.ndarray) -> float:
    """Delta-method relative standard error of a ratio of means.

    The published methods report no uncertainty at all, but chaining needs an
    edge cost, so one is supplied here and recorded as ours rather than theirs.
    """
    n = len(target)
    mean_t, mean_s = target.mean(), source.mean()
    if n < 2 or mean_t <= 0 or mean_s <= 0:
        return np.inf
    var_t, var_s = target.var(ddof=1), source.var(ddof=1)
    covariance = np.cov(target, source, ddof=1)[0, 1]
    relative = (var_t / mean_t**2 + var_s / mean_s**2 - 2 * covariance / (mean_t * mean_s)) / n
    return float(np.sqrt(max(relative, 1e-12)))


def years_of(days: np.ndarray) -> np.ndarray:
    """Calendar year of each ordinal, vectorised.

    ``datetime.date.fromordinal`` in a comprehension is the single slowest thing
    in this package when called per edge; 1970-01-01 is ordinal 719163, so the
    whole array converts at once.
    """
    return pd.to_datetime(
        np.asarray(days, dtype=np.int64) - UNIX_EPOCH_ORDINAL, unit="D"
    ).year.to_numpy()


class HoSc98(PairwiseEstimator):
    """``k = sum(G*) / sum(G)`` over days both observers saw at least one group."""

    name = "HoSc98"

    def estimate(self, left, right, prepared_left, prepared_right):
        del prepared_left, prepared_right
        target, source = matched_counts(left, right)
        both_spotted = (target > 0) & (source > 0)
        target, source = target[both_spotted], source[both_spotted]
        if len(target) < MIN_POINTS or source.sum() == 0:
            return PairEstimate(np.nan, np.inf, len(target), usable=False)

        factor = float(target.sum() / source.sum())
        low, high = HOSC98_K_RANGE
        return PairEstimate(
            factor=factor,
            relative_sd=_ratio_relative_sd(target, source),
            n_days=int(len(target)),
            usable=bool(low < factor < high),
        )

    def invert(self, estimate):
        """HoSc98 gates on ``0.6 < k < 1.4``, and that gate is not symmetric."""
        inverted = super().invert(estimate)
        if not inverted.usable:
            return inverted
        low, high = HOSC98_K_RANGE
        return PairEstimate(
            factor=inverted.factor,
            relative_sd=inverted.relative_sd,
            n_days=inverted.n_days,
            usable=bool(low < inverted.factor < high),
        )


class SvSc16(PairwiseEstimator):
    """Regression of annual means through the origin, on each observer's own days."""

    name = "SvSc16"

    def prepare(self, series):
        days, counts = series
        if len(days) == 0:
            return {}
        years = years_of(days)
        frame = pd.DataFrame({"year": years, "count": counts})
        return frame.groupby("year")["count"].mean().to_dict()

    def estimate(self, left, right, prepared_left, prepared_right):
        del left, right
        shared = sorted(set(prepared_left) & set(prepared_right))
        if len(shared) < 2:
            return PairEstimate(np.nan, np.inf, len(shared), usable=False)

        target = np.array([prepared_left[y] for y in shared])
        source = np.array([prepared_right[y] for y in shared])
        denominator = float((source**2).sum())
        if denominator <= 0:
            return PairEstimate(np.nan, np.inf, len(shared), usable=False)

        factor = float((target * source).sum() / denominator)
        residual = target - factor * source
        if len(shared) > 2 and factor > 0:
            sd = float(np.sqrt((residual**2).sum() / (len(shared) - 1) / denominator))
            relative = sd / factor
        else:
            relative = np.inf
        return PairEstimate(
            factor=factor,
            relative_sd=relative,
            n_days=int(len(shared)),
            usable=bool(np.isfinite(relative) and factor > 0),
        )


class VHS24(PairwiseEstimator):
    """``k = sd(G*) / sd(G)`` over the whole records, with an additive intercept."""

    name = "VHS24"

    def prepare(self, series):
        _days, counts = series
        if len(counts) < MIN_POINTS:
            return None
        return float(counts.mean()), float(counts.std(ddof=1)), int(len(counts))

    def estimate(self, left, right, prepared_left, prepared_right):
        del left, right
        if prepared_left is None or prepared_right is None:
            return PairEstimate(np.nan, np.inf, 0, usable=False)
        mean_t, sd_t, n_t = prepared_left
        mean_s, sd_s, n_s = prepared_right
        if sd_s <= 0 or sd_t <= 0:
            return PairEstimate(np.nan, np.inf, min(n_t, n_s), usable=False)

        factor = sd_t / sd_s
        intercept = float(mean_t - factor * mean_s)
        relative = float(np.sqrt(1.0 / (2 * (n_t - 1)) + 1.0 / (2 * (n_s - 1))))
        return PairEstimate(
            factor=factor,
            relative_sd=relative,
            n_days=int(min(n_t, n_s)),
            intercept=intercept,
            usable=True,
        )


def calibrate_linear(
    observations,
    graph,
    reference: int,
    estimator,
    name: str,
    *,
    use_intercept: bool = False,
):
    """Drive one pairwise estimator through the graph into a daily series."""
    from gn_next_gen.baselines.base import (
        CalibrationResult,
        chain_from_reference,
        combine_observers,
        estimate_edges,
        station_series,
    )

    series = station_series(observations)
    estimates = estimate_edges(graph, series, estimator)
    scales = chain_from_reference(graph, estimates, reference)

    intercepts = None
    if use_intercept:
        intercepts = pd.Series(scales["intercept"].to_numpy(), index=scales["station"].to_numpy())

    daily = combine_observers(observations, scales, intercepts=intercepts)
    observers = scales.assign(recovered_threshold_msh=np.nan)
    return CalibrationResult(
        series=daily,
        observers=observers,
        name=name,
        meta={
            "n_edges_estimated": len(estimates) // 2,
            "n_reached": int(scales["reached"].sum()),
            "n_unreached": int((~scales["reached"]).sum()),
        },
    )


def calibrate_hosc98(observations, graph, reference: int, **_kwargs):
    """Hoyt & Schatten (1998)."""
    return calibrate_linear(observations, graph, reference, HoSc98(), "HoSc98")


def calibrate_svsc16(observations, graph, reference: int, **_kwargs):
    """Svalgaard & Schatten (2016)."""
    return calibrate_linear(observations, graph, reference, SvSc16(), "SvSc16")


def calibrate_vhs24(observations, graph, reference: int, **_kwargs):
    """Velasco Herrera et al. (2024)."""
    return calibrate_linear(observations, graph, reference, VHS24(), "VHS24", use_intercept=True)
