"""Shared machinery for the ported calibration baselines.

Five of the seven methods share a shape: estimate a scale factor between two
observers who overlapped, then carry the reference's scale out to everyone else
along the overlap graph.  Only the pairwise estimator differs, so it is the only
thing each method supplies.

**Daisy-chaining is the point.**  Chatzistergos et al. (2025) compared every
synthetic observer directly to RGO and wrote that "the effect of the daisy-chaining
process was not evaluated in this study, but it would be important to address it in
forthcoming studies".  Here the reference observer sees only its own anchored
epoch (ADR-009), so every other observer is reached through the graph and the
error of that chaining is exactly what the scorer measures.

Error propagation follows the legacy MATLAB composite
(``GN_Reconstruction_MultiPath.m``): factors multiply along a path and relative
uncertainties add in quadrature.  One legacy bug is **not** reproduced -- l. 668-669
scale by ``LnkVlP`` twice, inflating the propagated uncertainty by one factor of
``a`` -- because there is no longer a published number to match (the plan's
decision 7) and reproducing a defect that makes a baseline look *less* certain
would flatter every method compared against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from gn_next_gen.graph.overlap import OverlapGraph


@dataclass
class CalibrationResult:
    """What a baseline produces, in the one shape the scorer reads.

    Attributes
    ----------
    series
        ``date_ord, gn, gn_sd`` -- the reconstructed reference-equivalent count
        and its standard deviation.  ``gn_sd`` may be NaN for methods that report
        no uncertainty at all, which is itself a finding the coverage figure shows.

        Two further columns say **how a day's uncertainty behaves over time**, which
        the scorer needs and a single ``sd`` cannot carry: ``gn_sd_factor`` is the
        part that comes from each observer's calibration factor, identical on every
        day of that observer's career and therefore *correlated* within any
        averaging window; ``gn_sd_spread`` is the part that comes from observers
        disagreeing on the day itself, which *averages down*.  A yearly interval
        that pools the two as though both averaged down is too narrow by up to
        ``sqrt(n_days)`` -- see
        :func:`gn_next_gen.benchmark.scoring.uncertainty_scores`.  ``gn_sd`` itself
        is left exactly as each method reports it, so no point score and no
        published interval width moves when these columns appear.
    observers
        ``station, factor, factor_sd, recovered_threshold_msh, n_hops, reached``.
        ``recovered_threshold_msh`` is NaN for methods that calibrate by scaling
        rather than by estimating an acuity.
    name
        Baseline name, as it appears in every score table and figure.
    meta
        Free-form: parameters used, counts of excluded observers, timings.
    """

    series: pd.DataFrame
    observers: pd.DataFrame
    name: str
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PairEstimate:
    """A scale factor between two observers, with its relative uncertainty."""

    factor: float
    relative_sd: float
    n_days: int
    intercept: float = 0.0
    usable: bool = True


#: A station's record as ``(days, counts)``.
Series = tuple[np.ndarray, np.ndarray]


class PairwiseEstimator:
    """A pairwise scale estimator, with per-station work hoisted out of the loop.

    Estimators see both observers' **whole records**, not only their matched
    days, because the methods differ on exactly that point: HoSc98 uses common
    spotted days, SvSc16 computes annual means on each observer's own days (which
    its own authors flag as a source of bias), and VHS24 uses the entire records.
    Flattening that away would erase part of what the benchmark measures.

    :meth:`prepare` runs once per station and :meth:`estimate` once per
    undirected edge.  On a 470-observer network that is ~470 preparations and
    ~20,000 estimates instead of 40,000 of each, which is the difference between
    seconds and many minutes.
    """

    name = "pairwise"

    def prepare(self, series: Series):  # noqa: D102 - trivial default
        del series
        return None

    def estimate(self, left: Series, right: Series, prepared_left, prepared_right) -> PairEstimate:
        raise NotImplementedError

    def invert(self, estimate: PairEstimate) -> PairEstimate:
        """The same link read the other way.

        The legacy MATLAB writes one computed value into both halves of the link
        matrix, so the reverse direction is the reciprocal by construction rather
        than an independent fit.  Keeping that is both faithful and half the work;
        an estimator whose reverse is genuinely different overrides this.
        """
        if not estimate.usable or not np.isfinite(estimate.factor) or estimate.factor <= 0:
            return PairEstimate(np.nan, np.inf, estimate.n_days, usable=False)
        inverse = 1.0 / estimate.factor
        return PairEstimate(
            factor=inverse,
            relative_sd=estimate.relative_sd,
            n_days=estimate.n_days,
            intercept=-estimate.intercept * inverse,
            usable=True,
        )


PairEstimator = PairwiseEstimator


def station_series(observations: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Split a long observation table into per-station ``(days, counts)``."""
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for station, frame in observations.groupby("station", sort=True):
        ordered = frame.sort_values("date_ord")
        out[int(station)] = (
            ordered["date_ord"].to_numpy(dtype=np.int64),
            ordered["groups"].to_numpy(dtype=np.int64),
        )
    return out


def matched_counts(left: Series, right: Series) -> tuple[np.ndarray, np.ndarray]:
    """Counts of two observers on the days they both reported."""
    days_a, counts_a = left
    days_b, counts_b = right
    _shared, index_a, index_b = np.intersect1d(
        days_a, days_b, assume_unique=True, return_indices=True
    )
    return counts_a[index_a], counts_b[index_b]


def estimate_edges(
    graph: OverlapGraph,
    series: dict[int, tuple[np.ndarray, np.ndarray]],
    estimator: PairEstimator,
) -> dict[tuple[int, int], PairEstimate]:
    """Run a pairwise estimator over every edge of the overlap graph.

    Each undirected edge is estimated once and the reverse direction derived by
    :meth:`PairwiseEstimator.invert`, which is what the legacy symmetric write
    amounts to and halves the work.
    """
    prepared = {station: estimator.prepare(record) for station, record in series.items()}
    estimates: dict[tuple[int, int], PairEstimate] = {}
    edges = graph.edge_frame()
    for a, b in zip(edges["station_a"].astype(int), edges["station_b"].astype(int), strict=True):
        if a not in series or b not in series:
            continue
        forward = estimator.estimate(series[a], series[b], prepared[a], prepared[b])
        estimates[(a, b)] = forward
        estimates[(b, a)] = estimator.invert(forward)
    return estimates


def chain_from_reference(
    graph: OverlapGraph,
    estimates: dict[tuple[int, int], PairEstimate],
    reference: int,
    *,
    divide_cost_by_overlap: bool = True,
) -> pd.DataFrame:
    """Carry the reference's scale to every observer along the cheapest path.

    Cost of an edge is its relative uncertainty, divided by the number of
    overlapping days as the legacy MATLAB scripts do, so a long well-sampled
    overlap is preferred to a short noisy one.  Dijkstra then gives the
    minimum-variance *single* path -- which is precisely the chained-calibration
    limiting case the joint model generalises, and precisely why its error grows
    with effective resistance.
    """
    index = graph.index
    n = graph.n_stations
    rows, cols, costs = [], [], []
    for (a, b), estimate in estimates.items():
        if not estimate.usable or estimate.factor <= 0:
            continue
        cost = max(estimate.relative_sd, 1e-6)
        if divide_cost_by_overlap:
            cost /= max(estimate.n_days, 1)
        rows.append(index[a])
        cols.append(index[b])
        costs.append(cost)

    if not rows:
        return pd.DataFrame(
            {
                "station": graph.stations,
                "factor": np.nan,
                "factor_sd": np.nan,
                "intercept": np.nan,
                "n_hops": -1,
                "reached": False,
            }
        )

    matrix = sp.csr_matrix((np.asarray(costs), (np.asarray(rows), np.asarray(cols))), shape=(n, n))
    _distance, predecessor = sp.csgraph.dijkstra(
        matrix, directed=True, indices=index[reference], return_predecessors=True
    )

    factor = np.full(n, np.nan)
    variance = np.full(n, np.nan)
    intercept = np.zeros(n)
    hops = np.full(n, -1, dtype=np.int64)
    start = index[reference]
    factor[start], variance[start], hops[start] = 1.0, 0.0, 0

    order = np.argsort(_distance)
    for node in order:
        if node == start or not np.isfinite(_distance[node]):
            continue
        parent = predecessor[node]
        if parent < 0 or not np.isfinite(factor[parent]):
            continue
        estimate = estimates[(int(graph.stations[parent]), int(graph.stations[node]))]
        factor[node] = factor[parent] * estimate.factor
        variance[node] = variance[parent] + estimate.relative_sd**2
        # An affine link composes as G* = b_parent + k_parent (b_edge + k_edge G),
        # so the cumulative intercept picks up the parent's factor. Only VHS24
        # carries a non-zero one.
        intercept[node] = intercept[parent] + factor[parent] * estimate.intercept
        hops[node] = hops[parent] + 1

    reached = np.isfinite(factor)
    return pd.DataFrame(
        {
            "station": graph.stations,
            "factor": factor,
            "factor_sd": np.where(reached, factor * np.sqrt(variance), np.nan),
            "intercept": np.where(reached, intercept, np.nan),
            "n_hops": hops,
            "reached": reached,
        }
    )


def combine_observers(
    observations: pd.DataFrame,
    scales: pd.DataFrame,
    *,
    intercepts: pd.Series | None = None,
) -> pd.DataFrame:
    """Inverse-variance combination of calibrated observers into one daily series.

    The spread *between* observers on a day is used as the day's uncertainty
    whenever two or more reported, because it carries the disagreement a single
    observer's propagated factor cannot.  With one observer the propagated
    uncertainty is all there is, which is exactly where chained methods turn out
    to be overconfident.
    """
    usable = scales[scales["reached"] & np.isfinite(scales["factor"])]
    merged = observations.merge(
        usable[["station", "factor", "factor_sd"]], on="station", how="inner"
    )
    if merged.empty:
        return pd.DataFrame(columns=["date_ord", "gn", "gn_sd", "n_observers"])

    if intercepts is not None:
        merged["offset"] = merged["station"].map(intercepts).fillna(0.0)
    else:
        merged["offset"] = 0.0

    merged["calibrated"] = merged["offset"] + merged["factor"] * merged["groups"]
    merged["sd"] = np.where(
        np.isfinite(merged["factor_sd"]) & (merged["factor_sd"] > 0),
        merged["factor_sd"] * merged["groups"],
        np.nan,
    )
    merged["weight"] = np.where(merged["sd"] > 0, 1.0 / merged["sd"] ** 2, 1.0)

    grouped = merged.groupby("date_ord")
    total_weight = grouped["weight"].sum()
    estimate = grouped.apply(
        lambda block: np.average(block["calibrated"], weights=block["weight"]),
        include_groups=False,
    )
    n_observers = grouped.size()

    spread = grouped["calibrated"].std(ddof=1)
    propagated = np.sqrt(1.0 / total_weight)
    combined = np.where(n_observers > 1, np.maximum(spread, propagated), propagated)

    return pd.DataFrame(
        {
            "date_ord": estimate.index.to_numpy(),
            "gn": estimate.to_numpy(),
            "gn_sd": combined,
            # A decomposition, not a second opinion: `gn_sd` above is unchanged.  The
            # propagated part is a function of the observers' factors, which do not
            # change from one day to the next, so it is the correlated part; the
            # between-observer disagreement is the part that averages down.
            "gn_sd_factor": propagated,
            "gn_sd_spread": np.where(n_observers > 1, spread.to_numpy(), 0.0),
            "n_observers": n_observers.to_numpy(),
        }
    ).reset_index(drop=True)


def inverse_variance_daily(
    frame: pd.DataFrame,
    *,
    value: str,
    sd: str,
    sd_floor: float,
) -> pd.DataFrame:
    """Inverse-variance combination of per-observer values into a daily series.

    The shape CEA17, UEA21, GRP and the Pathfinding re-pool all share: one row per
    observer-day carrying a calibrated ``value`` and the uncertainty ``sd`` that the
    observer's own calibration implies, combined as ``1 / sum(1 / sd^2)``.

    ``gn_sd`` is exactly what each of those four already produced, so adopting this
    helper moves no score.  What it adds is the decomposition: for these four the
    reported uncertainty is **entirely** a propagated per-observer calibration
    uncertainty -- the per-row ``sd`` is a property of the observer's career, not of
    the day -- so ``gn_sd_factor`` carries all of it and ``gn_sd_spread`` is zero.
    The between-observer scatter is reported beside it as ``day_spread``, for
    diagnostics only: it is deliberately not folded into ``gn_sd``, because these
    methods do not claim it.
    """
    weight = 1.0 / np.maximum(frame[sd].to_numpy(), sd_floor) ** 2
    frame = frame.assign(_weight=weight)
    grouped = frame.groupby("date_ord")
    propagated = np.sqrt(1.0 / grouped["_weight"].sum().to_numpy())
    return pd.DataFrame(
        {
            "date_ord": grouped[value].mean().index.to_numpy(),
            "gn": grouped.apply(
                lambda block: float(np.average(block[value], weights=block["_weight"])),
                include_groups=False,
            ).to_numpy(),
            "gn_sd": propagated,
            "gn_sd_factor": propagated,
            "gn_sd_spread": np.zeros(len(propagated)),
            "day_spread": grouped[value].std(ddof=1).fillna(0.0).to_numpy(),
            "n_observers": grouped.size().to_numpy(),
        }
    )
