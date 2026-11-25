"""CEA17: calibration by a probability-mass conversion matrix.

Chatzistergos et al. (2017), as described in their 2025 assessment §2.2.3, which
settles a question this project had recorded as open
(``docs/02_research/notes/raters_linking_ratings.md`` l. 41, 72 -- "verify the
conditioning direction in their Sect. 2").  The direction is:

    a **column** is indexed by the secondary observer's count ``G``, and holds a
    normalised histogram over the **reference** observer's count ``G*`` on the
    days the secondary reported ``G``.

So the matrix is ``P(G* | G)``, column-stochastic.  Two consequences this
benchmark cares about.  First, the method is **non-linear** by construction, which
is why A157 finds it does not skew weak against strong cycles the way a scaling
does.  Second, it is the only one of the seven that returns a distribution rather
than a number -- "the calibration returns a PMF series, which also includes
information about the uncertainty" -- so it is the only baseline with a native
interval to score coverage on.  A157 never scores that interval; the coverage
figure here does.

Chaining is where this benchmark departs from A157, which never evaluated it.
Conditional distributions compose: ``P(a|c) = sum_b P(a|b) P(b|c)``, so chaining
along the overlap graph is a matrix product.  That is exact under a Markov
assumption -- that ``a`` is independent of ``c`` given ``b`` -- which is false
whenever two observers share an error, and the ``copier`` case study exists to
make the consequence visible.

Sparse columns are filled as the paper does, by Monte-Carlo bootstrap over the fit
``<G*|G> - G = R0 - B exp(-a G)``.
"""

from __future__ import annotations

import heapq
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit

from gn_next_gen.baselines.base import (
    CalibrationResult,
    Series,
    combine_observers,
    inverse_variance_daily,
    matched_counts,
    station_series,
)
from gn_next_gen.graph.overlap import OverlapGraph

#: Largest count the matrices represent.  Step 1 fixed K = 41 for the state space
#: (D17); the same truncation applies here for the same reason.
MAX_COUNT = 41

#: CEA17's quality criteria (their §2.2.3).  The matrix must carry statistics for
#: "at least three columns **and one-quarter of the range of counts by the
#: secondary observer**" -- the second clause is what makes the criterion bite at
#: high acuity, where the secondary's counts span a wide range but only its low
#: columns are populated.
MIN_OVERLAP_DAYS = 20
MIN_YEARS_FOR_LONG_RECORDS = 4
MIN_POPULATED_COLUMNS = 3
MIN_POPULATED_FRACTION_OF_RANGE = 0.25
MAX_CHI2_PER_DOF = 6.0

#: Candidate parents tried per observer when routing around a failed edge.
#: Unbounded retrying turns the chain into an all-pairs search; five keeps the
#: "route around a bad link" behaviour at a bounded cost.
MAX_PARENT_CANDIDATES = 5

#: Bootstrap draws used to fill sparse columns.  The paper uses 1,000; 100 is
#: enough here because the draws only smooth columns the data cannot support,
#: and the fits dominate this baseline's runtime.  Recorded as a departure.
N_BOOTSTRAP = 100


def _offset_model(count: np.ndarray, r0: float, b: float, a: float) -> np.ndarray:
    return r0 - b * np.exp(-a * count)


def conversion_matrix(
    target: np.ndarray,
    source: np.ndarray,
    *,
    max_count: int = MAX_COUNT,
    rng: np.random.Generator | None = None,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[np.ndarray, dict]:
    """``P(G* | G)`` as a column-stochastic matrix, sparse columns filled.

    Returns the matrix and a diagnostics dict carrying the populated-column count
    and the bootstrap fit's chi-square per degree of freedom, which is one of the
    paper's exclusion criteria.
    """
    rng = rng or np.random.default_rng(0)
    size = max_count + 1
    matrix = np.zeros((size, size))
    target = np.clip(target, 0, max_count)
    source = np.clip(source, 0, max_count)

    populated = []
    for value in range(size):
        rows = target[source == value]
        if len(rows) >= 3:
            counts = np.bincount(rows, minlength=size).astype(float)
            matrix[:, value] = counts / counts.sum()
            populated.append(value)

    count_range = int(source.max() - source.min()) + 1
    diagnostics = {
        "n_populated": len(populated),
        "count_range": count_range,
        "populated_fraction_of_range": len(populated) / max(count_range, 1),
        "chi2_per_dof": np.nan,
    }
    if len(populated) < 2:
        return matrix, diagnostics

    # The bootstrap exists to invent columns the data cannot support.  When every
    # column up to the highest count the secondary ever reported is already
    # populated, there is nothing to invent, and skipping it is most of this
    # method's runtime on a well-sampled edge.
    highest = int(source.max())
    if all(matrix[:, value].sum() > 0 for value in range(highest + 1)):
        return matrix, diagnostics

    # Bootstrap the mean-offset curve on half the days, as the paper does, and
    # use it to invent the columns the data cannot support.
    observed = np.array(populated, dtype=float)
    offsets = np.array([float((target[source == v]).mean() - v) for v in populated], dtype=float)
    draws = []
    for _ in range(n_bootstrap):
        half = rng.random(len(source)) < 0.5
        sub_t, sub_s = target[half], source[half]
        values, means = [], []
        for value in populated:
            rows = sub_t[sub_s == value]
            if len(rows) >= 2:
                values.append(value)
                means.append(rows.mean() - value)
        if len(values) < 4:
            continue
        try:
            with warnings.catch_warnings():
                # A flat bootstrap resample gives a singular covariance; the fit
                # itself is still usable and the spread across draws is what we
                # want, so the warning is noise rather than news.
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, _ = curve_fit(
                    _offset_model,
                    np.array(values, dtype=float),
                    np.array(means, dtype=float),
                    p0=(float(np.mean(means)), 1.0, 0.2),
                    maxfev=2000,
                )
        except (RuntimeError, ValueError):
            continue
        draws.append(popt)

    if not draws:
        return matrix, diagnostics

    draws = np.asarray(draws)
    predicted = np.array([_offset_model(observed, *p) for p in draws])
    residual = offsets - predicted.mean(axis=0)
    spread = predicted.std(axis=0, ddof=1)
    # A perfect pair -- an observer calibrated against itself, which is exactly
    # the acuity-0 end of the A157 grid -- has zero residual and zero bootstrap
    # spread, and residual/0 reads as a catastrophically failed fit rather than
    # a flawless one.  Score only the bins that carry spread, and call a fit
    # with no scatter and no residual what it is.
    informative = spread > 1e-9
    if informative.any():
        chi2 = float(np.mean((residual[informative] / spread[informative]) ** 2))
    else:
        chi2 = 0.0 if np.allclose(residual, 0.0, atol=1e-6) else np.inf
    diagnostics["chi2_per_dof"] = chi2

    width = max(float(np.std(target - source, ddof=1)), 0.5)
    grid = np.arange(size)
    for value in range(size):
        if matrix[:, value].sum() > 0:
            continue
        # An extrapolated exponential can run away far outside the count range;
        # clip to the representable support before it overflows the Gaussian.
        centres = np.clip(
            value + _offset_model(np.full(len(draws), value), *draws.T), 0.0, size - 1
        )
        column = np.exp(-0.5 * ((grid[:, None] - centres[None, :]) / width) ** 2).sum(axis=1)
        total = column.sum()
        if total > 0:
            matrix[:, value] = column / total
    return matrix, diagnostics


def passes_criteria(diagnostics: dict) -> bool:
    """CEA17's own exclusion rule, applied to one conversion matrix."""
    if diagnostics["n_populated"] < MIN_POPULATED_COLUMNS:
        return False
    if diagnostics.get("populated_fraction_of_range", 1.0) < MIN_POPULATED_FRACTION_OF_RANGE:
        return False
    chi2 = diagnostics["chi2_per_dof"]
    return not (np.isfinite(chi2) and chi2 > MAX_CHI2_PER_DOF)


def compose(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Chain two conditional matrices, renormalising the result.

    ``P(a|c) = sum_b P(a|b) P(b|c)`` holds only if the intermediate's whole
    support is covered.  It is not: wherever ``first`` has an empty column -- a
    count value the upstream pair never jointly produced -- that part of the
    intermediate distribution has nowhere to go and its mass vanishes.  Left
    alone the chained matrix is sub-stochastic in the tail and the implied mean
    is biased **low**, which would make a long chain look falsely small.

    Renormalising each surviving column keeps the result a distribution and makes
    the loss explicit instead of silent.  The lost mass is itself a diagnostic:
    it says the chain passed through a link that had never seen counts that high.
    """
    chained = first @ second
    total = chained.sum(axis=0)
    safe = total > 0
    chained[:, safe] /= total[safe]
    return chained


def _usable(left: Series, right: Series) -> tuple[bool, int, int]:
    """CEA17's quality gate: enough overlapping days, and enough of a span."""
    target, source = matched_counts(left, right)
    n_days = len(target)
    span_years = 0
    if n_days:
        shared_days = np.intersect1d(left[0], right[0], assume_unique=True)
        span_years = int((shared_days[-1] - shared_days[0]) // 365)
    long_record = min(len(left[1]), len(right[1])) > 4 * 365
    ok = n_days >= MIN_OVERLAP_DAYS and (
        not long_record or span_years >= MIN_YEARS_FOR_LONG_RECORDS
    )
    return ok, n_days, span_years


def calibrate_cea17(
    observations: pd.DataFrame,
    graph: OverlapGraph,
    reference: int,
    *,
    seed: int = 0,
    max_count: int = MAX_COUNT,
) -> CalibrationResult:
    """Calibrate every observer to the reference by chained conversion matrices."""
    rng = np.random.default_rng(seed)
    series = station_series(observations)
    index = graph.index
    size = max_count + 1

    # Edge cost: more overlapping days is better.  The published method states
    # quality *criteria* but no edge cost, because it never chains; this is ours
    # and is recorded as such.
    import scipy.sparse as sp

    edges = graph.edge_frame()
    rows, cols, costs = [], [], []
    for a, b, overlap in zip(
        edges["station_a"].astype(int),
        edges["station_b"].astype(int),
        edges["overlap_days"].astype(int),
        strict=True,
    ):
        if a not in series or b not in series:
            continue
        cost = 1.0 / max(int(overlap), 1)
        rows += [index[a], index[b]]
        cols += [index[b], index[a]]
        costs += [cost, cost]

    if not rows:
        empty = pd.DataFrame(columns=["date_ord", "gn", "gn_sd", "n_observers"])
        return CalibrationResult(empty, pd.DataFrame(), "CEA17", {"reason": "no edges"})

    matrix = sp.csr_matrix(
        (np.asarray(costs), (np.asarray(rows), np.asarray(cols))),
        shape=(graph.n_stations, graph.n_stations),
    )
    distance, predecessor = sp.csgraph.dijkstra(
        matrix, directed=False, indices=index[reference], return_predecessors=True
    )

    # Grow the calibrated set outward from the reference with a priority queue,
    # Prim-style rather than along a fixed Dijkstra tree.  CEA17's criteria
    # exclude an *observer pair*, not a route, so when an edge fails its quality
    # gate the observer behind it should be reached another way if one exists.
    # A fixed tree instead severs the whole subtree beyond the first bad edge,
    # which on one S1 world cost a third of the day coverage for reasons that had
    # nothing to do with the method.  (A repeated rescan expresses the same thing
    # and is 8x slower, because most rounds chain only a handful of nodes.)
    neighbours: dict[int, list[tuple[float, int]]] = {}
    for (a, b), cost in zip(
        zip(edges["station_a"].astype(int), edges["station_b"].astype(int), strict=True),
        1.0 / np.maximum(edges["overlap_days"].astype(float).to_numpy(), 1.0),
        strict=True,
    ):
        neighbours.setdefault(a, []).append((float(cost), b))
        neighbours.setdefault(b, []).append((float(cost), a))

    chained: dict[int, np.ndarray] = {index[reference]: np.eye(size)}
    done = {reference}
    records: list[dict] = []
    attempts: dict[int, int] = {}

    queue: list[tuple[float, int, int]] = []
    for cost, neighbour in neighbours.get(reference, ()):
        heapq.heappush(queue, (cost, reference, neighbour))

    while queue:
        _cost, parent, station = heapq.heappop(queue)
        if station in done or station not in series:
            continue
        if attempts.get(station, 0) >= MAX_PARENT_CANDIDATES:
            continue
        attempts[station] = attempts.get(station, 0) + 1

        ok, n_days, _span = _usable(series[parent], series[station])
        if not ok:
            continue
        target, source = matched_counts(series[parent], series[station])
        edge_matrix, diagnostics = conversion_matrix(target, source, max_count=max_count, rng=rng)
        if not passes_criteria(diagnostics):
            continue

        chained[index[station]] = compose(chained[index[parent]], edge_matrix)
        done.add(station)
        records.append({"station": station, "n_days": n_days, **diagnostics})
        for cost, neighbour in neighbours.get(station, ()):
            if neighbour not in done:
                heapq.heappush(queue, (cost, station, neighbour))

    excluded = sum(1 for s in series if s not in done)

    values = np.arange(size)
    calibrated = []
    observer_rows = []
    for node, composite in chained.items():
        station = int(graph.stations[node])
        if station not in series:
            continue
        mean = composite.T @ values
        second = composite.T @ (values**2)
        sd = np.sqrt(np.maximum(second - mean**2, 0.0))

        days, counts = series[station]
        clipped = np.clip(counts, 0, max_count)
        calibrated.append(
            pd.DataFrame(
                {
                    "date_ord": days,
                    "value": mean[clipped],
                    "sd": sd[clipped],
                    "station": station,
                }
            )
        )
        observer_rows.append(
            {
                "station": station,
                "factor": float(mean[1]) if size > 1 else np.nan,
                "factor_sd": float(sd[1]) if size > 1 else np.nan,
                "recovered_threshold_msh": np.nan,
                "n_hops": int(np.isfinite(distance[node])),
                "reached": True,
            }
        )

    stacked = pd.concat(calibrated, ignore_index=True)
    series_out = inverse_variance_daily(stacked, value="value", sd="sd", sd_floor=0.25)

    return CalibrationResult(
        series=series_out,
        observers=pd.DataFrame(observer_rows),
        name="CEA17",
        meta={
            "excluded_observers": excluded,
            "n_calibrated": len(observer_rows),
            "max_count": max_count,
            "diagnostics": pd.DataFrame(records),
        },
    )


__all__ = ["calibrate_cea17", "combine_observers", "compose", "conversion_matrix"]
