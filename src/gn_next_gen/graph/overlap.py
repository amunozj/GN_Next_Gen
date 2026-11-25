"""Who observed alongside whom, and how far every observer sits from the anchor.

Chained calibration's random error grows like the **effective resistance** between
an observer and the reference on this graph, which is identifiability trap 3 of
the assessment (§6) and the quantity the pathfinding baseline is judged against.  The
graph is also what makes this benchmark differ from Chatzistergos et al. (2025):
they compare every synthetic observer pairwise to RGO, so no graph exists and
daisy-chaining is, in their words, "not evaluated in this study".

Construction follows ``legacy/GN_Series_matlab/A1_Matrix_Calculation_weighted_fit.m``
l. 71-117 and the ``AMJ_JMV_MM_Perspective`` notebook: expand each observer's
observing days by +/- ``day_tolerance`` and intersect with a partner's days.  Two
legacy choices are deliberately **not** carried over, per
``docs/03_legacy/reuse_map.md``'s Drop list:

* **no forward-in-time edge direction** -- the legacy code keeps an edge only if it
  runs from an earlier to a later observer, so information never flows back.  The
  graph here is undirected; a baseline that wants direction imposes it itself.
* **no Spearman-rho cut at construction** -- a correlation floor is a property of a
  calibration method, not of who observed together, so it belongs in the baseline.

Symmetry: the legacy code computes ``|dilate(a) & days(b)|`` in one direction and
writes it into both halves of the matrix, which makes the result depend on loop
order.  Here both directions are computed and the **maximum** is taken, which is
the same number whenever two calendars are comparable and is order-independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

#: Legacy day tolerance: two days counted as simultaneous (``DayTol = 2``).
DEFAULT_DAY_TOLERANCE = 2

#: Legacy minimum overlap for an edge (``MinOvr = 60``).
DEFAULT_MIN_OVERLAP_DAYS = 60

#: Legacy minimum distinct reported counts for a usable observer (``minObVal = 4``).
DEFAULT_MIN_DISTINCT_VALUES = 4


@dataclass
class OverlapGraph:
    """An undirected weighted graph over observing stations.

    Attributes
    ----------
    stations
        Station ids, in the row/column order of every matrix here.
    overlap
        Dense ``(n, n)`` integer matrix of matched days; zero off the kept edges.
    first_day, last_day, n_days
        Per-station calendar summaries, for diagnostics and endpoint choice.
    day_tolerance, min_overlap_days, min_distinct_values
        The construction parameters, carried so a run records them.
    """

    stations: np.ndarray
    overlap: np.ndarray
    first_day: np.ndarray
    last_day: np.ndarray
    n_days: np.ndarray
    day_tolerance: int = DEFAULT_DAY_TOLERANCE
    min_overlap_days: int = DEFAULT_MIN_OVERLAP_DAYS
    min_distinct_values: int = DEFAULT_MIN_DISTINCT_VALUES
    index: dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        self.index = {int(s): i for i, s in enumerate(self.stations)}

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    @property
    def n_edges(self) -> int:
        return int(np.count_nonzero(np.triu(self.overlap, 1)))

    def edge_frame(self) -> pd.DataFrame:
        """The kept edges as ``station_a, station_b, overlap_days``."""
        a, b = np.triu_indices(self.n_stations, 1)
        keep = self.overlap[a, b] > 0
        return pd.DataFrame(
            {
                "station_a": self.stations[a[keep]],
                "station_b": self.stations[b[keep]],
                "overlap_days": self.overlap[a[keep], b[keep]],
            }
        )


def _dilate(matrix: sp.csr_matrix, tolerance: int) -> sp.csr_matrix:
    """Union of a station-by-day indicator with its +/- ``tolerance`` shifts."""
    if tolerance <= 0:
        return matrix
    n_days = matrix.shape[1]
    out = matrix.copy()
    for shift in range(1, tolerance + 1):
        left = sp.hstack(
            [matrix[:, shift:], sp.csr_matrix((matrix.shape[0], shift), dtype=matrix.dtype)]
        )
        right = sp.hstack(
            [
                sp.csr_matrix((matrix.shape[0], shift), dtype=matrix.dtype),
                matrix[:, : n_days - shift],
            ]
        )
        out = out + left + right
    out = out.tocsr()
    out.data[:] = 1
    return out


def build_overlap_graph(
    observations: pd.DataFrame,
    *,
    day_tolerance: int = DEFAULT_DAY_TOLERANCE,
    min_overlap_days: int = DEFAULT_MIN_OVERLAP_DAYS,
    min_distinct_values: int = DEFAULT_MIN_DISTINCT_VALUES,
) -> OverlapGraph:
    """Build the overlap graph from a long-format observation table.

    Parameters
    ----------
    observations
        Columns ``date_ord, station, groups``; one row per observer-day.  Long
        format throughout -- the day x observer matrix is never densified, only a
        sparse indicator of *who observed when*, which is ~1.1M nonzeros for the
        real record.
    day_tolerance
        Days either side counted as simultaneous.
    min_overlap_days
        Edges below this many matched days are dropped.
    min_distinct_values
        Stations reporting fewer distinct counts than this are dropped entirely:
        an observer who only ever wrote the same number carries no calibration
        information and would otherwise anchor a spurious edge.
    """
    required = {"date_ord", "station", "groups"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"observations is missing {sorted(missing)}")

    distinct = observations.groupby("station")["groups"].nunique()
    usable = distinct[distinct >= min_distinct_values].index
    frame = observations[observations["station"].isin(usable)]
    if frame.empty:
        raise ValueError("no station reports enough distinct counts to build a graph")

    stations = np.sort(frame["station"].unique())
    station_row = {int(s): i for i, s in enumerate(stations)}
    days = np.sort(frame["date_ord"].unique())
    day_col = {int(d): i for i, d in enumerate(days)}

    rows = frame["station"].map(station_row).to_numpy()
    cols = frame["date_ord"].map(day_col).to_numpy()
    indicator = sp.csr_matrix(
        (np.ones(len(frame), dtype=np.int32), (rows, cols)),
        shape=(len(stations), len(days)),
    )
    indicator.sum_duplicates()
    indicator.data[:] = 1

    dilated = _dilate(indicator, day_tolerance)
    # |dilate(a) & days(b)| in both directions; max() removes the order dependence
    # the legacy single-direction write introduces.
    forward = (dilated @ indicator.T).toarray()
    overlap = np.maximum(forward, forward.T).astype(np.int64)
    np.fill_diagonal(overlap, 0)
    overlap[overlap < min_overlap_days] = 0

    grouped = frame.groupby("station")["date_ord"]
    first = grouped.min().reindex(stations).to_numpy()
    last = grouped.max().reindex(stations).to_numpy()
    count = grouped.size().reindex(stations).to_numpy()

    return OverlapGraph(
        stations=stations,
        overlap=overlap,
        first_day=first,
        last_day=last,
        n_days=count,
        day_tolerance=day_tolerance,
        min_overlap_days=min_overlap_days,
        min_distinct_values=min_distinct_values,
    )


def components(graph: OverlapGraph) -> np.ndarray:
    """Connected-component label per station, labelled by decreasing size."""
    n_comp, labels = sp.csgraph.connected_components(
        sp.csr_matrix(graph.overlap > 0), directed=False
    )
    sizes = np.bincount(labels, minlength=n_comp)
    order = np.argsort(-sizes, kind="stable")
    remap = np.empty(n_comp, dtype=np.int64)
    remap[order] = np.arange(n_comp)
    return remap[labels]


def bridges(graph: OverlapGraph) -> list[tuple[int, int]]:
    """Edges whose removal disconnects the graph, as ``(station_a, station_b)``.

    A bridge is a single point of failure in the calibration chain: everything
    beyond it reaches the anchor through exactly one link, so a systematic error on
    that link transfers coherently to every observer behind it.
    """
    import networkx as nx

    network = nx.Graph()
    network.add_nodes_from(int(s) for s in graph.stations)
    edges = graph.edge_frame()
    network.add_edges_from(
        zip(edges["station_a"].astype(int), edges["station_b"].astype(int), strict=True)
    )
    return [(int(a), int(b)) for a, b in nx.bridges(network)]


def _laplacian_pinv(graph: OverlapGraph, weights: np.ndarray) -> np.ndarray:
    degree = np.diag(weights.sum(axis=1))
    return np.linalg.pinv(degree - weights, hermitian=True)


def effective_resistance(
    graph: OverlapGraph, source: int, target: int, *, weight: str = "overlap_days"
) -> float:
    """Effective resistance between two stations.

    Variance accumulated by a chained calibration grows like this quantity
    (assessment §6, trap 3; report "Ubercal: simultaneous solution..."), so it is
    the natural x-axis for "how far is this observer from the anchor".

    ``weight`` is ``"overlap_days"`` (conductance proportional to matched days) or
    ``"unit"`` (every edge conductance 1, i.e. pure hop structure).  Returns
    ``inf`` when the two stations sit in different components.
    """
    weights = _edge_weights(graph, weight)
    if components(graph)[graph.index[source]] != components(graph)[graph.index[target]]:
        return float("inf")
    pinv = _laplacian_pinv(graph, weights)
    i, j = graph.index[source], graph.index[target]
    return float(pinv[i, i] + pinv[j, j] - 2 * pinv[i, j])


def resistance_to_set(
    graph: OverlapGraph, anchor: list[int] | np.ndarray, *, weight: str = "overlap_days"
) -> pd.DataFrame:
    """Effective resistance from every station to an anchor set, in one solve.

    The anchor set is shorted to a single node, which is what "distance to the
    reference epoch" means when the anchor is several observers (ADR-009's anchors
    are whole surveys over whole cycles, not single stations).
    """
    weights = _edge_weights(graph, weight)
    anchor_rows = sorted({graph.index[int(a)] for a in anchor})
    if not anchor_rows:
        raise ValueError("anchor set is empty")

    keep = [i for i in range(graph.n_stations) if i not in set(anchor_rows)]
    short = np.zeros((len(keep) + 1, len(keep) + 1))
    short[: len(keep), : len(keep)] = weights[np.ix_(keep, keep)]
    short[: len(keep), -1] = weights[np.ix_(keep, anchor_rows)].sum(axis=1)
    short[-1, : len(keep)] = short[: len(keep), -1]

    degree = np.diag(short.sum(axis=1))
    pinv = np.linalg.pinv(degree - short, hermitian=True)
    anchor_node = len(keep)
    label = components(graph)
    anchor_labels = {label[i] for i in anchor_rows}

    resistance = np.full(graph.n_stations, np.inf)
    for position, row in enumerate(keep):
        if label[row] not in anchor_labels:
            continue
        resistance[row] = (
            pinv[position, position]
            + pinv[anchor_node, anchor_node]
            - 2 * pinv[position, anchor_node]
        )
    resistance[anchor_rows] = 0.0
    return pd.DataFrame(
        {
            "station": graph.stations,
            "component": label,
            "resistance": resistance,
            "n_days": graph.n_days,
        }
    )


def _edge_weights(graph: OverlapGraph, weight: str) -> np.ndarray:
    if weight == "overlap_days":
        return graph.overlap.astype(float)
    if weight == "unit":
        return (graph.overlap > 0).astype(float)
    raise ValueError(f"unknown weight {weight!r}; expected 'overlap_days' or 'unit'")


def decimal_year(date_ord: np.ndarray | float) -> np.ndarray:
    """Calendar year plus the fraction of it elapsed.

    Takes fractional ordinals -- a career midpoint lands between two days -- so it
    floors to a day, converts that day with integer ordinals (pandas rejects a
    *float* ordinal before 1677, and the record starts in 1610), and adds the
    fraction back.
    """
    values = np.atleast_1d(np.asarray(date_ord, dtype=float))
    day = np.floor(values)
    stamps = pd.to_datetime(day.astype(np.int64) - 719_163, unit="D")
    length = np.where(stamps.is_leap_year, 366.0, 365.0)
    elapsed = stamps.day_of_year.to_numpy() - 1.0 + (values - day)
    return stamps.year.to_numpy() + elapsed / length


def temporal_cut_profile(graph: OverlapGraph, years: np.ndarray | None = None) -> pd.DataFrame:
    """How much of the calibration chain crosses each year boundary.

    Stations are split by the midpoint of their career, so each year ``y`` induces
    the bipartition *finished before y* / *still to come*.  The edges crossing it
    are every link a chained method could use to carry calibration from one side
    to the other, and their total matched days is the chain's capacity there.  A
    year with **zero** crossing edges is a severed graph: no chain of overlaps
    reaches across it at all, whatever the method does with the data.

    ``n_before``/``n_after`` say how many stations fall on each side, because a
    cut with almost nothing on one side is empty for a trivial reason -- the
    first and last years of the record always are.

    Returns a frame of ``year, n_edges, overlap_days, n_before, n_after``.
    """
    mid = decimal_year((graph.first_day + graph.last_day) / 2.0)
    if years is None:
        years = np.arange(int(np.floor(mid.min())) + 1, int(np.ceil(mid.max())) + 1)
    years = np.asarray(years)

    n_before = np.searchsorted(np.sort(mid), years, side="right")

    edges = graph.edge_frame()
    if edges.empty:
        return pd.DataFrame(
            {
                "year": years,
                "n_edges": 0,
                "overlap_days": 0,
                "n_before": n_before,
                "n_after": len(mid) - n_before,
            }
        )
    row_a = edges["station_a"].map(graph.index).to_numpy()
    row_b = edges["station_b"].map(graph.index).to_numpy()
    lo = np.minimum(mid[row_a], mid[row_b])
    hi = np.maximum(mid[row_a], mid[row_b])
    weight = edges["overlap_days"].to_numpy().astype(float)

    # An edge crosses every year y with lo < y <= hi; accumulate on a difference
    # array so the whole profile costs one pass over the edges.
    start = np.searchsorted(years, lo, side="right")
    stop = np.searchsorted(years, hi, side="right")
    counts = np.zeros(len(years) + 1)
    weights = np.zeros(len(years) + 1)
    np.add.at(counts, start, 1.0)
    np.add.at(counts, stop, -1.0)
    np.add.at(weights, start, weight)
    np.add.at(weights, stop, -weight)
    return pd.DataFrame(
        {
            "year": years,
            "n_edges": np.cumsum(counts)[:-1].astype(int),
            "overlap_days": np.cumsum(weights)[:-1].astype(np.int64),
            "n_before": n_before,
            "n_after": len(mid) - n_before,
        }
    )


# ---------------------------------------------------------------------------
# Matched day pairs and their lags (Step 3)
#
# ``overlap`` above counts ``|dilate(a) & days(b)|``, which is what the legacy
# MATLAB needed: a scalar edge weight.  A likelihood needs more than a weight --
# it needs the actual matched observations and the lag between them, because the
# pairwise kernel is a function of ``(y_a, y_b, dt)`` and the ``dt`` is what the
# lifetime-based decorrelation acts on.
#
# Two things change from the legacy count.  **The matching is one-to-one**: a
# dilated day can touch up to ``2*tolerance + 1`` of the partner's days, so the
# dilation count double-uses observations and would enter a likelihood as though
# one observer-day were several.  Each day is used at most once per edge here,
# paired with the nearest unused partner day.  And **the lag is kept**, rather
# than collapsed into "simultaneous", which is the assumption
# ``docs/03_legacy/reuse_map.md`` puts on the Drop list.
# ---------------------------------------------------------------------------

#: Largest reported count the pair table represents, matching Step 1's state
#: space (``K = 41``, issue D17).  Counts above it are clipped and the number
#: clipped is reported, never silently dropped.
PAIR_MAX_COUNT = 41


def matched_pairs(
    days_a: np.ndarray,
    days_b: np.ndarray,
    tolerance: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-to-one nearest matching of two observing calendars within a lag.

    Returns ``(index_a, index_b, lag)`` into the two *sorted* day arrays, with
    ``lag = days_b[index_b] - days_a[index_a]``.  Exact-day matches are taken
    first, then the remaining days are paired with the nearest unused partner
    within ``tolerance``; ties go to the earlier partner day, which makes the
    result independent of input order.

    Parameters
    ----------
    days_a, days_b
        Sorted, unique day ordinals.
    tolerance
        Largest ``|lag|`` allowed.  ``0`` gives exact-day matching only.

    Notes
    -----
    Greedy nearest matching is not the global minimum-cost assignment, but at
    these tolerances the two coincide except in rare interleavings, and the
    greedy version is linear rather than cubic.  The difference is reported by
    :func:`pair_table` as ``n_unmatched`` so it can never hide.
    """
    days_a = np.asarray(days_a, dtype=np.int64)
    days_b = np.asarray(days_b, dtype=np.int64)
    if len(days_a) == 0 or len(days_b) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty.astype(np.int16)

    used_a = np.zeros(len(days_a), dtype=bool)
    used_b = np.zeros(len(days_b), dtype=bool)
    out_a: list[np.ndarray] = []
    out_b: list[np.ndarray] = []
    out_lag: list[np.ndarray] = []

    # Pass 0 is the exact-day intersection, which is the overwhelming majority of
    # matches for calendars that overlap at all, and is vectorized.
    shared, index_a, index_b = np.intersect1d(
        days_a, days_b, assume_unique=True, return_indices=True
    )
    if len(shared):
        used_a[index_a] = True
        used_b[index_b] = True
        out_a.append(index_a)
        out_b.append(index_b)
        out_lag.append(np.zeros(len(shared), dtype=np.int16))

    # Then |lag| = 1, 2, ... so the nearest partner always wins, and within a lag
    # the earlier partner wins because the scan runs in day order.
    for distance in range(1, int(tolerance) + 1):
        for sign in (-1, 1):
            free_a = np.flatnonzero(~used_a)
            if len(free_a) == 0:
                break
            targets = days_a[free_a] + sign * distance
            position = np.searchsorted(days_b, targets)
            valid = (position < len(days_b)) & (
                days_b[np.minimum(position, len(days_b) - 1)] == targets
            )
            free_a = free_a[valid]
            position = position[valid]
            open_b = ~used_b[position]
            free_a = free_a[open_b]
            position = position[open_b]
            if len(free_a) == 0:
                continue
            # A target day can be claimed by only one source day; keep the first.
            _unique, first = np.unique(position, return_index=True)
            free_a = free_a[first]
            position = position[first]
            used_a[free_a] = True
            used_b[position] = True
            out_a.append(free_a)
            out_b.append(position)
            out_lag.append(np.full(len(free_a), sign * distance, dtype=np.int16))

    if not out_a:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty.astype(np.int16)
    index_a = np.concatenate(out_a)
    index_b = np.concatenate(out_b)
    lag = np.concatenate(out_lag)
    order = np.argsort(index_a, kind="stable")
    return index_a[order], index_b[order], lag[order]


@dataclass
class PairTable:
    """Every matched observation pair on the graph, in one long table.

    One row per matched ``(observer a day, observer b day)``.  The columns are
    compact on purpose: the real 1610-2026 network produces millions of rows, and
    this table is read once per likelihood evaluation.

    Attributes
    ----------
    edge
        Row index into :attr:`edges` -- which pair of stations this row belongs to.
    year
        Calendar year of observer a's day, the resolution the drift prior works at.
    count_a, count_b
        Reported counts, clipped at :data:`PAIR_MAX_COUNT`.
    lag
        ``day_b - day_a``, in days.
    weight
        How many identical rows this one stands for.  ``1`` everywhere unless the
        table was collapsed (:func:`collapse_pairs`).
    edges
        ``station_a, station_b`` per edge index.
    meta
        Construction parameters and the counts of what was clipped or unmatched.
    """

    edge: np.ndarray
    year: np.ndarray
    count_a: np.ndarray
    count_b: np.ndarray
    lag: np.ndarray
    weight: np.ndarray
    edges: pd.DataFrame
    meta: dict = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return len(self.edge)

    @property
    def n_pairs(self) -> int:
        """Matched observation pairs represented, counting collapsed weight."""
        return int(self.weight.sum())

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def nbytes(self) -> int:
        return int(
            sum(
                a.nbytes
                for a in (
                    self.edge,
                    self.year,
                    self.count_a,
                    self.count_b,
                    self.lag,
                    self.weight,
                )
            )
        )

    def station_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """``(station_a, station_b)`` arrays aligned with :attr:`edges`."""
        return (
            self.edges["station_a"].to_numpy(dtype=np.int64),
            self.edges["station_b"].to_numpy(dtype=np.int64),
        )


def pair_table(
    observations: pd.DataFrame,
    graph: OverlapGraph,
    *,
    tolerance: int | None = None,
    collapse: bool = True,
    max_count: int = PAIR_MAX_COUNT,
) -> PairTable:
    """Build the matched-pair table the pairwise kernels consume.

    Parameters
    ----------
    observations
        Long format: ``date_ord, station, groups``.
    graph
        The overlap graph; only its kept edges are walked.
    tolerance
        Lag allowed, defaulting to the graph's own ``day_tolerance`` so the table
        and the graph always describe the same overlap.
    collapse
        Aggregate identical ``(edge, year, count_a, count_b, lag)`` rows into one
        weighted row.  **Lossless**, because the pairwise kernel depends on a row
        only through those five numbers: the detection ratio is evaluated at the
        year's drift knot and the year's activity level, not per day.  This is what
        makes the real network affordable -- see the compression in ``meta``.
    max_count
        Counts above this are clipped, and ``meta["n_clipped"]`` says how many.
        D17: the two largest reported counts in v1.3 are implausible outliers above
        a tail that stops at 53, so a state space must never be sized from them.
    """
    if tolerance is None:
        tolerance = graph.day_tolerance
    series = {}
    for station, frame in observations.groupby("station", sort=True):
        ordered = frame.sort_values("date_ord")
        series[int(station)] = (
            ordered["date_ord"].to_numpy(dtype=np.int64),
            ordered["groups"].to_numpy(dtype=np.int64),
        )

    edges = graph.edge_frame()
    keep_edges: list[tuple[int, int]] = []
    chunks: list[tuple[np.ndarray, ...]] = []
    n_clipped = 0
    n_unmatched = 0
    for station_a, station_b in zip(
        edges["station_a"].astype(int), edges["station_b"].astype(int), strict=True
    ):
        if station_a not in series or station_b not in series:
            continue
        days_a, counts_a = series[station_a]
        days_b, counts_b = series[station_b]
        index_a, index_b, lag = matched_pairs(days_a, days_b, tolerance)
        if len(index_a) == 0:
            continue
        n_unmatched += (len(days_a) - len(index_a)) + (len(days_b) - len(index_b))
        edge_index = len(keep_edges)
        keep_edges.append((station_a, station_b))
        year = np.floor(decimal_year(days_a[index_a])).astype(np.int32)
        count_a = counts_a[index_a]
        count_b = counts_b[index_b]
        n_clipped += int(np.count_nonzero(count_a > max_count)) + int(
            np.count_nonzero(count_b > max_count)
        )
        chunks.append(
            (
                np.full(len(index_a), edge_index, dtype=np.int32),
                year,
                np.clip(count_a, 0, max_count).astype(np.int16),
                np.clip(count_b, 0, max_count).astype(np.int16),
                lag.astype(np.int8),
            )
        )

    if not chunks:
        raise ValueError("no edge of the graph has a matched observation pair")

    edge = np.concatenate([c[0] for c in chunks])
    year = np.concatenate([c[1] for c in chunks])
    count_a = np.concatenate([c[2] for c in chunks])
    count_b = np.concatenate([c[3] for c in chunks])
    lag = np.concatenate([c[4] for c in chunks])
    weight = np.ones(len(edge), dtype=np.int32)

    table = PairTable(
        edge=edge,
        year=year,
        count_a=count_a,
        count_b=count_b,
        lag=lag,
        weight=weight,
        edges=pd.DataFrame(keep_edges, columns=["station_a", "station_b"]),
        meta={
            "tolerance": int(tolerance),
            "max_count": int(max_count),
            "n_clipped": int(n_clipped),
            "n_unmatched_days": int(n_unmatched),
            "n_rows_raw": int(len(edge)),
            "collapsed": bool(collapse),
        },
    )
    if collapse:
        table = collapse_pairs(table)
    return table


def collapse_pairs(table: PairTable) -> PairTable:
    """Aggregate identical ``(edge, year, count_a, count_b, lag)`` rows.

    Lossless for the pairwise kernels, which read a row only through those five
    numbers.  The legacy MATLAB did the same thing for its robust fits -- it
    "collapse[d] matched (count_1, count_2) pairs to unique pairs weighted by
    frequency" (``A1_Matrix_Calculation_weighted_fit.m``) -- for the same reason,
    one era earlier.
    """
    # The five fields are packed into one int64 so the aggregation is a sort of a
    # 1-D array rather than a lexicographic sort of an (n, 5) one.  On the real
    # network that is the difference between 37 s and a few seconds, and the field
    # widths are asserted rather than assumed.
    year0 = int(table.year.min())
    lag0 = int(table.lag.min())
    fields = (
        (table.edge.astype(np.int64), table.edges.shape[0], 21),
        (table.year.astype(np.int64) - year0, int(table.year.max()) - year0 + 1, 11),
        (table.count_a.astype(np.int64), int(table.count_a.max()) + 1, 7),
        (table.count_b.astype(np.int64), int(table.count_b.max()) + 1, 7),
        (table.lag.astype(np.int64) - lag0, int(table.lag.max()) - lag0 + 1, 7),
    )
    key = np.zeros(table.n_rows, dtype=np.int64)
    for values, extent, bits in fields:
        if extent > (1 << bits):
            raise ValueError(
                f"pair-table field needs more than {bits} bits (extent {extent}); "
                "widen the packing in collapse_pairs"
            )
        key = (key << bits) | values
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    starts = np.flatnonzero(np.concatenate(([True], key_sorted[1:] != key_sorted[:-1])))
    weight = np.add.reduceat(table.weight[order].astype(np.int64), starts).astype(np.int32)
    first = order[starts]
    unique = np.stack(
        [
            table.edge[first].astype(np.int64),
            table.year[first].astype(np.int64),
            table.count_a[first].astype(np.int64),
            table.count_b[first].astype(np.int64),
            table.lag[first].astype(np.int64),
        ],
        axis=1,
    )
    meta = dict(table.meta)
    meta.update(
        collapsed=True,
        n_rows_collapsed=int(len(unique)),
        compression=float(table.n_rows / max(len(unique), 1)),
    )
    return PairTable(
        edge=unique[:, 0].astype(np.int32),
        year=unique[:, 1].astype(np.int32),
        count_a=unique[:, 2].astype(np.int16),
        count_b=unique[:, 3].astype(np.int16),
        lag=unique[:, 4].astype(np.int8),
        weight=weight,
        edges=table.edges,
        meta=meta,
    )
