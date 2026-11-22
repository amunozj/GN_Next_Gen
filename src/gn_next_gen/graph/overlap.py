"""Who observed alongside whom, and how far every observer sits from the anchor.

Chained calibration's random error grows like the **effective resistance** between
an observer and the reference on this graph, which is identifiability trap 3 of
the assessment (§6) and the quantity the backbone baseline is judged against.  The
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
