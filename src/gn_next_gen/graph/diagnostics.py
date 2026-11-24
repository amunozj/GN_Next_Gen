"""How well each observer, and each year, is tied to an anchored epoch.

Two research streams derived the same result independently and both flagged it as
their own inference rather than a quoted one: in a weighted least-squares link
graph the variance of an observer's scale relative to the reference **is** the
effective resistance between the two nodes, it grows linearly along a single
chain, and parallel paths reduce it like parallel resistors
(``adjacent-methods-report.md``, "Ubercal: simultaneous solution..." and "Equating
quantifies chain drift..."; both *(derived)*).

That makes resistance the natural axis for "how far is this observer from the
anchor", and it is why a concurrent fit is never worse than the best single chain.

**One caveat has to travel with every number this module produces.** Ranked idea
7's own cost column reads: "Very low. **Captures random error only; systematic
bias adds coherently.**" Resistance is the random part of the scale uncertainty
and nothing else. A systematic error on one link transfers at full size to
everything behind it however many parallel paths exist, which is what the
loop-closure test here and the two-kernel comparison elsewhere in Step 3 probe
instead.

What Step 2 measured on this graph, for orientation: 507 usable stations, ~20,000
edges, 14 components with 383 in the largest, 20 bridges, and **39 of 422 stations
with no path to the anchor at all**; 1727-1745 and 1777-1798 have *zero* edges
crossing the year cut.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from gn_next_gen.graph.overlap import (
    OverlapGraph,
    _edge_weights,
    bridges,
    components,
    decimal_year,
    resistance_to_set,
)

#: Resistance is reported against this weighting unless asked otherwise.
#: Conductance proportional to matched days is the one that matches the
#: least-squares derivation: a long well-sampled overlap is a thick wire.
DEFAULT_WEIGHT = "overlap_days"


def _rows(graph: OverlapGraph, stations: Sequence[int] | np.ndarray) -> list[int]:
    """Matrix rows for a set of station ids, de-duplicated and sorted."""
    return sorted({graph.index[int(s)] for s in stations})


def _resistance_between_rows(
    weights: np.ndarray, rows_a: Sequence[int], rows_b: Sequence[int]
) -> float:
    """Effective resistance between two sets, each shorted to a single node.

    Solved as a potential problem rather than through a pseudo-inverse: hold set
    ``a`` at potential 1 and set ``b`` at 0, solve the Laplacian on the interior,
    and read the injected current ``I``.  Since the potential difference is 1,
    ``R = 1 / I``.  One dense solve of the interior, which is what makes a
    per-year sweep affordable -- shorting a set is not something a single
    pseudo-inverse of the unshorted Laplacian can answer.
    """
    rows_a = list(rows_a)
    rows_b = list(rows_b)
    if not rows_a or not rows_b:
        raise ValueError("both station sets must be non-empty")
    overlap = set(rows_a) & set(rows_b)
    if overlap:
        # The sets share an observer, so they are the same node: zero resistance.
        return 0.0

    # Restrict to the component(s) holding set ``b``.  Without this the interior
    # block is singular whenever an unreachable node is present, and the
    # least-squares fallback returns a huge-but-finite resistance that reads as a
    # connection -- on the real graph, year 1610 came back as 7.8e12 rather than
    # inf.  Restricting is also what makes the per-year sweep fast, since the
    # largest component holds 383 of 507 stations.
    n_components, labels = sp.csgraph.connected_components(
        sp.csr_matrix(weights > 0), directed=False
    )
    target_labels = {labels[i] for i in rows_b}
    keep = np.flatnonzero(np.isin(labels, list(target_labels)))
    position = {int(node): i for i, node in enumerate(keep)}
    local_a = [position[i] for i in rows_a if i in position]
    local_b = [position[i] for i in rows_b if i in position]
    if not local_a:
        # Nothing in set ``a`` shares a component with the anchor.
        return float("inf")

    sub = weights[np.ix_(keep, keep)]
    n = sub.shape[0]
    laplacian = np.diag(sub.sum(axis=1)) - sub
    potential = np.zeros(n)
    potential[local_a] = 1.0
    fixed = set(local_a) | set(local_b)
    free = [i for i in range(n) if i not in fixed]
    if free:
        rhs = -laplacian[np.ix_(free, local_a)].sum(axis=1)
        block = laplacian[np.ix_(free, free)]
        potential[free] = np.linalg.solve(block, rhs)
    current = float((laplacian[local_a, :] @ potential).sum())
    if not np.isfinite(current) or current <= 0:
        return float("inf")
    return 1.0 / current


def resistance_between_sets(
    graph: OverlapGraph,
    set_a: Sequence[int] | np.ndarray,
    set_b: Sequence[int] | np.ndarray,
    *,
    weight: str = DEFAULT_WEIGHT,
) -> float:
    """Effective resistance between two anchor sets, each shorted to one node.

    This is the number ADR-009 asks Step 3 for: "the resistance **between the two
    anchors**, which is the number that says whether the two anchorings are tied
    together or floating independently".  Returns ``inf`` when no path connects
    them, which is the statement that the two anchored series float relative to
    one another and that their difference is not a measurement.
    """
    return _resistance_between_rows(
        _edge_weights(graph, weight), _rows(graph, set_a), _rows(graph, set_b)
    )


def active_stations(graph: OverlapGraph, year: int) -> np.ndarray:
    """Stations whose career covers any part of a calendar year."""
    start = decimal_year(graph.first_day)
    end = decimal_year(graph.last_day)
    covered = (start < year + 1) & (end >= year)
    return graph.stations[covered]


def resistance_by_year(
    graph: OverlapGraph,
    anchor: Sequence[int] | np.ndarray,
    *,
    years: np.ndarray | None = None,
    weight: str = DEFAULT_WEIGHT,
) -> pd.DataFrame:
    """Effective resistance from each calendar year to an anchor set.

    A year is represented by every observer active in it, shorted to one node:
    "how well is this year tied to the anchor" is a question about the year's
    observers taken together, not about any one of them.  A year containing an
    anchor observer has resistance 0 by construction, which is the right answer.

    Returns ``year, n_active, resistance, anchored``.
    """
    weights = _edge_weights(graph, weight)
    anchor_rows = _rows(graph, anchor)
    if years is None:
        lo = int(np.floor(decimal_year(graph.first_day).min()))
        hi = int(np.ceil(decimal_year(graph.last_day).max()))
        years = np.arange(lo, hi + 1)
    years = np.asarray(years, dtype=np.int64)

    out = []
    for year in years:
        stations = active_stations(graph, int(year))
        if len(stations) == 0:
            out.append((int(year), 0, float("inf")))
            continue
        rows = _rows(graph, stations)
        resistance = _resistance_between_rows(weights, rows, anchor_rows)
        out.append((int(year), len(stations), resistance))

    frame = pd.DataFrame(out, columns=["year", "n_active", "resistance"])
    frame["anchored"] = np.isfinite(frame["resistance"])
    return frame


def cycle_basis(graph: OverlapGraph, *, max_loops: int | None = None) -> list[list[int]]:
    """A cycle basis of the overlap graph, as lists of station ids.

    Every independent loop in the graph is a path-consistency test waiting to be
    run: the calibration carried around a loop must come back to where it started.
    Battauz developed exactly such tests of "whether different chains give
    different results", though that reference is **title-level only** in
    ``docs/02_research/`` and is not the source of the construction here.

    ``max_loops`` keeps the shortest loops, which are the most informative: a
    two-edge disagreement localizes the inconsistency, a thirty-edge one does not.
    """
    import networkx as nx

    network = nx.Graph()
    network.add_nodes_from(int(s) for s in graph.stations)
    edges = graph.edge_frame()
    network.add_edges_from(
        zip(edges["station_a"].astype(int), edges["station_b"].astype(int), strict=True)
    )
    loops = [list(map(int, loop)) for loop in nx.cycle_basis(network)]
    loops.sort(key=len)
    if max_loops is not None:
        loops = loops[:max_loops]
    return loops


def loop_closure(
    graph: OverlapGraph,
    log_ratios: dict[tuple[int, int], float],
    *,
    max_loops: int | None = 2000,
) -> pd.DataFrame:
    """How far independently estimated pairwise links fail to close around a loop.

    ``log_ratios[(a, b)]`` is the estimated ``log(k_a / k_b)`` for that edge.  A
    consistent set of per-observer scales makes every loop sum to exactly zero,
    because a difference of potentials always does.  A non-zero sum therefore
    cannot be fixed by any choice of per-observer scale: it is misfit, drift or a
    data error, and it is the part of the error that a concurrent fit **absorbs
    into residuals rather than removes**.

    This is the diagnostic the equating section asks to be kept beside the
    concurrent model: "Keep a Haberman-style linking of separately estimated
    pairwise calibrations as a diagnostic. Treat disagreement between the two as a
    misspecification signal."

    Returns ``loop_id, length, closure, abs_closure``, one row per loop that has
    an estimate on every one of its edges.
    """
    rows = []
    for loop_id, loop in enumerate(cycle_basis(graph, max_loops=max_loops)):
        total = 0.0
        complete = True
        for position, station_a in enumerate(loop):
            station_b = loop[(position + 1) % len(loop)]
            if (station_a, station_b) in log_ratios:
                total += log_ratios[(station_a, station_b)]
            elif (station_b, station_a) in log_ratios:
                total -= log_ratios[(station_b, station_a)]
            else:
                complete = False
                break
        if complete:
            rows.append((loop_id, len(loop), total, abs(total)))
    return pd.DataFrame(rows, columns=["loop_id", "length", "closure", "abs_closure"])


def observer_diagnostics(
    graph: OverlapGraph,
    anchors: dict[str, Sequence[int]],
    *,
    weight: str = DEFAULT_WEIGHT,
) -> pd.DataFrame:
    """One diagnostic row per station -- the step's "every observer" criterion.

    Columns: ``station, component, component_size, degree, overlap_days,
    on_bridge, n_days, first_year, last_year``, then ``resistance_<name>`` and
    ``anchored_<name>`` for each named anchor set, then ``anchored`` (reachable
    from *any* anchor) and ``fingerprint_only`` (reachable from none, so its
    calibration rests entirely on the fingerprint term).

    Every station in the graph gets a row, including the ones in components the
    anchor cannot reach.  Step 2 found 39 such stations out of 422, and the
    Laplacian's one exact null mode per connected component is why their level is
    not identified by overlap at all.
    """
    labels = components(graph)
    sizes = np.bincount(labels)
    adjacency = graph.overlap > 0
    bridge_stations: set[int] = set()
    for station_a, station_b in bridges(graph):
        bridge_stations.add(station_a)
        bridge_stations.add(station_b)
    on_bridge = np.isin(graph.stations, list(bridge_stations))

    frame = pd.DataFrame(
        {
            "station": graph.stations,
            "component": labels,
            "component_size": sizes[labels],
            "degree": adjacency.sum(axis=1).astype(np.int64),
            "overlap_days": graph.overlap.sum(axis=1).astype(np.int64),
            "on_bridge": on_bridge,
            "n_days": graph.n_days,
            "first_year": np.floor(decimal_year(graph.first_day)).astype(np.int64),
            "last_year": np.floor(decimal_year(graph.last_day)).astype(np.int64),
        }
    )

    reachable = np.zeros(graph.n_stations, dtype=bool)
    for name, anchor in anchors.items():
        present = [int(s) for s in anchor if int(s) in graph.index]
        if not present:
            frame[f"resistance_{name}"] = np.inf
            frame[f"anchored_{name}"] = False
            continue
        table = resistance_to_set(graph, present, weight=weight)
        merged = table.set_index("station")["resistance"].reindex(frame["station"])
        frame[f"resistance_{name}"] = merged.to_numpy()
        anchored = np.isfinite(merged.to_numpy())
        frame[f"anchored_{name}"] = anchored
        reachable |= anchored

    frame["anchored"] = reachable
    frame["fingerprint_only"] = ~reachable
    return frame


def anchor_bridge_report(
    graph: OverlapGraph,
    anchors: dict[str, Sequence[int]],
    *,
    weight: str = DEFAULT_WEIGHT,
) -> pd.DataFrame:
    """Resistance between every pair of named anchor sets.

    ADR-009's consequence section is specific about why this matters: cycle 20 is
    in neither anchor, is "complete in **KMAS alone**", and "KMAS bridges the two
    anchors ... which is what makes the Step 3 effective-resistance diagnostics
    between them meaningful instead of degenerate".  If this table says ``inf``,
    the two anchored series are not on a common scale and their difference is a
    convention rather than a measurement.
    """
    names = sorted(anchors)
    rows = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            rows.append(
                (
                    left,
                    right,
                    resistance_between_sets(graph, anchors[left], anchors[right], weight=weight),
                )
            )
    frame = pd.DataFrame(rows, columns=["anchor_a", "anchor_b", "resistance"])
    frame["connected"] = np.isfinite(frame["resistance"])
    return frame
