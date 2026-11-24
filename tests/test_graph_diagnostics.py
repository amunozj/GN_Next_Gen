"""Graph diagnostics: resistance between sets, per-year resistance, loop closure.

Every resistance assertion here is against a value computable by hand from the
series/parallel rules, because the whole point of the quantity is that it is the
variance a chained calibration accumulates -- a number this project reports, so a
number that has to be right rather than merely plausible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gn_next_gen.graph import diagnostics as dg
from gn_next_gen.graph.overlap import (
    OverlapGraph,
    build_overlap_graph,
    collapse_pairs,
    matched_pairs,
    pair_table,
)


def toy_graph(overlap: np.ndarray, *, first=None, last=None) -> OverlapGraph:
    n = overlap.shape[0]
    return OverlapGraph(
        stations=np.arange(n, dtype=np.int64),
        overlap=np.asarray(overlap, dtype=np.int64),
        first_day=np.asarray(first if first is not None else [700_000] * n, dtype=np.int64),
        last_day=np.asarray(last if last is not None else [700_100] * n, dtype=np.int64),
        n_days=np.full(n, 100, dtype=np.int64),
    )


def path_graph(n: int, conductance: int = 1) -> OverlapGraph:
    overlap = np.zeros((n, n), dtype=np.int64)
    for i in range(n - 1):
        overlap[i, i + 1] = overlap[i + 1, i] = conductance
    return toy_graph(overlap)


# --------------------------------------------------------------------------
# Effective resistance between sets
# --------------------------------------------------------------------------


def test_two_nodes_is_one_over_conductance():
    graph = toy_graph(np.array([[0, 4], [4, 0]]))
    assert dg.resistance_between_sets(graph, [0], [1]) == pytest.approx(0.25)


@pytest.mark.parametrize("n_edges", [1, 2, 3, 5, 9])
def test_series_resistance_grows_linearly_along_a_chain(n_edges):
    """The chained-calibration result: variance grows linearly with path length."""
    graph = path_graph(n_edges + 1)
    assert dg.resistance_between_sets(graph, [0], [n_edges]) == pytest.approx(float(n_edges))


def test_two_parallel_paths_halve_the_resistance():
    """ "Two independent parallel paths halve link variance" -- the equating result."""
    square = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.int64)
    graph = toy_graph(square)
    # Two parallel routes of two unit edges each: 2 || 2 = 1.
    assert dg.resistance_between_sets(graph, [0], [2]) == pytest.approx(1.0)
    assert dg.resistance_between_sets(graph, [0], [2]) < dg.resistance_between_sets(
        path_graph(3), [0], [2]
    )


def test_disconnected_sets_are_infinite_not_merely_large():
    """A floating component must read inf, never a huge finite number.

    This is a regression test with a measured provenance: before the component
    restriction, the real graph's year 1610 came back as 7.8e12 and
    ``anchored = True`` because the singular interior block fell through to a
    least-squares solve.  A huge finite resistance reads as a connection.
    """
    overlap = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.int64)
    graph = toy_graph(overlap)
    assert dg.resistance_between_sets(graph, [0], [2]) == float("inf")
    assert dg.resistance_between_sets(graph, [0, 2], [3]) == pytest.approx(1.0)


def test_shorting_a_set_is_not_the_same_as_picking_one_member():
    """ADR-009's anchors are whole surveys over whole cycles, not single stations."""
    graph = path_graph(5)
    both = dg.resistance_between_sets(graph, [0], [3, 4])
    just_far = dg.resistance_between_sets(graph, [0], [4])
    assert both < just_far
    assert both == pytest.approx(3.0)


def test_overlapping_sets_are_the_same_node():
    graph = path_graph(4)
    assert dg.resistance_between_sets(graph, [0, 1], [1, 2]) == 0.0


def test_removing_a_bridge_severs_the_two_anchors():
    """The bridge is the single point of failure the diagnostic exists to find."""
    overlap = np.zeros((6, 6), dtype=np.int64)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 2), (3, 5)]:
        overlap[a, b] = overlap[b, a] = 1
    graph = toy_graph(overlap)
    assert np.isfinite(dg.resistance_between_sets(graph, [0], [5]))
    assert (2, 3) in dg.bridges(graph) or (3, 2) in dg.bridges(graph)
    cut = overlap.copy()
    cut[2, 3] = cut[3, 2] = 0
    assert dg.resistance_between_sets(toy_graph(cut), [0], [5]) == float("inf")


def test_unit_weighting_counts_hops_and_day_weighting_does_not():
    overlap = np.array([[0, 1000], [1000, 0]], dtype=np.int64)
    graph = toy_graph(overlap)
    assert dg.resistance_between_sets(graph, [0], [1], weight="unit") == pytest.approx(1.0)
    assert dg.resistance_between_sets(graph, [0], [1], weight="overlap_days") == pytest.approx(
        0.001
    )


# --------------------------------------------------------------------------
# Per-year resistance
# --------------------------------------------------------------------------


def test_resistance_by_year_marks_the_anchor_year_zero_and_gaps_infinite():
    overlap = np.zeros((3, 3), dtype=np.int64)
    overlap[0, 1] = overlap[1, 0] = 1
    import datetime as dt

    def ordinal(year):
        return dt.date(year, 1, 1).toordinal()

    graph = toy_graph(
        overlap,
        first=[ordinal(1900), ordinal(1905), ordinal(1800)],
        last=[ordinal(1910), ordinal(1915), ordinal(1810)],
    )
    table = dg.resistance_by_year(graph, [1], years=np.array([1805, 1901, 1906]))
    byyear = table.set_index("year")
    assert byyear.loc[1805, "n_active"] == 1
    assert not byyear.loc[1805, "anchored"]  # station 2 floats in its own component
    assert byyear.loc[1901, "resistance"] == pytest.approx(1.0)
    assert byyear.loc[1906, "resistance"] == 0.0  # the anchor itself is observing


def test_every_year_asked_for_gets_a_row():
    graph = path_graph(4)
    years = np.arange(1900, 1910)
    table = dg.resistance_by_year(graph, [0], years=years)
    assert list(table["year"]) == list(years)


# --------------------------------------------------------------------------
# Path consistency
# --------------------------------------------------------------------------


def test_loop_closure_is_exactly_zero_for_consistent_scales():
    """A set of per-observer scales is a potential, and a potential always closes."""
    overlap = np.zeros((4, 4), dtype=np.int64)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]:
        overlap[a, b] = overlap[b, a] = 1
    graph = toy_graph(overlap)
    log_k = {0: 0.0, 1: 0.3, 2: -0.2, 3: 0.7}
    ratios = {}
    edges = graph.edge_frame()
    for a, b in zip(edges["station_a"], edges["station_b"], strict=True):
        ratios[(int(a), int(b))] = log_k[int(a)] - log_k[int(b)]
    closure = dg.loop_closure(graph, ratios)
    assert len(closure) >= 2
    assert np.allclose(closure["closure"].to_numpy(), 0.0, atol=1e-12)


def test_loop_closure_recovers_a_planted_inconsistency():
    overlap = np.zeros((3, 3), dtype=np.int64)
    for a, b in [(0, 1), (1, 2), (0, 2)]:
        overlap[a, b] = overlap[b, a] = 1
    graph = toy_graph(overlap)
    ratios = {(0, 1): 0.1, (1, 2): 0.1, (0, 2): 0.2}
    assert dg.loop_closure(graph, ratios)["abs_closure"].max() == pytest.approx(0.0)
    ratios[(0, 2)] = 0.5  # inconsistent by 0.3
    closure = dg.loop_closure(graph, ratios)
    assert closure["abs_closure"].max() == pytest.approx(0.3)


def test_loop_closure_skips_loops_with_a_missing_edge():
    overlap = np.zeros((3, 3), dtype=np.int64)
    for a, b in [(0, 1), (1, 2), (0, 2)]:
        overlap[a, b] = overlap[b, a] = 1
    graph = toy_graph(overlap)
    assert dg.loop_closure(graph, {(0, 1): 0.1, (1, 2): 0.1}).empty


# --------------------------------------------------------------------------
# The per-observer table -- the step's "every observer" criterion
# --------------------------------------------------------------------------


def test_observer_diagnostics_covers_every_station_and_flags_floaters():
    overlap = np.zeros((5, 5), dtype=np.int64)
    for a, b in [(0, 1), (1, 2)]:
        overlap[a, b] = overlap[b, a] = 10
    overlap[3, 4] = overlap[4, 3] = 10
    graph = toy_graph(overlap)
    table = dg.observer_diagnostics(graph, {"main": [0], "other": [3]})

    assert len(table) == graph.n_stations
    assert set(table["station"]) == set(graph.stations)
    assert table["anchored_main"].tolist() == [True, True, True, False, False]
    assert table["anchored_other"].tolist() == [False, False, False, True, True]
    # Reachable from *some* anchor, so nothing is fingerprint-only here.
    assert not table["fingerprint_only"].any()

    only_one = dg.observer_diagnostics(graph, {"main": [0]})
    assert only_one["fingerprint_only"].tolist() == [False, False, False, True, True]


def test_anchor_bridge_report_says_whether_two_anchorings_are_tied():
    overlap = np.zeros((4, 4), dtype=np.int64)
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        overlap[a, b] = overlap[b, a] = 1
    graph = toy_graph(overlap)
    report = dg.anchor_bridge_report(graph, {"left": [0], "right": [3]})
    assert len(report) == 1
    assert bool(report.loc[0, "connected"])
    assert report.loc[0, "resistance"] == pytest.approx(3.0)

    severed = overlap.copy()
    severed[1, 2] = severed[2, 1] = 0
    report = dg.anchor_bridge_report(toy_graph(severed), {"left": [0], "right": [3]})
    assert not bool(report.loc[0, "connected"])


# --------------------------------------------------------------------------
# Matched pairs and the pair table
# --------------------------------------------------------------------------


def test_matched_pairs_prefers_exact_days():
    days_a = np.array([1, 2, 3, 10])
    days_b = np.array([2, 3, 4, 20])
    index_a, index_b, lag = matched_pairs(days_a, days_b, 2)
    assert list(zip(days_a[index_a], days_b[index_b], lag, strict=True)) == [
        (2, 2, 0),
        (3, 3, 0),
    ]


def test_matched_pairs_is_one_to_one():
    """The legacy dilation count double-uses observations; a likelihood must not.

    One day of ``a``, dilated by +/-2, touches five of ``b``'s days.  The legacy
    count calls that five matches; here it is one.
    """
    days_a = np.array([10])
    days_b = np.array([8, 9, 10, 11, 12])
    index_a, index_b, lag = matched_pairs(days_a, days_b, 2)
    assert len(index_a) == 1
    assert lag.tolist() == [0]

    days_a = np.array([8, 9, 10, 11, 12])
    days_b = np.array([10])
    index_a, index_b, _ = matched_pairs(days_a, days_b, 2)
    assert len(index_a) == 1


def test_matched_pairs_takes_the_nearest_partner():
    days_a = np.array([5])
    days_b = np.array([7, 9])
    _index_a, index_b, lag = matched_pairs(days_a, days_b, 4)
    assert days_b[index_b].tolist() == [7]
    assert lag.tolist() == [2]


def test_matched_pairs_respects_the_tolerance():
    days_a = np.array([5])
    days_b = np.array([9])
    assert len(matched_pairs(days_a, days_b, 3)[0]) == 0
    assert len(matched_pairs(days_a, days_b, 4)[0]) == 1
    assert len(matched_pairs(days_a, days_b, 0)[0]) == 0


def test_matched_pairs_is_order_independent_in_lag_sign():
    days_a = np.array([5, 6])
    days_b = np.array([4, 7])
    _ia, ib, lag = matched_pairs(days_a, days_b, 2)
    # 5 -> 4 (lag -1) and 6 -> 7 (lag +1); both at distance 1, both one-to-one.
    assert sorted(lag.tolist()) == [-1, 1]
    assert sorted(days_b[ib].tolist()) == [4, 7]


def _toy_observations() -> tuple[pd.DataFrame, OverlapGraph]:
    rng = np.random.default_rng(0)
    frames = []
    for station in (1, 2, 3):
        days = np.arange(700_000, 700_400)
        frames.append(
            pd.DataFrame(
                {
                    "date_ord": days,
                    "station": station,
                    "groups": rng.integers(0, 8, len(days)),
                }
            )
        )
    observations = pd.concat(frames, ignore_index=True)
    graph = build_overlap_graph(observations, min_overlap_days=10)
    return observations, graph


def test_pair_table_collapse_is_lossless():
    observations, graph = _toy_observations()
    raw = pair_table(observations, graph, collapse=False)
    collapsed = pair_table(observations, graph, collapse=True)

    assert collapsed.n_pairs == raw.n_rows
    assert collapsed.n_rows <= raw.n_rows
    assert collapsed.meta["compression"] >= 1.0

    def contingency(table):
        return (
            pd.DataFrame(
                {
                    "edge": table.edge,
                    "year": table.year,
                    "a": table.count_a,
                    "b": table.count_b,
                    "lag": table.lag,
                    "w": table.weight,
                }
            )
            .groupby(["edge", "year", "a", "b", "lag"])["w"]
            .sum()
        )

    pd.testing.assert_series_equal(contingency(raw), contingency(collapsed))


def test_pair_table_collapse_is_idempotent():
    observations, graph = _toy_observations()
    once = pair_table(observations, graph, collapse=True)
    twice = collapse_pairs(once)
    assert twice.n_rows == once.n_rows
    assert twice.n_pairs == once.n_pairs


def test_pair_table_clips_counts_and_says_how_many():
    observations, graph = _toy_observations()
    observations = observations.copy()
    observations.loc[observations.index[:3], "groups"] = 99
    table = pair_table(observations, graph, max_count=41)
    assert table.meta["n_clipped"] > 0
    assert int(table.count_a.max()) <= 41
    assert int(table.count_b.max()) <= 41


def test_pair_table_uses_the_graphs_own_tolerance_by_default():
    observations, graph = _toy_observations()
    table = pair_table(observations, graph)
    assert table.meta["tolerance"] == graph.day_tolerance


def test_pair_table_edges_are_a_subset_of_the_graphs():
    observations, graph = _toy_observations()
    table = pair_table(observations, graph)
    graph_edges = {
        (int(a), int(b))
        for a, b in zip(
            graph.edge_frame()["station_a"], graph.edge_frame()["station_b"], strict=True
        )
    }
    for a, b in zip(table.edges["station_a"], table.edges["station_b"], strict=True):
        assert (int(a), int(b)) in graph_edges
