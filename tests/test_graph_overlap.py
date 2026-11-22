"""The overlap graph, and the resistance that measures chained-calibration error."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gn_next_gen.graph import (
    bridges,
    build_overlap_graph,
    components,
    effective_resistance,
    resistance_to_set,
)


def _chain(spans, groups_cycle=7):
    """Stations observing over the given day spans, with varied counts."""
    rows = [
        {"date_ord": day, "station": station, "groups": day % groups_cycle}
        for station, span in spans.items()
        for day in span
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def path_graph():
    return build_overlap_graph(_chain({1: range(0, 400), 2: range(200, 600), 3: range(500, 900)}))


def test_a_path_graph_has_additive_resistance(path_graph):
    """The property the whole diagnostic rests on: resistance adds along a chain."""
    assert effective_resistance(path_graph, 1, 2, weight="unit") == pytest.approx(1.0)
    assert effective_resistance(path_graph, 2, 3, weight="unit") == pytest.approx(1.0)
    assert effective_resistance(path_graph, 1, 3, weight="unit") == pytest.approx(2.0)


def test_parallel_paths_lower_the_resistance():
    """Why concurrent calibration beats chaining -- two routes halve the variance."""
    series = _chain(
        {
            1: range(0, 400),
            2: range(200, 600),  # route A
            3: range(200, 600),  # route B, same span
            4: range(500, 900),
        }
    )
    graph = build_overlap_graph(series)
    assert effective_resistance(graph, 1, 4, weight="unit") == pytest.approx(1.0)

    one_route = build_overlap_graph(
        _chain({1: range(0, 400), 2: range(200, 600), 4: range(500, 900)})
    )
    assert effective_resistance(one_route, 1, 4, weight="unit") == pytest.approx(2.0)


def test_disconnected_stations_have_infinite_resistance():
    graph = build_overlap_graph(_chain({1: range(0, 400), 2: range(5000, 5400)}))
    assert graph.n_edges == 0
    assert effective_resistance(graph, 1, 2) == float("inf")
    assert len(set(components(graph))) == 2


def test_every_link_in_a_chain_is_a_bridge(path_graph):
    assert sorted(bridges(path_graph)) == [(1, 2), (2, 3)]


def test_a_triangle_has_no_bridge():
    graph = build_overlap_graph(
        _chain(
            {1: range(0, 400), 2: range(200, 600), 3: list(range(300, 500)) + list(range(0, 200))}
        )
    )
    assert graph.n_edges == 3
    assert bridges(graph) == []


def test_the_day_tolerance_creates_overlap_that_strict_matching_misses():
    """DayTol = 2: observers on alternating days still overlap."""
    series = _chain({1: range(0, 400, 2), 2: range(1, 401, 2)})
    strict = build_overlap_graph(series, day_tolerance=0, min_overlap_days=1)
    tolerant = build_overlap_graph(series, day_tolerance=2, min_overlap_days=1)
    assert strict.n_edges == 0
    assert tolerant.n_edges == 1


def test_overlap_is_symmetric_regardless_of_station_order():
    """The legacy single-direction write made this depend on loop order."""
    series = _chain({1: range(0, 400), 2: range(200, 600)})
    graph = build_overlap_graph(series)
    assert np.array_equal(graph.overlap, graph.overlap.T)


def test_an_observer_with_too_few_distinct_counts_is_dropped():
    """minObVal = 4: a station that always writes the same number anchors nothing."""
    rows = [{"date_ord": d, "station": 1, "groups": d % 7} for d in range(400)]
    rows += [{"date_ord": d, "station": 2, "groups": 3} for d in range(200, 600)]
    graph = build_overlap_graph(pd.DataFrame(rows))
    assert graph.stations.tolist() == [1]


def test_short_overlaps_are_dropped():
    series = _chain({1: range(0, 400), 2: range(390, 800)})
    assert build_overlap_graph(series, min_overlap_days=60).n_edges == 0
    assert build_overlap_graph(series, min_overlap_days=5).n_edges == 1


def test_resistance_to_an_anchor_set_shorts_it_to_one_node(path_graph):
    table = resistance_to_set(path_graph, [1], weight="unit")
    assert table.set_index("station")["resistance"].to_dict() == pytest.approx(
        {1: 0.0, 2: 1.0, 3: 2.0}
    )

    both_ends = resistance_to_set(path_graph, [1, 3], weight="unit")
    assert both_ends.set_index("station")["resistance"][2] == pytest.approx(0.5)


def test_unreachable_stations_report_infinity_not_zero():
    series = _chain({1: range(0, 400), 2: range(200, 600), 3: range(9000, 9400)})
    graph = build_overlap_graph(series)
    table = resistance_to_set(graph, [1], weight="unit").set_index("station")
    assert np.isinf(table["resistance"][3])
    assert np.isfinite(table["resistance"][2])


def test_an_unknown_weight_is_rejected(path_graph):
    with pytest.raises(ValueError, match="unknown weight"):
        effective_resistance(path_graph, 1, 2, weight="inverse-variance")


@pytest.mark.slow
def test_the_real_network_is_mostly_one_component():
    """The graph Step 3 will diagnose; here only that it builds and hangs together."""
    from gn_next_gen.io import load_observations

    graph = build_overlap_graph(
        load_observations(include_extension=True, kmas_role="observer_counts")
    )
    labels = components(graph)
    largest = np.bincount(labels).max()
    assert graph.n_stations > 400
    assert largest / graph.n_stations > 0.8
    assert len(bridges(graph)) > 0
