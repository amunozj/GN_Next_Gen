"""The observer overlap graph, its matched pairs, and the diagnostics on it.

Step 2 needed the graph as a structure to chain along.  Step 3 needs two more
things from it: the **matched observation pairs and their lags**, which is what a
pairwise likelihood consumes, and resistance between *sets* of observers, because
ADR-009's anchors are whole surveys over whole cycles rather than single stations.
"""

from gn_next_gen.graph.diagnostics import (
    active_stations,
    anchor_bridge_report,
    cycle_basis,
    loop_closure,
    observer_diagnostics,
    resistance_between_sets,
    resistance_by_year,
)
from gn_next_gen.graph.overlap import (
    PAIR_MAX_COUNT,
    OverlapGraph,
    PairTable,
    bridges,
    build_overlap_graph,
    collapse_pairs,
    components,
    decimal_year,
    effective_resistance,
    matched_pairs,
    pair_table,
    resistance_to_set,
    temporal_cut_profile,
)

__all__ = [
    "PAIR_MAX_COUNT",
    "OverlapGraph",
    "PairTable",
    "active_stations",
    "anchor_bridge_report",
    "bridges",
    "build_overlap_graph",
    "collapse_pairs",
    "components",
    "cycle_basis",
    "decimal_year",
    "effective_resistance",
    "loop_closure",
    "matched_pairs",
    "observer_diagnostics",
    "pair_table",
    "resistance_between_sets",
    "resistance_by_year",
    "resistance_to_set",
    "temporal_cut_profile",
]
