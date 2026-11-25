"""The overlap graph's diagnostics on the real record -- fast, and needs no fit.

Everything here is a property of **who observed alongside whom**, so it is
available before any calibration runs and is what f29 and f39 are drawn from.
Minutes, not hours, so a session may run it.

    python scripts/graph_diagnostics.py --config configs/step3_network.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gn_next_gen.graph import (  # noqa: E402
    anchor_bridge_report,
    build_overlap_graph,
    cycle_basis,
    loop_closure,
    observer_diagnostics,
    resistance_by_year,
    temporal_cut_profile,
)
from gn_next_gen.io import load_observations  # noqa: E402
from gn_next_gen.models import network as nw  # noqa: E402
from gn_next_gen.paths import OUTPUTS  # noqa: E402

OUT = OUTPUTS / "step3" / "graph"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/step3_network.yaml")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)

    from gn_next_gen.config import load_run_config

    config = load_run_config(args.config)
    options = dict(config.params["observations"])
    graph_options = dict(config.params["graph"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    observations = load_observations(
        release=config.release,
        duplicate_rule=config.duplicate_rule,
        kmas_role=config.kmas_role,
        include_extension=options.get("include_extension", True),
        extension_threshold=options.get("extension_threshold", "anchor"),
        extension_post_v13_only=options.get("extension_post_v13_only", False),
    )

    # Each anchoring joins the network as its own pseudo-station, so the two can be
    # present at once and the resistance *between* them is well posed -- ADR-009's
    # "the number that says whether the two anchorings are tied together".
    anchors: dict[str, list[int]] = {}
    frames = [observations[["date_ord", "station", "groups"]]]
    for index, entry in enumerate(config.params["released"]):
        anchoring = entry["anchoring"]
        station = nw.ANCHOR_STATION + index
        rows, _threshold = nw.anchor_observations(anchoring, station=station)
        frames.append(rows[["date_ord", "station", "groups"]])
        anchors[anchoring] = [station]
    network = pd.concat(frames, ignore_index=True)

    graph = build_overlap_graph(network, **graph_options)
    print(
        f"graph: {graph.n_stations} stations, {graph.n_edges} edges ({time.time() - started:.1f}s)",
        flush=True,
    )

    diagnostics = observer_diagnostics(graph, anchors)
    diagnostics.to_parquet(out / "diagnostics.parquet", index=False)

    for anchoring, stations in anchors.items():
        table = resistance_by_year(graph, stations)
        table.to_parquet(out / f"resistance_by_year_{anchoring}.parquet", index=False)
        reachable = int(table["anchored"].sum())
        first = table.loc[table["anchored"], "year"]
        print(
            f"  {anchoring}: {reachable} of {len(table)} years reach the anchor; "
            f"earliest {int(first.min()) if len(first) else 'none'}",
            flush=True,
        )

    pairs = anchor_bridge_report(graph, anchors)
    pairs.to_parquet(out / "anchor_pairs.parquet", index=False)
    for row in pairs.itertuples():
        print(
            f"  {row.anchor_a} <-> {row.anchor_b}: "
            f"{f'R = {row.resistance:.3e}' if row.connected else 'SEVERED'}",
            flush=True,
        )

    cut = temporal_cut_profile(graph)
    cut.to_parquet(out / "temporal_cut.parquet", index=False)
    usable = cut[(cut["n_before"] >= 10) & (cut["n_after"] >= 10)]
    severed = usable[usable["n_edges"] == 0]
    print(f"  severed years: {len(severed)} of {len(usable)}", flush=True)

    # Path consistency, using the legacy-style robust pairwise fits as the
    # independently estimated links -- the Haberman-style diagnostic the equating
    # section asks to be kept beside the concurrent model.
    from gn_next_gen.baselines.base import estimate_edges, station_series
    from gn_next_gen.baselines.pathfinding import PathfindingPair

    series = station_series(network)
    estimates = estimate_edges(graph, series, PathfindingPair())
    ratios = {
        key: float(np.log(estimate.factor))
        for key, estimate in estimates.items()
        if estimate.usable and estimate.factor > 0
    }
    closure = loop_closure(graph, ratios, max_loops=4000)
    closure.to_parquet(out / "loop_closure.parquet", index=False)
    if not closure.empty:
        print(
            f"  loop closure: {len(closure)} loops, median |closure| "
            f"{closure['abs_closure'].median():.4f}, "
            f"p90 {closure['abs_closure'].quantile(0.9):.4f}",
            flush=True,
        )

    summary = {
        "n_stations": int(graph.n_stations),
        "n_edges": int(graph.n_edges),
        "n_components": int(diagnostics["component"].nunique()),
        "largest_component": int(diagnostics["component_size"].max()),
        "n_bridges": int(diagnostics["on_bridge"].sum()),
        "n_fingerprint_only": int(diagnostics["fingerprint_only"].sum()),
        "n_loops": int(len(closure)),
        "median_abs_closure": float(closure["abs_closure"].median()) if len(closure) else None,
        "graph": graph_options,
        "seconds": time.time() - started,
        "n_cycle_basis": len(cycle_basis(graph, max_loops=4000)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
