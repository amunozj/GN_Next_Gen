"""Rebuild Step 3's figures, f29-f40, in both light and dark.

Like ``make_step2_figures.py``, this **skips a figure with a reason** rather than
failing when its inputs are not on disk yet.  Most of Step 3's figures need either
the benchmark grid or the real-data runs, both of which the PI launches detached,
so a session can legitimately produce only some of them -- and should say which.

    python scripts/make_step3_figures.py [--out docs/results/figures/step-3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gn_next_gen.paths import DOCS, OUTPUTS  # noqa: E402
from gn_next_gen.viz import network as viz  # noqa: E402
from gn_next_gen.viz.style import save_both_modes  # noqa: E402

FIGURE_DIR = DOCS / "results" / "figures" / "step-3"
STEP3 = OUTPUTS / "step3"
RUN_ROOT = STEP3 / "runs"
SCORES = OUTPUTS / "step2" / "scores"


def _runs() -> dict[str, dict]:
    """Every completed real-data run, by name."""
    out: dict[str, dict] = {}
    if not RUN_ROOT.is_dir():
        return out
    for directory in sorted(RUN_ROOT.glob("*")):
        meta_path = directory / "meta.json"
        if not (directory / "series.parquet").is_file() or not meta_path.is_file():
            continue
        out[directory.name] = {
            "dir": directory,
            "meta": json.loads(meta_path.read_text()),
        }
    return out


def _skip(code: str, reason: str) -> None:
    print(f"  {code}: skipped -- {reason}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(FIGURE_DIR))
    parser.add_argument(
        "--graph-only", action="store_true", help="only the figures that need no fitted run"
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    runs = _runs()
    print(f"found {len(runs)} completed run(s) under {RUN_ROOT}", flush=True)

    # --- f29, f39: the graph, which needs no fit -----------------------------
    graph_inputs = STEP3 / "graph"
    diagnostics_path = graph_inputs / "diagnostics.parquet"
    if diagnostics_path.is_file():
        diagnostics = pd.read_parquet(diagnostics_path)
        by_year = {
            path.stem.replace("resistance_by_year_", ""): pd.read_parquet(path)
            for path in sorted(graph_inputs.glob("resistance_by_year_*.parquet"))
        }
        anchor_pairs = pd.read_parquet(graph_inputs / "anchor_pairs.parquet")
        cut = pd.read_parquet(graph_inputs / "temporal_cut.parquet")
        save_both_modes(
            lambda mode: viz.resistance_figure(diagnostics, by_year, anchor_pairs, mode),
            out,
            "f29_resistance",
        )
        made.append("f29")
        save_both_modes(
            lambda mode: viz.reach_figure(diagnostics, cut, next(iter(by_year.values())), mode),
            out,
            "f39_reach",
        )
        made.append("f39")
    else:
        _skip("f29/f39", f"run scripts/graph_diagnostics.py first (no {diagnostics_path})")

    # --- f30: path consistency ----------------------------------------------
    closure_path = graph_inputs / "loop_closure.parquet"
    if closure_path.is_file():
        closure = pd.read_parquet(closure_path)
        comparison_path = STEP3 / "concurrent_vs_chained.parquet"
        comparison = pd.read_parquet(comparison_path) if comparison_path.is_file() else None
        save_both_modes(
            lambda mode: viz.loop_closure_figure(closure, comparison, mode),
            out,
            "f30_path_consistency",
        )
        made.append("f30")
    else:
        _skip("f30", "no loop_closure.parquet")

    # --- f40: runtime --------------------------------------------------------
    runtime_path = STEP3 / "runtime.csv"
    if runtime_path.is_file():
        stages = pd.read_csv(runtime_path)
        save_both_modes(lambda mode: viz.runtime_figure(stages, mode), out, "f40_runtime")
        made.append("f40")
    else:
        _skip("f40", "no runtime.csv")

    if args.graph_only:
        print(f"made: {', '.join(made) if made else 'nothing'}", flush=True)
        return 0

    # --- f36, f37: the real-data runs ---------------------------------------
    released = {
        info["meta"]["anchoring"]: pd.read_parquet(info["dir"] / "series.parquet")
        for info in runs.values()
        if info["meta"].get("released")
    }
    if released:
        save_both_modes(lambda mode: viz.series_figure(released, None, mode), out, "f36_series")
        made.append("f36")
    else:
        _skip("f36", "no released run on disk (run_network_calibration.py --runs R1 R2)")

    primary = next((info for info in runs.values() if info["meta"].get("label") == "primary"), None)
    if primary is not None and (primary["dir"] / "slots.parquet").is_file():
        observers = pd.read_parquet(primary["dir"] / "observers.parquet")
        slots = pd.read_parquet(primary["dir"] / "slots.parquet")
        save_both_modes(
            lambda mode: viz.parameter_figure(observers, slots, mode), out, "f37_parameters"
        )
        made.append("f37")
    else:
        _skip("f37", "no primary run with slots.parquet")

    # --- f31-f34, f35, f38: the benchmark grid and the sensitivities ---------
    if not (SCORES / "scores.csv").is_file():
        _skip("f31-f34", "no outputs/step2/scores/scores.csv")
    else:
        scores = pd.read_csv(SCORES / "scores.csv")
        if not scores["method"].isin(["NetCal", "NetCal-M"]).any():
            _skip(
                "f31-f34",
                "the grid has no NetCal cells yet (run_baselines.py --methods NetCal NetCal-M)",
            )

    drift_runs = {
        info["meta"]["label"]: pd.read_parquet(info["dir"] / "series.parquet")
        for name, info in runs.items()
        if "drift_log_sigma_mean" in name
    }
    if len(drift_runs) >= 2:
        save_both_modes(
            lambda mode: viz.series_figure(drift_runs, None, mode), out, "f35_drift_prior"
        )
        made.append("f35")
    else:
        _skip("f35", "needs the two drift-prior sensitivity runs")

    print(f"made: {', '.join(made) if made else 'nothing'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
