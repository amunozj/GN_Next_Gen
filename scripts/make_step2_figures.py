"""Build f13-f28 and f41-f42 from whatever the grid has produced.

Short (< 3 minutes) and safe to re-run: each figure is skipped, with a reason, if
its inputs are not on disk yet, so the session can build the set from a two-world
slice and the PI can rebuild it from the full grid with the same command.

Usage
-----
    python scripts/make_step2_figures.py [--out-dir docs/results/figures/step-2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gn_next_gen.benchmark.observers import CASE_STUDY_KINDS  # noqa: E402
from gn_next_gen.benchmark.scoring import summarise_case_studies  # noqa: E402
from gn_next_gen.benchmark.store import BENCHMARK_ROOT, read_observations  # noqa: E402
from gn_next_gen.graph import (  # noqa: E402
    bridges,
    build_overlap_graph,
    decimal_year,
    resistance_to_set,
    temporal_cut_profile,
)
from gn_next_gen.paths import DATA_RAW, DOCS, OUTPUTS  # noqa: E402
from gn_next_gen.viz import benchmark as viz  # noqa: E402
from gn_next_gen.viz.style import save_both_modes  # noqa: E402

SCORE_DIR = OUTPUTS / "step2" / "scores"
SAMPLING_DIR = OUTPUTS / "step2" / "sampling_only"
A157_DIR = OUTPUTS / "step2" / "a157"
DEFAULT_OUT = DOCS / "results" / "figures" / "step-2"
DATA_OUT = DOCS / "results" / "data"


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.is_file() else None


def _first_realization(root: Path, scenario: str) -> int | None:
    seeds = sorted(
        int(p.name.split("-")[1])
        for p in (Path(root) / scenario).glob("seed-*")
        if "__" not in p.name
    )
    return seeds[0] if seeds else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--data-root", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument(
        "--scenario",
        default="S2",
        help="which worlds the design figures use; the full record by default, so a "
        "caption about 1610-2026 is drawn from a world that spans it",
    )
    parser.add_argument(
        "--network-scenario",
        default="S2",
        help="which worlds f28 draws the overlap graph from; the full record by default",
    )
    args = parser.parse_args()

    out = args.out_dir
    written: list[Path] = []
    skipped: list[str] = []

    written += save_both_modes(viz.design_figure, out, "f13_benchmark_design")

    seed = _first_realization(args.data_root, args.scenario)
    if seed is None:
        skipped.append(f"f14, f15, f16, f17: no {args.scenario} realization generated")
    else:
        from gn_next_gen.benchmark.scenarios import REFERENCE_STATION
        from gn_next_gen.benchmark.store import read_sealed_parameters, read_sealed_truth

        truth = read_sealed_truth(args.scenario, seed, root=args.data_root)
        parameters = read_sealed_parameters(args.scenario, seed, root=args.data_root)
        observations = read_observations(args.scenario, seed, root=args.data_root)

        counts = truth.reference_counts()
        yearly = (
            pd.Series(counts.to_numpy(), index=viz._years(counts.index.to_numpy()))
            .groupby(level=0)
            .mean()
        )
        written += save_both_modes(
            lambda mode: viz.world_figure(yearly, truth.groups, mode), out, "f14_replay_world"
        )

        graph = build_overlap_graph(observations)
        resistance = resistance_to_set(graph, [REFERENCE_STATION])
        first_day = observations.groupby("station")["date_ord"].min().reindex(resistance["station"])
        resistance = resistance.assign(first_ord=first_day.to_numpy())
        written += save_both_modes(
            lambda mode: viz.graph_figure(resistance, graph.edge_frame(), mode),
            out,
            "f15_overlap_graph",
        )
        written += save_both_modes(
            lambda mode: viz.observer_population_figure(parameters, mode),
            out,
            "f16_observer_population",
        )

        summary = (
            observations.groupby("station")["date_ord"]
            .agg(first_ord="min", last_ord="max", n_days="size")
            .reset_index()
        )
        written += save_both_modes(
            lambda mode: viz.calendar_figure(summary, mode), out, "f17_calendars"
        )

    scores = _read(SCORE_DIR / "scores.csv")
    if scores is None or scores.empty:
        skipped.append("f18, f19: no scores.csv; run the baselines and score them")
    else:
        written += save_both_modes(
            lambda mode: viz.scores_figure(scores, mode), out, "f18_baseline_scores"
        )
        written += save_both_modes(
            lambda mode: viz.amplitude_slope_figure(scores, mode), out, "f19_cycle_amplitude"
        )

    coverage = _read(SCORE_DIR / "coverage.csv")
    if coverage is None or coverage.empty:
        skipped.append("f20: no coverage.csv")
    else:
        written += save_both_modes(
            lambda mode: viz.coverage_figure(coverage, mode), out, "f20_interval_coverage"
        )

    bootstrap = _read(SAMPLING_DIR / "sampling_only_bootstrap.csv")
    trend = _read(SAMPLING_DIR / "sampling_only_trend.csv")
    if bootstrap is None or trend is None:
        skipped.append("f21: no sampling-only output; run sampling_only_experiment.py")
    else:
        written += save_both_modes(
            lambda mode: viz.sampling_only_figure(bootstrap, trend, mode),
            out,
            "f21_sampling_only",
        )

    structural = _read(SCORE_DIR / "structural.csv")
    if structural is None or structural.empty or scores is None:
        skipped.append("f22: no structural.csv; run the baselines with --structural")
    else:
        written += save_both_modes(
            lambda mode: viz.structural_figure(structural, scores, mode),
            out,
            "f22_structural_ensemble",
        )

    cases = _read(SCORE_DIR / "case_study_observers.csv")
    if cases is None or cases.empty:
        skipped.append("f23, f25, f26: no case_study_observers.csv")
    else:
        written += save_both_modes(
            lambda mode: viz.acuity_recovery_figure(cases, mode), out, "f23_acuity_recovery"
        )
        # f25/f26 and the committed table come from the same summary, so a
        # number in a caption cannot drift from the file behind it.
        summary = summarise_case_studies(cases)
        DATA_OUT.mkdir(parents=True, exist_ok=True)
        table = DATA_OUT / "step2_case_study_observers.csv"
        summary.to_csv(table, index=False, lineterminator="\n", float_format="%.4f")
        print(f"wrote {table} ({len(summary)} rows)")
        written += save_both_modes(
            lambda mode: viz.case_study_figure(
                summary, list(CASE_STUDY_KINDS[:5]), mode, title="Case studies 1-5"
            ),
            out,
            "f25_case_studies_easy",
        )
        written += save_both_modes(
            lambda mode: viz.case_study_figure(
                summary, list(CASE_STUDY_KINDS[5:]), mode, title="Case studies 6-10"
            ),
            out,
            "f26_case_studies_hard",
        )

    a157 = _read(A157_DIR / "a157_validation.csv")
    if a157 is None or a157.empty:
        skipped.append("f24: no a157_validation.csv; run validate_against_a157.py")
    else:
        written += save_both_modes(
            lambda mode: viz.a157_figure(a157, mode), out, "f24_a157_validation"
        )

    # f28 draws the graph over the whole record, so it uses the full-network
    # scenario whatever --scenario the design figures were built from.
    network_seed = _first_realization(args.data_root, args.network_scenario)
    if network_seed is None:
        skipped.append(f"f28: no {args.network_scenario} realization generated")
    else:
        from gn_next_gen.benchmark.scenarios import REFERENCE_STATION

        network_obs = read_observations(args.network_scenario, network_seed, root=args.data_root)
        network = build_overlap_graph(network_obs)
        nodes = resistance_to_set(network, [REFERENCE_STATION]).assign(
            mid_year=decimal_year((network.first_day + network.last_day) / 2.0)
        )
        cut = temporal_cut_profile(network)
        DATA_OUT.mkdir(parents=True, exist_ok=True)
        cut_table = DATA_OUT / "step2_overlap_graph_cut.csv"
        cut.to_csv(cut_table, index=False, lineterminator="\n")
        print(f"wrote {cut_table} ({len(cut)} rows)")
        written += save_both_modes(
            lambda mode: viz.network_figure(
                nodes, network.edge_frame(), cut, bridges(network), mode
            ),
            out,
            "f28_overlap_network",
        )

    # f27 needs no grid output: it reads the frozen comparison table directly.
    published = _read(DATA_RAW / "gn_published_series_2026-11" / "published_gn_series.csv")
    if published is None:
        skipped.append("f27: no published_gn_series.csv; run fetch_published_gn_series.py")
    else:
        written += save_both_modes(
            lambda mode: viz.published_series_figure(published, mode), out, "f27_published_series"
        )

    # f41, f42: the error resolved in time.  These need the per-year table, which
    # only the plain (non-structural) scoring pass writes.
    pooled = _read(SCORE_DIR / "yearly_pooled.csv")
    era_table = _read(SCORE_DIR / "era_scores.csv")
    if pooled is None or pooled.empty or era_table is None or era_table.empty:
        skipped.append("f41, f42: no yearly_pooled.csv / era_scores.csv; re-run score_benchmark.py")
    else:
        written += save_both_modes(
            lambda mode: viz.error_through_time_figure(pooled, mode),
            out,
            "f41_error_through_time",
        )
        written += save_both_modes(
            lambda mode: viz.bias_and_era_coverage_figure(pooled, era_table, mode),
            out,
            "f42_bias_and_era_coverage",
        )
        # The committed table behind every number in the new section: the pooled
        # rows only, which is 11 methods x the eras they reach.
        DATA_OUT.mkdir(parents=True, exist_ok=True)
        era_out = DATA_OUT / "step2_era_errors.csv"
        columns = [
            "method",
            "era",
            "rmse",
            "bias",
            "truth",
            "reached",
            "coverage_50",
            "coverage_90",
            "n_years",
            "n_worlds",
        ]
        (
            era_table[era_table["scenario"] == "pooled"][columns]
            .round(4)
            .to_csv(era_out, index=False, lineterminator="\n")
        )
        print(f"wrote {era_out}")

    oversized = [p for p in written if p.stat().st_size > 400_000]
    print(f"wrote {len(written)} files to {out}")
    for note in skipped:
        print(f"  skipped {note}")
    if oversized:
        print("\nOVER 400 kB (the committed-figure limit):")
        for path in oversized:
            print(f"  {path.name}: {path.stat().st_size / 1000:.0f} kB")
        raise SystemExit(1)
    print(f"largest figure {max((p.stat().st_size for p in written), default=0) / 1000:.0f} kB")


if __name__ == "__main__":
    main()
