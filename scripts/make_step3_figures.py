"""Rebuild Step 3's figures, f29-f40 and f43-f47, in both light and dark.

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

from gn_next_gen.benchmark.observers import CASE_STUDY_KINDS  # noqa: E402
from gn_next_gen.benchmark.scoring import summarise_case_studies  # noqa: E402
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

    # --- f31-f34, f38: the comparison with Step 2's eleven -------------------
    #
    # f31, f32 and f33 reuse Step 2's own builders: they are method-generic and
    # NetCal is registered in METHOD_ORDER, so they draw it with no new plotting
    # code.  They are written under Step 3's codes into figures/step-3/, so Step
    # 2's committed figures are never touched.
    from gn_next_gen.viz import benchmark as bviz

    if not (SCORES / "scores.csv").is_file():
        _skip("f31-f34", "no outputs/step2/scores/scores.csv -- run score_benchmark.py")
    else:
        scores = pd.read_csv(SCORES / "scores.csv")
        step3 = scores[scores["method"].isin(["NetCal", "NetCal-M"])]
        if step3.empty:
            _skip(
                "f31-f34",
                "the grid has no NetCal cells yet (run_baselines.py --methods NetCal NetCal-M)",
            )
        else:
            print(
                f"  scoring {scores['method'].nunique()} methods ({len(step3)} NetCal cells)",
                flush=True,
            )
            save_both_modes(
                lambda mode: bviz.scores_figure(scores, mode), out, "f32_baseline_scores"
            )
            made.append("f32")

            era_path = SCORES / "era_scores.csv"
            era = pd.read_csv(era_path) if era_path.is_file() else None
            save_both_modes(
                lambda mode: viz.reach_and_maunder_figure(scores, era, mode),
                out,
                "f34_reach_and_maunder",
            )
            made.append("f34")

            coverage_path = SCORES / "coverage.csv"
            if coverage_path.is_file():
                coverage = pd.read_csv(coverage_path)
                save_both_modes(
                    lambda mode: bviz.coverage_figure(coverage, mode),
                    out,
                    "f33_interval_coverage",
                )
                made.append("f33")
            else:
                _skip("f33", "no coverage.csv")

            cases_path = SCORES / "case_study_observers.csv"
            if cases_path.is_file():
                cases = pd.read_csv(cases_path)
                save_both_modes(
                    lambda mode: bviz.acuity_recovery_figure(cases, mode),
                    out,
                    "f31_acuity_recovery",
                )
                made.append("f31")
            else:
                _skip("f31", "no case_study_observers.csv")

            # --- f43-f47: the rest of Step 2's method comparison, redrawn ----
            #
            # Step 2's report compares the eleven on six method-generic figures.
            # f32/f33/f31 above already redraw three of them with NetCal in.
            # These are the other three (plus the two case-study panels), built
            # from the same Step 2 builders so the only difference from the
            # committed Step 2 version is that NetCal and NetCal-M are present.
            save_both_modes(
                lambda mode: bviz.amplitude_slope_figure(scores, mode),
                out,
                "f43_cycle_amplitude",
            )
            made.append("f43")

            pooled_path = SCORES / "yearly_pooled.csv"
            if pooled_path.is_file() and era is not None and not era.empty:
                pooled = pd.read_csv(pooled_path)
                save_both_modes(
                    lambda mode: bviz.error_through_time_figure(pooled, mode),
                    out,
                    "f44_error_through_time",
                )
                made.append("f44")
                save_both_modes(
                    lambda mode: bviz.bias_and_era_coverage_figure(pooled, era, mode),
                    out,
                    "f45_bias_and_era_coverage",
                )
                made.append("f45")
            else:
                _skip("f44/f45", "no yearly_pooled.csv or era_scores.csv")

            if cases_path.is_file():
                summary = summarise_case_studies(pd.read_csv(cases_path))
                save_both_modes(
                    lambda mode: bviz.case_study_figure(
                        summary, list(CASE_STUDY_KINDS[:5]), mode, title="Case studies 1-5"
                    ),
                    out,
                    "f46_case_studies_easy",
                )
                made.append("f46")
                save_both_modes(
                    lambda mode: bviz.case_study_figure(
                        summary, list(CASE_STUDY_KINDS[5:]), mode, title="Case studies 6-10"
                    ),
                    out,
                    "f47_case_studies_hard",
                )
                made.append("f47")
            else:
                _skip("f46/f47", "no case_study_observers.csv")

            # Step 2's f22 has no Step 3 twin on purpose: the structural
            # ensemble re-runs a method over its own free graph choices, and
            # that arm was never run for NetCal.  Say so rather than skip it
            # silently -- it is an open PI decision, not a missing input.
            structural_path = SCORES / "structural.csv"
            if structural_path.is_file():
                structural = pd.read_csv(structural_path)
                have = structural["method"].isin(["NetCal", "NetCal-M"]).sum()
                if not have:
                    _skip(
                        "f22 twin",
                        "structural.csv has no NetCal rows; the structural arm was "
                        "not run for the new methods (open PI decision)",
                    )

    # f38: the two kernels, on a world that has both arms.  Worlds whose tables
    # carry `first_year` are preferred, because that is what the per-era panel is
    # drawn against and a cell surviving from an older run may not have it -- one
    # such cell does survive the first grid's failure.  A world without it still
    # works; the panel says so rather than drawing nothing.
    pair = None
    fallback = None
    baselines = OUTPUTS / "step2" / "baselines"
    for directory in sorted(baselines.glob("*/seed-*")):
        left_path = directory / "NetCal_observers.parquet"
        right_path = directory / "NetCal-M_observers.parquet"
        if not (left_path.is_file() and right_path.is_file()):
            continue
        left = pd.read_parquet(left_path)
        right = pd.read_parquet(right_path)
        candidate = (left, right, directory.name)
        if "first_year" in left.columns and "first_year" in right.columns:
            pair = candidate
            break
        fallback = fallback or candidate
    pair = pair or fallback
    if pair is not None and "first_year" not in pair[0].columns:
        print("  f38: no world carries first_year; the per-era panel will say so", flush=True)
    if pair is not None:
        left, right, label = pair
        print(f"  f38 from {label}", flush=True)
        save_both_modes(
            lambda mode: viz.kernel_disagreement_figure(left, right, mode),
            out,
            "f38_kernel_disagreement",
        )
        made.append("f38")
    else:
        _skip("f38", "no world has both NetCal and NetCal-M observer tables yet")

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
