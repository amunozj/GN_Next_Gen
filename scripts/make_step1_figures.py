"""Regenerate the Step 1 figure set from committed inputs.

Verification in Step 1 is figure-first (PI, 2026-11-21), so the figures have to be
reproducible from the repository alone: the data figures read ``data/raw`` and the
committed per-year release diff, and the check figures read the CSVs the two check
scripts wrote.  Every figure is rendered in both light and dark.

Usage
-----
    python scripts/make_step1_figures.py [--out-dir docs/results/figures/step-1]
                                         [--check-a outputs/step1/check_a] ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from gn_next_gen.paths import DOCS, OUTPUTS  # noqa: E402
from gn_next_gen.viz import checks, data  # noqa: E402
from gn_next_gen.viz.style import save_both_modes  # noqa: E402

RELEASE_DIFF = DOCS / "results" / "data" / "gn_v13_release_diff.csv"
DEFAULT_OUT = DOCS / "results" / "figures" / "step-1"


def build_data_figures(out_dir: Path) -> list[Path]:
    written = []
    written += save_both_modes(data.coverage_figure, out_dir, "f01_coverage")
    if RELEASE_DIFF.is_file():
        diff = pd.read_csv(RELEASE_DIFF)
        written += save_both_modes(
            lambda mode: data.release_diff_figure(diff, mode), out_dir, "f02_v13_release_diff"
        )
    else:
        print(f"  skipping f02: run scripts/diff_gn_releases.py first ({RELEASE_DIFF})")
    written += save_both_modes(data.groups_tail_figure, out_dir, "f03_groups_tail_and_K")
    written += save_both_modes(data.leap_day_figure, out_dir, "f04_leapday_calendar")
    written += save_both_modes(data.day_status_figure, out_dir, "f05_reference_daystatus")
    written += save_both_modes(data.lifetime_figure, out_dir, "f06_lifetimes")
    return written


def build_check_figures(out_dir: Path, check_a: Path, check_b: Path, benchmark: Path) -> list[Path]:
    written = []
    if (check_a / "acf.csv").is_file():
        acf = pd.read_csv(check_a / "acf.csv")
        written += save_both_modes(
            lambda mode: checks.acf_figure(acf, mode), out_dir, "f07_checkA_acf"
        )
        bins = pd.read_csv(check_a / "zero_by_bin.csv")
        pmf = pd.read_csv(check_a / "count_pmf.csv")
        runs = pd.read_csv(check_a / "run_lengths.csv")
        written += save_both_modes(
            lambda mode: checks.zero_and_counts_figure(bins, pmf, runs, mode),
            out_dir,
            "f08_checkA_zero_counts",
        )
        scores = pd.read_csv(check_a / "scores.csv")
        written += save_both_modes(
            lambda mode: checks.scores_figure(scores, mode), out_dir, "f09_checkA_scores"
        )
    else:
        print(f"  skipping f07-f09: no Check A output in {check_a}")

    if (check_b / "stage2_fixed_weibull.csv").is_file():
        from gn_next_gen.io import load_group_single_entry
        from gn_next_gen.reference.area_law import load_upstream_parameters

        stage2 = pd.read_csv(check_b / "stage2_fixed_weibull.csv")
        power = pd.read_csv(check_b / "power_laws.csv")
        gate = pd.read_csv(check_b / "weibull_gate.csv").to_dict("records")
        surveys = list(dict.fromkeys(stage2["survey"]))

        prepared = {}
        for survey in surveys:
            base = load_upstream_parameters(survey)
            groups = load_group_single_entry([survey])
            prepared[survey] = (groups[groups["area_corr"] > base.threshold_msh], base)

        written += save_both_modes(
            lambda mode: checks.area_law_figure(prepared, None, mode),
            out_dir,
            "f10_checkB_area_law",
        )
        written += save_both_modes(
            lambda mode: checks.activity_dependence_figure(stage2, power, gate, mode),
            out_dir,
            "f11_checkB_activity",
        )
    else:
        print(f"  skipping f10-f11: no Check B output in {check_b}")

    if (benchmark / "filter_benchmark.csv").is_file():
        timings = pd.read_csv(benchmark / "filter_benchmark.csv")
        written += save_both_modes(
            lambda mode: checks.filter_benchmark_figure(timings, mode),
            out_dir,
            "f12_filter_benchmark",
        )
    else:
        print(f"  skipping f12: no benchmark output in {benchmark}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check-a", type=Path, default=OUTPUTS / "step1" / "check_a")
    ap.add_argument("--check-b", type=Path, default=OUTPUTS / "step1" / "check_b")
    ap.add_argument("--benchmark", type=Path, default=OUTPUTS / "step1" / "benchmark")
    ap.add_argument("--data-only", action="store_true")
    args = ap.parse_args()

    print(f"writing figures to {args.out_dir}")
    written = build_data_figures(args.out_dir)
    if not args.data_only:
        written += build_check_figures(args.out_dir, args.check_a, args.check_b, args.benchmark)

    total = sum(p.stat().st_size for p in written)
    print(f"\n{len(written)} files, {total / 1e6:.1f} MB total")
    for path in sorted(written):
        print(f"  {path.name}  {path.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
