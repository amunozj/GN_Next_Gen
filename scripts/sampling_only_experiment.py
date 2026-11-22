"""The Hay diagnostic: what the observing calendar alone does to the answer.

Hay et al. (2015) reconstructed global mean sea level and found that binning the
real, gappy tide-gauge network reproduced the earlier, higher trend **from the
sampling pattern alone** -- 1.6 mm/yr against 1.2 -- and that bootstrapping over
25 to 600 gauges levelled off about 0.4 mm/yr above the probabilistic estimate.
"Combined spatial sparsity and missing data generate an upward bias."

The GN analogue is scenario **S5**: perfect observers, with no threshold, no
drift, no noise and no missing-not-at-random omission, placed on the real
historical calendars.  Every observer reports the estimand exactly on the days
it looked, so **any** error in a naive estimate is the calendar's doing and
nothing else.  That isolates sampling from calibration, which no amount of
comparing calibration methods can do.

Two things are measured:

* the **induced trend bias** of a naive estimator -- the mean of reported counts,
  aggregated to years -- against the truth over all days, which is the
  reconstruction every non-probabilistic analysis implicitly performs;
* how that bias behaves as the network thins, bootstrapping over subsets of
  observers, which is Hay's levelling-off curve.

Usage
-----
    python scripts/sampling_only_experiment.py [--scenario S5] [--draws 200]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gn_next_gen.benchmark.store import (  # noqa: E402
    BENCHMARK_ROOT,
    read_observations,
    read_sealed_truth,
)
from gn_next_gen.paths import OUTPUTS  # noqa: E402

#: Network sizes to bootstrap over, echoing Hay's 25-600 gauges.
SUBSET_SIZES = (5, 10, 25, 50, 100, 200, 400)

UNIX_EPOCH_ORDINAL = 719_163


def _years(date_ord: np.ndarray) -> np.ndarray:
    return pd.to_datetime(np.asarray(date_ord) - UNIX_EPOCH_ORDINAL, unit="D").year.to_numpy()


def naive_yearly(observations: pd.DataFrame) -> pd.Series:
    """What a non-probabilistic analysis does: average whoever reported."""
    frame = observations.assign(year=_years(observations["date_ord"].to_numpy()))
    daily = frame.groupby(["year", "date_ord"])["groups"].mean()
    return daily.groupby(level="year").mean()


def trend_per_century(series: pd.Series, first_year: int | None = None) -> float:
    window = series if first_year is None else series[series.index >= first_year]
    if len(window) < 20:
        return float("nan")
    years = window.index.to_numpy(dtype=float)
    return float(np.polyfit(years, window.to_numpy(), 1)[0] * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="S5")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--data-root", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS / "step2" / "sampling_only")
    args = parser.parse_args()

    scenario_dir = Path(args.data_root) / args.scenario
    seeds = sorted(int(p.name.split("-")[1]) for p in scenario_dir.glob("seed-*"))
    if not seeds:
        raise SystemExit(f"no {args.scenario} realization under {args.data_root}")
    seed = args.seed if args.seed is not None else seeds[0]

    truth = read_sealed_truth(args.scenario, seed, root=args.data_root)
    observations = read_observations(args.scenario, seed, root=args.data_root)

    reference = truth.reference_counts()
    truth_yearly = (
        pd.Series(reference.to_numpy(), index=_years(reference.index.to_numpy()))
        .groupby(level=0)
        .mean()
    )
    sampled_yearly = naive_yearly(observations)

    common = truth_yearly.index.intersection(sampled_yearly.index)
    truth_yearly, sampled_yearly = truth_yearly.loc[common], sampled_yearly.loc[common]

    full = pd.DataFrame(
        [
            {
                "window": label,
                "truth_trend_per_century": trend_per_century(truth_yearly, start),
                "sampled_trend_per_century": trend_per_century(sampled_yearly, start),
            }
            for label, start in (("1610-", None), ("1700-", 1700), ("1850-", 1850))
        ]
    )
    full["induced_bias"] = full["sampled_trend_per_century"] - full["truth_trend_per_century"]

    rng = np.random.default_rng(seed)
    stations = np.array(sorted(observations["station"].unique()))
    rows = []
    for size in SUBSET_SIZES:
        if size > len(stations):
            continue
        for draw in range(args.draws):
            chosen = rng.choice(stations, size=size, replace=False)
            subset = observations[observations["station"].isin(chosen)]
            yearly = naive_yearly(subset)
            shared = truth_yearly.index.intersection(yearly.index)
            if len(shared) < 20:
                continue
            rows.append(
                {
                    "n_observers": size,
                    "draw": draw,
                    "trend_per_century": trend_per_century(yearly.loc[shared]),
                    "truth_trend_per_century": trend_per_century(truth_yearly.loc[shared]),
                    "mean_level": float(yearly.loc[shared].mean()),
                    "truth_mean_level": float(truth_yearly.loc[shared].mean()),
                }
            )
    bootstrap = pd.DataFrame(rows)
    bootstrap["trend_bias"] = bootstrap["trend_per_century"] - bootstrap["truth_trend_per_century"]
    bootstrap["level_bias"] = bootstrap["mean_level"] - bootstrap["truth_mean_level"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    full.to_csv(args.out_dir / "sampling_only_trend.csv", index=False, lineterminator="\n")
    bootstrap.to_csv(args.out_dir / "sampling_only_bootstrap.csv", index=False, lineterminator="\n")

    summary = (
        bootstrap.groupby("n_observers")[["trend_bias", "level_bias"]].agg(["mean", "std"]).round(4)
    )
    summary.to_csv(args.out_dir / "sampling_only_summary.csv", lineterminator="\n")

    print(f"scenario {args.scenario} seed {seed}, {len(stations)} observers\n")
    print("Trend, truth over all days against a naive mean of what was reported:")
    print(full.round(4).to_string(index=False))
    print("\nBias against network size (Hay's levelling-off curve):")
    print(summary.to_string())
    print(f"\nwrote {args.out_dir}")


if __name__ == "__main__":
    main()
