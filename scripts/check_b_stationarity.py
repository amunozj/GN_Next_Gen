"""Check B -- is the area distribution, conditional on activity, stationary?

The second assumption Step 4 rests on.  The detection layer's effective detection
fraction is ``q_o = E_F[pi_o(a)]``, an expectation of the acuity curve over the area
law F; if F moves with epoch, an observer's count ratio is not a fixed number and
the historical extrapolation changes meaning (research report, "Mark-dependent
thinning explains why k-factors depend on activity").

Two design choices follow the PI's decisions of 2026-11-21:

* **condition on the smoothed group count**, not on smoothed area.  Conditioning the
  area distribution on an area-derived activity index would condition on a function
  of the quantity under test; ``amp_whole_sm91`` is reported as a sensitivity;
* the primary test restricts to ``area >= 30 MSH`` (upstream's recommended floor,
  issue D14), so per-survey completeness floors are not read as solar change.

Significance uses a **27-day block bootstrap**: groups within a day, and days within
a rotation, are dependent, and an i.i.d. test would reject everywhere.

Usage
-----
    python scripts/check_b_stationarity.py --config configs/step1_check_b.yaml [--quick]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gn_next_gen.config import load_run_config, write_lineage  # noqa: E402
from gn_next_gen.io import daily_counts, load_group_days  # noqa: E402
from gn_next_gen.paths import OUTPUTS  # noqa: E402
from gn_next_gen.stats.discrepancy import ks_distance, smooth_centered  # noqa: E402
from gn_next_gen.viz.checks import area_distribution_figure, mixture_figure  # noqa: E402
from gn_next_gen.viz.style import save_both_modes  # noqa: E402


def smoothed_activity(survey: str, smoothing: int) -> pd.Series:
    """91-day smoothed unthresholded group count per day, indexed by ``date_ord``."""
    counts = daily_counts(0.0, [survey]).sort_values("date_ord")
    grid = np.arange(int(counts["date_ord"].min()), int(counts["date_ord"].max()) + 1)
    series = np.full(grid.shape, np.nan)
    series[counts["date_ord"].to_numpy() - grid[0]] = counts["n_groups"].to_numpy()
    return pd.Series(smooth_centered(series, smoothing), index=grid)


def lognormal_mixture(
    values: np.ndarray, n_components: int, seed: int, iterations: int = 200
) -> dict:
    """Fit a 1- or 2-component lognormal mixture by EM, and report its BIC."""
    x = np.log10(values[values > 0])
    n = len(x)
    if n < 50:
        return {"bic": np.nan, "mu": [np.nan] * n_components, "weight": [np.nan] * n_components}
    if n_components == 1:
        mu, sigma = x.mean(), x.std()
        loglik = float(np.sum(-0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))))
        return {"bic": 2 * np.log(n) - 2 * loglik, "mu": [mu], "weight": [1.0], "sigma": [sigma]}

    rng = np.random.default_rng(seed)
    mu = np.quantile(x, [0.25, 0.75]) + rng.normal(0, 0.01, 2)
    sigma = np.full(2, x.std() / 2)
    weight = np.full(2, 0.5)
    for _ in range(iterations):
        dens = np.stack(
            [
                weight[c]
                * np.exp(-0.5 * ((x - mu[c]) / sigma[c]) ** 2)
                / (sigma[c] * np.sqrt(2 * np.pi))
                for c in range(2)
            ]
        )
        total = dens.sum(axis=0)
        total[total <= 0] = 1e-300
        resp = dens / total
        counts = resp.sum(axis=1)
        weight = counts / n
        mu = (resp * x).sum(axis=1) / counts
        sigma = np.sqrt((resp * (x - mu[:, None]) ** 2).sum(axis=1) / counts)
        sigma = np.maximum(sigma, 1e-3)
    loglik = float(np.sum(np.log(total)))
    order = np.argsort(mu)
    return {
        "bic": 5 * np.log(n) - 2 * loglik,
        "mu": mu[order].tolist(),
        "weight": weight[order].tolist(),
        "sigma": sigma[order].tolist(),
    }


def block_bootstrap_ks(
    a: np.ndarray, b: np.ndarray, blocks_a: np.ndarray, blocks_b: np.ndarray, n_boot: int, rng
) -> tuple[float, float]:
    """KS distance between two epochs, and its block-bootstrap p-value.

    Blocks are 27-day rotation blocks; resampling whole blocks keeps the
    within-rotation dependence that makes an i.i.d. p-value meaningless here.
    """
    observed = ks_distance(a, b)
    pooled = np.concatenate([a, b])
    pooled_blocks = np.concatenate([blocks_a, blocks_b])
    unique_blocks = np.unique(pooled_blocks)
    n_a = len(np.unique(blocks_a))
    exceed = 0
    for _ in range(n_boot):
        shuffled = rng.permutation(unique_blocks)
        left = np.isin(pooled_blocks, shuffled[:n_a])
        null = ks_distance(pooled[left], pooled[~left])
        exceed += null >= observed
    return float(observed), float((exceed + 1) / (n_boot + 1))


def run(config_path: Path, *, quick: bool, out_dir: Path) -> pd.DataFrame:
    config = load_run_config(config_path)
    params = config.params
    smoothing = int(params.get("smoothing_days", 91))
    n_bins = int(params.get("activity_bins", 5))
    floor = float(params.get("area_floor_msh", 30.0))
    block_days = int(params.get("block_bootstrap_days", 27))
    n_boot = 50 if quick else int(params.get("n_bootstrap", 1000))
    rng = np.random.default_rng(config.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_lineage(config, out_dir, extra={"script": "check_b_stationarity.py", "quick": quick})

    groups = load_group_days(
        config.surveys, columns=["survey", "group_id", "date_ord", "area_corr", "cycle"]
    )
    activity = {survey: smoothed_activity(survey, smoothing) for survey in config.surveys}

    epoch_rows, epoch_areas = [], {}
    for name, spec in params["epochs"].items():
        survey, cycles = spec["survey"], set(spec["cycles"])
        sub = groups[(groups["survey"] == survey) & (groups["cycle"].isin(cycles))].copy()
        if sub.empty:
            continue
        sub["activity"] = sub["date_ord"].map(activity[survey])
        sub["block"] = sub["date_ord"] // block_days
        sub["epoch"] = name
        epoch_rows.append(sub)
        epoch_areas[name] = sub["area_corr"].to_numpy()
    table = pd.concat(epoch_rows, ignore_index=True)

    # Activity quintiles are defined once, over all epochs, so a bin means the same
    # thing in each of them.
    finite = table["activity"].to_numpy()
    edges = np.nanquantile(finite, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    table["activity_bin"] = np.digitize(table["activity"], edges[1:-1])

    primary = table[table["area_corr"] >= floor]

    summary = []
    for epoch, sub in primary.groupby("epoch", sort=False):
        logs = np.log10(sub["area_corr"].to_numpy())
        summary.append(
            {
                "epoch": epoch,
                "n_groups": len(sub),
                "p10": float(np.quantile(logs, 0.10)),
                "p50": float(np.quantile(logs, 0.50)),
                "p90": float(np.quantile(logs, 0.90)),
                "above_30_fraction": float(
                    (table.loc[table["epoch"] == epoch, "area_corr"] >= floor).mean()
                ),
                "above_100_fraction": float(
                    (table.loc[table["epoch"] == epoch, "area_corr"] >= 100.0).mean()
                ),
            }
        )
    percentiles = pd.DataFrame(summary)

    pairs = []
    epochs = list(percentiles["epoch"])
    for i, left in enumerate(epochs):
        for right in epochs[i + 1 :]:
            for b in range(n_bins):
                a_sel = primary[(primary["epoch"] == left) & (primary["activity_bin"] == b)]
                b_sel = primary[(primary["epoch"] == right) & (primary["activity_bin"] == b)]
                if len(a_sel) < 100 or len(b_sel) < 100:
                    continue
                ks, pvalue = block_bootstrap_ks(
                    np.log10(a_sel["area_corr"].to_numpy()),
                    np.log10(b_sel["area_corr"].to_numpy()),
                    a_sel["block"].to_numpy(),
                    b_sel["block"].to_numpy(),
                    n_boot,
                    rng,
                )
                left_logs = np.log10(a_sel["area_corr"].to_numpy())
                right_logs = np.log10(b_sel["area_corr"].to_numpy())
                pairs.append(
                    {
                        "epoch_a": left,
                        "epoch_b": right,
                        "activity_bin": b,
                        "n_a": len(a_sel),
                        "n_b": len(b_sel),
                        "ks": ks,
                        "p_block": pvalue,
                        "d_p10": float(np.quantile(right_logs, 0.1) - np.quantile(left_logs, 0.1)),
                        "d_p50": float(np.quantile(right_logs, 0.5) - np.quantile(left_logs, 0.5)),
                        "d_p90": float(np.quantile(right_logs, 0.9) - np.quantile(left_logs, 0.9)),
                    }
                )
    pair_table = pd.DataFrame(pairs)

    mixtures = []
    for epoch in epochs:
        values = primary.loc[primary["epoch"] == epoch, "area_corr"].to_numpy()
        one = lognormal_mixture(values, 1, config.seed)
        two = lognormal_mixture(values, 2, config.seed)
        mixtures.append(
            {
                "epoch": epoch,
                "bic1": one["bic"],
                "bic2": two["bic"],
                "delta_bic": one["bic"] - two["bic"],
                "mu1": two["mu"][0],
                "mu2": two["mu"][1],
                "weight1": two["weight"][0],
            }
        )
    mixture_table = pd.DataFrame(mixtures)

    percentiles.to_csv(out_dir / "percentiles.csv", index=False, lineterminator="\n")
    pair_table.to_csv(out_dir / "epoch_pairs.csv", index=False, lineterminator="\n")
    mixture_table.to_csv(out_dir / "mixtures.csv", index=False, lineterminator="\n")

    figures = out_dir / "figures"
    save_both_modes(
        lambda mode: area_distribution_figure(epoch_areas, percentiles, mode),
        figures,
        "f10_checkB_areas",
    )
    save_both_modes(
        lambda mode: mixture_figure(mixture_table, mode), figures, "f11_checkB_mixtures"
    )

    spread = percentiles[["p10", "p50", "p90"]].max() - percentiles[["p10", "p50", "p90"]].min()
    print(f"\nwrote {out_dir}")
    print(
        "  percentile spread across epochs (dex):",
        {k: round(float(v), 3) for k, v in spread.items()},
    )
    print(f"  epoch pairs tested: {len(pair_table)}")
    return percentiles


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/step1_check_b.yaml"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out = args.out_dir or (OUTPUTS / "step1" / ("check_b_quick" if args.quick else "check_b"))
    run(args.config, quick=args.quick, out_dir=out)


if __name__ == "__main__":
    main()
