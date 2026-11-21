"""Check A -- do immigration-death dynamics plus rotation reproduce the reference?

The viability question Step 4 rests on (assessment section 4; research report's
closing "Two empirical checks underpin the whole engine, and neither has been run").
For each survey and acuity threshold this fits the chain by exact-filter maximum
penalized likelihood, simulates from the fit on the **real observing calendar**, and
scores five pre-registered discrepancy measures against advisory targets.

Output is figure-first: a ``scores.csv`` that every number in the results doc traces
to, plus the figures f07-f09.  Nothing is printed but progress.

Usage
-----
    python scripts/check_a_dynamics.py --config configs/step1_check_a.yaml [--quick]
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gn_next_gen.config import load_run_config, write_lineage  # noqa: E402
from gn_next_gen.inference.forward_filter import fit_immigration_death  # noqa: E402
from gn_next_gen.io import daily_counts, load_day_status, load_group_single_entry  # noqa: E402
from gn_next_gen.models.dynamics import simulate_counts  # noqa: E402
from gn_next_gen.models.visibility import simulate_marked_process  # noqa: E402
from gn_next_gen.paths import OUTPUTS  # noqa: E402
from gn_next_gen.stats.discrepancy import (  # noqa: E402
    TARGETS,
    activity_bins,
    autocorrelation,
    count_pmf,
    run_lengths,
    score_check_a,
    smooth_centered,
)
from gn_next_gen.viz.checks import acf_figure, scores_figure, zero_and_counts_figure  # noqa: E402
from gn_next_gen.viz.style import save_both_modes  # noqa: E402


def daily_series(survey: str, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The survey's counts on a contiguous daily grid, with NaN where unobserved.

    A contiguous grid is essential: the gaps carry information, and compacting the
    series would make two days a month apart look adjacent.
    """
    counts = daily_counts(threshold, [survey])
    status = load_day_status([survey])
    first, last = int(status["date_ord"].min()), int(status["date_ord"].max())
    grid = np.arange(first, last + 1)
    series = np.full(grid.shape, np.nan)
    index = counts["date_ord"].to_numpy() - first
    series[index] = counts["n_groups"].to_numpy()
    observed = np.isfinite(series)
    return grid, series, observed


def observed_spans(survey: str) -> np.ndarray:
    """Visible spans for the survey, D22-clean, as the snapshot measures them."""
    single = load_group_single_entry([survey])
    return single["obs_duration"].dropna().to_numpy()


def simulate_like_observed(
    fit, series: np.ndarray, observed: np.ndarray, k: int, seed: int
) -> np.ndarray:
    """One simulated series with the observing calendar transplanted onto it.

    The transplant is the legacy missing-day imprinting idea
    (``legacy/ADFSSN/scripts/SSN_ADF.py`` l. 867-871): a simulated series compared
    against an observed one must be missing the same days, or every gap-sensitive
    statistic is measured on different supports.
    """
    import jax

    path = np.asarray(
        simulate_counts(jax.random.key(seed), float(fit.phi[0]), fit.lam_t, k)
    ).astype(float)
    path[~observed] = np.nan
    del series
    return path


def run(config_path: Path, *, quick: bool, out_dir: Path) -> pd.DataFrame:
    config = load_run_config(config_path)
    params = config.params
    k = int(params["truncation_k"])
    thresholds = [30.0] if quick else list(config.thresholds)
    surveys = ["rgo"] if quick else list(config.surveys)
    class_counts = [1] if quick else list(params.get("area_classes", [1]))
    smoothing = int(params.get("smoothing_days", 91))
    n_bins = int(params.get("activity_bins", 5))
    max_lag = int(params.get("acf_max_lag", 30))
    n_sims = 3 if quick else int(params.get("n_simulations", 200))
    # A two-class fit squares the state space, so it is run where the decision rule
    # reads it off -- the primary threshold, for the two anchor surveys.
    two_class_thresholds = set(params.get("two_class_thresholds", [30.0]))
    two_class_surveys = set(params.get("two_class_surveys", ["rgo", "debrecen"]))
    maxiter = 25 if quick else 300

    out_dir.mkdir(parents=True, exist_ok=True)
    write_lineage(config, out_dir, extra={"script": "check_a_dynamics.py", "quick": quick})

    rows, acf_rows, bin_rows, pmf_rows, run_rows = [], [], [], [], []
    rng = np.random.default_rng(config.seed)

    for survey in surveys:
        spans = observed_spans(survey)
        for threshold in thresholds:
            _, series, observed = daily_series(survey, threshold)
            if quick:
                series, observed = series[:4000], observed[:4000]
            activity = smooth_centered(series, smoothing)
            counts_for_fit = np.nan_to_num(series, nan=0.0).astype(int)

            holdout = np.zeros(len(series), dtype=bool)
            holdout[rng.random(len(series)) < float(params.get("holdout_fraction", 0.1))] = True

            two_class_here = threshold in two_class_thresholds and survey in two_class_surveys
            for n_classes in class_counts:
                if n_classes == 2 and not two_class_here:
                    continue  # the 1-vs-2 decision is made where the go rule looks
                print(f"  {survey} T={threshold:g} classes={n_classes}: fitting", flush=True)
                fit = fit_immigration_death(
                    counts_for_fit,
                    observed,
                    k=k,
                    n_classes=n_classes,
                    knot_days=int(params.get("lambda_knot_days", 27)),
                    penalty=float(
                        params.get("lambda_penalties", [10.0])[
                            min(1, len(params.get("lambda_penalties", [10.0])) - 1)
                        ]
                    ),
                    holdout_mask=holdout,
                    maxiter=maxiter,
                )

                scored = []
                sim_keep = None
                for index in range(n_sims):
                    simulated = simulate_like_observed(
                        fit, series, observed, k, config.seed + index
                    )
                    sim_activity = smooth_centered(simulated, smoothing)
                    marked = simulate_marked_process(
                        fit.lam_t,
                        float(1.0 / (1.0 - fit.phi[0])),
                        seed=config.seed + index,
                    )
                    scored.append(
                        score_check_a(
                            series,
                            simulated,
                            activity,
                            sim_activity,
                            spans,
                            marked.first_apparition_spans,
                            k=k,
                            max_lag=max_lag,
                            n_bins=n_bins,
                            detrend_days=smoothing,
                        ).as_dict()
                        | {
                            "aggregate": score_check_a(
                                series,
                                simulated,
                                activity,
                                sim_activity,
                                spans,
                                marked.first_apparition_spans,
                                k=k,
                                max_lag=max_lag,
                                n_bins=n_bins,
                                detrend_days=smoothing,
                            ).aggregate
                        }
                    )
                    if index == 0:
                        sim_keep = (simulated, sim_activity, marked)

                mean_scores = pd.DataFrame(scored).mean().to_dict()
                rows.append(
                    {
                        "survey": survey,
                        "threshold_msh": threshold,
                        "n_classes": n_classes,
                        "phi": float(fit.phi[0]),
                        "mean_visible_days": float(fit.mean_visible_days[0]),
                        "class_weight": fit.class_weight,
                        "loglik": fit.loglik,
                        "holdout_loglik": fit.holdout_loglik,
                        "converged": fit.success,
                        "n_observed_days": int(observed.sum()),
                        **mean_scores,
                    }
                )

                if n_classes == class_counts[0] and threshold == 30.0:
                    simulated, sim_activity, marked = sim_keep
                    for detrended in (False, True):
                        obs_acf = autocorrelation(series, max_lag, smoothing if detrended else None)
                        sim_acf = autocorrelation(
                            simulated, max_lag, smoothing if detrended else None
                        )
                        for lag in range(max_lag):
                            acf_rows.append(
                                {
                                    "survey": survey,
                                    "detrended": detrended,
                                    "lag": lag + 1,
                                    "observed": obs_acf[lag],
                                    "simulated": sim_acf[lag],
                                }
                            )
                    if survey == surveys[0]:
                        obs_bins = activity_bins(activity, n_bins)
                        sim_bins = activity_bins(sim_activity, n_bins)
                        for b in range(n_bins):
                            obs_sel = (obs_bins == b) & np.isfinite(series)
                            sim_sel = (sim_bins == b) & np.isfinite(simulated)
                            bin_rows.append(
                                {
                                    "bin": b,
                                    "observed": float((series[obs_sel] == 0).mean())
                                    if obs_sel.sum()
                                    else np.nan,
                                    "simulated": float((simulated[sim_sel] == 0).mean())
                                    if sim_sel.sum()
                                    else np.nan,
                                }
                            )
                        obs_pmf, sim_pmf = count_pmf(series, k), count_pmf(simulated, k)
                        for value in range(min(k, 32)):
                            pmf_rows.append(
                                {
                                    "count": value,
                                    "observed": obs_pmf[value],
                                    "simulated": sim_pmf[value],
                                }
                            )
                        obs_runs = run_lengths(series)
                        sim_runs = run_lengths(simulated)
                        longest = int(
                            max(
                                obs_runs.max() if len(obs_runs) else 1,
                                sim_runs.max() if len(sim_runs) else 1,
                            )
                        )
                        for length in range(1, longest + 1):
                            run_rows.append(
                                {
                                    "length": length,
                                    "observed": int((obs_runs == length).sum()),
                                    "simulated": int((sim_runs == length).sum()),
                                }
                            )

    scores = pd.DataFrame(rows)
    scores.to_csv(out_dir / "scores.csv", index=False, lineterminator="\n")
    acf_table = pd.DataFrame(acf_rows)
    acf_table.to_csv(out_dir / "acf.csv", index=False, lineterminator="\n")
    bins_table = pd.DataFrame(bin_rows)
    pmf_table = pd.DataFrame(pmf_rows)
    runs_table = pd.DataFrame(run_rows)
    for name, table in (
        ("zero_by_bin", bins_table),
        ("count_pmf", pmf_table),
        ("run_lengths", runs_table),
    ):
        table.to_csv(out_dir / f"{name}.csv", index=False, lineterminator="\n")

    figures = out_dir / "figures"
    if not acf_table.empty:
        save_both_modes(lambda mode: acf_figure(acf_table, mode), figures, "f07_checkA_acf")
    if not bins_table.empty:
        save_both_modes(
            lambda mode: zero_and_counts_figure(bins_table, pmf_table, runs_table, mode),
            figures,
            "f08_checkA_zero_counts",
        )
    if not scores.empty:
        save_both_modes(lambda mode: scores_figure(scores, mode), figures, "f09_checkA_scores")

    summary = {
        "run": config.name,
        "written_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "targets": TARGETS,
    }
    pd.Series(summary).to_json(out_dir / "summary.json", indent=2)
    print(f"\nwrote {out_dir}/scores.csv and {figures}")
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/step1_check_a.yaml"))
    ap.add_argument("--quick", action="store_true", help="one survey, one threshold, few sims")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out = args.out_dir or (OUTPUTS / "step1" / ("check_a_quick" if args.quick else "check_a"))
    run(args.config, quick=args.quick, out_dir=out)


if __name__ == "__main__":
    main()
