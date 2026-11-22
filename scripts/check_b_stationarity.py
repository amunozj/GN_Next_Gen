"""Check B -- the group-area law, its activity dependence, and whether it is stationary.

Step 4's detection layer consumes one functional of the area distribution: the
effective detection fraction ``q_o = E_F[pi_o(a)]``.  Check B therefore has to say
what ``F`` is, whether it moves with the solar cycle, and whether the same law held
in 1874 as in 2014.

The law is a **Weibull + log-normal** mixture on log-area (ADR-011), calibrated
per survey by ``sunspot-homogenization`` and **not refitted here**.  Fitting two
log-normals instead -- which an earlier version of this script did -- folds
observational effects into the small-object component.

Three stages, following Munoz-Jaramillo et al. (2015) Secs. 6-8:

1. **Free.** All five parameters per activity bin. Does the Weibull component move
   with activity?  The paper finds it does not -- the magnetic baseline -- and that
   finding is what licenses stage 2.  **If it does move, stage 2 does not run.**
2. **Fixed Weibull.** ``(k, lambda)`` held at upstream's per-survey values; only
   ``mu``, ``sigma`` and the mixing weight free, then power laws in activity level,
   compared against the published coefficients.
3. **Stationarity.** At matched activity, are the log-normal parameters the same
   across epochs?  That is the question Step 1 owes Step 4.

Then the **grain bridge**: the same fit on group-*days* rather than groups, since
the published law describes one area per group while ``q_o`` averages over what is
on the disk that day.

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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gn_next_gen.config import load_run_config, write_lineage  # noqa: E402
from gn_next_gen.io import load_day_status, load_group_days, load_group_single_entry  # noqa: E402
from gn_next_gen.paths import OUTPUTS  # noqa: E402
from gn_next_gen.reference.area_fit import (  # noqa: E402
    activity_bins,
    fit_bin,
    fit_by_activity,
    fit_power_law,
)
from gn_next_gen.reference.area_law import (  # noqa: E402
    PUBLISHED_2015,
    TruncatedLikelihood,
    activity_level,
    load_upstream_parameters,
)
from gn_next_gen.viz.checks import activity_dependence_figure, area_law_figure  # noqa: E402
from gn_next_gen.viz.style import save_both_modes  # noqa: E402

#: Confidence above which a Spearman correlation counts as detected, as the paper
#: reports it (it quotes P = 1 - p_value).
CONFIDENCE = 0.95

LOGNORMAL_PARAMS = ("mu_LN", "sigma_LN", "c_LN")
WEIBULL_PARAMS = ("k_W", "lambda_W")


def prepare(survey: str, *, weighting: str, grain: str, scope: str = "whole"):
    """Truncated groups for one survey, annotated with activity level."""
    status = load_day_status([survey])
    single = load_group_single_entry([survey])
    activity = activity_level(status, single, scope=scope, weighting=weighting)
    params = load_upstream_parameters(survey)

    frame = (
        single
        if grain == "group"
        else load_group_days(
            [survey],
            columns=["survey", "group_id", "date_ord", "area_corr", "area_corr_unc", "cycle"],
        )
    )
    kept = frame[frame["area_corr"] > params.threshold_msh].copy()
    kept["activity"] = kept["date_ord"].map(activity)
    kept = kept[np.isfinite(kept["activity"])]
    return kept, params


def correlations(fits, parameters) -> list[dict]:
    rows = []
    frame = pd.DataFrame([f.as_row() for f in fits])
    for name in parameters:
        rho, p_value = spearmanr(frame["activity"], frame[name])
        rows.append(
            {
                "parameter": name,
                "spearman_rho": float(rho),
                "confidence": float(1 - p_value),
                "detected": bool((1 - p_value) >= CONFIDENCE),
                "n_bins": len(frame),
            }
        )
    return rows


def weibull_gate(stage1: dict[str, list]) -> tuple[bool, list[dict]]:
    """Does the Weibull component move with activity?  ADR-011's licensing test."""
    rows = []
    for survey, fits in stage1.items():
        for row in correlations(fits, WEIBULL_PARAMS):
            rows.append({"survey": survey} | row)
    table = pd.DataFrame(rows)
    detected = table[table["detected"]]
    # One survey disagreeing in sign with the others is noise, not a trend; the gate
    # fails only if a Weibull parameter is detected in the same direction everywhere.
    failed = False
    for name in WEIBULL_PARAMS:
        hits = detected[detected["parameter"] == name]
        if len(hits) == len(stage1) and len({np.sign(r) for r in hits["spearman_rho"]}) == 1:
            failed = True
    return (not failed), rows


def run(config_path: Path, *, quick: bool, out_dir: Path) -> pd.DataFrame:
    config = load_run_config(config_path)
    params_cfg = config.params
    surveys = ["rgo"] if quick else list(config.surveys)
    n_bins = 5 if quick else int(params_cfg.get("activity_bins", 10))
    weighting = params_cfg.get("activity_weighting", "emergence")
    min_groups = int(params_cfg.get("min_groups_per_bin", 200))

    out_dir.mkdir(parents=True, exist_ok=True)
    write_lineage(config, out_dir, extra={"script": "check_b_stationarity.py", "quick": quick})

    prepared = {s: prepare(s, weighting=weighting, grain="group") for s in surveys}

    # --- stage 1: is the Weibull component activity-independent? -----------
    print("stage 1 -- all five parameters free per activity bin")
    stage1 = {}
    for survey, (groups, base) in prepared.items():
        stage1[survey] = fit_by_activity(
            groups, base, n_bins=n_bins, free_weibull=True, stage="free", min_groups=min_groups
        )
        print(f"  {survey}: {len(stage1[survey])} bins")
    stage1_rows = [f.as_row() for fits in stage1.values() for f in fits]
    pd.DataFrame(stage1_rows).to_csv(out_dir / "stage1_free.csv", index=False, lineterminator="\n")

    licensed, gate_rows = weibull_gate(stage1)
    pd.DataFrame(gate_rows).to_csv(out_dir / "weibull_gate.csv", index=False, lineterminator="\n")
    print("\nWeibull gate (ADR-011):")
    for row in gate_rows:
        print(
            f"  {row['survey']:9s} {row['parameter']:9s} rho={row['spearman_rho']:+.2f} "
            f"P={row['confidence']:.2f}{'  DETECTED' if row['detected'] else ''}"
        )
    if not licensed:
        print(
            "\nSTOP: the Weibull component moves with activity in every survey, in the "
            "same direction. Holding (k, lambda) fixed is not licensed, which "
            "contradicts Munoz-Jaramillo et al. (2015). Stage 2 is not run; report "
            "this finding."
        )
        raise SystemExit(2)
    print("  -> no consistent dependence; holding (k, lambda) fixed is licensed\n")

    # --- stage 2: the log-normal's activity dependence ---------------------
    print("stage 2 -- (k, lambda) fixed at upstream's values")
    stage2 = {}
    for survey, (groups, base) in prepared.items():
        stage2[survey] = fit_by_activity(
            groups,
            base,
            n_bins=n_bins,
            free_weibull=False,
            stage="fixed_weibull",
            min_groups=min_groups,
        )
        print(f"  {survey}: {len(stage2[survey])} bins")
    stage2_rows = [f.as_row() for fits in stage2.values() for f in fits]
    stage2_table = pd.DataFrame(stage2_rows)
    stage2_table.to_csv(out_dir / "stage2_fixed_weibull.csv", index=False, lineterminator="\n")

    power_rows = []
    for survey, fits in stage2.items():
        frame = pd.DataFrame([f.as_row() for f in fits])
        for name, published in (
            ("mu_LN", PUBLISHED_2015["mu_LN"]),
            ("sigma_LN", PUBLISHED_2015["sigma_LN"]),
            ("c_LN", PUBLISHED_2015["c_LN"]),
        ):
            law = fit_power_law(
                frame["activity"], frame[name], name, published=published, weights=frame["n_groups"]
            )
            power_rows.append({"survey": survey} | law.as_row())
    power_table = pd.DataFrame(power_rows)
    power_table.to_csv(out_dir / "power_laws.csv", index=False, lineterminator="\n")
    print("\npower laws (fitted vs published):")
    for row in power_rows:
        print(
            f"  {row['survey']:9s} {row['parameter']:9s} "
            f"a={row['a']:.3f} b={row['b']:+.3f}  "
            f"published a={row['published_a']:.2f} b={row['published_b']:+.3f}  "
            f"rho={row['spearman_rho']:+.2f} P={1 - row['p_value']:.2f}"
        )

    # --- stage 3: stationarity at matched activity -------------------------
    stationarity = []
    for survey, fits in stage2.items():
        frame = pd.DataFrame([f.as_row() for f in fits])
        for name in LOGNORMAL_PARAMS:
            stationarity.append(
                {
                    "survey": survey,
                    "parameter": name,
                    "min": float(frame[name].min()),
                    "max": float(frame[name].max()),
                    "spread": float(frame[name].max() - frame[name].min()),
                }
            )
    across = []
    for name in LOGNORMAL_PARAMS:
        per_survey = {s: pd.DataFrame([f.as_row() for f in fits]) for s, fits in stage2.items()}
        # Compare surveys at their closest-matching activity bins.
        reference = per_survey[surveys[0]]
        for other in surveys[1:]:
            frame = per_survey[other]
            matched = [
                float(frame.iloc[np.argmin(np.abs(frame["activity"] - a))][name]) - float(v)
                for a, v in zip(reference["activity"], reference[name], strict=True)
            ]
            across.append(
                {
                    "parameter": name,
                    "pair": f"{surveys[0]}_vs_{other}",
                    "mean_abs_difference": float(np.mean(np.abs(matched))),
                    "max_abs_difference": float(np.max(np.abs(matched))),
                }
            )
    pd.DataFrame(stationarity).to_csv(
        out_dir / "stationarity_within.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(across).to_csv(
        out_dir / "stationarity_across.csv", index=False, lineterminator="\n"
    )
    print("\nstationarity across surveys at matched activity:")
    for row in across:
        print(
            f"  {row['parameter']:9s} {row['pair']:20s} "
            f"mean |diff| {row['mean_abs_difference']:.3f}  "
            f"max {row['max_abs_difference']:.3f}"
        )

    # --- the grain bridge --------------------------------------------------
    print("\ngrain bridge -- the same fit on group-days")
    bridge = []
    for survey in surveys:
        day_groups, base = prepare(survey, weighting="instantaneous", grain="group_day")
        day_fits = fit_by_activity(
            day_groups,
            base,
            n_bins=n_bins,
            free_weibull=False,
            stage="group_day",
            min_groups=min_groups,
        )
        day_frame = pd.DataFrame([f.as_row() for f in day_fits])
        group_frame = pd.DataFrame([f.as_row() for f in stage2[survey]])
        bridge.append(
            {
                "survey": survey,
                "n_bins": len(day_frame),
                "mu_group": float(group_frame["mu_LN"].mean()),
                "mu_group_day": float(day_frame["mu_LN"].mean()),
                "mu_shift": float(day_frame["mu_LN"].mean() - group_frame["mu_LN"].mean()),
                "sigma_group": float(group_frame["sigma_LN"].mean()),
                "sigma_group_day": float(day_frame["sigma_LN"].mean()),
                "c_group": float(group_frame["c_LN"].mean()),
                "c_group_day": float(day_frame["c_LN"].mean()),
            }
        )
        print(
            f"  {survey}: mu {bridge[-1]['mu_group']:.3f} (per group) -> "
            f"{bridge[-1]['mu_group_day']:.3f} (per group-day), "
            f"shift {bridge[-1]['mu_shift']:+.3f}"
        )
        day_frame.to_csv(out_dir / f"group_day_{survey}.csv", index=False, lineterminator="\n")
    pd.DataFrame(bridge).to_csv(out_dir / "grain_bridge.csv", index=False, lineterminator="\n")

    # --- model comparison by AIC -------------------------------------------
    comparison = []
    for survey, (groups, base) in prepared.items():
        log_area = np.log(groups["area_corr"].to_numpy())
        sigma = np.nan_to_num(
            groups["area_corr_unc"].to_numpy() / groups["area_corr"].to_numpy(),
            nan=0.0,
            posinf=0.0,
        )
        pooled_params, pooled_ll, _ = fit_bin(log_area, sigma, base, free_weibull=True)
        index, _ = activity_bins(groups["activity"].to_numpy(), n_bins)

        free_ll = sum(f.loglik for f in stage1[survey])
        fixed_ll = sum(f.loglik for f in stage2[survey])

        # Analytic form: parameters from the fitted power laws, evaluated per bin.
        frame = pd.DataFrame([f.as_row() for f in stage2[survey]])
        laws = {
            name: fit_power_law(frame["activity"], frame[name], name, weights=frame["n_groups"])
            for name in LOGNORMAL_PARAMS
        }
        analytic_ll = 0.0
        for fit in stage2[survey]:
            subset = groups[index == fit.bin_index]
            if subset.empty:
                continue
            likelihood = TruncatedLikelihood(
                np.log(subset["area_corr"].to_numpy()),
                np.nan_to_num(
                    subset["area_corr_unc"].to_numpy() / subset["area_corr"].to_numpy(),
                    nan=0.0,
                    posinf=0.0,
                ),
                float(np.log(base.threshold_msh)),
                sigma_floor=base.sigma_floor,
            )
            analytic_ll += likelihood(
                base.replace(
                    mu_LN=float(laws["mu_LN"].predict(fit.activity)),
                    sigma_LN=float(laws["sigma_LN"].predict(fit.activity)),
                    pi_W=float(1.0 - laws["c_LN"].predict(fit.activity)),
                )
            )

        n_fitted = len(stage2[survey])
        for label, loglik, free in (
            ("pooled, no activity dependence", pooled_ll, 5),
            ("free per activity bin", free_ll, 5 * n_fitted),
            ("(k, lambda) fixed, free per bin", fixed_ll, 2 + 3 * n_fitted),
            ("(k, lambda) fixed, analytic power laws", analytic_ll, 2 + 6),
        ):
            comparison.append(
                {
                    "survey": survey,
                    "model": label,
                    "loglik": loglik,
                    "n_free": free,
                    "aic": 2 * free - 2 * loglik,
                }
            )
    comparison_table = pd.DataFrame(comparison)
    for _, group in comparison_table.groupby("survey"):
        comparison_table.loc[group.index, "delta_aic"] = group["aic"] - group["aic"].min()
    comparison_table.to_csv(out_dir / "model_comparison.csv", index=False, lineterminator="\n")
    print("\nmodel comparison (AIC, lower is better):")
    for _, row in comparison_table.iterrows():
        print(f"  {row['survey']:9s} {row['model']:40s} dAIC {row['delta_aic']:10.1f}")

    figures = out_dir / "figures"
    save_both_modes(
        lambda mode: area_law_figure(prepared, stage2, mode), figures, "f10_checkB_area_law"
    )
    save_both_modes(
        lambda mode: activity_dependence_figure(stage2_table, power_table, gate_rows, mode),
        figures,
        "f11_checkB_activity",
    )
    print(f"\nwrote {out_dir}")
    return stage2_table


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
