"""Figures for Check A and Check B -- the panels the go/no-go is read off.

Every measure is drawn against its **advisory** target (PI, 2026-11-21), as a dashed
reference line with a word on it rather than a pass/fail colour wash: the thresholds
are targets on this first pass, and the binding rule is set in the results doc from
the magnitudes these figures show.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gn_next_gen.reference.area_law import PUBLISHED_2015
from gn_next_gen.stats.discrepancy import TARGETS
from gn_next_gen.viz.style import (
    SURVEY_COLORS,
    annotate_target,
    caption,
    figure_style,
    palette,
    role_color,
    sequential_colors,
)

SURVEY_LABELS = {"rgo": "RGO", "debrecen": "Debrecen", "kmas": "KMAS"}

#: The 2015 published power laws, for overlay. Keys match the fitted parameter names.
PUBLISHED_TARGETS = {
    "mu_LN": PUBLISHED_2015["mu_LN"],
    "sigma_LN": PUBLISHED_2015["sigma_LN"],
    "c_LN": PUBLISHED_2015["c_LN"],
}
TROUGH_LAG, BUMP_LAG = 13, 27


def acf_figure(acf_table: pd.DataFrame, mode: str = "light") -> plt.Figure:
    """f07 -- observed against simulated ACF, raw and detrended, per survey."""
    theme = palette(mode)
    surveys = list(dict.fromkeys(acf_table["survey"]))
    with figure_style(mode):
        fig, axes = plt.subplots(
            2, len(surveys), figsize=(3.6 * len(surveys), 6.0), sharex=True, sharey="row"
        )
        axes = np.asarray(axes).reshape(2, len(surveys))
        for col, survey in enumerate(surveys):
            for row, detrended in enumerate((False, True)):
                ax = axes[row, col]
                sub = acf_table[
                    (acf_table["survey"] == survey) & (acf_table["detrended"] == detrended)
                ]
                ax.plot(
                    sub["lag"],
                    sub["observed"],
                    color=role_color("observed", mode),
                    label="observed",
                )
                ax.plot(
                    sub["lag"],
                    sub["simulated"],
                    color=role_color("simulated", mode),
                    label="simulated",
                )
                for lag, text in ((TROUGH_LAG, "transit"), (BUMP_LAG, "recurrence")):
                    ax.axvline(
                        lag, color=theme["text_secondary"], linewidth=0.9, linestyle=(0, (3, 3))
                    )
                    # Annotated on the lower row only, where there is clear space;
                    # the legend lives in the upper row.
                    if row == 1 and col == 0:
                        ax.annotate(
                            text,
                            xy=(lag - 0.6, 0.03),
                            xycoords=("data", "axes fraction"),
                            rotation=90,
                            ha="right",
                            va="bottom",
                            fontsize=7,
                            color=theme["text_secondary"],
                        )
                ax.axhline(0.0, color=theme["grid"], linewidth=0.8)
                if row == 0:
                    ax.set_title(SURVEY_LABELS.get(survey, survey))
                if col == 0:
                    ax.set_ylabel("ACF, detrended" if detrended else "ACF, raw")
                if row == 1:
                    ax.set_xlabel("lag (days)")
                if row == 0 and col == 0:
                    ax.legend(loc="upper right")
        caption(
            fig,
            "The raw ACF rides on the solar cycle; detrending against a 91-day mean is "
            "what exposes the ~13-day transit trough and the ~27-day recurrence bump.",
            mode,
        )
    return fig


def zero_and_counts_figure(
    bin_table: pd.DataFrame, pmf_table: pd.DataFrame, run_table: pd.DataFrame, mode: str = "light"
) -> plt.Figure:
    """f08 -- zero fraction and count distribution per activity bin, and run lengths."""
    theme = palette(mode)
    with figure_style(mode):
        fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))

        ax = axes[0]
        width = 0.38
        positions = np.arange(len(bin_table))
        ax.bar(
            positions - width / 2,
            bin_table["observed"],
            width=width,
            color=role_color("observed", mode),
            label="observed",
        )
        ax.bar(
            positions + width / 2,
            bin_table["simulated"],
            width=width,
            color=role_color("simulated", mode),
            label="simulated",
        )
        ax.set_xticks(positions)
        ax.set_xticklabels([f"Q{int(b) + 1}" for b in bin_table["bin"]])
        ax.set_xlabel("activity quintile (91-day smoothed count)")
        ax.set_ylabel("fraction of observed days with zero groups")
        ax.set_title("Zero fraction by activity")
        ax.legend(loc="upper right")

        ax = axes[1]
        ax.step(
            pmf_table["count"],
            pmf_table["observed"],
            where="mid",
            color=role_color("observed", mode),
            label="observed",
        )
        ax.step(
            pmf_table["count"],
            pmf_table["simulated"],
            where="mid",
            color=role_color("simulated", mode),
            label="simulated",
        )
        ax.set_yscale("log")
        nonzero = pmf_table.loc[(pmf_table[["observed", "simulated"]] > 0).any(axis=1), "count"]
        if len(nonzero):
            ax.set_xlim(-0.5, float(nonzero.max()) + 1.5)
        ax.set_xlabel("groups in a day")
        ax.set_ylabel("probability")
        ax.set_title("Daily count distribution")
        ax.legend(loc="upper right")

        ax = axes[2]
        bins = np.arange(0.5, max(12.5, run_table["length"].max() + 1.5), 1.0)
        ax.hist(
            np.repeat(run_table["length"], run_table["observed"].astype(int)),
            bins=bins,
            histtype="step",
            linewidth=2.0,
            density=True,
            color=role_color("observed", mode),
            label="observed",
        )
        ax.hist(
            np.repeat(run_table["length"], run_table["simulated"].astype(int)),
            bins=bins,
            histtype="step",
            linewidth=2.0,
            density=True,
            color=role_color("simulated", mode),
            label="simulated",
        )
        ax.set_yscale("log")
        ax.set_xlabel("length of a spotless run (days)")
        ax.set_ylabel("density")
        ax.set_title("Spotless run lengths")
        ax.legend(loc="upper right")

        caption(
            fig,
            "The three statistics ADF compresses into one number, kept apart: how often a "
            "day is blank, how the counts are distributed, and how blank days cluster.",
            mode,
        )
        del theme
    return fig


def scores_figure(scores: pd.DataFrame, mode: str = "light") -> plt.Figure:
    """f09 -- every measure against its advisory target, 1-class against 2-class."""
    theme = palette(mode)
    measures = [m for m in TARGETS if m in scores.columns]
    with figure_style(mode):
        fig, axes = plt.subplots(
            1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [2.1, 1.0]}
        )

        ax = axes[0]
        positions = np.arange(len(measures))
        width = 0.38
        for offset, (n_classes, label) in enumerate(((1, "1 area class"), (2, "2 area classes"))):
            sub = scores[scores["n_classes"] == n_classes]
            if sub.empty:
                continue
            ratios = [
                abs(sub[m].mean()) / TARGETS[m] if np.isfinite(sub[m].mean()) else np.nan
                for m in measures
            ]
            ax.bar(
                positions + (offset - 0.5) * width,
                ratios,
                width=width,
                color=role_color("observed" if n_classes == 1 else "simulated", mode),
                label=label,
            )
        ax.axhline(1.0, color=theme["text_secondary"], linestyle=(0, (4, 3)), linewidth=1.2)
        ax.annotate(
            "advisory target",
            xy=(0.995, 1.04),
            xycoords=("axes fraction", "data"),
            ha="right",
            fontsize=7.5,
            color=theme["text_secondary"],
        )
        ax.set_xticks(positions)
        ax.set_xticklabels([m.replace("_", "\n") for m in measures], fontsize=7)
        ax.set_ylabel("measure / its target  (lower is better)")
        ax.set_yscale("log")
        ax.set_title("Discrepancy measures against their advisory targets")
        ax.legend(loc="upper left")

        ax = axes[1]
        by_threshold = scores.groupby(["survey", "threshold_msh"])["aggregate"].mean()
        surveys = list(dict.fromkeys(scores["survey"]))
        for survey in surveys:
            sub = by_threshold.loc[survey]
            ax.plot(
                sub.index,
                sub.to_numpy(),
                marker="o",
                color=SURVEY_COLORS[mode].get(survey, theme["categorical"][6]),
                label=SURVEY_LABELS.get(survey, survey),
            )
        annotate_target(ax, 1.0, mode, label="all targets met")
        ax.set_xlabel("acuity threshold (MSH)")
        ax.set_ylabel("aggregate score")
        ax.set_yscale("log")
        ax.set_title("Aggregate by threshold")
        ax.legend(loc="upper right")

        caption(
            fig,
            "Bars above the dashed line are measures the fitted chain does not yet match. "
            "The thresholds are advisory on this pass; the results doc sets the binding rule.",
            mode,
        )
    return fig


def area_law_figure(prepared: dict, stage2: dict, mode: str = "light") -> plt.Figure:
    """f10 -- the fitted Weibull + log-normal law against the data, per survey.

    The panel that retires the two-log-normal form: the small-object component is a
    left-skewed Gumbel in log-area, not a second Gaussian, and most of it sits below
    each survey's own truncation threshold -- which is why that regime cannot be read
    as solar structure.
    """
    from gn_next_gen.reference.area_law import log_pdf_latent, log_pdf_weibull, z_grid

    theme = palette(mode)
    surveys = list(prepared)
    with figure_style(mode):
        fig, axes = plt.subplots(1, len(surveys), figsize=(3.8 * len(surveys), 4.1), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, survey in zip(axes, surveys, strict=True):
            groups, base = prepared[survey]
            grid = z_grid(2048)
            areas = np.log(groups["area_corr"].to_numpy())
            ax.hist(
                areas,
                bins=60,
                density=True,
                histtype="stepfilled",
                color=role_color("observed", mode),
                alpha=0.28,
                linewidth=0,
                label="observed, above threshold",
            )

            # Renormalize the latent law onto the retained support for comparison.
            density = np.exp(
                log_pdf_latent(
                    grid, base.k_W, base.log_lambda_W, base.mu_LN, base.sigma_LN, base.pi_W
                )
            )
            keep = grid > np.log(base.threshold_msh)
            norm_const = np.trapezoid(density[keep], grid[keep])
            ax.plot(
                grid[keep],
                density[keep] / norm_const,
                color=role_color("simulated", mode),
                label="Weibull + log-normal",
            )

            weibull = base.pi_W * np.exp(log_pdf_weibull(grid, base.k_W, base.log_lambda_W))
            lognormal = (
                (1 - base.pi_W)
                * np.exp(-0.5 * ((grid - base.mu_LN) / base.sigma_LN) ** 2)
                / (base.sigma_LN * np.sqrt(2 * np.pi))
            )
            ax.plot(
                grid,
                weibull / norm_const,
                color=role_color("alternative", mode),
                linestyle=(0, (4, 2)),
                linewidth=1.5,
                label="Weibull (small objects)",
            )
            ax.plot(
                grid,
                lognormal / norm_const,
                color=theme["categorical"][6],
                linestyle=(0, (2, 2)),
                linewidth=1.5,
                label="log-normal (active regions)",
            )

            ax.axvline(
                np.log(base.threshold_msh),
                color=theme["text_secondary"],
                linestyle=(0, (4, 3)),
                linewidth=1.1,
            )
            ax.annotate(
                f"threshold\n{base.threshold_msh:.0f} MSH",
                xy=(np.log(base.threshold_msh) - 0.15, 0.92),
                xycoords=("data", "axes fraction"),
                ha="right",
                fontsize=7.5,
                color=theme["text_secondary"],
            )
            ax.set_xlim(0, 9)
            ax.set_ylim(0, None)
            ax.set_xlabel(r"$\ln$ area (MSH)")
            ax.set_title(SURVEY_LABELS.get(survey, survey))
            if survey == surveys[0]:
                ax.set_ylabel("density")
                ax.legend(loc="upper right", fontsize=7)
        caption(
            fig,
            "Two populations, and they are not two log-normals: most of the Weibull "
            "component lies below each survey's own completeness threshold, so counts "
            "from that regime measure the instrument as much as the Sun.",
            mode,
        )
    return fig


def activity_dependence_figure(
    stage2: pd.DataFrame, power_laws: pd.DataFrame, gate_rows: list, mode: str = "light"
) -> plt.Figure:
    """f11 -- the log-normal moves with activity, the Weibull does not."""
    theme = palette(mode)
    surveys = list(dict.fromkeys(stage2["survey"]))
    panels = ("mu_LN", "sigma_LN", "c_LN")
    titles = {
        "mu_LN": r"log-normal mean $\mu$",
        "sigma_LN": r"log-normal width $\sigma$",
        "c_LN": "log-normal weight $c$",
    }

    with figure_style(mode):
        fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.9))
        grid = np.linspace(0.02, 0.6, 100)

        for ax, name in zip(axes[:3], panels, strict=True):
            for survey in surveys:
                sub = stage2[stage2["survey"] == survey]
                ax.plot(
                    sub["activity"],
                    sub[name],
                    marker="o",
                    markersize=4.5,
                    linestyle="none",
                    color=SURVEY_COLORS[mode].get(survey, theme["categorical"][6]),
                    label=SURVEY_LABELS.get(survey, survey),
                )
                law = power_laws[
                    (power_laws["survey"] == survey) & (power_laws["parameter"] == name)
                ]
                if len(law):
                    row = law.iloc[0]
                    ax.plot(
                        grid,
                        row["a"] * grid ** row["b"],
                        linewidth=1.3,
                        color=SURVEY_COLORS[mode].get(survey, theme["categorical"][6]),
                    )
            published = PUBLISHED_TARGETS.get(name)
            if published:
                ax.plot(
                    grid,
                    published[0] * grid ** published[1],
                    color=theme["text_secondary"],
                    linestyle=(0, (4, 3)),
                    linewidth=1.6,
                    label="published (2015)",
                )
            ax.set_xscale("log")
            ax.set_xlabel("activity level (mHem)")
            ax.set_title(titles[name])
            if name == "mu_LN":
                ax.legend(loc="upper left", fontsize=7.5)

        ax = axes[3]
        gate = pd.DataFrame(gate_rows)
        positions = np.arange(len(gate))
        colors = [SURVEY_COLORS[mode].get(s, theme["categorical"][6]) for s in gate["survey"]]
        ax.barh(positions, gate["spearman_rho"], color=colors, height=0.62)
        ax.axvline(0, color=theme["grid"], linewidth=0.8)
        ax.set_yticks(positions)
        ax.set_yticklabels([f"{r.survey}\n{r.parameter}" for r in gate.itertuples()], fontsize=7)
        ax.set_xlim(-1, 1)
        ax.set_xlabel(r"Spearman $\rho$ vs activity")
        ax.set_title("Weibull: no consistent trend")
        caption(
            fig,
            "The log-normal half of the law tracks activity, as the 2015 paper found; "
            "the Weibull half does not, which is what licenses holding it fixed.",
            mode,
        )
    return fig


def filter_benchmark_figure(timings: pd.DataFrame, mode: str = "light") -> plt.Figure:
    """f12 -- measured filter cost against the research report's estimates."""
    theme = palette(mode)
    with figure_style(mode):
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))

        ax = axes[0]
        for index, (device, group) in enumerate(timings.groupby("device")):
            for states, sub in group.groupby("states"):
                ax.plot(
                    sub["n_days"],
                    sub["seconds"],
                    marker="o",
                    color=sequential_colors(2, mode)[index % 2],
                    linestyle="-" if states <= 64 else (0, (3, 2)),
                    label=f"{device}, {states} states",
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("days in the series, $T$")
        ax.set_ylabel("seconds per likelihood evaluation")
        ax.set_title("Measured forward-filter cost")
        ax.legend(loc="upper left", fontsize=7.5)

        ax = axes[1]
        full = timings[timings["n_days"] == timings["n_days"].max()]
        positions = np.arange(len(full))
        ax.bar(positions, full["seconds"], width=0.6, color=role_color("observed", mode))
        for pos, (_, row) in zip(positions, full.iterrows(), strict=True):
            ax.annotate(
                f"{row['seconds']:.2f}s",
                xy=(pos, row["seconds"]),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=theme["text_primary"],
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [f"{row['device']}\n{int(row['states'])} states" for _, row in full.iterrows()],
            fontsize=7.5,
        )
        ax.set_ylabel("seconds at full length")
        ax.set_title(f"At T = {int(full['n_days'].max()):,} days")

        caption(
            fig,
            "The research report's 0.2-1 s CPU and 1-3 s single-chain GPU figures were "
            "estimates; these are measurements, and they replace them (Step 4 early task 1).",
            mode,
        )
    return fig
