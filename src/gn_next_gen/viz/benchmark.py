"""Step 2's figures, f13-f26, in the house style of Step 1.

Figure-first, as `CLAUDE.md` requires: each figure carries the claim it supports
and every number in it comes from a committed CSV.  Two of them -- `f25` and
`f26` -- exist because an aggregate score says which method wins and hides the
observers that break one.
"""

from __future__ import annotations

import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gn_next_gen.viz.style import (
    annotate_target,
    caption,
    figure_style,
    palette,
    sequential_colors,
)

#: Method order, used by every table and figure so a reader learns it once: the
#: three linear scalings, the two non-parametric methods, the count-distribution
#: variant, the backbone, then the nulls.
METHOD_ORDER = [
    "HoSc98",
    "SvSc16",
    "VHS24",
    "CEA17",
    "UEA21",
    "GRP",
    "Backbone",
    "null-segmentation",
    "null-prior",
]

UNIX_EPOCH_ORDINAL = 719_163


def _order(methods) -> list[str]:
    present = [m for m in METHOD_ORDER if m in set(methods)]
    return present + sorted(set(methods) - set(present))


def _method_colors(methods, mode: str) -> dict[str, str]:
    theme = palette(mode)
    categorical = theme["categorical"]
    colors = {}
    for index, method in enumerate(_order(methods)):
        colors[method] = (
            theme["text_secondary"] if method.startswith("null-") else categorical[index % 8]
        )
    return colors


#: Nulls are drawn in the same recessive grey, so they need a second channel to
#: tell them apart -- colour alone is never the identity (the house rule).
NULL_LINESTYLE = {"null-prior": (0, (1, 2)), "null-segmentation": (0, (5, 2))}


def _linestyle(method: str):
    return NULL_LINESTYLE.get(method, "-")


def _years(date_ord) -> np.ndarray:
    return pd.to_datetime(np.asarray(date_ord) - UNIX_EPOCH_ORDINAL, unit="D").year.to_numpy()


# --- f18: the headline score table, as a figure ---------------------------


def scores_figure(scores: pd.DataFrame, mode: str = "light") -> plt.Figure:
    """f18 -- bias and centred RMSE at three averaging scales, per method."""
    with figure_style(mode) as theme:
        methods = _order(scores["method"].unique())
        colors = _method_colors(methods, mode)
        panels = [
            ("bias_daily", "crmse_daily", "daily"),
            ("bias_monthly", "crmse_monthly", "monthly"),
            ("bias_yearly", "crmse_yearly", "yearly"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
        for ax, (bias_key, crmse_key, label) in zip(axes, panels, strict=True):
            grouped = scores.groupby("method")[[bias_key, crmse_key]].mean()
            positions = np.arange(len(methods))
            ax.barh(
                positions,
                [grouped.loc[m, crmse_key] for m in methods],
                color=[colors[m] for m in methods],
                alpha=0.45,
                label="centred RMSE",
            )
            ax.scatter(
                [grouped.loc[m, bias_key] for m in methods],
                positions,
                color=[colors[m] for m in methods],
                zorder=3,
                s=26,
                label="bias",
            )
            ax.axvline(0.0, color=theme["grid"], linewidth=1.0, zorder=1)
            ax.set_yticks(positions)
            # With sharey, blanking the labels on one axis blanks them on all of
            # them, so hide the ticks per axis instead of emptying the list.
            ax.set_yticklabels(methods)
            ax.tick_params(labelleft=(ax is axes[0]))
            ax.set_xlabel(f"groups, {label}")
            ax.set_title(label, fontsize=9, color=theme["text_secondary"])
        axes[0].legend(loc="lower right", fontsize=7.5, frameon=False)
        caption(
            fig,
            "Bars are centred RMSE, dots are bias. A method can be unbiased daily "
            "and wrong per cycle, which is why three scales are shown.",
            mode,
        )
        fig.tight_layout()
        return fig


# --- f19: the A157 secular-bias signature ---------------------------------


def amplitude_slope_figure(scores: pd.DataFrame, mode: str = "light") -> plt.Figure:
    """f19 -- does a method amplify strong cycles against weak ones?"""
    with figure_style(mode) as theme:
        methods = _order(scores["method"].unique())
        colors = _method_colors(methods, mode)
        grouped = scores.groupby("method")["amplitude_slope"]
        mean = grouped.mean()
        spread = grouped.std(ddof=1).fillna(0.0)

        fig, ax = plt.subplots(figsize=(7.4, 3.6))
        positions = np.arange(len(methods))
        ax.barh(
            positions,
            [mean.get(m, np.nan) for m in methods],
            xerr=[spread.get(m, 0.0) for m in methods],
            color=[colors[m] for m in methods],
            error_kw={"ecolor": theme["text_secondary"], "elinewidth": 0.9},
        )
        ax.axvline(0.0, color=theme["text_secondary"], linewidth=1.0)
        ax.set_yticks(positions)
        ax.set_yticklabels(methods)
        ax.set_xlabel("slope of (estimated - true) cycle amplitude against true amplitude")
        ax.annotate(
            "amplifies strong cycles ->",
            xy=(0.995, 1.02),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=theme["text_secondary"],
        )
        caption(
            fig,
            "Positive means strong cycles are inflated relative to weak ones - the "
            "secular bias Chatzistergos et al. (2025) attribute to the linear scalings.",
            mode,
        )
        fig.tight_layout()
        return fig


# --- f20: coverage, the load-bearing metric -------------------------------


def coverage_figure(coverage: pd.DataFrame, mode: str = "light") -> plt.Figure:
    """f20 -- empirical against nominal interval coverage."""
    with figure_style(mode) as theme:
        methods = _order(coverage["method"].unique())
        colors = _method_colors(methods, mode)
        fig, ax = plt.subplots(figsize=(6.0, 5.2))
        ax.plot([0, 1], [0, 1], color=theme["text_secondary"], linestyle=(0, (4, 3)), linewidth=1.0)
        ax.annotate(
            "honest intervals",
            xy=(0.62, 0.66),
            rotation=38,
            fontsize=7.5,
            color=theme["text_secondary"],
        )
        for method in methods:
            block = coverage[coverage["method"] == method].groupby("nominal")["empirical"].mean()
            ax.plot(
                block.index,
                block.to_numpy(),
                marker="o",
                markersize=4,
                linewidth=1.4,
                color=colors[method],
                linestyle=_linestyle(method),
                label=method,
            )
        ax.set_xlabel("nominal coverage")
        ax.set_ylabel("empirical coverage")
        ax.set_xlim(0.4, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=7.5, frameon=False, ncol=2)
        caption(
            fig,
            "Every curve below the diagonal is an overconfident method. "
            "A157 scores four point-error metrics and never scores an interval.",
            mode,
        )
        fig.tight_layout()
        return fig


# --- f21: the Hay sampling-only diagnostic --------------------------------


def sampling_only_figure(
    bootstrap: pd.DataFrame, trend: pd.DataFrame, mode: str = "light"
) -> plt.Figure:
    """f21 -- what the observing calendar alone does, with no detection error."""
    with figure_style(mode) as theme:
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
        grouped = bootstrap.groupby("n_observers")["trend_bias"]
        sizes = grouped.mean().index.to_numpy()
        mean = grouped.mean().to_numpy()
        spread = grouped.std(ddof=1).to_numpy()

        ax = axes[0]
        ax.fill_between(
            sizes, mean - spread, mean + spread, color=theme["categorical"][0], alpha=0.22
        )
        ax.plot(sizes, mean, marker="o", markersize=4, color=theme["categorical"][0])
        annotate_target(ax, 0.0, mode, label="no bias")
        ax.set_xscale("log")
        ax.set_xlabel("observers in the network")
        ax.set_ylabel("induced trend bias\n(groups per century)")
        ax.set_title("bias against network size", fontsize=9, color=theme["text_secondary"])

        ax = axes[1]
        positions = np.arange(len(trend))
        ax.barh(positions, trend["induced_bias"], color=theme["categorical"][1])
        ax.axvline(0.0, color=theme["text_secondary"], linewidth=1.0)
        ax.set_yticks(positions)
        ax.set_yticklabels(trend["window"])
        ax.set_xlabel("sampled minus true trend (groups per century)")
        ax.set_title("by window", fontsize=9, color=theme["text_secondary"])

        caption(
            fig,
            "Perfect observers on real calendars: any error here is the sampling "
            "pattern alone. Hay et al. found the equivalent bias levelled off well above zero.",
            mode,
        )
        fig.tight_layout()
        return fig


# --- f23: observer-parameter recovery -------------------------------------


def acuity_recovery_figure(cases: pd.DataFrame, mode: str = "light") -> plt.Figure:
    """f23 -- recovered against true acuity, for the methods that estimate one."""
    with figure_style(mode) as theme:
        usable = cases[np.isfinite(cases["recovered_threshold_msh"])]
        fig, ax = plt.subplots(figsize=(5.8, 5.2))
        if usable.empty:
            ax.text(
                0.5,
                0.5,
                "No ported baseline estimates an observer acuity\n"
                "except UEA21 and GRP; none ran in this slice.",
                ha="center",
                va="center",
                fontsize=9,
                color=theme["text_secondary"],
            )
            ax.set_axis_off()
        else:
            methods = _order(usable["method"].unique())
            colors = _method_colors(methods, mode)
            limit = float(
                max(usable["true_threshold_msh"].max(), usable["recovered_threshold_msh"].max())
            )
            ax.plot(
                [0, limit],
                [0, limit],
                color=theme["text_secondary"],
                linestyle=(0, (4, 3)),
                linewidth=1.0,
            )
            for method in methods:
                block = usable[usable["method"] == method]
                ax.scatter(
                    block["true_threshold_msh"],
                    block["recovered_threshold_msh"],
                    s=34,
                    color=colors[method],
                    label=method,
                    alpha=0.85,
                )
            ax.set_xlabel("true acuity threshold (MSH)")
            ax.set_ylabel("recovered acuity threshold (MSH)")
            ax.legend(fontsize=7.5, frameon=False)
        caption(
            fig,
            "Points above the diagonal are observers judged worse than they were. "
            "A157 reports UEA21 systematically overestimates acuity.",
            mode,
        )
        fig.tight_layout()
        return fig


# --- f24: the A157 validation ---------------------------------------------


def a157_figure(table: pd.DataFrame, mode: str = "light") -> plt.Figure:
    """f24 -- our ports against the published Fig. 3b behaviour."""
    with figure_style(mode) as theme:
        methods = _order(table["method"].unique())
        colors = _method_colors(methods, mode)
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))

        ax = axes[0]
        for method in methods:
            block = table[table["method"] == method].sort_values("acuity_msd")
            ax.plot(
                block["acuity_msd"],
                block["mean_difference"],
                linewidth=1.5,
                color=colors[method],
                label=method,
                # CEA17 and UEA21 coincide exactly in this idealised case -- "by
                # design, CEA17 and UEA21 are identical in these metrics in this
                # case" (A157 §3.1) -- so one is dashed to keep both visible.
                linestyle=(0, (6, 3)) if method == "UEA21" else _linestyle(method),
            )
        ax.axhline(0.0, color=theme["text_secondary"], linewidth=1.0)
        ax.set_xscale("log")
        ax.set_xlabel("acuity threshold (msd)")
        ax.set_ylabel("mean difference (groups)")
        ax.set_title("their Fig. 3b, rebuilt", fontsize=9, color=theme["text_secondary"])
        ax.legend(fontsize=7.5, frameon=False)

        ax = axes[1]
        for method in methods:
            block = table[table["method"] == method].sort_values("acuity_msd")
            ax.plot(
                block["acuity_msd"],
                block["rms_difference"],
                linewidth=1.5,
                color=colors[method],
                label=method,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("acuity threshold (msd)")
        ax.set_ylabel("RMS difference (groups)")
        ax.set_title("errors grow with acuity, for all", fontsize=9, color=theme["text_secondary"])

        caption(
            fig,
            "HoSc98 and SvSc16 fall away with acuity; CEA17, UEA21 and VHS24 stay near "
            "zero - the sign pattern A157 published, reproduced by these ports.",
            mode,
        )
        fig.tight_layout()
        return fig


# --- f25, f26: the ten case-study observers -------------------------------


def case_study_figure(
    cases: pd.DataFrame, kinds: list[str], mode: str = "light", *, title: str = ""
) -> plt.Figure:
    """f25/f26 -- one column per named observer: acuity above, scale below.

    Two rows because only UEA21 and GRP estimate an acuity, and a figure whose
    job is to show *why* a method fails cannot leave seven of nine out.  Every
    method produces a scale factor, and the truth implies one, so the lower row
    is comparable across all of them.
    """
    with figure_style(mode) as theme:
        fig, axes = plt.subplots(2, len(kinds), figsize=(2.35 * len(kinds), 5.0), sharey="row")
        axes = np.atleast_2d(axes)
        methods = _order(cases["method"].unique())
        colors = _method_colors(methods, mode)

        for column, kind in enumerate(kinds):
            block = cases[cases["kind"] == kind]
            top, bottom = axes[0, column], axes[1, column]
            top.set_title(kind, fontsize=8, color=theme["text_primary"])

            if block.empty:
                for ax in (top, bottom):
                    ax.text(
                        0.5,
                        0.5,
                        "not in slice",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color=theme["text_secondary"],
                        transform=ax.transAxes,
                    )
                    ax.set_xticks([])
                continue

            true_acuity = float(block["true_threshold_msh"].iloc[0])
            top.axhline(
                true_acuity, color=theme["text_secondary"], linestyle=(0, (4, 3)), linewidth=1.1
            )
            drawn = 0
            for index, method in enumerate(methods):
                row = block[block["method"] == method]
                if row.empty:
                    continue
                value = row["recovered_threshold_msh"].iloc[0]
                if np.isfinite(value):
                    top.scatter(index, value, s=28, color=colors[method])
                    drawn += 1
            if drawn == 0:
                top.annotate(
                    "no method\nestimates acuity",
                    xy=(0.5, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=theme["text_secondary"],
                )
            top.set_xticks([])
            top.set_xlim(-1, len(methods))

            bottom.axhline(1.0, color=theme["text_secondary"], linestyle=(0, (4, 3)), linewidth=1.1)
            for index, method in enumerate(methods):
                row = block[block["method"] == method]
                if row.empty or "factor_ratio" not in row:
                    continue
                ratio = row["factor_ratio"].iloc[0]
                if np.isfinite(ratio) and ratio > 0:
                    bottom.scatter(
                        index,
                        ratio,
                        s=28,
                        color=colors[method],
                        label=method if column == 0 else None,
                    )
            bottom.set_yscale("log")
            bottom.set_xticks([])
            bottom.set_xlim(-1, len(methods))

        axes[0, 0].set_ylabel("acuity threshold (MSH)")
        axes[1, 0].set_ylabel("recovered / implied\nscale factor")
        handles, labels = axes[1, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=min(len(labels), 5),
                fontsize=7,
                frameon=False,
                bbox_to_anchor=(0.5, -0.06),
            )
        if title:
            fig.suptitle(title, fontsize=9, color=theme["text_secondary"])
        caption(
            fig,
            "Dashed lines are the truth: the observer's acuity above, a scale factor of "
            "one below. An aggregate score hides exactly these observers.",
            mode,
        )
        fig.tight_layout()
        return fig


# --- f15: the overlap graph through time ----------------------------------


def graph_figure(resistance: pd.DataFrame, edges: pd.DataFrame, mode: str = "light") -> plt.Figure:
    """f15 -- how far each observer sits from the anchor, and when."""
    with figure_style(mode) as theme:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))

        ax = axes[0]
        finite = resistance[np.isfinite(resistance["resistance"])]
        unreachable = resistance[~np.isfinite(resistance["resistance"])]
        ax.scatter(
            _years(finite["first_ord"]),
            finite["resistance"],
            s=14,
            color=theme["categorical"][0],
            alpha=0.75,
            label=f"reachable ({len(finite)})",
        )
        if not unreachable.empty:
            ax.scatter(
                _years(unreachable["first_ord"]),
                np.full(len(unreachable), finite["resistance"].max() * 1.3 + 1e-9),
                s=16,
                marker="^",
                color=theme["categorical"][3],
                label=f"unreachable ({len(unreachable)})",
            )
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_xlabel("observer's first year")
        ax.set_ylabel("effective resistance to the anchor")
        ax.legend(fontsize=7.5, frameon=False)

        ax = axes[1]
        ax.hist(
            edges["overlap_days"],
            bins=np.geomspace(max(edges["overlap_days"].min(), 1), edges["overlap_days"].max(), 40),
            color=theme["categorical"][2],
        )
        ax.set_xscale("log")
        ax.set_xlabel("overlapping days per edge")
        ax.set_ylabel("edges")

        caption(
            fig,
            "Chained calibration's variance grows like effective resistance, so the left "
            "panel is the x-axis of every chained method's error.",
            mode,
        )
        fig.tight_layout()
        return fig


# --- f14, f17: the worlds and their calendars -----------------------------


def world_figure(yearly_truth: pd.Series, donors: pd.DataFrame, mode: str = "light") -> plt.Figure:
    """f14 -- the replayed truth, and which real cycle each era came from."""
    with figure_style(mode) as theme:
        fig, axes = plt.subplots(2, 1, figsize=(10.0, 4.4), sharex=True, height_ratios=[3, 1])
        ax = axes[0]
        ax.fill_between(
            yearly_truth.index, yearly_truth.to_numpy(), color=theme["categorical"][0], alpha=0.75
        )
        ax.set_ylabel("true reference-equivalent\ngroups per day")

        ax = axes[1]
        cycles = sorted(donors["donor_cycle"].unique())
        ramp = sequential_colors(max(len(cycles), 1), mode)
        for color, cycle in zip(ramp, cycles, strict=False):
            block = donors[donors["donor_cycle"] == cycle]
            ax.scatter(
                _years(block["emergence_ord"]),
                np.full(len(block), cycle),
                s=1.2,
                color=color,
            )
        ax.set_ylabel("donor cycle")
        ax.set_xlabel("year")

        caption(
            fig,
            "Inside the donor survey's own span the truth is the real record, day for day; "
            "outside it whole real cycles are shifted and thinned.",
            mode,
        )
        fig.tight_layout()
        return fig


def calendar_figure(summary: pd.DataFrame, mode: str = "light") -> plt.Figure:
    """f17 -- the real observing calendars the synthetic network inherits."""
    with figure_style(mode) as theme:
        fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.4))
        years = _years(summary["first_ord"])
        ax = axes[0]
        ax.scatter(
            years,
            summary["n_days"],
            s=12,
            color=theme["categorical"][0],
            alpha=0.75,
        )
        ax.set_yscale("log")
        ax.set_xlabel("observer's first year")
        ax.set_ylabel("observing days")

        ax = axes[1]
        edges = np.arange(1600, 2041, 10)
        ax.hist(years, bins=edges, color=theme["categorical"][2])
        ax.set_xlabel("decade of first observation")
        ax.set_ylabel("observers")

        caption(
            fig,
            "Transplanted verbatim, including the sparse Maunder-era calendars where the "
            "overlap graph is thinnest. A157 instead deleted days at random.",
            mode,
        )
        fig.tight_layout()
        return fig


def design_figure(mode: str = "light") -> plt.Figure:
    """f13 -- what the benchmark does, as a diagram."""
    with figure_style(mode) as theme:
        fig, ax = plt.subplots(figsize=(10.0, 3.2))
        ax.set_axis_off()
        stages = [
            ("Truth", "real groups,\nareas, lifetimes\n1610-2026"),
            ("Observers", "soft threshold,\ndrift, breaks,\nsplit/merge, MNAR"),
            ("Sampling", "real calendars\nand the real\noverlap graph"),
            ("Sealed", "truth hidden\nby API, audited\nby test"),
            ("Methods", "7 baselines\n+ 2 nulls,\nrun blind"),
            ("Scores", "4 time scales,\ncoverage, CRPS,\ntrend, MM level"),
        ]
        width, gap = 1.0, 0.42
        for index, (title, body) in enumerate(stages):
            left = index * (width + gap)
            color = theme["categorical"][index % 8]
            ax.add_patch(
                plt.Rectangle((left, 0.0), width, 1.0, facecolor=color, alpha=0.18, edgecolor=color)
            )
            ax.text(
                left + width / 2,
                0.84,
                title,
                ha="center",
                fontsize=9,
                color=theme["text_primary"],
                weight="bold",
            )
            ax.text(
                left + width / 2,
                0.42,
                body,
                ha="center",
                va="center",
                fontsize=7.2,
                color=theme["text_secondary"],
            )
            if index < len(stages) - 1:
                ax.annotate(
                    "",
                    xy=(left + width + gap * 0.85, 0.5),
                    xytext=(left + width + gap * 0.15, 0.5),
                    arrowprops={"arrowstyle": "->", "color": theme["text_secondary"]},
                )
        ax.set_xlim(-0.2, len(stages) * (width + gap))
        ax.set_ylim(-0.1, 1.15)
        caption(
            fig,
            "The generator differs from the Step 4 model by design, so the benchmark "
            "cannot flatter it.",
            mode,
        )
        fig.tight_layout()
        return fig


def observer_population_figure(parameters: pd.DataFrame, mode: str = "light") -> plt.Figure:
    """f16 -- the drawn observer population, and the ten named case studies."""
    with figure_style(mode) as theme:
        drawn = parameters[parameters["kind"] == "population"]
        named = parameters[~parameters["kind"].isin({"population", "reference"})]
        fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))

        ax = axes[0]
        ax.hist(
            np.log10(drawn["base_threshold_msh"].clip(lower=1e-3)),
            bins=30,
            color=theme["categorical"][0],
        )
        for value in named["base_threshold_msh"]:
            ax.axvline(np.log10(max(value, 1e-3)), color=theme["categorical"][1], linewidth=0.9)
        ax.set_xlabel("log10 acuity threshold (MSH)")
        ax.set_ylabel("observers")
        ax.set_title("acuity", fontsize=9, color=theme["text_secondary"])

        ax = axes[1]
        ax.hist(drawn["mnar_omega"], bins=25, color=theme["categorical"][2])
        ax.set_xlabel("MNAR omission at zero activity")
        ax.set_title("spotless-day reporting", fontsize=9, color=theme["text_secondary"])

        ax = axes[2]
        counts = {
            "breaks": int((drawn["n_breaks"] > 0).sum()),
            "documented": int((drawn["n_documented_breaks"] > 0).sum()),
            "copiers": int((drawn["copies_of"] >= 0).sum()),
            "blanket": int(drawn["blanket"].sum()),
        }
        ax.barh(list(counts), list(counts.values()), color=theme["categorical"][3])
        ax.set_xlabel("observers")
        ax.set_title("career effects", fontsize=9, color=theme["text_secondary"])

        caption(
            fig,
            "Orange lines are the ten named case studies. Everything here is excluded from "
            "A157's generator by their own statement.",
            mode,
        )
        fig.tight_layout()
        return fig


def structural_figure(structural: pd.DataFrame, scores: pd.DataFrame, mode="light") -> plt.Figure:
    """f22 -- the spread a method's own free choices produce."""
    with figure_style(mode) as theme:
        methods = _order(structural["method"].unique())
        colors = _method_colors(methods, mode)
        fig, ax = plt.subplots(figsize=(7.4, 3.6))
        positions = np.arange(len(methods))
        for index, method in enumerate(methods):
            block = structural[structural["method"] == method]["bias_yearly"].dropna()
            if block.empty:
                continue
            ax.plot(
                [block.min(), block.max()],
                [index, index],
                linewidth=3.0,
                color=colors[method],
                alpha=0.7,
                solid_capstyle="round",
            )
            ax.scatter(block.mean(), index, s=28, color=colors[method], zorder=3)
            parametric = scores[scores["method"] == method]["bias_yearly"].dropna()
            if not parametric.empty:
                ax.scatter(
                    parametric.mean(),
                    index,
                    s=28,
                    marker="x",
                    color=theme["text_secondary"],
                    zorder=4,
                )
        ax.axvline(0.0, color=theme["text_secondary"], linewidth=1.0)
        ax.set_yticks(positions)
        ax.set_yticklabels(methods)
        ax.set_xlabel("yearly bias (groups)")
        caption(
            fig,
            "Bars span the method's own free choices (graph tolerance, minimum overlap); "
            "crosses are the default setting. Structural spread sits beside the parametric one.",
            mode,
        )
        fig.tight_layout()
        return fig


def _year_of(date_ord: int) -> int:
    return dt.date.fromordinal(int(date_ord)).year
