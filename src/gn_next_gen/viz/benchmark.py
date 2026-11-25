"""Step 2's figures, f13-f28 and f41-f42, in the house style of Step 1.

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from gn_next_gen.benchmark.scoring import ERA_ORDER, ERAS
from gn_next_gen.viz.style import (
    annotate_target,
    caption,
    figure_style,
    palette,
    sequential_colors,
)

#: Method order, used by every table and figure so a reader learns it once: the
#: three linear scalings, the two non-parametric methods, the count-distribution
#: variant, the pathfinding chain, then the four floors -- hardest first, so the
#: bar a method has to clear is the first grey line a reader meets.
METHOD_ORDER = [
    "HoSc98",
    "SvSc16",
    "VHS24",
    "CEA17",
    "UEA21",
    "GRP",
    "Pathfinding",
    # Step 3's concurrent fit and its cross-check arm, kept together and ahead of
    # the floors so every figure orders methods the same way.
    "NetCal",
    "NetCal-M",
    "naive-median",
    "naive-mean",
    "null-segmentation",
    "null-prior",
]

#: The floors, which are drawn recessively and never ranked with the seven.
FLOOR_PREFIXES = ("null-", "naive-")


def is_floor(method: str) -> bool:
    return method.startswith(FLOOR_PREFIXES)


#: A *variant* is a second arm of the same estimator rather than an independent
#: method, so it takes its parent's colour and is told apart by shape and
#: linestyle.  ``NetCal-M`` swaps only the pairwise kernel, and reporting it as a
#: ninth colour would both overstate it and overflow the palette: the house rule
#: is eight categorical slots, never cycled, and the seven published methods plus
#: NetCal already fill eight exactly.
VARIANT_OF = {"NetCal-M": "NetCal"}


def is_variant(method: str) -> bool:
    return method in VARIANT_OF


UNIX_EPOCH_ORDINAL = 719_163


def _order(methods) -> list[str]:
    present = [m for m in METHOD_ORDER if m in set(methods)]
    return present + sorted(set(methods) - set(present))


def _method_colors(methods, mode: str) -> dict[str, str]:
    """A method's colour, fixed by `METHOD_ORDER` rather than by who is present.

    Keyed off the position in the full order, not off the position in this
    figure's subset: with the subset indexing, UEA21 was pink in f20, where all
    nine methods appear, and blue in f23, where only two do.  The shapes in
    `METHOD_MARKERS` are fixed, so the colours have to be as well, or the two
    channels disagree about which method a mark belongs to.
    """
    theme = palette(mode)
    categorical = theme["categorical"]
    # Slots are allocated over the *coloured* methods only -- floors share one grey
    # and variants share their parent's hue -- so the count that has to fit in the
    # eight-slot palette is the count of independent methods, and adding a variant
    # can never push anything onto a second lap of the palette.
    coloured = [m for m in METHOD_ORDER if not is_floor(m) and not is_variant(m)]
    extras = [
        m
        for m in _order(methods)
        if m not in METHOD_ORDER and not is_floor(m) and not is_variant(m)
    ]
    slot_of = {method: index for index, method in enumerate(coloured)}
    for offset, method in enumerate(extras):
        slot_of[method] = len(coloured) + offset

    colors = {}
    for method in _order(methods):
        if is_floor(method):
            colors[method] = theme["text_secondary"]
            continue
        parent = VARIANT_OF.get(method, method)
        slot = slot_of.get(parent, slot_of.get(method, 0))
        colors[method] = categorical[slot % len(categorical)]
    return colors


#: The floors are drawn in the same recessive grey, so they need a second channel
#: to tell them apart -- colour alone is never the identity (the house rule).
NULL_LINESTYLE = {
    "null-prior": (0, (1, 2)),
    "null-segmentation": (0, (5, 2)),
    "naive-mean": (0, (3, 1, 1, 1)),
    "naive-median": (0, (6, 1, 1, 1)),
    # A variant shares its parent's hue, so it needs the second channel too.
    "NetCal-M": (0, (4, 2)),
}


def _linestyle(method: str):
    return NULL_LINESTYLE.get(method, "-")


#: One marker shape per method, carried by every figure that draws methods as
#: points, so shape and colour both say which method a mark belongs to: the
#: identity survives a greyscale print, a reader who cannot separate two hues,
#: and the two nulls sharing one recessive grey.  `x` is left out on purpose --
#: `x` is left out on purpose -- f22 reserves it for the default-setting cross,
#: which is not a method.  The two naive floors take the pentagon and the hexagon,
#: which read as a pair without colliding with anything the seven use.
METHOD_MARKERS = {
    "HoSc98": "o",
    "SvSc16": "s",
    "VHS24": "D",
    "CEA17": "^",
    "UEA21": "v",
    "GRP": "P",
    "Pathfinding": "*",
    "NetCal": "X",
    "NetCal-M": "d",
    "naive-median": "p",
    "naive-mean": "h",
    "null-segmentation": ">",
    "null-prior": "<",
}

#: Shapes for methods this module does not know about yet.  Allocation skips any
#: shape a named method already holds: indexing this tuple by position was fine
#: while there were nine methods and ten shapes, and started handing a new method
#: SvSc16's square the moment the two naive floors were registered.
MARKER_FALLBACK = ("8", "d", "H", "+", "1", "2", "3", "4", "|", "_")

#: A star of a given nominal size reads much smaller than a circle of it, and a
#: diamond larger; these are linear size factors that even out visual weight.
MARKER_SCALE = {"*": 1.75, "P": 1.2, "p": 1.15, "h": 1.15, "D": 0.9}


def _method_markers(methods) -> dict[str, str]:
    """One shape per method, and never the same shape twice."""
    ordered = _order(methods)
    taken = {METHOD_MARKERS[m] for m in ordered if m in METHOD_MARKERS}
    spare = [shape for shape in MARKER_FALLBACK if shape not in taken]
    markers, next_spare = {}, 0
    for method in ordered:
        if method in METHOD_MARKERS:
            markers[method] = METHOD_MARKERS[method]
            continue
        markers[method] = spare[next_spare % len(spare)] if spare else "o"
        next_spare += 1
    return markers


def _marker_area(marker: str, base: float) -> float:
    """`scatter` sizes a marker by area, so the factor has to be squared."""
    return base * MARKER_SCALE.get(marker, 1.0) ** 2


def _marker_size(marker: str, base: float) -> float:
    """`plot` sizes a marker linearly."""
    return base * MARKER_SCALE.get(marker, 1.0)


def _years(date_ord) -> np.ndarray:
    return pd.to_datetime(np.asarray(date_ord) - UNIX_EPOCH_ORDINAL, unit="D").year.to_numpy()


# --- f18: the headline score table, as a figure ---------------------------


def scores_figure(scores: pd.DataFrame, mode: str = "light") -> plt.Figure:
    """f18 -- bias and centred RMSE at three averaging scales, per method."""
    with figure_style(mode) as theme:
        methods = _order(scores["method"].unique())
        colors = _method_colors(methods, mode)
        markers = _method_markers(methods)
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
            # One call per method, because a scatter takes a single marker:
            # the shape is the method's identity everywhere else in the set and
            # a bar chart is no reason to drop it.
            for position, method in zip(positions, methods, strict=True):
                ax.scatter(
                    grouped.loc[method, bias_key],
                    position,
                    color=colors[method],
                    marker=markers[method],
                    zorder=3,
                    s=_marker_area(markers[method], 26),
                )
            ax.axvline(0.0, color=theme["grid"], linewidth=1.0, zorder=1)
            ax.set_yticks(positions)
            # With sharey, blanking the labels on one axis blanks them on all of
            # them, so hide the ticks per axis instead of emptying the list.
            ax.set_yticklabels(methods)
            ax.tick_params(labelleft=(ax is axes[0]))
            ax.set_xlabel(f"groups, {label}")
            ax.set_title(label, fontsize=9, color=theme["text_secondary"])
        # The bias handle is built by hand: its marker now varies by method, so
        # no single drawn point stands for "bias" and taking the legend off the
        # axis would label one method's shape as the whole quantity.
        axes[0].legend(
            handles=[
                Patch(facecolor=theme["text_secondary"], alpha=0.45, label="centred RMSE"),
                Line2D(
                    [],
                    [],
                    marker="o",
                    markersize=5.0,
                    markerfacecolor=theme["text_secondary"],
                    markeredgecolor=theme["text_secondary"],
                    linestyle="none",
                    label="bias (one shape per method)",
                ),
            ],
            loc="lower right",
            fontsize=7.5,
            frameon=False,
        )
        caption(
            fig,
            "Bars are centred RMSE, markers are bias -- one shape per method, the same "
            "shape it keeps in f20-f26. A method can be unbiased daily and wrong per "
            "cycle, which is why three scales are shown.",
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
            "Positive inflates strong cycles relative to weak ones. The two families "
            "separate as A157 found; on the real graph the linear group sits at zero "
            "rather than above it.",
            mode,
        )
        fig.tight_layout()
        return fig


# --- f20: coverage, the load-bearing metric -------------------------------


#: S5 is the sampling-only diagnostic: detection error is switched off, so an
#: exact method scores a coverage of 1.0 there and it is not a test of
#: calibration. Averaging scenarios equally would let one degenerate world carry
#: a fifth of the headline, so the coverage figure leaves it out and says so.
COVERAGE_EXCLUDED_SCENARIOS = ("S5",)


def coverage_figure(coverage: pd.DataFrame, mode: str = "light") -> plt.Figure:
    """f20 -- empirical against nominal interval coverage, excluding S5."""
    with figure_style(mode) as theme:
        if "scenario" in coverage:
            coverage = coverage[~coverage["scenario"].isin(COVERAGE_EXCLUDED_SCENARIOS)]
        # A method that reports no interval -- the two naive floors -- has no curve
        # here, and listing it in the legend would imply one that is merely hidden.
        answered = coverage.groupby("method")["empirical"].apply(lambda b: np.isfinite(b).any())
        coverage = coverage[coverage["method"].isin(answered[answered].index)]
        methods = _order(coverage["method"].unique())
        colors = _method_colors(methods, mode)
        markers = _method_markers(methods)
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
                marker=markers[method],
                markersize=_marker_size(markers[method], 4.4),
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
            "A curve below the diagonal is an overconfident method -- but read it with "
            "the interval's width: the prior null sits near the diagonal with a 90% "
            "interval nineteen times wider than HoSc98's. S5 is excluded: with "
            "detection error off it is not a test of calibration.",
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
    """f23 -- recovered against true acuity, and against the era that confounds it.

    The right panel is the diagnosis, not decoration.  UEA21 and GRP match an
    observer's statistic against a library built once over the anchor epoch, with
    no conditioning on activity: a quiet era and a blind observer produce the same
    fingerprint, so the recovered acuity tracks how active the observer's own window
    truly was.  Reporting only the left panel states a bias without saying what
    drives it.
    """
    with figure_style(mode) as theme:
        usable = cases[np.isfinite(cases["recovered_threshold_msh"])]
        has_era = "era_mean_truth" in cases.columns and bool(
            np.isfinite(cases.get("era_mean_truth", pd.Series(dtype=float))).any()
        )
        if has_era and not usable.empty:
            fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.0))
            ax = axes[0]
        else:
            fig, ax = plt.subplots(figsize=(5.8, 5.2))
            axes = [ax]
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
            markers = _method_markers(methods)
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
                    s=_marker_area(markers[method], 34),
                    marker=markers[method],
                    color=colors[method],
                    label=method,
                    alpha=0.85,
                )
            ax.set_xlabel("true acuity threshold (MSH)")
            ax.set_ylabel("recovered acuity threshold (MSH)")
            ax.legend(fontsize=7.5, frameon=False)
            ax.set_title("(a) recovered against true", fontsize=8, loc="left")

        if has_era and not usable.empty:
            ax = axes[1]
            era = usable[np.isfinite(usable["era_mean_truth"])]
            for method in _order(era["method"].unique()):
                block = era[era["method"] == method]
                ax.scatter(
                    block["era_mean_truth"],
                    block["recovered_threshold_msh"],
                    s=_marker_area(markers[method], 34),
                    marker=markers[method],
                    color=colors[method],
                    label=method,
                    alpha=0.85,
                )
            if len(era) > 2:
                rho = (
                    era[["era_mean_truth", "recovered_threshold_msh"]]
                    .corr(method="spearman")
                    .iloc[0, 1]
                )
                ax.annotate(
                    f"Spearman $\\rho$ = {rho:+.2f}",
                    xy=(0.97, 0.95),
                    xycoords="axes fraction",
                    ha="right",
                    va="top",
                    fontsize=8,
                    color=theme["text_secondary"],
                )
            ax.set_xlabel("true mean group count over the observer's own window")
            ax.set_ylabel("recovered acuity threshold (MSH)")
            ax.set_title(
                "(b) the same points against the era, not the observer",
                fontsize=8,
                loc="left",
            )

        caption(
            fig,
            "Left: points above the diagonal are observers judged worse than they "
            "were, which A157 reports for UEA21. Right: the recovered acuity falls "
            "as the observer's own era gets more active -- the statistic cannot tell "
            "a quiet Sun from a blind observer.",
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
        markers = _method_markers(methods)
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))

        ax = axes[0]
        # Markers are staggered along the curve rather than drawn at the same
        # abscissae: CEA17 and UEA21 coincide exactly here, so a shared offset
        # would stack one shape on top of the other and hide the lower one.
        for slot, method in enumerate(methods):
            block = table[table["method"] == method].sort_values("acuity_msd")
            ax.plot(
                block["acuity_msd"],
                block["mean_difference"],
                linewidth=1.5,
                color=colors[method],
                marker=markers[method],
                markersize=_marker_size(markers[method], 4.0),
                markevery=(slot % 5, 5),
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
        for slot, method in enumerate(methods):
            block = table[table["method"] == method].sort_values("acuity_msd")
            ax.plot(
                block["acuity_msd"],
                block["rms_difference"],
                linewidth=1.5,
                color=colors[method],
                marker=markers[method],
                markersize=_marker_size(markers[method], 4.0),
                markevery=(slot % 5, 5),
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


def _draw_spread(
    ax, index: int, row, prefix: str, color: str, theme: dict, marker: str = "o"
) -> bool:
    """One method's replicate distribution: full range, IQR, median.

    Returns whether anything was drawn, so the legend can be built from the
    methods that actually appear rather than from one panel's handles.
    """
    if row[f"{prefix}_n_finite"] == 0:
        return False
    ax.vlines(
        index, row[f"{prefix}_min"], row[f"{prefix}_max"], color=color, linewidth=1.0, alpha=0.45
    )
    ax.vlines(
        index, row[f"{prefix}_q1"], row[f"{prefix}_q3"], color=color, linewidth=3.2, alpha=0.85
    )
    ax.scatter(
        index,
        row[f"{prefix}_median"],
        s=_marker_area(marker, 26),
        marker=marker,
        color=color,
        zorder=3,
        edgecolor=theme["surface"],
        linewidth=0.6,
    )
    return True


def case_study_figure(
    summary: pd.DataFrame, kinds: list[str], mode: str = "light", *, title: str = ""
) -> plt.Figure:
    """f25/f26 -- one column per named observer: acuity above, scale below.

    Two rows because only UEA21 and GRP estimate an acuity, and a figure whose
    job is to show *why* a method fails cannot leave seven of nine out.  Every
    method produces a scale factor, and the truth implies one, so the lower row
    is comparable across all of them.

    Each marker is the **median across the fifty replicates** behind that
    observer, with the interquartile range as the thick bar and the full range
    as the thin one -- so a claim about typical behaviour and a claim about the
    worst case are different marks on the same figure, and neither rests on one
    draw.  Where a method answered for only some replicates, the count of finite
    answers is printed under its slot; a method that never answers anywhere in
    the row is left unannotated, because that absence is structural rather than
    a failure.

    The legend is built from every method drawn anywhere in the figure, not from
    one panel's handles: a method can be missing from the first column and
    present in the rest, and reading the legend off that column silently drops
    it.  The dashed truth lines get an entry too, so nothing is unlabelled.
    """
    with figure_style(mode) as theme:
        fig, axes = plt.subplots(2, len(kinds), figsize=(2.35 * len(kinds), 5.4), sharey="row")
        axes = np.atleast_2d(axes)
        methods = _order(summary["method"].unique())
        colors = _method_colors(methods, mode)
        markers = _method_markers(methods)
        shown = summary[summary["kind"].isin(kinds)]
        #: Methods that answer somewhere in this row; the rest are never counted.
        answering = {
            prefix: {
                method
                for method in methods
                if (shown.loc[shown["method"] == method, f"{prefix}_n_finite"] > 0).any()
            }
            for prefix in ("acuity", "scale")
        }

        plotted: set[str] = set()
        for column, kind in enumerate(kinds):
            block = summary[summary["kind"] == kind].set_index("method")
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
            bottom.axhline(1.0, color=theme["text_secondary"], linestyle=(0, (4, 3)), linewidth=1.1)

            for ax, prefix in ((top, "acuity"), (bottom, "scale")):
                drawn = 0
                labels = []
                for index, method in enumerate(methods):
                    if method not in block.index:
                        labels.append("")
                        continue
                    row = block.loc[method]
                    if _draw_spread(ax, index, row, prefix, colors[method], theme, markers[method]):
                        plotted.add(method)
                        drawn += 1
                    finite, total = row[f"{prefix}_n_finite"], row[f"{prefix}_n_replicates"]
                    partial = finite < total and method in answering[prefix]
                    labels.append(f"{int(finite)}" if partial else "")
                if drawn == 0 and prefix == "acuity":
                    ax.annotate(
                        "no method\nestimates acuity",
                        xy=(0.5, 0.5),
                        xycoords="axes fraction",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=theme["text_secondary"],
                    )
                ax.set_xlim(-1, len(methods))
                # The slot identifies the method, so the count is drawn in the
                # recessive text colour: a yellow numeral on a light surface
                # would be the one unreadable mark in the figure.
                ax.set_xticks(
                    range(len(methods)),
                    labels=labels,
                    fontsize=5.5,
                    color=theme["text_secondary"],
                )
                ax.tick_params(axis="x", length=0, pad=1.5)
            bottom.set_yscale("log")

        axes[0, 0].set_ylabel("acuity threshold (MSH)")
        axes[1, 0].set_ylabel("recovered / implied\nscale factor")

        handles = [
            Line2D(
                [],
                [],
                marker=markers[method],
                markersize=_marker_size(markers[method], 5.5),
                linestyle="none",
                color=colors[method],
                label=method,
            )
            for method in methods
            if method in plotted
        ]
        if handles:
            handles.append(
                Line2D(
                    [],
                    [],
                    color=theme["text_secondary"],
                    linestyle=(0, (4, 3)),
                    linewidth=1.1,
                    label="truth",
                )
            )
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=min(len(handles), 5),
                fontsize=7,
                frameon=False,
                bbox_to_anchor=(0.5, -0.06),
            )
        if title:
            fig.suptitle(title, fontsize=9, color=theme["text_secondary"])
        caption(
            fig,
            "Marker is the median over 50 replicates -- one shape per method, as in the "
            "legend -- thick bar the interquartile range, thin bar "
            "the full range; dashed lines are the truth. A small number under a method counts "
            "the replicates it answered at all, where that was not all 50.",
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


# --- f28: the overlap graph drawn as a network ----------------------------


def network_figure(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    cut: pd.DataFrame,
    bridge_pairs: list[tuple[int, int]] | None = None,
    mode: str = "light",
    *,
    min_side: int = 10,
) -> plt.Figure:
    """f28 -- the overlap graph itself: who links to whom, and where it is severed.

    Parameters
    ----------
    nodes
        ``station, mid_year, resistance`` -- one row per station, ``resistance``
        infinite for stations the anchor cannot reach.
    edges
        ``station_a, station_b, overlap_days`` from ``OverlapGraph.edge_frame``.
    cut
        ``year, n_edges, overlap_days`` from ``graph.temporal_cut_profile``.
    bridge_pairs
        Edges whose removal disconnects the graph, drawn in the accent colour.
    min_side
        The right panel shows only years with at least this many stations on each
        side of the cut, so the trivially empty cuts at the ends of the record are
        not drawn as if they were bottlenecks.
    """
    from matplotlib.collections import LineCollection

    with figure_style(mode) as theme:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), width_ratios=[1.35, 1])

        finite = nodes[np.isfinite(nodes["resistance"])]
        ceiling = float(finite["resistance"].max()) if not finite.empty else 1.0
        pinned = ceiling * 4.0
        y = np.where(np.isfinite(nodes["resistance"]), nodes["resistance"], pinned)
        position = {
            int(s): (float(x), float(v))
            for s, x, v in zip(nodes["station"], nodes["mid_year"], y, strict=True)
        }

        ax = axes[0]
        bridge_set = {tuple(sorted(pair)) for pair in (bridge_pairs or [])}
        plain, marked = [], []
        for a, b in zip(edges["station_a"], edges["station_b"], strict=True):
            segment = (position[int(a)], position[int(b)])
            (marked if tuple(sorted((int(a), int(b)))) in bridge_set else plain).append(segment)
        ax.add_collection(
            LineCollection(plain, colors=theme["text_secondary"], linewidths=0.25, alpha=0.10)
        )
        if marked:
            ax.add_collection(
                LineCollection(
                    marked, colors=theme["categorical"][3], linewidths=1.1, alpha=0.9, zorder=2
                )
            )

        reachable = np.isfinite(nodes["resistance"]).to_numpy()
        ax.scatter(
            nodes["mid_year"][reachable],
            y[reachable],
            s=13,
            color=theme["categorical"][0],
            zorder=3,
            label=f"reachable ({int(reachable.sum())})",
        )
        ax.scatter(
            nodes["mid_year"][~reachable],
            y[~reachable],
            s=20,
            marker="^",
            color=theme["categorical"][3],
            zorder=3,
            label=f"unreachable ({int((~reachable).sum())})",
        )
        is_anchor = (nodes["resistance"].to_numpy() == 0.0) & reachable
        ax.scatter(
            nodes["mid_year"][is_anchor],
            y[is_anchor],
            s=110,
            marker="*",
            color=theme["categorical"][1],
            edgecolor=theme["surface"],
            linewidth=0.6,
            zorder=4,
            label="anchor (reference survey)",
        )
        ax.axhline(pinned / 2, color=theme["grid"], linewidth=0.8, linestyle=(0, (4, 3)))
        if marked:
            ax.plot(
                [],
                [],
                color=theme["categorical"][3],
                linewidth=1.1,
                label=f"bridge ({len(marked)})",
            )
        ax.plot(
            [],
            [],
            color=theme["text_secondary"],
            linewidth=0.8,
            alpha=0.5,
            label=f"overlap edge ({len(edges)})",
        )
        ax.set_yscale("symlog", linthresh=1e-5)
        ax.set_ylim(0, pinned * 1.8)
        ax.set_xlabel("observer's mid-career year")
        ax.set_ylabel("effective resistance to the anchor")
        ax.legend(fontsize=7.0, frameon=False, loc="lower left")

        ax = axes[1]
        inner = cut[(cut["n_before"] >= min_side) & (cut["n_after"] >= min_side)]
        severed = inner["n_edges"].to_numpy() == 0
        ax.fill_between(
            inner["year"], inner["n_edges"], step="mid", color=theme["categorical"][2], alpha=0.85
        )
        for start, stop in _spans(inner["year"].to_numpy(), severed):
            ax.axvspan(start - 0.5, stop + 0.5, color=theme["categorical"][3], alpha=0.22, lw=0)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(0, inner["n_edges"].max() * 2)
        ax.set_xlim(inner["year"].min(), inner["year"].max())
        ax.set_xlabel("year")
        ax.set_ylabel("edges crossing the year's cut")
        ax.plot(
            [],
            [],
            color=theme["categorical"][3],
            alpha=0.35,
            linewidth=6,
            label=f"severed ({int(severed.sum())} years)",
        )
        ax.legend(fontsize=7.0, frameon=False, loc="upper left")

        caption(
            fig,
            "Every observer reaches the anchor only through this graph, and in the shaded "
            "years nothing crosses it at all.",
            mode,
        )
        fig.tight_layout()
        return fig


# --- f41, f42: the error through time -------------------------------------

#: The floor every method is measured against through time.  `naive-median` rather
#: than `naive-mean` because it is the harder of the two in section 4's paired test.
TIME_FLOOR = "naive-median"

#: One solar cycle, as a smoothing window on a yearly series.  Narrower and the
#: curves are cycle-phase noise; wider and an era boundary smears across itself.
CYCLE_YEARS = 11


def _smooth(frame: pd.DataFrame, column: str, *, years=None) -> pd.Series:
    """A cycle-long centred mean on a year-indexed series, gaps left as gaps.

    Reindexed onto the full year range first, so a method that reports nothing for
    a decade gets NaN there rather than having the window reach across the hole and
    draw a line through an era it never entered.
    """
    if years is None:
        years = np.arange(int(frame["year"].min()), int(frame["year"].max()) + 1)
    series = frame.set_index("year")[column].reindex(years)
    smoothed = series.rolling(CYCLE_YEARS, center=True, min_periods=CYCLE_YEARS // 2).mean()
    smoothed[series.isna()] = np.nan
    return smoothed


def _window_rmse(frame: pd.DataFrame, *, years=None) -> pd.Series:
    """RMSE over a cycle-long window: the squared error is pooled, then rooted.

    Not the rolling mean of the yearly RMSE, and the difference is not cosmetic.
    The floor's pooled RMSE passes through ~0.0008 in 1878 and ~0.0002 in 1968 --
    years where the median of the raw counts happens to sit on the truth in every
    world at once -- so a year-by-year ratio against it spikes past 1000 and buries
    the comparison the panel exists to make.  Pooling the squared error across the
    window first removes the singularity without clipping anything away.
    """
    return np.sqrt(_smooth(frame, "mse", years=years))


def _era_bands(ax, mode: str, *, label: bool = False, first: int = 1610) -> None:
    """The nine eras as alternating faint bands, so a boundary is read, not counted."""
    theme = palette(mode)
    for index, (name, start, end) in enumerate(ERAS):
        if end < first:
            continue
        if index % 2:
            ax.axvspan(max(start, first), end, color=theme["grid"], alpha=0.35, zorder=0, lw=0)
        if name == "anchor-epoch":
            # The era the whole figure turns on: the only window the reference
            # observer actually sees (ADR-009, RGO cycles 15-19).
            ax.axvspan(start, end, color=theme["categorical"][3], alpha=0.14, zorder=0, lw=0)
        if label:
            ax.annotate(
                name.replace("-", "\n"),
                xy=((max(start, first) + end) / 2, 1.02),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="bottom",
                fontsize=5.6,
                color=theme["text_secondary"],
                linespacing=0.95,
            )


def _winner_by_year(pooled: pd.DataFrame) -> pd.DataFrame:
    """The lowest-RMSE real method each year, and whether it beats the floor.

    Floors are excluded from the ranking because they are the bar, not entrants;
    ``beats_floor`` is what keeps the ribbon honest, since a method can lead the
    seven in an era where all seven are worse than leaving the counts alone.
    """
    real = pooled[~pooled["method"].map(is_floor) & np.isfinite(pooled["rmse"])]
    if real.empty:
        return pd.DataFrame(columns=["year", "method", "rmse", "floor", "beats_floor"])
    best = real.loc[real.groupby("year")["rmse"].idxmin(), ["year", "method", "rmse"]]
    floor = (
        pooled[pooled["method"] == TIME_FLOOR]
        .set_index("year")["rmse"]
        .rename("floor")
        .reset_index()
    )
    out = best.merge(floor, on="year", how="left")
    out["beats_floor"] = out["rmse"] < out["floor"]
    return out.sort_values("year").reset_index(drop=True)


def _draw_ribbon(ax, winners: pd.DataFrame, colors: dict[str, str], mode: str) -> None:
    """One bar per year in the winner's own colour, hatched where no one beats the floor."""
    theme = palette(mode)
    for _, row in winners.iterrows():
        ax.axvspan(
            row["year"] - 0.5,
            row["year"] + 0.5,
            ymin=0.42,
            ymax=1.0,
            color=colors.get(row["method"], theme["text_secondary"]),
            lw=0,
        )
        ax.axvspan(
            row["year"] - 0.5,
            row["year"] + 0.5,
            ymin=0.0,
            ymax=0.30,
            color=theme["categorical"][0] if row["beats_floor"] else theme["grid"],
            lw=0,
        )


def error_through_time_figure(
    pooled: pd.DataFrame, mode: str = "light", *, first_year: int = 1645
) -> plt.Figure:
    """f41 -- where in the record each method's error actually sits.

    Section 4's table is pooled over the whole record and says HoSc98 wins.  That
    is true and it hides the mechanism: the methods that calibrate through overlap
    are only good near the epoch the reference observer sees, and the pooled number
    averages that over four centuries.
    """
    with figure_style(mode) as theme:
        methods = _order(pooled["method"].unique())
        colors = _method_colors(methods, mode)
        # Five rows, the last one blank and holding the legend.  The house style
        # turns constrained layout on globally, so tight_layout and subplots_adjust
        # are both ignored at draw time, and an "outside" figure legend lands in the
        # same strip as the caption.  A real axes is the one place the layout engine
        # is guaranteed to reserve.
        fig, axes = plt.subplots(
            5,
            1,
            figsize=(9.2, 8.6),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 2.1, 1.7, 0.42, 0.30], "hspace": 0.12},
        )
        top, absolute, relative, ribbon, legend_row = axes
        legend_row.set_axis_off()

        # (a) where we are: the activity level the errors sit on top of.
        level = pooled[pooled["method"] == TIME_FLOOR]
        _era_bands(top, mode, label=True, first=first_year)
        top.plot(
            level["year"],
            _smooth(level, "truth").reindex(level["year"]).to_numpy(),
            color=theme["text_primary"],
            linewidth=1.3,
        )
        top.set_yscale("log")
        top.set_ylabel("true groups/day")

        # (b) the error itself, in groups.
        _era_bands(absolute, mode, first=first_year)
        span = np.arange(first_year, 2027)
        floor_rmse = _window_rmse(
            pooled[pooled["method"] == TIME_FLOOR].sort_values("year"), years=span
        )
        handles = []
        for method in methods:
            block = pooled[pooled["method"] == method].sort_values("year")
            if block.empty:
                continue
            smoothed = _window_rmse(block, years=span)
            absolute.plot(
                smoothed.index,
                smoothed.to_numpy(),
                color=colors[method],
                linestyle=_linestyle(method),
                linewidth=1.9 if not is_floor(method) else 1.2,
                alpha=1.0 if not is_floor(method) else 0.85,
            )
            handles.append(
                Line2D(
                    [],
                    [],
                    color=colors[method],
                    linestyle=_linestyle(method),
                    marker="" if is_floor(method) else _method_markers(methods)[method],
                    markersize=4.6,
                    label=method,
                )
            )
            if is_floor(method):
                continue
            # A marker at the method's own best era, so the shape channel is present
            # on a line chart too and the legend is not the only way in.
            best = smoothed.idxmin()
            if pd.notna(best):
                absolute.plot(
                    best,
                    smoothed.loc[best],
                    marker=_method_markers(methods)[method],
                    markersize=_marker_size(_method_markers(methods)[method], 5.0),
                    color=colors[method],
                    linestyle="none",
                    zorder=4,
                )
        absolute.set_yscale("log")
        absolute.set_ylabel("pooled RMSE (groups)")

        # (c) the same error against the floor, which removes the activity level.
        _era_bands(relative, mode, first=first_year)
        for method in methods:
            if method == TIME_FLOOR:
                continue
            block = pooled[pooled["method"] == method].sort_values("year")
            if block.empty:
                continue
            ratio = _window_rmse(block, years=span) / floor_rmse
            relative.plot(
                ratio.index,
                ratio.to_numpy(),
                color=colors[method],
                linestyle=_linestyle(method),
                linewidth=1.9 if not is_floor(method) else 1.2,
                alpha=1.0 if not is_floor(method) else 0.85,
            )
        relative.axhline(
            1.0, color=theme["text_secondary"], linestyle=(0, (4, 3)), linewidth=1.0, zorder=1
        )
        relative.annotate(
            "below this line a method beats leaving the counts alone",
            xy=(0.012, 0.035),
            xycoords="axes fraction",
            ha="left",
            va="bottom",
            fontsize=7.2,
            color=theme["text_secondary"],
        )
        relative.set_yscale("log")
        relative.set_ylabel("RMSE / naive-median")

        # (d) who wins, and whether winning means anything.
        winners = _winner_by_year(pooled)
        _draw_ribbon(ribbon, winners, colors, mode)
        ribbon.set_yticks([0.15, 0.71])
        ribbon.set_yticklabels(["beats floor", "lowest RMSE"], fontsize=7)
        ribbon.set_ylim(0, 1)
        # sharex hides the tick labels on every row but the last, and the last row
        # is the blank legend strip, so the ribbon has to ask for its own back.
        ribbon.tick_params(labelbottom=True)
        ribbon.set_xlabel("year")
        for spine in ribbon.spines.values():
            spine.set_visible(False)
        ribbon.tick_params(length=0)

        absolute.set_xlim(first_year, 2026)
        legend_row.legend(
            handles=handles,
            ncol=6,
            fontsize=7,
            frameon=False,
            loc="center",
            borderaxespad=0.0,
        )
        caption(
            fig,
            "Scenario-balanced over S2/S3/S4 and pooled over one solar cycle; a line stops\n"
            "where the method reports nothing, because absence is not zero error. The tinted\n"
            "band is the anchor epoch -- the only window the reference observer sees -- and it\n"
            "is the one era where a published method beats the median of the raw counts.",
            mode,
        )
        return fig


def bias_and_era_coverage_figure(
    pooled: pd.DataFrame, era_table: pd.DataFrame, mode: str = "light", *, first_year: int = 1645
) -> plt.Figure:
    """f42 -- the sign of the error through time, and the intervals per era.

    f19 reports one amplitude slope per method over the whole record; the left panel
    is where in time that slope is earned.  The right panel is the era-stratified
    coverage the Step 2 plan's criterion A5 asked for and the scorer never produced.
    """
    with figure_style(mode) as theme:
        methods = _order(pooled["method"].unique())
        colors = _method_colors(methods, mode)
        markers = _method_markers(methods)
        fig, (bias_ax, cover_ax) = plt.subplots(
            1, 2, figsize=(11.4, 4.3), gridspec_kw={"width_ratios": [1.25, 1.0]}
        )

        _era_bands(bias_ax, mode, first=first_year)
        for method in methods:
            block = pooled[pooled["method"] == method].sort_values("year")
            if block.empty:
                continue
            smoothed = _smooth(block, "bias")
            bias_ax.plot(
                smoothed.index,
                smoothed.to_numpy(),
                color=colors[method],
                linestyle=_linestyle(method),
                linewidth=1.8 if not is_floor(method) else 1.1,
                alpha=1.0 if not is_floor(method) else 0.8,
                label=method,
            )
        bias_ax.axhline(0.0, color=theme["text_secondary"], linewidth=1.0, zorder=2)
        bias_ax.set_xlim(first_year, 2026)
        bias_ax.set_xlabel("year")
        bias_ax.set_ylabel("pooled bias (groups)")
        bias_ax.set_title(
            "sign of the error through time", fontsize=9, color=theme["text_secondary"]
        )
        # Lower left: the upper left is where UEA21 and GRP spend the whole record.
        bias_ax.legend(ncol=3, fontsize=6.4, frameon=False, loc="lower left")

        # Coverage per era, at the spec's headline level.  Methods that report no
        # interval at all (the naive floors) have nothing to draw and are left out.
        pooled_eras = era_table[era_table["scenario"] == "pooled"]
        present = [e for e in ERA_ORDER if e in set(pooled_eras["era"])]
        positions = {era: index for index, era in enumerate(present)}
        drawn = [
            m
            for m in methods
            if np.isfinite(pooled_eras.loc[pooled_eras["method"] == m, "coverage_50"]).any()
        ]
        for method in drawn:
            block = pooled_eras[pooled_eras["method"] == method]
            x = [positions[e] for e in block["era"] if e in positions]
            y = [
                v for e, v in zip(block["era"], block["coverage_50"], strict=True) if e in positions
            ]
            cover_ax.plot(
                x,
                y,
                marker=markers[method],
                markersize=_marker_size(markers[method], 5.4),
                color=colors[method],
                linestyle=_linestyle(method) if is_floor(method) else "-",
                linewidth=1.4,
                alpha=1.0 if not is_floor(method) else 0.8,
                label=method,
            )
        annotate_target(cover_ax, 0.5, mode, label="nominal 50%")
        cover_ax.set_xticks(list(positions.values()))
        cover_ax.set_xticklabels(
            [e.replace("-", " ") for e in present], fontsize=6.6, rotation=32, ha="right"
        )
        cover_ax.set_ylim(-0.03, 1.03)
        cover_ax.set_ylabel("empirical coverage at nominal 50%")
        cover_ax.set_title(
            "are the intervals honest, era by era?", fontsize=9, color=theme["text_secondary"]
        )

        caption(
            fig,
            "Left: pooled yearly bias, cycle-smoothed -- a method can be centred over the\n"
            "record and wrong in every era of it. Right: coverage at nominal 50% per era, the\n"
            "stratification criterion A5 asked for; a point is missing where the method\n"
            "reaches no year of that era.",
            mode,
        )
        return fig


def _spans(values: np.ndarray, flag: np.ndarray) -> list[tuple[float, float]]:
    """Contiguous runs of ``True`` in ``flag``, as ``(first, last)`` of ``values``."""
    spans, start = [], None
    for index, active in enumerate(flag):
        if active and start is None:
            start = index
        elif not active and start is not None:
            spans.append((values[start], values[index - 1]))
            start = None
    if start is not None:
        spans.append((values[start], values[-1]))
    return spans


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
    """f22 -- the spread a method's own free choices produce.

    The bar is the **within-world** range: for each world, the spread across the
    fifteen settings, averaged over worlds, drawn about the method's mean.  Pooling
    worlds and settings into one min-max bar instead -- which this did until
    2026-11-24 -- measures the two together and labels the result structural.  For
    UEA21 and GRP that is not a quibble but the whole bar: neither method touches
    the overlap graph, so their within-world range is *exactly zero* and a pooled
    bar reports world-to-world variation as if the knobs had caused it.

    The default-setting marker is taken from the **same worlds** the ensemble ran
    on.  An earlier version drew the ensemble's S2 bars against the default run's S1
    bias, which compared two different scenarios and made every method look as
    though its structural choices had moved it.
    """
    with figure_style(mode) as theme:
        methods = _order(structural["method"].unique())
        colors = _method_colors(methods, mode)
        markers = _method_markers(methods)

        matched = scores
        if not scores.empty and {"scenario", "seed"} <= set(scores.columns):
            matched = scores[
                scores["scenario"].isin(set(structural["scenario"]))
                & scores["seed"].isin(set(structural["seed"]))
            ]

        fig, ax = plt.subplots(figsize=(7.4, 3.6))
        positions = np.arange(len(methods))
        drew_default = False
        for index, method in enumerate(methods):
            rows = structural[structural["method"] == method].dropna(subset=["bias_yearly"])
            block = rows["bias_yearly"]
            if block.empty:
                continue
            per_world = rows.groupby("seed")["bias_yearly"].agg(lambda x: x.max() - x.min())
            half = float(per_world.mean()) / 2.0
            centre = float(block.mean())
            ax.plot(
                [centre - half, centre + half],
                [index, index],
                linewidth=3.0,
                color=colors[method],
                alpha=0.7,
                solid_capstyle="round",
            )
            # Drawn larger than elsewhere, and with a surface-coloured rim: the
            # default-setting cross often lands on the mean, and at f18's size
            # the cross would swallow the shape that says which method this is.
            ax.scatter(
                block.mean(),
                index,
                s=_marker_area(markers[method], 46),
                marker=markers[method],
                color=colors[method],
                edgecolor=theme["surface"],
                linewidth=0.6,
                zorder=3,
            )
            default = matched[matched["method"] == method]["bias_yearly"].dropna()
            if not default.empty:
                ax.scatter(
                    default.mean(),
                    index,
                    s=34,
                    marker="x",
                    color=theme["text_secondary"],
                    zorder=4,
                )
                drew_default = True

        ax.axvline(0.0, color=theme["text_secondary"], linewidth=1.0)
        ax.set_yticks(positions)
        ax.set_yticklabels(methods)
        ax.set_xlabel("yearly bias (groups)")
        if not drew_default:
            ax.annotate(
                "no default-setting run on these worlds yet",
                xy=(0.5, 1.02),
                xycoords="axes fraction",
                ha="center",
                fontsize=7.5,
                color=theme["text_secondary"],
            )
        caption(
            fig,
            "Bars are the mean within-world spread across the method's own free choices "
            "(graph tolerance, minimum overlap, minimum distinct counts); the marker is "
            "the method's mean and the grey cross the default, on the same worlds. "
            "UEA21 and GRP need no graph, so their bar is zero by construction.",
            mode,
        )
        fig.tight_layout()
        return fig


def _year_of(date_ord: int) -> int:
    return dt.date.fromordinal(int(date_ord)).year


#: Shown in the order the comparison reads: the two long reconstructions a reader
#: is most likely to know, then the rest, then the two Maunder-window-only studies
#: whose disagreement is the point of the figure.
SERIES_ORDER = (
    "hoyt_schatten_1998",
    "velasco_herrera_2024_hoyt_update",
    "svalgaard_schatten_2016_backbone",
    "usoskin_series",
    "chatzistergos_2017",
    "velasco_herrera_2024",
    "vaquero",
    "zolotova",
)

SERIES_LABEL = {
    "hoyt_schatten_1998": "Hoyt & Schatten 1998",
    "velasco_herrera_2024_hoyt_update": "Velasco Herrera 2024 (H&S update)",
    "svalgaard_schatten_2016_backbone": "Svalgaard & Schatten 2016",
    "usoskin_series": "Usoskin et al. (ADF family)",
    "chatzistergos_2017": "Chatzistergos et al. 2017",
    "velasco_herrera_2024": "Velasco Herrera et al. 2024",
    "vaquero": "Vaquero et al.",
    "zolotova": "Zolotova & Ponyavin",
}

#: Marker per series, so the identity survives greyscale and the two H&S variants
#: -- which lie on top of each other by construction -- stay separable.
SERIES_MARKER = {
    "hoyt_schatten_1998": "o",
    "velasco_herrera_2024_hoyt_update": "s",
    "svalgaard_schatten_2016_backbone": "D",
    "usoskin_series": "^",
    "chatzistergos_2017": "v",
    "velasco_herrera_2024": "P",
    "vaquero": "X",
    "zolotova": "*",
}

#: The two windows the comparison is made in.  The modern one is where all six
#: full-record series overlap; the Maunder one is ADR-009's grand-minimum window and
#: the one Step 2 section 6 scores against.
COMPARISON_WINDOWS = (("Maunder Minimum", 1645, 1715), ("modern overlap", 1749, 1995))


def series_window_means(table: pd.DataFrame) -> pd.DataFrame:
    """Each series' mean daily group count in each comparison window.

    The numbers the comparison rests on, returned rather than drawn so the figure
    and the report quote the same arithmetic.
    """
    rows = []
    year = np.floor(table["year"]).astype(int)
    for label, first, last in COMPARISON_WINDOWS:
        inside = table[(year >= first) & (year <= last)].dropna(subset=["gn"])
        for name, block in inside.groupby("series"):
            rows.append(
                {
                    "window": label,
                    "first_year": first,
                    "last_year": last,
                    "series": name,
                    "mean_gn": float(block["gn"].mean()),
                    "n_years": int(len(block)),
                }
            )
    return pd.DataFrame(rows)


def published_series_figure(table: pd.DataFrame, mode: str = "light") -> plt.Figure:
    """f27 -- the eight published series compared, on one scale.

    Not a record of which arrived when: all eight are drawn alike.  What the figure
    is for is the *shape* of their disagreement -- a quarter over the modern
    overlap, a factor of sixty-six in the Maunder Minimum -- because that is the
    quantity this benchmark's section 6 explains.
    """
    with figure_style(mode) as theme:
        fig, axes = plt.subplots(3, 1, figsize=(10.0, 8.2), height_ratios=[3, 3, 1.5])
        colors = {name: theme["categorical"][i % 8] for i, name in enumerate(SERIES_ORDER)}
        present = [s for s in SERIES_ORDER if s in set(table["series"])]

        # --- (a) every series, drawn alike ---------------------------------
        ax = axes[0]
        ax.axvspan(1645, 1715, color=theme["text_secondary"], alpha=0.10, zorder=0)
        for name in present:
            block = table[table["series"] == name].dropna(subset=["gn"]).sort_values("year")
            # Annual values are a hairball at this width and, worse, a *misleading*
            # one: eight cycle-resolved series on one axis hide exactly the slow
            # disagreement the figure is about.  An 11-year centred mean is one
            # cycle, so what is left is the secular level each series claims.
            smooth = block["gn"].rolling(11, center=True, min_periods=4).mean()
            ax.plot(
                block["year"],
                smooth.clip(lower=0.02),
                color=colors[name],
                linewidth=1.3,
                label=SERIES_LABEL[name],
                zorder=2,
            )
        ax.set_yscale("log")
        # Headroom for the legend, so it never sits on a curve or on the shaded band.
        ax.set_ylim(0.018, 90.0)
        ax.set_ylabel("mean daily group count\n(11-year centred mean)")
        ax.set_xlim(1600, 2030)
        ax.legend(loc="upper center", fontsize=6.5, frameon=False, ncol=4)
        ax.set_title(
            "(a) eight published series, smoothed over one cycle, log axis; "
            "shaded: the Maunder Minimum 1645-1715",
            fontsize=8,
            loc="left",
        )

        # --- (b) the comparison itself -------------------------------------
        ax = axes[1]
        means = series_window_means(table)
        rows = [s for s in present if s in set(means["series"])]
        positions = {name: len(rows) - 1 - i for i, name in enumerate(rows)}
        for window, style in zip([w[0] for w in COMPARISON_WINDOWS], ["full", "open"], strict=True):
            block = means[means["window"] == window]
            for _, row in block.iterrows():
                name = row["series"]
                if name not in positions:
                    continue
                ax.scatter(
                    max(row["mean_gn"], 1e-3),
                    positions[name],
                    marker=SERIES_MARKER[name],
                    s=110 * MARKER_SCALE.get(SERIES_MARKER[name], 1.0),
                    facecolor=colors[name] if style == "full" else "none",
                    edgecolor=colors[name],
                    linewidth=1.3,
                    zorder=3,
                )
        for name in rows:
            pair = means[means["series"] == name].set_index("window")["mean_gn"]
            if len(pair) == 2:
                ax.plot(
                    [max(pair.min(), 1e-3), pair.max()],
                    [positions[name]] * 2,
                    color=colors[name],
                    linewidth=0.9,
                    alpha=0.6,
                    zorder=2,
                )
        maunder = means[means["window"] == "Maunder Minimum"]["mean_gn"]
        if len(maunder) > 1 and maunder.min() > 0:
            ax.annotate(
                f"Maunder spread: {maunder.max() / maunder.min():.0f}$\\times$\n"
                f"({maunder.min():.2f} to {maunder.max():.2f} groups/day)",
                xy=(0.02, 0.06),
                xycoords="axes fraction",
                ha="left",
                fontsize=7.5,
                color=theme["text_secondary"],
            )
        ax.set_xscale("log")
        ax.set_yticks(list(positions.values()))
        ax.set_yticklabels([SERIES_LABEL[n] for n in rows], fontsize=7)
        ax.set_xlabel("mean daily group count in the window (log scale)")
        ax.set_title(
            "(b) filled: Maunder Minimum 1645-1715   open: modern overlap 1749-1995",
            fontsize=8,
            loc="left",
        )

        # --- (c) the scale check that licenses one axis --------------------
        ax = axes[2]
        old_hs = table[table["series"] == "hoyt_schatten_1998"].copy()
        new_hs = table[table["series"] == "velasco_herrera_2024_hoyt_update"].copy()
        old_hs["key"] = np.floor(old_hs["year"]).astype(int)
        new_hs["key"] = np.floor(new_hs["year"]).astype(int)
        both = old_hs.merge(new_hs, on="key", suffixes=("_a", "_b"))
        ax.axhline(0.0, color=theme["grid"], linewidth=0.8)
        ax.plot(
            both["key"],
            (both["gn_b"] - both["gn_a"]) * 1e3,
            color=colors["velasco_herrera_2024_hoyt_update"],
            linewidth=0.9,
        )
        ax.set_ylabel("(update / 12.08)\n$-$ 1998  [$10^{-3}$]", fontsize=7)
        ax.set_xlabel("year")
        ax.set_xlim(1600, 2030)
        ax.set_title(
            "(c) the 12.08 check (D24): the two Hoyt & Schatten files agree to "
            "under a thousandth of a group",
            fontsize=8,
            loc="left",
        )

        caption(
            fig,
            "All eight series carry the same units, so one axis compares them: they "
            "agree to about a quarter where they overlap in the modern era and diverge "
            "by more than an order of magnitude in the Maunder Minimum.",
            mode,
        )
        fig.tight_layout()
        return fig
