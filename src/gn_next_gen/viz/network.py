"""Step 3's figures, f29-f40.

Figure-first, per ``CLAUDE.md``: the results doc has to be readable without
opening a file, so every claim names a figure and every figure carries a caption
that states the claim rather than labelling the axes again.

One caption rule specific to this step and repeated in several of these panels:
**effective resistance is the random part of the scale uncertainty and nothing
else.**  Ranked idea 7's own cost column says so -- "Captures random error only;
systematic bias adds coherently" -- so a resistance panel must never be read as
the 17th century's total scale uncertainty.  What probes the systematic part is
the loop closure in f30 and the two-kernel disagreement in f38.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from gn_next_gen.graph.overlap import decimal_year
from gn_next_gen.viz.style import STATUS, caption, figure_style

#: Resistance spans many decades, so every resistance axis is symlog with one
#: shared threshold -- comparability across panels matters more than filling each.
RESISTANCE_LINTHRESH = 1e-6


def _years(ordinals) -> np.ndarray:
    return np.asarray(decimal_year(np.asarray(ordinals, dtype=float)))


def _unreachable_marker(ax, values: np.ndarray, finite_max: float, theme, label: str):
    if len(values) == 0:
        return
    ax.scatter(
        values,
        np.full(len(values), finite_max * 3.0),
        s=18,
        marker="^",
        color=STATUS["critical"],
        label=label,
        zorder=3,
    )


def resistance_figure(
    diagnostics: pd.DataFrame,
    by_year: dict[str, pd.DataFrame],
    anchor_pairs: pd.DataFrame,
    mode: str = "light",
) -> plt.Figure:
    """f29 -- how far each observer, and each year, sits from each anchor."""
    with figure_style(mode) as theme:
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
        names = [
            c[len("resistance_") :] for c in diagnostics.columns if c.startswith("resistance_")
        ]

        ax = axes[0]
        finite_max = 1.0
        for slot, name in enumerate(names):
            column = f"resistance_{name}"
            values = diagnostics[column].to_numpy()
            good = np.isfinite(values)
            if good.any():
                finite_max = max(finite_max, float(np.nanmax(values[good])))
            ax.scatter(
                diagnostics.loc[good, "first_year"],
                values[good],
                s=12,
                alpha=0.7,
                color=theme["categorical"][slot],
                label=f"{name} ({int(good.sum())} reachable)",
            )
        unreachable = diagnostics[~diagnostics["anchored"]]
        _unreachable_marker(
            ax,
            unreachable["first_year"].to_numpy(),
            finite_max,
            theme,
            f"no path to any anchor ({len(unreachable)})",
        )
        ax.set_yscale("symlog", linthresh=RESISTANCE_LINTHRESH)
        ax.set_xlabel("observer's first year")
        ax.set_ylabel("effective resistance to the anchor")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        ax.set_title("per observer", fontsize=9)

        ax = axes[1]
        for slot, (name, table) in enumerate(sorted(by_year.items())):
            inside = table[table["anchored"]]
            ax.plot(
                inside["year"],
                inside["resistance"],
                lw=1.2,
                color=theme["categorical"][slot],
                label=name,
            )
            severed = table[~table["anchored"] & (table["n_active"] > 0)]
            if not severed.empty:
                ax.scatter(
                    severed["year"],
                    np.full(len(severed), RESISTANCE_LINTHRESH / 3.0),
                    s=5,
                    marker="x",
                    color=STATUS["critical"],
                )
        ax.set_yscale("symlog", linthresh=RESISTANCE_LINTHRESH)
        ax.set_xlabel("year")
        ax.set_ylabel("resistance from the year to the anchor")
        ax.legend(fontsize=7.5, frameon=False)
        ax.set_title("per year (x = severed)", fontsize=9)

        ax = axes[2]
        if anchor_pairs.empty:
            ax.text(0.5, 0.5, "no anchor pair", ha="center", va="center", fontsize=8)
            ax.set_axis_off()
        else:
            labels = [f"{row.anchor_a}\n{row.anchor_b}" for row in anchor_pairs.itertuples()]
            values = anchor_pairs["resistance"].to_numpy(dtype=float)
            drawn = np.where(np.isfinite(values), values, RESISTANCE_LINTHRESH / 3.0)
            colors = [
                theme["categorical"][2] if ok else STATUS["critical"]
                for ok in anchor_pairs["connected"]
            ]
            ax.bar(labels, drawn, color=colors)
            for index, (value, ok) in enumerate(
                zip(values, anchor_pairs["connected"], strict=True)
            ):
                ax.text(
                    index,
                    drawn[index],
                    f"{value:.2e}" if ok else "severed",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
            ax.set_yscale("symlog", linthresh=RESISTANCE_LINTHRESH)
            ax.set_ylabel("resistance between anchors")
        ax.set_title("between the two anchorings", fontsize=9)

        caption(
            fig,
            "Resistance is the random part of the scale uncertainty only; systematic "
            "per-link bias adds coherently and is not in these panels.",
            mode,
        )
    return fig


def reach_figure(
    diagnostics: pd.DataFrame,
    cut: pd.DataFrame,
    by_year: pd.DataFrame,
    mode: str = "light",
) -> plt.Figure:
    """f39 -- components, bridges, and the years no chain crosses."""
    with figure_style(mode) as theme:
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))

        ax = axes[0]
        sizes = diagnostics.groupby("component").size().sort_values(ascending=False)
        ax.bar(
            np.arange(len(sizes)),
            sizes.to_numpy(),
            color=theme["categorical"][0],
        )
        ax.set_yscale("log")
        ax.set_xlabel("connected component (largest first)")
        ax.set_ylabel("observers")
        ax.set_title(f"{len(sizes)} components, largest {int(sizes.iloc[0])}", fontsize=9)

        ax = axes[1]
        anchored = diagnostics["anchored"].sum()
        floating = len(diagnostics) - anchored
        on_bridge = diagnostics["on_bridge"].sum()
        ax.bar(
            ["anchored", "fingerprint\nonly", "on a bridge"],
            [anchored, floating, on_bridge],
            color=[
                theme["categorical"][2],
                STATUS["critical"],
                theme["categorical"][3],
            ],
        )
        for index, value in enumerate([anchored, floating, on_bridge]):
            ax.text(index, value, f"{int(value)}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("observers")
        ax.set_title("who the anchor can reach", fontsize=9)

        ax = axes[2]
        usable = cut[(cut["n_before"] >= 10) & (cut["n_after"] >= 10)]
        ax.fill_between(
            usable["year"], usable["n_edges"], step="mid", color=theme["categorical"][0], alpha=0.8
        )
        severed = usable[usable["n_edges"] == 0]
        if not severed.empty:
            ax.scatter(
                severed["year"],
                np.zeros(len(severed)),
                s=10,
                marker="x",
                color=STATUS["critical"],
                label=f"{len(severed)} severed years",
                zorder=3,
            )
            ax.legend(fontsize=7.5, frameon=False)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("year")
        ax.set_ylabel("edges crossing the year")
        ax.set_title("where the chain is severed", fontsize=9)

        caption(
            fig,
            "A floating component has one exact Laplacian null mode, so its level "
            "rests entirely on the smoothness prior rather than on any overlap.",
            mode,
        )
    return fig


def loop_closure_figure(
    closure: pd.DataFrame,
    comparison: pd.DataFrame | None = None,
    mode: str = "light",
) -> plt.Figure:
    """f30 -- path consistency, and whether concurrent beats chained with distance."""
    with figure_style(mode) as theme:
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))

        ax = axes[0]
        if closure.empty:
            ax.text(
                0.5,
                0.5,
                "no closed loop with an estimate on every edge",
                ha="center",
                va="center",
                fontsize=8,
            )
            ax.set_axis_off()
        else:
            ax.hist(
                closure["closure"],
                bins=40,
                color=theme["categorical"][0],
            )
            ax.axvline(0.0, color=theme["text_secondary"], lw=1.0, ls="--")
            median = float(closure["abs_closure"].median())
            ax.set_xlabel("loop closure in log scale")
            ax.set_ylabel("loops")
            ax.set_title(f"{len(closure)} loops, median |closure| {median:.3f}", fontsize=9)

        ax = axes[1]
        if comparison is None or comparison.empty:
            ax.text(
                0.5,
                0.5,
                "awaiting the benchmark grid\n(NetCal vs Pathfinding by resistance)",
                ha="center",
                va="center",
                fontsize=8,
                color=theme["text_secondary"],
            )
            ax.set_axis_off()
        else:
            ax.scatter(
                comparison["resistance"],
                comparison["advantage"],
                s=14,
                color=theme["categorical"][1],
                alpha=0.75,
            )
            ax.axhline(0.0, color=theme["text_secondary"], lw=1.0, ls="--")
            ax.set_xscale("symlog", linthresh=RESISTANCE_LINTHRESH)
            ax.set_xlabel("effective resistance to the anchor")
            ax.set_ylabel("chained error - concurrent error")
            ax.set_title("concurrent's advantage against distance", fontsize=9)

        caption(
            fig,
            "A consistent set of per-observer scales closes every loop exactly, so a "
            "non-zero closure is misfit that no choice of scale can remove.",
            mode,
        )
    return fig


def runtime_figure(stages: pd.DataFrame, mode: str = "light") -> plt.Figure:
    """f40 -- measured wall clock per stage, against the stated target."""
    with figure_style(mode) as theme:
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        order = stages.sort_values("seconds")
        colors = [
            STATUS["critical"] if row.seconds > row.budget else theme["categorical"][0]
            for row in order.itertuples()
        ]
        ax.barh(order["stage"], order["seconds"], color=colors)
        for index, row in enumerate(order.itertuples()):
            ax.text(
                row.seconds,
                index,
                f" {row.seconds:.1f}s",
                va="center",
                fontsize=7.5,
            )
        ax.set_xscale("log")
        ax.set_xlabel("wall clock (s), measured")
        caption(
            fig,
            "Measured on this machine, not estimated; a bar past its budget is drawn "
            "in the warning colour.",
            mode,
        )
    return fig


def parameter_figure(
    observers: pd.DataFrame, slots: pd.DataFrame, mode: str = "light"
) -> plt.Figure:
    """f37 -- the observer parameter table as a picture."""
    with figure_style(mode) as theme:
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))

        ax = axes[0]
        ax.scatter(
            slots["year"],
            slots["threshold_msh"],
            s=4,
            alpha=0.25,
            color=theme["categorical"][0],
        )
        ax.set_yscale("log")
        ax.set_xlabel("year")
        ax.set_ylabel("fitted acuity threshold (MSH)")
        ax.set_title("acuity through time", fontsize=9)

        ax = axes[1]
        ax.hist(
            observers["drift_log_range"].dropna(),
            bins=40,
            color=theme["categorical"][1],
        )
        ax.set_xlabel("log range of the rate over a career")
        ax.set_ylabel("observers")
        ax.set_title("how much each career drifts", fontsize=9)

        ax = axes[2]
        finite = observers[np.isfinite(observers["factor"])]
        ax.scatter(
            finite["recovered_threshold_msh"],
            finite["factor"],
            s=14,
            alpha=0.7,
            color=theme["categorical"][2],
        )
        ax.set_xscale("log")
        ax.set_xlabel("fitted acuity threshold (MSH)")
        ax.set_ylabel("calibration factor 1/r")
        ax.set_title("the two are not independent", fontsize=9)

        caption(
            fig,
            "The acuity is weakly identified -- d log k / d log T is about -0.24 -- so "
            "the factor is the quantity to read, not the threshold.",
            mode,
        )
    return fig


def series_figure(
    series: dict[str, pd.DataFrame],
    sensitivity: dict[str, pd.DataFrame] | None = None,
    mode: str = "light",
) -> plt.Figure:
    """f36 -- the released series under both anchorings, and their difference."""
    with figure_style(mode) as theme:
        fig, axes = plt.subplots(2, 1, figsize=(10.0, 5.6), sharex=True)

        ax = axes[0]
        yearly = {}
        for slot, (name, table) in enumerate(sorted(series.items())):
            frame = table.copy()
            frame["year"] = np.floor(_years(frame["date_ord"]))
            grouped = frame.groupby("year")
            mean = grouped["gn"].mean()
            sd = grouped["gn_sd"].mean()
            yearly[name] = mean
            ax.plot(
                mean.index, mean.to_numpy(), lw=1.1, color=theme["categorical"][slot], label=name
            )
            ax.fill_between(
                mean.index,
                (mean - 2 * sd).clip(lower=0).to_numpy(),
                (mean + 2 * sd).to_numpy(),
                color=theme["categorical"][slot],
                alpha=0.18,
                lw=0,
            )
        ax.set_ylabel("reference-equivalent groups/day")
        ax.legend(fontsize=7.5, frameon=False)
        ax.set_title("the released series, with 2-sigma ensemble bands", fontsize=9)

        ax = axes[1]
        names = sorted(yearly)
        if len(names) >= 2:
            left, right = yearly[names[0]], yearly[names[1]]
            common = left.index.intersection(right.index)
            ax.plot(
                common,
                (left.reindex(common) - right.reindex(common)).to_numpy(),
                lw=1.0,
                color=theme["categorical"][3],
            )
            ax.axhline(0.0, color=theme["text_secondary"], lw=1.0, ls="--")
            ax.set_ylabel(f"{names[0]} - {names[1]}")
        else:
            ax.text(
                0.5,
                0.5,
                "awaiting the second anchoring",
                ha="center",
                va="center",
                fontsize=8,
                color=theme["text_secondary"],
            )
            ax.set_axis_off()
        ax.set_xlabel("year")

        caption(
            fig,
            "The two anchorings are reported side by side because their difference "
            "is the anchor-sensitivity result ADR-003 asks for, not a robustness check.",
            mode,
        )
    return fig
