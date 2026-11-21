"""Figures that audit the data, in place of printed reports.

Each function answers one question a reviewer would otherwise have to take on trust
from a number in a document:

* :func:`coverage_figure` (f01) -- who observed when, and where the record thins;
* :func:`release_diff_figure` (f02) -- what changed between v1.3_3 and v1.3_4;
* :func:`groups_tail_figure` (f03) -- why ``K = 41`` and not ``max(GROUPS)`` (D17);
* :func:`leap_day_figure` (f04) -- that the calendar is sound across four centuries;
* :func:`day_status_figure` (f05) -- that two surveys record no spotless days (D23);
* :func:`lifetime_figure` (f06) -- that "lifetimes" are disk-passage-censored spans.
"""

from __future__ import annotations

import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gn_next_gen.io import (
    load_day_status,
    load_group_single_entry,
    load_observations,
    load_observers,
)
from gn_next_gen.io.reference import CORE_FIVE
from gn_next_gen.models.visibility import max_transit_days, simulate_marked_process
from gn_next_gen.viz.style import (
    SURVEY_COLORS,
    caption,
    figure_style,
    palette,
    role_color,
    sequential_colors,
)

SURVEY_LABELS = {
    "schwabe": "Schwabe",
    "sporer": "Spörer",
    "rgo": "RGO",
    "debrecen": "Debrecen",
    "kmas": "KMAS",
}


def _years(date_ord: np.ndarray) -> np.ndarray:
    """Calendar year for an array of proleptic-Gregorian ordinals."""
    unique, inverse = np.unique(np.asarray(date_ord), return_inverse=True)
    years = np.array([dt.date.fromordinal(int(o)).year for o in unique])
    return years[inverse]


def coverage_figure(mode: str = "light") -> plt.Figure:
    """f01 -- observer-days and active stations per year, and where the record thins."""
    theme = palette(mode)
    observations = load_observations(include_extension=True)
    observers = load_observers()
    observations = observations.assign(year=_years(observations["date_ord"].to_numpy()))
    # The station count has to be split: v1.3's observers are the historical network,
    # and the extension adds survey-derived stations that are not observers in the
    # same sense (D21). Counting them together would overstate the network.
    network = observations[observations["source"] == "v1.3_4"]

    per_year = observations.groupby("year").agg(
        observer_days=("station", "size"),
        stations=("station", "nunique"),
        mean_groups=("groups", "mean"),
    )
    spotless = observations.assign(zero=observations["groups"] == 0).groupby("year")["zero"].mean()

    with figure_style(mode):
        fig, axes = plt.subplots(4, 1, figsize=(9.5, 9.0), sharex=True)

        ax = axes[0]
        ax.fill_between(
            per_year.index,
            per_year["observer_days"],
            color=role_color("observed", mode),
            alpha=0.25,
            linewidth=0,
        )
        ax.plot(per_year.index, per_year["observer_days"], color=role_color("observed", mode))
        ax.set_yscale("log")
        ax.set_ylabel("observer-days")
        ax.set_title("Observer-days per year, v1.3_4 plus the 2026 count extension")
        ax.axvline(2010.5, color=theme["text_secondary"], linestyle=(0, (4, 3)), linewidth=1.0)
        ax.annotate(
            "v1.3 ends 2010;\nextension carries to 2026",
            xy=(2011, ax.get_ylim()[0] * 3),
            fontsize=7.5,
            color=theme["text_secondary"],
            ha="right",
        )

        ax = axes[1]
        network_per_year = network.groupby("year")["station"].nunique()
        ax.plot(
            network_per_year.index, network_per_year.to_numpy(), color=role_color("simulated", mode)
        )
        ax.set_ylabel("stations reporting")
        ax.set_title(
            f"Distinct stations per year — {network['station'].nunique()} of "
            f"{len(observers)} listed v1.3 observers ever report"
        )

        ax = axes[2]
        ax.plot(per_year.index, per_year["mean_groups"], color=role_color("observed", mode))
        ax.set_ylabel("groups per report")
        ax.set_title("Mean groups reported")

        # Its own panel rather than a scaled overlay: a second series squeezed onto
        # another series' axis cannot be read quantitatively, and a twin axis is worse.
        ax = axes[3]
        ax.fill_between(
            spotless.index,
            spotless,
            color=role_color("alternative", mode),
            alpha=0.35,
            linewidth=0,
        )
        ax.plot(spotless.index, spotless, color=role_color("alternative", mode))
        ax.set_ylim(0, 1)
        ax.set_ylabel("zero-count fraction")
        ax.set_xlabel("year")
        ax.set_title("Fraction of reports that are spotless")

        caption(
            fig,
            "The record thins towards 1610 and the spotless fraction rises into the "
            "Maunder Minimum — the regime the reporting layer has to model.",
            mode,
        )
    return fig


def release_diff_figure(diff: pd.DataFrame, mode: str = "light") -> plt.Figure:
    """f02 -- v1.3_3 against v1.3_4, from the committed per-year diff summary."""
    theme = palette(mode)
    with figure_style(mode):
        fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.6), sharex=True)

        ax = axes[0]
        width = 0.42
        ax.bar(
            diff["year"] - width / 2,
            diff["only_in_v4"],
            width=width,
            color=role_color("observed", mode),
            label="new in v1.3_4",
        )
        ax.bar(
            diff["year"] + width / 2,
            -diff["only_in_v3"],
            width=width,
            color=role_color("simulated", mode),
            label="removed from v1.3_3",
        )
        ax.axhline(0, color=theme["grid"], linewidth=0.8)
        ax.set_ylabel("observer-days")
        ax.set_title(
            f"Observer-days added and removed: {int(diff['only_in_v4'].sum())} new, "
            f"{int(diff['only_in_v3'].sum())} gone"
        )
        ax.legend(loc="upper left")

        ax = axes[1]
        ax.bar(
            diff["year"],
            diff["changed_groups"],
            width=0.9,
            color=role_color("alternative", mode),
            label="GROUPS changed",
        )
        ax.bar(
            diff["year"],
            diff["duplicate_keys_v3"],
            width=0.9,
            color=theme["categorical"][7],
            alpha=0.85,
            label="duplicate (date, station) keys in v1.3_3",
        )
        ax.set_ylabel("observer-days")
        ax.set_xlabel("year")
        ax.set_title(
            f"{int(diff['changed_groups'].sum())} counts corrected; the "
            f"{int(diff['duplicate_keys_v3'].sum())} duplicate keys of D16 are down to "
            f"{int(diff['duplicate_keys_v4'].sum())}"
        )
        ax.legend(loc="upper left")

        # The cluster is the finding, so it is named rather than left to be noticed.
        early = diff[(diff["year"] >= 1715) & (diff["year"] <= 1730)]
        share = 100 * early["changed_groups"].sum() / max(diff["changed_groups"].sum(), 1)
        for ax in axes:
            ax.annotate(
                f"1715–1730:\n{share:.0f}% of all corrections",
                xy=(1731, ax.get_ylim()[1] * 0.75),
                fontsize=7.5,
                color=theme["text_secondary"],
            )
        caption(
            fig,
            "v1.3_4 is a correction-and-deduplication release, and almost all of it lands in "
            "1719–1721 — the end of the Maunder Minimum, which is the era this project's "
            "grand-minimum question turns on.",
            mode,
        )
    return fig


def groups_tail_figure(mode: str = "light") -> plt.Figure:
    """f03 -- the ``GROUPS`` tail, the two outliers, and the chosen truncation ``K``."""
    theme = palette(mode)
    observations = load_observations()
    counts = observations["groups"].to_numpy()
    reference_max = 31  # KMAS; RGO 26, Debrecen 21

    grid = np.arange(0, counts.max() + 2)
    survival = np.array([(counts >= g).mean() for g in grid])

    with figure_style(mode):
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))

        ax = axes[0]
        ax.step(grid, np.maximum(survival, 1e-9), where="post", color=role_color("observed", mode))
        ax.set_yscale("log")
        ax.set_xlabel("reported groups in a day, $g$")
        ax.set_ylabel(r"$P(\mathrm{GROUPS} \geq g)$")
        ax.set_title("The reported-count tail, v1.3_4")
        ax.axhline(1e-5, color=theme["text_secondary"], linestyle=(0, (4, 3)), linewidth=1.0)
        ax.annotate(
            r"$10^{-5}$: the rule's cut",
            xy=(1, 1.3e-5),
            fontsize=7.5,
            color=theme["text_secondary"],
        )
        ax.axvline(41, color=theme["categorical"][5], linewidth=1.6)
        ax.annotate("K = 41", xy=(41.6, 2e-4), fontsize=8.5, color=theme["categorical"][5])
        ax.axvline(
            reference_max,
            color=role_color("alternative", mode),
            linewidth=1.2,
            linestyle=(0, (2, 2)),
        )
        ax.annotate(
            f"reference max {reference_max}",
            xy=(reference_max - 0.8, 3e-6),
            rotation=90,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=role_color("alternative", mode),
        )
        ax.scatter(
            [58],
            [(counts >= 58).mean()],
            s=42,
            zorder=5,
            color=theme["categorical"][7],
            label="58 — inherited from v1.22",
        )
        ax.scatter(
            [93],
            [1e-6],
            s=42,
            marker="x",
            zorder=5,
            color=theme["text_secondary"],
            label="93 — removed by v1.3_4",
        )
        ax.legend(loc="lower left")

        ax = axes[1]
        quantiles = [0.99, 0.999, 0.9999, 0.99999]
        values = [np.quantile(counts, q) for q in quantiles]
        positions = np.arange(len(quantiles))
        colors = sequential_colors(len(quantiles), mode)
        bars = ax.bar(positions, values, color=colors, width=0.62)
        for bar, value in zip(bars, values, strict=True):
            ax.annotate(
                f"{value:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=theme["text_primary"],
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(["99th", "99.9th", "99.99th", "99.999th"])
        ax.axhline(41, color=theme["categorical"][5], linewidth=1.6)
        ax.annotate(
            "K = 41",
            xy=(0.02, 41.8),
            xycoords=("axes fraction", "data"),
            fontsize=8.5,
            color=theme["categorical"][5],
        )
        ax.axhline(
            counts.max(), color=theme["categorical"][7], linewidth=1.2, linestyle=(0, (3, 3))
        )
        ax.annotate(
            f"max {counts.max()} — not used",
            xy=(0.97, counts.max() - 4.0),
            xycoords=("axes fraction", "data"),
            ha="right",
            va="top",
            fontsize=7.5,
            color=theme["categorical"][7],
        )
        ax.set_ylim(0, counts.max() * 1.08)
        ax.set_ylabel("reported groups")
        ax.set_title("Robust quantiles, which size the state space")

        caption(
            fig,
            "D17: K comes from a robust quantile and the reference's own daily maximum, "
            "never from max(GROUPS). The rule's minimum is 39 here and was 41 for v1.3_3, "
            "so 41 is stable across both releases.",
            mode,
        )
    return fig


def leap_day_figure(mode: str = "light") -> plt.Figure:
    """f04 -- every 29 February in the record against the Gregorian rule."""
    theme = palette(mode)
    observations = load_observations()
    ordinals = np.unique(observations["date_ord"].to_numpy())
    dates = [dt.date.fromordinal(int(o)) for o in ordinals]
    leap_years = sorted({d.year for d in dates if (d.month, d.day) == (2, 29)})

    centurial = [y for y in range(1600, 2101, 100)]
    gregorian_leap = [y for y in centurial if y % 400 == 0]

    with figure_style(mode):
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))

        ax = axes[0]
        divisible = [y for y in range(1610, 2011) if y % 4 == 0]
        present = [y in leap_years for y in divisible]
        ax.scatter(
            [y for y, p in zip(divisible, present, strict=True) if p],
            [1] * sum(present),
            s=14,
            color=role_color("observed", mode),
            label="29 Feb present",
        )
        missing = [y for y, p in zip(divisible, present, strict=True) if not p]
        ax.scatter(
            missing,
            [1] * len(missing),
            s=44,
            marker="x",
            color=theme["categorical"][7],
            label="29 Feb absent (no observation that day)",
        )
        for year in centurial:
            if 1610 <= year <= 2010:
                colour = (
                    role_color("alternative", mode)
                    if year in gregorian_leap
                    else theme["text_secondary"]
                )
                ax.axvline(year, color=colour, linewidth=1.0, linestyle=(0, (3, 3)))
        ax.set_yticks([])
        ax.set_xlabel("year divisible by 4")
        ax.set_title(f"29 February: {len(leap_years)} years carry one")
        ax.legend(loc="lower center", ncol=1)

        ax = axes[1]
        # A zero-height bar renders as nothing, which reads as "no data" rather than
        # "absent". Markers state both states explicitly.
        centuries = [1700, 1800, 1900, 2000]
        positions = np.arange(len(centuries))
        in_data = [1 if year in leap_years else 0 for year in centuries]
        by_rule = [1 if year % 400 == 0 else 0 for year in centuries]
        ax.scatter(
            positions - 0.12,
            in_data,
            s=90,
            color=role_color("observed", mode),
            label="present in the data",
            zorder=3,
        )
        ax.scatter(
            positions + 0.12,
            by_rule,
            s=90,
            marker="D",
            color=role_color("simulated", mode),
            label="Gregorian rule",
            zorder=3,
        )
        for position in positions:
            ax.plot(
                [position - 0.12, position + 0.12],
                [0, 0],
                color=theme["grid"],
                linewidth=0.8,
                zorder=1,
            )
        agree = all(a == b for a, b in zip(in_data, by_rule, strict=True))
        ax.set_xticks(positions)
        ax.set_xticklabels([str(year) for year in centuries])
        ax.set_xlim(-0.5, len(centuries) - 0.5)
        ax.set_ylim(-0.35, 1.35)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["absent", "present"])
        ax.set_title(
            "Centurial years: data against the rule"
            + ("  — agree everywhere" if agree else "  — MISMATCH")
        )
        ax.legend(loc="center left")

        caption(
            fig,
            "1700, 1800 and 1900 are not Gregorian leap years and carry no 29 February; "
            "2000 is and does. Absences in other years are simply days nobody observed.",
            mode,
        )
    return fig


def day_status_figure(mode: str = "light") -> plt.Figure:
    """f05 -- ``day_status`` per survey through time: the figure that shows D23."""
    theme = palette(mode)
    days = load_day_status(CORE_FIVE)
    days = days.assign(year=_years(days["date_ord"].to_numpy()))

    with figure_style(mode):
        fig, axes = plt.subplots(len(CORE_FIVE), 1, figsize=(9.5, 7.8), sharex=True)
        for ax, survey in zip(axes, CORE_FIVE, strict=True):
            sub = days[days["survey"] == survey]
            per_year = sub.groupby("year")["day_status"].value_counts().unstack(fill_value=0)
            for status in (0, 1, 2):
                if status not in per_year:
                    per_year[status] = 0
            colour = SURVEY_COLORS[mode].get(survey, theme["categorical"][6])
            ax.fill_between(
                per_year.index, per_year[2], color=colour, alpha=0.9, linewidth=0, label="active"
            )
            ax.fill_between(
                per_year.index, per_year[1], color=colour, alpha=0.35, linewidth=0, label="spotless"
            )
            ax.plot(
                per_year.index,
                per_year[0],
                color=theme["text_secondary"],
                linewidth=1.0,
                label="no observation",
            )
            total_spotless = int(per_year[1].sum())
            flag = "" if total_spotless else "   ← no spotless days at all (D23)"
            ax.set_title(
                f"{SURVEY_LABELS[survey]} — {total_spotless:,} spotless days{flag}", loc="left"
            )
            ax.set_ylabel("days/yr")
            if survey == CORE_FIVE[0]:
                ax.legend(loc="upper right", ncol=3)
        axes[-1].set_xlabel("year")
        caption(
            fig,
            "Schwabe and Spörer record no spotless day anywhere in their span, so neither "
            "can supply a zero fraction or a spotless run length. Both are excluded from "
            "Step 1 (PI, 2026-11-21); Schwabe's handling is deferred to Step 4.",
            mode,
        )
    return fig


def lifetime_figure(mode: str = "light") -> plt.Figure:
    """f06 -- observed visible spans against the rotation kernel's prediction."""
    theme = palette(mode)
    single = load_group_single_entry(["rgo", "debrecen", "kmas"])
    simulated = simulate_marked_process(np.full(40_000, 1.2), 8.0, seed=11)
    transit = max_transit_days()

    with figure_style(mode):
        fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))

        ax = axes[0]
        bins = np.arange(0.5, 26.5, 1.0)
        for survey in ("rgo", "debrecen", "kmas"):
            spans = single.loc[single["survey"] == survey, "obs_duration"].dropna()
            ax.hist(
                spans,
                bins=bins,
                histtype="step",
                linewidth=2.0,
                density=True,
                color=SURVEY_COLORS[mode][survey],
                label=SURVEY_LABELS[survey],
            )
        ax.axvline(transit, color=theme["text_secondary"], linestyle=(0, (4, 3)), linewidth=1.2)
        ax.annotate(
            f"half a rotation\n{transit:.1f} d",
            xy=(transit + 0.5, ax.get_ylim()[1] * 0.6),
            fontsize=7.5,
            color=theme["text_secondary"],
        )
        ax.set_xlabel("obs_duration (days)")
        ax.set_ylabel("density")
        ax.set_title("Observed spans stop at disk passage")
        ax.legend(loc="upper right")

        ax = axes[1]
        rgo = single.loc[single["survey"] == "rgo", "obs_duration"].dropna().to_numpy()
        ax.hist(
            rgo,
            bins=bins,
            histtype="step",
            linewidth=2.0,
            density=True,
            color=role_color("observed", mode),
            label="RGO observed",
        )
        ax.hist(
            simulated.first_apparition_spans,
            bins=bins,
            histtype="step",
            linewidth=2.0,
            density=True,
            color=role_color("simulated", mode),
            label="rotation kernel, first apparition",
        )
        ax.hist(
            simulated.visible_spans,
            bins=bins,
            histtype="step",
            linewidth=1.3,
            density=True,
            linestyle=(0, (3, 2)),
            color=role_color("alternative", mode),
            label="same, every apparition",
        )
        ax.axvline(transit, color=theme["text_secondary"], linestyle=(0, (4, 3)), linewidth=1.2)
        ax.set_xlabel("span (days)")
        ax.set_title("Which simulated quantity is comparable")
        ax.legend(loc="upper right", fontsize=7.5)

        ax = axes[2]
        for survey in ("rgo", "debrecen"):
            spans = single.loc[single["survey"] == survey, "obs_duration"].dropna().to_numpy()
            grid = np.arange(0, 20)
            surv = np.array([(spans > g).mean() for g in grid])
            ratio = np.divide(
                surv[1:], surv[:-1], out=np.full(len(grid) - 1, np.nan), where=surv[:-1] > 0
            )
            ax.plot(
                grid[:-1],
                ratio,
                marker="o",
                markersize=4,
                color=SURVEY_COLORS[mode][survey],
                label=SURVEY_LABELS[survey],
            )
        ax.axhline(0.88, color=theme["text_secondary"], linestyle=(0, (4, 3)), linewidth=1.0)
        ax.annotate(
            r"geometric, $\varphi \approx 0.88$",
            xy=(0.3, 0.895),
            fontsize=7.5,
            color=theme["text_secondary"],
        )
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("day $t$")
        ax.set_ylabel(r"$S(t+1)/S(t)$")
        ax.set_title("Flat, then a cliff — not geometric")
        ax.legend(loc="lower left")

        caption(
            fig,
            "obs_duration is visible residence, not intrinsic lifetime: flat hazard for "
            "nine days, then disk passage ends every apparition. phi in the chain is "
            "therefore a visible-residence parameter.",
            mode,
        )
    return fig
