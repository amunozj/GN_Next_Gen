"""House style for every Step 1 figure, in both light and dark.

Verification in this step is **figure-first** (PI, 2026-11-21): a script prints
progress and writes numbers to CSV, but what a reviewer reads is the figure.  That
only works if the figures are legible, consistent and honest, so the palette is not
chosen per plot.

The colours are the validated default palette of the ``dataviz`` skill, used
**unchanged**: categorical slots in fixed order (never cycled), a single-hue blue
ramp for ordered quantities such as the acuity thresholds, and the skill's own
light and dark surfaces.  Its published validation therefore applies as written --
worst adjacent-pair CVD Delta E 9.1 light / 8.4 dark, and the first three slots clear
the all-pairs floors in both modes.  Forms where every pair is compared at once
(scatter, small multiples) use at most those first three slots, which is why the
three Step 1 surveys map onto them exactly.  The skill's ``validate_palette.js``
could not be re-run here (no Node on this machine), which is precisely why nothing
is re-stepped or invented.

Rules carried from the skill that are easy to break by accident:

* identity is never colour alone -- every multi-series panel has a legend, and
  series are directly labelled where there is room;
* one y-axis per panel, never a twin scale;
* ordered quantities get the sequential ramp, not categorical hues;
* grid and spines recede; data is the darkest thing in the frame.
"""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- the palette, unchanged from the skill's reference instance -------------

CATEGORICAL_LIGHT = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
CATEGORICAL_DARK = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#0ca30c",
    "#9085e9",
    "#e66767",
]

#: Blue, light -> dark.  Steps 250..700, so the lightest still clears 2:1 on the
#: light surface (the skill's ordinal rule).
SEQUENTIAL_LIGHT = ["#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL_DARK = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "grid": "#d9d8d4",
        "categorical": CATEGORICAL_LIGHT,
        "sequential": SEQUENTIAL_LIGHT,
    },
    "dark": {
        "surface": "#1a1a19",
        "text_primary": "#f0efec",
        "text_secondary": "#c3c2b7",
        "grid": "#383835",
        "categorical": CATEGORICAL_DARK,
        "sequential": SEQUENTIAL_DARK,
    },
}

#: The three Step 1 surveys, mapped to the first three categorical slots -- the
#: ones that clear the all-pairs floors, which small multiples need.
SURVEY_COLORS = {"light": {}, "dark": {}}
for _mode in ("light", "dark"):
    for _i, _survey in enumerate(("rgo", "debrecen", "kmas")):
        SURVEY_COLORS[_mode][_survey] = THEMES[_mode]["categorical"][_i]

#: Observed against simulated: slot 1 against slot 2, everywhere, so the reader
#: learns the pairing once.
ROLE_SLOT = {"observed": 0, "simulated": 1, "alternative": 2}


def palette(mode: str = "light") -> dict:
    """The theme dictionary for one mode."""
    if mode not in THEMES:
        raise ValueError(f"unknown mode {mode!r}; expected 'light' or 'dark'")
    return THEMES[mode]


def role_color(role: str, mode: str = "light") -> str:
    """Colour for a fixed role (observed / simulated / alternative)."""
    return palette(mode)["categorical"][ROLE_SLOT[role]]


def sequential_colors(n: int, mode: str = "light") -> list[str]:
    """``n`` steps of the single-hue ramp, light to dark, for an ordered quantity."""
    ramp = palette(mode)["sequential"]
    if n <= 1:
        return [ramp[len(ramp) // 2]]
    step = (len(ramp) - 1) / (n - 1)
    return [ramp[round(i * step)] for i in range(n)]


def rc_params(mode: str = "light") -> dict:
    """Matplotlib settings for one mode: recessive frame, readable text, no cycling."""
    theme = palette(mode)
    return {
        "figure.facecolor": theme["surface"],
        "axes.facecolor": theme["surface"],
        "savefig.facecolor": theme["surface"],
        "savefig.edgecolor": theme["surface"],
        "text.color": theme["text_primary"],
        "axes.labelcolor": theme["text_primary"],
        "axes.titlecolor": theme["text_primary"],
        "xtick.color": theme["text_secondary"],
        "ytick.color": theme["text_secondary"],
        "axes.edgecolor": theme["grid"],
        "grid.color": theme["grid"],
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.0,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "figure.constrained_layout.use": True,
        "axes.prop_cycle": mpl.cycler(color=theme["categorical"]),
    }


@contextmanager
def figure_style(mode: str = "light"):
    """Apply the house style for one mode, then restore whatever was set."""
    with plt.rc_context(rc_params(mode)):
        yield palette(mode)


def annotate_target(
    ax, value: float, mode: str = "light", *, label: str = "target", orientation: str = "h"
) -> None:
    """Draw an advisory target line.

    Check A's thresholds are advisory on this first pass, so they are drawn as a
    reference line with a word on it, never as a pass/fail colour fill that would
    read as a verdict the data has not yet earned.
    """
    theme = palette(mode)
    drawer = ax.axhline if orientation == "h" else ax.axvline
    drawer(value, color=theme["text_secondary"], linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
    if orientation == "h":
        ax.annotate(
            label,
            xy=(0.995, value),
            xycoords=("axes fraction", "data"),
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=theme["text_secondary"],
        )


def caption(fig, text: str, mode: str = "light") -> None:
    """One line under the figure saying what the reader should take from it."""
    fig.supxlabel(text, fontsize=8, color=palette(mode)["text_secondary"], ha="center")


def save_both_modes(builder, out_dir, stem: str, *, formats=("png",)) -> list:
    """Render ``builder(mode)`` once per mode and write ``<stem>_<mode>.<ext>``.

    ``builder`` takes a mode name and returns a figure.  Writing both modes from the
    same code is what keeps the dark variant a *selected* design rather than an
    automatic inversion: the palette's dark steps were chosen for the dark surface.
    """
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for mode in ("light", "dark"):
        fig = builder(mode)
        for ext in formats:
            path = out_dir / f"{stem}_{mode}.{ext}"
            fig.savefig(path, bbox_inches="tight")
            written.append(path)
        plt.close(fig)
    return written
