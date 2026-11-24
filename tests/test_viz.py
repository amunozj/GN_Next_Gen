"""Figures are the Step 1 deliverable, so they get tested like code.

These do not judge whether a figure is beautiful. They check the things that would
silently produce a misleading picture: a palette that cycles, a dark mode that is an
automatic inversion rather than its own steps, a figure builder that mutates global
state, and a target line drawn as if it were a verdict.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from gn_next_gen.stats.discrepancy import TARGETS  # noqa: E402
from gn_next_gen.viz import checks  # noqa: E402
from gn_next_gen.viz.style import (  # noqa: E402
    CATEGORICAL_DARK,
    CATEGORICAL_LIGHT,
    ROLE_SLOT,
    SURVEY_COLORS,
    THEMES,
    palette,
    rc_params,
    role_color,
    save_both_modes,
    sequential_colors,
)

MODES = ("light", "dark")


# --- the palette ----------------------------------------------------------


def test_both_modes_are_defined_with_their_own_steps():
    """Dark is a selected palette, not an inversion of light."""
    assert set(THEMES) == set(MODES)
    assert CATEGORICAL_LIGHT != CATEGORICAL_DARK
    assert len(CATEGORICAL_LIGHT) == len(CATEGORICAL_DARK) == 8
    assert palette("light")["surface"] != palette("dark")["surface"]


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown mode"):
        palette("sepia")


def test_roles_keep_a_fixed_colour_across_every_figure():
    """Observed is always slot 1 and simulated always slot 2, so the reader learns it once."""
    for mode in MODES:
        assert role_color("observed", mode) == palette(mode)["categorical"][0]
        assert role_color("simulated", mode) == palette(mode)["categorical"][1]
    assert len(set(ROLE_SLOT.values())) == len(ROLE_SLOT)


def test_surveys_use_the_first_three_slots():
    """Forms that compare every pair at once are only safe on the first three slots."""
    for mode in MODES:
        colours = [SURVEY_COLORS[mode][s] for s in ("rgo", "debrecen", "kmas")]
        assert colours == palette(mode)["categorical"][:3]
        assert len(set(colours)) == 3


def test_sequential_ramp_is_ordered_and_single_hue():
    """An ordered quantity gets the ramp, never categorical hues."""
    for mode in MODES:
        ramp = sequential_colors(5, mode)
        assert len(ramp) == 5
        assert len(set(ramp)) == 5
        assert all(c in palette(mode)["sequential"] for c in ramp)
    assert sequential_colors(1)[0] in palette("light")["sequential"]


def test_rc_params_do_not_cycle_colours_beyond_the_palette():
    cycle = rc_params("light")["axes.prop_cycle"].by_key()["color"]
    assert cycle == CATEGORICAL_LIGHT


def test_style_does_not_leak_into_global_state():
    before = dict(plt.rcParams)
    from gn_next_gen.viz.style import figure_style

    with figure_style("dark"):
        pass
    assert plt.rcParams["figure.facecolor"] == before["figure.facecolor"]


# --- the figure builders --------------------------------------------------


@pytest.fixture
def acf_table():
    rows = []
    for survey in ("rgo", "debrecen"):
        for detrended in (False, True):
            for lag in range(1, 31):
                rows.append(
                    {
                        "survey": survey,
                        "detrended": detrended,
                        "lag": lag,
                        "observed": 0.9**lag,
                        "simulated": 0.88**lag,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def score_table():
    rows = []
    for survey in ("rgo", "debrecen"):
        for threshold in (0.0, 30.0):
            for n_classes in (1, 2):
                row = {
                    "survey": survey,
                    "threshold_msh": threshold,
                    "n_classes": n_classes,
                    "aggregate": 1.3,
                }
                row.update({measure: target * 0.8 for measure, target in TARGETS.items()})
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("mode", MODES)
def test_check_figures_build_in_both_modes(acf_table, score_table, mode):
    figure = checks.acf_figure(acf_table, mode)
    assert figure.get_facecolor()[:3] == matplotlib.colors.to_rgb(palette(mode)["surface"])
    plt.close(figure)

    bins = pd.DataFrame(
        {
            "bin": range(5),
            "observed": np.linspace(0.8, 0.0, 5),
            "simulated": np.linspace(0.85, 0.0, 5),
        }
    )
    pmf = pd.DataFrame(
        {"count": range(10), "observed": np.full(10, 0.1), "simulated": np.full(10, 0.1)}
    )
    runs = pd.DataFrame(
        {"length": range(1, 8), "observed": np.arange(7, 0, -1), "simulated": np.arange(7, 0, -1)}
    )
    plt.close(checks.zero_and_counts_figure(bins, pmf, runs, mode))
    plt.close(checks.scores_figure(score_table, mode))

    # The Check B figures: the law itself, and its activity dependence.
    from gn_next_gen.reference.area_law import AreaLawParams

    base = AreaLawParams(
        survey="rgo",
        k_W=0.45,
        log_lambda_W=float(np.log(5.7)),
        mu_LN=5.47,
        sigma_LN=0.89,
        pi_W=0.79,
        sigma_floor=0.17,
        threshold_msh=28.08,
    )
    rng = np.random.default_rng(0)
    groups = pd.DataFrame({"area_corr": rng.lognormal(5.47, 0.89, 3000)})
    plt.close(checks.area_law_figure({"rgo": (groups, base)}, None, mode))

    stage2 = pd.DataFrame(
        {
            "survey": ["rgo"] * 5 + ["kmas"] * 5,
            "activity": list(np.linspace(0.05, 0.5, 5)) * 2,
            "mu_LN": list(np.linspace(5.4, 5.7, 5)) * 2,
            "sigma_LN": list(np.linspace(0.85, 0.95, 5)) * 2,
            "c_LN": list(np.linspace(0.2, 0.4, 5)) * 2,
            "n_groups": [1500] * 10,
        }
    )
    power = pd.DataFrame(
        {
            "survey": ["rgo", "rgo", "rgo", "kmas", "kmas", "kmas"],
            "parameter": ["mu_LN", "sigma_LN", "c_LN"] * 2,
            "a": [5.7, 0.94, 0.23, 5.9, 0.97, 0.34],
            "b": [0.025, 0.03, 0.057, 0.021, 0.109, 0.289],
        }
    )
    gate = [
        {"survey": "rgo", "parameter": "k_W", "spearman_rho": -0.1},
        {"survey": "rgo", "parameter": "lambda_W", "spearman_rho": 0.2},
    ]
    plt.close(checks.activity_dependence_figure(stage2, power, gate, mode))

    timings = pd.DataFrame(
        {
            "device": ["cpu", "cpu", "gpu", "gpu"],
            "states": [41, 1681, 41, 1681],
            "n_days": [10_000, 10_000, 10_000, 10_000],
            "seconds": [0.04, 0.27, 0.45, 0.64],
        }
    )
    plt.close(checks.filter_benchmark_figure(timings, mode))


def test_acf_figure_handles_a_single_survey(acf_table):
    """The panel grid must not transpose itself when there is one column."""
    single = acf_table[acf_table["survey"] == "rgo"]
    figure = checks.acf_figure(single, "light")
    assert len(figure.axes) == 2
    plt.close(figure)


def test_scores_figure_draws_the_target_as_a_reference_not_a_verdict(score_table):
    """Thresholds are advisory on this pass, so nothing may be coloured pass/fail."""
    figure = checks.scores_figure(score_table, "light")
    texts = [t.get_text() for ax in figure.axes for t in ax.texts]
    assert any("advisory" in t or "target" in t for t in texts)
    plt.close(figure)


@pytest.mark.parametrize("mode", MODES)
def test_network_figure_draws_every_node_including_the_unreachable(mode):
    """f28's point is the observers the anchor cannot reach, so none may be dropped."""
    from gn_next_gen.viz import benchmark

    nodes = pd.DataFrame(
        {
            "station": [1, 2, 3, 4],
            "mid_year": [1700.5, 1800.0, 1900.0, 1990.0],
            "resistance": [np.inf, 1e-2, 1e-4, 0.0],
        }
    )
    edges = pd.DataFrame({"station_a": [2, 3], "station_b": [3, 4], "overlap_days": [120, 4000]})
    cut = pd.DataFrame(
        {
            "year": [1750, 1850, 1950],
            "n_edges": [0, 1, 2],
            "overlap_days": [0, 120, 4120],
            "n_before": [1, 2, 3],
            "n_after": [3, 2, 1],
        }
    )
    figure = benchmark.network_figure(nodes, edges, cut, [(2, 3)], mode, min_side=1)
    scatters = [
        c
        for c in figure.axes[0].collections
        if isinstance(c, matplotlib.collections.PathCollection)
    ]
    # Every station once, plus the anchor again as the star that marks it.
    assert sum(len(c.get_offsets()) for c in scatters) == len(nodes) + 1
    labels = [t.get_text() for t in figure.axes[0].get_legend().get_texts()]
    assert any("unreachable (1)" in label for label in labels)
    assert any("bridge (1)" in label for label in labels)
    plt.close(figure)


def test_each_method_keeps_one_marker_and_they_are_all_distinct():
    """Shape is a method's identity, so no two methods may share one."""
    from gn_next_gen.viz import benchmark

    markers = benchmark._method_markers(benchmark.METHOD_ORDER)
    assert set(markers) == set(benchmark.METHOD_ORDER)
    assert len(set(markers.values())) == len(benchmark.METHOD_ORDER)
    # f22 draws the default-setting run as a grey cross, which is not a method.
    assert "x" not in set(markers.values())
    # A method the module has not met yet still gets a shape of its own.
    extended = benchmark._method_markers([*benchmark.METHOD_ORDER, "NewPort"])
    assert extended["NewPort"] not in {markers[m] for m in benchmark.METHOD_ORDER}
    assert all(extended[m] == markers[m] for m in benchmark.METHOD_ORDER)


def test_a_method_keeps_its_colour_when_the_others_are_absent():
    """f23 draws two methods and f20 draws nine; UEA21 is the same colour in both."""
    from gn_next_gen.viz import benchmark

    full = benchmark._method_colors(benchmark.METHOD_ORDER, "light")
    subset = benchmark._method_colors(["UEA21", "GRP"], "light")
    assert subset["UEA21"] == full["UEA21"]
    assert subset["GRP"] == full["GRP"]
    coloured = [m for m in benchmark.METHOD_ORDER if not m.startswith("null-")]
    assert len({full[m] for m in coloured}) == len(coloured)


def test_save_both_modes_writes_one_file_per_mode(tmp_path):
    def builder(mode):
        figure, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], color=role_color("observed", mode))
        return figure

    written = save_both_modes(builder, tmp_path, "toy")
    assert {p.name for p in written} == {"toy_light.png", "toy_dark.png"}
    assert all(p.stat().st_size > 1000 for p in written)


def _figure_folders():
    """Every committed step folder, so a new step is covered the day it lands."""
    from gn_next_gen.paths import DOCS

    return sorted(p for p in (DOCS / "results" / "figures").glob("step-*") if p.is_dir())


def test_every_committed_figure_has_both_modes():
    """The results doc can only offer a dark variant if every figure has one."""
    folders = _figure_folders()
    if not folders:
        pytest.skip("figures not built in this checkout")
    for folder in folders:
        stems = {p.name.rsplit("_", 1)[0] for p in folder.glob("*.png")}
        for stem in stems:
            assert (folder / f"{stem}_light.png").is_file(), f"{folder.name}/{stem}"
            assert (folder / f"{stem}_dark.png").is_file(), f"{folder.name}/{stem}"


def test_committed_figures_stay_small():
    """Only small summary figures are committed; the rest live in git-ignored outputs/."""
    folders = _figure_folders()
    if not folders:
        pytest.skip("figures not built in this checkout")
    for folder in folders:
        for path in folder.glob("*.png"):
            assert path.stat().st_size < 400_000, (
                f"{folder.name}/{path.name} is {path.stat().st_size / 1e3:.0f} kB"
            )


def test_every_committed_figure_is_embedded_and_captioned():
    """CLAUDE.md's "Reports and figures" rule, as a test rather than as prose.

    A committed figure that no report shows is a figure the reader has to go hunting for,
    and an uncaptioned one makes the reader guess what it proves. Both are easy to do by
    accident and invisible in review.

    The embed must be the **light** variant in plain Markdown. A ``<picture>`` block with
    ``prefers-color-scheme`` was tried and reverted on 2026-11-22: GitHub resolves a
    relative path in ``<img src>`` but not in a ``<source srcset>``, so on a dark theme the
    matching source won with a dead path and the fallback was never reached -- every figure
    in the report broke at once. The dark variant still has to exist, for slides, which
    ``test_every_committed_figure_has_both_modes`` covers.
    """
    from gn_next_gen.paths import DOCS

    results = DOCS / "results"
    reports = {path: path.read_text() for path in results.glob("*.md")}
    if not reports:
        pytest.skip("no reports in this checkout")

    for light in sorted(results.glob("figures/step-*/*_light.png")):
        relative = light.relative_to(results).as_posix()
        code = light.name.split("_", 1)[0]
        hosts = [text for text in reports.values() if relative in text]
        assert hosts, f"{relative} is committed but no report embeds it"
        for text in hosts:
            assert f"]({relative})" in text, (
                f"{relative} is referenced but not embedded as a Markdown image; "
                "a <picture>/<img> embed breaks on GitHub in dark mode"
            )
            assert f"**Figure {code}**" in text, f"{relative} is embedded without a caption"


def test_reports_do_not_use_a_picture_embed():
    """The regression that sent this report's figures blank for dark-theme readers.

    GitHub rewrites a relative path in ``<img src>`` and leaves ``<source srcset>`` alone,
    so a ``<picture>`` block resolves in light mode and fails in dark mode -- which is the
    worst kind of breakage, because it passes a casual look.
    """
    from gn_next_gen.paths import DOCS

    for path in (DOCS / "results").glob("*.md"):
        # Markup only: a prose mention inside backticks is how the reports explain the revert.
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            assert stripped != "<picture>", f"{path.name}:{number} opens a <picture> embed"
            assert 'srcset="' not in stripped, (
                f"{path.name}:{number} sets srcset, which GitHub leaves unresolved"
            )
