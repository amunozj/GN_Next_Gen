"""The seven ported baselines, each checked where its own construction is checkable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gn_next_gen.baselines import BASELINES, NULLS
from gn_next_gen.baselines.acuity import (
    ACUITY_GRID,
    build_library,
    recover_acuity,
)
from gn_next_gen.baselines.base import (
    PairEstimate,
    chain_from_reference,
    matched_counts,
    station_series,
)
from gn_next_gen.baselines.linear import VHS24, HoSc98, SvSc16, years_of
from gn_next_gen.baselines.pathfinding import robust_proportional_fit
from gn_next_gen.baselines.pmf import compose, conversion_matrix
from gn_next_gen.graph.overlap import build_overlap_graph

EPOCH = 700_000


def _chain_observations(factors: dict[int, float], n_days: int = 2000, seed: int = 0):
    """A network whose observers are exact multiples of a latent signal.

    Every scaling method must recover these factors, because there is nothing
    else in the data to get wrong.
    """
    rng = np.random.default_rng(seed)
    latent = rng.poisson(5.0, size=n_days)
    rows = []
    for station, factor in factors.items():
        counts = np.round(latent * factor).astype(int)
        rows.append(
            pd.DataFrame(
                {
                    "station": station,
                    "date_ord": np.arange(EPOCH, EPOCH + n_days),
                    "groups": counts,
                }
            )
        )
    return pd.concat(rows, ignore_index=True), latent


# --- the pairwise estimators ---------------------------------------------


def test_years_of_matches_the_calendar():
    import datetime as dt

    days = np.array([dt.date(y, 3, 1).toordinal() for y in (1610, 1750, 1899, 2026)])
    assert years_of(days).tolist() == [1610, 1750, 1899, 2026]


@pytest.mark.parametrize("estimator", [HoSc98(), SvSc16(), VHS24()])
def test_each_scaling_recovers_an_exact_factor(estimator):
    # 0.8 rather than 0.5 so the implied k = 1.25 sits inside HoSc98's own
    # 0.6 < k < 1.4 gate; the other two have no such gate.
    observations, _latent = _chain_observations({1: 1.0, 2: 0.8})
    series = station_series(observations)
    estimate = estimator.estimate(
        series[1], series[2], estimator.prepare(series[1]), estimator.prepare(series[2])
    )
    assert estimate.usable
    assert estimate.factor == pytest.approx(1.25, rel=0.1)


def test_hosc98_refuses_a_factor_outside_its_own_gate():
    """0.6 < k < 1.4 is the method's own rule, and it is not symmetric."""
    observations, _latent = _chain_observations({1: 1.0, 2: 0.2})
    series = station_series(observations)
    estimator = HoSc98()
    estimate = estimator.estimate(series[1], series[2], None, None)
    assert estimate.factor > 1.4
    assert not estimate.usable


def test_inverting_an_edge_gives_the_reciprocal():
    estimate = PairEstimate(factor=2.0, relative_sd=0.1, n_days=100, intercept=1.0)
    inverted = VHS24().invert(estimate)
    assert inverted.factor == pytest.approx(0.5)
    assert inverted.intercept == pytest.approx(-0.5)
    assert inverted.relative_sd == pytest.approx(0.1)


def test_svsc16_uses_each_observers_own_days_not_the_common_ones():
    """Its authors flag this as a bias source, so it must not be smoothed away."""
    rng = np.random.default_rng(0)
    n = 1500
    latent = rng.poisson(5.0, size=n)
    days = np.arange(EPOCH, EPOCH + n)
    # Observer 2 only looks on the quietest half of the days.
    quiet = np.argsort(latent)[: n // 2]
    frames = [
        pd.DataFrame({"station": 1, "date_ord": days, "groups": latent}),
        pd.DataFrame(
            {"station": 2, "date_ord": days[np.sort(quiet)], "groups": latent[np.sort(quiet)]}
        ),
    ]
    series = station_series(pd.concat(frames, ignore_index=True))
    estimator = SvSc16()
    factor = estimator.estimate(
        series[1], series[2], estimator.prepare(series[1]), estimator.prepare(series[2])
    ).factor
    # Identical counts on shared days, yet the annual means differ, so k != 1.
    on_common = matched_counts(series[1], series[2])
    assert np.array_equal(on_common[0], on_common[1])
    assert factor > 1.1


def test_robust_fit_ignores_a_gross_outlier():
    source = np.arange(1.0, 50.0)
    target = 2.0 * source
    target[10] = 500.0
    slope, _sd = robust_proportional_fit(target, source)
    assert slope == pytest.approx(2.0, rel=0.05)


# --- chaining -------------------------------------------------------------


def test_chaining_multiplies_factors_along_a_path():
    observations, _latent = _chain_observations({1: 1.0, 2: 0.5, 3: 0.25})
    graph = build_overlap_graph(observations, min_overlap_days=10)
    series = station_series(observations)
    estimator = HoSc98()
    prepared = {s: estimator.prepare(series[s]) for s in series}
    estimates = {}
    for a in series:
        for b in series:
            if a != b:
                estimates[(a, b)] = estimator.estimate(
                    series[a], series[b], prepared[a], prepared[b]
                )
                estimates[(a, b)] = PairEstimate(estimates[(a, b)].factor, 0.1, 100, usable=True)
    scales = chain_from_reference(graph, estimates, reference=1).set_index("station")
    assert scales.loc[1, "factor"] == pytest.approx(1.0)
    assert scales.loc[2, "factor"] == pytest.approx(2.0, rel=0.1)
    assert scales.loc[3, "factor"] == pytest.approx(4.0, rel=0.1)
    assert scales.loc[3, "factor_sd"] > scales.loc[2, "factor_sd"]


def test_uncertainty_grows_along_the_chain():
    """Chained variance accumulates -- the whole reason for the graph diagnostics."""
    observations, _latent = _chain_observations({1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
    graph = build_overlap_graph(observations, min_overlap_days=10)
    estimates = {
        (a, b): PairEstimate(1.0, 0.2, 100, usable=True)
        for a in (1, 2, 3, 4)
        for b in (1, 2, 3, 4)
        if a != b
    }
    # Keep only a path 1-2-3-4 so the chain has to walk it.
    estimates = {k: v for k, v in estimates.items() if abs(k[0] - k[1]) == 1}
    scales = chain_from_reference(graph, estimates, reference=1).set_index("station")
    sds = [scales.loc[s, "factor_sd"] for s in (2, 3, 4)]
    assert sds[0] < sds[1] < sds[2]


# --- the PMF matrix -------------------------------------------------------


def test_the_conversion_matrix_is_column_stochastic():
    rng = np.random.default_rng(0)
    source = rng.poisson(4.0, size=4000)
    target = source + rng.poisson(1.0, size=4000)
    matrix, diagnostics = conversion_matrix(target, source, max_count=20, rng=rng)
    populated = matrix.sum(axis=0) > 0
    assert np.allclose(matrix[:, populated].sum(axis=0), 1.0)
    assert diagnostics["n_populated"] >= 3


def test_the_matrix_conditions_on_the_secondary_as_a157_states():
    """Column = secondary's count, column content = PMF over the reference's.

    This settles a question the research notes recorded as open (the conditioning
    direction of the CEA17 matrix).
    """
    source = np.array([1] * 500 + [5] * 500)
    target = np.array([2] * 500 + [10] * 500)
    matrix, _diagnostics = conversion_matrix(target, source, max_count=12)
    assert matrix[2, 1] == pytest.approx(1.0)
    assert matrix[10, 5] == pytest.approx(1.0)


def test_chained_matrices_compose_like_conditional_distributions():
    rng = np.random.default_rng(1)
    middle = rng.poisson(4.0, size=6000)
    outer = middle * 2
    inner = middle + rng.poisson(1.0, size=6000)
    first, _ = conversion_matrix(outer, middle, max_count=30, rng=rng)
    second, _ = conversion_matrix(middle, inner, max_count=30, rng=rng)
    raw = first @ second
    # Composition leaks mass wherever the upstream matrix has an empty column,
    # so the raw product is sub-stochastic and its implied mean is biased low.
    leaky = raw.sum(axis=0)
    assert ((leaky > 0) & (leaky < 0.99)).any(), "no leak to renormalise; weaken the fixture"

    composed = compose(first, second)
    populated = composed.sum(axis=0) > 0
    assert populated.sum() >= 3
    assert np.allclose(composed[:, populated].sum(axis=0), 1.0)

    # And the bias the renormalisation removes is real, not cosmetic.
    values = np.arange(composed.shape[0])
    leaked_mean = (raw.T @ values)[populated]
    fixed_mean = (composed.T @ values)[populated]
    assert (fixed_mean >= leaked_mean - 1e-9).all()
    assert (fixed_mean > leaked_mean + 1e-6).any()


# --- the acuity methods ---------------------------------------------------


def _reference_areas(n_days=4000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for offset in range(n_days):
        for _ in range(rng.poisson(5.0)):
            rows.append(
                {
                    "group_id": f"g{offset}_{len(rows)}",
                    "date_ord": EPOCH + offset,
                    "area_msh": float(np.exp(rng.normal(4.0, 1.1))),
                }
            )
    return pd.DataFrame(rows)


def test_a_higher_acuity_library_entry_reports_fewer_active_days():
    areas = _reference_areas()
    library = build_library(areas, fingerprint="adf")
    low = library[float(ACUITY_GRID[0])].mean()
    high = library[float(ACUITY_GRID[-1])].mean()
    assert low > high


@pytest.mark.slow
def test_uea21_recovers_a_known_acuity_from_active_day_statistics():
    """The method's whole claim: an observer's own statistics give its acuity."""
    areas = _reference_areas(n_days=6000, seed=2)
    library = build_library(areas, fingerprint="adf")
    days = np.unique(areas["date_ord"].to_numpy())
    for true_acuity in (20.0, 60.0):
        counts = (
            areas[areas["area_msh"] > true_acuity]
            .groupby("date_ord")
            .size()
            .reindex(days, fill_value=0)
            .to_numpy()
        )
        recovered = recover_acuity(days, counts, library, fingerprint="adf")
        assert abs(recovered - true_acuity) <= 10.0


# --- the nulls ------------------------------------------------------------


def test_the_prior_null_ignores_the_data_entirely():
    observations, _latent = _chain_observations({1: 1.0, 2: 2.0})
    result = NULLS["null-prior"](observations)
    assert result.series["gn"].nunique() == 1
    assert (result.series["gn_sd"] > 0).all()


def test_the_segmentation_null_produces_a_series_with_the_right_level():
    observations, _latent = _chain_observations({1: 1.0, 2: 1.0})
    result = NULLS["null-segmentation"](observations, seed=3)
    assert len(result.series) > 1000
    assert result.series["gn"].mean() == pytest.approx(observations["groups"].mean(), rel=0.15)


def test_every_registered_baseline_has_the_same_signature():
    import inspect

    for name, function in BASELINES.items():
        parameters = inspect.signature(function).parameters
        assert "observations" in parameters, name
        assert any(p.kind == p.VAR_KEYWORD for p in parameters.values()), name
