"""The detection ratio and the emission tables.

``k_o`` multiplies every one of an observer's counts, so an error in it is an
error of the same size in that observer's contribution to the series.  These
tests pin the three properties that make the estimand well defined at all:
``k = 1`` exactly at the anchor's own threshold, monotonicity in the threshold,
and emission rows that sum to one without leaking mass.
"""

from __future__ import annotations

import numpy as np
import pytest

from gn_next_gen.models import emissions as em
from gn_next_gen.reference.area_law import load_upstream_parameters

MAX = em.MAX_COUNT


def tiny_grid(survey: str = "rgo", source: str = "parametric") -> em.DetectionGrid:
    """A small grid, so the fast suite does not pay for a 160x64 quadrature."""
    return em.build_detection_grid(
        survey,
        thresholds=np.exp(np.linspace(np.log(2.0), np.log(400.0), 24)),
        activities=np.linspace(0.01, 0.4, 8),
        source=source,
    )


# --------------------------------------------------------------------------
# Emission tables
# --------------------------------------------------------------------------


def test_identical_to_the_anchor_is_the_identity():
    """An observer declared identical to the anchor reports the estimand exactly.

    The benchmark's reference observer depends on this: ``Truth.reference_counts``
    is what it must reproduce, day for day.
    """
    table = em.emission_matrix(1.0, 0.0, max_count=12)
    assert np.allclose(table, np.eye(13), atol=1e-12)


@pytest.mark.parametrize("k", [0.1, 0.5, 0.9, 1.0, 1.5, 3.0])
@pytest.mark.parametrize("split", [0.0, 0.05, 0.3])
def test_rows_are_stochastic(k, split):
    table = em.emission_matrix(k, split, max_count=MAX)
    assert np.all(table >= 0.0)
    assert np.allclose(table.sum(axis=1), 1.0, atol=1e-12)


@pytest.mark.parametrize("k", [0.25, 0.6, 1.0])
@pytest.mark.parametrize("split", [0.0, 0.1, 0.25])
def test_mean_is_k_times_n_times_one_plus_split(k, split):
    """``k`` and ``s`` are separate mechanisms with a shared, exact mean."""
    table = em.emission_matrix(k, split, max_count=MAX)
    counts = np.arange(MAX + 1)
    for n in (1, 5, 10):
        got = float((table[n] * counts).sum())
        assert got == pytest.approx(k * n * (1.0 + split), rel=1e-9)


def test_thinning_alone_cannot_report_more_than_the_anchor_counted():
    """With no splitting and k <= 1 every reported count satisfies y <= N."""
    table = em.emission_matrix(0.5, 0.0, max_count=20)
    for n in range(21):
        assert table[n, n + 1 :].sum() == pytest.approx(0.0, abs=1e-12)


def test_splitting_is_what_lets_y_exceed_n():
    table = em.emission_matrix(1.0, 0.3, max_count=MAX)
    assert table[5, 6:].sum() > 0.1


def test_a_finer_observer_sees_more_than_the_anchor():
    table = em.emission_matrix(2.0, 0.0, max_count=MAX)
    counts = np.arange(MAX + 1)
    assert float((table[4] * counts).sum()) == pytest.approx(8.0, rel=1e-9)
    assert table[0, 0] == pytest.approx(1.0)  # no groups to double


def test_upper_tail_is_folded_in_not_dropped():
    """Step 2's recorded trap: composing PMF matrices leaks mass and biases low.

    Here the truncation is handled by accumulating the lost tail in the last
    column, so no row ever sums to less than one even when ``k`` pushes mass past
    the truncation.
    """
    table = em.emission_matrix(3.0, 0.5, max_count=10)
    assert np.allclose(table.sum(axis=1), 1.0, atol=1e-12)
    assert table[9, -1] > 0.0


def test_log_library_matches_the_matrices_it_tabulates():
    log_table, k_values, split_values = em.log_emission_library(
        k_values=np.array([0.4, 1.0, 2.0]),
        split_values=np.array([0.0, 0.1]),
        max_count=8,
    )
    assert log_table.shape == (3, 2, 9, 9)
    for i, k in enumerate(k_values):
        for j, split in enumerate(split_values):
            direct = em.emission_matrix(float(k), float(split), max_count=8)
            assert np.allclose(np.exp(log_table[i, j]), direct, atol=1e-12)


def test_k_grid_brackets_every_plausible_observer():
    grid = em.k_grid()
    assert grid[0] <= 0.02 + 1e-12
    assert grid[-1] >= 5.0 - 1e-12
    assert np.all(np.diff(grid) > 0)


# --------------------------------------------------------------------------
# The area law at an activity level
# --------------------------------------------------------------------------


def test_activity_scaling_leaves_the_weibull_component_alone():
    """Step 1 §2.1 found no detectable activity dependence in (k, lambda), with
    signs disagreeing across surveys -- which is what licenses holding them fixed."""
    base = load_upstream_parameters("rgo")
    scaled = em.params_at_activity(base, 0.3, 0.1)
    assert scaled.k_W == base.k_W
    assert scaled.log_lambda_W == base.log_lambda_W


def test_activity_scaling_is_the_identity_at_the_reference_except_for_grain():
    """At the reference activity only the per-group-day shift should move mu."""
    base = load_upstream_parameters("rgo")
    scaled = em.params_at_activity(base, 0.1, 0.1)
    assert scaled.mu_LN == pytest.approx(base.mu_LN + em.GRAIN_SHIFT_MU["rgo"])
    assert scaled.sigma_LN == pytest.approx(base.sigma_LN)
    assert scaled.c_LN == pytest.approx(base.c_LN)


def test_the_grain_shift_is_negative_which_overturned_the_expectation():
    """Step 1 §2.4: the per-day law is *smaller*, not larger.

    The plan for Step 1 expected size-biasing to make it larger (big groups live
    longer, so they appear on more days).  The measurement went the other way,
    because the per-group area is each group's *peak* while a detection integral
    sees instantaneous areas.  A positive shift here would be the old expectation
    silently reinstated.
    """
    assert all(shift < 0 for shift in em.GRAIN_SHIFT_MU.values())
    assert em.GRAIN_SHIFT_DEFAULT < 0


def test_mu_rises_with_activity():
    base = load_upstream_parameters("rgo")
    quiet = em.params_at_activity(base, 0.02, 0.1)
    busy = em.params_at_activity(base, 0.40, 0.1)
    assert busy.mu_LN > quiet.mu_LN


def test_params_at_activity_vectorizes():
    base = load_upstream_parameters("rgo")
    many = em.params_at_activity(base, np.array([0.05, 0.1, 0.2]), 0.1)
    assert isinstance(many, list) and len(many) == 3
    assert many[0].mu_LN < many[1].mu_LN < many[2].mu_LN


# --------------------------------------------------------------------------
# Bilinear lookup
# --------------------------------------------------------------------------


def test_bilinear_is_exact_on_nodes_and_clamps_outside():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 10.0])
    table = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    assert em.bilinear(x, y, table, np.array(1.0), np.array(10.0)) == pytest.approx(3.0)
    assert em.bilinear(x, y, table, np.array(0.5), np.array(0.0)) == pytest.approx(1.0)
    # Outside the grid, the edge value rather than an extrapolation.
    assert em.bilinear(x, y, table, np.array(-5.0), np.array(0.0)) == pytest.approx(0.0)
    assert em.bilinear(x, y, table, np.array(99.0), np.array(99.0)) == pytest.approx(5.0)


def test_bilinear_differentiates_under_jax():
    import jax
    import jax.numpy as jnp

    x = jnp.array([0.0, 1.0, 2.0])
    y = jnp.array([0.0, 1.0])
    table = jnp.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]])

    def f(value):
        return em.bilinear(x, y, table, value, jnp.array(0.5))

    assert float(jax.grad(f)(jnp.array(0.5))) == pytest.approx(2.0, rel=1e-6)


# --------------------------------------------------------------------------
# The detection grid
# --------------------------------------------------------------------------


@pytest.mark.parametrize("survey", ["rgo", "debrecen"])
def test_k_is_exactly_one_at_the_anchor_threshold(survey):
    """The zero that pins the absolute scale (ADR-003).

    Regression test with a provenance: before ``T_ref`` was forced onto the
    threshold grid, interpolating near it returned 0.9988 -- a 0.1% scale error
    injected at the anchor, where the whole series is pinned.
    """
    grid = tiny_grid(survey)
    for activity in (0.01, 0.05, 0.2, 0.4):
        got = float(np.exp(grid(grid.reference_threshold_msh, activity)))
        assert got == pytest.approx(1.0, abs=1e-12)


def test_k_falls_monotonically_as_the_observer_gets_coarser():
    grid = tiny_grid()
    assert np.all(np.diff(grid.log_k, axis=0) <= 1e-9)
    fine = float(np.exp(grid(5.0, grid.activity_ref)))
    coarse = float(np.exp(grid(300.0, grid.activity_ref)))
    assert fine > 1.0 > coarse


def test_a_deeper_survey_is_the_finer_instrument_under_the_others_anchor():
    """Debrecen cuts at 20.4 MSH and RGO at 28.1, so each is finer than the other."""
    under_rgo = tiny_grid("rgo")
    under_debrecen = tiny_grid("debrecen")
    assert float(np.exp(under_rgo(20.42, under_rgo.activity_ref))) > 1.0
    assert float(np.exp(under_debrecen(28.08, under_debrecen.activity_ref))) < 1.0


def test_parametric_and_empirical_sources_are_both_available():
    parametric = tiny_grid(source="parametric")
    assert parametric.meta["source"] == "parametric"
    assert parametric.meta["empirical_fraction"] == 0.0


def test_unknown_source_is_refused():
    with pytest.raises(ValueError, match="unknown source"):
        em.build_detection_grid(
            "rgo",
            thresholds=np.array([10.0, 30.0]),
            activities=np.array([0.05, 0.1]),
            source="vibes",
        )


@pytest.mark.slow
def test_empirical_grid_reproduces_the_catalogues_own_count_ratio():
    """The reason the empirical branch exists (PI, 2026-11-24).

    ``k(T)`` measured as a count ratio in the reference catalogue is what the
    detection model is *trying* to be, so the grid has to reproduce it.  The
    parametric law does not: measured on RGO it under-predicts ``k`` for coarse
    observers by 4% at 40 MSH and 13% at 200 MSH, always in the same direction.
    """
    from gn_next_gen.io.reference import daily_counts

    grid = em.detection_grid("rgo")
    base = daily_counts(grid.reference_threshold_msh, ["rgo"])
    for threshold, tolerance in [(20.0, 0.02), (40.0, 0.02), (60.0, 0.02), (100.0, 0.03)]:
        other = daily_counts(threshold, ["rgo"])
        merged = base.merge(other, on=["survey", "date_ord"], suffixes=("_ref", "_o"))
        empirical = merged["n_groups_o"].sum() / merged["n_groups_ref"].sum()
        got = float(np.exp(grid(threshold, grid.activity_ref)))
        assert got == pytest.approx(empirical, rel=tolerance)


@pytest.mark.slow
def test_empirical_grid_extrapolates_below_the_catalogues_coverage():
    """Below the reference's own activity range the law takes over, continuously.

    That range stops well above the Maunder Minimum, which is precisely the era
    the step has to reach, so something parametric has to speak out there -- and
    ``meta`` records that it is speaking.
    """
    grid = em.detection_grid("rgo")
    assert grid.meta["source"] == "empirical"
    assert 0.0 < grid.meta["coverage_lo"] < grid.meta["coverage_hi"]
    assert grid.activity.min() < grid.meta["coverage_lo"]

    boundary = grid.meta["coverage_lo"]
    inside = float(grid(60.0, boundary * 1.02))
    outside = float(grid(60.0, boundary * 0.5))
    # Continuous across the boundary rather than stepping.
    assert abs(inside - float(grid(60.0, boundary))) < 0.05
    assert np.isfinite(outside)


@pytest.mark.slow
def test_detection_grid_cache_round_trips_exactly():
    first = em.detection_grid("rgo", force=True)
    second = em.detection_grid("rgo")
    assert np.array_equal(first.log_k, second.log_k)
    assert np.array_equal(first.activity, second.activity)
    assert second.meta["source"] == first.meta["source"]
    assert second.reference_threshold_msh == first.reference_threshold_msh


@pytest.mark.slow
def test_reference_activity_is_positive_and_survey_specific():
    values = {survey: em.reference_activity(survey) for survey in ("rgo", "debrecen", "kmas")}
    assert all(value > 0 for value in values.values())
    assert len(set(np.round(list(values.values()), 6))) == 3
