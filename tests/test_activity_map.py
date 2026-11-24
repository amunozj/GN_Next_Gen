"""The GN -> activity-level map, which is how detection stays activity-conditioned
before 1874, where there are no areas to measure an activity level from."""

from __future__ import annotations

import numpy as np
import pytest

from gn_next_gen.models.activity_map import (
    ActivityMap,
    fit_activity_map,
    smooth_counts,
)


def toy_map() -> ActivityMap:
    return ActivityMap(
        survey="toy",
        amplitude=0.04,
        exponent=1.08,
        r_log=0.99,
        residual_sd=0.08,
        gn_lo=0.2,
        gn_hi=10.0,
        n_days=1000,
        threshold_msh=28.0,
    )


def test_map_passes_through_the_origin():
    """A grand minimum has no groups, so it must have no area.

    A polynomial or a spline fitted over 1874-2026 gives no such guarantee; the
    power law does, and that is why it is the form.
    """
    assert float(toy_map()(0.0)) == 0.0
    assert float(toy_map()(1e-6)) < 1e-6


def test_map_is_monotone_and_slightly_superlinear():
    mapping = toy_map()
    counts = np.array([0.1, 0.5, 1.0, 3.0, 9.0])
    values = mapping(counts)
    assert np.all(np.diff(values) > 0)
    # Doubling the count more than doubles the area, because an active Sun has
    # bigger groups as well as more of them.
    assert float(mapping(2.0)) > 2.0 * float(mapping(1.0))


def test_map_refuses_negative_counts_silently_rather_than_returning_nan():
    assert float(toy_map()(-1.0)) == 0.0


def test_extrapolating_flags_the_range_it_was_not_measured_on():
    mapping = toy_map()
    assert bool(mapping.extrapolating(0.05))
    assert not bool(mapping.extrapolating(1.0))
    assert bool(mapping.extrapolating(50.0))


def test_smooth_counts_treats_a_gap_as_missing_not_as_spotless():
    """A run of unobserved days must not read as a run of zero-group days."""
    days = np.array([0, 1, 2, 500, 501, 502])
    counts = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    smoothed = smooth_counts(days, counts, sigma_days=5.0)
    assert float(smoothed.loc[1]) == pytest.approx(5.0, rel=1e-6)
    assert float(smoothed.loc[501]) == pytest.approx(5.0, rel=1e-6)
    # The middle is NaN because nothing was observed anywhere near it.
    assert not np.isfinite(float(smoothed.loc[250]))


def test_smooth_counts_covers_the_requested_span():
    smoothed = smooth_counts(
        np.array([10, 11, 12]), np.array([1.0, 2.0, 3.0]), sigma_days=2.0, first=5, last=20
    )
    assert smoothed.index.min() == 5
    assert smoothed.index.max() == 20


@pytest.mark.slow
@pytest.mark.parametrize("survey,threshold", [("rgo", 28.08), ("debrecen", 20.42)])
def test_fitted_map_is_tight_and_superlinear_on_real_data(survey, threshold):
    """Measured twice, independently, a century apart -- and they agree.

    RGO gives ``AL = 0.0382 GN^1.0742`` and Debrecen ``0.0350 GN^1.0906``, both at
    ``r > 0.996`` in log-log.  That agreement is what licenses using the map
    outside the era it was fitted on.
    """
    mapping = fit_activity_map(survey, threshold)
    assert mapping.r_log > 0.99
    assert mapping.residual_sd < 0.12
    assert 1.0 < mapping.exponent < 1.2
    assert 0.03 < mapping.amplitude < 0.05
    assert mapping.n_days > 10_000


@pytest.mark.slow
def test_the_two_surveys_agree_within_a_tenth():
    rgo = fit_activity_map("rgo", 28.08)
    debrecen = fit_activity_map("debrecen", 20.42)
    assert abs(rgo.exponent - debrecen.exponent) < 0.05
    # Amplitudes differ because the two anchors cut at different thresholds, so
    # "one group" means a slightly different object in each.
    assert abs(np.log(rgo.amplitude) - np.log(debrecen.amplitude)) < 0.15


@pytest.mark.slow
def test_map_cache_round_trips():
    first = fit_activity_map("rgo", 28.08, force=True)
    second = fit_activity_map("rgo", 28.08)
    assert first.as_row() == second.as_row()
