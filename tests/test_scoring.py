"""The scorer, on cases whose right answer is known by construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from gn_next_gen.baselines.base import CalibrationResult
from gn_next_gen.benchmark.scoring import (
    COVERAGE_LEVELS,
    align,
    break_detection_skill,
    crps_gaussian,
    cycle_boundaries,
    cycle_errors,
    maunder_level,
    scale_errors,
    score,
    summarise_case_studies,
    trend_errors,
    uncertainty_scores,
)
from gn_next_gen.benchmark.truth import Truth

EPOCH = 700_000


def _truth_from(counts: np.ndarray, start: int = EPOCH, threshold: float = 10.0) -> Truth:
    """A truth whose reference counts are exactly ``counts``."""
    rows, meta = [], []
    for offset, n in enumerate(counts):
        for k in range(int(n)):
            gid = f"g{offset}_{k}"
            rows.append({"group_id": gid, "date_ord": start + offset, "area_msh": 100.0})
            meta.append(
                {
                    "group_id": gid,
                    "emergence_ord": start + offset,
                    "visible_span": 1,
                    "peak_area_msh": 100.0,
                    "donor_survey": "toy",
                    "donor_cycle": 0,
                }
            )
    return Truth(
        groups=pd.DataFrame(
            meta,
            columns=[
                "group_id",
                "emergence_ord",
                "visible_span",
                "peak_area_msh",
                "donor_survey",
                "donor_cycle",
            ],
        ),
        group_days=pd.DataFrame(rows, columns=["group_id", "date_ord", "area_msh"]),
        days=np.arange(start, start + len(counts)),
        reference_threshold_msh=threshold,
        anchoring="rgo_15_19",
    )


def _result(days, gn, sd, name="toy"):
    return CalibrationResult(
        series=pd.DataFrame({"date_ord": days, "gn": gn, "gn_sd": sd}),
        observers=pd.DataFrame(
            columns=["station", "factor", "factor_sd", "recovered_threshold_msh", "reached"]
        ),
        name=name,
    )


def test_a_perfect_reconstruction_scores_zero_error():
    counts = np.random.default_rng(0).poisson(4.0, size=3000)
    truth = _truth_from(counts)
    days = truth.days
    result = _result(days, counts.astype(float), np.full(len(days), 0.5))
    aligned = align(truth, result)
    errors = scale_errors(aligned)
    assert errors["bias_daily"] == pytest.approx(0.0, abs=1e-12)
    assert errors["crmse_daily"] == pytest.approx(0.0, abs=1e-12)
    assert errors["crmse_yearly"] == pytest.approx(0.0, abs=1e-12)


def test_a_constant_offset_is_recovered_exactly_as_bias_not_as_scatter():
    counts = np.random.default_rng(1).poisson(4.0, size=3000)
    truth = _truth_from(counts)
    result = _result(truth.days, counts + 1.5, np.full(len(counts), 0.5))
    errors = scale_errors(align(truth, result))
    assert errors["bias_daily"] == pytest.approx(1.5)
    assert errors["crmse_daily"] == pytest.approx(0.0, abs=1e-12)


def test_crps_of_a_point_forecast_is_the_absolute_error():
    """As sigma -> 0 the CRPS of a Gaussian collapses onto |y - mu|."""
    truth = np.array([3.0, 5.0])
    mean = np.array([2.0, 5.5])
    tiny = np.full(2, 1e-6)
    assert crps_gaussian(truth, mean, tiny) == pytest.approx(np.abs(truth - mean), abs=1e-5)


def test_a_correctly_specified_ensemble_has_nominal_coverage():
    """The test the whole coverage figure rests on."""
    rng = np.random.default_rng(3)
    n_years = 4000
    counts = np.repeat(rng.poisson(5.0, size=n_years), 1)
    truth = _truth_from(counts)
    # One day per year keeps the yearly aggregation a no-op, so the nominal
    # Gaussian is exactly the predictive the scorer is handed.
    days = truth.days
    sigma = 1.0
    estimate = counts + rng.normal(0.0, sigma, size=len(counts))
    aligned = align(truth, _result(days, estimate, np.full(len(days), sigma)))
    aligned["year"] = np.arange(len(aligned))  # one "year" per day
    scores = uncertainty_scores(aligned)
    for level in COVERAGE_LEVELS:
        assert scores[f"coverage_{int(level * 100)}"] == pytest.approx(level, abs=0.03)
    assert scores["pit_ks"] < 0.05


def test_an_overconfident_ensemble_fails_coverage():
    rng = np.random.default_rng(4)
    counts = rng.poisson(5.0, size=3000)
    truth = _truth_from(counts)
    estimate = counts + rng.normal(0.0, 1.0, size=len(counts))
    aligned = align(truth, _result(truth.days, estimate, np.full(len(counts), 0.2)))
    aligned["year"] = np.arange(len(aligned))
    scores = uncertainty_scores(aligned)
    assert scores["coverage_90"] < 0.5
    assert scores["pit_ks"] > 0.1


def test_coverage_is_reported_at_all_four_levels():
    assert set(COVERAGE_LEVELS) == {0.50, 0.68, 0.90, 0.95}


def test_cycle_boundaries_find_the_minima_of_a_sinusoid():
    years = np.arange(1900, 1990)
    series = pd.Series(5 + 4 * np.sin(2 * np.pi * (years - 1900) / 11.0), index=years)
    cycles = cycle_boundaries(series)
    assert 6 <= len(cycles) <= 9
    lengths = [end - start for start, end in cycles]
    assert 9 <= np.median(lengths) <= 13


def test_amplitude_slope_detects_a_method_that_amplifies_strong_cycles():
    """A157's headline: the linear scalings skew strong against weak cycles.

    The scorer has to be able to see that, so here it is fed exactly that
    behaviour and must report a positive slope.
    """
    rng = np.random.default_rng(5)
    years = np.arange(1900, 2000)
    amplitude = np.where((years // 11) % 2 == 0, 2.0, 8.0)
    truth_yearly = amplitude * (1 + 0.5 * np.sin(2 * np.pi * (years - 1900) / 11.0))
    skewed = truth_yearly * (0.8 + 0.06 * truth_yearly)

    aligned = pd.DataFrame(
        {
            "date_ord": np.arange(len(years)),
            "truth": truth_yearly,
            "estimate": skewed,
            "sd": 1.0,
            "year": years,
            "month": years * 12,
        }
    )
    del rng
    errors = cycle_errors(aligned)
    assert errors["n_cycles"] >= 3
    assert errors["amplitude_slope"] > 0

    faithful = aligned.assign(estimate=aligned["truth"])
    assert cycle_errors(faithful)["amplitude_slope"] == pytest.approx(0.0, abs=1e-6)


def test_trend_error_is_zero_for_a_faithful_reconstruction_and_signed_otherwise():
    years = np.arange(1700, 2000)
    truth_yearly = 3.0 + 0.002 * (years - 1700)
    aligned = pd.DataFrame(
        {
            "date_ord": np.arange(len(years)),
            "truth": truth_yearly,
            "estimate": truth_yearly,
            "sd": 1.0,
            "year": years,
            "month": years * 12,
        }
    )
    assert trend_errors(aligned)["trend_error_1700"] == pytest.approx(0.0, abs=1e-9)

    steeper = aligned.assign(estimate=3.0 + 0.004 * (years - 1700))
    assert trend_errors(steeper)["trend_error_1700"] == pytest.approx(0.2, abs=1e-6)


def test_maunder_level_reads_the_right_window():
    years = np.arange(1600, 1800)
    truth_yearly = np.where((years >= 1645) & (years <= 1715), 0.1, 4.0)
    aligned = pd.DataFrame(
        {
            "date_ord": np.arange(len(years)),
            "truth": truth_yearly,
            "estimate": truth_yearly * 2,
            "sd": 1.0,
            "year": years,
            "month": years * 12,
        }
    )
    level = maunder_level(aligned)
    assert level["maunder_truth"] == pytest.approx(0.1)
    assert level["maunder_error"] == pytest.approx(0.1)


def test_break_detection_is_nan_when_a_method_detects_nothing():
    """None of the seven ported baselines detects breaks, and that is the point."""
    truth_breaks = pd.DataFrame({"station": [1, 2], "date_ord": [EPOCH, EPOCH + 100]})
    skill = break_detection_skill(None, truth_breaks)
    assert all(np.isnan(v) for v in skill.values())


def test_break_detection_rewards_a_hit_and_penalises_a_false_alarm():
    truth_breaks = pd.DataFrame({"station": [1, 2], "date_ord": [EPOCH, EPOCH + 10_000]})
    detections = pd.DataFrame({"station": [1, 3], "date_ord": [EPOCH + 30, EPOCH + 50]})
    skill = break_detection_skill(detections, truth_breaks)
    assert skill["break_pod"] == pytest.approx(0.5)
    assert skill["break_far"] == pytest.approx(0.5)


def test_score_returns_one_flat_row_per_cell():
    counts = np.random.default_rng(6).poisson(4.0, size=4000)
    truth = _truth_from(counts)
    result = _result(truth.days, counts + 0.2, np.full(len(counts), 0.6), name="toy")
    parameters = pd.DataFrame(
        {"station": [1], "base_threshold_msh": [30.0], "kind": ["population"]}
    )
    row = score(truth, result, parameters, scenario="S1", seed=7)
    assert row["method"] == "toy"
    assert row["scenario"] == "S1"
    assert row["seed"] == 7
    assert row["anchoring"] == "rgo_15_19"
    assert row["bias_daily"] == pytest.approx(0.2)
    assert all(not isinstance(v, (list, dict, pd.DataFrame)) for v in row.values())


def test_scoring_an_empty_reconstruction_does_not_raise():
    truth = _truth_from(np.ones(100, dtype=int))
    empty = _result(np.array([], dtype=np.int64), np.array([]), np.array([]))
    row = score(truth, empty, pd.DataFrame(columns=["station", "base_threshold_msh", "kind"]))
    assert row["n_days_scored"] == 0


def test_the_gaussian_crps_matches_a_monte_carlo_estimate():
    rng = np.random.default_rng(8)
    mean, sd, observed = 3.0, 1.5, 4.2
    draws = rng.normal(mean, sd, size=400_000)
    empirical = np.mean(np.abs(draws - observed)) - 0.5 * np.mean(
        np.abs(draws[:200_000] - draws[200_000:])
    )
    analytic = float(crps_gaussian(np.array([observed]), np.array([mean]), np.array([sd]))[0])
    assert analytic == pytest.approx(empirical, rel=0.02)
    assert analytic == pytest.approx(
        sd
        * (
            (observed - mean) / sd * (2 * norm.cdf((observed - mean) / sd) - 1)
            + 2 * norm.pdf((observed - mean) / sd)
            - 1 / np.sqrt(np.pi)
        ),
        rel=1e-9,
    )


def _case_rows(kind, method, ratios, acuities, true_acuity=30.0):
    return pd.DataFrame(
        {
            "scenario": "S1",
            "seed": range(len(ratios)),
            "method": method,
            "station": 7,
            "kind": kind,
            "true_threshold_msh": true_acuity,
            "recovered_threshold_msh": acuities,
            "factor_ratio": ratios,
        }
    )


def test_the_case_study_summary_collapses_replicates_to_median_and_range():
    cases = _case_rows("copier", "UEA21", [1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
    row = summarise_case_studies(cases).iloc[0]

    assert row["kind"] == "copier" and row["method"] == "UEA21"
    assert row["true_threshold_msh"] == 30.0
    assert row["scale_n_replicates"] == 4 and row["scale_n_finite"] == 4
    assert row["scale_median"] == pytest.approx(2.5)
    assert (row["scale_q1"], row["scale_q3"]) == pytest.approx((1.75, 3.25))
    assert (row["scale_min"], row["scale_max"]) == pytest.approx((1.0, 4.0))
    assert row["acuity_median"] == pytest.approx(25.0)


def test_the_case_study_summary_counts_the_replicates_a_method_answered():
    """A method that answers rarely is a different failure from one that answers badly."""
    cases = _case_rows(
        "leaf-node",
        "CEA17",
        [np.nan, 0.0, -1.0, 2.0],
        [np.nan, np.nan, np.nan, 50.0],
    )
    row = summarise_case_studies(cases).iloc[0]

    # NaN, zero and negative scale factors are all "no usable answer".
    assert row["scale_n_finite"] == 1 and row["scale_n_replicates"] == 4
    assert row["scale_median"] == pytest.approx(2.0)
    assert row["acuity_n_finite"] == 1 and row["acuity_median"] == pytest.approx(50.0)


def test_the_case_study_summary_reports_off_by_in_whichever_direction_the_method_erred():
    """Half the truth and twice the truth are both 'off by 2x'; the worst case is the extreme."""
    cases = _case_rows("zero-omitter", "HoSc98", [0.5, 0.25, 4.0], [np.nan] * 3)
    row = summarise_case_studies(cases).iloc[0]

    assert row["scale_median"] == pytest.approx(0.5)
    assert row["scale_off_by_median"] == pytest.approx(2.0)
    assert row["scale_off_by_worst"] == pytest.approx(4.0)
    assert row["acuity_n_finite"] == 0 and np.isnan(row["acuity_median"])


def test_the_case_study_summary_keeps_the_ten_observers_in_their_published_order():
    cases = pd.concat(
        [
            _case_rows("blanket-statements", "GRP", [1.0], [1.0]),
            _case_rows("clean-modern", "GRP", [1.0], [1.0]),
            _case_rows("clean-modern", "CEA17", [1.0], [1.0]),
        ]
    )
    summary = summarise_case_studies(cases)

    assert list(summary["kind"]) == ["clean-modern", "clean-modern", "blanket-statements"]
    assert list(summary["method"]) == ["CEA17", "GRP", "GRP"]
