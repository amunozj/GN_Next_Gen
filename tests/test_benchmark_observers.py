"""Synthetic observers: each effect switched on alone, so a panel means something."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from gn_next_gen.benchmark.calendars import calendar_summary, transplant_calendar
from gn_next_gen.benchmark.observers import (
    CASE_STUDY_KINDS,
    ObserverParams,
    draw_population,
    observe,
    observe_one,
    threshold_trajectory,
)
from gn_next_gen.benchmark.truth import Truth

#: Toy worlds start at a real ordinal (1917-01-01). Ordinal 0 is not a date, and
#: anything that formats a year -- blanket statements, for one -- rejects it.
TOY_EPOCH = 700_000


def _toy_truth(n_days=400, groups_per_day=4, area=60.0, seed=0):
    """A flat world: the same number of identical groups every day.

    Flatness is the point -- anything the observer layer does to the counts is
    then visibly the observer's doing and not the Sun's.
    """
    rng = np.random.default_rng(seed)
    rows, meta = [], []
    for offset in range(n_days):
        day = TOY_EPOCH + offset
        for k in range(groups_per_day):
            gid = f"g{day}_{k}"
            rows.append({"group_id": gid, "date_ord": day, "area_msh": area})
            meta.append(
                {
                    "group_id": gid,
                    "emergence_ord": day,
                    "visible_span": 1,
                    "peak_area_msh": area,
                    "donor_survey": "toy",
                    "donor_cycle": 0,
                }
            )
    del rng
    return Truth(
        groups=pd.DataFrame(meta),
        group_days=pd.DataFrame(rows),
        days=np.arange(TOY_EPOCH, TOY_EPOCH + n_days),
        reference_threshold_msh=10.0,
        anchoring="rgo_15_19",
    )


def _toy_days(n_days, start=0):
    return np.arange(TOY_EPOCH + start, TOY_EPOCH + start + n_days)


def _run(params, truth, days, seed=0, activity=None):
    ordered = truth.group_days.sort_values("date_ord")
    return observe_one(
        truth,
        params,
        days,
        np.random.default_rng(seed),
        date_ord=ordered["date_ord"].to_numpy(),
        areas=ordered["area_msh"].to_numpy(),
        activity=activity,
    )


def test_a_lower_threshold_detects_more():
    """The monotonicity every calibration method assumes."""
    truth = _toy_truth()
    days = _toy_days(400)
    sharp = dict(softness=0.1, seeing_sigma=0.0, p_split=0.0, p_merge=0.0)
    low, _ = _run(ObserverParams(1, 20.0, **sharp), truth, days)
    high, _ = _run(ObserverParams(1, 200.0, **sharp), truth, days)
    assert low["groups"].mean() > high["groups"].mean()


def test_a_hard_threshold_is_the_zero_softness_limit():
    """ADF's assumption is a limiting case of this generator, not a rival to it."""
    truth = _toy_truth(area=60.0)
    days = _toy_days(400)
    below, _ = _run(
        ObserverParams(1, 50.0, softness=1e-3, seeing_sigma=0.0, p_split=0.0, p_merge=0.0),
        truth,
        days,
    )
    above, _ = _run(
        ObserverParams(1, 70.0, softness=1e-3, seeing_sigma=0.0, p_split=0.0, p_merge=0.0),
        truth,
        days,
    )
    assert (below["groups"] == 4).all()  # every group cleared a 50 MSH threshold
    assert (above["groups"] == 0).all()  # none cleared 70 MSH


def test_seeing_noise_blurs_a_hard_threshold():
    truth = _toy_truth(area=60.0)
    days = _toy_days(400)
    clean, _ = _run(
        ObserverParams(1, 50.0, softness=1e-3, seeing_sigma=0.0, p_split=0.0, p_merge=0.0),
        truth,
        days,
    )
    blurred, _ = _run(
        ObserverParams(1, 50.0, softness=1e-3, seeing_sigma=0.5, p_split=0.0, p_merge=0.0),
        truth,
        days,
    )
    assert clean["groups"].std() == pytest.approx(0.0, abs=1e-9)
    assert blurred["groups"].std() > 0.5


def test_splitting_inflates_and_merging_deflates_the_count():
    truth = _toy_truth(area=200.0)
    days = _toy_days(400)
    base = dict(softness=0.1, seeing_sigma=0.0)
    plain, _ = _run(ObserverParams(1, 20.0, p_split=0.0, p_merge=0.0, **base), truth, days)
    split, _ = _run(ObserverParams(1, 20.0, p_split=0.5, p_merge=0.0, **base), truth, days, seed=2)
    merge, _ = _run(ObserverParams(1, 20.0, p_split=0.0, p_merge=0.5, **base), truth, days, seed=2)
    assert split["groups"].mean() > plain["groups"].mean()
    assert merge["groups"].mean() < plain["groups"].mean()


def test_mnar_omission_removes_blank_days_and_only_blank_days():
    """ADF removes days at random; this observer removes them because they are
    blank, which is the assumption the Maunder Minimum turns on."""
    truth = _toy_truth(area=10.0)  # far below threshold, so every day is blank
    days = _toy_days(400)
    reported, _ = _run(
        ObserverParams(
            1, 500.0, softness=0.1, seeing_sigma=0.0, p_split=0.0, p_merge=0.0, mnar_omega=0.9
        ),
        truth,
        days,
        activity=np.zeros(400),
    )
    assert len(reported) < 100
    assert (reported["groups"] == 0).all()

    kept, _ = _run(
        ObserverParams(
            1, 500.0, softness=0.1, seeing_sigma=0.0, p_split=0.0, p_merge=0.0, mnar_omega=0.0
        ),
        truth,
        days,
    )
    assert len(kept) == 400


def test_mnar_omission_fades_as_activity_rises():
    truth = _toy_truth(area=10.0)
    days = _toy_days(400)
    quiet, _ = _run(
        ObserverParams(1, 500.0, softness=0.1, seeing_sigma=0.0, mnar_omega=0.9),
        truth,
        days,
        activity=np.zeros(400),
    )
    busy, _ = _run(
        ObserverParams(1, 500.0, softness=0.1, seeing_sigma=0.0, mnar_omega=0.9),
        truth,
        days,
        activity=np.ones(400),
    )
    assert len(busy) > len(quiet)


def test_a_break_steps_the_threshold_and_drift_wanders():
    days = np.arange(TOY_EPOCH, TOY_EPOCH + 4000, 10)
    stepped = threshold_trajectory(
        ObserverParams(
            1, 50.0, break_days=(TOY_EPOCH + 2000,), break_jumps=(-0.5,), break_documented=(False,)
        ),
        days,
        np.random.default_rng(0),
    )
    assert stepped[days < TOY_EPOCH + 2000].std() == pytest.approx(0.0, abs=1e-9)
    assert stepped[days >= TOY_EPOCH + 2000].mean() < stepped[days < TOY_EPOCH + 2000].mean()

    drifting = threshold_trajectory(
        ObserverParams(1, 50.0, drift_sigma=0.3), days, np.random.default_rng(0)
    )
    assert drifting.std() > 0
    assert drifting[0] == pytest.approx(50.0)


def test_only_documented_breaks_reach_the_open_half():
    truth = _toy_truth(n_days=200)
    calendars = {1: _toy_days(200), 2: _toy_days(200)}
    population = [
        ObserverParams(
            1, 50.0, break_days=(TOY_EPOCH + 100,), break_jumps=(-0.3,), break_documented=(True,)
        ),
        ObserverParams(
            2, 50.0, break_days=(TOY_EPOCH + 100,), break_jumps=(-0.3,), break_documented=(False,)
        ),
    ]
    network = observe(truth, population, calendars, np.random.default_rng(0))
    assert network.metadata["station"].tolist() == [1]


def test_a_copier_reproduces_its_partner_rather_than_drawing_afresh():
    """Correlated observer error: nothing in Step 4 models it, so it has to be here."""
    truth = _toy_truth(area=200.0, n_days=300)
    calendars = {1: _toy_days(300), 2: _toy_days(300)}
    # The threshold sits *at* the group area on purpose: detection is then a coin
    # flip per group and the daily count genuinely varies.  With a threshold far
    # below the areas both observers would report 4 every day, and a correlation
    # between two constants measures nothing.
    sharp = dict(softness=0.2, seeing_sigma=0.3, p_split=0.0, p_merge=0.0)
    population = [
        ObserverParams(1, 200.0, **sharp),
        ObserverParams(2, 200.0, copies_of=1, **sharp),
    ]
    network = observe(truth, population, calendars, np.random.default_rng(1))
    wide = network.observations.pivot(index="date_ord", columns="station", values="groups")
    copied = wide.corr().loc[1, 2]

    independent = observe(
        truth,
        [ObserverParams(1, 200.0, **sharp), ObserverParams(2, 200.0, **sharp)],
        calendars,
        np.random.default_rng(1),
    )
    wide2 = independent.observations.pivot(index="date_ord", columns="station", values="groups")
    independently = wide2.corr().loc[1, 2]
    assert copied > 0.7
    assert copied > independently + 0.5


def test_blanket_statements_replace_daily_counts_entirely():
    truth = _toy_truth(n_days=800, area=200.0)
    calendars = {1: _toy_days(800)}
    network = observe(
        truth, [ObserverParams(1, 40.0, blanket=True)], calendars, np.random.default_rng(0)
    )
    assert network.observations.empty
    assert not network.statements.empty
    assert set(network.statements["kind"]) <= {"literal_none", "rhetorical"}


def test_a_drawn_population_spans_the_calendars_it_is_given():
    calendars = {s: np.arange(s * 10, s * 10 + 500) for s in range(1, 40)}
    population = draw_population(calendars, np.random.default_rng(0))
    assert [p.station for p in population] == sorted(calendars)
    assert all(p.base_threshold_msh > 0 for p in population)
    assert any(p.copies_of is not None for p in population)
    assert any(p.blanket for p in population)


def test_transplanting_a_calendar_blanks_unobserved_days():
    series = np.arange(5.0)
    observed = np.array([True, False, True, False, True])
    out = transplant_calendar(series, observed)
    assert np.isnan(out[[1, 3]]).all()
    assert out[[0, 2, 4]].tolist() == [0.0, 2.0, 4.0]


def test_calendar_summary_is_sorted_by_first_day():
    summary = calendar_summary({5: np.array([10, 11]), 2: np.array([1, 2, 3])})
    assert summary["station"].tolist() == [2, 5]
    assert summary["n_days"].tolist() == [3, 2]


def test_all_ten_case_studies_are_placed_in_a_real_scenario():
    from gn_next_gen.benchmark.scenarios import SCENARIOS, build_realization

    scenario = SCENARIOS["S1"]
    _truth, network, _config = build_realization(scenario, seed=1)
    # "reference" is the anchor observer the scenario adds (ADR-003); it is not
    # a case study and is not drawn from the population.
    kinds = set(network.parameters["kind"]) - {"population", "reference"}
    assert kinds == set(CASE_STUDY_KINDS)


def test_perfect_observers_report_the_estimand_exactly():
    """S5's premise: with detection error off, the only error left is the calendar."""
    from gn_next_gen.benchmark.scenarios import _perfect

    truth = _toy_truth(area=60.0, n_days=200)
    params = _perfect(ObserverParams(1, 999.0, drift_sigma=0.5), truth.reference_threshold_msh)
    reported, _ = _run(params, truth, _toy_days(200))
    expected = truth.reference_counts().reindex(reported["date_ord"]).to_numpy()
    assert np.array_equal(reported["groups"].to_numpy(), expected)


def test_dates_outside_an_observers_calendar_are_never_reported():
    truth = _toy_truth(n_days=300, area=200.0)
    days = _toy_days(70, start=50)
    reported, _ = _run(ObserverParams(1, 40.0), truth, days)
    assert reported["date_ord"].min() >= TOY_EPOCH + 50
    assert reported["date_ord"].max() <= TOY_EPOCH + 119


def test_a_day_with_no_visible_group_is_reported_as_zero_not_dropped():
    """The empty-slice case that np.add.reduceat could not express."""
    truth = Truth(
        groups=pd.DataFrame(
            {
                "group_id": ["g"],
                "emergence_ord": [TOY_EPOCH + 5],
                "visible_span": [1],
                "peak_area_msh": [200.0],
                "donor_survey": ["toy"],
                "donor_cycle": [0],
            }
        ),
        group_days=pd.DataFrame(
            {"group_id": ["g"], "date_ord": [TOY_EPOCH + 5], "area_msh": [200.0]}
        ),
        days=_toy_days(10),
        reference_threshold_msh=10.0,
        anchoring="rgo_15_19",
    )
    reported, _ = _run(
        ObserverParams(1, 20.0, softness=0.1, seeing_sigma=0.0, p_split=0.0, p_merge=0.0),
        truth,
        _toy_days(10),
    )
    assert len(reported) == 10
    assert reported.set_index("date_ord")["groups"].loc[TOY_EPOCH + 5] == 1
    assert reported.set_index("date_ord")["groups"].loc[TOY_EPOCH + 9] == 0


def test_the_toy_world_has_a_sane_date_type():
    assert dt.date.fromordinal(700_000).year == 1917
