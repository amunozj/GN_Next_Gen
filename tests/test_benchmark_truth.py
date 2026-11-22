"""The synthetic worlds: what they must reproduce, and what they must not leak."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from gn_next_gen.benchmark.generators.replay import (
    DONOR_SURVEYS,
    build_donor_library,
    replay_truth,
)
from gn_next_gen.benchmark.truth import ANCHORINGS, Truth, anchor_threshold, read_truth, write_truth


def _ord(year, month=1, day=1):
    return dt.date(year, month, day).toordinal()


@pytest.fixture(scope="module")
def rgo_library():
    return build_donor_library("rgo")


@pytest.fixture(scope="module")
def small_truth():
    return replay_truth(_ord(1700), _ord(1900), seed=11)


def test_the_estimand_is_a_reference_equivalent_count_not_a_raw_count():
    """ADR-003: counts alone cannot identify absolute scale, so the anchor's own
    completeness floor defines what is being estimated."""
    truth = Truth(
        groups=pd.DataFrame(
            {
                "group_id": ["g1"],
                "emergence_ord": [10],
                "visible_span": [3],
                "peak_area_msh": [100.0],
                "donor_survey": ["rgo"],
                "donor_cycle": [12],
            }
        ),
        group_days=pd.DataFrame(
            {"group_id": ["g1"] * 3, "date_ord": [10, 11, 12], "area_msh": [50.0, 100.0, 40.0]}
        ),
        days=np.arange(8, 16),
        reference_threshold_msh=45.0,
        anchoring="rgo_15_19",
    )
    assert truth.counts(0.0).sum() == 3
    assert truth.reference_counts().sum() == 2  # the 40 MSH day is below the anchor


def test_the_two_anchorings_are_adr_009s_and_differ():
    assert set(ANCHORINGS) == {"debrecen_21_23", "rgo_15_19"}
    assert anchor_threshold("rgo_15_19") == pytest.approx(28.084, abs=1e-3)
    assert anchor_threshold("debrecen_21_23") == pytest.approx(20.424, abs=1e-3)


def test_an_unknown_anchoring_is_rejected():
    with pytest.raises(ValueError, match="unknown anchoring"):
        anchor_threshold("kmas_20_24")


def test_sporer_cannot_donate_because_it_has_no_lifetimes():
    """D23: all 5,421 Spörer groups carry obs_duration = 1."""
    assert "sporer" not in DONOR_SURVEYS
    with pytest.raises(ValueError, match="cannot donate"):
        build_donor_library("sporer")


def test_schwabe_can_donate_even_though_step_1_excluded_it():
    """Step 1 needed spotless days, which Schwabe lacks; a truth needs groups."""
    assert "schwabe" in DONOR_SURVEYS
    library = build_donor_library("schwabe")
    assert library
    assert all(donor.n_groups > 0 for donor in library.values())


def test_the_donor_library_is_adr_009s_complete_cycles(rgo_library):
    assert sorted(rgo_library) == [12, 13, 14, 15, 16, 17, 18, 19]
    for donor in rgo_library.values():
        assert donor.length_days > 2000
        assert len(donor.group_days) > donor.n_groups  # multi-day groups


@pytest.mark.slow
def test_the_surveys_own_span_is_replayed_verbatim(rgo_library):
    """The strongest fidelity claim this generator makes: inside RGO's own span
    the truth *is* RGO, day for day, not a resampling of it."""
    truth = replay_truth(_ord(1610), _ord(2026, 12, 31), seed=1)
    low, high = truth.meta["real_span_ord"]

    ids = set().union(*(set(d.groups["group_id"]) for d in rgo_library.values()))
    from gn_next_gen.io import load_group_days

    rows = load_group_days(surveys=["rgo"])
    rows = rows[rows["group_id"].isin(ids)]
    real = rows[rows["area_corr"] > truth.reference_threshold_msh].groupby("date_ord").size()
    real = real[(real.index >= low) & (real.index <= high)]

    mine = truth.reference_counts().reindex(real.index)
    assert len(real) > 20_000
    assert np.array_equal(mine.to_numpy(), real.to_numpy())


def test_a_grand_minimum_suppresses_activity_by_about_the_factor_asked_for():
    quiet = replay_truth(
        _ord(1610),
        _ord(1874),
        seed=2,
        grand_minimum=(_ord(1645), _ord(1715, 12, 31)),
        grand_minimum_factor=30.0,
    )
    counts = quiet.reference_counts()
    years = np.array([dt.date.fromordinal(int(d)).year for d in counts.index])
    inside = counts.to_numpy()[(years >= 1645) & (years <= 1715)].mean()
    outside = counts.to_numpy()[(years < 1645) | (years > 1715)].mean()
    assert inside < outside / 10


def test_a_grand_minimum_deletes_whole_groups_rather_than_shrinking_them():
    """A weaker cycle has fewer regions, not smaller fragments of the same ones.

    Tested structurally rather than by comparing two runs: suppressing amplitudes
    changes which donor cycles are *eligible*, so a quiet world is a different
    world, not the same one thinned.  What must hold is that every area in it is
    an untouched real reference area -- nothing has been scaled down.
    """
    from gn_next_gen.io import load_group_days

    quiet = replay_truth(
        _ord(1610), _ord(1874), seed=2, grand_minimum=(_ord(1645), _ord(1715, 12, 31))
    )
    window = (_ord(1645), _ord(1715, 12, 31))
    rows = quiet.group_days
    inside = rows[(rows["date_ord"] >= window[0]) & (rows["date_ord"] <= window[1])]
    assert len(inside) > 0

    real_areas = set(load_group_days(surveys=["rgo"])["area_corr"].to_numpy().tolist())
    assert set(inside["area_msh"].to_numpy().tolist()) <= real_areas

    # And every surviving group keeps its whole apparition, not a clipped stub.
    spans = inside.groupby("group_id").size()
    declared = quiet.groups.set_index("group_id")["visible_span"].reindex(spans.index)
    assert (spans <= declared).all()


def test_generation_is_deterministic_in_its_seed():
    a = replay_truth(_ord(1700), _ord(1800), seed=5)
    b = replay_truth(_ord(1700), _ord(1800), seed=5)
    c = replay_truth(_ord(1700), _ord(1800), seed=6)
    pd.testing.assert_frame_equal(a.group_days, b.group_days)
    assert not a.group_days.equals(c.group_days)


def test_synthetic_cycles_do_not_spill_into_the_verbatim_span():
    """They tile past the gap edge by construction, so the clip is load-bearing:
    without it the real era would carry two Suns at once."""
    truth = replay_truth(_ord(1610), _ord(2026, 12, 31), seed=4)
    low, _high = truth.meta["real_span_ord"]
    synthetic = truth.groups[truth.groups["group_id"].str.startswith("s")]
    emergences = synthetic["emergence_ord"].to_numpy()
    spill = truth.group_days[truth.group_days["group_id"].str.startswith("s")]
    assert len(emergences) > 0
    assert spill["date_ord"].max() < low or spill[spill["date_ord"] >= low].empty or True
    # The real test: no day inside the verbatim span carries a synthetic group.
    in_real = spill[(spill["date_ord"] >= low) & (spill["date_ord"] <= _high)]
    assert in_real.empty


def test_a_groups_daily_areas_sit_below_its_peak_as_step_1_found():
    """Step 1 §2.4: "the per-group area is each group's **peak**, while the per-day
    law sees **instantaneous** areas".  Replay carries real areas, so the mechanism
    must come through.

    Tested per group, which is where the claim lives.  The *aggregate* log-area
    mean runs the other way (+0.33 here) because a 20-day group contributes 20
    rows and long-lived groups are large -- that is size-biasing, not a
    contradiction, and Step 1's -0.19 was a truncated fitted mu rather than a raw
    mean of either sample.
    """
    truth = replay_truth(_ord(1880), _ord(1960), seed=3)
    peak = np.log(truth.groups.set_index("group_id")["peak_area_msh"])
    daily = truth.group_days.assign(log_area=np.log(truth.group_days["area_msh"]))
    mean_daily = daily.groupby("group_id")["log_area"].mean()
    delta = mean_daily - peak.reindex(mean_daily.index)
    delta = delta[np.isfinite(delta)]

    assert delta.median() < -0.2
    assert (delta < 0).mean() > 0.6


def test_activity_level_is_in_mhem_and_tracks_the_cycle(small_truth):
    activity = small_truth.activity_level()
    assert activity.index.equals(pd.Index(small_truth.days))
    assert 0.0 < float(activity.max()) < 20.0  # mHem, not microhemispheres
    assert float(activity.std()) > 0


def test_truth_round_trips_through_its_sealed_directory(tmp_path, small_truth):
    write_truth(small_truth, tmp_path)
    back = read_truth(tmp_path)
    pd.testing.assert_frame_equal(back.groups, small_truth.groups)
    pd.testing.assert_frame_equal(back.group_days, small_truth.group_days)
    assert back.anchoring == small_truth.anchoring
    assert back.reference_threshold_msh == small_truth.reference_threshold_msh


def test_a_truth_missing_a_column_is_rejected():
    with pytest.raises(ValueError, match="group_days is missing"):
        Truth(
            groups=pd.DataFrame(
                {
                    "group_id": [],
                    "emergence_ord": [],
                    "visible_span": [],
                    "peak_area_msh": [],
                    "donor_survey": [],
                    "donor_cycle": [],
                }
            ),
            group_days=pd.DataFrame({"group_id": []}),
            days=np.arange(3),
            reference_threshold_msh=1.0,
            anchoring="rgo_15_19",
        )
