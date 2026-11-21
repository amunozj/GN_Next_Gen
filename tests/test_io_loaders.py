"""Loader contracts: counts against the manifest, sentinels, dates, D16, D22, the KMAS rule.

The measured constants here are the same ones the data dictionary and the Step 1
results doc quote.  They are checked against ``data/raw/MANIFEST.csv`` rather than
against each other wherever the manifest records them, so a refreshed snapshot
cannot pass by agreeing with a stale test.
"""

from __future__ import annotations

import csv
import datetime as dt

import pandas as pd
import pytest

from gn_next_gen.io import (
    DUPLICATE_RULES,
    STEP1_SURVEYS,
    daily_counts,
    kmas_colliding_group_ids,
    load_day_status,
    load_group_days,
    load_group_single_entry,
    load_observations,
    load_observers,
    load_silso_daily,
    load_silso_monthly,
    resolve_duplicate_keys,
)
from gn_next_gen.io.cache import cache_key, manifest_checksum, manifest_checksums
from gn_next_gen.io.reference import CORE_FIVE, DAY_NO_OBSERVATION
from gn_next_gen.io.silso import COMPARISON_ONLY
from gn_next_gen.paths import GN_V13_OBSERVATIONS, MANIFEST

# v1.3_4, after the D16 'max' rule collapses 3 duplicate keys into 3 fewer rows.
V13_ROWS_WITH_DATA = 1_099_986
OBSERVATIONS_AFTER_MAX_RULE = 1_099_983
STATIONS_WITH_DATA = 740
KMAS_COLLIDING_IDS = 289


@pytest.fixture(scope="module")
def obs():
    return load_observations(use_cache=False)


@pytest.fixture(scope="module")
def manifest_rows():
    with MANIFEST.open(newline="") as handle:
        return {row["path"]: row for row in csv.DictReader(handle)}


# --- counts and coverage against the manifest ------------------------------


def test_row_count_matches_manifest(obs, manifest_rows):
    """The manifest's own `lines` for the observations table, minus the sentinels."""
    row = manifest_rows["gn_database_v1.3/gn_v13_observations.parquet"]
    total = int(row["lines"])
    raw = pd.read_parquet(GN_V13_OBSERVATIONS, columns=["STATION"])
    assert len(raw) == total
    assert int((raw["STATION"] > 0).sum()) == V13_ROWS_WITH_DATA
    # Loading drops sentinel rows and collapses the 3 duplicate keys.
    assert len(obs) == OBSERVATIONS_AFTER_MAX_RULE


def test_coverage_matches_manifest(obs, manifest_rows):
    """The manifest's coverage is the table's *calendar* span, sentinels included.

    Loading drops the no-observation rows, so the first real observation is later:
    Harriot on 1610-12-18, the record's opening observation.  The last day carries
    observations, so that end does match.
    """
    row = manifest_rows["gn_database_v1.3/gn_v13_observations.parquet"]
    start = dt.date.fromisoformat(row["coverage_start"]).toordinal()
    end = dt.date.fromisoformat(row["coverage_end"]).toordinal()
    raw = pd.read_parquet(GN_V13_OBSERVATIONS, columns=["ORDINAL"])
    assert raw["ORDINAL"].min() == start
    assert raw["ORDINAL"].max() == end
    assert start <= obs["date_ord"].min() <= end
    assert obs["date_ord"].min() == dt.date(1610, 12, 18).toordinal()
    assert obs["date_ord"].max() == end


def test_every_station_is_listed(obs):
    observers = load_observers()
    assert obs["station"].nunique() == STATIONS_WITH_DATA
    assert set(obs["station"]) <= set(observers["STATION"])


# --- sentinels and dates ---------------------------------------------------


def test_sentinel_rows_do_not_survive(obs):
    """STATION = 0 and GROUPS = -1 are sentinels, not data."""
    assert (obs["station"] > 0).all()
    assert (obs["groups"] >= 0).all()


def test_dates_round_trip_and_leap_days_are_valid(obs):
    """Every date_ord is a real date, and no 29 February lands in a non-leap year."""
    days = pd.Index(obs["date_ord"].unique())
    dates = [dt.date.fromordinal(int(o)) for o in days]
    feb29 = [d for d in dates if (d.month, d.day) == (2, 29)]
    assert feb29, "the record spans four centuries; it must contain leap days"
    for d in feb29:
        assert d.year % 4 == 0 and (d.year % 100 != 0 or d.year % 400 == 0), d
    for bad_year in (1700, 1800, 1900):
        assert not any(d.year == bad_year and (d.month, d.day) == (2, 29) for d in dates)


def test_ordinal_convention_is_v122s():
    """ORDINAL is derived, so its convention is worth pinning: 1610-01-01 = 587676."""
    raw = pd.read_parquet(GN_V13_OBSERVATIONS, columns=["ORDINAL"])
    assert raw["ORDINAL"].min() == dt.date(1610, 1, 1).toordinal() == 587_676


# --- D16 ------------------------------------------------------------------


def _toy_duplicates() -> pd.DataFrame:
    """Two identical rows on one key, two conflicting on another, one single."""
    return pd.DataFrame(
        {
            "date_ord": [10, 10, 20, 20, 30],
            "station": [1, 1, 2, 2, 3],
            "groups": [0, 0, 7, 8, 5],
        }
    )


@pytest.mark.parametrize("rule", DUPLICATE_RULES)
def test_duplicate_rules_behave_as_documented(rule):
    out = resolve_duplicate_keys(_toy_duplicates(), rule)
    counts = out.groupby(["date_ord", "station"]).size().to_dict()
    if rule == "keep_both":
        assert len(out) == 5
        return
    assert counts[(10, 1)] == 1, "identical duplicates always collapse"
    if rule == "drop_conflicting":
        assert (20, 2) not in counts
        assert len(out) == 2
    else:
        assert counts[(20, 2)] == 1
        assert len(out) == 3
        got = int(out.loc[(out["date_ord"] == 20), "groups"].iloc[0])
        assert got == (8 if rule == "max" else 7)


def test_default_rule_keeps_the_takuma_day_at_eight(obs):
    """The single surviving D16 conflict, resolved by the default rule."""
    day = dt.date(1982, 10, 13).toordinal()
    row = obs[(obs["date_ord"] == day) & (obs["station"] == 768)]
    assert len(row) == 1
    assert int(row["groups"].iloc[0]) == 8


def test_drop_conflicting_loses_exactly_that_one_observer_day():
    dropped = load_observations(duplicate_rule="drop_conflicting", use_cache=False)
    day = dt.date(1982, 10, 13).toordinal()
    assert dropped[(dropped["date_ord"] == day) & (dropped["station"] == 768)].empty
    assert len(dropped) == OBSERVATIONS_AFTER_MAX_RULE - 1


def test_unknown_duplicate_rule_is_rejected():
    with pytest.raises(ValueError, match="unknown duplicate rule"):
        resolve_duplicate_keys(_toy_duplicates(), "whatever")


# --- the KMAS single-use rule (ADR-009) ------------------------------------


def test_kmas_counts_are_withheld_while_kmas_is_the_reference():
    ext = load_observations(include_extension=True, kmas_role="reference", use_cache=False)
    assert "extension_kmas" not in set(ext["source"])
    assert "extension_swpc" in set(ext["source"])


def test_kmas_counts_appear_only_when_declared_as_observer_counts():
    ext = load_observations(include_extension=True, kmas_role="observer_counts", use_cache=False)
    assert "extension_kmas" in set(ext["source"])


def test_unknown_kmas_role_is_rejected():
    with pytest.raises(ValueError, match="kmas_role"):
        load_observations(include_extension=True, kmas_role="both", use_cache=False)


def test_extension_carries_the_record_past_v13():
    ext = load_observations(include_extension=True, use_cache=False)
    assert ext["date_ord"].max() > dt.date(2010, 12, 31).toordinal()


# --- the reference, D22 and the Step 1 survey set --------------------------


def test_step1_excludes_sporer_and_schwabe_by_default():
    """PI, 2026-11-21: Spörer tracks no groups, Schwabe records no spotless days (D23)."""
    assert set(load_group_days()["survey"]) == set(STEP1_SURVEYS)
    assert "sporer" not in STEP1_SURVEYS and "schwabe" not in STEP1_SURVEYS
    both = load_group_days(CORE_FIVE)
    assert {"sporer", "schwabe"} <= set(both["survey"]), "still one argument away"


def test_d23_the_excluded_surveys_have_no_spotless_days():
    """The measurement behind the exclusion, pinned so it cannot drift silently."""
    days = load_day_status(CORE_FIVE)
    spotless = days[days["day_status"] == 1].groupby("survey").size()
    assert "sporer" not in spotless.index
    assert "schwabe" not in spotless.index
    for survey in STEP1_SURVEYS:
        assert spotless[survey] > 0


def test_d23_sporer_tracks_no_group_across_days():
    single = load_group_single_entry(["sporer"])
    assert (single["obs_duration"] == 1).all()
    assert (single["group_obs_kind"] == 1).all()


def test_d22_collisions_are_dropped_from_group_keyed_tables():
    bad = kmas_colliding_group_ids()
    assert len(bad) == KMAS_COLLIDING_IDS
    kept = load_group_single_entry(["kmas"])
    assert not set(kept["group_id"]) & bad
    with_bad = load_group_single_entry(["kmas"], drop_kmas_collisions=False)
    assert len(with_bad) - len(kept) == KMAS_COLLIDING_IDS


def test_d22_daily_counts_keep_every_row():
    """Counts count rows, which is the correct unit, so collisions must NOT be dropped."""
    bad = kmas_colliding_group_ids()
    rows = load_group_days(["kmas"])
    assert set(rows["group_id"]) & bad


def test_unknown_survey_is_rejected():
    with pytest.raises(ValueError, match="unknown survey"):
        load_group_days(["soon"])


# --- daily counts ---------------------------------------------------------


def test_counts_are_strictly_above_the_threshold():
    groups = pd.DataFrame(
        {
            "survey": ["rgo"] * 3,
            "group_id": ["a", "b", "c"],
            "date_ord": [1, 1, 1],
            "area_corr": [29.0, 30.0, 31.0],
        }
    )
    days = pd.DataFrame(
        {
            "survey": ["rgo"],
            "date_ord": [1],
            "day_status": [2],
            "n_groups": [3],
            "area_total": [90.0],
        }
    )
    out = daily_counts(30.0, group_days=groups, day_status=days)
    assert int(out["n_groups"].iloc[0]) == 1, "strictly greater, as the legacy code had it"


def test_spotless_days_are_zeros_and_unobserved_days_are_absent():
    counts = daily_counts(0.0, ["rgo"])
    days = load_day_status(["rgo"])
    assert len(counts) == int((days["day_status"] > DAY_NO_OBSERVATION).sum())
    spotless = counts[counts["day_status"] == 1]
    assert (spotless["n_groups"] == 0).all()
    assert counts["n_groups"].min() == 0


def test_unthresholded_counts_match_the_snapshots_own_n_groups():
    """At T = 0 the row count must equal upstream's n_groups, including for KMAS (D22)."""
    counts = daily_counts(0.0, ["kmas"])
    days = load_day_status(["kmas"])
    merged = counts.merge(days, on=["survey", "date_ord"], suffixes=("", "_snapshot"))
    assert (merged["n_groups"] == merged["n_groups_snapshot"]).all()


def test_counts_are_monotone_in_the_threshold():
    base = daily_counts(0.0, ["debrecen"]).set_index("date_ord")["n_groups"]
    for threshold in (10.0, 30.0, 60.0, 100.0):
        higher = daily_counts(threshold, ["debrecen"]).set_index("date_ord")["n_groups"]
        assert (higher <= base.reindex(higher.index)).all(), threshold


# --- SILSO is comparison-only ---------------------------------------------


def test_silso_is_tagged_comparison_only():
    for frame in (load_silso_daily(), load_silso_monthly()):
        assert frame.attrs["role"] == COMPARISON_ONLY


def test_silso_sentinels_become_nan():
    daily = load_silso_daily()
    assert daily["DAILYSN"].isna().any()
    assert (daily["DAILYSN"].dropna() >= 0).all()


# --- caching --------------------------------------------------------------


def test_cache_key_tracks_options_and_manifest_checksums():
    one = cache_key([GN_V13_OBSERVATIONS], {"rule": "max"})
    two = cache_key([GN_V13_OBSERVATIONS], {"rule": "first"})
    assert one != two, "a different loader option must not reuse a cache"
    assert one == cache_key([GN_V13_OBSERVATIONS], {"rule": "max"})
    assert (
        manifest_checksum(GN_V13_OBSERVATIONS)
        == (manifest_checksums()["gn_database_v1.3/gn_v13_observations.parquet"])
    )


def test_unmanifested_input_cannot_be_cached(tmp_path):
    """No provenance, no cache key: an unmanifested raw input is an error (ADR-006)."""
    with pytest.raises((KeyError, ValueError)):
        cache_key([tmp_path / "mystery.parquet"], {})


def test_cached_frame_round_trips(tmp_path, monkeypatch):
    from gn_next_gen.io import cache as cache_mod

    monkeypatch.setattr(cache_mod, "DATA_INTERIM", tmp_path)
    calls = []

    def build():
        calls.append(1)
        return pd.DataFrame({"a": [1, 2, 3]})

    first = cache_mod.cached_frame("toy", [GN_V13_OBSERVATIONS], {"x": 1}, build)
    second = cache_mod.cached_frame("toy", [GN_V13_OBSERVATIONS], {"x": 1}, build)
    assert len(calls) == 1, "the second call must read the cache"
    pd.testing.assert_frame_equal(first, second)
    third = cache_mod.cached_frame("toy", [GN_V13_OBSERVATIONS], {"x": 1}, build, force=True)
    assert len(calls) == 2
    pd.testing.assert_frame_equal(first, third)
