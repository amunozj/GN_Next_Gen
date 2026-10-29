"""Invariants of the frozen reference-survey snapshot.

The cycle-completeness table and the two anchor cycle sets are pinned here because
ADR-009's estimand rests on them: if an upstream refit moved a cycle boundary enough
to change a cycle's completeness status, that has to fail loudly rather than quietly
redefine the anchor.
"""

import datetime as dt
import json

import pandas as pd
import pytest

from gn_next_gen.paths import (
    REFERENCE_CYCLE_COVERAGE,
    REFERENCE_DAY_STATUS,
    REFERENCE_GROUP_DAY,
    REFERENCE_GROUP_SINGLE,
    REFERENCE_GROUPS,
    REFERENCE_SURVEY_CALIBRATION,
)

CORE_FIVE = {"schwabe", "sporer", "rgo", "debrecen", "kmas"}

# Measured from the snapshot; see ADR-009.  A cycle is complete in a survey only if
# that survey also carries its neighbours.
COMPLETE_CYCLES = {
    "schwabe": {8, 9},
    "sporer": {9, 10},
    "rgo": {12, 13, 14, 15, 16, 17, 18, 19},
    "debrecen": {21, 22, 23},
    "kmas": {20, 21, 22, 23, 24},
}
PRIMARY_ANCHOR = ("debrecen", {21, 22, 23})
SECONDARY_ANCHOR = ("rgo", {15, 16, 17, 18, 19})

MAX_PLAUSIBLE_AREA_MSH = 11_500

# Issue D22: KMAS reuses a group_id for physically distinct groups on these days.
KMAS_COLLIDING_ROWS = 839
KMAS_COLLIDING_DAY_KEYS = 414
KMAS_COLLIDING_IDS = 289


@pytest.fixture(scope="module")
def group_day():
    return pd.read_parquet(REFERENCE_GROUP_DAY)


@pytest.fixture(scope="module")
def single():
    return pd.read_parquet(REFERENCE_GROUP_SINGLE)


@pytest.fixture(scope="module")
def day_status():
    return pd.read_parquet(REFERENCE_DAY_STATUS)


@pytest.fixture(scope="module")
def coverage():
    return pd.read_csv(REFERENCE_CYCLE_COVERAGE)


@pytest.fixture(scope="module")
def provenance():
    return json.loads((REFERENCE_GROUPS / "provenance.json").read_text())


# --- schema and scope -------------------------------------------------------


def test_survey_set_is_exactly_the_core_five(group_day, single, day_status):
    for df in (group_day, single, day_status):
        assert set(df["survey"]) == CORE_FIVE
    assert "soon" not in set(group_day["survey"])
    assert "swpc" not in set(group_day["survey"]), "SWPC contributes counts only, not areas"


def test_obs_type_was_renamed_at_the_boundary(group_day, single, day_status):
    """D10: obs_type means two different things upstream; neither name may survive."""
    for df in (group_day, single, day_status):
        assert "obs_type" not in df.columns
    assert "group_obs_kind" in single.columns
    assert "day_status" in day_status.columns
    assert "obs_duration" in single.columns


def test_model_facing_and_calibrated_areas_both_present(group_day):
    for col in ("area_corr", "area_corr_unc", "area_umbra_corr"):
        assert col in group_day.columns
    for col in ("area_corr_dbcal", "area_corr_dbcal_unc_corr", "area_corr_dbcal_unc_indep"):
        assert col in group_day.columns


def test_diagnostic_columns_are_marked_as_such(provenance):
    """ADR-003: cycle phase stays latent, so these must be documented non-inputs."""
    assert provenance["model_facing_area_column"] == "area_corr"
    assert "prior" in provenance["calibrated_area_use"]
    assert "diagnostic only" in provenance["data_lineage"]


def test_firewall_lineage_statement(provenance):
    lineage = provenance["data_lineage"].lower()
    for forbidden in ("isotope", "aurora", "proxy", "silso"):
        assert forbidden in lineage, "the lineage statement must name what is excluded"
    assert "no isotope" in lineage


# --- granularity and cross-table consistency --------------------------------


def test_group_day_grain_is_one_row_per_group_observation(group_day):
    """One row per group observation. `group_id` is the key for four surveys of five.

    KMAS reuses a `group_id` for physically distinct groups on a small number of days
    (issue D22), so the ROW is the unit of observation, not (group_id, date_ord). The
    counts are pinned so the collision cannot grow unnoticed.
    """
    assert group_day["group_id"].notna().all()
    assert group_day["date_ord"].notna().all()

    for survey in CORE_FIVE - {"kmas"}:
        part = group_day[group_day["survey"] == survey]
        assert not part.duplicated(["group_id", "date_ord"]).any(), survey

    kmas = group_day[group_day["survey"] == "kmas"]
    dup = kmas.duplicated(["group_id", "date_ord"], keep=False)
    assert int(dup.sum()) == KMAS_COLLIDING_ROWS
    assert kmas[dup].groupby(["group_id", "date_ord"]).ngroups == KMAS_COLLIDING_DAY_KEYS
    assert kmas.loc[dup, "group_id"].nunique() == KMAS_COLLIDING_IDS


def test_d22_collisions_are_distinct_groups_not_repeat_measurements(group_day):
    """The colliding rows sit at different places on the disk, so they are real groups.

    This is what rules out the benign reading (the same group measured twice in a day)
    and is why counting rows is right while counting `group_id` would under-count.
    """
    kmas = group_day[group_day["survey"] == "kmas"]
    dup = kmas.duplicated(["group_id", "date_ord"], keep=False)
    spread = kmas[dup].groupby(["group_id", "date_ord"])[["latitude", "longitude"]].nunique()
    assert (spread.max(axis=1) > 1).any(), "at least some collisions must be at distinct positions"


def test_day_status_is_one_row_per_survey_day(day_status):
    assert not day_status.duplicated(["survey", "date_ord"]).any()
    assert set(day_status["day_status"].unique()) <= {0, 1, 2}


def test_day_status_calendar_is_contiguous_per_survey(day_status):
    for survey, part in day_status.groupby("survey"):
        ords = part["date_ord"].sort_values().to_numpy()
        assert (ords[1:] - ords[:-1] == 1).all(), f"{survey} has a gap in its calendar"


def test_day_status_agrees_with_the_group_rows(group_day, day_status):
    """status 1 means spotless, 2 means active, 0 means no observation."""
    counted = group_day.groupby(["survey", "date_ord"]).size().rename("observed").reset_index()
    merged = day_status.merge(counted, on=["survey", "date_ord"], how="left")
    merged["observed"] = merged["observed"].fillna(0).astype(int)

    assert (merged.loc[merged["day_status"] == 1, "observed"] == 0).all()
    assert (merged.loc[merged["day_status"] == 2, "observed"] > 0).all()
    assert (merged.loc[merged["day_status"] == 0, "observed"] == 0).all()
    # n_groups is the group-row count on every day the survey observed.
    active = merged[merged["day_status"] == 2]
    assert (active["n_groups"] == active["observed"]).all()


def test_single_entry_is_consistent_with_the_group_days(group_day, single):
    assert not single.duplicated(["survey", "group_id"]).any()
    gd = set(zip(group_day["survey"], group_day["group_id"], strict=True))
    sg = set(zip(single["survey"], single["group_id"], strict=True))
    assert sg == gd, "every group must appear in both tables"

    # The canonical day must lie within the group's observed span.
    obs = group_day.groupby(["survey", "group_id"])["date_ord"].agg(["min", "max"])
    can = single.set_index(["survey", "group_id"])["date_ord"]
    joined = obs.join(can.rename("canonical")).dropna()
    assert (joined["canonical"] >= joined["min"]).all()
    assert (joined["canonical"] <= joined["max"]).all()


def test_d22_inflates_lifetimes_only_for_the_colliding_kmas_groups(group_day, single):
    """Merged identities lengthen the apparent lifetime, so lifetime work must exclude them.

    Step 1 Check A compares lifetime distributions; the 289 affected KMAS groups have a
    markedly longer median duration than the clean ones precisely because two groups'
    apparitions are being read as one.
    """
    kmas_days = group_day[group_day["survey"] == "kmas"]
    dup = kmas_days.duplicated(["group_id", "date_ord"], keep=False)
    affected = set(kmas_days.loc[dup, "group_id"])

    kmas = single[single["survey"] == "kmas"]
    hit = kmas[kmas["group_id"].isin(affected)]
    clean = kmas[~kmas["group_id"].isin(affected)]
    assert len(hit) == KMAS_COLLIDING_IDS
    assert hit["obs_duration"].median() > clean["obs_duration"].median(), (
        "if this ever stops holding, re-check what D22 actually is"
    )
    # Small enough to drop: well under 2% of KMAS groups.
    assert len(hit) / len(kmas) < 0.02


def test_obs_duration_is_consistent_with_the_observed_span(single, group_day):
    span = (
        group_day.groupby(["survey", "group_id"])["date_ord"]
        .agg(lambda s: s.max() - s.min())
        .rename("span")
    )
    joined = single.set_index(["survey", "group_id"]).join(span)
    assert (joined["obs_duration"] >= 0).all()
    assert (joined["obs_duration"] >= joined["span"]).all(), (
        "a group's reported duration cannot be shorter than the days it was seen on"
    )


# --- values -----------------------------------------------------------------


def test_areas_are_positive_and_physically_plausible(group_day):
    for col in ("area_obs", "area_corr"):
        vals = group_day[col].dropna()
        assert (vals > 0).all(), f"{col} has non-positive values"
        assert vals.max() < MAX_PLAUSIBLE_AREA_MSH, f"{col} max {vals.max()} MSH"
    umbra = group_day["area_umbra_corr"].dropna()
    assert (umbra >= 0).all()


def test_coverage_spans_1825_to_2026(day_status, provenance):
    first = dt.date.fromordinal(int(day_status["date_ord"].min()))
    last = dt.date.fromordinal(int(day_status["date_ord"].max()))
    assert first.year == 1825
    assert last.year >= 2026
    spans = provenance["coverage_by_survey"]
    assert set(spans) == CORE_FIVE


def test_the_union_of_survey_spans_has_no_gap(provenance):
    """1825-2026 must be covered continuously by the five surveys together."""
    spans = sorted(
        (dt.date.fromisoformat(a).toordinal(), dt.date.fromisoformat(b).toordinal())
        for a, b in provenance["coverage_by_survey"].values()
    )
    reach = spans[0][1]
    for start, end in spans[1:]:
        assert start <= reach + 1, f"gap before ordinal {start}"
        reach = max(reach, end)


def test_survey_calibration_table(provenance):
    calib = pd.read_csv(REFERENCE_SURVEY_CALIBRATION).set_index("survey")
    assert set(calib.index) == CORE_FIVE
    # Debrecen is the identity map, which is why it is the primary anchor.
    assert calib.loc["debrecen", "CF_med"] == 1.0
    assert calib.loc["debrecen", "map_kind"] == "identity"
    assert (calib["CF_lo"] <= calib["CF_med"]).all()
    assert (calib["CF_med"] <= calib["CF_hi"]).all()
    assert (calib["threshold_MSH"] > 0).all()


# --- ADR-009's anchors ------------------------------------------------------


def test_cycle_completeness_matches_adr_009(coverage):
    got = {
        survey: set(part.loc[part["complete"], "cycle"])
        for survey, part in coverage.groupby("survey")
    }
    assert got == COMPLETE_CYCLES


def test_anchors_are_complete_and_disjoint(coverage):
    primary_survey, primary = PRIMARY_ANCHOR
    secondary_survey, secondary = SECONDARY_ANCHOR
    assert primary <= COMPLETE_CYCLES[primary_survey]
    assert secondary <= COMPLETE_CYCLES[secondary_survey]
    assert not primary & secondary, "the two anchors must share no cycle"

    flags = coverage.set_index(["survey", "cycle"])["complete"]
    for survey, cycles in ((primary_survey, primary), (secondary_survey, secondary)):
        for c in cycles:
            assert bool(flags.loc[(survey, c)]), f"{survey} cycle {c} is not complete"


def test_cycle_20_is_in_neither_anchor_and_bridges_them(coverage):
    """Cycle 20 is the one cycle RGO and Debrecen would have split; KMAS carries it."""
    assert 20 not in PRIMARY_ANCHOR[1]
    assert 20 not in SECONDARY_ANCHOR[1]
    flags = coverage.set_index(["survey", "cycle"])["complete"]
    assert not bool(flags.loc[("rgo", 20)]), "RGO stops mid-cycle-20"
    assert not bool(flags.loc[("debrecen", 20)]), "Debrecen starts mid-cycle-20"
    assert bool(flags.loc[("kmas", 20)]), "KMAS must carry cycle 20 whole to bridge the anchors"
    # The bridge also has to overlap the primary anchor.
    assert PRIMARY_ANCHOR[1] <= COMPLETE_CYCLES["kmas"]


def test_clipped_cycles_are_flagged_incomplete(coverage):
    """A cycle at the edge of a survey's coverage is clipped, however contiguous it looks."""
    flags = coverage.set_index(["survey", "cycle"])["complete"]
    for survey, cycle in (
        ("schwabe", 7),
        ("schwabe", 10),
        ("sporer", 8),
        ("sporer", 11),
        ("rgo", 11),
        ("rgo", 20),
        ("debrecen", 20),
        ("debrecen", 24),
        ("kmas", 19),
        ("kmas", 25),
    ):
        assert not bool(flags.loc[(survey, cycle)]), f"{survey} cycle {cycle}"
