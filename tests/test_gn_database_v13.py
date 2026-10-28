"""Invariants of the frozen v1.3 observation database.

Several of these are *characterization* tests rather than correctness tests: they
pin the known data issues D16-D18 and D20 at their measured values so the numbers
quoted in the data dictionary, the folder README and the report to the v1.3 authors
cannot drift unnoticed. If one of them fails, the data changed -- check whether the
documentation and the report still say the right thing before relaxing it.
"""

import datetime as dt

import pandas as pd
import pytest

from gn_next_gen.paths import (
    GN_V13_CHANGELOG,
    GN_V13_OBSERVATIONS,
    GN_V13_OBSERVERS,
    GN_V13_XLSX,
)

SHEETS = ("1610-1946", "1947-1988", "1989-2010")
SHEET_ROWS = {"1610-1946": 405_140, "1947-1988": 347_248, "1989-2010": 404_285}
TOTAL_ROWS = 1_156_673
STATION_ROWS = 1_100_144
STATIONS_WITH_DATA = 740
OBSERVERS_LISTED = 775
CHANGELOG_COUNTS = {"v1.12": 403, "v1.2": 72, "v1.21": 2, "v1.3": 109}


@pytest.fixture(scope="module")
def obs():
    return pd.read_parquet(GN_V13_OBSERVATIONS)


@pytest.fixture(scope="module")
def observers():
    return pd.read_parquet(GN_V13_OBSERVERS)


@pytest.fixture(scope="module")
def changelog():
    return pd.read_csv(GN_V13_CHANGELOG)


def test_source_workbook_present():
    assert GN_V13_XLSX.is_file(), "the authoritative xlsx must stay beside the conversion"


def test_schema(obs, observers):
    assert list(obs.columns) == [
        "YEAR", "MONTH", "DAY", "ORDINAL", "STATION", "OBSER", "GROUPS", "FRACYEAR", "SHEET",
    ]
    assert list(observers.columns) == [
        "STATION", "INITIAL", "FINAL", "TOT_OBS", "OBSERVER", "changed_in_v13",
    ]


def test_row_counts(obs, observers):
    assert len(obs) == TOTAL_ROWS
    assert int((obs["STATION"] > 0).sum()) == STATION_ROWS
    assert obs.loc[obs["STATION"] > 0, "STATION"].nunique() == STATIONS_WITH_DATA
    assert len(observers) == OBSERVERS_LISTED


def test_sheet_row_counts_and_no_blank_rows(obs):
    """The 1610-1946 sheet has ~96,000 trailing blank rows; none may survive."""
    assert obs.groupby("SHEET").size().to_dict() == SHEET_ROWS
    assert obs["YEAR"].notna().all()
    assert obs["ORDINAL"].notna().all()


def test_sheet_seams_do_not_gap_or_overlap(obs):
    """The three sheets must tile 1610-2010 exactly at 1946/1947 and 1988/1989."""
    spans = obs.groupby("SHEET")["YEAR"].agg(["min", "max"]).to_dict("index")
    assert spans["1610-1946"]["min"] == 1610
    for earlier, later in (("1610-1946", "1947-1988"), ("1947-1988", "1989-2010")):
        assert spans[earlier]["max"] + 1 == spans[later]["min"], (earlier, later, spans)
    assert spans["1989-2010"]["max"] == 2010


def test_coverage_and_no_invalid_dates(obs):
    assert obs["ORDINAL"].min() == dt.date(1610, 1, 1).toordinal()
    assert obs["ORDINAL"].max() == dt.date(2010, 12, 31).toordinal()
    # ORDINAL must round-trip to (YEAR, MONTH, DAY) for every row.
    sample = obs.sample(5000, random_state=0)
    rebuilt = [dt.date.fromordinal(int(o)) for o in sample["ORDINAL"]]
    assert all(
        (d.year, d.month, d.day) == (y, m, dd)
        for d, y, m, dd in zip(rebuilt, sample["YEAR"], sample["MONTH"], sample["DAY"])
    )


def test_ordinal_matches_v122_convention(obs):
    """1610-01-01 = 587676, as the v1.22 data dictionary records."""
    first = obs.loc[obs["ORDINAL"] == obs["ORDINAL"].min()].iloc[0]
    assert int(first["ORDINAL"]) == 587_676


def test_leap_days_absent(obs):
    """Feb 29 of 1700/1800/1900 is not a Gregorian leap day; v1.21 removed them."""
    bad = obs[(obs["MONTH"] == 2) & (obs["DAY"] == 29) & obs["YEAR"].isin([1700, 1800, 1900])]
    assert bad.empty


def test_sentinels(obs):
    """GROUPS = -1 only ever accompanies STATION = 0, and vice versa."""
    assert (obs.loc[obs["STATION"] > 0, "GROUPS"] >= 0).all()
    assert (obs.loc[obs["STATION"] == 0, "GROUPS"] == -1).all()


def test_every_station_with_data_is_listed(obs, observers):
    with_data = set(obs.loc[obs["STATION"] > 0, "STATION"].unique())
    assert with_data <= set(observers["STATION"])
    assert observers["STATION"].min() == 1
    assert observers["STATION"].max() == OBSERVERS_LISTED


# --- characterization of the known issues ----------------------------------

def test_d16_duplicate_date_station_keys(obs):
    """217 duplicate keys, 122 of them with identical GROUPS -- see D16.

    v1.22 had none. 95 carry conflicting GROUPS for the same observer and day, so
    no loader may key on (date, station) until a resolution rule is chosen.
    """
    rows = obs[obs["STATION"] > 0]
    dup_mask = rows.duplicated(["ORDINAL", "STATION"], keep=False)
    dups = rows[dup_mask]
    keys = dups.groupby(["ORDINAL", "STATION"])["GROUPS"].nunique()
    assert len(keys) == 217
    assert int((keys == 1).sum()) == 122
    assert int((keys > 1).sum()) == 95
    by_station = dups.groupby(["ORDINAL", "STATION"]).size().reset_index()
    counts = by_station["STATION"].value_counts().to_dict()
    assert counts == {755: 94, 329: 76, 245: 41, 773: 4, 768: 2}


def test_d17_extreme_group_counts(obs):
    """Only two rows exceed 53 groups; 93 is implausible -- see D17.

    State-space truncation K must come from a robust quantile, not from this maximum.
    """
    extreme = obs[(obs["STATION"] > 0) & (obs["GROUPS"] > 53)]
    assert len(extreme) == 2
    got = {(int(r.YEAR), int(r.MONTH), int(r.DAY), int(r.STATION), int(r.GROUPS))
           for r in extreme.itertuples()}
    assert got == {(1982, 9, 26, 768, 93), (1992, 3, 18, 712, 58)}
    assert int(obs.loc[obs["STATION"] > 0, "GROUPS"].quantile(0.9999)) < 40


def test_d18_placeholder_observers(observers):
    """33 observers are listed with no observations -- see D18."""
    placeholder = observers[observers["TOT_OBS"] == 0]
    assert len(placeholder) == 33
    assert placeholder["INITIAL"].isna().all()
    assert placeholder["FINAL"].isna().all()
    active = observers[observers["TOT_OBS"] > 0]
    assert len(active) == OBSERVERS_LISTED - 33
    assert active["INITIAL"].notna().all()
    assert (active["FINAL"] >= active["INITIAL"]).all()


def test_d20_change_flags_recovered_from_fill(observers):
    """94 observers are highlighted in the workbook -- see D20.

    This exists nowhere in the cell values, so a brittle fill read must fail here
    rather than silently produce an all-false column.
    """
    assert int(observers["changed_in_v13"].sum()) == 94


def test_changelog_blocks(changelog):
    assert list(changelog.columns) == ["version", "name", "year", "note"]
    assert changelog["version"].value_counts().to_dict() == CHANGELOG_COUNTS
    v13 = changelog[changelog["version"] == "v1.3"]
    assert v13["note"].notna().all()
    # D7's correction literature must actually be cited in the v1.3 block.
    notes = " ".join(v13["note"].astype(str))
    for author in ("Hayakawa", "Carrasco", "Neuh", "Vokhmyanin"):
        assert author in notes, author
