"""The superseded inputs Step 2 needs for the legacy baselines.

These exist because ADR-009 keeps ``reference_areas/`` and ``gn_database_v1.22/``
for exactly one purpose.  The tests pin the two shapes that bit the legacy code:
the missing header row (**D1**) and the positional ``AREA26`` slicing (**D9**).
"""

from __future__ import annotations

import csv
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from gn_next_gen.io import legacy_inputs as legacy
from gn_next_gen.paths import MANIFEST


def _manifest() -> dict[str, dict[str, str]]:
    with MANIFEST.open(newline="") as handle:
        return {row["path"]: row for row in csv.DictReader(handle)}


@pytest.mark.parametrize(
    ("name", "path", "rows"),
    [
        ("sc_sp_rg_db_km", "reference_areas/SC_SP_RG_DB_KM_group_areas_by_day.csv", 70_354),
        (
            "sc_sp_rg_db_km_pre5898ab8",
            "reference_areas/SC_SP_RG_DB_KM_group_areas_by_day_pre5898ab8.csv",
            52_659,
        ),
        ("drgo", "reference_areas/DRGO_group_areas_by_day.csv", 37_514),
    ],
)
def test_reference_area_files_read_to_their_manifest_coverage(name, path, rows):
    """Row counts and coverage come from MANIFEST.csv, not from constants here."""
    frame = legacy.load_reference_areas(name)
    assert len(frame) == rows

    row = _manifest()[path]
    assert dt.date.fromordinal(int(frame["date_ord"].iloc[0])).isoformat() == row["coverage_start"]
    assert dt.date.fromordinal(int(frame["date_ord"].iloc[-1])).isoformat() == row["coverage_end"]


def test_the_headerless_file_does_not_lose_its_first_day_to_D1():
    """D1: reading with ``header=0`` would make 1825-11-05 the column names."""
    frame = legacy.load_reference_areas("sc_sp_rg_db_km")
    assert list(frame.columns) == ["date_ord", *legacy.REFERENCE_AREA_COLUMNS]
    assert dt.date.fromordinal(int(frame["date_ord"].iloc[0])) == dt.date(1825, 11, 5)
    assert frame["YEAR"].dtype.kind in "iu"


def test_a_headed_file_keeps_every_data_row():
    """The header is consumed once, not twice -- the bug this loader was born with."""
    headed = legacy.load_reference_areas("drgo")
    with (legacy.REFERENCE_AREA_FILES["drgo"]).open() as handle:
        raw = sum(1 for _ in handle)
    assert len(headed) == raw - 1


def test_areas_are_addressed_by_name_so_AREA26_cannot_be_dropped():
    """D9: the legacy code slices positionally and loses AREA26 in ten places."""
    assert legacy.AREA_COLUMNS[-1] == "AREA26"
    assert len(legacy.AREA_COLUMNS) == 26

    frame = legacy.load_reference_areas("sc_sp_rg_db_km")
    with_26 = frame[frame["AREA26"].notna() & (frame["AREA26"] > 0)]
    assert len(with_26) > 0, "no 26-group day, so the regression cannot be exercised"

    counts = legacy.legacy_daily_counts(frame, 0.0)
    dropped = legacy.legacy_daily_counts(frame.drop(columns=["AREA26"]).assign(AREA26=np.nan), 0.0)
    on_those_days = frame.index.isin(with_26.index)
    assert (counts.loc[on_those_days, "n_groups"] > dropped.loc[on_those_days, "n_groups"]).all()


def test_counting_is_strictly_above_the_threshold():
    """Ported from SSN_ADF.py l. 837-841: ``np.greater``, not ``>=``."""
    areas = pd.DataFrame(
        {
            "date_ord": [1, 2],
            "AREA1": [30.0, np.nan],
            **{c: [np.nan, np.nan] for c in legacy.AREA_COLUMNS[1:]},
        }
    )
    assert legacy.legacy_daily_counts(areas, 30.0)["n_groups"].iloc[0] == 0
    assert legacy.legacy_daily_counts(areas, 29.9)["n_groups"].iloc[0] == 1


def test_a_nan_in_AREA1_means_no_observation_not_a_spotless_day():
    areas = pd.DataFrame(
        {
            "date_ord": [1, 2],
            **{c: [0.0, np.nan] for c in legacy.AREA_COLUMNS},
        }
    )
    counts = legacy.legacy_daily_counts(areas, 0.0)
    assert counts["observed"].tolist() == [True, False]
    assert counts["n_groups"].iloc[0] == 0  # observed and spotless
    assert np.isnan(counts["n_groups"].iloc[1])  # not observed at all


def test_v122_observations_match_the_file_and_carry_no_minus_one():
    frame = legacy.load_v122_observations()
    assert len(frame) == 1_047_761
    assert frame["station"].min() > 0
    assert (frame["groups"] >= 0).all()
    assert frame["station"].nunique() == 715


def test_v122_observers_list_every_station():
    observers = legacy.load_v122_observers()
    assert len(observers) == 743
    assert {"STATION", "TOT_OBS", "OBSERVER"} <= set(observers.columns)


def test_pathfinding_endpoints_resolve_by_name_not_by_the_matlab_numbers():
    """The MATLAB scripts' V1-12 445/590 are Koyama and Janssens under v1.22."""
    resolved = {k: legacy.v122_station_by_name(v) for k, v in legacy.PATHFINDING_ENDPOINTS.items()}
    assert resolved == {"staudach": 466, "locarno": 614, "rgo": 332, "schwabe": 279}

    observers = legacy.load_v122_observers().set_index("STATION")["OBSERVER"]
    assert "KOYAMA" in observers.loc[445].upper()
    assert "JANSSENS" in observers.loc[590].upper()


def test_an_ambiguous_name_raises_rather_than_picking_one():
    with pytest.raises(LookupError, match="matches 2"):
        legacy.v122_station_by_name("STAUDACH")
