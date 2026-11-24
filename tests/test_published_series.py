"""The published-series comparison table must stay on one scale and one shape.

These series are comparison only (ADR-002, ADR-003), so nothing here scores a
method.  What is worth pinning is the thing that would corrupt a comparison
silently: ``gn`` means the **mean daily group count** for all eight series, and
two of the files in the Velasco Herrera supplement are published in Hoyt &
Schatten's 12.08-scaled units instead (D24).  A future exporter that forgets the
division, or divides twice, fails here rather than in a figure.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

from gn_next_gen.paths import DATA_RAW, SCRIPTS

FOLDER = DATA_RAW / "gn_published_series_2026-11"
TABLE = FOLDER / "published_gn_series.csv"

#: series -> (first year, last year, rows), as published.
EXPECTED = {
    "hoyt_schatten_1998": (1610.5, 1995.5, 386),
    "svalgaard_schatten_2016_backbone": (1610.5, 2015.5, 406),
    "usoskin_series": (1749.5, 1995.5, 247),
    "vaquero": (1639.0, 1714.0, 26),
    "zolotova": (1635.347078, 1720.859131, 127),
    "chatzistergos_2017": (1739.5, 2010.5, 272),
    "velasco_herrera_2024": (1610.0, 2022.0, 413),
    "velasco_herrera_2024_hoyt_update": (1610.0, 2022.0, 413),
}


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return pd.read_csv(TABLE)


@pytest.fixture(scope="module")
def exporter():
    """Import ``scripts/fetch_published_gn_series.py`` as a module."""
    path = SCRIPTS / "fetch_published_gn_series.py"
    spec = importlib.util.spec_from_file_location("fetch_published_gn_series", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_columns(table):
    assert list(table.columns) == [
        "series",
        "year",
        "gn",
        "gn_sd",
        "gn_sd_hi",
        "gn_sd_lo",
        "n_observations",
    ]


def test_every_series_present_with_its_published_span(table):
    assert set(table["series"]) == set(EXPECTED)
    for name, (first, last, rows) in EXPECTED.items():
        group = table[table["series"] == name]
        assert len(group) == rows, name
        assert group["year"].min() == pytest.approx(first), name
        assert group["year"].max() == pytest.approx(last), name


def test_everything_is_in_group_counts_not_12_08_scaled(table):
    """The whole point of D24: a 12.08-scaled series would be ~12x too big.

    No annual mean daily group count in any published reconstruction approaches
    20; Hoyt & Schatten's scaled units reach 175 over the same years.
    """
    peaks = table.groupby("series")["gn"].max()
    assert peaks.max() < 20.0, f"a series looks 12.08-scaled:\n{peaks}"
    assert table["gn"].min() >= 0.0


def test_hoyt_update_reproduces_the_1998_series_it_updates(table):
    """The cross-check that licenses the 12.08 division (D24).

    Velasco Herrera et al. recompute Hoyt & Schatten's series and extend it to
    2022.  Divided by 12.08 it must land on the 1998 series this folder already
    carries, over every year the two share.
    """
    old = table[table["series"] == "hoyt_schatten_1998"].copy()
    new = table[table["series"] == "velasco_herrera_2024_hoyt_update"].copy()
    old["key"] = np.floor(old["year"]).astype(int)
    new["key"] = np.floor(new["year"]).astype(int)

    both = old.merge(new, on="key", suffixes=("_1998", "_2024")).dropna(
        subset=["gn_1998", "gn_2024"]
    )
    assert len(both) == 380, (
        "the two overlap on 1610-1995 minus the six years Hoyt & Schatten "
        "record as no-data, which the 2024 file also leaves empty"
    )
    difference = (both["gn_1998"] - both["gn_2024"]).abs()
    assert difference.max() < 1e-3, f"max |diff| = {difference.max()}"
    # And it genuinely extends the record, which is why it is carried at all.
    assert new["key"].max() == 2022


def test_chatzistergos_errors_are_asymmetric_and_not_averaged(table):
    group = table[table["series"] == "chatzistergos_2017"]
    assert group["gn_sd"].isna().all(), "symmetric gn_sd must stay empty"
    assert group["gn_sd_hi"].notna().sum() == 266
    assert group["gn_sd_lo"].notna().sum() == 266
    observed = group.dropna(subset=["gn_sd_hi", "gn_sd_lo"])
    assert (observed["gn_sd_hi"] != observed["gn_sd_lo"]).any()
    # -99 is the sentinel, so it must never survive as a value.
    assert (group[["gn", "gn_sd_hi", "gn_sd_lo"]].min(numeric_only=True) >= 0).all()


#: series -> how many of its rows carry a value, the rest being no-data years.
#: Pinned because every sentinel here is a different number (-1 in the snapshot's
#: yearly files, -99 at CDS, NaN in the Velasco Herrera supplement) and each one
#: would otherwise read as a group count.
NON_NULL = {
    "hoyt_schatten_1998": 380,
    "svalgaard_schatten_2016_backbone": 406,
    "usoskin_series": 246,
    "vaquero": 22,
    "zolotova": 127,
    "chatzistergos_2017": 266,
    "velasco_herrera_2024": 401,
    "velasco_herrera_2024_hoyt_update": 407,
}


def test_no_sentinel_survives_as_a_value(table):
    """-1, -99 and NaN all mean "no data"; none may reach the gn column."""
    assert not (table["gn"] < 0).any()
    counts = table.groupby("series")["gn"].count().to_dict()
    assert counts == NON_NULL


def test_the_snapshots_no_data_years_are_null(table):
    """Seven rows the earlier exporter carried through as gn = -1."""
    hoyt = table[table["series"] == "hoyt_schatten_1998"]
    empty = hoyt[hoyt["gn"].isna()]["year"].astype(int).tolist()
    assert empty == [1636, 1637, 1641, 1744, 1745, 1747]
    assert (empty_rows := hoyt[hoyt["gn"].isna()])["n_observations"].eq(0).all()
    assert empty_rows["gn_sd"].isna().all()

    usoskin = table[table["series"] == "usoskin_series"]
    assert usoskin[usoskin["gn"].isna()]["year"].astype(int).tolist() == [1811]


def test_scale_constant_is_the_hoyt_schatten_factor(exporter):
    assert exporter.HOYT_GSN_SCALE == 12.08
    scaled = exporter.ARCHIVE_SERIES["velasco_herrera_2024_hoyt_update"]["scale"]
    assert scaled == exporter.HOYT_GSN_SCALE
    # The count-scale series must not be rescaled.
    assert "scale" not in exporter.ARCHIVE_SERIES["velasco_herrera_2024"]
    assert "scale" not in exporter.ARCHIVE_SERIES["chatzistergos_2017"]


def test_export_is_reproducible_from_the_archived_sources(exporter):
    """``--check`` in test form: rebuilding must reproduce the committed CSV."""
    rebuilt, _ = exporter.build()
    committed = pd.read_csv(TABLE)
    pd.testing.assert_frame_equal(rebuilt, committed, check_dtype=False)
