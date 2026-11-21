"""Unit tests for the v1.3 converter's tricky reads, in both release layouts.

Everything is exercised on workbooks built in the test, so neither 28 MB artifact is a
test fixture. The two layouts differ in exactly the places that would silently produce
wrong tables:

* the changelog is four blocks side by side in one ``Differences with HS98`` sheet in
  ``v1.3_3`` (columns A/I/Q/Y), and four separate ``Differences-*`` sheets in ``v1.3_4``;
* ``v1.3_3`` marks v1.3 changes only as green cell fill, which no cell value records
  (D20), while ``v1.3_4`` adds an explicit column that wins -- and the two can disagree,
  which must be reported rather than hidden;
* ``v1.3_3`` writes the string ``'NA'`` for a placeholder observer's years (D18),
  ``v1.3_4`` leaves them blank. Both must come out null.

The output schema must be identical for both, so that loaders never branch on release.
"""

import datetime as dt
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

openpyxl = pytest.importorskip("openpyxl")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_gn_database_v13.py"


@pytest.fixture(scope="module")
def conv():
    spec = importlib.util.spec_from_file_location("convert_gn_database_v13", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _workbook(path: Path) -> None:
    from openpyxl.styles import PatternFill

    wb = openpyxl.Workbook()
    green = PatternFill(start_color="FF92D050", end_color="FF92D050", fill_type="solid")

    first = wb.active
    first.title = "1610-1946"
    first.append(["YEAR", "MONTH", "DAY", "STATION", "OBSER", "GROUPS"])
    first.append([1610, 1, 1, 0, 0, -1])
    first.append([1611, 3, 4, 2, 1, 3])
    first.append([None, None, None, None, None, None])  # trailing blank row

    second = wb.create_sheet("1947-1988")
    second.append(["YEAR", "MONTH", "DAY", "STATION", "OBSER", "GROUPS"])
    second.append([1947, 1, 1, 332, 1, 7])

    third = wb.create_sheet("1989-2010")
    third.append(["YEAR", "MONTH", "DAY", "STATION", "OBSER", "GROUPS"])
    third.append([2010, 12, 31, 445, 1, 2])

    diff = wb.create_sheet("Differences with HS98")
    diff.append(
        ["v1.12"]
        + [None] * 7
        + ["v1.2"]
        + [None] * 7
        + ["v1.21"]
        + [None] * 7
        + ["v1.3"]
        + [None] * 2
    )
    diff.append(
        ["NAME", "YEAR", "NOTES"]
        + [None] * 5
        + ["NAME", "YEAR", "NOTES"]
        + [None] * 5
        + ["NAME", "YEAR", "NOTES"]
        + [None] * 5
        + ["NAME", "YEAR", "NOTES"]
    )
    row = [None] * 27
    row[0:3] = ["OLD, A.", 1612, "old note"]
    row[8:11] = ["MID, B.", 1616, "mid note"]
    row[24:27] = ["NEW, C.", "1610-1613", "new note with doi"]
    diff.append(row)
    row2 = [None] * 27
    row2[24:27] = ["NEW, D.", 1611, "second new note"]
    diff.append(row2)

    obsv = wb.create_sheet("Observers")
    obsv.append(
        [
            "STATION",
            "INITIAL",
            "FINAL",
            "TOT.OBS",
            "OBSERVER",
            None,
            None,
            None,
            "All new changes are indicated in green",
        ]
    )
    obsv.append([1, 1610, 1613, 214, "HARRIOT, T., OXFORD"])
    obsv.append([2, "NA", "NA", 0, "MARIUS, S., NUREMBERG"])
    obsv.append([3, 1947, 1988, 900, "CHANGED, X."])
    obsv["A4"].fill = green  # only station 3 is flagged

    wb.save(path)


def _workbook_v4(path: Path) -> None:
    """The v1.3_4 layout: OBSERVER header, four changelog sheets, explicit flag column."""
    from openpyxl.styles import PatternFill

    wb = openpyxl.Workbook()
    green = PatternFill(start_color="FF92D050", end_color="FF92D050", fill_type="solid")

    first = wb.active
    first.title = "1610-1946"
    first.append(["YEAR", "MONTH", "DAY", "STATION", "OBSERVER", "GROUPS"])
    first.append([1610, 1, 1, 0, 0, -1])
    first.append([1611, 3, 4, 2, 1, 3])
    first.append([None, None, None, None, None, None])

    second = wb.create_sheet("1947-1988")
    second.append(["YEAR", "MONTH", "DAY", "STATION", "OBSERVER", "GROUPS"])
    second.append([1947, 1, 1, 332, 1, 7])

    third = wb.create_sheet("1989-2010")
    third.append(["YEAR", "MONTH", "DAY", "STATION", "OBSERVER", "GROUPS"])
    third.append([2010, 12, 31, 445, 1, 2])

    obsv = wb.create_sheet("Observers")
    obsv.append(
        [
            "STATION",
            "INITIAL",
            "FINAL",
            "TOTAL OBSERVATIONS",
            "OBSERVER",
            "CHANGED IN v1.3 (1=YES, 0=NOT)",
        ]
    )
    obsv.append([1, 1610, 1613, 214, "HARRIOT, T., OXFORD", 0])
    obsv.append([2, None, None, 0, "MARIUS, S., NUREMBERG", 1])  # column says changed...
    obsv.append([3, 1947, 1988, 900, "CHANGED, X.", 1])
    obsv["A4"].fill = green  # ...but only station 3 carries the old green fill

    for sheet, entries in {
        "Differences-v1_12-HS98": [["OLD, A.", 1612, "old note"]],
        "Differences-v1_2-v1_12": [["MID, B.", 1616, "mid note"]],
        "Differences-v1_21-v1_2": [],
        "Differences-v1_3-v1_21": [
            ["NEW, C.", "1610-1613", "new note with doi"],
            ["NEW, D.", 1611, "second new note"],
        ],
    }.items():
        ws = wb.create_sheet(sheet)
        ws.append([f"CHANGES IN {sheet.split('-')[1]}"])
        ws.append(["NAME", "YEAR", "NOTES"])
        for row in entries:
            ws.append(row)

    wb.save(path)


@pytest.fixture(scope="module")
def book(tmp_path_factory):
    p = tmp_path_factory.mktemp("v13") / "book.xlsx"
    _workbook(p)
    return p


@pytest.fixture(scope="module")
def book_v4(tmp_path_factory):
    p = tmp_path_factory.mktemp("v134") / "book4.xlsx"
    _workbook_v4(p)
    return p


@pytest.fixture(scope="module")
def rel3(conv):
    return conv.RELEASES["v1.3_3"]


@pytest.fixture(scope="module")
def rel4(conv):
    return conv.RELEASES["v1.3_4"]


def test_observations_concatenate_and_derive(conv, book):
    obs = conv.read_observations(book)
    assert len(obs) == 4, "the trailing blank row must be dropped"
    assert obs.groupby("SHEET").size().to_dict() == {"1610-1946": 2, "1947-1988": 1, "1989-2010": 1}
    first = obs.iloc[0]
    assert int(first["ORDINAL"]) == dt.date(1610, 1, 1).toordinal() == 587_676
    assert first["FRACYEAR"] == pytest.approx(1610.0)
    # Last day of a non-leap year: (365 - 1) / 365 past Jan 1.
    last = obs[obs["YEAR"] == 2010].iloc[0]
    assert last["FRACYEAR"] == pytest.approx(2010 + 364 / 365)
    assert list(obs.columns) == [
        "YEAR",
        "MONTH",
        "DAY",
        "ORDINAL",
        "STATION",
        "OBSER",
        "GROUPS",
        "FRACYEAR",
        "SHEET",
    ]


def test_observers_na_sentinel_and_fill(conv, book, rel3):
    df = conv.read_observers(book, rel3)
    assert len(df) == 3
    assert df.loc[df["STATION"] == 2, "INITIAL"].isna().all(), "'NA' must become null (D18)"
    assert df.loc[df["STATION"] == 2, "TOT_OBS"].iloc[0] == 0
    assert df.loc[df["STATION"] == 1, "INITIAL"].iloc[0] == 1610
    # Only the filled row is flagged -- the information is in formatting only (D20).
    assert df.set_index("STATION")["changed_in_v13"].to_dict() == {1: False, 2: False, 3: True}


def test_v134_observers_prefer_the_explicit_column(conv, book_v4, rel4):
    """The column wins, the fill is kept for comparison, and blanks are null (D18, D20)."""
    df = conv.read_observers(book_v4, rel4)
    assert df.set_index("STATION")["changed_in_v13"].to_dict() == {1: False, 2: True, 3: True}
    assert df.set_index("STATION")["_from_fill"].to_dict() == {1: False, 2: False, 3: True}
    assert df.loc[df["STATION"] == 2, "INITIAL"].isna().all()


def test_v134_fill_vs_column_disagreement_is_reported(conv, book_v4, rel4, capsys):
    """A column/fill mismatch is a finding for the authors, not a silent choice."""
    df = conv.read_observers(book_v4, rel4)
    report = conv.fill_vs_column_report(df, rel4)
    assert report == {"n_observers": 3, "from_column": 2, "from_fill": 1, "disagree": 1}
    assert "stations differing: [2]" in capsys.readouterr().out


def test_output_schema_is_release_independent(conv, book, book_v4, rel3, rel4):
    """Loaders must never have to branch on release."""
    three = conv.read_observers(book, rel3).drop(columns=["_from_fill"])
    four = conv.read_observers(book_v4, rel4).drop(columns=["_from_fill"])
    assert list(three.columns) == list(four.columns)
    assert conv.read_observations(book).columns.equals(conv.read_observations(book_v4).columns)
    assert list(conv.read_changelog(book, rel3).columns) == list(
        conv.read_changelog(book_v4, rel4).columns
    )


def test_v134_changelog_reads_four_sheets(conv, book_v4, rel4):
    df = conv.read_changelog(book_v4, rel4)
    assert df["version"].value_counts().to_dict() == {"v1.3": 2, "v1.12": 1, "v1.2": 1}
    assert "v1.21" not in set(df["version"]), "an empty sheet contributes no rows"
    v13 = df[df["version"] == "v1.3"]
    assert list(v13["name"]) == ["NEW, C.", "NEW, D."]
    assert list(v13["year"]) == ["1610-1613", "1611"]


def test_changelog_unstacks_four_blocks(conv, book, rel3):
    df = conv.read_changelog(book, rel3)
    assert df["version"].value_counts().to_dict() == {"v1.3": 2, "v1.12": 1, "v1.2": 1}
    assert "v1.21" not in set(df["version"]), "an empty block contributes no rows"
    v13 = df[df["version"] == "v1.3"]
    assert list(v13["name"]) == ["NEW, C.", "NEW, D."]
    # Years are kept as text because the sheet mixes ints and ranges like '1610-1613'.
    assert list(v13["year"]) == ["1610-1613", "1611"]


def test_parquet_write_is_deterministic(conv, tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    one, two = tmp_path / "one.parquet", tmp_path / "two.parquet"
    conv.write_parquet(df, one)
    conv.write_parquet(df, two)
    assert conv.sha256(one) == conv.sha256(two)
    # No pandas metadata, so the bytes do not carry a library version.
    import pyarrow.parquet as pq

    assert pq.ParquetFile(one).schema_arrow.metadata in (None, {})


def test_superseded_release_may_not_be_written_into_data_raw(conv):
    """ADR-010's third condition, enforced rather than documented."""
    from gn_next_gen.paths import DATA_INTERIM, GN_DATABASE_V13

    assert conv.default_out_dir(conv.RELEASES["v1.3_4"]) == GN_DATABASE_V13
    out = conv.default_out_dir(conv.RELEASES["v1.3_3"])
    assert DATA_INTERIM in out.parents
    assert GN_DATABASE_V13 not in out.parents
