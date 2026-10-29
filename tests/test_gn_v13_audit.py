"""Keep the report to the v1.3 authors tied to the data and to the other tests.

``docs/results/gn-v13-feedback-report.md`` goes outside this group, so the numbers
in it must not drift away from the workbook. The audit script is the single source
of those numbers; this checks that the script agrees with the figures pinned in
``test_gn_database_v13.py`` and that the report quotes the same ones.

The audit itself parses a 28 MB workbook twice, so running it is marked slow. The
cheap half -- that the report's prose matches the pinned constants -- always runs.
"""

import json
import re
import subprocess
import sys
import pytest

from gn_next_gen.paths import DOCS, REPO_ROOT, SCRIPTS

from test_gn_database_v13 import (  # noqa: I001  (sibling test module, shared constants)
    CHANGELOG_COUNTS,
    OBSERVERS_LISTED,
    STATION_ROWS,
    STATIONS_WITH_DATA,
    TOTAL_ROWS,
)

REPORT = DOCS / "results" / "gn-v13-feedback-report.md"
AUDIT = SCRIPTS / "audit_gn_database_v13.py"

# The findings the report is built around.  Keep in step with D16-D20.
DUPLICATE_KEYS = 217
DUPLICATE_IDENTICAL = 122
DUPLICATE_CONFLICTING = 95
DUPLICATES_BY_STATION = {755: 94, 329: 76, 245: 41, 773: 4, 768: 2}
PLACEHOLDER_OBSERVERS = 33
FILLED_OBSERVERS = 94
BLANK_TAIL_1610 = 96_662


def _numbers(text: str) -> set[int]:
    """Every integer in the prose, with thousands separators resolved."""
    return {
        int(m.replace(",", ""))
        for m in re.findall(r"\b\d[\d,]*\b", text)
        if len(m.replace(",", "")) <= 9
    }


@pytest.fixture(scope="module")
def report() -> str:
    assert REPORT.is_file(), f"missing report: {REPORT}"
    return REPORT.read_text(encoding="utf-8")


def test_report_states_its_audience(report):
    """It is written for the v1.3 authors, not for the PI; that must be on its face."""
    head = report[:600]
    assert "**For:**" in head
    assert "v1.3 revised collection" in head


def test_report_quotes_the_pinned_numbers(report):
    present = _numbers(report)
    for value in (
        TOTAL_ROWS,
        STATION_ROWS,
        STATIONS_WITH_DATA,
        OBSERVERS_LISTED,
        DUPLICATE_KEYS,
        DUPLICATE_IDENTICAL,
        DUPLICATE_CONFLICTING,
        PLACEHOLDER_OBSERVERS,
        FILLED_OBSERVERS,
        BLANK_TAIL_1610,
        CHANGELOG_COUNTS["v1.3"],
    ):
        assert value in present, f"report no longer quotes {value}"


def test_report_names_the_reproduction_command(report):
    assert "audit_gn_database_v13.py" in report
    assert AUDIT.is_file()


def test_report_asks_rather_than_asserts_on_the_ambiguous_finding(report):
    """The duplicate keys may be deliberate, so the report must not call them an error."""
    section = report[report.index("## 1."): report.index("## 2.")]
    assert "question" in section.lower()
    assert "not guessed" in section.lower() or "not** guessed" in section.lower()


def test_report_records_what_verified_clean(report):
    """An audit, not a complaint: the clean checks have to be in there too."""
    assert "verified as clean" in report.lower()
    for phrase in ("February 29", "sentinel", "Coverage"):
        assert phrase in report


@pytest.mark.slow
def test_audit_script_reproduces_the_pinned_numbers(tmp_path):
    out = tmp_path / "audit.json"
    proc = subprocess.run(
        [sys.executable, str(AUDIT), "--json", str(out)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    got = json.loads(out.read_text())

    assert got["total_rows"] == TOTAL_ROWS
    assert got["station_rows"] == STATION_ROWS
    assert got["stations_with_data"] == STATIONS_WITH_DATA
    assert got["observers_listed"] == OBSERVERS_LISTED
    assert got["duplicate_keys"] == DUPLICATE_KEYS
    assert got["duplicate_identical"] == DUPLICATE_IDENTICAL
    assert got["duplicate_conflicting"] == DUPLICATE_CONFLICTING
    assert {int(k): v for k, v in got["duplicates_by_station"].items()} == DUPLICATES_BY_STATION
    assert got["placeholder_observers"] == PLACEHOLDER_OBSERVERS
    assert got["filled_observers"] == FILLED_OBSERVERS
    assert got["changelog"] == CHANGELOG_COUNTS
    assert got["blank_tail"]["1610-1946"] == BLANK_TAIL_1610
    assert got["coverage"] == ["1610-01-01", "2010-12-31"]
    # The extremes the report discusses, exactly these two and no others.
    assert sorted(tuple(e) for e in got["extremes"]) == [
        [1982, 9, 26, 768, 93, "1947-1988"],
        [1992, 3, 18, 712, 58, "1989-2010"],
    ] or sorted(got["extremes"]) == [
        [1982, 9, 26, 768, 93, "1947-1988"],
        [1992, 3, 18, 712, 58, "1989-2010"],
    ]
    # Everything that should be zero.
    assert got["clean"] == {
        "bad_dates": 0,
        "feb29": 0,
        "neg_with_station": 0,
        "neg_without_minus1": 0,
        "stations_missing_from_observers": 0,
    }
