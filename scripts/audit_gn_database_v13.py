"""Audit the v1.3 revised collection workbook and print every number in the report.

``docs/results/gn-v13-feedback-report.md`` is written for the authors of the v1.3
collection, so every claim in it has to be reproducible by them from the workbook
alone.  This script reads the .xlsx directly (not this repository's converted
parquet) and prints each figure the report quotes, so a reader can re-run it
against their own copy.

Usage
-----
    python scripts/audit_gn_database_v13.py [--json OUT.json]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gn_next_gen.paths import GN_V13_XLSX  # noqa: E402

OBS_SHEETS = ("1610-1946", "1947-1988", "1989-2010")
CHANGELOG_BLOCKS = {"v1.12": 0, "v1.2": 8, "v1.21": 16, "v1.3": 24}
EXTREME_FLOOR = 54  # report every GROUPS value at or above this


def _hr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, help="also write the findings as JSON")
    args = ap.parse_args()

    import openpyxl

    print(f"workbook: {GN_V13_XLSX}")
    print(f"size:     {GN_V13_XLSX.stat().st_size:,} bytes")

    wb = openpyxl.load_workbook(GN_V13_XLSX, read_only=True, data_only=True)
    print(f"sheets:   {wb.sheetnames}")

    per_sheet: dict[str, int] = {}
    blank_tail: dict[str, int] = {}
    station_rows = 0
    stations_with_data: set[int] = set()
    groups_hist: Counter[int] = Counter()
    extremes: list[tuple] = []
    pairs: dict[tuple, list] = defaultdict(list)
    feb29 = []
    bad_dates = []
    sentinel_violations = {"neg_with_station": 0, "neg_without_minus1": 0}
    ord_min, ord_max = None, None

    for sheet in OBS_SHEETS:
        ws = wb[sheet]
        declared = ws.max_row
        n = 0
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i == 0:
                continue
            if row[0] is None:
                continue
            n += 1
            y, m, d, st, _ob, g = (row[j] for j in range(6))
            try:
                o = dt.date(int(y), int(m), int(d)).toordinal()
            except ValueError:
                bad_dates.append((y, m, d, sheet))
                continue
            ord_min = o if ord_min is None else min(ord_min, o)
            ord_max = o if ord_max is None else max(ord_max, o)
            if m == 2 and d == 29 and y in (1700, 1800, 1900):
                feb29.append((y, sheet))
            if st and st > 0:
                station_rows += 1
                stations_with_data.add(int(st))
                groups_hist[g] += 1
                if g is not None and g >= EXTREME_FLOOR:
                    extremes.append((int(y), int(m), int(d), int(st), int(g), sheet))
                if g is not None and g < 0:
                    sentinel_violations["neg_with_station"] += 1
            elif g != -1:
                sentinel_violations["neg_without_minus1"] += 1
            pairs[((int(y), int(m), int(d)), int(st or 0))].append(g)
        per_sheet[sheet] = n
        blank_tail[sheet] = declared - 1 - n
        print(f"  {sheet}: {n:,} data rows ({blank_tail[sheet]:,} trailing blank rows)")

    ws = wb["Observers"]
    observers = [[c.value for c in r[:5]] for r in ws.iter_rows(min_row=2, max_col=5)]
    observers = [r for r in observers if r[0] is not None]
    placeholders = [
        r for r in observers if not isinstance(r[1], int) or not isinstance(r[2], int) or not r[3]
    ]
    wb.close()

    # Fills need a non-read-only workbook.
    wb2 = openpyxl.load_workbook(GN_V13_XLSX, data_only=True)
    ws2 = wb2["Observers"]
    filled = [
        (r[0].value, r[4].value)
        for r in ws2.iter_rows(min_row=2, max_col=5)
        if r[0].value is not None and r[0].fill.patternType is not None
    ]
    ws3 = wb2["Differences with HS98"]
    rows = list(ws3.iter_rows(min_row=3, values_only=True))
    changelog = {
        v: sum(1 for row in rows if any(x is not None for x in row[o : o + 3]))
        for v, o in CHANGELOG_BLOCKS.items()
    }
    wb2.close()

    total_rows = sum(per_sheet.values())
    dups = {k: v for k, v in pairs.items() if len(v) > 1}
    dup_identical = sum(1 for v in dups.values() if len(set(v)) == 1)
    dup_conflicting = len(dups) - dup_identical
    listed_stations = {int(r[0]) for r in observers}

    _hr("SCALE")
    print(f"total data rows            {total_rows:,}")
    print(f"rows with STATION > 0      {station_rows:,}")
    print(f"distinct stations w/ data  {len(stations_with_data)}")
    print(f"observers listed           {len(observers)}")
    span = f"{dt.date.fromordinal(ord_min)} -> {dt.date.fromordinal(ord_max)}"
    print(f"coverage                   {span}")

    _hr("ISSUE 1 - duplicate (date, station) keys")
    print(f"duplicate keys             {len(dups)}")
    print(f"  identical GROUPS         {dup_identical}")
    print(f"  conflicting GROUPS       {dup_conflicting}")
    print(f"by station: {Counter(k[1] for k in dups).most_common()}")
    print(f"by year (top 10): {Counter(k[0][0] for k in dups).most_common(10)}")
    print("examples with conflicting values:")
    shown = 0
    for k, v in sorted(dups.items()):
        if len(set(v)) > 1:
            print(f"  station {k[1]:>4}  {k[0][0]:04d}-{k[0][1]:02d}-{k[0][2]:02d}  GROUPS = {v}")
            shown += 1
            if shown == 8:
                break

    _hr("ISSUE 2 - extreme GROUPS values")
    print(f"values >= {EXTREME_FLOOR}: {len(extremes)}")
    for e in sorted(extremes, key=lambda x: -x[4]):
        print(f"  {e[0]:04d}-{e[1]:02d}-{e[2]:02d}  station {e[3]}  GROUPS = {e[4]}  [{e[5]}]")
    tail = sorted(g for g in groups_hist if g is not None and g > 40)
    print(f"all GROUPS values above 40: {tail}")

    _hr("ISSUE 3 - observers listed with no observations")
    print(f"placeholder observers      {len(placeholders)}")
    for r in placeholders[:8]:
        print(f"  station {r[0]:>4}  INITIAL={r[1]!r} FINAL={r[2]!r} TOT.OBS={r[3]!r}  {r[4]}")

    _hr("ISSUE 4 - change flags exist only as cell fill")
    print(f"observers with a filled STATION cell  {len(filled)} of {len(observers)}")
    print(f"  e.g. {[f[0] for f in filled[:12]]}")

    _hr("ISSUE 5 - layout and schema notes")
    print(f"changelog entries per version  {changelog}")
    print("  (four blocks side by side in one sheet at columns A/I/Q/Y)")
    print(f"trailing blank rows per sheet  {blank_tail}")
    print("observation columns are YEAR, MONTH, DAY, STATION, OBSER, GROUPS")
    print("  (v1.22 as distributed downstream also carried ORDINAL and FRACYEAR)")

    _hr("WHAT VERIFIED CLEAN")
    print(f"invalid calendar dates                      {len(bad_dates)}")
    print(f"Feb 29 of 1700/1800/1900 present            {len(feb29)} (expected 0)")
    print(f"GROUPS < 0 while STATION > 0                {sentinel_violations['neg_with_station']}")
    print("STATION == 0 rows whose GROUPS is not -1   ", sentinel_violations["neg_without_minus1"])
    print("stations with data missing from Observers   ", len(stations_with_data - listed_stations))
    print(
        "max station code in data / listed           ",
        f"{max(stations_with_data)} / {max(listed_stations)}",
    )

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "workbook_bytes": GN_V13_XLSX.stat().st_size,
                    "per_sheet_rows": per_sheet,
                    "blank_tail": blank_tail,
                    "total_rows": total_rows,
                    "station_rows": station_rows,
                    "stations_with_data": len(stations_with_data),
                    "observers_listed": len(observers),
                    "coverage": [
                        str(dt.date.fromordinal(ord_min)),
                        str(dt.date.fromordinal(ord_max)),
                    ],
                    "duplicate_keys": len(dups),
                    "duplicate_identical": dup_identical,
                    "duplicate_conflicting": dup_conflicting,
                    "duplicates_by_station": dict(Counter(k[1] for k in dups)),
                    "extremes": extremes,
                    "placeholder_observers": len(placeholders),
                    "filled_observers": len(filled),
                    "changelog": changelog,
                    "clean": {
                        "bad_dates": len(bad_dates),
                        "feb29": len(feb29),
                        **sentinel_violations,
                        "stations_missing_from_observers": len(
                            stations_with_data - listed_stations
                        ),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nJSON written to {args.json}")


if __name__ == "__main__":
    main()
