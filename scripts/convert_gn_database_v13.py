"""Convert the v1.3 revised collection workbook into frozen, checksummed tables.

The upstream artifact is an .xlsx (``Revised collection_v1.3_3.xlsx``).  Parsing
1.16 M rows out of it with openpyxl takes minutes, so the conversion is frozen
beside it in ``data/raw/`` and recorded in ``MANIFEST.csv`` rather than being
redone on every run.  This mirrors ``gn_database_v1.22/``, whose manifest row
likewise records it as a conversion ("converted to CSV with ORDINAL and FRACYEAR
added").

Two things in the workbook are not in its cell *values* and would be lost by any
re-export, so they are rescued here:

* the ``Observers`` sheet marks what changed in v1.3 only as green cell fill
  (its own header says "All new changes are indicated in green") -> ``changed_in_v13``;
* ``Differences with HS98`` is four changelogs side by side in one sheet
  (v1.12, v1.2, v1.21, v1.3) -> unstacked to one long table.

Outputs are written with the pandas metadata stripped from the parquet schema so
the files are byte-reproducible for a given input workbook.

Usage
-----
    python scripts/convert_gn_database_v13.py [--check]

``--check`` rewrites to a temporary directory and compares against the committed
files instead of overwriting them.
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import hashlib
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gn_next_gen.paths import GN_DATABASE_V13, GN_V13_XLSX  # noqa: E402

OBS_SHEETS = ("1610-1946", "1947-1988", "1989-2010")
CHANGELOG_BLOCKS = {"v1.12": 0, "v1.2": 8, "v1.21": 16, "v1.3": 24}
OBS_COLUMNS = ["YEAR", "MONTH", "DAY", "STATION", "OBSER", "GROUPS"]


def _ordinal(year: int, month: int, day: int) -> int:
    return dt.date(year, month, day).toordinal()


def _fracyear(year: int, ordinal: int) -> float:
    """YEAR + (ORDINAL - ordinal(Jan 1)) / days_in_year, as in v1.22."""
    jan1 = dt.date(year, 1, 1).toordinal()
    n_days = 366 if calendar.isleap(year) else 365
    return year + (ordinal - jan1) / n_days


def read_observations(xlsx: Path) -> pd.DataFrame:
    """Concatenate the three observation sheets and add ORDINAL and FRACYEAR.

    The ``1610-1946`` sheet carries roughly 96,000 trailing blank rows; they are
    dropped here, so the row counts this returns are the post-drop truth.
    """
    import openpyxl

    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    frames = []
    for sheet in OBS_SHEETS:
        ws = wb[sheet]
        rows = []
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i == 0:
                continue
            if row[0] is None:  # trailing blank row
                continue
            rows.append(row[:6])
        df = pd.DataFrame(rows, columns=OBS_COLUMNS)
        df["SHEET"] = sheet
        print(f"  {sheet}: {len(df):,} data rows")
        frames.append(df)
    wb.close()

    obs = pd.concat(frames, ignore_index=True)
    for col in OBS_COLUMNS:
        obs[col] = obs[col].astype("int64")
    obs["ORDINAL"] = [
        _ordinal(y, m, d) for y, m, d in zip(obs["YEAR"], obs["MONTH"], obs["DAY"], strict=True)
    ]
    obs["FRACYEAR"] = [_fracyear(y, o) for y, o in zip(obs["YEAR"], obs["ORDINAL"], strict=True)]
    return obs[
        [
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
    ]


def read_observers(xlsx: Path) -> pd.DataFrame:
    """Read the Observers sheet, converting the 'NA' sentinel and reading fills.

    ``INITIAL``/``FINAL`` carry the *string* ``'NA'`` for observers that are listed
    but contribute no observations (``TOT.OBS == 0``); those become nulls.
    ``changed_in_v13`` comes from the green cell fill, which is the only place that
    information exists.
    """
    import openpyxl

    wb = openpyxl.load_workbook(xlsx, data_only=True)
    ws = wb["Observers"]
    recs = []
    for row in ws.iter_rows(min_row=2, max_col=5):
        station = row[0].value
        if station is None:
            continue
        recs.append(
            {
                "STATION": int(station),
                "INITIAL": row[1].value,
                "FINAL": row[2].value,
                "TOT_OBS": int(row[3].value or 0),
                "OBSERVER": row[4].value,
                "changed_in_v13": row[0].fill.patternType is not None,
            }
        )
    wb.close()

    df = pd.DataFrame(recs)
    for col in ("INITIAL", "FINAL"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df.sort_values("STATION", ignore_index=True)


def read_changelog(xlsx: Path) -> pd.DataFrame:
    """Unstack the four side-by-side version blocks of 'Differences with HS98'."""
    import openpyxl

    wb = openpyxl.load_workbook(xlsx, data_only=True)
    ws = wb["Differences with HS98"]
    rows = list(ws.iter_rows(min_row=3, values_only=True))
    wb.close()

    recs = []
    for version, off in CHANGELOG_BLOCKS.items():
        for row in rows:
            block = row[off : off + 3]
            if all(v is None for v in block):
                continue
            recs.append(
                {
                    "version": version,
                    "name": block[0],
                    "year": None if block[1] is None else str(block[1]),
                    "note": block[2],
                }
            )
    df = pd.DataFrame(recs)
    print("  changelog entries per version:", df["version"].value_counts().to_dict())
    return df


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write deterministically: no pandas metadata, fixed compression."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata(None)
    pq.write_table(table, path, compression="zstd", compression_level=9)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="compare instead of overwrite")
    ap.add_argument("--out-dir", type=Path, default=GN_DATABASE_V13)
    args = ap.parse_args()

    xlsx = GN_V13_XLSX
    if not xlsx.is_file():
        raise SystemExit(f"missing workbook: {xlsx}")
    print(f"reading {xlsx.name} ({xlsx.stat().st_size:,} bytes)")

    print("observations:")
    obs = read_observations(xlsx)
    print("observers:")
    obsv = read_observers(xlsx)
    print("changelog:")
    chg = read_changelog(xlsx)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    targets = {
        "gn_v13_observations.parquet": (write_parquet, obs),
        "gn_v13_observers.parquet": (write_parquet, obsv),
    }

    if args.check:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ok = True
            for name, (writer, df) in targets.items():
                writer(df, tmp / name)
                a, b = sha256(tmp / name), sha256(out / name)
                print(f"  {name}: {'MATCH' if a == b else 'DIFFERS'}")
                ok &= a == b
            chg.to_csv(tmp / "gn_v13_changelog.csv", index=False, lineterminator="\n")
            a = sha256(tmp / "gn_v13_changelog.csv")
            b = sha256(out / "gn_v13_changelog.csv")
            print(f"  gn_v13_changelog.csv: {'MATCH' if a == b else 'DIFFERS'}")
            ok &= a == b
        raise SystemExit(0 if ok else 1)

    for name, (writer, df) in targets.items():
        writer(df, out / name)
        print(f"  wrote {name}: {len(df):,} rows, {(out / name).stat().st_size:,} bytes")
    chg.to_csv(out / "gn_v13_changelog.csv", index=False, lineterminator="\n")
    print(f"  wrote gn_v13_changelog.csv: {len(chg):,} rows")

    print("\nsummary")
    print(f"  data rows                {len(obs):,}")
    print(f"  rows with STATION > 0    {(obs['STATION'] > 0).sum():,}")
    print(f"  distinct stations w/data  {obs.loc[obs['STATION'] > 0, 'STATION'].nunique()}")
    print(f"  observers listed         {len(obsv)}")
    print(f"  changed_in_v13           {int(obsv['changed_in_v13'].sum())}")
    print(f"  coverage                 {obs['ORDINAL'].min()} -> {obs['ORDINAL'].max()}")
    print(f"  max GROUPS               {obs['GROUPS'].max()}")


if __name__ == "__main__":
    main()
