"""Convert a v1.3 revised-collection workbook into frozen, checksummed tables.

Two releases of the same database are in ``data/raw/gn_database_v1.3/``:

* ``Revised collection_v1.3_4.xlsx`` -- **current**.  Landed 2026-11-21; fixes most of
  what ``docs/results/gn-v13-feedback-report.md`` reported upstream (D16 duplicate keys
  217 -> 3, D17's implausible ``GROUPS = 93`` gone, D18's ``'NA'`` sentinel gone, D19's
  ``OBSER`` renamed ``OBSERVER``, D20's green-fill-only flag now an explicit column).
* ``Revised collection_v1.3_3.xlsx`` -- superseded, retained byte-exact.

Per **ADR-010**, the derived tables in ``data/raw/`` track the newest release and an
earlier release is regenerated on demand into ``data/interim/``; this script refuses to
write a superseded release into ``data/raw/``.

Parsing 1.16 M rows out of an .xlsx with openpyxl takes minutes, so the conversion is
frozen beside the workbook and recorded in ``MANIFEST.csv`` rather than redone per run.
This mirrors ``gn_database_v1.22/``, whose manifest row likewise records it as a
conversion ("converted to CSV with ORDINAL and FRACYEAR added").

What the releases do not express in cell *values*, and would lose in any re-export, is
rescued here:

* in ``v1.3_3`` the ``Observers`` sheet marks what changed only as green cell fill (its
  own header says "All new changes are indicated in green") -> ``changed_in_v13``;
  ``v1.3_4`` adds an explicit ``CHANGED IN v1.3`` column, which wins, with the fill read
  alongside as a cross-check (D20);
* ``v1.3_3`` holds four changelogs side by side in one ``Differences with HS98`` sheet;
  ``v1.3_4`` splits them into four ``Differences-*`` sheets.  Both unstack to one long
  table with the same schema.

The output schema is identical for both releases -- including the flag column, which
stays ``OBSER`` whatever the sheet header says -- so loaders never branch on release.
``ORDINAL`` and ``FRACYEAR`` are derived with v1.22's definitions (1610-01-01 = 587676),
since no release carries them (D19).

Outputs are written with the pandas metadata stripped from the parquet schema so the
files are byte-reproducible for a given input workbook.

Usage
-----
    python scripts/convert_gn_database_v13.py [--release v1.3_4] [--check]

``--check`` rewrites to a temporary directory and compares against the files already in
the output directory instead of overwriting them.
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gn_next_gen.paths import (  # noqa: E402
    DATA_INTERIM,
    GN_DATABASE_V13,
    GN_V13_RELEASES,
    GN_V13_CURRENT_RELEASE,
)

OBS_SHEETS = ("1610-1946", "1947-1988", "1989-2010")
CHANGELOG_BLOCKS = {"v1.12": 0, "v1.2": 8, "v1.21": 16, "v1.3": 24}
CHANGELOG_SHEETS = {
    "Differences-v1_12-HS98": "v1.12",
    "Differences-v1_2-v1_12": "v1.2",
    "Differences-v1_21-v1_2": "v1.21",
    "Differences-v1_3-v1_21": "v1.3",
}
# The output column stays OBSER for every release, whatever the sheet header reads (D19).
OBS_COLUMNS = ["YEAR", "MONTH", "DAY", "STATION", "OBSER", "GROUPS"]


@dataclass(frozen=True)
class Release:
    """How one workbook release differs in layout.  Output schema is identical."""

    name: str
    changed_flag_from: str  # "fill" (v1.3_3) or "column" (v1.3_4)
    changelog_layout: str  # "blocks" (one sheet) or "sheets" (four sheets)
    observer_max_col: int

    @property
    def is_current(self) -> bool:
        return self.name == GN_V13_CURRENT_RELEASE


RELEASES = {
    "v1.3_3": Release("v1.3_3", changed_flag_from="fill", changelog_layout="blocks", observer_max_col=5),
    "v1.3_4": Release("v1.3_4", changed_flag_from="column", changelog_layout="sheets", observer_max_col=6),
}


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


def read_observers(xlsx: Path, release: Release) -> pd.DataFrame:
    """Read the Observers sheet: the 'NA' sentinel, and where the v1.3 flag lives.

    ``INITIAL``/``FINAL`` carry the *string* ``'NA'`` in ``v1.3_3`` for observers that are
    listed but contribute no observations (``TOT.OBS == 0``); those become nulls (D18).
    ``v1.3_4`` no longer uses the sentinel, and the same coercion leaves its blanks null.

    ``changed_in_v13`` comes from the green cell fill in ``v1.3_3``, which is the only
    place the information exists, and from the explicit ``CHANGED IN v1.3`` column in
    ``v1.3_4``, which wins.  The fill is read for both so the two can be compared: a
    disagreement is reported by :func:`fill_vs_column_report` rather than hidden (D20).
    """
    import openpyxl

    wb = openpyxl.load_workbook(xlsx, data_only=True)
    ws = wb["Observers"]
    recs = []
    for row in ws.iter_rows(min_row=2, max_col=release.observer_max_col):
        station = row[0].value
        if station is None:
            continue
        from_fill = row[0].fill.patternType is not None
        if release.changed_flag_from == "column":
            flag = bool(int(row[5].value or 0))
        else:
            flag = from_fill
        recs.append(
            {
                "STATION": int(station),
                "INITIAL": row[1].value,
                "FINAL": row[2].value,
                "TOT_OBS": int(row[3].value or 0),
                "OBSERVER": row[4].value,
                "changed_in_v13": flag,
                "_from_fill": from_fill,
            }
        )
    wb.close()

    df = pd.DataFrame(recs)
    for col in ("INITIAL", "FINAL"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df.sort_values("STATION", ignore_index=True)


def fill_vs_column_report(obsv: pd.DataFrame, release: Release) -> dict[str, int]:
    """Compare the green-fill flag with the explicit column (D20).

    Returns counts rather than raising: in ``v1.3_4`` the fill marks 99 observers and the
    column marks its own set, and which observers differ is a finding to report upstream,
    not a reason to refuse the conversion.  A test pins these numbers so a silent change
    in either encoding fails loudly.
    """
    report = {
        "n_observers": len(obsv),
        "from_column": int(obsv["changed_in_v13"].sum()),
        "from_fill": int(obsv["_from_fill"].sum()),
        "disagree": int((obsv["changed_in_v13"] != obsv["_from_fill"]).sum()),
    }
    print(
        f"  changed_in_v13 ({release.changed_flag_from} is authoritative): "
        f"column {report['from_column']}, fill {report['from_fill']}, "
        f"disagree on {report['disagree']}"
    )
    if report["disagree"]:
        who = obsv.loc[obsv["changed_in_v13"] != obsv["_from_fill"], "STATION"].tolist()
        print(f"    stations differing: {who}")
    return report


def read_changelog(xlsx: Path, release: Release) -> pd.DataFrame:
    """Unstack the changelog to one long table, whichever layout the release uses.

    ``v1.3_3`` holds four version blocks side by side in ``Differences with HS98``;
    ``v1.3_4`` splits them into four ``Differences-*`` sheets, each with a title row, a
    ``NAME/YEAR/NOTES`` header and then entries.
    """
    import openpyxl

    wb = openpyxl.load_workbook(xlsx, data_only=True)
    recs = []
    if release.changelog_layout == "blocks":
        rows = list(wb["Differences with HS98"].iter_rows(min_row=3, values_only=True))
        for version, off in CHANGELOG_BLOCKS.items():
            for row in rows:
                block = row[off : off + 3]
                if all(v is None for v in block):
                    continue
                recs.append(_changelog_record(version, block))
    else:
        for sheet, version in CHANGELOG_SHEETS.items():
            for row in wb[sheet].iter_rows(min_row=3, values_only=True):
                block = row[:3]
                if all(v is None for v in block):
                    continue
                recs.append(_changelog_record(version, block))
    wb.close()

    df = pd.DataFrame(recs)
    print("  changelog entries per version:", df["version"].value_counts().to_dict())
    return df


def _changelog_record(version: str, block: tuple) -> dict:
    return {
        "version": version,
        "name": block[0],
        "year": None if block[1] is None else str(block[1]),
        "note": block[2],
    }


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


def default_out_dir(release: Release) -> Path:
    """ADR-010: the current release owns ``data/raw/``; older ones go to ``data/interim/``."""
    if release.is_current:
        return GN_DATABASE_V13
    return DATA_INTERIM / f"gn_database_{release.name}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--release",
        choices=sorted(RELEASES),
        default=GN_V13_CURRENT_RELEASE,
        help=f"workbook release to convert (default {GN_V13_CURRENT_RELEASE})",
    )
    ap.add_argument("--check", action="store_true", help="compare instead of overwrite")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    release = RELEASES[args.release]
    xlsx = GN_V13_RELEASES[release.name]
    if not xlsx.is_file():
        raise SystemExit(f"missing workbook: {xlsx}")

    out = args.out_dir or default_out_dir(release)
    if not release.is_current and not args.check and GN_DATABASE_V13 in (out, *out.parents):
        raise SystemExit(
            f"refusing to write superseded release {release.name} into {GN_DATABASE_V13}: "
            "ADR-010 keeps data/raw on the current release and regenerates older ones "
            "into data/interim/"
        )

    print(f"reading {xlsx.name} ({xlsx.stat().st_size:,} bytes) as release {release.name}")

    print("observations:")
    obs = read_observations(xlsx)
    print("observers:")
    obsv = read_observers(xlsx, release)
    flag_report = fill_vs_column_report(obsv, release)
    obsv = obsv.drop(columns=["_from_fill"])
    print("changelog:")
    chg = read_changelog(xlsx, release)

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

    real = obs.loc[obs["STATION"] > 0]
    dup = real.groupby(["ORDINAL", "STATION"])["GROUPS"].agg(["size", "nunique"])
    dup = dup[dup["size"] > 1]
    print(f"\nsummary ({release.name} -> {out})")
    print(f"  data rows                 {len(obs):,}")
    print(f"  rows with STATION > 0     {len(real):,}")
    print(f"  distinct stations w/data  {real['STATION'].nunique()}")
    print(f"  observers listed          {len(obsv)}")
    print(f"  changed_in_v13            {flag_report['from_column']} "
          f"(fill {flag_report['from_fill']}, disagree {flag_report['disagree']})")
    print(f"  coverage                  {obs['ORDINAL'].min()} -> {obs['ORDINAL'].max()}")
    print(f"  max GROUPS                {obs['GROUPS'].max()}  (D17)")
    print(f"  duplicate (date,station)  {len(dup)} keys, "
          f"{int((dup['nunique'] > 1).sum())} conflicting  (D16)")


if __name__ == "__main__":
    main()
