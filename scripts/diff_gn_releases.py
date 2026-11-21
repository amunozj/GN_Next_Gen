"""Diff two v1.3 releases and write the small per-year summary the figures plot.

ADR-010 keeps only the current release's tables in ``data/raw``, so this regenerates
the superseded one into ``data/interim/`` first (condition 3) and compares there.
The output is a committed summary small enough to live in the repository, which is
what ``gn_next_gen.viz.data.release_diff_figure`` reads -- so the figure never needs
a 28 MB workbook or a ten-minute conversion.

Usage
-----
    python scripts/diff_gn_releases.py [--old v1.3_3] [--new v1.3_4]
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gn_next_gen.paths import DATA_INTERIM, DOCS, GN_V13_OBSERVATIONS, SCRIPTS  # noqa: E402

OUT = DOCS / "results" / "data" / "gn_v13_release_diff.csv"


def ensure_release(release: str) -> Path:
    """Regenerate a superseded release's tables into ``data/interim`` if needed."""
    out_dir = DATA_INTERIM / f"gn_database_{release}"
    table = out_dir / "gn_v13_observations.parquet"
    if table.is_file():
        print(f"  {release}: reusing {table}")
        return table
    print(f"  {release}: regenerating into {out_dir} (parses a 28 MB workbook)")
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "convert_gn_database_v13.py"),
            "--release",
            release,
            "--out-dir",
            str(out_dir),
        ],
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"conversion of {release} failed")
    return table


def real_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["ORDINAL", "STATION", "GROUPS", "YEAR"])
    return frame[frame["STATION"] > 0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", default="v1.3_3")
    ap.add_argument("--new", default="v1.3_4")
    args = ap.parse_args()

    print("releases:")
    old = real_rows(ensure_release(args.old))
    new = real_rows(GN_V13_OBSERVATIONS)

    old_keys = pd.MultiIndex.from_arrays([old["ORDINAL"], old["STATION"]])
    new_keys = pd.MultiIndex.from_arrays([new["ORDINAL"], new["STATION"]])

    old_max = old.groupby(["ORDINAL", "STATION"])["GROUPS"].max()
    new_max = new.groupby(["ORDINAL", "STATION"])["GROUPS"].max()
    shared = old_max.index.intersection(new_max.index)
    changed = shared[(old_max.loc[shared] != new_max.loc[shared]).to_numpy()]

    only_old = old_max.index.difference(new_max.index)
    only_new = new_max.index.difference(old_max.index)

    def dup_keys(frame: pd.DataFrame) -> pd.Series:
        sizes = frame.groupby(["ORDINAL", "STATION"]).size()
        return sizes[sizes > 1]

    dup_old, dup_new = dup_keys(old), dup_keys(new)

    def years_of(index) -> np.ndarray:
        ordinals = index.get_level_values(0).to_numpy()
        unique, inverse = np.unique(ordinals, return_inverse=True)
        years = np.array([dt.date.fromordinal(int(o)).year for o in unique])
        return years[inverse]

    pieces = {
        "only_in_v3": years_of(only_old),
        "only_in_v4": years_of(only_new),
        "changed_groups": years_of(changed),
        "duplicate_keys_v3": years_of(dup_old.index),
        "duplicate_keys_v4": years_of(dup_new.index),
    }
    all_years = sorted({int(y) for arr in pieces.values() for y in arr})
    table = pd.DataFrame({"year": all_years}).set_index("year")
    for name, years in pieces.items():
        table[name] = pd.Series(years).value_counts().reindex(all_years, fill_value=0).to_numpy()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.reset_index().to_csv(OUT, index=False, lineterminator="\n")

    print(f"\nwrote {OUT} ({len(table)} years with any difference)")
    print(f"  rows            {len(old):,} -> {len(new):,}")
    print(
        f"  observer-days   only in {args.old}: {len(only_old)}, "
        f"only in {args.new}: {len(only_new)}"
    )
    print(f"  changed GROUPS  {len(changed)}")
    print(f"  duplicate keys  {len(dup_old)} -> {len(dup_new)}")
    print(f"  max GROUPS      {old['GROUPS'].max()} -> {new['GROUPS'].max()}")
    assert not old_keys.duplicated().all()
    assert not new_keys.duplicated().all()


if __name__ == "__main__":
    main()
