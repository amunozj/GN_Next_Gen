"""Freeze the published GN series this project compares against.

Moved into Step 2 by the PI on 2026-11-21: the benchmark is where they are
actually consumed.  They are **comparison only** -- the Step 2 scorer never
reads them and no run may calibrate on them (ADR-002, ADR-003); they exist so a
reader can see the benchmark's baselines beside the real series the same methods
produced.

Two provenance routes, deliberately kept apart:

**From the read-only 2017 MATLAB snapshot** (``legacy/GN_Series_matlab/data/``):
Hoyt & Schatten 1998, the Svalgaard & Schatten backbone, a Usoskin ADF-family
series, and Vaquero's and Zolotova's reconstructions.  Exporting from the
snapshot beats fetching them again -- these are the exact bytes the legacy
analyses used, so a comparison against the legacy baselines is like for like.
Their one departure from verbatim: ``gn = -1`` with ``n_observations = 0`` is the
snapshot's no-data sentinel, and is mapped to null rather than carried through as
a negative group count (seven rows, 2026-11-23).

**From upstream supplements frozen under this folder** (2026-11-23), the two the
Step 2 plan named and Step 2 could not get:

- ``chatzistergos_2017/`` -- CDS catalogue J/A+A/602/A69, downloaded from the
  NAOJ VizieR mirror.  CDS's own ``/ftp/`` front returns a placeholder page;
  ``https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/602/A69`` redirects to the
  mirror, which serves the files.  Yearly, monthly and daily, with
  **asymmetric** error bars (``sigmaU``/``sigmaL``), 1739-2010.
- ``velasco_herrera_2024/velasco_herrera_2024.zip`` -- the Adv. Space Res.
  73, 2788 supplement ("Group Number series VERSION 3.14"), supplied by the PI
  on 2026-11-23 from his ScienceDirect access; the article is closed and the
  series is in no open repository, so the zip is the archival copy.  Read
  in place: nothing is extracted into ``data/raw/``.

Scale trap (D24): the supplement ships two families of file whose columns are
both labelled "GSN".  ``New_Reconstruction_GSN/`` is in **group counts**;
``Database_GSN_HOYT/`` is in **Hoyt & Schatten's 12.08-scaled units**.  Dividing
the latter by 12.08 reproduces this folder's ``hoyt_schatten_1998`` on all 386
overlapping years to 9e-4 (pinned in ``tests/test_published_series.py``), which
is what licenses ``gn`` to mean one thing in the output.

Usage
-----
    python scripts/fetch_published_gn_series.py [--check] [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gn_next_gen.paths import DATA_RAW, LEGACY  # noqa: E402

OUT_DIR = DATA_RAW / "gn_published_series_2026-11"
SOURCE_DIR = LEGACY / "GN_Series_matlab" / "data"

#: Hoyt & Schatten's group sunspot number is 12.08 x the mean daily group count.
#: Their 1998 series is carried here in group counts, so the 12.08-scaled files
#: in the Velasco Herrera supplement are divided by this to join that column.
HOYT_GSN_SCALE = 12.08

#: Output columns, in order.  ``gn_sd_hi``/``gn_sd_lo`` exist because
#: Chatzistergos et al. publish asymmetric error bars; the five legacy series
#: carry a symmetric ``gn_sd`` and the two 2026-11-23 additions carry neither.
COLUMNS = ["series", "year", "gn", "gn_sd", "gn_sd_hi", "gn_sd_lo", "n_observations"]

#: Series read from the read-only 2017 MATLAB snapshot.
LEGACY_SERIES = {
    "hoyt_schatten_1998": {
        "file": "GNhs_y.txt",
        "columns": ["year", "gn", "gn_sd", "n_observations", "aux"],
        "reference": "Hoyt & Schatten 1998, Sol. Phys. 179, 189",
        "note": "the original group sunspot number; reads blanket statements"
        " as daily observations (D7). Group counts, not 12.08-scaled (D24)",
    },
    "svalgaard_schatten_2016_backbone": {
        "file": "GNbb2_y.txt",
        "columns": ["year", "gn", "gn_sd"],
        "reference": "Svalgaard & Schatten 2016, Sol. Phys. 291, 2653",
        "note": "the backbone series; A157 finds linear scaling amplifies strong cycles",
    },
    "usoskin_series": {
        "file": "GNiu_y2.txt",
        "columns": ["year", "gn", "gn_sd", "n_observations", "aux"],
        "reference": "Usoskin et al., active-day-fraction family"
        " (exact version not recorded upstream)",
        "note": "ADF-family series as carried by the 2017 MATLAB snapshot;"
        " the version string was not recorded there, so it is NOT asserted"
        " to be Usoskin et al. 2021",
    },
    "vaquero": {
        "file": "Vaquero.csv",
        "columns": None,
        "reference": "Vaquero et al., as carried by the 2017 snapshot",
        "note": "sparse yearly reconstruction",
    },
    "zolotova": {
        "file": "Zolotova.csv",
        "columns": None,
        "reference": "Zolotova & Ponyavin, as carried by the 2017 snapshot",
        "note": "sparse yearly reconstruction",
    },
}

#: Series read from the upstream supplements frozen under this folder.
#: ``files`` lists every archived file the series came with, for the manifest and
#: the README; ``yearly`` names the one the output CSV is built from.
ARCHIVE_SERIES = {
    "chatzistergos_2017": {
        "reader": "chatzistergos",
        "root": "chatzistergos_2017",
        "yearly": "ch_gsn_y.dat",
        "files": ["ReadMe", "ch_gsn_y.dat", "ch_gsn_m.dat", "ch_gsn_d.dat.gz"],
        "reference": "Chatzistergos, Usoskin, Kovaltsov, Krivova & Solanki 2017,"
        " A&A 602, A69 (CDS J/A+A/602/A69)",
        "upstream": "http://vizier.nao.ac.jp/ftp/J/A+A/602/A69/",
        "acquired": "2026-11-23",
        "note": "non-parametric PDF-matrix daisy chain, 314 observers, 1739-2010."
        " ASYMMETRIC errors: sigmaU -> gn_sd_hi, sigmaL -> gn_sd_lo, and gn_sd is"
        " left null rather than averaging them. -99 is the missing-value sentinel,"
        " so 266 of 272 years carry a value",
    },
    "velasco_herrera_2024": {
        "reader": "velasco_tab",
        "root": "velasco_herrera_2024",
        "archive": "velasco_herrera_2024.zip",
        "yearly": "Group Number series VERSION 3.14/New_Reconstruction_GSN/"
        "New_annual_GSN_1610_2022.txt",
        "files": ["velasco_herrera_2024.zip"],
        "reference": "Velasco Herrera, Soon, Babynets, Murakozy, Tlatov, Nagovitsyn,"
        " Qiu, Svanda & Velasco Herrera 2024, Adv. Space Res. 73, 2788"
        " (doi:10.1016/j.asr.2023.12.011)",
        "upstream": "ScienceDirect supplement (closed access), PII S0273117723009651;"
        " supplied by the PI",
        "acquired": "2026-11-23",
        "note": "the V-ELSY/3.14 reconstruction, 1610-2022, in GROUP COUNTS."
        " No uncertainty is published, so gn_sd is null. 401 of 413 years carry a"
        " value; NaN marks the rest",
    },
    "velasco_herrera_2024_hoyt_update": {
        "reader": "velasco_tab",
        "root": "velasco_herrera_2024",
        "archive": "velasco_herrera_2024.zip",
        "yearly": "Group Number series VERSION 3.14/Database_GSN_HOYT/"
        "ANNUAL_GSN_HOYT_1610_2022.txt",
        "files": [],  # same archive as the row above
        "scale": HOYT_GSN_SCALE,
        "reference": "Velasco Herrera et al. 2024 (as above), updating Hoyt & Schatten 1998",
        "upstream": "ScienceDirect supplement (closed access), PII S0273117723009651;"
        " supplied by the PI",
        "acquired": "2026-11-23",
        "note": "H&S 1998 recomputed and extended to 2022. Published 12.08-scaled"
        " and DIVIDED BY 12.08 here so gn means group counts throughout (D24);"
        " it reproduces this folder's hoyt_schatten_1998 to 9e-4 over 1610-1995"
        " and carries it 27 years further. -99 is the missing-value sentinel",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_legacy_series(name: str, spec: dict) -> pd.DataFrame:
    """Read one series from the 2017 MATLAB snapshot.

    The yearly files use ``gn = -1`` with ``n_observations = 0`` for a year with
    no data at all: six such years in Hoyt & Schatten (1636, 1637, 1641, 1744,
    1745, 1747) and one in the Usoskin file (1811).  Earlier versions of this
    exporter carried the -1 through as if it were a group count, which put a
    negative spike into a comparison plot; it is mapped to null here.
    """
    path = SOURCE_DIR / spec["file"]
    if spec["columns"] is None:
        frame = pd.read_csv(path)
        frame = frame.rename(columns={frame.columns[0]: "year", frame.columns[1]: "gn"})
    else:
        frame = pd.read_csv(path, sep=r"\s+", header=None, names=spec["columns"], engine="python")
    if "n_observations" in frame:
        no_data = frame["n_observations"] == 0
        frame.loc[no_data, [c for c in ("gn", "gn_sd") if c in frame]] = np.nan
    frame.insert(0, "series", name)
    return frame


def read_chatzistergos(text: str) -> pd.DataFrame:
    """Yearly ``ch_gsn_y.dat``: fixed width, per the catalogue ReadMe."""
    import io

    frame = pd.read_fwf(
        io.StringIO(text),
        colspecs=[(0, 6), (7, 13), (14, 20), (21, 27)],
        header=None,
        names=["year", "gn", "gn_sd_hi", "gn_sd_lo"],
    )
    return frame.replace(-99.0, np.nan)


def read_velasco_tab(text: str) -> pd.DataFrame:
    """A V-ELSY/3.14 annual file: tab separated under a free-text preamble.

    The preamble has no comment marker, so the data is found by the first line
    whose first four characters are a year.  Trailing empty tab-separated fields
    are present in some files and dropped.
    """
    import io

    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line[:4].isdigit())
    frame = pd.read_csv(
        io.StringIO("\n".join(lines[start:])),
        sep="\t",
        header=None,
        usecols=[0, 1],
        names=["year", "gn"],
        na_values=["NaN"],
        engine="python",
    )
    return frame.replace(-99.0, np.nan)


READERS = {"chatzistergos": read_chatzistergos, "velasco_tab": read_velasco_tab}


def read_archive_series(name: str, spec: dict, root: Path) -> tuple[pd.DataFrame, list[dict], str]:
    """Read one archived series, and checksum every file it was archived with."""
    folder = root / spec["root"]
    if "archive" in spec:
        archive = folder / spec["archive"]
        if not archive.is_file():
            raise SystemExit(f"missing archive {archive}")
        with zipfile.ZipFile(archive) as zf:
            text = zf.read(spec["yearly"]).decode("utf-8", errors="replace")
        read_from = f"{spec['archive']}::{spec['yearly']}"
    else:
        path = folder / spec["yearly"]
        if not path.is_file():
            raise SystemExit(f"missing raw file {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        read_from = spec["yearly"]

    frame = READERS[spec["reader"]](text)
    if "scale" in spec:
        for column in ("gn", "gn_sd", "gn_sd_hi", "gn_sd_lo"):
            if column in frame:
                frame[column] = frame[column] / spec["scale"]
    frame.insert(0, "series", name)

    archived = [
        {"file": f, "sha256": sha256(folder / f), "bytes": (folder / f).stat().st_size}
        for f in spec["files"]
    ]
    return frame, archived, read_from


def build() -> tuple[pd.DataFrame, dict]:
    frames, provenance = [], {}

    for name, spec in LEGACY_SERIES.items():
        path = SOURCE_DIR / spec["file"]
        if not path.is_file():
            raise SystemExit(f"missing legacy source {path}")
        frames.append(read_legacy_series(name, spec))
        provenance[name] = {
            "source": str(path.relative_to(LEGACY.parent)),
            "sha256": sha256(path),
            "reference": spec["reference"],
            "note": spec["note"],
        }

    for name, spec in ARCHIVE_SERIES.items():
        frame, archived, read_from = read_archive_series(name, spec, OUT_DIR)
        frames.append(frame)
        provenance[name] = {
            "source": f"{spec['root']}/{read_from}",
            "archived_files": archived,
            "upstream": spec["upstream"],
            "acquired": spec["acquired"],
            "reference": spec["reference"],
            "note": spec["note"],
        }
        if "scale" in spec:
            provenance[name]["published_scale_divided_by"] = spec["scale"]

    table = pd.concat(frames, ignore_index=True)
    table = table.reindex(columns=COLUMNS)
    return table, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--check", action="store_true", help="verify without writing")
    args = parser.parse_args()

    table, provenance = build()
    target = args.out / "published_gn_series.csv"

    if args.check:
        if not target.is_file():
            raise SystemExit(f"{target} does not exist; run without --check first")
        # Compare the rendered text, not two parsed frames: a round trip through
        # read_csv changes dtypes for the all-null columns and would report a
        # difference where the bytes are identical.
        same = target.read_text() == table.to_csv(index=False, lineterminator="\n")
        print(f"{target.name}: {'MATCH' if same else 'DIFFERS'}")
        raise SystemExit(0 if same else 1)

    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(target, index=False, lineterminator="\n")

    (args.out / "provenance.json").write_text(
        json.dumps(
            {
                "exported_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "exporter": "scripts/fetch_published_gn_series.py",
                "role": "comparison_only",
                "units": "gn is the mean daily group COUNT for every series;"
                " 12.08-scaled sources are divided by 12.08 on the way in (D24)",
                "year_convention": "as published per series: mid-year for three,"
                " integer for Vaquero and both Velasco Herrera series,"
                " exact fractional dates for Zolotova",
                "series": provenance,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (args.out / "README.md").write_text(_readme(table, provenance))

    print(f"wrote {target} ({len(table)} rows, {table['series'].nunique()} series)")
    print(table.groupby("series")["year"].agg(["min", "max", "count"]).to_string())


def _readme(table: pd.DataFrame, provenance: dict) -> str:
    spans = table.groupby("series")["year"].agg(["min", "max", "count"])
    refs = {
        **{k: v["reference"] for k, v in LEGACY_SERIES.items()},
        **{k: v["reference"] for k, v in ARCHIVE_SERIES.items()},
    }
    legacy_rows = "\n".join(
        f"| `{name}` | {int(spans.loc[name, 'min'])}-{int(spans.loc[name, 'max'])} |"
        f" {int(spans.loc[name, 'count'])} | {refs[name]} |"
        for name in LEGACY_SERIES
    )
    archive_rows = "\n".join(
        f"| `{name}` | {int(spans.loc[name, 'min'])}-{int(spans.loc[name, 'max'])} |"
        f" {int(spans.loc[name, 'count'])} | {refs[name]} |"
        for name in ARCHIVE_SERIES
    )
    return f"""# Published group-number series (2026-11)

**Comparison only, and enforced in code.** The Step 2 scorer never reads this
folder, and ADR-003 forbids calibrating on a published series -- doing so would
close a loop through the same observers the model is meant to recalibrate. Since
2026-11-23 `gn_next_gen.config.validate_run_config` **refuses** a run that gives
a dataset of `kind: published_series` the role `calibration`, in *either*
variant, and every run's `lineage.json` carries a
`published_series_calibration_inputs` list and a `circularity_statement` as the
evidence. That rule is deliberately separate from the ADR-002 proxy firewall:
these are sunspot data, so listing them as proxy-derived would make the run's own
firewall statement false. Declare this folder as:

```yaml
datasets:
  - path: gn_published_series_2026-11
    role: comparison
    kind: published_series
```

`published_gn_series.csv` is yearly and long: one row per
`(series, year)`, with `gn` the **mean daily group count** for every series.
`provenance.json` carries each source file's SHA-256.

## From the read-only 2017 MATLAB snapshot

`legacy/GN_Series_matlab/data/`, **not** re-downloaded: those are the exact bytes
the legacy analyses used, which makes a comparison against the Step 2 legacy
baselines like for like.

| series | years | rows | reference |
|---|---|---|---|
{legacy_rows}

## From upstream supplements, frozen here (2026-11-23)

The two the Step 2 plan named and Step 2 could not get. Both are archived in
this folder and read in place; nothing is extracted into `data/raw/`.

| series | years | rows | reference |
|---|---|---|---|
{archive_rows}

- **`chatzistergos_2017/`** -- CDS catalogue J/A+A/602/A69 (yearly, monthly,
  daily, plus the byte-by-byte `ReadMe`), from the NAOJ VizieR mirror
  `http://vizier.nao.ac.jp/ftp/J/A+A/602/A69/`. CDS's own `/ftp/` path serves a
  placeholder page, which is what Step 2 recorded as unreachable; the
  `viz-bin/cat` endpoint redirects to the mirror. Errors are **asymmetric**, so
  they land in `gn_sd_hi`/`gn_sd_lo` and `gn_sd` stays null.
- **`velasco_herrera_2024/velasco_herrera_2024.zip`** -- the Adv. Space Res.
  supplement, "Group Number series VERSION 3.14", supplied by the PI from his
  ScienceDirect access. The article is closed and the series is in no open
  repository, so this zip is the archival copy. It also contains monthly and
  daily files, a corrected raw observation database to 2022, the v1.12 and v1.21
  Vaquero databases and a list of excluded values -- see the next section.

## The 12.08 scale trap (D24)

The Velasco Herrera supplement ships two families of file whose data column is
labelled `GSN` in both:

| folder in the zip | units | at 1610 |
|---|---|---|
| `New_Reconstruction_GSN/` | mean daily group **count** | 3.27 |
| `Database_GSN_HOYT/` | Hoyt & Schatten's **12.08 x count** | 72.0 |

The exporter divides the second by 12.08, which reproduces this folder's
`hoyt_schatten_1998` on all **386** overlapping years to within **9e-4** and
extends it from 1995 to 2022. `tests/test_published_series.py` pins that
agreement, so a future change of scale convention fails loudly rather than
silently rescaling a series by a factor of twelve.

## Other things in the Velasco Herrera zip, deliberately not used

`New_Raw_Database_GSN/corrected_raw_GSN_database_1610_2022_Elsy.txt` is a
competing revision of the raw observation database, 1610-2022, in the same shape
as `gn_database_v1.22/` (year, month, day, fractional year, station, observer
flag, count). It is **not** the project's database of record -- ADR-010 makes
that `gn_database_v1.3/` release `_4` -- and it is a *corrected* database, so its
corrections are an author's judgement to be compared against v1.3's changelog,
not adopted. It is interesting for one reason worth a later look: it runs to
**2022**, where v1.3 stops at 2010.

## Year convention

`year` is **as published**: mid-year (`1610.5`) for Hoyt & Schatten, the
backbone, Usoskin and Chatzistergos; integer for Vaquero and both Velasco
Herrera series; exact fractional dates for Zolotova. Floor it before joining
series.

## A caveat on the Usoskin series

The snapshot's `GNiu_y2.txt` carries no version string, so it is recorded as
"Usoskin et al., active-day-fraction family" and **not** asserted to be Usoskin
et al. 2021. Do not cite it as a specific publication without checking.
"""


if __name__ == "__main__":
    main()
