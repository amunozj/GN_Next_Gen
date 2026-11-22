"""Freeze the published GN series this project compares against.

Moved into Step 2 by the PI on 2026-11-21: the benchmark is where they are
actually consumed.  They are **comparison only** -- the firewall validator
refuses them as a calibration input (ADR-002, ADR-003) and the scorer never
reads them; they exist so a reader can see the benchmark's baselines beside the
real series the same methods produced.

Three are already in the repository, inside the read-only 2017 MATLAB snapshot,
and are exported here into a dated raw folder with their provenance and
checksums rather than re-downloaded: ``legacy/GN_Series_matlab/data/`` holds
Hoyt & Schatten 1998, the Svalgaard & Schatten backbone and a Usoskin series,
plus Vaquero's and Zolotova's reconstructions.  Exporting from the snapshot is
better than fetching them again -- these are the exact bytes the legacy analyses
used, so a comparison against the legacy baselines is like for like.

Two of the five the plan named are **not** acquired: Chatzistergos et al. (2017)
and Velasco Herrera et al. (2024).  Their supplements live at CDS/VizieR, which
is unreachable from this machine (connection timeout), and no alternative source
was found.  Recorded in ``docs/04_data/refresh_todo.md`` rather than quietly
dropped; the benchmark does not depend on them.

Usage
-----
    python scripts/fetch_published_gn_series.py [--check] [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gn_next_gen.paths import DATA_RAW, LEGACY  # noqa: E402

OUT_DIR = DATA_RAW / "gn_published_series_2026-11"
SOURCE_DIR = LEGACY / "GN_Series_matlab" / "data"

#: Each series: source file, its columns, and what it is.
SERIES = {
    "hoyt_schatten_1998": {
        "file": "GNhs_y.txt",
        "columns": ["year", "gn", "gn_sd", "n_observations", "aux"],
        "reference": "Hoyt & Schatten 1998, Sol. Phys. 179, 189",
        "note": "the original group sunspot number; reads blanket statements"
        " as daily observations (D7)",
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

#: Named in the plan but not acquired, with the reason.
UNAVAILABLE = {
    "chatzistergos_2017": "A&A 602, A69 supplement at CDS/VizieR -- unreachable from this machine",
    "velasco_herrera_2024": "Adv. Space Res. 73, 2788 supplement -- no open source found",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_series(name: str, spec: dict) -> pd.DataFrame:
    path = SOURCE_DIR / spec["file"]
    if spec["columns"] is None:
        frame = pd.read_csv(path)
        frame = frame.rename(columns={frame.columns[0]: "year", frame.columns[1]: "gn"})
    else:
        frame = pd.read_csv(path, sep=r"\s+", header=None, names=spec["columns"], engine="python")
    frame.insert(0, "series", name)
    keep = ["series", "year", "gn"] + [c for c in ("gn_sd", "n_observations") if c in frame]
    return frame[keep]


def build() -> tuple[pd.DataFrame, dict]:
    frames, provenance = [], {}
    for name, spec in SERIES.items():
        path = SOURCE_DIR / spec["file"]
        if not path.is_file():
            raise SystemExit(f"missing legacy source {path}")
        frames.append(read_series(name, spec))
        provenance[name] = {
            "source": str(path.relative_to(LEGACY.parent)),
            "sha256": sha256(path),
            "reference": spec["reference"],
            "note": spec["note"],
        }
    return pd.concat(frames, ignore_index=True), provenance


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
        existing = pd.read_csv(target)
        same = existing.equals(table)
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
                "series": provenance,
                "not_acquired": UNAVAILABLE,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (args.out / "README.md").write_text(_readme(table))

    print(f"wrote {target} ({len(table)} rows, {table['series'].nunique()} series)")
    print(table.groupby("series")["year"].agg(["min", "max", "count"]).to_string())
    print("\nNot acquired:")
    for name, reason in UNAVAILABLE.items():
        print(f"  {name}: {reason}")


def _readme(table: pd.DataFrame) -> str:
    spans = table.groupby("series")["year"].agg(["min", "max", "count"])
    rows = "\n".join(
        f"| `{name}` | {int(row['min'])}-{int(row['max'])} | {int(row['count'])} | "
        f"{SERIES[name]['reference']} |"
        for name, row in spans.iterrows()
    )
    missing = "\n".join(f"- **{name}** -- {reason}" for name, reason in UNAVAILABLE.items())
    return f"""# Published group-number series (2026-11)

**Comparison only.** The firewall validator refuses these as a calibration input
(ADR-002, ADR-003) and the Step 2 scorer never reads them. They are here so the
benchmark's baseline scores can be read beside the real series the same methods
produced.

Exported by `scripts/fetch_published_gn_series.py` from the read-only 2017
MATLAB snapshot in `legacy/GN_Series_matlab/data/`, **not** re-downloaded: those
are the exact bytes the legacy analyses used, which makes a comparison against
the legacy baselines like for like. `provenance.json` carries each source file's
SHA-256.

| series | years | rows | reference |
|---|---|---|---|
{rows}

## Not acquired

{missing}

Both are recorded in `docs/04_data/refresh_todo.md`. The benchmark does not
depend on them.

## A caveat on the Usoskin series

The snapshot's `GNiu_y2.txt` carries no version string, so it is recorded as
"Usoskin et al., active-day-fraction family" and **not** asserted to be Usoskin
et al. 2021. Do not cite it as a specific publication without checking.
"""


if __name__ == "__main__":
    main()
