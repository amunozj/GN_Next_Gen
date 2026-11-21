"""Refresh the SILSO sunspot-number snapshots into a new dated raw folder.

SILSO SN V2.0 is **comparison only** and may never enter calibration: it is itself a
composite, so using it would make the new series partly a re-derivation of the old
one (ADR-002, ADR-003).  Cycle phase stays latent in the model.  It is refreshed
anyway because the committed snapshots end 2018-07 and 2017-10 (issue **D6**) and
every comparison figure in the paper will want the present day.

Bytes are saved exactly as served -- semicolon-separated, no header, which is
SILSO's own format and *not* the comma-separated converted form of the 2018
snapshot.  ``gn_next_gen.io.silso`` reads both.  A new dated folder, never an
overwrite (ADR-006).

Usage
-----
    python scripts/fetch_silso.py [--out data/raw/silso_2026-11] [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gn_next_gen.paths import DATA_RAW  # noqa: E402

SOURCES = {
    "SN_d_tot_V2.0.csv": "https://www.sidc.be/SILSO/INFO/sndtotcsv.php",
    "SN_m_tot_V2.0.csv": "https://www.sidc.be/SILSO/INFO/snmtotcsv.php",
}

README = """# `{folder}/` — SILSO sunspot number V2.0, refreshed {date}

International Sunspot Number V2.0 from WDC-SILSO, Royal Observatory of Belgium
(https://www.sidc.be/SILSO/). Supersedes the 2018-era snapshot in `../silso/`, which is retained.

## ⚠ Comparison only

**SN V2.0 may not enter calibration of the new model** (ADR-002, ADR-003). It is a composite, so
calibrating against it would make this series partly a re-derivation of the one it is meant to be
compared with. Cycle phase is latent in the model. `gn_next_gen.io.silso` tags what it loads
`comparison_only`, and `gn_next_gen.config.validate_run_config` refuses any sunspot-only run that
lists SILSO as a calibration input.

## Files

| File | Grain | Coverage |
|---|---|---|
| `SN_d_tot_V2.0.csv` | one row per day | {daily_coverage} |
| `SN_m_tot_V2.0.csv` | one row per month | {monthly_coverage} |

**Format differs from `../silso/`.** These are SILSO's own downloads: **semicolon-separated, no
header row**. The 2018 snapshot was converted to comma-separated with a header during the ADFSSN
era. Both are byte-exact as received, and the loader accepts either.

Daily columns: year; month; day; fractional year; daily total SN (**-1 = missing**); standard
deviation; number of observations; definitive/provisional marker (1/0).

Monthly columns: year; month; fractional year; monthly mean SN (**-1 = missing**); standard
deviation; number of observations; definitive/provisional marker.

## Citation

WDC-SILSO, Royal Observatory of Belgium, Brussels. Distributed under CC BY-NC 4.0.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def coverage(path: Path, monthly: bool) -> tuple[str, str]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]

    def stamp(line: str) -> str:
        parts = [p for p in line.replace(";", " ").split() if p]
        return f"{int(parts[0]):04d}-{int(parts[1]):02d}" + (
            "" if monthly else f"-{int(parts[2]):02d}"
        )

    return stamp(lines[0]), stamp(lines[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DATA_RAW / f"silso_{dt.date.today():%Y-%m}")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows, spans = [], {}
    for name, url in SOURCES.items():
        target = args.out / name
        print(f"fetching {url}")
        if not args.dry_run:
            with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
                target.write_bytes(response.read())
        start, end = coverage(target, monthly=name.startswith("SN_m"))
        spans[name] = f"{start} → {end}"
        lines = len(target.read_text().splitlines())
        print(f"  {name}: {target.stat().st_size:,} bytes, {lines:,} rows, {start} → {end}")
        rows.append(
            {
                "path": f"{args.out.name}/{name}",
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
                "lines": lines,
                "eol": "LF" if b"\r\n" not in target.read_bytes()[:4096] else "CRLF",
                "header": "none (SILSO ships headerless, semicolon-separated)",
                "coverage_start": start,
                "coverage_end": end,
                "origin": f"Downloaded from {url} on {dt.date.today():%Y-%m-%d}",
                "upstream": "WDC-SILSO, Royal Observatory of Belgium; Sunspot Number "
                "V2.0; CC BY-NC 4.0",
                "notes": "COMPARISON ONLY - never a calibration input (ADR-002, ADR-003). "
                "-1 marks a missing value. Supersedes ../silso/, which is retained; "
                "that older snapshot is comma-separated with a header (D6)",
            }
        )

    readme = args.out / "README.md"
    readme.write_text(
        README.format(
            folder=args.out.name,
            date=f"{dt.date.today():%Y-%m}",
            daily_coverage=spans["SN_d_tot_V2.0.csv"],
            monthly_coverage=spans["SN_m_tot_V2.0.csv"],
        )
    )
    rows.append(
        {
            "path": f"{args.out.name}/README.md",
            "bytes": readme.stat().st_size,
            "sha256": sha256(readme),
            "lines": len(readme.read_text().splitlines()),
            "eol": "LF",
            "header": "n/a (markdown)",
            "coverage_start": "n/a",
            "coverage_end": "n/a",
            "origin": "Written by scripts/fetch_silso.py",
            "upstream": "n/a",
            "notes": "Format, the comparison-only rule, and how this differs from ../silso/",
        }
    )

    import csv

    manifest = DATA_RAW / "MANIFEST.csv"
    with manifest.open(newline="") as handle:
        existing = list(csv.DictReader(handle))
        fields = list(existing[0])
    known = {row["path"] for row in existing}
    added = [row for row in rows if row["path"] not in known]
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing + added)
    print(f"\nwrote {args.out} and added {len(added)} MANIFEST rows")


if __name__ == "__main__":
    main()
