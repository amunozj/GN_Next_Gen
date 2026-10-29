"""Export daily group counts from KMAS and SWPC as an observation-record extension.

The v1.3 revised collection ends 2010-12-31. KMAS and SWPC both run to 2026 through
the sunspot-homogenization pipeline, so both are added here as modern observers,
over their full span with post-2010 rows flagged -- the pre-2010 overlap (KMAS from
1954, SWPC from 1996) is what lets each be calibrated against the existing observer
network rather than grafted on unanchored at the end.

Counts, not areas.  A count needs an acuity threshold, so the table is long over
thresholds rather than committing to one: the Step 1 sweep grid plus each survey's
own anchor threshold from the joint posterior.  Counts are emitted on both area
scales, because the thresholds are defined on the common Debrecen scale while native
areas are the model-facing input (ADR-009).

SWPC areas never land in data/raw: this script reads its group catalogue to derive
counts and writes only counts, so the decision to keep SWPC out of the reference
(known issue D13) cannot be reached around by accident.

THE KMAS SINGLE-USE RULE.  KMAS is also a reference survey in
reference_groups_2026-10/, and its counts are derivable from its areas, so using
both puts the same observations into the likelihood twice.  Any run must declare
KMAS's role as reference areas OR observer counts, never both.  SWPC carries no
such constraint.  The ``role`` column and the folder README record this; enforcing
it in the run configuration is a Step 1 follow-up.

Usage
-----
    python scripts/export_observer_extension.py [--check] [--allow-commit-mismatch]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gn_next_gen.paths import CONFIGS, GN_OBSERVER_EXTENSION  # noqa: E402

# v1.3 uses station codes 1-775; these are upstream's own survey tags, so they
# cannot collide and are self-documenting.
STATIONS = {
    "kmas": dict(
        station=1006,
        name="KISLOVODSK MOUNTAIN ASTRONOMICAL STATION (KMAS), PULKOVO",
        role="reference_or_observer_choose_one",
        note="Also a reference survey in reference_groups_2026-10/. Single-use rule: a run "
        "may use KMAS as reference areas OR as observer counts, never both.",
    ),
    "swpc": dict(
        station=1008,
        name="NOAA/SWPC SOLAR REGION SUMMARY",
        role="observer_counts_only",
        note="Not in the reference (D13: its areas sit ~43% below Debrecen). That is an "
        "area-scale issue and does not affect counts, so its counts are unconstrained.",
    ),
}

# The Step 1 acuity sweep; each survey's own anchor threshold is added to this.
BASE_THRESHOLDS = (0.0, 10.0, 30.0, 60.0, 100.0)

V13_LAST_DAY = dt.date(2010, 12, 31).toordinal()

LINEAGE = (
    "Sunspot-survey observations only (Kislovodsk KMAS and NOAA/SWPC Solar Region Summary), "
    "processed by sunspot-homogenization at the pinned commit and reduced here to daily group "
    "counts above a set of acuity thresholds. No isotope, aurora, proxy or SILSO sunspot-number "
    "input anywhere in this lineage (ADR-002 firewall)."
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_parquet(df: pd.DataFrame, path: Path, cfg: dict) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata(None)
    pq.write_table(
        table,
        path,
        compression=cfg["output"]["compression"],
        compression_level=cfg["output"]["compression_level"],
    )


def _find(group_dir: Path, survey: str, stage: str) -> Path:
    hits = sorted(group_dir.glob(f"{survey}/{survey}_{stage}_*.csv"))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one {survey} {stage} CSV, found {hits}")
    return hits[0]


def survey_counts(group_dir: Path, survey: str, thresholds: tuple[float, ...]) -> pd.DataFrame:
    """Daily group counts above each threshold, on both area scales."""
    groups = pd.read_csv(
        _find(group_dir, survey, "12_final_all"),
        usecols=["date_ord", "group_id", "area_corr", "area_corr_dbcal"],
        low_memory=False,
    )
    groups = groups[groups["group_id"].notna()]

    flags = pd.read_csv(_find(group_dir, survey, "04_flags"), low_memory=False)
    flags = flags.rename(columns={"obs_type": "day_status"})
    days = flags[["date_ord", "date_frac", "year", "month", "day", "day_status", "n_groups"]]

    meta = STATIONS[survey]
    frames = []
    for thr in thresholds:
        native = (
            groups.loc[groups["area_corr"] > thr].groupby("date_ord").size().rename("groups_native")
        )
        dbcal = (
            groups.loc[groups["area_corr_dbcal"] > thr]
            .groupby("date_ord")
            .size()
            .rename("groups_dbcal")
        )
        d = days.join(native, on="date_ord").join(dbcal, on="date_ord")
        # A day the survey observed has a real count, zero included; a day it did
        # not observe has none.  day_status: 0 no observation, 1 spotless, 2 active.
        observed = d["day_status"] > 0
        for col in ("groups_native", "groups_dbcal"):
            d[col] = d[col].fillna(0).where(observed, other=pd.NA).astype("Int64")
        # v1.3's own sentinel, so the table concatenates onto it without a join rule.
        d["groups_v13_convention"] = d["groups_native"].fillna(-1).astype("int64")
        d["threshold_msh"] = thr
        d["survey"] = survey
        d["station"] = meta["station"]
        d["role"] = meta["role"]
        d["extends_v13"] = d["date_ord"] > V13_LAST_DAY
        frames.append(d)

    out = pd.concat(frames, ignore_index=True)
    return out[
        [
            "survey",
            "station",
            "date_ord",
            "date_frac",
            "year",
            "month",
            "day",
            "threshold_msh",
            "groups_native",
            "groups_dbcal",
            "groups_v13_convention",
            "day_status",
            "n_groups",
            "extends_v13",
            "role",
        ]
    ].sort_values(["survey", "threshold_msh", "date_ord"], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=CONFIGS / "reference_groups_export.yaml")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--allow-commit-mismatch", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    up = Path(cfg["upstream"]["repo"])
    head = subprocess.run(
        ["git", "-C", str(up), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != cfg["upstream"]["commit"]:
        msg = f"upstream HEAD {head} != pinned {cfg['upstream']['commit']}"
        if not args.allow_commit_mismatch:
            raise SystemExit(msg + " (pass --allow-commit-mismatch to override)")
        print("WARNING: " + msg)

    # Each survey's own anchor threshold, from the joint posterior.
    offsets = pd.read_csv(up / cfg["upstream"]["offsets_csv"]).set_index("survey")
    group_dir = Path(cfg["upstream"]["group_output_dir"])

    frames, anchors = [], {}
    for survey in STATIONS:
        anchor = round(float(offsets.loc[survey, "threshold_MSH"]), 2)
        anchors[survey] = anchor
        thresholds = tuple(sorted(set(BASE_THRESHOLDS) | {anchor}))
        df = survey_counts(group_dir, survey, thresholds)
        obs_days = int((df["day_status"] > 0).sum() / len(thresholds))
        print(
            f"  {survey}: {len(df):,} rows, {len(thresholds)} thresholds "
            f"(anchor {anchor} MSH), {obs_days:,} observed days"
        )
        frames.append(df)

    counts = pd.concat(frames, ignore_index=True)
    stations = pd.DataFrame(
        [
            dict(
                survey=s,
                station=m["station"],
                name=m["name"],
                role=m["role"],
                anchor_threshold_msh=anchors[s],
                first_day=str(
                    dt.date.fromordinal(int(counts.loc[counts["survey"] == s, "date_ord"].min()))
                ),
                last_day=str(
                    dt.date.fromordinal(int(counts.loc[counts["survey"] == s, "date_ord"].max()))
                ),
                note=m["note"],
            )
            for s, m in STATIONS.items()
        ]
    )

    # SWPC areas must not leak: assert no area column survived.
    leaked = [c for c in counts.columns if "area" in c.lower()]
    if leaked:
        raise SystemExit(f"area columns must not reach the extension: {leaked}")

    out = GN_OBSERVER_EXTENSION
    out.mkdir(parents=True, exist_ok=True)

    def emit(target: Path) -> dict[str, str]:
        write_parquet(counts, target / "observer_counts.parquet", cfg)
        stations.to_csv(target / "station_assignment.csv", index=False, lineterminator="\n")
        return {
            p.name: sha256(p)
            for p in sorted(target.iterdir())
            if p.name not in ("provenance.json", "README.md")
        }

    if args.check:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            new = emit(Path(td))
        ok = True
        for name, digest in new.items():
            old = sha256(out / name)
            print(f"  {name}: {'MATCH' if old == digest else 'DIFFERS'}")
            ok &= old == digest
        raise SystemExit(0 if ok else 1)

    digests = emit(out)

    provenance = {
        "generated_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/export_observer_extension.py",
        "purpose": (
            "Extend the observation record past v1.3's 2010-12-31 end with daily group counts "
            "from KMAS and SWPC, over their full span so the pre-2010 overlap calibrates them "
            "against the existing observer network."
        ),
        "upstream": {
            **{k: v for k, v in cfg["upstream"].items() if k != "group_output_dir"},
            "head_at_export": head,
            "license": "MIT",
        },
        "stations": stations.drop(columns=["note"]).to_dict("records"),
        "thresholds_msh": {s: sorted(set(BASE_THRESHOLDS) | {a}) for s, a in anchors.items()},
        "threshold_convention": (
            "a group counts when its area is strictly greater than the threshold"
        ),
        "single_use_rule": (
            "KMAS is also a reference survey in reference_groups_2026-10/. Its counts are "
            "derivable from its areas, so any run must declare KMAS's role as reference areas "
            "OR observer counts, never both. SWPC has no such constraint."
        ),
        "row_counts": {"observer_counts": int(len(counts))},
        "contains_areas": False,
        "data_lineage": LINEAGE,
        "sha256": digests,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    print("\nwrote:")
    for name, digest in digests.items():
        print(f"  {name:32s} {(out / name).stat().st_size:>10,} bytes  {digest[:12]}")
    print("\nstations:")
    print(stations.drop(columns=["note"]).to_string(index=False))
    post = counts[counts["extends_v13"]]
    print(
        f"\nrows extending past 2010-12-31: {len(post):,} "
        f"({post['date_ord'].min()} -> {post['date_ord'].max()})"
    )


if __name__ == "__main__":
    main()
