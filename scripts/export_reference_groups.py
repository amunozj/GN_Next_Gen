"""Export the frozen reference-survey snapshot from sunspot-homogenization.

Reads the per-survey group CSVs that upstream's ``run_survey`` writes and lands a
checksummed, long-format snapshot under ``data/raw/``.  Three tables per the
plan, concatenated across surveys with a ``survey`` column:

* ``group_day_observations`` -- one row per (survey, group, day): THE reference.
  Taken from upstream's ``12_final_all`` stage, which is the full multi-observation
  catalogue.  Upstream's ``05_daily`` stage is deliberately NOT used: its
  ``area_total`` is each group's *peak* area deposited on that group's single
  canonical max-area day, which is an emergence-weighted rate, not instantaneous
  disk area.
* ``group_single_entry`` -- one row per (survey, group), carrying ``obs_duration``
  (group lifetimes, which the old day-by-area matrices could not provide).
* ``day_status`` -- one row per (survey, day): the explicit observed /
  spotless / no-observation status that replaces the old ``AREA1`` NaN/0 convention.

Two renames happen here because ``obs_type`` means two different things upstream
(known issue D10):

* in ``final_single`` it is the group's observation kind -> ``group_obs_kind``
  (0 placeholder row, 1 single-observation group, 2 multi-observation group);
* in ``flags`` it is the day's observation status -> ``day_status``
  (0 no observation, 1 observed spotless, 2 observed active).

Also derives ``survey_cycle_coverage.csv``, which is what ADR-009's anchor
definition is checked against: a cycle counts as complete in a survey only if that
survey also carries its neighbours, which is what catches a cycle clipped at the
edge of a survey's coverage.

Usage
-----
    python scripts/export_reference_groups.py [--check] [--allow-commit-mismatch]
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

from gn_next_gen.paths import CONFIGS, REPO_ROOT  # noqa: E402

GROUP_DAY_COLUMNS = [
    "survey",
    "group_id",
    "group_id_orig",
    "date_ord",
    "date_frac",
    "year",
    "month",
    "day",
    "area_obs",
    "area_obs_unc",
    "area_corr",
    "area_corr_unc",
    "area_umbra_obs",
    "area_umbra_obs_unc",
    "area_umbra_corr",
    "area_umbra_corr_unc",
    "area_corr_dbcal",
    "area_corr_dbcal_unc",
    "area_corr_dbcal_unc_corr",
    "area_corr_dbcal_unc_indep",
    "area_corr_dbcal_above_thr",
    "cycle",
    "cycle_parity",
    "amp_whole_sm91",
    "amp_hemi_sm91",
    "amp_whole_dbcal_sm91",
    "amp_hemi_dbcal_sm91",
    # QA / provenance metadata only.  ADR-004 bars positions as model inputs.
    "hemisphere",
    "dist_center",
    "latitude",
    "longitude",
]
SINGLE_EXTRA = ["group_obs_kind", "obs_duration"]
DAY_STATUS_COLUMNS = [
    "survey",
    "date_ord",
    "date_frac",
    "year",
    "month",
    "day",
    "day_status",
    "n_groups",
    "area_total",
    "area_north",
    "area_south",
]

LINEAGE = (
    "Sunspot-survey observations only (Schwabe, Spoerer, RGO/Greenwich, Debrecen DPD, "
    "Kislovodsk KMAS), processed by sunspot-homogenization at the pinned commit. "
    "No isotope, aurora, proxy or SILSO sunspot-number input anywhere in this lineage "
    "(ADR-002 firewall). The cycle/cycle_parity and amp_* columns are derived from the "
    "surveys' own group positions and smoothed areas, not from SILSO, and are diagnostic "
    "only: they are never calibration inputs, so cycle phase stays latent (ADR-003)."
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_parquet(df: pd.DataFrame, path: Path, cfg: dict) -> None:
    """Write deterministically: no pandas metadata, fixed compression."""
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


def load_survey(group_dir: Path, survey: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    read = dict(dtype={"group_id_orig": "string"}, low_memory=False)
    fa = pd.read_csv(_find(group_dir, survey, "12_final_all"), **read)
    fs = pd.read_csv(_find(group_dir, survey, "13_final_single"), **read)
    fl = pd.read_csv(_find(group_dir, survey, "04_flags"), low_memory=False)

    # D10: the two incompatible meanings of obs_type, separated at the boundary.
    fs = fs.rename(columns={"obs_type": "group_obs_kind"})
    fl = fl.rename(columns={"obs_type": "day_status"})
    if "obs_type" in fa.columns:
        raise SystemExit(f"{survey}: unexpected obs_type in final_all")
    fl["survey"] = survey

    # Placeholder calendar rows (no group) belong in day_status, not the group tables.
    fa = fa[fa["group_id"].notna()].copy()
    fs = fs[fs["group_id"].notna()].copy()

    missing = [c for c in GROUP_DAY_COLUMNS if c not in fa.columns]
    if missing:
        raise SystemExit(f"{survey}: final_all missing {missing}")
    fa = fa[GROUP_DAY_COLUMNS]
    fs = fs[[c for c in GROUP_DAY_COLUMNS + SINGLE_EXTRA if c in fs.columns]]
    fl = fl[DAY_STATUS_COLUMNS]
    return fa, fs, fl


def cycle_coverage(group_day: pd.DataFrame) -> pd.DataFrame:
    """Per (survey, cycle) extent and the completeness flag used by ADR-009.

    A cycle is complete in a survey only if the survey also carries its
    neighbours; a cycle at the edge of a survey's coverage is clipped even when
    its own groups look contiguous.
    """
    g = group_day.dropna(subset=["cycle"])
    agg = (
        g.groupby(["survey", "cycle"])
        .agg(
            n_groups=("group_id", "nunique"),
            n_observations=("group_id", "size"),
            first_ord=("date_ord", "min"),
            last_ord=("date_ord", "max"),
        )
        .reset_index()
    )
    agg["cycle"] = agg["cycle"].astype(int)
    present = {(s, c) for s, c in zip(agg["survey"], agg["cycle"], strict=True)}
    agg["complete"] = [
        ((s, c - 1) in present) and ((s, c + 1) in present)
        for s, c in zip(agg["survey"], agg["cycle"], strict=True)
    ]
    agg["first_day"] = [str(dt.date.fromordinal(int(o))) for o in agg["first_ord"]]
    agg["last_day"] = [str(dt.date.fromordinal(int(o))) for o in agg["last_ord"]]
    return agg[
        [
            "survey",
            "cycle",
            "n_groups",
            "n_observations",
            "first_day",
            "last_day",
            "first_ord",
            "last_ord",
            "complete",
        ]
    ].sort_values(["survey", "cycle"], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=CONFIGS / "reference_groups_export.yaml")
    ap.add_argument("--check", action="store_true", help="compare instead of overwrite")
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
    pinned = cfg["upstream"]["commit"]
    if head != pinned:
        msg = f"upstream HEAD {head} != pinned {pinned}"
        if not args.allow_commit_mismatch:
            raise SystemExit(msg + " (pass --allow-commit-mismatch to override)")
        print("WARNING: " + msg)

    pkl = up / cfg["upstream"]["posterior"]
    pkl_sha = sha256(pkl)
    if pkl_sha != cfg["upstream"]["posterior_sha256"]:
        raise SystemExit(f"posterior checksum changed: {pkl_sha}")

    group_dir = Path(cfg["upstream"]["group_output_dir"])
    surveys = cfg["surveys"]
    fas, fss, fls = [], [], []
    for s in surveys:
        fa, fs, fl = load_survey(group_dir, s)
        print(f"  {s}: {len(fa):,} group-days, {len(fs):,} groups, {len(fl):,} calendar days")
        fas.append(fa)
        fss.append(fs)
        fls.append(fl)

    group_day = pd.concat(fas, ignore_index=True)
    single = pd.concat(fss, ignore_index=True)
    day_status = pd.concat(fls, ignore_index=True)
    coverage = cycle_coverage(group_day)

    offsets = pd.read_csv(up / cfg["upstream"]["offsets_csv"])
    calib = offsets[offsets["survey"].isin(surveys)].reset_index(drop=True)

    out = REPO_ROOT / cfg["output"]["dir"]
    out.mkdir(parents=True, exist_ok=True)

    def emit(target: Path) -> dict[str, str]:
        write_parquet(group_day, target / "group_day_observations.parquet", cfg)
        write_parquet(single, target / "group_single_entry.parquet", cfg)
        write_parquet(day_status, target / "day_status.parquet", cfg)
        calib.to_csv(target / "survey_calibration.csv", index=False, lineterminator="\n")
        coverage.to_csv(target / "survey_cycle_coverage.csv", index=False, lineterminator="\n")
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

    def _span(survey: str) -> list[str]:
        ords = day_status.loc[day_status["survey"] == survey, "date_ord"]
        return [
            str(dt.date.fromordinal(int(ords.min()))),
            str(dt.date.fromordinal(int(ords.max()))),
        ]

    spans = {s: _span(s) for s in surveys}
    provenance = {
        "generated_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "scripts/export_reference_groups.py",
        "upstream": {
            **{k: v for k, v in cfg["upstream"].items() if k != "group_output_dir"},
            "head_at_export": head,
            "license": "MIT",
        },
        "surveys": surveys,
        "excluded_surveys": {
            "soon": "excluded from the reference on area-scale grounds (D13)",
            "swpc": "excluded from the reference; contributes observer counts only (D13)",
        },
        "anchors": cfg["anchors"],
        "coverage_by_survey": spans,
        "row_counts": {
            "group_day_observations": int(len(group_day)),
            "group_single_entry": int(len(single)),
            "day_status": int(len(day_status)),
        },
        "model_facing_area_column": "area_corr",
        "calibrated_area_column": "area_corr_dbcal",
        "calibrated_area_use": (
            "prior and cross-check only; never applied as a hard pre-correction, so the "
            "upstream calibration is not applied twice (ADR-003, D5)"
        ),
        "data_lineage": LINEAGE,
        "sha256": digests,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    print("\nwrote:")
    for name, digest in digests.items():
        print(f"  {name:38s} {(out / name).stat().st_size:>12,} bytes  {digest[:12]}")
    print(f"\ncoverage: {spans}")
    print("\ncycle completeness:")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
