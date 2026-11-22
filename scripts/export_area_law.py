"""Freeze the upstream group-area law calibration into ``data/raw/``.

The area distribution of sunspot groups is a **Weibull + log-normal** mixture on
log-area: a Weibull for the small objects (a fragmentation regime) and a log-normal
for the large ones (the multiplicative growth of bipolar active regions).  Fitting
*two log-normals* instead folds observational effects into the small-object
component and is the form upstream's own paper explicitly rejects
(``paper/ms.tex`` l. 207; Nagovitsyn & Pevtsov 2021 is named as the alternative).

``sunspot-homogenization`` has already fitted this law per survey by nested
sampling, hard-truncated at each survey's own completeness threshold, and that
calibration is mature: **this project does not refit it**.  Its posteriors live in
a git-ignored tree in that repository, so nothing here may read them at run time.
This script copies the summaries into a small, checksummed snapshot under
``data/raw/`` (a new folder, overwriting nothing -- ADR-006) and refuses to run if
the upstream repository is not at the commit the reference snapshot was exported
from.

The fit is **one area per group** -- upstream's ``n_kept`` reproduces exactly from
``group_single_entry.parquet`` with ``area_corr`` strictly above the per-survey
``threshold_MSH`` already frozen in ``reference_groups_2026-10`` (RGO 16,880;
Debrecen 8,306; KMAS 15,662).

Usage
-----
    python scripts/export_area_law.py [--check] [--allow-commit-mismatch]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gn_next_gen.paths import AREA_LAW, DATA_RAW, REFERENCE_GROUPS  # noqa: E402

UPSTREAM = Path("/d0/amunozj/git_repos/sunspot-homogenization")
SOLO_WLN = UPSTREAM / "data/processed/calibration/solo_wln"

#: Only the surveys this project's reference carries.  SOON is excluded entirely
#: (ADR-009) and SWPC contributes counts, not areas (D13), but both are exported
#: so the snapshot documents the full upstream fit rather than a filtered view.
SURVEYS = (
    "rgo_total",
    "debrecen_total",
    "kmas_total",
    "sporer_total",
    "soon_total",
    "swpc_total",
    "rgo_umbra",
    "debrecen_umbra",
)

PARAMS = ("k_W", "log_lambda_W", "mu_LN", "sigma_LN", "pi_W", "sigma_floor")

META_KEYS = (
    "condition",
    "area_column",
    "filter_column",
    "trunc_factor",
    "threshold_source",
    "plateau_tol",
    "y_trunc_R",
    "threshold_corr_R",
    "n_kept_R",
    "n_below_R",
    "sampler",
    "log_Z",
    "log_Z_uncert",
)

README = """# `area_law_2026-11/` — the group-area law: Weibull + log-normal, per survey

Frozen summaries of `sunspot-homogenization`'s per-survey fits of the **group-area
distribution**. Fixed by **ADR-011**. This project **does not refit them**.

## The law

The area of a sunspot group is modelled on **log-area** `z = ln A` as a two-component mixture:

    p(z) = pi_W * Weibull_k,lambda(z)  +  (1 - pi_W) * Normal(z; mu, sigma)

where the Weibull's density in log-area is

    log p_W(z) = log k - k*log(lambda) + k*z - exp(k*(z - log lambda))

a left-skewed Gumbel. The Weibull captures **small objects** (pores and ephemeral
regions, a fragmentation regime); the log-normal captures **large objects** (bipolar
active regions, multiplicative growth).

**This is not a two-log-normal mixture.** Fitting two log-normals folds observational
effects — resolution, detection bias, cadence-vs-lifetime convolution — into the
small-object component. Upstream's paper adopts Weibull+log-normal because it is the
form Muñoz-Jaramillo et al. (2015, ApJ 804:68) found to best reconcile eleven
databases, and names the two-log-normal form as the alternative it rejected.

## ⚠ Two mixing-weight conventions

| source | weights | relation |
|---|---|---|
| upstream, and this file | `pi_W` weights the **Weibull** | — |
| Muñoz-Jaramillo et al. (2015) | `c` weights the **log-normal** | **`c = 1 - pi_W`** |

Getting this backwards silently swaps the two populations. The 2015 paper's published
activity-level power law `c(AL) = 0.66*AL^0.230` is on the **log-normal** weight.

## Truncation

Each survey is hard-truncated at its own completeness threshold before fitting — the
recipe of the 2015 paper §2.4 (an order of magnitude above the smallest detected
structure), implemented upstream as `threshold_method = "lifetime_divergence"` with
`trunc_factor = 10`. Those thresholds are the `threshold_MSH` column of
`../reference_groups_2026-10/survey_calibration.csv`, and the fit uses **one area per
group** (`group_single_entry.parquet`, column `area_corr`, strictly greater than the
threshold). The kept counts reproduce exactly:

| survey | threshold (MSH) | `n_kept` |
|---|---|---|
| rgo | 28.084 | 16,880 |
| debrecen | 20.424 | 8,306 |
| kmas | 23.415 | 15,662 |

Schwabe is calibrated on the **umbral** scale, so its threshold applies to a different
area column and does not reproduce against `area_corr`; Spörer differs by 13 groups.
Neither is used in Step 1 (issue **D23**).

## Activity-level dependence

The 2015 paper finds the **Weibull component independent of activity level** — the
"magnetic baseline" — while the **log-normal component varies strongly** with it. The
fits frozen here are pooled over all activity levels; Step 1 holds `(k, lambda)` fixed
at these values and fits the log-normal's activity dependence on top. See ADR-011 and
`docs/results/step-1-data-foundation.md`.

## Files

| File | What |
|---|---|
| `solo_wln_posterior.csv` | one row per survey: parameter medians and 16/84
  percentiles, truncation, kept counts, evidence |
| `provenance.json` | upstream commit, source paths, checksums, lineage |

Produced by `scripts/export_area_law.py`, which pins the upstream commit and verifies
byte reproducibility with `--check`.

## Citation

`sunspot-homogenization` (MIT, same PI) for the calibration; Muñoz-Jaramillo et al.
(2015, ApJ 804:68) for the Weibull+log-normal form and its activity-level dependence.
"""


def upstream_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def pinned_commit() -> str:
    provenance = json.loads((REFERENCE_GROUPS / "provenance.json").read_text())
    return provenance["upstream"]["commit"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(name: str) -> dict | None:
    path = SOLO_WLN / f"{name}.pkl"
    if not path.is_file():
        print(f"  {name}: absent upstream, skipped")
        return None
    sys.path.insert(0, str(UPSTREAM))
    with path.open("rb") as handle:
        posterior = pickle.load(handle)

    samples = posterior.posterior_samples
    row: dict = {
        "survey": name.rsplit("_", 1)[0],
        "scale": name.rsplit("_", 1)[1],
        "condition": name,
    }
    for key in PARAMS:
        values = np.asarray(samples[key], dtype=float).ravel()
        row[f"{key}_med"] = float(np.median(values))
        row[f"{key}_lo"] = float(np.percentile(values, 16))
        row[f"{key}_hi"] = float(np.percentile(values, 84))
    row["lambda_W_med"] = float(np.exp(row["log_lambda_W_med"]))
    row["n_samples"] = int(np.asarray(samples[PARAMS[0]]).size)

    meta = posterior.metadata
    for key in META_KEYS:
        if key in meta:
            value = meta[key]
            row[key] = float(value) if isinstance(value, (int, float, np.floating)) else value
    row["model_name"] = posterior.model_name
    row["source_sha256"] = sha256(path)
    print(
        f"  {name}: k_W={row['k_W_med']:.4f} lambda_W={row['lambda_W_med']:.2f} "
        f"mu_LN={row['mu_LN_med']:.4f} sigma_LN={row['sigma_LN_med']:.4f} "
        f"pi_W={row['pi_W_med']:.4f}"
    )
    return row


def build() -> tuple[pd.DataFrame, dict]:
    print("reading upstream solo_wln posteriors:")
    rows = [row for row in (summarize(name) for name in SURVEYS) if row is not None]
    table = pd.DataFrame(rows)
    front = ["survey", "scale", "condition", "model_name"]
    table = table[front + [c for c in table.columns if c not in front]]

    provenance = {
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "generated_by": "scripts/export_area_law.py",
        "upstream": {
            "repo": str(UPSTREAM),
            "commit": upstream_commit(),
            "source_dir": "data/processed/calibration/solo_wln",
            "license": "MIT",
        },
        "model": {
            "form": "Weibull + log-normal mixture on log-area z = ln A",
            "weight_convention": "pi_W weights the WEIBULL; the 2015 paper's c weights "
            "the LOG-NORMAL, so c = 1 - pi_W",
            "grain": "one area per group (group_single_entry), column area_corr",
            "truncation": "hard, at the per-survey threshold_MSH in "
            "reference_groups_2026-10/survey_calibration.csv "
            "(lifetime_divergence threshold, trunc_factor 10)",
            "refit_here": False,
        },
        "data_lineage": (
            "Sunspot-derived only: per-survey group areas and their calibration. No "
            "isotope, aurora, proxy or SILSO input (ADR-002)."
        ),
    }
    return table, provenance


def write(table: pd.DataFrame, provenance: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "solo_wln_posterior.csv", index=False, lineterminator="\n")
    (out / "README.md").write_text(README)
    provenance["sha256"] = {
        name: sha256(out / name) for name in ("solo_wln_posterior.csv", "README.md")
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="compare instead of overwrite")
    ap.add_argument("--allow-commit-mismatch", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=AREA_LAW)
    args = ap.parse_args()

    head, pinned = upstream_commit(), pinned_commit()
    print(f"upstream HEAD {head}\npinned        {pinned}")
    if head != pinned and not args.allow_commit_mismatch:
        raise SystemExit(
            "upstream is not at the commit the reference snapshot was exported from; "
            "re-export the reference too, or pass --allow-commit-mismatch knowing the "
            "area law and the reference would then come from different code"
        )

    table, provenance = build()

    if args.check:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp)
            write(table, provenance, scratch)
            ok = True
            for name in ("solo_wln_posterior.csv", "README.md"):
                same = sha256(scratch / name) == sha256(args.out_dir / name)
                print(f"  {name}: {'MATCH' if same else 'DIFFERS'}")
                ok &= same
        raise SystemExit(0 if ok else 1)

    write(table, provenance, args.out_dir)
    print(f"\nwrote {args.out_dir} ({len(table)} surveys)")
    print(f"  relative to {DATA_RAW}")


if __name__ == "__main__":
    main()
