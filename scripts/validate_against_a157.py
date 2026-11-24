"""Criterion A7: do our ports behave as Chatzistergos et al. (2025) published?

The plan dropped the ``legacy/ADFSSN/runcsvs/`` reproduction -- those files were
not produced by the committed driver settings, and the snapshot already carries
``uncommitted.diff``, so the target is not well defined.  This replaces it with an
**external, published** target: rebuild A157's own setup and check that our ports
reproduce the behaviour their figures show.

Their setup, from §2.1 and §3.1: RGO after **1900**, 121 synthetic observers made
by imposing a hard acuity threshold on the RGO catalogue, **full overlap** with
RGO, compared as **annual means**.  No graph, no chaining, no drift -- so this
calls the pairwise estimators directly rather than the chained calibrators, which
is the faithful reproduction.

**Units matter here and are easy to get wrong.**  A157 quotes acuity in millionths
of the solar *disc* (msd); this project works in millionths of the *hemisphere*
(MSH), and the disc is half the hemisphere, so ``MSH = msd / 2`` (issue D4).
Their 300 msd grid is 150 MSH, their HoSc98 exclusion at 31 msd is 15.5 MSH, and
their CEA17 exclusion at 87 msd is 43.5 MSH.

What must hold (their Fig. 3b and §4):

* **HoSc98 and SvSc16** show a mean difference that *falls* with acuity -- "the
  overall mean value is consistently underestimated by these methods";
* **CEA17, UEA21 and VHS24** stay near zero -- they "return rather symmetric
  errors that almost completely balance out over solar cycle timescales";
* the exclusion thresholds land near 31 msd for HoSc98 and 87 msd for CEA17.

Takes roughly 30-60 minutes, so the PI launches it::

    nohup python scripts/validate_against_a157.py > outputs/step2/logs/a157.log 2>&1 &
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gn_next_gen.baselines.linear import VHS24, HoSc98, SvSc16  # noqa: E402
from gn_next_gen.baselines.pmf import conversion_matrix, passes_criteria  # noqa: E402
from gn_next_gen.io import load_group_days  # noqa: E402
from gn_next_gen.paths import OUTPUTS  # noqa: E402

#: A157 works in millionths of the solar disc; this project in millionths of the
#: hemisphere.  The disc is half the hemisphere (D4).
MSD_TO_MSH = 0.5

#: Their grid: every integer 0-100 msd, then every tenth to 300.
ACUITY_MSD = np.concatenate([np.arange(0, 101, 1), np.arange(110, 301, 10)])

#: Their published exclusion thresholds, in msd.
PUBLISHED_EXCLUSIONS = {"HoSc98": 31.0, "CEA17": 87.0}

#: "only data after 1900 were considered" (their §2.1).
FIRST_YEAR = 1900

MAX_COUNT = 41


def rgo_daily_counts(first_year: int = FIRST_YEAR) -> tuple[np.ndarray, pd.DataFrame]:
    """RGO group-days after ``first_year``, and the days it observed."""
    groups = load_group_days(surveys=["rgo"])
    cutoff = dt.date(first_year, 1, 1).toordinal()
    groups = groups[groups["date_ord"] >= cutoff]
    days = np.sort(groups["date_ord"].unique())
    return days, groups


def counts_at(groups: pd.DataFrame, days: np.ndarray, acuity_msh: float) -> np.ndarray:
    above = groups[groups["area_msh" if "area_msh" in groups else "area_corr"] > acuity_msh]
    return above.groupby("date_ord").size().reindex(days, fill_value=0).to_numpy(dtype=np.int64)


def annual_mean(days: np.ndarray, counts: np.ndarray) -> pd.Series:
    years = pd.to_datetime(days - 719_163, unit="D").year
    return pd.Series(counts, index=years).groupby(level=0).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS / "step2" / "a157")
    parser.add_argument("--max-acuity-msd", type=float, default=300.0)
    args = parser.parse_args()

    days, groups = rgo_daily_counts()
    reference = counts_at(groups, days, 0.0)
    reference_annual = annual_mean(days, reference)
    print(f"RGO from {FIRST_YEAR}: {len(days)} days, mean count {reference.mean():.3f}")

    estimators = {"HoSc98": HoSc98(), "SvSc16": SvSc16(), "VHS24": VHS24()}
    rows = []
    for acuity_msd in ACUITY_MSD:
        if acuity_msd > args.max_acuity_msd:
            continue
        acuity_msh = float(acuity_msd) * MSD_TO_MSH
        observer = counts_at(groups, days, acuity_msh)
        if observer.sum() == 0:
            continue

        left, right = (days, reference), (days, observer)
        for name, estimator in estimators.items():
            estimate = estimator.estimate(
                left, right, estimator.prepare(left), estimator.prepare(right)
            )
            calibrated = estimate.intercept + estimate.factor * observer
            difference = annual_mean(days, calibrated) - reference_annual
            rows.append(
                {
                    "acuity_msd": float(acuity_msd),
                    "acuity_msh": acuity_msh,
                    "method": name,
                    "factor": estimate.factor,
                    "usable": bool(estimate.usable),
                    "mean_difference": float(difference.mean()),
                    "rms_difference": float(np.sqrt((difference**2).mean())),
                }
            )

        # CEA17 and UEA21 share the conversion matrix in this idealised case --
        # "by design, CEA17 and UEA21 are identical in these metrics in this
        # case", because UEA21's acuity is determined from the very data that
        # built the reference curves (their §3.1).
        matrix, diagnostics = conversion_matrix(reference, observer, max_count=MAX_COUNT)
        values = np.arange(MAX_COUNT + 1)
        mean_map = matrix.T @ values
        calibrated = mean_map[np.clip(observer, 0, MAX_COUNT)]
        difference = annual_mean(days, calibrated) - reference_annual
        for name in ("CEA17", "UEA21"):
            rows.append(
                {
                    "acuity_msd": float(acuity_msd),
                    "acuity_msh": acuity_msh,
                    "method": name,
                    "factor": float(mean_map[1]),
                    "usable": bool(passes_criteria(diagnostics)),
                    "mean_difference": float(difference.mean()),
                    "rms_difference": float(np.sqrt((difference**2).mean())),
                }
            )
        if acuity_msd % 25 == 0:
            print(f"  acuity {acuity_msd:5.0f} msd done", flush=True)

    table = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "a157_validation.csv"
    table.to_csv(out, index=False, lineterminator="\n")

    verdict = assess(table)
    verdict.to_csv(args.out_dir / "a157_verdict.csv", index=False, lineterminator="\n")

    # Why an exclusion limit may differ even when every sign agrees: the
    # catalogues are not the same.  A157 used the 2017 RGO; this project uses the
    # renovated per-group catalogue, whose small-group completeness differs
    # (D14 puts RGO's floor at 28.1 MSH).  Thresholding a catalogue with fewer
    # small groups removes less, so every criterion bites later.
    catalogue = _catalogue_profile(groups, days)
    catalogue.to_csv(args.out_dir / "a157_catalogue.csv", index=False, lineterminator="\n")

    print(f"\nwrote {out}")
    print()
    print(verdict.to_string(index=False))
    print()
    print("Catalogue profile -- why an exclusion limit can differ:")
    print(catalogue.to_string(index=False))


def _catalogue_profile(groups: pd.DataFrame, days: np.ndarray) -> pd.DataFrame:
    """Fraction of group-days removed at each acuity, and the k it implies.

    HoSc98's factor is a ratio of sums, so losing a fraction ``f`` of group-days
    gives ``k ~ 1/(1-f)``.  Its 0.6-1.4 gate therefore trips once ``f`` reaches
    about 0.29, and where that happens is a property of the catalogue, not of
    the port.
    """
    areas = groups["area_corr"].to_numpy()
    rows = []
    for acuity_msd in (10, 20, 31, 60, 87, 150, 300):
        lost = float((areas < acuity_msd * MSD_TO_MSH).mean())
        rows.append(
            {
                "acuity_msd": acuity_msd,
                "acuity_msh": acuity_msd * MSD_TO_MSH,
                "fraction_of_group_days_lost": round(lost, 4),
                "implied_hosc98_k": round(1.0 / max(1.0 - lost, 1e-9), 3),
            }
        )
    del days
    return pd.DataFrame(rows)


def _exclusion_limit(block: pd.DataFrame) -> float:
    """The acuity above which a method excludes an observer *and stays* excluding.

    A157 reports an upper limit -- "CEA17 would exclude observers with A >= 87
    msd" -- not the first acuity at which a criterion happens to trip.  Isolated
    exclusions at low acuity are chi-square noise on a near-perfect fit; what is
    comparable is where the method gives up for good.
    """
    usable = block["usable"].to_numpy()
    acuity = block["acuity_msd"].to_numpy()
    for index in range(len(usable)):
        if not usable[index:].any():
            return float(acuity[index])
    return float("nan")


def assess(table: pd.DataFrame) -> pd.DataFrame:
    """Sign agreement with A157's Fig. 3b, and the exclusion limits."""
    rows = []
    expected_falling = {"HoSc98", "SvSc16"}
    for method, block in table.groupby("method"):
        block = block.sort_values("acuity_msd")
        slope = float(np.polyfit(block["acuity_msd"], block["mean_difference"], 1)[0])
        falling = slope < -1e-5
        expected = method in expected_falling
        rows.append(
            {
                "method": method,
                "mean_difference_slope": round(slope, 6),
                "falls_with_acuity": falling,
                "a157_expects_falling": expected,
                "sign_agrees": bool(falling == expected),
                "excluded_above_msd": _exclusion_limit(block),
                "a157_excluded_msd": PUBLISHED_EXCLUSIONS.get(method, np.nan),
                "n_excluded": int((~block["usable"]).sum()),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
