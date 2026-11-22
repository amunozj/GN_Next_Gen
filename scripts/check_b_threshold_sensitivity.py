"""Is the cross-survey area-law difference the Sun, the survey, or the truncation point?

Check B's stationarity stage compares ``(mu, sigma, c)`` across surveys at matched
activity, but each survey is fitted on its **own** support: upstream truncates every
survey at its own ``threshold_MSH`` (the ``lifetime_divergence`` value under the 10x rule
of Munoz-Jaramillo et al. 2015, section 2.4), which is 28.08 MSH for RGO, 20.42 for
Debrecen and 23.41 for KMAS.  Three fits on three supports are not obviously comparable,
so this refits all three on a **common** support and re-runs the same matched-activity
comparison.  If a difference is an artifact of where the data were cut, it shrinks; if it
is a property of the survey, it does not.

``(k, lambda)`` stay fixed at upstream's per-survey values throughout (ADR-011), which is
what makes the comparison a clean one-factor change: only the truncation point moves.
Those values were calibrated under the per-survey cut, so the common-threshold fits are a
**diagnostic, not a recalibration** -- which is also why nothing here is refitted into the
frozen snapshot.

Usage
-----
    python scripts/check_b_threshold_sensitivity.py [--thresholds 30 40] [--bins 10]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gn_next_gen.io import load_day_status, load_group_single_entry  # noqa: E402
from gn_next_gen.paths import OUTPUTS  # noqa: E402
from gn_next_gen.reference.area_fit import fit_by_activity  # noqa: E402
from gn_next_gen.reference.area_law import (  # noqa: E402
    activity_level,
    load_upstream_parameters,
)

SURVEYS = ("rgo", "debrecen", "kmas")
PARAMETERS = ("mu_LN", "sigma_LN", "c_LN")


def prepare(survey: str, threshold: float | None):
    """One survey's groups above ``threshold``, or above its own if None."""
    status = load_day_status([survey])
    single = load_group_single_entry([survey])
    activity = activity_level(status, single, scope="whole", weighting="emergence")
    base = load_upstream_parameters(survey)
    cut = base.threshold_msh if threshold is None else float(threshold)
    kept = single[single["area_corr"] > cut].copy()
    kept["activity"] = kept["date_ord"].map(activity)
    kept = kept[np.isfinite(kept["activity"])]
    # The truncated likelihood reads its cut off the parameters, so move it too.
    return kept, dataclasses.replace(base, threshold_msh=cut)


def compare(threshold: float | None, *, n_bins: int, min_groups: int) -> pd.DataFrame:
    """Mean |difference| in each parameter between RGO and each other survey."""
    fits, kept = {}, {}
    for survey in SURVEYS:
        groups, base = prepare(survey, threshold)
        kept[survey] = len(groups)
        fits[survey] = pd.DataFrame(
            [
                f.as_row()
                for f in fit_by_activity(
                    groups,
                    base,
                    n_bins=n_bins,
                    free_weibull=False,
                    stage="fixed_weibull",
                    min_groups=min_groups,
                )
            ]
        )

    reference = fits[SURVEYS[0]]
    rows = []
    for other in SURVEYS[1:]:
        frame = fits[other]
        for name in PARAMETERS:
            matched = [
                float(frame.iloc[int(np.argmin(np.abs(frame["activity"] - a)))][name]) - float(v)
                for a, v in zip(reference["activity"], reference[name], strict=True)
            ]
            rows.append(
                {
                    "threshold": "per-survey" if threshold is None else f"{threshold:g}",
                    "pair": f"{SURVEYS[0]}_vs_{other}",
                    "parameter": name,
                    "mean_abs_difference": float(np.mean(np.abs(matched))),
                    "max_abs_difference": float(np.max(np.abs(matched))),
                    "n_kept_reference": kept[SURVEYS[0]],
                    "n_kept_other": kept[other],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[30.0, 40.0])
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--min-groups", type=int, default=200)
    ap.add_argument("--out-dir", type=Path, default=OUTPUTS / "step1" / "check_b")
    args = ap.parse_args()

    tables = [compare(None, n_bins=args.bins, min_groups=args.min_groups)]
    tables += [compare(t, n_bins=args.bins, min_groups=args.min_groups) for t in args.thresholds]
    table = pd.concat(tables, ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "threshold_sensitivity.csv"
    table.to_csv(out, index=False, lineterminator="\n")

    wide = table.pivot_table(
        index=["threshold", "pair"], columns="parameter", values="mean_abs_difference"
    )
    print("mean |difference| at matched activity, by truncation point:\n")
    print(wide[list(PARAMETERS)].round(3).to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
