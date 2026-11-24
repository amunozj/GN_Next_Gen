"""UEA21 and GRP: calibration from a record's own statistics, with no overlap.

These are the two methods that do not need a direct overlap, which is their great
advantage and their hidden assumption.  A157 puts the assumption plainly: they
"carry the implicit assumption that the statistical property they use remains
stable in time".  UEA21 uses the fraction of active days; GRP -- the GRPSSN fork
of the same machinery -- uses the distribution of daily group counts.

How it works, following A157 §2.2.4.  The **reference survey's own area catalogue**
is public (RGO's is; here it is in the open half for the same reason), so it can be
rethresholded at any acuity to produce a library of what an observer of that acuity
would have recorded.  Matching an observer's statistic against that library gives
its acuity, and the matching acuity's conversion matrix gives the calibration.  No
overlap, no chain, no graph.

**Cycle phase does not enter.**  The legacy ADFSSN engine in ``legacy/ADFSSN/``
takes cycle phases and activity limits from SILSO v2 and scans phase-matched
windows; the reuse map lists that machinery under Drop, and it is not what A157
describes as UEA21.  Porting the published method rather than the 2017 scanning-
window variant means no SILSO is needed anywhere in this step -- the planning
session's decision 3 is satisfied by construction rather than by substitution.
The scanning-window/EMD variant is the thing we did not port.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from gn_next_gen.baselines.base import CalibrationResult, station_series
from gn_next_gen.baselines.pmf import MAX_COUNT, conversion_matrix

#: Acuity grid searched, in MSH.  Dense where real observers sit (A157: most real
#: observers back to 1749 are below 60 msd), sparse in the poor-instrument tail.
ACUITY_GRID = np.concatenate([np.arange(0.0, 100.0, 2.0), np.arange(100.0, 320.0, 20.0)])

#: Month length used for the active-day fraction, in days.  The legacy driver's
#: fixed-length "month" (``MoLngt``), kept because the statistic is defined on it.
MONTH_DAYS = 30

#: Minimum usable months before an acuity is claimed.  The method rests on the
#: *distribution* of monthly active-day fractions, and a handful of months does
#: not have one.  Measured on S1: recovery correlation with the true threshold
#: rises from 0.27 over all observers to 0.61 for those with 3,000+ observing
#: days, while the bias stays put -- so coverage controls the scatter and the
#: bias is the method's own.
MIN_MONTHS = 12

#: GRP discards 0- and 1-group days (``LowLim = 2`` in the GRPSSN driver), which
#: is also its documented weakness: those days carry the most threshold
#: information at low activity.
GRP_LOW_LIMIT = 2


def _reference_counts_at(areas: pd.DataFrame, acuity: float) -> pd.Series:
    """Counts the reference catalogue implies for an observer of this acuity."""
    above = areas[areas["area_msh"] > acuity]
    counts = above.groupby("date_ord").size()
    return counts.reindex(np.unique(areas["date_ord"].to_numpy()), fill_value=0)


def _active_day_fractions(days: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Active-day fraction per fixed-length month, for months with coverage."""
    if len(days) == 0:
        return np.empty(0)
    block = (days - days[0]) // MONTH_DAYS
    frame = pd.DataFrame({"block": block, "active": counts > 0})
    grouped = frame.groupby("block")["active"]
    observed = grouped.size()
    fraction = grouped.mean()
    keep = observed >= MONTH_DAYS // 3
    return fraction[keep].to_numpy()


def _count_distribution(counts: np.ndarray, max_count: int = MAX_COUNT) -> np.ndarray:
    """Normalised histogram of daily counts, above GRP's low limit."""
    kept = counts[counts >= GRP_LOW_LIMIT]
    if len(kept) == 0:
        return np.zeros(max_count + 1)
    histogram = np.bincount(np.clip(kept, 0, max_count), minlength=max_count + 1).astype(float)
    return histogram / histogram.sum()


def _cumulative_distance(sample: np.ndarray, library: np.ndarray, grid: np.ndarray) -> float:
    """Sum of squared residuals between two cumulative distributions."""
    if len(sample) == 0 or len(library) == 0:
        return np.inf
    sample_cdf = np.searchsorted(np.sort(sample), grid, side="right") / len(sample)
    library_cdf = np.searchsorted(np.sort(library), grid, side="right") / len(library)
    return float(((sample_cdf - library_cdf) ** 2).sum())


def build_library(
    reference_areas: pd.DataFrame, *, fingerprint: str = "adf"
) -> dict[float, np.ndarray]:
    """Rethreshold the reference catalogue across the acuity grid.

    ``fingerprint`` is ``"adf"`` (UEA21: monthly active-day fractions) or
    ``"counts"`` (GRP: the daily count distribution).
    """
    days = np.unique(reference_areas["date_ord"].to_numpy())
    library: dict[float, np.ndarray] = {}
    for acuity in ACUITY_GRID:
        counts = _reference_counts_at(reference_areas, acuity).to_numpy()
        if fingerprint == "adf":
            library[float(acuity)] = _active_day_fractions(days, counts)
        elif fingerprint == "counts":
            library[float(acuity)] = _count_distribution(counts)
        else:
            raise ValueError(f"unknown fingerprint {fingerprint!r}")
    return library


def recover_acuity(
    days: np.ndarray,
    counts: np.ndarray,
    library: dict[float, np.ndarray],
    *,
    fingerprint: str = "adf",
) -> float:
    """The library acuity whose fingerprint best matches this observer's.

    Returns NaN when the record cannot support the statistic, which the score
    table reports as an unreached observer rather than hiding.

    Two published behaviours show up here and are **not** bugs.  UEA21
    "systematically overestimates the acuity of the observer, thus in the end
    also amplifies the group numbers" (A157 §3.2), and its acuities pile up at
    the top of the grid because "300 msd was the highest acuity considered for
    the synthetic observers used to produce the reference curves. This introduces
    a cap in the determined acuities with the UEA21 method."
    """
    if fingerprint == "adf":
        sample = _active_day_fractions(days, counts)
        if len(sample) < MIN_MONTHS:
            return np.nan
        grid = np.linspace(0.0, 1.0, 51)
        scores = {a: _cumulative_distance(sample, lib, grid) for a, lib in library.items()}
    else:
        sample = _count_distribution(counts)
        if sample.sum() == 0:
            return np.nan
        scores = {a: float(((sample - lib) ** 2).sum()) for a, lib in library.items()}
    best = min(scores, key=lambda a: scores[a])
    return float(best)


def calibrate_acuity(
    observations: pd.DataFrame,
    reference_areas: pd.DataFrame,
    reference_threshold_msh: float,
    *,
    fingerprint: str = "adf",
    name: str | None = None,
    seed: int = 0,
    max_count: int = MAX_COUNT,
) -> CalibrationResult:
    """Calibrate every observer from its own statistics, with no overlap needed."""
    rng = np.random.default_rng(seed)
    series = station_series(observations)
    library = build_library(reference_areas, fingerprint=fingerprint)

    anchor = _reference_counts_at(reference_areas, reference_threshold_msh)
    matrices: dict[float, np.ndarray] = {}
    for acuity in ACUITY_GRID:
        at_acuity = _reference_counts_at(reference_areas, acuity)
        matrix, _diagnostics = conversion_matrix(
            anchor.to_numpy(), at_acuity.to_numpy(), max_count=max_count, rng=rng
        )
        matrices[float(acuity)] = matrix

    values = np.arange(max_count + 1)
    calibrated, observer_rows = [], []
    for station, (days, counts) in series.items():
        acuity = recover_acuity(days, counts, library, fingerprint=fingerprint)
        if not np.isfinite(acuity):
            observer_rows.append(
                {
                    "station": station,
                    "factor": np.nan,
                    "factor_sd": np.nan,
                    "recovered_threshold_msh": np.nan,
                    "n_hops": 0,
                    "reached": False,
                }
            )
            continue
        matrix = matrices[acuity]
        mean = matrix.T @ values
        sd = np.sqrt(np.maximum(matrix.T @ (values**2) - mean**2, 0.0))
        clipped = np.clip(counts, 0, max_count)
        calibrated.append(
            pd.DataFrame(
                {"date_ord": days, "value": mean[clipped], "sd": sd[clipped], "station": station}
            )
        )
        observer_rows.append(
            {
                "station": station,
                "factor": float(mean[1]) if max_count > 0 else np.nan,
                "factor_sd": float(sd[1]) if max_count > 0 else np.nan,
                "recovered_threshold_msh": acuity,
                "n_hops": 0,
                "reached": True,
            }
        )

    label = name or ("UEA21" if fingerprint == "adf" else "GRP")
    if not calibrated:
        return CalibrationResult(
            pd.DataFrame(columns=["date_ord", "gn", "gn_sd", "n_observers"]),
            pd.DataFrame(observer_rows),
            label,
            {"reason": "no observer matched the library"},
        )

    stacked = pd.concat(calibrated, ignore_index=True)
    stacked["weight"] = 1.0 / np.maximum(stacked["sd"].to_numpy(), 0.25) ** 2
    grouped = stacked.groupby("date_ord")
    series_out = pd.DataFrame(
        {
            "date_ord": grouped["value"].mean().index.to_numpy(),
            "gn": grouped.apply(
                lambda b: float(np.average(b["value"], weights=b["weight"])),
                include_groups=False,
            ).to_numpy(),
            "gn_sd": np.sqrt(1.0 / grouped["weight"].sum().to_numpy()),
            "n_observers": grouped.size().to_numpy(),
        }
    )
    return CalibrationResult(
        series=series_out,
        observers=pd.DataFrame(observer_rows),
        name=label,
        meta={"fingerprint": fingerprint, "n_acuities": len(ACUITY_GRID)},
    )


def _year(date_ord: int) -> int:
    return dt.date.fromordinal(int(date_ord)).year
