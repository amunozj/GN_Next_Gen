"""Scoring: point error at four time scales, and whether the intervals are honest.

ADR-007 asks for coverage, not only point errors, and the research notes are
specific about why.  Every one of the three approaches this project compares fails
in a way that shows up as a coverage failure before it shows up as a bias: the
joint model through misspecification, the global network through double-counted
days, simulation-based inference through extrapolation.  A157 scores RMS, mean,
relative and maximum-absolute difference and never scores an interval, so this is
the axis on which the comparison is new.

Metrics, per (scenario, realization, method):

* **centred RMSE** at daily, monthly, yearly and cycle-mean scales, beside the
  plain bias, because HOME scores at several averaging scales and a method can be
  unbiased daily and badly wrong per cycle;
* **cycle amplitude error** and **cycle-to-cycle ratio error** -- the quantity
  A157 found the linear methods skew, amplifying strong cycles and suppressing
  weak ones;
* **secular trend error** over 1610-, 1700- and 1850-present;
* **Maunder-Minimum mean level**, for S3;
* **CRPS** and **PIT** on yearly means, and empirical **interval coverage** at
  50/68/90/95%;
* **observer acuity recovery**, where a method estimates one.

Coverage is reported at four levels because the sources disagree -- the spec says
50 and 90, the 7-step design says 68 and 95, the SBI note says 50/68/95 -- and
computing all four costs nothing.  The spec's 50 and 90 are the headline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kstest, norm

from gn_next_gen.baselines.base import CalibrationResult
from gn_next_gen.benchmark.observers import CASE_STUDY_KINDS
from gn_next_gen.benchmark.truth import Truth

#: Nominal central-interval levels.  The first two are the spec's headline.
COVERAGE_LEVELS = (0.50, 0.90, 0.68, 0.95)

#: Trend windows, as start years.  The record ends at the scenario's last day.
TREND_START_YEARS = (1610, 1700, 1850)

#: The Maunder Minimum, for the S3 level metric.
MAUNDER_YEARS = (1645, 1715)

#: 1970-01-01 as a proleptic-Gregorian ordinal.
UNIX_EPOCH_ORDINAL = 719_163


def _calendar(date_ord: np.ndarray) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(date_ord, dtype=np.int64) - UNIX_EPOCH_ORDINAL, unit="D")


def align(truth: Truth, result: CalibrationResult) -> pd.DataFrame:
    """Truth beside estimate on the days both have, with the calendar attached."""
    reference = truth.reference_counts()
    frame = result.series.set_index("date_ord")
    common = frame.index.intersection(reference.index)
    out = pd.DataFrame(
        {
            "date_ord": common,
            "truth": reference.loc[common].to_numpy(dtype=float),
            "estimate": frame.loc[common, "gn"].to_numpy(dtype=float),
            "sd": frame.loc[common, "gn_sd"].to_numpy(dtype=float),
        }
    )
    calendar = _calendar(out["date_ord"].to_numpy())
    out["year"] = calendar.year
    out["month"] = calendar.year * 12 + calendar.month
    return out


def _bias_and_crmse(truth: np.ndarray, estimate: np.ndarray) -> tuple[float, float]:
    """Bias, and RMSE after removing it -- HOME's centred RMSE."""
    if len(truth) == 0:
        return np.nan, np.nan
    error = estimate - truth
    bias = float(error.mean())
    return bias, float(np.sqrt(((error - bias) ** 2).mean()))


def scale_errors(aligned: pd.DataFrame) -> dict[str, float]:
    """Bias and centred RMSE at daily, monthly and yearly scales."""
    out: dict[str, float] = {}
    bias, crmse = _bias_and_crmse(aligned["truth"].to_numpy(), aligned["estimate"].to_numpy())
    out["bias_daily"], out["crmse_daily"] = bias, crmse

    for label, key in (("monthly", "month"), ("yearly", "year")):
        grouped = aligned.groupby(key)[["truth", "estimate"]].mean()
        bias, crmse = _bias_and_crmse(grouped["truth"].to_numpy(), grouped["estimate"].to_numpy())
        out[f"bias_{label}"], out[f"crmse_{label}"] = bias, crmse
    return out


def cycle_boundaries(yearly: pd.Series, *, min_length: int = 7) -> list[tuple[int, int]]:
    """Cycle edges found as minima of the smoothed yearly truth.

    Derived from the truth itself rather than from a published cycle list, so the
    scorer never needs a date table the firewall would have to vet, and so it
    works on synthetic cycles whose timing is jittered by construction.
    """
    if len(yearly) < 2 * min_length:
        return []
    smoothed = yearly.rolling(3, center=True, min_periods=1).mean()
    values = smoothed.to_numpy()
    years = smoothed.index.to_numpy()

    minima = [0]
    for index in range(1, len(values) - 1):
        window = values[max(0, index - 3) : index + 4]
        if values[index] == window.min() and years[index] - years[minima[-1]] >= min_length:
            minima.append(index)
    minima.append(len(values) - 1)
    return [(int(years[minima[i]]), int(years[minima[i + 1]])) for i in range(len(minima) - 1)]


def cycle_errors(aligned: pd.DataFrame) -> dict[str, float]:
    """Amplitude and cycle-to-cycle ratio error -- A157's headline quantity."""
    yearly = aligned.groupby("year")[["truth", "estimate"]].mean()
    cycles = cycle_boundaries(yearly["truth"])
    if len(cycles) < 2:
        return {
            "n_cycles": 0,
            "cycle_amplitude_bias": np.nan,
            "cycle_amplitude_crmse": np.nan,
            "cycle_ratio_error": np.nan,
            "amplitude_slope": np.nan,
        }

    truth_amp, estimate_amp = [], []
    for start, end in cycles:
        window = yearly.loc[start:end]
        if window.empty:
            continue
        truth_amp.append(float(window["truth"].max()))
        estimate_amp.append(float(window["estimate"].max()))
    truth_amp, estimate_amp = np.array(truth_amp), np.array(estimate_amp)

    bias, crmse = _bias_and_crmse(truth_amp, estimate_amp)
    ratio_error = np.nan
    if len(truth_amp) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            truth_ratio = truth_amp[1:] / truth_amp[:-1]
            estimate_ratio = estimate_amp[1:] / estimate_amp[:-1]
        finite = np.isfinite(truth_ratio) & np.isfinite(estimate_ratio)
        if finite.any():
            ratio_error = float(np.mean(estimate_ratio[finite] - truth_ratio[finite]))

    # The A157 signature: does the method amplify strong cycles relative to weak
    # ones?  A positive slope of (estimate - truth) against truth amplitude is
    # exactly the secular bias they attribute to the linear scalings.
    slope = np.nan
    if len(truth_amp) > 2 and np.ptp(truth_amp) > 0:
        slope = float(np.polyfit(truth_amp, estimate_amp - truth_amp, 1)[0])

    return {
        "n_cycles": len(truth_amp),
        "cycle_amplitude_bias": bias,
        "cycle_amplitude_crmse": crmse,
        "cycle_ratio_error": ratio_error,
        "amplitude_slope": slope,
    }


def trend_errors(aligned: pd.DataFrame) -> dict[str, float]:
    """Error in the linear trend, per century, over several start years."""
    yearly = aligned.groupby("year")[["truth", "estimate"]].mean()
    out: dict[str, float] = {}
    for start in TREND_START_YEARS:
        window = yearly[yearly.index >= start]
        key = f"trend_error_{start}"
        if len(window) < 20:
            out[key] = np.nan
            continue
        years = window.index.to_numpy(dtype=float)
        truth_slope = np.polyfit(years, window["truth"].to_numpy(), 1)[0]
        estimate_slope = np.polyfit(years, window["estimate"].to_numpy(), 1)[0]
        out[key] = float((estimate_slope - truth_slope) * 100.0)
    return out


def maunder_level(aligned: pd.DataFrame) -> dict[str, float]:
    """Mean level over the Maunder window, truth and estimate."""
    low, high = MAUNDER_YEARS
    window = aligned[(aligned["year"] >= low) & (aligned["year"] <= high)]
    if window.empty:
        return {"maunder_truth": np.nan, "maunder_estimate": np.nan, "maunder_error": np.nan}
    truth = float(window["truth"].mean())
    estimate = float(window["estimate"].mean())
    return {
        "maunder_truth": truth,
        "maunder_estimate": estimate,
        "maunder_error": estimate - truth,
    }


def crps_gaussian(truth: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Closed-form CRPS of a Gaussian predictive distribution."""
    sd = np.where(sd > 0, sd, np.nan)
    z = (truth - mean) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def uncertainty_scores(aligned: pd.DataFrame) -> dict[str, float]:
    """Coverage, CRPS and PIT on yearly means.

    Yearly rather than daily because that is the scale the series is used at, and
    because daily counts are integers whose Gaussian intervals would be scored
    against a discreteness the methods never claimed to model.
    """
    yearly = aligned.groupby("year").agg(
        truth=("truth", "mean"),
        estimate=("estimate", "mean"),
        sd=("sd", lambda block: float(np.sqrt(np.nanmean(np.square(block)) / max(len(block), 1)))),
    )
    out: dict[str, float] = {"n_years": int(len(yearly))}

    valid = yearly[np.isfinite(yearly["sd"]) & (yearly["sd"] > 0)]
    if valid.empty:
        for level in COVERAGE_LEVELS:
            out[f"coverage_{int(level * 100)}"] = np.nan
        out |= {"crps_yearly": np.nan, "pit_ks": np.nan, "interval_width_90": np.nan}
        return out

    truth = valid["truth"].to_numpy()
    mean = valid["estimate"].to_numpy()
    sd = valid["sd"].to_numpy()

    for level in COVERAGE_LEVELS:
        half = norm.ppf(0.5 + level / 2) * sd
        inside = np.abs(truth - mean) <= half
        out[f"coverage_{int(level * 100)}"] = float(inside.mean())

    out["interval_width_90"] = float(np.mean(2 * norm.ppf(0.95) * sd))
    out["crps_yearly"] = float(np.nanmean(crps_gaussian(truth, mean, sd)))
    pit = norm.cdf((truth - mean) / sd)
    out["pit_ks"] = float(kstest(pit, "uniform").statistic)
    return out


def acuity_recovery(result: CalibrationResult, parameters: pd.DataFrame) -> dict[str, float]:
    """How well a method recovered each observer's true acuity threshold.

    Only UEA21 and GRP estimate an acuity at all; the scalings report NaN, which
    the score table shows rather than hides.  That absence is informative: a
    method that never estimates an observer's acuity cannot report how uncertain
    it is about one.
    """
    if result.observers.empty or "recovered_threshold_msh" not in result.observers:
        return {"n_acuity": 0, "acuity_bias": np.nan, "acuity_rmse": np.nan}
    merged = result.observers.merge(
        parameters[["station", "base_threshold_msh", "kind"]], on="station", how="inner"
    )
    merged = merged[np.isfinite(merged["recovered_threshold_msh"])]
    merged = merged[merged["kind"] != "reference"]
    if merged.empty:
        return {"n_acuity": 0, "acuity_bias": np.nan, "acuity_rmse": np.nan}
    error = merged["recovered_threshold_msh"] - merged["base_threshold_msh"]
    return {
        "n_acuity": int(len(merged)),
        "acuity_bias": float(error.mean()),
        "acuity_rmse": float(np.sqrt((error**2).mean())),
    }


def break_detection_skill(
    detections: pd.DataFrame | None, truth_breaks: pd.DataFrame, *, tolerance_days: int = 365
) -> dict[str, float]:
    """Contingency skill for break detection, HOME-style.

    None of the seven ported baselines detects breaks, so this returns NaN for
    all of them -- which is the point.  Break attribution is one of the things a
    joint model adds, and the benchmark has to be able to score it when Steps 3-4
    produce a method that tries.
    """
    if detections is None or detections.empty:
        return {"break_pod": np.nan, "break_far": np.nan, "break_heidke": np.nan}

    hits = 0
    for station, day in zip(truth_breaks["station"], truth_breaks["date_ord"], strict=True):
        near = detections[
            (detections["station"] == station)
            & (np.abs(detections["date_ord"] - day) <= tolerance_days)
        ]
        hits += int(not near.empty)
    misses = len(truth_breaks) - hits
    false_alarms = max(len(detections) - hits, 0)
    pod = hits / max(hits + misses, 1)
    far = false_alarms / max(hits + false_alarms, 1)
    denominator = (hits + misses) * (misses + 0) + (hits + false_alarms) * (false_alarms + 0)
    heidke = (2 * (hits * 0 - misses * false_alarms) / denominator) if denominator else np.nan
    return {"break_pod": float(pod), "break_far": float(far), "break_heidke": float(heidke)}


def score(
    truth: Truth,
    result: CalibrationResult,
    parameters: pd.DataFrame,
    *,
    scenario: str = "",
    seed: int = 0,
) -> dict:
    """Every metric for one (scenario, realization, method) cell."""
    aligned = align(truth, result)
    row: dict = {
        "scenario": scenario,
        "seed": seed,
        "method": result.name,
        "anchoring": truth.anchoring,
        "n_days_scored": int(len(aligned)),
    }
    if aligned.empty:
        return row
    row |= scale_errors(aligned)
    row |= cycle_errors(aligned)
    row |= trend_errors(aligned)
    row |= maunder_level(aligned)
    row |= uncertainty_scores(aligned)
    row |= acuity_recovery(result, parameters)
    return row


def _spread(values: pd.Series, total: int, prefix: str) -> dict[str, float]:
    """Median, quartiles and full range of the finite values, with their count."""
    finite = values[np.isfinite(values)]
    if prefix == "scale":
        finite = finite[finite > 0]
    if finite.empty:
        return {
            f"{prefix}_n_finite": 0,
            f"{prefix}_n_replicates": total,
            f"{prefix}_median": np.nan,
            f"{prefix}_q1": np.nan,
            f"{prefix}_q3": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_n_finite": int(len(finite)),
        f"{prefix}_n_replicates": total,
        f"{prefix}_median": float(finite.median()),
        f"{prefix}_q1": float(finite.quantile(0.25)),
        f"{prefix}_q3": float(finite.quantile(0.75)),
        f"{prefix}_min": float(finite.min()),
        f"{prefix}_max": float(finite.max()),
    }


def _off_by(ratio: float) -> float:
    """How far a scale factor sits from the truth, as a factor in either direction."""
    if not np.isfinite(ratio) or ratio <= 0:
        return np.nan
    return float(max(ratio, 1.0 / ratio))


def summarise_case_studies(cases: pd.DataFrame) -> pd.DataFrame:
    """One row per case-study observer and method, across all replicates.

    ``case_study_observers.csv`` holds one row per (scenario, seed, observer,
    method) -- fifty replicates behind every point `f25`/`f26` draw.  Reading a
    single one of them would put an arbitrary draw in the figure and let a claim
    about typical behaviour rest on it, so the figure and the committed table are
    both built here: median, interquartile range, full range, and how many
    replicates produced a finite answer at all.

    ``scale_off_by_median`` and ``scale_off_by_worst`` are the typical and
    worst-case distance from the truth, as a factor in whichever direction the
    method erred, so a "this method is off by more than 2x" count is checkable
    from the table without re-deriving it.

    Parameters
    ----------
    cases
        Per-replicate rows, as ``score_benchmark.py`` writes them.

    Returns
    -------
    pandas.DataFrame
        One row per ``(kind, method)``, ordered by case-study kind.
    """
    rows = []
    for (kind, method), block in cases.groupby(["kind", "method"], sort=False):
        true_acuity = block["true_threshold_msh"].dropna().unique()
        row = {
            "kind": kind,
            "method": method,
            "true_threshold_msh": float(true_acuity[0]) if len(true_acuity) else np.nan,
        }
        row |= _spread(block["recovered_threshold_msh"], len(block), "acuity")
        row |= _spread(block["factor_ratio"], len(block), "scale")
        row["scale_off_by_median"] = _off_by(row["scale_median"])
        worst = [_off_by(value) for value in (row["scale_min"], row["scale_max"])]
        row["scale_off_by_worst"] = float(np.nanmax(worst)) if np.isfinite(worst).any() else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    order = {kind: index for index, kind in enumerate(CASE_STUDY_KINDS)}
    summary["_kind"] = summary["kind"].map(lambda k: order.get(k, len(order)))
    return summary.sort_values(["_kind", "method"]).drop(columns="_kind").reset_index(drop=True)
