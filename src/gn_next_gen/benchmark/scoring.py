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
  50/68/90/95%, aggregated by :func:`_yearly_sd`, which keeps the correlated and
  day-level parts of a method's uncertainty apart -- the single choice that decides
  whether the coverage figure is a measurement or an artifact;
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

#: The record cut into eras, as ``(label, first_year, last_year)`` inclusive.
#:
#: Pooled scores say which method is best overall and hide which method is best
#: *when*.  These nine windows are the grain the era tables and ``f41``'s winner
#: ribbon use, and every boundary is drawn from something in this repository
#: rather than from solar history -- the truth here is a replay of RGO cycles, so
#: only the observing *calendars* are real, and an era named for a real observer
#: would claim more than the benchmark can support.
ERAS: tuple[tuple[str, int, int], ...] = (
    # Almost no observers survive the 50-day minimum on the transplanted calendars.
    ("pre-network", 1610, 1644),
    # MAUNDER_YEARS, and where S3 inserts its grand minimum.
    ("maunder-window", 1645, 1715),
    # The overlap graph has zero edges across 1727-1745 (f28), so no chain crosses it.
    ("first-disconnection", 1716, 1748),
    # The second cut with zero edges, 1777-1798 (f28).
    ("second-disconnection", 1749, 1798),
    # Ends where S1's modern sub-network begins.
    ("pre-modern", 1799, 1873),
    # Dense network, but no overlap with the anchor's own epoch yet.
    ("pre-anchor", 1874, 1911),
    # RGO cycles 15-19 -- the anchoring itself (ADR-009, ``rgo_15_19``).
    ("anchor-epoch", 1912, 1969),
    # To the last year the v1.3 observation database covers.
    ("post-anchor", 1970, 2010),
    # Only gn_observer_extension_2026-10 reports here; v1.3 has ended.
    ("extension-only", 2011, 2026),
)

#: Era labels in record order, for a categorical axis.
ERA_ORDER: tuple[str, ...] = tuple(label for label, _, _ in ERAS)

#: The scenarios the era pooling runs over.  S1 begins in 1874 and holds 20 of the
#: 51 worlds, so pooling it with the rest would step the curve at 1874 from the
#: pool changing rather than from any method's behaviour; S5 switches detection
#: error off entirely, which flatters an exact method (the reason
#: ``COVERAGE_EXCLUDED_SCENARIOS`` drops it from f20).  Both are scored and both
#: appear in the era table as their own rows -- they are excluded from the pooled
#: *curve*, not from the benchmark.
POOLED_SCENARIOS: tuple[str, ...] = ("S2", "S3", "S4")


def era_of(year: int | np.ndarray) -> str | np.ndarray:
    """The era a year falls in, or ``"outside"`` beyond :data:`ERAS`."""
    years = np.asarray(year)
    out = np.full(years.shape, "outside", dtype=object)
    for label, first, last in ERAS:
        out[(years >= first) & (years <= last)] = label
    return out if out.ndim else str(out.item())


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
    # The decomposition, when the method supplies it.  A series written before the
    # columns existed falls back to the old independent-days rule, which
    # `uncertainty_scores` states in its own output as `sd_decomposed = 0`.
    for column in ("gn_sd_factor", "gn_sd_spread"):
        if column in frame.columns:
            out[column.removeprefix("gn_")] = frame.loc[common, column].to_numpy(dtype=float)
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


def _yearly_sd(block: pd.DataFrame) -> float:
    r"""The sd of a year's *mean*, respecting which part of the error averages down.

    A year's daily errors are not independent.  The dominant term in a chained
    method's uncertainty is the observer's calibration factor, which is the same on
    every day of that observer's career: averaging 300 days of it removes none of
    it.  Pooling the whole daily ``sd`` as ``sqrt(mean(sd^2) / n_days)`` -- which is
    what this scorer did until 2026-11-24 -- therefore shrinks a yearly interval by
    up to ``sqrt(365) ~ 19`` and reports an overconfidence the method never claimed.

    So the two parts are pooled differently.  The correlated part carries at its
    mean, because a constant error survives averaging unchanged; the day-level part
    divides by ``sqrt(n_days)``, because that part genuinely is independent:

    .. math::

        \sigma_{\rm year} = \sqrt{\langle\sigma_{\rm factor}\rangle^2
                                  + \langle\sigma_{\rm spread}^2\rangle / n}

    A method that supplies no decomposition keeps the old rule, so its number stays
    comparable with what was published, and ``sd_decomposed`` records which of the
    two was used.
    """
    n = max(len(block), 1)
    if not np.isfinite(block["sd"]).any() and not (
        "sd_factor" in block and np.isfinite(block["sd_factor"]).any()
    ):
        # A method that reports no uncertainty at all -- the naive floors, by
        # design.  NaN keeps it out of the coverage table instead of warning.
        return float("nan")
    if "sd_factor" in block and np.isfinite(block["sd_factor"]).any():
        factor = float(np.nanmean(block["sd_factor"]))
        spread = float(np.nanmean(np.square(block["sd_spread"])))
        return float(np.sqrt(factor**2 + spread / n))
    return float(np.sqrt(np.nanmean(np.square(block["sd"])) / n))


def yearly_frame(aligned: pd.DataFrame) -> pd.DataFrame:
    """One row per year: the means, the day count, and the sd of the year's mean.

    Both the pooled uncertainty scores and the per-era tables are built from this,
    deliberately.  They were two code paths once, and they drifted: the era-stratified
    coverage reported as finding F5 on 2026-11-22 was computed by hand and never
    recomputed after the 2026-11-24 fix to :func:`_yearly_sd`, so it went on quoting
    pre-fix numbers that the pooled table contradicted.  One frame, one aggregation
    rule, and the two cannot disagree again.
    """
    columns = ["sd"] + [c for c in ("sd_factor", "sd_spread") if c in aligned.columns]
    grouped = aligned.groupby("year")
    yearly = grouped.agg(
        truth=("truth", "mean"),
        estimate=("estimate", "mean"),
        n_days=("truth", "size"),
    )
    yearly["sd"] = grouped[columns].apply(_yearly_sd)
    return yearly


def uncertainty_scores(aligned: pd.DataFrame) -> dict[str, float]:
    """Coverage, CRPS and PIT on yearly means.

    Yearly rather than daily because that is the scale the series is used at, and
    because daily counts are integers whose Gaussian intervals would be scored
    against a discreteness the methods never claimed to model.  :func:`_yearly_sd`
    is where the aggregation lives, and it is the load-bearing choice in this
    module.
    """
    decomposed = "sd_factor" in aligned.columns and bool(np.isfinite(aligned["sd_factor"]).any())
    yearly = yearly_frame(aligned)
    out: dict[str, float] = {"n_years": int(len(yearly)), "sd_decomposed": int(decomposed)}

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


def yearly_errors(
    aligned: pd.DataFrame, *, scenario: str = "", seed: int = 0, method: str = ""
) -> pd.DataFrame:
    """The per-year residual of one cell, with its era and its interval flags.

    :func:`score` reduces a cell to a single row, which is what the pooled tables
    need and is also why nothing in Step 2 could say *when* a method is wrong.  This
    keeps the year axis.  ``within_50`` and friends are the same central-interval
    tests :func:`uncertainty_scores` applies, on the same :func:`yearly_frame`, so an
    era's coverage and the pooled coverage are the same measurement at two grains.
    """
    yearly = yearly_frame(aligned)
    out = pd.DataFrame(
        {
            "scenario": scenario,
            "seed": seed,
            "method": method,
            "year": yearly.index.to_numpy(dtype=int),
            "era": era_of(yearly.index.to_numpy(dtype=int)),
            "n_days": yearly["n_days"].to_numpy(dtype=int),
            "truth": yearly["truth"].to_numpy(dtype=float),
            "estimate": yearly["estimate"].to_numpy(dtype=float),
            "sd": yearly["sd"].to_numpy(dtype=float),
        }
    )
    out["error"] = out["estimate"] - out["truth"]
    usable = np.isfinite(out["sd"]) & (out["sd"] > 0)
    for level in COVERAGE_LEVELS:
        half = norm.ppf(0.5 + level / 2) * out["sd"].to_numpy()
        inside = np.abs(out["error"].to_numpy()) <= half
        out[f"within_{int(level * 100)}"] = np.where(usable, inside, np.nan)
    return out


def _error_moments(block: pd.DataFrame) -> pd.Series:
    """Mean squared error, bias and level over whatever rows are handed in."""
    error = block["error"].to_numpy(dtype=float)
    finite = np.isfinite(error)
    if not finite.any():
        return pd.Series({"mse": np.nan, "bias": np.nan, "truth": np.nan, "n_years": 0})
    return pd.Series(
        {
            "mse": float(np.mean(np.square(error[finite]))),
            "bias": float(np.mean(error[finite])),
            "truth": float(np.nanmean(block["truth"].to_numpy(dtype=float)[finite])),
            "n_years": int(finite.sum()),
        }
    )


def _pool_scenarios(per_scenario: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Average per-scenario moments with equal weight per scenario.

    Equal weight, not equal weight per *world*: S1 holds 20 of the 51 worlds and S5
    holds one, so pooling rows would let one scenario's observer network decide the
    answer.  Squared error is averaged and then rooted, so the result is the RMSE of
    a scenario-balanced pool rather than the mean of three RMSEs.
    """
    grouped = per_scenario.groupby(keys, dropna=False)
    out = grouped.agg(
        mse=("mse", "mean"),
        bias=("bias", "mean"),
        truth=("truth", "mean"),
        n_scenarios=("scenario", "nunique"),
        n_worlds=("n_worlds", "sum"),
        n_years=("n_years", "sum"),
    ).reset_index()
    out["rmse"] = np.sqrt(out["mse"])
    return out


def pool_yearly(yearly: pd.DataFrame, *, scenarios: tuple[str, ...] = POOLED_SCENARIOS):
    """Per method and calendar year, the scenario-balanced RMSE, bias and level.

    One row per ``(method, year)``.  A method that emitted no day in a year has no
    row, so a curve drawn from this stops where the method's reach stops instead of
    reading as zero error -- which is :func:`score`'s reach result made continuous.
    """
    frame = yearly[yearly["scenario"].isin(scenarios)]
    if frame.empty:
        return pd.DataFrame(
            columns=["method", "year", "mse", "rmse", "bias", "truth", "n_scenarios", "n_worlds"]
        )
    per_scenario = (
        frame.groupby(["scenario", "method", "year"], dropna=False)
        .apply(_error_moments, include_groups=False)
        .reset_index()
    )
    counts = (
        frame.groupby(["scenario", "method", "year"], dropna=False)["seed"]
        .nunique()
        .rename("n_worlds")
        .reset_index()
    )
    per_scenario = per_scenario.merge(counts, on=["scenario", "method", "year"])
    return _pool_scenarios(per_scenario, ["method", "year"]).sort_values(["method", "year"])


def era_errors(yearly: pd.DataFrame, *, scenarios: tuple[str, ...] = POOLED_SCENARIOS):
    """Error, coverage and reach per method and era, per scenario and pooled.

    Every scenario gets its own rows and ``scenario = "pooled"`` carries the
    scenario-balanced average over ``scenarios``.  Keeping S1 and S5 as their own
    rows rather than dropping them is what makes the pooling checkable: if S1's
    ranking in an era disagrees with the pooled one, the table says so.

    ``reached`` is the share of the scenario's worlds in which the method emitted
    any day in that era -- the quantity that turns f41's gaps into a number, and
    the reason a 0.0 RMSE cell is never mistaken for a perfect one.
    """
    worlds = yearly.groupby("scenario")["seed"].nunique()

    moments = (
        yearly.groupby(["scenario", "method", "era"], dropna=False)
        .apply(_error_moments, include_groups=False)
        .reset_index()
    )
    extra = yearly.groupby(["scenario", "method", "era"], dropna=False).agg(
        n_worlds=("seed", "nunique"),
        **{
            f"coverage_{int(level * 100)}": (f"within_{int(level * 100)}", "mean")
            for level in COVERAGE_LEVELS
        },
    )
    per_scenario = moments.merge(extra.reset_index(), on=["scenario", "method", "era"])
    per_scenario["reached"] = per_scenario["n_worlds"] / per_scenario["scenario"].map(worlds)
    per_scenario["rmse"] = np.sqrt(per_scenario["mse"])

    pooled = _pool_scenarios(
        per_scenario[per_scenario["scenario"].isin(scenarios)], ["method", "era"]
    )
    for level in COVERAGE_LEVELS:
        key = f"coverage_{int(level * 100)}"
        pooled = pooled.merge(
            per_scenario[per_scenario["scenario"].isin(scenarios)]
            .groupby(["method", "era"], dropna=False)[key]
            .mean()
            .reset_index(),
            on=["method", "era"],
        )
    pooled = pooled.merge(
        per_scenario[per_scenario["scenario"].isin(scenarios)]
        .groupby(["method", "era"], dropna=False)["reached"]
        .mean()
        .reset_index(),
        on=["method", "era"],
    )
    pooled["scenario"] = "pooled"

    out = pd.concat([per_scenario, pooled], ignore_index=True)
    out["era"] = pd.Categorical(out["era"], categories=[*ERA_ORDER, "outside"], ordered=True)
    return out.sort_values(["scenario", "method", "era"]).reset_index(drop=True)


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
    aligned: pd.DataFrame | None = None,
) -> dict:
    """Every metric for one (scenario, realization, method) cell.

    ``aligned`` lets a caller that also wants the per-year table (:func:`yearly_errors`)
    hand in the frame it already built, so the join runs once per cell rather than twice.
    """
    if aligned is None:
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
