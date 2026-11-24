"""Four floors every real method must clear.

The homogenization literature is blunt about why a benchmark needs them.  Lindau
& Venema (2018) found that at realistic signal-to-noise **a random segmentation
explains about half the break variance**, so "explains half the variance" is not
evidence of skill.  And the SBI calibration literature notes that a coverage test
which cannot reject an estimator that simply returns the prior is not testing
anything.

There are two kinds of floor here and both are needed, because they answer
different questions.  ``null-prior`` and ``null-segmentation`` keep the *form* of
the methods under test -- segment a career, scale each segment -- with the content
randomised, so losing to them says a method's breakpoints and factors carried no
information.  ``naive-mean`` and ``naive-median`` do not calibrate at all, so losing
to *them* says something stronger: the calibration made the series worse than
leaving the counts alone.  The second kind is the harder floor, because a random
segmentation's factors average out over hundreds of observers and a year of days --
it is the naive mean plus noise.

None of the four is a calibration method and none is ranked with the seven.  They
sit beside them in the score table so a reader can see where zero is.

The naive two are **shape comparators only**.  They have no anchoring -- nothing
carries the reference observer's scale to anyone -- so their bias is not
interpretable, and they report no uncertainty, which keeps them out of the coverage,
CRPS and PIT columns by construction rather than by a rule in the scorer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gn_next_gen.baselines.base import CalibrationResult, station_series


def calibrate_prior_null(
    observations: pd.DataFrame, *, reference=None, seed: int = 0, **_kwargs
) -> CalibrationResult:
    """Return the same wide interval every day, ignoring the data entirely.

    Centred on the network's own grand mean reported count with a deliberately
    generous spread.  A coverage test that this passes and a real method fails is
    measuring interval width, not calibration -- which is exactly the failure mode
    TARP-style reasoning warns about.
    """
    del seed
    mean = float(observations["groups"].mean())
    spread = float(observations["groups"].std(ddof=1))
    days = np.unique(observations["date_ord"].to_numpy())
    return CalibrationResult(
        series=pd.DataFrame(
            {
                "date_ord": days,
                "gn": np.full(len(days), mean),
                "gn_sd": np.full(len(days), spread),
                # One number for the whole record, so its error is the same error
                # every day: entirely correlated, nothing to average down.
                "gn_sd_factor": np.full(len(days), spread),
                "gn_sd_spread": np.zeros(len(days)),
                "n_observers": 0,
            }
        ),
        observers=pd.DataFrame(
            columns=["station", "factor", "factor_sd", "recovered_threshold_msh", "reached"]
        ),
        name="null-prior",
        meta={"mean": mean, "sd": spread},
    )


def calibrate_random_segmentation(
    observations: pd.DataFrame,
    *,
    reference: int | None = None,
    seed: int = 0,
    n_segments: int = 12,
    **_kwargs,
) -> CalibrationResult:
    """Scale each observer by a random factor drawn per random career segment.

    The Lindau & Venema floor: cut careers at arbitrary points, apply arbitrary
    scalings, and see how much of the answer falls out anyway.  Anything a real
    method does beyond this is its actual contribution.
    """
    rng = np.random.default_rng(seed)
    del reference
    series = station_series(observations)
    target = float(observations["groups"].mean())

    rows = []
    for _station, (days, counts) in series.items():
        if len(days) == 0:
            continue
        cuts = np.sort(rng.choice(len(days), size=min(n_segments, len(days)), replace=False))
        segment = np.searchsorted(cuts, np.arange(len(days)), side="right")
        factors = np.exp(rng.normal(0.0, 0.3, size=segment.max() + 1))
        scaled = counts * factors[segment]
        rows.append(pd.DataFrame({"date_ord": days, "value": scaled}))

    if not rows:
        return CalibrationResult(
            pd.DataFrame(columns=["date_ord", "gn", "gn_sd", "n_observers"]),
            pd.DataFrame(),
            "null-segmentation",
            {},
        )

    stacked = pd.concat(rows, ignore_index=True)
    grouped = stacked.groupby("date_ord")["value"]
    mean = grouped.mean()
    # Rescale so the series sits at the network's own level: a random
    # segmentation should be judged on shape, not on an arbitrary offset.
    scale = target / max(float(mean.mean()), 1e-9)
    return CalibrationResult(
        series=pd.DataFrame(
            {
                "date_ord": mean.index.to_numpy(),
                "gn": mean.to_numpy() * scale,
                "gn_sd": grouped.std(ddof=1).fillna(0.0).to_numpy() * scale,
                # The interval is the day's between-observer scatter and nothing
                # else, so all of it averages down.  That this null is *also* badly
                # calibrated is the caveat f20 carries: wrongly centred, not
                # overconfident.
                "gn_sd_factor": np.zeros(len(mean)),
                "gn_sd_spread": grouped.std(ddof=1).fillna(0.0).to_numpy() * scale,
                "n_observers": grouped.size().to_numpy(),
            }
        ),
        observers=pd.DataFrame(
            columns=["station", "factor", "factor_sd", "recovered_threshold_msh", "reached"]
        ),
        name="null-segmentation",
        meta={"n_segments": n_segments},
    )


def _naive(observations: pd.DataFrame, how: str) -> CalibrationResult:
    """One statistic of the raw reported counts per day, and no calibration at all.

    The floor that turns out to be hard.  A chained method exists to turn an
    observer's 4 groups into the reference's 10; if it cannot beat the median of the
    uncompensated counts on centred RMSE, its factors inject more error than they
    remove.  Measured on S2, six of the seven published methods do not.

    No ``gn_sd``: this estimator makes no claim about its own uncertainty, and
    inventing one would put a meaningless row in the coverage table.
    """
    grouped = observations.groupby("date_ord")["groups"]
    value = grouped.median() if how == "median" else grouped.mean()
    return CalibrationResult(
        series=pd.DataFrame(
            {
                "date_ord": value.index.to_numpy(),
                "gn": value.to_numpy(dtype=float),
                "gn_sd": np.nan,
                "gn_sd_factor": np.nan,
                "gn_sd_spread": np.nan,
                "n_observers": grouped.size().to_numpy(),
            }
        ),
        observers=pd.DataFrame(
            columns=["station", "factor", "factor_sd", "recovered_threshold_msh", "reached"]
        ),
        name=f"naive-{how}",
        meta={"statistic": how, "calibrated": False},
    )


def calibrate_naive_mean(observations: pd.DataFrame, **_kwargs) -> CalibrationResult:
    """Daily mean of the raw reported counts."""
    return _naive(observations, "mean")


def calibrate_naive_median(observations: pd.DataFrame, **_kwargs) -> CalibrationResult:
    """Daily median of the raw reported counts."""
    return _naive(observations, "median")
