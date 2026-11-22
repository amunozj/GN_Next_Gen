"""Two floors every real method must clear.

The homogenization literature is blunt about why a benchmark needs them.  Lindau
& Venema (2018) found that at realistic signal-to-noise **a random segmentation
explains about half the break variance**, so "explains half the variance" is not
evidence of skill.  And the SBI calibration literature notes that a coverage test
which cannot reject an estimator that simply returns the prior is not testing
anything.

Neither of these is a calibration method and neither is ranked with the seven.
They sit beside them in the score table so a reader can see where zero is.
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
                "n_observers": grouped.size().to_numpy(),
            }
        ),
        observers=pd.DataFrame(
            columns=["station", "factor", "factor_sd", "recovered_threshold_msh", "reached"]
        ),
        name="null-segmentation",
        meta={"n_segments": n_segments},
    )
