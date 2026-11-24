"""Activity level from group count, so detection can be activity-conditioned before 1874.

The detection ratio ``k_o(t)`` depends on the activity level, because the group-area
law moves with the cycle (ADR-011, Step 1 §2.2) and a fixed acuity threshold
therefore removes a different fraction of groups at solar minimum than at maximum.
Step 2 measured what ignoring this costs: the published acuity methods' recovered
threshold tracks the era's own activity at Spearman **-0.62** and **-0.73**, which
is "not a per-observer error, it is a missing covariate".

But activity level is computed from *areas*, and areas exist only from 1825.  For
an observer in 1750 there is no activity level to condition on.  This module
supplies one, from the quantity the network is estimating anyway.

The map
-------
Measured on the anchor survey's own span, where the reference-equivalent count and
the activity level both exist, both smoothed on the same 182-day kernel::

    AL = a * GN ** b

It is a very tight relation and it is measured twice, independently, a century
apart:

Both coefficients are measured per anchoring and recorded in the run's own
metadata; the fit is tight (``r > 0.99`` in log-log, residual sd under 0.12) and
the two anchors agree on the exponent, which is what licenses using the map
outside the era it was fitted on.

Two things make it usable where it has to be used.  It passes through the origin,
so ``GN -> 0`` gives ``AL -> 0``, which is the only sane behaviour in a grand
minimum and is not something a polynomial or a spline would give for free.  And
``b`` slightly above 1 is physically the right sign: a more active Sun has not
only more groups but bigger ones, so area grows a little faster than count.

**The circularity, and why it is not one.**  ``AL`` is needed to compute ``k_o``,
which is needed to estimate ``GN``, which is the input to this map.  That is a
fixed point, solved by iterating (``network.calibrate`` runs 3-5 passes), not a
logical circle -- the same structure Step 4 formalizes with a latent activity
state.  The research notes are explicit that this belongs inside one likelihood
rather than in a two-stage pre-calibration: "A fingerprint has to condition on an
activity estimate that comes from other observers. **Embedding it in one joint
likelihood avoids the two-stage loop**, which is the main argument against running
ADF as a separate pre-calibration step."

**Firewall (ADR-002).**  The inputs are the reference survey's own areas and group
counts, and the observers' own reported counts.  No SILSO, no isotope, no aurora,
no published series.  Cycle phase never enters: the map is a function of the
count on a day, not of where in a cycle that day sits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gn_next_gen.io.cache import cached_frame
from gn_next_gen.io.reference import daily_counts, load_day_status
from gn_next_gen.paths import AREA_LAW_POSTERIOR, REFERENCE_DAY_STATUS
from gn_next_gen.reference.area_law import (
    ACTIVITY_SIGMA_DAYS,
    activity_level,
    gaussian_smooth,
)

#: Smoothed counts below this are dropped from the fit.  The relation is fitted in
#: log space and a 182-day smoothed count can sit arbitrarily close to zero at the
#: ends of a survey's span, where the smoother is running out of data rather than
#: the Sun running out of groups.
MIN_SMOOTHED_COUNT = 0.02


@dataclass(frozen=True)
class ActivityMap:
    """``AL = amplitude * GN ** exponent``, fitted on one survey's own span.

    Attributes
    ----------
    amplitude, exponent
        The power law, in mHem per (group/day) ** exponent.
    r_log
        Pearson correlation in log-log, the fit quality.
    residual_sd
        Scatter in log activity about the fit -- the honest uncertainty on an
        activity level inferred this way, carried so a run can report it.
    gn_lo, gn_hi
        Smoothed-count range the fit was measured over.  Outside it the map is an
        extrapolation, and :meth:`extrapolating` says so.
    """

    survey: str
    amplitude: float
    exponent: float
    r_log: float
    residual_sd: float
    gn_lo: float
    gn_hi: float
    n_days: int
    threshold_msh: float
    meta: dict = field(default_factory=dict)

    def __call__(self, group_number) -> np.ndarray:
        """Activity level in mHem for a (smoothed) reference-equivalent count."""
        counts = np.maximum(np.asarray(group_number, dtype=float), 0.0)
        return self.amplitude * np.power(counts, self.exponent)

    def extrapolating(self, group_number) -> np.ndarray:
        """Where a count falls outside the range the map was measured on."""
        counts = np.asarray(group_number, dtype=float)
        return (counts < self.gn_lo) | (counts > self.gn_hi)

    def as_row(self) -> dict:
        return {
            "survey": self.survey,
            "amplitude": self.amplitude,
            "exponent": self.exponent,
            "r_log": self.r_log,
            "residual_sd": self.residual_sd,
            "gn_lo": self.gn_lo,
            "gn_hi": self.gn_hi,
            "n_days": self.n_days,
            "threshold_msh": self.threshold_msh,
        }


def smooth_counts(
    date_ord: np.ndarray,
    counts: np.ndarray,
    *,
    sigma_days: float = ACTIVITY_SIGMA_DAYS,
    first: int | None = None,
    last: int | None = None,
) -> pd.Series:
    """Put counts on a daily grid and smooth them like the activity level is.

    Unobserved days are NaN rather than zero, and :func:`gaussian_smooth` is
    NaN-aware, so a gap in the record widens the effective kernel instead of
    being read as a run of spotless days.  Using the *same* kernel on both sides
    is what makes the two series comparable at all.
    """
    date_ord = np.asarray(date_ord, dtype=np.int64)
    counts = np.asarray(counts, dtype=float)
    lo = int(date_ord.min()) if first is None else int(first)
    hi = int(date_ord.max()) if last is None else int(last)
    grid = np.arange(lo, hi + 1)
    dense = np.full(grid.shape, np.nan)
    inside = (date_ord >= lo) & (date_ord <= hi)
    dense[date_ord[inside] - lo] = counts[inside]
    return pd.Series(gaussian_smooth(dense, sigma_days), index=grid, name="gn_smooth")


def fit_activity_map(
    survey: str,
    threshold_msh: float,
    *,
    sigma_days: float = ACTIVITY_SIGMA_DAYS,
    use_cache: bool = True,
    force: bool = False,
) -> ActivityMap:
    """Fit ``AL = a GN^b`` on a reference survey's own span."""
    frame = cached_frame(
        "activity_map",
        [REFERENCE_DAY_STATUS, AREA_LAW_POSTERIOR],
        {
            "survey": survey,
            "threshold_msh": round(float(threshold_msh), 6),
            "sigma_days": float(sigma_days),
            "weighting": "instantaneous",
        },
        lambda: _fit_frame(survey, float(threshold_msh), float(sigma_days)),
        use_cache=use_cache,
        force=force,
    )
    row = frame.iloc[0]
    return ActivityMap(
        survey=str(row["survey"]),
        amplitude=float(row["amplitude"]),
        exponent=float(row["exponent"]),
        r_log=float(row["r_log"]),
        residual_sd=float(row["residual_sd"]),
        gn_lo=float(row["gn_lo"]),
        gn_hi=float(row["gn_hi"]),
        n_days=int(row["n_days"]),
        threshold_msh=float(row["threshold_msh"]),
    )


def _fit_frame(survey: str, threshold_msh: float, sigma_days: float) -> pd.DataFrame:
    counts = daily_counts(threshold_msh, [survey])
    day_status = load_day_status([survey])
    # Instantaneous weighting, to match the activity axis of the detection grid
    # (``emissions.Catalogue``): every group present on the day, which is what a
    # per-group-day detection integral sees and the only weighting a benchmark
    # world's open ``reference_areas`` can supply.  The two weightings differ by
    # about six, so mixing them would silently misplace every observer on the
    # grid's activity axis.
    activity = activity_level(day_status, weighting="instantaneous", sigma_days=sigma_days)

    smoothed = smooth_counts(
        counts["date_ord"].to_numpy(),
        counts["n_groups"].to_numpy(),
        sigma_days=sigma_days,
    )
    joined = pd.DataFrame({"gn": smoothed}).join(activity.rename("al"), how="inner")
    usable = joined[
        np.isfinite(joined["gn"])
        & np.isfinite(joined["al"])
        & (joined["gn"] > MIN_SMOOTHED_COUNT)
        & (joined["al"] > 0)
    ]
    if len(usable) < 100:
        raise ValueError(f"{survey}: too few usable days to fit an activity map")

    log_gn = np.log(usable["gn"].to_numpy())
    log_al = np.log(usable["al"].to_numpy())
    exponent, log_amplitude = np.polyfit(log_gn, log_al, 1)
    residual = log_al - (log_amplitude + exponent * log_gn)
    return pd.DataFrame(
        {
            "survey": [survey],
            "amplitude": [float(np.exp(log_amplitude))],
            "exponent": [float(exponent)],
            "r_log": [float(np.corrcoef(log_gn, log_al)[0, 1])],
            "residual_sd": [float(np.std(residual))],
            "gn_lo": [float(usable["gn"].min())],
            "gn_hi": [float(usable["gn"].max())],
            "n_days": [int(len(usable))],
            "threshold_msh": [threshold_msh],
        }
    )
