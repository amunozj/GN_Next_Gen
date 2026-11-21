"""SILSO sunspot number, **comparison only**.

SN V2.0 is itself a composite, so it may not enter calibration of the new model:
not as an activity level, not as a cycle phase, not as a prior (ADR-002, ADR-003).
Cycle phase is latent.  These loaders tag what they return with
``frame.attrs["role"] = "comparison_only"`` and
:func:`gn_next_gen.config.validate.validate_run_config` refuses a run that lists
SILSO as a calibration input, so the rule is enforced in two places rather than
remembered.
"""

from __future__ import annotations

import pandas as pd

from gn_next_gen.paths import SILSO

COMPARISON_ONLY = "comparison_only"

DAILY = SILSO / "SN_d_tot_V2.0.csv"
MONTHLY = SILSO / "SN_m_tot_V2.0.csv"


def _tag(frame: pd.DataFrame) -> pd.DataFrame:
    frame.attrs["role"] = COMPARISON_ONLY
    return frame


def load_silso_daily(path=DAILY) -> pd.DataFrame:
    """Daily SN: ``YEAR, MONTH, DAY, FRACYEAR, DAILYSN, DAILYSTD, NOBS, MARKER``.

    ``DAILYSN = -1`` marks a missing value and becomes NaN.  ``MARKER`` is 1 for
    definitive and 0 for provisional.
    """
    frame = pd.read_csv(path)
    for col in ("DAILYSN", "DAILYSTD"):
        frame[col] = frame[col].where(frame[col] >= 0)
    return _tag(frame)


def load_silso_monthly(path=MONTHLY) -> pd.DataFrame:
    """Monthly mean SN; ``MSTD`` and ``NOBS`` of -1 mark missing values."""
    frame = pd.read_csv(path)
    for col in ("MMEAN", "MSTD", "NOBS"):
        frame[col] = frame[col].where(frame[col] >= 0)
    return _tag(frame)
