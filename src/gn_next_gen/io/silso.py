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

DAILY_COLUMNS = ["YEAR", "MONTH", "DAY", "FRACYEAR", "DAILYSN", "DAILYSTD", "NOBS", "MARKER"]
MONTHLY_COLUMNS = ["YEAR", "MONTH", "FRACYEAR", "MMEAN", "MSTD", "NOBS", "MARKER"]


def _read(path, columns: list[str]) -> pd.DataFrame:
    """Read either SILSO layout.

    The 2026-11 refresh is SILSO's own download: semicolon-separated and headerless.
    The retained 2018 snapshot was converted to comma-separated with a header during
    the ADFSSN era. Both are byte-exact as received, so the loader sniffs instead of
    anyone having to remember which is which.
    """
    first = path.read_text().splitlines()[0]
    if ";" in first:
        frame = pd.read_csv(
            path,
            sep=";",
            header=None,
            usecols=range(len(columns)),
            names=columns,
            skipinitialspace=True,
        )
    else:
        frame = pd.read_csv(path)
    return frame


def _tag(frame: pd.DataFrame) -> pd.DataFrame:
    frame.attrs["role"] = COMPARISON_ONLY
    return frame


def load_silso_daily(path=DAILY) -> pd.DataFrame:
    """Daily SN: ``YEAR, MONTH, DAY, FRACYEAR, DAILYSN, DAILYSTD, NOBS, MARKER``.

    ``DAILYSN = -1`` marks a missing value and becomes NaN.  ``MARKER`` is 1 for
    definitive and 0 for provisional.
    """
    frame = _read(path, DAILY_COLUMNS)
    for col in ("DAILYSN", "DAILYSTD"):
        frame[col] = frame[col].where(frame[col] >= 0)
    return _tag(frame)


def load_silso_monthly(path=MONTHLY) -> pd.DataFrame:
    """Monthly mean SN; ``MSTD`` and ``NOBS`` of -1 mark missing values."""
    frame = _read(path, MONTHLY_COLUMNS)
    for col in ("MMEAN", "MSTD", "NOBS"):
        frame[col] = frame[col].where(frame[col] >= 0)
    return _tag(frame)
