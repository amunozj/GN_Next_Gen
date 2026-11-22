"""Real observing calendars, transplanted onto a synthetic Sun.

Who looked, and when, is not incidental: Hay et al. showed for tide gauges that the
sampling pattern alone moved a century-scale trend by about 30%, and the research
notes carry that across as the diagnostic this step owes
(``intercalibration_homogenization.md`` Q4).  A benchmark on idealised calendars
cannot show it.  Chatzistergos et al. (2025) reduce overlap by deleting days at
random; here every synthetic observer inherits a real station's actual observing
days, including the sparse Maunder-era patterns where the overlap graph is thinnest.

The transplant idea is the legacy missing-day imprinting of
``legacy/ADFSSN/scripts/SSN_ADF.py`` l. 867-871, ported as the reuse map asks:
the idea, not the shape -- long format throughout, never a dense day x observer
matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gn_next_gen.io import load_observations


def observer_calendars(
    first_ord: int,
    last_ord: int,
    *,
    release: str = "v1.3_4",
    include_extension: bool = True,
    kmas_role: str = "reference",
    min_days: int = 50,
    stations: list[int] | None = None,
) -> dict[int, np.ndarray]:
    """Real observing days per station, clipped to a window.

    Parameters
    ----------
    first_ord, last_ord
        The window to clip to.
    kmas_role
        Passed through to :func:`gn_next_gen.io.load_observations`, which withholds
        KMAS counts when it is ``"reference"`` -- the ADR-009 single-use rule, which
        a benchmark scenario must respect exactly as a real run does.
    min_days
        Stations with fewer observing days inside the window are dropped: they
        cannot support any calibration method and only inflate the station count.
    stations
        Restrict to these stations; ``None`` takes every station available.

    Returns
    -------
    dict
        ``{station: sorted array of date_ord}``.
    """
    observations = load_observations(
        release=release, include_extension=include_extension, kmas_role=kmas_role
    )
    inside = (observations["date_ord"] >= first_ord) & (observations["date_ord"] <= last_ord)
    observations = observations[inside]
    if stations is not None:
        observations = observations[observations["station"].isin(stations)]

    calendars: dict[int, np.ndarray] = {}
    for station, frame in observations.groupby("station"):
        days = np.unique(frame["date_ord"].to_numpy(dtype=np.int64))
        if len(days) >= min_days:
            calendars[int(station)] = days
    return calendars


def transplant_calendar(series: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Blank a simulated daily series wherever the real observer did not look.

    Promoted from ``scripts/check_a_dynamics.py``, where Step 1 used it to score
    simulated against observed counts on a like-for-like calendar.
    """
    out = np.asarray(series, dtype=float).copy()
    out[~np.asarray(observed, dtype=bool)] = np.nan
    return out


def calendar_summary(calendars: dict[int, np.ndarray]) -> pd.DataFrame:
    """Per-station first day, last day, and observing-day count."""
    return pd.DataFrame(
        {
            "station": list(calendars),
            "first_ord": [int(days[0]) for days in calendars.values()],
            "last_ord": [int(days[-1]) for days in calendars.values()],
            "n_days": [int(len(days)) for days in calendars.values()],
        }
    ).sort_values("first_ord", ignore_index=True)
