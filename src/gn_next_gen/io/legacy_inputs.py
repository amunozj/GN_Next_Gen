"""The two superseded raw folders, kept only so Step 2 can run the legacy baselines.

ADR-009 retains ``reference_areas/`` and ``gn_database_v1.22/`` for exactly one
reason: the legacy methods were written against *those* files, and scoring a port
against the renovated inputs would differ for reasons that have nothing to do with
the port.  Nothing else in the package may read them -- the model-facing reference
is ``reference_groups_2026-10/`` (``io.reference``) and the model-facing observation
database is v1.3_4 (``io.observations``).

Two shapes differ from everything else in this package and are handled here rather
than at the call site:

* **The reference-area files are day x group matrices**, ``YEAR, MONTH, DAY,
  AREA1..AREA26``, where ``AREA1`` being NaN means *no observation that day* and an
  all-zero row means *observed and spotless*.  The current ``SC_SP_RG_DB_KM`` file
  **has no header** -- it was lost in ADFSSN commit ``5898ab8`` -- so reading it with
  ``header=0`` silently promotes the first data row to column names (**D1**).  The
  loader always supplies names and never trusts a header row.
* **v1.22 observations carry 15 leading ``%`` comment lines** before the column
  header, and CRLF endings.

Counting convention, ported from ``legacy/ADFSSN/scripts/SSN_ADF.py`` l. 837-841
(``np.nansum(np.greater(areas, Thr), axis=1)``): a group counts when its area is
**strictly greater** than the threshold, and a NaN ``AREA1`` makes the whole day
missing.  This is the shape-specific twin of ``io.reference.daily_counts``.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

from gn_next_gen.paths import GN_DATABASE_V122, REFERENCE_AREAS_LEGACY

#: The reference-area files, by the short name Step 2 configs use.
REFERENCE_AREA_FILES = {
    "sc_sp_rg_db_km": REFERENCE_AREAS_LEGACY / "SC_SP_RG_DB_KM_group_areas_by_day.csv",
    "sc_sp_rg_db_km_pre5898ab8": (
        REFERENCE_AREAS_LEGACY / "SC_SP_RG_DB_KM_group_areas_by_day_pre5898ab8.csv"
    ),
    "drgo": REFERENCE_AREAS_LEGACY / "DRGO_group_areas_by_day.csv",
}

#: The 26 area slots the legacy files carry.  Always addressed by name: the legacy
#: code slices them positionally and drops ``AREA26`` in ten places (**D9**).
AREA_COLUMNS = [f"AREA{i}" for i in range(1, 27)]
REFERENCE_AREA_COLUMNS = ["YEAR", "MONTH", "DAY", *AREA_COLUMNS]

#: v1.22's own columns, after its 15 ``%`` comment lines.
V122_OBSERVATION_COLUMNS = [
    "YEAR",
    "MONTH",
    "DAY",
    "ORDINAL",
    "STATION",
    "OBSERVER",
    "GROUPS",
    "FRACYEAR",
]


def _ordinal(frame: pd.DataFrame) -> np.ndarray:
    """Proleptic-Gregorian ordinals, the convention v1.22 and v1.3 share."""
    return np.array(
        [
            _dt.date(int(y), int(m), int(d)).toordinal()
            for y, m, d in zip(frame["YEAR"], frame["MONTH"], frame["DAY"], strict=True)
        ],
        dtype=np.int64,
    )


def load_reference_areas(name: str = "sc_sp_rg_db_km") -> pd.DataFrame:
    """One legacy reference-area file as ``date_ord`` plus ``AREA1..AREA26``.

    Parameters
    ----------
    name
        A key of :data:`REFERENCE_AREA_FILES`.

    Returns
    -------
    pandas.DataFrame
        Columns ``date_ord, YEAR, MONTH, DAY, AREA1..AREA26``, one row per calendar
        day the file carries, in file order.  Column names are supplied, never read
        from the file, so **D1** cannot bite.
    """
    if name not in REFERENCE_AREA_FILES:
        raise ValueError(
            f"unknown reference-area file {name!r}; expected {sorted(REFERENCE_AREA_FILES)}"
        )
    path = REFERENCE_AREA_FILES[name]
    first = path.open("r", encoding="utf-8", errors="replace").readline()
    has_header = first.lstrip().upper().startswith("YEAR")
    # header=None with an explicit skiprows: passing header=0 *and* skiprows=1
    # discards two rows, because pandas skips first and then consumes a header.
    frame = pd.read_csv(
        path,
        header=None,
        names=REFERENCE_AREA_COLUMNS,
        skiprows=1 if has_header else 0,
    )
    frame.insert(0, "date_ord", _ordinal(frame))
    return frame


def legacy_daily_counts(areas: pd.DataFrame, threshold_msh: float = 0.0) -> pd.DataFrame:
    """Daily group counts above a threshold, in the legacy day x group shape.

    A group counts when ``area > threshold_msh`` *strictly*; a day whose ``AREA1``
    is NaN is not an observation and its count is NaN.  Ported from
    ``legacy/ADFSSN/scripts/SSN_ADF.py`` l. 837-841, by name rather than by
    position so the inconsistent ``AREA26`` slicing (**D9**) cannot recur.

    Returns
    -------
    pandas.DataFrame
        ``date_ord, n_groups, observed`` -- ``n_groups`` NaN where unobserved.
    """
    values = areas[AREA_COLUMNS].to_numpy(dtype=float)
    counts = np.nansum(values > threshold_msh, axis=1).astype(float)
    observed = ~np.isnan(values[:, 0])
    counts[~observed] = np.nan
    return pd.DataFrame(
        {"date_ord": areas["date_ord"].to_numpy(), "n_groups": counts, "observed": observed}
    )


def load_v122_observations() -> pd.DataFrame:
    """The v1.22 observation database as ``date_ord, station, groups``.

    Sentinel rows (``STATION == 0``) are dropped, as in ``io.load_observations``.
    v1.22 carries no ``GROUPS == -1`` on any row with ``STATION > 0``, so that
    sentinel never appears here; the loader asserts it rather than assuming it.
    """
    # ``skiprows`` counts raw lines and is applied *before* ``comment``, so it
    # must not be combined with it: 15 '%' lines precede the real header, and
    # ``comment="%"`` already removes them, leaving the header for ``header=0``.
    frame = pd.read_csv(
        GN_DATABASE_V122 / "GNobservations_JV_V1.22.csv",
        comment="%",
        header=0,
    )
    if list(frame.columns) != V122_OBSERVATION_COLUMNS:
        raise AssertionError(f"unexpected v1.22 columns {list(frame.columns)}")
    frame = frame[frame["STATION"] > 0].copy()
    if (frame["GROUPS"] == -1).any():
        raise AssertionError("v1.22 carries GROUPS == -1 on an observing station; handling needed")
    out = pd.DataFrame(
        {
            "date_ord": frame["ORDINAL"].to_numpy(dtype=np.int64),
            "station": frame["STATION"].to_numpy(dtype=np.int64),
            "groups": frame["GROUPS"].to_numpy(dtype=np.int64),
        }
    )
    return out.reset_index(drop=True)


def load_v122_observers() -> pd.DataFrame:
    """The v1.22 observer list: ``STATION, INITIAL, FINAL, TOT_OBS, OBSERVER``."""
    frame = pd.read_csv(GN_DATABASE_V122 / "GNobservers_JV_V1.22.csv")
    return frame.rename(columns={"TOT.OBS": "TOT_OBS"})


#: Canonical ``OBSERVER`` patterns for the pathfinding endpoints, resolved by name
#: because the MATLAB scripts' V1-12 numbers do not transfer (see
#: :func:`v122_station_by_name`).  These are **regexes**, and several are anchored
#: or spelled out because v1.22 carries near-duplicates: ``STAUDACH`` also matches
#: ``STAUDACHER, J.C., NUREMBERG``, ``LOCARNO`` also matches ``RAPP,
#: LOCARNO-MONTI``, and ``SCHWABE, H., DESSAU`` is a prefix of
#: ``SCHWABE, H., DESSAU-ARLT``.  Resolved values under v1.22: Staudach 466,
#: Locarno 614, RGO 332 -- against the MATLAB scripts' V1-12 445 and 590, which
#: are Koyama and Janssens here.
PATHFINDING_ENDPOINTS = {
    "staudach": r"STAUDACH, J\.C\.",
    "locarno": "LOCARNO, SWITZERLAND",
    "rgo": "ROYAL GREENWICH",
    "schwabe": "SCHWABE, H., DESSAU$",
}


def v122_station_by_name(fragment: str) -> int:
    """The v1.22 station number whose ``OBSERVER`` matches ``fragment`` (a regex).

    The MATLAB pathfinding code hard-codes **V1-12** station numbers, and V1-12 is not
    in this repository.  Its ``stnd = 445`` / ``endnd = 590`` are Koyama and
    Janssens under v1.22; the intended nodes are 466 (Staudach) and 614 (Locarno).
    Every port therefore resolves an endpoint by name, never by a copied number.

    Raises
    ------
    LookupError
        If the fragment matches no station or more than one.
    """
    observers = load_v122_observers()
    hit = observers[observers["OBSERVER"].str.contains(fragment, case=False, na=False)]
    if len(hit) != 1:
        raise LookupError(
            f"{fragment!r} matches {len(hit)} v1.22 observers: {list(hit['OBSERVER'])[:5]}"
        )
    return int(hit["STATION"].iloc[0])
