"""Ten named observers, so the results doc can show *why* a method fails.

Aggregate scores say which baseline wins; they hide the observers that break one.
These ten are fixed by name and seed rather than sampled, each chosen to isolate a
failure mode, and each gets a panel in ``f25``/``f26`` showing truth, the raw
record and every baseline's calibrated series with true and recovered acuity.

The selection is by *calendar and graph position*, not by station number, so the
set survives a change of scenario window or of observation release.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from gn_next_gen.benchmark.observers import CASE_STUDY_KINDS, ObserverParams

#: Observing days a ``clean-modern`` candidate must have.  `acuity.py` measures
#: UEA21's recovery correlation rising from 0.27 over all observers to 0.61 for
#: those with 3,000+ observing days, so this is where the method's own statistic
#: starts to exist rather than an arbitrary round number.
MIN_CLEAN_MODERN_DAYS = 3_000


#: How each case study differs from a population draw.  Everything not named here
#: is left at the population value, so a panel isolates one effect at a time.
CASE_STUDY_OVERRIDES: dict[str, dict] = {
    "clean-modern": dict(
        base_threshold_msh=10.0,
        softness=0.3,
        seeing_sigma=0.10,
        p_split=0.0,
        p_merge=0.0,
        mnar_omega=0.0,
        drift_sigma=0.0,
    ),
    "sparse-maunder": dict(
        base_threshold_msh=60.0,
        softness=0.6,
        seeing_sigma=0.35,
        mnar_omega=0.5,
        drift_sigma=0.15,
    ),
    "break-undocumented": dict(
        base_threshold_msh=45.0,
        drift_sigma=0.0,
        mnar_omega=0.1,
    ),
    "break-documented": dict(
        base_threshold_msh=45.0,
        drift_sigma=0.0,
        mnar_omega=0.1,
    ),
    "slow-drift": dict(
        base_threshold_msh=35.0,
        drift_sigma=0.30,
        mnar_omega=0.1,
    ),
    "poor-instrument": dict(
        base_threshold_msh=100.0,
        softness=0.6,
        seeing_sigma=0.30,
        drift_sigma=0.05,
        mnar_omega=0.1,
    ),
    "leaf-node": dict(
        base_threshold_msh=40.0,
        drift_sigma=0.10,
        mnar_omega=0.2,
    ),
    "copier": dict(
        base_threshold_msh=30.0,
        drift_sigma=0.05,
        mnar_omega=0.1,
    ),
    "zero-omitter": dict(
        base_threshold_msh=30.0,
        drift_sigma=0.05,
        mnar_omega=0.8,
    ),
    "blanket-statements": dict(
        base_threshold_msh=50.0,
        drift_sigma=0.10,
        mnar_omega=0.3,
        blanket=True,
    ),
}


def _span_years(days: np.ndarray) -> float:
    return max((days[-1] - days[0]) / 365.25, 1.0)


def _pick_stations(
    calendars: dict[int, np.ndarray], degrees: dict[int, int] | None
) -> dict[str, int]:
    """Choose one real station per case study, by calendar and graph position."""
    stations = sorted(calendars)
    if not stations:
        raise ValueError("no calendars to choose case-study observers from")

    density = {s: len(calendars[s]) / _span_years(calendars[s]) for s in stations}
    n_days = {s: len(calendars[s]) for s in stations}
    span = {s: _span_years(calendars[s]) for s in stations}
    start = {s: int(calendars[s][0]) for s in stations}
    taken: set[int] = set()

    def take(candidates: list[int]) -> int:
        for station in candidates:
            if station not in taken:
                taken.add(station)
                return station
        remaining = [s for s in stations if s not in taken]
        taken.add(remaining[0])
        return remaining[0]

    by_density = sorted(stations, key=lambda s: -density[s])
    by_span = sorted(stations, key=lambda s: -span[s])
    by_start = sorted(stations, key=lambda s: start[s])
    sparse_first = sorted(stations, key=lambda s: (density[s], start[s]))
    if degrees:
        by_degree = sorted(stations, key=lambda s: degrees.get(s, 0))
    else:
        by_degree = sparse_first

    modern = [s for s in by_density if start[s] > 715_000]
    chosen = {
        # Dense, late and -- the part the first version of this left out -- *long*.
        # Density alone picked a 517-day career, which is ~17 of the fixed months
        # UEA21's statistic is defined on, against its own floor of 12.  A record
        # that short does not have a distribution, and what the acuity methods then
        # recovered was the activity of whichever 17 months the replay had put under
        # it, not the observer's acuity: Spearman -0.93 against the window's true
        # mean count, across the ten S2 worlds.  The case every method should get
        # right has to be a clean *record*, not merely a dense one.
        "clean-modern": take(
            [s for s in modern if n_days[s] >= MIN_CLEAN_MODERN_DAYS] or modern or by_density
        ),
        # Earliest and thinnest calendar available.
        "sparse-maunder": take([s for s in by_start if density[s] < 60] or by_start),
        "break-undocumented": take(by_span),
        "break-documented": take(by_span),
        "slow-drift": take(by_span),
        "poor-instrument": take(by_density),
        # Fewest overlap partners in the graph.
        "leaf-node": take(by_degree),
        "copier": take(by_density),
        "zero-omitter": take(by_span),
        "blanket-statements": take(by_start),
    }
    return chosen


def apply_case_studies(
    population: list[ObserverParams],
    calendars: dict[int, np.ndarray],
    rng: np.random.Generator,
    *,
    degrees: dict[int, int] | None = None,
) -> list[ObserverParams]:
    """Overwrite ten of the drawn observers with the named case studies.

    Returns a new population list.  The ten keep their real calendars -- that is
    the point of choosing them by calendar -- and gain the parameters that make
    each one the case it is named for.
    """
    chosen = _pick_stations(calendars, degrees)
    by_station = {p.station: index for index, p in enumerate(population)}
    out = list(population)

    for kind in CASE_STUDY_KINDS:
        station = chosen[kind]
        index = by_station.get(station)
        if index is None:
            continue
        overrides = dict(CASE_STUDY_OVERRIDES[kind])
        days = calendars[station]

        if kind in ("break-undocumented", "break-documented"):
            # One clean mid-career acuity improvement, so the only difference
            # between these two panels is whether the method is told about it.
            middle = int(days[len(days) // 2])
            overrides |= {
                "break_days": (middle,),
                "break_jumps": (-0.5,),
                "break_documented": (kind == "break-documented",),
            }
        else:
            overrides |= {"break_days": (), "break_jumps": (), "break_documented": ()}

        if kind == "copier":
            partner = _nearest_overlapping(station, calendars, exclude=set(chosen.values()))
            overrides["copies_of"] = partner
        else:
            overrides["copies_of"] = None

        out[index] = replace(out[index], kind=kind, **overrides)

    return out


def _nearest_overlapping(
    station: int, calendars: dict[int, np.ndarray], *, exclude: set[int]
) -> int | None:
    """The overlapping partner with the most shared days, for the copier."""
    mine = set(calendars[station].tolist())
    best, best_shared = None, 0
    for other, days in calendars.items():
        if other == station or other in exclude:
            continue
        shared = len(mine.intersection(days.tolist()))
        if shared > best_shared:
            best, best_shared = other, shared
    return best
