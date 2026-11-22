"""Synthetic observers: what a person with a small telescope writes down.

Everything here that Chatzistergos et al. (2025) list and then set aside -- "actual
observational reports may also vary due to differences in grouping conventions,
spot-counting practices, the possible use of weighting schemes, or changes in
observer acuity over time ... which we do not consider here" -- is modelled.  Their
observer is a hard area threshold; this one has a soft threshold that drifts and
breaks, miscounts groups by splitting and merging them, sees through variable
seeing, omits spotless days at a rate that depends on activity, sometimes copies a
neighbour, and sometimes writes a sentence instead of a number.

**The generator is deliberately misspecified against the Step 4 model.** The
research notes are blunt about why: "Benchmarks that reuse the same generative model
as the inference method will overstate performance."  Four departures are structural
rather than cosmetic, and each is switchable so its contribution can be isolated:

=====================  ==========================  ==========================
layer                  here                        Step 4 will assume
=====================  ==========================  ==========================
detection              logistic in log-area with   smooth ``q_o`` with no
                       per-day seeing jitter       per-day noise
career drift           abrupt breaks + random      smooth random walk / spline
                       walk
observer errors        a fraction copy a           independent observers
                       neighbour
spotless reporting     MNAR, activity-dependent    MNAR (matched by design)
=====================  ==========================  ==========================

The other two departures live in the truth generator (lifetime law and clustered
emergence), not here.

Detection, per observer *o*, day *t*, visible group *i* of area *a*:

.. math::

    a' = a \\, e^{\\epsilon},\\; \\epsilon \\sim N(0, \\sigma_{\\rm seeing}^2)
    \\qquad
    P(\\text{detect}) = \\operatorname{logistic}\\!\\left(
        \\frac{\\ln a' - \\ln \\theta_o(t)}{w_o} \\right)

so :math:`\\theta_o(t)` is the area at which the observer detects half the time --
the soft twin of the legacy hard acuity threshold, and the parameter the scorer
asks each baseline to recover.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.special import expit

from gn_next_gen.benchmark.truth import Truth

#: The ten case-study observers, by the name their panel carries in f25/f26.
CASE_STUDY_KINDS = (
    "clean-modern",
    "sparse-maunder",
    "break-undocumented",
    "break-documented",
    "slow-drift",
    "poor-instrument",
    "leaf-node",
    "copier",
    "zero-omitter",
    "blanket-statements",
)


@dataclass(frozen=True)
class ObserverParams:
    """One synthetic observer's truth. Sealed -- no baseline may read this.

    Attributes
    ----------
    station
        The real station whose calendar this observer inherits.
    base_threshold_msh
        Career-median acuity: the area detected half the time, in MSH.
    softness
        Logistic width in log-area.  Small is nearly a hard threshold, which is
        the limiting case the legacy ADF method assumes.
    seeing_sigma
        Per-day, per-group multiplicative noise on area, in log units.
    p_split, p_merge
        Per-detected-group probability of being written as two groups, and of a
        pair being written as one.
    mnar_omega
        Probability of omitting a spotless day at zero activity, falling to zero
        at high activity.  Zero means the observer reports every blank day.
    drift_sigma
        Random-walk scale on log threshold, per sqrt(year).
    break_days, break_jumps, break_documented
        Abrupt career changes: when, how big in log threshold, and whether the
        break is exposed to methods as metadata.
    copies_of
        Station copied from, or ``None``.  A copier reproduces its partner's
        counts on shared days, which is correlated error no method under test
        models.
    blanket
        Whether the record is reported as yearly statements instead of counts.
    kind
        A label, set for the ten case-study observers and ``"population"``
        otherwise.
    """

    station: int
    base_threshold_msh: float
    softness: float = 0.45
    seeing_sigma: float = 0.25
    p_split: float = 0.02
    p_merge: float = 0.02
    mnar_omega: float = 0.0
    drift_sigma: float = 0.0
    break_days: tuple[int, ...] = ()
    break_jumps: tuple[float, ...] = ()
    break_documented: tuple[bool, ...] = ()
    copies_of: int | None = None
    blanket: bool = False
    kind: str = "population"

    def as_row(self) -> dict:
        return {
            "station": self.station,
            "kind": self.kind,
            "base_threshold_msh": self.base_threshold_msh,
            "softness": self.softness,
            "seeing_sigma": self.seeing_sigma,
            "p_split": self.p_split,
            "p_merge": self.p_merge,
            "mnar_omega": self.mnar_omega,
            "drift_sigma": self.drift_sigma,
            "n_breaks": len(self.break_days),
            "n_documented_breaks": int(sum(self.break_documented)),
            "copies_of": -1 if self.copies_of is None else self.copies_of,
            "blanket": self.blanket,
        }


@dataclass
class ObservedNetwork:
    """What the methods get to see, and what they do not.

    ``observations`` and ``statements`` are open; ``parameters`` and
    ``thresholds`` are sealed and written to the sealed root.
    """

    observations: pd.DataFrame
    statements: pd.DataFrame
    parameters: pd.DataFrame
    thresholds: pd.DataFrame
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)


def threshold_trajectory(
    params: ObserverParams, days: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """The observer's acuity on each of its own observing days, in MSH.

    A random walk in log threshold, plus a step at each break.  The walk is
    evaluated on the observer's *own* days with increments scaled by the real gap
    between them, so a career with a ten-year interruption drifts across it.
    """
    log_theta = np.full(len(days), np.log(params.base_threshold_msh), dtype=float)
    if params.drift_sigma > 0 and len(days) > 1:
        gap_years = np.diff(days) / 365.25
        steps = rng.normal(0.0, params.drift_sigma * np.sqrt(gap_years))
        log_theta[1:] += np.cumsum(steps)
    for day, jump in zip(params.break_days, params.break_jumps, strict=True):
        log_theta[days >= day] += jump
    return np.exp(log_theta)


def _day_slices(truth: Truth) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index the truth's group-days so any day's areas are one slice.

    ``group_days`` is sorted by ``date_ord``, so a day's rows are contiguous and a
    pair of ``searchsorted`` bounds locates them without building a dense matrix.
    """
    ordered = truth.group_days.sort_values("date_ord")
    date_ord = ordered["date_ord"].to_numpy(dtype=np.int64)
    areas = ordered["area_msh"].to_numpy(dtype=float)
    return date_ord, areas, ordered["group_id"].to_numpy()


def _gather(
    date_ord: np.ndarray, areas: np.ndarray, days: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Areas of every group visible on each of ``days``, flattened.

    Returns the flat area array and the number of groups per day, which together
    are a ragged array without the memory of a padded one.
    """
    lo = np.searchsorted(date_ord, days, side="left")
    hi = np.searchsorted(date_ord, days, side="right")
    per_day = (hi - lo).astype(np.int64)
    total = int(per_day.sum())
    if total == 0:
        return np.empty(0, dtype=float), per_day
    starts = np.repeat(lo, per_day)
    offsets = np.arange(total) - np.repeat(np.concatenate([[0], np.cumsum(per_day)[:-1]]), per_day)
    return areas[starts + offsets], per_day


def observe_one(
    truth: Truth,
    params: ObserverParams,
    days: np.ndarray,
    rng: np.random.Generator,
    *,
    date_ord: np.ndarray,
    areas: np.ndarray,
    activity: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """One observer's reported counts, and its acuity trajectory.

    Returns ``(frame, thresholds)`` where ``frame`` has ``date_ord, groups`` for the
    days the observer both looked and reported.
    """
    thresholds = threshold_trajectory(params, days, rng)
    flat_areas, per_day = _gather(date_ord, areas, days)

    if len(flat_areas):
        theta = np.repeat(thresholds, per_day)
        seen = np.log(flat_areas)
        if params.seeing_sigma > 0:
            seen = seen + rng.normal(0.0, params.seeing_sigma, size=len(seen))
        # expit, not 1/(1+exp(-x)): a near-hard threshold makes the exponent
        # huge and the naive form overflows exactly where ADF's limiting case lives.
        probability = expit((seen - np.log(theta)) / params.softness)
        detected = rng.random(len(probability)) < probability
        # bincount, not add.reduceat: a day with no visible group gives an empty
        # slice, and reduceat rejects a start index equal to the array length.
        day_index = np.repeat(np.arange(len(days)), per_day)
        counts = np.bincount(day_index[detected], minlength=len(days)).astype(np.int64)
    else:
        counts = np.zeros(len(days), dtype=np.int64)

    # Grouping convention: a detected group is occasionally written as two, and a
    # pair occasionally as one.  Both are miscounts of the same Sun.
    if params.p_split > 0:
        counts = counts + rng.binomial(counts, params.p_split)
    if params.p_merge > 0:
        pairs = np.maximum(counts - 1, 0)
        counts = counts - rng.binomial(pairs, params.p_merge)
    counts = np.maximum(counts, 0)

    keep = np.ones(len(days), dtype=bool)
    if params.mnar_omega > 0:
        # MNAR: a blank day is likelier to go unrecorded when the Sun is quiet,
        # which is exactly where the Maunder Minimum is decided and exactly what
        # ADF's missing-at-random assumption denies.
        scale = np.ones(len(days)) if activity is None else activity
        omit = params.mnar_omega * np.clip(1.0 - scale, 0.0, 1.0)
        keep &= ~((counts == 0) & (rng.random(len(days)) < omit))

    frame = pd.DataFrame({"date_ord": days[keep], "groups": counts[keep]})
    frame.insert(0, "station", params.station)
    return frame, thresholds


def observe(
    truth: Truth,
    population: list[ObserverParams],
    calendars: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> ObservedNetwork:
    """Run the whole synthetic network over one truth.

    Copiers are resolved after their partners, so a copier reproduces what its
    partner actually wrote rather than an independent draw from the same
    parameters -- the difference between correlated error and none.
    """
    date_ord, areas, _ = _day_slices(truth)
    activity = truth.activity_level()
    activity_scaled = (activity / max(float(activity.max()), 1e-12)).to_dict()

    independent = [p for p in population if p.copies_of is None]
    copiers = [p for p in population if p.copies_of is not None]

    frames: dict[int, pd.DataFrame] = {}
    trajectories: list[pd.DataFrame] = []

    for params in independent:
        days = calendars[params.station]
        scale = np.array([activity_scaled.get(int(d), 0.0) for d in days])
        frame, thresholds = observe_one(
            truth, params, days, rng, date_ord=date_ord, areas=areas, activity=scale
        )
        frames[params.station] = frame
        trajectories.append(
            pd.DataFrame({"station": params.station, "date_ord": days, "threshold_msh": thresholds})
        )

    for params in copiers:
        days = calendars[params.station]
        source = frames.get(params.copies_of)
        if source is None or source.empty:
            # Nothing to copy: fall back to observing independently rather than
            # dropping the observer, and record the fallback in the parameters.
            scale = np.array([activity_scaled.get(int(d), 0.0) for d in days])
            frame, thresholds = observe_one(
                truth, params, days, rng, date_ord=date_ord, areas=areas, activity=scale
            )
        else:
            lookup = source.set_index("date_ord")["groups"]
            shared = lookup.reindex(days)
            thresholds = threshold_trajectory(params, days, rng)
            have = shared.notna().to_numpy()
            counts = shared.fillna(0).to_numpy().astype(np.int64)
            # A copier is not a photocopier: it drops the occasional group.
            counts = np.maximum(counts - rng.binomial(counts, 0.05), 0)
            frame = pd.DataFrame(
                {"station": params.station, "date_ord": days[have], "groups": counts[have]}
            )
        frames[params.station] = frame
        trajectories.append(
            pd.DataFrame({"station": params.station, "date_ord": days, "threshold_msh": thresholds})
        )

    statements = []
    for params in population:
        if not params.blanket:
            continue
        frame = frames.pop(params.station, None)
        if frame is None or frame.empty:
            continue
        statements.append(_to_statements(frame, params, rng))

    observations = (
        pd.concat(frames.values(), ignore_index=True)
        if frames
        else pd.DataFrame(columns=["station", "date_ord", "groups"])
    )
    observations = observations.sort_values(["date_ord", "station"], ignore_index=True)

    return ObservedNetwork(
        observations=observations,
        statements=(
            pd.concat(statements, ignore_index=True)
            if statements
            else pd.DataFrame(columns=["station", "year", "kind", "n_days", "n_active_days"])
        ),
        parameters=pd.DataFrame([p.as_row() for p in population]),
        thresholds=pd.concat(trajectories, ignore_index=True),
        metadata=_documented_breaks(population),
    )


def _to_statements(
    frame: pd.DataFrame, params: ObserverParams, rng: np.random.Generator
) -> pd.DataFrame:
    """Collapse a daily record into yearly statements of unknown literalness.

    Hoyt & Schatten read generic statements as daily observations, which
    Neuhäuser et al. (2018) showed inflates the implied observing effort.  Here
    the statement carries a *kind*: a literal one means every covered day really
    was spotless, a rhetorical one means only that the year was quiet.  A method
    that reads both as daily zeros should pay for it.
    """
    years = np.array([_year_of(d) for d in frame["date_ord"].to_numpy()])
    rows = []
    for year in np.unique(years):
        block = frame[years == year]
        active = int((block["groups"] > 0).sum())
        literal = active == 0 and rng.random() < 0.5
        rows.append(
            {
                "station": params.station,
                "year": int(year),
                "kind": "literal_none" if literal else "rhetorical",
                "n_days": int(len(block)),
                "n_active_days": active,
            }
        )
    return pd.DataFrame(rows)


def _year_of(date_ord: int) -> int:
    import datetime as dt

    return dt.date.fromordinal(int(date_ord)).year


def _documented_breaks(population: list[ObserverParams]) -> pd.DataFrame:
    """The breaks a method is allowed to know about -- station metadata, no more."""
    rows = [
        {"station": p.station, "date_ord": int(day), "note": "instrument change"}
        for p in population
        for day, documented in zip(p.break_days, p.break_documented, strict=True)
        if documented
    ]
    return pd.DataFrame(rows, columns=["station", "date_ord", "note"])


def draw_population(
    calendars: dict[int, np.ndarray],
    rng: np.random.Generator,
    *,
    copier_fraction: float = 0.05,
    blanket_fraction: float = 0.03,
    break_rate_per_century: float = 0.8,
    documented_fraction: float = 0.5,
    threshold_log_mean: float = np.log(35.0),
    threshold_log_sd: float = 0.75,
) -> list[ObserverParams]:
    """Draw a realistic observer population over a set of calendars.

    Acuity is log-normal around 35 MSH, which puts most observers inside the
    0-100 msd band where Usoskin et al. (2016b) place real observers back to 1749,
    with a tail of poor instruments.  Earlier careers drift and break more often,
    since that is where instruments and conventions actually changed.
    """
    stations = sorted(calendars)
    population: list[ObserverParams] = []
    for station in stations:
        days = calendars[station]
        span_years = max((days[-1] - days[0]) / 365.25, 1.0)
        n_breaks = rng.poisson(break_rate_per_century * span_years / 100.0)
        break_days = (
            tuple(
                sorted(
                    int(d)
                    for d in rng.choice(
                        days[1:-1], size=min(n_breaks, max(len(days) - 2, 0)), replace=False
                    )
                )
            )
            if n_breaks and len(days) > 2
            else ()
        )
        population.append(
            ObserverParams(
                station=station,
                base_threshold_msh=float(np.exp(rng.normal(threshold_log_mean, threshold_log_sd))),
                softness=float(np.exp(rng.normal(np.log(0.45), 0.3))),
                seeing_sigma=float(abs(rng.normal(0.25, 0.08))),
                p_split=float(abs(rng.normal(0.02, 0.01))),
                p_merge=float(abs(rng.normal(0.02, 0.01))),
                mnar_omega=float(np.clip(rng.normal(0.25, 0.15), 0.0, 0.8)),
                drift_sigma=float(abs(rng.normal(0.08, 0.04))),
                break_days=break_days,
                break_jumps=tuple(float(rng.normal(0.0, 0.35)) for _ in break_days),
                break_documented=tuple(
                    bool(rng.random() < documented_fraction) for _ in break_days
                ),
            )
        )

    n_copiers = int(round(copier_fraction * len(population)))
    if n_copiers and len(population) > 1:
        chosen = rng.choice(len(population), size=n_copiers, replace=False)
        for index in chosen:
            partner = _overlapping_partner(population[index].station, calendars, rng)
            if partner is not None:
                population[index] = replace(population[index], copies_of=partner)

    n_blanket = int(round(blanket_fraction * len(population)))
    if n_blanket:
        order = np.argsort([calendars[p.station][0] for p in population])
        for index in order[:n_blanket]:
            population[index] = replace(population[index], blanket=True)

    return population


def _overlapping_partner(
    station: int, calendars: dict[int, np.ndarray], rng: np.random.Generator
) -> int | None:
    """A station whose career overlaps this one, to copy from."""
    mine = calendars[station]
    candidates = [
        other
        for other, days in calendars.items()
        if other != station and days[0] <= mine[-1] and days[-1] >= mine[0]
    ]
    if not candidates:
        return None
    return int(candidates[rng.integers(len(candidates))])
