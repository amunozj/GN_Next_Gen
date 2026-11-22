"""Reference replay: a 1610-2026 truth built from real groups, areas and lifetimes.

The reference covers 1825-2026 against a network spanning ~400 years, so a replay
has a gap to bridge.  This generator bridges it by **resequencing whole cycles**:
inside the donor survey's own span the real record is used verbatim, and outside it
synthetic cycles are filled by drawing a complete real cycle, shifting it rigidly in
time, and deleting whole groups at random to set the amplitude.

Three properties are deliberate.

* **Real areas and real lifetimes.** A drawn group keeps its own daily area profile
  and its own visible span; only its date moves.  The spec asks for real lifetimes
  and ``group_single_entry.parquet`` finally supplies them, so nothing here is
  parametric.  Rigid shifting also keeps the rotation signature Step 1 found the
  reference carries (the lag-13 trough and lag-27 recurrence) rather than
  re-imposing it from a model.
* **One donor survey per realization, by default.** Schwabe's areas are *umbral*
  with a 7.1 MSH floor against RGO's whole-spot 28.1, so mixing donors across
  surveys would plant a scale jump at a cycle boundary and the benchmark would
  score methods on an artefact.  ADR-011's per-survey log-normal offset makes a
  mixed-donor variant possible; it is a flagged sensitivity, not the default.
* **Amplitudes are synthetic.** The cycle-amplitude sequence comes from a seeded
  quasi-periodic model, never from SILSO, a proxy or a published GN series.  The
  firewall (ADR-002) requires it, and it is also what keeps the truth genuinely
  unknown rather than a copy of a series a method might have been tuned on.

Spörer is excluded as a donor: all 5,421 of its groups carry ``obs_duration = 1``,
so it supplies no lifetimes (**D23**).  Schwabe is *not* excluded here, although
Step 1 excluded it -- Step 1 needed spotless days, which Schwabe lacks, while a
truth generator needs groups, areas and lifetimes, which Schwabe has.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gn_next_gen.benchmark.truth import Truth, anchor_threshold
from gn_next_gen.io import load_group_days, load_group_single_entry
from gn_next_gen.paths import REFERENCE_CYCLE_COVERAGE

#: Surveys that can donate cycles.  Spörer is out (D23: no multi-day groups).
DONOR_SURVEYS = ("schwabe", "rgo", "debrecen", "kmas")

#: Mean solar cycle length in days, and the jitter the amplitude model applies.
MEAN_CYCLE_DAYS = 4015.0
CYCLE_JITTER_DAYS = 300.0


@dataclass
class DonorCycle:
    """One complete real cycle, ready to be shifted and thinned."""

    survey: str
    cycle: int
    first_ord: int
    last_ord: int
    groups: pd.DataFrame
    group_days: pd.DataFrame

    @property
    def length_days(self) -> int:
        return int(self.last_ord - self.first_ord + 1)

    @property
    def n_groups(self) -> int:
        return len(self.groups)


def build_donor_library(survey: str = "rgo") -> dict[int, DonorCycle]:
    """Every **complete** cycle of one survey, as shiftable donors.

    Completeness is ADR-009's rule -- a cycle counts only if the survey also
    carries its neighbours -- and is read from the frozen
    ``survey_cycle_coverage.csv`` rather than recomputed, so a future upstream
    refit moves it in one place.
    """
    if survey not in DONOR_SURVEYS:
        raise ValueError(f"{survey!r} cannot donate cycles; expected one of {DONOR_SURVEYS}")

    coverage = pd.read_csv(REFERENCE_CYCLE_COVERAGE)
    complete = coverage[(coverage["survey"] == survey) & coverage["complete"]]
    if complete.empty:
        raise ValueError(f"{survey!r} has no complete cycle in survey_cycle_coverage.csv")

    singles = load_group_single_entry(surveys=[survey])
    day_rows = load_group_days(surveys=[survey])
    day_rows = day_rows[day_rows["group_id"].isin(set(singles["group_id"]))]

    library: dict[int, DonorCycle] = {}
    for row in complete.itertuples():
        members = singles[singles["cycle"] == row.cycle]
        if members.empty:
            continue
        ids = set(members["group_id"])
        library[int(row.cycle)] = DonorCycle(
            survey=survey,
            cycle=int(row.cycle),
            first_ord=int(row.first_ord),
            last_ord=int(row.last_ord),
            groups=members.reset_index(drop=True),
            group_days=day_rows[day_rows["group_id"].isin(ids)].reset_index(drop=True),
        )
    return library


def _cycle_boundaries(
    first_ord: int, last_ord: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """Synthetic cycle edges with a jittered period, covering ``[first, last]``."""
    edges = [int(first_ord)]
    while edges[-1] <= last_ord:
        length = rng.normal(MEAN_CYCLE_DAYS, CYCLE_JITTER_DAYS)
        edges.append(edges[-1] + int(max(2500.0, length)))
    return [(edges[i], edges[i + 1] - 1) for i in range(len(edges) - 1)]


def _amplitudes(
    n_cycles: int,
    rng: np.random.Generator,
    *,
    grand_minimum: tuple[int, int] | None,
    boundaries: list[tuple[int, int]],
    grand_minimum_factor: float,
) -> np.ndarray:
    """Relative cycle amplitudes in ``(0, 1]``, from a seeded synthetic model.

    An AR(1) in log amplitude gives the modest cycle-to-cycle persistence the real
    record shows without copying any published series.  No proxy, no SILSO: the
    firewall (ADR-002) and the point of a blind benchmark both require it.
    """
    log_amplitude = np.empty(n_cycles)
    log_amplitude[0] = rng.normal(0.0, 0.35)
    for k in range(1, n_cycles):
        log_amplitude[k] = 0.45 * log_amplitude[k - 1] + rng.normal(0.0, 0.35)
    amplitude = np.exp(log_amplitude)
    amplitude = np.clip(amplitude / amplitude.max(), 0.15, 1.0)

    if grand_minimum is not None:
        start, end = grand_minimum
        for k, (lo, hi) in enumerate(boundaries):
            if hi >= start and lo <= end:
                amplitude[k] /= grand_minimum_factor
    return amplitude


def _shift_cycle(
    donor: DonorCycle,
    target_start: int,
    keep_fraction: float,
    rng: np.random.Generator,
    tag: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shift a donor cycle to a new start and delete whole groups.

    Deleting *whole* groups rather than thinning group-days is what the spec asks
    for and what keeps the result physical: a weaker cycle has fewer regions, not
    smaller fragments of the same regions.  Each surviving group keeps its own
    area profile and visible span, so lifetimes and the rotation signature survive
    the move.
    """
    keep = rng.random(donor.n_groups) < keep_fraction
    groups = donor.groups[keep].copy()
    if groups.empty:
        empty_days = pd.DataFrame(columns=["group_id", "date_ord", "area_msh"])
        return groups, empty_days

    delta = int(target_start) - donor.first_ord
    rename = {old: f"{tag}_{old}" for old in groups["group_id"]}

    day_rows = donor.group_days[donor.group_days["group_id"].isin(set(groups["group_id"]))].copy()
    day_rows["group_id"] = day_rows["group_id"].map(rename)
    day_rows["date_ord"] = day_rows["date_ord"] + delta
    day_rows = day_rows.rename(columns={"area_corr": "area_msh"})[
        ["group_id", "date_ord", "area_msh"]
    ]

    out = pd.DataFrame(
        {
            "group_id": groups["group_id"].map(rename).to_numpy(),
            "emergence_ord": groups["date_ord"].to_numpy() + delta,
            "visible_span": groups["obs_duration"].to_numpy(),
            "peak_area_msh": groups["area_corr"].to_numpy(),
            "donor_survey": donor.survey,
            "donor_cycle": donor.cycle,
        }
    )
    return out, day_rows


def replay_truth(
    first_ord: int,
    last_ord: int,
    *,
    survey: str = "rgo",
    anchoring: str = "rgo_15_19",
    seed: int = 0,
    grand_minimum: tuple[int, int] | None = None,
    grand_minimum_factor: float = 30.0,
) -> Truth:
    """Build a replay truth spanning ``[first_ord, last_ord]``.

    Parameters
    ----------
    first_ord, last_ord
        Proleptic-Gregorian ordinals bounding the world.
    survey
        The donor survey.  Its own span is replayed verbatim; everything outside is
        resequenced from its complete cycles.
    anchoring
        ADR-009 anchor, which fixes the estimand's threshold.
    seed
        Everything random here derives from this.
    grand_minimum
        ``(start_ord, end_ord)`` over which cycle amplitudes are divided by
        ``grand_minimum_factor``.  Whole groups are deleted, never shrunk.
    grand_minimum_factor
        10-100 is the range the research notes suggest; 30 is the default.

    Returns
    -------
    Truth
    """
    rng = np.random.default_rng(seed)
    library = build_donor_library(survey)
    donors = list(library.values())
    biggest = max(d.n_groups for d in donors)

    real_first = min(d.first_ord for d in donors)
    real_last = max(d.last_ord for d in donors)
    real_lo, real_hi = max(first_ord, real_first), min(last_ord, real_last)

    group_frames: list[pd.DataFrame] = []
    day_frames: list[pd.DataFrame] = []

    # --- the survey's own span, verbatim -----------------------------------
    if real_lo <= real_hi:
        for donor in donors:
            inside = (donor.groups["date_ord"] >= real_lo) & (donor.groups["date_ord"] <= real_hi)
            members = donor.groups[inside]
            if members.empty:
                continue
            ids = set(members["group_id"])
            rows = donor.group_days[donor.group_days["group_id"].isin(ids)]
            group_frames.append(
                pd.DataFrame(
                    {
                        "group_id": members["group_id"].to_numpy(),
                        "emergence_ord": members["date_ord"].to_numpy(),
                        "visible_span": members["obs_duration"].to_numpy(),
                        "peak_area_msh": members["area_corr"].to_numpy(),
                        "donor_survey": donor.survey,
                        "donor_cycle": donor.cycle,
                    }
                )
            )
            day_frames.append(
                rows.rename(columns={"area_corr": "area_msh"})[["group_id", "date_ord", "area_msh"]]
            )

    # --- everything outside it, resequenced --------------------------------
    gaps = [
        (int(first_ord), int(min(last_ord, real_lo - 1))),
        (int(max(first_ord, real_hi + 1)), int(last_ord)),
    ]
    tag = 0
    for gap_lo, gap_hi in gaps:
        if gap_lo > gap_hi:
            continue
        boundaries = _cycle_boundaries(gap_lo, gap_hi, rng)
        amplitude = _amplitudes(
            len(boundaries),
            rng,
            grand_minimum=grand_minimum,
            boundaries=boundaries,
            grand_minimum_factor=grand_minimum_factor,
        )
        for (start, _end), target in zip(boundaries, amplitude, strict=True):
            wanted = target * biggest
            eligible = [d for d in donors if d.n_groups >= wanted] or [
                max(donors, key=lambda d: d.n_groups)
            ]
            donor = eligible[rng.integers(len(eligible))]
            tag += 1
            groups, day_rows = _shift_cycle(
                donor, start, min(1.0, wanted / donor.n_groups), rng, f"s{tag:03d}"
            )
            # Cycles overlap in reality and may overlap each other here, but a
            # synthetic cycle must not spill into the survey's own span: the real
            # record already carries that era in full, and a spill would count the
            # same Sun twice.  Tiling always overruns the last gap boundary, so the
            # clip is the rule rather than an edge case.
            if not day_rows.empty:
                inside_gap = (day_rows["date_ord"] >= gap_lo) & (day_rows["date_ord"] <= gap_hi)
                day_rows = day_rows[inside_gap]
                groups = groups[groups["group_id"].isin(set(day_rows["group_id"]))]
            if not groups.empty:
                group_frames.append(groups)
                day_frames.append(day_rows)

    groups_all = pd.concat(group_frames, ignore_index=True)
    days_all = pd.concat(day_frames, ignore_index=True)

    inside = (days_all["date_ord"] >= first_ord) & (days_all["date_ord"] <= last_ord)
    days_all = days_all[inside].reset_index(drop=True)
    groups_all = groups_all[groups_all["group_id"].isin(set(days_all["group_id"]))]
    groups_all = groups_all.sort_values("emergence_ord").reset_index(drop=True)

    return Truth(
        groups=groups_all,
        group_days=days_all.sort_values(["date_ord", "group_id"]).reset_index(drop=True),
        days=np.arange(int(first_ord), int(last_ord) + 1, dtype=np.int64),
        reference_threshold_msh=anchor_threshold(anchoring),
        anchoring=anchoring,
        meta={
            "generator": "replay",
            "donor_survey": survey,
            "seed": int(seed),
            "grand_minimum": list(grand_minimum) if grand_minimum else None,
            "grand_minimum_factor": grand_minimum_factor if grand_minimum else None,
            "real_span_ord": [int(real_lo), int(real_hi)] if real_lo <= real_hi else None,
        },
    )
