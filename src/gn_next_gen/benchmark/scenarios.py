"""The six scenarios, and the one function that turns a scenario into a world.

Each scenario declares its **anchoring** (ADR-009), because the estimand is a
reference-equivalent count and the scorer must never compare two anchorings'
numbers as though they were the same quantity.

===  ==========================  ====  ===============================================
id   scenario                    reps  what it is for
===  ==========================  ====  ===============================================
S1   modern sub-network          20    development, and enough worlds for coverage
S2   full network 1610-2026      10    the headline
S3   grand minimum               10    the Maunder question, by whole-group deletion
S4   synchronous breaks          10    era-wide acuity change: where pairwise methods
                                       fail and common-mode drift looks like trend
S5   sampling-only (Hay)         1     perfect observers on real calendars, so any
                                       bias is the sampling pattern alone
S6   structural ensemble         --    no new data: S2 re-scored with each baseline's
                                       free choices perturbed
===  ==========================  ====  ===============================================

S6 generates nothing, which is deliberate: a structural ensemble perturbs the
*method's* free choices, not the world, so it reuses S2's worlds and lives in the
baseline runner.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from gn_next_gen.benchmark.calendars import observer_calendars
from gn_next_gen.benchmark.case_studies import apply_case_studies
from gn_next_gen.benchmark.generators.replay import replay_truth
from gn_next_gen.benchmark.observers import (
    ObservedNetwork,
    ObserverParams,
    draw_population,
    observe,
)
from gn_next_gen.benchmark.store import GENERATOR_VERSION
from gn_next_gen.benchmark.truth import ANCHORINGS, Truth

#: Station id of the reference observer.  Far above any real station number so it
#: cannot collide with a transplanted calendar.
REFERENCE_STATION = 9001


def _ord(year: int, month: int = 1, day: int = 1) -> int:
    return dt.date(year, month, day).toordinal()


@dataclass(frozen=True)
class Scenario:
    """One benchmark scenario.

    Attributes
    ----------
    grand_minimum
        ``(start_ord, end_ord)`` over which cycle amplitudes are divided by
        ``grand_minimum_factor``.  Whole groups are deleted, never shrunk.
    synchronous_break
        ``(date_ord, log_jump)`` applied to **every** observer: an era-wide
        telescope improvement.  This is the MULTITEST lesson -- synchronous
        breaks defeat pairwise methods and are confounded with real trend.
    perfect_observers
        Detection error switched off entirely.  Only S5 uses it: with a perfect
        observer, every residual bias is the observing calendar's doing, which is
        the Hay et al. sampling-only diagnostic.
    """

    name: str
    first_ord: int
    last_ord: int
    n_realizations: int
    donor_survey: str = "rgo"
    anchoring: str = "rgo_15_19"
    grand_minimum: tuple[int, int] | None = None
    grand_minimum_factor: float = 30.0
    synchronous_break: tuple[int, float] | None = None
    perfect_observers: bool = False
    min_calendar_days: int = 50
    description: str = ""
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


#: The scenario grid.  Realization counts follow the approved plan.
SCENARIOS: dict[str, Scenario] = {
    "S1": Scenario(
        name="S1",
        first_ord=_ord(1874),
        last_ord=_ord(2026, 12, 31),
        n_realizations=20,
        description="Modern sub-network: the cheap scenario, used for development "
        "and for enough worlds to pin interval coverage.",
    ),
    "S2": Scenario(
        name="S2",
        first_ord=_ord(1610),
        last_ord=_ord(2026, 12, 31),
        n_realizations=10,
        description="The full network over the whole record. The headline scenario.",
    ),
    "S3": Scenario(
        name="S3",
        first_ord=_ord(1610),
        last_ord=_ord(2026, 12, 31),
        n_realizations=10,
        grand_minimum=(_ord(1645), _ord(1715, 12, 31)),
        grand_minimum_factor=30.0,
        description="A Maunder-like grand minimum inserted by deleting whole groups. "
        "Thinned modern groups are not a real regime change -- see the results doc.",
    ),
    "S4": Scenario(
        name="S4",
        first_ord=_ord(1610),
        last_ord=_ord(2026, 12, 31),
        n_realizations=10,
        synchronous_break=(_ord(1850), -0.45),
        description="An era-wide acuity improvement in 1850, applied to every observer "
        "at once: common-mode drift that looks exactly like real solar change.",
    ),
    "S5": Scenario(
        name="S5",
        first_ord=_ord(1610),
        last_ord=_ord(2026, 12, 31),
        n_realizations=1,
        perfect_observers=True,
        description="Hay-style sampling-only: perfect observers on real calendars, so "
        "any bias in a naive estimator comes from the sampling pattern alone.",
    ),
}

#: S6 adds no world; it perturbs each baseline's free choices over S2's worlds.
STRUCTURAL_ENSEMBLE_SOURCE = "S2"


def reference_calendar(anchoring: str) -> np.ndarray:
    """The anchor epoch's own observing days.

    ADR-003 defines GN as the count a **fixed reference observer** would have
    made, and ADR-009 fixes that observer as a survey over whole cycles.  The
    reference is therefore an observer in the network like any other, with a real
    calendar covering only its own epoch -- which is what makes the benchmark
    hard: every method has to carry the anchor's scale out to 1610 along the
    overlap graph, and that carrying is what the scorer measures.
    """
    import pandas as pd

    from gn_next_gen.io import load_day_status
    from gn_next_gen.paths import REFERENCE_CYCLE_COVERAGE

    spec = ANCHORINGS[anchoring]
    coverage = pd.read_csv(REFERENCE_CYCLE_COVERAGE)
    rows = coverage[(coverage["survey"] == spec["survey"]) & coverage["cycle"].isin(spec["cycles"])]
    low, high = int(rows["first_ord"].min()), int(rows["last_ord"].max())

    status = load_day_status(surveys=[spec["survey"]])
    observed = status[
        (status["day_status"] > 0) & (status["date_ord"] >= low) & (status["date_ord"] <= high)
    ]
    return np.sort(observed["date_ord"].to_numpy(dtype=np.int64))


def _perfect(params: ObserverParams, threshold: float) -> ObserverParams:
    """Strip every detection effect, leaving only the calendar."""
    return replace(
        params,
        base_threshold_msh=threshold,
        softness=0.0,
        seeing_sigma=0.0,
        p_split=0.0,
        p_merge=0.0,
        mnar_omega=0.0,
        drift_sigma=0.0,
        break_days=(),
        break_jumps=(),
        break_documented=(),
        copies_of=None,
        blanket=False,
        kind="perfect",
    )


def _with_synchronous_break(
    params: ObserverParams, when: int, jump: float, days: np.ndarray
) -> ObserverParams:
    """Add one common break to an observer whose career spans it."""
    if not (days[0] < when < days[-1]):
        return params
    return replace(
        params,
        break_days=tuple(sorted((*params.break_days, when))),
        break_jumps=(*params.break_jumps, jump),
        break_documented=(*params.break_documented, False),
    )


def build_realization(
    scenario: Scenario, seed: int, *, degrees: dict[int, int] | None = None
) -> tuple[Truth, ObservedNetwork, dict]:
    """Generate one world: truth, the network that observed it, and its config.

    Every random draw descends from ``seed``, and the two consumers are seeded
    apart so that changing the observer population does not reshuffle the truth.
    """
    truth = replay_truth(
        scenario.first_ord,
        scenario.last_ord,
        survey=scenario.donor_survey,
        anchoring=scenario.anchoring,
        seed=seed,
        grand_minimum=scenario.grand_minimum,
        grand_minimum_factor=scenario.grand_minimum_factor,
    )

    calendars = observer_calendars(
        scenario.first_ord, scenario.last_ord, min_days=scenario.min_calendar_days
    )
    rng = np.random.default_rng(seed + 1_000_000)
    population = draw_population(calendars, rng)

    if scenario.perfect_observers:
        population = [_perfect(p, truth.reference_threshold_msh) for p in population]
    else:
        population = apply_case_studies(population, calendars, rng, degrees=degrees)

    if scenario.synchronous_break is not None:
        when, jump = scenario.synchronous_break
        population = [
            _with_synchronous_break(p, when, jump, calendars[p.station]) for p in population
        ]

    # The reference observer joins the network, openly declared.  It is the one
    # observer whose scale every method may assume (ADR-003), and it sees only
    # its own epoch.
    anchor_days = reference_calendar(scenario.anchoring)
    inside = (anchor_days >= scenario.first_ord) & (anchor_days <= scenario.last_ord)
    anchor_days = anchor_days[inside]
    if len(anchor_days) == 0:
        raise ValueError(
            f"anchoring {scenario.anchoring!r} has no observing day inside "
            f"{scenario.first_ord}-{scenario.last_ord}"
        )
    calendars = {**calendars, REFERENCE_STATION: anchor_days}
    population = [
        *population,
        replace(
            _perfect(
                ObserverParams(REFERENCE_STATION, truth.reference_threshold_msh),
                truth.reference_threshold_msh,
            ),
            kind="reference",
        ),
    ]

    network = observe(truth, population, calendars, rng)

    # The anchor survey's catalogue, as published data rather than as truth.
    visible = truth.group_days[truth.group_days["date_ord"].isin(set(anchor_days.tolist()))]
    network.reference_areas = visible.rename(columns={"area_msh": "area_msh"}).reset_index(
        drop=True
    )

    config = {
        **scenario.as_dict(),
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "n_observers": len(population),
        "n_groups": truth.n_groups,
        "reference_station": REFERENCE_STATION,
    }
    return truth, network, config
