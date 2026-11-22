"""The blind benchmark: synthetic worlds, synthetic observers, and the scorer.

Truth lives in :mod:`gn_next_gen.benchmark.truth` and is not re-exported here, so
an accidental ``from gn_next_gen.benchmark import ...`` cannot reach it.
"""

from gn_next_gen.benchmark.calendars import observer_calendars, transplant_calendar
from gn_next_gen.benchmark.observers import (
    ObservedNetwork,
    ObserverParams,
    draw_population,
    observe,
)
from gn_next_gen.benchmark.scenarios import SCENARIOS, Scenario, build_realization

__all__ = [
    "SCENARIOS",
    "ObservedNetwork",
    "ObserverParams",
    "Scenario",
    "build_realization",
    "draw_population",
    "observe",
    "observer_calendars",
    "transplant_calendar",
]
