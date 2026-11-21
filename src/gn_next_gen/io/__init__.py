"""Loaders for every raw input, long format throughout.

Nothing here densifies a day x observer matrix: the joint model reads long
(observer, day, count) records and gathers emissions with ``segment_sum``, so a
dense array would be ~9e7 mostly empty cells (see the research report, "A
41-state forward filter makes the 1610-present posterior tractable").

Cleaned tables are cached to git-ignored ``data/interim/`` keyed by the raw
inputs' manifest checksums; see :mod:`gn_next_gen.io.cache`.
"""

from gn_next_gen.io.observations import (
    DUPLICATE_RULES,
    load_observations,
    load_observers,
    resolve_duplicate_keys,
)
from gn_next_gen.io.reference import (
    CORE_FIVE,
    STEP1_SURVEYS,
    daily_counts,
    kmas_colliding_group_ids,
    load_day_status,
    load_group_days,
    load_group_single_entry,
)
from gn_next_gen.io.silso import load_silso_daily, load_silso_monthly

__all__ = [
    "CORE_FIVE",
    "DUPLICATE_RULES",
    "STEP1_SURVEYS",
    "daily_counts",
    "kmas_colliding_group_ids",
    "load_day_status",
    "load_group_days",
    "load_group_single_entry",
    "load_observations",
    "load_observers",
    "load_silso_daily",
    "load_silso_monthly",
    "resolve_duplicate_keys",
]
