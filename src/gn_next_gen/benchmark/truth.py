"""What the Sun did, in a synthetic world -- and the only module allowed to read it.

A benchmark is worth nothing if the methods under test can see the answer, so truth
lives behind this module and behind a separate directory, and
``tests/test_benchmark_blindness.py`` fails if anything under ``baselines``,
``models`` or ``inference`` imports it.  The spec calls this "API discipline; tests
enforce it".

Said plainly, because the results doc has to say it too: this is **not** a blind
trial in the HOME sense, where 25 outside groups submitted reconstructions before
the inserted inhomogeneities were revealed.  One developer writes both sides here.
Blindness is enforced by API and audited by test, and the baselines are frozen by
commit before the scoring run.  That is weaker than HOME and stronger than nothing.

The estimand is **not** the number of groups that exist (ADR-003: counts alone
cannot identify absolute scale).  It is the count a fixed reference observer would
have made, so a :class:`Truth` carries ``reference_threshold_msh`` -- the anchor
survey's own completeness floor -- and :meth:`Truth.reference_counts` is the series
every baseline is scored against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

#: Columns of the per-group table.
GROUP_COLUMNS = [
    "group_id",
    "emergence_ord",
    "visible_span",
    "peak_area_msh",
    "donor_survey",
    "donor_cycle",
]

#: Columns of the per-group-day table -- the instantaneous areas a detection layer
#: integrates over.  Step 1 §2.4 measured these as ~0.2 *smaller* in log area than
#: the per-group peaks, which is why both grains are carried.
GROUP_DAY_COLUMNS = ["group_id", "date_ord", "area_msh"]


@dataclass
class Truth:
    """One synthetic world's latent population.

    Attributes
    ----------
    groups
        One row per group: emergence day, visible span, peak area, and which real
        (survey, cycle) it was drawn from.
    group_days
        One row per group per visible day, with that day's instantaneous area.
    days
        The contiguous daily grid the world spans.
    reference_threshold_msh
        The anchor survey's completeness floor.  Defines the estimand (ADR-003).
    anchoring
        ``"debrecen_21_23"`` or ``"rgo_15_19"`` -- ADR-009's two co-equal anchors.
        A scenario states which one it uses; the scorer never mixes them.
    meta
        Generator name, seed, donor survey and anything else a run must record.
    """

    groups: pd.DataFrame
    group_days: pd.DataFrame
    days: np.ndarray
    reference_threshold_msh: float
    anchoring: str
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = set(GROUP_COLUMNS) - set(self.groups.columns)
        if missing:
            raise ValueError(f"groups is missing {sorted(missing)}")
        missing = set(GROUP_DAY_COLUMNS) - set(self.group_days.columns)
        if missing:
            raise ValueError(f"group_days is missing {sorted(missing)}")
        if self.anchoring not in ANCHORINGS:
            raise ValueError(f"unknown anchoring {self.anchoring!r}; expected {sorted(ANCHORINGS)}")

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    def counts(self, threshold_msh: float = 0.0) -> pd.Series:
        """Daily true count of groups whose area exceeds ``threshold_msh``.

        Strictly greater, the legacy convention kept throughout this project
        (``reuse_map.md`` row 1).  Indexed by ``date_ord`` on the full daily grid,
        with zeros on days no group was visible -- a synthetic world is never
        "unobserved", which is the whole point of having one.
        """
        visible = self.group_days[self.group_days["area_msh"] > threshold_msh]
        counted = visible.groupby("date_ord").size()
        return counted.reindex(self.days, fill_value=0).astype(np.int64).rename("n_groups")

    def reference_counts(self) -> pd.Series:
        """The estimand: counts at the anchor's own completeness threshold."""
        return self.counts(self.reference_threshold_msh)

    def activity_level(self, sigma_days: float = 182.0) -> pd.Series:
        """Whole-Sun activity level in mHem, the ADR-011 convention.

        Daily total area, smoothed with a 182-day Gaussian and divided by 1000
        (``area_msh`` is MSH; the published power laws are in mHem).  Carries no
        positional information, so it may feed Step 4 (ADR-004).
        """
        from gn_next_gen.reference.area_law import MUHEM_PER_MHEM, gaussian_smooth

        total = self.group_days.groupby("date_ord")["area_msh"].sum()
        total = total.reindex(self.days, fill_value=0.0)
        smoothed = gaussian_smooth(total.to_numpy(dtype=float), sigma_days)
        return pd.Series(smoothed / MUHEM_PER_MHEM, index=self.days, name="activity")

    def visible_on(self, date_ord: int) -> pd.DataFrame:
        """The groups on the disk on one day, with their instantaneous areas."""
        return self.group_days[self.group_days["date_ord"] == date_ord]


#: ADR-009's two co-equal anchored epochs, and the survey threshold each implies.
ANCHORINGS = {
    "debrecen_21_23": {"survey": "debrecen", "cycles": (21, 22, 23)},
    "rgo_15_19": {"survey": "rgo", "cycles": (15, 16, 17, 18, 19)},
}


def anchor_threshold(anchoring: str) -> float:
    """The anchor survey's completeness floor, from the frozen calibration table."""
    from gn_next_gen.reference.area_law import load_upstream_parameters

    if anchoring not in ANCHORINGS:
        raise ValueError(f"unknown anchoring {anchoring!r}; expected {sorted(ANCHORINGS)}")
    return float(load_upstream_parameters(ANCHORINGS[anchoring]["survey"]).threshold_msh)


def write_truth(truth: Truth, directory: str | Path) -> Path:
    """Write a :class:`Truth` to its sealed directory.

    The caller is responsible for that directory being the sealed one; see
    :mod:`gn_next_gen.benchmark.store`, which is what scenarios actually use.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    truth.groups.to_parquet(out / "groups.parquet", index=False)
    truth.group_days.to_parquet(out / "group_days.parquet", index=False)
    (out / "truth_meta.json").write_text(
        json.dumps(
            {
                "first_ord": int(truth.days[0]),
                "last_ord": int(truth.days[-1]),
                "reference_threshold_msh": truth.reference_threshold_msh,
                "anchoring": truth.anchoring,
                "meta": truth.meta,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return out


def read_truth(directory: str | Path) -> Truth:
    """Read back a :class:`Truth`.  Only scoring and diagnostics may call this."""
    source = Path(directory)
    meta = json.loads((source / "truth_meta.json").read_text())
    return Truth(
        groups=pd.read_parquet(source / "groups.parquet"),
        group_days=pd.read_parquet(source / "group_days.parquet"),
        days=np.arange(meta["first_ord"], meta["last_ord"] + 1, dtype=np.int64),
        reference_threshold_msh=meta["reference_threshold_msh"],
        anchoring=meta["anchoring"],
        meta=meta["meta"],
    )
