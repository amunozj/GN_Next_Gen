"""Run configuration: what a run reads, in which role, under which variant.

Every run is reproducible from a committed YAML plus a seed, and every run records
its data lineage (ADR-002).  The two rules the configuration exists to enforce are
the ones no amount of documentation has so far prevented someone from breaking:

* the **firewall** -- a ``sunspot_only`` run may not take an isotope, aurora,
  proxy or SILSO input as a calibration input (ADR-002, ADR-003);
* the **KMAS single-use rule** -- KMAS is reference areas *or* observer counts,
  never both, because its counts are derivable from its areas (ADR-009).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gn_next_gen.paths import DATA_RAW, GN_V13_CURRENT_RELEASE

SUNSPOT_ONLY = "sunspot_only"
PROXY_INFORMED = "proxy_informed"
VARIANTS = (SUNSPOT_ONLY, PROXY_INFORMED)

#: Roles a dataset can play in a run.  ``calibration`` is the only role that feeds
#: the likelihood; the other two are read but may not influence the posterior.
ROLES = ("calibration", "comparison", "diagnostic")

#: Inputs the firewall bars from calibration in the sunspot-only variant.
#: SILSO is included because SN V2.0 is itself a composite (ADR-003).
FIREWALLED_KINDS = ("silso", "isotope", "aurora", "proxy")

DEFAULT_THRESHOLDS = (0.0, 10.0, 30.0, 60.0, 100.0)


@dataclass(frozen=True)
class DatasetRef:
    """One raw input and the role it plays in this run."""

    path: str
    role: str = "calibration"
    kind: str = "sunspot"

    @property
    def resolved(self) -> Path:
        return DATA_RAW / self.path


@dataclass(frozen=True)
class RunConfig:
    """A reproducible run specification.

    Attributes
    ----------
    kmas_role
        ``"reference"`` (the default) keeps KMAS as a reference survey, where it is
        the only survey bridging the two ADR-009 anchors through cycle 20;
        ``"observer_counts"`` makes it a modern observer instead.
    duplicate_rule
        The D16 resolution rule; ``"max"`` by default.
    """

    name: str
    variant: str = SUNSPOT_ONLY
    seed: int = 0
    release: str = GN_V13_CURRENT_RELEASE
    kmas_role: str = "reference"
    duplicate_rule: str = "max"
    surveys: tuple[str, ...] = ("rgo", "debrecen", "kmas")
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    datasets: tuple[DatasetRef, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)

    def datasets_in_role(self, role: str) -> tuple[DatasetRef, ...]:
        return tuple(d for d in self.datasets if d.role == role)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_run_config(path: str | Path) -> RunConfig:
    """Read a YAML run configuration and validate it before returning it."""
    from gn_next_gen.config.validate import validate_run_config

    raw = yaml.safe_load(Path(path).read_text()) or {}
    datasets = tuple(DatasetRef(**d) for d in raw.pop("datasets", []))
    for key in ("thresholds", "surveys"):
        if key in raw and raw[key] is not None:
            raw[key] = tuple(raw[key])
    config = RunConfig(datasets=datasets, **raw)
    validate_run_config(config)
    return config
