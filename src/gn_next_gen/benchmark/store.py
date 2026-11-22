"""Where a synthetic world lives on disk, and which half a method may read.

Two roots per realization::

    outputs/benchmark/<scenario>/seed-<n>/open/     observations, statements, metadata
    outputs/benchmark/<scenario>/seed-<n>/sealed/   truth, observer parameters

Fitting code reads ``open/``.  Only scoring and diagnostics read ``sealed/``, and
``tests/test_benchmark_blindness.py`` fails if a module under ``baselines``,
``models`` or ``inference`` so much as imports the module that can.

Caching deliberately does **not** go through :mod:`gn_next_gen.io.cache`: that keys
on ``MANIFEST.csv`` checksums and refuses unmanifested inputs by design (ADR-006),
which is right for raw data and wrong for a generated world.  A dataset is keyed by
the hash of its own generating config instead, so "same config + seed -> identical
dataset" is checkable by comparing one hex string.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from gn_next_gen.benchmark.observers import ObservedNetwork
from gn_next_gen.benchmark.truth import Truth, read_truth, write_truth
from gn_next_gen.paths import OUTPUTS

#: Default root for generated worlds.  Git-ignored, like everything in outputs/.
BENCHMARK_ROOT = OUTPUTS / "benchmark"

OPEN_FILES = ("observations.parquet", "statements.parquet", "metadata.parquet")
SEALED_FILES = ("groups.parquet", "group_days.parquet", "observer_parameters.parquet")


def config_hash(config: dict) -> str:
    """A stable 12-hex digest of a scenario's generating configuration."""
    payload = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def realization_dir(scenario: str, seed: int, *, root: Path | str = BENCHMARK_ROOT) -> Path:
    return Path(root) / scenario / f"seed-{seed:04d}"


def open_dir(scenario: str, seed: int, *, root: Path | str = BENCHMARK_ROOT) -> Path:
    return realization_dir(scenario, seed, root=root) / "open"


def sealed_dir(scenario: str, seed: int, *, root: Path | str = BENCHMARK_ROOT) -> Path:
    return realization_dir(scenario, seed, root=root) / "sealed"


def write_dataset(
    scenario: str,
    seed: int,
    truth: Truth,
    network: ObservedNetwork,
    config: dict,
    *,
    root: Path | str = BENCHMARK_ROOT,
) -> Path:
    """Write one realization, truth sealed away from what the methods read."""
    base = realization_dir(scenario, seed, root=root)
    opened, sealed = base / "open", base / "sealed"
    opened.mkdir(parents=True, exist_ok=True)
    sealed.mkdir(parents=True, exist_ok=True)

    network.observations.to_parquet(opened / "observations.parquet", index=False)
    network.statements.to_parquet(opened / "statements.parquet", index=False)
    network.metadata.to_parquet(opened / "metadata.parquet", index=False)

    write_truth(truth, sealed)
    network.parameters.to_parquet(sealed / "observer_parameters.parquet", index=False)
    network.thresholds.to_parquet(sealed / "observer_thresholds.parquet", index=False)

    (base / "config.json").write_text(
        json.dumps(
            {"scenario": scenario, "seed": seed, "hash": config_hash(config), **config},
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    return base


def is_complete(scenario: str, seed: int, *, root: Path | str = BENCHMARK_ROOT) -> bool:
    """Whether a realization is already on disk, so a resumed run can skip it."""
    base = realization_dir(scenario, seed, root=root)
    if not (base / "config.json").is_file():
        return False
    return all((base / "open" / name).is_file() for name in OPEN_FILES) and all(
        (base / "sealed" / name).is_file() for name in SEALED_FILES
    )


def read_observations(
    scenario: str, seed: int, *, root: Path | str = BENCHMARK_ROOT
) -> pd.DataFrame:
    """The open half: what every baseline is allowed to see."""
    return pd.read_parquet(open_dir(scenario, seed, root=root) / "observations.parquet")


def read_statements(scenario: str, seed: int, *, root: Path | str = BENCHMARK_ROOT) -> pd.DataFrame:
    return pd.read_parquet(open_dir(scenario, seed, root=root) / "statements.parquet")


def read_metadata(scenario: str, seed: int, *, root: Path | str = BENCHMARK_ROOT) -> pd.DataFrame:
    """Documented breaks -- the only observer truth a method may legitimately use."""
    return pd.read_parquet(open_dir(scenario, seed, root=root) / "metadata.parquet")


def read_sealed_truth(scenario: str, seed: int, *, root: Path | str = BENCHMARK_ROOT) -> Truth:
    """Scoring only."""
    return read_truth(sealed_dir(scenario, seed, root=root))


def read_sealed_parameters(
    scenario: str, seed: int, *, root: Path | str = BENCHMARK_ROOT
) -> pd.DataFrame:
    """Scoring only: the observer parameters a baseline is asked to recover."""
    return pd.read_parquet(sealed_dir(scenario, seed, root=root) / "observer_parameters.parquet")


def read_sealed_thresholds(
    scenario: str, seed: int, *, root: Path | str = BENCHMARK_ROOT
) -> pd.DataFrame:
    return pd.read_parquet(sealed_dir(scenario, seed, root=root) / "observer_thresholds.parquet")


def dataset_digest(
    scenario: str, seed: int, *, root: Path | str = BENCHMARK_ROOT
) -> dict[str, str]:
    """SHA-256 per written file -- the determinism criterion, as one dict."""
    base = realization_dir(scenario, seed, root=root)
    digests = {}
    for half in ("open", "sealed"):
        for path in sorted((base / half).glob("*")):
            digests[f"{half}/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests
