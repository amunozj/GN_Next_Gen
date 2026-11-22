"""Truth must not be reachable from anything that fits a model to the data.

The spec asks for truth "stored separately from what fitting code may read (API
discipline; tests enforce it)".  This file is that enforcement, and it is static:
an import audit over the source, not a runtime check, so a path that is never
exercised still fails the suite.

What this does *not* claim: that the benchmark is blind in the HOME sense.  One
developer writes both the generator and the baselines here, where HOME had 25
outside groups submit before the answer was revealed.  Blindness is enforced by
API and audited here, the baselines are frozen by commit before the scoring run,
and the results doc says all of that plainly instead of claiming a blind trial.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from gn_next_gen.paths import REPO_ROOT

SRC = REPO_ROOT / "src" / "gn_next_gen"

#: Modules that may read truth: the container itself, the store's sealed readers,
#: the scorer, and the scenario builder that creates a world in the first place.
TRUTH_READERS = {
    "benchmark/truth.py",
    "benchmark/store.py",
    "benchmark/scoring.py",
    "benchmark/scenarios.py",
    "benchmark/case_studies.py",
    "benchmark/observers.py",
    "benchmark/generators/replay.py",
    "benchmark/generators/__init__.py",
    "benchmark/__init__.py",
}

#: Packages that fit models to data and must stay blind.
BLIND_PACKAGES = ("baselines", "models", "inference")

#: Names that reach sealed data, by import or by path.
SEALED_NAMES = ("benchmark.truth", "read_sealed", "sealed_dir", "read_truth")


def _python_files(package: str) -> list[Path]:
    root = SRC / package
    return sorted(root.rglob("*.py")) if root.is_dir() else []


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


@pytest.mark.parametrize("package", BLIND_PACKAGES)
def test_fitting_code_does_not_import_the_truth_module(package):
    offenders = []
    for path in _python_files(package):
        imports = _imported_modules(path)
        if any("benchmark.truth" in name for name in imports):
            offenders.append(str(path.relative_to(SRC)))
        if any(name.endswith(".read_sealed_truth") for name in imports):
            offenders.append(str(path.relative_to(SRC)))
    assert offenders == [], f"{package} reaches truth: {offenders}"


@pytest.mark.parametrize("package", BLIND_PACKAGES)
def test_fitting_code_never_names_the_sealed_directory(package):
    """Even a string literal is a leak waiting to happen."""
    offenders = [
        str(path.relative_to(SRC))
        for path in _python_files(package)
        if "sealed" in path.read_text()
    ]
    assert offenders == [], f"{package} names the sealed half: {offenders}"


def test_only_the_declared_modules_read_truth():
    """A new module that reads truth has to be added to TRUTH_READERS on purpose."""
    readers = []
    for path in sorted(SRC.rglob("*.py")):
        relative = str(path.relative_to(SRC))
        text = path.read_text()
        if any(name in text for name in SEALED_NAMES):
            readers.append(relative)
    unexpected = set(readers) - TRUTH_READERS
    assert unexpected == set(), f"undeclared truth readers: {sorted(unexpected)}"


def test_the_two_halves_are_written_to_different_directories(tmp_path):
    from gn_next_gen.benchmark.store import open_dir, sealed_dir

    assert open_dir("S1", 0, root=tmp_path) != sealed_dir("S1", 0, root=tmp_path)
    assert "sealed" not in str(open_dir("S1", 0, root=tmp_path))


def test_the_open_half_carries_no_truth_column():
    """Observations, statements and documented breaks -- nothing else."""
    import numpy as np

    from gn_next_gen.benchmark.scenarios import SCENARIOS, build_realization

    scenario = SCENARIOS["S1"]
    _truth, network, _config = build_realization(scenario, seed=0)

    assert set(network.observations.columns) == {"station", "date_ord", "groups"}
    assert set(network.statements.columns) == {
        "station",
        "year",
        "kind",
        "n_days",
        "n_active_days",
    }
    assert set(network.metadata.columns) == {"station", "date_ord", "note"}
    # The acuity trajectory and the parameters are sealed, not open.
    assert "threshold_msh" not in network.observations.columns
    assert np.isfinite(network.parameters["base_threshold_msh"]).all()
